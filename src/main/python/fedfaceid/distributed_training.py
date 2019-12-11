import dataclasses
from functools import reduce
from typing import Dict, List

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import federated as fd
from dataloaders import facemetadataset
from dataloaders.facemetadataset import PeopleDataset, FaceMetaSamples
from fedfaceid import fedfacedataset as ffd
from fedfaceid.commons import EarlyStopping
from fedfaceid.settings import Settings


def merge_subsets(s1: Subset, s2: Subset) -> Subset:
    s1.indices += s2.indices
    return s1


def average_results(results: List[fd.TrainingResult]) -> fd.TrainingResult:
    results_avg = fd.TrainingResult(0.0, 0, 0.0)
    for r in results:
        results_avg.loss += r.loss
        results_avg.learning_rate += r.learning_rate
        results_avg.steps += r.steps

    results_avg.loss /= len(results)
    results_avg.learning_rate /= len(results)
    results_avg.steps /= len(results)

    return results_avg


@dataclasses.dataclass
class EdgeDeviceSettings:
    batch_size: int
    epochs: int
    batches_in_epoch: int
    embedding_dim: int
    learning_rate: float
    learning_rate_decay: float
    loss_margin: float

    device: str

    num_local_images_to_use: int
    num_remote_images_to_use: int


class Training:
    def __init__(self, model: Module,
                 settings: EdgeDeviceSettings,
                 faces_metadata_local: FaceMetaSamples,
                 faces_metadata_remote: FaceMetaSamples):

        self.model: Module = model
        self.settings: EdgeDeviceSettings = settings
        self.faces_metadata_local: FaceMetaSamples = faces_metadata_local
        self.faces_metadata_remote: FaceMetaSamples = faces_metadata_remote

    def _calculate_embeddings(self, model: Module, people_dataset: PeopleDataset):
        image_loader = DataLoader(people_dataset,
                                  batch_size=self.settings.batch_size,
                                  shuffle=False)
        num_examples = len(people_dataset)

        embeddings = np.zeros((num_examples, self.settings.embedding_dim))

        start_idx = 0

        for i, image in tqdm(enumerate(image_loader)):
            batch_size = min(num_examples - i * self.settings.batch_size, self.settings.batch_size)
            image = image.cuda()
            embedding = model(image).cpu().detach().numpy()
            embeddings[start_idx: start_idx + batch_size, :] = embedding

            start_idx += self.settings.batch_size

        return embeddings

    def train_for_epoch(self):
        self.model.train()

        num_batches: int = 0
        while num_batches < self.settings.batches_in_epoch:
            # Selecting faces from available images
            faces_local: PeopleDataset = ffd.select_faces(
                self.faces_metadata_local,
                self.settings.num_local_images_to_use
            )

            faces_remote: PeopleDataset = ffd.select_faces(
                self.faces_metadata_remote,
                self.settings.num_remote_images_to_use
            )

            self.model.eval()
            with torch.no_grad():
                embeddings_local: np.array = self._calculate_embeddings(self.model, faces_local)
                embeddings_remote: np.array = self._calculate_embeddings(self.model, faces_remote)

                triplets = facemetadataset.select_triplets(
                    embeddings_local,
                    embeddings_remote,
                    self.settings.loss_margin
                )

                triplet_dataset = TripletsDataset(triplets, people_dataset)

                self.model.train()
                results: TrainStepResults = self.train_step(triplet_dataset)
                num_batches += results.steps

    def train_step(self, triplet_dataset: TripletsDataset) -> TrainStepResults:
        losses: List[float] = []
        local_step: int = 0

        triplet_loader = DataLoader(triplet_dataset,
                                    batch_size=self.settings.batch_size,
                                    shuffle=True)

        num_batches = int(np.ceil(len(triplet_dataset) / self.settings.batch_size))
        for triplets in tqdm(triplet_loader, total=num_batches):
            # Calculate triplet loss
            triplet_loss = self.loss_fn(anchor=self.model(triplets["anchor"].cuda()),
                                        positive=self.model(triplets["positive"].cuda()),
                                        negative=self.model(triplets["negative"].cuda())).cuda()

            # Backward pass
            self.optimizer.zero_grad()
            triplet_loss.backward()
            self.optimizer.step()

            self.global_step += 1
            local_step += 1
            losses.append(triplet_loss.item())

            if self.global_step % self.log_every_step == 0:
                self.tensorboard.add_scalar(name="loss_train",
                                            value=sum(losses[-self.log_every_step:]) / len(losses),
                                            global_step=self.global_step)
                losses: List[float] = []
            if self.global_step % self.evaluate_every_step == 0:
                print("Validating on LFW!")
                self.model.eval()

                metrics: EvaluationMetrics = evaluate(self.model,
                                                      self.distance_fn,
                                                      self.dataset_eval_loader,
                                                      None)
                self.tensorboard.add_dict(dataclasses.asdict(metrics), self.global_step)
                self.model.train()

        return TrainStepResults(0.0, local_step)


def train_distributed(model: Module,
                      loader_distributed: DataLoader,
                      loader_generated: DataLoader,
                      settings: Settings) -> Module:
    dataset_iter_validate = DataLoader(dataset_validate, batch_size=settings.num_global_batch)
    # num_users: int = len(dataset_train.classes)
    num_users = settings.num_users
    subsets_per_user = settings.num_subsets_per_user

    lr = settings.learning_rate * (2 - settings.learning_rate_decay)
    settings_edge_device = fd.EdgeDeviceSettings(epochs=settings.num_local_epochs,
                                                 batch_size=settings.num_local_batch,
                                                 learning_rate=lr,
                                                 learning_rate_decay=settings.learning_rate_decay,
                                                 device=settings.device)

    subsets: List[Subset]
    if settings.non_iid:
        subsets = data.split_dataset_non_iid(dataset_train, num_users * subsets_per_user)
    else:
        subsets = data.split_dataset_iid(dataset_train, num_users)

    users = []
    subsets_indices = list(range(len(subsets)))
    for i in range(num_users):
        indices = np.random.choice(subsets_indices, size=subsets_per_user, replace=False)

        [subsets_indices.remove(i) for i in indices]
        subset_for_user = reduce(merge_subsets, [subsets[i] for i in indices])

        user = fd.EdgeDevice(device_id=i, subset=subset_for_user, settings=settings_edge_device)
        users.append(user)

    max_users_in_round = max(int(settings.user_fraction * num_users), 1)

    early_stopping = EarlyStopping(settings.stopping_rounds)
    if settings.skip_stopping:
        early_stopping.disable()

    writer = SummaryWriter(str(settings.save_path.joinpath("tensorboard").joinpath(settings.id)))

    global_step = 0
    for i_epoch in range(settings.num_global_epochs):
        model.cuda()
        model.train()

        # local_models: Dict[int, Module] = {}
        local_results: Dict[int, fd.TrainingResult] = {}

        free_users = list(range(num_users))
        while free_users:
            users_in_round_ids = np.random.choice(free_users,
                                                  max_users_in_round,
                                                  replace=False)
            [free_users.remove(i) for i in users_in_round_ids]

            models: List[Module] = []
            for i_user in users_in_round_ids:
                user = users[i_user]
                user.download(model)
                local_results[i_user] = user.train()
                models.append(user.upload())

            # update global weights
            model = fd.federated_averaging(models)

            global_step += 1

        results_train: fd.TrainingResult = average_results(list(local_results.values()))
        results_eval: EvaluationResult = evaluate(model.cpu(), dataset_iter_validate)

        for key, value in dataclasses.asdict(results_eval).items():
            writer.add_scalar(key, value, global_step=global_step)
        writer.add_scalar("train_loss", results_train.loss, global_step=global_step)

        print(f"epoch={i_epoch}  "
              f"global_step={global_step}  "
              f"lr={results_train.learning_rate:.4f}  "
              f"train_loss={results_train.loss:.3f}  "
              f"eval_loss={results_eval.loss:.3f}  "
              f"eval_f1={results_eval.f1_score:.3f}  "
              f"eval_acc={results_eval.accuracy:.3f}")

        if early_stopping.is_best(results_eval.loss):
            torch.save(model.state_dict(),
                       settings.save_path.joinpath("model.pt"))

        if early_stopping.update(results_eval.loss).should_break:
            print("Early stopping! Loading best model.")
            model.load_state_dict(torch.load(settings.save_path.joinpath("model.pt")))
            break

    return model

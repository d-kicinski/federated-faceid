import copy
import dataclasses
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.nn import Module, PairwiseDistance
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

import federated as fd
from facenet.commons import ModelBuilder, get_validation_data_loader, load_checkpoint
from facenet.dataloaders.facemetadataset import (
    FaceMetaDataset,
    FaceMetaSamples,
    PeopleDataset,
)
from facenet.evaluation import EvaluationMetrics, evaluate
from facenet.train_triplet import Tensorboard, TrainStepResults
from fedfaceid import fedfacedataset as ffd
from fedfaceid.fedfacedataset import FederatedTripletsDataset
from fedfaceid.settings import DataSettings, FederatedSettings, ModelSettings


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
    loss_margin: float

    num_local_images_to_use: int
    num_remote_images_to_use: int


class DeviceTraining:
    def __init__(
        self,
        device_id: int,
        settings: EdgeDeviceSettings,
        faces_metadata_local: FaceMetaSamples,
        faces_metadata_remote: FaceMetaSamples,
    ):
        self.device_id: int = device_id
        self.settings: EdgeDeviceSettings = copy.deepcopy(settings)
        self.faces_metadata_local: FaceMetaSamples = faces_metadata_local
        self.faces_metadata_remote: FaceMetaSamples = faces_metadata_remote

        self.model: Optional[Module] = None

        self.loss_fn = torch.nn.TripletMarginLoss(
            margin=settings.loss_margin, reduction="mean"
        )

        self.global_step: int = 0

    def download(self, model: Module):
        self.model = copy.deepcopy(model).cpu()

    def upload(self) -> Module:
        if self.model is not None:
            model = copy.deepcopy(self.model)
            del self.model
            return model
        else:
            raise ValueError("Model not found on this device!")

    def _calculate_embeddings(self, model: Module, people_dataset: PeopleDataset):
        image_loader = DataLoader(
            people_dataset, batch_size=self.settings.batch_size, shuffle=False
        )
        num_examples = len(people_dataset)

        embeddings = np.zeros((num_examples, self.settings.embedding_dim))

        start_idx = 0

        for i, image in enumerate(image_loader):
            batch_size = min(
                num_examples - i * self.settings.batch_size, self.settings.batch_size
            )
            image = image.cuda()
            embedding = model(image).cpu().detach().numpy()
            embeddings[start_idx : start_idx + batch_size, :] = embedding

            start_idx += self.settings.batch_size

        return embeddings

    def train(self):
        self.model.cuda()
        self.model.train()
        optimizer = torch.optim.SGD(
            params=self.model.parameters(), lr=self.settings.learning_rate
        )

        epoch_loss = []
        local_steps: int = 0
        for _ in range(self.settings.epochs):
            # num_batches: int = 0

            # while num_batches < self.settings.batches_in_epoch:
            # Selecting faces from available images
            faces_local: PeopleDataset = ffd.select_faces(
                self.faces_metadata_local, self.settings.num_local_images_to_use
            )

            num_remote_image_to_use = sum(
                range(
                    1,
                    min(
                        self.settings.num_local_images_to_use,
                        len(self.faces_metadata_local),
                    ),
                )
            )
            faces_remote: PeopleDataset = ffd.select_faces(
                self.faces_metadata_remote, num_remote_image_to_use
            )

            self.model.eval()
            with torch.no_grad():
                embeddings_local: np.array = self._calculate_embeddings(
                    self.model, faces_local
                )
                embeddings_remote: np.array = self._calculate_embeddings(
                    self.model, faces_remote
                )

                triplets = ffd.select_triplets(
                    embeddings_local, embeddings_remote, self.settings.loss_margin
                )

                if len(triplets) == 0:
                    continue  # :(

                triplet_dataset = FederatedTripletsDataset(
                    triplets, faces_local, faces_remote
                )

            self.model.train()
            results: TrainStepResults = self.train_steps(optimizer, triplet_dataset)
            # num_batches += results.steps

            local_steps += results.steps
            epoch_loss.append(results.loss)

            # if len(faces_local) < self.settings.num_local_images_to_use:
            #     break

        # mean_loss = sum(epoch_loss) / len(epoch_loss)
        self.model.cpu()
        return fd.TrainingResult(
            loss=0.0, steps=local_steps, learning_rate=self.settings.learning_rate
        )

    def train_steps(
        self, optimizer: Optimizer, triplet_dataset: FederatedTripletsDataset
    ) -> TrainStepResults:
        losses: List[float] = []
        local_step: int = 0

        triplet_loader = DataLoader(
            triplet_dataset, batch_size=self.settings.batch_size, shuffle=True
        )

        for triplets in triplet_loader:
            # Calculate triplet loss
            triplet_loss = self.loss_fn(
                anchor=self.model(triplets["anchor"].cuda()),
                positive=self.model(triplets["positive"].cuda()),
                negative=self.model(triplets["negative"].cuda()),
            ).cuda()

            # Backward pass
            optimizer.zero_grad()
            triplet_loss.backward()
            optimizer.step()

            self.global_step += 1
            local_step += 1
            losses.append(triplet_loss.item())

        loss_mean = sum(losses) / len(losses)
        return TrainStepResults(loss_mean, local_step)


def federated_training(
    model: Module,
    global_step: int,
    start_epoch: int,
    face_local_meta_dataset: FaceMetaDataset,
    face_remote_meta_dataset: FaceMetaDataset,
    validate_dataloader: DataLoader,
    settings_federated: FederatedSettings,
    settings_model: ModelSettings,
    tensorboard: Tensorboard,
    distance_fn: Module,
    checkpoint_path: Path,
) -> Module:
    if settings_federated.num_users < 0:
        num_users = len(face_local_meta_dataset)
    else:
        num_users = settings_federated.num_users

    settings_edge_device = EdgeDeviceSettings(
        epochs=settings_federated.num_local_epochs,
        batch_size=settings_model.batch_size,
        learning_rate=settings_model.learning_rate,
        batches_in_epoch=settings_model.batches_in_epoch,
        loss_margin=settings_model.triplet_loss_margin,
        embedding_dim=settings_model.embedding_dim,
        num_local_images_to_use=settings_model.num_local_images_to_use,
        num_remote_images_to_use=settings_model.num_remote_images_to_use,
    )

    users = []
    for i in range(num_users):
        user = DeviceTraining(
            device_id=i,
            settings=settings_edge_device,
            faces_metadata_local=face_local_meta_dataset[i],
            faces_metadata_remote=face_remote_meta_dataset[0],
        )
        users.append(user)

    max_users_in_round = max(int(settings_federated.user_fraction * num_users), 1)
    model_accumulator = fd.ModelAccumulator()
    for i_epoch in range(start_epoch, settings_federated.num_global_epochs):
        model.cuda()
        model.train()

        local_results: Dict[int, fd.TrainingResult] = {}

        free_users = list(range(num_users))
        while free_users:
            users_in_round_ids = np.random.choice(
                free_users, max_users_in_round, replace=False
            )
            [free_users.remove(i) for i in users_in_round_ids]

            for i_user in users_in_round_ids:
                # print(f"Training user {i_user}")
                user = users[i_user]
                user.download(model)
                local_results[i_user] = user.train()
                model_accumulator.update(user.upload())

                # print(f"Did {local_results[i_user].steps} steps")

            # update global weights
            model = model_accumulator.get()
            model_accumulator.reset()

            global_step += 1

        results_train: fd.TrainingResult = average_results(list(local_results.values()))

        metrics: EvaluationMetrics = evaluate(
            model.cuda(), distance_fn, validate_dataloader, None
        )

        pprint(dataclasses.asdict(metrics))
        tensorboard.add_dict(dataclasses.asdict(metrics), global_step)
        tensorboard.add_scalar(
            "loss_train", results_train.loss, global_step=global_step
        )

        # Save model checkpoint
        state = {
            "model_architecture": settings_model.model_architecture,
            "epoch": i_epoch,
            "global_step": global_step,
            "embedding_dimension": settings_model.embedding_dim,
            "batch_size_training": settings_model.batch_size,
            "model_state_dict": model.state_dict(),
        }

        # Save model checkpoint
        checkpoint_name = (
            f"{settings_model.model_architecture}_{i_epoch}_{global_step}.pt"
        )
        torch.save(state, checkpoint_path.joinpath(checkpoint_name))
    return model


def train(
    settings_model: ModelSettings,
    settings_data: DataSettings,
    settings_federated: FederatedSettings,
):
    output_dir: Path = settings_data.output_dir
    output_dir_logs = output_dir.joinpath("logs")
    output_dir_plots = output_dir.joinpath("plots")
    output_dir_checkpoints = output_dir.joinpath("checkpoints")
    output_dir_tensorboard = output_dir.joinpath("tensorboard")

    output_dir_logs.mkdir(exist_ok=True, parents=True)
    output_dir_plots.mkdir(exist_ok=True, parents=True)
    output_dir_checkpoints.mkdir(exist_ok=True, parents=True)

    model_architecture = settings_model.model_architecture

    start_epoch: int = 0
    global_step: int = 0

    data_loader_validate: DataLoader = get_validation_data_loader(
        settings_model, settings_data
    )

    model: Module = ModelBuilder.build(
        settings_model.model_architecture,
        settings_model.embedding_dim,
        settings_model.pretrained_on_imagenet,
    )

    print("Using {} model architecture.".format(model_architecture))

    # Load model to GPU or multiple GPUs if available
    if torch.cuda.is_available():
        print("Using single-gpu training.")
        model.cuda()

    if settings_data.checkpoint_path:
        checkpoint = load_checkpoint(settings_data.checkpoint_path, model, None)
        model = checkpoint.model
        start_epoch = checkpoint.epoch
        global_step = checkpoint.global_step

    # Start Training loop

    face_local__meta_dataset = FaceMetaDataset(
        root_dir=settings_data.dataset_local_dir,
        csv_name=settings_data.dataset_local_csv_file,
        min_images_per_class=2,
    )

    face_remote_meta_dataset = FaceMetaDataset(
        root_dir=settings_data.dataset_remote_dir,
        csv_name=settings_data.dataset_remote_csv_file,
        min_images_per_class=1,
    )

    l2_distance = PairwiseDistance(2).cuda()

    tensorboard = Tensorboard(output_dir_tensorboard)

    federated_training(
        model=model,
        global_step=global_step,
        start_epoch=start_epoch,
        face_local_meta_dataset=face_local__meta_dataset,
        face_remote_meta_dataset=face_remote_meta_dataset,
        validate_dataloader=data_loader_validate,
        settings_federated=settings_federated,
        settings_model=settings_model,
        tensorboard=tensorboard,
        distance_fn=l2_distance,
        checkpoint_path=output_dir_checkpoints,
    )

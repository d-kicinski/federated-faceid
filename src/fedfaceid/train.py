import argparse
from pathlib import Path

from fedfaceid import distributed_training
from fedfaceid.settings import DataSettings, FederatedSettings, ModelSettings


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training FaceNet facial recognition model using Triplet Loss."
    )

    # Dataset settings
    parser.add_argument(
        "--output_dir", type=lambda p: Path(p), default=DataSettings.output_dir
    )

    parser.add_argument(
        "--lfw_dir", type=lambda p: Path(p), default=DataSettings.lfw_dir
    )

    parser.add_argument(
        "--dataset_local_dir",
        type=lambda p: Path(p),
        default=DataSettings.dataset_local_dir,
    )
    parser.add_argument(
        "--dataset_local_csv_file",
        type=lambda p: Path(p),
        default=DataSettings.dataset_local_csv_file,
    )

    parser.add_argument(
        "--dataset_remote_dir",
        type=lambda p: Path(p),
        default=DataSettings.dataset_remote_dir,
    )
    parser.add_argument(
        "--dataset_remote_csv_file",
        type=lambda p: Path(p),
        default=DataSettings.dataset_remote_csv_file,
    )

    parser.add_argument(
        "--checkpoint_path",
        type=lambda p: Path(p) if p else None,
        default=DataSettings.checkpoint_path,
    )

    # Model settings
    parser.add_argument(
        "--lfw_batch_size", default=ModelSettings.lfw_batch_size, type=int
    )
    parser.add_argument(
        "--lfw_validation_epoch_interval",
        type=int,
        default=ModelSettings.lfw_validation_epoch_interval,
    )

    parser.add_argument(
        "--model_architecture",
        type=str,
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "inceptionresnetv1",
            "inceptionresnetv2",
        ],
        default=ModelSettings.model_architecture,
    )

    parser.add_argument(
        "--batch_size",
        default=ModelSettings.batch_size,
        type=int,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=ModelSettings.learning_rate
    )
    parser.add_argument(
        "--embedding_dim", default=ModelSettings.embedding_dim, type=int
    )
    parser.add_argument(
        "--triplet_loss_margin", type=float, default=ModelSettings.triplet_loss_margin
    )
    parser.add_argument(
        "--pretrained_on_imagenet",
        action="store_true",
        default=ModelSettings.pretrained_on_imagenet,
    )

    parser.add_argument(
        "--num_workers",
        default=ModelSettings.num_workers,
        type=int,
        help="Number of workers for data loaders (default: 4)",
    )

    parser.add_argument(
        "--people_per_batch", default=ModelSettings.people_per_batch, type=int
    )
    parser.add_argument(
        "--images_per_person", default=ModelSettings.images_per_person, type=int
    )
    parser.add_argument(
        "--batches_in_epoch", default=ModelSettings.batches_in_epoch, type=int
    )

    parser.add_argument(
        "--num_local_images_to_use",
        type=int,
        default=ModelSettings.num_local_images_to_use,
    )
    parser.add_argument(
        "--num_remote_images_to_use",
        type=int,
        default=ModelSettings.num_remote_images_to_use,
    )

    # Federated settings
    parser.add_argument(
        "--num_global_epochs", type=int, default=FederatedSettings.num_global_epochs
    )
    parser.add_argument(
        "--num_global_batch", type=int, default=FederatedSettings.num_global_batch
    )
    parser.add_argument(
        "--num_users",
        type=int,
        default=FederatedSettings.num_users,
        help="number of users: K",
    )
    parser.add_argument(
        "--user_fraction",
        type=float,
        default=FederatedSettings.user_fraction,
        help="the fraction of clients: C",
    )
    parser.add_argument(
        "--num_local_epochs",
        type=int,
        default=FederatedSettings.num_local_epochs,
        help="the number of local epochs: E",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    settings_data = DataSettings(
        output_dir=args.output_dir,
        lfw_dir=args.lfw_dir,
        dataset_local_dir=args.dataset_local_dir,
        dataset_local_csv_file=args.dataset_local_csv_file,
        dataset_remote_dir=args.dataset_remote_dir,
        dataset_remote_csv_file=args.dataset_remote_csv_file,
        checkpoint_path=args.checkpoint_path,
    )

    settings_model = ModelSettings(
        lfw_batch_size=args.lfw_batch_size,
        lfw_validation_epoch_interval=args.lfw_validation_epoch_interval,
        model_architecture=args.model_architecture,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim,
        triplet_loss_margin=args.triplet_loss_margin,
        pretrained_on_imagenet=args.pretrained_on_imagenet,
        num_workers=args.num_workers,
        people_per_batch=args.people_per_batch,
        images_per_person=args.images_per_person,
        batches_in_epoch=args.batches_in_epoch,
        num_local_images_to_use=args.num_local_images_to_use,
        num_remote_images_to_use=args.num_remote_images_to_use,
    )

    settings_federated = FederatedSettings(
        num_global_batch=args.num_global_batch,
        num_global_epochs=args.num_global_epochs,
        num_local_epochs=args.num_local_epochs,
        num_users=args.num_users,
        user_fraction=args.user_fraction,
    )

    distributed_training.train(
        settings_data=settings_data,
        settings_model=settings_model,
        settings_federated=settings_federated,
    )


if __name__ == "__main__":
    main()

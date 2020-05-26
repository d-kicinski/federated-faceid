import argparse
import dataclasses
from pathlib import Path
from pprint import pprint

from torch.nn import Module, PairwiseDistance
from torch.utils.data import DataLoader

from facenet.commons import ModelBuilder, get_validation_data_loader, load_checkpoint
from facenet.evaluation import EvaluationMetrics, evaluate
from facenet.settings import DataSettings, ModelSettings


def evaluate_model(settings_model: ModelSettings, settings_data: DataSettings):
    data_loader_validate: DataLoader = get_validation_data_loader(
        settings_model, settings_data
    )

    distance_l2: Module = PairwiseDistance(2).cuda()
    model: Module = ModelBuilder.build(
        settings_model.model_architecture,
        settings_model.embedding_dim,
        imagenet_pretrained=False,
    )
    model = model.cuda()

    checkpoint = load_checkpoint(
        checkpoint_path=settings_data.checkpoint_path, model=model
    )
    model = checkpoint.model
    epoch_last = checkpoint.epoch

    model.eval()

    figure_name = f"roc_eval_{epoch_last}.png"
    figure_path: Path = settings_data.output_dir.joinpath(figure_name)

    metrics: EvaluationMetrics = evaluate(
        model, distance_l2, data_loader_validate, figure_path
    )
    pprint(dataclasses.asdict(metrics))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on LFW dataset")
    parser.add_argument(
        "--output_dir",
        type=lambda p: Path(p) if p else None,
        default=DataSettings.output_dir,
    )
    parser.add_argument(
        "--checkpoint_path", type=lambda p: Path(p) if p else None, required=True
    )
    parser.add_argument(
        "--lfw_dir", type=lambda p: Path(p), default=DataSettings.lfw_dir
    )
    parser.add_argument(
        "--lfw_batch_size", type=int, default=ModelSettings.lfw_batch_size
    )
    parser.add_argument(
        "--model_architecture",
        type=str,
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "inception_resnet_v1_vggface2",
            "inception_resnet_v1_casia",
            "inceptionresnetv2",
        ],
        default=ModelSettings.model_architecture,
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=ModelSettings.embedding_dim
    )

    return parser.parse_args()


def main():
    args = parse_args()
    model_settings = ModelSettings(
        model_architecture=args.model_architecture,
        embedding_dim=args.embedding_dim,
        lfw_batch_size=args.lfw_batch_size,
    )

    data_settings = DataSettings(
        checkpoint_path=args.checkpoint_path,
        lfw_dir=args.lfw_dir,
        output_dir=args.output_dir,
    )

    evaluate_model(model_settings, data_settings)


if __name__ == "__main__":
    main()

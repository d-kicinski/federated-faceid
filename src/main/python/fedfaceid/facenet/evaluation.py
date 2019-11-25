from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from plots import plot_roc_lfw
from validate_on_LFW import evaluate_lfw


@dataclass
class EvaluationMetrics:
    accuracy: float
    precision: float
    recall: float
    roc_auc: float
    tar: float
    far: float
    distance: float


def evaluate(model: Module, distance_fn: Module, data_loader: DataLoader, figure_path: Path) \
        -> EvaluationMetrics:
    with torch.no_grad():
        distances, labels = [], []

        for batch_index, (data_a, data_b, label) in enumerate(tqdm(data_loader)):
            data_a, data_b, label = data_a.cuda(), data_b.cuda(), label.cuda()

            output_a, output_b = model(data_a), model(data_b)
            distance = distance_fn(output_a, output_b)  # Euclidean distance

            distances.append(distance.cpu().detach().numpy())
            labels.append(label.cpu().detach().numpy())

        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for distance in distances for subdist in distance])

        (true_positive_rate, false_positive_rate,
         precision, recall, accuracy,
         roc_auc, best_distances, tar, far) = evaluate_lfw(distances=distances, labels=labels)

        # Plot ROC curve
        plot_roc_lfw(false_positive_rate=false_positive_rate,
                     true_positive_rate=true_positive_rate,
                     figure_name=str(figure_path))

        return EvaluationMetrics(accuracy=float(np.mean(accuracy)),
                                 precision=float(np.mean(precision)),
                                 recall=float(np.mean(recall)),
                                 roc_auc=roc_auc,
                                 tar=float(np.mean(tar)),
                                 far=float(np.mean(far)),
                                 distance=float(np.mean(best_distances)))

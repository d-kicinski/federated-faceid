from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.nn import Module, functional
from torch.utils.data import DataLoader


@dataclass
class EvaluationResult:
    f1_score: float
    accuracy: float
    loss: float

    def __str__(self):
        return (
            f"eval_loss={self.loss:.3f}\t"
            f"macro_f1={self.f1_score:.3f}\t"
            f"accuracy={self.accuracy}"
        )


def evaluate(
    model: Module, data_loader: DataLoader, verbose: bool = False
) -> EvaluationResult:
    loss: float = 0.0

    labels_golden: List[int] = []
    labels_system: List[int] = []

    for idx, (data, target) in enumerate(data_loader):
        logits: torch.Tensor = model(data)
        loss += functional.cross_entropy(logits, target).item()

        predictions = np.argmax(logits.data, axis=-1)

        labels_golden.extend(target.data.numpy().tolist())
        labels_system.extend(predictions.tolist())

    f1 = f1_score(labels_golden, labels_system, average="macro")
    accuracy = accuracy_score(labels_golden, labels_system)

    if verbose:
        print(classification_report(labels_golden, labels_system))

    return EvaluationResult(f1, accuracy, loss)

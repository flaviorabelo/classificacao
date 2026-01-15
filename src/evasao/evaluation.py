from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from .data import DatasetSplit


@dataclass
class EvaluationResult:
    model_name: str
    accuracy: float
    report: str


def evaluate(model_name: str, model: object, dataset: DatasetSplit) -> EvaluationResult:
    """Calcula métricas de avaliação para o conjunto informado."""
    predictions = model.predict(dataset.features)
    accuracy = accuracy_score(dataset.labels, predictions)
    report = classification_report(dataset.labels, predictions)
    return EvaluationResult(model_name=model_name, accuracy=accuracy, report=report)


def compute_baseline(dataset: DatasetSplit) -> float:
    """Calcula a acurácia de um modelo baseline que escolhe a classe majoritária."""
    majority_class = dataset.labels.value_counts().idxmax()
    predictions = np.full(shape=len(dataset.labels), fill_value=majority_class)
    return accuracy_score(dataset.labels, predictions)

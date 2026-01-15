from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from sklearn.base import BaseEstimator

from .config import DATASET_PATH
from .data import DatasetBundle, DatasetSplit, load_dataset, split_dataset
from .evaluation import EvaluationResult, compute_baseline, evaluate
from .models import get_models


@dataclass
class TrainingOutcome:
    dataset: DatasetBundle
    models: Dict[str, BaseEstimator]
    evaluations: List[EvaluationResult]
    winner_name: str
    baseline_accuracy: float
    test_evaluation: EvaluationResult


def _train_and_validate(dataset: DatasetBundle, models: Dict[str, BaseEstimator]) -> List[EvaluationResult]:
    evaluations: List[EvaluationResult] = []

    for model_name, model in models.items():
        model.fit(dataset.train.features, dataset.train.labels)
        evaluation = evaluate(model_name, model, dataset.validation)
        evaluations.append(evaluation)

    return evaluations


def _merge_train_and_validation(dataset: DatasetBundle) -> DatasetSplit:
    """Combina treino e validação para usar o máximo de dados no ajuste final."""
    features = pd.concat([dataset.train.features, dataset.validation.features])
    labels = pd.concat([dataset.train.labels, dataset.validation.labels])
    return DatasetSplit(features=features, labels=labels)


def _evaluate_on_test(dataset: DatasetBundle, models: Dict[str, BaseEstimator], winner_name: str) -> EvaluationResult:
    winner_model = models[winner_name]
    full_training = _merge_train_and_validation(dataset)
    winner_model.fit(full_training.features, full_training.labels)
    return evaluate(winner_name, winner_model, dataset.test)


def train_pipeline(dataset_path: str | None = None) -> TrainingOutcome:
    """Executa o fluxo completo de treinamento, seleção e avaliação final.

    Args:
        dataset_path: caminho opcional para sobrescrever o dataset padrão.
    """
    features, labels = load_dataset(dataset_path or DATASET_PATH)
    dataset = split_dataset(features, labels)

    models = get_models()
    evaluations = _train_and_validate(dataset, models)
    winner = max(evaluations, key=lambda item: item.accuracy)

    baseline_accuracy = compute_baseline(dataset.validation)
    test_evaluation = _evaluate_on_test(dataset, models, winner.model_name)

    return TrainingOutcome(
        dataset=dataset,
        models=models,
        evaluations=evaluations,
        winner_name=winner.model_name,
        baseline_accuracy=baseline_accuracy,
        test_evaluation=test_evaluation,
    )

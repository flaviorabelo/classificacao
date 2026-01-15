from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATASET_PATH, FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE, TRAIN_SIZE


@dataclass
class DatasetSplit:
    features: pd.DataFrame
    labels: pd.Series


@dataclass
class DatasetBundle:
    train: DatasetSplit
    validation: DatasetSplit
    test: DatasetSplit


def load_dataset(dataset_path: Path | str = DATASET_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    """Carrega o dataset de evasão escolar.

    Args:
        dataset_path: caminho para o arquivo CSV separado por `;`.

    Returns:
        Uma tupla com o DataFrame de atributos e a Série de rótulos.
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Arquivo do dataset não encontrado: {dataset_path}")

    df = pd.read_csv(dataset_path, sep=";")
    features = df[FEATURE_COLUMNS]
    labels = df[TARGET_COLUMN]
    return features, labels


def split_dataset(
    features: pd.DataFrame,
    labels: pd.Series,
    train_size: float = TRAIN_SIZE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> DatasetBundle:
    """Divide o conjunto em treino, validação e teste."""
    temp_size = 1 - train_size
    features_train, features_temp, labels_train, labels_temp = train_test_split(
        features, labels, train_size=train_size, random_state=random_state, stratify=labels
    )

    validation_ratio = test_size / temp_size
    features_validation, features_test, labels_validation, labels_test = train_test_split(
        features_temp,
        labels_temp,
        test_size=validation_ratio,
        random_state=random_state,
        stratify=labels_temp,
    )

    return DatasetBundle(
        train=DatasetSplit(features_train, labels_train),
        validation=DatasetSplit(features_validation, labels_validation),
        test=DatasetSplit(features_test, labels_test),
    )

from __future__ import annotations

from typing import Dict

from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB

from .config import RANDOM_STATE


def get_models() -> Dict[str, object]:
    """Retorna o conjunto de modelos que serão avaliados."""
    return {
        "MultinomialNB": MultinomialNB(),
        "AdaBoostClassifier": AdaBoostClassifier(random_state=RANDOM_STATE),
    }

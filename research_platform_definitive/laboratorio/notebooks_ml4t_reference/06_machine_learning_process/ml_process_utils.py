"""Portable helpers for the ML process chapter."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import learning_curve, validation_curve


def make_regression_frame(n_samples: int = 800, n_features: int = 12, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    x, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=6, noise=12.0, random_state=seed)
    return pd.DataFrame(x, columns=[f"feature_{i:02d}" for i in range(n_features)]), pd.Series(y, name="target")


def make_classification_frame(n_samples: int = 800, n_features: int = 10, seed: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    x, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=5, n_redundant=2, random_state=seed)
    return pd.DataFrame(x, columns=[f"feature_{i:02d}" for i in range(n_features)]), pd.Series(y, name="target")


def learning_curve_frame(estimator, x: pd.DataFrame, y: pd.Series, cv: int = 5, scoring: str = "r2") -> pd.DataFrame:
    train_sizes, train_scores, test_scores = learning_curve(estimator, x, y, cv=cv, scoring=scoring, train_sizes=np.linspace(0.2, 1.0, 5))
    return pd.DataFrame({"train_size": train_sizes, "train_score_mean": train_scores.mean(axis=1), "test_score_mean": test_scores.mean(axis=1), "train_score_std": train_scores.std(axis=1), "test_score_std": test_scores.std(axis=1)})


def validation_curve_frame(estimator, x: pd.DataFrame, y: pd.Series, param_name: str, param_range, cv: int = 5, scoring: str = "r2") -> pd.DataFrame:
    train_scores, test_scores = validation_curve(estimator, x, y, param_name=param_name, param_range=param_range, cv=cv, scoring=scoring)
    return pd.DataFrame({"param_value": list(param_range), "train_score_mean": train_scores.mean(axis=1), "test_score_mean": test_scores.mean(axis=1), "train_score_std": train_scores.std(axis=1), "test_score_std": test_scores.std(axis=1)})

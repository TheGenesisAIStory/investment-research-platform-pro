"""Expected-return prediction models with sklearn-like APIs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .valuation import _make_model


class ExpectedReturnModel:
    """Predict future returns: r_hat_i,t+h = g_t(X_i,t)."""

    def __init__(self, model: str = "rf", random_state: int = 42, **kwargs: Any) -> None:
        self.model = model
        self.random_state = random_state
        self.kwargs = kwargs
        self.estimator_ = _make_model(model, random_state, **kwargs)
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ExpectedReturnModel":
        X = pd.DataFrame(X).astype(float).replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
        y = pd.to_numeric(y, errors="coerce")
        mask = y.notna()
        self.feature_names_ = list(X.columns)
        if self.estimator_ is not None:
            self.estimator_.fit(X.loc[mask], y.loc[mask])
        else:
            x = np.c_[np.ones(len(X.loc[mask])), X.loc[mask].to_numpy()]
            self.beta_ = np.linalg.pinv(x.T @ x) @ x.T @ y.loc[mask].to_numpy()
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X = pd.DataFrame(X).reindex(columns=self.feature_names_).astype(float).replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
        if self.estimator_ is not None:
            pred = self.estimator_.predict(X)
        else:
            pred = np.c_[np.ones(len(X)), X.to_numpy()] @ self.beta_
        return pd.Series(pred, index=X.index, name="expected_return")

    def fit_predict(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        return self.fit(X, y).predict(X)


def temporal_train_test_split(panel: pd.DataFrame, test_fraction: float = 0.25, date_col: str = "date") -> tuple[pd.Index, pd.Index]:
    if date_col not in panel.columns:
        cutoff = int(len(panel) * (1 - test_fraction))
        return panel.index[:cutoff], panel.index[cutoff:]
    dates = pd.Series(pd.to_datetime(panel[date_col], errors="coerce")).sort_values().dropna().unique()
    if len(dates) < 2:
        cutoff = int(len(panel) * (1 - test_fraction))
        return panel.index[:cutoff], panel.index[cutoff:]
    split_date = dates[max(0, int(len(dates) * (1 - test_fraction)) - 1)]
    train_idx = panel[pd.to_datetime(panel[date_col], errors="coerce") <= split_date].index
    test_idx = panel[pd.to_datetime(panel[date_col], errors="coerce") > split_date].index
    return train_idx, test_idx


def _date_or_na(values: pd.Series, op: str) -> Any:
    clean = pd.to_datetime(values, errors="coerce").dropna()
    if clean.empty:
        return pd.NA
    value = clean.min() if op == "min" else clean.max()
    return value.date().isoformat()


def describe_temporal_split(
    panel: pd.DataFrame,
    train_idx: pd.Index,
    test_idx: pd.Index,
    date_col: str = "date",
) -> dict[str, Any]:
    """Describe the split used for expected-return experiments."""
    meta: dict[str, Any] = {
        "split_type": "row_order",
        "split_criterion": f"row_position < {len(train_idx)}; row_position >= {len(train_idx)}",
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_date_count": pd.NA,
        "test_date_count": pd.NA,
        "train_start_date": pd.NA,
        "train_end_date": pd.NA,
        "test_start_date": pd.NA,
        "test_end_date": pd.NA,
    }
    if date_col not in panel.columns:
        return meta

    train_dates = pd.to_datetime(panel.loc[train_idx, date_col], errors="coerce")
    test_dates = pd.to_datetime(panel.loc[test_idx, date_col], errors="coerce")
    train_end = _date_or_na(train_dates, "max")
    meta.update({
        "split_type": "temporal",
        "split_criterion": f"{date_col} <= {train_end}; {date_col} > {train_end}",
        "train_date_count": int(train_dates.dropna().nunique()),
        "test_date_count": int(test_dates.dropna().nunique()),
        "train_start_date": _date_or_na(train_dates, "min"),
        "train_end_date": train_end,
        "test_start_date": _date_or_na(test_dates, "min"),
        "test_end_date": _date_or_na(test_dates, "max"),
    })
    return meta

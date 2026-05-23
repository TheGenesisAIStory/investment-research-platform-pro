"""Expected-return prediction models and temporal split helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml_stock_lab.valuation.peer_ols import _make_estimator


class ExpectedReturnModel:
    """Predict future returns from stock features with a sklearn-like API."""

    def __init__(self, model: str = "rf", random_state: int = 42, **kwargs: Any) -> None:
        self.model = model
        self.random_state = random_state
        self.kwargs = kwargs
        self.estimator_ = _make_estimator(model, random_state=random_state, **kwargs)
        self.feature_names_: list[str] = []
        self.beta_: np.ndarray | None = None

    def _prepare_X(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).astype(float).replace([np.inf, -np.inf], np.nan)
        return frame.fillna(frame.median(numeric_only=True)).fillna(0.0)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ExpectedReturnModel":
        """Fit the expected-return model."""
        Xp = self._prepare_X(X)
        yp = pd.to_numeric(y, errors="coerce")
        mask = yp.notna()
        self.feature_names_ = list(Xp.columns)
        if self.estimator_ is not None:
            self.estimator_.fit(Xp.loc[mask], yp.loc[mask])
        else:
            x = np.c_[np.ones(mask.sum()), Xp.loc[mask].to_numpy()]
            self.beta_ = np.linalg.pinv(x.T @ x) @ x.T @ yp.loc[mask].to_numpy()
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict expected returns for rows in `X`."""
        Xp = self._prepare_X(pd.DataFrame(X).reindex(columns=self.feature_names_))
        if self.estimator_ is not None:
            values = self.estimator_.predict(Xp)
        else:
            beta = self.beta_ if self.beta_ is not None else np.zeros(Xp.shape[1] + 1)
            values = np.c_[np.ones(len(Xp)), Xp.to_numpy()] @ beta
        return pd.Series(values, index=Xp.index, name="expected_return")

    def fit_predict(self, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame | None = None) -> pd.Series:
        """Fit on `X, y` and predict on `X_test` or `X` if omitted."""
        return self.fit(X, y).predict(X if X_test is None else X_test)


def temporal_train_test_split(panel: pd.DataFrame, test_fraction: float = 0.25, date_col: str = "date") -> tuple[pd.Index, pd.Index]:
    """Return train/test indices using dates when available, else row order."""
    if date_col not in panel.columns:
        cutoff = int(len(panel) * (1 - test_fraction))
        return panel.index[:cutoff], panel.index[cutoff:]
    dates = pd.Series(pd.to_datetime(panel[date_col], errors="coerce")).sort_values().dropna().unique()
    if len(dates) < 2:
        cutoff = int(len(panel) * (1 - test_fraction))
        return panel.index[:cutoff], panel.index[cutoff:]
    split_date = dates[max(0, int(len(dates) * (1 - test_fraction)) - 1)]
    dt = pd.to_datetime(panel[date_col], errors="coerce")
    return panel[dt <= split_date].index, panel[dt > split_date].index


def describe_temporal_split(panel: pd.DataFrame, train_idx: pd.Index, test_idx: pd.Index, date_col: str = "date") -> dict[str, Any]:
    """Describe the split used to guard against look-ahead leakage."""
    meta: dict[str, Any] = {
        "split_type": "row_order",
        "split_criterion": f"row_position < {len(train_idx)}; row_position >= {len(train_idx)}",
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_end_date": pd.NA,
        "test_start_date": pd.NA,
    }
    if date_col not in panel.columns:
        return meta
    train_dates = pd.to_datetime(panel.loc[train_idx, date_col], errors="coerce").dropna()
    test_dates = pd.to_datetime(panel.loc[test_idx, date_col], errors="coerce").dropna()
    train_end = train_dates.max().date().isoformat() if not train_dates.empty else pd.NA
    test_start = test_dates.min().date().isoformat() if not test_dates.empty else pd.NA
    meta.update({
        "split_type": "temporal",
        "split_criterion": f"{date_col} <= {train_end}; {date_col} > {train_end}",
        "train_start_date": train_dates.min().date().isoformat() if not train_dates.empty else pd.NA,
        "train_end_date": train_end,
        "test_start_date": test_start,
        "test_end_date": test_dates.max().date().isoformat() if not test_dates.empty else pd.NA,
        "train_date_count": int(train_dates.nunique()),
        "test_date_count": int(test_dates.nunique()),
    })
    return meta


def oos_r2(y_true: pd.Series, y_pred: pd.Series, benchmark: pd.Series | float | None = None) -> float:
    """Compute out-of-sample R2 against a mean or supplied benchmark."""
    y = pd.to_numeric(y_true, errors="coerce")
    p = pd.to_numeric(y_pred, errors="coerce")
    if benchmark is None:
        b = pd.Series(y.mean(), index=y.index)
    elif isinstance(benchmark, (int, float)):
        b = pd.Series(float(benchmark), index=y.index)
    else:
        b = pd.to_numeric(benchmark, errors="coerce")
    mask = y.notna() & p.notna() & b.notna()
    if mask.sum() == 0:
        return float("nan")
    denom = ((y[mask] - b[mask]) ** 2).sum()
    return float(1 - ((y[mask] - p[mask]) ** 2).sum() / denom) if denom else float("nan")


class FundamentalPredictorRF:
    """Panel wrapper for Random Forest expected-return prediction."""

    def __init__(
        self,
        feature_cols: list[str],
        target_col: str = "target_ret_1m_fwd",
        train_end: str | None = None,
        pred_col: str = "pred_ret_1m_fwd",
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.train_end = train_end
        self.pred_col = pred_col
        self.random_state = random_state
        self.kwargs = kwargs
        self.model_ = ExpectedReturnModel(model="rf", random_state=random_state, **kwargs)
        self.feature_cols_: list[str] = []

    def fit(self, df: pd.DataFrame) -> "FundamentalPredictorRF":
        """Fit the predictor on the training slice of a panel."""
        train = df.copy()
        if self.train_end is not None and "date" in train.columns:
            train = train[pd.to_datetime(train["date"], errors="coerce") <= pd.Timestamp(self.train_end)]
        self.feature_cols_ = [col for col in self.feature_cols if col in train.columns]
        if not self.feature_cols_:
            raise ValueError("No requested expected-return features are present in the panel.")
        if self.target_col not in train.columns:
            raise ValueError(f"Target column not found: {self.target_col}")
        y = pd.to_numeric(train[self.target_col], errors="coerce")
        mask = y.notna()
        self.model_.fit(train.loc[mask, self.feature_cols_], y.loc[mask])
        return self

    def predict_panel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append expected-return predictions to the panel."""
        if not self.feature_cols_:
            raise RuntimeError("Call fit before predict_panel.")
        out = df.copy()
        out[self.pred_col] = self.model_.predict(out.reindex(columns=self.feature_cols_))
        return out

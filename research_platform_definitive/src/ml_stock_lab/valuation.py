"""Fair-value estimation models with sklearn-like APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _make_model(model: str, random_state: int = 42, **kwargs: Any):
    try:
        if model == "lasso":
            from sklearn.linear_model import Lasso
            return Lasso(alpha=float(kwargs.get("alpha", 0.001)), max_iter=10000, random_state=random_state)
        if model == "rf":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=int(kwargs.get("n_estimators", 100)), min_samples_leaf=2, random_state=random_state, n_jobs=-1)
        if model in {"gbrt", "gbm"}:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(random_state=random_state)
        if model == "ensemble":
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
            from sklearn.linear_model import Lasso, LinearRegression
            return VotingRegressor([
                ("ols", LinearRegression()),
                ("lasso", Lasso(alpha=0.001, max_iter=10000, random_state=random_state)),
                ("rf", RandomForestRegressor(n_estimators=80, min_samples_leaf=2, random_state=random_state, n_jobs=-1)),
                ("gbrt", GradientBoostingRegressor(random_state=random_state)),
            ])
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    except Exception:
        return None


@dataclass
class PeerImpliedValuator:
    """Estimate fair value: V_hat = f_t(X)."""

    model: str = "ols"
    random_state: int = 42
    kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.estimator_ = _make_model(self.model, self.random_state, **(self.kwargs or {}))
        self.feature_names_: list[str] = []
        self.coef_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PeerImpliedValuator":
        X = pd.DataFrame(X).astype(float).replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
        y = pd.to_numeric(y, errors="coerce")
        mask = y.notna()
        self.feature_names_ = list(X.columns)
        if self.estimator_ is not None:
            self.estimator_.fit(X.loc[mask], y.loc[mask])
            self.coef_ = getattr(self.estimator_, "coef_", None)
        else:
            x = np.c_[np.ones(len(X.loc[mask])), X.loc[mask].to_numpy()]
            beta = np.linalg.pinv(x.T @ x) @ x.T @ y.loc[mask].to_numpy()
            self.coef_ = beta
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X = pd.DataFrame(X).reindex(columns=self.feature_names_).astype(float).replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
        if self.estimator_ is not None:
            values = self.estimator_.predict(X)
        else:
            beta = self.coef_
            values = np.c_[np.ones(len(X)), X.to_numpy()] @ beta
        return pd.Series(values, index=X.index, name="fair_value_hat")

    def fit_predict(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        return self.fit(X, y).predict(X)


class RollingCrossSectionalValuator:
    """Date-aware fair value estimator; fits per date when enough data exist."""

    def __init__(self, model: str = "ols", min_obs: int = 20, random_state: int = 42) -> None:
        self.model = model
        self.min_obs = min_obs
        self.random_state = random_state

    def fit_predict(self, panel: pd.DataFrame, feature_cols: list[str], target_col: str = "market_value") -> pd.Series:
        if "date" not in panel.columns:
            X = panel[feature_cols]
            y = panel[target_col]
            return PeerImpliedValuator(self.model, self.random_state).fit_predict(X, y)
        preds = pd.Series(index=panel.index, dtype=float, name="fair_value_hat")
        for _, idx in panel.groupby("date").groups.items():
            idx = list(idx)
            if len(idx) < max(3, min(self.min_obs, len(idx))):
                continue
            X = panel.loc[idx, feature_cols]
            y = panel.loc[idx, target_col]
            try:
                preds.loc[idx] = PeerImpliedValuator(self.model, self.random_state).fit_predict(X, y)
            except Exception:
                continue
        if preds.isna().all() and feature_cols and target_col in panel:
            preds = PeerImpliedValuator(self.model, self.random_state).fit_predict(panel[feature_cols], panel[target_col])
        return preds

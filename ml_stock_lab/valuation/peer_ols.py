"""Cross-sectional fair-value models with a sklearn-like API."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _make_estimator(model: str = "ols", random_state: int = 42, **kwargs: Any):
    """Create an optional sklearn estimator, returning `None` for numpy OLS."""
    try:
        if model == "lasso":
            from sklearn.linear_model import Lasso

            return Lasso(alpha=float(kwargs.get("alpha", 0.001)), max_iter=10000, random_state=random_state)
        if model == "rf":
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(
                n_estimators=int(kwargs.get("n_estimators", 100)),
                max_depth=kwargs.get("max_depth", None),
                min_samples_leaf=int(kwargs.get("min_samples_leaf", 2)),
                random_state=random_state,
                n_jobs=-1,
            )
        if model in {"gbrt", "gbm"}:
            from sklearn.ensemble import GradientBoostingRegressor

            return GradientBoostingRegressor(
                n_estimators=int(kwargs.get("n_estimators", 100)),
                max_depth=kwargs.get("max_depth", 3),
                min_samples_leaf=int(kwargs.get("min_samples_leaf", 1)),
                learning_rate=float(kwargs.get("learning_rate", 0.05)),
                random_state=random_state,
            )
        if model == "ensemble":
            from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
            from sklearn.linear_model import Lasso, LinearRegression

            return VotingRegressor([
                ("ols", LinearRegression()),
                ("lasso", Lasso(alpha=0.001, max_iter=10000, random_state=random_state)),
                ("rf", RandomForestRegressor(n_estimators=80, min_samples_leaf=2, random_state=random_state, n_jobs=-1)),
                ("gbrt", GradientBoostingRegressor(random_state=random_state)),
            ])
        if model == "ols":
            from sklearn.linear_model import LinearRegression

            return LinearRegression()
    except Exception:
        return None
    return None


class PeerOLSValuator:
    """Estimate peer-implied fair value from cross-sectional features.

    Parameters are set at construction time, while `fit`, `predict`, and
    `fit_predict` follow the familiar sklearn estimator pattern.
    """

    def __init__(self, model: str = "ols", random_state: int = 42, **kwargs: Any) -> None:
        self.model = model
        self.random_state = random_state
        self.kwargs = kwargs
        self.estimator_ = _make_estimator(model, random_state=random_state, **kwargs)
        self.feature_names_: list[str] = []
        self.beta_: np.ndarray | None = None

    def _prepare_X(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).astype(float).replace([np.inf, -np.inf], np.nan)
        return frame.fillna(frame.median(numeric_only=True)).fillna(0.0)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PeerOLSValuator":
        """Fit the fair-value model on numeric features and target values."""
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
        """Predict fair value for rows in `X`."""
        Xp = self._prepare_X(pd.DataFrame(X).reindex(columns=self.feature_names_))
        if self.estimator_ is not None:
            values = self.estimator_.predict(Xp)
        else:
            beta = self.beta_ if self.beta_ is not None else np.zeros(Xp.shape[1] + 1)
            values = np.c_[np.ones(len(Xp)), Xp.to_numpy()] @ beta
        return pd.Series(values, index=Xp.index, name="fair_value_hat")

    def fit_predict(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Fit the model and return in-sample fair-value estimates."""
        return self.fit(X, y).predict(X)


PeerImpliedValuator = PeerOLSValuator


class _PeerImpliedPanelValuator:
    """Panel-oriented fair-value wrapper used by notebook experiments."""

    model_name = "ols"

    def __init__(
        self,
        feature_cols: list[str],
        target_col: str = "target_log_mcap",
        market_value_col: str = "mkt_market_cap",
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.market_value_col = market_value_col
        self.random_state = random_state
        self.kwargs = kwargs
        self.estimator_: PeerOLSValuator | None = None

    def _usable_feature_cols(self, df: pd.DataFrame) -> list[str]:
        return [col for col in self.feature_cols if col in df.columns]

    def fit(self, df: pd.DataFrame, train_end: str | None = None) -> "_PeerImpliedPanelValuator":
        """Fit on rows up to `train_end` if a date column is present."""
        train = df.copy()
        if train_end is not None and "date" in train.columns:
            train = train[pd.to_datetime(train["date"], errors="coerce") <= pd.Timestamp(train_end)]
        feature_cols = self._usable_feature_cols(train)
        if not feature_cols:
            raise ValueError("No requested fair-value features are present in the panel.")
        if self.target_col not in train.columns:
            raise ValueError(f"Target column not found: {self.target_col}")

        y = pd.to_numeric(train[self.target_col], errors="coerce")
        X = train[feature_cols]
        mask = y.notna()
        self.estimator_ = PeerOLSValuator(self.model_name, random_state=self.random_state, **self.kwargs)
        self.estimator_.fit(X.loc[mask], y.loc[mask])
        self.feature_cols_ = feature_cols
        return self

    def predict_panel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the input panel with fair-value predictions appended."""
        if self.estimator_ is None:
            raise RuntimeError("Call fit before predict_panel.")
        out = df.copy()
        pred = self.estimator_.predict(out.reindex(columns=self.feature_cols_))
        out["pred_log_mcap"] = pred
        if "log" in self.target_col:
            out["fair_value"] = np.exp(pred)
        else:
            out["fair_value"] = pred
        if self.market_value_col not in out.columns:
            if self.target_col in out.columns and "log" in self.target_col:
                out[self.market_value_col] = np.exp(pd.to_numeric(out[self.target_col], errors="coerce"))
            elif self.target_col in out.columns:
                out[self.market_value_col] = pd.to_numeric(out[self.target_col], errors="coerce")
        if self.market_value_col in out.columns:
            out["valuation_error"] = out["fair_value"] - pd.to_numeric(out[self.market_value_col], errors="coerce")
        return out


class PeerImpliedOLS(_PeerImpliedPanelValuator):
    """OLS peer-implied fair-value model."""

    model_name = "ols"


class PeerImpliedLasso(_PeerImpliedPanelValuator):
    """LASSO peer-implied fair-value model."""

    model_name = "lasso"


class PeerImpliedRF(_PeerImpliedPanelValuator):
    """Random Forest peer-implied fair-value model."""

    model_name = "rf"


class PeerImpliedGBRT(_PeerImpliedPanelValuator):
    """Gradient Boosting peer-implied fair-value model."""

    model_name = "gbrt"

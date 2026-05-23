"""Rolling and expanding wrappers for cross-sectional valuation."""

from __future__ import annotations

import pandas as pd

from .peer_ols import PeerOLSValuator


class RollingPeerValuator:
    """Date-aware wrapper that fits a peer model per date when possible.

    This initial implementation is intentionally simple: it estimates each
    date cross-section independently when enough observations are available,
    and falls back to one pooled fit if all per-date fits fail.
    """

    def __init__(self, model: str = "ols", min_obs: int = 20, random_state: int = 42) -> None:
        self.model = model
        self.min_obs = min_obs
        self.random_state = random_state

    def fit_predict(self, panel: pd.DataFrame, feature_cols: list[str], target_col: str = "market_value") -> pd.Series:
        """Return fair-value estimates indexed like `panel`."""
        if "date" not in panel.columns:
            return PeerOLSValuator(self.model, self.random_state).fit_predict(panel[feature_cols], panel[target_col])

        preds = pd.Series(index=panel.index, dtype=float, name="fair_value_hat")
        for _, idx in panel.groupby("date").groups.items():
            idx = list(idx)
            if len(idx) < max(3, min(self.min_obs, len(idx))):
                continue
            try:
                preds.loc[idx] = PeerOLSValuator(self.model, self.random_state).fit_predict(panel.loc[idx, feature_cols], panel.loc[idx, target_col])
            except Exception:
                continue
        if preds.isna().all() and feature_cols and target_col in panel:
            preds = PeerOLSValuator(self.model, self.random_state).fit_predict(panel[feature_cols], panel[target_col])
        return preds

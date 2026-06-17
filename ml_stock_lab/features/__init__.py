"""Feature engineering namespaces used by Colab research notebooks.

The definitive Streamlit workstation imports the canonical ``ml_stock_lab``
training module from ``research_platform_definitive/src``.  When tests run from
the repository root, this legacy package is imported first, so we expose the
small panel helpers expected by the canonical training code as compatibility
wrappers.
"""

from __future__ import annotations

import pandas as pd

from . import fundamentals, technical
from ml_stock_lab.datasets.panel import (
    add_basic_features as _add_basic_features,
    make_forward_returns as _make_forward_returns,
    select_numeric_features as _select_numeric_features,
)


def add_basic_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper for the canonical training import path."""

    return _add_basic_features(panel)


def make_forward_returns(
    panel: pd.DataFrame,
    price_col: str = "price",
    horizon: int = 21,
) -> pd.DataFrame:
    """Compatibility wrapper for forward-return target generation."""

    return _make_forward_returns(panel, price_col=price_col, horizon=horizon)


def select_numeric_features(
    panel: pd.DataFrame,
    target: str = "market_value",
    min_non_null: int = 5,
    feature_blocks: list[str] | tuple[str, ...] | None = None,
    include_extra_numeric: bool = False,
) -> list[str]:
    """Return numeric feature names while accepting canonical keyword args."""

    return _select_numeric_features(panel, target=target, min_non_null=min_non_null)


__all__ = [
    "fundamentals",
    "technical",
    "add_basic_features",
    "make_forward_returns",
    "select_numeric_features",
]

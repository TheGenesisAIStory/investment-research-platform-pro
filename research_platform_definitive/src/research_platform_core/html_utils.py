"""Small HTML helpers shared by notebook dashboard hooks."""

from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd

from .data_utils import as_df


def table_html(df: Any, title: str, max_rows: int = 50, css_class: str = "data-table") -> str:
    """Render a defensive HTML table block used by static dashboard exports."""
    frame = as_df(df)
    if frame.empty:
        return f"<section class='card'><h2>{escape(title)}</h2><p class='muted'>No data available.</p></section>"
    return (
        f"<section class='card'><h2>{escape(title)}</h2>"
        + frame.head(max_rows).to_html(index=False, classes=css_class, border=0, escape=True)
        + "</section>"
    )


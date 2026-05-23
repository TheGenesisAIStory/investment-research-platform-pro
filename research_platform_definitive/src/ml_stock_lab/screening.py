"""Ranking, screening and quantile assignment."""

from __future__ import annotations

import pandas as pd


def rank_scores(df: pd.DataFrame, score_col: str, ascending: bool = False, group_col: str | None = None) -> pd.DataFrame:
    out = df.copy()
    if score_col not in out.columns:
        out["rank"] = pd.NA
        return out
    if group_col and group_col in out.columns:
        out["rank"] = out.groupby(group_col)[score_col].rank(ascending=ascending, method="first")
        out["percentile_rank"] = out.groupby(group_col)[score_col].rank(ascending=ascending, pct=True)
    else:
        out["rank"] = out[score_col].rank(ascending=ascending, method="first")
        out["percentile_rank"] = out[score_col].rank(ascending=ascending, pct=True)
    return out.sort_values("rank")


def assign_quantiles(df: pd.DataFrame, score_col: str, q: int = 5, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    if score_col not in out.columns:
        out["quantile"] = pd.NA
        return out

    def qcut_safe(s: pd.Series) -> pd.Series:
        ranks = s.rank(method="first")
        try:
            return pd.qcut(ranks, q=q, labels=False, duplicates="drop") + 1
        except Exception:
            return pd.Series(pd.NA, index=s.index)

    if date_col in out.columns:
        out["quantile"] = out.groupby(date_col)[score_col].transform(qcut_safe)
    else:
        out["quantile"] = qcut_safe(out[score_col])
    return out


def top_bottom(df: pd.DataFrame, score_col: str, n: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = rank_scores(df, score_col, ascending=False)
    return ranked.head(n), ranked.tail(n).sort_values(score_col, ascending=True)


class StockScreener:
    """Minimal sklearn-like screener over score columns."""

    def __init__(self, score_col: str = "zscore", top_n: int = 25) -> None:
        self.score_col = score_col
        self.top_n = top_n

    def fit(self, df: pd.DataFrame) -> "StockScreener":
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return rank_scores(df, self.score_col, ascending=False).head(self.top_n)

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).predict(df)

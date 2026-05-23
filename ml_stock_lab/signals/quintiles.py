"""Ranking and quintile portfolio construction."""

from __future__ import annotations

import pandas as pd


def rank_scores(df: pd.DataFrame, score_col: str, ascending: bool = False, group_col: str | None = None) -> pd.DataFrame:
    """Add rank and percentile-rank columns for a score."""
    out = df.copy()
    if score_col not in out.columns:
        out["rank"] = pd.NA
        out["percentile_rank"] = pd.NA
        return out
    if group_col and group_col in out.columns:
        out["rank"] = out.groupby(group_col)[score_col].rank(ascending=ascending, method="first")
        out["percentile_rank"] = out.groupby(group_col)[score_col].rank(ascending=ascending, pct=True)
    else:
        out["rank"] = out[score_col].rank(ascending=ascending, method="first")
        out["percentile_rank"] = out[score_col].rank(ascending=ascending, pct=True)
    return out.sort_values("rank")


def assign_quantiles(df: pd.DataFrame, score_col: str, q: int = 5, date_col: str = "date") -> pd.DataFrame:
    """Assign quantile labels from 1 to `q`, optionally per date."""
    out = df.copy()
    if score_col not in out.columns:
        out["quantile"] = pd.NA
        return out

    def qcut_safe(s: pd.Series) -> pd.Series:
        ranks = pd.to_numeric(s, errors="coerce").rank(method="first")
        try:
            return pd.qcut(ranks, q=q, labels=False, duplicates="drop") + 1
        except Exception:
            return pd.Series(pd.NA, index=s.index)

    out["quantile"] = out.groupby(date_col)[score_col].transform(qcut_safe) if date_col in out.columns else qcut_safe(out[score_col])
    return out


def top_bottom(df: pd.DataFrame, score_col: str, n: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return top and bottom `n` rows by score."""
    ranked = rank_scores(df, score_col, ascending=False)
    return ranked.head(n), ranked.tail(n).sort_values(score_col, ascending=True)


def make_quantile_portfolios(panel: pd.DataFrame, signal_col: str, return_col: str = "forward_return", q: int = 5) -> pd.DataFrame:
    """Return long-form date/quantile returns plus a Qhigh-Qlow long-short leg."""
    if panel is None or panel.empty or signal_col not in panel.columns or return_col not in panel.columns:
        return pd.DataFrame(columns=["date", "quantile", "return", "name_count"])

    out = assign_quantiles(panel, signal_col, q=q)
    out[return_col] = pd.to_numeric(out[return_col], errors="coerce")
    if "date" in out.columns:
        qret = (
            out.groupby(["date", "quantile"], dropna=True)[return_col]
            .agg(**{"return": "mean", "name_count": "count"})
            .reset_index()
        )
        if qret.empty:
            return pd.DataFrame(columns=["date", "quantile", "return", "name_count"])
        ret_wide = qret.pivot(index="date", columns="quantile", values="return").sort_index()
        count_wide = qret.pivot(index="date", columns="quantile", values="name_count").sort_index()
        label_map = {str(c).replace(".0", ""): c for c in ret_wide.columns}
        low_col = label_map.get("1")
        high_col = label_map.get(str(q))
        if low_col is not None and high_col is not None:
            ret_wide["LS"] = ret_wide[high_col] - ret_wide[low_col]
            count_wide["LS"] = count_wide[high_col].fillna(0) + count_wide[low_col].fillna(0)
        returns = ret_wide.reset_index().melt(id_vars="date", var_name="quantile", value_name="return")
        counts = count_wide.reset_index().melt(id_vars="date", var_name="quantile", value_name="name_count")
        return returns.merge(counts, on=["date", "quantile"], how="left")

    qret = (
        out.groupby("quantile", dropna=True)[return_col]
        .agg(**{"return": "mean", "name_count": "count"})
        .reset_index()
    )
    qret["date"] = pd.Timestamp.today().normalize()
    return qret[["date", "quantile", "return", "name_count"]]


def quintile_returns_wide(quintile_returns: pd.DataFrame) -> pd.DataFrame:
    """Convert long-form quintile returns to a wide table with Q1..Q5 and LS."""
    if quintile_returns is None or quintile_returns.empty or not {"date", "quantile", "return"}.issubset(quintile_returns.columns):
        return pd.DataFrame()
    wide = quintile_returns.pivot_table(index="date", columns="quantile", values="return", aggfunc="mean").sort_index()
    wide.columns = [f"Q{int(c)}" if str(c).replace(".0", "").isdigit() else str(c) for c in wide.columns]
    return wide

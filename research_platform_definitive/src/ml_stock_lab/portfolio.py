"""Portfolio construction from ML scores."""

from __future__ import annotations

import pandas as pd

from .screening import assign_quantiles


def make_quantile_portfolios(panel: pd.DataFrame, signal_col: str, return_col: str = "forward_return", q: int = 5) -> pd.DataFrame:
    """Return date/quantile portfolio returns and long-short Qhigh-Qlow."""
    if panel.empty or signal_col not in panel.columns or return_col not in panel.columns:
        return pd.DataFrame(columns=["date", "quantile", "return", "name_count"])
    out = assign_quantiles(panel, signal_col, q=q)
    ret = pd.to_numeric(out[return_col], errors="coerce")
    out[return_col] = ret
    date_col = "date" if "date" in out.columns else None
    if date_col:
        qret = (
            out.groupby([date_col, "quantile"], dropna=True)[return_col]
            .agg(**{"return": "mean", "name_count": "count"})
            .reset_index()
        )
        if qret.empty:
            return pd.DataFrame(columns=["date", "quantile", "return", "name_count"])
        ret_pivot = qret.pivot(index=date_col, columns="quantile", values="return").sort_index()
        count_pivot = qret.pivot(index=date_col, columns="quantile", values="name_count").sort_index()
        col_map = {str(c).replace(".0", ""): c for c in ret_pivot.columns}
        low_col = col_map.get("1")
        high_col = col_map.get(str(q))
        if low_col is not None and high_col is not None:
            ret_pivot["long_short"] = ret_pivot[high_col] - ret_pivot[low_col]
            count_pivot["long_short"] = count_pivot[high_col].fillna(0) + count_pivot[low_col].fillna(0)
        returns = ret_pivot.reset_index().melt(id_vars=date_col, var_name="quantile", value_name="return")
        counts = count_pivot.reset_index().melt(id_vars=date_col, var_name="quantile", value_name="name_count")
        return returns.merge(counts, on=[date_col, "quantile"], how="left").rename(columns={date_col: "date"})
    qret = (
        out.groupby("quantile", dropna=True)[return_col]
        .agg(**{"return": "mean", "name_count": "count"})
        .reset_index()
    )
    col_map = {str(c).replace(".0", ""): c for c in qret["quantile"].tolist()}
    low_col = col_map.get("1")
    high_col = col_map.get(str(q))
    if low_col is not None and high_col is not None:
        low = qret[qret["quantile"].eq(low_col)].iloc[0]
        high = qret[qret["quantile"].eq(high_col)].iloc[0]
        qret = pd.concat([
            qret,
            pd.DataFrame([{
                "quantile": "long_short",
                "return": high["return"] - low["return"],
                "name_count": high["name_count"] + low["name_count"],
            }]),
        ], ignore_index=True)
    qret["date"] = pd.Timestamp.today().normalize()
    return qret[["date", "quantile", "return", "name_count"]]


def long_short_weights(panel: pd.DataFrame, signal_col: str, q: int = 5) -> pd.DataFrame:
    out = assign_quantiles(panel, signal_col, q=q)
    out["weight"] = 0.0
    out.loc[out["quantile"].eq(q), "weight"] = 1.0 / max(1, int(out["quantile"].eq(q).sum()))
    out.loc[out["quantile"].eq(1), "weight"] = -1.0 / max(1, int(out["quantile"].eq(1).sum()))
    return out


def mean_variance_from_forecasts(panel: pd.DataFrame, forecast_col: str = "expected_return", risk_col: str | None = None) -> pd.DataFrame:
    out = panel.copy()
    if forecast_col not in out.columns:
        out["weight"] = 0.0
        return out
    score = pd.to_numeric(out[forecast_col], errors="coerce").clip(lower=0)
    if risk_col and risk_col in out.columns:
        risk = pd.to_numeric(out[risk_col], errors="coerce").replace(0, pd.NA)
        score = score / risk
    total = score.sum()
    out["weight"] = score / total if total and pd.notna(total) else 0.0
    return out

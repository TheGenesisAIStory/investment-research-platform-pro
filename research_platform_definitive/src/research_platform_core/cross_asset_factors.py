"""Cross-asset factors from Macro DB and factor signal inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


CORE_CROSS_ASSET_SYMBOLS: tuple[str, ...] = ("SPY", "QQQ", "DX-Y.NYB", "TLT", "HYG", "LQD", "BZ=F", "GC=F", "HG=F", "BTC-USD")
RATES_KEYS: tuple[str, ...] = ("fi", "fixed_income", "rates", "bond", "bonds")


def sanitize_symbol(symbol: object) -> str:
    text = str(symbol or "").upper().replace("=X", "").replace("^", "")
    replacements = {"DX-Y.NYB": "DXY", "BZ=F": "BRENT", "GC=F": "GOLD", "HG=F": "COPPER", "BTC-USD": "BTC"}
    text = replacements.get(text, text)
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def _wide_from_macro_history(macro_history_df: pd.DataFrame) -> pd.DataFrame:
    if macro_history_df is None or macro_history_df.empty or "date" not in macro_history_df.columns:
        return pd.DataFrame()
    frame = macro_history_df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    value_col = "close" if "close" in frame.columns else "last_close" if "last_close" in frame.columns else ""
    if "symbol" in frame.columns and value_col:
        frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
        frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
        return frame.pivot_table(index="date", columns="symbol", values=value_col, aggfunc="last").sort_index()
    numeric = frame.select_dtypes("number").copy()
    numeric.index = frame["date"]
    return numeric.sort_index()


def _numeric_signal_frame(frame: pd.DataFrame | pd.Series, prefix: str) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    if isinstance(frame, pd.Series):
        out = frame.to_frame(name=frame.name or prefix)
    else:
        out = frame.copy()
    if "date" in out.columns:
        out = out.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.set_index("date")
    out = out.apply(pd.to_numeric, errors="coerce").sort_index()
    out = out.loc[:, ~out.columns.duplicated()]
    out.columns = [f"{prefix}_{sanitize_symbol(col)}" for col in out.columns]
    return out


def _numeric_panel(frame: pd.DataFrame | pd.Series | None) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    if isinstance(frame, pd.Series):
        out = frame.to_frame(name=frame.name or "value")
    else:
        out = frame.copy()
    if "date" in out.columns:
        out = out.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.set_index("date")
    out = out.apply(pd.to_numeric, errors="coerce").sort_index()
    out = out.loc[:, ~out.columns.duplicated()]
    out.columns = [sanitize_symbol(col) for col in out.columns]
    return out


def _rolling_zscore(frame: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    mean = frame.rolling(window, min_periods=max(12, min(window, 20))).mean()
    std = frame.rolling(window, min_periods=max(12, min(window, 20))).std().replace(0, np.nan)
    return (frame - mean) / std


def _cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1, skipna=True)
    std = frame.std(axis=1, skipna=True).replace(0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0)


def _roll_bid_ask_spread_proxy(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    values = pd.DataFrame(index=returns.index)
    min_periods = max(5, min(int(window), 10))
    for col in returns.columns:
        series = pd.to_numeric(returns[col], errors="coerce")
        cov = series.rolling(int(window), min_periods=min_periods).cov(series.shift(1))
        values[col] = 2.0 * np.sqrt(np.maximum(-cov, 0.0))
    return values


def build_cross_asset_momentum(macro_history_df: pd.DataFrame) -> pd.DataFrame:
    """Build lagged momentum-everywhere features across Macro DB proxies.

    Reference: Asness, Moskowitz and Pedersen (2013); Bartram et al. (2021).
    """
    close = _wide_from_macro_history(macro_history_df).sort_index().ffill()
    if close.empty:
        return pd.DataFrame()
    selected = [col for col in close.columns if str(col).upper() in {s.upper() for s in CORE_CROSS_ASSET_SYMBOLS}]
    if not selected:
        selected = list(close.columns[:20])
    out = pd.DataFrame(index=close.index)
    for col in selected:
        sid = sanitize_symbol(col)
        series = pd.to_numeric(close[col], errors="coerce").shift(1)
        mom = series.shift(21) / series.shift(252) - 1
        vol = series.pct_change().rolling(63, min_periods=20).std() * np.sqrt(252)
        ma50 = series.rolling(50, min_periods=20).mean()
        ma200 = series.rolling(200, min_periods=60).mean()
        out[f"xasset_mom_{sid}"] = mom
        out[f"xasset_trend_{sid}"] = np.sign(ma50 - ma200)
        out[f"xasset_vol_adj_mom_{sid}"] = mom / vol.replace(0, np.nan)
    out["date"] = out.index
    return out.reset_index(drop=True)


def cross_asset_momentum(macro_history_df: pd.DataFrame) -> pd.DataFrame:
    """Alias for the implemented cross-asset momentum builder."""
    return build_cross_asset_momentum(macro_history_df)


def cross_asset_value(
    value_signals_dict: dict[str, pd.DataFrame | pd.Series],
    *,
    zscore_window: int = 60,
) -> pd.DataFrame:
    """Combine equity, FX, FI and commodity value signals with anti-leakage.

    Each supplied signal matrix is shifted by one observation, normalized by a
    rolling z-score independently, prefixed by asset class, then averaged into
    `xasset_value_score`.  Inputs are expected to be point-in-time value signals
    such as HML, PPP deviation, yield reversion or commodity value proxies.

    Reference: Asness, Moskowitz and Pedersen (2013).
    """
    if not value_signals_dict:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    for asset_class, frame in value_signals_dict.items():
        numeric = _numeric_signal_frame(frame, str(asset_class).lower())
        if numeric.empty:
            continue
        parts.append(_rolling_zscore(numeric.shift(1), window=zscore_window))
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, axis=1).sort_index()
    out = pd.DataFrame(index=combined.index)
    for col in combined.columns:
        out[f"xasset_value_{col}"] = combined[col]
    out["xasset_value_score"] = combined.mean(axis=1, skipna=True)
    out["date"] = out.index
    return out.reset_index(drop=True)


def _stack_asset_returns(asset_ret_dict: dict[str, pd.DataFrame | pd.Series]) -> tuple[pd.DataFrame, list[str], list[str]]:
    parts: list[pd.DataFrame] = []
    equity_cols: list[str] = []
    rates_cols: list[str] = []
    for asset_class, frame in (asset_ret_dict or {}).items():
        key = str(asset_class).lower()
        numeric = _numeric_signal_frame(frame, key)
        if numeric.empty:
            continue
        parts.append(numeric)
        if key == "equity":
            equity_cols.extend(numeric.columns)
        if key in RATES_KEYS:
            rates_cols.extend(numeric.columns)
    if not parts:
        return pd.DataFrame(), equity_cols, rates_cols
    return pd.concat(parts, axis=1).sort_index(), equity_cols, rates_cols


def global_risk_factor(
    asset_ret_dict: dict[str, pd.DataFrame | pd.Series],
    *,
    window: int = 60,
) -> pd.DataFrame:
    """Rolling PCA global risk factor from cross-asset returns.

    The PCA window uses observations available up to `t-1`.  PC1 orientation is
    anchored so equity loadings are positive and rates/fixed-income loadings are
    negative when those groups are available.

    Returns columns: `global_risk_factor` and
    `global_risk_explained_variance`.
    """
    returns, equity_cols, rates_cols = _stack_asset_returns(asset_ret_dict)
    if returns.empty:
        return pd.DataFrame()
    returns = returns.replace([np.inf, -np.inf], np.nan).sort_index()
    shifted = returns.shift(1)
    out = pd.DataFrame(index=returns.index, columns=["global_risk_factor", "global_risk_explained_variance"], dtype=float)
    min_periods = max(12, min(int(window), 24))
    for pos in range(len(shifted)):
        start = max(0, pos - int(window) + 1)
        sample = shifted.iloc[start : pos + 1].dropna(axis=1, how="all").dropna(how="any")
        if len(sample) < min_periods or sample.shape[1] < 2:
            continue
        centered = sample - sample.mean(axis=0)
        cov = np.cov(centered.to_numpy(dtype=float), rowvar=False)
        cov = np.atleast_2d(np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0))
        if cov.shape[0] != sample.shape[1]:
            continue
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvec = eigvecs[:, order[0]]
        cols = list(sample.columns)

        anchor = np.zeros(len(cols), dtype=float)
        for col in equity_cols:
            if col in cols:
                anchor[cols.index(col)] = 1.0
        for col in rates_cols:
            if col in cols:
                anchor[cols.index(col)] = -1.0
        if np.any(anchor):
            if float(np.dot(eigvec, anchor)) < 0:
                eigvec = -eigvec
        elif eigvec.sum() < 0:
            eigvec = -eigvec

        current = shifted.iloc[pos][cols].astype(float)
        out.iloc[pos, out.columns.get_loc("global_risk_factor")] = float(np.dot(current.fillna(0.0).to_numpy(), eigvec))
        total_var = float(np.nansum(np.maximum(eigvals, 0.0)))
        out.iloc[pos, out.columns.get_loc("global_risk_explained_variance")] = float(eigvals[0] / total_var) if total_var > 0 else np.nan
    out["date"] = out.index
    return out.reset_index(drop=True)


def liquidity_factor(
    ret_df: pd.DataFrame | pd.Series,
    volume_df: pd.DataFrame | pd.Series | None = None,
    *,
    window: int = 21,
) -> pd.DataFrame:
    """Cross-asset Amihud illiquidity factor with a spread-proxy fallback.

    Primary signal: rolling mean of `abs(return) / dollar_volume`.
    If dollar-volume data is unavailable for an asset, the function falls back
    to a Roll-style implicit bid-ask spread proxy from return autocovariance.
    The final values are shifted by one observation and transformed into a
    cross-sectional z-score each date.

    Positive values indicate higher illiquidity / stronger trading-friction
    exposure. Reference: Amihud (2002), Journal of Financial Markets.
    """
    returns = _numeric_panel(ret_df).replace([np.inf, -np.inf], np.nan).sort_index()
    if returns.empty:
        return pd.DataFrame()

    volumes = _numeric_panel(volume_df).replace([np.inf, -np.inf], np.nan).sort_index()
    volumes = volumes.reindex(index=returns.index, columns=returns.columns) if not volumes.empty else pd.DataFrame(index=returns.index, columns=returns.columns)
    positive_volume = volumes.where(volumes > 0)

    min_periods = max(5, min(int(window), 10))
    amihud = (returns.abs() / positive_volume).rolling(int(window), min_periods=min_periods).mean()
    spread_proxy = _roll_bid_ask_spread_proxy(returns, int(window))
    raw_illiquidity = amihud.where(amihud.notna(), spread_proxy).shift(1)
    zscore = _cross_sectional_zscore(raw_illiquidity)

    out = pd.DataFrame(index=returns.index)
    for col in zscore.columns:
        out[f"xasset_liquidity_{sanitize_symbol(col)}"] = zscore[col]
    out["xasset_liquidity_score"] = zscore.mean(axis=1, skipna=True)
    out["date"] = out.index
    return out.reset_index(drop=True)


__all__ = [
    "CORE_CROSS_ASSET_SYMBOLS",
    "build_cross_asset_momentum",
    "cross_asset_momentum",
    "cross_asset_value",
    "global_risk_factor",
    "liquidity_factor",
    "sanitize_symbol",
]

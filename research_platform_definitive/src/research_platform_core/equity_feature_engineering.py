"""Advanced equity feature engineering for optional factor blocks.

All features are computed from current or lagged observations only. The module
does not mutate existing factor panels destructively: callers opt in by applying
the returned columns to a copy of the panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _num(frame: pd.DataFrame, column: str, default: float | pd.Series = np.nan) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    if isinstance(default, pd.Series):
        return pd.to_numeric(default.reindex(frame.index), errors="coerce")
    return pd.Series(default, index=frame.index, dtype=float)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num.astype(float) / den.replace(0, np.nan).astype(float)


def _rolling_downside_std(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window, min_periods=max(5, window // 4)).apply(
        lambda values: pd.Series(values)[pd.Series(values) < 0].std(ddof=1),
        raw=False,
    ) * np.sqrt(252)


def compute_advanced_technical_features(price_frame: pd.DataFrame) -> pd.DataFrame:
    """Compute advanced price/volume features for one ticker price frame."""
    if price_frame.empty:
        return price_frame.copy()
    out = price_frame.copy().sort_values("date") if "date" in price_frame.columns else price_frame.copy()
    close = _num(out, "price", np.nan)
    if close.isna().all():
        close = _num(out, "close", _num(out, "adjclose", np.nan))
    high = _num(out, "high", close)
    low = _num(out, "low", close)
    volume = _num(out, "volume", np.nan)
    returns = close.pct_change()

    out["ret_1d"] = returns
    for days in [5, 21, 63, 126, 252]:
        out[f"ret_{days}d"] = close.pct_change(days)
        if f"ret{days}d" not in out.columns:
            out[f"ret{days}d"] = out[f"ret_{days}d"]
    high_52w = close.rolling(252, min_periods=20).max()
    low_52w = close.rolling(252, min_periods=20).min()
    out["ret_52w_high_proximity"] = _safe_div(close - high_52w, high_52w)
    out["ret_52w_low_proximity"] = _safe_div(close - low_52w, low_52w)
    out["price_to_52w_high"] = _safe_div(close, high_52w)
    out["momentum_reversal_1m"] = -out["ret_21d"]
    out["momentum_12m_1m"] = out["ret_252d"] - out["ret_21d"]
    out["vol_21d"] = returns.rolling(21, min_periods=5).std() * np.sqrt(252)
    out["vol_63d"] = returns.rolling(63, min_periods=15).std() * np.sqrt(252)
    out["vol_126d"] = returns.rolling(126, min_periods=30).std() * np.sqrt(252)
    out["vol_252d"] = returns.rolling(252, min_periods=60).std() * np.sqrt(252)
    out["realized_vol_daily"] = out["vol_21d"]
    out["vol_ratio"] = _safe_div(out["vol_21d"], out["vol_252d"])
    out["downside_vol_21d"] = _rolling_downside_std(returns, 21)
    out["max_drawdown_1y"] = close.rolling(252, min_periods=20).apply(
        lambda values: float(pd.Series(values).div(pd.Series(values).cummax()).sub(1).min()),
        raw=False,
    )
    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    out["avg_true_range_21d"] = _safe_div(tr.rolling(21, min_periods=5).mean(), close)
    out["avg_volume_21d"] = volume.rolling(21, min_periods=5).mean()
    out["avg_volume_252d"] = volume.rolling(252, min_periods=20).mean()
    out["volume_ratio"] = _safe_div(out["avg_volume_21d"], out["avg_volume_252d"])
    volume_usd = volume * close
    out["amihud_illiquidity"] = (_safe_div(returns.abs(), volume_usd) * 1_000_000).rolling(21, min_periods=5).mean()
    shares = _num(out, "shares_outstanding", np.nan)
    out["turnover_ratio_21d"] = _safe_div(volume, shares).rolling(21, min_periods=5).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=5).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean()
    rs = _safe_div(gain, loss)
    out["rsi_14"] = 100 - (100 / (1 + rs))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    out["macd_signal"] = macd - macd.ewm(span=9, adjust=False).mean()
    sma20 = close.rolling(20, min_periods=10).mean()
    std20 = close.rolling(20, min_periods=10).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    out["bb_position"] = _safe_div(close - bb_lower, bb_upper - bb_lower)
    sma50 = close.rolling(50, min_periods=20).mean()
    sma200 = close.rolling(200, min_periods=60).mean()
    out["price_to_sma_50"] = _safe_div(close, sma50)
    out["price_to_sma_200"] = _safe_div(close, sma200)
    out["golden_cross"] = (sma50 > sma200).astype(float)
    return out


def add_idiosyncratic_features(panel: pd.DataFrame, benchmark_returns: pd.Series | None = None) -> pd.DataFrame:
    """Add rolling idiosyncratic vol/momentum versus a benchmark return series."""
    if panel.empty:
        return panel.copy()
    out = panel.copy()
    returns = _num(out, "ret_1d", _num(out, "ret1d", np.nan))
    if benchmark_returns is None or benchmark_returns.empty:
        out["idiosyncratic_vol"] = np.nan
        out["idiosyncratic_momentum"] = np.nan
        return out
    benchmark = pd.to_numeric(benchmark_returns.reindex(out.index), errors="coerce")

    def residual_std(values: np.ndarray) -> float:
        y = values[:, 0]
        x = values[:, 1]
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 30 or np.nanvar(x[mask]) == 0:
            return np.nan
        beta = np.cov(y[mask], x[mask])[0, 1] / np.var(x[mask])
        residual = y[mask] - beta * x[mask]
        return float(np.std(residual, ddof=1) * np.sqrt(252))

    joint = pd.concat([returns, benchmark], axis=1)
    vals = []
    for idx in range(len(joint)):
        window = joint.iloc[max(0, idx - 251): idx + 1].to_numpy(dtype=float)
        vals.append(residual_std(window) if len(window) >= 60 else np.nan)
    out["idiosyncratic_vol"] = vals
    out["idiosyncratic_momentum"] = out.get("momentum_12m_1m", out.get("momentum_12_1", pd.Series(np.nan, index=out.index))) - _safe_div(benchmark.rolling(252).sum(), pd.Series(252, index=out.index))
    return out


def compute_advanced_fundamental_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute advanced quality/value/growth/size features from fundamentals."""
    if df.empty:
        return df.copy()
    out = df.copy()
    revenue = _num(out, "revenue_ttm", _num(out, "revenue", np.nan))
    net_income = _num(out, "net_income", np.nan)
    assets = _num(out, "total_assets", np.nan)
    equity = _num(out, "total_equity", np.nan)
    debt = _num(out, "total_debt", np.nan)
    cash = _num(out, "cash", np.nan)
    ebit = _num(out, "ebit", _num(out, "ebit_ttm", np.nan))
    ebitda = _num(out, "ebitda", _num(out, "ebitda_ttm", np.nan))
    gross_profit = _num(out, "gross_profit", np.nan)
    operating_income = _num(out, "operating_income", np.nan)
    fcf = _num(out, "free_cash_flow", np.nan)
    ocf = _num(out, "operating_cash_flow", np.nan)
    capex = abs(_num(out, "capex", _num(out, "capital_expenditure", np.nan)))
    rd = _num(out, "research_development", _num(out, "rd_expense", np.nan))
    current_assets = _num(out, "current_assets", np.nan)
    current_liabilities = _num(out, "current_liabilities", np.nan)
    inventory = _num(out, "inventory", 0.0)
    interest = abs(_num(out, "interest_expense", np.nan))
    market_cap = _num(out, "market_cap", _num(out, "market_value", np.nan))
    price = _num(out, "price", np.nan)
    shares = _num(out, "shares_outstanding", np.nan)
    eps = _num(out, "eps_ttm", _num(out, "eps", _safe_div(net_income, shares)))
    book_value_per_share = _safe_div(equity, shares)
    tax_rate = _num(out, "tax_rate", 0.21).clip(lower=0.0, upper=0.5)
    nopat = ebit * (1 - tax_rate)
    invested_capital = equity + debt - cash

    out["roe"] = out.get("roe", _safe_div(net_income, equity))
    out["roa"] = out.get("roa", _safe_div(net_income, assets))
    out["roic"] = out.get("roic", _safe_div(nopat, invested_capital))
    out["gross_margin"] = out.get("gross_margin", _safe_div(gross_profit, revenue))
    out["operating_margin"] = out.get("operating_margin", _safe_div(operating_income, revenue))
    out["net_margin"] = out.get("net_margin", _safe_div(net_income, revenue))
    out["ebitda_margin"] = out.get("ebitda_margin", _safe_div(ebitda, revenue))
    out["fcf_margin"] = out.get("fcf_margin", _safe_div(fcf, revenue))
    out["asset_turnover"] = out.get("asset_turnover", _safe_div(revenue, assets))
    out["current_ratio"] = out.get("current_ratio", _safe_div(current_assets, current_liabilities))
    out["quick_ratio"] = out.get("quick_ratio", _safe_div(current_assets - inventory, current_liabilities))
    out["cash_ratio"] = out.get("cash_ratio", _safe_div(cash, current_liabilities))
    out["debt_to_equity"] = out.get("debt_to_equity", _safe_div(debt, equity))
    out["debt_to_ebitda"] = out.get("debt_to_ebitda", _safe_div(debt, ebitda))
    out["net_debt_to_ebitda"] = out.get("net_debt_to_ebitda", _safe_div(debt - cash, ebitda))
    out["interest_coverage"] = out.get("interest_coverage", _safe_div(ebit, interest))
    working_capital = current_assets - current_liabilities
    retained_earnings = _num(out, "retained_earnings", np.nan)
    liabilities = _num(out, "total_liabilities", np.nan)
    out["altman_z_score"] = (
        6.56 * _safe_div(working_capital, assets)
        + 3.26 * _safe_div(retained_earnings, assets)
        + 6.72 * _safe_div(ebit, assets)
        + 1.05 * _safe_div(equity, liabilities)
    )
    out["book_to_market"] = _safe_div(equity, market_cap)
    out["earnings_yield"] = _safe_div(net_income, market_cap)
    out["fcf_yield"] = _safe_div(fcf, market_cap)
    enterprise_value = market_cap + debt - cash
    out["ebitda_yield"] = _safe_div(ebitda, enterprise_value)
    out["sales_to_price"] = _safe_div(revenue, market_cap)
    out["pe_ratio"] = out.get("pe_ratio", _safe_div(price, eps))
    out["pb_ratio"] = out.get("pb_ratio", _safe_div(price, book_value_per_share))
    out["ps_ratio"] = out.get("ps_ratio", _safe_div(market_cap, revenue))
    out["pcf_ratio"] = out.get("pcf_ratio", _safe_div(market_cap, ocf))
    out["ev_ebitda"] = out.get("ev_ebitda", _safe_div(enterprise_value, ebitda))
    out["ev_ebit"] = out.get("ev_ebit", _safe_div(enterprise_value, ebit))
    out["ev_sales"] = out.get("ev_sales", _safe_div(enterprise_value, revenue))
    out["ev_fcf"] = out.get("ev_fcf", _safe_div(enterprise_value, fcf))
    earnings_growth_rate = _num(out, "earnings_growth_rate_5y", _num(out, "earnings_growth", np.nan))
    out["peg_ratio"] = out.get("peg_ratio", _safe_div(out["pe_ratio"], earnings_growth_rate.where(earnings_growth_rate.abs() > 1, earnings_growth_rate * 100)))
    if "ticker" in out.columns:
        group = out["ticker"].astype(str)
        out["revenue_growth_1y"] = revenue.groupby(group).pct_change(252)
        out["revenue_growth_3y"] = (revenue.groupby(group).pct_change(756) + 1).pow(1 / 3) - 1
        out["earnings_growth_1y"] = eps.groupby(group).pct_change(252)
        out["earnings_growth_3y"] = (eps.groupby(group).pct_change(756) + 1).pow(1 / 3) - 1
        out["fcf_growth_1y"] = fcf.groupby(group).pct_change(252)
        out["asset_growth"] = assets.groupby(group).pct_change(252)
    else:
        out["revenue_growth_1y"] = revenue.pct_change()
        out["revenue_growth_3y"] = (revenue.pct_change(3) + 1).pow(1 / 3) - 1
        out["earnings_growth_1y"] = eps.pct_change()
        out["earnings_growth_3y"] = (eps.pct_change(3) + 1).pow(1 / 3) - 1
        out["fcf_growth_1y"] = fcf.pct_change()
        out["asset_growth"] = assets.pct_change()
    out["capex_intensity"] = _safe_div(capex, revenue)
    out["rd_intensity"] = _safe_div(rd, revenue)
    out["accruals_ratio"] = _safe_div(net_income - ocf, assets)
    out["log_market_cap"] = np.log(market_cap.where(market_cap > 0))
    out["log_total_assets"] = np.log(assets.where(assets > 0))
    out["log_revenue"] = np.log(revenue.where(revenue > 0))
    out["piotroski_f_score"] = compute_piotroski_f_score(out)
    return out


def compute_piotroski_f_score(df: pd.DataFrame) -> pd.Series:
    """Compute the 0-9 Piotroski F-Score using point-in-time columns.

    Criteria:
    F1 ROA > 0, F2 CFO > 0, F3 delta ROA > 0, F4 accruals < 0,
    F5 leverage down, F6 current ratio up, F7 no new shares,
    F8 gross margin up, F9 asset turnover up.
    """
    if df.empty:
        return pd.Series(dtype=float)
    frame = df.copy()
    group = frame["ticker"].astype(str) if "ticker" in frame.columns else pd.Series("all", index=frame.index)
    roa = _num(frame, "roa", _safe_div(_num(frame, "net_income", np.nan), _num(frame, "total_assets", np.nan)))
    cfo = _num(frame, "operating_cash_flow", np.nan)
    assets = _num(frame, "total_assets", np.nan)
    accruals = _num(frame, "accruals_ratio", _safe_div(_num(frame, "net_income", np.nan) - cfo, assets))
    leverage = _num(frame, "debt_to_equity", _safe_div(_num(frame, "total_debt", np.nan), _num(frame, "total_equity", np.nan)))
    current_ratio = _num(frame, "current_ratio", _safe_div(_num(frame, "current_assets", np.nan), _num(frame, "current_liabilities", np.nan)))
    shares = _num(frame, "shares_outstanding", np.nan)
    gross_margin = _num(frame, "gross_margin", _safe_div(_num(frame, "gross_profit", np.nan), _num(frame, "revenue_ttm", _num(frame, "revenue", np.nan))))
    asset_turnover = _num(frame, "asset_turnover", _safe_div(_num(frame, "revenue_ttm", _num(frame, "revenue", np.nan)), assets))
    delta_roa = roa.groupby(group).diff()
    delta_leverage = leverage.groupby(group).diff()
    delta_current_ratio = current_ratio.groupby(group).diff()
    delta_shares = shares.groupby(group).diff()
    delta_gross_margin = gross_margin.groupby(group).diff()
    delta_asset_turnover = asset_turnover.groupby(group).diff()
    criteria = [
        roa > 0,
        cfo > 0,
        delta_roa > 0,
        accruals < 0,
        delta_leverage < 0,
        delta_current_ratio > 0,
        delta_shares <= 0,
        delta_gross_margin > 0,
        delta_asset_turnover > 0,
    ]
    score = sum(condition.fillna(False).astype(int) for condition in criteria)
    return score.astype(float)


def _price_panel_to_wide(df_prices: pd.DataFrame) -> pd.DataFrame:
    if df_prices is None or df_prices.empty:
        return pd.DataFrame()
    frame = df_prices.copy()
    if {"ticker", "date"}.issubset(frame.columns):
        price_col = next((col for col in ["adjclose", "adj_close", "close", "price"] if col in frame.columns), None)
        if price_col is None:
            return pd.DataFrame()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        return frame.pivot_table(index="date", columns="ticker", values=price_col, aggfunc="last").sort_index()
    wide = frame.copy()
    if not isinstance(wide.index, pd.DatetimeIndex):
        maybe_date = next((col for col in ["date", "Date"] if col in wide.columns), None)
        if maybe_date:
            wide[maybe_date] = pd.to_datetime(wide[maybe_date], errors="coerce")
            wide = wide.set_index(maybe_date)
    return wide.apply(pd.to_numeric, errors="coerce").sort_index()


def _aligned_returns(asset: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    asset_ret = pd.to_numeric(asset, errors="coerce").pct_change()
    bench_ret = pd.to_numeric(benchmark, errors="coerce").pct_change()
    return pd.concat({"asset": asset_ret, "benchmark": bench_ret}, axis=1).dropna()


def _capture_ratio(joint: pd.DataFrame, up: bool) -> float:
    mask = joint["benchmark"] > 0 if up else joint["benchmark"] < 0
    sample = joint.loc[mask]
    if sample.empty or sample["benchmark"].mean() == 0:
        return np.nan
    return float(sample["asset"].mean() / sample["benchmark"].mean())


def compute_alpha_factors(
    df_prices: pd.DataFrame,
    df_fundamentals: pd.DataFrame | None,
    benchmark_series: pd.Series,
    windows: list[int] | tuple[int, ...] = (252, 756),
) -> pd.DataFrame:
    """Compute benchmark-relative alpha and active-risk features.

    The function uses only trailing observations.  If regional factor returns
    are unavailable, multi-factor alpha columns fall back to the same
    point-in-time Jensen-alpha estimate and remain explicitly model-ready.
    """
    prices = _price_panel_to_wide(df_prices)
    if prices.empty or benchmark_series is None or len(benchmark_series) == 0:
        return pd.DataFrame()
    benchmark = pd.Series(benchmark_series).copy()
    benchmark.index = pd.to_datetime(benchmark.index, errors="coerce")
    benchmark = benchmark.sort_index()
    rows: list[dict[str, float | str]] = []
    for ticker in prices.columns:
        joint = _aligned_returns(prices[ticker], benchmark).tail(max(windows))
        row: dict[str, float | str] = {"ticker": str(ticker)}
        for window in windows:
            sample = joint.tail(window)
            suffix = "1y" if window <= 252 else "3y" if window >= 756 else f"{window}d"
            if len(sample) < max(30, min(window // 4, 126)) or sample["benchmark"].var() == 0:
                row[f"alpha_{suffix}"] = np.nan
                continue
            beta = float(np.cov(sample["asset"], sample["benchmark"])[0, 1] / np.var(sample["benchmark"]))
            alpha_daily = float(sample["asset"].mean() - beta * sample["benchmark"].mean())
            row[f"alpha_{suffix}"] = alpha_daily * 252
            row[f"beta_{suffix}"] = beta
            if suffix == "1y":
                active = sample["asset"] - sample["benchmark"]
                tracking_error = float(active.std(ddof=1) * np.sqrt(252))
                active_return = float(active.mean() * 252)
                downside = sample["asset"] - sample["asset"].clip(lower=0)
                threshold = 0.0
                gains = np.maximum(sample["asset"] - threshold, 0).sum()
                losses = np.abs(np.minimum(sample["asset"] - threshold, 0).sum())
                row["tracking_error_1y"] = tracking_error
                row["information_ratio_1y"] = active_return / tracking_error if tracking_error else np.nan
                row["treynor_ratio_1y"] = float(sample["asset"].mean() * 252 / beta) if beta else np.nan
                row["m2_measure_1y"] = float((sample["asset"].mean() / sample["asset"].std(ddof=1)) * sample["benchmark"].std(ddof=1) * 252) if sample["asset"].std(ddof=1) else np.nan
                row["upside_capture_ratio"] = _capture_ratio(sample, up=True)
                row["downside_capture_ratio"] = _capture_ratio(sample, up=False)
                row["batting_average_1y"] = float((active > 0).mean())
                row["omega_ratio_1y"] = float(gains / losses) if losses else np.nan
                row["idiosyncratic_return_1y"] = active_return
                row["downside_active_vol_1y"] = float(downside.std(ddof=1) * np.sqrt(252)) if downside.notna().sum() > 2 else np.nan
        alpha_proxy = row.get("alpha_1y", np.nan)
        row.setdefault("alpha_3factor", alpha_proxy)
        row.setdefault("alpha_5factor", alpha_proxy)
        row.setdefault("alpha_carhart4", alpha_proxy)
        row.setdefault("alpha_q5", alpha_proxy)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_investment_factors(df_fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Compute point-in-time-safe investment factor signals.

    Signals are calculated from reported fundamentals and shifted by one
    reporting observation per ticker, representing the minimum post-filing lag
    needed before they can be used as predictors.
    """
    if df_fundamentals is None or df_fundamentals.empty:
        return pd.DataFrame()
    out = df_fundamentals.copy()
    date_col = next((col for col in ["filed_date", "report_date", "period_of_report", "date"] if col in out.columns), None)
    if date_col:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        sort_cols = ["ticker", date_col] if "ticker" in out.columns else [date_col]
        out = out.sort_values(sort_cols)
    group = out["ticker"].astype(str) if "ticker" in out.columns else pd.Series("all", index=out.index)
    assets = _num(out, "total_assets", np.nan)
    capex = abs(_num(out, "capex", _num(out, "capital_expenditure", np.nan)))
    shares = _num(out, "shares_outstanding", np.nan)
    debt = _num(out, "total_debt", np.nan)
    equity = _num(out, "total_equity", np.nan)
    net_income = _num(out, "net_income", np.nan)
    revenue = _num(out, "revenue_ttm", _num(out, "revenue", np.nan))
    roe = _num(out, "roe", _safe_div(net_income, equity))
    ppe = _num(out, "ppe", _num(out, "property_plant_equipment", np.nan))
    working_capital = _num(out, "working_capital", _num(out, "current_assets", np.nan) - _num(out, "current_liabilities", np.nan))

    out["asset_growth"] = assets.groupby(group).pct_change()
    out["capex_to_assets"] = _safe_div(capex, assets)
    out["capex_growth"] = capex.groupby(group).pct_change()
    out["net_stock_issues"] = np.log(_safe_div(shares, shares.groupby(group).shift(1)))
    out["net_debt_issues"] = _safe_div(debt - debt.groupby(group).shift(1), assets.groupby(group).shift(1))
    out["investment_to_assets"] = _safe_div(assets - assets.groupby(group).shift(1), assets.groupby(group).shift(1))
    out["roe_growth"] = roe - roe.groupby(group).shift(1)
    out["external_financing_ratio"] = _safe_div((equity - equity.groupby(group).shift(1)) + (debt - debt.groupby(group).shift(1)), assets.groupby(group).shift(1))
    out["pp_and_e_growth"] = ppe.groupby(group).pct_change()
    out["working_capital_change"] = _safe_div(working_capital - working_capital.groupby(group).shift(1), assets.groupby(group).shift(1))
    out["capex_intensity"] = _safe_div(capex, revenue)

    factor_cols = [
        "asset_growth",
        "capex_to_assets",
        "capex_growth",
        "net_stock_issues",
        "net_debt_issues",
        "investment_to_assets",
        "roe_growth",
        "external_financing_ratio",
        "pp_and_e_growth",
        "working_capital_change",
        "capex_intensity",
    ]
    out[factor_cols] = out.groupby(group)[factor_cols].shift(1)
    out["pit_lag_days"] = 45
    return out


def _extract_first_numeric(payload: object, keys: tuple[str, ...]) -> float:
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if not isinstance(payload, dict):
        return np.nan
    for key in keys:
        value = payload.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return np.nan


def compute_sentiment_alternative_features(
    tickers: list[str],
    api_client,
    as_of_date: str | None = None,
) -> pd.DataFrame:
    """Fetch alternative/sentiment features through a multi-provider client.

    Missing credentials or provider errors return ``PARTIAL`` rows instead of
    raising, which keeps local research and Streamlit startup deterministic.
    """
    rows: list[dict[str, object]] = []
    for ticker in tickers:
        row: dict[str, object] = {
            "ticker": ticker,
            "as_of_date": as_of_date or pd.Timestamp.utcnow().date().isoformat(),
            "short_interest_ratio": np.nan,
            "institutional_ownership_pct": np.nan,
            "insider_net_buying": np.nan,
            "analyst_coverage_count": np.nan,
            "earnings_estimate_dispersion": np.nan,
            "data_status": "FAILED",
            "source_provider": "none",
        }
        try:
            waterfall = api_client.get_fundamentals_waterfall(ticker) if hasattr(api_client, "get_fundamentals_waterfall") else {}
            data = waterfall.get("data", waterfall) if isinstance(waterfall, dict) else {}
            row["source_provider"] = waterfall.get("source_provider", "unknown") if isinstance(waterfall, dict) else "unknown"
            row["institutional_ownership_pct"] = _extract_first_numeric(data, ("institutionalOwnershipPercentage", "institutional_ownership_pct", "heldPercentInstitutions"))
            row["analyst_coverage_count"] = _extract_first_numeric(data, ("analystCoverage", "analyst_coverage_count", "numberOfAnalystOpinions"))
            row["earnings_estimate_dispersion"] = _extract_first_numeric(data, ("earningsEstimateDispersion", "earnings_estimate_dispersion"))
            row["short_interest_ratio"] = _extract_first_numeric(data, ("shortInterestRatio", "short_interest_ratio", "shortRatio"))
            row["insider_net_buying"] = _extract_first_numeric(data, ("insiderNetBuying", "insider_net_buying"))
            row["data_status"] = "OK" if any(pd.notna(row[col]) for col in [
                "short_interest_ratio",
                "institutional_ownership_pct",
                "insider_net_buying",
                "analyst_coverage_count",
                "earnings_estimate_dispersion",
            ]) else "PARTIAL"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def add_advanced_equity_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add optional technical and fundamental columns to an equity panel."""
    if panel.empty:
        return panel.copy()
    out = panel.copy()
    if "ticker" in out.columns and "date" in out.columns:
        out = out.sort_values(["ticker", "date"])
        out = out.groupby("ticker", group_keys=False).apply(compute_advanced_technical_features).reset_index(drop=True)
        spy_mask = out["ticker"].astype(str).str.upper().eq("SPY")
        if spy_mask.any() and "ret_1d" in out.columns:
            spy_returns = (
                out.loc[spy_mask, ["date", "ret_1d"]]
                .dropna(subset=["date"])
                .drop_duplicates("date", keep="last")
                .set_index("date")["ret_1d"]
            )
            out["_benchmark_ret"] = pd.to_datetime(out["date"], errors="coerce").map(spy_returns)
            out = out.groupby("ticker", group_keys=False).apply(
                lambda frame: add_idiosyncratic_features(frame, frame["_benchmark_ret"])
            ).reset_index(drop=True)
            out = out.drop(columns=["_benchmark_ret"], errors="ignore")
    else:
        out = compute_advanced_technical_features(out)
    out = compute_advanced_fundamental_features(out)
    return out

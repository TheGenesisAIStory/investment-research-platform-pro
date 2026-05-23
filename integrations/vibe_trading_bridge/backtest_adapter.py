"""Backtest adapter that standardizes strategy evaluation results."""

from __future__ import annotations

from dataclasses import dataclass, field
from inspect import signature
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from .data_adapter import DataBridgeConfig, get_feature_matrix, get_price_history
from .strategy_adapter import StrategyBase, validate_signal_frame


BacktestRunner = Callable[..., Any]


@dataclass(slots=True)
class BacktestResult:
    """Standardized backtest output returned to Vibe-Trading and local callers."""

    returns_curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]
    weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable summary while keeping frames accessible."""

        return {
            "metrics": self.metrics,
            "diagnostics": self.diagnostics,
            "returns_rows": int(len(self.returns_curve)),
            "trade_rows": int(len(self.trades)),
            "weight_rows": int(len(self.weights)),
        }


def run_backtest(
    strategy: StrategyBase | Callable[..., pd.DataFrame],
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    initial_capital: float = 100_000.0,
    params: Mapping[str, Any] | None = None,
) -> BacktestResult:
    """Run a strategy through the host backtest engine or a vectorized fallback.

    ``params`` may include ``universe``/``symbols``, ``price_history``,
    ``features``, ``data_config``, ``backtest_runner``, ``transaction_cost_bps``,
    ``max_gross_exposure`` and ``periods_per_year``.
    """

    options = dict(params or {})
    data_config = options.get("data_config")
    if data_config is not None and not isinstance(data_config, DataBridgeConfig):
        raise TypeError("params['data_config'] must be a DataBridgeConfig when provided")
    data_config = data_config or DataBridgeConfig()

    symbols = options.get("symbols") or options.get("universe") or options.get("tickers")
    price_history = options.get("price_history")
    if price_history is None:
        price_history = get_price_history(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            fields=options.get("fields"),
            config=data_config,
        )

    features = options.get("features")
    if features is None and options.get("load_features", True):
        features = get_feature_matrix(
            universe=symbols,
            start_date=start_date,
            end_date=end_date,
            feature_set_name=str(options.get("feature_set_name", "all")),
            config=data_config,
        )

    custom_runner = options.get("backtest_runner")
    if custom_runner is not None:
        return _standardize_custom_result(
            _call_runner(
                custom_runner,
                strategy=strategy,
                price_history=price_history,
                features=features,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                params=options,
            )
        )

    signals = _generate_strategy_signals(strategy, price_history, features, options)
    return _run_vectorized_backtest(
        price_history=price_history,
        signals=signals,
        initial_capital=initial_capital,
        params=options,
    )


def _generate_strategy_signals(
    strategy: StrategyBase | Callable[..., pd.DataFrame],
    price_history: pd.DataFrame,
    features: pd.DataFrame | None,
    params: Mapping[str, Any],
) -> pd.DataFrame:
    if isinstance(strategy, StrategyBase):
        signals = strategy.generate_signals(params.get("current_state", {}), price_history, features)
    else:
        signals = _call_runner(strategy, historical_data=price_history, price_history=price_history, features=features, params=params)
    if not isinstance(signals, pd.DataFrame):
        raise TypeError("Strategy must return a pandas.DataFrame of signals")
    return validate_signal_frame(signals)


def _run_vectorized_backtest(
    price_history: pd.DataFrame,
    signals: pd.DataFrame,
    initial_capital: float,
    params: Mapping[str, Any],
) -> BacktestResult:
    prices = _prices_to_wide(price_history, price_col=params.get("price_col"))
    if prices.empty:
        return BacktestResult(
            returns_curve=pd.DataFrame(columns=["date", "return", "equity"]),
            trades=pd.DataFrame(),
            metrics={},
            diagnostics={"engine_status": "no_price_data"},
        )

    weights = _signals_to_weights(
        signals=signals,
        dates=prices.index,
        tickers=prices.columns,
        allow_short=bool(params.get("allow_short", True)),
        max_gross_exposure=float(params.get("max_gross_exposure", 1.0)),
        normalize_target_weights=bool(params.get("normalize_target_weights", False)),
    )
    asset_returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    held_weights = weights.shift(1).fillna(0.0)
    gross_returns = held_weights.mul(asset_returns, axis=0).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    transaction_cost = turnover * (float(params.get("transaction_cost_bps", 0.0)) / 10_000.0)
    portfolio_returns = (gross_returns - transaction_cost).fillna(0.0)
    equity = initial_capital * (1.0 + portfolio_returns).cumprod()

    returns_curve = pd.DataFrame(
        {
            "date": portfolio_returns.index,
            "return": portfolio_returns.values,
            "gross_return": gross_returns.reindex(portfolio_returns.index).values,
            "transaction_cost": transaction_cost.reindex(portfolio_returns.index).values,
            "equity": equity.values,
        }
    )
    trades = _build_trade_log(weights, prices)
    periods_per_year = int(params.get("periods_per_year") or _infer_periods_per_year(portfolio_returns.index))
    metrics = _compute_metrics(portfolio_returns, equity, turnover, periods_per_year)
    weights_long = weights.reset_index(names="date").melt(id_vars="date", var_name="ticker", value_name="target_weight")
    diagnostics = {
        "engine_status": "vectorized_fallback",
        "periods_per_year": periods_per_year,
        "price_rows": int(len(price_history)),
        "signal_rows": int(len(signals)),
        "transaction_cost_bps": float(params.get("transaction_cost_bps", 0.0)),
    }
    return BacktestResult(returns_curve=returns_curve, trades=trades, metrics=metrics, weights=weights_long, diagnostics=diagnostics)


def _prices_to_wide(price_history: pd.DataFrame, price_col: str | None = None) -> pd.DataFrame:
    if price_history is None or price_history.empty:
        return pd.DataFrame()
    data = price_history.copy()
    if isinstance(data.index, pd.DatetimeIndex) and "ticker" not in data.columns:
        return data.apply(pd.to_numeric, errors="coerce").sort_index()
    if "date" not in data.columns:
        raise ValueError("price_history must include a date column")
    if "ticker" not in data.columns:
        data["ticker"] = "ASSET"
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    selected_price_col = _resolve_price_col(data, price_col)
    if selected_price_col is None:
        return pd.DataFrame()
    wide = data.pivot_table(index="date", columns="ticker", values=selected_price_col, aggfunc="last")
    return wide.sort_index().apply(pd.to_numeric, errors="coerce").ffill().dropna(axis=1, how="all")


def _signals_to_weights(
    signals: pd.DataFrame,
    dates: pd.Index,
    tickers: pd.Index,
    allow_short: bool,
    max_gross_exposure: float,
    normalize_target_weights: bool,
) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame(0.0, index=dates, columns=tickers)
    values = signals.copy()
    values["date"] = pd.to_datetime(values["date"], errors="coerce")
    value_col = "target_weight" if "target_weight" in values.columns else "signal"
    wide = values.pivot_table(index="date", columns="ticker", values=value_col, aggfunc="last")
    wide = wide.reindex(index=dates, columns=tickers).ffill().fillna(0.0)
    if not allow_short:
        wide = wide.clip(lower=0.0)
    if value_col == "target_weight" and not normalize_target_weights:
        gross = wide.abs().sum(axis=1).replace(0, np.nan)
        scale = (max_gross_exposure / gross).clip(upper=1.0).fillna(0.0)
        return wide.mul(scale, axis=0)
    gross = wide.abs().sum(axis=1).replace(0, np.nan)
    return wide.div(gross, axis=0).fillna(0.0) * max_gross_exposure


def _build_trade_log(weights: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    deltas = weights.diff().fillna(weights)
    rows: list[dict[str, Any]] = []
    for date, row in deltas.iterrows():
        changed = row[row.abs() > 1e-12]
        for ticker, delta in changed.items():
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "delta_weight": float(delta),
                    "target_weight": float(weights.at[date, ticker]),
                    "previous_weight": float(weights.shift(1).fillna(0.0).at[date, ticker]),
                    "price": float(prices.at[date, ticker]) if ticker in prices.columns and pd.notna(prices.at[date, ticker]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _compute_metrics(
    returns: pd.Series,
    equity: pd.Series,
    turnover: pd.Series,
    periods_per_year: int,
) -> dict[str, float]:
    try:
        from ml_stock_lab.evaluation import annualized_volatility, max_drawdown, sharpe_ratio
    except Exception:
        annualized_volatility = None
        max_drawdown = None
        sharpe_ratio = None

    clean_returns = pd.to_numeric(returns, errors="coerce").dropna()
    if clean_returns.empty:
        return {}
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 and equity.iloc[0] else 0.0
    annualized_return = float((1.0 + total_return) ** (periods_per_year / max(len(clean_returns), 1)) - 1.0)
    vol = (
        float(annualized_volatility(clean_returns, periods_per_year=periods_per_year))
        if annualized_volatility is not None
        else float(clean_returns.std(ddof=0) * np.sqrt(periods_per_year))
    )
    sharpe = (
        float(sharpe_ratio(clean_returns, periods_per_year=periods_per_year))
        if sharpe_ratio is not None
        else float((clean_returns.mean() / clean_returns.std(ddof=0)) * np.sqrt(periods_per_year))
    )
    drawdown = (
        float(max_drawdown(clean_returns))
        if max_drawdown is not None
        else float((equity / equity.cummax() - 1.0).min())
    )
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "mean_period_return": float(clean_returns.mean()),
        "avg_turnover": float(pd.to_numeric(turnover, errors="coerce").mean()),
        "observations": float(len(clean_returns)),
    }


def _infer_periods_per_year(index: pd.Index) -> int:
    dates = pd.DatetimeIndex(index).dropna()
    if len(dates) < 3:
        return 252
    median_days = pd.Series(dates).diff().dt.days.median()
    if pd.isna(median_days) or median_days <= 2:
        return 252
    if median_days <= 10:
        return 52
    return 12


def _resolve_price_col(data: pd.DataFrame, requested: str | None) -> str | None:
    for column in (requested, "adj_close", "close", "price", "mkt_price", "market_value"):
        if column and column in data.columns:
            return column
    return None


def _call_runner(runner: BacktestRunner, **kwargs: Any) -> Any:
    try:
        params = signature(runner).parameters
    except (TypeError, ValueError):
        return runner(**kwargs)
    accepts_kwargs = any(param.kind.name == "VAR_KEYWORD" for param in params.values())
    filtered = kwargs if accepts_kwargs else {key: value for key, value in kwargs.items() if key in params}
    return runner(**filtered)


def _standardize_custom_result(raw: Any) -> BacktestResult:
    if isinstance(raw, BacktestResult):
        return raw
    if not isinstance(raw, Mapping):
        raise TypeError("Custom backtest runner must return BacktestResult or a mapping")
    returns_curve = raw.get("returns_curve") or raw.get("backtest") or raw.get("returns")
    if isinstance(returns_curve, pd.Series):
        returns_curve = returns_curve.rename("return").reset_index().rename(columns={"index": "date"})
    if returns_curve is None:
        returns_curve = pd.DataFrame()
    trades = raw.get("trades", pd.DataFrame())
    weights = raw.get("weights", pd.DataFrame())
    return BacktestResult(
        returns_curve=returns_curve if isinstance(returns_curve, pd.DataFrame) else pd.DataFrame(returns_curve),
        trades=trades if isinstance(trades, pd.DataFrame) else pd.DataFrame(trades),
        metrics=dict(raw.get("metrics", {})),
        weights=weights if isinstance(weights, pd.DataFrame) else pd.DataFrame(weights),
        diagnostics=dict(raw.get("diagnostics", {"engine_status": "custom_runner"})),
    )


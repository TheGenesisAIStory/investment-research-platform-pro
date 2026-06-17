"""WorldQuant 101 Formulaic Alphas.

Reference: Kakushadze (2015), "101 Formulaic Alphas",
https://arxiv.org/abs/1601.00991.

Input conventions use wide daily panels with dates on the index and tickers on
columns.  Each public ``alphaNNN`` function returns a DataFrame with the same
shape as the input and a final cross-sectional rank normalization in ``[0, 1]``
for stable downstream ML usage.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


_WARNED: set[str] = set()


def _w(d: float | int) -> int:
    return max(int(np.floor(float(d))), 1)


def _df(x: pd.DataFrame | pd.Series | float | int | bool, like: pd.DataFrame | None = None) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        out = x.copy()
    elif isinstance(x, pd.Series):
        out = x.to_frame().T if like is None else pd.DataFrame(np.tile(x.to_numpy(), (len(like.index), 1)), index=like.index, columns=like.columns)
    else:
        if like is None:
            raise ValueError("Scalar inputs require a template DataFrame")
        out = pd.DataFrame(float(x), index=like.index, columns=like.columns)
    if like is not None:
        out = out.reindex(index=like.index, columns=like.columns)
    return out.apply(pd.to_numeric, errors="coerce")


def _clean(x: pd.DataFrame, like: pd.DataFrame | None = None) -> pd.DataFrame:
    out = _df(x, like=like).replace([np.inf, -np.inf], np.nan)
    return out


def rank(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank in each row."""
    return _clean(x).rank(axis=1, pct=True)


def delay(x: pd.DataFrame, d: float | int = 1) -> pd.DataFrame:
    return _clean(x).shift(_w(d))


def correlation(x: pd.DataFrame, y: pd.DataFrame, d: float | int) -> pd.DataFrame:
    return _clean(x).rolling(_w(d), min_periods=max(2, _w(d) // 2)).corr(_clean(y, x))


def covariance(x: pd.DataFrame, y: pd.DataFrame, d: float | int) -> pd.DataFrame:
    return _clean(x).rolling(_w(d), min_periods=max(2, _w(d) // 2)).cov(_clean(y, x))


def scale(x: pd.DataFrame, a: float = 1.0) -> pd.DataFrame:
    out = _clean(x)
    denom = out.abs().sum(axis=1).replace(0, np.nan)
    return out.div(denom, axis=0) * float(a)


def delta(x: pd.DataFrame, d: float | int = 1) -> pd.DataFrame:
    out = _clean(x)
    return out - out.shift(_w(d))


def signedpower(x: pd.DataFrame, a: pd.DataFrame | float | int) -> pd.DataFrame:
    base = _clean(x)
    if isinstance(a, pd.DataFrame):
        exp = _clean(a, base).clip(lower=-5, upper=5).fillna(1.0)
        return np.sign(base) * np.power(base.abs().clip(lower=1e-12), exp)
    return np.sign(base) * np.power(base.abs(), float(a))


def decay_linear(x: pd.DataFrame, d: float | int) -> pd.DataFrame:
    window = _w(d)
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()
    return _clean(x).rolling(window, min_periods=max(2, window // 2)).apply(lambda arr: float(np.dot(arr, weights[-len(arr):])), raw=True)


def indneutralize(x: pd.DataFrame, g: pd.Series | dict[str, str] | None = None) -> pd.DataFrame:
    """Demean within groups; no-op with a one-time warning when groups absent."""
    out = _clean(x)
    if g is None:
        if "industry" not in _WARNED:
            warnings.warn("indneutralize skipped: no industry mapping available", RuntimeWarning, stacklevel=2)
            _WARNED.add("industry")
        return out
    groups = pd.Series(g)
    groups.index = groups.index.astype(str)
    mapped = pd.Series(out.columns.astype(str), index=out.columns).map(groups).fillna("UNKNOWN")
    result = out.copy()
    for group in mapped.unique():
        cols = mapped[mapped.eq(group)].index
        result[cols] = out[cols].sub(out[cols].mean(axis=1), axis=0)
    return result


def ts_min(x: pd.DataFrame, d: float | int) -> pd.DataFrame:
    return _clean(x).rolling(_w(d), min_periods=max(1, _w(d) // 2)).min()


def ts_max(x: pd.DataFrame, d: float | int) -> pd.DataFrame:
    return _clean(x).rolling(_w(d), min_periods=max(1, _w(d) // 2)).max()


def ts_argmin(x: pd.DataFrame, d: float | int) -> pd.DataFrame:
    window = _w(d)
    return _clean(x).rolling(window, min_periods=max(2, window // 2)).apply(lambda arr: float(np.nanargmin(arr) + 1) if np.isfinite(arr).any() else np.nan, raw=True)


def ts_argmax(x: pd.DataFrame, d: float | int) -> pd.DataFrame:
    window = _w(d)
    return _clean(x).rolling(window, min_periods=max(2, window // 2)).apply(lambda arr: float(np.nanargmax(arr) + 1) if np.isfinite(arr).any() else np.nan, raw=True)


def ts_rank(x: pd.DataFrame, d: float | int) -> pd.DataFrame:
    window = _w(d)

    def _last_rank(arr: np.ndarray) -> float:
        series = pd.Series(arr).dropna()
        if series.empty:
            return np.nan
        return float(series.rank(pct=True).iloc[-1])

    return _clean(x).rolling(window, min_periods=max(2, window // 2)).apply(_last_rank, raw=True)


def ts_sum(x: pd.DataFrame, d: float | int) -> pd.DataFrame:
    return _clean(x).rolling(_w(d), min_periods=max(1, _w(d) // 2)).sum()


def ts_mean(x: pd.DataFrame, d: float | int) -> pd.DataFrame:
    return _clean(x).rolling(_w(d), min_periods=max(1, _w(d) // 2)).mean()


def product(x: pd.DataFrame, d: float | int) -> pd.DataFrame:
    return _clean(x).rolling(_w(d), min_periods=max(1, _w(d) // 2)).apply(np.nanprod, raw=True)


def stddev(x: pd.DataFrame, d: float | int) -> pd.DataFrame:
    return _clean(x).rolling(_w(d), min_periods=max(2, _w(d) // 2)).std()


def log(x: pd.DataFrame) -> pd.DataFrame:
    return np.log(_clean(x).where(_clean(x) > 0))


def abs(x: pd.DataFrame) -> pd.DataFrame:  # noqa: A001 - WQ operator name
    return _clean(x).abs()


def sign(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(_clean(x))


def _where(cond: pd.DataFrame, a: pd.DataFrame | float | int, b: pd.DataFrame | float | int, like: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.where(cond, _df(a, like), _df(b, like)), index=like.index, columns=like.columns)


def _min2(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.minimum(_clean(a), _clean(b, a)), index=a.index, columns=a.columns)


def _max2(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.maximum(_clean(a), _clean(b, a)), index=a.index, columns=a.columns)


def _final(x: pd.DataFrame, like: pd.DataFrame) -> pd.DataFrame:
    out = _clean(x, like=like)
    out = rank(out)
    return out.clip(lower=0.0, upper=1.0)


def _adv(volume: pd.DataFrame, d: float | int) -> pd.DataFrame:
    return ts_mean(volume, d)


def _cap(close: pd.DataFrame, volume: pd.DataFrame, cap: pd.DataFrame | None = None) -> pd.DataFrame:
    if cap is not None:
        return _clean(cap, close)
    if "cap" not in _WARNED:
        warnings.warn("cap unavailable: using close * adv20 proxy", RuntimeWarning, stacklevel=2)
        _WARNED.add("cap")
    return close * _adv(volume, 20)


def alpha001(close: pd.DataFrame, returns: pd.DataFrame, **kw) -> pd.DataFrame:
    inner = pd.DataFrame(np.where(returns < 0, stddev(returns, 20), close), index=close.index, columns=close.columns)
    return _final(rank(ts_argmax(signedpower(inner, 2), 5)) - 0.5, close)


def alpha002(close: pd.DataFrame, open_: pd.DataFrame, volume: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(-correlation(rank(delta(log(volume), 2)), rank((close - open_) / open_), 6), close)


def alpha003(open_: pd.DataFrame, volume: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(-correlation(rank(open_), rank(volume), 10), open_)


def alpha004(low: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(-ts_rank(rank(low), 9), low)


def alpha005(open_: pd.DataFrame, close: pd.DataFrame, vwap: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(rank(open_ - ts_sum(vwap, 10) / 10) * (-abs(rank(close - vwap))), close)


def alpha006(open_: pd.DataFrame, volume: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(-correlation(open_, volume, 10), open_)


def alpha007(close: pd.DataFrame, volume: pd.DataFrame, adv20: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv20 = adv20 if adv20 is not None else _adv(volume, 20)
    val = -ts_rank(abs(delta(close, 7)), 60) * sign(delta(close, 7))
    return _final(_where(adv20 < volume, val, -1, close), close)


def alpha008(open_: pd.DataFrame, returns: pd.DataFrame, **kw) -> pd.DataFrame:
    x = ts_sum(open_, 5) * ts_sum(returns, 5)
    return _final(-rank(x - delay(x, 10)), open_)


def alpha009(close: pd.DataFrame, **kw) -> pd.DataFrame:
    dc = delta(close, 1)
    return _final(_where((ts_min(dc, 5) > 0) | (ts_max(dc, 5) < 0), dc, -dc, close), close)


def alpha010(close: pd.DataFrame, **kw) -> pd.DataFrame:
    dc = delta(close, 1)
    return _final(rank(_where((ts_min(dc, 4) > 0) | (ts_max(dc, 4) < 0), dc, -dc, close)), close)


def alpha011(close: pd.DataFrame, volume: pd.DataFrame, vwap: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final((rank(ts_max(vwap - close, 3)) + rank(ts_min(vwap - close, 3))) * rank(delta(volume, 3)), close)


def alpha012(close: pd.DataFrame, volume: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(sign(delta(volume, 1)) * (-delta(close, 1)), close)


def alpha013(close: pd.DataFrame, volume: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(-rank(covariance(rank(close), rank(volume), 5)), close)


def alpha014(open_: pd.DataFrame, volume: pd.DataFrame, returns: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(-rank(delta(returns, 3)) * correlation(open_, volume, 10), open_)


def alpha015(high: pd.DataFrame, volume: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(-ts_sum(rank(correlation(rank(high), rank(volume), 3)), 3), high)


def alpha016(high: pd.DataFrame, volume: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(-rank(covariance(rank(high), rank(volume), 5)), high)


def alpha017(close: pd.DataFrame, volume: pd.DataFrame, adv20: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv20 = adv20 if adv20 is not None else _adv(volume, 20)
    return _final((-rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1)) * rank(ts_rank(volume / adv20, 5)), close)


def alpha018(open_: pd.DataFrame, close: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(-rank(stddev(abs(close - open_), 5) + (close - open_) + correlation(close, open_, 10)), close)


def alpha019(close: pd.DataFrame, returns: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final((-sign((close - delay(close, 7)) + delta(close, 7))) * (1 + rank(1 + ts_sum(returns, 250))), close)


def alpha020(open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final((-rank(open_ - delay(high, 1))) * rank(open_ - delay(close, 1)) * rank(open_ - delay(low, 1)), close)


def alpha021(close: pd.DataFrame, volume: pd.DataFrame, adv20: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv20 = adv20 if adv20 is not None else _adv(volume, 20)
    cond1 = (ts_sum(close, 8) / 8 + stddev(close, 8)) < (ts_sum(close, 2) / 2)
    cond2 = (ts_sum(close, 2) / 2) < (ts_sum(close, 8) / 8 - stddev(close, 8))
    cond3 = volume / adv20 >= 1
    raw = _where(cond1, -1, _where(cond2, 1, _where(cond3, 1, -1, close), close), close)
    return _final(raw, close)


def alpha022(high: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(-(delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))), close)


def alpha023(high: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(_where((ts_sum(high, 20) / 20) < high, -delta(high, 2), 0, high), high)


def alpha024(close: pd.DataFrame, **kw) -> pd.DataFrame:
    trend = delta(ts_sum(close, 100) / 100, 100) / delay(close, 100)
    return _final(_where(trend <= 0.05, -(close - ts_min(close, 100)), -delta(close, 3), close), close)


def alpha025(high: pd.DataFrame, close: pd.DataFrame, returns: pd.DataFrame, volume: pd.DataFrame, vwap: pd.DataFrame, adv20: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv20 = adv20 if adv20 is not None else _adv(volume, 20)
    return _final(rank((-returns) * adv20 * vwap * (high - close)), close)


def alpha026(high: pd.DataFrame, volume: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(-ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3), high)


def alpha027(volume: pd.DataFrame, vwap: pd.DataFrame, **kw) -> pd.DataFrame:
    raw = _where(0.5 < rank(ts_sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0), -1, 1, volume)
    return _final(raw, volume)


def alpha028(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, adv20: pd.DataFrame | None = None, volume: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv20 = adv20 if adv20 is not None else _adv(volume, 20)
    return _final(scale(correlation(adv20, low, 5) + ((high + low) / 2) - close), close)


def alpha029(close: pd.DataFrame, returns: pd.DataFrame, **kw) -> pd.DataFrame:
    base = rank(rank(-rank(delta(close - 1, 5))))
    nested = rank(rank(scale(log(ts_sum(ts_min(base, 2), 1)))))
    raw = ts_min(product(nested, 1), 5) + ts_rank(delay(-returns, 6), 5)
    return _final(raw, close)


def alpha030(close: pd.DataFrame, volume: pd.DataFrame, **kw) -> pd.DataFrame:
    inner = sign(close - delay(close, 1)) + sign(delay(close, 1) - delay(close, 2)) + sign(delay(close, 2) - delay(close, 3))
    return _final(((1 - rank(inner)) * ts_sum(volume, 5)) / ts_sum(volume, 20), close)


def alpha031(close: pd.DataFrame, low: pd.DataFrame, adv20: pd.DataFrame | None = None, volume: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv20 = adv20 if adv20 is not None else _adv(volume, 20)
    raw = rank(rank(rank(decay_linear(-rank(rank(delta(close, 10))), 10)))) + rank(-delta(close, 3)) + sign(scale(correlation(adv20, low, 12)))
    return _final(raw, close)


def alpha032(close: pd.DataFrame, vwap: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(scale(ts_sum(close, 7) / 7 - close) + 20 * scale(correlation(vwap, delay(close, 5), 230)), close)


def alpha033(open_: pd.DataFrame, close: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(rank(-(1 - open_ / close)), close)


def alpha034(close: pd.DataFrame, returns: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(rank((1 - rank(stddev(returns, 2) / stddev(returns, 5))) + (1 - rank(delta(close, 1)))), close)


def alpha035(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, returns: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(ts_rank(volume, 32) * (1 - ts_rank((close + high) - low, 16)) * (1 - ts_rank(returns, 32)), close)


def alpha036(open_: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, returns: pd.DataFrame, vwap: pd.DataFrame, adv20: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv20 = adv20 if adv20 is not None else _adv(volume, 20)
    raw = (
        2.21 * rank(correlation(close - open_, delay(volume, 1), 15))
        + 0.7 * rank(open_ - close)
        + 0.73 * rank(ts_rank(delay(-returns, 6), 5))
        + rank(abs(correlation(vwap, adv20, 6)))
        + 0.6 * rank((ts_sum(close, 200) / 200 - open_) * (close - open_))
    )
    return _final(raw, close)


def alpha037(open_: pd.DataFrame, close: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(rank(correlation(delay(open_ - close, 1), close, 200)) + rank(open_ - close), close)


def alpha038(open_: pd.DataFrame, close: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(-rank(ts_rank(close, 10)) * rank(close / open_), close)


def alpha039(close: pd.DataFrame, volume: pd.DataFrame, returns: pd.DataFrame, adv20: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv20 = adv20 if adv20 is not None else _adv(volume, 20)
    return _final((-rank(delta(close, 7) * (1 - rank(decay_linear(volume / adv20, 9))))) * (1 + rank(ts_sum(returns, 250))), close)


def alpha040(high: pd.DataFrame, volume: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(-rank(stddev(high, 10)) * correlation(high, volume, 10), high)


def alpha041(high: pd.DataFrame, low: pd.DataFrame, vwap: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(np.sqrt(high * low) - vwap, high)


def alpha042(close: pd.DataFrame, vwap: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(rank(vwap - close) / rank(vwap + close), close)


def alpha043(close: pd.DataFrame, volume: pd.DataFrame, adv20: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv20 = adv20 if adv20 is not None else _adv(volume, 20)
    return _final(ts_rank(volume / adv20, 20) * ts_rank(-delta(close, 7), 8), close)


def alpha044(high: pd.DataFrame, volume: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(-correlation(high, rank(volume), 5), high)


def alpha045(close: pd.DataFrame, volume: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(-(rank(ts_sum(delay(close, 5), 20) / 20) * correlation(close, volume, 2) * rank(correlation(ts_sum(close, 5), ts_sum(close, 20), 2))), close)


def alpha046(close: pd.DataFrame, **kw) -> pd.DataFrame:
    inner = ((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)
    raw = _where(0.25 < inner, -1, _where(inner < 0, 1, -(close - delay(close, 1)), close), close)
    return _final(raw, close)


def alpha047(high: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, vwap: pd.DataFrame, adv20: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv20 = adv20 if adv20 is not None else _adv(volume, 20)
    return _final((((rank(1 / close) * volume) / adv20) * ((high * rank(high - close)) / (ts_sum(high, 5) / 5))) - rank(vwap - delay(vwap, 5)), close)


def alpha048(close: pd.DataFrame, **kw) -> pd.DataFrame:
    raw = indneutralize((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close, kw.get("industries"))
    return _final(raw / ts_sum((delta(close, 1) / delay(close, 1)) ** 2, 250), close)


def alpha049(close: pd.DataFrame, **kw) -> pd.DataFrame:
    inner = ((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)
    return _final(_where(inner < -0.1, 1, -(close - delay(close, 1)), close), close)


def alpha050(volume: pd.DataFrame, vwap: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(-ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5), volume)


def alpha051(close: pd.DataFrame, **kw) -> pd.DataFrame:
    inner = ((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)
    return _final(_where(inner < -0.05, 1, -(close - delay(close, 1)), close), close)


def alpha052(low: pd.DataFrame, volume: pd.DataFrame, returns: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(((-ts_min(low, 5) + delay(ts_min(low, 5), 5)) * rank((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220)) * ts_rank(volume, 5), low)


def alpha053(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(-delta(((close - low) - (high - close)) / (close - low + 1e-6), 9), close)


def alpha054(open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final((-(low - close) * (open_**5)) / ((low - high).replace(0, -1e-6) * (close**5)), close)


def alpha055(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(-correlation(rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12) + 1e-6)), rank(volume), 6), close)


def alpha056(returns: pd.DataFrame, cap: pd.DataFrame | None = None, close: pd.DataFrame | None = None, volume: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    cap_frame = _cap(close, volume, cap)
    return _final(-(rank(ts_sum(returns, 10) / ts_sum(ts_sum(returns, 2), 3)) * rank(returns * cap_frame)), returns)


def alpha057(close: pd.DataFrame, vwap: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(-((close - vwap) / (decay_linear(rank(ts_argmax(close, 30)), 2) + 1e-6)), close)


def alpha058(volume: pd.DataFrame, vwap: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(-ts_rank(decay_linear(correlation(indneutralize(vwap, kw.get("industries")), volume, 3.92795), 7.89291), 5.50322), vwap)


def alpha059(volume: pd.DataFrame, vwap: pd.DataFrame, **kw) -> pd.DataFrame:
    neutral = indneutralize((vwap * 0.728317) + (vwap * (1 - 0.728317)), kw.get("industries"))
    return _final(-ts_rank(decay_linear(correlation(neutral, volume, 4.25197), 16.2289), 8.19648), vwap)


def alpha060(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, **kw) -> pd.DataFrame:
    raw = -((2 * scale(rank((((close - low) - (high - close)) / (high - low + 1e-6)) * volume))) - scale(rank(ts_argmax(close, 10))))
    return _final(raw, close)


def alpha061(volume: pd.DataFrame, vwap: pd.DataFrame, adv180: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv180 = adv180 if adv180 is not None else _adv(volume, 180)
    return _final((rank(vwap - ts_min(vwap, 16.1219)) < rank(correlation(vwap, adv180, 17.9282))).astype(float), vwap)


def alpha062(open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, vwap: pd.DataFrame, adv20: pd.DataFrame | None = None, volume: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv20 = adv20 if adv20 is not None else _adv(volume, 20)
    raw = (rank(correlation(vwap, ts_sum(adv20, 22.4101), 9.91009)) < rank((rank(open_) + rank(open_)) < (rank((high + low) / 2) + rank(high)))) * -1
    return _final(raw.astype(float), vwap)


def alpha063(close: pd.DataFrame, open_: pd.DataFrame, vwap: pd.DataFrame, volume: pd.DataFrame, adv180: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv180 = adv180 if adv180 is not None else _adv(volume, 180)
    p1 = rank(decay_linear(delta(indneutralize(close, kw.get("industries")), 2.25164), 8.22237))
    p2 = rank(decay_linear(correlation((vwap * 0.318108) + (open_ * (1 - 0.318108)), ts_sum(adv180, 37.2467), 13.557), 12.2883))
    return _final((p1 - p2) * -1, close)


def alpha064(open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, vwap: pd.DataFrame, adv120: pd.DataFrame | None = None, volume: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv120 = adv120 if adv120 is not None else _adv(volume, 120)
    raw = rank(correlation(ts_sum((open_ * 0.178404) + (low * (1 - 0.178404)), 12.7054), ts_sum(adv120, 12.7054), 16.6208)) < rank(delta((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404)), 3.69741))
    return _final(raw.astype(float) * -1, vwap)


def alpha065(open_: pd.DataFrame, volume: pd.DataFrame, vwap: pd.DataFrame, adv60: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv60 = adv60 if adv60 is not None else _adv(volume, 60)
    raw = rank(correlation((open_ * 0.00817205) + (vwap * (1 - 0.00817205)), ts_sum(adv60, 8.6911), 6.40374)) < rank(open_ - ts_min(open_, 13.635))
    return _final(raw.astype(float) * -1, open_)


def alpha066(open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, vwap: pd.DataFrame, **kw) -> pd.DataFrame:
    raw = rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + ts_rank(decay_linear((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open_ - ((high + low) / 2) + 1e-6), 11.4157), 6.72611)
    return _final(raw * -1, vwap)


def alpha067(high: pd.DataFrame, vwap: pd.DataFrame, adv20: pd.DataFrame | None = None, volume: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv20 = adv20 if adv20 is not None else _adv(volume, 20)
    raw = rank(high - ts_min(high, 2.14593)).pow(rank(correlation(indneutralize(vwap, kw.get("industries")), indneutralize(adv20, kw.get("industries")), 6.02936)))
    return _final(raw * -1, high)


def alpha068(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, adv15: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv15 = adv15 if adv15 is not None else _adv(volume, 15)
    raw = ts_rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) < rank(delta((close * 0.518371) + (low * (1 - 0.518371)), 1.06157))
    return _final(raw.astype(float) * -1, close)


def alpha069(close: pd.DataFrame, vwap: pd.DataFrame, adv20: pd.DataFrame | None = None, volume: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv20 = adv20 if adv20 is not None else _adv(volume, 20)
    p1 = rank(ts_max(delta(indneutralize(vwap, kw.get("industries")), 2.72412), 4.79344))
    p2 = ts_rank(correlation((close * 0.490655) + (vwap * (1 - 0.490655)), adv20, 4.92416), 9.0615)
    return _final(p1.pow(p2) * -1, close)


def alpha070(close: pd.DataFrame, vwap: pd.DataFrame, adv50: pd.DataFrame | None = None, volume: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv50 = adv50 if adv50 is not None else _adv(volume, 50)
    raw = rank(delta(vwap, 1.29456)).pow(ts_rank(correlation(indneutralize(close, kw.get("industries")), adv50, 17.8256), 17.9171))
    return _final(raw * -1, close)


def alpha071(open_: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, vwap: pd.DataFrame, adv180: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv180 = adv180 if adv180 is not None else _adv(volume, 180)
    p1 = ts_rank(decay_linear(correlation(ts_rank(close, 3.43976), ts_rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948)
    p2 = ts_rank(decay_linear(rank((low + open_) - (vwap + vwap)).pow(2), 16.4662), 4.4388)
    return _final(_max2(p1, p2), close)


def alpha072(high: pd.DataFrame, low: pd.DataFrame, volume: pd.DataFrame, vwap: pd.DataFrame, adv40: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv40 = adv40 if adv40 is not None else _adv(volume, 40)
    return _final(rank(decay_linear(correlation((high + low) / 2, adv40, 8.93345), 10.1519)) / rank(decay_linear(correlation(ts_rank(vwap, 3.72469), ts_rank(volume, 18.5188), 6.86671), 2.95011)), vwap)


def alpha073(open_: pd.DataFrame, low: pd.DataFrame, vwap: pd.DataFrame, **kw) -> pd.DataFrame:
    base = (open_ * 0.147155) + (low * (1 - 0.147155))
    p1 = rank(decay_linear(delta(vwap, 4.72775), 2.91864))
    p2 = ts_rank(decay_linear((delta(base, 2.03608) / (base + 1e-6)) * -1, 3.33829), 16.7411)
    return _final(_max2(p1, p2) * -1, vwap)


def alpha074(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, vwap: pd.DataFrame, adv30: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv30 = adv30 if adv30 is not None else _adv(volume, 30)
    raw = rank(correlation(close, ts_sum(adv30, 37.4843), 15.1365)) < rank(correlation(rank((high * 0.0261661) + (vwap * (1 - 0.0261661))), rank(volume), 11.4791))
    return _final(raw.astype(float) * -1, close)


def alpha075(low: pd.DataFrame, volume: pd.DataFrame, vwap: pd.DataFrame, adv50: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv50 = adv50 if adv50 is not None else _adv(volume, 50)
    return _final((rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50), 12.4413))).astype(float), low)


def alpha076(low: pd.DataFrame, volume: pd.DataFrame, adv81: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv81 = adv81 if adv81 is not None else _adv(volume, 81)
    p1 = rank(decay_linear(delta(low, 1.24383), 11.8259))
    p2 = ts_rank(decay_linear(ts_rank(correlation(indneutralize(low, kw.get("industries")), adv81, 8.14941), 19.569), 17.1543), 19.383)
    return _final(_max2(p1, p2) * -1, low)


def alpha077(high: pd.DataFrame, low: pd.DataFrame, volume: pd.DataFrame, vwap: pd.DataFrame, adv40: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv40 = adv40 if adv40 is not None else _adv(volume, 40)
    p1 = rank(decay_linear((((high + low) / 2) + high) - (vwap + high), 20.0451))
    p2 = rank(decay_linear(correlation((high + low) / 2, adv40, 3.1614), 5.64125))
    return _final(_min2(p1, p2), high)


def alpha078(low: pd.DataFrame, volume: pd.DataFrame, vwap: pd.DataFrame, adv40: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv40 = adv40 if adv40 is not None else _adv(volume, 40)
    raw = rank(correlation(ts_sum((low * 0.352233) + (vwap * (1 - 0.352233)), 19.7428), ts_sum(adv40, 19.7428), 6.83313)).pow(rank(correlation(rank(vwap), rank(volume), 5.77492)))
    return _final(raw, vwap)


def alpha079(open_: pd.DataFrame, close: pd.DataFrame, vwap: pd.DataFrame, volume: pd.DataFrame, adv150: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv150 = adv150 if adv150 is not None else _adv(volume, 150)
    raw = rank(delta(indneutralize((close * 0.60733) + (open_ * (1 - 0.60733)), kw.get("industries")), 1.23438)) < rank(correlation(ts_rank(vwap, 3.60973), ts_rank(adv150, 9.18637), 14.6644))
    return _final(raw.astype(float), close)


def alpha080(open_: pd.DataFrame, high: pd.DataFrame, volume: pd.DataFrame, adv10: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv10 = adv10 if adv10 is not None else _adv(volume, 10)
    raw = rank(sign(delta(indneutralize((open_ * 0.868128) + (high * (1 - 0.868128)), kw.get("industries")), 4.04545))).pow(ts_rank(correlation(high, adv10, 5.11456), 5.53756))
    return _final(raw * -1, high)


def alpha081(volume: pd.DataFrame, vwap: pd.DataFrame, adv10: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv10 = adv10 if adv10 is not None else _adv(volume, 10)
    raw = rank(log(product(rank(rank(correlation(vwap, ts_sum(adv10, 49.6054), 8.47743)).pow(4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))
    return _final(raw.astype(float) * -1, vwap)


def alpha082(open_: pd.DataFrame, volume: pd.DataFrame, adv20: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    p1 = rank(decay_linear(delta(open_, 1.46063), 14.8717))
    p2 = ts_rank(decay_linear(correlation(indneutralize(volume, kw.get("industries")), open_, 17.4842), 6.92131), 13.4283)
    return _final(_min2(p1, p2) * -1, open_)


def alpha083(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, vwap: pd.DataFrame, **kw) -> pd.DataFrame:
    spread = (high - low) / (ts_sum(close, 5) / 5)
    return _final((rank(delay(spread, 2)) * rank(rank(volume))) / (spread / (vwap - close + 1e-6)), close)


def alpha084(close: pd.DataFrame, vwap: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final(signedpower(ts_rank(vwap - ts_max(vwap, 15.3217), 20.7127), delta(close, 4.96796)), close)


def alpha085(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, adv30: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv30 = adv30 if adv30 is not None else _adv(volume, 30)
    raw = rank(correlation((high * 0.876703) + (close * (1 - 0.876703)), adv30, 9.61331)).pow(rank(correlation(ts_rank((high + low) / 2, 3.70596), ts_rank(volume, 10.1595), 7.11408)))
    return _final(raw, close)


def alpha086(open_: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, vwap: pd.DataFrame, adv20: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv20 = adv20 if adv20 is not None else _adv(volume, 20)
    raw = ts_rank(correlation(close, ts_sum(adv20, 14.7444), 6.00049), 20.4195) < rank((open_ + close) - (vwap + open_))
    return _final(raw.astype(float) * -1, close)


def alpha087(close: pd.DataFrame, volume: pd.DataFrame, vwap: pd.DataFrame, adv81: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv81 = adv81 if adv81 is not None else _adv(volume, 81)
    p1 = rank(decay_linear(delta((close * 0.369701) + (vwap * (1 - 0.369701)), 1.91233), 2.65461))
    p2 = ts_rank(decay_linear(abs(correlation(indneutralize(adv81, kw.get("industries")), close, 13.4132)), 4.89768), 14.4535)
    return _final(_max2(p1, p2) * -1, close)


def alpha088(open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, adv60: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv60 = adv60 if adv60 is not None else _adv(volume, 60)
    p1 = rank(decay_linear((rank(open_) + rank(low)) - (rank(high) + rank(close)), 8.06882))
    p2 = ts_rank(decay_linear(correlation(ts_rank(close, 8.44728), ts_rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957)
    return _final(_min2(p1, p2), close)


def alpha089(low: pd.DataFrame, volume: pd.DataFrame, vwap: pd.DataFrame, adv10: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv10 = adv10 if adv10 is not None else _adv(volume, 10)
    p1 = ts_rank(decay_linear(correlation(low, adv10, 6.94279), 5.51607), 3.79744)
    p2 = ts_rank(decay_linear(delta(indneutralize(vwap, kw.get("industries")), 3.48158), 10.1466), 15.3012)
    return _final(p1 - p2, vwap)


def alpha090(close: pd.DataFrame, low: pd.DataFrame, adv40: pd.DataFrame | None = None, volume: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv40 = adv40 if adv40 is not None else _adv(volume, 40)
    raw = rank(close - ts_max(close, 4.66719)).pow(ts_rank(correlation(indneutralize(adv40, kw.get("industries")), low, 5.38375), 3.21856))
    return _final(raw * -1, close)


def alpha091(close: pd.DataFrame, volume: pd.DataFrame, vwap: pd.DataFrame, adv30: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv30 = adv30 if adv30 is not None else _adv(volume, 30)
    p1 = ts_rank(decay_linear(decay_linear(correlation(indneutralize(close, kw.get("industries")), volume, 9.74928), 16.398), 3.83219), 4.8667)
    p2 = rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))
    return _final((p1 - p2) * -1, close)


def alpha092(open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, adv30: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv30 = adv30 if adv30 is not None else _adv(volume, 30)
    p1 = ts_rank(decay_linear(((((high + low) / 2) + close) < (low + open_)).astype(float), 14.7221), 18.8683)
    p2 = ts_rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024), 6.80584)
    return _final(_min2(p1, p2), close)


def alpha093(close: pd.DataFrame, vwap: pd.DataFrame, adv81: pd.DataFrame | None = None, volume: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv81 = adv81 if adv81 is not None else _adv(volume, 81)
    num = ts_rank(decay_linear(correlation(indneutralize(vwap, kw.get("industries")), adv81, 17.4193), 19.848), 7.54455)
    den = rank(decay_linear(delta((close * 0.524434) + (vwap * (1 - 0.524434)), 2.77377), 16.2664))
    return _final(num / (den + 1e-6), close)


def alpha094(volume: pd.DataFrame, vwap: pd.DataFrame, adv60: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv60 = adv60 if adv60 is not None else _adv(volume, 60)
    raw = rank(vwap - ts_min(vwap, 11.5783)).pow(ts_rank(correlation(ts_rank(vwap, 19.6462), ts_rank(adv60, 4.02992), 18.0926), 2.70756))
    return _final(raw * -1, vwap)


def alpha095(open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, adv40: pd.DataFrame | None = None, volume: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv40 = adv40 if adv40 is not None else _adv(volume, 40)
    raw = rank(open_ - ts_min(open_, 12.4105)) < ts_rank(rank(correlation(ts_sum((high + low) / 2, 19.1351), ts_sum(adv40, 19.1351), 12.8742)).pow(5), 11.7584)
    return _final(raw.astype(float), open_)


def alpha096(close: pd.DataFrame, volume: pd.DataFrame, vwap: pd.DataFrame, adv60: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv60 = adv60 if adv60 is not None else _adv(volume, 60)
    p1 = ts_rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151)
    p2 = ts_rank(decay_linear(ts_argmax(correlation(ts_rank(close, 7.45404), ts_rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)
    return _final(_max2(p1, p2) * -1, close)


def alpha097(low: pd.DataFrame, vwap: pd.DataFrame, adv60: pd.DataFrame | None = None, volume: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv60 = adv60 if adv60 is not None else _adv(volume, 60)
    p1 = rank(decay_linear(delta(indneutralize((low * 0.721001) + (vwap * (1 - 0.721001)), kw.get("industries")), 3.3705), 20.4523))
    p2 = ts_rank(decay_linear(ts_rank(correlation(ts_rank(low, 7.87871), ts_rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)
    return _final((p1 - p2) * -1, low)


def alpha098(open_: pd.DataFrame, volume: pd.DataFrame, vwap: pd.DataFrame, adv5: pd.DataFrame | None = None, adv15: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv5 = adv5 if adv5 is not None else _adv(volume, 5)
    adv15 = adv15 if adv15 is not None else _adv(volume, 15)
    raw = rank(decay_linear(correlation(vwap, ts_sum(adv5, 26.4719), 4.58418), 7.18088)) - rank(decay_linear(ts_rank(ts_argmin(correlation(rank(open_), rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206))
    return _final(raw, vwap)


def alpha099(high: pd.DataFrame, low: pd.DataFrame, volume: pd.DataFrame, adv60: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv60 = adv60 if adv60 is not None else _adv(volume, 60)
    raw = rank(correlation(ts_sum((high + low) / 2, 19.8975), ts_sum(adv60, 19.8975), 8.8136)) < rank(correlation(low, volume, 6.28259))
    return _final(raw.astype(float) * -1, high)


def alpha100(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, adv20: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    adv20 = adv20 if adv20 is not None else _adv(volume, 20)
    part1 = 1.5 * scale(indneutralize(indneutralize(rank((((close - low) - (high - close)) / (high - low + 1e-6)) * volume), kw.get("industries")), kw.get("industries")))
    part2 = scale(indneutralize(correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30)), kw.get("industries")))
    return _final(-((part1 - part2) * (volume / adv20)), close)


def alpha101(open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, **kw) -> pd.DataFrame:
    return _final((close - open_) / ((high - low) + 0.001), close)


ALPHA_FUNCTIONS: dict[str, Callable[..., pd.DataFrame]] = {
    f"alpha{i:03d}": globals()[f"alpha{i:03d}"] for i in range(1, 102)
}


REQUIRES_INDUSTRY = {
    "alpha048",
    "alpha058",
    "alpha059",
    "alpha063",
    "alpha067",
    "alpha069",
    "alpha070",
    "alpha076",
    "alpha079",
    "alpha080",
    "alpha082",
    "alpha087",
    "alpha089",
    "alpha090",
    "alpha091",
    "alpha093",
    "alpha097",
    "alpha100",
}
REQUIRES_CAP = {"alpha056"}


def _category_for_alpha(i: int) -> str:
    if i in {1, 4, 7, 8, 9, 10, 17, 19, 24, 29, 31, 32, 33, 34, 38, 39, 43, 46, 49, 51, 52, 57, 61, 64, 65, 66, 68, 73, 84, 86, 94, 96, 98}:
        return "momentum"
    if i in {2, 3, 5, 6, 11, 12, 13, 14, 15, 16, 18, 20, 22, 25, 26, 27, 28, 35, 36, 37, 40, 41, 42, 44, 45, 47, 50, 53, 54, 55, 60, 62, 71, 72, 74, 75, 77, 78, 81, 83, 85, 88, 92, 95, 99, 101}:
        return "technical"
    if i in {48, 56, 58, 59, 63, 67, 69, 70, 76, 79, 80, 82, 87, 89, 90, 91, 93, 97, 100}:
        return "risk"
    return "technical"


FORMULA_TEXT: dict[str, str] = {
    f"alpha{i:03d}": f"WorldQuant Alpha#{i} formulaic signal from Kakushadze (2015). See source paper Appendix A for exact expression."
    for i in range(1, 102)
}
FORMULA_TEXT.update(
    {
        "alpha001": "(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)",
        "alpha002": "(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))",
        "alpha003": "(-1 * correlation(rank(open), rank(volume), 10))",
        "alpha004": "(-1 * Ts_Rank(rank(low), 9))",
        "alpha005": "(rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))",
        "alpha006": "(-1 * correlation(open, volume, 10))",
        "alpha007": "((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : -1)",
        "alpha056": "(0 - (rank(sum(returns, 10) / sum(sum(returns, 2), 3)) * rank(returns * cap)))",
        "alpha100": "(0 - (((1.5 * scale(indneutralize(indneutralize(rank(price_pressure*volume))))) - scale(indneutralize(correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))))) * (volume / adv20)))",
        "alpha101": "((close - open) / ((high - low) + .001))",
    }
)


@dataclass
class Alpha101Suite:
    """Compute WorldQuant Alpha101 features on wide OHLCV panels."""

    failed_alphas: dict[str, str] | None = None

    ALPHA_REGISTRY = ALPHA_FUNCTIONS
    METADATA = {
        alpha_id: {
            "id": alpha_id,
            "paper": "Kakushadze 2015",
            "arxiv": "1601.00991",
            "formula_text": FORMULA_TEXT[alpha_id],
            "inputs": ["open_", "high", "low", "close", "volume", "vwap", "returns"],
            "requires_industry": alpha_id in REQUIRES_INDUSTRY,
            "requires_cap": alpha_id in REQUIRES_CAP,
            "factor_zoo_category": _category_for_alpha(int(alpha_id[-3:])),
            "economic_rationale": "Formulaic price-volume alpha capturing short-horizon cross-sectional mean reversion, momentum, liquidity or risk-state effects.",
        }
        for alpha_id in ALPHA_FUNCTIONS
    }

    def _kwargs(
        self,
        close: pd.DataFrame,
        open_: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        volume: pd.DataFrame,
        vwap: pd.DataFrame | None = None,
        returns: pd.DataFrame | None = None,
        cap: pd.DataFrame | None = None,
        industries: pd.Series | dict[str, str] | None = None,
    ) -> dict[str, pd.DataFrame | pd.Series | dict[str, str] | None]:
        vwap = _clean(vwap, close) if vwap is not None else (open_ + high + low + close) / 4
        returns = _clean(returns, close) if returns is not None else close.pct_change()
        payload: dict[str, pd.DataFrame | pd.Series | dict[str, str] | None] = {
            "close": _clean(close),
            "open_": _clean(open_, close),
            "high": _clean(high, close),
            "low": _clean(low, close),
            "volume": _clean(volume, close),
            "vwap": _clean(vwap, close),
            "returns": _clean(returns, close),
            "cap": _clean(cap, close) if cap is not None else None,
            "industries": industries,
        }
        for d in [5, 10, 15, 20, 30, 40, 50, 60, 81, 120, 150, 180]:
            payload[f"adv{d}"] = _adv(payload["volume"], d)  # type: ignore[arg-type]
        return payload

    def compute_all(self, close, open_, high, low, volume, vwap=None, returns=None, cap=None, industries=None, verbose: bool = False) -> dict[str, pd.DataFrame]:
        kwargs = self._kwargs(close, open_, high, low, volume, vwap=vwap, returns=returns, cap=cap, industries=industries)
        results: dict[str, pd.DataFrame] = {}
        self.failed_alphas = {}
        for alpha_id, func in self.ALPHA_REGISTRY.items():
            try:
                results[alpha_id] = func(**kwargs)  # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover - defensive path
                self.failed_alphas[alpha_id] = f"{type(exc).__name__}: {exc}"
                if verbose:
                    warnings.warn(f"{alpha_id} failed: {exc}", RuntimeWarning, stacklevel=2)
                results[alpha_id] = pd.DataFrame(np.nan, index=kwargs["close"].index, columns=kwargs["close"].columns)  # type: ignore[index, union-attr]
        return results

    def compute_subset(self, alpha_ids: list[str], close, open_, high, low, volume, vwap=None, returns=None, cap=None, industries=None, verbose: bool = False) -> dict[str, pd.DataFrame]:
        kwargs = self._kwargs(close, open_, high, low, volume, vwap=vwap, returns=returns, cap=cap, industries=industries)
        results: dict[str, pd.DataFrame] = {}
        self.failed_alphas = {}
        for alpha_id in alpha_ids:
            normalized = alpha_id if alpha_id.startswith("alpha") else f"alpha{int(alpha_id):03d}"
            func = self.ALPHA_REGISTRY[normalized]
            try:
                results[normalized] = func(**kwargs)  # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover
                self.failed_alphas[normalized] = f"{type(exc).__name__}: {exc}"
                if verbose:
                    warnings.warn(f"{normalized} failed: {exc}", RuntimeWarning, stacklevel=2)
        return results

    def to_flat_panel(self, results: dict[str, pd.DataFrame]) -> pd.DataFrame:
        frames = []
        for alpha_id, frame in results.items():
            if frame.empty:
                continue
            try:
                stacked = frame.stack(future_stack=True).rename(alpha_id)
            except TypeError:  # pragma: no cover - pandas <2.1 fallback
                stacked = frame.stack(dropna=False).rename(alpha_id)
            stacked.index = stacked.index.set_names(["date", "ticker"])
            frames.append(stacked)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).sort_index()

    def get_metadata_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.METADATA.values()).sort_values("id").reset_index(drop=True)

    def write_audit(self, path: str | Path, existing: list[str] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = existing or []
        missing = [alpha_id for alpha_id in self.ALPHA_REGISTRY if alpha_id not in set(existing)]
        payload = [
            "AUDIT RESULT:",
            "- alpha101 module presente: YES (path: src/research_platform_core/alpha101.py)",
            f"- Alpha implementate esistenti: {sorted(existing)}",
            f"- Totale già presenti: {len(existing)}/101",
            f"- Mancanti: {missing}",
            "- Libreria esterna usata: NO",
        ]
        path.write_text("\n".join(payload) + "\n", encoding="utf-8")
        return path


__all__ = [
    "Alpha101Suite",
    "ALPHA_FUNCTIONS",
    "FORMULA_TEXT",
    "abs",
    "correlation",
    "covariance",
    "decay_linear",
    "delay",
    "delta",
    "indneutralize",
    "log",
    "product",
    "rank",
    "scale",
    "sign",
    "signedpower",
    "stddev",
    "ts_argmax",
    "ts_argmin",
    "ts_max",
    "ts_min",
    "ts_rank",
    "ts_sum",
    *[f"alpha{i:03d}" for i in range(1, 102)],
]

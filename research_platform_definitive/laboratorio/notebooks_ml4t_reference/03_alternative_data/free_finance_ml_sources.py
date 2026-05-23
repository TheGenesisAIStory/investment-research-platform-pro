"""Curated free/low-friction finance ML data source catalog."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class FinanceMLSource:
    name: str
    url: str
    data_type: str
    ml_use: str
    quality: str
    access_notes: str


FREE_FINANCE_ML_SOURCES: tuple[FinanceMLSource, ...] = (
    FinanceMLSource("FRED", "https://fred.stlouisfed.org/", "Macro, rates, inflation, labor", "Regime detection, macro alpha, risk features", "Alta", "API key recommended; free official source."),
    FinanceMLSource("SEC EDGAR", "https://www.sec.gov/edgar", "10-K, 10-Q, 8-K, 13F filings", "NLP, event detection, fundamentals", "Alta", "Use fair-access User-Agent and rate limits."),
    FinanceMLSource("Yahoo Finance", "https://finance.yahoo.com/", "Historical prices, volumes, basic fundamentals", "Backtest prototypes and feature engineering", "Media-alta", "Unofficial access libraries can break; cache locally."),
    FinanceMLSource("Stooq", "https://stooq.com/", "Historical prices, indices, FX", "Backtest and factor research", "Media-alta", "Prefer light/manual downloads; avoid aggressive scraping."),
    FinanceMLSource("GDELT", "https://www.gdeltproject.org/", "Global news, events, media attention", "Sentiment, event trading, attention signals", "Media-alta", "Public APIs and BigQuery options."),
    FinanceMLSource("Google Trends", "https://trends.google.com/", "Search interest", "Demand proxy, attention factor", "Media", "Unofficial clients can be rate-limited."),
    FinanceMLSource("Wikipedia Pageviews", "https://pageviews.wmcloud.org/", "Page traffic", "Attention factor and event studies", "Media", "Public Wikimedia APIs."),
    FinanceMLSource("OpenInsider", "https://openinsider.com/", "Form 4 insider trading", "Insider-sentiment features", "Media-alta", "Respect website terms and low request rates."),
    FinanceMLSource("TradingView free", "https://www.tradingview.com/", "Charts, screening, indicators", "Research, validation, workflow", "Media", "Use manually/free features; do not bypass restrictions."),
    FinanceMLSource("OpenBB", "https://openbb.co/", "Finance data access layer", "Research pipeline and provider abstraction", "Alta come strumento", "Quality depends on configured providers."),
    FinanceMLSource("Polymarket", "https://polymarket.com/", "Prediction market prices/probabilities", "Event-risk and expectation features", "Media", "Use public APIs only; cache and validate markets."),
    FinanceMLSource("Stocktwits", "https://stocktwits.com/", "Social market messages", "Retail attention and sentiment features", "Media", "Use official/public endpoints where available."),
)


def sources_to_dataframe() -> pd.DataFrame:
    return pd.DataFrame([asdict(source) for source in FREE_FINANCE_ML_SOURCES])


if __name__ == "__main__":
    print(sources_to_dataframe().to_string(index=False))

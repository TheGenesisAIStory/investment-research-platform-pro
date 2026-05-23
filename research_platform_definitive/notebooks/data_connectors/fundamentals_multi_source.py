import numpy as np
import pandas as pd
import yfinance as yf
import requests
import time
import logging
from tqdm import tqdm

def normalize_ticker(value):
    if pd.isna(value): return None
    return str(value).strip().upper().replace("/", "-")

def get_panel_tickers(df_panel, max_tickers=None):
    if df_panel is None or df_panel.empty: return []
    ticker_col = next((c for c in df_panel.columns if str(c).lower() in ["ticker", "symbol", "asset", "stock"]), None)
    if ticker_col is None: return []
    tickers = sorted(set(pd.Series(df_panel[ticker_col].dropna().map(normalize_ticker).unique()).dropna().tolist()))
    return tickers[:int(max_tickers)] if max_tickers else tickers

# --- Provider Implementations ---
def fetch_yfinance_fundamentals(ticker):
    try:
        yt = yf.Ticker(ticker)
        inc = yt.financials.T if yt.financials is not None else pd.DataFrame()
        bal = yt.balance_sheet.T if yt.balance_sheet is not None else pd.DataFrame()
        cf = yt.cashflow.T if yt.cashflow is not None else pd.DataFrame()
        if inc.empty and bal.empty and cf.empty: return pd.DataFrame(), "No data"
        merged = pd.concat([inc, bal, cf], axis=1)
        merged = merged.loc[:,~merged.columns.duplicated()]
        merged = merged.reset_index().rename(columns={"index": "date"})
        merged["ticker"] = ticker
        merged["provider"] = "yfinance"
        rename_map = {"Total Revenue": "revenue", "Net Income": "net_income", "Total Assets": "total_assets", "Stockholders Equity": "total_equity"}
        return merged.rename(columns={k: v for k, v in rename_map.items() if k in merged.columns}), None
    except Exception as e:
        return pd.DataFrame(), str(e)

def fetch_fmp_fundamentals(ticker, api_key, config):
    # FMP implementation logic placeholder
    return pd.DataFrame(), "FMP execution not fully scaffolded here but consumes registry"

def get_provider_strategy(market):
    return ["fmp", "eodhd", "finnhub", "alphavantage", "yfinance"]

def standardize_dataframe_schema(raw, config):
    if raw is None or raw.empty: return pd.DataFrame()
    df = raw.copy()
    if "date" in df.columns: df["effective_fundamental_date"] = pd.to_datetime(df["date"])
    return df.dropna(subset=["ticker"])

def run_fundamentals_ingestion(df_panel, api_keys, config, market, max_tickers, logger=None):
    if logger is None: logger = logging.getLogger(__name__)
    selected_tickers = get_panel_tickers(df_panel, max_tickers=max_tickers)
    provider_strategy = get_provider_strategy(market)
    fundamental_diagnostics = []
    frames = []
    fundamentals_provider_used = {}

    for ticker in tqdm(selected_tickers, desc="Fetching Fundamentals"):
        ticker_success = False
        logger.info(f"[{ticker}] Starting fundamentals extraction. Fallback hierarchy: {provider_strategy}")
        for provider in provider_strategy:
            result_df, error_msg = pd.DataFrame(), None
            # Registry integration and reporting
            has_cred = bool(api_keys.get(provider)) or provider == "yfinance"
            logger.info(f"[{ticker}] Evaluating {provider}: Credential Present={has_cred}")

            if has_cred:
                if provider == "yfinance": result_df, error_msg = fetch_yfinance_fundamentals(ticker)
                elif provider == "fmp": result_df, error_msg = fetch_fmp_fundamentals(ticker, api_keys["fmp"], config)
            else:
                error_msg = "Credential missing in registry"

            if not result_df.empty:
                std_df = standardize_dataframe_schema(result_df, config)
                frames.append(std_df)
                ticker_success = True
                fundamentals_provider_used[ticker] = provider
                fundamental_diagnostics.append({"ticker": ticker, "provider": provider, "status": "success", "error": None})
                logger.info(f"[{ticker}] SUCCESS via {provider}. Fallback halted.")
                break
            else:
                logger.warning(f"[{ticker}] FAILED via {provider}: {error_msg}. Triggering next fallback.")
                fundamental_diagnostics.append({"ticker": ticker, "provider": provider, "status": "failed", "error": error_msg})

        if not ticker_success:
            fundamentals_provider_used[ticker] = "none"

    df_fund_standardized = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return df_fund_standardized, fundamental_diagnostics, fundamentals_provider_used

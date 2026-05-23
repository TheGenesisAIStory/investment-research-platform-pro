import pandas as pd
import numpy as np
import yfinance as yf
import requests
import logging
from pathlib import Path
from tqdm import tqdm

def get_dynamic_tickers_from_request(master_request=None):
    if master_request:
        target = master_request.get('ticker', 'AAPL')
        peers = master_request.get('manual_peers', [])
        return [target] + [p for p in peers if p and p != target]
    return ["AAPL"]

def load_prices_from_api(tickers, start, end, api_keys=None, logger=None):
    if logger is None: logger = logging.getLogger(__name__)
    api_keys = api_keys or {}
    logger.info(f"[API] Fetching prices for {len(tickers)} tickers with explicit fallbacks")
    frames = []
    diagnostics = []

    for t in tqdm(tickers, desc="Prices Download"):
        df = pd.DataFrame()
        success = False
        provider_selected = None
        error_msg = None

        # 1. Polygon Fallback
        if not success and api_keys.get('polygon'):
            logger.info(f"[{t}] Trying Polygon (Credential found)")
            try:
                url = f"https://api.polygon.io/v2/aggs/ticker/{t}/range/1/day/{start}/{end}?adjusted=true&apiKey={api_keys['polygon']}"
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200 and 'results' in resp.json():
                    raw = pd.DataFrame(resp.json()['results'])
                    if not raw.empty:
                        raw['date'] = pd.to_datetime(raw['t'], unit='ms')
                        raw = raw[['date', 'c', 'v']].rename(columns={'c': 'adj_close', 'v': 'volume'})
                        raw['ticker'] = t
                        df = raw
                        success = True
                        provider_selected = 'polygon'
            except Exception as e:
                error_msg = str(e)

        # 2. AlphaVantage Fallback
        if not success and api_keys.get('alphavantage'):
            logger.info(f"[{t}] Trying AlphaVantage (Credential found)")
            try:
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={t}&outputsize=full&apikey={api_keys['alphavantage']}"
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200 and 'Time Series (Daily)' in resp.json():
                    raw = pd.DataFrame.from_dict(resp.json()['Time Series (Daily)'], orient='index')
                    raw.index = pd.to_datetime(raw.index)
                    raw = raw.reset_index().rename(columns={'index': 'date', '4. close': 'adj_close', '5. volume': 'volume'})
                    raw = raw[(raw['date'] >= pd.to_datetime(start)) & (raw['date'] <= pd.to_datetime(end))]
                    raw['adj_close'] = pd.to_numeric(raw['adj_close'])
                    raw['volume'] = pd.to_numeric(raw['volume'])
                    raw['ticker'] = t
                    df = raw
                    success = True
                    provider_selected = 'alphavantage'
            except Exception as e:
                error_msg = str(e)

        # 3. yfinance Fallback
        if not success:
            logger.info(f"[{t}] Fallback triggered: Using yfinance")
            try:
                raw = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
                if not raw.empty:
                    if isinstance(raw.columns, pd.MultiIndex):
                        raw.columns = raw.columns.droplevel(1)
                    if 'Close' in raw.columns:
                        raw = raw[["Close","Volume"]].copy()
                        raw.columns = ["adj_close","volume"]
                        raw.index.name = "date"
                        raw.reset_index(inplace=True)
                        raw["ticker"] = t
                        df = raw
                        success = True
                        provider_selected = 'yfinance'
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"[API] yfinance {t} error: {e}")

        if success:
            frames.append(df)

        diagnostics.append({
            'ticker': t,
            'provider_selected': provider_selected or 'None',
            'credential_source': 'registry' if provider_selected != 'yfinance' else 'open_access',
            'fallback_triggered': provider_selected == 'yfinance',
            'success_or_failure': 'success' if success else f'failure: {error_msg}'
        })

    df_final = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not df_final.empty:
        df_final["date"] = pd.to_datetime(df_final["date"])
        df_final.sort_values(["ticker","date"], inplace=True)
        df_final.reset_index(drop=True, inplace=True)

    return df_final, diagnostics

def load_dataframe_safe(path: Path, logger=None):
    if logger is None: logger = logging.getLogger(__name__)
    for candidate in [path, Path(str(path)+".parquet"), Path(str(path)+".csv")]:
        if not candidate.exists(): continue
        try:
            if candidate.suffix in [".parquet",".pq"]: df = pd.read_parquet(candidate)
            elif candidate.suffix == ".csv": df = pd.read_csv(candidate, low_memory=False)
            elif candidate.suffix in [".xlsx",".xls"]: df = pd.read_excel(candidate)
            else: continue
            logger.info(f"Loaded {candidate.name} shape={df.shape}")
            return df, str(candidate)
        except Exception as e:
            logger.warning(f"Errore lettura {candidate}: {e}")
    return None, None

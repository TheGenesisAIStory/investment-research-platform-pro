import pandas as pd
import numpy as np

def standardize_to_canonical(ticker: str) -> str:
    """
    Standardize any ticker into a canonical internal format (base YF format).
    """
    if pd.isna(ticker) or not str(ticker).strip():
        return None
    t = str(ticker).strip().upper()
    t = t.replace("/", "-")
    return t

def get_provider_ticker(canonical_ticker: str, provider: str) -> str:
    """
    Map a canonical ticker to a provider-specific ticker format.
    """
    if not canonical_ticker:
        return None

    t = canonical_ticker

    if provider == "yfinance":
        return t
    elif provider == "fmp":
        if t.endswith(".MI"):
            return t.replace(".MI", ".MIL")
        elif t.endswith(".L"):
            return t.replace(".L", ".L") # FMP often keeps L
        elif t.endswith(".PA"):
            return t.replace(".PA", ".PA")
        return t
    elif provider == "alphavantage":
        if t.endswith(".MI"):
            return t.replace(".MI", ".MIL")
        elif t.endswith(".L"):
            return t.replace(".L", ".LON")
        elif t.endswith(".PA"):
            return t.replace(".PA", ".PAR")
        return t
    elif provider == "polygon":
        if "." in t:
            # Polygon requires a different handling for international, often not fully supported without prefix
            pass
        return t
    elif provider == "fred":
        # FRED uses specific series IDs, not standard tickers
        return t

    return t

class TickerRegistry:
    def __init__(self):
        self.registry = {}

    def register_ticker(self, raw_ticker, asset_class="equity"):
        canonical = standardize_to_canonical(raw_ticker)
        if not canonical:
            return None

        suffix = canonical.split(".")[-1] if "." in canonical else ""
        base = canonical.split(".")[0]

        self.registry[canonical] = {
            "raw_ticker": raw_ticker,
            "canonical_ticker": canonical,
            "base_ticker": base,
            "exchange_suffix": suffix,
            "asset_class": asset_class,
            "provider_mapping": {
                "yfinance": get_provider_ticker(canonical, "yfinance"),
                "fmp": get_provider_ticker(canonical, "fmp"),
                "alphavantage": get_provider_ticker(canonical, "alphavantage"),
                "polygon": get_provider_ticker(canonical, "polygon"),
                "eodhd": get_provider_ticker(canonical, "eodhd"),
                "finnhub": get_provider_ticker(canonical, "finnhub")
            }
        }
        return canonical

    def get_registry_df(self):
        if not self.registry:
            return pd.DataFrame()
        records = []
        for can, data in self.registry.items():
            record = {
                "canonical_ticker": can,
                "raw_ticker": data["raw_ticker"],
                "base_ticker": data["base_ticker"],
                "exchange_suffix": data["exchange_suffix"],
            }
            record.update({f"{k}_ticker": v for k, v in data["provider_mapping"].items()})
            records.append(record)
        return pd.DataFrame(records)

    def get_provider_map(self, provider):
        return {can: data["provider_mapping"].get(provider, can) for can, data in self.registry.items()}

# Instantiate a global instance for the module
GLOBAL_REGISTRY = TickerRegistry()

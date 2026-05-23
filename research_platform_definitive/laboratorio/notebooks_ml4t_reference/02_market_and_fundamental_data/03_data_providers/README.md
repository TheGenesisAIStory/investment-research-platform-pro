# 02 API access to market data

## Modern API Registry

The current safe integration layer is:

| File | Purpose |
|---|---|
| [`api_provider_registry.py`](api_provider_registry.py) | Provider metadata, masked credential status, OpenBB bridge, unauthenticated health checks, credentialed smoke checks |
| [`03_integrated_api_provider_registry_definitive.ipynb`](03_integrated_api_provider_registry_definitive.ipynb) | Executable notebook that exports provider catalog/checks to `data/data_providers/api_registry/` |
| [`../../data/00_data_api_services_catalog_definitive.ipynb`](../../data/00_data_api_services_catalog_definitive.ipynb) | Repository-wide data/API catalog |
| [`../../data/01_private_active_api_services.ipynb`](../../data/01_private_active_api_services.ipynb) | Private local smoke tests for active API credentials |

Credentials are read from environment variables only. Keep `.env` and `.env.local` out of git.

| Provider | Env var | Category |
|---|---|---|
| Alpha Vantage | `ALPHA_VANTAGE_API_KEY` | Market data / fundamentals |
| Tiingo | `TIINGO_TOKEN` | Market data / news |
| Financial Modeling Prep | `FMP_API_KEY` | Fundamentals / market data |
| Polygon | `POLYGON_API_KEY` | Market data |
| Intrinio | `INTRINIO_API_KEY` | Fundamentals / market data |
| FRED | `FRED_API_KEY` | Macro |
| BLS | `BLS_API_KEY` | Macro / labor |
| CFTC | `CFTC_APP_TOKEN` | Futures positioning |
| Congress.gov | `CONGRESS_GOV_API_KEY` | Government / policy |
| EconDB | `ECONDB_API_KEY` | Global macro |
| EIA | `EIA_API_KEY` | Energy / macro |
| Nasdaq Data Link | `NASDAQ_API_KEY` | Market / alternative / fundamentals |

There are several options to access market data via API using Python.

## pandas datareader

The notebook [01_pandas_datareader_demo](01_pandas_datareader_demo.ipynb) presents a few sources built into the pandas library. 
- The `pandas` library enables access to data displayed on websites using the read_html function 
- the related `pandas-datareader` library provides access to the API endpoints of various data providers through a standard interface 

## yfinance: Yahoo! Finance market and fundamental data 

The notebook [yfinance_demo](02_yfinance_demo.ipynb) shows how to use yfinance to download a variety of data from Yahoo! Finance. The library works around the deprecation of the historical data API by scraping data from the website in a reliable, efficient way with a Pythonic API.

## LOBSTER tick data

The notebook [03_lobster_itch_data](03_lobster_itch_data.ipynb) demonstrates the use of order book data made available by LOBSTER (Limit Order Book System - The Efficient Reconstructor), an [online](https://lobsterdata.com/info/WhatIsLOBSTER.php) limit order book data tool that aims to provide easy-to-use, high-quality limit order book data.

Since 2013 LOBSTER acts as a data provider for the academic community, giving access to reconstructed limit order book data for the entire universe of NASDAQ traded stocks. More recently, it started offering a commercial service.

## Quandl / Nasdaq Data Link

The notebook [04_quandl_demo](04_quandl_demo.ipynb) shows legacy Quandl usage. For new work, prefer Nasdaq Data Link credentials through `NASDAQ_API_KEY` and validate dataset-level licensing.

## zipline & Quantopian

The notebook [05_zipline_data_demo](05_zipline_data_demo.ipynb) briefly introduces the backtesting library `zipline`. For installation please refer to the instructions [here](../../installation).

# Data Sources used in the book

## Modern Provider Registry

Start here for the current data/API inventory:

| Entry point | Purpose |
|---|---|
| [`00_data_api_services_catalog_definitive.ipynb`](00_data_api_services_catalog_definitive.ipynb) | Safe provider catalog, masked credential status, health checks, dataset inventory, and CSV/Parquet manifest exports |
| [`01_private_active_api_services.ipynb`](01_private_active_api_services.ipynb) | Private local credentialed API smoke checks; writes only masked diagnostics under ignored `data/private_api_checks/` |
| [`create_datasets_definitive.ipynb`](create_datasets_definitive.ipynb) | Optimized dataset creation pipeline for modern Parquet outputs |
| [`data_catalog.py`](data_catalog.py) | Reusable metadata and inventory helpers |
| [`create_dataset_utils.py`](create_dataset_utils.py) | Reusable dataset builders with local fallbacks |
| [`private_api_activation.py`](private_api_activation.py) | Terminal helper that writes `.env.local` with hidden input |
| [`02_local_market_database_definitive.ipynb`](02_local_market_database_definitive.ipynb) | Builds a local SQLite market database smoke sample |
| [`market_data_database.py`](market_data_database.py) | CLI/database builder for equities, ETFs, FX, crypto and optional FMP global top companies |

Recommended commands:

```bash
python3.11 data/private_api_activation.py
cd data
python3.11 -m nbconvert --to notebook --execute --inplace 00_data_api_services_catalog_definitive.ipynb
python3.11 -m nbconvert --to notebook --execute --inplace 01_private_active_api_services.ipynb
python3.11 -m nbconvert --to notebook --execute --inplace create_datasets_definitive.ipynb
python3.11 data/market_data_database.py --mode smoke --years 5 --max-symbols 8
```

Full local market database command, private/local only:

```bash
python3.11 data/market_data_database.py --mode full --years 7
```

The SQLite file is written under `data/local_market_data/`, which is ignored by git. Full mode includes seed universes for S&P 500, Euro STOXX 50, FTSE MIB, major ETFs, major FX, and major crypto; when `FMP_API_KEY` is present it also attempts to add a `global_top_500` equity universe from FMP and downloads daily time series through Yahoo Chart-compatible symbols.

Supported API/provider environment variables:

| Provider | Env var | Primary use |
|---|---|---|
| Alpha Vantage | `ALPHA_VANTAGE_API_KEY` | Prices, fundamentals, FX, crypto, indicators |
| Tiingo | `TIINGO_TOKEN` | Daily prices, IEX data, news, crypto |
| Financial Modeling Prep | `FMP_API_KEY` | Fundamentals, ratios, profiles, prices |
| Polygon | `POLYGON_API_KEY` | Market data, intraday/event research |
| Intrinio | `INTRINIO_API_KEY` | Fundamentals, estimates, ownership, prices |
| FRED | `FRED_API_KEY` | Macro regimes, rates, inflation, labor |
| BLS | `BLS_API_KEY` | Labor market, CPI, PPI, productivity |
| CFTC | `CFTC_APP_TOKEN` | Commitments of Traders and positioning |
| Congress.gov | `CONGRESS_GOV_API_KEY` | Policy events and legislative attention |
| EconDB | `ECONDB_API_KEY` | Global macro panels |
| EIA | `EIA_API_KEY` | Energy prices, inventories, production |
| Nasdaq Data Link | `NASDAQ_API_KEY` | Research datasets and vendor data |

Copy `.env.example` to `.env` or use `python3.11 data/private_api_activation.py`. Do not commit `.env`, `.env.local`, generated private API reports, or raw licensed/heavy datasets.

We will use freely available historical data from market, fundamental and alternative sources. Chapter 2, Market and Fundamental Data and Chapter 3, Alternative Data for Finance  cover characteristics and access to these data sources and introduce key providers that we will use throughout the book. 

A few sample data sources that we will source and work with include, among others:
- Quandl daily prices and other data points for over 3,000 US stocks
- Algoseek minute bar trade and quote price data for NASDAQ 100 stocks
- Stooq daily price data on Japanese equities and US ETFs and stocks
- Yahoo finance daily price data and fundamentals for US stocks  
- NASDAQ ITCH order book data
- Electronic Data Gathering, Analysis, and Retrieval (EDGAR) SEC filings
- Earnings call transcripts from Seeking Alpha
- Various macro fundamental data from the Federal Reserve and others
- Financial news data from Reuters, etc.
- Twitter sentiment data
- Yelp business reviews sentiment data

## How to source the Data

There are several notebooks that guide you through the data sourcing process:
- The notebook [create_datasets](create_datasets.ipynb) contains information on downloading the **Quandl Wiki stock prices** and a few other sources that we use throughout the book, such as S&P500 benchmark, and US equities metadata.
- The notebook [create_stooq_data](create_stooq_data.ipynb) demonstrates how to download historical prices for Japanese stocks and US stocks and ETFs from STOOQ.
  > Please note that STOOQ will disable automatic downloads and require CAPTCHA starting Dec 10, 2020 so that the code that downloads and unpacks the zip files will no longer work; please navigate to their website for manual download.
- The notebook [create_yelp_review_data](create_yelp_review_data.ipynb) combines text data with additional numerical features for sentiment analysis from Yelp user reviews. 
- The notebook [glove_word_vectors](glove_word_vectors.ipynb) downloads pre-trained word vectors.
- The notebook [twitter_sentiment](twitter_sentiment.ipynb) downloads and extracts twitter data for sentiment analysis.

In addition, instructions to obtain data sources for specific applications are provided in the relevant directories and notebooks. 

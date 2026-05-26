# Macro & Sentiment Data Layer

Questa nota descrive il nuovo layer separato da Smart Money:

- **Macro View**: cross-asset board per Global, USA, EU, Italy, Crypto.
- **Asset coverage**: FX, commodities, ETF, fixed income, crypto e country/regional proxies.
- **Sentiment**: provider opzionali StockTwits, Reddit e X.

## Output principali

Il job `compile_macro_asset_database()` scrive:

- `Database Finanziario/MarketData/Macro/MacroAssetCatalog.csv`
- `Database Finanziario/MarketData/Macro/MacroAssetManifest.csv`
- `research_platform_definitive/output/macro_market/tables/MacroAssetCatalog.csv`
- `research_platform_definitive/output/macro_market/tables/MacroAssetManifest.csv`
- `research_platform_definitive/output/macro_market/tables/MacroLatestSnapshot.csv`
- `research_platform_definitive/output/macro_market/tables/MacroHistorySample.csv`

## Policy provider

- yfinance viene usato solo come proxy market-data senza chiave.
- ECB, Banca d'Italia e FRED restano nei loader ufficiali esistenti.
- StockTwits e Reddit sono best-effort; X richiede `X_BEARER_TOKEN`.
- Il sentiment e' rumore contestuale: non e' mai un segnale operativo autonomo.

## UX

Smart Money ora resta focalizzato su issuer evidence:
ownership, insider, activism, government exposure e official-source flow.

Macro View gestisce invece:

- regime map cross-asset;
- regioni Global, USA, EU, Italy, Crypto;
- metadata di strumenti e proxy;
- compile/refresh del macro database;
- social sentiment esplorativo.

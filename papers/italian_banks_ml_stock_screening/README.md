# Italian Banks ML Stock Screening

Paper/esperimento: **Machine Learning per la Valutazione e lo Stock Screening delle Banche Italiane Quotate**.

File principali:

- `paper.md`: brief metodologico del paper da testare.
- `Italian_Banks_ML_Stock_Lab_Colab.ipynb`: notebook Google Colab riutilizzabile basato su `ml_stock_lab`.
- `data/`: cartella per il panel CSV reale e note sulle colonne attese.

Il notebook include anche un **Italian Banks ML Lab Control Center** con interfaccia `ipywidgets` per configurare:

- universo e ticker target;
- peer manuali o derivati dall'universo;
- path CSV o fallback automatico `yfinance`;
- modelli fair-value peer-implied;
- split temporale train/test;
- numero di quantili e weighting;
- soglie CET1/NPL/LDR;
- salvataggio del panel generato.

Dataset:

- CSV panel date/ticker, default: `data/italian_banks_panel.csv`.
- Se il CSV non esiste e `data_mode='auto'`, il notebook scarica un panel prezzi da `yfinance` e lo salva localmente.
- Colonne minime consigliate: `date`, `ticker`, `mkt_price`, `mkt_market_cap`, fondamentali bancari `fund_*`, multipli/mercato `mkt_*`.
- Le feature fondamentali vengono ritardate nel notebook con suffisso `_lag` per ridurre look-ahead bias.
- Se i fondamentali bancari non sono disponibili, il notebook usa solo feature di mercato/tecniche e segnala `target_source` / `data_source`.

Stato: notebook operativo con fallback dati. Per analisi research-grade completa resta preferibile sostituire o arricchire il CSV con fondamentali bancari storici reali.

## Banking data pipeline integrata

Il notebook ora usa anche il modulo condiviso:

- `research_platform_definitive/src/research_platform_core/banking_data.py`

Output prodotti dal notebook/pipeline:

- `output/banks_pipeline/banks_universe.csv`
- `output/banks_pipeline/banks_macro_regulatory.csv`
- `output/banks_pipeline/banks_fundamentals_panel.csv`
- `output/banks_pipeline/banks_market_panel.csv`
- `output/banks_pipeline/banks_data.sqlite`

La pipeline costruisce un universo banche da fonti pubbliche/open:

- Wikipedia come seed non regolamentare delle banche italiane;
- ECB supervised entities PDF come fonte SI/LSI quando parsabile;
- Banca d'Italia BDS tramite link export CSV/XLSX configurabili in `USER_CONFIG['bds_exports']`;
- yfinance solo come fallback market data per banche quotate.

In Streamlit la vista corrispondente è:

- `research_platform_definitive/research_platform_app/pages/12_Banking_Data_Lab.py`

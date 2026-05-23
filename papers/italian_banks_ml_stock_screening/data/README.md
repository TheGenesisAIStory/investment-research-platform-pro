# Dataset

Metti qui il CSV reale dell'esperimento con questo nome:

```text
italian_banks_panel.csv
```

Percorso locale atteso:

```text
/Users/itsgennymac/GitHub/machine-learning-for-trading/papers/italian_banks_ml_stock_screening/data/italian_banks_panel.csv
```

Colonne minime consigliate:

- `date`
- `ticker`
- `country` o `meta_country`
- `sector` o `meta_sector`
- `mkt_price`
- `mkt_market_cap`
- `fund_roa`
- `fund_roe`
- `fund_nim`
- `fund_cost_income`
- `fund_ldr`
- `fund_npl_ratio`
- `fund_cet1`
- `fund_assets`
- `mkt_pb`
- `mkt_pe`
- `mkt_beta_bank`
- `mkt_vol_1y`

Il notebook crea automaticamente:

- feature fondamentali laggate con suffisso `_lag`
- `target_log_mcap`
- `target_ret_1m_fwd`

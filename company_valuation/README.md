# Company Valuation Research Pipeline

Cartella dedicata alla **valutazione aziendale** e all'integrazione tra dati di mercato, fondamentali e dashboard finale.

Questa directory è la posizione canonica del notebook `notebooks/Company_Valuatio.ipynb` e dei moduli Python collegati. La copia top-level `Company_Valuatio.ipynb` è stata rimossa: la fonte di verità GitHub-ready rimane questa cartella.

## Contenuto

```text
company_valuation/
├── notebooks/
│   └── Company_Valuatio.ipynb        # Notebook Colab/Jupyter completo
├── src/
│   ├── __init__.py
│   ├── company_valuation_utils.py    # Helper riusabili estratti/organizzati dal notebook
│   └── sws_company_analysis_model.py # Modello SWS-style eseguibile
├── output/
│   ├── tables/.gitkeep               # Tabelle CSV/parquet generate dal runtime
│   ├── figures/.gitkeep              # Figure statiche o HTML chart generate dal runtime
│   ├── dashboard/.gitkeep            # Dashboard HTML esportata dal runtime
│   └── logs/.gitkeep                 # Log di esecuzione
├── requirements.txt                  # Dipendenze principali
└── README.md                         # Questa guida
```

## Risoluzione conflitti e percorso canonico

Per ridurre conflitti con il ramo principale e con l'editor GitHub, il progetto usa **un solo notebook canonico**:

```text
company_valuation/notebooks/Company_Valuatio.ipynb
```

La copia in root (`Company_Valuatio.ipynb`) non viene più mantenuta. I vecchi link o riferimenti devono puntare al percorso canonico nella cartella del progetto.

## Pipeline finale

Il notebook implementa un flusso end-to-end:

1. **Setup e configurazione**
   - Logging, cartelle di output, configurazione esperimento.
   - Supporto Colab/GitHub.

2. **Ingestion dati di mercato**
   - Caricamento `df_panel` da file locali, Drive o API.
   - Normalizzazione colonne chiave (`date`, `ticker`, `adj_close`).

3. **Fondamentali FMP / cache locale**
   - Lettura cache se presente.
   - Fallback API FMP con secret `FMP_API_KEY` o `fmp_api_key`.
   - Download di income statement, balance sheet e cash flow statement.
   - Costruzione canonica di `df_fund`.

4. **Merge anti look-ahead**
   - Calcolo di `effective_fundamental_date`.
   - Join as-of ticker-safe tra mercato e fondamentali.
   - Diagnostica coverage, staleness e unmatched ticker.

5. **Feature, target e scoring**
   - Returns, momentum, volatilità, ranking cross-section.
   - Scorecard value / quality / momentum / risk.
   - Score Simply Wall St-style: Value, Future, Past, Health, Income.

6. **Valutazione**
   - Fair value stimato con modello FCF a due stadi.
   - Fallback excess returns.
   - Fallback relative valuation.

7. **Diagnostica, robustness e dashboard**
   - QA tables, missingness, staleness, ablation, scenario analysis.
   - Dashboard finale HTML con KPI card, ranking, chart e summary analyst-style.

## Output runtime attesi

Quando il notebook viene eseguito, gli artifact principali vengono salvati in `output/` o nella directory configurata dal notebook:

- `Table_V_company_ranking.csv`
- `Table_V_sws_style_snowflake_scores.csv`
- `Table_VIII_intrinsic_value_estimates.csv`
- `Table_VIII_performance_summary.csv`
- `Table_XI_scenario_expected_returns.csv`
- `Table_XIII_robustness_checks.csv`
- `company_valuation_dashboard.html`
- `final_notebook_report.json`

## Uso rapido in Colab

Aprire il notebook:

```text
company_valuation/notebooks/Company_Valuatio.ipynb
```

Per usare FMP in Colab, configurare uno dei segreti:

- `FMP_API_KEY`
- `fmp_api_key`

Se il secret non è disponibile, il notebook tenta prima la cache locale e poi prosegue in modalità market-only senza interrompere il flusso.

## Uso locale

```bash
python -m pip install -r company_valuation/requirements.txt
jupyter notebook company_valuation/notebooks/Company_Valuatio.ipynb
```

I dataset pesanti devono restare fuori dal repo. Usa `DATA_PATH` per puntare alla cartella locale sincronizzata da Drive:

```bash
export ML_TRADING_DB_BASE="/Users/itsgennymac/Library/CloudStorage/GoogleDrive-sfn.gns@gmail.com/Il mio Drive/Database Finanziario"
```

Vedi anche [`../data/README_drive.md`](../data/README_drive.md).

## Nota metodologica

La sezione scorecard è ispirata al modello pubblico Simply Wall St / Company Analysis Model a livello metodologico: assi interpretabili per Value, Future, Past, Health e Income, implementati qui con controlli trasparenti sui dati disponibili nel notebook.

## Salvataggio su Google Drive

Da Colab, dopo `drive.mount('/content/drive')`, puoi salvare questa cartella e gli altri asset research con:

```bash
python scripts/sync_research_to_drive.py
```

Destinazione default:

```text
/content/drive/MyDrive/ml-trading-thesis-bot_research_exports/company_valuation
```

## Implementazione completa Simply Wall St-style

Il progetto ora include anche `company_valuation/src/sws_company_analysis_model.py`, una traduzione eseguibile in Python del modello documentato in `SimplyWallSt/Company-Analysis-Model`.

La logica implementa in modo auditabile:

- **Value**: fair value DCF/DDM/excess-returns/relative valuation, discount to fair value, PE, PEG e PB checks.
- **Future Performance**: crescita utili, crescita ricavi, high-growth threshold e ROE prospettico.
- **Past Performance**: EPS growth, ROE, ROCE e ROA.
- **Health**: checks separati per aziende non-finanziarie e financial institutions / banche.
- **Income / Dividends**: dividend yield, stabilità dividend per share, crescita dividend e payout coverage.
- **Management**: CEO compensation, allineamento compensi/performance, tenure management/board e insider buying/selling.

Esempio rapido:

```python
from company_valuation.src import SWSModelConfig, score_companies

config = SWSModelConfig(
    market_pe=18.0,
    industry_pe=14.0,
    industry_pb=1.2,
    discount_rate=0.09,
    terminal_growth=0.025,
)

scores, checks = score_companies(latest_cross_section, config=config)
```

Output principali:

- `scores`: una riga per società con score Value/Future/Past/Health/Income/Management.
- `checks`: audit trail check-by-check con pass/fail, disponibilità del dato, valore osservato e soglia.

Nota: il repository Simply Wall St documenta formule e soglie ma non fornisce codice Python. Questa implementazione è indipendente, trasparente e usa solo i dati disponibili nel progetto.

# Smart Money Government Data Engine

## Audit iniziale

La piattaforma usa come base primaria dati governativi e regolamentari ufficiali:

- SEC 13F per institutional holdings.
- SEC Form 4 / Forms 3, 4, 5 per insider activity.
- SEC EDGAR 13D/13G per beneficial ownership e activist pressure.
- CFTC COT per macro futures positioning.
- U.S. Treasury TIC per cross-border capital flow proxy.
- USAspending per government demand beneficiaries.
- ESMA FIRDS, ECB SDW, TED procurement e registri nazionali per EU/Italia.

Gap critici:

- 13F è trimestrale e ritardato, non trade tape.
- Form 4 richiede interpretazione di transaction code, piani e grant.
- TIC/COT sono proxy macro, non security-level trading.
- USAspending richiede entity resolution recipient-to-listed-parent.
- EU/Italia non hanno un EDGAR unico: copertura frammentata, connector paese per paese.

## Architettura proposta

Layer implementati:

- `src/smart_money_engine/schemas.py`: schema, source registry, matrici governance.
- `ingestion.py`: scansione Drive/cache e caricamento file ufficiali locali.
- `normalization.py`: normalizzazione 13F, Form 4, 13D/G, COT, TIC, USAspending.
- `resolution.py`: entity master e matching auditabile.
- `analytics.py`: scoring ownership, insider, activism, government spending, macro/flows.
- `visualization.py`: grafici Plotly per ranking, heatmap, COT/TIC, procurement.
- `reporting.py`: report HTML.
- `pipeline.py`: orchestrazione end-to-end senza network calls.

Output principali:

- `SmartMoney_smart_money_scores.csv`
- `SmartMoney_event_feed.csv`
- `SmartMoney_coverage.csv`
- `SmartMoney_source_registry.csv`
- `SmartMoney_entity_master.csv`
- `smart_money_government_data_report.html`

## Piano implementazione

1. MVP USA: 13F + Form 4 + 13D/G parser locali, coverage, screener.
2. Smart money scoring: componenti explainable e coverage-aware.
3. Dashboard: pagina Streamlit `8_Smart_Money_Gov_Data.py`.
4. EU expansion: ESMA FIRDS, ECB SDW, TED, CONSOB/AMF/BaFin/CNMV connectors separati.
5. Alerting: event feed, weekly report, thresholds e scheduler.

## Specifiche tecniche

Campi standard:

- `issuer_name`, `filer_name`, `cik`, `lei`, `isin`, `cusip`, `ticker`
- `country`, `sector`, `industry`
- `filing_date`, `report_date`, `source_form`
- `position_value_usd`, `shares`, `percent_portfolio`, `ownership_percent`
- `insider_role`, `transaction_code`
- `award_amount`, `procurement_agency`
- `asset_class`, `long_short_flag`, `instrument_type`

QA:

- Ogni dataset ha `source` e `source_path`.
- Se una fonte manca, la pipeline emette schema vuoto e coverage `MISSING`.
- Nessuno score viene inventato: il composite è media dei componenti disponibili.
- `coverage_count` e `coverage_note` indicano qualità del segnale.

## Caveat USA vs EU/Italia

USA:

- Ecosistema più centralizzato.
- SEC/CFTC/Treasury/USAspending consentono pipeline più standardizzate.

EU/Italia:

- ESMA supporta reference data, non ownership intelligence completa.
- ECB/TED sono proxy macro/procurement.
- Major holdings e insider dealing dipendono da registri nazionali.
- I segnali EU devono essere marcati come `complete`, `partial`, `proxy` o `unavailable`.

## Repository open-source → funzione nel progetto

| Repository | Funzione |
|---|---|
| QuantStats | HTML reporting, summary layout, monitor settimanali |
| pyfolio / pyfolio-reloaded | tear sheets, rolling diagnostics, caveat-first analytics |
| Riskfolio-Lib | concentration, overlap, contribution analysis |
| Ghostfolio | UX dashboard investimento, navigation, allocation views |
| Reverse-DCF-Data-Analysis | narrative valuation UX e scenario communication |
| Vibe-Trading | orchestration flow, auditabilità, research automation |

## Database governativo → use case → frequenza → limiti

La matrice completa è generata in `SmartMoney_government_dataset_matrix.csv`.

## MVP 30 giorni

La tabella operativa è generata in `SmartMoney_mvp_30_day_plan.csv`.

## Dataset priority ranking

La priorità è generata in `SmartMoney_dataset_priority_ranking.csv`.

## Quick wins analista

- Buy-side: rising 13F holders + insider net buying.
- Event-driven: 13D/13G activist feed.
- Macro: COT percentiles + TIC flow shifts.
- Industrial policy: USAspending acceleration.
- Portfolio manager: clone portfolios and crowdedness.
- EU/Italia: ESMA identifier map + TED procurement proxy + national disclosures.

# Memory Recovery Plan

Versione di lavoro per recuperare nel progetto v1.0 le idee migliori del precedente `ml-trading-thesis-bot`.

## Scopo

Investment Research Platform Pro ha gia' una struttura canonica, un Data Center e un LLM Lab. Il vecchio progetto `ml-trading-thesis-bot` va trattato come archivio metodologico: factor library, backtest, protocolli di validazione, notebook di tesi e casi studio.

Questo documento evita che la memoria della tesi originaria venga persa durante il refactor.

## Principi di recupero

- Recuperare concetti e formule prima dei file.
- Importare codice solo quando e' piu' robusto dell'attuale.
- Ogni modulo recuperato deve avere test, documentazione e posizione nel nuovo layout.
- I notebook del vecchio progetto diventano case study o reference, non nuove fonti operative parallele.

## Blocchi da recuperare o ricostruire

| Blocco | Stato atteso nel vecchio progetto | Target v1.0 |
|---|---|---|
| Factor taxonomy | Value, quality, momentum, size, low-volatility, profitability, investment | `src/research_platform/quant/factors/factor_definitions.py` |
| Factor registry | Nome, formula, frequenza, fonte, interpretazione | `src/research_platform/quant/factors/factor_registry.py` |
| IC / Rank IC | Validazione cross-sectional dei segnali | `src/research_platform/quant/factors/ic.py` |
| Factor quintile backtest | Spread Q5-Q1, turnover, costi | `src/research_platform/quant/factors/factor_backtest.py` |
| Walk-forward validation | Protocollo temporale anti-leakage | `src/research_platform/quant/ml/validation.py` |
| Data leakage checks | Controlli su feature future, normalizzazione e universi | `src/research_platform/quant/validation/leakage.py` |
| Performance attribution | Beta, fattori, active return, residual | `src/research_platform/quant/portfolio/attribution.py` |
| Thesis notebooks | Casi studio narrativi | `research_platform_definitive/docs/case_studies/` |
| Factor docs | Spiegazione famiglie fattoriali | `research_platform_definitive/docs/FACTOR_METHODS.md` |

## Memory Recovery Tasks

1. Recuperare o ricostruire modulo per IC/RankIC cross-sectional.
   - Done: funzioni `information_coefficient` e `rank_information_coefficient` testate su dati toy.

2. Reintrodurre documentazione fattoriale.
   - Done: documento con famiglie `value`, `quality`, `momentum`, `size`, `low_volatility`, `profitability`, `investment`.

3. Creare un factor registry.
   - Done: ogni fattore ha formula, frequenza, fonte, ipotesi economica e rischio.

4. Ricostruire factor quintile backtest.
   - Done: report con Q1-Q5, spread, Sharpe, turnover e costi.

5. Reintrodurre walk-forward validation.
   - Done: protocollo temporale usato dal ML Stock Lab e documentato in `ACADEMIC_METHODS.md`.

6. Recuperare notebook di tesi come case study.
   - Done: almeno un caso studio convertito in documento o notebook paper-ready nel nuovo layout.

7. Aggiungere leakage checklist automatizzabile.
   - Done: test o validator segnala feature future, target leakage e ranking non point-in-time.

8. Collegare factor outputs al Research Database.
   - Done: tabelle o viste per factor scores, IC history e factor backtest summary.

9. Ricreare performance attribution.
   - Done: report che separa benchmark, factor contribution e residual.

10. Creare issue GitHub per ogni blocco recuperato.
    - Done: milestone `v1.0.0` contiene task di memory recovery o collegamenti alle issue di fase.

## Path da cercare nel vecchio repo

Da compilare quando il vecchio repository viene riaperto:

| Vecchio path | Cosa contiene | Decisione |
|---|---|---|
| `24_alpha_factor_library/` | factor library e notebook fattoriali | Da mappare |
| `backtest/` o simili | engine backtest e metriche | Da mappare |
| notebook di tesi | casi studio e testo accademico | Da convertire |
| validation utilities | IC, RankIC, leakage checks | Da recuperare |

## Regola finale

Un blocco recuperato entra in v1.0 solo se ha:

- modulo nel nuovo layout;
- test minimo;
- documentazione;
- esempio d'uso;
- collegamento a notebook o report.

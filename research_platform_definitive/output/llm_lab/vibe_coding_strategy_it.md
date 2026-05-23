# vibe_coding_strategy_it

Generated: 2026-05-23T18:20:59.272527+00:00

Genera una strategia Python compatibile con:

from integrations.vibe_trading_bridge.strategy_adapter import StrategyBase, StrategyContext

Idea utente:
Migliora il blend momentum/z-score con controllo volatilita' e costi realistici.

Universo: AAPL,MSFT,NVDA,SPY,ENEL.MI,ISP.MI,ASML.AS,SAP.DE
Finestra backtest: 2021-01-04 -> 2026-05-22

Requisiti:
- codice semplice e leggibile;
- nessun path locale hard-coded;
- usa solo dati disponibili fino a context.date;
- spiega formula, parametri, filtri, ribilanciamento e risk management nella docstring;
- aggiungi uno snippet di test con run_backtest;
- output finale: codice, assunzioni, limiti e metriche attese.

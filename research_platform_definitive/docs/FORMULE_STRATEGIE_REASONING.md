# Formule, Strategie e Reasoning

Questa nota spiega le formule principali in modo leggibile. Le API e il codice restano in inglese, ma il reasoning e' volutamente in italiano semplice.

## Rendimenti

Rendimento semplice:

```text
r_t = P_t / P_{t-1} - 1
```

Rendimento a 21 sedute:

```text
r_{21,t} = P_t / P_{t-21} - 1
```

Uso: misura quanto si e' mosso lo strumento. Il rendimento semplice e' intuitivo e si combina bene con backtest discreti.

Limite: non dice nulla sul rischio preso per ottenere quel rendimento.

## Momentum

```text
momentum_{63,t} = P_t / P_{t-63} - 1
```

Intuizione: se un titolo ha salito negli ultimi tre mesi, una parte del movimento puo' continuare per inerzia informativa, flussi o revisione delle aspettative.

Uso nel baseline:

```text
score_i,t = momentum_{63,i,t}
w_i,t = max(score_i,t, 0) / sum_j max(score_j,t, 0)
```

Limiti:

- puo' comprare dopo un movimento gia' maturo;
- soffre inversioni rapide;
- aumenta turnover se il ranking cambia spesso.

## Z-score di prezzo

```text
z_{63,t} = (P_t - SMA_63(P_t)) / std_63(P_t)
```

Intuizione: misura quanto il prezzo e' lontano dalla propria media recente in unita' di volatilita'. Uno z-score alto segnala forza o eccesso; uno z-score basso segnala debolezza o possibile mean reversion.

Nel progetto lo usiamo nel blend come segnale di forza normalizzata:

```text
raw_i,t = 0.60 * zscore_i,t + 0.40 * momentum_i,t / volatility_i,t
```

Limite: da solo puo' confondere un vero cambio di regime con un eccesso temporaneo.

## Volatilita'

```text
vol_{21,t} = std(r_{1d,t-20:t}) * sqrt(252)
```

Intuizione: stima il rischio annualizzato recente. Nel sizing, strumenti piu' volatili ricevono meno peso.

Uso:

```text
weight_i,t proportional max(raw_i,t, 0) / vol_{21,i,t}
```

Limite: la volatilita' passata non garantisce quella futura; dopo shock improvvisi puo' arrivare in ritardo.

## Risk Balanced Blend

Strategia finale sample:

```text
raw_i,t = 0.60 * zscore_{63,i,t} + 0.40 * momentum_{63,i,t} / vol_{21,i,t}
positive_i,t = max(raw_i,t, 0)
w_i,t = positive_i,t / sum_j positive_j,t
w_i,t = min(w_i,t, 25%)
w_i,t = w_i,t / sum_j w_j,t
```

Reasoning:

- momentum evita di comprare solo perche' qualcosa e' sceso;
- z-score normalizza la forza rispetto alla storia recente;
- volatilita' riduce concentrazione su strumenti rumorosi;
- cap 25% evita che un singolo titolo domini l'universo.

Assunzioni:

- il ranking cross-section contiene informazione;
- il costo di transazione e' abbastanza basso;
- l'universo non e' troppo piccolo o totalmente correlato.

## Backtest

Pseudocodice:

```text
for each date t:
    compute features using data <= t
    compute target weights w_t
    apply weights at t+1
    portfolio_return_t = sum_i w_{i,t-1} * r_{i,t}
    cost_t = turnover_t * transaction_cost_bps / 10000
    net_return_t = portfolio_return_t - cost_t
    equity_t = equity_{t-1} * (1 + net_return_t)
```

Metriche:

```text
total_return = equity_T / equity_0 - 1
annualized_volatility = std(net_returns) * sqrt(252)
sharpe = mean(net_returns) / std(net_returns) * sqrt(252)
drawdown_t = equity_t / max(equity_0...equity_t) - 1
max_drawdown = min(drawdown_t)
turnover_t = sum_i abs(w_i,t - w_i,t-1)
```

Limiti:

- sample sintetico non sostituisce dati reali;
- il backtest non considera slippage complesso;
- metriche aggregate non bastano: servono walk-forward, ablation e stress test.

## Risk management friendly

Regole consigliate prima di promuovere una strategia:

- nessun peso singolo sopra 25%;
- costi espliciti in bps;
- turnover monitorato;
- confronto con `momentum_baseline`;
- nessun uso di feature future;
- log esperimento salvato in `experiment_logs`.

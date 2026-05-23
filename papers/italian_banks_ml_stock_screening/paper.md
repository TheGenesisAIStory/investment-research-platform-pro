# Machine Learning per la Valutazione e lo Stock Screening delle Banche Italiane Quotate

## 1. Introduzione

Il settore bancario italiano ha attraversato fasi di crisi, ristrutturazione e rafforzamento patrimoniale, con effetti importanti sui prezzi azionari e sulla percezione del rischio. La complessita' dei bilanci bancari, la regolamentazione prudenziale e il peso delle esposizioni sovrane rendono utile verificare se i fondamentali di bilancio si riflettano nei rendimenti azionari delle banche italiane.

Il paper propone un framework di machine learning per:

- stimare fair value peer-implied orientati ai ratio bancari;
- misurare mispricing relativo e z-score cross-sectional;
- costruire stock screener con filtri regolamentari e ranking multifattoriali;
- prevedere rendimenti futuri con modelli ML;
- testare portafogli long-only, quintile e long-short su banche italiane quotate.

## 2. Contributi

1. Specializzare l'agnostic fundamental analysis al settore bancario italiano.
2. Confrontare OLS, LASSO, Random Forest e Gradient Boosting nella stima del fair value.
3. Costruire uno screener bancario basato su mispricing, CET1, NPL e liquidita'/funding.
4. Valutare se ratio bancari e mispricing spiegano rendimenti futuri meglio di benchmark lineari o di mercato.

## 3. Dati e Universo Empirico

Universo: principali banche italiane quotate, ad esempio Intesa Sanpaolo, UniCredit, Banco BPM, BPER Banca, Monte dei Paschi di Siena e Credito Emiliano. Il peer set puo' essere esteso a banche europee per aumentare la profondita' cross-sectional.

Periodo target: 2000-2025, con sottoperiodi per robustezza:

- pre-crisi finanziaria globale;
- crisi sovrana 2011-2012;
- post-2012;
- Covid;
- post-2020.

Frequenza:

- fondamentali annuali o trimestrali, laggati per evitare look-ahead bias;
- prezzi giornalieri o mensili, aggregati a rendimenti mensili.

## 4. Variabili

Feature fondamentali bancarie:

- redditivita': ROA, ROE, net interest margin, cost/income ratio;
- funding/liquidita': loan-to-deposit ratio, proxy LCR/NSFR, funding wholesale;
- qualita' del credito: NPL ratio, coverage ratio, cost of risk, UTP;
- capitale: CET1, Tier1, Total Capital ratio, leverage ratio;
- esposizione sovrana: BTP/assets, duration del portafoglio titoli;
- dimensione e business mix: log total assets, commissioni/ricavi, retail vs corporate.

Feature di mercato:

- multipli: P/B, P/E, dividend yield;
- rischio: beta vs indice bancario, volatilita' storica, drawdown;
- liquidita': turnover e proxy di bid-ask spread.

Target:

- `target_log_mcap`: log della market capitalization;
- `target_ret_1m_fwd`: rendimento futuro mensile.

## 5. Modello di Fair Value Peer-Implied

Per banca `i` e data `t`:

```text
Y_i,t = log(MarketCap_i,t)
Yhat_i,t = f_t(X_i,t)
FairValue_i,t = exp(Yhat_i,t)
```

Modelli testati:

- OLS;
- LASSO;
- Random Forest;
- Gradient Boosting Regression Trees.

Il mispricing relativo e':

```text
MP_i,t = (FairValue_i,t - MarketValue_i,t) / MarketValue_i,t
```

Lo z-score viene calcolato cross-sectionalmente per data. Un segnale ensemble combina RF e GBRT tramite media degli z-score.

## 6. Screening Bancario

La screening function applica filtri hard:

```text
screen_pass = 1[CET1 >= soglia] * 1[NPL <= soglia] * 1[liquidita' >= soglia]
```

Lo score finale:

```text
score_final = screen_pass * score_fund
```

dove `score_fund` puo' essere il mispricing ensemble o uno score ML di rendimento atteso.

## 7. Strategie

Strategie da testare:

- quintile portfolios ordinati per mispricing ensemble;
- long-short Q5-Q1;
- screening long-short dopo filtri CET1/NPL/liquidita';
- prediction-sorted portfolios basati su expected return ML.

Metriche:

- rendimento medio e annualizzato;
- volatilita' annualizzata;
- Sharpe ratio;
- max drawdown;
- turnover;
- alpha fattoriale se sono disponibili fattori di benchmark.

## 8. Ipotesi

- H1: RF/GBRT stimano mispricing piu' informativo di OLS e semplici multipli come P/B.
- H2: redditivita', liquidita' e asset quality mantengono potere predittivo sui rendimenti bancari italiani.
- H3: lo screener con mispricing e filtri regolamentari produce performance risk-adjusted superiori a strategie value/momentum semplici.
- H4: modelli ML di expected returns ottengono R2 out-of-sample positivo ma moderato, dato l'universo ristretto.
- H5: un eventuale layer di timing tecnico puo' migliorare il profilo rischio/rendimento, ma e' sensibile ai costi di transazione.

## 9. Notebook di Test

Il notebook associato implementa:

1. setup e import;
2. configurazione dell'esperimento;
3. caricamento e preprocessing panel;
4. fair value e mispricing OLS/RF/GBRT;
5. ensemble mispricing e quintile portfolios;
6. screening bancario con soglie regolamentari;
7. expected returns con Random Forest;
8. valutazione performance e alpha;
9. funzione wrapper per rieseguire la pipeline cambiando parametri.

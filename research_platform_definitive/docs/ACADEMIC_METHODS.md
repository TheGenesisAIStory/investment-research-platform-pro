# Academic Methods

Versione di lavoro per Investment Research Platform Pro v1.0.

Questo documento raccoglie la metodologia accademica della piattaforma. Il suo scopo e' rendere esplicite le ipotesi quantitative che nel codice e nei notebook sono gia' in parte implementate: modelli di valutazione, costruzione di portafoglio, ricerca ML, governance dati e uso assistito degli LLM.

## 1. Introduzione

Investment Research Platform Pro e' una piattaforma notebook-first e data-center-first per ricerca su equity, portafogli, segnali ML e workflow LLM-assisted. La piattaforma combina:

- app Streamlit per workstation e Data/API control;
- moduli Python riusabili per valuation, portfolio, ML e data engineering;
- un Research Database SQLite rigenerabile per strumenti, OHLCV, feature, segnali, backtest ed esperimenti;
- un LLM Lab offline-first per prompt packet, review e documentazione assistita.

La piattaforma non produce raccomandazioni operative. Produce artefatti di ricerca, diagnostiche e report verificabili. Ogni risultato deve essere interpretato come output di ricerca condizionato a dati, ipotesi, costi e limiti dichiarati.

## 2. Valuation Framework

### 2.1 Domanda di ricerca

La domanda centrale della componente valuation e': il prezzo corrente di uno strumento equity e' coerente con i flussi di cassa attesi, il rischio dell'impresa e il confronto con societa' comparabili?

Il valore stimato deve essere espresso come intervallo, non come punto singolo. La piattaforma combina metodi assoluti, metodi relativi e scenari probabilistici.

### 2.2 Discounted Cash Flow

Il DCF assume che il valore dell'impresa derivi dai flussi di cassa futuri attualizzati a un tasso coerente con il rischio operativo e finanziario.

Formula base:

```text
EV_0 = sum_{t=1}^{T} FCFF_t / (1 + WACC)^t + TV_T / (1 + WACC)^T
EquityValue_0 = EV_0 - NetDebt
FairValuePerShare = EquityValue_0 / SharesOutstanding
```

Terminal value Gordon growth:

```text
TV_T = FCFF_{T+1} / (WACC - g)
```

Il terminal growth `g` deve essere inferiore al WACC e coerente con crescita nominale di lungo periodo, inflazione, paese e maturita' del business.

### 2.3 Costo del capitale

Il costo dell'equity puo' essere stimato con CAPM:

```text
r_e = r_f + beta * ERP + CRP
```

dove:

- `r_f` e' il tasso risk-free coerente con valuta e durata;
- `beta` misura esposizione sistematica;
- `ERP` e' il premio per il rischio azionario;
- `CRP` e' un eventuale premio paese.

Il WACC e':

```text
WACC = E / (D + E) * r_e + D / (D + E) * r_d * (1 - TaxRate)
```

Ogni notebook valuation deve dichiarare fonte e data del risk-free rate, ipotesi su ERP, trattamento della leva, tax rate e valuta.

### 2.4 Multipli e peer group

I metodi relativi confrontano lo strumento con un gruppo di peer:

```text
FairValue_i = Metric_i * Multiple_peer
```

Esempi: `P/E`, `P/B`, `EV/EBITDA`, `EV/Sales`. Il peer group deve essere motivato per settore, area geografica, crescita, marginalita', leva e qualita' del business. La piattaforma deve evidenziare quando il peer group e' manuale, automatico o incompleto.

### 2.5 Reverse DCF

Il reverse DCF risponde alla domanda opposta: quali crescita e margini sono impliciti nel prezzo corrente?

Testo modello:

> Il reverse DCF non stima un fair value indipendente; traduce il prezzo di mercato in aspettative implicite. Se le ipotesi implicite appaiono incompatibili con dati storici, settore o vincoli macroeconomici, il prezzo puo' essere considerato aggressivo o conservativo rispetto allo scenario analizzato.

### 2.6 Monte Carlo e sensibilita'

Il DCF Monte Carlo campiona ipotesi su crescita, WACC, terminal growth e margini. L'output corretto e' una distribuzione:

- media e mediana del fair value;
- percentili 5/25/75/95;
- probabilita' di upside/downside rispetto al prezzo corrente;
- sensitivita' a WACC e terminal growth.

Limiti principali: instabilita' dei forecast, qualita' dei fondamentali, scelta delle distribuzioni e rischio di false precision.

## 3. Portfolio Construction & Risk

### 3.1 Domanda di ricerca

La componente portfolio risponde alla domanda: quale allocazione massimizza rendimento atteso per unita' di rischio sotto vincoli realistici e costi espliciti?

La piattaforma deve confrontare sempre strategie ottimizzate con baseline semplici: equal-weight, benchmark e/o allocazione risk-adjusted interna.

### 3.2 Rendimenti e rischio

Rendimento di portafoglio:

```text
R_{p,t} = sum_i w_{i,t-1} * R_{i,t}
```

Rendimento atteso e varianza:

```text
E[R_p] = w' * mu
sigma_p^2 = w' * Sigma * w
```

Sharpe ratio:

```text
Sharpe = (E[R_p] - r_f) / sigma_p
```

Drawdown:

```text
DD_t = Equity_t / max(Equity_0 ... Equity_t) - 1
```

### 3.3 Vincoli e implementabilita'

La ricerca di portafoglio deve dichiarare:

- long-only o short-enabled;
- peso massimo per singolo strumento;
- leverage massimo;
- turnover limit;
- transaction cost bps;
- benchmark e tracking error target, se presenti.

Un portafoglio e' research-valid solo se la soluzione e' implementabile sotto questi vincoli.

### 3.4 Downside risk e CVaR

Il CVaR misura la perdita media condizionata alla coda sinistra:

```text
CVaR_alpha = E[Loss | Loss >= VaR_alpha]
```

Va usato quando la distribuzione dei rendimenti e' asimmetrica o fat-tailed. Il CVaR non sostituisce drawdown e stress test; li integra.

### 3.5 Benchmark e attribution

Ogni analisi deve confrontare il portafoglio con benchmark investibile:

```text
ActiveReturn_t = R_{p,t} - R_{b,t}
TrackingError = std(ActiveReturn_t) * sqrt(252)
InformationRatio = mean(ActiveReturn_t) / std(ActiveReturn_t) * sqrt(252)
```

Estensione v1.0 consigliata: attribution tra market beta, fattori e stock-specific return.

## 4. ML Equity Research

### 4.1 Domanda di ricerca

Il ML Stock Lab testa se feature fondamentali, tecniche o fattoriali contengono informazione predittiva sui rendimenti futuri o sul mispricing relativo.

Il risultato principale non e' solo un modello, ma una catena verificabile: panel dati, feature, label, split temporale, metriche, segnali e robustness.

### 4.2 Target e label

Forward return:

```text
r_{i,t+h} = P_{i,t+h} / P_{i,t} - 1
```

Label direzionale:

```text
y_{i,t+h} = 1 if r_{i,t+h} > 0 else 0
```

Ogni target deve dichiarare horizon, frequenza, trattamento dei missing, corporate actions e disponibilita' point-in-time.

### 4.3 Feature e fattori

Esempi di feature:

- momentum;
- value;
- quality;
- profitability;
- low volatility;
- size;
- revisioni o segnali alternativi, se disponibili.

Ogni feature deve avere formula, fonte, data availability e ragione economica.

### 4.4 Validazione temporale

La validazione deve rispettare l'ordine informativo:

- train/test split temporale;
- walk-forward validation;
- nessuna normalizzazione usando dati futuri;
- nessun ranking cross-section calcolato con informazioni future.

OOS R-squared:

```text
OOS_R2 = 1 - SSE_model / SSE_benchmark
```

### 4.5 Signal evaluation

Information Coefficient:

```text
IC_t = corr(signal_{i,t}, return_{i,t+h})
```

Rank IC:

```text
RankIC_t = corr(rank(signal_{i,t}), rank(return_{i,t+h}))
```

Quintile spread:

```text
Spread_t = Return_Q5,t - Return_Q1,t
```

Le metriche devono essere lette insieme a turnover, drawdown, costi e stabilita' per periodo.

### 4.6 Model risk

Rischi da dichiarare:

- lookahead bias;
- survivorship bias;
- data snooping;
- multiple testing;
- overfitting;
- regime dependence;
- instabilita' del provider dati.

## 5. Data Center & Data/API Governance

### 5.1 Ruolo del Data Center

Il Data Center e' il contratto operativo della piattaforma. Deve rendere riproducibili strumenti, prezzi, feature, segnali, backtest e log esperimenti.

Lo schema canonico include: `instruments`, `ohlcv_daily`, `features_labels`, `ml_signals`, `backtest_results`, `backtest_equity`, `experiment_logs`, `strategy_registry`, `ingestion_runs`, `data_catalog_entries`.

### 5.2 Provenance

Ogni dataset deve dichiarare:

- fonte;
- timestamp di ingestione;
- frequenza;
- chiave primaria;
- trasformazioni applicate;
- stato `synthetic`, `sample`, `live` o `derived`.

### 5.3 Freshness e fallback

La policy di freshness deve indicare quando un dato e' stale e quale fallback usare. Un dato stale non invalida automaticamente un'analisi, ma deve essere segnalato nel report.

### 5.4 Bias e data quality

Controlli minimi:

- copertura per ticker e date;
- missingness;
- outlier;
- duplicati;
- corporate actions;
- survivorship bias;
- lookahead negli universi.

### 5.5 Synthetic data

I dati sintetici sono ammessi per demo, test e sviluppo. Non devono essere presentati come evidenza empirica. Ogni output generato da dati sintetici deve dichiararlo in apertura.

## 6. LLM-Assisted Research

### 6.1 Ruolo dell'LLM

L'LLM e' un assistente di ricerca. Trasforma dati strutturati in memo, checklist, spiegazioni e bozze di codice ispezionabile. Non e' un decisore e non produce raccomandazioni operative.

### 6.2 Prompt packet

Il prompt packet v1.0 deve contenere:

- `manifest.json`;
- `00_system_policy.md`;
- prompt core per data sanity, equity research, portfolio diagnostics, strategy review, backtest diagnostics, risk scenarios, code review e academic methods.

Ogni prompt deve dichiarare ruolo, input obbligatori, output atteso, fonti richieste, limiti e rischio.

### 6.3 Policy di incertezza e responsabilita'

L'assistente deve distinguere risultati calcolati, inferenze qualitative e informazioni non verificate. Ogni affermazione su dati, metriche o modelli deve indicare la fonte usata o dichiarare che la fonte non e' disponibile.

L'output non costituisce raccomandazione di investimento, consulenza finanziaria o istruzione operativa. Quando i dati sono sintetici, incompleti, stale o derivati da fallback, l'assistente deve segnalarlo in apertura. Ogni conclusione deve includere almeno un controllo consigliato prima dell'uso in ricerca o reportistica.

### 6.4 Output schema e verifica umana

Ogni risposta LLM deve includere:

- sintesi;
- dati usati;
- metodologia;
- risultati o interpretazione;
- incertezza;
- rischi;
- checklist di verifica.

La review umana e' obbligatoria prima di includere output LLM in documenti finali.

## 7. Collegamento con ml-trading-thesis-bot

Il vecchio progetto `ml-trading-thesis-bot` rappresenta la memoria metodologica della piattaforma: factor library, protocolli di backtest, validazione IC/RankIC e notebook di tesi.

Per la v1.0, il recupero deve concentrarsi su:

- factor taxonomy;
- IC e Rank IC;
- factor quintile backtest;
- walk-forward validation;
- thesis-style case studies.

Questi blocchi devono essere reinseriti in `research_platform/quant/factors/`, `research_platform/quant/ml/validation.py`, `research_platform/quant/portfolio/attribution.py` e `docs/case_studies/`.

## 8. Notebook Paper-Ready Contract

Ogni notebook canonico deve contenere:

1. domanda di ricerca;
2. dataset e ipotesi;
3. metodologia;
4. risultati;
5. robustezza;
6. limiti;
7. link ai moduli Python usati.

Il notebook deve orchestrare e spiegare. La logica core deve vivere in libreria.

## 9. Reproducibility Checklist

Prima di una release:

- rigenerare Research Database;
- eseguire test e validator;
- rigenerare prompt packet;
- verificare notebook canonici;
- aggiornare `PROJECT_STATUS_FINAL.md`;
- sincronizzare Drive;
- pubblicare tag GitHub.

## 10. Limiti Generali

La piattaforma e' progettata per ricerca e didattica avanzata. I risultati dipendono da dati, provider, assunzioni, sample period e modelli. Nessun output deve essere interpretato come consiglio finanziario o sistema automatico di trading.

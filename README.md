# ML Trading Thesis Bot

> **Autore:** Stefano Genise — Master II Livello in Data Science, Finance Data Science  
> **Repo:** [TheGenesisAIStory/ml-trading-thesis-bot](https://github.com/TheGenesisAIStory/ml-trading-thesis-bot)  
> **Stato:** Blocco operativo completato fino al capitolo `24_alpha_factor_library` · Database locale attivo

Questo progetto è la base pratica della mia tesi magistrale su **Machine Learning for Trading**. Copre l'intera pipeline end-to-end: dalla raccolta e ingegnerizzazione dei dati di mercato, alla costruzione di modelli ML/DL, fino al backtest e alla valutazione delle strategie algoritmiche.

Il lavoro è stato interamente modernizzato su **Python 3.11** con stack aggiornato (2025/2026), rimuovendo tutte le dipendenze legacy incompatibili con gli ambienti attuali.

---

## Struttura del Progetto

Il repo contiene **oltre 164 notebook eseguiti**, organizzati in 24 capitoli tematici:

| Parte | Capitoli | Focus |
|---|---|---|
| **I — Dati e Strategie** | 01–05 | Market data, feature engineering, portfolio management |
| **II — ML Fondamentale** | 06–13 | Modelli supervisionati, non supervisionati, time series |
| **III — NLP per il Trading** | 14–16 | Sentiment analysis, topic modeling, word embeddings |
| **IV — Deep & RL** | 17–24 | CNN, RNN, Autoencoder, GAN, Reinforcement Learning |

In più, il workspace master include:

- `00_notebooks_final/`: notebook finali integrati dal repo `ml-trading-thesis-bot-final`;
- `src/analysis/`: motori Analysis Studio riusabili da notebook, CLI, dashboard e scheduler;
- `dashboard/`: dashboard Streamlit multipagina per le 8 analisi professionali;
- `company_valuation/notebooks/Company_Valuatio.ipynb`: percorso canonico del notebook Company Valuation;
- `pead_european_banks_ifrs9/`: esperimento PEAD European Banks / IFRS9;
- `data/README_drive.md`: istruzioni per collegare i database pesanti su Google Drive senza versionarli;
- `scripts/sync_drive_data.sh`: sync locale via `rclone` verso la cartella dati esterna.

---

## Database e Google Drive

I database pesanti non devono essere committati su GitHub. La convenzione comune per notebook locali e Colab è:

```python
import os
from pathlib import Path

DB_BASE = Path(os.environ.get(
    "ML_TRADING_DB_BASE",
    "/Users/itsgennymac/Library/CloudStorage/GoogleDrive-sfn.gns@gmail.com/Il mio Drive/Database Finanziario",
))
DATA_PATH = DB_BASE
```

In Colab, `DATA_PATH` punta normalmente a:

```text
/content/drive/MyDrive/Database Finanziario
```

Guida completa: [`data/README_drive.md`](data/README_drive.md).

Per sincronizzare in locale:

```bash
bash scripts/sync_drive_data.sh
```

---

## Analysis Studio

Analysis Studio aggiunge 8 workflow professionali eseguibili da notebook, terminale, dashboard e scheduler:

```bash
python -m src.cli.analysis_commands screen --universe sp500 --sector technology
python -m src.cli.analysis_commands risk --portfolio-file positions.csv
python -m src.cli.analysis_commands earnings --ticker NVDA
python -m src.cli.analysis_commands build --risk-profile moderate
python -m src.cli.analysis_commands technical --ticker AAPL
python -m src.cli.analysis_commands competitive --sector semiconductors
python -m src.cli.analysis_commands quant --ticker MSFT
python -m src.cli.analysis_commands macro --portfolio-file positions.csv
```

Dashboard:

```bash
streamlit run dashboard/Analysis_Studio.py
```

Aggiornamento batch:

```bash
python scripts/weekly_update.py --offline
```

Documentazione:

- [`docs/ANALYSIS_STUDIO.md`](docs/ANALYSIS_STUDIO.md)
- [`docs/NOTEBOOKS_USAGE.md`](docs/NOTEBOOKS_USAGE.md)

---

## Database Locale di Mercato

Il progetto include un database locale SQLite con dati di mercato reali scaricati da Yahoo Finance.

### Smoke Test (rapido)
```bash
python3.11 data/market_data_database.py --mode smoke --years 5 --max-symbols 8
```

### Full Build (7 anni, universe completo)
```bash
python3.11 data/market_data_database.py --mode full --years 7
```

Con `FMP_API_KEY` nell'ambiente aggiunge automaticamente `global_top_500`.  
Senza API key usa il seed universe: **S&P 500, Euro STOXX 50, FTSE MIB, ETF, FX e crypto**.

**Risultati smoke test verificati:**
- Strumenti: `8`
- Righe prezzi: `10.751`
- Simboli Yahoo: `8`
- Fallback demo: `0`

> I dati generati sono esclusi da git via `.gitignore` (`data/local_market_data/`).

---

## Stack Tecnico Modernizzato

**Rimossi** tutti i runtime legacy incompatibili:
`PyMC3`, `Theano`, `Zipline`, `Alphalens`, `TA-Lib`, `LightGBM`, `CatBoost`, `SHAP`, `TensorFlow`, `PyTorch`, `Gym`, `spaCy`, `TextBlob`, `Gensim`

**Sostituiti** con alternative moderne compatibili Python 3.11 — ogni capitolo ha il suo `requirements.txt` e modulo helper dedicato.

---

## Installazione

```bash
git clone https://github.com/TheGenesisAIStory/ml-trading-thesis-bot.git
cd ml-trading-thesis-bot
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # oppure usa il requirements.txt del singolo capitolo
```

> Consigliato installare le dipendenze capitolo per capitolo per evitare conflitti di versione.

---

## Verifiche di Qualità

| Check | Risultato |
|---|---|
| Notebook verificati | ✅ 164 |
| Error output nei notebook | ✅ 0 |
| Python files compilati | ✅ 55 |
| Compile errors | ✅ 0 |
| Notebook database eseguito via nbconvert | ✅ |

---

## Contenuto per Capitolo

### Parte I — Dati e Strategia

#### 01 Machine Learning for Trading: Introduzione
Trend del settore, posizionamento dell'ML nel processo di investimento, casi d'uso principali.

#### 02 Market & Fundamental Data
Dati tick NASDAQ, minute bar Algoseek, order book, XBRL filings SEC, API dati finanziari.

#### 03 Alternative Data
Categorie e fonti di dati alternativi, scraping earnings call transcripts, sentiment da dati non strutturati.

#### 04 Financial Feature Engineering — Alpha Factors
Indicatori tecnici, Kalman filter, wavelet denoising, costruzione e valutazione di alpha factor.

#### 05 Portfolio Optimization
Mean-variance optimization, hierarchical risk parity, valutazione performance con pyfolio.

### Parte II — ML Fondamentale

#### 06 The Machine Learning Process
Workflow ML completo: training, tuning, cross-validation time-series, bias-variance tradeoff.

#### 07 Linear Models
Regressione lineare, Ridge, Lasso, regressione logistica per previsione rendimenti e direzione.

#### 08 ML4T Workflow — Backtesting
Backtest vettorizzato ed event-driven con backtrader, pipeline ML integrata nel backtest.

#### 09 Time Series Models
ARIMA, GARCH, VAR, cointegrazione, pairs trading statistico.

#### 10 Bayesian ML
Programmazione probabilistica, dynamic Sharpe ratio, rolling regression bayesiana, volatilità stocastica.

#### 11 Random Forests
Decision tree, random forest, long-short strategy su azioni giapponesi.

#### 12 Gradient Boosting
XGBoost, LightGBM, CatBoost, SHAP values, strategia intraday ad alta frequenza.

#### 13 Unsupervised Learning
PCA, ICA, t-SNE, UMAP, k-means, hierarchical clustering, hierarchical risk parity.

### Parte III — NLP per il Trading

#### 14 Text Data & Sentiment Analysis
Pipeline NLP, document-term matrix, Naive Bayes, analisi sentiment su news e social.

#### 15 Topic Modeling
LSI, pLSA, LDA con scikit-learn e gensim, analisi earnings call e news finanziarie.

#### 16 Word Embeddings
Word2Vec, GloVe, Doc2Vec, embeddings da SEC filings per previsione ritorni, BERT fine-tuning.

### Parte IV — Deep Learning & Reinforcement Learning

#### 17 Deep Learning
Feedforward NN, backpropagation, TensorFlow 2 e PyTorch, ottimizzazione architettura per trading.

#### 18 CNN
Convolutional NN per time series e immagini satellitari, transfer learning, classificazione attività economica.

#### 19 RNN
LSTM, GRU, time series multivariate, sentiment analysis con embeddings pretrainati.

#### 20 Autoencoders
Autoencoder per risk factors condizionali, replica paper AQR (Gu, Kelly, Xiu 2019).

#### 21 GAN
Generative Adversarial Networks per dati sintetici, replica TimeGAN (NeurIPS 2019).

#### 22 Deep Reinforcement Learning
MDP, Q-learning, DDQN, trading agent con OpenAI Gym, ottimizzazione long-term.

#### 23 Conclusioni
Sintesi del workflow ML4T, prossimi passi, integrazione ML nel processo di investimento.

#### 24 Alpha Factor Library (Appendice)
Oltre 100 alpha factor tecnici (TA-Lib) e formulaici (WorldQuant 101 Alphas), valutazione con IC, SHAP, Alphalens.

---

## Configurazione Variabili d'Ambiente

Copia `.env.example` in `.env` e compila le chiavi necessarie:

```bash
cp .env.example .env
```

```env
FMP_API_KEY=la_tua_chiave    # Financial Modeling Prep (opzionale, per universe esteso)
```

---

## Note Git

- `.DS_Store` escluso da tutti i commit
- `data/local_market_data/` escluso via `.gitignore` (dati pesanti locali)
- database Drive e output Analysis Studio sincronizzati in `/Users/itsgennymac/Library/CloudStorage/GoogleDrive-sfn.gns@gmail.com/Il mio Drive/Database Finanziario`
- File `.env` mai committato — usa solo `.env.example`

---

## Licenza

Questo progetto è originale. Il codice è distribuito sotto licenza **MIT**.  
I dati di mercato sono soggetti ai termini dei rispettivi provider (Yahoo Finance, FMP).

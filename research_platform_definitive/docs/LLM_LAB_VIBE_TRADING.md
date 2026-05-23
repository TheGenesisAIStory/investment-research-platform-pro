# Laboratorio LLM / Vibe Trading

Il Laboratorio LLM e' offline-first: genera prompt italiani, controlla quali provider sono configurati e prepara pacchetti per vibe coding/trading senza obbligare a chiamare subito un'API esterna.

## Comandi

Lista template:

```bash
python3 research_platform_definitive/scripts/llm_lab_cli.py templates
```

Provider readiness:

```bash
python3 research_platform_definitive/scripts/llm_lab_cli.py providers
```

Render di un prompt:

```bash
python3 research_platform_definitive/scripts/llm_lab_cli.py render strategy_review_it
```

Pacchetto completo:

```bash
python3 research_platform_definitive/scripts/llm_lab_cli.py packet
```

Output:

```text
research_platform_definitive/output/llm_lab/
```

## Template disponibili

| Template | Uso |
|---|---|
| `strategy_review_it` | review strategia con segnali, backtest, rischi e prossimi esperimenti |
| `vibe_coding_strategy_it` | generazione di una `StrategyBase` compatibile con Vibe-Trading |
| `backtest_diagnostic_it` | diagnostica di metriche, equity curve e fragilita' |
| `data_quality_brief_it` | brief Data Center su copertura, sample e query |
| `quantdinger_mobile_summary_it` | sintesi breve per app mobile QuantDinger-style |

## Provider

Il registry legge solo variabili ambiente:

- `OPENAI_API_KEY`, `OPENAI_MODEL`
- `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL`
- `VIBE_TRADING_ROOT`

Se nessun provider cloud e' configurato, il laboratorio resta utile: produce prompt e contesto pronti per revisione manuale o per essere incollati in un tool esterno.

## Bridge con Vibe-Trading

Per generare codice strategia:

```bash
python3 scripts/vibe_cli.py generate-strategy \
  --prompt "Blend momentum e z-score con controllo volatilita" \
  --universe AAPL,MSFT,NVDA \
  --output-path research_platform_definitive/output/llm_lab/generated_strategy.py \
  --no-vibe
```

Per backtest da prompt:

```bash
python3 scripts/vibe_cli.py backtest-from-prompt \
  --prompt "Long only sui migliori score momentum risk-adjusted" \
  --start 2021-01-04 \
  --end 2026-05-22 \
  --universe AAPL,MSFT,NVDA \
  --output-dir research_platform_definitive/output/llm_lab/backtest_demo \
  --no-vibe
```

## Regola di sicurezza

Il laboratorio non deve trasformare output LLM in trading reale. Ogni strategia generata deve passare:

- controllo formule;
- test senza lookahead;
- backtest con costi;
- confronto baseline;
- review umana prima di promozione.

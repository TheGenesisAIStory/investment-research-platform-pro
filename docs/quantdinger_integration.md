# QuantDinger Integration

This integration keeps QuantDinger upstream code isolated under `external/`
and exposes this repository's ML/data/backtest capabilities through a sidecar
service that joins QuantDinger's Docker network.

## QuantDinger Summary

QuantDinger is a self-hosted quantitative trading platform with:

- Python/Flask backend API under `backend_api_python/`.
- PostgreSQL 16 metadata store, SQLite-friendly code paths, and Redis cache.
- Nginx-served Vue SPA frontend for dashboards, charts, AI analysis, strategy
  management and backtests.
- Existing database tables for strategies, positions, trades, orders,
  backtest runs, backtest trades, equity points, market symbols, credentials
  and notifications.
- REST market routes such as `/api/indicator/kline`, `/api/indicator/price`,
  `/api/market/symbols/search`, plus Agent Gateway read routes such as
  `/api/agent/v1/klines`.
- QuantDinger-Mobile as a Vue 3, Vite and Capacitor 6 app that can be extended
  with new mobile-first screens.

## Target Layout

```text
machine-learning-for-trading/
  ml_stock_lab/
  research_platform_definitive/
  scripts/
  external/
    QuantDinger/
    QuantDinger-Mobile/
  integrations/
    quantdinger_bridge/
      __init__.py
      api_adapter.py
      config.py
      db_adapter.py
      ml_service.py
      Dockerfile
      vue/
        web/
          MLDashboard.vue
          MLBacktests.vue
          MLSignals.vue
          mlApi.js
          router-snippet.js
        mobile/
          MLSignalsList.vue
          MLBacktestSummary.vue
          mlApi.js
  docker-compose.override.yml
  docs/
    quantdinger_integration.md
```

## Architecture

```text
QuantDinger Web / Mobile
        |
        |  /api/v1/ml/signals, /api/v1/ml/backtest, /api/v1/ml/models
        v
ml-service FastAPI sidecar
        |
        +--> QuantDinger Flask API for K-lines, symbols, positions and orders
        +--> QuantDinger PostgreSQL for persisted runs, metrics and signals
        +--> ml_stock_lab and research_platform_definitive for ML logic
```

Data flow:

```text
QuantDinger market/account infrastructure
        |
        v
quantdinger_bridge.api_adapter / db_adapter
        |
        v
ML features, signals, backtests
        |
        v
qd_backtest_runs, qd_backtest_trades, qd_backtest_equity_points,
qd_ml_signals, qd_ml_model_metrics, qd_ml_backtests
        |
        v
QuantDinger UI and mobile screens
```

## Onboarding

Clone the upstream apps as submodules:

```bash
git submodule add https://github.com/brokermr810/QuantDinger.git external/QuantDinger
git submodule add https://github.com/brokermr810/QuantDinger-Mobile.git external/QuantDinger-Mobile
git submodule update --init --recursive
```

Prepare QuantDinger's backend environment:

```bash
cp external/QuantDinger/backend_api_python/env.example external/QuantDinger/backend_api_python/.env
python3 - <<'PY'
import secrets
print("SECRET_KEY=" + secrets.token_hex(32))
PY
# Put the generated SECRET_KEY into external/QuantDinger/backend_api_python/.env
```

Start the combined stack from the ML4T repo root:

```bash
export ML4T_REPO_ROOT="$(pwd)"
docker compose \
  -f external/QuantDinger/docker-compose.yml \
  -f docker-compose.override.yml \
  up -d --build
```

Services:

- `postgres`: QuantDinger PostgreSQL metadata store.
- `redis`: QuantDinger cache.
- `backend`: QuantDinger Flask API at `http://localhost:5000`.
- `frontend`: QuantDinger Vue SPA at `http://localhost:8888`.
- `ml-service`: this repo's FastAPI ML bridge at `http://localhost:8000`.

Inside Docker, `ml-service` uses:

- `QUANTDINGER_API_BASE_URL=http://backend:5000/api`
- `QUANTDINGER_DATABASE_URL=postgresql+psycopg2://...@postgres:5432/quantdinger`
- `QUANTDINGER_REDIS_URL=redis://redis:6379/0`
- `PYTHONPATH=/workspace/ml4t:.../research_platform_definitive/src`

## Quickstart: Verify The Bridge

Start the stack:

```bash
export ML4T_REPO_ROOT="$(pwd)"
docker compose \
  -f external/QuantDinger/docker-compose.yml \
  -f docker-compose.override.yml \
  up -d --build
```

Check service health:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/ml/models
```

Run smoke diagnostics from inside the container:

```bash
docker compose \
  -f external/QuantDinger/docker-compose.yml \
  -f docker-compose.override.yml \
  exec ml-service python - <<'PY'
from integrations.quantdinger_bridge import smoke_test_bridge
print(smoke_test_bridge())
PY
```

Run a tiny API-only backtest:

```bash
curl -X POST http://localhost:8000/api/v1/ml/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "universe": ["AAPL", "MSFT"],
    "market": "USStock",
    "timeframe": "1D",
    "start_date": "2023-01-01",
    "end_date": "2024-01-01",
    "initial_capital": 100000,
    "persist": true
  }'
```

Confirm a persisted row:

```bash
docker compose \
  -f external/QuantDinger/docker-compose.yml \
  -f docker-compose.override.yml \
  exec postgres psql -U "${POSTGRES_USER:-quantdinger}" -d "${POSTGRES_DB:-quantdinger}" \
  -c "select id, qd_run_id, model_name, symbol, created_at from qd_ml_backtests order by id desc limit 5;"
```

## API Workflows

Get signals:

```bash
curl "http://localhost:8000/api/v1/ml/signals?universe=AAPL,MSFT,NVDA&market=USStock&timeframe=1D"
```

Run and persist a backtest:

```bash
curl -X POST http://localhost:8000/api/v1/ml/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "universe": ["AAPL", "MSFT", "NVDA"],
    "market": "USStock",
    "timeframe": "1D",
    "start_date": "2020-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 100000,
    "model_name": "momentum_baseline",
    "transaction_cost_bps": 2,
    "persist": true
  }'
```

The persisted run appears in QuantDinger-compatible tables:

- `qd_backtest_runs` with `run_type='ml'`
- `qd_backtest_trades`
- `qd_backtest_equity_points`
- `qd_ml_model_metrics`
- `qd_ml_backtests`

Current ML signals are persisted into `qd_ml_signals`.

## Backend Bridge Modules

- `config.py`: central environment-driven configuration.
- `api_adapter.py`: wraps QuantDinger REST routes and returns pandas frames or
  typed position/order objects.
- `db_adapter.py`: SQLAlchemy persistence and table reads.
- `ml_service.py`: FastAPI sidecar exposing:
  - `GET /api/v1/ml/models`
  - `GET /api/v1/ml/strategies`
  - `GET /api/v1/ml/signals`
  - `GET /api/v1/ml/signals/snapshot`
  - `GET /api/v1/ml/features`
  - `GET /api/v1/ml/backtests/summary`
  - `GET /api/v1/ml/metrics`
  - `POST /api/v1/ml/backtest`

Legacy `/ml/...` aliases remain for backward compatibility, but new frontend
work should use `/api/v1/ml/...`.

The default model is a conservative momentum baseline so the stack works before
you wire specific `ml_stock_lab` predictors. Replace `_momentum_signals` and
`_run_momentum_backtest` dispatch with your production feature/model/backtest
engines when ready.

## Web Frontend Integration

Copy these templates into a local QuantDinger frontend source checkout:

```text
integrations/quantdinger_bridge/vue/web/mlApi.js
integrations/quantdinger_bridge/vue/web/router-snippet.js
integrations/quantdinger_bridge/vue/web/MLDashboard.vue
integrations/quantdinger_bridge/vue/web/MLSignals.vue
integrations/quantdinger_bridge/vue/web/MLBacktests.vue
```

Proposed routes:

| Route | Endpoint | UI |
| --- | --- | --- |
| `/ml/dashboard` | `/api/v1/ml/models`, `/api/v1/ml/signals` | model count, signal summary, latest signal table |
| `/ml/signals` | `/api/v1/ml/signals`, `/api/v1/ml/signals/snapshot` | filterable cards/table for symbol, signal, confidence, target, stop |
| `/ml/backtests` | `/api/v1/ml/backtest`, `/api/v1/ml/backtests/summary` | run form, metrics, equity curve and stored run summary |
| `/ml/strategies` | `/api/v1/ml/strategies` | formulas, assumptions and limits from the Research Platform registry |

For same-origin deployment, add a reverse proxy rule in the QuantDinger Nginx
frontend image or your outer reverse proxy:

```nginx
location /api/v1/ml/ {
  proxy_pass http://ml-service:8000/api/v1/ml/;
  proxy_set_header Host $host;
  proxy_set_header Authorization $http_authorization;
}
```

For development, set `VITE_ML_API_URL=http://localhost:8000`.

## Mobile Integration

QuantDinger-Mobile already centralizes API base URL and bearer-token handling.
Add:

```text
integrations/quantdinger_bridge/vue/mobile/mlApi.js
integrations/quantdinger_bridge/vue/mobile/MLSignalsList.vue
integrations/quantdinger_bridge/vue/mobile/MLBacktestSummary.vue
```

Suggested router entries:

```js
{
  path: '/ml/signals',
  name: 'MLSignals',
  component: () => import('@/views/ml/MLSignalsList.vue'),
  meta: { title: 'ML Signals', showTabbar: true }
},
{
  path: '/ml/backtest',
  name: 'MLBacktestSummary',
  component: () => import('@/views/ml/MLBacktestSummary.vue'),
  meta: { title: 'ML Backtest', showTabbar: false }
}
```

Screen behavior:

- `ML Signals`: calls `GET /api/v1/ml/signals`, falls back to
  `GET /api/v1/ml/signals/snapshot`, supports pull-to-refresh, shows strategy
  tags, signal direction, confidence, target and stop-loss.
- `ML Backtest Summary`: calls `POST /api/v1/ml/backtest`, shows Sharpe, total return,
  drawdown, a compact equity curve and stored summaries from
  `GET /api/v1/ml/backtests/summary`.

Keep mobile navigation shallow: expose `ML Signals` as a tab or profile menu
entry, then link to `ML Backtest Summary` for deeper experiment review.

## Main End-To-End Workflows

### 1. Web UI Launches An ML Backtest

1. User opens `/ml/backtests`.
2. User enters universe, dates, market and initial capital.
3. Vue posts to `/api/v1/ml/backtest`.
4. `ml-service` fetches K-lines from QuantDinger backend.
5. ML4T model/backtest logic computes returns, trades and metrics.
6. `db_adapter.write_backtest_result` persists the run into QuantDinger tables.
7. The UI renders returned metrics immediately, and existing QuantDinger
   backtest tables can display persisted history.

### 2. Web/Mobile Displays ML Signals

1. User opens `/ml/signals`.
2. Client calls `/api/v1/ml/signals?universe=...`.
3. `ml-service` fetches recent bars, computes signals and optionally writes
   them to `qd_ml_signals`.
4. Web/mobile shows signal, confidence, target, stop-loss and horizon.

## Upstream-Safe Extension Pattern

Keep custom code in this repo:

- Sidecar backend: `integrations/quantdinger_bridge`.
- Docker composition: root `docker-compose.override.yml`.
- Frontend/mobile code: copyable templates under `integrations/.../vue`.

Avoid editing `external/QuantDinger` or `external/QuantDinger-Mobile` directly
unless you are intentionally maintaining a fork. This keeps upstream pulls and
submodule updates low-conflict.

## Runbook / Troubleshooting

Check service status:

```bash
docker compose \
  -f external/QuantDinger/docker-compose.yml \
  -f docker-compose.override.yml \
  ps
```

Follow logs:

```bash
docker compose -f external/QuantDinger/docker-compose.yml -f docker-compose.override.yml logs -f ml-service
docker compose -f external/QuantDinger/docker-compose.yml -f docker-compose.override.yml logs -f backend
docker compose -f external/QuantDinger/docker-compose.yml -f docker-compose.override.yml logs -f postgres
```

Exec into `ml-service`:

```bash
docker compose -f external/QuantDinger/docker-compose.yml -f docker-compose.override.yml exec ml-service bash
python - <<'PY'
from integrations.quantdinger_bridge import smoke_test_bridge
print(smoke_test_bridge())
PY
```

Common issues:

- `ml-service` cannot reach `backend`: confirm both services are on
  `quantdinger-network` and `QUANTDINGER_API_BASE_URL=http://backend:5000/api`.
- `connection refused` for PostgreSQL: wait for the `postgres` healthcheck, then
  inspect `QUANTDINGER_DATABASE_URL`.
- `401` from `/api/v1/ml/...`: unset `ML_SERVICE_AUTH_TOKEN` for local dev, or
  send `Authorization: Bearer <ML_SERVICE_AUTH_TOKEN>`.
- Frontend cannot call `/api/v1/ml`: add the Nginx proxy rule above, or set
  `VITE_ML_API_URL=http://localhost:8000` during frontend development.
- Docker Compose validation fails with missing paths: run commands from the
  repo root and export `ML4T_REPO_ROOT="$(pwd)"`.

Safe restart:

```bash
docker compose -f external/QuantDinger/docker-compose.yml -f docker-compose.override.yml restart ml-service
```

Safe shutdown without deleting volumes:

```bash
docker compose -f external/QuantDinger/docker-compose.yml -f docker-compose.override.yml down
```

Destructive shutdown, only when you intentionally want a clean database:

```bash
docker compose -f external/QuantDinger/docker-compose.yml -f docker-compose.override.yml down -v
```

## API-Only Python Example

```python
from integrations.quantdinger_bridge import QuantDingerAPIClient, get_ml_signals_for_universe

client = QuantDingerAPIClient()
bars = client.get_market_data("AAPL", start="2023-01-01", end="2024-01-01")
signals = get_ml_signals_for_universe(["AAPL", "MSFT"], date="2024-01-01")

print(bars.tail())
print(signals)
```

## UI Button To Database Row Example

1. User clicks `Run Backtest` in `MLBacktests.vue`.
2. The component posts a `BacktestRequest` to `/api/v1/ml/backtest`.
3. `ml-service` fetches QuantDinger K-lines from `/api/indicator/kline`.
4. The bridge runs the ML4T model/backtest path.
5. `db_adapter.write_backtest_result` writes:
   - `qd_backtest_runs`
   - `qd_backtest_trades`
   - `qd_backtest_equity_points`
   - `qd_ml_backtests`
   - `qd_ml_model_metrics`
6. The UI renders returned metrics immediately; persisted rows remain available
   for QuantDinger dashboards and future history pages.

<template>
  <div class="ml-screen">
    <van-nav-bar title="ML Backtest" />

    <van-form class="form" @submit="runBacktest">
      <van-field v-model="form.universe" label="Universe" placeholder="AAPL,MSFT,NVDA" />
      <van-field v-model="form.start_date" label="Start" type="date" />
      <van-field v-model="form.end_date" label="End" type="date" />
      <van-field v-model.number="form.initial_capital" label="Capital" type="number" />
      <div class="button-wrap">
        <van-button block type="primary" native-type="submit" :loading="loading">
          Run Backtest
        </van-button>
      </div>
    </van-form>

    <section v-if="result" class="summary">
      <div class="metric">
        <span>Sharpe</span>
        <strong>{{ metric('sharpe') }}</strong>
      </div>
      <div class="metric">
        <span>Return</span>
        <strong>{{ pct(result.metrics.total_return) }}</strong>
      </div>
      <div class="metric">
        <span>Drawdown</span>
        <strong>{{ pct(result.metrics.max_drawdown) }}</strong>
      </div>
    </section>

    <section v-if="result" class="chart-card">
      <h3>Equity Curve</h3>
      <div class="chart-placeholder">
        <span
          v-for="point in bars"
          :key="point.date"
          :style="{ height: `${point.height}%` }"
        />
      </div>
    </section>

    <section v-if="summaries.length" class="history-card">
      <h3>Stored Backtests</h3>
      <van-cell-group>
        <van-cell
          v-for="row in summaries"
          :key="row.backtest_id"
          :title="row.strategy_name"
          :label="`${row.start_date} -> ${row.end_date}`"
        >
          <template #value>
            <span>{{ pct(row.total_return) }} / S {{ metricValue(row.sharpe) }}</span>
          </template>
        </van-cell>
      </van-cell-group>
    </section>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { showToast } from 'vant'
import { mlApi } from './mlApi'

const loading = ref(false)
const result = ref(null)
const summaries = ref([])
const form = reactive({
  universe: 'AAPL,MSFT,NVDA',
  market: 'USStock',
  timeframe: '1D',
  start_date: '2020-01-01',
  end_date: '2024-12-31',
  initial_capital: 100000
})

const runBacktest = async () => {
  loading.value = true
  try {
    result.value = await mlApi.runBacktest({
      ...form,
      universe: form.universe.split(',').map((x) => x.trim()).filter(Boolean)
    })
  } catch (err) {
    showToast({ type: 'fail', message: err?.message || 'Backtest failed' })
  } finally {
    loading.value = false
  }
}

const loadSummary = async () => {
  try {
    summaries.value = await mlApi.getBacktestSummary(5)
  } catch (_) {
    summaries.value = []
  }
}

const bars = computed(() => {
  const rows = result.value?.equity_curve || []
  if (!rows.length) return []
  const step = Math.max(1, Math.floor(rows.length / 40))
  const sampled = rows.filter((_, idx) => idx % step === 0)
  const values = sampled.map((row) => Number(row.equity || 0))
  const min = Math.min(...values)
  const max = Math.max(...values)
  return sampled.map((row) => ({
    date: row.date,
    height: max === min ? 50 : 12 + ((Number(row.equity) - min) / (max - min)) * 78
  }))
})

const metric = (key) => Number(result.value?.metrics?.[key] || 0).toFixed(2)
const metricValue = (value) => Number(value || 0).toFixed(2)
const pct = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`

onMounted(loadSummary)
</script>

<style scoped>
.ml-screen { min-height: 100vh; background: #f7f8fa; padding-bottom: 24px; }
.form { margin-top: 8px; }
.button-wrap { padding: 16px; }
.summary { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; padding: 0 12px 12px; }
.metric, .chart-card { background: #fff; border-radius: 8px; padding: 12px; }
.metric span { color: #969799; font-size: 12px; }
.metric strong { display: block; margin-top: 6px; font-size: 18px; }
.chart-card { margin: 0 12px; }
.chart-card h3, .history-card h3 { margin: 0 0 12px; font-size: 15px; }
.chart-placeholder { height: 160px; display: flex; align-items: end; gap: 2px; border-bottom: 1px solid #ebedf0; }
.chart-placeholder span { flex: 1; min-width: 2px; background: #1989fa; border-radius: 2px 2px 0 0; }
.history-card { background: #fff; border-radius: 8px; margin: 12px; padding: 12px; }
</style>

<template>
  <main class="ml-backtests">
    <form class="config" @submit.prevent="runBacktest">
      <input v-model="form.universe" placeholder="AAPL,MSFT,NVDA" />
      <input v-model="form.start_date" type="date" />
      <input v-model="form.end_date" type="date" />
      <input v-model.number="form.initial_capital" type="number" min="1000" step="1000" />
      <button :disabled="loading">{{ loading ? 'Running...' : 'Run Backtest' }}</button>
    </form>

    <p v-if="error" class="error">{{ error }}</p>

    <section v-if="result" class="summary">
      <article v-for="(value, key) in result.metrics" :key="key">
        <span>{{ key }}</span>
        <strong>{{ formatMetric(value) }}</strong>
      </article>
    </section>

    <section v-if="result" class="chart-panel">
      <h2>Equity Curve</h2>
      <div class="placeholder-chart">
        <span
          v-for="point in sampledCurve"
          :key="point.date"
          :style="{ height: `${point.height}%` }"
        />
      </div>
    </section>
  </main>
</template>

<script setup>
import { computed, reactive, ref } from 'vue'
import { mlApi } from './mlApi'

const form = reactive({
  universe: 'AAPL,MSFT,NVDA',
  market: 'USStock',
  timeframe: '1D',
  start_date: '2020-01-01',
  end_date: '2024-12-31',
  initial_capital: 100000,
  transaction_cost_bps: 2
})
const loading = ref(false)
const error = ref('')
const result = ref(null)

const runBacktest = async () => {
  loading.value = true
  error.value = ''
  try {
    result.value = await mlApi.backtest({
      ...form,
      universe: form.universe.split(',').map((x) => x.trim()).filter(Boolean)
    })
  } catch (err) {
    error.value = err?.message || 'Backtest failed.'
  } finally {
    loading.value = false
  }
}

const sampledCurve = computed(() => {
  const rows = result.value?.equity_curve || []
  if (!rows.length) return []
  const step = Math.max(1, Math.floor(rows.length / 60))
  const sampled = rows.filter((_, idx) => idx % step === 0)
  const values = sampled.map((row) => Number(row.equity || 0))
  const min = Math.min(...values)
  const max = Math.max(...values)
  return sampled.map((row) => ({
    date: row.date,
    height: max === min ? 40 : 10 + ((Number(row.equity) - min) / (max - min)) * 80
  }))
})

const formatMetric = (value) => Number(value).toFixed(Math.abs(Number(value)) < 2 ? 4 : 2)
</script>

<style scoped>
.ml-backtests { padding: 24px; display: grid; gap: 16px; }
.config { display: flex; gap: 8px; flex-wrap: wrap; }
input, button { height: 36px; border: 1px solid #d0d5dd; border-radius: 6px; padding: 0 10px; }
button { background: #1677ff; color: white; border: 0; }
.summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
.summary article, .chart-panel { border: 1px solid #eaecf0; border-radius: 8px; background: white; padding: 16px; }
.summary span { color: #667085; display: block; font-size: 13px; }
.summary strong { font-size: 20px; margin-top: 6px; display: block; }
.placeholder-chart { height: 220px; display: flex; align-items: end; gap: 3px; border-bottom: 1px solid #d0d5dd; }
.placeholder-chart span { flex: 1; background: #1677ff; min-width: 2px; border-radius: 2px 2px 0 0; }
.error { color: #b42318; }
</style>


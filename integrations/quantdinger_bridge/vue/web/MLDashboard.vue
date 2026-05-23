<template>
  <main class="ml-page">
    <header class="toolbar">
      <div>
        <h1>ML Dashboard</h1>
        <p>Model health, latest signals, and backtest summary.</p>
      </div>
      <button class="primary" @click="refresh" :disabled="loading">Refresh</button>
    </header>

    <section v-if="error" class="error">{{ error }}</section>

    <section class="metric-grid">
      <article v-for="item in metrics" :key="item.label" class="metric">
        <span>{{ item.label }}</span>
        <strong>{{ item.value }}</strong>
      </article>
    </section>

    <section class="panel">
      <h2>Latest Signals</h2>
      <table>
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Signal</th>
            <th>Confidence</th>
            <th>Target</th>
            <th>Stop</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="signal in signals" :key="`${signal.symbol}-${signal.date}`">
            <td>{{ signal.symbol }}</td>
            <td :class="signal.signal > 0 ? 'up' : signal.signal < 0 ? 'down' : ''">
              {{ signal.signal > 0 ? 'Long' : signal.signal < 0 ? 'Short' : 'Flat' }}
            </td>
            <td>{{ pct(signal.confidence) }}</td>
            <td>{{ money(signal.target_price) }}</td>
            <td>{{ money(signal.stop_loss) }}</td>
          </tr>
        </tbody>
      </table>
    </section>
  </main>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { mlApi } from './mlApi'

const loading = ref(false)
const error = ref('')
const signals = ref([])
const models = ref([])

const metrics = computed(() => [
  { label: 'Models', value: models.value.length },
  { label: 'Signals', value: signals.value.length },
  { label: 'Long Bias', value: pct(signals.value.filter((s) => Number(s.signal) > 0).length / Math.max(signals.value.length, 1)) }
])

const refresh = async () => {
  loading.value = true
  error.value = ''
  try {
    const [modelRows, signalRows] = await Promise.all([
      mlApi.models(),
      mlApi.signals({ universe: 'AAPL,MSFT,NVDA', market: 'USStock', timeframe: '1D' })
    ])
    models.value = modelRows
    signals.value = signalRows
  } catch (err) {
    error.value = err?.message || 'Failed to load ML dashboard.'
  } finally {
    loading.value = false
  }
}

const pct = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`
const money = (value) => Number(value || 0).toFixed(2)

onMounted(refresh)
</script>

<style scoped>
.ml-page { padding: 24px; display: grid; gap: 16px; }
.toolbar { display: flex; align-items: center; justify-content: space-between; gap: 16px; }
.toolbar h1 { margin: 0; font-size: 24px; }
.toolbar p { margin: 4px 0 0; color: #667085; }
.primary { border: 0; background: #1677ff; color: white; padding: 10px 14px; border-radius: 6px; }
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
.metric, .panel { border: 1px solid #eaecf0; border-radius: 8px; background: white; padding: 16px; }
.metric span { color: #667085; font-size: 13px; }
.metric strong { display: block; margin-top: 8px; font-size: 24px; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 10px; border-bottom: 1px solid #eaecf0; text-align: left; }
.up { color: #039855; }
.down { color: #d92d20; }
.error { background: #fef3f2; color: #b42318; padding: 12px; border-radius: 6px; }
</style>


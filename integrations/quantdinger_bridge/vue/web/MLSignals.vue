<template>
  <main class="ml-signals">
    <header class="filters">
      <input v-model="universe" placeholder="AAPL,MSFT,NVDA" />
      <select v-model="market">
        <option>USStock</option>
        <option>Crypto</option>
        <option>Forex</option>
        <option>Futures</option>
      </select>
      <button @click="loadSignals" :disabled="loading">Load</button>
    </header>

    <p v-if="error" class="error">{{ error }}</p>
    <div v-if="loading" class="empty">Loading signals...</div>

    <section v-else class="cards">
      <article v-for="row in signals" :key="`${row.symbol}-${row.date}`" class="signal-card">
        <div>
          <strong>{{ row.symbol }}</strong>
          <small>{{ row.model_name || 'ML model' }}</small>
        </div>
        <span :class="badgeClass(row.signal)">
          {{ label(row.signal) }}
        </span>
        <dl>
          <dt>Confidence</dt><dd>{{ pct(row.confidence) }}</dd>
          <dt>Target</dt><dd>{{ num(row.target_price) }}</dd>
          <dt>Stop</dt><dd>{{ num(row.stop_loss) }}</dd>
        </dl>
      </article>
    </section>
  </main>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { mlApi } from './mlApi'

const universe = ref('AAPL,MSFT,NVDA')
const market = ref('USStock')
const signals = ref([])
const loading = ref(false)
const error = ref('')

const loadSignals = async () => {
  loading.value = true
  error.value = ''
  try {
    signals.value = await mlApi.signals({ universe: universe.value, market: market.value, timeframe: '1D' })
  } catch (err) {
    error.value = err?.message || 'Unable to load ML signals.'
  } finally {
    loading.value = false
  }
}

const label = (signal) => Number(signal) > 0 ? 'Long' : Number(signal) < 0 ? 'Short' : 'Flat'
const badgeClass = (signal) => Number(signal) > 0 ? 'long' : Number(signal) < 0 ? 'short' : 'flat'
const pct = (value) => `${(Number(value || 0) * 100).toFixed(0)}%`
const num = (value) => Number(value || 0).toFixed(2)

onMounted(loadSignals)
</script>

<style scoped>
.ml-signals { padding: 24px; display: grid; gap: 16px; }
.filters { display: flex; gap: 8px; flex-wrap: wrap; }
input, select, button { height: 36px; border: 1px solid #d0d5dd; border-radius: 6px; padding: 0 10px; }
button { background: #111827; color: white; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }
.signal-card { border: 1px solid #eaecf0; border-radius: 8px; padding: 14px; background: white; display: grid; gap: 12px; }
.signal-card small { display: block; color: #667085; }
.long, .short, .flat { justify-self: start; border-radius: 999px; padding: 4px 8px; font-size: 12px; }
.long { color: #027a48; background: #ecfdf3; }
.short { color: #b42318; background: #fef3f2; }
.flat { color: #344054; background: #f2f4f7; }
dl { display: grid; grid-template-columns: 1fr auto; gap: 6px; margin: 0; }
dt { color: #667085; }
dd { margin: 0; font-variant-numeric: tabular-nums; }
.error { color: #b42318; }
.empty { color: #667085; }
</style>


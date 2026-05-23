<template>
  <div class="ml-screen">
    <van-nav-bar title="ML Signals" />

    <van-search
      v-model="universe"
      placeholder="AAPL,MSFT,NVDA"
      shape="round"
      @search="loadSignals"
    />

    <div v-if="strategies.length" class="strategy-strip">
      <van-tag
        v-for="row in strategies"
        :key="row.strategy_name"
        plain
        type="primary"
      >
        {{ row.strategy_name }}
      </van-tag>
    </div>

    <van-pull-refresh v-model="refreshing" @refresh="loadSignals">
      <van-loading v-if="loading" class="center" />
      <van-empty v-else-if="!signals.length" description="No signals" />

      <van-cell-group v-else inset>
        <van-cell v-for="row in signals" :key="`${row.symbol}-${row.date}`">
          <template #title>
            <div class="row-title">
              <strong>{{ row.symbol }}</strong>
              <van-tag :type="tagType(row.signal)">{{ label(row.signal) }}</van-tag>
            </div>
          </template>
          <template #label>
            <div class="metrics">
              <span>Confidence {{ pct(row.confidence) }}</span>
              <span>Target {{ price(row.target_price) }}</span>
              <span>Stop {{ price(row.stop_loss) }}</span>
            </div>
          </template>
        </van-cell>
      </van-cell-group>
    </van-pull-refresh>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { showToast } from 'vant'
import { mlApi } from './mlApi'

const universe = ref('AAPL,MSFT,NVDA')
const signals = ref([])
const strategies = ref([])
const loading = ref(false)
const refreshing = ref(false)

const loadStrategies = async () => {
  try {
    strategies.value = await mlApi.getStrategies()
  } catch (_) {
    strategies.value = []
  }
}

const loadSignals = async () => {
  loading.value = !refreshing.value
  try {
    signals.value = await mlApi.getSignals({
      universe: universe.value,
      market: 'USStock',
      timeframe: '1D'
    })
  } catch (err) {
    showToast({ type: 'fail', message: err?.message || 'Failed to load signals' })
  } finally {
    loading.value = false
    refreshing.value = false
  }
}

const label = (signal) => Number(signal) > 0 ? 'Long' : Number(signal) < 0 ? 'Short' : 'Flat'
const tagType = (signal) => Number(signal) > 0 ? 'success' : Number(signal) < 0 ? 'danger' : 'default'
const pct = (value) => `${(Number(value || 0) * 100).toFixed(0)}%`
const price = (value) => Number(value || 0).toFixed(2)

onMounted(() => {
  loadStrategies()
  loadSignals()
})
</script>

<style scoped>
.ml-screen { min-height: 100vh; background: #f7f8fa; }
.center { display: block; margin: 40px auto; }
.strategy-strip { display: flex; gap: 6px; overflow-x: auto; padding: 0 12px 10px; }
.row-title { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.metrics { display: grid; gap: 4px; color: #646566; font-size: 12px; }
</style>

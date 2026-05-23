import { getBaseUrl } from '@/api'

const getMlBaseUrl = () => {
  const explicit = import.meta.env?.VITE_ML_API_URL || window.__ML_API_URL__
  return (explicit || getBaseUrl()).replace(/\/$/, '')
}

const parseError = async (response) => {
  try {
    const payload = await response.json()
    return payload?.detail || payload?.message || `ML service error ${response.status}`
  } catch (_) {
    return `ML service error ${response.status}`
  }
}

const request = async (path, options = {}) => {
  const token = localStorage.getItem('token')
  const response = await fetch(`${getMlBaseUrl()}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(options.headers || {})
    }
  })
  if (!response.ok) throw new Error(await parseError(response))
  return response.json()
}

export const mlApi = {
  getStrategies: async () => {
    const data = await request('/api/v1/ml/strategies')
    return data.strategies || []
  },
  getSignals: async (params) => {
    const query = new URLSearchParams(params).toString()
    let data
    try {
      data = await request(`/api/v1/ml/signals?${query}`)
    } catch (_) {
      data = await request('/api/v1/ml/signals/snapshot?limit=50')
    }
    return data.signals || []
  },
  getSignalSnapshot: async (limit = 50) => {
    const data = await request(`/api/v1/ml/signals/snapshot?limit=${limit}`)
    return data.signals || []
  },
  runBacktest: async (payload) => {
    return request('/api/v1/ml/backtest', {
      method: 'POST',
      body: JSON.stringify(payload)
    })
  },
  getBacktestSummary: async (limit = 10) => {
    const data = await request(`/api/v1/ml/backtests/summary?limit=${limit}`)
    return data.backtests || []
  }
}

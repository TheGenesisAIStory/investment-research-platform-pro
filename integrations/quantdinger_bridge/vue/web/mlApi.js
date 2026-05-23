import axios from 'axios'

const mlHttp = axios.create({
  baseURL: import.meta.env.VITE_ML_API_URL || '',
  timeout: 60000,
  headers: { 'Content-Type': 'application/json' }
})

mlHttp.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

export const mlApi = {
  models: () => mlHttp.get('/api/v1/ml/models').then((res) => res.data?.models || res.data?.data?.models || []),
  strategies: () => mlHttp.get('/api/v1/ml/strategies').then((res) => res.data?.strategies || []),
  signals: (params) => mlHttp.get('/api/v1/ml/signals', { params }).then((res) => res.data?.signals || []),
  signalSnapshot: (limit = 50) => mlHttp.get('/api/v1/ml/signals/snapshot', { params: { limit } }).then((res) => res.data?.signals || []),
  backtest: (payload) => mlHttp.post('/api/v1/ml/backtest', payload).then((res) => res.data),
  backtestSummary: (limit = 10) => mlHttp.get('/api/v1/ml/backtests/summary', { params: { limit } }).then((res) => res.data?.backtests || []),
  features: (params) => mlHttp.get('/api/v1/ml/features', { params }).then((res) => res.data?.features || res.data?.data?.features || [])
}

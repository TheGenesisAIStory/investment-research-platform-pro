import axios from 'axios'

const mlHttp = axios.create({
  // TODO: In the QuantDinger Vue app, either set VITE_ML_API_URL to the
  // ml-service origin in development, or proxy /api/v1/ml through Nginx.
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
  signals: (params) => mlHttp.get('/api/v1/ml/signals', { params }).then((res) => res.data?.signals || []),
  backtest: (payload) => mlHttp.post('/api/v1/ml/backtest', payload).then((res) => res.data),
  features: (params) => mlHttp.get('/api/v1/ml/features', { params }).then((res) => res.data?.features || res.data?.data?.features || [])
}

import { getBaseUrl } from '@/api'

const request = async (path, options = {}) => {
  const token = localStorage.getItem('token')
  const response = await fetch(`${getBaseUrl()}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(options.headers || {})
    }
  })
  if (!response.ok) throw new Error(`Request failed: ${response.status}`)
  return response.json()
}

export const mlApi = {
  getSignals: async (params) => {
    const query = new URLSearchParams(params).toString()
    // TODO: Configure the QuantDinger-Mobile server URL so /api/v1/ml is
    // reverse-proxied to ml-service, matching the web deployment.
    const data = await request(`/api/v1/ml/signals?${query}`)
    return data.signals || []
  },
  runBacktest: async (payload) => {
    return request('/api/v1/ml/backtest', {
      method: 'POST',
      body: JSON.stringify(payload)
    })
  }
}

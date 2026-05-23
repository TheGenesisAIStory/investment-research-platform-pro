// TODO: Copy these route records into QuantDinger's main Vue router once a
// local frontend source checkout is available. Keep the components under
// src/views/ml/ to avoid conflicts with upstream QuantDinger routes.
export const mlRoutes = [
  {
    path: '/ml/dashboard',
    name: 'MLDashboard',
    component: () => import('@/views/ml/MLDashboard.vue'),
    meta: { title: 'ML Dashboard' }
  },
  {
    path: '/ml/backtests',
    name: 'MLBacktests',
    component: () => import('@/views/ml/MLBacktests.vue'),
    meta: { title: 'ML Backtests' }
  },
  {
    path: '/ml/signals',
    name: 'MLSignals',
    component: () => import('@/views/ml/MLSignals.vue'),
    meta: { title: 'ML Signals' }
  }
]

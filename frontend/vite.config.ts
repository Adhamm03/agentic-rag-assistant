import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxy /api/* to the FastAPI backend so the app uses relative paths (no CORS).
// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})

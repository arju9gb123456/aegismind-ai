import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// During development the React app runs on :5173 and proxies API calls to FastAPI on :8000.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { '/api': 'http://localhost:8000' },
  },
  build: {
    chunkSizeWarningLimit: 2500,
  },
})

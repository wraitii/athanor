import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  plugins: [vue()],
  server: { headers: isolation },
  preview: { headers: isolation },
  worker: { format: 'es' },
});

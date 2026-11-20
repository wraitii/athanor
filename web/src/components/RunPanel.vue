<script setup lang="ts">
import type { CallStats, LogLevel } from '../run/protocol';

defineProps<{ status: string; running: boolean; logs: { level: LogLevel; text: string }[]; stats: CallStats[] }>();
const gpu = defineModel<boolean>('gpu', { required: true });
const emit = defineEmits<{ run: []; stop: [] }>();

function ms(total: number, count: number): string {
  return count ? `${(total / count).toFixed(1)} ms` : '—';
}

function speedup(row: CallStats): string {
  if (!row.gpu) return '—';
  return `${(row.v8Ms / row.calls / (row.gpuMs / row.gpu)).toFixed(1)}×`;
}

function error(row: CallStats): string {
  if (!row.gpu) return '—';
  return row.maxRel.toExponential(1);
}

function fallbacks(row: CallStats): string {
  const entries = Object.entries(row.fallbacks);
  if (!entries.length) return '—';
  return entries.map(([reason, count]) => `${reason} ×${count}`).join(', ');
}
</script>

<template>
  <section class="run-panel">
    <header>
      <span class="title">Run</span>
      <button v-if="running" class="run" @click="emit('stop')">Stop</button>
      <button v-else class="run" @click="emit('run')">Run <kbd>⇧⌘↵</kbd></button>
      <label class="toggle"><input v-model="gpu" type="checkbox" /> WebGPU</label>
      <span class="status">{{ status }}</span>
    </header>
    <div class="body">
      <p v-if="!stats.length && !logs.length" class="hint">
        Runs the script as plain JS in a worker, then calls <code>main()</code> if it exists.
        With WebGPU on, calls to functions Athanor compiles to a kernel also run on the GPU;
        the JS version runs on each call too, as the timing and correctness reference.
      </p>
      <table v-if="stats.length">
        <thead>
          <tr>
            <th>function</th>
            <th>calls</th>
            <th>JS / call</th>
            <th>WebGPU / call</th>
            <th>speed-up</th>
            <th title="Largest difference from the JS result, relative to the largest JS value">max rel. error</th>
            <th>ran in JS only</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in stats" :key="row.name">
            <td class="name">{{ row.name }}</td>
            <td>{{ row.calls }}</td>
            <td>{{ ms(row.v8Ms, row.calls) }}</td>
            <td>{{ ms(row.gpuMs, row.gpu) }}</td>
            <td class="speedup">{{ speedup(row) }}</td>
            <td>{{ error(row) }}</td>
            <td class="reasons">{{ fallbacks(row) }}</td>
          </tr>
        </tbody>
      </table>
      <div v-for="(log, index) in logs" :key="index" class="log" :class="log.level">{{ log.text }}</div>
    </div>
  </section>
</template>

<style scoped>
.run-panel {
  display: grid;
  grid-template-rows: auto 1fr;
  border-top: 1px solid var(--border);
  min-height: 0;
}
header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 6px 12px;
  border-bottom: 1px solid var(--border);
}
.title {
  font-family: var(--mono);
  font-weight: 600;
}
.run {
  background: var(--accent);
  color: white;
  border: none;
  border-radius: 6px;
  padding: 4px 12px;
  cursor: pointer;
}
.run kbd {
  font-family: inherit;
  opacity: 0.75;
  margin-left: 4px;
}
.toggle {
  color: var(--muted);
  display: flex;
  gap: 6px;
  align-items: center;
}
.status {
  margin-left: auto;
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}
.body {
  overflow: auto;
  padding: 8px 0;
  font-family: var(--mono);
  font-size: 12px;
}
.hint {
  margin: 0 12px;
  color: var(--muted);
  font-family: var(--sans);
  font-size: 13px;
  line-height: 1.5;
  max-width: 60ch;
}
table {
  border-collapse: collapse;
  margin: 0 12px 8px;
  font-variant-numeric: tabular-nums;
}
th,
td {
  text-align: right;
  padding: 2px 10px;
  white-space: nowrap;
}
th {
  color: var(--muted);
  font-weight: normal;
}
th:first-child,
td.name,
td.reasons {
  text-align: left;
}
td.speedup {
  color: var(--added-strong);
  font-weight: 600;
}
td.reasons {
  white-space: normal;
  color: var(--muted);
}
.log {
  white-space: pre-wrap;
  padding: 0 12px;
}
.log.info {
  color: var(--muted);
}
.log.warn {
  color: var(--accent);
}
.log.error {
  color: var(--removed-strong);
}
</style>

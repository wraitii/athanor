<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue';
import SourceEditor from './components/SourceEditor.vue';
import IrListing from './components/IrListing.vue';
import { Compiler } from './compiler';
import { examples } from './examples';
import { byteOffsets, changeCount, diffListing, type Span } from './listing';
import type { Pass, Report } from './types';

const STORAGE_KEY = 'athanor-inspector-source';

function loadSource(): string {
  try {
    return localStorage.getItem(STORAGE_KEY) ?? examples['Gaussian blur'];
  } catch {
    return examples['Gaussian blur'];
  }
}

const compiler = new Compiler();
const source = ref(loadSource());
const filter = ref('');
const reports = ref<Report[]>([]);
const compiledSource = ref('');
const reportIndex = ref(0);
const passIndex = ref(0);
const showDiff = ref(true);
const hover = ref<Span | null>(null);
const cursor = ref<number | null>(null);
const status = ref('Loading compiler…');
const error = ref<string | null>(null);
const compiling = ref(false);
let latest = 0;

const report = computed<Report | undefined>(() => reports.value[reportIndex.value]);
const passes = computed<Pass[]>(() => report.value?.passes ?? []);
const toUnits = computed(() => byteOffsets(compiledSource.value));

function previousPass(index: number): Pass | undefined {
  const current = passes.value[index];
  if (!current || current.pass === 'wgsl') return undefined;
  for (let i = index - 1; i >= 0; i--) {
    if (passes.value[i].pass !== 'wgsl') return passes.value[i];
  }
  return undefined;
}

const summaries = computed(() =>
  passes.value.map((pass, index) => ({ name: pass.pass, ...changeCount(previousPass(index)?.text, pass.text) })),
);

const lines = computed(() => {
  const current = passes.value[passIndex.value];
  if (!current) return [];
  return diffListing(previousPass(passIndex.value)?.text, current.text, toUnits.value);
});

async function run() {
  const id = ++latest;
  const text = source.value;
  compiling.value = true;
  const response = await compiler.compile(text, filter.value.trim());
  if (id !== latest) return;
  compiling.value = false;
  if (!response.ok) {
    error.value = response.error;
    status.value = response.fatal ? 'Compiler crashed; restarted' : 'Compile failed';
    return;
  }
  error.value = null;
  const previous = report.value;
  const previousPassName = passes.value[passIndex.value]?.pass;
  reports.value = response.reports;
  compiledSource.value = text;
  const keep = response.reports.findIndex((r) => r.label === previous?.label);
  reportIndex.value = keep >= 0 ? keep : 0;
  const pass = passes.value.findIndex((p) => p.pass === previousPassName);
  passIndex.value = pass >= 0 ? pass : 0;
  const count = response.reports.length;
  status.value = `${count} compilation${count === 1 ? '' : 's'} · ${Math.round(response.milliseconds)} ms`;
}

let timer: number | undefined;
watch([source, filter], () => {
  try {
    localStorage.setItem(STORAGE_KEY, source.value);
  } catch {}
  clearTimeout(timer);
  timer = window.setTimeout(run, 500);
});

watch(reportIndex, () => {
  passIndex.value = Math.min(passIndex.value, Math.max(passes.value.length - 1, 0));
});

function stepPass(event: KeyboardEvent) {
  const delta = event.key === 'ArrowDown' ? 1 : event.key === 'ArrowUp' ? -1 : 0;
  if (!delta) return;
  event.preventDefault();
  passIndex.value = Math.min(Math.max(passIndex.value + delta, 0), passes.value.length - 1);
}

function loadExample(event: Event) {
  const name = (event.target as HTMLSelectElement).value;
  if (examples[name]) source.value = examples[name];
  (event.target as HTMLSelectElement).value = '';
}

onMounted(run);
</script>

<template>
  <div class="app">
    <header class="toolbar">
      <div class="brand">
        <strong>Athanor</strong>
        <span>semantic IR inspector</span>
      </div>
      <select class="examples" aria-label="Load example" @change="loadExample">
        <option value="">Examples…</option>
        <option v-for="name in Object.keys(examples)" :key="name" :value="name">{{ name }}</option>
      </select>
      <input v-model="filter" class="filter" placeholder="Filter functions" aria-label="Function filter" />
      <button class="compile" :disabled="compiling" @click="run">Compile <kbd>⌘↵</kbd></button>
      <span class="status" :class="{ failed: error }">{{ compiling ? 'Compiling…' : status }}</span>
    </header>

    <main class="workspace">
      <section class="panel editor">
        <SourceEditor v-model:source="source" :highlight="hover" @cursor="cursor = $event" @run="run" />
      </section>

      <nav class="panel rail" tabindex="0" aria-label="Passes" @keydown="stepPass">
        <select v-if="reports.length" v-model="reportIndex" class="report" aria-label="Compilation">
          <option v-for="(r, index) in reports" :key="index" :value="index">{{ r.label }}</option>
        </select>
        <ol class="passes">
          <li v-for="(summary, index) in summaries" :key="summary.name + index">
            <button
              :class="{ active: index === passIndex, unchanged: index > 0 && !summary.added && !summary.removed }"
              @click="passIndex = index"
            >
              <span class="name">{{ summary.name }}</span>
              <span v-if="index > 0 && (summary.added || summary.removed)" class="delta">
                <span class="plus">+{{ summary.added }}</span>
                <span class="minus">−{{ summary.removed }}</span>
              </span>
            </button>
          </li>
        </ol>
      </nav>

      <section class="panel output">
        <div class="output-header">
          <span class="title">{{ passes[passIndex]?.pass ?? '—' }}</span>
          <label class="toggle"><input v-model="showDiff" type="checkbox" /> diff vs previous pass</label>
        </div>
        <pre v-if="error" class="error">{{ error }}</pre>
        <IrListing v-else :lines="lines" :diff="showDiff" :cursor="cursor" @hover="hover = $event" />
      </section>
    </main>
  </div>
</template>

<style scoped>
.app {
  display: grid;
  grid-template-rows: auto 1fr;
  height: 100%;
}
.toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 16px;
  border-bottom: 1px solid var(--border);
  background: var(--panel);
  flex-wrap: wrap;
}
.brand {
  display: flex;
  align-items: baseline;
  gap: 8px;
  margin-right: auto;
}
.brand strong {
  font-size: 16px;
  letter-spacing: 0.02em;
}
.brand span {
  color: var(--muted);
}
.examples,
.filter,
.report {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 5px 8px;
}
.filter {
  width: 160px;
}
.compile {
  background: var(--accent);
  color: white;
  border: none;
  border-radius: 6px;
  padding: 6px 12px;
  cursor: pointer;
}
.compile:disabled {
  opacity: 0.6;
  cursor: progress;
}
.compile kbd {
  font-family: inherit;
  opacity: 0.75;
  margin-left: 4px;
}
.status {
  color: var(--muted);
  font-variant-numeric: tabular-nums;
  min-width: 140px;
}
.status.failed {
  color: var(--removed-strong);
}
.workspace {
  display: grid;
  grid-template-columns: minmax(320px, 2fr) 230px minmax(0, 3fr);
  gap: 1px;
  background: var(--border);
  min-height: 0;
}
.panel {
  background: var(--panel);
  min-height: 0;
  min-width: 0;
}
.rail {
  display: flex;
  flex-direction: column;
  outline: none;
}
.rail:focus-visible {
  box-shadow: inset 0 0 0 2px var(--accent);
}
.report {
  margin: 10px;
}
.passes {
  list-style: none;
  margin: 0;
  padding: 0 6px 10px;
  overflow: auto;
  flex: 1;
}
.passes button {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  width: 100%;
  border: none;
  background: none;
  text-align: left;
  padding: 4px 8px;
  border-radius: 5px;
  cursor: pointer;
  font-family: var(--mono);
  font-size: 12px;
}
.passes button:hover {
  background: var(--hover);
}
.passes button.active {
  background: var(--accent-soft);
  color: var(--accent);
  font-weight: 600;
}
.passes button.unchanged .name {
  color: var(--muted);
}
.delta {
  display: flex;
  gap: 6px;
  font-variant-numeric: tabular-nums;
}
.plus {
  color: var(--added-strong);
}
.minus {
  color: var(--removed-strong);
}
.output {
  display: grid;
  grid-template-rows: auto 1fr;
}
.output-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  border-bottom: 1px solid var(--border);
}
.output-header .title {
  font-family: var(--mono);
  font-weight: 600;
}
.toggle {
  color: var(--muted);
  display: flex;
  gap: 6px;
  align-items: center;
}
.error {
  margin: 0;
  padding: 12px;
  color: var(--removed-strong);
  white-space: pre-wrap;
  overflow: auto;
  font-family: var(--mono);
  font-size: 12px;
}

@media (max-width: 900px) {
  .workspace {
    grid-template-columns: 1fr;
    grid-template-rows: 40vh auto 60vh;
  }
  .passes {
    max-height: 30vh;
  }
  .filter {
    width: 120px;
  }
}
</style>

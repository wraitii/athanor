<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { EditorView, basicSetup } from 'codemirror';
import { javascript } from '@codemirror/lang-javascript';
import { oneDark } from '@codemirror/theme-one-dark';
import { Compartment, EditorState, Prec, StateEffect, StateField } from '@codemirror/state';
import { Decoration, type DecorationSet, keymap } from '@codemirror/view';
import type { Span } from '../listing';

const props = defineProps<{ source: string; highlight: Span | null; lines: number[] }>();
const emit = defineEmits<{ 'update:source': [string]; cursor: [number]; run: []; execute: [] }>();

const host = ref<HTMLElement>();
let view: EditorView | undefined;

const setHighlight = StateEffect.define<Span | null>();
const mark = Decoration.mark({ class: 'cm-ir-span' });
const highlightField = StateField.define<DecorationSet>({
  create: () => Decoration.none,
  update(set, transaction) {
    for (const effect of transaction.effects) {
      if (!effect.is(setHighlight)) continue;
      const span = effect.value;
      const length = transaction.state.doc.length;
      if (!span || span[0] >= span[1] || span[0] >= length) return Decoration.none;
      return Decoration.set([mark.range(span[0], Math.min(span[1], length))]);
    }
    return set.map(transaction.changes);
  },
  provide: (field) => EditorView.decorations.from(field),
});

const setLines = StateEffect.define<number[]>();
const lineDecorations = Array.from({ length: 8 }, (_, index) => Decoration.line({ class: `cm-sourced src-${index}` }));
const linesField = StateField.define<DecorationSet>({
  create: () => Decoration.none,
  update(set, transaction) {
    for (const effect of transaction.effects) {
      if (!effect.is(setLines)) continue;
      const doc = transaction.state.doc;
      const ranges = effect.value
        .filter((line) => line < doc.lines)
        .map((line) => lineDecorations[line % 8].range(doc.line(line + 1).from));
      return Decoration.set(ranges);
    }
    return set.map(transaction.changes);
  },
  provide: (field) => EditorView.decorations.from(field),
});

const theme = new Compartment();
const dark = window.matchMedia('(prefers-color-scheme: dark)');
const applyTheme = () => view?.dispatch({ effects: theme.reconfigure(dark.matches ? oneDark : []) });

onMounted(() => {
  view = new EditorView({
    parent: host.value!,
    state: EditorState.create({
      doc: props.source,
      extensions: [
        Prec.highest(
          keymap.of([
            { key: 'Mod-Enter', run: () => (emit('run'), true) },
            { key: 'Shift-Mod-Enter', run: () => (emit('execute'), true) },
          ]),
        ),
        basicSetup,
        javascript(),
        highlightField,
        linesField,
        theme.of(dark.matches ? oneDark : []),
        EditorView.updateListener.of((update) => {
          if (update.docChanged) emit('update:source', update.state.doc.toString());
          if (update.selectionSet) emit('cursor', update.state.selection.main.head);
        }),
      ],
    }),
  });
  view.dispatch({ effects: setLines.of(props.lines) });
  dark.addEventListener('change', applyTheme);
});

onBeforeUnmount(() => {
  dark.removeEventListener('change', applyTheme);
  view?.destroy();
});

watch(
  () => props.source,
  (source) => {
    if (!view || source === view.state.doc.toString()) return;
    view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: source } });
  },
);

watch(
  () => props.lines,
  (lines) => view?.dispatch({ effects: setLines.of(lines) }),
);

watch(
  () => props.highlight,
  (span) => {
    if (!view) return;
    const scroll = span && span[0] < view.state.doc.length ? [EditorView.scrollIntoView(span[0], { y: 'nearest' })] : [];
    view.dispatch({ effects: [setHighlight.of(span), ...scroll] });
  },
);
</script>

<template>
  <div ref="host" class="source-editor" />
</template>

<style scoped>
.source-editor {
  height: 100%;
  overflow: hidden;
}
.source-editor :deep(.cm-editor) {
  height: 100%;
}
.source-editor :deep(.cm-scroller) {
  font-family: var(--mono);
  font-size: 13px;
}
.source-editor :deep(.cm-sourced) {
  background: color-mix(in srgb, var(--src) var(--src-tint), transparent);
  box-shadow: inset 3px 0 0 var(--src);
}
.source-editor :deep(.cm-ir-span) {
  background: var(--span-highlight);
  border-radius: 2px;
}
</style>

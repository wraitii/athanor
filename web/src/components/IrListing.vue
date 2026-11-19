<script setup lang="ts">
import { nextTick, ref, watch } from 'vue';
import type { Line, Span } from '../listing';

const props = defineProps<{ lines: Line[]; diff: boolean; cursor: number | null }>();
const emit = defineEmits<{ hover: [Span | null] }>();

const host = ref<HTMLElement>();

function atCursor(line: Line): boolean {
  return props.cursor !== null && line.span !== null && line.span[0] <= props.cursor && props.cursor < line.span[1];
}

watch(
  () => props.cursor,
  async () => {
    await nextTick();
    host.value?.querySelector('.at-cursor')?.scrollIntoView({ block: 'nearest' });
  },
);
</script>

<template>
  <div ref="host" class="ir-listing" @mouseleave="emit('hover', null)">
    <template v-for="(line, index) in lines" :key="index">
      <div
        v-if="diff || line.kind !== 'removed'"
        class="line"
        :class="[diff ? line.kind : 'same', { 'at-cursor': atCursor(line), spanned: line.span }]"
        @mouseenter="emit('hover', line.span)"
      ><span class="gutter">{{ diff && line.kind === 'added' ? '+' : diff && line.kind === 'removed' ? '−' : ' ' }}</span>{{ line.text }}</div>
    </template>
  </div>
</template>

<style scoped>
.ir-listing {
  height: 100%;
  overflow: auto;
  font-family: var(--mono);
  font-size: 12px;
  line-height: 1.55;
  padding: 8px 0;
}
.line {
  white-space: pre;
  padding: 0 12px 0 4px;
  min-width: max-content;
}
.gutter {
  display: inline-block;
  width: 1.5ch;
  color: var(--muted);
  user-select: none;
}
.line.added {
  background: var(--added);
}
.line.added .gutter {
  color: var(--added-strong);
}
.line.removed {
  background: var(--removed);
  color: var(--muted);
  text-decoration: line-through;
  text-decoration-color: var(--removed-strong);
}
.line.removed .gutter {
  color: var(--removed-strong);
}
.line.spanned:hover {
  background: var(--hover);
}
.line.at-cursor {
  box-shadow: inset 3px 0 0 var(--accent);
  background: var(--cursor-line);
}
</style>

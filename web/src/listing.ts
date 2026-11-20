import { diffArrays } from 'diff';

export type Span = [number, number];

export interface Line {
  text: string;
  span: Span | null;
  source: number | null;
  kind: 'same' | 'added' | 'removed';
}

const SPAN = /;\s*(\d+)\.\.(\d+)/;
const LOCATION = /\s+@[0-9a-f]+$/;
const BLOCK = /^(block\d+)[(:]/;
const MAX_EDITS = 600;

type Kind = Line['kind'];

interface Segment {
  kind: Kind;
  lines: string[];
}

export function byteOffsets(source: string): (byte: number) => number {
  if (/^[\x00-\x7f]*$/.test(source)) {
    return (byte) => byte;
  }
  const table: number[] = [];
  const encoder = new TextEncoder();
  let unit = 0;
  for (const char of source) {
    const bytes = encoder.encode(char).length;
    for (let i = 0; i < bytes; i++) table.push(unit);
    unit += char.length;
  }
  table.push(unit);
  return (byte) => table[Math.min(byte, table.length - 1)];
}

export function sourceLines(source: string): (unit: number) => number {
  const starts = [0];
  for (let i = source.indexOf('\n'); i >= 0; i = source.indexOf('\n', i + 1)) starts.push(i + 1);
  return (unit) => {
    let low = 0;
    let high = starts.length - 1;
    while (low < high) {
      const middle = (low + high + 1) >> 1;
      if (starts[middle] <= unit) low = middle;
      else high = middle - 1;
    }
    return low;
  };
}

function lines(text: string | undefined): string[] {
  if (!text) return [];
  const split = text.split('\n');
  return split[split.length - 1] === '' ? split.slice(0, -1) : split;
}

function span(text: string, toUnits: (byte: number) => number): Span | null {
  const match = SPAN.exec(text);
  if (!match) return null;
  return [toUnits(Number(match[1])), toUnits(Number(match[2]))];
}

interface Group {
  key: string;
  lines: string[];
}

function groups(lines: string[]): Group[] {
  const result: Group[] = [{ key: '', lines: [] }];
  const seen = new Map<string, number>();
  for (const line of lines) {
    const label = BLOCK.exec(line)?.[1];
    if (label) {
      const count = seen.get(label) ?? 0;
      seen.set(label, count + 1);
      result.push({ key: count ? `${label}#${count}` : label, lines: [] });
    }
    result[result.length - 1].lines.push(line);
  }
  return result;
}

function diffLines(previous: string | undefined, current: string): Segment[] {
  const after = lines(current);
  if (previous === current) return [{ kind: 'same', lines: after }];
  const before = groups(lines(previous));
  const matched = new Map(before.map((group) => [group.key, group]));
  const currentGroups = groups(after);
  const present = new Set(currentGroups.map((group) => group.key));
  const leading: Segment[] = [];
  const orphans = new Map<string, Segment[]>();
  let anchor: string | null = null;
  for (const group of before) {
    if (present.has(group.key)) {
      anchor = group.key;
      continue;
    }
    const removed: Segment = { kind: 'removed', lines: group.lines };
    if (anchor === null) leading.push(removed);
    else orphans.set(anchor, [...(orphans.get(anchor) ?? []), removed]);
  }
  const segments = leading;
  for (const group of currentGroups) {
    const old = matched.get(group.key);
    if (old) segments.push(...diffGroup(old.lines, group.lines));
    else segments.push({ kind: 'added', lines: group.lines });
    segments.push(...(orphans.get(group.key) ?? []));
  }
  return segments;
}

function diffGroup(before: string[], after: string[]): Segment[] {
  const strip = (text: string) => text.replace(LOCATION, '');
  const changes = diffArrays(before.map(strip), after.map(strip), { maxEditLength: MAX_EDITS });
  if (!changes) {
    return [
      { kind: 'removed', lines: before },
      { kind: 'added', lines: after },
    ];
  }
  const segments: Segment[] = [];
  let i = 0;
  let j = 0;
  for (const change of changes) {
    const count = change.value.length;
    if (change.added) {
      segments.push({ kind: 'added', lines: after.slice(j, (j += count)) });
    } else if (change.removed) {
      segments.push({ kind: 'removed', lines: before.slice(i, (i += count)) });
    } else {
      segments.push({ kind: 'same', lines: after.slice(j, (j += count)) });
      i += count;
    }
  }
  return segments;
}

export function diffListing(
  previous: string | undefined,
  current: string,
  toUnits: (byte: number) => number,
  lineOf: (unit: number) => number,
): Line[] {
  const result: Line[] = [];
  for (const { kind, lines } of diffLines(previous, current)) {
    for (const text of lines) {
      const range = span(text, toUnits);
      result.push({ text, span: range, source: range && lineOf(range[0]), kind });
    }
  }
  return result;
}

export function changeCount(previous: string | undefined, current: string) {
  let added = 0;
  let removed = 0;
  for (const { kind, lines } of diffLines(previous, current)) {
    if (kind === 'added') added += lines.length;
    if (kind === 'removed') removed += lines.length;
  }
  return { added, removed };
}

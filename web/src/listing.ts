import { diffArrays } from 'diff';

export type Span = [number, number];

export interface Line {
  text: string;
  span: Span | null;
  kind: 'same' | 'added' | 'removed';
}

const SPAN = /;\s*(\d+)\.\.(\d+)/;

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

export function diffListing(
  previous: string | undefined,
  current: string,
  toUnits: (byte: number) => number,
): Line[] {
  const result: Line[] = [];
  for (const change of diffArrays(lines(previous), lines(current))) {
    const kind = change.added ? 'added' : change.removed ? 'removed' : 'same';
    for (const text of change.value) {
      result.push({ text, span: span(text, toUnits), kind });
    }
  }
  return result;
}

export function changeCount(previous: string | undefined, current: string) {
  let added = 0;
  let removed = 0;
  for (const change of diffArrays(lines(previous), lines(current))) {
    if (change.added) added += change.value.length;
    if (change.removed) removed += change.value.length;
  }
  return { added, removed };
}

import type { Bound, GpuKernel } from '../types';
import { DISPATCHED, type CallStats, type LogLevel, type RunEvent, type RunRequest } from './protocol';

type Plan =
  | { kind: 'fallback'; reason: string }
  | { kind: 'dispatch'; groups: [number, number]; uniforms: Int32Array; buffers: Float32Array[] };

const MAX_GROUPS = 65535;
const FLUSH_MS = 250;

const post = (event: RunEvent) => self.postMessage(event);

const stats = new Map<string, CallStats>();
let flushed = 0;

function flush() {
  flushed = performance.now();
  post({ type: 'stats', stats: [...stats.values()] });
}

function entry(name: string): CallStats {
  let current = stats.get(name);
  if (!current) {
    current = { name, calls: 0, v8Ms: 0, gpu: 0, gpuMs: 0, maxAbs: 0, maxRel: 0, fallbacks: {} };
    stats.set(name, current);
  }
  return current;
}

function recorded() {
  if (performance.now() - flushed > FLUSH_MS) flush();
}

function format(value: unknown): string {
  if (typeof value === 'string') return value;
  if (ArrayBuffer.isView(value) && !(value instanceof DataView)) {
    const items = Array.from((value as Float32Array).subarray(0, 8), String);
    const more = (value as Float32Array).length > 8 ? ', …' : '';
    return `${value.constructor.name}(${(value as Float32Array).length}) [${items.join(', ')}${more}]`;
  }
  if (value instanceof Error) return value.stack ?? String(value);
  if (typeof value === 'object' && value !== null) {
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }
  return String(value);
}

for (const level of ['log', 'info', 'warn', 'error'] as LogLevel[]) {
  console[level] = (...args: unknown[]) => post({ type: 'log', level, text: args.map(format).join(' ') });
}

function int32(value: unknown): number | null {
  return typeof value === 'number' && (value | 0) === value ? value : null;
}

function bound(spec: Bound, args: unknown[]): number | null {
  if (spec.arg === null) return spec.add;
  const value = int32(args[spec.arg]);
  if (value === null) return null;
  const sum = value + spec.add;
  return (sum | 0) === sum ? sum : null;
}

function overlaps(a: Float32Array, b: Float32Array): boolean {
  return a.buffer === b.buffer && a.byteOffset < b.byteOffset + b.byteLength && b.byteOffset < a.byteOffset + a.byteLength;
}

function plan(kernel: GpuKernel, args: unknown[]): Plan {
  const fallback = (reason: string): Plan => ({ kind: 'fallback', reason });
  const uniforms = new Int32Array(Math.ceil(kernel.uniforms.length / 4) * 4);
  for (const [index, arg] of kernel.uniforms.entries()) {
    const value = int32(args[arg]);
    if (value === null) return fallback(`argument ${arg} is not an int32`);
    uniforms[index] = value;
  }
  const extents: number[] = [];
  for (const axis of kernel.axes) {
    const start = bound(axis.start, args);
    const end = bound(axis.end, args);
    if (start === null || end === null) return fallback('grid bounds are not int32');
    extents.push(end - start);
  }
  if (extents.some((extent) => extent <= 0)) return fallback('empty grid');
  for (const check of kernel.strides) {
    const { arg, factor } = check.stride;
    const scale = arg === null ? 1 : int32(args[arg]);
    if (scale === null) return fallback(`argument ${arg} is not an int32`);
    if (Math.abs(scale * factor) < check.inner * extents[check.axis]) return fallback('rows overlap');
  }
  const buffers: Float32Array[] = [];
  for (const { arg } of kernel.buffers) {
    const buffer = args[arg];
    if (!(buffer instanceof Float32Array)) return fallback(`argument ${arg} is not a Float32Array`);
    if (buffer.byteLength === 0) return fallback('empty buffer');
    buffers.push(buffer);
  }
  for (const [index, { writable }] of kernel.buffers.entries()) {
    if (!writable) continue;
    if (buffers.some((other, j) => j !== index && overlaps(buffers[index], other))) return fallback('buffers alias');
  }
  const groups: [number, number] = [
    Math.ceil(extents[0] / kernel.workgroup[0]),
    Math.ceil((extents[1] ?? 1) / kernel.workgroup[1]),
  ];
  if (groups[0] > MAX_GROUPS || groups[1] > MAX_GROUPS) return fallback('grid too large');
  return { kind: 'dispatch', groups, uniforms, buffers };
}

function dispatch(index: number, kernel: GpuKernel, plan: Extract<Plan, { kind: 'dispatch' }>): boolean {
  const shared = plan.buffers.map((buffer) => {
    const copy = new Float32Array(new SharedArrayBuffer(buffer.byteLength));
    copy.set(buffer);
    return copy;
  });
  const signal = new Int32Array(new SharedArrayBuffer(4));
  post({ type: 'dispatch', kernel: index, buffers: shared, uniforms: plan.uniforms, groups: plan.groups, signal });
  Atomics.wait(signal, 0, 0);
  if (Atomics.load(signal, 0) !== DISPATCHED) return false;
  for (const [i, { writable }] of kernel.buffers.entries()) {
    if (writable) plan.buffers[i].set(shared[i].subarray(0, plan.buffers[i].length));
  }
  return true;
}

function difference(reference: Float32Array, actual: Float32Array, totals: CallStats) {
  let scale = 0;
  for (const value of reference) scale = Math.max(scale, Math.abs(value));
  for (let i = 0; i < reference.length; i++) {
    const a = reference[i];
    const b = actual[i];
    const abs = Number.isNaN(a) || Number.isNaN(b) ? (Number.isNaN(a) && Number.isNaN(b) ? 0 : Infinity) : Math.abs(a - b);
    totals.maxAbs = Math.max(totals.maxAbs, abs);
    totals.maxRel = Math.max(totals.maxRel, scale > 0 ? abs / scale : abs);
  }
}

function wrap(fn: Function, index: number, name: string, kernel: GpuKernel, request: RunRequest): Function {
  return function (this: unknown, ...args: unknown[]) {
    const totals = entry(name);
    totals.calls++;
    const v8 = (target: unknown[]) => {
      const started = performance.now();
      const result = fn.apply(this, target);
      totals.v8Ms += performance.now() - started;
      return result;
    };
    const fallback = (reason: string) => {
      totals.fallbacks[reason] = (totals.fallbacks[reason] ?? 0) + 1;
      const result = v8(args);
      recorded();
      return result;
    };
    if (!request.gpu) {
      const result = v8(args);
      recorded();
      return result;
    }
    if (typeof SharedArrayBuffer === 'undefined') return fallback('page is not cross-origin isolated');
    const call = plan(kernel, args);
    if (call.kind === 'fallback') return fallback(call.reason);
    const copies = args.slice();
    const references: [Float32Array, Float32Array][] = [];
    for (const [i, { arg, writable }] of kernel.buffers.entries()) {
      if (!writable) continue;
      const copy = call.buffers[i].slice();
      copies[arg] = copy;
      references.push([copy, call.buffers[i]]);
    }
    v8(copies);
    const started = performance.now();
    if (!dispatch(index, kernel, call)) {
      for (const [reference, actual] of references) actual.set(reference);
      totals.fallbacks['WebGPU dispatch failed'] = (totals.fallbacks['WebGPU dispatch failed'] ?? 0) + 1;
      recorded();
      return undefined;
    }
    totals.gpu++;
    totals.gpuMs += performance.now() - started;
    for (const [reference, actual] of references) difference(reference, actual, totals);
    recorded();
    return undefined;
  };
}

self.onmessage = (event: MessageEvent<RunRequest>) => {
  const request = event.data;
  const scope = globalThis as unknown as Record<string, unknown>;
  scope.__athanor = {
    wrap: (fn: unknown, index: number) =>
      typeof fn === 'function' ? wrap(fn, index, request.kernels[index].name, request.kernels[index].kernel, request) : fn,
  };
  const prelude = request.kernels
    .map(({ name }, index) => `if (typeof ${name} === 'function') ${name} = __athanor.wrap(${name}, ${index});`)
    .join(' ');
  const started = performance.now();
  try {
    (0, eval)(prelude + request.source);
    if (typeof scope.main === 'function') (scope.main as () => unknown)();
    flush();
    post({ type: 'done', ms: performance.now() - started });
  } catch (error) {
    flush();
    post({ type: 'failed', text: error instanceof Error ? (error.stack ?? String(error)) : String(error) });
  }
};

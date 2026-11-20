import type { GpuKernel } from '../types';

export interface RunKernel {
  name: string;
  kernel: GpuKernel;
}

export interface RunRequest {
  source: string;
  kernels: RunKernel[];
  gpu: boolean;
}

export interface DispatchRequest {
  type: 'dispatch';
  kernel: number;
  buffers: Float32Array[];
  uniforms: Int32Array;
  groups: [number, number];
  signal: Int32Array;
}

export interface CallStats {
  name: string;
  calls: number;
  v8Ms: number;
  gpu: number;
  gpuMs: number;
  maxAbs: number;
  maxRel: number;
  fallbacks: Record<string, number>;
}

export type LogLevel = 'log' | 'info' | 'warn' | 'error';

export type RunEvent =
  | { type: 'log'; level: LogLevel; text: string }
  | { type: 'stats'; stats: CallStats[] }
  | DispatchRequest
  | { type: 'done'; ms: number }
  | { type: 'failed'; text: string };

export const DISPATCHED = 1;
export const DECLINED = 2;

export interface Pass {
  pass: string;
  text: string;
}

export interface Bound {
  arg: number | null;
  add: number;
}

export interface Stride {
  stride: { arg: number | null; factor: number };
  inner: number;
  axis: number;
}

export interface GpuKernel {
  wgsl: string;
  buffers: { arg: number; writable: boolean }[];
  uniforms: number[];
  axes: { start: Bound; end: Bound }[];
  strides: Stride[];
  workgroup: [number, number, number];
}

export type GpuReport = GpuKernel | { error: string };

export interface Report {
  name: string;
  label: string;
  direct: boolean;
  semantic: string | null;
  passes: Pass[];
  gpu: GpuReport | null;
}

export interface CompileOptions {
  inlining: boolean;
  versioning: boolean;
  continuations: boolean;
}

export interface CompileRequest {
  id: number;
  source: string;
  filter: string;
  options: CompileOptions;
}

export type CompileResponse =
  | { id: number; ok: true; reports: Report[]; milliseconds: number }
  | { id: number; ok: false; error: string; fatal: boolean };

export interface Pass {
  pass: string;
  text: string;
}

export interface Report {
  name: string;
  label: string;
  direct: boolean;
  semantic: string | null;
  passes: Pass[];
}

export interface CompileRequest {
  id: number;
  source: string;
  filter: string;
}

export type CompileResponse =
  | { id: number; ok: true; reports: Report[]; milliseconds: number }
  | { id: number; ok: false; error: string; fatal: boolean };

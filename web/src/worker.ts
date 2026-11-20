import init, { compile } from './wasm/js_web.js';
import type { CompileRequest, CompileResponse, Report } from './types';

const ready = init();

self.onmessage = async (event: MessageEvent<CompileRequest>) => {
  await ready;
  const { id, source, filter, options } = event.data;
  let response: CompileResponse;
  try {
    const started = performance.now();
    const reports = JSON.parse(compile(source, filter || undefined, options.inlining, options.versioning, options.continuations)) as Report[];
    response = { id, ok: true, reports, milliseconds: performance.now() - started };
  } catch (error) {
    response = { id, ok: false, error: String(error), fatal: error instanceof WebAssembly.RuntimeError };
  }
  self.postMessage(response);
};

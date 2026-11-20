import { GpuDispatcher } from './gpu';
import type { RunEvent, RunRequest } from './protocol';

export class Runner {
  private worker: Worker | undefined;
  private gpu = new GpuDispatcher();

  run(request: RunRequest, listener: (event: RunEvent) => void) {
    this.stop();
    const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    this.worker = worker;
    const failures = new Set<string>();
    worker.onmessage = async (event: MessageEvent<RunEvent>) => {
      const message = event.data;
      if (message.type !== 'dispatch') {
        listener(message);
        if (message.type === 'done' || message.type === 'failed') this.stop(worker);
        return;
      }
      const failure = await this.gpu.dispatch(message, request.kernels[message.kernel].kernel);
      if (failure === null || failures.has(failure)) return;
      failures.add(failure);
      listener({ type: 'log', level: 'warn', text: `GPU: ${failure}` });
    };
    worker.onerror = (event) => {
      listener({ type: 'failed', text: event.message });
      this.stop(worker);
    };
    worker.postMessage(request);
  }

  stop(worker = this.worker) {
    worker?.terminate();
    if (worker === this.worker) this.worker = undefined;
  }

  get running(): boolean {
    return this.worker !== undefined;
  }
}

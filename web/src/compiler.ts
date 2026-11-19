import type { CompileResponse } from './types';

type Pending = (response: CompileResponse) => void;

export class Compiler {
  private worker = Compiler.spawn();
  private pending = new Map<number, Pending>();
  private next = 0;

  private static spawn(): Worker {
    return new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
  }

  constructor() {
    this.listen();
  }

  private listen() {
    this.worker.onmessage = (event: MessageEvent<CompileResponse>) => {
      const response = event.data;
      this.pending.get(response.id)?.(response);
      this.pending.delete(response.id);
      if (!response.ok && response.fatal) {
        this.worker.terminate();
        this.worker = Compiler.spawn();
        this.listen();
      }
    };
  }

  compile(source: string, filter: string): Promise<CompileResponse> {
    const id = this.next++;
    return new Promise((resolve) => {
      this.pending.set(id, resolve);
      this.worker.postMessage({ id, source, filter });
    });
  }
}

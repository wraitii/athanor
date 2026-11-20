import { changeCount } from './listing';

export interface DiffRequest {
  id: number;
  pairs: [string | undefined, string][];
}

export interface DiffResponse {
  id: number;
  counts: { added: number; removed: number }[];
}

self.onmessage = (event: MessageEvent<DiffRequest>) => {
  const { id, pairs } = event.data;
  const response: DiffResponse = { id, counts: pairs.map(([previous, current]) => changeCount(previous, current)) };
  self.postMessage(response);
};

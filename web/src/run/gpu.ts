import type { GpuKernel } from '../types';
import { DECLINED, DISPATCHED, type DispatchRequest } from './protocol';

export class GpuDispatcher {
  private device: Promise<GPUDevice> | undefined;
  private pipelines = new Map<string, Promise<GPUComputePipeline>>();

  private acquire(): Promise<GPUDevice> {
    this.device ??= (async () => {
      if (!navigator.gpu) throw new Error('WebGPU is unavailable in this browser');
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error('no WebGPU adapter');
      const device = await adapter.requestDevice({
        requiredLimits: {
          maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
          maxBufferSize: adapter.limits.maxBufferSize,
        },
      });
      void device.lost.then(() => {
        this.device = undefined;
        this.pipelines.clear();
      });
      return device;
    })();
    this.device.catch(() => (this.device = undefined));
    return this.device;
  }

  private pipeline(device: GPUDevice, wgsl: string): Promise<GPUComputePipeline> {
    let pipeline = this.pipelines.get(wgsl);
    if (!pipeline) {
      const module = device.createShaderModule({ code: wgsl });
      pipeline = device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
      pipeline.catch(() => this.pipelines.delete(wgsl));
      this.pipelines.set(wgsl, pipeline);
    }
    return pipeline;
  }

  async dispatch(request: DispatchRequest, kernel: GpuKernel): Promise<string | null> {
    let failure: string | null = null;
    try {
      await this.run(request, kernel);
    } catch (error) {
      failure = error instanceof Error ? error.message : String(error);
    }
    Atomics.store(request.signal, 0, failure === null ? DISPATCHED : DECLINED);
    Atomics.notify(request.signal, 0);
    return failure;
  }

  private async run(request: DispatchRequest, kernel: GpuKernel) {
    const device = await this.acquire();
    const pipeline = await this.pipeline(device, kernel.wgsl);
    const created: GPUBuffer[] = [];
    const create = (descriptor: GPUBufferDescriptor) => {
      const buffer = device.createBuffer(descriptor);
      created.push(buffer);
      return buffer;
    };
    device.pushErrorScope('validation');
    device.pushErrorScope('out-of-memory');
    try {
      const storage = request.buffers.map((data, index) => {
        const copySource = kernel.buffers[index].writable ? GPUBufferUsage.COPY_SRC : 0;
        const buffer = create({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | copySource });
        device.queue.writeBuffer(buffer, 0, data);
        return buffer;
      });
      const entries: GPUBindGroupEntry[] = storage.map((buffer, binding) => ({ binding, resource: { buffer } }));
      if (request.uniforms.length) {
        const uniforms = create({ size: request.uniforms.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(uniforms, 0, request.uniforms);
        entries.push({ binding: storage.length, resource: { buffer: uniforms } });
      }
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }));
      pass.dispatchWorkgroups(request.groups[0], request.groups[1], 1);
      pass.end();
      const readbacks: [number, GPUBuffer][] = [];
      for (const [index, { writable }] of kernel.buffers.entries()) {
        if (!writable) continue;
        const staging = create({ size: storage[index].size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        encoder.copyBufferToBuffer(storage[index], 0, staging, 0, storage[index].size);
        readbacks.push([index, staging]);
      }
      device.queue.submit([encoder.finish()]);
      const outOfMemory = await device.popErrorScope();
      const invalid = await device.popErrorScope();
      if (outOfMemory ?? invalid) throw new Error((outOfMemory ?? invalid)!.message);
      for (const [index, staging] of readbacks) {
        await staging.mapAsync(GPUMapMode.READ);
        request.buffers[index].set(new Float32Array(staging.getMappedRange()));
        staging.unmap();
      }
    } finally {
      for (const buffer of created) buffer.destroy();
    }
  }
}

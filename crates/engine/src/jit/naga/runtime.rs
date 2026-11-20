use std::borrow::Cow;
use std::sync::{Arc, LazyLock, Mutex, OnceLock};

use super::kernel::{AxisSpec, BoundSpec, GpuKernel, Scale};
use crate::FxHashMap;
use crate::jit::session::JitCtx;
use crate::value::Value;

fn int_arg(args: &[Value], index: usize) -> Option<i32> {
    let number = args.get(index)?.as_number()?;
    let integer = number as i32;
    (f64::from(integer) == number).then_some(integer)
}

impl BoundSpec {
    fn eval(&self, args: &[Value]) -> Option<i32> {
        match *self {
            Self::Const(value) => Some(value),
            Self::Arg(index) => int_arg(args, index),
            Self::ArgAdd(index, delta) => int_arg(args, index)?.checked_add(delta),
        }
    }
}

impl Scale {
    fn eval(&self, args: &[Value]) -> Option<i64> {
        match *self {
            Self::Const(value) => Some(value),
            Self::Arg(index, factor) => i64::from(int_arg(args, index)?).checked_mul(factor),
        }
    }
}

static KERNELS: LazyLock<Mutex<Vec<Arc<GpuKernel>>>> = LazyLock::new(|| Mutex::new(Vec::new()));

pub(super) fn register_kernel(kernel: GpuKernel) -> usize {
    let mut kernels = KERNELS.lock().unwrap();
    kernels.push(Arc::new(kernel));
    kernels.len() - 1
}

struct GpuRuntime {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Mutex<FxHashMap<usize, wgpu::ComputePipeline>>,
}

static RUNTIME: OnceLock<Result<GpuRuntime, String>> = OnceLock::new();

impl GpuRuntime {
    fn get() -> Result<&'static Self, String> {
        RUNTIME
            .get_or_init(|| {
                let instance =
                    wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
                let adapter = pollster::block_on(
                    instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
                )
                .map_err(|e| format!("wgpu adapter failed: {e}"))?;
                let (device, queue) =
                    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                        .map_err(|e| format!("wgpu device failed: {e}"))?;
                Ok(Self {
                    device,
                    queue,
                    pipelines: Mutex::new(FxHashMap::default()),
                })
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    fn dispatch(
        &self,
        kernel_id: usize,
        kernel: &GpuKernel,
        args: &[Value],
        ctx: &JitCtx,
    ) -> Result<bool, String> {
        let mut uniform_bytes = Vec::with_capacity(kernel.uniforms.len().div_ceil(4) * 16);
        for &index in &kernel.uniforms {
            let Some(value) = int_arg(args, index) else {
                return Ok(false);
            };
            uniform_bytes.extend_from_slice(&value.to_le_bytes());
        }
        uniform_bytes.resize(uniform_bytes.len().div_ceil(16) * 16, 0);
        let extent = |axis: &AxisSpec| {
            Some(i64::from(axis.end.eval(args)?) - i64::from(axis.start.eval(args)?))
        };
        let Some(dim_x) = extent(&kernel.axis_x) else {
            return Ok(false);
        };
        let dim_y = match &kernel.axis_y {
            Some(axis_y) => {
                let Some(dim_y) = extent(axis_y) else {
                    return Ok(false);
                };
                dim_y
            }
            None => 1,
        };
        if dim_x <= 0 || dim_y <= 0 {
            return Ok(true);
        }
        let extents = [dim_x.unsigned_abs(), dim_y.unsigned_abs()];
        for check in &kernel.strides {
            let Some(stride) = check.stride.eval(args) else {
                return Ok(false);
            };
            if stride.unsigned_abs() < check.inner.saturating_mul(extents[check.inner_axis]) {
                return Ok(false);
            }
        }
        let heap = ctx.global.heap();
        let mut host_buffers: Vec<(Value, Vec<f32>, bool)> =
            Vec::with_capacity(kernel.buffers.len());
        for buf in &kernel.buffers {
            let Some(&val) = args.get(buf.arg_index) else {
                return Ok(false);
            };
            let Some(data) = heap.float32_copy(val) else {
                return Ok(false);
            };
            host_buffers.push((val, data, buf.writable));
        }
        for (i, (val_a, _, writable_a)) in host_buffers.iter().enumerate() {
            if !writable_a {
                continue;
            }
            for (j, (val_b, _, _)) in host_buffers.iter().enumerate() {
                if i != j && val_a.bits() == val_b.bits() {
                    return Ok(false);
                }
            }
        }
        let pipeline = {
            let mut map = self.pipelines.lock().unwrap();
            map.entry(kernel_id)
                .or_insert_with(|| {
                    let shader = self
                        .device
                        .create_shader_module(wgpu::ShaderModuleDescriptor {
                            label: None,
                            source: wgpu::ShaderSource::Naga(Cow::Owned(kernel.module.clone())),
                        });
                    self.device
                        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: None,
                            layout: None,
                            module: &shader,
                            entry_point: Some("main"),
                            compilation_options: Default::default(),
                            cache: None,
                        })
                })
                .clone()
        };
        let mut gpu_buffers = Vec::with_capacity(host_buffers.len());
        for (_, data, writable) in &host_buffers {
            let bytes = f32_as_bytes(data);
            let usage = if *writable {
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC
            } else {
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
            };
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: bytes.len().max(4) as u64,
                usage,
                mapped_at_creation: false,
            });
            if !bytes.is_empty() {
                self.queue.write_buffer(&buffer, 0, bytes);
            }
            gpu_buffers.push(buffer);
        }
        let uniform_buffer = (!uniform_bytes.is_empty()).then(|| {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: uniform_bytes.len() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&buffer, 0, &uniform_bytes);
            buffer
        });
        let flag = kernel.flag.map(|binding| {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&buffer, 0, &[0; 4]);
            (binding, buffer)
        });
        let mut entries = Vec::with_capacity(
            gpu_buffers.len() + usize::from(uniform_buffer.is_some()) + usize::from(flag.is_some()),
        );
        for (i, buffer) in gpu_buffers.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            });
        }
        if let Some(ref ubuf) = uniform_buffer {
            entries.push(wgpu::BindGroupEntry {
                binding: gpu_buffers.len() as u32,
                resource: ubuf.as_entire_binding(),
            });
        }
        if let Some((binding, buffer)) = &flag {
            entries.push(wgpu::BindGroupEntry {
                binding: *binding,
                resource: buffer.as_entire_binding(),
            });
        }
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let gx = (dim_x as u32).div_ceil(kernel.workgroup_size[0]);
            let gy = (dim_y as u32).div_ceil(kernel.workgroup_size[1]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        let staging = |size: u64| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let flag = flag.map(|(_, buffer)| {
            let copy = staging(4);
            encoder.copy_buffer_to_buffer(&buffer, 0, &copy, 0, 4);
            copy
        });
        let mut readbacks = Vec::new();
        for (i, (val, data, writable)) in host_buffers.iter().enumerate() {
            if !writable || data.is_empty() {
                continue;
            }
            let size = (data.len() * 4) as u64;
            let copy = staging(size);
            encoder.copy_buffer_to_buffer(&gpu_buffers[i], 0, &copy, 0, size);
            readbacks.push((*val, data.len(), copy));
        }
        self.queue.submit(Some(encoder.finish()));
        if let Some(flag) = flag {
            let bytes = self.read(&flag)?;
            if bytes.iter().any(|&byte| byte != 0) {
                return Ok(false);
            }
        }
        let mut results = Vec::with_capacity(readbacks.len());
        for (val, len, copy) in readbacks {
            let bytes = self.read(&copy)?;
            let mut out = vec![0.0f32; len];
            for (slot, chunk) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
                *slot = f32::from_le_bytes(*chunk);
            }
            results.push((val, out));
        }
        for (val, out) in results {
            heap.float32_write(val, &out);
        }
        Ok(true)
    }

    fn read(&self, buffer: &wgpu::Buffer) -> Result<Vec<u8>, String> {
        let slice = buffer.slice(..);
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = done.clone();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            if res.is_ok() {
                flag.store(true, std::sync::atomic::Ordering::Release);
            }
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        if !done.load(std::sync::atomic::Ordering::Acquire) {
            return Err("readback failed".into());
        }
        let bytes = slice
            .get_mapped_range()
            .map_err(|e| format!("{e}"))?
            .to_vec();
        buffer.unmap();
        Ok(bytes)
    }
}

fn f32_as_bytes(data: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) }
}

fn gpu_log_enabled() -> bool {
    std::env::var("JS_JIT_GPU_LOG").as_deref() == Ok("1")
}

pub(crate) extern "C" fn js_gpu_dispatch(
    ctx: *mut JitCtx,
    kernel_id: usize,
    args_array: Value,
) -> u64 {
    let log = gpu_log_enabled();
    let ctx = unsafe { &*ctx };
    let Some(args) = ctx.global.heap().array_elements_copy(args_array) else {
        if log {
            eprintln!("[js jit gpu] kernel={kernel_id} fallback: invalid args");
        }
        return 0;
    };
    let Some(kernel) = KERNELS.lock().unwrap().get(kernel_id).cloned() else {
        return 0;
    };
    let runtime = match GpuRuntime::get() {
        Ok(runtime) => runtime,
        Err(error) => {
            if log {
                eprintln!("[js jit gpu] kernel={kernel_id} fallback: {error}");
            }
            return 0;
        }
    };
    match runtime.dispatch(kernel_id, &kernel, &args, ctx) {
        Ok(true) => {
            if log {
                eprintln!("[js jit gpu] kernel={kernel_id} dispatched on GPU");
            }
            1
        }
        Ok(false) => {
            if log {
                eprintln!("[js jit gpu] kernel={kernel_id} fallback: guard declined");
            }
            0
        }
        Err(error) => {
            if log {
                eprintln!("[js jit gpu] kernel={kernel_id} fallback: {error}");
            }
            0
        }
    }
}

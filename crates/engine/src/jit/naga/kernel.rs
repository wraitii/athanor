use std::fmt;

pub(super) enum BoundSpec {
    Const(i32),
    Arg(usize),
    ArgAdd(usize, i32),
}

pub(super) struct AxisSpec {
    pub(super) start: BoundSpec,
    pub(super) end: BoundSpec,
}

pub(super) struct KernelBuffer {
    pub(super) arg_index: usize,
    pub(super) writable: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Scale {
    Const(i64),
    Arg(usize, i64),
}

pub(super) struct StrideCheck {
    pub(super) stride: Scale,
    pub(super) inner: u64,
    pub(super) inner_axis: usize,
}

pub(super) struct GpuKernel {
    pub(super) wgsl: String,
    #[cfg(feature = "gpu")]
    pub(super) module: naga::Module,
    pub(super) buffers: Vec<KernelBuffer>,
    pub(super) uniforms: Vec<usize>,
    pub(super) axis_x: AxisSpec,
    pub(super) axis_y: Option<AxisSpec>,
    pub(super) strides: Vec<StrideCheck>,
    pub(super) workgroup_size: [u32; 3],
}

impl fmt::Display for BoundSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Const(value) => write!(f, "{value}"),
            Self::Arg(index) => write!(f, "arg{index}"),
            Self::ArgAdd(index, delta) => write!(f, "arg{index}{delta:+}"),
        }
    }
}

impl fmt::Display for Scale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Const(value) => write!(f, "{value}"),
            Self::Arg(index, 1) => write!(f, "arg{index}"),
            Self::Arg(index, factor) => write!(f, "{factor}*arg{index}"),
        }
    }
}

impl fmt::Display for GpuKernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "// grid: x in [{}, {})", self.axis_x.start, self.axis_x.end)?;
        if let Some(axis_y) = &self.axis_y {
            write!(f, ", y in [{}, {})", axis_y.start, axis_y.end)?;
        }
        writeln!(f, ", workgroup {:?}", self.workgroup_size)?;
        for buffer in &self.buffers {
            let access = if buffer.writable { "read_write" } else { "read" };
            writeln!(f, "// buffer arg{}: Float32Array, {access}", buffer.arg_index)?;
        }
        for index in &self.uniforms {
            writeln!(f, "// uniform arg{index}: int32")?;
        }
        for check in &self.strides {
            let axis = ["x", "y"][check.inner_axis];
            writeln!(f, "// guard |{}| >= {} * extent.{axis}", check.stride, check.inner)?;
        }
        f.write_str(&self.wgsl)
    }
}

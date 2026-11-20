use std::fmt;

#[derive(Clone, Debug)]
pub enum BoundSpec {
    Const(i32),
    Arg(usize),
    ArgAdd(usize, i32),
}

#[derive(Clone, Debug)]
pub struct AxisSpec {
    pub start: BoundSpec,
    pub end: BoundSpec,
}

#[derive(Clone, Debug)]
pub struct KernelBuffer {
    pub arg_index: usize,
    pub writable: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scale {
    Const(i64),
    Arg(usize, i64),
}

#[derive(Clone, Debug)]
pub struct StrideCheck {
    pub stride: Scale,
    pub inner: u64,
    pub inner_axis: usize,
}

#[derive(Clone, Debug)]
pub struct GpuKernel {
    pub wgsl: String,
    #[cfg(feature = "gpu")]
    pub module: naga::Module,
    pub buffers: Vec<KernelBuffer>,
    pub uniforms: Vec<usize>,
    pub axis_x: AxisSpec,
    pub axis_y: Option<AxisSpec>,
    pub strides: Vec<StrideCheck>,
    pub workgroup_size: [u32; 3],
    pub flag: Option<u32>,
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
        write!(
            f,
            "// grid: x in [{}, {})",
            self.axis_x.start, self.axis_x.end
        )?;
        if let Some(axis_y) = &self.axis_y {
            write!(f, ", y in [{}, {})", axis_y.start, axis_y.end)?;
        }
        writeln!(f, ", workgroup {:?}", self.workgroup_size)?;
        for buffer in &self.buffers {
            let access = if buffer.writable {
                "read_write"
            } else {
                "read"
            };
            writeln!(
                f,
                "// buffer arg{}: Float32Array, {access}",
                buffer.arg_index
            )?;
        }
        for index in &self.uniforms {
            writeln!(f, "// uniform arg{index}: int32")?;
        }
        if let Some(binding) = self.flag {
            writeln!(f, "// bails to the CPU through binding {binding}")?;
        }
        for check in &self.strides {
            let axis = ["x", "y"][check.inner_axis];
            writeln!(
                f,
                "// guard |{}| >= {} * extent.{axis}",
                check.stride, check.inner
            )?;
        }
        f.write_str(&self.wgsl)
    }
}

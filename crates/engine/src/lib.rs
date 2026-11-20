mod builtins;
mod continuation;
mod engine;
mod environment;
mod error;
mod heap;
mod host;
mod jit;
mod parse;
mod regex;
mod value;

pub use engine::{Backend, Engine, EngineOptions, OptimizationLevel};
pub use environment::Environment;
pub use error::Error;
pub use heap::OwnedValue;
pub use host::Host;
pub use jit::{CompileOptions, Report, ReportSink};
#[cfg(feature = "wgsl")]
pub use jit::{AxisSpec, BoundSpec, GpuKernel, KernelBuffer, Scale, StrideCheck};
pub use value::Value;

pub(crate) use rustc_hash::{FxHashMap, FxHashSet};

pub type Result<T> = std::result::Result<T, Error>;

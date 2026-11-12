mod continuation;
mod engine;
mod environment;
mod error;
mod heap;
mod host;
mod jit;
mod parse;
mod value;

pub use engine::{Backend, Engine, EngineOptions, OptLevel};
pub use environment::Environment;
pub use error::Error;
pub use heap::OwnedValue;
pub use host::Host;
pub use jit::CompileOptions;
pub use value::Value;

pub(crate) use rustc_hash::{FxHashMap, FxHashSet};

pub type Result<T> = std::result::Result<T, Error>;

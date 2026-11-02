mod engine;
mod environment;
mod error;
mod heap;
mod host;
mod jit;
mod parse;
mod value;

pub use engine::{Engine, EngineOptions, OptLevel};
pub use environment::Environment;
pub use error::Error;
pub use heap::OwnedValue;
pub use host::Host;
pub use value::Value;

pub type Result<T> = std::result::Result<T, Error>;

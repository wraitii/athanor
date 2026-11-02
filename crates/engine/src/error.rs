use std::fmt;

use oxc_span::Span;

#[derive(Debug, Clone)]
pub enum Error {
    Parse(String),
    Compile(String),
    Unsupported(String),
    Runtime(String),
}

pub(crate) fn unsupported_at(span: Span, message: impl fmt::Display) -> Error {
    Error::Unsupported(format!("{message} at {}..{}", span.start, span.end))
}

pub(crate) fn compile_at(span: Span, message: impl fmt::Display) -> Error {
    Error::Compile(format!("{message} at {}..{}", span.start, span.end))
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(message) => write!(f, "parse error: {message}"),
            Self::Compile(message) => write!(f, "compile error: {message}"),
            Self::Unsupported(message) => write!(f, "unsupported: {message}"),
            Self::Runtime(message) => write!(f, "runtime error: {message}"),
        }
    }
}

impl std::error::Error for Error {}

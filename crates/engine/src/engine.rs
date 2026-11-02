use std::collections::HashMap;

use oxc_allocator::Allocator;
use oxc_semantic::SemanticBuilder;

use crate::environment::Environment;
use crate::error::Error;
use crate::heap::{CodeId, OwnedValue};
use crate::jit::{self, Compiler};
use crate::parse;
use crate::value::Value;

struct Function {
    arity: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OptLevel {
    None,
    #[default]
    Speed,
}

impl OptLevel {
    pub(crate) fn as_flag(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Speed => "speed",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EngineOptions {
    pub opt_level: OptLevel,
}

pub struct Engine {
    compiler: Compiler,
    global: Environment,
    functions: HashMap<String, Function>,
    code: HashMap<u64, usize>,
}

impl Engine {
    pub fn new() -> Result<Self, Error> {
        Self::with_options(EngineOptions::default())
    }

    pub fn with_options(options: EngineOptions) -> Result<Self, Error> {
        let global = Environment::new();
        global.define("globalThis", Value::GLOBAL_OBJECT);
        Ok(Self {
            compiler: Compiler::new(options)?,
            global,
            functions: HashMap::new(),
            code: HashMap::new(),
        })
    }

    pub fn global(&self) -> &Environment {
        &self.global
    }

    pub fn compile(&mut self, source: &str, filename: &str) -> Result<(), Error> {
        let allocator = Allocator::default();
        let program = parse::parse(&allocator, source, filename)?;
        let analysis = SemanticBuilder::new_compiler()
            .with_build_nodes(true)
            .build(&program);
        if analysis.diagnostics.has_errors() {
            let message = analysis
                .diagnostics
                .errors()
                .map(|error| format!("{error:?}"))
                .collect::<Vec<_>>()
                .join("\n");
            return Err(Error::Parse(message));
        }
        let (functions, globals, initializer, code) = self.compiler.compile_program(
            &program,
            analysis.semantic.scoping(),
            analysis.semantic.nodes(),
        )?;
        self.code.extend(code);
        for function in functions {
            let value = self
                .global
                .heap()
                .closure(CodeId(function.code), &[])
                .ok_or_else(|| Error::Runtime("heap exhausted".into()))?;
            if !self.global.set(&function.name, value) {
                self.global.define(function.name.clone(), value);
            }
            self.global.heap().release(value);
            self.functions.insert(
                function.name,
                Function {
                    arity: function.arity,
                },
            );
        }
        for name in globals {
            self.global.define(name, Value::UNDEFINED);
        }
        if let Some(initializer) = initializer {
            unsafe {
                let value =
                    jit::call(initializer, &[], &self.global, &self.code, Value::UNDEFINED)?;
                self.global.heap().release(value);
            }
        }
        Ok(())
    }

    /// Calls a function with borrowed arguments and returns an owning handle.
    pub fn call(&self, name: &str, args: &[Value]) -> Result<OwnedValue, Error> {
        if !args.iter().all(|value| self.global.heap().valid(*value)) {
            return Err(Error::Runtime("invalid or foreign heap value".into()));
        }
        let callable = self
            .global
            .get(name)
            .ok_or_else(|| Error::Runtime(format!("unknown function {name}")))?;
        let entry = self
            .global
            .heap()
            .closure_code(callable.value())
            .and_then(|code| self.code.get(&code.0).copied())
            .ok_or_else(|| {
                Error::Runtime(format!("value `{name}` is not callable in this engine"))
            })?;
        let value = unsafe { jit::call(entry, args, &self.global, &self.code, callable.value())? };
        self.global.heap().drain(64);
        Ok(OwnedValue::from_owned(value, self.global.heap().clone()))
    }

    /// Processes at most `budget` disposal steps, counting edges and objects.
    pub fn drain_disposals(&self, budget: usize) {
        self.global.heap().drain(budget);
    }

    pub fn arity(&self, name: &str) -> Option<usize> {
        self.functions.get(name).map(|function| function.arity)
    }

    pub fn has_function(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }

    pub fn function_names(&self) -> impl Iterator<Item = &str> {
        self.functions.keys().map(String::as_str)
    }
}

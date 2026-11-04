use crate::host::Host;
use std::cell::{Cell, RefCell};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EngineOptions {
    pub opt_level: OptLevel,
    pub jitdump: bool,
    pub jitdump_ir: bool,
    pub ssa_opt: bool,
    pub ssa_dump: bool,
    pub compile_dump: bool,
    pub compile_ir: bool,
    pub compile_ast: bool,
    pub compile_semantic_ir: bool,
    pub compile_asm: bool,
}

impl Default for EngineOptions {
    fn default() -> Self {
        let jitdump = std::env::var("JS_JIT_JITDUMP").ok();
        let jitdump = jitdump.as_deref();
        Self {
            opt_level: OptLevel::default(),
            jitdump: matches!(jitdump, Some("1" | "2")),
            jitdump_ir: jitdump == Some("2"),
            ssa_opt: std::env::var("JS_JIT_SSA").as_deref() != Ok("0"),
            ssa_dump: std::env::var("JS_JIT_SSA_DUMP").as_deref() == Ok("1"),
            compile_dump: false,
            compile_ir: false,
            compile_ast: false,
            compile_semantic_ir: false,
            compile_asm: false,
        }
    }
}

pub struct Engine {
    compiler: RefCell<Compiler>,
    random_state: Cell<u64>,
    global: Environment,
    functions: RefCell<HashMap<String, Function>>,
    code: RefCell<HashMap<u64, usize>>,
    pub(crate) host: Option<Box<dyn Host>>,
}

impl Engine {
    pub fn new() -> Result<Self, Error> {
        Self::with_options(EngineOptions::default())
    }

    pub fn with_options(options: EngineOptions) -> Result<Self, Error> {
        let global = Environment::new();
        global.define("globalThis", Value::GLOBAL_OBJECT);
        global.define_readonly("Infinity", Value::number(f64::INFINITY));
        global.define_readonly("NaN", Value::number(f64::NAN));
        let constructor = global
            .heap()
            .native_function(crate::heap::NativeFunction::Error)
            .ok_or_else(|| Error::Runtime("heap exhausted".into()))?;
        global.define("Error", constructor);
        global.heap().release(constructor);
        let constructor = global
            .heap()
            .native_function(crate::heap::NativeFunction::Set)
            .ok_or_else(|| Error::Runtime("heap exhausted".into()))?;
        global.define("Set", constructor);
        global.heap().release(constructor);
        let constructor = global
            .heap()
            .native_function(crate::heap::NativeFunction::Array)
            .ok_or_else(|| Error::Runtime("heap exhausted".into()))?;
        global.define("Array", constructor);
        global.heap().release(constructor);
        let print = global
            .heap()
            .native_function(crate::heap::NativeFunction::Print)
            .ok_or_else(|| Error::Runtime("heap exhausted".into()))?;
        global.define("print", print);
        global.heap().release(print);
        let console = global
            .heap()
            .object()
            .ok_or_else(|| Error::Runtime("heap exhausted".into()))?;
        let log = global
            .heap()
            .native_function(crate::heap::NativeFunction::ConsoleLog)
            .ok_or_else(|| Error::Runtime("heap exhausted".into()))?;
        global.heap().define_property(console, "log", log, false);
        global.heap().release(log);
        global.define("console", console);
        global.heap().release(console);
        use crate::heap::{NativeFunction as Native, TypedKind};
        for (name, kind) in [
            ("Uint8Array", TypedKind::Uint8),
            ("Uint16Array", TypedKind::Uint16),
            ("Uint32Array", TypedKind::Uint32),
            ("Int8Array", TypedKind::Int8),
            ("Int16Array", TypedKind::Int16),
            ("Int32Array", TypedKind::Int32),
            ("Float32Array", TypedKind::Float32),
            ("Float64Array", TypedKind::Float64),
        ] {
            let value = global
                .heap()
                .native_function(Native::TypedConstructor(kind))
                .ok_or_else(|| Error::Runtime("heap exhausted".into()))?;
            global.define(name, value);
            global.heap().release(value);
        }
        let math = global
            .heap()
            .object()
            .ok_or_else(|| Error::Runtime("heap exhausted".into()))?;
        for (name, value) in [("PI", std::f64::consts::PI), ("E", std::f64::consts::E)] {
            global
                .heap()
                .define_property(math, name, Value::number(value), false);
        }
        use crate::heap::{MathFunction as M, NativeFunction};
        for (name, function) in [
            ("random", M::Random),
            ("abs", M::Abs),
            ("ceil", M::Ceil),
            ("floor", M::Floor),
            ("round", M::Round),
            ("trunc", M::Trunc),
            ("sqrt", M::Sqrt),
            ("sin", M::Sin),
            ("cos", M::Cos),
            ("tan", M::Tan),
            ("asin", M::Asin),
            ("acos", M::Acos),
            ("atan", M::Atan),
            ("atan2", M::Atan2),
            ("exp", M::Exp),
            ("log", M::Log),
            ("pow", M::Pow),
            ("min", M::Min),
            ("max", M::Max),
            ("imul", M::Imul),
            ("sign", M::Sign),
        ] {
            let value = global
                .heap()
                .native_function(NativeFunction::Math(function))
                .ok_or_else(|| Error::Runtime("heap exhausted".into()))?;
            global.heap().define_property(math, name, value, false);
            global.heap().release(value);
        }
        global.define("Math", math);
        global.heap().release(math);
        let json = global
            .heap()
            .object()
            .ok_or_else(|| Error::Runtime("heap exhausted".into()))?;
        for (name, function) in [
            ("parse", NativeFunction::JsonParse),
            ("stringify", NativeFunction::JsonStringify),
        ] {
            let value = global
                .heap()
                .native_function(function)
                .ok_or_else(|| Error::Runtime("heap exhausted".into()))?;
            global.heap().define_property(json, name, value, false);
            global.heap().release(value);
        }
        global.define("JSON", json);
        global.heap().release(json);
        let object = global
            .heap()
            .object()
            .ok_or_else(|| Error::Runtime("heap exhausted".into()))?;
        for (name, function) in [
            ("keys", NativeFunction::ObjectKeys),
            ("values", NativeFunction::ObjectValues),
            (
                "getOwnPropertyNames",
                NativeFunction::ObjectGetOwnPropertyNames,
            ),
            ("freeze", NativeFunction::ObjectFreeze),
            ("isFrozen", NativeFunction::ObjectIsFrozen),
        ] {
            let value = global
                .heap()
                .native_function(function)
                .ok_or_else(|| Error::Runtime("heap exhausted".into()))?;
            global.heap().define_property(object, name, value, false);
            global.heap().release(value);
        }
        global.define("Object", object);
        global.heap().release(object);
        let date = global
            .heap()
            .object()
            .ok_or_else(|| Error::Runtime("heap exhausted".into()))?;
        let now = global
            .heap()
            .native_function(NativeFunction::DateNow)
            .ok_or_else(|| Error::Runtime("heap exhausted".into()))?;
        global.heap().define_property(date, "now", now, false);
        global.heap().release(now);
        global.define("Date", date);
        global.heap().release(date);
        let seed_random = global
            .heap()
            .native_function(NativeFunction::HostSeedRandom)
            .ok_or_else(|| Error::Runtime("heap exhausted".into()))?;
        global.define("hostSeedRandom", seed_random);
        global.heap().release(seed_random);
        Ok(Self {
            compiler: RefCell::new(Compiler::new(options)?),
            random_state: Cell::new(0x330e),
            global,
            functions: RefCell::new(HashMap::new()),
            code: RefCell::new(HashMap::new()),
            host: None,
        })
    }

    pub(crate) fn seed_random(&self, seed: u32) {
        self.random_state.set(((seed as u64) << 16) | 0x330e);
    }

    pub(crate) fn random(&self) -> f64 {
        let state = self
            .random_state
            .get()
            .wrapping_mul(0x5deece66d)
            .wrapping_add(11)
            & ((1_u64 << 48) - 1);
        self.random_state.set(state);
        (state >> 17) as f64 / 2147483648.0
    }

    pub fn install_host(&mut self, host: impl Host + 'static) -> Result<(), Error> {
        use crate::heap::NativeFunction;
        for (name, function) in [
            ("hostReadFile", NativeFunction::HostReadFile),
            ("hostListFiles", NativeFunction::HostListFiles),
            ("hostEvaluateSource", NativeFunction::HostEvaluateSource),
        ] {
            let value = self
                .global
                .heap()
                .native_function(function)
                .ok_or_else(|| Error::Runtime("heap exhausted".into()))?;
            if !self.global.set(name, value) {
                self.global.define(name, value);
            }
            self.global.heap().release(value);
        }
        self.host = Some(Box::new(host));
        Ok(())
    }

    pub(crate) fn entry(&self, code: u64) -> Option<usize> {
        self.code.borrow().get(&code).copied()
    }

    pub fn global(&self) -> &Environment {
        &self.global
    }

    pub fn string(&self, text: &str) -> Result<OwnedValue, Error> {
        let value = self
            .global
            .heap()
            .string(text)
            .ok_or_else(|| Error::Runtime("heap exhausted".into()))?;
        Ok(OwnedValue::from_owned(value, self.global.heap().clone()))
    }

    pub fn compile(&mut self, source: &str, filename: &str) -> Result<(), Error> {
        if let Some(initializer) = self.prepare_source(source, filename)? {
            let value = unsafe {
                jit::call(
                    initializer,
                    &[],
                    self,
                    Value::UNDEFINED,
                    Value::GLOBAL_OBJECT,
                )?
            };
            self.global.heap().release(value);
        }
        Ok(())
    }

    pub fn compile_only(&self, source: &str, filename: &str) -> Result<(), Error> {
        self.prepare_source(source, filename).map(|_| ())
    }

    pub(crate) fn prepare_source(
        &self,
        source: &str,
        filename: &str,
    ) -> Result<Option<usize>, Error> {
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
        let mut lexical = Vec::new();
        for statement in &program.body {
            match statement {
                oxc_ast::ast::Statement::VariableDeclaration(declaration) => {
                    for declarator in &declaration.declarations {
                        if let oxc_ast::ast::BindingPattern::BindingIdentifier(identifier) =
                            &declarator.id
                        {
                            let name = identifier.name.as_str();
                            if declaration.kind == oxc_ast::ast::VariableDeclarationKind::Var {
                                if self.global.has_lexical(name) {
                                    return Err(Error::Compile(format!(
                                        "{filename}: global binding `{name}` already declared"
                                    )));
                                }
                            } else {
                                if self.global.get_raw(name).is_some() {
                                    return Err(Error::Compile(format!(
                                        "{filename}: global binding `{name}` already declared"
                                    )));
                                }
                                lexical.push((
                                    name.to_string(),
                                    declaration.kind == oxc_ast::ast::VariableDeclarationKind::Let,
                                ));
                            }
                        }
                    }
                }
                oxc_ast::ast::Statement::FunctionDeclaration(function) => {
                    if let Some(identifier) = &function.id
                        && self.global.has_lexical(identifier.name.as_str())
                    {
                        return Err(Error::Compile(format!(
                            "{filename}: global binding `{}` already declared",
                            identifier.name
                        )));
                    }
                }
                oxc_ast::ast::Statement::ClassDeclaration(class) => {
                    if let Some(identifier) = &class.id
                        && self.global.has_lexical(identifier.name.as_str())
                    {
                        return Err(Error::Compile(format!(
                            "{filename}: global binding `{}` already declared",
                            identifier.name
                        )));
                    }
                }
                _ => {}
            }
        }
        let (functions, globals, initializer, code) = self
            .compiler
            .borrow_mut()
            .compile_program(
                &program,
                analysis.semantic.scoping(),
                analysis.semantic.nodes(),
                source,
                filename,
            )
            .map_err(|error| match error {
                Error::Unsupported(message) => Error::Unsupported(format!("{filename}: {message}")),
                Error::Compile(message) => Error::Compile(format!("{filename}: {message}")),
                other => other,
            })?;
        for (name, mutable) in lexical {
            self.global.define_lexical(name, mutable);
        }
        self.code.borrow_mut().extend(code);
        for function in functions {
            let value = self
                .global
                .heap()
                .closure(CodeId(function.code), &[])
                .ok_or_else(|| Error::Runtime("heap exhausted".into()))?;
            if function.generator {
                self.global.heap().disable_constructor(value);
            }
            if !self.global.set(&function.name, value) {
                self.global.define(function.name.clone(), value);
            }
            self.global.heap().release(value);
            self.functions.borrow_mut().insert(
                function.name,
                Function {
                    arity: function.arity,
                },
            );
        }
        for name in globals {
            self.global.define(name, Value::UNDEFINED);
        }
        Ok(initializer)
    }

    /// Calls a function with borrowed arguments and returns an owning handle.
    /// Heap arguments must be live tokens from this engine.
    pub fn call(&self, name: &str, args: &[Value]) -> Result<OwnedValue, Error> {
        if !args.iter().all(|value| self.global.heap().owns(*value)) {
            return Err(Error::Runtime("invalid heap value".into()));
        }
        let callable = self
            .global
            .get(name)
            .ok_or_else(|| Error::Runtime(format!("unknown function {name}")))?;
        if self.global.heap().is_class(callable.value()) {
            return Err(Error::Runtime("class constructor requires new".into()));
        }
        let entry = self
            .global
            .heap()
            .closure_code(callable.value())
            .and_then(|code| self.entry(code.0))
            .ok_or_else(|| {
                Error::Runtime(format!("value `{name}` is not callable in this engine"))
            })?;
        let receiver = self
            .global
            .heap()
            .lexical_this(callable.value())
            .unwrap_or(Value::UNDEFINED);
        let value = unsafe { jit::call(entry, args, self, callable.value(), receiver)? };
        self.global.heap().drain(64);
        Ok(OwnedValue::from_owned(value, self.global.heap().clone()))
    }

    /// Processes at most `budget` disposal steps, counting edges and objects.
    pub fn drain_disposals(&self, budget: usize) {
        self.global.heap().drain(budget);
    }

    pub fn arity(&self, name: &str) -> Option<usize> {
        self.functions
            .borrow()
            .get(name)
            .map(|function| function.arity)
    }

    pub fn has_function(&self, name: &str) -> bool {
        self.functions.borrow().contains_key(name)
    }

    pub fn function_names(&self) -> impl Iterator<Item = String> {
        self.functions
            .borrow()
            .keys()
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
    }
}

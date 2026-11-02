use crate::host::Host;
use std::cell::RefCell;
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
    compiler: RefCell<Compiler>,
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
        Ok(Self {
            compiler: RefCell::new(Compiler::new(options)?),
            global,
            functions: RefCell::new(HashMap::new()),
            code: RefCell::new(HashMap::new()),
            host: None,
        })
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
    pub fn call(&self, name: &str, args: &[Value]) -> Result<OwnedValue, Error> {
        if !args.iter().all(|value| self.global.heap().valid(*value)) {
            return Err(Error::Runtime("invalid or foreign heap value".into()));
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

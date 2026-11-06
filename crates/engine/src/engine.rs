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
    pub get_refinement: bool,
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
            get_refinement: std::env::var("JS_JIT_GET_REFINEMENT").as_deref() != Ok("0"),
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
    planner: RefCell<jit::Planner>,
    random_state: Cell<u64>,
    global: Environment,
    functions: RefCell<HashMap<String, Function>>,
    // Code IDs are function indices in this engine's persistent Cranelift module.
    code: RefCell<Vec<Option<jit::entries::Entry>>>,
    direct: jit::DirectEntries,
    pub(crate) locals: jit::LocalSlots,
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
        let direct = jit::DirectEntries::new();
        Ok(Self {
            compiler: RefCell::new(Compiler::new(options, direct.address())?),
            planner: RefCell::new(jit::Planner::default()),
            random_state: Cell::new(0x330e),
            global,
            functions: RefCell::new(HashMap::new()),
            code: RefCell::new(Vec::new()),
            direct,
            locals: jit::LocalSlots::new(),
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

    pub(crate) fn observe_call(&self, cache: &mut jit::planner::ObservationCache, target: u64) {
        self.planner
            .borrow_mut()
            .observe(cache.owner, cache.site, target);
        self.direct.withdraw(cache.owner);
        cache.recorded = 1;
    }

    pub(crate) fn observe_get(
        &self,
        cache: &mut jit::planner::ObservationCache,
        observed: jit::planner::ReadType,
        kind: jit::planner::ReadKind,
    ) {
        self.planner
            .borrow_mut()
            .observe_get(cache.owner, cache.site, kind, observed);
        self.direct.withdraw(cache.owner);
        cache.recorded = 1;
    }

    pub(crate) fn entry(&self, code: u64) -> Option<jit::entries::Entry> {
        let recompile = self.planner.borrow_mut().take(code);
        if recompile {
            let result = self
                .compiler
                .borrow_mut()
                .recompile(code, &self.planner.borrow());
            match result {
                Ok(Some(entry)) => self.code.borrow_mut()[code as usize] = Some(entry),
                Ok(None) => {}
                Err(error) => eprintln!("js jit: recompilation failed: {error}"),
            }
        }
        let entry = self
            .code
            .borrow()
            .get(usize::try_from(code).ok()?)
            .copied()
            .flatten()?;
        self.publish(code, entry);
        Some(entry)
    }

    fn publish(&self, code: u64, entry: jit::entries::Entry) {
        match entry.adapter {
            Some(_) => self.direct.publish(code, entry),
            None => self.direct.withdraw(code),
        }
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
                    jit::entries::Entry::generic(initializer),
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
        {
            let mut entries = self.code.borrow_mut();
            for (code, entry) in code {
                let index = usize::try_from(code).unwrap();
                if entries.len() <= index {
                    entries.resize(index + 1, None);
                }
                entries[index] = Some(entry);
                self.publish(code, entry);
            }
        }
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
    /// Heap arguments must satisfy the lifetime and runtime requirements of [`Value`].
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

#[cfg(test)]
mod feedback_tests {
    use super::*;

    fn code(engine: &Engine, name: &str) -> u64 {
        engine
            .global
            .heap()
            .closure_code(engine.global.get_raw(name).unwrap())
            .unwrap()
            .0
    }

    fn entry(engine: &Engine, code: u64) -> jit::entries::Entry {
        engine.code.borrow()[code as usize].unwrap()
    }

    #[test]
    fn method_feedback_reemits_once_and_guards_replacements() {
        let mut engine = Engine::new().unwrap();
        engine
            .compile("Math.square=function(x){return x*x;};", "math.js")
            .unwrap();
        engine
            .compile(
                r#"
            function distance(x, y) { return Math.square(x) + Math.square(y); }
            function replace() { Math.square=function(x){return x+10;}; }
        "#,
                "distance.js",
            )
            .unwrap();
        let code = code(&engine, "distance");
        let original = entry(&engine, code);
        assert_eq!(
            engine
                .call("distance", &[Value::number(3.0), Value::number(4.0)])
                .unwrap()
                .value(),
            Value::number(25.0)
        );
        assert_eq!(entry(&engine, code), original);
        assert_eq!(
            engine
                .call("distance", &[Value::number(3.0), Value::number(4.0)])
                .unwrap()
                .value(),
            Value::number(25.0)
        );
        let optimized = entry(&engine, code);
        assert_ne!(original, optimized);
        engine.call("replace", &[]).unwrap();
        for _ in 0..3 {
            assert_eq!(
                engine
                    .call("distance", &[Value::number(3.0), Value::number(4.0)])
                    .unwrap()
                    .value(),
                Value::number(27.0)
            );
            assert_eq!(entry(&engine, code), optimized);
        }
    }

    #[test]
    fn feedback_keeps_captures_receivers_and_heap_returns() {
        let mut engine = Engine::new().unwrap();
        engine
            .compile(
                r#"
            function make(n) { return function(x) { return {value:this.base+n+x}; }; }
            var first={base:10, run:make(2)};
            var second={base:20, run:make(3)};
            function invoke(o) { return o.run(4).value; }
        "#,
                "captures.js",
            )
            .unwrap();
        let code = code(&engine, "invoke");
        let original = entry(&engine, code);
        for _ in 0..4 {
            for (name, expected) in [("first", 16.0), ("second", 27.0)] {
                let receiver = engine.global.get_raw(name).unwrap();
                assert_eq!(
                    engine.call("invoke", &[receiver]).unwrap().value(),
                    Value::number(expected)
                );
                engine.drain_disposals(1000);
            }
        }
        assert_ne!(entry(&engine, code), original);
    }

    #[test]
    fn previous_entry_fallback_preserves_arguments_heap_returns_and_errors() {
        for invoke in [
            "function invoke(x){return object.run(x);}",
            "function invoke(){return object.run(arguments[0]);}",
        ] {
            let mut engine = Engine::new().unwrap();
            engine.compile(&format!(
                "var object={{run:function(x){{return x;}}}}; {invoke} function replace(){{object.run=function(x){{return {{value:x}};}};}} function fail(){{object.run=function(x){{throw x;}};}} function restore(){{object.run=function(x){{return x;}};}}"
            ), "previous-entry.js").unwrap();
            let code = code(&engine, "invoke");
            let original = entry(&engine, code);
            let input = engine.global.heap().object().unwrap();
            for _ in 0..3 {
                assert_eq!(engine.call("invoke", &[input]).unwrap().value(), input);
            }
            assert_ne!(entry(&engine, code), original);
            engine.call("replace", &[]).unwrap();
            for _ in 0..3 {
                let result = engine.call("invoke", &[input]).unwrap();
                engine.drain_disposals(1000);
                assert_eq!(engine.global.heap().property(result.value(), "value"), Some(input));
            }
            engine.call("fail", &[]).unwrap();
            assert!(engine.call("invoke", &[input]).is_err());
            engine.call("restore", &[]).unwrap();
            assert_eq!(engine.call("invoke", &[input]).unwrap().value(), input);
            engine.global.heap().release(input);
        }
    }

    #[test]
    fn feedback_preserves_foreign_source_errors_and_recovery() {
        let mut engine = Engine::new().unwrap();
        engine
            .compile(
                "var object={run:function(x){return x.value;}};",
                "callee.js",
            )
            .unwrap();
        engine
            .compile("function invoke(x){return object.run(x);}", "caller.js")
            .unwrap();
        let object = engine.global.heap().object().unwrap();
        engine
            .global
            .heap()
            .set_property(object, "value", Value::number(7.0));
        assert_eq!(
            engine.call("invoke", &[object]).unwrap().value(),
            Value::number(7.0)
        );
        let error = engine
            .call("invoke", &[Value::NULL])
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("callee.js:"), "{error}");
        assert_eq!(
            engine.call("invoke", &[object]).unwrap().value(),
            Value::number(7.0)
        );
        engine.global.heap().release(object);
    }

    #[test]
    fn constructor_body_feedback_survives_prototype_replacement() {
        let mut engine = Engine::new().unwrap();
        engine
            .compile(
                r#"
            function C(x){this.set(x);}
            C.prototype.set=function(x){this.value=x;};
            function make(K,x){return new K(x).value;}
            function replace(){C.prototype.set=function(x){this.value=x+10;};}
        "#,
                "constructors.js",
            )
            .unwrap();
        let code = code(&engine, "make");
        let original = entry(&engine, code);
        let constructor = engine.global.get("C").unwrap();
        let args = [constructor.value(), Value::number(3.0)];
        for _ in 0..2 {
            assert_eq!(
                engine.call("make", &args).unwrap().value(),
                Value::number(3.0)
            );
        }
        assert_ne!(entry(&engine, code), original);
        engine.call("replace", &[]).unwrap();
        assert_eq!(
            engine.call("make", &args).unwrap().value(),
            Value::number(13.0)
        );
    }

    #[test]
    fn recursive_dispatch_can_replace_code_with_old_frames_active() {
        let mut engine = Engine::new().unwrap();
        engine
            .compile(
                r#"
            var object={run:function(x){return x+1;}};
            function recurse(n){let x=object.run(n);if(n>0)return x+recurse(n-1);return x;}
        "#,
                "recursive.js",
            )
            .unwrap();
        let code = code(&engine, "recurse");
        let original = entry(&engine, code);
        assert_eq!(
            engine
                .call("recurse", &[Value::number(5.0)])
                .unwrap()
                .value(),
            Value::number(21.0)
        );
        assert_ne!(entry(&engine, code), original);
        assert_eq!(
            engine
                .call("recurse", &[Value::number(5.0)])
                .unwrap()
                .value(),
            Value::number(21.0)
        );
    }

    #[test]
    fn ineligible_feedback_keeps_the_existing_entry() {
        let mut engine = Engine::new().unwrap();
        engine.compile(
            "var object={run:function(x){return arguments[0];}};function invoke(x){return object.run(x);}",
            "ineligible.js",
        ).unwrap();
        let code = code(&engine, "invoke");
        let original = entry(&engine, code);
        for _ in 0..3 {
            assert_eq!(
                engine
                    .call("invoke", &[Value::number(7.0)])
                    .unwrap()
                    .value(),
                Value::number(7.0)
            );
            assert_eq!(entry(&engine, code), original);
        }
    }

    #[test]
    fn entry_guards_specialize_nested_reads_and_guard_replacements() {
        let mut engine = Engine::new().unwrap();
        engine.compile(
            "var matrix=[Float64Array.from([1.5,2.5])]; function read(i,j){return matrix[i][j];} function replace(){matrix[0]=[7,8];}",
            "get_feedback.js",
        ).unwrap();
        let code = code(&engine, "read");
        let original = entry(&engine, code);
        let args = [Value::number(0.0), Value::number(1.0)];
        assert_eq!(
            engine.call("read", &args).unwrap().value(),
            Value::number(2.5)
        );
        assert_eq!(entry(&engine, code), original);
        assert_eq!(
            engine.call("read", &args).unwrap().value(),
            Value::number(2.5)
        );
        let optimized = entry(&engine, code);
        assert_eq!(optimized, original);
        assert_eq!(
            engine
                .call("read", &[Value::number(0.0), Value::number(-1.0)])
                .unwrap()
                .value(),
            Value::UNDEFINED
        );
        for index in [0.5, f64::NAN, f64::INFINITY, 2.0] {
            assert_eq!(
                engine
                    .call("read", &[Value::number(0.0), Value::number(index)])
                    .unwrap()
                    .value(),
                Value::UNDEFINED
            );
        }
        let key = engine.string("0").unwrap();
        assert_eq!(
            engine
                .call("read", &[Value::number(0.0), key.value()])
                .unwrap()
                .value(),
            Value::number(1.5)
        );
        engine.call("replace", &[]).unwrap();
        assert_eq!(
            engine.call("read", &args).unwrap().value(),
            Value::number(8.0)
        );
    }

    #[test]
    fn get_feedback_specializes_arrays_and_preserves_heap_results() {
        let mut engine = Engine::new().unwrap();
        engine.compile(
            "var holder={values:new Array(3)};holder.values[0]={value:3};holder.values[2]=9; function read(i){return holder.values[i];} function replace(){holder.values={0:7};}",
            "array_feedback.js",
        ).unwrap();
        let code = code(&engine, "read");
        let original = entry(&engine, code);
        for _ in 0..2 {
            assert_eq!(
                engine.call("read", &[Value::number(2.0)]).unwrap().value(),
                Value::number(9.0)
            );
        }
        assert_ne!(entry(&engine, code), original);
        for index in [1.0, 3.0] {
            assert_eq!(
                engine
                    .call("read", &[Value::number(index)])
                    .unwrap()
                    .value(),
                Value::UNDEFINED
            );
        }
        for index in [-1.0, 0.5, f64::NAN, f64::INFINITY] {
            assert!(engine.call("read", &[Value::number(index)]).is_err());
        }
        let key = engine.string("2").unwrap();
        assert_eq!(
            engine.call("read", &[key.value()]).unwrap().value(),
            Value::number(9.0)
        );
        let retained = engine.call("read", &[Value::number(0.0)]).unwrap();
        engine.call("replace", &[]).unwrap();
        engine.drain_disposals(1000);
        assert_eq!(
            engine.global.heap().property(retained.value(), "value"),
            Some(Value::number(3.0))
        );
        assert_eq!(
            engine.call("read", &[Value::number(0.0)]).unwrap().value(),
            Value::number(7.0)
        );
    }

    #[test]
    fn get_feedback_specializes_typed_rows_inside_loops() {
        for ssa_opt in [false, true] {
            let mut engine = Engine::with_options(EngineOptions {
                ssa_opt,
                ..Default::default()
            })
            .unwrap();
            engine.compile(
                "var height=[Float64Array.from([1.5,2.5]),Float64Array.from([3.5,4.5])]; function sum(){let result=0;for(let x=0;x<height.length;++x)for(let y=0;y<height[x].length;++y)result+=height[x][y];return result;} function replace(){height[1]=[5,6];}",
                "row_feedback.js",
            ).unwrap();
            let code = code(&engine, "sum");
            let original = entry(&engine, code);
            for _ in 0..3 {
                assert_eq!(
                    engine.call("sum", &[]).unwrap().value(),
                    Value::number(12.0)
                );
            }
            assert_eq!(entry(&engine, code) != original, ssa_opt);
            engine.call("replace", &[]).unwrap();
            assert_eq!(
                engine.call("sum", &[]).unwrap().value(),
                Value::number(15.0)
            );
        }
    }

    #[test]
    fn entry_guards_specialize_named_typed_array_properties() {
        let mut engine = Engine::new().unwrap();
        engine.compile(
            "var holder={values:Int32Array.from([3,4])}; function read(i){return holder.values[i];} function replace(){holder.values=Float64Array.from([5.5,6.5]);}",
            "property_feedback.js",
        ).unwrap();
        let code = code(&engine, "read");
        let original = entry(&engine, code);
        let args = [Value::number(1.0)];
        for _ in 0..3 {
            assert_eq!(
                engine.call("read", &args).unwrap().value(),
                Value::number(4.0)
            );
        }
        let optimized = entry(&engine, code);
        assert_eq!(optimized, original);
        engine.call("replace", &[]).unwrap();
        assert_eq!(
            engine.call("read", &args).unwrap().value(),
            Value::number(6.5)
        );
        assert_eq!(entry(&engine, code), optimized);
    }

    #[test]
    fn disabling_get_refinement_keeps_call_feedback() {
        let mut engine = Engine::with_options(EngineOptions {
            get_refinement: false,
            ..Default::default()
        })
        .unwrap();
        engine.compile(
            "var holder={values:[1,2],run:function(){return 7;}}; function read(i){return holder.values[i];} function invoke(){return holder.run();}",
            "disabled_get_feedback.js",
        ).unwrap();
        let read_code = code(&engine, "read");
        let original = entry(&engine, read_code);
        for _ in 0..3 {
            assert_eq!(
                engine.call("read", &[Value::number(1.0)]).unwrap().value(),
                Value::number(2.0)
            );
            assert_eq!(entry(&engine, read_code), original);
            assert!(!engine.planner.borrow_mut().take(read_code));
        }
        let invoke_code = code(&engine, "invoke");
        let original = entry(&engine, invoke_code);
        for _ in 0..2 {
            assert_eq!(
                engine.call("invoke", &[]).unwrap().value(),
                Value::number(7.0)
            );
        }
        assert_ne!(entry(&engine, invoke_code), original);
    }

    #[test]
    fn feedback_is_disabled_with_semantic_optimization() {
        let mut engine = Engine::with_options(EngineOptions {
            ssa_opt: false,
            ..Default::default()
        })
        .unwrap();
        engine
            .compile(
                "var o={f:function(x){return x*x;}};function run(){return o.f(3);}",
                "disabled.js",
            )
            .unwrap();
        let code = code(&engine, "run");
        let original = entry(&engine, code);
        for _ in 0..3 {
            assert_eq!(engine.call("run", &[]).unwrap().value(), Value::number(9.0));
        }
        assert_eq!(entry(&engine, code), original);
        assert!(!engine.planner.borrow_mut().take(code));
    }
}

use crate::FxHashMap;
use crate::host::Host;
use std::cell::{Cell, RefCell};
use std::sync::Arc;

use oxc_allocator::Allocator;
use oxc_semantic::SemanticBuilder;

use crate::environment::Environment;
use crate::error::Error;
use crate::heap::{CodeId, OwnedValue};
use crate::jit;
use crate::parse;
use crate::value::Value;

struct Function {
    arity: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Backend {
    Cranelift,
    Native,
    #[default]
    NativeOnly,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OptimizationLevel {
    None,
    Baseline,
    Initial,
    #[default]
    Full,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineOptions {
    pub optimization: OptimizationLevel,
    pub jitdump: bool,
    pub jitdump_ir: bool,
    pub ssa_opt: bool,
    pub get_refinement: bool,
    pub background_compile: bool,
    pub compiler_threads: usize,
    pub inlining: bool,
    pub call_variants: bool,
    pub call_specialization: bool,
    pub recompile_inline_multiplier: usize,
    pub osr: bool,
    pub continuations: bool,
    pub generic_first: bool,
    pub recompile_log: bool,
    pub specialization_log: Option<String>,
    pub call_profile: bool,
    pub backend: Backend,
    pub compile: Option<Arc<crate::CompileOptions>>,
}

impl Default for EngineOptions {
    fn default() -> Self {
        let jitdump = std::env::var("JS_JIT_JITDUMP").ok();
        let jitdump = jitdump.as_deref();
        Self {
            optimization: match std::env::var("JS_JIT_OPT").as_deref() {
                Ok("none") => OptimizationLevel::None,
                Ok("baseline") => OptimizationLevel::Baseline,
                Ok("initial") => OptimizationLevel::Initial,
                _ => OptimizationLevel::Full,
            },
            jitdump: matches!(jitdump, Some("1" | "2")),
            jitdump_ir: jitdump == Some("2"),
            ssa_opt: true,
            get_refinement: true,
            background_compile: std::env::var("JS_JIT_BACKGROUND").as_deref() != Ok("0"),
            compiler_threads: std::env::var("JS_JIT_WORKERS")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(jit::recompile::DEFAULT_COMPILER_THREADS),
            inlining: std::env::var("JS_JIT_INLINE").as_deref() != Ok("0"),
            call_variants: std::env::var("JS_JIT_CALL_VARIANTS").as_deref() != Ok("0"),
            call_specialization: std::env::var("JS_JIT_CALL_SPECIALIZATION").as_deref() != Ok("0"),
            recompile_inline_multiplier: std::env::var("JS_JIT_RECOMPILE_INLINE_MULTIPLIER")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(jit::DEFAULT_RECOMPILE_INLINE_MULTIPLIER),
            osr: std::env::var("JS_JIT_OSR").as_deref() != Ok("0"),
            continuations: true,
            generic_first: true,
            recompile_log: std::env::var("JS_JIT_RECOMPILE_LOG").as_deref() == Ok("1"),
            specialization_log: std::env::var("JS_JIT_SPECIALIZATION_LOG")
                .ok()
                .filter(|filter| !filter.is_empty() && filter != "0"),
            call_profile: false,
            backend: match std::env::var("JS_JIT_BACKEND").as_deref() {
                Ok("cranelift") => Backend::Cranelift,
                Ok("native") => Backend::Native,
                Ok("native-only") => Backend::NativeOnly,
                _ => Backend::default(),
            },
            compile: None,
        }
    }
}

pub struct Engine {
    pub(crate) tiering: jit::Tiering,
    random_state: Cell<u64>,
    global: Environment,
    functions: RefCell<FxHashMap<String, Function>>,
    pub(crate) locals: jit::LocalSlots,
    pub(crate) host: Option<Box<dyn Host>>,
}

impl Drop for Engine {
    fn drop(&mut self) {
        if std::env::var_os("JS_JIT_HEAP_CYCLES").is_some_and(|value| value == "1") {
            let start = std::time::Instant::now();
            let stats = self.global.heap().cycle_stats();
            eprintln!(
                "heap cycles: arena_bytes={}, scanned_slots={}, live={}, unreachable={}, cyclic_components={}, cyclic_nodes={}, unreachable_components={}, unreachable_cyclic_nodes={}, cyclic_kinds={:?}, unreachable_kinds={:?}; snapshot={:?}",
                stats.arena_bytes,
                stats.scanned_slots,
                stats.live,
                stats.unreachable,
                stats.cyclic_components,
                stats.cyclic_nodes,
                stats.unreachable_components,
                stats.unreachable_cyclic_nodes,
                stats.cyclic_kinds,
                stats.unreachable_kinds,
                start.elapsed()
            );
        }
        self.tiering.log_prefetch_uses();
        crate::jit::bailouts::report();
        self.tiering.report();
    }
}

impl Engine {
    pub fn new() -> Result<Self, Error> {
        Self::with_options(EngineOptions::default())
    }

    pub fn with_options(options: EngineOptions) -> Result<Self, Error> {
        if options.compiler_threads == 0 {
            return Err(Error::Compile("compiler_threads must be positive".into()));
        }
        let global = Environment::new();
        crate::builtins::install(&global)?;
        Ok(Self {
            tiering: jit::Tiering::new(options)?,
            random_state: Cell::new(0x330e),
            global,
            functions: RefCell::new(FxHashMap::default()),
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
        let span = tracing::info_span!("Script compilation", filename);
        let _entered = span.enter();
        let started = std::env::var_os("JS_JIT_STATS").map(|_| std::time::Instant::now());
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
        let (functions, globals, initializer) = self
            .tiering
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
        self.tiering.queue_unlazification();
        if let Some(started) = started {
            eprintln!(
                "js jit script compile: {:.3} ms: {filename}",
                started.elapsed().as_secs_f64() * 1000.0
            );
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
            .map(|code| self.tiering.entry(code.0))
            .transpose()?
            .flatten()
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

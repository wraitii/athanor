use crate::host::Host;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{Duration, Instant};

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineOptions {
    pub opt_level: OptLevel,
    pub jitdump: bool,
    pub jitdump_ir: bool,
    pub ssa_opt: bool,
    pub get_refinement: bool,
    pub background_compile: bool,
    pub compiler_threads: usize,
    pub recompile_inline_multiplier: usize,
    pub osr: bool,
    pub recompile_log: bool,
    pub call_profile: bool,
    pub ssa_dump: bool,
    pub compile: Option<Arc<crate::CompileOptions>>,
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
            background_compile: std::env::var("JS_JIT_BACKGROUND").as_deref() != Ok("0"),
            compiler_threads: std::env::var("JS_JIT_WORKERS")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(jit::recompile::DEFAULT_COMPILER_THREADS),
            recompile_inline_multiplier: std::env::var("JS_JIT_RECOMPILE_INLINE_MULTIPLIER")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(jit::DEFAULT_RECOMPILE_INLINE_MULTIPLIER),
            osr: std::env::var("JS_JIT_OSR").as_deref() == Ok("1"),
            recompile_log: std::env::var("JS_JIT_RECOMPILE_LOG").as_deref() == Ok("1"),
            call_profile: std::env::var("JS_JIT_CALL_PROFILE").as_deref() == Ok("1"),
            ssa_dump: std::env::var("JS_JIT_SSA_DUMP").as_deref() == Ok("1"),
            compile: None,
        }
    }
}

#[derive(Default)]
struct PrefetchStats {
    unused: HashSet<u64>,
    installed: usize,
    used: usize,
    discarded: usize,
}

#[derive(Default)]
struct LazyWaitStats {
    count: usize,
    total: Duration,
    longest: Duration,
}

pub struct Engine {
    // Join the worker before releasing resources referenced by its compilation snapshots.
    worker: RefCell<Option<jit::recompile::Service>>,
    unlazify_seen: RefCell<HashSet<u64>>,
    compilation_errors: RefCell<HashMap<u64, Error>>,
    lazy_wait_stats: RefCell<LazyWaitStats>,
    compilation_state: Arc<AtomicU8>,
    options: EngineOptions,
    recompile_log: jit::recompile::Log,
    prefetch_stats: RefCell<PrefetchStats>,
    compiler: RefCell<Compiler>,
    planner: RefCell<jit::Planner>,
    random_state: Cell<u64>,
    global: Environment,
    functions: RefCell<HashMap<String, Function>>,
    // Code IDs are function indices in this engine's persistent Cranelift module.
    code: RefCell<Vec<Option<jit::entries::Entry>>>,
    direct: jit::DirectEntries,
    overflow_calls: RefCell<HashMap<u64, usize>>,
    pub(crate) locals: jit::LocalSlots,
    pub(crate) host: Option<Box<dyn Host>>,
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.log_prefetch_uses();
        if self.recompile_log.enabled() {
            let stats = self.prefetch_stats.borrow();
            self.recompile_log.event(format_args!(
                "prefetch summary installed={} used={} unused={} discarded={}",
                stats.installed,
                stats.used,
                stats.unused.len(),
                stats.discarded
            ));
        }
        if self.recompile_log.enabled() {
            let stats = self.lazy_wait_stats.borrow();
            self.recompile_log.event(format_args!(
                "lazy wait summary count={} total_ms={:.3} max_ms={:.3}",
                stats.count,
                stats.total.as_secs_f64() * 1000.0,
                stats.longest.as_secs_f64() * 1000.0,
            ));
        }
        if self.options.call_profile {
            eprint!(
                "{}",
                self.compiler.borrow().call_profile(&self.planner.borrow())
            );
        }
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
        let compilation_state = Arc::new(AtomicU8::new(0));
        let recompile_log = jit::recompile::Log::new(options.recompile_log);
        let state_address = if options.background_compile {
            Arc::as_ptr(&compilation_state) as usize
        } else {
            0
        };
        Ok(Self {
            worker: RefCell::new(None),
            unlazify_seen: RefCell::new(HashSet::new()),
            compilation_errors: RefCell::new(HashMap::new()),
            lazy_wait_stats: RefCell::new(LazyWaitStats::default()),
            compilation_state,
            options: options.clone(),
            recompile_log,
            prefetch_stats: RefCell::new(PrefetchStats::default()),
            compiler: RefCell::new(Compiler::new(
                options,
                jit::CompilationTier::Optimized,
                direct.address(),
                state_address,
                recompile_log,
            )?),
            planner: RefCell::new(jit::Planner::default()),
            random_state: Cell::new(0x330e),
            global,
            functions: RefCell::new(HashMap::new()),
            code: RefCell::new(Vec::new()),
            direct,
            overflow_calls: RefCell::new(HashMap::new()),
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

    pub(crate) fn record_calls<'s>(
        &self,
        samples: impl Iterator<Item = (&'s jit::planner::CallSite, u64)>,
    ) {
        let mut planner = self.planner.borrow_mut();
        let mut run: Option<(&jit::planner::CallSite, u64, u64)> = None;
        for (site, code) in samples {
            match &mut run {
                Some((last, last_code, count))
                    if std::ptr::eq(*last, site) && *last_code == code =>
                {
                    *count += 1;
                }
                _ => {
                    if let Some((last, last_code, count)) = run.replace((site, code, 1)) {
                        planner.record_call(last, last_code, count);
                    }
                }
            }
        }
        if let Some((site, code, count)) = run {
            planner.record_call(site, code, count);
        }
    }

    pub(crate) fn observe_call(&self, cache: &mut jit::planner::ObservationCache, target: u64) {
        let observed = self
            .planner
            .borrow_mut()
            .observe(cache.owner, cache.site, target);
        if observed {
            self.log_recompilation(
                cache.owner,
                format_args!("call feedback site={} target={target}", cache.site),
            );
            self.request_compilation(cache.owner);
        }
        cache.recorded = 1;
    }

    pub(crate) fn observe_native(
        &self,
        cache: &mut jit::planner::ObservationCache,
        target: crate::heap::NativeFunction,
    ) {
        let observed = self
            .planner
            .borrow_mut()
            .observe_native(cache.owner, cache.site, target);
        if observed {
            self.request_compilation(cache.owner);
        }
        cache.recorded = 1;
    }

    pub(crate) fn observe_get(
        &self,
        cache: &mut jit::planner::ObservationCache,
        observed: jit::planner::ReadType,
        kind: jit::planner::ReadKind,
    ) {
        let recorded =
            self.planner
                .borrow_mut()
                .observe_get(cache.owner, cache.site, kind, observed);
        if recorded && observed != jit::planner::ReadType::Unknown {
            self.log_recompilation(
                cache.owner,
                format_args!(
                    "read feedback site={} kind={kind:?} type={observed:?}",
                    cache.site
                ),
            );
            self.request_compilation(cache.owner);
        }
        cache.recorded = 1;
    }

    fn request_compilation(&self, code: u64) {
        if self.planner.borrow_mut().request(code) && self.recompile_log.enabled() {
            let compiling = self.worker.borrow().as_ref().is_some_and(|worker| {
                worker
                    .in_flight
                    .contains_key(&jit::recompile::Key::Function(code))
            });
            self.log_recompilation(
                code,
                format_args!(
                    "requested mode={} compiling={compiling}",
                    if self.options.background_compile {
                        "background"
                    } else {
                        "synchronous"
                    }
                ),
            );
        }
        if self.options.background_compile {
            self.direct.set_pending(code, true);
            self.compilation_state
                .fetch_or(jit::recompile::REQUESTED, Ordering::Relaxed);
        } else {
            self.direct.withdraw(code);
        }
    }

    fn log_recompilation(&self, code: u64, event: std::fmt::Arguments<'_>) {
        if self.recompile_log.enabled() {
            let compiler = self.compiler.borrow();
            self.recompile_log.event(format_args!(
                "code={code} function={} {event}",
                compiler.function_name(code)
            ));
        }
    }

    pub(crate) fn has_code(&self, code: u64) -> bool {
        self.code
            .borrow()
            .get(code as usize)
            .is_some_and(Option::is_some)
            || self.compiler.borrow().has_lazy(code)
    }

    pub(crate) fn entry(&self, code: u64) -> Result<Option<jit::entries::Entry>, Error> {
        self.poll_compilation();
        if !self.options.background_compile && self.planner.borrow_mut().take(code) {
            let mut compiler = self.compiler.borrow_mut();
            compiler.update_feedback(&self.planner.borrow());
            let previous = self.code.borrow().get(code as usize).copied().flatten();
            if let Some(previous) = previous
                && let Some(job) = compiler.recompilation(code, previous)
            {
                match compiler.recompile(job) {
                    Ok(Some(replacement)) => {
                        let (code, entry) = compiler.install(replacement);
                        self.code.borrow_mut()[code as usize] = Some(entry);
                        self.recompile_log.event(format_args!(
                            "installed code={code} function={} entry={:#x}",
                            compiler.function_name(code),
                            entry.code
                        ));
                    }
                    Ok(None) => {}
                    Err(error) => eprintln!("js jit: recompilation failed: {error}"),
                }
            }
        }
        let Ok(index) = usize::try_from(code) else {
            return Ok(None);
        };
        let mut entry = self.code.borrow().get(index).copied().flatten();
        if entry.is_none() {
            entry = if self.options.background_compile {
                self.wait_for_entry(code)?
            } else {
                self.compiler.borrow_mut().compile_lazy(code)?
            };
            if let Some(compiled) = entry {
                let mut entries = self.code.borrow_mut();
                if entries.len() <= index {
                    entries.resize(index + 1, None);
                }
                entries[index] = Some(compiled);
            }
            if entry.is_some() {
                self.queue_callees(code);
            }
        }
        if let Some(entry) = entry {
            self.publish(code, entry);
        }
        Ok(entry)
    }

    fn queue_callees(&self, code: u64) {
        if !self.options.background_compile {
            return;
        }
        let jobs = self.compiler.borrow().lazy_callees(code);
        for mut job in jobs {
            job.tier = jit::CompilationTier::Baseline;
            if let Err(error) = self.enqueue_compilation(job, jit::recompile::Priority::Prefetch) {
                eprintln!("js jit: callee compilation failed: {error}");
            }
        }
    }

    fn wait_for_entry(&self, code: u64) -> Result<Option<jit::entries::Entry>, Error> {
        if let Some(error) = self.compilation_errors.borrow().get(&code) {
            return Err(error.clone());
        }
        let Some(mut job) = self.compiler.borrow().lazy_compilation(code) else {
            return Ok(None);
        };
        job.tier = jit::CompilationTier::Baseline;
        let started = self.recompile_log.enabled().then(Instant::now);
        if started.is_some() {
            self.log_recompilation(code, format_args!("lazy wait begin"));
        }
        let result = (|| {
            self.enqueue_compilation(job, jit::recompile::Priority::Demanded)?;
            self.queue_callees(code);
            loop {
                self.poll_compilation();
                if let Some(entry) = self.code.borrow().get(code as usize).copied().flatten() {
                    return Ok(Some(entry));
                }
                if let Some(error) = self.compilation_errors.borrow().get(&code) {
                    return Err(error.clone());
                }
                let mut service = self.worker.borrow_mut();
                let service = service.as_mut().unwrap();
                if !service
                    .in_flight
                    .contains_key(&jit::recompile::Key::Function(code))
                {
                    return Err(Error::Compile(format!(
                        "compilation produced no entry for code={code}"
                    )));
                }
                service.wait_for_completion()?;
                self.compilation_state
                    .fetch_or(jit::recompile::READY, Ordering::Release);
            }
        })();
        if let Some(started) = started {
            let elapsed = started.elapsed();
            let mut stats = self.lazy_wait_stats.borrow_mut();
            stats.count += 1;
            stats.total += elapsed;
            stats.longest = stats.longest.max(elapsed);
            self.log_recompilation(
                code,
                format_args!(
                    "lazy wait end wait_ms={:.3} success={}",
                    elapsed.as_secs_f64() * 1000.0,
                    result.is_ok(),
                ),
            );
        }
        result
    }

    fn queue_unlazification(&self) {
        if !self.options.background_compile {
            return;
        }
        let codes = self.compiler.borrow().lazy_codes();
        for code in codes {
            if !self.unlazify_seen.borrow_mut().insert(code) {
                continue;
            }
            let mut job = self.compiler.borrow().lazy_compilation(code).unwrap();
            job.tier = jit::CompilationTier::Baseline;
            if let Err(error) = self.enqueue_compilation(job, jit::recompile::Priority::Unlazify) {
                eprintln!("js jit: unlazification failed: {error}");
                return;
            }
        }
    }

    fn log_prefetch_uses(&self) {
        if !self.recompile_log.enabled() {
            return;
        }
        let mut stats = self.prefetch_stats.borrow_mut();
        stats.unused.retain(|&code| {
            let called = if code < jit::entries::DIRECT_ENTRIES as u64 {
                self.direct.called(code)
            } else {
                self.overflow_calls
                    .borrow()
                    .get(&code)
                    .is_some_and(|&count| count != 0)
            };
            if called {
                self.log_recompilation(
                    code,
                    format_args!("prefetch used sync_compilation=avoided"),
                );
            }
            !called
        });
        stats.used = stats.installed - stats.unused.len();
    }

    pub(crate) fn leave_function(&self, code: u64) {
        if !self.options.background_compile {
            return;
        }
        if code >= jit::entries::DIRECT_ENTRIES as u64 {
            let mut counts = self.overflow_calls.borrow_mut();
            let count = counts.entry(code).or_default();
            *count = (*count + 1).min(jit::entries::RECOMPILE_WARMUP);
            if *count < jit::entries::RECOMPILE_WARMUP {
                return;
            }
        } else if !self.direct.warmed(code) {
            return;
        }
        self.poll_compilation();
        self.submit_compilation(code);
    }

    fn submit_compilation(&self, code: u64) {
        let warmed = if code < jit::entries::DIRECT_ENTRIES as u64 {
            self.direct.warmed(code)
        } else {
            self.overflow_calls
                .borrow()
                .get(&code)
                .copied()
                .unwrap_or(0)
                >= jit::entries::RECOMPILE_WARMUP
        };
        if !warmed {
            return;
        }
        if self.worker.borrow().as_ref().is_some_and(|worker| {
            worker
                .in_flight
                .contains_key(&jit::recompile::Key::Function(code))
        }) {
            return;
        }
        if !self.planner.borrow_mut().take(code) {
            return;
        }
        self.direct.set_pending(code, false);
        if !self.planner.borrow().has_pending() {
            self.compilation_state
                .fetch_and(!jit::recompile::REQUESTED, Ordering::Relaxed);
        }
        let Some(previous) = self.code.borrow().get(code as usize).copied().flatten() else {
            return;
        };
        let mut compiler = self.compiler.borrow_mut();
        compiler.update_feedback(&self.planner.borrow());
        let Some(job) = compiler.recompilation(code, previous) else {
            return;
        };
        if let Err(error) = self.enqueue_compilation(job, jit::recompile::Priority::Optimize) {
            eprintln!("js jit: recompilation failed: {error}");
        }
    }

    fn enqueue_compilation(
        &self,
        job: jit::RecompileJob,
        priority: jit::recompile::Priority,
    ) -> Result<jit::recompile::Ticket, Error> {
        let mut service = self.worker.borrow_mut();
        if service.is_none() {
            *service = Some(jit::recompile::Service::new(
                self.options.clone(),
                self.direct.address() as usize,
                self.compilation_state.clone(),
                self.recompile_log,
            )?);
        }
        service.as_mut().unwrap().submit(job, priority, &[])
    }

    pub(crate) fn osr_entered(&self, site: &jit::osr::Site) {
        let transfers = site.transfers.get();
        site.transfers.set(transfers.saturating_add(1));
        if transfers == 0 {
            self.log_recompilation(
                site.owner,
                format_args!(
                    "osr entered header={} entries={}",
                    site.header(),
                    site.entries.get()
                ),
            );
        }
    }

    pub(crate) fn osr_entry(&self, site: &jit::osr::Site) -> usize {
        if site.entry.get() != 0 {
            return site.entry.get();
        }
        if site.entries.get() % 32 != 0 {
            return 0;
        }
        self.poll_compilation();
        if site.entry.get() != 0 {
            return site.entry.get();
        }
        if site.requested.get() {
            return 0;
        }
        let job = {
            let mut compiler = self.compiler.borrow_mut();
            compiler.update_feedback(&self.planner.borrow());
            compiler.osr_compilation(site)
        };
        let Some(job) = job else { return 0 };
        site.requested.set(true);
        self.log_recompilation(
            site.owner,
            format_args!(
                "osr requested header={} entries={}",
                site.header(),
                site.entries.get()
            ),
        );
        if !self.options.background_compile {
            let mut compiler = self.compiler.borrow_mut();
            match compiler.recompile(job) {
                Ok(Some(replacement)) => site.entry.set(compiler.install(replacement).1.code),
                Ok(None) => {}
                Err(error) => eprintln!("js jit: OSR compilation failed: {error}"),
            }
            return site.entry.get();
        }
        if let Err(error) = self.enqueue_compilation(job, jit::recompile::Priority::Optimize) {
            eprintln!("js jit: OSR compilation failed: {error}");
        }
        if site.owner < jit::entries::DIRECT_ENTRIES as u64 {
            self.direct.warm(site.owner);
        } else {
            self.overflow_calls
                .borrow_mut()
                .insert(site.owner, jit::entries::RECOMPILE_WARMUP);
        }
        self.request_compilation(site.owner);
        self.submit_compilation(site.owner);
        0
    }

    pub(crate) fn poll_compilation(&self) {
        self.log_prefetch_uses();
        if self.compilation_state.load(Ordering::Acquire) & jit::recompile::READY == 0 {
            return;
        }
        if self
            .compilation_state
            .fetch_and(!jit::recompile::READY, Ordering::Acquire)
            & jit::recompile::READY
            == 0
        {
            return;
        }
        let mut completed = Vec::new();
        {
            let mut worker = self.worker.borrow_mut();
            let Some(worker) = worker.as_mut() else {
                return;
            };
            loop {
                match worker.receive() {
                    Ok(Some(completion)) => {
                        let jit::recompile::Completion {
                            ticket,
                            key,
                            result,
                        } = completion;
                        if let jit::recompile::Key::Function(code) = key {
                            completed.push(code);
                        }
                        match result {
                            Ok(Some(compiled)) => {
                                let initial = compiled.initial();
                                let installed =
                                    self.compiler.borrow_mut().install_unpublished(compiled);
                                let Some((code, entry)) = installed else {
                                    if initial && self.recompile_log.enabled() {
                                        self.prefetch_stats.borrow_mut().discarded += 1;
                                        self.recompile_log.event(format_args!(
                                            "prefetch discarded target={key:?}"
                                        ));
                                    }
                                    worker.complete(ticket, key);
                                    continue;
                                };
                                match key {
                                    jit::recompile::Key::Function(_) => {
                                        if initial && self.recompile_log.enabled() {
                                            let mut stats = self.prefetch_stats.borrow_mut();
                                            stats.installed += 1;
                                            stats.unused.insert(code);
                                            self.log_recompilation(
                                                code,
                                                format_args!("prefetch installed"),
                                            );
                                        }
                                        {
                                            let mut entries = self.code.borrow_mut();
                                            if entries.len() <= code as usize {
                                                entries.resize(code as usize + 1, None);
                                            }
                                            entries[code as usize] = Some(entry);
                                        }
                                        self.publish(code, entry);
                                    }
                                    jit::recompile::Key::Loop(address) => {
                                        let site = unsafe { &*(address as *const jit::osr::Site) };
                                        site.entry.set(entry.code);
                                        self.log_recompilation(
                                            code,
                                            format_args!(
                                                "osr installed header={} entries={}",
                                                site.header(),
                                                site.entries.get()
                                            ),
                                        );
                                    }
                                }
                                self.log_recompilation(
                                    code,
                                    format_args!("installed entry={:#x}", entry.code),
                                );
                            }
                            Ok(None) => {}
                            Err(error) => {
                                if let jit::recompile::Key::Function(code) = key
                                    && self.compiler.borrow().has_lazy(code)
                                {
                                    self.compilation_errors.borrow_mut().insert(code, error);
                                } else {
                                    eprintln!("js jit: recompilation failed: {error}");
                                }
                            }
                        }
                        worker.complete(ticket, key);
                    }
                    Ok(None) => break,
                    Err(error) => {
                        eprintln!("js jit: recompilation failed: {error}");
                        break;
                    }
                }
            }
        }
        for code in completed {
            self.submit_compilation(code);
        }
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
        self.queue_unlazification();
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
            .map(|code| self.entry(code.0))
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

#[cfg(test)]
mod feedback_tests {
    use super::*;

    fn new_engine() -> Result<Engine, Error> {
        Engine::with_options(EngineOptions {
            background_compile: false,
            ..Default::default()
        })
    }

    fn code(engine: &Engine, name: &str) -> u64 {
        engine
            .global
            .heap()
            .closure_code(engine.global.get_raw(name).unwrap())
            .unwrap()
            .0
    }

    fn entry(engine: &Engine, code: u64) -> jit::entries::Entry {
        if !engine
            .code
            .borrow()
            .get(code as usize)
            .is_some_and(Option::is_some)
        {
            engine.entry(code).unwrap().unwrap();
        }
        engine.code.borrow()[code as usize].unwrap()
    }

    #[test]
    fn ordinary_functions_compile_on_first_call_across_sources() {
        let mut engine = new_engine().unwrap();
        engine
            .compile(
                "function first(x) { return x + 2; } function unused() { return 7; }",
                "first.js",
            )
            .unwrap();
        let first = code(&engine, "first");
        let unused = code(&engine, "unused");
        assert!(
            engine
                .code
                .borrow()
                .get(first as usize)
                .copied()
                .flatten()
                .is_none()
        );
        engine
            .compile("function second() { return 40; }", "second.js")
            .unwrap();
        assert_eq!(
            engine
                .call("first", &[Value::number(40.0)])
                .unwrap()
                .value(),
            Value::number(42.0)
        );
        assert!(
            engine
                .code
                .borrow()
                .get(first as usize)
                .copied()
                .flatten()
                .is_some()
        );
        assert_eq!(
            engine.call("second", &[]).unwrap().value(),
            Value::number(40.0)
        );
        assert!(
            engine
                .code
                .borrow()
                .get(unused as usize)
                .copied()
                .flatten()
                .is_none()
        );
    }

    #[test]
    fn closure_creation_does_not_compile_its_body() {
        let mut engine = new_engine().unwrap();
        engine
            .compile("function make(x) { return () => x; }", "closure.js")
            .unwrap();
        let closure = engine.call("make", &[Value::number(42.0)]).unwrap();
        let id = engine
            .global
            .heap()
            .closure_code(closure.value())
            .unwrap()
            .0;
        assert!(
            engine
                .code
                .borrow()
                .get(id as usize)
                .copied()
                .flatten()
                .is_none()
        );
        assert!(engine.compiler.borrow().has_lazy(id));
    }

    #[test]
    fn method_feedback_reemits_once_and_guards_replacements() {
        let mut engine = new_engine().unwrap();
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
        let mut engine = new_engine().unwrap();
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
            let mut engine = new_engine().unwrap();
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
                assert_eq!(
                    engine.global.heap().property(result.value(), "value"),
                    Some(input)
                );
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
        let mut engine = new_engine().unwrap();
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
        let mut engine = new_engine().unwrap();
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
        let mut engine = new_engine().unwrap();
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
    fn property_integer_feedback_preserves_type_changes() {
        let mut engine = new_engine().unwrap();
        engine.compile(
            "var point={x:2}; function sum(n){let total=0;for(let i=0;i<n;++i)total+=point.x+i;return total;} function replace(){point.x='3';}",
            "property-numbers.js",
        ).unwrap();
        let code = code(&engine, "sum");
        let original = entry(&engine, code);
        for _ in 0..3 {
            assert_eq!(
                engine.call("sum", &[Value::number(3.0)]).unwrap().value(),
                Value::number(9.0)
            );
        }
        assert_ne!(entry(&engine, code), original);
        engine.call("replace", &[]).unwrap();
        let result = engine.call("sum", &[Value::number(3.0)]).unwrap();
        let expected = engine.string("0303132").unwrap();
        assert!(
            engine
                .global
                .heap()
                .strings_equal(result.value(), expected.value())
        );
    }

    #[test]
    fn ineligible_feedback_keeps_the_existing_entry() {
        let mut engine = new_engine().unwrap();
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
        let mut engine = new_engine().unwrap();
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
        assert_eq!(
            engine.call("read", &args).unwrap().value(),
            Value::number(2.5)
        );
        let optimized = entry(&engine, code);
        assert_ne!(optimized, original);
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
        let mut engine = new_engine().unwrap();
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
        for index in [
            -1.0,
            0.5,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MAX,
            f64::MIN,
        ] {
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
    fn array_read_guards_follow_layout_length_and_storage_changes() {
        let mut engine = new_engine().unwrap();
        engine
            .compile(
                "var values=new Array(3);values[0]=5;values[2]=9;
             function read(i){return values[i];}
             function fill(){values[1]=7;}
             function shrink(){values.length=1;}
             function grow(){for(let i=1;i<100;++i)values.push(i);}
             function replace(){values=new Array(3);values[2]=11;}",
                "array_layout_feedback.js",
            )
            .unwrap();
        let code = code(&engine, "read");
        let original = entry(&engine, code);
        for _ in 0..3 {
            assert_eq!(
                engine.call("read", &[Value::number(2.0)]).unwrap().value(),
                Value::number(9.0)
            );
        }
        assert_ne!(entry(&engine, code), original);
        assert_eq!(
            engine.call("read", &[Value::number(1.0)]).unwrap().value(),
            Value::UNDEFINED
        );
        engine.call("fill", &[]).unwrap();
        assert_eq!(
            engine.call("read", &[Value::number(1.0)]).unwrap().value(),
            Value::number(7.0)
        );
        assert_eq!(
            engine.call("read", &[Value::number(-0.0)]).unwrap().value(),
            Value::number(5.0)
        );
        engine.call("shrink", &[]).unwrap();
        assert_eq!(
            engine.call("read", &[Value::number(2.0)]).unwrap().value(),
            Value::UNDEFINED
        );
        engine.call("grow", &[]).unwrap();
        assert_eq!(
            engine.call("read", &[Value::number(99.0)]).unwrap().value(),
            Value::number(99.0)
        );
        engine.call("replace", &[]).unwrap();
        assert_eq!(
            engine.call("read", &[Value::number(0.0)]).unwrap().value(),
            Value::UNDEFINED
        );
        assert_eq!(
            engine.call("read", &[Value::number(2.0)]).unwrap().value(),
            Value::number(11.0)
        );
    }

    #[test]
    fn get_feedback_specializes_typed_rows_inside_loops() {
        for ssa_opt in [false, true] {
            let mut engine = Engine::with_options(EngineOptions {
                background_compile: false,
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
        let mut engine = new_engine().unwrap();
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
        assert_ne!(optimized, original);
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
            background_compile: false,
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
            background_compile: false,
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

    fn background_engine() -> Engine {
        Engine::with_options(EngineOptions {
            background_compile: true,
            ..Default::default()
        })
        .unwrap()
    }

    fn osr_engine(background_compile: bool) -> Engine {
        Engine::with_options(EngineOptions {
            osr: true,
            background_compile,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn numeric_feedback_recompiles_without_read_or_call_specialization() {
        let mut engine = Engine::with_options(EngineOptions {
            background_compile: false,
            ..Default::default()
        })
        .unwrap();
        engine
            .compile(
                "function sum(x){let result=1/x;for(let i=0;i<2;++i)result+=x;return result;}",
                "integer-feedback.js",
            )
            .unwrap();
        let code = code(&engine, "sum");
        let original = entry(&engine, code);
        for _ in 0..3 {
            assert_eq!(
                engine.call("sum", &[Value::number(2.0)]).unwrap().value(),
                Value::number(4.5)
            );
        }
        assert_ne!(entry(&engine, code), original);
        for value in [
            0.5,
            -0.0,
            0.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            9007199254740992.0,
        ] {
            let expected = (1.0 / value + value) + value;
            assert_eq!(
                engine.call("sum", &[Value::number(value)]).unwrap().value(),
                Value::number(expected)
            );
        }
    }

    #[test]
    fn numeric_osr_feedback_preserves_fractional_guard_fallbacks() {
        let source = "function main(){const holder={x:2};let result=0;for(let j=0;j<160;++j){if(j===100)holder.x=0.5;const x=holder.x;let sum=0;for(let i=0;i<3;++i)sum+=(x+i)*1;result+=sum;}return result;}";
        let mut baseline = Engine::with_options(EngineOptions {
            ssa_opt: false,
            background_compile: false,
            ..Default::default()
        })
        .unwrap();
        baseline.compile(source, "numeric-osr.js").unwrap();
        let expected = baseline.call("main", &[]).unwrap().value();
        let mut engine = osr_engine(false);
        engine.compile(source, "numeric-osr.js").unwrap();
        assert_eq!(engine.call("main", &[]).unwrap().value(), expected);
        assert!(engine.compiler.borrow().osr_transfers() > 0);
    }

    #[test]
    fn typed_read_feedback_preserves_kind_and_index_checks() {
        let mut engine = Engine::with_options(EngineOptions {
            background_compile: false,
            ..Default::default()
        })
        .unwrap();
        engine.compile("var f32=Float32Array.from([1.5,2.5]);var f64=Float64Array.from([7.25]);function read(a,k){return a[k];}", "typed-read.js").unwrap();
        let f32 = engine.global.get_raw("f32").unwrap();
        assert_eq!(
            engine
                .call("read", &[f32, Value::number(0.0)])
                .unwrap()
                .value(),
            Value::number(1.5)
        );
        for index in [-1.0, 0.5, 2.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                engine
                    .call("read", &[f32, Value::number(index)])
                    .unwrap()
                    .value(),
                Value::UNDEFINED
            );
        }
        assert_eq!(
            engine
                .call("read", &[f32, Value::number(-0.0)])
                .unwrap()
                .value(),
            Value::number(1.5)
        );
        let f64 = engine.global.get_raw("f64").unwrap();
        assert_eq!(
            engine
                .call("read", &[f64, Value::number(0.0)])
                .unwrap()
                .value(),
            Value::number(7.25)
        );
    }

    #[test]
    fn loop_replacement_preserves_multiple_accumulators_and_iteration_order() {
        let source = "function main(){const rows=[Float32Array.from([1,2,3]),Float32Array.from([4,5,6])];let total=0;for(let y=0;y<200;++y){let weighted=0;let weights=0;for(let k=0;k<3;++k){if(y%2 && k===1)continue;weighted+=rows[y%2][k]/(k+1);weights+=1/(k+1);}total+=weighted/weights;}return total;}";
        let mut baseline = Engine::with_options(EngineOptions {
            osr: false,
            background_compile: false,
            ..Default::default()
        })
        .unwrap();
        baseline.compile(source, "smoothing.js").unwrap();
        let expected = baseline.call("main", &[]).unwrap().value();
        let mut engine = osr_engine(false);
        engine.compile(source, "smoothing.js").unwrap();
        assert_eq!(engine.call("main", &[]).unwrap().value(), expected);
        assert!(engine.compiler.borrow().osr_transfers() > 0);
    }

    #[test]
    fn loop_replacement_keeps_effects_and_guard_failure_continuations() {
        let mut engine = osr_engine(false);
        engine.compile("function main(){const rows=[Float32Array.from([1,2,3])];const effects={n:0};let total=0;for(let y=0;y<200;++y){if(y===60)rows[0]={0:4,1:5,2:6};let sum=0;for(let k=0;k<3;++k){effects.n+=1;sum+=rows[0][k]/1;}total+=sum;}return total+effects.n;}", "effects.js").unwrap();
        assert_eq!(
            engine.call("main", &[]).unwrap().value(),
            Value::number(3060.0)
        );
        assert!(engine.compiler.borrow().osr_transfers() > 0);
    }

    #[test]
    fn loop_replacement_propagates_errors_and_releases_its_frame() {
        let mut engine = osr_engine(false);
        engine.compile("function main(){const rows=[Float32Array.from([1,2,3])];let total=0;for(let y=0;y<200;++y){if(y===60)rows[0]=null;let sum=0;for(let k=0;k<3;++k)sum+=rows[0][k]/1;total+=sum;}return total;} function good(){return 42;}", "loop-error.js").unwrap();
        let error = engine.call("main", &[]).err().unwrap();
        assert!(error.to_string().contains("loop-error.js"), "{error}");
        assert!(engine.compiler.borrow().osr_transfers() > 0);
        assert_eq!(
            engine.call("good", &[]).unwrap().value(),
            Value::number(42.0)
        );
    }

    #[test]
    fn background_loop_replacement_runs_before_the_first_function_return() {
        let mut engine = osr_engine(true);
        engine.compile("function main(){const rows=[[1,2,3]];let total=0;for(let y=0;y<2000000;++y){let sum=0;for(let k=0;k<3;++k)sum+=rows[0][k]/1;total+=sum;}return total;}", "background-loop.js").unwrap();
        let code = code(&engine, "main");
        entry(&engine, code);
        assert_eq!(
            engine.call("main", &[]).unwrap().value(),
            Value::number(12000000.0)
        );
        assert!(engine.compiler.borrow().osr_transfers() > 0);
        finish_compilation(&engine);
    }

    #[test]
    fn osr_request_queues_the_enclosing_function_before_return() {
        let mut engine = osr_engine(true);
        engine
            .compile(
            "var values=[1,2,3];function hot(xs){let total=0;for(let i=0;i<3;++i)total+=xs[i]/1;return total;}",
                "osr-warmup.js",
            )
            .unwrap();
        let code = code(&engine, "hot");
        entry(&engine, code);
        let (site, location, kind) = engine.compiler.borrow().osr_read_site(code).unwrap();
        let site = unsafe { &*site };
        let mut cache = jit::planner::ObservationCache {
            recorded: 0,
            owner: code,
            site: location,
        };
        engine.observe_get(&mut cache, jit::planner::ReadType::Array, kind);
        assert!(!engine.direct.warmed(code));
        site.entries.set(32);
        engine.osr_entry(site);
        assert!(engine.direct.warmed(code));
        {
            let worker = engine.worker.borrow();
            let worker = worker.as_ref().unwrap();
            assert!(
                worker
                    .in_flight
                    .contains_key(&jit::recompile::Key::Loop(site as *const _ as usize))
            );
            assert!(
                worker
                    .in_flight
                    .contains_key(&jit::recompile::Key::Function(code))
            );
        }
        finish_compilation(&engine);
        engine.request_compilation(code);
        let values = engine.global.get_raw("values").unwrap();
        assert_eq!(
            engine.call("hot", &[values]).unwrap().value(),
            Value::number(6.0)
        );
        assert!(
            engine
                .worker
                .borrow()
                .as_ref()
                .unwrap()
                .in_flight
                .contains_key(&jit::recompile::Key::Function(code))
        );
        finish_compilation(&engine);
    }

    fn wait_ready(engine: &Engine) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while engine.compilation_state.load(Ordering::Acquire) & jit::recompile::READY == 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "compilation did not complete"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    fn finish_compilation(engine: &Engine) {
        loop {
            if engine
                .worker
                .borrow()
                .as_ref()
                .is_none_or(|worker| worker.in_flight.is_empty())
            {
                break;
            }
            wait_ready(engine);
            engine.poll_compilation();
        }
    }

    #[test]
    fn background_pool_compiles_multiple_targets() {
        for compiler_threads in [1, 2, 3] {
            let mut engine = Engine::with_options(EngineOptions {
                background_compile: true,
                compiler_threads,
                ..Default::default()
            })
            .unwrap();
            let source = (0..4)
                .map(|i| format!("function read{i}(a){{return a.items[0];}}"))
                .collect::<String>();
            engine
                .compile(&format!("var holder={{items:[3]}};{source}"), "pool.js")
                .unwrap();
            let holder = engine.global.get_raw("holder").unwrap();
            let originals: Vec<_> = (0..4)
                .map(|i| {
                    let name = format!("read{i}");
                    let code = code(&engine, &name);
                    (name, code, entry(&engine, code))
                })
                .collect();
            for (name, _, _) in &originals {
                for _ in 0..=jit::entries::RECOMPILE_WARMUP {
                    assert_eq!(
                        engine.call(name, &[holder]).unwrap().value(),
                        Value::number(3.0)
                    );
                }
            }
            finish_compilation(&engine);
            for (name, code, original) in originals {
                assert_ne!(entry(&engine, code), original);
                assert_eq!(
                    engine.call(&name, &[holder]).unwrap().value(),
                    Value::number(3.0)
                );
            }
        }
    }

    #[test]
    fn baseline_bodies_upgrade_without_specialization_changes() {
        let mut engine = Engine::with_options(EngineOptions {
            background_compile: false,
            ..Default::default()
        })
        .unwrap();
        engine
            .compile("function f(a){return a+1;}", "tier-upgrade.js")
            .unwrap();
        let code = code(&engine, "f");
        let mut job = engine.compiler.borrow().lazy_compilation(code).unwrap();
        job.tier = jit::CompilationTier::Baseline;
        let mut baseline = Compiler::new(
            engine.options.clone(),
            jit::CompilationTier::Baseline,
            engine.direct.address(),
            0,
            engine.recompile_log,
        )
        .unwrap();
        let replacement = baseline.recompile(job).unwrap().unwrap();
        let unpublished = baseline.take_unpublished(replacement);
        let (_, original) = engine
            .compiler
            .borrow_mut()
            .install_unpublished(unpublished)
            .unwrap();
        engine.code.borrow_mut().resize(code as usize + 1, None);
        engine.code.borrow_mut()[code as usize] = Some(original);
        assert_eq!(
            engine.call("f", &[Value::number(2.0)]).unwrap().value(),
            Value::number(3.0)
        );
        let job = engine
            .compiler
            .borrow()
            .recompilation(code, original)
            .unwrap();
        let replacement = engine
            .compiler
            .borrow_mut()
            .recompile(job)
            .unwrap()
            .unwrap();
        let (_, optimized) = engine.compiler.borrow_mut().install(replacement);
        engine.code.borrow_mut()[code as usize] = Some(optimized);
        assert_ne!(original.code, optimized.code);
        assert_eq!(
            engine.call("f", &[Value::number(2.0)]).unwrap().value(),
            Value::number(3.0)
        );
        let job = engine
            .compiler
            .borrow()
            .recompilation(code, optimized)
            .unwrap();
        assert!(
            engine
                .compiler
                .borrow_mut()
                .recompile(job)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn workers_compile_independent_lazy_bodies_without_publication() {
        let engine = Engine::with_options(EngineOptions {
            background_compile: true,
            compiler_threads: 1,
            ..Default::default()
        })
        .unwrap();
        engine.compile_only(
            "function first(a){return a.x;} function second(a){return a.y;} function third(a){return a.z;}",
            "independent.js",
        ).unwrap();
        let count = engine.compiler.borrow().lazy_codes().len();
        assert_eq!(count, 3);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let mut service = engine.worker.borrow_mut();
        let service = service.as_mut().unwrap();
        for _ in 0..count {
            let completion = loop {
                assert!(
                    std::time::Instant::now() < deadline,
                    "compilation did not complete"
                );
                service.wait_for_completion().unwrap();
                if let Some(completion) = service.receive().unwrap() {
                    break completion;
                }
            };
            assert!(completion.result.unwrap().unwrap().initial());
        }
        assert_eq!(engine.compiler.borrow().lazy_codes().len(), count);
    }

    #[test]
    fn background_unlazification_drains_uncalled_bodies_across_sources() {
        let mut engine = background_engine();
        for source in [
            "function unused(a){return a.items[0];} function factory(a){return ()=>a;}",
            "function later(a){return a.items[1];}",
        ] {
            engine.compile(source, "unlazify.js").unwrap();
            finish_compilation(&engine);
            assert!(engine.compiler.borrow().lazy_codes().is_empty());
        }
    }

    #[test]
    fn synchronous_compilation_prefetches_callees_exposed_by_inlining() {
        let mut engine = Engine::with_options(EngineOptions {
            background_compile: true,
            recompile_log: true,
            ..Default::default()
        })
        .unwrap();
        engine.compile("function caller(a){return callee(a);} function callee(a){return leaf(a);} function leaf(a){return a.items[0];}", "prefetch.js").unwrap();
        let caller = code(&engine, "caller");
        let callee = code(&engine, "callee");
        let leaf = code(&engine, "leaf");
        engine.entry(caller).unwrap().unwrap();
        finish_compilation(&engine);
        assert!(!engine.compiler.borrow().has_lazy(callee));
        assert!(!engine.compiler.borrow().has_lazy(leaf));
        assert!((2..=3).contains(&engine.prefetch_stats.borrow().installed));
        assert_eq!(engine.prefetch_stats.borrow().used, 0);
        engine
            .compile("var holder={items:[3]};", "holder.js")
            .unwrap();
        let holder = engine.global.get_raw("holder").unwrap();
        let entry = engine.code.borrow()[callee as usize].unwrap();
        let callable = engine.global.get_raw("callee").unwrap();
        let result =
            unsafe { jit::call(entry, &[holder], &engine, callable, Value::UNDEFINED) }.unwrap();
        assert_eq!(result, Value::number(3.0));
        engine.global.heap().release(result);
        finish_compilation(&engine);
        engine.log_prefetch_uses();
        assert_eq!(engine.prefetch_stats.borrow().used, 1);
        engine.log_prefetch_uses();
        assert_eq!(engine.prefetch_stats.borrow().used, 1);
    }

    #[test]
    fn synchronous_compilation_supersedes_an_unpublished_initial_entry() {
        let mut engine = background_engine();
        engine
            .compile("function f(a){return a.items[0];}", "takeover.js")
            .unwrap();
        let code = code(&engine, "f");
        let job = engine.compiler.borrow().lazy_compilation(code).unwrap();
        let mut worker = Compiler::new(
            engine.options.clone(),
            jit::CompilationTier::Optimized,
            engine.direct.address(),
            Arc::as_ptr(&engine.compilation_state) as usize,
            engine.recompile_log,
        )
        .unwrap();
        let replacement = worker.recompile(job).unwrap().unwrap();
        let unpublished = worker.take_unpublished(replacement);
        let original = engine.entry(code).unwrap().unwrap();
        assert!(
            engine
                .compiler
                .borrow_mut()
                .install_unpublished(unpublished)
                .is_none()
        );
        assert_eq!(engine.entry(code).unwrap(), Some(original));
    }

    #[test]
    fn prefetched_closures_preserve_captures_and_lexical_this() {
        let mut engine = background_engine();
        let body = "total += a;".repeat(80);
        engine.compile(&format!("function caller(a){{const f=()=>{{let total=this.base+a;{body}return total;}};return f();}} var obj={{base:10,run:caller}}; function run(){{return obj.run(2);}}"), "prefetch-closure.js").unwrap();
        let caller = code(&engine, "caller");
        let jobs = engine.compiler.borrow().lazy_callees(caller);
        assert_eq!(jobs.len(), 1);
        let closure = u64::from(jobs[0].code.as_u32());
        engine.entry(caller).unwrap().unwrap();
        finish_compilation(&engine);
        assert!(!engine.compiler.borrow().has_lazy(closure));
        assert!(engine.code.borrow()[closure as usize].unwrap().lexical_this);
        assert_eq!(
            engine.call("run", &[]).unwrap().value(),
            Value::number(172.0)
        );
        finish_compilation(&engine);
    }

    #[test]
    fn background_recompilation_starts_after_eight_returns_and_guards_replacements() {
        let mut engine = background_engine();
        engine.compile(
            "var holder={items:[3]}; function read(a){return a.items[0];} function replace(){holder.items={0:9};}",
            "read.js",
        ).unwrap();
        finish_compilation(&engine);
        let code = code(&engine, "read");
        let original = entry(&engine, code);
        let holder = engine.global.get_raw("holder").unwrap();
        for _ in 0..7 {
            assert_eq!(
                engine.call("read", &[holder]).unwrap().value(),
                Value::number(3.0)
            );
            assert_eq!(entry(&engine, code), original);
            assert!(
                engine
                    .worker
                    .borrow()
                    .as_ref()
                    .unwrap()
                    .in_flight
                    .is_empty()
            );
        }
        assert_eq!(
            engine.call("read", &[holder]).unwrap().value(),
            Value::number(3.0)
        );
        assert!(
            engine
                .worker
                .borrow()
                .as_ref()
                .unwrap()
                .in_flight
                .contains_key(&jit::recompile::Key::Function(code))
        );
        engine
            .compile("function unrelated(){return 42;}", "later.js")
            .unwrap();
        finish_compilation(&engine);
        let optimized = entry(&engine, code);
        assert_ne!(optimized, original);
        assert_eq!(
            engine.call("read", &[holder]).unwrap().value(),
            Value::number(3.0)
        );
        assert_eq!(
            engine.compilation_state.load(Ordering::Relaxed) & jit::recompile::REQUESTED,
            0
        );
        engine.call("replace", &[]).unwrap();
        assert_eq!(
            engine.call("read", &[holder]).unwrap().value(),
            Value::number(9.0)
        );
        assert_eq!(entry(&engine, code), optimized);
    }

    #[test]
    fn generated_dispatch_installs_background_results() {
        let mut engine = background_engine();
        let body = "n += a.extra;".repeat(40);
        engine.compile(
            &format!("var holder={{items:[3],extra:0}}; function read(a){{let n=0;{body}return a.items[0]+n;}} function drive(a){{return read(a);}}"),
            "dispatch.js",
        ).unwrap();
        finish_compilation(&engine);
        let read = code(&engine, "read");
        let original = entry(&engine, read);
        let holder = engine.global.get_raw("holder").unwrap();
        for _ in 0..8 {
            engine.call("read", &[holder]).unwrap();
        }
        wait_ready(&engine);
        let drive = entry(&engine, code(&engine, "drive"));
        let callable = engine.global.get_raw("drive").unwrap();
        let value =
            unsafe { jit::call(drive, &[holder], &engine, callable, Value::UNDEFINED) }.unwrap();
        assert_eq!(value, Value::number(3.0));
        engine.global.heap().release(value);
        assert_ne!(entry(&engine, read), original);
        finish_compilation(&engine);
    }

    #[test]
    fn background_warmup_counts_direct_calls_before_feedback() {
        let mut engine = background_engine();
        engine.compile(
            "var holder={items:[3]}; function read(a){if(!a)return 0;return a.items[0];} function drive(f,a){return f(a);}",
            "warmup.js",
        ).unwrap();
        let code = code(&engine, "read");
        let callable = engine.global.get_raw("read").unwrap();
        for _ in 0..7 {
            assert_eq!(
                engine
                    .call("drive", &[callable, Value::NULL])
                    .unwrap()
                    .value(),
                Value::number(0.0)
            );
        }
        assert!(!engine.direct.warmed(code));
        let holder = engine.global.get_raw("holder").unwrap();
        assert_eq!(
            engine.call("drive", &[callable, holder]).unwrap().value(),
            Value::number(3.0)
        );
        assert!(engine.direct.warmed(code));
        assert!(
            engine
                .worker
                .borrow()
                .as_ref()
                .unwrap()
                .in_flight
                .contains_key(&jit::recompile::Key::Function(code))
        );
        finish_compilation(&engine);
    }

    #[test]
    fn requests_during_background_compilation_are_coalesced_and_retained() {
        let mut engine = background_engine();
        engine
            .compile(
                "var holder={items:[3]}; function read(a){return a.items[0];}",
                "requests.js",
            )
            .unwrap();
        let code = code(&engine, "read");
        let holder = engine.global.get_raw("holder").unwrap();
        for _ in 0..8 {
            engine.call("read", &[holder]).unwrap();
        }
        for _ in 0..3 {
            engine.request_compilation(code);
            engine.submit_compilation(code);
        }
        assert_eq!(engine.worker.borrow().as_ref().unwrap().in_flight.len(), 1);
        assert!(engine.planner.borrow().has_pending());
        finish_compilation(&engine);
        assert!(!engine.planner.borrow().has_pending());
        assert_eq!(
            engine.call("read", &[holder]).unwrap().value(),
            Value::number(3.0)
        );
    }

    #[test]
    fn background_versions_preserve_captures_heap_returns_and_foreign_errors() {
        let mut engine = background_engine();
        engine.compile(
            "function make(offset){return function(x){return {value:this.base+offset+x.value};};} var object={base:10,run:make(2)}; var input={value:3}; function replace(){object.run=function(x){return x;};}",
            "callee.js",
        ).unwrap();
        engine
            .compile("function invoke(x){return object.run(x);}", "caller.js")
            .unwrap();
        let input = engine.global.get_raw("input").unwrap();
        let code = code(&engine, "invoke");
        let original = entry(&engine, code);
        let first = engine.call("invoke", &[input]).unwrap();
        for _ in 1..8 {
            engine.call("invoke", &[input]).unwrap();
        }
        finish_compilation(&engine);
        assert_ne!(entry(&engine, code), original);
        let second = engine.call("invoke", &[input]).unwrap();
        engine.drain_disposals(1000);
        for result in [first, second] {
            assert_eq!(
                engine.global.heap().property(result.value(), "value"),
                Some(Value::number(15.0))
            );
        }
        let error = engine
            .call("invoke", &[Value::NULL])
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("callee.js:"), "{error}");
        engine.call("replace", &[]).unwrap();
        assert_eq!(engine.call("invoke", &[input]).unwrap().value(), input);
        finish_compilation(&engine);
    }
}

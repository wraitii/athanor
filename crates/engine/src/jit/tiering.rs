use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{Duration, Instant};

use oxc_ast::ast::Program;
use oxc_semantic::{AstNodes, Scoping};

use crate::EngineOptions;
use crate::error::Error;
use crate::jit::{self, Compiler};

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

pub(crate) struct Tiering {
    // Stop the workers first; in-flight compiles finish detached on their own snapshots.
    worker: RefCell<Option<jit::recompile::Service>>,
    unlazify_seen: RefCell<HashSet<u64>>,
    compilation_errors: RefCell<HashMap<u64, Error>>,
    lazy_wait_stats: RefCell<LazyWaitStats>,
    compilation_state: Arc<AtomicU8>,
    options: EngineOptions,
    recompile_log: jit::recompile::Log,
    prefetch_stats: RefCell<PrefetchStats>,
    compiler: RefCell<Compiler>,
    profile: RefCell<jit::Profile>,
    recompiles: RefCell<jit::Recompiles>,
    // Code IDs are function indices in this engine's persistent Cranelift module.
    code: RefCell<Vec<Option<jit::entries::Entry>>>,
    direct: jit::DirectEntries,
    overflow_calls: RefCell<HashMap<u64, usize>>,
}

impl Tiering {
    pub(crate) fn new(options: EngineOptions) -> Result<Self, Error> {
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
            profile: RefCell::new(jit::Profile::default()),
            recompiles: RefCell::new(jit::Recompiles::default()),
            code: RefCell::new(Vec::new()),
            direct,
            overflow_calls: RefCell::new(HashMap::new()),
        })
    }

    pub(crate) fn report(&self) {
        if self.recompile_log.enabled() {
            let stats = self.prefetch_stats.borrow();
            self.recompile_log.event(format_args!(
                "prefetch summary installed={} used={} unused={} discarded={}",
                stats.installed,
                stats.used,
                stats.unused.len(),
                stats.discarded
            ));
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
                self.compiler.borrow().call_profile(&self.profile.borrow())
            );
        }
    }

    pub(crate) fn compile_program<'a>(
        &self,
        program: &'a Program<'a>,
        scoping: &Scoping,
        nodes: &AstNodes<'a>,
        source: &str,
        filename: &str,
    ) -> Result<(Vec<jit::CompiledFunction>, Vec<String>, Option<usize>), Error> {
        let (functions, globals, initializer, code) = self
            .compiler
            .borrow_mut()
            .compile_program(program, scoping, nodes, source, filename)?;
        let mut entries = self.code.borrow_mut();
        for (code, entry) in code {
            let index = usize::try_from(code).unwrap();
            if entries.len() <= index {
                entries.resize(index + 1, None);
            }
            entries[index] = Some(entry);
            self.publish(code, entry);
        }
        Ok((functions, globals, initializer))
    }

    pub(crate) fn record_calls<'s>(
        &self,
        samples: impl Iterator<Item = (&'s jit::profile::CallSite, u64)>,
    ) {
        let mut profile = self.profile.borrow_mut();
        let mut run: Option<(&jit::profile::CallSite, u64, u64)> = None;
        for (site, code) in samples {
            match &mut run {
                Some((last, last_code, count))
                    if std::ptr::eq(*last, site) && *last_code == code =>
                {
                    *count += 1;
                }
                _ => {
                    if let Some((last, last_code, count)) = run.replace((site, code, 1)) {
                        profile.record_call(last, last_code, count);
                    }
                }
            }
        }
        if let Some((site, code, count)) = run {
            profile.record_call(site, code, count);
        }
    }

    pub(crate) fn observe_call(&self, cache: &mut jit::profile::ObservationCache, target: u64) {
        let observed = self
            .profile
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
        cache: &mut jit::profile::ObservationCache,
        target: crate::heap::NativeFunction,
    ) {
        let observed = self
            .profile
            .borrow_mut()
            .observe_native(cache.owner, cache.site, target);
        if observed {
            self.request_compilation(cache.owner);
        }
        cache.recorded = 1;
    }

    pub(crate) fn observe_get(
        &self,
        cache: &mut jit::profile::ObservationCache,
        observed: jit::profile::ReadType,
        kind: jit::profile::ReadKind,
    ) {
        let recorded =
            self.profile
                .borrow_mut()
                .observe_get(cache.owner, cache.site, kind, observed);
        if recorded && observed != jit::profile::ReadType::Unknown {
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

    pub(crate) fn observe_shape(
        &self,
        owner: u64,
        site: jit::profile::Site,
        property: jit::profile::PropertyAccess,
        access: jit::profile::ShapeAccess,
    ) {
        let Some(merged) = self
            .profile
            .borrow_mut()
            .observe_shape(site, property, access)
        else {
            return;
        };
        self.log_recompilation(
            owner,
            format_args!("shape feedback site={site} {property:?} access={merged:?}"),
        );
        self.request_compilation(owner);
    }

    fn request_compilation(&self, code: u64) {
        if self.recompiles.borrow_mut().request(code) && self.recompile_log.enabled() {
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
        if !self.options.background_compile && self.recompiles.borrow_mut().take(code) {
            let mut compiler = self.compiler.borrow_mut();
            compiler.update_feedback(&self.profile.borrow());
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

    pub(crate) fn queue_unlazification(&self) {
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

    pub(crate) fn log_prefetch_uses(&self) {
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
        if !self.recompiles.borrow_mut().take(code) {
            return;
        }
        self.direct.set_pending(code, false);
        if !self.recompiles.borrow().has_pending() {
            self.compilation_state
                .fetch_and(!jit::recompile::REQUESTED, Ordering::Relaxed);
        }
        let Some(previous) = self.code.borrow().get(code as usize).copied().flatten() else {
            return;
        };
        let mut compiler = self.compiler.borrow_mut();
        compiler.update_feedback(&self.profile.borrow());
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
        if site.entries.get() % jit::osr::TICK != 0 {
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
            compiler.update_feedback(&self.profile.borrow());
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

    pub(crate) fn publish(&self, code: u64, entry: jit::entries::Entry) {
        match entry.adapter {
            Some(_) => self.direct.publish(code, entry),
            None => self.direct.withdraw(code),
        }
    }
}

#[cfg(test)]
mod tests;

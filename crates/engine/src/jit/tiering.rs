use crate::{FxHashMap, FxHashSet};
use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{Duration, Instant};

use oxc_ast::ast::Program;
use oxc_semantic::{AstNodes, Scoping};

use super::entries::{VariantId, Variants};
use crate::EngineOptions;
use crate::error::Error;
use crate::jit::{self, Compiler};

#[derive(Default)]
struct PrefetchStats {
    unused: FxHashSet<u64>,
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
    unlazify_seen: RefCell<FxHashSet<u64>>,
    compilation_errors: RefCell<FxHashMap<u64, Error>>,
    lazy_wait_stats: RefCell<LazyWaitStats>,
    compilation_state: Arc<AtomicU8>,
    options: EngineOptions,
    recompile_log: jit::recompile::Log,
    prefetch_stats: RefCell<PrefetchStats>,
    compiler: RefCell<Compiler>,
    profile: RefCell<jit::Profile>,
    recompiles: RefCell<jit::Recompiles>,
    variants: RefCell<Variants>,
    direct: jit::DirectEntries,
    overflow_calls: RefCell<FxHashMap<u64, usize>>,
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
            unlazify_seen: RefCell::new(FxHashSet::default()),
            compilation_errors: RefCell::new(FxHashMap::default()),
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
            variants: RefCell::new(Variants::default()),
            direct,
            overflow_calls: RefCell::new(FxHashMap::default()),
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
        let mut entries = self.variants.borrow_mut();
        for (code, entry) in code {
            entries.publish(VariantId::ordinary(code), entry);
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
        let update = self
            .profile
            .borrow_mut()
            .observe(cache.owner, cache.site, target);
        let site = cache.site;
        self.apply_feedback(
            cache,
            update,
            format_args!("call feedback site={site} target={target}"),
        );
    }

    fn apply_feedback(
        &self,
        cache: &mut jit::profile::ObservationCache,
        update: jit::profile::Update,
        event: std::fmt::Arguments<'_>,
    ) {
        cache.complete = u64::from(update.complete);
        if update.changed {
            self.log_recompilation(cache.owner, event);
            self.request_compilation(cache.owner);
        }
    }

    pub(crate) fn observe_native(
        &self,
        cache: &mut jit::profile::ObservationCache,
        target: crate::heap::NativeFunction,
    ) {
        let changed = self
            .profile
            .borrow_mut()
            .observe_native(cache.owner, cache.site, target);
        let site = cache.site;
        self.apply_feedback(
            cache,
            jit::profile::Update {
                changed,
                complete: false,
            },
            format_args!("native feedback site={site} target={target:?}"),
        );
    }

    pub(crate) fn observe_get(
        &self,
        cache: &mut jit::profile::ObservationCache,
        observed: jit::profile::ReadType,
        kind: jit::profile::ReadKind,
    ) {
        let changed =
            self.profile
                .borrow_mut()
                .observe_get(cache.owner, cache.site, kind, observed)
                && observed != jit::profile::ReadType::Unknown;
        let site = cache.site;
        self.apply_feedback(
            cache,
            jit::profile::Update {
                changed,
                complete: true,
            },
            format_args!("read feedback site={site} kind={kind:?} type={observed:?}"),
        );
    }

    pub(crate) fn observe_number_failure(
        &self,
        cache: &mut jit::profile::ObservationCache,
        observed: jit::profile::ReadType,
    ) {
        let update =
            self.profile
                .borrow_mut()
                .observe_number_failure(cache.owner, cache.site, observed);
        let site = cache.site;
        self.apply_feedback(
            cache,
            update,
            format_args!("number guard failure site={site} type={observed:?}"),
        );
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
            .observe_shape(owner, site, property, access)
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
                    .contains_key(&jit::recompile::Key::ordinary(code))
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
            self.direct.withdraw(VariantId::ordinary(code));
            for variant in self.compiler.borrow().variants_for_function(code) {
                variant.entry.store(0, Ordering::Release);
            }
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
        self.variants
            .borrow()
            .get(VariantId::ordinary(code))
            .is_some()
            || self.compiler.borrow().has_lazy(code)
    }

    pub(crate) fn entry(&self, code: u64) -> Result<Option<jit::entries::Entry>, Error> {
        self.poll_compilation();
        if !self.options.background_compile && self.recompiles.borrow_mut().take(code) {
            for job in self.recompilations(code) {
                let replacement = self.compiler.borrow_mut().recompile(job);
                match replacement {
                    Ok(Some(replacement)) => {
                        let installed = self.compiler.borrow_mut().install(replacement);
                        self.variants
                            .borrow_mut()
                            .publish(installed.variant, installed.entry);
                        self.publish_variant(installed.variant, installed.entry);
                    }
                    Ok(None) => {}
                    Err(error) => eprintln!("js jit: recompilation failed: {error}"),
                }
            }
            for variant in self.compiler.borrow().variants_for_function(code) {
                if let Some(entry) = self.variants.borrow().get(variant.id()) {
                    variant.publish(entry);
                }
            }
        }

        let variant = VariantId::ordinary(code);
        let mut entry = self.variants.borrow().get(variant);
        if entry.is_none() {
            entry = if self.options.background_compile {
                self.wait_for_entry(code)?
            } else {
                self.compiler.borrow_mut().compile_lazy(code)?
            };
            if let Some(compiled) = entry {
                self.variants.borrow_mut().publish(variant, compiled);
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
                if let Some(entry) = self.variants.borrow().get(VariantId::ordinary(code)) {
                    return Ok(Some(entry));
                }
                if let Some(error) = self.compilation_errors.borrow().get(&code) {
                    return Err(error.clone());
                }
                let mut service = self.worker.borrow_mut();
                let service = service.as_mut().unwrap();
                if !service
                    .in_flight
                    .contains_key(&jit::recompile::Key::ordinary(code))
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
            worker.in_flight.keys().any(|key| {
                matches!(key,
                    jit::recompile::Key::Variant { function, .. } if *function == code
                )
            })
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
        for job in self.recompilations(code) {
            if let Err(error) = self.enqueue_compilation(job, jit::recompile::Priority::Optimize) {
                eprintln!("js jit: recompilation failed: {error}");
            }
        }
    }

    fn recompilations(&self, code: u64) -> Vec<jit::RecompileJob> {
        let mut compiler = self.compiler.borrow_mut();
        compiler.update_feedback(&self.profile.borrow());
        let entries = self.variants.borrow();
        let mut jobs = Vec::new();
        if let Some(previous) = entries.get(VariantId::ordinary(code))
            && let Some(job) = compiler.recompilation(code, previous)
        {
            jobs.push(job);
        }
        for variant in compiler.variants_for_function(code) {
            if let Some(previous) = entries.get(variant.id())
                && let Some(job) = compiler.variant_compilation(variant.id(), previous)
            {
                jobs.push(job);
            }
        }
        jobs
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

    pub(crate) fn variant_entry(&self, variant: &jit::variants::Variant) -> usize {
        if variant.entry.load(Ordering::Acquire) != 0 {
            return variant.entry.load(Ordering::Acquire);
        }
        let ordinary = match self.entry(variant.function) {
            Ok(Some(entry)) if entry.adapter.is_some() && !entry.lexical_this => entry,
            _ => return 0,
        };
        let current = variant.entry.load(Ordering::Acquire);
        if current != 0 {
            return current;
        }
        variant.publish(ordinary);
        self.variants.borrow_mut().publish(variant.id(), ordinary);
        let job = {
            let mut compiler = self.compiler.borrow_mut();
            compiler.update_feedback(&self.profile.borrow());
            compiler.variant_compilation(variant.id(), ordinary)
        };
        let Some(job) = job else { return ordinary.code };
        if self.options.background_compile {
            if let Err(error) = self.enqueue_compilation(job, jit::recompile::Priority::Optimize) {
                eprintln!("js jit: variant compilation failed: {error}");
            }
        } else {
            let replacement = self.compiler.borrow_mut().recompile(job);
            match replacement {
                Ok(Some(replacement)) => {
                    let installed = self.compiler.borrow_mut().install(replacement);
                    self.variants
                        .borrow_mut()
                        .publish(installed.variant, installed.entry);
                    variant.publish(installed.entry);
                }
                Ok(None) => {}
                Err(error) => eprintln!("js jit: variant compilation failed: {error}"),
            }
        }
        variant.entry.load(Ordering::Acquire)
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
        if !site.entries.get().is_multiple_of(jit::osr::TICK) {
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
                Ok(Some(replacement)) => site.entry.set(compiler.install(replacement).entry.code),
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
                        if let jit::recompile::Key::Variant { function: code, .. } = key {
                            completed.push(code);
                        }
                        match result {
                            Ok(Some(compiled)) => {
                                let initial = compiled.initial();
                                let installed =
                                    self.compiler.borrow_mut().install_unpublished(compiled);
                                let Some(jit::compiler::Installed {
                                    function: code,
                                    variant,
                                    entry,
                                }) = installed
                                else {
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
                                    jit::recompile::Key::Variant { .. } => {
                                        if initial && self.recompile_log.enabled() {
                                            let mut stats = self.prefetch_stats.borrow_mut();
                                            stats.installed += 1;
                                            stats.unused.insert(code);
                                            self.log_recompilation(
                                                code,
                                                format_args!("prefetch installed"),
                                            );
                                        }
                                        self.variants.borrow_mut().publish(variant, entry);
                                        self.publish_variant(variant, entry);
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
                                if let jit::recompile::Key::Variant { function: code, .. } = key
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
        self.publish_variant(VariantId::ordinary(code), entry);
    }

    fn publish_variant(&self, variant: VariantId, entry: jit::entries::Entry) {
        if let Some(variant) = self.compiler.borrow().variant(variant) {
            variant.publish(entry);
            return;
        }
        match entry.adapter {
            Some(_) => self.direct.publish(variant, entry),
            None => self.direct.withdraw(variant),
        }
    }
}

#[cfg(test)]
mod tests;

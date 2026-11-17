use super::Tiering;
use crate::jit;

impl Tiering {
    pub(crate) fn baseline_warmed(&self, code: u64) {
        self.log_recompilation(
            code,
            format_args!("baseline warmup calls={}", self.options.baseline_warmup),
        );
        self.request_hot_compilation(code);
    }

    pub(crate) fn baseline_osr_entered(&self, site: &jit::baseline_tiering::Site) {
        let transfers = site.transfers.get();
        site.transfers.set(transfers.saturating_add(1));
        if transfers == 0 {
            self.recompile_log.event(format_args!(
                "baseline OSR entered header={} entries={}",
                site.header,
                site.entries.get(),
            ));
        }
    }

    pub(crate) fn baseline_osr_entry(&self, site: &jit::baseline_tiering::Site) -> usize {
        if site.entry.get() != 0 {
            return site.entry.get();
        }
        self.poll_compilation();
        if site.entry.get() != 0 || site.requested.replace(true) {
            return site.entry.get();
        }
        let job = self.compiler.borrow().baseline_osr_compilation(site);
        let job = match job {
            Ok(Some(job)) => job,
            Ok(None) => return 0,
            Err(error) => {
                self.recompile_log.event(format_args!(
                    "baseline OSR declined header={}: {error}",
                    site.header
                ));
                return 0;
            }
        };
        self.recompile_log.event(format_args!(
            "baseline OSR requested header={} entries={}",
            site.header,
            site.entries.get()
        ));
        if self.options.background_compile {
            if let Err(error) = self.enqueue_compilation(job, jit::recompile::Priority::Optimize) {
                eprintln!("js jit: baseline OSR compilation failed: {error}");
            }
        } else {
            let mut compiler = self.compiler.borrow_mut();
            match compiler.recompile(job) {
                Ok(Some(replacement)) => site.entry.set(compiler.install(replacement).entry.code),
                Ok(None) => {}
                Err(error) => eprintln!("js jit: baseline OSR compilation failed: {error}"),
            }
        }
        site.entry.get()
    }
}

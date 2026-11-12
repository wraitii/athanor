use crate::FxHashMap;
use std::sync::{Mutex, OnceLock};

struct Site {
    description: String,
    hits: u64,
}

static SITES: Mutex<Vec<Site>> = Mutex::new(Vec::new());
static GUARDS: Mutex<Option<FxHashMap<(u32, u32), String>>> = Mutex::new(None);

pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("JS_JIT_BAILOUT_LOG").as_deref() == Ok("1"))
}

pub(crate) fn note_guard(span: oxc_span::Span, text: String) {
    GUARDS
        .lock()
        .unwrap()
        .get_or_insert_with(FxHashMap::default)
        .insert((span.start, span.end), text);
}

pub(crate) fn guard_text(span: oxc_span::Span) -> String {
    GUARDS
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|guards| guards.get(&(span.start, span.end)).cloned())
        .unwrap_or_default()
}

pub(crate) fn register(description: String) -> u64 {
    let mut sites = SITES.lock().unwrap();
    sites.push(Site {
        description,
        hits: 0,
    });
    sites.len() as u64 - 1
}

pub(crate) fn hit(site: u64) {
    if let Some(site) = SITES.lock().unwrap().get_mut(site as usize) {
        site.hits += 1;
    }
}

pub(crate) fn report() {
    if !enabled() {
        return;
    }
    let mut sites = SITES.lock().unwrap();
    sites.retain(|site| site.hits > 0);
    sites.sort_by_key(|site| std::cmp::Reverse(site.hits));
    let total: u64 = sites.iter().map(|site| site.hits).sum();
    eprintln!("js jit bailouts: {total} transfers into continuations");
    for site in sites.iter().take(40) {
        eprintln!("  {:>10}  {}", site.hits, site.description);
    }
}

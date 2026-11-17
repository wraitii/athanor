use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use super::*;

static SITES: Mutex<Vec<String>> = Mutex::new(Vec::new());

pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("JS_JIT_HEAP_STATS").as_deref() == Ok("1"))
}

pub(crate) fn register(name: String) -> u64 {
    let mut sites = SITES.lock().unwrap();
    if let Some(index) = sites.iter().position(|site| site == &name) {
        return index as u64 + 1;
    }
    sites.push(name);
    sites.len() as u64
}

pub(crate) fn location(source: (usize, usize), span: oxc_span::Span) -> String {
    if source.0 == 0 {
        return format!("runtime {}..{}", span.start, span.end);
    }
    let filename = unsafe {
        std::str::from_utf8_unchecked(std::slice::from_raw_parts(source.0 as *const u8, source.1))
    };
    format!("{filename} {}..{}", span.start, span.end)
}

#[derive(Default)]
struct Counts {
    allocated: u64,
    queued: u64,
    cascading: u64,
    disposed: u64,
    reused: u64,
}

#[derive(Default)]
pub(super) struct Stats {
    pub(super) site: u64,
    pub(super) disposing: bool,
    allocations: FxHashMap<u64, (u64, &'static str)>,
    counts: BTreeMap<(u64, &'static str), Counts>,
    pub(super) drains: u64,
    pub(super) empty_drains: u64,
    pub(super) work: u64,
    pub(super) edges: u64,
    pub(super) heap_edges: u64,
}

impl Stats {
    pub(super) fn allocate(&mut self, id: u64, kind: &'static str) {
        let key = (self.site, kind);
        self.allocations.insert(id, key);
        self.counts.entry(key).or_default().allocated += 1;
    }

    pub(super) fn queue(&mut self, id: u64) {
        let Some(key) = self.allocations.get(&id) else {
            return;
        };
        let counts = self.counts.get_mut(key).unwrap();
        counts.queued += 1;
        counts.cascading += u64::from(self.disposing);
    }

    pub(super) fn dispose(&mut self, id: u64, reused: bool) {
        let Some(key) = self.allocations.remove(&id) else {
            return;
        };
        let counts = self.counts.get_mut(&key).unwrap();
        counts.disposed += 1;
        counts.reused += u64::from(reused);
    }
}

impl Heap {
    pub(crate) fn allocation_site(&self, site: u64) {
        if let Some(stats) = &mut self.0.borrow_mut().stats {
            stats.site = site;
        }
    }

    pub(crate) fn report_stats(&self) {
        let data = self.0.borrow();
        let Some(stats) = &data.stats else { return };
        eprintln!(
            "heap stats: drains={} empty={} work={} edges={} heap_edges={} pending={}",
            stats.drains,
            stats.empty_drains,
            stats.work,
            stats.edges,
            stats.heap_edges,
            data.pending.len()
        );
        eprintln!("  allocated queued cascading disposed reused remaining kind site");
        let sites = SITES.lock().unwrap();
        let mut rows: Vec<_> = stats.counts.iter().collect();
        rows.sort_by_key(|(_, counts)| std::cmp::Reverse(counts.allocated));
        for (&(site, kind), counts) in rows {
            let name = site
                .checked_sub(1)
                .and_then(|index| sites.get(index as usize));
            eprintln!(
                "  {} {} {} {} {} {} {} {}",
                counts.allocated,
                counts.queued,
                counts.cascading,
                counts.disposed,
                counts.reused,
                counts.allocated - counts.disposed,
                kind,
                name.map_or("runtime/unattributed", String::as_str)
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disposal_counts_distinguish_roots_and_cascading_edges() {
        let heap = Heap::default();
        heap.0.borrow_mut().stats = Some(Box::default());
        let string = heap.string("leaf").unwrap();
        let array = heap.array(&[string]).unwrap();
        let root = heap.object().unwrap();
        assert!(heap.set_property(root, "array", array));
        heap.release(string);
        heap.release(array);
        heap.release(root);
        heap.drain(100);
        let data = heap.0.borrow();
        let stats = data.stats.as_ref().unwrap();
        assert_eq!(
            stats.counts.values().map(|row| row.allocated).sum::<u64>(),
            3
        );
        assert_eq!(stats.counts.values().map(|row| row.queued).sum::<u64>(), 3);
        assert_eq!(
            stats.counts.values().map(|row| row.cascading).sum::<u64>(),
            2
        );
        assert_eq!(
            stats.counts.values().map(|row| row.disposed).sum::<u64>(),
            3
        );
        assert!(stats.allocations.is_empty());
        assert_eq!((stats.drains, stats.empty_drains, stats.work), (4, 3, 5));
        assert_eq!((stats.edges, stats.heap_edges), (2, 2));
    }

    #[test]
    fn object_reuse_counts_a_disposal_and_a_new_allocation() {
        let heap = Heap::default();
        heap.0.borrow_mut().stats = Some(Box::default());
        let first = heap.object().unwrap();
        heap.release(first);
        let second = heap.object().unwrap();
        assert_eq!(first, second);
        let data = heap.0.borrow();
        let stats = data.stats.as_ref().unwrap();
        let row = stats.counts.get(&(0, "object")).unwrap();
        assert_eq!(
            (row.allocated, row.queued, row.disposed, row.reused),
            (2, 1, 1, 1)
        );
        assert_eq!(stats.allocations.len(), 1);
    }
}

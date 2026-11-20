use std::fmt::{self, Write};
use std::sync::{Arc, Mutex};

use super::semantic;

const ALL_PASSES: &str = "all";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompileOptions {
    pub filter: Option<String>,
    pub semantic_ir: bool,
    pub verbose: bool,
    pub after: Option<String>,
    pub diff: bool,
    pub planner: bool,
    pub reports: Option<ReportSink>,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            filter: None,
            semantic_ir: true,
            verbose: false,
            after: None,
            diff: false,
            planner: false,
            reports: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Report {
    pub name: String,
    pub label: String,
    pub direct: bool,
    pub semantic: Option<String>,
    pub passes: Vec<(String, String)>,
    pub planner: Option<String>,
    #[cfg(feature = "wgsl")]
    pub gpu: Option<Result<super::naga::GpuKernel, String>>,
}

#[derive(Clone, Default)]
pub struct ReportSink(Arc<Mutex<Vec<Report>>>);

impl ReportSink {
    pub fn take(&self) -> Vec<Report> {
        std::mem::take(&mut self.0.lock().unwrap())
    }
}

impl PartialEq for ReportSink {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ReportSink {}

impl fmt::Debug for ReportSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ReportSink")
    }
}

pub(super) fn selects(pass: Option<&str>, name: &str) -> bool {
    pass.is_some_and(|pass| {
        pass == ALL_PASSES
            || pass == name
            || name.split_once('/').is_some_and(|(stage, _)| stage == pass)
    })
}

#[derive(Clone, Default)]
pub(super) struct Inspection {
    label: Option<String>,
    semantic: Option<String>,
    after: Vec<(String, String)>,
    planner: Option<String>,
    #[cfg(feature = "wgsl")]
    gpu: Option<Result<super::naga::GpuKernel, String>>,
}

pub(super) fn graph_text(graph: &semantic::Graph, verbose: bool) -> String {
    let facts = semantic::Facts::compute(graph);
    semantic_text(graph.listing(&facts).verbose(verbose).lines())
}

fn semantic_text(lines: Vec<semantic::ListingLine>) -> String {
    let mut text = String::new();
    for (id, _, line) in lines {
        if id == 0 {
            writeln!(text, "{line}").unwrap();
        } else {
            writeln!(text, "{line}  @{id:04x}").unwrap();
        }
    }
    text
}

pub(super) struct PassInspection {
    pass: Option<String>,
    verbose: bool,
    snapshots: Vec<(String, String)>,
    #[cfg(feature = "wgsl")]
    gpu: Option<Result<super::naga::GpuKernel, String>>,
}

impl PassInspection {
    pub(super) fn new(options: Option<&CompileOptions>) -> Self {
        Self {
            pass: options
                .filter(|options| options.semantic_ir)
                .and_then(|options| options.after.clone()),
            verbose: options.is_some_and(|options| options.verbose),
            snapshots: Vec::new(),
            #[cfg(feature = "wgsl")]
            gpu: None,
        }
    }

    pub(super) fn observe(&mut self, name: &str, graph: &semantic::Graph) {
        if selects(self.pass.as_deref(), name) {
            self.snapshots.push((name.to_owned(), graph_text(graph, self.verbose)));
        }
        #[cfg(feature = "wgsl")]
        if name == "inline" && selects(self.pass.as_deref(), "wgsl") {
            let kernel = super::naga::compile_kernel(graph);
            let text = match &kernel {
                Ok(kernel) => kernel.to_string(),
                Err(error) => format!("// wgsl unsupported: {error}\n"),
            };
            self.snapshots.push(("wgsl".to_owned(), text));
            self.gpu = Some(kernel);
        }
    }

    pub(super) fn lower_runtime(
        &mut self,
        state: &mut semantic::FunctionState,
        baseline: bool,
    ) -> Result<(), String> {
        state.lower_runtime_observed(baseline, &mut |name, graph| self.observe(name, graph))?;
        let Some(pass) = self.pass.as_deref() else {
            return Ok(());
        };
        if pass == ALL_PASSES || (cfg!(feature = "wgsl") && pass == "wgsl") {
            return Ok(());
        }
        let passes = semantic::SPECIALIZATION_PASSES
            .iter()
            .chain(semantic::RUNTIME_PASSES)
            .chain(semantic::LOW_LEVEL_PASSES)
            .copied()
            .collect::<Vec<_>>();
        let stage = pass.split_once('/').map_or(pass, |(stage, _)| stage);
        if !passes.contains(&stage) {
            return Err(format!(
                "unknown pass {pass}; available passes: {}",
                passes.join(", ")
            ));
        }
        Ok(())
    }
}

impl Inspection {
    pub(super) fn new(
        options: Option<&CompileOptions>,
        graph: &semantic::FunctionState,
        repository: &semantic::specialization::calls::Repository,
        source: usize,
        inline_multiplier: usize,
    ) -> Self {
        let Some(options) = options else {
            return Self::default();
        };
        Self {
            semantic: options
                .semantic_ir
                .then(|| graph_text(&graph.graph, options.verbose)),
            planner: options
                .planner
                .then(|| graph.plan_report(repository, source, inline_multiplier)),
            ..Self::default()
        }
    }

    pub(super) fn from_graph(options: Option<&CompileOptions>, graph: &semantic::Graph) -> Self {
        let Some(options) = options else {
            return Self::default();
        };
        Self {
            semantic: options
                .semantic_ir
                .then(|| graph_text(graph, options.verbose)),
            ..Self::default()
        }
    }

    pub(super) fn named(mut self, label: &str) -> Self {
        self.label = Some(label.to_owned());
        self
    }

    pub(super) fn after(mut self, passes: PassInspection) -> Self {
        self.after = passes.snapshots;
        #[cfg(feature = "wgsl")]
        {
            self.gpu = passes.gpu;
        }
        self
    }

    pub(super) fn lowered(mut self, snapshots: Vec<(String, String)>) -> Self {
        self.after.extend(snapshots);
        self
    }

    pub(super) fn print(&self, name: &str, direct: bool, options: &CompileOptions) {
        let label = self.label.as_deref().unwrap_or(name);
        if let Some(reports) = &options.reports {
            reports.0.lock().unwrap().push(Report {
                name: name.to_owned(),
                label: label.to_owned(),
                direct,
                semantic: self.semantic.clone(),
                passes: self.after.clone(),
                planner: self.planner.clone(),
                #[cfg(feature = "wgsl")]
                gpu: self.gpu.clone(),
            });
            return;
        }
        if direct && (!self.after.is_empty() || self.semantic.is_some()) {
            println!("[{name}] direct baseline code");
        }
        if options.diff {
            self.print_diffs(name, label);
        } else {
            for (pass, semantic) in &self.after {
                println!("[{name}] semantic IR {label} after {pass}\n{semantic}");
            }
        }
        if let (true, Some(semantic)) = (self.after.is_empty(), &self.semantic) {
            println!("[{name}] semantic IR {label}\n{semantic}");
        }
        if let Some(planner) = &self.planner {
            println!("[{name}] planner {label}\n{planner}");
        }
    }
}

impl Inspection {
    fn print_diffs(&self, name: &str, label: &str) {
        let mut previous: Option<String> = None;
        let mut unchanged = Vec::new();
        let flush = |unchanged: &mut Vec<&str>| {
            if !unchanged.is_empty() {
                println!("[{name}] unchanged {label}: {}\n", unchanged.join(", "));
                unchanged.clear();
            }
        };
        for (pass, text) in &self.after {
            if pass == "wgsl" {
                flush(&mut unchanged);
                println!("[{name}] semantic IR {label} after {pass}\n{text}");
                continue;
            }
            let text = strip_locations(text);
            let Some(before) = previous.replace(text.clone()) else {
                println!("[{name}] semantic IR {label} after {pass}\n{text}");
                continue;
            };
            let diff = listing_diff(&before, &text);
            if diff.is_empty() {
                unchanged.push(pass.as_str());
                continue;
            }
            flush(&mut unchanged);
            println!("[{name}] diff {label} after {pass}\n{diff}");
        }
        flush(&mut unchanged);
    }
}

fn strip_locations(text: &str) -> String {
    let mut stripped = String::with_capacity(text.len());
    for line in text.lines() {
        let line = match line.rsplit_once("  @") {
            Some((code, id)) if id.chars().all(|c| c.is_ascii_hexdigit()) => code.trim_end(),
            _ => line,
        };
        stripped.push_str(line);
        stripped.push('\n');
    }
    stripped
}

fn listing_diff(before: &str, after: &str) -> String {
    use similar::{ChangeTag, TextDiff};
    let diff = TextDiff::from_lines(before, after);
    let lines: Vec<&str> = after.lines().collect();
    let mut text = String::new();
    for group in diff.grouped_ops(2) {
        let start = group
            .iter()
            .find(|op| op.tag() != similar::DiffTag::Equal)
            .map_or(0, |op| op.new_range().start);
        writeln!(text, "@@ {} @@", enclosing_block(&lines, start)).unwrap();
        for op in &group {
            for change in diff.iter_changes(op) {
                let sign = match change.tag() {
                    ChangeTag::Equal => ' ',
                    ChangeTag::Delete => '-',
                    ChangeTag::Insert => '+',
                };
                write!(text, "{sign}{}", change.value()).unwrap();
                if change.missing_newline() {
                    text.push('\n');
                }
            }
        }
    }
    text
}

fn enclosing_block<'a>(lines: &[&'a str], index: usize) -> &'a str {
    lines[..(index + 1).min(lines.len())]
        .iter()
        .rev()
        .find(|line| line.starts_with("block"))
        .and_then(|line| line.split(['(', ':']).next())
        .unwrap_or("function")
}

impl CompileOptions {
    pub(super) fn matches(&self, name: &str) -> bool {
        self.filter
            .as_ref()
            .is_none_or(|filter| name.contains(filter))
    }
}

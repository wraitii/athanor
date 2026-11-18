use std::fmt::Write;

use super::semantic;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompileOptions {
    pub filter: Option<String>,
    pub semantic_ir: bool,
    pub after: Option<String>,
    pub planner: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            filter: None,
            semantic_ir: true,
            after: None,
            planner: false,
        }
    }
}

#[derive(Clone, Default)]
pub(super) struct Inspection {
    label: Option<String>,
    semantic: Option<String>,
    after: Option<(String, String)>,
    planner: Option<String>,
}

pub(super) fn graph_text(graph: &semantic::Graph) -> String {
    semantic_text(graph.listing(&semantic::Facts::compute(graph)).lines())
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
    pub(super) snapshot: Option<(String, String)>,
}

impl PassInspection {
    pub(super) fn new(options: Option<&CompileOptions>) -> Self {
        Self {
            pass: options
                .filter(|options| options.semantic_ir)
                .and_then(|options| options.after.clone()),
            snapshot: None,
        }
    }

    pub(super) fn observe(&mut self, name: &str, state: &semantic::FunctionState) {
        if self.pass.as_deref() == Some(name) {
            self.snapshot = Some((name.to_owned(), semantic_text(state.listing_lines())));
        }
    }

    pub(super) fn lower_runtime(
        &mut self,
        state: &mut semantic::FunctionState,
        baseline: bool,
    ) -> Result<(), String> {
        state.lower_runtime_observed(baseline, &mut |name, state| self.observe(name, state))?;
        let Some(pass) = self.pass.as_deref() else {
            return Ok(());
        };
        let passes = semantic::SPECIALIZATION_PASSES
            .iter()
            .chain(semantic::RUNTIME_PASSES)
            .chain(semantic::LOW_LEVEL_PASSES)
            .copied()
            .collect::<Vec<_>>();
        if !passes.contains(&pass) {
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
                .then(|| semantic_text(graph.listing_lines())),
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
            semantic: options.semantic_ir.then(|| graph_text(graph)),
            ..Self::default()
        }
    }

    pub(super) fn named(mut self, label: &str) -> Self {
        self.label = Some(label.to_owned());
        self
    }

    pub(super) fn after(mut self, snapshot: Option<(String, String)>) -> Self {
        self.after = snapshot;
        self
    }

    pub(super) fn lowered(mut self, snapshot: Option<(String, String)>) -> Self {
        if snapshot.is_some() {
            self.after = snapshot;
        }
        self
    }

    pub(super) fn print(&self, name: &str, direct: bool) {
        let label = self.label.as_deref().unwrap_or(name);
        if direct && (self.after.is_some() || self.semantic.is_some()) {
            println!("[{name}] direct baseline code");
        }
        match (&self.after, &self.semantic) {
            (Some((pass, semantic)), _) => {
                println!("[{name}] semantic IR {label} after {pass}\n{semantic}")
            }
            (None, Some(semantic)) => println!("[{name}] semantic IR {label}\n{semantic}"),
            (None, None) => {}
        }
        if let Some(planner) = &self.planner {
            println!("[{name}] planner {label}\n{planner}");
        }
    }
}

impl CompileOptions {
    pub(super) fn matches(&self, name: &str) -> bool {
        self.filter
            .as_ref()
            .is_none_or(|filter| name.contains(filter))
    }
}

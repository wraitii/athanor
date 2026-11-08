use super::super::specialize::{ConditionalCall, ConditionalPlan};
use super::versioning::Version;
use super::*;

struct RegionPlan {
    body: HashSet<Block>,
    entry: Option<(Block, Block)>,
    requirements: Vec<Requirement>,
    calls: Vec<ConditionalCall>,
    identities: Vec<(String, cranelift_module::FuncId)>,
    reads: Vec<Inst>,
    checks: Vec<Value>,
    guards: Vec<Block>,
    exits: Vec<Block>,
    typed: Vec<(Value, Option<TypedKind>)>,
    bounds: Vec<(Value, Value)>,
    typed_reads: Vec<(Inst, TypedRead)>,
    children: Vec<RegionPlan>,
}

impl RegionPlan {
    fn report(&self, graph: &Graph, depth: usize, children: bool) -> String {
        use std::fmt::Write;

        let indent = "  ".repeat(depth);
        let mut out = String::new();
        match self.entry {
            Some((preheader, header)) => {
                writeln!(out, "{indent}region loop at {header} preheader={preheader}").unwrap()
            }
            None => writeln!(out, "{indent}region function").unwrap(),
        }
        let mut blocks: Vec<_> = self.body.iter().copied().collect();
        blocks.sort();
        writeln!(
            out,
            "{indent}  blocks: {}",
            blocks
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )
        .unwrap();
        writeln!(out, "{indent}  speculative: {}", self.speculative()).unwrap();
        for requirement in &self.requirements {
            writeln!(
                out,
                "{indent}  native {}: {}",
                requirement.call, requirement.native
            )
            .unwrap();
        }
        for call in &self.calls {
            writeln!(
                out,
                "{indent}  closed call {}: {} effects=0x{:04x}",
                call.inst, call.name, call.flags
            )
            .unwrap();
        }
        for (name, code) in &self.identities {
            writeln!(out, "{indent}  identity {name}={code}").unwrap();
        }
        for &value in &self.checks {
            let origin = graph
                .defining_op(value)
                .map(|op| graph.op_text(op))
                .unwrap_or_else(|| "block parameter".into());
            writeln!(out, "{indent}  number assumption {value}: {origin}").unwrap();
        }
        for &inst in &self.reads {
            writeln!(
                out,
                "{indent}  probe {inst}: {}",
                graph.op_text(&graph.insts[inst].op)
            )
            .unwrap();
        }
        for &block in &self.guards {
            writeln!(out, "{indent}  hoisted guard {block}").unwrap();
        }
        for &block in &self.exits {
            writeln!(out, "{indent}  fallback exit {block}").unwrap();
        }
        for (value, element) in &self.typed {
            writeln!(out, "{indent}  typed receiver {value}: {element:?}").unwrap();
        }
        for (object, bound) in &self.bounds {
            writeln!(out, "{indent}  length check {bound} <= length({object})").unwrap();
        }
        for (inst, read) in &self.typed_reads {
            writeln!(
                out,
                "{indent}  typed read {inst}: element={:?} bounded={}",
                read.element, read.bounded
            )
            .unwrap();
        }
        if children {
            for child in &self.children {
                out.push_str(&child.report(graph, depth + 1, true));
            }
        }
        out
    }

    fn speculative(&self) -> bool {
        !self.identities.is_empty()
            || !self.requirements.is_empty()
            || !self.checks.is_empty()
            || !self.guards.is_empty()
            || !self.typed.is_empty()
    }

    fn covers(&self, child: &Self, facts: &Facts) -> bool {
        child
            .checks
            .iter()
            .all(|&value| facts.of(value).within(NUMERIC))
            && child
                .calls
                .iter()
                .all(|call| self.calls.iter().any(|parent| parent == call))
            && child.guards.iter().all(|guard| self.guards.contains(guard))
            && child.requirements.iter().all(|requirement| {
                self.requirements
                    .iter()
                    .any(|parent| parent.call == requirement.call)
            })
            && child.typed_reads.iter().all(|(inst, read)| {
                self.typed_reads.iter().any(|(parent, specialized)| {
                    parent == inst
                        && specialized.element == read.element
                        && (!read.bounded || specialized.bounded)
                })
            })
    }

    fn merge(&mut self, child: Self, facts: &Facts) {
        if self.covers(&child, facts) {
            for descendant in child.children {
                self.merge(descendant, facts);
            }
        } else {
            self.children.push(child);
        }
    }

    fn remap(&mut self, version: &Version) {
        let block = |old: Block| version.blocks.get(&old).copied().unwrap_or(old);
        self.body = self.body.iter().map(|&old| block(old)).collect();
        self.entry = self
            .entry
            .map(|(preheader, header)| (block(preheader), block(header)));
        for call in &mut self.calls {
            call.inst = version.insts.get(&call.inst).copied().unwrap_or(call.inst);
            call.callee = version
                .values
                .get(&call.callee)
                .copied()
                .unwrap_or(call.callee);
        }
        for child in &mut self.children {
            child.remap(version);
        }
    }

    fn emit(
        mut self,
        graph: &mut Graph,
        fallback: Option<usize>,
        regions: &mut Vec<HashSet<Block>>,
        notes: &mut Vec<String>,
        report_regions: bool,
    ) -> Result<()> {
        let cfg = Cfg::new(graph);
        let live: HashSet<_> = cfg.rpo.iter().copied().collect();
        self.body.retain(|block| live.contains(block));
        if self.body.is_empty() {
            return Ok(());
        }
        // Earlier copies can introduce exit phis or move a child's probes to parent entry.
        let (mut current, _) = plan_region(graph, self.body, self.entry, &self.calls)?;
        current.children = self.children;
        self = current;
        if report_regions {
            notes.push(format!(
                "region emission:\n{}",
                self.report(graph, 1, false)
            ));
        }
        let label = self.entry.map_or_else(
            || "function".to_owned(),
            |(_, header)| format!("loop at {header}"),
        );
        if self.speculative() {
            let blocks: Vec<_> = Cfg::new(graph)
                .rpo
                .into_iter()
                .filter(|block| self.body.contains(block))
                .collect();
            let fallback = fallback.filter(|_| self.entry.is_none());
            let speculation = Speculation {
                fallback,
                requirements: &self.requirements,
                reads: &self.reads,
                checks: &self.checks,
                guards: &self.guards,
                exits: &self.exits,
                typed: &self.typed,
                bounds: &self.bounds,
                typed_reads: &self.typed_reads,
                code_guards: &self.identities,
                closed_calls: &self.calls,
                ..Default::default()
            };
            let version = apply_version(graph, &blocks, self.entry, &speculation);
            graph.verify()?;
            if let Some(previous) = fallback {
                notes.push(format!(
                    "function: entry guard failure reuses {previous:#x}"
                ));
            }
            if let Some(reason) = closure_failure(
                graph,
                &version.body,
                &Guarded::new(&self.requirements).with_code(&self.identities),
            ) {
                return Err(format!("{label}: planned copy {reason}"));
            }
            notes.push(format!(
                "{label}: combined entry guards: {} native calls, {} number assumptions, {} read probes, {} hoisted guards, {} typed receiver assumptions and {} length checks",
                self.requirements.len(), self.checks.len(), self.reads.len(), self.guards.len(), self.typed.len(), self.bounds.len()
            ));
            if !self.calls.is_empty() {
                notes.push(format!(
                    "{label}: {} global code checks prove {} read-only call(s)",
                    self.identities.len(),
                    self.calls.len()
                ));
            }
            for child in &mut self.children {
                child.remap(&version);
            }
            self.body = version.body;
        } else {
            notes.push(format!("{label}: closed without guards"));
        }
        for child in self.children {
            let first = regions.len();
            child.emit(graph, None, regions, notes, report_regions)?;
            for region in &regions[first..] {
                self.body.extend(region.iter().copied());
            }
        }
        regions.push(self.body);
        Ok(())
    }
}

fn plan_region(
    graph: &Graph,
    body: HashSet<Block>,
    entry: Option<(Block, Block)>,
    conditional: &[ConditionalCall],
) -> Result<(RegionPlan, Facts)> {
    let blocks: Vec<_> = Cfg::new(graph)
        .rpo
        .into_iter()
        .filter(|block| body.contains(block))
        .collect();
    let exits: Vec<_> = blocks
        .iter()
        .copied()
        .filter(|&block| guard_fallback(graph, block).is_some())
        .collect();
    let fallbacks: HashSet<_> = exits
        .iter()
        .filter_map(|&block| guard_fallback(graph, block))
        .collect();
    let written: HashSet<_> = body.difference(&fallbacks).copied().collect();
    let requirements = requirements(graph, &blocks);
    let calls: Vec<_> = conditional
        .iter()
        .filter(|call| {
            written.contains(&graph.insts[call.inst].block)
                && body.contains(&graph.def_block(call.callee))
                && matches!(graph.insts[call.inst].op, Op::Call { callee, .. } if callee == call.callee)
                && matches!(graph.defining_op(call.callee), Some(Op::GlobalRead(name)) if name == &call.name)
        })
        .cloned()
        .collect();
    let mut identities = HashMap::new();
    for call in &calls {
        for (name, code) in &call.identities {
            if identities
                .insert(name.clone(), *code)
                .is_some_and(|other| other != *code)
            {
                return Err(format!("conflicting code identities for {name}"));
            }
        }
    }
    let mut identities: Vec<_> = identities.into_iter().collect();
    identities.sort_by(|a, b| a.0.cmp(&b.0));
    let guarded = Guarded::new(&requirements).with_code(&identities);
    let mut planning = graph.clone();
    for call in &calls {
        planning
            .closed_calls
            .insert(call.inst, (call.callee, call.flags));
    }
    for requirement in &requirements {
        let Op::Call { args, .. } = &planning.insts[requirement.call].op else {
            unreachable!()
        };
        planning.insts[requirement.call].op = Op::CallNative(requirement.native, args.clone());
    }
    if let Some(reason) = closure_failure(&planning, &written, &guarded) {
        return Err(reason);
    }
    let mut uses = planning.use_counts();
    let mut unused: Vec<_> = requirements
        .iter()
        .filter_map(|requirement| match graph.insts[requirement.call].op {
            Op::Call { callee, .. } => Some(callee),
            _ => None,
        })
        .collect();
    while let Some(value) = unused.pop() {
        if uses[value] != 0 || !body.contains(&planning.def_block(value)) {
            continue;
        }
        let ValueDef::Result(inst) = planning.values[value] else {
            continue;
        };
        match planning.insts[inst].op {
            Op::Get(object, _) => {
                uses[object] -= 1;
                unused.push(object);
            }
            Op::GlobalRead(_) => {}
            _ => continue,
        }
        planning.remove_inst(inst);
    }
    let reads = probe_reads(&planning, &body, &written, &blocks, entry.is_none());
    let available: HashSet<_> = reads
        .iter()
        .filter_map(|&inst| planning.insts[inst].result)
        .collect();
    let checks = entry_checks(&planning, &body, &blocks, &reads);
    let guards = invariant_guards(&planning, &body, &blocks, &reads);
    // Candidate facts must not narrow the original graph before its guards execute.
    let assumptions = checks.iter().map(|&value| (value, Types::NUMBER)).collect();
    let facts = Facts::assuming(&planning, &assumptions);
    let objects = typed_objects(&planning, &body, &available, &facts);
    let typed: Vec<_> = objects
        .iter()
        .map(|&object| (object, element_guess(&planning, object)))
        .collect();
    let bounded = bounded_reads(
        &planning,
        &Cfg::new(&planning),
        &body,
        &objects,
        &available,
        &facts,
    );
    let mut bounds: Vec<_> = bounded.values().copied().collect();
    bounds.sort();
    bounds.dedup();
    let mut typed_reads = Vec::new();
    for &block in &blocks {
        for &inst in &planning.blocks[block].insts {
            if let Op::GetIndex(object, key) | Op::TypedIndex(object, key, _) =
                planning.insts[inst].op
                && let Some(&(_, element)) =
                    typed.iter().find(|(candidate, _)| *candidate == object)
                && facts.of(key).within(NUMERIC)
                && !reads.contains(&inst)
            {
                let previous = match planning.insts[inst].op {
                    Op::TypedIndex(_, _, read) => Some(read),
                    _ => None,
                };
                typed_reads.push((
                    inst,
                    TypedRead {
                        element: previous.and_then(|read| read.element).or(element),
                        bounded: previous.is_some_and(|read| read.bounded)
                            || bounded.contains_key(&inst),
                    },
                ));
            }
        }
    }
    if identities.is_empty()
        && requirements.is_empty()
        && checks.is_empty()
        && guards.is_empty()
        && typed.is_empty()
        && closure_failure(&planning, &body, &guarded).is_some()
    {
        return Err("no invariant entry guards for reentrant fallbacks".into());
    }
    Ok((
        RegionPlan {
            body,
            entry,
            requirements,
            calls,
            identities,
            reads,
            checks,
            guards,
            exits,
            typed,
            bounds,
            typed_reads,
            children: Vec::new(),
        },
        facts,
    ))
}

fn plan(graph: &Graph, conditional: &ConditionalPlan, notes: &mut Vec<String>) -> Vec<RegionPlan> {
    let cfg = Cfg::new(graph);
    let mut candidates: Vec<_> = natural_loops(graph, &cfg)
        .into_iter()
        .filter_map(|(header, body)| {
            if graph.cold.contains(&header) {
                return None;
            }
            let preheader = preheader(graph, &cfg, header, &body)?;
            Some((body, Some((preheader, header))))
        })
        .collect();
    candidates.push((cfg.rpo.iter().copied().collect(), None));
    let mut plans: Vec<RegionPlan> = Vec::new();
    for (body, entry) in candidates {
        let (mut parent, facts) = match plan_region(graph, body, entry, &conditional.calls) {
            Ok(plan) => plan,
            Err(reason) => {
                let label = entry.map_or_else(
                    || "function".to_owned(),
                    |(_, header)| format!("loop at {header}"),
                );
                notes.push(format!("{label}: not versioned: {reason}"));
                continue;
            }
        };
        let mut remaining = Vec::new();
        for child in plans {
            if child.body.is_subset(&parent.body) {
                parent.merge(child, &facts);
            } else {
                remaining.push(child);
            }
        }
        remaining.push(parent);
        plans = remaining;
    }
    plans
}

#[cfg(test)]
pub(in crate::jit::semantic) fn version(
    graph: &mut Graph,
    conditional: &ConditionalPlan,
    fallback: Option<usize>,
    notes: &mut Vec<String>,
) -> Result<Vec<HashSet<Block>>> {
    version_with_report(graph, conditional, fallback, notes, false)
}

pub(in crate::jit::semantic) fn version_with_report(
    graph: &mut Graph,
    conditional: &ConditionalPlan,
    fallback: Option<usize>,
    notes: &mut Vec<String>,
    report_regions: bool,
) -> Result<Vec<HashSet<Block>>> {
    let plans = plan(graph, conditional, notes);
    if report_regions {
        let mut report = String::from("region plan before versioning:\n");
        for plan in &plans {
            report.push_str(&plan.report(graph, 1, true));
        }
        notes.push(report);
    }
    let mut regions = Vec::new();
    for plan in plans {
        plan.emit(graph, fallback, &mut regions, notes, report_regions)?;
    }
    Ok(regions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxc_allocator::Allocator;
    use oxc_ast::ast::Statement;
    use oxc_parser::Parser;
    use oxc_span::SourceType;

    fn graph(source: &str) -> Graph {
        let allocator = Allocator::default();
        let parsed = Parser::new(&allocator, source, SourceType::script()).parse();
        assert!(parsed.diagnostics.is_empty());
        let Statement::FunctionDeclaration(function) = &parsed.program.body[0] else {
            panic!()
        };
        let semantic = oxc_semantic::SemanticBuilder::new_compiler()
            .build(&parsed.program)
            .semantic;
        let graph =
            super::super::super::build::lower(function, semantic.scoping(), &HashSet::new())
                .unwrap();
        let mut graph = super::super::super::canonicalize(graph, true)
            .unwrap()
            .graph;
        super::super::super::comparisons::expose(&mut graph);
        graph
    }

    #[test]
    fn versioning_keeps_feedback_specialized_typed_reads() {
        use crate::heap::TypedKind;
        use crate::jit::planner::{ReadKind, ReadType, Site};

        let graph = graph(
            "function f(xs,n){let sum=0;for(let i=0;i<n;++i)sum+=Math.floor(xs[i]);return sum;}",
        );
        let data = graph
            .insts
            .values()
            .find(|data| matches!(data.op, Op::GetIndex(..)))
            .unwrap();
        let site = Site {
            source: 7,
            start: data.span.start,
            end: data.span.end,
        };
        let mut repository = super::super::super::specialize::Repository::default();
        repository.read_feedback.insert(
            (site, ReadKind::Receiver),
            ReadType::Typed(TypedKind::Uint16),
        );
        let canonical = super::super::super::Canonical::from_graph(graph);
        let analysis =
            super::super::super::optimize_canonical(canonical, true, Some((&repository, 7)))
                .unwrap();
        let graph = &analysis.graph;
        let reads: Vec<_> = Cfg::new(graph)
            .rpo
            .iter()
            .filter(|block| !graph.cold.contains(block))
            .flat_map(|&block| graph.blocks[block].insts.iter())
            .filter_map(|&inst| match graph.insts[inst].op {
                Op::TypedIndex(_, _, read) => Some(read.element),
                _ => None,
            })
            .collect();
        assert_eq!(reads, [Some(TypedKind::Uint16)]);
    }

    #[test]
    fn nested_loop_assumptions_merge_to_function_entry_before_copying() {
        let graph = graph(
            "function f(grid,width){let sum=0;for(let y=0;y<2;++y){const offset=y*width;for(let x=0;x<2;++x)sum+=Math.sqrt(grid[offset+x]);}return sum;}",
        );
        let plans = plan(&graph, &ConditionalPlan::default(), &mut Vec::new());
        assert_eq!(plans.len(), 1);
        let root = &plans[0];
        assert!(root.entry.is_none());
        assert!(root.children.is_empty());
        assert!(!root.requirements.is_empty());
        assert!(!root.checks.is_empty());
        assert!(root.typed_reads.is_empty());
        assert_eq!(root.body.len(), Cfg::new(&graph).rpo.len());
        graph.verify().unwrap();
    }

    #[test]
    fn previous_entry_reuses_the_fast_body_without_a_generic_copy() {
        let mut graph = graph(
            "function f(xs,n){let sum=0;for(let i=0;i<n;++i)sum+=Math.floor(xs[i]);return sum;}",
        );
        let original = graph.entry;
        let loops = natural_loops(&graph, &Cfg::new(&graph)).len();
        let regions = version(
            &mut graph,
            &ConditionalPlan::default(),
            Some(0x1234),
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(regions.len(), 1);
        assert!(regions[0].contains(&original));
        let cfg = Cfg::new(&graph);
        assert_eq!(natural_loops(&graph, &cfg).len(), loops);
        assert_eq!(
            cfg.rpo
                .iter()
                .filter(|&&block| {
                    matches!(
                        graph.blocks[block].terminator.as_ref().unwrap().kind,
                        TerminatorKind::Fallback(0x1234)
                    )
                })
                .count(),
            1
        );
        assert!(!cfg.rpo.iter().any(|&block| {
            graph.blocks[block]
                .insts
                .iter()
                .any(|&inst| matches!(graph.insts[inst].op, Op::Call { .. }))
        }));
        graph.verify().unwrap();
    }

    #[test]
    fn loop_versions_keep_their_local_fallback() {
        let mut graph = graph(
            "function f(g,xs){g();let sum=0;for(let i=0;i<2;++i)sum+=Math.floor(xs[i]);return sum;}",
        );
        let regions = version(
            &mut graph,
            &ConditionalPlan::default(),
            Some(0x1234),
            &mut Vec::new(),
        )
        .unwrap();
        assert!(!regions.is_empty());
        assert!(Cfg::new(&graph).rpo.iter().all(|&block| {
            !matches!(
                graph.blocks[block].terminator.as_ref().unwrap().kind,
                TerminatorKind::Fallback(_)
            )
        }));
        graph.verify().unwrap();
    }

    #[test]
    fn unobserved_rows_do_not_speculate_on_typed_storage() {
        let graph = graph(
            "function f(rows){let sum=0;for(let y=0;y<2;++y){const row=rows[y];for(let x=0;x<2;++x)sum+=row[x];}return sum;}",
        );
        let plans = plan(&graph, &ConditionalPlan::default(), &mut Vec::new());
        assert_eq!(plans.len(), 1);
        let root = &plans[0];
        assert!(root.entry.is_none());
        assert!(root.children.is_empty());
        assert!(root.typed.is_empty());
        assert!(root.typed_reads.is_empty());
    }

    #[test]
    fn versioning_shares_repeated_typed_receiver_guards() {
        use crate::jit::planner::{ReadKind, ReadType, Site};

        let graph = graph(
            "function f(xs,n){unknown();let sum=0;for(let i=0;i<n;++i)sum+=xs[i]+xs[i+1];return sum;}",
        );
        let mut repository = super::super::super::specialize::Repository::default();
        for data in graph
            .insts
            .values()
            .filter(|data| matches!(data.op, Op::GetIndex(..)))
        {
            repository.read_feedback.insert(
                (
                    Site {
                        source: 7,
                        start: data.span.start,
                        end: data.span.end,
                    },
                    ReadKind::Receiver,
                ),
                ReadType::Typed(TypedKind::Uint16),
            );
        }
        let canonical = super::super::super::Canonical::from_graph(graph);
        let analysis =
            super::super::super::optimize_canonical(canonical, true, Some((&repository, 7)))
                .unwrap();
        let graph = &analysis.graph;
        let cfg = Cfg::new(graph);
        let checks = cfg
            .rpo
            .iter()
            .filter(|block| !graph.cold.contains(block))
            .flat_map(|&block| &graph.blocks[block].insts)
            .filter(|&&inst| {
                matches!(
                    graph.insts[inst].op,
                    Op::IsTyped(_, Some(TypedKind::Uint16))
                )
            })
            .count();
        assert_eq!(checks, 1);
        graph.verify().unwrap();
    }

    #[test]
    fn nested_regions_reuse_dominating_typed_guards() {
        let mut graph = graph(
            "function f(rows){const xs=new Float64Array(2);unknown();let sum=0;for(let y=0;y<2;++y){const n=rows[y];for(let x=0;x<n;++x)sum+=xs[x];}return sum;}",
        );
        let xs = graph
            .insts
            .values()
            .find_map(|data| {
                matches!(data.op, Op::Construct { .. })
                    .then_some(data.result)
                    .flatten()
            })
            .unwrap();
        let plans = plan(&graph, &ConditionalPlan::default(), &mut Vec::new());
        assert!(plans[0].typed.iter().any(|&(object, _)| object == xs));
        assert!(
            plans[0]
                .children
                .iter()
                .any(|child| { child.typed.iter().any(|&(object, _)| object == xs) })
        );
        version(
            &mut graph,
            &ConditionalPlan::default(),
            None,
            &mut Vec::new(),
        )
        .unwrap();
        let cfg = Cfg::new(&graph);
        let checks = cfg
            .rpo
            .iter()
            .flat_map(|&block| &graph.blocks[block].insts)
            .filter(|&&inst| matches!(graph.insts[inst].op, Op::IsTyped(object, _) if object == xs))
            .count();
        assert_eq!(checks, 1);
        graph.verify().unwrap();
    }

    #[test]
    fn parent_writes_keep_number_checks_in_the_inner_region() {
        let graph = graph(
            "function f(state){let sum=0;for(let y=0;y<2;++y){state.width=y+1;for(let x=0;x<2;++x)sum+=state.grid[x*state.width];}return sum;}",
        );
        let plans = plan(&graph, &ConditionalPlan::default(), &mut Vec::new());
        assert_eq!(plans.len(), 1);
        let root = &plans[0];
        assert!(root.entry.is_none());
        assert_eq!(root.children.len(), 1);
        let child = &root.children[0];
        assert!(child.checks.iter().any(|&value| {
            matches!(graph.defining_op(value), Some(Op::Get(_, name)) if name == "width")
        }));
        assert!(!root.checks.iter().any(|&value| {
            matches!(graph.defining_op(value), Some(Op::Get(_, name)) if name == "width")
        }));
    }
}

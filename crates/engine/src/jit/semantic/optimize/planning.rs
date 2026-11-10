use super::super::specialize::{ConditionalCall, ConditionalPlan};
use super::super::guards::Fact;
use super::placement::Reach;
use super::versioning::{Canonical, Version};
use super::*;

fn pure_guard(graph: &Graph, block: Block) -> bool {
    let Some(Terminator {
        kind: TerminatorKind::Branch(condition, ..),
        ..
    }) = &graph.blocks[block].terminator
    else {
        return false;
    };
    let condition = match graph.defining_op(*condition) {
        Some(Op::Unary(UnaryOp::Not, inner)) => *inner,
        _ => *condition,
    };
    match graph.defining_op(condition) {
        None => true,
        Some(op) => matches!(
            super::super::guards::implies(op),
            Some((_, Fact::Type(_) | Fact::Typed(_) | Fact::Indexed))
        ),
    }
}

fn consume_assumes(
    graph: &mut Graph,
    blocks: &[Block],
    entry: Option<(Block, Block)>,
    checks: &[(Value, Fact)],
) {
    let scope = blocks.iter().copied().chain(entry.map(|(before, _)| before));
    for block in scope.collect::<Vec<_>>() {
        let consumed: Vec<Inst> = graph.blocks[block]
            .insts
            .iter()
            .copied()
            .filter(|&inst| {
                matches!(graph.insts[inst].op, Op::Assume(value, fact, _)
                    if checks.iter().any(|&(checked, known)| checked == value && known.implies(fact)))
            })
            .collect();
        for inst in consumed {
            graph.detach_inst(inst);
        }
    }
}

#[derive(Default)]
pub(super) struct Candidates {
    facts: Vec<(Value, Fact)>,
    codes: Vec<(String, cranelift_module::FuncId)>,
}

struct RegionPlan {
    body: HashSet<Block>,
    entry: Option<(Block, Block)>,
    requirements: Vec<Requirement>,
    calls: Vec<ConditionalCall>,
    identities: Vec<(String, cranelift_module::FuncId)>,
    reads: Vec<Inst>,
    assumes: Vec<(Value, Fact)>,
    checks: Vec<(Value, Fact)>,
    guards: Vec<Block>,
    exits: Vec<Block>,
    typed_reads: Vec<(Inst, TypedRead)>,
    children: Vec<RegionPlan>,
}

impl RegionPlan {
    fn typed(&self) -> Vec<(Value, Option<TypedKind>)> {
        self.checks
            .iter()
            .filter_map(|&(value, fact)| match fact {
                Fact::Typed(element) => Some((value, element)),
                _ => None,
            })
            .collect()
    }

    fn bounds(&self) -> Vec<(Value, Value)> {
        self.checks
            .iter()
            .filter_map(|&(value, fact)| match fact {
                Fact::Bound(bound) => Some((value, bound)),
                _ => None,
            })
            .collect()
    }

    fn numbers(&self) -> usize {
        self.checks
            .iter()
            .filter(|(_, fact)| matches!(fact, Fact::Type(_)))
            .count()
    }

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
                "{indent}  closed call {}: {} {} {}",
                call.inst, call.name, call.memory, call.control
            )
            .unwrap();
        }
        for (name, code) in &self.identities {
            writeln!(out, "{indent}  identity {name}={code}").unwrap();
        }
        for &(value, fact) in &self.checks {
            let Fact::Type(types) = fact else {
                continue;
            };
            let origin = graph
                .defining_op(value)
                .map(|op| graph.op_text(op))
                .unwrap_or_else(|| "block parameter".into());
            writeln!(
                out,
                "{indent}  number assumption {value} as {types}: {origin}"
            )
            .unwrap();
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
        for (value, element) in &self.typed() {
            writeln!(out, "{indent}  typed receiver {value}: {element:?}").unwrap();
        }
        for (object, bound) in &self.bounds() {
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
    }

    fn covers(&self, child: &Self, facts: &Facts) -> bool {
        child
            .checks
            .iter()
            .filter(|(_, fact)| matches!(fact, Fact::Type(_)))
            .all(|&(value, fact)| Fact::Type(facts.of(value)).implies(fact))
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
        let value = |old: Value| {
            let copied = version.values.get(&old).copied().unwrap_or(old);
            version.aliases.get(&copied).copied().unwrap_or(copied)
        };
        for (subject, fact) in self.assumes.iter_mut().chain(self.checks.iter_mut()) {
            *subject = value(*subject);
            if let Fact::Bound(bound) = fact {
                *bound = value(*bound);
            }
        }
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
        regions: &mut Vec<RegionId>,
        notes: &mut Vec<String>,
        report_regions: bool,
        canonical: &Canonical<'_>,
    ) -> Result<()> {
        let cfg = Cfg::new(graph);
        let live: HashSet<_> = cfg.rpo.iter().copied().collect();
        self.body.retain(|block| live.contains(block));
        if self.body.is_empty() {
            return Ok(());
        }
        // Earlier copies can introduce exit phis or move a child's probes to parent entry.
        let candidates = Candidates {
            facts: std::mem::take(&mut self.assumes),
            codes: std::mem::take(&mut self.identities),
        };
        let (mut current, _) = plan_region(
            graph,
            self.body,
            self.entry,
            &self.calls,
            &candidates,
        )?;
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
        let mut child_canonical = None;
        let mut region_fallback = None;
        if self.speculative() {
            let blocks: Vec<_> = Cfg::new(graph)
                .rpo
                .into_iter()
                .filter(|block| self.body.contains(block))
                .collect();
            let fallback = fallback.filter(|_| self.entry.is_none());
            let speculation = Speculation {
                fallback,
                reads: &self.reads,
                assumes: &self.checks,
                guards: &self.guards,
                exits: &self.exits,
                typed_reads: &self.typed_reads,
                code_guards: &self.identities,
                ..Default::default()
            };
            let version = apply_version(graph, &blocks, self.entry, &speculation, Some(canonical));
            consume_assumes(graph, &blocks, self.entry, &self.checks);
            region_fallback = Some(if let Some(address) = fallback {
                RegionFallback::CompiledEntry(address)
            } else {
                let source = graph.regions.push(Region {
                    blocks: self.body.clone(),
                    fallback: None,
                });
                RegionFallback::Region(source)
            });
            notes.extend(version.notes.iter().cloned());
            graph.verify()?;
            if let Some(previous) = fallback {
                notes.push(format!(
                    "function: entry guard failure reuses {previous:#x}"
                ));
            }
            let probe_like = !self.identities.is_empty()
                || self.requirements.iter().any(|r| r.captured.is_none());
            if let Some(reason) = closure_failure(
                graph,
                &version.body,
                &guarded_accesses(&self.requirements, &self.identities),
                &HashMap::new(),
            )
            .filter(|_| probe_like)
            {
                return Err(format!("{label}: planned copy {reason}"));
            }
            notes.push(format!(
                "{label}: combined entry guards: {} native calls, {} number assumptions, {} read probes, {} hoisted guards, {} typed receiver assumptions and {} length checks",
                self.requirements.len(), self.numbers(), self.reads.len(), self.guards.len(), self.typed().len(), self.bounds().len()
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
            child_canonical = Some(canonical.copied(&version));
            self.body = version.body;
        } else {
            notes.push(format!("{label}: closed without guards"));
        }
        for child in self.children {
            let first = regions.len();
            child.emit(
                graph,
                None,
                regions,
                notes,
                report_regions,
                child_canonical.as_ref().unwrap_or(canonical),
            )?;
            for region in &regions[first..] {
                self.body
                    .extend(graph.regions[*region].blocks.iter().copied());
            }
        }
        regions.push(graph.regions.push(Region {
            blocks: self.body,
            fallback: region_fallback,
        }));
        Ok(())
    }
}

fn plan_region(
    graph: &Graph,
    body: HashSet<Block>,
    entry: Option<(Block, Block)>,
    conditional: &[ConditionalCall],
    candidates: &Candidates,
) -> Result<(RegionPlan, Facts)> {
    let blocks: Vec<_> = Cfg::new(graph)
        .rpo
        .into_iter()
        .filter(|block| body.contains(block))
        .collect();
    let exits: Vec<_> = blocks
        .iter()
        .copied()
        .filter(|&block| cold_exit(graph, block).is_some())
        .collect();
    let fallbacks: HashSet<_> = exits
        .iter()
        .filter_map(|&block| cold_exit(graph, block))
        .collect();
    let written: HashSet<_> = body.difference(&fallbacks).copied().collect();
    let mut requirements = requirements(graph, &blocks);
    let native_of = |requirement: &Requirement| match graph.insts[requirement.call].op {
        Op::Call { callee, .. } => (callee, Fact::Native(requirement.native)),
        _ => unreachable!(),
    };
    let mut calls: Vec<_> = conditional
        .iter()
        .filter(|call| {
            written.contains(&graph.insts[call.inst].block)
                && body.contains(&graph.def_block(call.callee))
                && matches!(graph.insts[call.inst].op, Op::Call { callee, .. } if callee == call.callee)
                && matches!(graph.defining_op(call.callee), Some(Op::GlobalRead(name)) if name == &call.name)
        })
        .cloned()
        .collect();
    let assumed_codes = |call: &ConditionalCall| {
        call.identities
            .iter()
            .all(|identity| candidates.codes.contains(identity))
    };
    let assumed_native =
        |requirement: &Requirement| candidates.facts.contains(&native_of(requirement));
    requirements.retain(|requirement| assumed_native(requirement));
    calls.retain(|call| assumed_codes(call));
    let build = |requirements: &[Requirement], calls: &[ConditionalCall]| {
        let mut planning = graph.clone();
        for requirement in requirements {
            let Op::Call { args, .. } = &planning.insts[requirement.call].op else {
                unreachable!()
            };
            planning.insts[requirement.call].op = Op::CallNative(requirement.native, args.clone());
        }
        let assumed: HashMap<Value, cranelift_module::FuncId> = calls
            .iter()
            .map(|call| (call.callee, call.target))
            .collect();
        (planning, assumed)
    };
    let identities_of = |calls: &[ConditionalCall]| {
        let mut identities = HashMap::new();
        for call in calls {
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
        Ok(identities)
    };
    let (mut planning, assumed, identities, guarded, reentry) = loop {
        let identities = identities_of(&calls)?;
        let (planning, assumed) = build(&requirements, &calls);
        let found = conflicts(&planning, &written, &assumed);
        let dead_identities: HashSet<&String> = identities
            .iter()
            .filter(|identity| {
                found.invalidates(&guarded_accesses(&[], std::slice::from_ref(identity)))
            })
            .map(|(name, _)| name)
            .collect();
        let live_requirements: Vec<bool> = requirements
            .iter()
            .map(|requirement| {
                requirement.captured.is_some()
                    || !found.invalidates(&guarded_accesses(
                        std::slice::from_ref(requirement),
                        &[],
                    ))
            })
            .collect();
        let live_calls: Vec<bool> = calls
            .iter()
            .map(|call| {
                !call
                    .identities
                    .iter()
                    .any(|(name, _)| dead_identities.contains(name))
            })
            .collect();
        if live_requirements.iter().all(|&live| live) && live_calls.iter().all(|&live| live) {
            let guarded = guarded_accesses(&requirements, &identities);
            break (planning, assumed, identities, guarded, found.reentry);
        }
        let mut index = 0;
        requirements.retain(|_| {
            index += 1;
            live_requirements[index - 1]
        });
        let mut index = 0;
        calls.retain(|_| {
            index += 1;
            live_calls[index - 1]
        });
    };
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
    let reads = probe_reads(
        &planning,
        &body,
        &written,
        &blocks,
        entry.is_none(),
        &assumed,
    );
    let available: HashSet<_> = reads
        .iter()
        .filter_map(|&inst| planning.insts[inst].result)
        .collect();
    let assumes = candidates.facts.clone();
    let numbers = entry_assumes(&planning, &body, &reads, &assumed, &assumes);
    let mut guards = invariant_guards(&planning, &body, &blocks, &reads);
    if reentry.is_some() {
        guards.retain(|&block| pure_guard(&planning, block));
    }
    // Candidate facts must not narrow the original graph before its guards execute.
    let assumptions = numbers
        .iter()
        .filter_map(|&(value, fact)| Some((value, fact.mask()?)))
        .collect();
    let facts = Facts::assuming(&planning, &guards::assumed(&assumptions, &assumed));
    let typed = typed_entry(&planning, &body, &available, &facts, &assumes);
    let objects: Vec<Value> = typed.iter().map(|&(object, _)| object).collect();
    let mut bounded = bounded_reads(
        &planning,
        &Cfg::new(&planning),
        &body,
        &objects,
        &available,
        &facts,
    );
    bounded.retain(|_, &mut (object, bound)| {
        assumes.contains(&(object, Fact::Bound(bound)))
            && (reentry.is_none() || typed.iter().any(|&(typed, _)| typed == object))
    });
    let mut bounds: Vec<_> = bounded.values().copied().collect();
    bounds.sort();
    bounds.dedup();
    let mut checks = numbers;
    checks.extend(typed.iter().map(|&(object, element)| (object, Fact::Typed(element))));
    checks.extend(bounds.iter().map(|&(object, bound)| (object, Fact::Bound(bound))));
    checks.extend(requirements.iter().map(native_of));
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
        && closure_failure(&planning, &body, &guarded, &assumed).is_some()
    {
        return Err(reentry
            .unwrap_or_else(|| "no invariant entry guards for reentrant fallbacks".into()));
    }
    Ok((
        RegionPlan {
            body,
            entry,
            requirements,
            calls,
            identities,
            reads,
            assumes,
            checks,
            guards,
            exits,
            typed_reads,
            children: Vec::new(),
        },
        facts,
    ))
}

fn plan(
    graph: &Graph,
    conditional: &ConditionalPlan,
    reached: &[Reach],
    notes: &mut Vec<String>,
) -> Vec<RegionPlan> {
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
    let bodies: HashMap<Option<Block>, HashSet<Block>> = candidates
        .iter()
        .map(|(body, entry)| (entry.map(|(_, header)| header), body.clone()))
        .collect();
    let mut plans: Vec<RegionPlan> = Vec::new();
    for (body, entry) in candidates {
        let mut proposals = Candidates::default();
        for reach in reached.iter().filter(|reach| {
            body.contains(&reach.use_block)
                && bodies
                    .get(&reach.header)
                    .is_some_and(|outer| outer.is_superset(&body))
        }) {
            match (reach.fact, &reach.global) {
                (Fact::Code(code), Some(name)) => proposals.codes.push((name.clone(), code)),
                (Fact::Code(_), None) => {}
                (fact, _) => proposals.facts.push((reach.value, fact)),
            }
        }
        let (mut parent, facts) = match plan_region(
            graph,
            body,
            entry,
            &conditional.calls,
            &proposals,
        ) {
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
) -> Result<Vec<RegionId>> {
    version_with_report(graph, conditional, fallback, notes, false)
}

pub(in crate::jit::semantic) fn version_with_report(
    graph: &mut Graph,
    conditional: &ConditionalPlan,
    fallback: Option<usize>,
    notes: &mut Vec<String>,
    report_regions: bool,
) -> Result<Vec<RegionId>> {
    let reached = super::propose::assume(graph, conditional);
    let plans = plan(graph, conditional, &reached, notes);
    let original = graph.clone();
    let canonical = Canonical::identity(&original);
    if report_regions {
        let mut report = String::from("region plan before versioning:\n");
        for plan in &plans {
            report.push_str(&plan.report(graph, 1, true));
        }
        notes.push(report);
    }
    let mut regions = Vec::new();
    for plan in plans {
        plan.emit(
            graph,
            fallback,
            &mut regions,
            notes,
            report_regions,
            &canonical,
        )?;
    }
    graph.strip_assumes();
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
    fn entry_numeric_observations_replace_copied_loop_observations() {
        use crate::jit::profile::{Observation, ReadKind};

        let mut graph = graph(
            "function sum(point,n){let sum=0;for(let i=0;i<n;++i)sum+=point.x+i;return sum;}",
        );
        let repository = super::super::super::specialize::Repository::default();
        super::super::super::observations::plan(&mut graph, Some((&repository, 7)));
        version(&mut graph, &Default::default(), None, &mut Vec::new()).unwrap();
        let cfg = Cfg::new(&graph);
        let observations: Vec<_> = cfg
            .rpo
            .iter()
            .flat_map(|&block| {
                graph.blocks[block]
                    .insts
                    .iter()
                    .filter_map(|&inst| match graph.insts[inst].op {
                        Op::Observe(value, Observation::Read(ReadKind::Number), site) => {
                            Some((block, value, site))
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        assert!(!observations.is_empty());
        for &(block, value, site) in &observations {
            assert!(!observations.iter().any(|&(other, candidate, origin)| {
                other != block
                    && candidate == value
                    && origin == site
                    && cfg.dominates(other, block)
            }));
        }
        graph.verify().unwrap();
    }

    #[test]
    fn observed_numeric_types_select_guarded_region_facts() {
        use crate::jit::profile::{ReadKind, ReadType, Site};
        for observed in [ReadType::Integer, ReadType::Number] {
            let mut graph =
                graph("function f(x,n){let sum=0;for(let i=0;i<n;++i)sum+=x+i;return sum;}");
            let (inst, value) = graph
                .insts
                .iter()
                .find_map(|(inst, data)| {
                    matches!(data.op, Op::Argument(0)).then(|| (inst, data.result.unwrap()))
                })
                .unwrap();
            let data = &graph.insts[inst];
            let site = Site {
                source: 7,
                start: data.span.start,
                end: data.span.end,
            };
            let mut repository = super::super::super::specialize::Repository::default();
            repository
                .read_feedback
                .insert((site, ReadKind::Number), observed);
            let decisions =
                super::super::super::observations::plan(&mut graph, Some((&repository, 7)));
            assert_eq!(decisions, vec![(site, observed)]);
            assert!(!Facts::compute(&graph).of(value).within(Types::INTEGER));
            version(&mut graph, &Default::default(), None, &mut Vec::new()).unwrap();
            let facts = Facts::compute(&graph);
            let integers: Vec<_> = Cfg::new(&graph)
                .rpo
                .iter()
                .flat_map(|&block| &graph.blocks[block].insts)
                .filter_map(|&inst| {
                    matches!(graph.insts[inst].op, Op::AsInteger(_))
                        .then(|| graph.insts[inst].result.unwrap())
                })
                .collect();
            assert_eq!(!integers.is_empty(), observed == ReadType::Integer);
            for value in integers {
                assert!(facts.of(value).within(Types::INTEGER));
            }
            graph.verify().unwrap();
        }
    }

    #[test]
    fn observed_storage_and_numbers_combine_for_bounded_integer_reads() {
        use crate::jit::profile::{ReadKind, ReadType, Site};
        let graph = graph(
            "function f(x,offsets){let sum=0;for(let k=0;k<offsets.length;++k)sum+=(x+offsets[k])*1;return sum;}",
        );
        let mut repository = super::super::super::specialize::Repository::default();
        for data in graph.insts.values() {
            let observed = match data.op {
                Op::Argument(0) => Some((ReadKind::Number, ReadType::Integer)),
                Op::GetIndex(..) => Some((ReadKind::Receiver, ReadType::Typed(TypedKind::Int32))),
                _ => None,
            };
            if let Some((kind, observed)) = observed {
                repository.read_feedback.insert(
                    (
                        Site {
                            source: 7,
                            start: data.span.start,
                            end: data.span.end,
                        },
                        kind,
                    ),
                    observed,
                );
            }
        }
        let canonical = super::super::super::Canonical::from_graph(graph);
        let analysis =
            super::super::super::optimize_canonical(canonical, true, Some((&repository, 7)))
                .unwrap();
        let graph = &analysis.graph;
        let facts = Facts::compute(graph);
        let integer_add = Cfg::new(graph)
            .rpo
            .iter()
            .flat_map(|&block| &graph.blocks[block].insts)
            .any(|&inst| {
                let data = &graph.insts[inst];
                let Op::Binary(BinaryOperator::Addition, _, right) = data.op else {
                    return false;
                };
                matches!(
                    graph.defining_op(right),
                    Some(Op::TypedIndex(
                        _,
                        _,
                        TypedRead {
                            element: Some(TypedKind::Int32),
                            bounded: true
                        }
                    ))
                ) && facts.of(data.result.unwrap()).within(Types::INTEGER)
            });
        assert!(integer_add);
        graph.verify().unwrap();
    }

    #[test]
    fn versioning_keeps_feedback_specialized_typed_reads() {
        use crate::heap::TypedKind;
        use crate::jit::profile::{ReadKind, ReadType, Site};

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
        let mut graph = graph(
            "function f(grid,width){let sum=0;for(let y=0;y<2;++y){const offset=y*width;for(let x=0;x<2;++x)sum+=Math.sqrt(grid[offset+x]);}return sum;}",
        );
        let reached = super::propose::assume(&mut graph, &ConditionalPlan::default());
        let plans = plan(
            &graph,
            &ConditionalPlan::default(),
            &reached,
            &mut Vec::new(),
        );
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
        assert!(matches!(
            graph.regions[regions[0]].fallback,
            Some(RegionFallback::CompiledEntry(0x1234))
        ));
        assert!(graph.regions[regions[0]].blocks.contains(&original));
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
        assert!(regions.iter().any(|&id| match graph.regions[id].fallback {
            Some(RegionFallback::Region(source)) => !graph.regions[source].blocks.is_empty(),
            _ => false,
        }));
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
        let mut graph = graph(
            "function f(rows){let sum=0;for(let y=0;y<2;++y){const row=rows[y];for(let x=0;x<2;++x)sum+=row[x];}return sum;}",
        );
        let reached = super::propose::assume(&mut graph, &ConditionalPlan::default());
        let plans = plan(
            &graph,
            &ConditionalPlan::default(),
            &reached,
            &mut Vec::new(),
        );
        assert_eq!(plans.len(), 1);
        let root = &plans[0];
        assert!(root.entry.is_none());
        assert!(root.children.is_empty());
        assert!(root.typed().is_empty());
        assert!(root.typed_reads.is_empty());
    }

    #[test]
    fn versioning_shares_repeated_typed_receiver_guards() {
        use crate::jit::profile::{ReadKind, ReadType, Site};

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
    fn nested_loop_continuations_survive_parent_pruning() {
        use crate::jit::profile::{ReadKind, ReadType, Site};

        let graph = graph(
            "function f(rows,offsets,weights,points){unknown();let total=0;for(let y=0;y<points.length;++y){let base=points[y].x;for(let i=0;i<weights.length;++i){let x=base+offsets[i];if(x<0||x>=rows.length)continue;total+=rows[x][y]/weights[i];}}return total;}",
        );
        let mut repository = super::super::super::specialize::Repository::default();
        for data in graph.insts.values() {
            if let Op::GetIndex(object, _) = data.op {
                let element = match graph.defining_op(object) {
                    Some(Op::GetIndex(..)) => TypedKind::Float64,
                    Some(Op::Argument(1)) => TypedKind::Int32,
                    Some(Op::Argument(2)) => TypedKind::Float64,
                    _ => continue,
                };
                repository.read_feedback.insert(
                    (
                        Site {
                            source: 7,
                            start: data.span.start,
                            end: data.span.end,
                        },
                        ReadKind::Receiver,
                    ),
                    ReadType::Typed(element),
                );
            }
            if matches!(data.op, Op::Binary(BinaryOperator::Addition, ..)) {
                repository.read_feedback.insert(
                    (
                        Site {
                            source: 7,
                            start: data.span.start,
                            end: data.span.end,
                        },
                        ReadKind::Number,
                    ),
                    ReadType::Integer,
                );
            }
        }
        let canonical = super::super::super::Canonical::from_graph(graph);
        let analysis =
            super::super::super::optimize_canonical(canonical, true, Some((&repository, 7)))
                .unwrap();
        let cfg = Cfg::new(&analysis.graph);
        assert_eq!(natural_loops(&analysis.graph, &cfg).len(), 2);
        assert!(cfg.rpo.iter().any(|&block| {
            analysis.graph.blocks[block]
                .insts
                .iter()
                .any(|&inst| matches!(analysis.graph.insts[inst].op, Op::Continuation(..)))
        }));
        analysis.graph.verify().unwrap();
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
        let reached = super::propose::assume(&mut graph, &ConditionalPlan::default());
        let plans = plan(
            &graph,
            &ConditionalPlan::default(),
            &reached,
            &mut Vec::new(),
        );
        assert!(plans[0].typed().iter().any(|&(object, _)| object == xs));
        assert!(
            plans[0]
                .children
                .iter()
                .any(|child| { child.typed().iter().any(|&(object, _)| object == xs) })
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
        let mut graph = graph(
            "function f(state){let sum=0;for(let y=0;y<2;++y){state.width=y+1;for(let x=0;x<2;++x)sum+=state.grid[x*state.width];}return sum;}",
        );
        let reached = super::propose::assume(&mut graph, &ConditionalPlan::default());
        let plans = plan(
            &graph,
            &ConditionalPlan::default(),
            &reached,
            &mut Vec::new(),
        );
        assert_eq!(plans.len(), 1);
        let root = &plans[0];
        assert!(root.entry.is_none());
        assert_eq!(root.children.len(), 1);
        let child = &root.children[0];
        assert!(child.checks.iter().any(|&(value, _)| {
            matches!(graph.defining_op(value), Some(Op::Get(_, name)) if name == "width")
        }));
        assert!(!root.checks.iter().any(|&(value, _)| {
            matches!(graph.defining_op(value), Some(Op::Get(_, name)) if name == "width")
        }));
    }
}

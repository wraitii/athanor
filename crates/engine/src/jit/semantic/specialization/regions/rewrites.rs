use super::super::super::guards::{Fact, Origin};
use super::super::super::ranges::{IntRange, Ranges};
use super::super::calls::ConditionalCall;
use super::*;
use crate::jit::semantic::JsOp;

#[derive(Default)]
pub(super) struct Candidates {
    pub(super) locals: Vec<Inst>,
    pub(super) facts: Vec<(Value, Fact)>,
    pub(super) codes: Vec<(String, cranelift_module::FuncId)>,
    pub(super) origins: Vec<(Value, Fact, Origin)>,
    pub(super) code_origins: Vec<(String, cranelift_module::FuncId, Origin)>,
}

pub(super) struct RegionPlan {
    pub(super) locals: Vec<Inst>,
    pub(super) body: FxHashSet<Block>,
    pub(super) entry: Option<(Block, Block)>,
    pub(super) requirements: Vec<Requirement>,
    pub(super) calls: Vec<ConditionalCall>,
    pub(super) identities: Vec<(String, cranelift_module::FuncId)>,
    pub(super) reads: Vec<Inst>,
    pub(super) checks: Vec<(Value, Fact)>,
    pub(super) guards: Vec<Block>,
    pub(super) exits: Vec<Block>,
    pub(super) typed_reads: Vec<(Inst, TypedRead)>,
    pub(super) array_reads: Vec<(Inst, IndexBound)>,
    pub(super) ranges: Vec<(Value, Interval)>,
    pub(super) survivors: Vec<(Fact, Origin)>,
    pub(super) reentry: Option<String>,
    pub(super) children: Vec<RegionPlan>,
}

impl RegionPlan {
    pub(super) fn typed(&self) -> Vec<(Value, Option<TypedKind>)> {
        self.checks
            .iter()
            .filter_map(|&(value, fact)| match fact {
                Fact::Typed(element) => Some((value, element)),
                _ => None,
            })
            .collect()
    }

    pub(super) fn bounds(&self) -> Vec<(Value, Value)> {
        self.checks
            .iter()
            .filter_map(|&(value, fact)| match fact {
                Fact::Bound(bound) => Some((value, bound)),
                _ => None,
            })
            .collect()
    }

    pub(super) fn numbers(&self) -> usize {
        self.checks
            .iter()
            .filter(|(_, fact)| matches!(fact, Fact::Type(_)))
            .count()
    }

    pub(super) fn report(&self, graph: &Graph, depth: usize, children: bool) -> String {
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
            let types = match fact {
                Fact::Shape(shape) => {
                    writeln!(out, "{indent}  shape assumption {value}: {shape}").unwrap();
                    continue;
                }
                Fact::Cached(cache, _) => {
                    writeln!(out, "{indent}  cache-hit assumption {value}: {cache}").unwrap();
                    continue;
                }
                Fact::IntegerRange(min, max) => {
                    writeln!(out, "{indent}  integer range {value}: {min}..={max}").unwrap();
                    continue;
                }
                Fact::Type(types) => types,
                _ => continue,
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
        for &inst in &self.locals {
            writeln!(
                out,
                "{indent}  local assumption {inst}: {}",
                graph.op_text(&graph.insts[inst].op)
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
        for &(inst, bound) in &self.array_reads {
            match bound {
                IndexBound::Max(max) => {
                    writeln!(out, "{indent}  packed array read {inst}: max index {max}").unwrap()
                }
                IndexBound::Below(bound) => writeln!(
                    out,
                    "{indent}  packed array read {inst}: index below {bound}"
                )
                .unwrap(),
            }
        }
        for (object, interval) in &self.ranges {
            writeln!(
                out,
                "{indent}  index range {interval:?} within length({object})"
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

    pub(super) fn speculative(&self) -> bool {
        !self.locals.is_empty()
            || !self.identities.is_empty()
            || !self.requirements.is_empty()
            || !self.checks.is_empty()
            || !self.guards.is_empty()
            || !self.array_reads.is_empty()
            || !self.ranges.is_empty()
    }
}

pub(super) fn plan_region(
    graph: &Graph,
    known: &Facts,
    body: FxHashSet<Block>,
    entry: Option<(Block, Block)>,
    conditional: &[ConditionalCall],
    candidates: &Candidates,
    protected: &FxHashSet<Block>,
) -> Result<(RegionPlan, Graph)> {
    let locals: Vec<_> = candidates
        .locals
        .iter()
        .copied()
        .filter(|&inst| {
            let block = graph.insts[inst].block;
            body.contains(&block)
                && graph.blocks[block].insts.contains(&inst)
                && match graph.insts[inst].op {
                    Op::Assume(value, Fact::Type(types), _) => !known.of(value).within(types),
                    Op::Assume(_, Fact::Typed(_) | Fact::Array, _) => true,
                    _ => false,
                }
        })
        .collect();
    let blocks: Vec<_> = known
        .cfg
        .rpo
        .iter()
        .copied()
        .filter(|block| body.contains(block) && !protected.contains(block))
        .collect();
    let exits: Vec<_> = blocks
        .iter()
        .copied()
        .filter(|&block| cold_exit(graph, block).is_some())
        .collect();
    let fallbacks: FxHashSet<_> = exits
        .iter()
        .filter_map(|&block| cold_exit(graph, block))
        .collect();
    let written: FxHashSet<_> = body.difference(&fallbacks).copied().collect();
    let mut requirements = requirements(graph, &blocks);
    let native_of = |requirement: &Requirement| match graph.insts[requirement.call].op {
        Op::Javascript(JsOp::Call { callee, .. }) => (callee, Fact::Native(requirement.native)),
        _ => unreachable!(),
    };
    let mut calls: Vec<_> = conditional
        .iter()
        .filter(|call| {
            written.contains(&graph.insts[call.inst].block)
                && body.contains(&graph.def_block(call.callee))
                && matches!(graph.insts[call.inst].op, Op::Javascript(JsOp::Call { callee, .. }) if callee == call.callee)
                && matches!(graph.defining_op(call.callee), Some(Op::Javascript(JsOp::GlobalRead(name))) if name == &call.name)
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
            let Op::Javascript(JsOp::Call { args, .. }) = &planning.insts[requirement.call].op
            else {
                unreachable!()
            };
            planning.set_op(
                requirement.call,
                Op::Javascript(JsOp::CallNative(requirement.native, args.clone())),
            );
        }
        let assumed: FxHashMap<Value, cranelift_module::FuncId> = calls
            .iter()
            .map(|call| (call.callee, call.target))
            .collect();
        (planning, assumed)
    };
    let identities_of = |calls: &[ConditionalCall]| {
        let mut identities = FxHashMap::default();
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
        let dead_identities: FxHashSet<&String> = identities
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
                    || !found.invalidates(&guarded_accesses(std::slice::from_ref(requirement), &[]))
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
            Op::Javascript(JsOp::Call { callee, .. }) => Some(callee),
            _ => None,
        })
        .collect();
    while let Some(value) = unused.pop() {
        if uses[value] != 0 || !body.contains(&planning.def_block(value)) {
            continue;
        }
        let ValueDef::Result(inst, _) = planning.values[value] else {
            continue;
        };
        match planning.insts[inst].op {
            Op::Javascript(JsOp::Get(object, _)) => {
                uses[object] -= 1;
                unused.push(object);
            }
            Op::Javascript(JsOp::GlobalRead(_)) => {}
            _ => continue,
        }
        planning.remove_inst(inst);
    }
    let shared = Facts::assuming(&planning, &guards::assumed(&FxHashMap::default(), &assumed));
    let reads = probe_reads(
        &planning,
        &body,
        &written,
        &blocks,
        entry.is_none(),
        &shared,
    );
    let available: FxHashSet<_> = reads
        .iter()
        .filter_map(|&inst| planning.insts[inst].result())
        .collect();
    let assumes = candidates.facts.clone();
    let numbers = entry_assumes(&planning, &body, &reads, &shared, &assumes);
    let mut guards = invariant_guards(&planning, &body, &blocks, &reads);
    let writes = Writes::with_facts(&planning, &written, &shared);
    guards.retain(|&block| {
        let TerminatorKind::Branch(condition, ..) =
            planning.blocks[block].terminator.as_ref().unwrap().kind
        else {
            return false;
        };
        let condition =
            super::super::super::guards::negated(&planning, condition).unwrap_or(condition);
        match planning.values[condition] {
            ValueDef::Result(inst, _) => writes.preserves(&planning, inst),
            ValueDef::Param(..) => true,
        }
    });
    // Candidate facts must not narrow the original graph before its guards execute.
    let assumptions = numbers
        .iter()
        .filter_map(|&(value, fact)| Some((value, fact.mask()?)))
        .collect();
    // Call rewrites and dead callee removal leave control flow unchanged.
    let facts = Facts::assuming_values(
        &planning,
        known.cfg.clone(),
        &guards::assumed(&assumptions, &assumed),
    );
    let array_reads = if let Some(entry) = entry {
        bounded_array_reads(
            &planning, &facts, &shared, &body, &blocks, entry, &assumes, &available,
        )
    } else {
        Vec::new()
    };
    let typed = typed_entry(&planning, &body, &available, &facts, &assumes);
    let objects: Vec<Value> = typed.iter().map(|&(object, _)| object).collect();
    let mut bounded = bounded_reads(&planning, &facts.cfg, &body, &objects, &available, &facts);
    bounded.retain(|_, &mut (object, bound)| assumes.contains(&(object, Fact::Bound(bound))));
    let ranges = entry.map_or_else(Vec::new, |(preheader, header)| {
        super::propose::integer_ranges(&planning, &facts, preheader, header, &body, &numbers)
    });
    let inputs: Vec<_> = ranges
        .iter()
        .filter_map(|&(value, fact)| match fact {
            Fact::IntegerRange(min, max) => Some((value, IntRange::new(min.into(), max.into()))),
            _ => None,
        })
        .collect();
    let (ranged, index_ranges) = ranged_reads(
        &planning, &facts, &body, &blocks, &objects, &bounded, &available, &inputs,
    );
    let mut bounds: Vec<_> = bounded.values().copied().collect();
    bounds.sort();
    bounds.dedup();
    let mut checks = numbers;
    checks.extend(ranges);
    checks.extend(
        typed
            .iter()
            .map(|&(object, element)| (object, Fact::Typed(element))),
    );
    checks.extend(
        bounds
            .iter()
            .map(|&(object, bound)| (object, Fact::Bound(bound))),
    );
    checks.extend(requirements.iter().map(native_of));
    let mut overwrites = shaped_overwrites(&planning, &written, &assumes);
    let preserved = loop {
        let writes = Writes::with_facts(&planning, &written, &shared)
            .preserving_shapes(&overwrites.iter().map(|&(inst, _)| inst).collect());
        let preserved: Vec<_> = assumes
            .iter()
            .copied()
            .filter(|&(value, fact)| {
                if body.contains(&planning.def_block(value)) && !available.contains(&value) {
                    return false;
                }
                match fact {
                    Fact::Array => true,
                    Fact::Shape(shape) => {
                        Access::read(graph, &Op::Javascript(JsOp::IsShape(value, shape)), &shared)
                            .is_some_and(|read| !writes.clobbers(&read))
                    }
                    Fact::ReadLayout(layout) => Access::read(
                        graph,
                        &Op::Javascript(JsOp::IsReadLayout(value, layout)),
                        &shared,
                    )
                    .is_some_and(|read| !writes.clobbers(&read)),
                    Fact::Cached(cache, builtins) => {
                        (!body.contains(&planning.def_block(cache)) || available.contains(&cache))
                            && Access::read(
                                graph,
                                &Op::Javascript(JsOp::IsPropertyCached(value, cache, builtins)),
                                &shared,
                            )
                            .is_some_and(|read| !writes.clobbers(&read))
                    }
                    _ => false,
                }
            })
            .collect();
        let count = overwrites.len();
        overwrites.retain(|(_, requirement)| preserved.contains(requirement));
        if overwrites.len() == count {
            break preserved;
        }
    };
    checks.extend(preserved);
    let mut typed_reads = Vec::new();
    for &block in &blocks {
        for &inst in &planning.blocks[block].insts {
            if let Op::Javascript(JsOp::GetIndex(object, key) | JsOp::TypedIndex(object, key, _)) =
                planning.insts[inst].op
                && let Some(&(_, element)) =
                    typed.iter().find(|(candidate, _)| *candidate == object)
                && facts.of(key).within(NUMERIC)
                && !reads.contains(&inst)
            {
                let previous = match planning.insts[inst].op {
                    Op::Javascript(JsOp::TypedIndex(_, _, read)) => Some(read),
                    _ => None,
                };
                typed_reads.push((
                    inst,
                    TypedRead {
                        element: previous.and_then(|read| read.element).or(element),
                        bounded: previous.is_some_and(|read| read.bounded)
                            || bounded.contains_key(&inst)
                            || ranged.contains(&inst),
                    },
                ));
            }
        }
    }
    if locals.is_empty()
        && identities.is_empty()
        && requirements.is_empty()
        && checks.is_empty()
        && guards.is_empty()
        && array_reads.is_empty()
        && closure_failure_with(&planning, &body, &guarded, &shared).is_some()
    {
        return Err(
            reentry.unwrap_or_else(|| "no invariant entry guards for reentrant fallbacks".into())
        );
    }
    let mut survivors: Vec<(Fact, Origin)> = candidates
        .origins
        .iter()
        .filter(|&&(value, fact, _)| {
            checks
                .iter()
                .any(|&(checked, known)| checked == value && known.implies(fact))
        })
        .map(|&(_, fact, origin)| (fact, origin))
        .collect();
    survivors.extend(
        candidates
            .code_origins
            .iter()
            .filter(|(name, code, _)| identities.contains(&(name.clone(), *code)))
            .map(|&(_, code, origin)| (Fact::Code(code), origin)),
    );
    Ok((
        RegionPlan {
            locals,
            body,
            entry,
            requirements,
            calls,
            identities,
            reads,
            checks,
            guards,
            exits,
            typed_reads,
            array_reads,
            ranges: index_ranges,
            survivors,
            reentry,
            children: Vec::new(),
        },
        planning,
    ))
}

/// Typed reads on `objects` whose key stays within an interval computable at region entry,
/// and those intervals.
#[allow(clippy::too_many_arguments)]
fn ranged_reads(
    graph: &Graph,
    facts: &Facts,
    body: &FxHashSet<Block>,
    blocks: &[Block],
    objects: &[Value],
    bounded: &FxHashMap<Inst, (Value, Value)>,
    available: &FxHashSet<Value>,
    assumed: &[(Value, IntRange)],
) -> (FxHashSet<Inst>, Vec<(Value, Interval)>) {
    let counters = Counters::discover(graph, &facts.cfg);
    let mut ranges = Ranges::new(graph, facts, assumed);
    let mut ranged = FxHashSet::default();
    let mut intervals = Vec::new();
    for &block in blocks.iter().filter(|block| !graph.cold.contains(block)) {
        for &inst in &graph.blocks[block].insts {
            let Op::Javascript(JsOp::GetIndex(object, key) | JsOp::TypedIndex(object, key, _)) =
                graph.insts[inst].op
            else {
                continue;
            };
            if !objects.contains(&object)
                || bounded.contains_key(&inst)
                || !facts.of(key).within(NUMERIC)
            {
                continue;
            }
            let mut invariant = |value: Value| {
                ((!body.contains(&graph.def_block(value)) || available.contains(&value))
                    && facts.of(value).within(ValueType::NUMBER))
                .then(|| ranges.integer_at(value, graph.def_block(value)))
            };
            let Some(interval) =
                Interval::of(graph, &facts.cfg, &counters, block, key, &mut invariant)
            else {
                continue;
            };
            // A failed entry check sends the whole region to generic code, so leave keys that
            // always go negative, such as `a[i - 1]` behind `i > 0`, to their per-read checks.
            if interval.low().max < 0 {
                continue;
            }
            ranged.insert(inst);
            let range = (object, interval);
            if !intervals.contains(&range) {
                intervals.push(range);
            }
        }
    }
    (ranged, intervals)
}

#[allow(clippy::too_many_arguments)]
fn bounded_array_reads(
    graph: &Graph,
    facts: &Facts,
    effects: &Facts,
    body: &FxHashSet<Block>,
    blocks: &[Block],
    (preheader, header): (Block, Block),
    assumes: &[(Value, Fact)],
    available: &FxHashSet<Value>,
) -> Vec<(Inst, IndexBound)> {
    use super::super::super::ranges::{LoopEndpoints, Ranges};

    let mut ranges = Ranges::new(graph, facts, &[]);
    let mut endpoints = None;
    let mut counters = None;
    let writes = Writes::with_facts(graph, body, effects);
    blocks
        .iter()
        .filter(|block| !graph.cold.contains(block))
        .flat_map(|&block| &graph.blocks[block].insts)
        .filter_map(|&inst| {
            let Op::Javascript(JsOp::GetIndex(object, key) | JsOp::ArrayGet(object, key)) =
                graph.insts[inst].op
            else {
                return None;
            };
            let block = graph.insts[inst].block;
            if !facts.cfg.dominates(graph.def_block(object), preheader)
                || !body.contains(&graph.def_block(key))
                || !assumes.contains(&(object, Fact::Array))
                    && !guards::holds(graph, &facts.cfg, preheader, object, Fact::Array)
            {
                return None;
            }
            for test in [JsOp::ArrayKind(object), JsOp::IndexedLength(object)] {
                let read = Access::read(graph, &Op::Javascript(test), facts)?;
                if writes.clobbers(&read) {
                    return None;
                }
            }
            if let Some(range) = ranges.integer_at(key, block)
                && range.min >= 0
                && range.max < i64::from(u32::MAX)
                && endpoints
                    .get_or_insert_with(|| LoopEndpoints::discover(graph, &facts.cfg, header))
                    .as_ref()
                    .is_some_and(|endpoints| endpoints.reaches(key, block, range.max))
            {
                return Some((inst, IndexBound::Max(range.max)));
            }
            let &(inside, bound) = counters
                .get_or_insert_with(|| loop_counters(graph, &facts.cfg, body, available, facts))
                .get(&key)?;
            facts
                .cfg
                .dominates(inside, block)
                .then_some((inst, IndexBound::Below(bound)))
        })
        .collect()
}

pub(super) fn shaped_overwrites(
    graph: &Graph,
    written: &FxHashSet<Block>,
    assumes: &[(Value, Fact)],
) -> Vec<(Inst, (Value, Fact))> {
    let canonical = |mut value| {
        while let Some(Op::Javascript(JsOp::AsObject(original))) = graph.defining_op(value) {
            value = *original;
        }
        value
    };
    written
        .iter()
        .flat_map(|&block| &graph.blocks[block].insts)
        .filter_map(|&inst| {
            let Op::Javascript(JsOp::Set(object, ..)) = graph.insts[inst].op else {
                return None;
            };
            let &[layout] = graph.property_accesses.get(&inst)?.shapes.as_slice() else {
                return None;
            };
            let requirement = (object, Fact::Shape(layout.shape));
            assumes
                .iter()
                .any(|&(value, fact)| {
                    canonical(value) == canonical(object) && fact == requirement.1
                })
                .then_some((inst, requirement))
        })
        .collect()
}

pub(super) fn property_assumptions(graph: &mut Graph, notes: &mut Vec<String>) {
    use super::super::super::available;

    let facts = Facts::compute(graph);
    let solved = available::solve(graph, &facts, &facts.cfg, Default::default(), false);
    let accesses: FxHashMap<_, _> = graph
        .property_accesses
        .iter()
        .filter_map(|(&inst, access)| Some((access.cache?, (inst, access.shapes.clone()))))
        .collect();
    let proposals: Vec<_> = facts
        .cfg
        .rpo
        .iter()
        .copied()
        .flat_map(|block| graph.blocks[block].insts.iter().copied())
        .filter_map(|inst| {
            let Op::Assume(object, Fact::Property(cache), origin) = graph.insts[inst].op else {
                return None;
            };
            let (access, shapes) = accesses.get(&cache)?;
            let fact = match (&graph.insts[*access].op, shapes.as_slice()) {
                (Op::Javascript(JsOp::Get(_, name)), []) => {
                    Some(Fact::Cached(cache, crate::builtins::builtin_method(&name)))
                }
                (Op::Javascript(JsOp::Get(..) | JsOp::Set(..)), &[layout]) => {
                    (!solved.as_ref().is_some_and(|solved| {
                        let state = solved.before(graph, &facts, &facts.cfg, inst);
                        solved.holds(graph, &state, object, layout.requirement())
                    }))
                    .then_some(layout.requirement())
                }
                _ => None,
            };
            Some((inst, object, fact, origin))
        })
        .collect();
    let canonical = |value| {
        solved
            .as_ref()
            .map_or(value, |solved| solved.canonical(value))
    };
    let mut shapes = FxHashMap::default();
    let mut conflicting = FxHashSet::default();
    for &(_, object, fact, _) in &proposals {
        if let Some(shape) = match fact {
            Some(Fact::Shape(shape)) => Some(shape),
            Some(Fact::ReadLayout(layout)) => Some(layout.shape),
            _ => None,
        } {
            let object = canonical(object);
            if shapes
                .insert(object, shape)
                .is_some_and(|other| other != shape)
            {
                conflicting.insert(object);
            }
        }
    }
    if !conflicting.is_empty() {
        notes.push(format!(
            "property ICs: conflicting shape requirements for {} receiver(s) stay inline",
            conflicting.len()
        ));
    }
    for (inst, object, fact, origin) in proposals {
        let fact = fact.filter(|fact| {
            !matches!(fact, Fact::Shape(_) | Fact::ReadLayout(_))
                || !conflicting.contains(&canonical(object))
        });
        if let Some(fact) = fact {
            graph.set_op(inst, Op::Assume(object, fact, origin));
        } else {
            graph.detach_inst(inst);
        }
    }
}

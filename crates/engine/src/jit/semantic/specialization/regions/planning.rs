use super::super::super::guards::Fact;
use super::super::calls::ConditionalPlan;
use super::assumptions::Proposal;
use super::cost::{Candidate, Verdict, worth};
use super::hoisting::RecoveryPoint;
use super::motion::JsMotion;
use super::rewrites::{Candidates, RegionPlan, plan_region, property_assumptions};
use super::versioning::{Bailouts, materialize};
use super::*;
#[cfg(test)]
use crate::jit::layout::ReadLayout;
use crate::{FxHashMap, FxHashSet};

fn plan(
    graph: &Graph,
    conditional: &ConditionalPlan,
    assumptions: &[Proposal],
    notes: &mut Vec<String>,
    shapes: &super::super::shapes::Plan,
) -> Result<(Graph, Vec<RegionPlan>)> {
    let known = Facts::values(graph);
    let cfg = &known.cfg;
    let mut candidates: Vec<_> = natural_loops(graph, cfg)
        .into_iter()
        .filter_map(|(header, body)| {
            if graph.cold.contains(&header) {
                return None;
            }
            let preheader = preheader(graph, cfg, header, &body)?;
            Some((body, Some((preheader, header))))
        })
        .collect();
    candidates.push((cfg.rpo.iter().copied().collect(), None));
    let bodies: FxHashMap<Option<Block>, FxHashSet<Block>> = candidates
        .iter()
        .map(|(body, entry)| (entry.map(|(_, header)| header), body.clone()))
        .collect();
    let points: Vec<_> = candidates
        .iter()
        .map(|(body, entry)| {
            let target = match entry {
                Some((before, _)) => {
                    let TerminatorKind::Jump(target) =
                        &graph.blocks[*before].terminator.as_ref().unwrap().kind
                    else {
                        unreachable!()
                    };
                    target.clone()
                }
                None => Target::new(graph.entry),
            };
            RecoveryPoint {
                before: entry.map(|(before, _)| before),
                target,
                body: body.clone(),
            }
        })
        .collect();
    let mut groups = Vec::new();
    let mut working = graph.clone();
    let mut values: FxHashMap<_, _> = graph.values.keys().map(|value| (value, value)).collect();
    let mut bailouts = Bailouts::new(graph);
    let mut plans: Vec<RegionPlan> = Vec::new();
    for (_, entry) in candidates {
        let known = Facts::values(&working);
        let (body, entry) = match entry {
            Some((_, header)) => {
                let Some(current) = natural_loop(&known.cfg, header) else {
                    continue;
                };
                let Some(before) = preheader(&working, &known.cfg, header, &current) else {
                    continue;
                };
                (current, Some((before, header)))
            }
            None => (known.cfg.rpo.iter().copied().collect(), None),
        };
        let protected: FxHashSet<_> = groups
            .iter()
            .flat_map(|group: &super::hoisting::AssumeCfg| group.blocks.iter().copied())
            .collect();
        let mut proposals = Candidates::default();
        for proposal in assumptions
            .iter()
            .filter(|proposal| body.contains(&proposal.point))
        {
            if !matches!(proposal.fact, Fact::Code(_))
                && !working.blocks[working.insts[proposal.inst].block]
                    .insts
                    .contains(&proposal.inst)
            {
                continue;
            }
            if entry.is_none()
                && matches!(proposal.fact, Fact::Shape(_) | Fact::ReadLayout(_))
                && known.cfg.rpo.iter().any(|&block| {
                    matches!(
                        working.blocks[block].terminator.as_ref().unwrap().kind,
                        TerminatorKind::Return(_)
                    ) && !known.cfg.dominates(proposal.point, block)
                })
            {
                continue;
            }
            if proposal.local {
                proposals.locals.push(proposal.inst);
            }
            match (proposal.fact, &proposal.global) {
                (Fact::Code(code), Some(name)) => {
                    proposals.codes.push((name.clone(), code));
                    proposals
                        .code_origins
                        .push((name.clone(), code, proposal.origin));
                }
                (Fact::Code(_), None) => {}
                (mut fact, _) => {
                    let value = values[&proposal.value];
                    if let Fact::Bound(bound) | Fact::Cached(bound, _) = &mut fact {
                        *bound = values[bound];
                    }
                    proposals.facts.push((value, fact));
                    proposals.origins.push((value, fact, proposal.origin));
                }
            }
        }
        let label = entry.map_or_else(
            || "function".to_owned(),
            |(_, header)| format!("loop at {header}"),
        );
        let (mut parent, mut candidate) = match plan_region(
            &working,
            &known,
            body,
            entry,
            &conditional.calls,
            &proposals,
            &protected,
        ) {
            Ok(plan) => plan,
            Err(reason) => {
                notes.push(format!("{label}: not versioned: {reason}"));
                continue;
            }
        };
        let cost = Candidate {
            size: parent
                .body
                .iter()
                .filter(|block| !candidate.cold.contains(block))
                .map(|&block| candidate.blocks[block].insts.len())
                .sum(),
            checks: parent.checks.len() + parent.identities.len() + parent.locals.len(),
            reentry: parent.reentry.as_deref(),
            survivors: &parent.survivors,
        };
        match worth(&cost) {
            Verdict::Open => notes.push(format!("{label}: gate open: {}", cost.summary())),
            Verdict::Skip(reason) => {
                notes.push(format!("{label}: not versioned: {reason}"));
                continue;
            }
        }
        let blocks: Vec<_> = known
            .cfg
            .rpo
            .iter()
            .copied()
            .filter(|block| parent.body.contains(block))
            .collect();
        let shaped_reads: Vec<_> = parent
            .reads
            .iter()
            .filter_map(|&inst| {
                let Op::Get(..) = candidate.insts[inst].op else {
                    return None;
                };
                // The probe validates its own shape without a region-wide shape assumption.
                shapes.read(&candidate, inst).map(|layout| (inst, layout))
            })
            .collect();
        let speculation = Speculation {
            locals: &parent.locals,
            reads: &parent.reads,
            shaped_reads: &shaped_reads,
            assumes: &parent.checks,
            guards: &parent.guards,
            exits: &parent.exits,
            typed_reads: &parent.typed_reads,
            code_guards: &parent.identities,
            ..Default::default()
        };
        let mut candidate_values = values.clone();
        let mut candidate_notes = Vec::new();
        let added = candidate.blocks.len();
        let added_groups = if parent.speculative() {
            materialize(
                &mut candidate,
                &blocks,
                parent.entry,
                points
                    .iter()
                    .find(|point| {
                        point.target.block == entry.map_or(graph.entry, |(_, header)| header)
                    })
                    .unwrap()
                    .target
                    .clone(),
                &speculation,
                &parent.requirements,
                &mut candidate_values,
                &mut bailouts,
                &mut candidate_notes,
            )
            .map_err(|reason| format!("{label}: {reason}"))?
        } else {
            Vec::new()
        };
        parent.body.extend(candidate.blocks.keys().skip(added));
        let live: FxHashSet<_> = Cfg::new(&candidate).rpo.into_iter().collect();
        parent.body.retain(|block| live.contains(block));
        if let Some(reason) = closure_failure(
            &candidate,
            &parent.body,
            &guarded_accesses(&parent.requirements, &parent.identities),
            &FxHashMap::default(),
        ) {
            notes.push(format!(
                "{label}: not versioned: guarded candidate {reason}"
            ));
            continue;
        }
        candidate
            .verify()
            .map_err(|reason| format!("{label}: {reason}"))?;
        notes.extend(candidate_notes);
        if !parent.calls.is_empty() {
            notes.push(format!(
                "{label}: {} global code checks prove {} read-only call(s)",
                parent.identities.len(),
                parent.calls.len()
            ));
        }
        notes.push(format!("{label}: combined entry guards: {} native calls, {} number assumptions, {} read probes, {} hoisted guards, {} typed receiver assumptions and {} length checks",
            parent.requirements.len(), parent.numbers(), parent.reads.len(), parent.guards.len(), parent.typed().len(), parent.bounds().len()));
        let mut remaining = Vec::new();
        for child in plans {
            if bodies[&child.entry.map(|(_, header)| header)]
                .is_subset(&bodies[&entry.map(|(_, header)| header)])
            {
                parent.children.push(child);
            } else {
                remaining.push(child);
            }
        }
        remaining.push(parent);
        plans = remaining;
        working = candidate;
        values = candidate_values;
        groups.extend(added_groups);
        super::hoisting::hoist(
            &mut working,
            &mut groups,
            &points,
            &bailouts,
            &values,
            &JsMotion,
            notes,
        );
        working.verify()?;
    }
    Ok((working, plans))
}

#[cfg(test)]
pub(in crate::jit::semantic) fn version(
    graph: &mut Graph,
    conditional: &ConditionalPlan,
    fallback: Option<usize>,
    notes: &mut Vec<String>,
) -> Result<Vec<RegionId>> {
    super::super::properties::assume_natives(graph);
    version_regions(
        graph,
        conditional,
        fallback,
        true,
        notes,
        false,
        &Default::default(),
    )
}

pub(in crate::jit::semantic) fn version_regions(
    graph: &mut Graph,
    conditional: &ConditionalPlan,
    fallback: Option<usize>,
    continuations: bool,
    notes: &mut Vec<String>,
    report_regions: bool,
    shapes: &super::super::shapes::Plan,
) -> Result<Vec<RegionId>> {
    property_assumptions(graph, notes);
    let reached = super::assumptions::prepare(graph, conditional);
    let original = graph.clone();
    let (mut candidate, plans) = plan(&original, conditional, &reached, notes, shapes)?;
    if report_regions {
        let mut report = String::from("region candidates after specialization:\n");
        for plan in &plans {
            report.push_str(&plan.report(&candidate, 1, true));
        }
        notes.push(report);
    }
    let mut regions = Vec::new();
    fn record(graph: &mut Graph, plan: RegionPlan, regions: &mut Vec<RegionId>) {
        regions.push(graph.regions.push(Region {
            blocks: plan.body,
            fallback: None,
        }));
    }
    for plan in plans {
        record(&mut candidate, plan, &mut regions);
    }
    super::versioning::finish(
        &mut candidate,
        &original,
        fallback,
        continuations,
        &regions,
        notes,
    )?;
    *graph = candidate;
    materialize_properties(graph)?;
    graph.strip_assumes();
    Ok(regions)
}

#[cfg(test)]
mod tests {
    use super::super::super::super::low_level::{Binary as LowBinary, Op as LowOp};
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
        let graph = super::super::super::super::build::lower(
            function,
            semantic.scoping(),
            &FxHashSet::default(),
        )
        .unwrap();
        let graph = super::super::super::super::canonicalize_graph(graph, true)
            .unwrap()
            .graph;
        let mut function = super::super::super::super::FunctionState::new(graph, Vec::new(), true);
        super::super::super::super::js::comparisons::expose(&mut function);
        let graph = function.graph;
        graph
    }

    #[test]
    fn assumption_cfg_hoisting_moves_the_recovery_boundary() {
        let mut graph = graph("function f(a,n){let s=0;for(let i=0;i<n;++i)s+=a*1;return s;}");
        let proposals = super::super::assumptions::prepare(&mut graph, &ConditionalPlan::default());
        let generic_entry = graph.entry;
        let mut notes = Vec::new();
        let (candidate, _) = plan(
            &graph,
            &ConditionalPlan::default(),
            &proposals,
            &mut notes,
            &Default::default(),
        )
        .unwrap();
        assert!(
            notes.iter().any(|note| note.starts_with("assumption CFG")),
            "{notes:?}"
        );
        let targets: Vec<_> = Cfg::new(&candidate)
            .rpo
            .iter()
            .filter_map(
                |&block| match &candidate.blocks[block].terminator.as_ref().unwrap().kind {
                    TerminatorKind::Bailout { target, .. } => Some(target.block),
                    _ => None,
                },
            )
            .collect();
        assert!(!targets.is_empty());
        assert!(
            targets.iter().all(|&target| target == generic_entry),
            "{targets:?}"
        );
        candidate.verify().unwrap();
    }

    #[test]
    fn guard_motion_keeps_a_stable_recovery_identity() {
        struct Motion;
        impl super::super::hoisting::Motion for Motion {
            fn dependencies(
                &self,
                _: &Graph,
                _: &Facts,
                _: &super::super::hoisting::AssumeCfg,
                _: &RecoveryPoint,
            ) -> Option<Vec<Inst>> {
                Some(Vec::new())
            }
        }
        let original = graph("function f(){return 1;}");
        let mut candidate = original.clone();
        let before = candidate.entry;
        let good = candidate.blocks.push(BlockData::default());
        candidate.blocks[good].terminator = candidate.blocks[before].terminator.take();
        let start = candidate.blocks.push(BlockData::default());
        let exit = candidate.blocks.push(BlockData::default());
        let bailouts = Bailouts::new(&original);
        let values = original.values.keys().map(|value| (value, value)).collect();
        let failure = bailouts.target(&mut candidate, Target::new(original.entry), &values);
        let condition =
            candidate.append_value(start, candidate.span, Op::Constant(Constant::Boolean(true)));
        candidate.terminate(
            before,
            candidate.span,
            TerminatorKind::Jump(Target::new(start)),
        );
        candidate.terminate(
            start,
            candidate.span,
            TerminatorKind::Branch(condition, Target::new(exit), failure.clone()),
        );
        candidate.terminate(
            exit,
            candidate.span,
            TerminatorKind::Jump(Target::new(good)),
        );
        let mut groups = [super::super::hoisting::AssumeCfg {
            start,
            exit,
            fallback: failure.block,
            recovery: original.entry,
            before: Some(before),
            blocks: [start, exit].into_iter().collect(),
            scope: FxHashSet::default(),
            dependencies: Vec::new(),
        }];
        let points = [RecoveryPoint {
            before: Some(before),
            target: Target::new(original.entry),
            body: [original.entry].into_iter().collect(),
        }];
        assert!(!super::super::hoisting::hoist(
            &mut candidate,
            &mut groups,
            &points,
            &bailouts,
            &values,
            &Motion,
            &mut Vec::new()
        ));
        assert!(
            matches!(&candidate.blocks[exit].terminator.as_ref().unwrap().kind, TerminatorKind::Jump(target) if target.block == good)
        );
        candidate.verify().unwrap();
    }

    #[test]
    fn proposal_recovery_points_exist_before_cloning() {
        let mut graph = graph("function f(a,n){let s=0;for(let i=0;i<n;++i)s+=a[i];return s;}");
        let proposals = super::super::assumptions::prepare(&mut graph, &ConditionalPlan::default());
        let original = graph.clone();
        for proposal in proposals {
            assert_eq!(original.insts[proposal.inst].block, proposal.point);
            assert!(
                original.blocks[proposal.point]
                    .insts
                    .contains(&proposal.inst)
            );
        }
        original.verify().unwrap();
    }

    #[test]
    fn invariant_guards_handle_block_parameters() {
        let mut graph = graph("function f(value){return value;}");
        let entry = graph.entry;
        let span = graph.span;
        let value = graph.append_value(entry, span, Op::Argument(0));
        let guard = graph.blocks.push(BlockData::default());
        let fast = graph.blocks.push(BlockData::default());
        let cold = graph.blocks.push(BlockData::default());
        let condition = graph.add_param(guard);
        graph.terminate(
            entry,
            span,
            TerminatorKind::Jump(Target {
                block: guard,
                args: vec![value],
            }),
        );
        graph.terminate(
            guard,
            span,
            TerminatorKind::Branch(condition, Target::new(fast), Target::new(cold)),
        );
        graph.terminate(fast, span, TerminatorKind::Return(value));
        graph.terminate(cold, span, TerminatorKind::Return(value));
        graph.cold.insert(cold);
        graph.verify().unwrap();
        assert!(
            invariant_guards(
                &graph,
                &FxHashSet::from_iter([guard, fast, cold]),
                &[guard],
                &[]
            )
            .is_empty()
        );
        assert_eq!(
            invariant_guards(&graph, &FxHashSet::from_iter([fast, cold]), &[guard], &[]),
            vec![guard]
        );
    }

    #[test]
    fn property_dispatch_uses_updated_receivers() {
        for source in [
            "function f(object){return object.child.value;}",
            "function f(object){object.child.value=1;return object;}",
        ] {
            let mut graph = graph(source);
            let accesses: Vec<_> = graph
                .insts
                .iter()
                .filter_map(|(inst, data)| {
                    matches!(data.op, Op::Get(..) | Op::Set(..)).then_some(inst)
                })
                .collect();
            for inst in accesses {
                graph.property_accesses.insert(
                    inst,
                    super::super::super::properties::Access {
                        cache: None,
                        shapes: vec![ReadLayout::own(10, 0), ReadLayout::own(20, 1)],
                    },
                );
            }
            materialize_properties(&mut graph).unwrap();
            super::super::super::properties::rewrite(&mut graph).unwrap();
            graph.verify().unwrap();
        }
    }

    #[test]
    fn native_calls_survive_entry_probe_replacement() {
        for fallback in [None, Some(0x1234)] {
            let mut graph = graph(
                "function f(object){const floor=Math.floor;return [floor(object.x),floor(object.y),floor];}",
            );
            let regions = version(
                &mut graph,
                &ConditionalPlan::default(),
                fallback,
                &mut Vec::new(),
            )
            .unwrap();
            assert!(!regions.is_empty());
            let facts = Facts::compute(&graph);
            assert!(regions.iter().all(|&region| {
                closure_failure_with(&graph, &graph.regions[region].blocks, &[], &facts).is_none()
            }));
            assert_eq!(
                facts
                    .cfg
                    .rpo
                    .iter()
                    .flat_map(|&block| &graph.blocks[block].insts)
                    .filter(|&&inst| matches!(graph.insts[inst].op, Op::CallNative(..)))
                    .count(),
                2
            );
            graph.verify().unwrap();
        }
    }

    #[test]
    fn entry_numeric_observations_replace_copied_loop_observations() {
        use crate::jit::profile::{Observation, ReadKind};

        let mut graph = graph(
            "function sum(point,n){let sum=0;for(let i=0;i<n;++i)sum+=point.x+i;return sum;}",
        );
        let repository = super::super::super::calls::Repository::default();
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
    fn observed_numeric_inputs_flow_through_parameters() {
        use crate::jit::profile::{ReadKind, ReadType, Site};

        for (source, expected) in [
            (
                "function f(a,n){let x=a.x;for(let i=0;i<n;++i)x=(x+1)*0.5;return x;}",
                true,
            ),
            (
                "function f(a,n,choose){let x=a.x;if(choose)x=1;for(let i=0;i<n;++i)x=(x+1)*0.5;return x;}",
                true,
            ),
            (
                "function f(a,n){let x=a.x;for(let i=0;i<n;++i)x=i;return x;}",
                false,
            ),
        ] {
            let mut graph = graph(source);
            let (read, value) = graph
                .insts
                .values()
                .find_map(|data| {
                    matches!(data.op, Op::Get(_, ref name) if name == "x")
                        .then(|| (data.span, data.result().unwrap()))
                })
                .unwrap();
            let mut repository = super::super::super::calls::Repository::default();
            repository.read_feedback.insert(
                (
                    Site {
                        inline_context: 0,
                        source: 7,
                        start: read.start,
                        end: read.end,
                    },
                    ReadKind::Number,
                ),
                ReadType::Number,
            );
            super::super::super::observations::plan(&mut graph, Some((&repository, 7)));
            let assumptions = super::super::propose::numbers(&mut graph);
            assert_eq!(
                assumptions.contains(&(value, Fact::Type(ValueType::NUMBER))),
                expected,
                "{source}",
            );
            if expected {
                let facts = Facts::assuming(&graph, &assumptions);
                for data in graph.insts.values() {
                    if let Op::Binary(BinaryOperator::Addition, left, _) = data.op {
                        assert!(facts.of(left).within(ValueType::NUMBER), "{source}");
                    }
                }
            }
            graph.verify().unwrap();
        }
    }

    #[test]
    fn observed_operands_prove_derived_numeric_values() {
        use crate::jit::profile::{ReadKind, ReadType, Site};

        let mut graph = graph(
            "function f(a,b,n){const x=a.x+b.x;let sum=0;for(let i=0;i<n;++i)sum+=x+i;return sum;}",
        );
        let mut repository = super::super::super::calls::Repository::default();
        for data in graph.insts.values() {
            if matches!(data.op, Op::Get(..)) {
                repository.read_feedback.insert(
                    (
                        Site {
                            inline_context: 0,
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
        super::super::super::observations::plan(&mut graph, Some((&repository, 7)));
        let derived = graph
            .insts
            .values()
            .find_map(|data| match data.op {
                Op::Binary(BinaryOperator::Addition, left, right)
                    if matches!(graph.defining_op(left), Some(Op::Get(..)))
                        && matches!(graph.defining_op(right), Some(Op::Get(..))) =>
                {
                    data.result()
                }
                _ => None,
            })
            .unwrap();
        let assumptions = super::super::propose::numbers(&mut graph);
        assert!(!assumptions.is_empty());
        assert!(!assumptions.iter().any(|&(value, _)| value == derived));
        assert!(
            Facts::assuming(&graph, &assumptions)
                .of(derived)
                .within(ValueType::INTEGER)
        );
        graph.verify().unwrap();
    }

    #[test]
    fn feedback_planning_reuses_properties_until_reentry() {
        for (source, expected) in [
            (
                "function f(holder){return holder.strength+(1-holder.strength);}",
                1,
            ),
            (
                "function f(holder,mutate){const first=holder.strength;mutate(holder);return first+(1-holder.strength);}",
                2,
            ),
        ] {
            let function = super::super::super::super::FunctionState::from_graph(graph(source));
            let planned =
                super::super::super::super::plan_specialization(function, true, None).unwrap();
            let reads = Cfg::new(&planned.graph)
                .rpo
                .iter()
                .flat_map(|&block| &planned.graph.blocks[block].insts)
                .filter(|&&inst| matches!(&planned.graph.insts[inst].op, Op::Get(_, name) if name == "strength"))
                .count();
            assert_eq!(reads, expected);
            planned.graph.verify().unwrap();
        }
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
                    matches!(data.op, Op::Argument(0)).then(|| (inst, data.result().unwrap()))
                })
                .unwrap();
            let data = &graph.insts[inst];
            let site = Site {
                inline_context: 0,
                source: 7,
                start: data.span.start,
                end: data.span.end,
            };
            let mut repository = super::super::super::calls::Repository::default();
            repository
                .read_feedback
                .insert((site, ReadKind::Number), observed);
            let decisions =
                super::super::super::observations::plan(&mut graph, Some((&repository, 7)));
            assert_eq!(decisions, vec![(site, observed)]);
            assert!(!Facts::compute(&graph).of(value).within(ValueType::INTEGER));
            version(&mut graph, &Default::default(), None, &mut Vec::new()).unwrap();
            let facts = Facts::compute(&graph);
            let integers: Vec<_> = Cfg::new(&graph)
                .rpo
                .iter()
                .flat_map(|&block| &graph.blocks[block].insts)
                .filter(|&&inst| matches!(graph.insts[inst].op, Op::AsInteger(_)))
                .map(|&inst| graph.insts[inst].result().unwrap())
                .collect();
            assert_eq!(!integers.is_empty(), observed == ReadType::Integer);
            for value in integers {
                assert!(facts.of(value).within(ValueType::INTEGER));
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
        let mut repository = super::super::super::calls::Repository::default();
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
                            inline_context: 0,
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
        let canonical = super::super::super::super::FunctionState::from_graph(graph);
        let analysis =
            super::super::super::super::optimize_canonical(canonical, true, Some((&repository, 7)))
                .unwrap();
        let graph = &analysis.graph;
        let facts = Facts::compute(graph);
        let integer_add = Cfg::new(graph)
            .rpo
            .iter()
            .flat_map(|&block| &graph.blocks[block].insts)
            .any(|&inst| {
                let data = &graph.insts[inst];
                let Op::LowLevel(LowOp::Binary {
                    op: LowBinary::Add,
                    right,
                    ..
                }) = data.op
                else {
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
                ) && facts.of(data.result().unwrap()).within(ValueType::INTEGER)
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
            inline_context: 0,
            source: 7,
            start: data.span.start,
            end: data.span.end,
        };
        let mut repository = super::super::super::calls::Repository::default();
        repository.read_feedback.insert(
            (site, ReadKind::Receiver),
            ReadType::Typed(TypedKind::Uint16),
        );
        let canonical = super::super::super::super::FunctionState::from_graph(graph);
        let analysis =
            super::super::super::super::optimize_canonical(canonical, true, Some((&repository, 7)))
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
    fn nested_loop_candidates_keep_one_specialized_body() {
        let mut graph = graph(
            "function f(grid,width){let sum=0;for(let y=0;y<2;++y){const offset=y*width;for(let x=0;x<2;++x)sum+=Math.sqrt(grid[offset+x]);}return sum;}",
        );
        super::super::super::properties::assume_natives(&mut graph);
        let reached = super::assumptions::prepare(&mut graph, &ConditionalPlan::default());
        let (candidate, plans) = plan(
            &graph,
            &ConditionalPlan::default(),
            &reached,
            &mut Vec::new(),
            &Default::default(),
        )
        .unwrap();
        assert_eq!(plans.len(), 1);
        let root = &plans[0];
        assert!(root.entry.is_none());
        assert!(!root.children.is_empty());
        let cfg = Cfg::new(&candidate);
        assert_eq!(natural_loops(&candidate, &cfg).len(), 2);
        assert!(cfg.rpo.iter().any(|&block| {
            candidate.blocks[block]
                .insts
                .iter()
                .any(|&inst| matches!(candidate.insts[inst].op, Op::CallNative(..)))
        }));
        assert!(cfg.rpo.iter().any(|&block| matches!(
            candidate.blocks[block].terminator.as_ref().unwrap().kind,
            TerminatorKind::Bailout { .. }
        )));
        let guards: Vec<_> = cfg
            .rpo
            .iter()
            .filter_map(|&block| {
                let TerminatorKind::Branch(condition, _, no) =
                    &candidate.blocks[block].terminator.as_ref()?.kind
                else {
                    return None;
                };
                let Op::GuardValid(guard) = candidate.defining_op(*condition)? else {
                    return None;
                };
                Some((*guard, no.block))
            })
            .collect();
        assert!(!guards.is_empty());
        for (guard, start) in guards {
            let mut pending = vec![start];
            let mut visited = FxHashSet::default();
            let mut committed = false;
            while let Some(block) = pending.pop() {
                if !visited.insert(block) {
                    continue;
                }
                committed |= candidate.blocks[block].insts.iter().any(
                    |&inst| matches!(candidate.insts[inst].op, Op::GuardCommit(id) if id == guard),
                );
                if let Some(term) = &candidate.blocks[block].terminator {
                    pending.extend(term.kind.targets().into_iter().map(|target| target.block));
                }
            }
            assert!(committed, "guard {guard} cannot be refreshed");
        }
        candidate.verify().unwrap();
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
        assert!(!super::super::super::super::continuation::bodies(&graph).is_empty());
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
        super::super::super::properties::assume_natives(&mut graph);
        let reached = super::assumptions::prepare(&mut graph, &ConditionalPlan::default());
        let (_, plans) = plan(
            &graph,
            &ConditionalPlan::default(),
            &reached,
            &mut Vec::new(),
            &Default::default(),
        )
        .unwrap();
        assert_eq!(plans.len(), 1);
        let root = &plans[0];
        assert!(root.entry.is_none());
        assert!(root.typed().is_empty());
        assert!(root.typed_reads.is_empty());
    }

    #[test]
    fn versioning_shares_repeated_typed_receiver_guards() {
        use crate::jit::profile::{ReadKind, ReadType, Site};

        let graph = graph(
            "function f(xs,n){unknown();let sum=0;for(let i=0;i<n;++i)sum+=xs[i]+xs[i+1];return sum;}",
        );
        let mut repository = super::super::super::calls::Repository::default();
        for data in graph
            .insts
            .values()
            .filter(|data| matches!(data.op, Op::GetIndex(..)))
        {
            repository.read_feedback.insert(
                (
                    Site {
                        inline_context: 0,
                        source: 7,
                        start: data.span.start,
                        end: data.span.end,
                    },
                    ReadKind::Receiver,
                ),
                ReadType::Typed(TypedKind::Uint16),
            );
        }
        let canonical = super::super::super::super::FunctionState::from_graph(graph);
        let analysis =
            super::super::super::super::optimize_canonical(canonical, true, Some((&repository, 7)))
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
    fn varying_numeric_reads_specialize_inside_the_outer_loop() {
        use crate::jit::profile::{ReadKind, ReadType, Site};

        let graph = graph(
            "function f(a,n,m){let total=0;for(let j=0;j<m;++j)for(let i=0;i<n;++i)total+=a[i];return total;}",
        );
        let mut repository = super::super::super::calls::Repository::default();
        for data in graph.insts.values() {
            if !matches!(data.op, Op::GetIndex(..)) {
                continue;
            }
            let site = Site {
                inline_context: 0,
                source: 7,
                start: data.span.start,
                end: data.span.end,
            };
            repository
                .read_feedback
                .insert((site, ReadKind::Number), ReadType::Integer);
            repository
                .read_feedback
                .insert((site, ReadKind::Receiver), ReadType::Array);
        }
        let canonical = super::super::super::super::FunctionState::from_graph(graph);
        let analysis =
            super::super::super::super::optimize_canonical(canonical, true, Some((&repository, 7)))
                .unwrap();
        let graph = &analysis.graph;
        let cfg = Cfg::new(graph);
        let additions: Vec<_> = cfg
            .rpo
            .iter()
            .flat_map(|&block| &graph.blocks[block].insts)
            .filter_map(|&inst| {
                matches!(
                    graph.insts[inst].op,
                    Op::LowLevel(LowOp::Binary {
                        op: LowBinary::Add | LowBinary::FloatAdd,
                        ..
                    })
                )
                .then(|| graph.insts[inst].result().unwrap())
            })
            .collect();
        assert!(!additions.is_empty());
        assert!(
            additions
                .iter()
                .all(|&value| analysis.facts.of(value).within(ValueType::NUMBER))
        );
        assert_eq!(natural_loops(graph, &cfg).len(), 2);
        assert!(
            analysis
                .notes
                .iter()
                .any(|note| note.starts_with("local assumption") && note.contains("bailout"))
        );
        assert!(!super::super::super::super::continuation::bodies(graph).is_empty());
        graph.verify().unwrap();
    }

    #[test]
    fn nested_loop_continuations_survive_parent_pruning() {
        use crate::jit::profile::{ReadKind, ReadType, Site};

        let graph = graph(
            "function f(rows,offsets,weights,points){unknown();let total=0;for(let y=0;y<points.length;++y){let base=points[y].x;for(let i=0;i<weights.length;++i){let x=base+offsets[i];if(x<0||x>=rows.length)continue;total+=rows[x][y]/weights[i];}}return total;}",
        );
        let mut repository = super::super::super::calls::Repository::default();
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
                            inline_context: 0,
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
                            inline_context: 0,
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
        let canonical = super::super::super::super::FunctionState::from_graph(graph);
        let analysis =
            super::super::super::super::optimize_canonical(canonical, true, Some((&repository, 7)))
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
                    .then_some(data.result())
                    .flatten()
            })
            .unwrap();
        super::super::super::properties::assume_natives(&mut graph);
        let reached = super::assumptions::prepare(&mut graph, &ConditionalPlan::default());
        let (candidate, plans) = plan(
            &graph,
            &ConditionalPlan::default(),
            &reached,
            &mut Vec::new(),
            &Default::default(),
        )
        .unwrap();
        let cfg = Cfg::new(&candidate);
        let (_, outer_header) = plans[0].entry.unwrap();
        assert!(cfg.rpo.iter().any(|&block| {
            let Some(Terminator { kind: TerminatorKind::Branch(condition, good, _), .. }) = &candidate.blocks[block].terminator else { return false };
            matches!(candidate.defining_op(*condition), Some(Op::IsTyped(object, _)) if *object == xs)
                && cfg.dominates(good.block, outer_header)
        }));
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
        super::super::super::properties::assume_natives(&mut graph);
        let reached = super::assumptions::prepare(&mut graph, &ConditionalPlan::default());
        let (candidate, plans) = plan(
            &graph,
            &ConditionalPlan::default(),
            &reached,
            &mut Vec::new(),
            &Default::default(),
        )
        .unwrap();
        assert_eq!(plans.len(), 1);
        let root = &plans[0];
        assert!(root.entry.is_none());
        let cfg = Cfg::new(&candidate);
        let checks: Vec<_> = cfg.rpo.iter().copied().filter(|&block| candidate.blocks[block].insts.iter().any(|&inst| {
            let (Op::IsNumber(value) | Op::IsInteger(value)) = candidate.insts[inst].op else { return false; };
            matches!(candidate.defining_op(value), Some(Op::ProbeProperty(_, name)) if name == "width")
        })).collect();
        assert!(!checks.is_empty());
        for block in checks {
            assert!(cfg.rpo.iter().any(|&write| cfg.dominates(write, block) && candidate.blocks[write].insts.iter()
                .any(|&inst| matches!(&candidate.insts[inst].op, Op::Set(_, name, _) if name == "width"))));
        }
        candidate.verify().unwrap();
    }

    #[test]
    fn reentrant_functions_are_not_versioned() {
        let mut graph = graph("function f(g,n){g();return n*2+1;}");
        let mut notes = Vec::new();
        let regions = version(&mut graph, &ConditionalPlan::default(), None, &mut notes).unwrap();
        assert!(regions.is_empty());
        assert!(
            notes
                .iter()
                .any(|note| note.starts_with("function: not versioned: reenters"))
        );
        graph.verify().unwrap();
    }

    #[test]
    fn closed_loops_are_versioned() {
        let mut graph = graph("function f(n){let sum=0;for(let i=0;i<n;++i)sum+=i*2;return sum;}");
        let mut notes = Vec::new();
        let regions = version(&mut graph, &ConditionalPlan::default(), None, &mut notes).unwrap();
        assert!(!regions.is_empty());
        assert!(notes.iter().any(|note| note.contains("gate open")));
        graph.verify().unwrap();
    }
}

// Overwrites guarded by their own shape requirement cannot change any shape, so they do not
// clobber the region's shape or cache checks.
pub(in crate::jit::semantic) fn materialize_properties(graph: &mut Graph) -> Result<()> {
    use super::super::super::{available, dispatch};
    use super::super::properties::{Access, shape_writes};

    shape_writes(graph);
    let facts = Facts::compute(graph);
    let solved = available::solve(graph, &facts, &facts.cfg, Default::default(), false);
    let accesses: Vec<_> = facts
        .cfg
        .rpo
        .iter()
        .copied()
        .flat_map(|block| graph.blocks[block].insts.iter().copied())
        .filter_map(|inst| {
            let (Op::Get(object, _) | Op::Set(object, ..)) = graph.insts[inst].op else {
                return None;
            };
            let access = graph.property_accesses.get(&inst)?;
            if access.shapes.is_empty() {
                return None;
            }
            if solved.as_ref().is_some_and(|solved| {
                let state = solved.before(graph, &facts, &facts.cfg, inst);
                access
                    .shapes
                    .iter()
                    .any(|&layout| solved.holds(graph, &state, object, layout.requirement()))
            }) {
                return None;
            }
            Some((inst, access.shapes.clone()))
        })
        .collect();
    for (inst, shapes) in accesses {
        let data = graph.insts[inst].clone();
        let (Op::Get(object, _) | Op::Set(object, ..)) = data.op else {
            unreachable!()
        };
        let alternatives = shapes
            .into_iter()
            .map(|layout| dispatch::Alternative {
                requirements: vec![(object, layout.requirement())],
                implementation: layout,
            })
            .collect();
        let first = graph.blocks.len();
        dispatch::apply(graph, inst, alternatives, |graph, block, layout| {
            let (copy, value) =
                graph.append(block, data.span, data.op.clone(), data.result().is_some());
            graph.insts[copy].source_name = data.source_name;
            graph.insts[copy].inline_context = data.inline_context;
            graph.property_accesses.insert(
                copy,
                Access {
                    cache: None,
                    shapes: vec![layout],
                },
            );
            value
        });
        graph
            .property_accesses
            .get_mut(&inst)
            .unwrap()
            .shapes
            .clear();
        dispatch::extend_regions(graph, data.block, first);
    }
    graph.verify()
}

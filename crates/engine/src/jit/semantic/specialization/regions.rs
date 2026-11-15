use super::super::transforms::loops::{natural_loops, preheader};
use crate::{FxHashMap, FxHashSet};

use oxc_syntax::operator::BinaryOperator;

use super::super::super::effects::Locations;
use super::super::access::{Access, Field};
use super::super::guards;
use super::super::js::NUMERIC;
use super::super::{
    Block, BlockData, Cfg, Constant, Facts, FunctionState, Graph, Inst, Native, Op, Region,
    RegionFallback, RegionId, Result, Target, Terminator, TerminatorKind, TypedRead, UnaryOp,
    Value, ValueDef, ValueType, resolve,
};
use crate::heap::TypedKind;

const SPECIALIZATION_REGION_LIMIT: usize = 512;

mod analysis;
mod cost;
mod placement;
mod planning;
mod propose;
mod versioning;

use super::super::transforms::repair;
use analysis::{Requirement, closure_failure, closure_failure_with, conflicts, guarded_accesses};
pub(in crate::jit::semantic) use analysis::{Writes, requirements};
#[cfg(test)]
pub(in crate::jit::semantic) use planning::version;
pub(in crate::jit::semantic) use planning::{materialize_properties, version_regions};
use versioning::Speculation;

pub(in crate::jit::semantic) fn version_string_leaves(
    graph: &mut Graph,
    continuations: bool,
    notes: &mut Vec<String>,
) -> Result<Vec<RegionId>> {
    let initial = Cfg::new(graph);
    let mut sites = Vec::new();
    for &block in &initial.rpo {
        let Some(Terminator {
            kind: TerminatorKind::Branch(condition, fast, _),
            ..
        }) = &graph.blocks[block].terminator
        else {
            continue;
        };
        if !matches!(graph.defining_op(*condition), Some(Op::IsStringSplit(_))) {
            continue;
        }
        let Some((receiver, header)) = graph.blocks[fast.block].insts.iter().find_map(|&inst| {
            let Op::StringSplit(receiver, _) = graph.insts[inst].op else {
                return None;
            };
            let Some(Terminator {
                kind: TerminatorKind::Jump(target),
                ..
            }) = &graph.blocks[fast.block].terminator
            else {
                return None;
            };
            Some((receiver, target.block))
        }) else {
            continue;
        };
        let string_branch = initial.rpo.iter().any(|&branch| {
            let Some(Terminator {
                kind: TerminatorKind::Branch(check, yes, no),
                ..
            }) = &graph.blocks[branch].terminator
            else {
                return false;
            };
            matches!(graph.defining_op(*check), Some(Op::IsString(value)) if *value == receiver)
                && yes.block != no.block
                && yes.block != graph.entry
                && initial.preds[yes.block].len() == 1
                && initial.preds[yes.block][0] == branch
                && initial.dominates(yes.block, block)
        });
        if string_branch {
            sites.push((fast.block, header));
        }
    }
    let mut regions = Vec::new();
    for (preheader, header) in sites {
        let cfg = Cfg::new(graph);
        if !cfg.rpo.contains(&preheader) || !cfg.rpo.contains(&header) {
            continue;
        }
        if !matches!(&graph.blocks[preheader].terminator, Some(Terminator { kind: TerminatorKind::Jump(target), .. }) if target.block == header)
        {
            continue;
        }
        let blocks: Vec<_> = cfg
            .rpo
            .iter()
            .copied()
            .filter(|&block| cfg.dominates(header, block))
            .collect();
        let cost: usize = blocks
            .iter()
            .map(|&block| graph.blocks[block].insts.len())
            .sum();
        if cost > SPECIALIZATION_REGION_LIMIT {
            continue;
        }
        let mut exits: Vec<_> = blocks
            .iter()
            .copied()
            .filter(|&block| cold_exit(graph, block).is_some())
            .collect();
        let exits_true: Vec<_> = blocks
            .iter()
            .copied()
            .filter(|&block| {
                let Some(Terminator {
                    kind: TerminatorKind::Branch(condition, yes, no),
                    ..
                }) = &graph.blocks[block].terminator
                else {
                    return false;
                };
                if yes.block == no.block {
                    return false;
                }
                match graph.defining_op(*condition) {
                    Some(Op::IsUninitialized(_)) => graph.cold.contains(&yes.block),
                    Some(Op::IsUndefined(_)) => true,
                    Some(Op::Binary(BinaryOperator::StrictEquality, left, right)) => {
                        [left, right].into_iter().any(|&value| {
                            matches!(
                                graph.defining_op(value),
                                Some(Op::Constant(Constant::Undefined))
                            )
                        })
                    }
                    _ => false,
                }
            })
            .collect();
        if exits.is_empty() && exits_true.is_empty() {
            continue;
        }
        for &block in &cfg.preds[preheader] {
            if matches!(&graph.blocks[block].terminator, Some(Terminator { kind: TerminatorKind::Branch(condition, yes, _), .. })
                if yes.block == preheader && matches!(graph.defining_op(*condition), Some(Op::IsStringSplit(_))))
            {
                exits.push(block);
            }
        }
        let original = graph.clone();
        let mut candidate = graph.clone();
        let speculation = Speculation {
            exits: &exits,
            exits_true: &exits_true,
            ..Default::default()
        };
        let mut values = original.values.keys().map(|value| (value, value)).collect();
        let mut bailouts = versioning::Bailouts::new(&original);
        let mut candidate_notes = Vec::new();
        let added = candidate.blocks.len();
        versioning::materialize(
            &mut candidate,
            &blocks,
            Some((preheader, header)),
            &speculation,
            &[],
            &mut values,
            &mut bailouts,
            &mut candidate_notes,
        )?;
        let mut copy: FxHashSet<_> = blocks.iter().copied().collect();
        copy.extend(candidate.blocks.keys().skip(added));
        let mut function = FunctionState::new(candidate, Vec::new(), true);
        super::super::js::simplify(&mut function);
        let mut candidate = function.graph;
        candidate.verify()?;
        let reachable: FxHashSet<_> = Cfg::new(&candidate).rpo.into_iter().collect();
        let copy: FxHashSet<_> = copy.intersection(&reachable).copied().collect();
        if let Some(reason) = closure_failure(&candidate, &copy, &[], &FxHashMap::default()) {
            notes.push(format!(
                "string split continuation not versioned: guarded copy {reason}"
            ));
            continue;
        }
        notes.push(format!(
            "string split continuation versioned after native identity check; {} guarded exits",
            exits.len() + exits_true.len()
        ));
        let region = candidate.regions.push(Region {
            blocks: copy,
            fallback: None,
        });
        versioning::finish(
            &mut candidate,
            &original,
            None,
            continuations,
            &[region],
            &mut candidate_notes,
        )?;
        notes.extend(candidate_notes);
        *graph = candidate;
        regions.push(region);
        break;
    }
    Ok(regions)
}

fn cold_exit(graph: &Graph, block: Block) -> Option<Block> {
    let Some(Terminator {
        kind: TerminatorKind::Branch(_, _, otherwise),
        ..
    }) = &graph.blocks[block].terminator
    else {
        return None;
    };
    graph
        .cold
        .contains(&otherwise.block)
        .then_some(otherwise.block)
}

fn invariant_guards(
    graph: &Graph,
    inside: &FxHashSet<Block>,
    scan: &[Block],
    probes: &[Inst],
) -> Vec<Block> {
    let probed: FxHashSet<Value> = probes
        .iter()
        .filter_map(|inst| graph.insts[*inst].result())
        .collect();
    let invariant =
        |value: Value| !inside.contains(&graph.def_block(value)) || probed.contains(&value);
    scan.iter()
        .copied()
        .filter(|&block| {
            cold_exit(graph, block).is_some()
                && matches!(
                    &graph.blocks[block].terminator,
                    Some(Terminator { kind: TerminatorKind::Branch(condition, ..), .. })
                        if invariant(*condition)
                            || graph.defining_op(*condition).unwrap().operands_iter().all(invariant)
                )
        })
        .collect()
}

fn typed_entry(
    graph: &Graph,
    body: &FxHashSet<Block>,
    available: &FxHashSet<Value>,
    facts: &Facts,
    candidates: &[(Value, guards::Fact)],
) -> Vec<(Value, Option<TypedKind>)> {
    let mut typed = FxHashMap::default();
    for &block in body {
        for &inst in &graph.blocks[block].insts {
            if let Op::GetIndex(object, key) | Op::TypedIndex(object, key, _) = graph.insts[inst].op
                && (!body.contains(&graph.def_block(object)) || available.contains(&object))
                && facts.of(key).within(NUMERIC)
                && let Some(element) = candidates.iter().find_map(|&(value, fact)| match fact {
                    guards::Fact::Typed(element) if value == object => Some(element),
                    _ => None,
                })
            {
                typed.insert(object, element);
            }
        }
    }
    let mut typed: Vec<_> = typed.into_iter().collect();
    typed.sort_by_key(|&(object, _)| object);
    typed
}

pub(in crate::jit::semantic) fn element_guess(graph: &Graph, object: Value) -> Option<TypedKind> {
    graph
        .blocks
        .iter()
        .flat_map(|(_, block)| block.insts.iter())
        .find_map(|&inst| match graph.insts[inst].op {
            Op::ReadHint(value, _, crate::jit::profile::ReadType::Typed(element), _)
                if value == object =>
            {
                Some(element)
            }
            _ => None,
        })
        .or_else(|| element_guess_at(graph, object, 0))
}

fn element_guess_at(graph: &Graph, object: Value, depth: usize) -> Option<TypedKind> {
    if depth > 4 {
        return None;
    }
    if let ValueDef::Param(block, index) = graph.values[object] {
        return graph
            .blocks
            .values()
            .filter_map(|data| data.terminator.as_ref())
            .flat_map(|term| term.kind.targets())
            .filter(|target| target.block == block)
            .find_map(|target| element_guess_at(graph, *target.args.get(index)?, depth + 1));
    }
    if let Some(Op::ConstructorResult(result, _)) = graph.defining_op(object) {
        return element_guess_at(graph, *result, depth + 1);
    }
    let constructor = match graph.defining_op(object)? {
        Op::Construct { callee, .. }
        | Op::ConstructResolved { callee, .. }
        | Op::ConstructorCall { callee, .. } => *callee,
        Op::Call { callee, .. } => match graph.defining_op(*callee)? {
            Op::Get(constructor, name) if name == "from" || name == "of" => *constructor,
            _ => return None,
        },
        _ => return None,
    };
    let Op::GlobalRead(name) = graph.defining_op(constructor)? else {
        return None;
    };
    Some(match name.as_str() {
        "Uint8Array" => TypedKind::Uint8,
        "Uint16Array" => TypedKind::Uint16,
        "Uint32Array" => TypedKind::Uint32,
        "Int8Array" => TypedKind::Int8,
        "Int16Array" => TypedKind::Int16,
        "Int32Array" => TypedKind::Int32,
        "Float32Array" => TypedKind::Float32,
        "Float64Array" => TypedKind::Float64,
        _ => return None,
    })
}

/// Index reads on `objects` inside `region` whose key is a loop counter starting at a
/// non-negative integer, stepping by one and tested against a bound defined before the region,
/// mapped to their receiver and that bound.
fn bounded_reads(
    graph: &Graph,
    cfg: &Cfg,
    region: &FxHashSet<Block>,
    objects: &[Value],
    available: &FxHashSet<Value>,
    facts: &Facts,
) -> FxHashMap<Inst, (Value, Value)> {
    let mut counters = FxHashMap::default();
    for (header, body) in natural_loops(graph, cfg) {
        if !region.contains(&header) {
            continue;
        }
        let Some(Terminator {
            kind: TerminatorKind::Branch(condition, inside, outside),
            ..
        }) = &graph.blocks[header].terminator
        else {
            continue;
        };
        let Some(
            &(Op::Binary(BinaryOperator::LessThan, counter, bound)
            | Op::GuardedCompare {
                operator: BinaryOperator::LessThan,
                left: counter,
                right: bound,
                ..
            }),
        ) = graph.defining_op(*condition)
        else {
            continue;
        };
        if !body.contains(&inside.block)
            || body.contains(&outside.block)
            || region.contains(&graph.def_block(bound)) && !available.contains(&bound)
            || !facts.of(bound).within(ValueType::NUMBER)
            || !counts_up(graph, cfg, header, &body, counter)
        {
            continue;
        }
        counters.insert(counter, (inside.block, bound));
    }
    let mut reads = FxHashMap::default();
    for &block in region {
        for &inst in &graph.blocks[block].insts {
            if let Op::GetIndex(object, key) | Op::TypedIndex(object, key, _) = graph.insts[inst].op
                && objects.contains(&object)
                && let Some(&(inside, bound)) = counters.get(&key)
                && cfg.dominates(inside, block)
            {
                reads.insert(inst, (object, bound));
            }
        }
    }
    reads
}

fn counts_up(
    graph: &Graph,
    cfg: &Cfg,
    header: Block,
    body: &FxHashSet<Block>,
    counter: Value,
) -> bool {
    let Some(index) = graph.blocks[header]
        .params
        .iter()
        .position(|&param| param == counter)
    else {
        return false;
    };
    cfg.preds[header].iter().all(|&pred| {
        let terminator = graph.blocks[pred].terminator.as_ref().unwrap();
        terminator
            .kind
            .targets()
            .filter(|target| target.block == header)
            .all(|target| {
                let arg = target.args[index];
                match graph.defining_op(arg) {
                    Some(Op::Constant(Constant::Number(start))) => {
                        !body.contains(&pred) && *start >= 0.0 && start.fract() == 0.0
                    }
                    Some(&Op::Unary(UnaryOp::Increment, operand)) => operand == counter,
                    Some(&Op::Binary(BinaryOperator::Addition, operand, step)) => {
                        operand == counter
                            && matches!(
                                graph.defining_op(step),
                                Some(Op::Constant(Constant::Number(step))) if *step == 1.0
                            )
                    }
                    _ => false,
                }
            })
    })
}

fn numeric_use(op: &Op, index: usize) -> bool {
    match op {
        Op::GuardedCompare { .. } => index < 2,
        Op::Binary(operator, ..) => !matches!(
            operator,
            BinaryOperator::In
                | BinaryOperator::Instanceof
                | BinaryOperator::Equality
                | BinaryOperator::Inequality
                | BinaryOperator::StrictEquality
                | BinaryOperator::StrictInequality
        ),
        Op::Unary(operator, _) => !matches!(operator, UnaryOp::Not | UnaryOp::Typeof),
        Op::ToNumeric(_) | Op::CallNative(..) => true,
        Op::GetIndex(..)
        | Op::TypedIndex(..)
        | Op::ArrayGet(..)
        | Op::ArrayLoad(..)
        | Op::SetIndex(..) => index == 1,
        _ => false,
    }
}

fn entry_assumes(
    graph: &Graph,
    inside: &FxHashSet<Block>,
    probes: &[Inst],
    facts: &Facts,
    candidates: &[(Value, guards::Fact)],
) -> Vec<(Value, guards::Fact)> {
    let probed: FxHashSet<Value> = probes
        .iter()
        .filter_map(|inst| graph.insts[*inst].result())
        .collect();
    let mut checks: Vec<_> = candidates
        .iter()
        .copied()
        .filter(|&(value, fact)| {
            if !matches!(fact, guards::Fact::Type(_)) {
                return false;
            }
            let desired = if fact == guards::Fact::Type(ValueType::INTEGER) {
                ValueType::INTEGER
            } else {
                NUMERIC
            };
            !facts.of(value).within(desired)
                && (!inside.contains(&graph.def_block(value)) || probed.contains(&value))
        })
        .collect();
    checks.sort_by_key(|&(value, _)| value);
    checks.dedup_by_key(|&mut (value, _)| value);
    checks
}

pub(in crate::jit::semantic) fn hoist(function: &mut FunctionState, region: RegionId) {
    function.ensure_facts();
    let scope = function.graph.regions[region].blocks.clone();
    if super::super::transforms::licm::hoist(
        &mut function.graph,
        &function.facts,
        Some(&scope),
        Some(&mut function.notes),
    ) {
        function.invalidate_values();
    }
}

fn probe_reads(
    graph: &Graph,
    inside: &FxHashSet<Block>,
    written: &FxHashSet<Block>,
    scan: &[Block],
    inputs: bool,
    facts: &Facts,
) -> Vec<Inst> {
    let writes = Writes::with_facts(graph, written, facts);
    let mut reads = Vec::new();
    let mut probed = FxHashSet::default();
    for &block in scan {
        for &inst in &graph.blocks[block].insts {
            let data = &graph.insts[inst];
            let input = matches!(data.op, Op::PropertyCache(..))
                || inputs && matches!(data.op, Op::Argument(_) | Op::This);
            if (input
                || matches!(
                    data.op,
                    Op::Get(..)
                        | Op::ShapedGet(..)
                        | Op::ProbeProperty(..)
                        | Op::ProbeShapeProperty(..)
                        | Op::GetIndex(..)
                        | Op::ArrayGet(..)
                        | Op::GlobalRead(_)
                ))
                && data.op.operands_iter().all(|value| {
                    !inside.contains(&graph.def_block(value)) || probed.contains(&value)
                })
                && (input || writes.preserves(graph, inst))
            {
                reads.push(inst);
                probed.extend(data.result());
            }
        }
    }
    reads
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxc_allocator::Allocator;
    use oxc_ast::ast::Statement;
    use oxc_parser::Parser;
    use oxc_span::SourceType;

    fn empty_graph() -> Graph {
        let allocator = Allocator::default();
        let parsed =
            Parser::new(&allocator, "function f(){return 0;}", SourceType::script()).parse();
        let Statement::FunctionDeclaration(function) = &parsed.program.body[0] else {
            panic!()
        };
        let semantic = oxc_semantic::SemanticBuilder::new_compiler()
            .build(&parsed.program)
            .semantic;
        super::super::super::build::lower(function, semantic.scoping(), &FxHashSet::default())
            .unwrap()
    }

    fn heap(base: Value, field: Field) -> Access {
        Access {
            location: Locations::HEAP,
            base: Some(base),
            field,
        }
    }

    fn global(name: &str) -> Access {
        Access {
            location: Locations::GLOBAL,
            base: None,
            field: Field::Named(name.into()),
        }
    }

    #[test]
    fn append_writes_preserve_properties_but_invalidate_elements_and_length() {
        for spread in [false, true] {
            let mut graph = empty_graph();
            let block = graph.entry;
            let span = graph.span;
            let array = graph.append_value(block, span, Op::Argument(0));
            let value = graph.append_value(block, span, Op::Argument(1));
            graph.append(block, span, Op::ArrayAppend(array, value, spread), false);
            let scope = Cfg::new(&graph).rpo.into_iter().collect();
            let writes = Writes::collect(&graph, &scope);
            assert!(writes.clobbers(&heap(array, Field::Elements)));
            assert!(writes.clobbers(&heap(array, Field::Length)));
            assert!(!writes.clobbers(&heap(array, Field::Named("allows".into()))));
            assert!(!writes.clobbers(&global("g_Map")));
        }
    }

    #[test]
    fn shape_exclusion_requires_a_live_positive_fact() {
        for writes_shape in [false, true] {
            let mut graph = empty_graph();
            let entry = graph.entry;
            let span = graph.span;
            let object = graph.append_value(entry, span, Op::Argument(0));
            let check = graph.append_value(entry, span, Op::IsShape(object, 2));
            let yes = graph.blocks.push(BlockData::default());
            let no = graph.blocks.push(BlockData::default());
            let join = graph.blocks.push(BlockData::default());
            graph.terminate(
                entry,
                span,
                TerminatorKind::Branch(check, Target::new(yes), Target::new(no)),
            );
            if writes_shape {
                graph.append(yes, span, Op::Set(object, "extra".into(), object), false);
            }
            let (positive, _) = graph.append(yes, span, Op::IsShape(object, 1), true);
            let (negative, _) = graph.append(no, span, Op::IsShape(object, 1), true);
            for block in [yes, no] {
                graph.terminate(block, span, TerminatorKind::Jump(Target::new(join)));
            }
            let (merged, result) = graph.append(join, span, Op::IsShape(object, 1), true);
            graph.terminate(join, span, TerminatorKind::Return(result.unwrap()));
            graph.verify().unwrap();
            let facts = Facts::compute(&graph);
            let cfg = Cfg::new(&graph);
            let solved = super::super::super::available::solve(
                &graph,
                &facts,
                &cfg,
                FxHashSet::default(),
                false,
            )
            .unwrap();
            assert_eq!(solved.folded.contains(&(positive, false)), !writes_shape);
            assert!(
                !solved
                    .folded
                    .iter()
                    .any(|&(inst, _)| inst == negative || inst == merged)
            );
        }
    }

    #[test]
    fn shape_checks_survive_slot_stores_but_not_shape_writes_on_the_object() {
        let mut graph = empty_graph();
        let block = graph.entry;
        let span = graph.span;
        let object = graph.append_value(block, span, Op::Argument(0));
        let value = graph.append_value(block, span, Op::Argument(1));
        let first = graph.append(block, span, Op::IsShape(object, 1), true).1;
        graph.append(
            block,
            span,
            Op::ShapedSet(object, "x".into(), 0, value),
            false,
        );
        let (second, _) = graph.append(block, span, Op::IsShape(object, 1), true);
        graph.append(block, span, Op::Set(object, "y".into(), value), false);
        let (third, _) = graph.append(block, span, Op::IsShape(object, 1), true);
        graph.terminate(block, span, TerminatorKind::Return(first.unwrap()));
        graph.verify().unwrap();
        let facts = Facts::compute(&graph);
        let cfg = Cfg::new(&graph);
        let solved = super::super::super::available::solve(
            &graph,
            &facts,
            &cfg,
            FxHashSet::default(),
            false,
        )
        .unwrap();
        let reused: Vec<Inst> = solved.reused.iter().map(|&(inst, _)| inst).collect();
        assert!(reused.contains(&second));
        assert!(!reused.contains(&third));
    }

    #[test]
    fn pack_mutations_do_not_clobber_javascript_properties() {
        let mut graph = empty_graph();
        let block = graph.entry;
        let span = graph.span;
        let pack = graph.append_value(block, span, Op::NewPack);
        let value = graph.append_value(block, span, Op::Argument(0));
        graph.append(block, span, Op::PackAppend(pack, value, false), false);
        graph.append(block, span, Op::PackRelease(pack), false);
        let scope = Cfg::new(&graph).rpo.into_iter().collect();
        let writes = Writes::collect(&graph, &scope);
        assert!(writes.clobbers(&heap(pack, Field::Packs)));
        assert!(!writes.clobbers(&heap(value, Field::Length)));
        assert!(!writes.clobbers(&heap(value, Field::Named("allows".into()))));
        assert!(!writes.clobbers(&global("g_Map")));
    }

    #[test]
    fn fresh_block_parameters_keep_aliases_and_reject_external_inputs_and_escapes() {
        for external in [false, true] {
            let mut graph = empty_graph();
            let entry = graph.entry;
            let span = graph.span;
            let object = graph.append_value(entry, span, Op::NewObject);
            let other = graph.append_value(entry, span, Op::NewObject);
            let argument = graph.append_value(entry, span, Op::Argument(0));
            let check = graph.append_value(entry, span, Op::IsUndefined(argument));
            let left = graph.blocks.push(BlockData::default());
            let right = graph.blocks.push(BlockData::default());
            let merge = graph.blocks.push(BlockData::default());
            let receiver = graph.add_param(merge);
            graph.terminate(
                entry,
                span,
                TerminatorKind::Branch(check, Target::new(left), Target::new(right)),
            );
            for (block, value) in [
                (left, object),
                (right, if external { argument } else { object }),
            ] {
                graph.terminate(
                    block,
                    span,
                    TerminatorKind::Jump(Target {
                        block: merge,
                        args: vec![value],
                    }),
                );
            }
            graph.append(merge, span, Op::Set(receiver, "x".into(), argument), false);
            graph.terminate(merge, span, TerminatorKind::Return(receiver));
            graph.verify().unwrap();
            let scope = Cfg::new(&graph).rpo.into_iter().collect();
            let writes = Writes::collect(&graph, &scope);
            assert!(writes.clobbers(&heap(object, Field::Named("x".into()))));
            assert_eq!(
                writes.clobbers(&heap(other, Field::Named("x".into()))),
                external
            );
            let mut after = graph.clone();
            after.append(merge, span, Op::GlobalWrite("leak".into(), receiver), false);
            let writes = Writes::collect(&after, &scope);
            assert_eq!(
                writes.clobbers(&heap(other, Field::Named("x".into()))),
                external
            );
            for value in [object, receiver] {
                let mut before = graph.clone();
                before.append(entry, span, Op::GlobalWrite("leak".into(), value), false);
                let writes = Writes::collect(&before, &scope);
                assert!(writes.clobbers(&heap(other, Field::Named("x".into()))));
            }
        }
    }

    #[test]
    fn branches_decided_by_a_dominating_edge_on_the_same_condition_fold() {
        for inverted in [false, true] {
            let mut graph = empty_graph();
            let entry = graph.entry;
            let span = graph.span;
            let argument = graph.append_value(entry, span, Op::Argument(0));
            let check = graph.append_value(entry, span, Op::IsUndefined(argument));
            let [yes, no, inner_yes, inner_no] =
                std::array::from_fn(|_| graph.blocks.push(BlockData::default()));
            graph.terminate(
                entry,
                span,
                TerminatorKind::Branch(check, Target::new(yes), Target::new(no)),
            );
            let repeated = if inverted {
                graph.append_value(yes, span, Op::Unary(UnaryOp::Not, check))
            } else {
                check
            };
            graph.terminate(
                yes,
                span,
                TerminatorKind::Branch(repeated, Target::new(inner_yes), Target::new(inner_no)),
            );
            for block in [no, inner_yes, inner_no] {
                let zero = graph.append_value(block, span, Op::Constant(Constant::Number(0.0)));
                graph.terminate(block, span, TerminatorKind::Return(zero));
            }
            graph.verify().unwrap();
            let mut function = FunctionState::new(graph, Vec::new(), true);
            super::super::super::js::simplify(&mut function);
            let graph = function.graph;
            let reachable: FxHashSet<Block> = Cfg::new(&graph).rpo.into_iter().collect();
            assert_eq!(reachable.contains(&inner_yes), !inverted);
            assert_eq!(reachable.contains(&inner_no), inverted);
        }
    }

    #[test]
    fn summaries_apply_only_while_the_guarded_global_is_unwritten() {
        use super::super::super::super::effects::{Control, Memory};
        let code = cranelift_module::FuncId::from_u32(0);
        for written in [false, true] {
            let mut graph = empty_graph();
            graph.summaries = std::sync::Arc::new(FxHashMap::from_iter([(
                code,
                super::super::super::CallSummary {
                    memory: Memory::default(),
                    control: Control::default(),
                    requires: vec![("f".into(), code)],
                },
            )]));
            let entry = graph.entry;
            let span = graph.span;
            let probe = graph.append_value(entry, span, Op::ProbeGlobal("f".into()));
            let check = graph.append_value(entry, span, Op::IsCode(probe, code));
            let body = graph.blocks.push(BlockData::default());
            let exit = graph.blocks.push(BlockData::default());
            graph.terminate(
                entry,
                span,
                TerminatorKind::Branch(check, Target::new(body), Target::new(exit)),
            );
            let value = graph.append_value(body, span, Op::Constant(Constant::Number(1.0)));
            if written {
                graph.append(body, span, Op::GlobalWrite("f".into(), value), false);
            }
            let (call, result) = graph.append(
                body,
                span,
                Op::Call {
                    callee: probe,
                    receiver: None,
                    args: Vec::new(),
                    target: None,
                },
                true,
            );
            graph.terminate(body, span, TerminatorKind::Return(result.unwrap()));
            let zero = graph.append_value(exit, span, Op::Constant(Constant::Number(0.0)));
            graph.terminate(exit, span, TerminatorKind::Return(zero));
            graph.verify().unwrap();
            let facts = Facts::compute(&graph);
            assert_eq!(facts.memory(call).is_empty(), !written);
        }
    }

    #[test]
    fn fresh_aliases_survive_loop_backedges() {
        let mut graph = empty_graph();
        let entry = graph.entry;
        let span = graph.span;
        let object = graph.append_value(entry, span, Op::NewObject);
        let header = graph.blocks.push(BlockData::default());
        let receiver = graph.add_param(header);
        graph.terminate(
            entry,
            span,
            TerminatorKind::Jump(Target {
                block: header,
                args: vec![object],
            }),
        );
        graph.terminate(
            header,
            span,
            TerminatorKind::Jump(Target {
                block: header,
                args: vec![receiver],
            }),
        );
        let other = graph.append_value(entry, span, Op::NewObject);
        let one = graph.append_value(header, span, Op::Constant(Constant::Number(1.0)));
        graph.append(header, span, Op::Set(receiver, "x".into(), one), false);
        graph.verify().unwrap();
        let scope = FxHashSet::from_iter([entry, header]);
        let writes = Writes::collect(&graph, &scope);
        assert!(writes.clobbers(&heap(object, Field::Named("x".into()))));
        assert!(writes.clobbers(&heap(receiver, Field::Named("x".into()))));
        assert!(!writes.clobbers(&heap(other, Field::Named("x".into()))));
        graph.append(
            header,
            span,
            Op::GlobalWrite("leak".into(), receiver),
            false,
        );
        let writes = Writes::collect(&graph, &scope);
        assert!(writes.clobbers(&heap(other, Field::Named("x".into()))));
    }

    #[test]
    fn numeric_and_typed_guards_share_one_loop_copy() {
        let allocator = Allocator::default();
        let source = "function f(width){const grid=new Float64Array(4);let sum=0;for(let y=0;y<2;++y){for(let x=0;x<2;++x)sum+=grid[y*width+x];}return sum;}";
        let parsed = Parser::new(&allocator, source, SourceType::script()).parse();
        let Statement::FunctionDeclaration(function) = &parsed.program.body[0] else {
            panic!()
        };
        let semantic = oxc_semantic::SemanticBuilder::new_compiler()
            .build(&parsed.program)
            .semantic;
        let analysis =
            super::super::super::analyze(function, semantic.scoping(), &FxHashSet::default())
                .unwrap();
        let graph = &analysis.graph;
        let loops = natural_loops(graph, &Cfg::new(graph));
        let outer = loops
            .iter()
            .filter(|(header, _)| {
                !loops
                    .iter()
                    .any(|(other, body)| other != header && body.contains(header))
            })
            .count();
        assert_eq!(outer, 1);
        assert!(
            super::super::super::continuation::bodies(graph)
                .iter()
                .any(|body| !natural_loops(&body.graph, &Cfg::new(&body.graph)).is_empty())
        );
        assert_eq!(
            Cfg::new(graph)
                .rpo
                .iter()
                .flat_map(|&block| &graph.blocks[block].insts)
                .filter(|&&inst| matches!(graph.insts[inst].op, Op::IsTyped(..)))
                .count(),
            1
        );
        assert!(
            analysis
                .notes
                .iter()
                .any(|note| note.contains("combined entry guards"))
        );
        graph.verify().unwrap();
    }

    #[test]
    fn fresh_constructor_result_remains_private_only_until_terminal_return() {
        let allocator = Allocator::default();
        let parsed =
            Parser::new(&allocator, "function f(){return 0;}", SourceType::script()).parse();
        let Statement::FunctionDeclaration(function) = &parsed.program.body[0] else {
            panic!()
        };
        let semantic = oxc_semantic::SemanticBuilder::new_compiler()
            .build(&parsed.program)
            .semantic;
        let mut graph =
            super::super::super::build::lower(function, semantic.scoping(), &FxHashSet::default())
                .unwrap();
        let block = graph.entry;
        let span = graph.span;
        let prototype = graph.append_value(block, span, Op::Constant(Constant::Null));
        let object = graph.append_value(block, span, Op::NewInstance(prototype));
        let undefined = graph.append_value(block, span, Op::Constant(Constant::Undefined));
        let result = graph.append_value(block, span, Op::ConstructorResult(undefined, object));
        graph.terminate(block, span, TerminatorKind::Return(result));
        let other = graph.append_value(block, span, Op::NewObject);
        let one = graph.append_value(block, span, Op::Constant(Constant::Number(1.0)));
        graph.append(block, span, Op::Set(object, "x".into(), one), false);
        let scope = FxHashSet::from_iter([block]);
        let writes = Writes::collect(&graph, &scope);
        assert!(!writes.clobbers(&heap(other, Field::Named("x".into()))));
        let position = graph.blocks[block]
            .insts
            .iter()
            .position(|&inst| matches!(graph.insts[inst].op, Op::Set(..)))
            .unwrap();
        graph.insert(
            block,
            position,
            span,
            Op::GlobalWrite("leak".into(), result),
            false,
        );
        let writes = Writes::collect(&graph, &scope);
        assert!(writes.clobbers(&heap(other, Field::Named("x".into()))));
    }
}

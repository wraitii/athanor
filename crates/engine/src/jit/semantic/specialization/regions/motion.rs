use super::hoisting::{AssumeCfg, Motion, RecoveryPoint};
use super::*;
use crate::jit::semantic::{ArrayLayout, JsOp};

pub(super) struct JsMotion;

impl Motion for JsMotion {
    fn dependencies(
        &self,
        graph: &Graph,
        facts: &Facts,
        group: &AssumeCfg,
        point: &RecoveryPoint,
    ) -> Option<Vec<Inst>> {
        let insertion = point.before.unwrap_or(graph.entry);
        let source = group.before?;
        if !facts.cfg.dominates(insertion, source) {
            return None;
        }
        let body: FxHashSet<_> = match point.before {
            Some(_) => natural_loop(&facts.cfg, point.target.block)?,
            None => facts.cfg.rpo.iter().copied().collect(),
        };
        let written: FxHashSet<_> = body
            .iter()
            .copied()
            .filter(|block| !graph.cold.contains(block) && !group.blocks.contains(block))
            .collect();
        let conditions: Vec<_> = group
            .blocks
            .iter()
            .flat_map(|&block| &graph.blocks[block].insts)
            .filter_map(|&inst| guards::implies(&graph.insts[inst].op))
            .collect();
        let overwrites = written
            .iter()
            .flat_map(|&block| &graph.blocks[block].insts)
            .filter(|&&inst| {
                let Op::Javascript(JsOp::Set(object, ..)) = graph.insts[inst].op else {
                    return false;
                };
                let Some(access) = graph.property_accesses.get(&inst) else {
                    return false;
                };
                let &[layout] = access.shapes.as_slice() else {
                    return false;
                };
                guards::holds(
                    graph,
                    &facts.cfg,
                    graph.insts[inst].block,
                    object,
                    guards::Fact::Shape(layout.shape),
                )
            })
            .copied()
            .collect();
        let writes = Writes::with_facts(graph, &written, facts).preserving_shapes(&overwrites);
        if group.dependencies.iter().any(|read| writes.clobbers(read)) {
            return None;
        }
        let mut dependencies = Vec::new();
        let mut seen = FxHashSet::default();
        for &block in &group.blocks {
            for &inst in &graph.blocks[block].insts {
                let op = &graph.insts[inst].op;
                let required = match *op {
                    // The maximum is only proven to be reached within the chosen loop.
                    Op::Javascript(JsOp::IsArray(_, ArrayLayout::Packed)) => return None,
                    Op::Javascript(JsOp::AsString(value)) => {
                        Some((value, guards::Fact::Type(ValueType::STRING)))
                    }
                    Op::Javascript(JsOp::AsNumber(value)) => {
                        Some((value, guards::Fact::Type(ValueType::NUMBER)))
                    }
                    Op::Javascript(JsOp::AsInteger(value)) => {
                        Some((value, guards::Fact::Type(ValueType::INTEGER)))
                    }
                    _ => None,
                };
                if let Some((value, required)) = required
                    && !required
                        .mask()
                        .is_some_and(|types| facts.of(value).within(types))
                    && !conditions
                        .iter()
                        .any(|&(checked, fact)| checked == value && fact.implies(required))
                    && !guards::holds(graph, &facts.cfg, insertion, value, required)
                {
                    return None;
                }
                if !matches!(
                    op,
                    Op::Argument(_)
                        | Op::Javascript(
                            JsOp::This
                                | JsOp::PropertyCache(..)
                                | JsOp::GuardValid(_)
                                | JsOp::GuardCommit(_)
                        )
                        | Op::Observe(..)
                        | Op::ObserveMethod(..)
                        | Op::Javascript(
                            JsOp::AsNumber(_)
                                | JsOp::AsInteger(_)
                                | JsOp::AsString(_)
                                | JsOp::AsObject(_)
                        )
                ) && !fixed_length(graph, facts, insertion, op)
                    && !writes.preserves(graph, inst)
                {
                    return None;
                }
                for value in op.operands_iter() {
                    collect(
                        graph,
                        facts,
                        group,
                        insertion,
                        &body,
                        value,
                        &mut seen,
                        &mut dependencies,
                    )?;
                }
            }
            for value in graph.blocks[block]
                .terminator
                .as_ref()
                .unwrap()
                .kind
                .operands_iter()
            {
                collect(
                    graph,
                    facts,
                    group,
                    insertion,
                    &body,
                    value,
                    &mut seen,
                    &mut dependencies,
                )?;
            }
        }
        Some(dependencies)
    }
}

/// Typed arrays never change length, so once one is known typed its length read cannot be
/// clobbered.
fn fixed_length(graph: &Graph, facts: &Facts, insertion: Block, op: &Op) -> bool {
    let &Op::Javascript(JsOp::IndexedLength(object)) = op else {
        return false;
    };
    guards::holds(
        graph,
        &facts.cfg,
        insertion,
        object,
        guards::Fact::Typed(None),
    )
}

fn collect(
    graph: &Graph,
    facts: &Facts,
    group: &AssumeCfg,
    insertion: Block,
    body: &FxHashSet<Block>,
    value: Value,
    seen: &mut FxHashSet<Value>,
    dependencies: &mut Vec<Inst>,
) -> Option<()> {
    let block = graph.def_block(value);
    if group.blocks.contains(&block) || !seen.insert(value) {
        return Some(());
    }
    if !body.contains(&block) && facts.cfg.dominates(block, insertion) {
        return Some(());
    }
    if block == insertion && insertion != graph.entry {
        return Some(());
    }
    let ValueDef::Result(inst, _) = graph.values[value] else {
        return None;
    };
    if !matches!(
        graph.insts[inst].op,
        Op::Argument(_) | Op::Javascript(JsOp::This | JsOp::PropertyCache(..))
    ) && !facts.is_pure(inst)
    {
        return None;
    }
    for operand in graph.insts[inst].op.operands_iter() {
        collect(
            graph,
            facts,
            group,
            insertion,
            body,
            operand,
            seen,
            dependencies,
        )?;
    }
    dependencies.push(inst);
    Some(())
}

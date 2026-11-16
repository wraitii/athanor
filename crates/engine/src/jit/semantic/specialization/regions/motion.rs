use super::hoisting::{AssumeCfg, Motion, RecoveryPoint};
use super::*;

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
            Some(_) => {
                natural_loops(graph, &facts.cfg)
                    .into_iter()
                    .find(|(header, _)| *header == point.target.block)?
                    .1
            }
            None => facts.cfg.rpo.iter().copied().collect(),
        };
        let written: FxHashSet<_> = body
            .iter()
            .copied()
            .filter(|block| !graph.cold.contains(block) && !group.blocks.contains(block))
            .collect();
        if written
            .iter()
            .flat_map(|&block| &graph.blocks[block].insts)
            .any(|&inst| facts.memory(inst).reenters())
        {
            return None;
        }
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
                let Op::Set(object, ..) = graph.insts[inst].op else {
                    return false;
                };
                let Some(access) = graph.property_accesses.get(&inst) else {
                    return false;
                };
                let &[(shape, _)] = access.shapes.as_slice() else {
                    return false;
                };
                guards::holds(
                    graph,
                    &facts.cfg,
                    graph.insts[inst].block,
                    object,
                    guards::Fact::Shape(shape),
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
                    Op::AsNumber(value) => Some((value, guards::Fact::Type(ValueType::NUMBER))),
                    Op::AsInteger(value) => Some((value, guards::Fact::Type(ValueType::INTEGER))),
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
                        | Op::This
                        | Op::PropertyCache(..)
                        | Op::GuardValid(_)
                        | Op::GuardCommit(_)
                        | Op::Observe(..)
                        | Op::AsNumber(_)
                        | Op::AsInteger(_)
                        | Op::AsObject(_)
                ) && !writes.preserves(graph, inst)
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
        Op::Argument(_) | Op::This | Op::PropertyCache(..)
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

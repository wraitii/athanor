use crate::FxHashMap;

use super::super::repr::{Step, steps};
use super::super::{Facts, Graph, Op, Repr, ValueDef, ValueType};

fn lossless(from: Repr, through: Repr) -> bool {
    matches!(
        (from, through),
        (Repr::I32, Repr::I64 | Repr::F64 | Repr::Boxed) | (Repr::Boolean, Repr::Boxed)
    )
}

pub(in crate::jit::semantic) fn simplify(graph: &mut Graph, facts: &Facts) {
    let mut aliases = FxHashMap::default();
    let mut removed = Vec::new();
    for &block in &facts.cfg.rpo {
        for &inst in &graph.blocks[block].insts {
            let Op::Convert(mut source, to) = graph.insts[inst].op else {
                continue;
            };
            while let Some(&alias) = aliases.get(&source) {
                source = alias;
            }
            let replacement = if graph.repr(source) == to {
                Some(source)
            } else if let Some(Op::Convert(original, through)) = graph.defining_op(source)
                && graph.repr(*original) == to
                && lossless(to, *through)
                && (to != Repr::Boolean || facts.of(*original) == ValueType::BOOLEAN)
            {
                Some(*original)
            } else {
                None
            };
            if let Some(replacement) = replacement {
                aliases.insert(graph.insts[inst].result().unwrap(), replacement);
                removed.push(inst);
            }
        }
    }
    graph.replace_uses(&aliases);
    for inst in removed {
        graph.remove_inst(inst);
    }
}

pub(in crate::jit::semantic) fn split(graph: &mut Graph, facts: &Facts) {
    let mut aliases = FxHashMap::default();
    for &block in &facts.cfg.rpo {
        for inst in graph.blocks[block].insts.clone() {
            let Op::Convert(source, to) = graph.insts[inst].op else {
                continue;
            };
            let steps = steps(graph.repr(source), to);
            if matches!(steps.as_slice(), [] | [Step::Convert(_)]) {
                continue;
            }
            let data = &graph.insts[inst];
            let (span, source_name) = (data.span, data.source_name);
            let at = graph.blocks[block]
                .insts
                .iter()
                .position(|&i| i == inst)
                .unwrap();
            let mut value = source;
            for (offset, step) in steps.into_iter().enumerate() {
                let (op, repr) = step.op(value);
                let (inserted, result) = graph.insert(block, at + offset, span, op, true);
                graph.insts[inserted].source_name = source_name;
                value = result.unwrap();
                graph.reprs[value] = Some(repr);
            }
            aliases.insert(graph.insts[inst].result().unwrap(), value);
            graph.detach_inst(inst);
        }
    }
    graph.replace_uses(&aliases);
}

pub(in crate::jit::semantic) fn hoist(graph: &mut Graph, facts: &Facts) {
    let loops = super::loops::natural_loops(graph, &facts.cfg);
    if loops.is_empty() {
        return;
    }
    let heap = ValueType::STRING
        .union(ValueType::OBJECT)
        .union(ValueType::OTHER);
    let mut available = FxHashMap::default();
    let mut aliases = FxHashMap::default();
    let mut removed = Vec::new();
    for &block in &facts.cfg.rpo {
        for inst in graph.blocks[block].insts.clone() {
            let Op::Convert(source, to) = graph.insts[inst].op else {
                continue;
            };
            let definition = graph.def_block(source);
            let invariant = !facts.of(source).intersects(heap)
                && loops
                    .iter()
                    .any(|(_, body)| body.contains(&block) && !body.contains(&definition));
            let destination = if invariant { definition } else { block };
            let result = graph.insts[inst].result().unwrap();
            if let Some(&previous) = available.get(&(destination, source, to)) {
                aliases.insert(result, previous);
                removed.push(inst);
                continue;
            }
            available.insert((destination, source, to), result);
            if destination == block {
                continue;
            }
            graph.blocks[block].insts.retain(|&other| other != inst);
            let at = match graph.values[source] {
                ValueDef::Result(source_inst, _) => {
                    graph.blocks[destination]
                        .insts
                        .iter()
                        .position(|&other| other == source_inst)
                        .unwrap()
                        + 1
                }
                ValueDef::Param(..) => 0,
            };
            graph.blocks[destination].insts.insert(at, inst);
            graph.insts[inst].block = destination;
        }
    }
    graph.replace_uses(&aliases);
    for inst in removed {
        graph.remove_inst(inst);
    }
}

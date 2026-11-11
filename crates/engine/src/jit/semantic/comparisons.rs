use std::collections::HashMap;

use oxc_syntax::operator::BinaryOperator;

use super::{Block, Constant, Op, Types, Value};

pub(super) fn expose(function: &mut super::FunctionState) {
    function.refresh_facts();
    let graph = &mut function.graph;
    let facts = &function.facts;
    let cfg = &facts.cfg;
    let mut checks: HashMap<Value, Vec<(Block, Value)>> = HashMap::new();
    let mut aliases = HashMap::new();
    let mut removed = Vec::new();
    for &block in &cfg.rpo {
        for inst in graph.blocks[block].insts.clone() {
            let data = graph.insts[inst].clone();
            if let Op::IsNumber(value) = data.op {
                let earlier = checks.get(&value).and_then(|checks| {
                    checks
                        .iter()
                        .find(|(definition, _)| cfg.dominates(*definition, block))
                });
                if let Some(&(_, earlier)) = earlier {
                    aliases.insert(data.result.unwrap(), earlier);
                    removed.push(inst);
                } else {
                    checks
                        .entry(value)
                        .or_default()
                        .push((block, data.result.unwrap()));
                }
                continue;
            }
            let Op::Binary(operator, left, right) = data.op else {
                continue;
            };
            if matches!(
                operator,
                BinaryOperator::StrictEquality | BinaryOperator::StrictInequality
            ) {
                let undefined = |value| {
                    let types = facts.of(value);
                    types.intersects(Types::UNDEFINED) && types.within(Types::UNDEFINED)
                };
                let value = if undefined(left) {
                    Some(right)
                } else if undefined(right) {
                    Some(left)
                } else {
                    None
                };
                if let Some(value) = value {
                    if operator == BinaryOperator::StrictEquality {
                        graph.insts[inst].op = Op::IsUndefined(value);
                    } else {
                        let position = graph.blocks[block]
                            .insts
                            .iter()
                            .position(|&current| current == inst)
                            .unwrap();
                        let (check, result) =
                            graph.insert(block, position, data.span, Op::IsUndefined(value), true);
                        graph.insts[check].source_name = data.source_name;
                        graph.insts[inst].op = Op::Unary(super::UnaryOp::Not, result.unwrap());
                    }
                    continue;
                }
            }
            if !matches!(
                operator,
                BinaryOperator::LessThan
                    | BinaryOperator::LessEqualThan
                    | BinaryOperator::GreaterThan
                    | BinaryOperator::GreaterEqualThan
            ) || [left, right].iter().all(|&value| {
                facts
                    .of(value)
                    .within(Types::NUMBER.union(Types::UNDEFINED))
            }) || [left, right]
                .iter()
                .any(|&value| !facts.of(value).intersects(Types::NUMBER))
            {
                continue;
            }
            let mut numbers = [left, right];
            for (number, value) in numbers.iter_mut().zip([left, right]) {
                if let Some(&(_, earlier)) = checks.get(&value).and_then(|checks| {
                    checks
                        .iter()
                        .find(|(definition, _)| cfg.dominates(*definition, block))
                }) {
                    *number = earlier;
                    continue;
                }
                let op = if facts.of(value).within(Types::NUMBER) {
                    Op::Constant(Constant::Boolean(true))
                } else {
                    Op::IsNumber(value)
                };
                let position = graph.blocks[block]
                    .insts
                    .iter()
                    .position(|&current| current == inst)
                    .unwrap();
                let (check, result) = graph.insert(block, position, data.span, op, true);
                graph.insts[check].source_name = data.source_name;
                *number = result.unwrap();
                checks.entry(value).or_default().push((block, *number));
            }
            graph.insts[inst].op = Op::GuardedCompare {
                operator,
                left,
                right,
                numbers,
            };
        }
    }
    graph.replace_uses(&aliases);
    for inst in removed {
        graph.remove_inst(inst);
    }
}

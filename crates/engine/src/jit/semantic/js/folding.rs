use crate::FxHashMap;

use oxc_syntax::operator::BinaryOperator;

use super::super::transforms::{self, Fold, Folder};
use super::super::{
    Block, Cfg, Constant, Facts, FunctionState, Graph, Inst, Op, Terminator, TerminatorKind,
    UnaryOp, Value, ValueType,
};
use super::NUMERIC;

pub(in crate::jit::semantic) fn simplify(function: &mut FunctionState) {
    transforms::simplify(function, &mut Rules);
}

struct Rules;

impl Folder for Rules {
    fn refine(&mut self, graph: &mut Graph, cfg: &Cfg) -> bool {
        refine_string_guards(graph, cfg)
    }

    fn fold(&mut self, graph: &Graph, facts: &Facts, inst: Inst) -> Option<Fold> {
        let data = &graph.insts[inst];
        let boolean = |value| Some(Fold::Op(Op::Constant(Constant::Boolean(value))));
        match data.op {
            Op::GuardedCompare {
                operator,
                left,
                right,
                ..
            } if [left, right]
                .iter()
                .all(|&value| facts.of(value).within(NUMERIC)) =>
            {
                Some(Fold::Op(Op::Binary(operator, left, right)))
            }
            Op::IsExactPosition(key, position)
                if facts.of(key).within(ValueType::INTEGER)
                    && matches!(graph.defining_op(position), Some(Op::ArrayPosition(value)) if *value == key) =>
            {
                boolean(true)
            }
            Op::BooleanAnd(left, right) => {
                for (constant, other) in [(left, right), (right, left)] {
                    if let Some(Op::Constant(Constant::Boolean(value))) =
                        graph.defining_op(constant)
                    {
                        return if *value {
                            Some(Fold::Alias(other))
                        } else {
                            boolean(false)
                        };
                    }
                }
                None
            }
            Op::IsIndexed(value) if facts.indexed(value) => boolean(true),
            Op::IsString(value) => boolean(known(facts.of(value), ValueType::STRING)?),
            Op::IsUndefined(value) => boolean(known(facts.of(value), ValueType::UNDEFINED)?),
            Op::Binary(
                operator @ (BinaryOperator::StrictEquality | BinaryOperator::StrictInequality),
                left,
                right,
            ) => {
                let (left, right) = (facts.of(left), facts.of(right));
                let compared = if known(left, ValueType::UNDEFINED) == Some(true) {
                    right
                } else if known(right, ValueType::UNDEFINED) == Some(true) {
                    left
                } else {
                    return None;
                };
                let equal = known(compared, ValueType::UNDEFINED)?;
                boolean(equal == (operator == BinaryOperator::StrictEquality))
            }
            Op::IsNumber(value) if facts.of(value).within(ValueType::NUMBER) => boolean(true),
            Op::IsNumber(value) if !facts.of(value).intersects(ValueType::NUMBER) => boolean(false),
            Op::ToNumeric(value) | Op::Unary(UnaryOp::Plus, value) | Op::AsNumber(value)
                if data.result().is_some() && facts.of(value).within(ValueType::NUMBER) =>
            {
                Some(Fold::Alias(value))
            }
            Op::ConstructorResult(returned, instance) if data.result().is_some() => {
                let types = facts.of(returned);
                if types == ValueType::default() {
                    None
                } else if !types.intersects(ValueType::OBJECT) {
                    Some(Fold::Alias(instance))
                } else if types.within(ValueType::OBJECT) {
                    Some(Fold::Alias(returned))
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

fn known(types: ValueType, kind: ValueType) -> Option<bool> {
    if types.within(kind) && types.intersects(kind) {
        Some(true)
    } else if types != ValueType::default() && !types.intersects(kind) {
        Some(false)
    } else {
        None
    }
}

fn refine_string_guards(graph: &mut Graph, cfg: &Cfg) -> bool {
    let mut checks: FxHashMap<(bool, Value), Vec<(Block, bool)>> = FxHashMap::default();
    for &block in &cfg.rpo {
        let Some(Terminator {
            kind: TerminatorKind::Branch(condition, yes, no),
            ..
        }) = &graph.blocks[block].terminator
        else {
            continue;
        };
        if yes.block == no.block {
            continue;
        }
        let key = match graph.defining_op(*condition) {
            Some(Op::IsString(value)) => (false, *value),
            Some(Op::IsStringSplit(value)) => (true, *value),
            _ => continue,
        };
        for (target, result) in [(yes.block, true), (no.block, false)] {
            if target != graph.entry
                && cfg.preds[target].len() == 1
                && cfg.preds[target][0] == block
            {
                checks.entry(key).or_default().push((target, result));
            }
        }
    }
    if checks.is_empty() {
        return false;
    }
    let mut changed = false;
    for &block in &cfg.rpo {
        for inst in graph.blocks[block].insts.clone() {
            let key = match graph.insts[inst].op {
                Op::IsString(value) => (false, value),
                Op::IsStringSplit(value) => (true, value),
                _ => continue,
            };
            let Some((_, result)) = checks
                .get(&key)
                .and_then(|edges| edges.iter().find(|(entry, _)| cfg.dominates(*entry, block)))
            else {
                continue;
            };
            graph.set_op(inst, Op::Constant(Constant::Boolean(*result)));
            changed = true;
        }
    }
    changed
}

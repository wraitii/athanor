use std::collections::HashMap;

use oxc_syntax::operator::BinaryOperator;

use super::super::transforms::{self, Fold, Folder};
use super::super::{
    Block, Cfg, Constant, Facts, Graph, Inst, Op, Terminator, TerminatorKind, Types, UnaryOp, Value,
};
use super::NUMERIC;

pub(in crate::jit::semantic) fn simplify(graph: &mut Graph) {
    transforms::simplify(graph, &mut Rules);
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
            Op::IsIndexed(value) if facts.indexed(value) => boolean(true),
            Op::IsString(value) => boolean(known(facts.of(value), Types::STRING)?),
            Op::IsUndefined(value) => boolean(known(facts.of(value), Types::UNDEFINED)?),
            Op::Binary(
                operator @ (BinaryOperator::StrictEquality | BinaryOperator::StrictInequality),
                left,
                right,
            ) => {
                let (left, right) = (facts.of(left), facts.of(right));
                let compared = if known(left, Types::UNDEFINED) == Some(true) {
                    right
                } else if known(right, Types::UNDEFINED) == Some(true) {
                    left
                } else {
                    return None;
                };
                let equal = known(compared, Types::UNDEFINED)?;
                boolean(equal == (operator == BinaryOperator::StrictEquality))
            }
            Op::IsNumber(value) if facts.of(value).within(Types::NUMBER) => boolean(true),
            Op::IsNumber(value) if !facts.of(value).intersects(Types::NUMBER) => boolean(false),
            Op::ToNumeric(value) | Op::Unary(UnaryOp::Plus, value) | Op::AsNumber(value)
                if data.result.is_some() && facts.of(value).within(Types::NUMBER) =>
            {
                Some(Fold::Alias(value))
            }
            Op::ConstructorResult(returned, instance) if data.result.is_some() => {
                let types = facts.of(returned);
                if types == Types::default() {
                    None
                } else if !types.intersects(Types::OBJECT) {
                    Some(Fold::Alias(instance))
                } else if types.within(Types::OBJECT) {
                    Some(Fold::Alias(returned))
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

fn known(types: Types, kind: Types) -> Option<bool> {
    if types.within(kind) && types.intersects(kind) {
        Some(true)
    } else if types != Types::default() && !types.intersects(kind) {
        Some(false)
    } else {
        None
    }
}

fn refine_string_guards(graph: &mut Graph, cfg: &Cfg) -> bool {
    let mut checks: HashMap<(bool, Value), Vec<(Block, bool)>> = HashMap::new();
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
        for &inst in &graph.blocks[block].insts {
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
            graph.insts[inst].op = Op::Constant(Constant::Boolean(*result));
            changed = true;
        }
    }
    changed
}

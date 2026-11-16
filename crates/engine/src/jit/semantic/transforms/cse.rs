use crate::FxHashMap;

use cranelift_codegen::entity::SecondaryMap;
use cranelift_codegen::ir::Type;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use oxc_syntax::operator::BinaryOperator;

use super::super::{Block, Constant, Facts, Graph, Op, Repr, Value, low_level, resolve};

#[derive(Clone, PartialEq, Eq, Hash)]
enum Expression {
    Undefined,
    Null,
    Boolean(bool),
    Number(u64),
    Integer(Type, u64),
    Float64(u64),
    ScalarBinary(low_level::Binary, Value, Value),
    ScalarUnary(low_level::Unary, Type, Value),
    Convert(Value, Repr),
    ToInt32(Value),
    NumericFloat(Value),
    IntCompare(IntCC, Value, Value),
    FloatCompare(FloatCC, Value, Value),
    SaturatingInteger(Value),
    Position(Value),
    ExactPosition(Value, Value),
    InBounds(Value, Value),
    BooleanAnd(Value, Value),
    This,
    Argument(usize),
    Binary(BinaryOperator, Value, Value),
}

fn expression(
    graph: &Graph,
    op: &Op,
    pure: bool,
    aliases: &SecondaryMap<Value, Option<Value>>,
) -> Option<Expression> {
    let at = |value: &Value| resolve(aliases, *value);
    Some(match op {
        Op::Constant(Constant::Undefined) => Expression::Undefined,
        Op::Constant(Constant::Null) => Expression::Null,
        Op::Constant(Constant::Boolean(value)) => Expression::Boolean(*value),
        Op::Constant(Constant::Number(value)) => Expression::Number(value.to_bits()),
        Op::LowLevel(low_level::Op::Integer { ty, bits }) => Expression::Integer(*ty, *bits),
        Op::LowLevel(low_level::Op::Float64(bits)) => Expression::Float64(*bits),
        Op::NumericFloat(value) => {
            let value = at(value);
            let value = match graph.defining_op(value) {
                Some(Op::AsInteger(input))
                    if matches!(graph.defining_op(at(input)), Some(Op::SaturatingInteger(_))) =>
                {
                    at(input)
                }
                _ => value,
            };
            Expression::NumericFloat(value)
        }
        Op::SaturatingInteger(value) => Expression::SaturatingInteger(at(value)),
        Op::ToInt32(value) => Expression::ToInt32(at(value)),
        Op::Convert(value, repr) if pure => Expression::Convert(at(value), *repr),
        Op::LowLevel(low_level::Op::Binary { op, left, right }) if pure => {
            Expression::ScalarBinary(*op, at(left), at(right))
        }
        Op::LowLevel(low_level::Op::Unary { op, ty, value }) if pure => {
            Expression::ScalarUnary(*op, *ty, at(value))
        }
        Op::LowLevel(low_level::Op::IntCompare {
            condition,
            left,
            right,
        }) if pure => Expression::IntCompare(*condition, at(left), at(right)),
        Op::LowLevel(low_level::Op::FloatCompare {
            condition,
            left,
            right,
        }) if pure => Expression::FloatCompare(*condition, at(left), at(right)),
        // The dominating conversion has already succeeded, including its internal error check.
        Op::ArrayPosition(value) => Expression::Position(at(value)),
        Op::IsExactPosition(key, position) => Expression::ExactPosition(at(key), at(position)),
        Op::BooleanAnd(left, right) => Expression::BooleanAnd(at(left), at(right)),
        Op::IsInBounds(position, length) => Expression::InBounds(at(position), at(length)),
        Op::This => Expression::This,
        Op::Argument(index) => Expression::Argument(*index),
        Op::Binary(operator, left, right)
        | Op::GuardedCompare {
            operator,
            left,
            right,
            ..
        } if pure => Expression::Binary(*operator, at(left), at(right)),
        _ => return None,
    })
}

pub(in crate::jit::semantic) fn eliminate(
    graph: &mut Graph,
    facts: &Facts,
    mut notes: Option<&mut Vec<String>>,
) -> bool {
    let cfg = &facts.cfg;
    let mut available = FxHashMap::default();
    let mut inserted = Vec::new();
    let mut scopes: Vec<(Block, usize)> = Vec::new();
    let mut aliases = SecondaryMap::new();
    let mut reused = Vec::new();
    for block in cfg.dominator_preorder() {
        while let Some(&(parent, start)) = scopes.last() {
            if cfg.dominates(parent, block) {
                break;
            }
            for key in inserted.drain(start..) {
                available.remove(&key);
            }
            scopes.pop();
        }
        scopes.push((block, inserted.len()));
        if graph.cold.contains(&block) {
            continue;
        }
        for &inst in &graph.blocks[block].insts {
            let data = &graph.insts[inst];
            let [result] = data.results.as_slice() else {
                continue;
            };
            let Some(expression) = expression(graph, &data.op, facts.is_pure(inst), &aliases)
            else {
                continue;
            };
            let key = (expression, graph.repr(*result));
            if let Some(&value) = available.get(&key) {
                aliases[*result] = Some(value);
                reused.push((inst, value));
            } else {
                available.insert(key.clone(), *result);
                inserted.push(key);
            }
        }
    }
    if reused.is_empty() {
        return false;
    }
    graph.replace_uses(&aliases);
    for &(inst, value) in &reused {
        if let Some(notes) = &mut notes {
            notes.push(format!("{} reuses {value}", graph.note_inst(inst)));
        }
    }
    graph.remove_insts(reused.iter().map(|&(inst, _)| inst));
    true
}

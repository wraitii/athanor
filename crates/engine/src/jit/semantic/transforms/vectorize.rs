use super::super::low_level::{Binary, Op as LowOp, Unary, Vector};
use super::super::{
    Block, BlockData, Cfg, Graph, Inst, MemoryFormat, MemoryOp, Op, Repr, Target, TerminatorKind,
    Value, ValueDef,
};
use super::loops::{natural_loops, preheader};
use crate::{FxHashMap, FxHashSet};
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{Type, types};

const LANES: i64 = 4;
const VECTOR: Type = types::I32X4;

/// Invariant values are computed once before the vector loop and scalars only for its first lane,
/// while vectors hold four consecutive iterations.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Invariant,
    Scalar,
    Vector,
}

enum Step {
    Inst(Inst),
    Guard(Value, bool),
}

struct Plan {
    preheader: Block,
    header: Block,
    index: Value,
    limit: Value,
    condition: IntCC,
    steps: Vec<Step>,
    kinds: FxHashMap<Value, Kind>,
    disjoint: Vec<(Value, Value)>,
}

/// Runs four iterations at once of innermost counted loops over typed elements. A prologue checks
/// that four iterations remain, the loop's invariant guards hold and its stores cannot overlap its
/// loads; otherwise, and for the remaining iterations, the scalar loop runs unchanged.
pub(in crate::jit::semantic) fn vectorize(graph: &mut Graph) -> bool {
    let cfg = Cfg::new(graph);
    let loops = natural_loops(graph, &cfg);
    let plans: Vec<Plan> = loops
        .iter()
        .filter(|(header, body)| {
            !loops
                .iter()
                .any(|(other, _)| other != header && body.contains(other))
        })
        .filter_map(|(header, body)| plan(graph, &cfg, *header, body))
        .collect();
    let changed = !plans.is_empty();
    for plan in plans {
        apply(graph, plan);
    }
    changed
}

fn plan(graph: &Graph, cfg: &Cfg, header: Block, body: &FxHashSet<Block>) -> Option<Plan> {
    let preheader = preheader(graph, cfg, header, body)?;
    let &[index] = graph.blocks[header].params.as_slice() else {
        return None;
    };
    let TerminatorKind::Branch(test, inside, _) = &graph.blocks[header].terminator.as_ref()?.kind
    else {
        return None;
    };
    if graph.blocks[header].insts.len() != 1 || !body.contains(&inside.block) {
        return None;
    }
    let Some(Op::LowLevel(LowOp::IntCompare {
        condition,
        left,
        right: limit,
    })) = graph.defining_op(*test)
    else {
        return None;
    };
    if *left != index
        || width(graph, index) != Some(types::I64)
        || width(graph, *limit) != Some(types::I64)
        || body.contains(&graph.def_block(*limit))
        || !matches!(
            condition,
            IntCC::SignedLessThan | IntCC::SignedLessThanOrEqual
        )
    {
        return None;
    }
    let mut kinds: FxHashMap<Value, Kind> = FxHashMap::from_iter([(index, Kind::Scalar)]);
    let kind = |kinds: &FxHashMap<Value, Kind>, value: Value| {
        if body.contains(&graph.def_block(value)) {
            kinds.get(&value).copied()
        } else {
            Some(Kind::Invariant)
        }
    };
    let mut steps = Vec::new();
    let mut loads: Vec<Value> = Vec::new();
    let mut stores = Vec::new();
    let mut visited = FxHashSet::from_iter([header]);
    let mut block = inside.block;
    if !inside.args.is_empty() {
        return None;
    }
    loop {
        if !visited.insert(block) || !graph.blocks[block].params.is_empty() {
            return None;
        }
        for &inst in &graph.blocks[block].insts {
            let data = &graph.insts[inst];
            let operands: Option<Vec<Kind>> = data
                .op
                .operands_iter()
                .map(|value| kind(&kinds, value))
                .collect();
            let operands = operands?;
            let result = match &data.op {
                Op::Memory(MemoryOp::IndexedLoad {
                    base,
                    index: offset,
                    format,
                }) => {
                    loads.push(*base);
                    match (operands[0], operands[1]) {
                        (Kind::Invariant, Kind::Invariant) => Kind::Invariant,
                        (Kind::Invariant, Kind::Scalar)
                            if lanes(*format)
                                && stride(graph, &kinds, index, *offset) == Some(1) =>
                        {
                            Kind::Vector
                        }
                        _ => return None,
                    }
                }
                Op::Memory(MemoryOp::IndexedStore {
                    base,
                    index: offset,
                    format,
                    ..
                }) => {
                    stores.push(*base);
                    if operands[0] != Kind::Invariant
                        || operands[1] != Kind::Scalar
                        || operands[2] == Kind::Scalar
                        || !lanes(*format)
                        || stride(graph, &kinds, index, *offset) != Some(1)
                    {
                        return None;
                    }
                    steps.push(Step::Inst(inst));
                    continue;
                }
                Op::LowLevel(LowOp::Load { .. }) if operands[0] == Kind::Invariant => {
                    Kind::Invariant
                }
                Op::LowLevel(LowOp::Integer { .. } | LowOp::Float64(_)) => Kind::Invariant,
                Op::LowLevel(
                    LowOp::Unary { .. }
                    | LowOp::Binary { .. }
                    | LowOp::IntCompare { .. }
                    | LowOp::FloatCompare { .. }
                    | LowOp::Select { .. }
                    | LowOp::ExtractBits { .. }
                    | LowOp::InRange { .. }
                    | LowOp::TestMask { .. },
                )
                | Op::Convert(..)
                    if !operands.contains(&Kind::Vector) =>
                {
                    if operands.contains(&Kind::Scalar) {
                        Kind::Scalar
                    } else {
                        Kind::Invariant
                    }
                }
                Op::LowLevel(op) if lanewise(graph, &kinds, op, &operands) => Kind::Vector,
                Op::Convert(value, Repr::I32 | Repr::I64) if exact(graph, &kinds, *value) => {
                    Kind::Vector
                }
                _ => return None,
            };
            kinds.insert(data.result()?, result);
            steps.push(Step::Inst(inst));
        }
        match &graph.blocks[block].terminator.as_ref()?.kind {
            TerminatorKind::Jump(next) if next.block == header => {
                let &[step] = next.args.as_slice() else {
                    return None;
                };
                let Some(Op::LowLevel(LowOp::Binary {
                    op: Binary::Add,
                    left,
                    right,
                })) = graph.defining_op(step)
                else {
                    return None;
                };
                if *left != index || super::super::ranges::integer_literal(graph, *right) != Some(1)
                {
                    return None;
                }
                break;
            }
            TerminatorKind::Jump(next) if next.args.is_empty() && body.contains(&next.block) => {
                block = next.block;
            }
            TerminatorKind::Branch(guard, yes, no) => {
                if kind(&kinds, *guard) != Some(Kind::Invariant) {
                    return None;
                }
                let stay = match (body.contains(&yes.block), body.contains(&no.block)) {
                    (true, false) => true,
                    (false, true) => false,
                    _ => return None,
                };
                let next = if stay { yes } else { no };
                if !next.args.is_empty() {
                    return None;
                }
                steps.push(Step::Guard(*guard, stay));
                block = next.block;
            }
            _ => return None,
        }
    }
    if visited.len() != body.len() || stores.is_empty() {
        return None;
    }
    // Each typed array owns its storage, so distinct data pointers never overlap.
    let mut disjoint = Vec::new();
    for (position, &store) in stores.iter().enumerate() {
        for &other in loads.iter().chain(&stores[position + 1..]) {
            if store == other {
                return None;
            }
            if !disjoint.contains(&(store, other)) {
                disjoint.push((store, other));
            }
        }
    }
    Some(Plan {
        preheader,
        header,
        index,
        limit: *limit,
        condition: *condition,
        steps,
        kinds,
        disjoint,
    })
}

fn lanes(format: MemoryFormat) -> bool {
    !matches!(format, MemoryFormat::F32 | MemoryFormat::F64)
}

/// Lanes hold the low 32 bits of each iteration's value, which these operations compute exactly.
fn lanewise(graph: &Graph, kinds: &FxHashMap<Value, Kind>, op: &LowOp, operands: &[Kind]) -> bool {
    if operands.contains(&Kind::Scalar) {
        return false;
    }
    match *op {
        LowOp::Unary {
            op: Unary::Reduce,
            ty,
            ..
        } => ty.is_int() && ty.bits() <= 32,
        LowOp::Unary {
            op: op @ (Unary::SignExtend | Unary::ZeroExtend),
            value,
            ..
        } => match graph.defining_op(value) {
            _ if width(graph, value) == Some(types::I32) => true,
            Some(Op::Memory(MemoryOp::IndexedLoad { format, .. })) => {
                kinds.get(&value) == Some(&Kind::Vector)
                    && (op == Unary::SignExtend) == signed(*format)
            }
            _ => false,
        },
        LowOp::Binary {
            op:
                Binary::Add
                | Binary::Subtract
                | Binary::Multiply
                | Binary::And
                | Binary::Or
                | Binary::Xor,
            left,
            ..
        } => width(graph, left).is_some_and(|ty| ty.is_int()),
        LowOp::Binary {
            op: Binary::ShiftLeft | Binary::ShiftRightSigned | Binary::ShiftRightUnsigned,
            left,
            right,
        } => {
            operands[0] == Kind::Vector
                && super::super::ranges::integer_literal(graph, right).is_some()
                && width(graph, left) == Some(types::I32)
        }
        _ => false,
    }
}

/// Lanes of full-width integers and of loaded elements hold their exact values' low bits, which a
/// representation change keeps.
fn exact(graph: &Graph, kinds: &FxHashMap<Value, Kind>, value: Value) -> bool {
    if kinds.get(&value) != Some(&Kind::Vector) {
        return false;
    }
    matches!(width(graph, value), Some(types::I32 | types::I64))
        || matches!(
            graph.defining_op(value),
            Some(Op::Memory(MemoryOp::IndexedLoad { .. }))
        )
}

fn signed(format: MemoryFormat) -> bool {
    matches!(
        format,
        MemoryFormat::I8 | MemoryFormat::I16 | MemoryFormat::I32
    )
}

fn width(graph: &Graph, value: Value) -> Option<Type> {
    let ValueDef::Result(inst, _) = graph.values[value] else {
        return graph.reprs[value].map(Repr::machine);
    };
    match graph.insts[inst].op {
        Op::LowLevel(
            LowOp::Integer { ty, .. } | LowOp::Unary { ty, .. } | LowOp::Load { ty, .. },
        ) => Some(ty),
        Op::LowLevel(LowOp::Binary { op, left, .. }) if !op.is_boolean() => width(graph, left),
        Op::Memory(MemoryOp::IndexedLoad { format, .. }) => Some(format.repr().machine()),
        _ => graph.reprs[value].map(Repr::machine),
    }
}

fn stride(
    graph: &Graph,
    kinds: &FxHashMap<Value, Kind>,
    index: Value,
    value: Value,
) -> Option<i64> {
    if value == index {
        return Some(1);
    }
    if kinds
        .get(&value)
        .is_none_or(|&kind| kind == Kind::Invariant)
    {
        return Some(0);
    }
    let Some(Op::LowLevel(LowOp::Binary { op, left, right })) = graph.defining_op(value) else {
        return None;
    };
    if width(graph, value) != Some(types::I64) {
        return None;
    }
    let (left, right) = (
        stride(graph, kinds, index, *left)?,
        stride(graph, kinds, index, *right)?,
    );
    match op {
        Binary::Add => Some(left + right),
        Binary::Subtract => Some(left - right),
        _ => None,
    }
}

fn apply(graph: &mut Graph, plan: Plan) {
    let Plan {
        preheader,
        header,
        index,
        limit,
        condition,
        steps,
        kinds,
        disjoint,
    } = plan;
    let first = graph.blocks.len();
    let TerminatorKind::Jump(entry) = &graph.blocks[preheader].terminator.as_ref().unwrap().kind
    else {
        unreachable!()
    };
    let start = entry.args[0];
    let scalar = |value| Target {
        block: header,
        args: vec![value],
    };
    let mut block = graph.blocks.push(BlockData::default());
    graph.blocks[preheader].terminator.as_mut().unwrap().kind =
        TerminatorKind::Jump(Target::new(block));
    let enter = remaining(graph, block, start, limit, condition);
    block = guard(graph, block, enter, true, scalar(start));
    let mut values: FxHashMap<Value, Value> = FxHashMap::default();
    for step in &steps {
        match *step {
            Step::Inst(inst) => {
                let invariant = graph.insts[inst]
                    .result()
                    .is_some_and(|result| kinds[&result] == Kind::Invariant);
                if invariant {
                    super::unroll::clone_inst(graph, inst, block, &mut values);
                }
            }
            Step::Guard(condition, stay) => {
                let condition = resolve(&values, condition);
                block = guard(graph, block, condition, stay, scalar(start));
            }
        }
    }
    for (store, load) in disjoint {
        let (store, load) = (resolve(&values, store), resolve(&values, load));
        let distinct = low(
            graph,
            block,
            LowOp::IntCompare {
                condition: IntCC::NotEqual,
                left: store,
                right: load,
            },
        );
        block = guard(graph, block, distinct, true, scalar(start));
    }
    let prologue = block;
    let head = graph.blocks.push(BlockData::default());
    let lanes = graph.add_param(head);
    graph.reprs[lanes] = graph.reprs[index];
    let body = graph.blocks.push(BlockData::default());
    let more = remaining(graph, head, lanes, limit, condition);
    graph.terminate(
        head,
        graph.span,
        TerminatorKind::Branch(more, Target::new(body), scalar(lanes)),
    );
    values.insert(index, lanes);
    let mut splats: FxHashMap<Value, Value> = FxHashMap::default();
    for step in &steps {
        let Step::Inst(inst) = *step else {
            continue;
        };
        let op = graph.insts[inst].op.clone();
        let kind = graph.insts[inst].result().map(|result| kinds[&result]);
        if kind == Some(Kind::Invariant) {
            continue;
        }
        if kind == Some(Kind::Scalar) {
            super::unroll::clone_inst(graph, inst, body, &mut values);
            continue;
        }
        let mut lane = |graph: &mut Graph, value: Value| {
            if kinds.get(&value) == Some(&Kind::Vector) {
                return values[&value];
            }
            *splats
                .entry(value)
                .or_insert_with(|| splat(graph, prologue, resolve(&values, value)))
        };
        let vector = match op {
            Op::Memory(MemoryOp::IndexedStore {
                base,
                index: offset,
                value,
                format,
            }) => {
                let value = lane(graph, value);
                let store = Vector::Store {
                    base: resolve(&values, base),
                    index: resolve(&values, offset),
                    value,
                    format,
                };
                graph.append(body, graph.span, Op::LowLevel(LowOp::Vector(store)), false);
                continue;
            }
            Op::Memory(MemoryOp::IndexedLoad {
                base,
                index: offset,
                format,
            }) => Vector::Load {
                ty: VECTOR,
                base: resolve(&values, base),
                index: resolve(&values, offset),
                format,
            },
            Op::LowLevel(LowOp::Unary { value, .. }) | Op::Convert(value, _) => {
                let result = graph.insts[inst].result().unwrap();
                values.insert(result, values[&value]);
                continue;
            }
            Op::LowLevel(LowOp::Binary { op, left, right }) => {
                let left = lane(graph, left);
                let right = match op {
                    Binary::ShiftLeft | Binary::ShiftRightSigned | Binary::ShiftRightUnsigned => {
                        resolve(&values, right)
                    }
                    _ => lane(graph, right),
                };
                Vector::Binary { op, left, right }
            }
            _ => unreachable!(),
        };
        let result = graph.insts[inst].result().unwrap();
        let value = graph.append_value(body, graph.span, Op::LowLevel(LowOp::Vector(vector)));
        values.insert(result, value);
    }
    let step = low(
        graph,
        body,
        LowOp::Integer {
            ty: types::I64,
            bits: LANES as u64,
        },
    );
    let next = low(
        graph,
        body,
        LowOp::Binary {
            op: Binary::Add,
            left: lanes,
            right: step,
        },
    );
    graph.terminate(
        body,
        graph.span,
        TerminatorKind::Jump(Target {
            block: head,
            args: vec![next],
        }),
    );
    graph.terminate(
        prologue,
        graph.span,
        TerminatorKind::Jump(Target {
            block: head,
            args: vec![start],
        }),
    );
    super::super::dispatch::extend_regions(graph, header, first);
}

fn resolve(values: &FxHashMap<Value, Value>, value: Value) -> Value {
    values.get(&value).copied().unwrap_or(value)
}

fn low(graph: &mut Graph, block: Block, op: LowOp) -> Value {
    graph.append_value(block, graph.span, Op::LowLevel(op))
}

fn remaining(graph: &mut Graph, block: Block, at: Value, limit: Value, condition: IntCC) -> Value {
    let span = LANES - 1;
    let span = low(
        graph,
        block,
        LowOp::Integer {
            ty: types::I64,
            bits: span as u64,
        },
    );
    let last = low(
        graph,
        block,
        LowOp::Binary {
            op: Binary::Add,
            left: at,
            right: span,
        },
    );
    low(
        graph,
        block,
        LowOp::IntCompare {
            condition,
            left: last,
            right: limit,
        },
    )
}

fn guard(graph: &mut Graph, block: Block, condition: Value, stay: bool, fallback: Target) -> Block {
    let next = graph.blocks.push(BlockData::default());
    let (yes, no) = if stay {
        (Target::new(next), fallback)
    } else {
        (fallback, Target::new(next))
    };
    graph.terminate(
        block,
        graph.span,
        TerminatorKind::Branch(condition, yes, no),
    );
    next
}

fn splat(graph: &mut Graph, block: Block, value: Value) -> Value {
    let lane = VECTOR.lane_type();
    let value = match width(graph, value) {
        Some(ty) if ty == lane => value,
        Some(ty) if ty.bits() > lane.bits() => low(
            graph,
            block,
            LowOp::Unary {
                op: Unary::Reduce,
                ty: lane,
                value,
            },
        ),
        _ => low(
            graph,
            block,
            LowOp::Unary {
                op: Unary::SignExtend,
                ty: lane,
                value,
            },
        ),
    };
    low(
        graph,
        block,
        LowOp::Vector(Vector::Splat { ty: VECTOR, value }),
    )
}

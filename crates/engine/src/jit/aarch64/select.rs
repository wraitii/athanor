use crate::jit::semantic::JsOp;
use cranelift_codegen::entity::SecondaryMap;
use cranelift_codegen::ir::condcodes::{CondCode, FloatCC, IntCC};
use cranelift_codegen::ir::{ArgumentExtension, Type, types};

use super::Unsupported;
use super::encode::{
    self as e, Alu, Cond, Extend, Float, FloatOp, FloatUnary, Index, Logical, Mem, Shift, VectorOp,
    Width,
};
use super::inst::{self as a, CompareOperand, Condition, Mask, ShiftAmount};
use crate::jit::alloc::RegClass;
use crate::jit::semantic::Repr;
use crate::jit::semantic::low_level::Builder;
use crate::jit::semantic::{
    Block, Constant, Graph, Inst, MemoryFormat, MemoryOp, Op, TerminatorKind, Value, ValueDef,
    low_level,
};

#[derive(Clone, Copy, Debug)]
enum Address {
    Offset {
        base: Value,
        offset: i64,
    },
    Indexed {
        base: Value,
        index: Value,
        kind: Index,
        scaled: bool,
    },
}

#[derive(Clone, Copy)]
struct Operand {
    position: usize,
    value: Value,
    kind: Index,
    shift: u32,
}

fn signed_constant(graph: &Graph, value: Value) -> Option<i64> {
    let bits = constant(graph, value)?;
    let width = graph.reprs[value].map(Repr::machine)?.bits();
    Some(if width >= 64 {
        bits as i64
    } else {
        ((bits << (64 - width)) as i64) >> (64 - width)
    })
}

fn binary(graph: &Graph, inst: Inst) -> Option<a::Op> {
    let Op::LowLevel(low_level::Op::Binary { op, left, right }) = graph.insts[inst].op else {
        return None;
    };
    use low_level::Binary as B;
    if !matches!(op, B::Add | B::Subtract | B::And | B::Or | B::Xor) {
        return None;
    }
    let (x, value) = match signed_constant(graph, right) {
        Some(value) => (left, value),
        None if !matches!(op, B::Subtract) => (right, signed_constant(graph, left)?),
        None => return None,
    };
    let ty = graph.reprs[graph.insts[inst].results[0]]?.machine();
    let width = width(ty);
    let immediate = if matches!(op, B::Add | B::Subtract) {
        let amount = if matches!(op, B::Subtract) {
            value.checked_neg()?
        } else {
            value
        };
        if let Some((imm, shift)) = e::arith_imm(amount) {
            a::Immediate::Add(imm, shift)
        } else {
            let (imm, shift) = amount.checked_neg().and_then(e::arith_imm)?;
            a::Immediate::Sub(imm, shift)
        }
    } else {
        let logical = match op {
            B::And => Logical::And,
            B::Or => Logical::Orr,
            _ => Logical::Eor,
        };
        let bits = match width {
            Width::X => value as u64,
            Width::W => value as u64 & 0xffff_ffff,
        };
        e::logical_imm(logical, width, 0, 0, bits)?;
        a::Immediate::Logical(logical, bits)
    };
    Some(a::Op::AluImmediate {
        width,
        value: x,
        immediate,
    })
}

fn comparison(graph: &Graph, cond: IntCC, x: Value, y: Value) -> Condition {
    let is_constant = |value| constant(graph, value).is_some();
    let (cond, x, y) = if is_constant(x) && !is_constant(y) {
        (cond.swap_args(), y, x)
    } else {
        (cond, x, y)
    };
    let signed = matches!(
        cond,
        IntCC::SignedLessThan
            | IntCC::SignedLessThanOrEqual
            | IntCC::SignedGreaterThan
            | IntCC::SignedGreaterThanOrEqual
    );
    let extended = matches!(
        graph.reprs[x].map(Repr::machine),
        Some(types::I8 | types::I16)
    );
    let unsigned = constant(graph, y).map(|bits| bits as i64);
    let sign_extended = signed_constant(graph, y);
    let negatable = signed || (!extended && matches!(cond, IntCC::Equal | IntCC::NotEqual));
    let immediate = (if signed { sign_extended } else { unsigned })
        .and_then(e::arith_imm)
        .map(|(imm, shift)| CompareOperand::Subs(imm, shift))
        .or_else(|| {
            sign_extended
                .filter(|_| negatable)
                .and_then(i64::checked_neg)
                .and_then(e::arith_imm)
                .map(|(imm, shift)| CompareOperand::Adds(imm, shift))
        });
    let ty = graph.repr(x).machine();
    let extend = match (ty, signed) {
        (types::I8, false) => Some(Extend::Uxtb),
        (types::I8, true) => Some(Extend::Sxtb),
        (types::I16, false) => Some(Extend::Uxth),
        (types::I16, true) => Some(Extend::Sxth),
        _ => None,
    };
    Condition::Int {
        cond: int_cond(cond),
        width: width(ty),
        extend,
        left: x,
        right: immediate.unwrap_or(CompareOperand::Reg(y)),
    }
}

fn indexed_load(graph: &Graph, inst: Inst) -> Option<(Address, Mem)> {
    let (base, index, format, mem) = match graph.insts[inst].op {
        Op::Memory(MemoryOp::IndexedLoad {
            base,
            index,
            format,
        }) => (base, index, format, format_mem(format)),
        Op::Memory(MemoryOp::IndexedStore {
            base,
            index,
            format,
            ..
        }) => (base, index, format, format_store(format)),
        _ => return None,
    };
    let kind = match graph.reprs[index].map(Repr::machine)? {
        types::I64 => Index::Lsl,
        types::I32 => Index::Uxtw,
        _ => return None,
    };
    Some((
        Address::Indexed {
            base,
            index,
            kind,
            scaled: format.bytes() > 1,
        },
        mem,
    ))
}

struct Patterns {
    uses: SecondaryMap<Value, u32>,
}

fn fused_compare(graph: &Graph, value: Value, uses: &SecondaryMap<Value, u32>) -> Option<Inst> {
    if uses[value] != 1 {
        return None;
    }
    let ValueDef::Result(inst, 0) = graph.values[value] else {
        return None;
    };
    match graph.insts[inst].op {
        Op::LowLevel(
            low_level::Op::IntCompare { .. }
            | low_level::Op::TestMask { .. }
            | low_level::Op::InRange { .. },
        ) => Some(inst),
        Op::LowLevel(low_level::Op::FloatCompare { condition, .. })
            if float_conds(condition).1.is_none() =>
        {
            Some(inst)
        }
        _ => None,
    }
}

fn memory_address(graph: &Graph, value: Value, offset: i32, size: u32) -> Option<Address> {
    if graph.reprs[value].map(Repr::machine) != Some(types::I64) {
        return None;
    }
    let ValueDef::Result(add, 0) = graph.values[value] else {
        return None;
    };
    let Op::LowLevel(low_level::Op::Binary {
        op: low_level::Binary::Add,
        left,
        right,
    }) = graph.insts[add].op
    else {
        return None;
    };
    if let Some(constant) = constant(graph, right) {
        return Some(Address::Offset {
            base: left,
            offset: i64::from(offset).wrapping_add(constant as i64),
        });
    }
    if let Some(constant) = constant(graph, left) {
        return Some(Address::Offset {
            base: right,
            offset: i64::from(offset).wrapping_add(constant as i64),
        });
    }
    if offset != 0 {
        return None;
    }
    let mut index = right;
    let mut scaled = false;
    if let ValueDef::Result(inst, 0) = graph.values[index]
        && let Op::LowLevel(low_level::Op::Binary {
            op: low_level::Binary::ShiftLeft,
            left,
            right,
        }) = graph.insts[inst].op
        && constant(graph, right) == Some(u64::from(size.trailing_zeros()))
        && size > 1
    {
        index = left;
        scaled = true;
    }
    let mut kind = Index::Lsl;
    if let ValueDef::Result(inst, 0) = graph.values[index]
        && let Some((value, extension)) = extended_index(graph, inst)
    {
        index = value;
        kind = extension;
    }
    Some(Address::Indexed {
        base: left,
        index,
        kind,
        scaled,
    })
}

fn shifted_operand(
    graph: &Graph,
    value: Value,
    position: usize,
    uses: &SecondaryMap<Value, u32>,
) -> Option<Operand> {
    if uses[value] != 1 {
        return None;
    }
    let wide = graph.reprs[value].map(Repr::machine) == Some(types::I64);
    let ValueDef::Result(inst, 0) = graph.values[value] else {
        return None;
    };
    let (mut leaf, shift, mut changed) = match graph.insts[inst].op {
        Op::LowLevel(low_level::Op::Binary {
            op: low_level::Binary::ShiftLeft,
            left,
            right,
        }) => {
            let bits = graph.reprs[value].map(Repr::machine)?.bits();
            (left, constant(graph, right)? as u32 & (bits - 1), true)
        }
        _ => (value, 0, false),
    };
    let mut kind = Index::Lsl;
    if wide
        && shift <= 4
        && uses[leaf] == 1
        && let ValueDef::Result(inst, 0) = graph.values[leaf]
        && let Some((value, extension)) = extended_index(graph, inst)
    {
        leaf = value;
        kind = extension;
        changed = true;
    }
    if !changed {
        return None;
    }
    Some(Operand {
        position,
        value: leaf,
        kind,
        shift,
    })
}

impl Patterns {
    fn new(graph: &Graph) -> Self {
        Self {
            uses: graph.use_counts(),
        }
    }

    fn select_compare(&self, graph: &Graph, inst: Inst) -> Option<Inst> {
        let Op::LowLevel(low_level::Op::Select { condition, .. }) = graph.insts[inst].op else {
            return None;
        };
        fused_compare(graph, condition, &self.uses)
    }

    fn branch_compare(&self, graph: &Graph, block: Block) -> Option<Inst> {
        let TerminatorKind::Branch(condition, ..) = graph.blocks[block].terminator.as_ref()?.kind
        else {
            return None;
        };
        fused_compare(graph, condition, &self.uses)
    }

    fn address(&self, graph: &Graph, inst: Inst) -> Option<Address> {
        let (address, offset, size) = match graph.insts[inst].op {
            Op::LowLevel(low_level::Op::Load {
                ty,
                address,
                offset,
                flags,
            }) if flags.trap_code().is_none() => (address, offset, ty.bytes()),
            Op::LowLevel(low_level::Op::Store {
                value,
                address,
                offset,
                flags,
            }) if flags.trap_code().is_none() => {
                (address, offset, graph.repr(value).machine().bytes())
            }
            _ => return None,
        };
        memory_address(graph, address, offset, size)
    }

    fn operand(&self, graph: &Graph, inst: Inst) -> Option<Operand> {
        let Op::LowLevel(low_level::Op::Binary { op, left, right }) = graph.insts[inst].op else {
            return None;
        };
        if !matches!(op, low_level::Binary::Add | low_level::Binary::Subtract)
            || constant(graph, left).is_some()
            || constant(graph, right).is_some()
        {
            return None;
        }
        shifted_operand(graph, right, 1, &self.uses).or_else(|| {
            if matches!(op, low_level::Binary::Add) {
                shifted_operand(graph, left, 0, &self.uses)
            } else {
                None
            }
        })
    }
}

fn width(ty: Type) -> Width {
    if ty == types::I64 { Width::X } else { Width::W }
}

fn float(ty: Type) -> Float {
    if ty == types::F32 { Float::S } else { Float::D }
}

fn int_cond(cond: IntCC) -> Cond {
    match cond {
        IntCC::Equal => Cond::Eq,
        IntCC::NotEqual => Cond::Ne,
        IntCC::SignedLessThan => Cond::Lt,
        IntCC::SignedGreaterThanOrEqual => Cond::Ge,
        IntCC::SignedGreaterThan => Cond::Gt,
        IntCC::SignedLessThanOrEqual => Cond::Le,
        IntCC::UnsignedLessThan => Cond::Lo,
        IntCC::UnsignedGreaterThanOrEqual => Cond::Hs,
        IntCC::UnsignedGreaterThan => Cond::Hi,
        IntCC::UnsignedLessThanOrEqual => Cond::Ls,
    }
}

fn float_conds(cond: FloatCC) -> (Cond, Option<Cond>) {
    match cond {
        FloatCC::Ordered => (Cond::Vc, None),
        FloatCC::Unordered => (Cond::Vs, None),
        FloatCC::Equal => (Cond::Eq, None),
        FloatCC::NotEqual => (Cond::Ne, None),
        FloatCC::OrderedNotEqual => (Cond::Mi, Some(Cond::Gt)),
        FloatCC::UnorderedOrEqual => (Cond::Eq, Some(Cond::Vs)),
        FloatCC::LessThan => (Cond::Mi, None),
        FloatCC::LessThanOrEqual => (Cond::Ls, None),
        FloatCC::GreaterThan => (Cond::Gt, None),
        FloatCC::GreaterThanOrEqual => (Cond::Ge, None),
        FloatCC::UnorderedOrLessThan => (Cond::Lt, None),
        FloatCC::UnorderedOrLessThanOrEqual => (Cond::Le, None),
        FloatCC::UnorderedOrGreaterThan => (Cond::Hi, None),
        FloatCC::UnorderedOrGreaterThanOrEqual => (Cond::Pl, None),
    }
}

pub(in crate::jit) fn run(graph: &mut Graph) -> Result<(), Unsupported> {
    graph.verify_machine().map_err(Unsupported)?;
    let order = crate::jit::semantic::Cfg::reverse_postorder(graph);
    check_function(graph, &order)?;
    let patterns = Patterns::new(graph);
    let mut replacements = Vec::new();
    let mut branches = Vec::new();
    for &block in &order {
        let mut position = 0;
        for inst in graph.blocks[block].insts.clone() {
            let len = graph.blocks[block].insts.len();
            replacements.push((inst, instruction(graph, inst, position, &patterns)?));
            position += graph.blocks[block].insts.len() - len + 1;
        }
        let Some(term) = graph.blocks[block].terminator.clone() else {
            continue;
        };
        if let TerminatorKind::Branch(value, yes, no) = term.kind {
            let mut b = Builder::new(graph);
            b.switch_to_block(block);
            b.set_span(term.span);
            b.set_source_name(term.source_name);
            b.set_location(term.emission_location);
            let mut select = Selector { b };
            let condition = match patterns.branch_compare(select.b.graph, block) {
                Some(inst) => select.predicate(&select.b.graph.insts[inst].op.clone())?,
                None => select.truth(value),
            };
            let condition = match condition {
                Condition::Int {
                    cond: cond @ (Cond::Eq | Cond::Ne),
                    width,
                    extend: None,
                    left,
                    right: CompareOperand::Subs(0, _),
                } => Condition::Zero {
                    width,
                    value: left,
                    nonzero: cond == Cond::Ne,
                },
                other => other,
            };
            branches.push((block, TerminatorKind::Aarch64Branch(condition, yes, no)));
        }
    }
    for (inst, op) in replacements {
        graph.insts[inst].op = Op::Aarch64(op);
    }
    for (block, kind) in branches {
        graph.blocks[block].terminator.as_mut().unwrap().kind = kind;
    }
    remove_unused(graph);
    graph.verify_machine().map_err(Unsupported)
}

fn instruction(
    graph: &mut Graph,
    inst: Inst,
    position: usize,
    patterns: &Patterns,
) -> Result<a::Op, Unsupported> {
    let data = graph.insts[inst].clone();
    let result = data.results.first().copied();
    let ty = result
        .and_then(|value| graph.reprs[value])
        .map(Repr::machine);
    let selected_binary = binary(graph, inst);
    let indexed = indexed_load(graph, inst);
    let mut b = Builder::new(graph);
    b.insert_at(data.block, position);
    b.set_span(data.span);
    b.set_source_name(data.source_name);
    b.set_location(data.emission_location);
    let mut select = Selector { b };
    let load = |ty| super::abi::memory(ty, true).ok_or_else(|| Unsupported(format!("load {ty}")));
    let store =
        |ty| super::abi::memory(ty, false).ok_or_else(|| Unsupported(format!("store {ty}")));
    match &data.op {
        Op::Convert(value, to) => return select.convert(*value, *to),
        Op::Javascript(JsOp::Constant(Constant::Number(number))) => {
            return Ok(match ty.unwrap() {
                types::I32 => a::Op::Constant {
                    bits: *number as i32 as u32 as u64,
                    class: RegClass::Int,
                },
                types::I64 => a::Op::Constant {
                    bits: *number as i64 as u64,
                    class: RegClass::Int,
                },
                types::F64 => a::Op::Constant {
                    bits: number.to_bits(),
                    class: RegClass::Float,
                },
                ty => return Err(Unsupported(format!("number constant {ty}"))),
            });
        }
        Op::Javascript(JsOp::Constant(Constant::Boolean(value))) => {
            return Ok(a::Op::Constant {
                bits: u64::from(*value),
                class: RegClass::Int,
            });
        }
        _ => {}
    }
    if let Op::Memory(memory) = data.op {
        return Ok(match memory {
            MemoryOp::IndexedLoad { .. } => {
                let (address, op) = indexed.ok_or_else(|| Unsupported("indexed load".into()))?;
                a::Op::Load {
                    op,
                    address: select.address(op, address),
                }
            }
            MemoryOp::IndexedStore { value, .. } => {
                let (address, op) = indexed.ok_or_else(|| Unsupported("indexed store".into()))?;
                a::Op::Store {
                    op,
                    address: select.address(op, address),
                    value: Some(value),
                }
            }
            MemoryOp::Load { address, format } => {
                let op = format_mem(format);
                a::Op::Load {
                    op,
                    address: select.address(
                        op,
                        Address::Offset {
                            base: address,
                            offset: 0,
                        },
                    ),
                }
            }
            MemoryOp::Address {
                base,
                index,
                format,
            } => {
                if select.ty(index) != types::I64 {
                    return Err(Unsupported("memory address index type".into()));
                }
                a::Op::AddShifted {
                    sub: false,
                    width: Width::X,
                    left: base,
                    right: index,
                    kind: Index::Lsl,
                    shift: format.bytes().trailing_zeros(),
                }
            }
        });
    }
    let Op::LowLevel(op) = data.op else {
        return Err(Unsupported("non-low-level operation".into()));
    };
    use low_level::{Binary as B, Op as L, Unary as U};
    Ok(match op {
        L::Integer { bits, ty } => a::Op::Constant {
            bits: bits & (u64::MAX >> (64 - ty.bits())),
            class: RegClass::Int,
        },
        L::Float64(bits) => a::Op::Constant {
            bits,
            class: RegClass::Float,
        },
        L::Call {
            target,
            signature,
            args,
            ..
        } => a::Op::Call {
            target,
            signature,
            args,
        },
        L::Binary { op, left, right } => {
            let ty = ty.unwrap();
            if let Some(binary) = selected_binary {
                return Ok(binary);
            }
            if let Some(operand) = patterns.operand(select.b.graph, inst) {
                return Ok(a::Op::AddShifted {
                    sub: matches!(op, B::Subtract),
                    width: width(ty),
                    left: if operand.position == 0 { right } else { left },
                    right: operand.value,
                    kind: operand.kind,
                    shift: operand.shift,
                });
            }
            match op {
                B::Add
                | B::Subtract
                | B::And
                | B::Or
                | B::Xor
                | B::BooleanAnd
                | B::BooleanOr
                | B::BooleanXor => a::Op::Alu {
                    op: match op {
                        B::Add => Alu::Add,
                        B::Subtract => Alu::Sub,
                        B::And | B::BooleanAnd => Alu::And,
                        B::Or | B::BooleanOr => Alu::Orr,
                        _ => Alu::Eor,
                    },
                    width: width(ty),
                    left,
                    right,
                },
                B::Multiply => a::Op::Multiply {
                    width: width(ty),
                    left,
                    right,
                },
                B::BooleanEqual => a::Op::Compare(select.compare(IntCC::Equal, left, right)),
                B::ShiftLeft | B::ShiftRightSigned | B::ShiftRightUnsigned => {
                    if !matches!(ty, types::I32 | types::I64) {
                        return Err(Unsupported(format!("shift {ty}")));
                    }
                    let amount = match constant(select.b.graph, right) {
                        Some(bits) => ShiftAmount::Immediate(bits as u32 & (ty.bits() - 1)),
                        None => ShiftAmount::Reg(right),
                    };
                    a::Op::Shift {
                        op: match op {
                            B::ShiftLeft => Shift::Lsl,
                            B::ShiftRightSigned => Shift::Asr,
                            _ => Shift::Lsr,
                        },
                        width: width(ty),
                        value: left,
                        amount,
                    }
                }
                B::FloatAdd
                | B::FloatSubtract
                | B::FloatMultiply
                | B::FloatDivide
                | B::FloatMin
                | B::FloatMax => a::Op::FloatBinary {
                    op: match op {
                        B::FloatAdd => FloatOp::Add,
                        B::FloatSubtract => FloatOp::Sub,
                        B::FloatMultiply => FloatOp::Mul,
                        B::FloatDivide => FloatOp::Div,
                        B::FloatMin => FloatOp::Min,
                        _ => FloatOp::Max,
                    },
                    ty: float(ty),
                    left,
                    right,
                },
            }
        }
        L::Unary { op, value, .. } => {
            let ty = ty.unwrap();
            let from = select.ty(value);
            match op {
                U::BitNot => a::Op::BitNot {
                    width: width(ty),
                    value,
                },
                U::Reduce | U::Bitcast if RegClass::of(from) == RegClass::of(ty) => {
                    a::Op::Copy { value }
                }
                U::Bitcast => a::Op::Bitcast {
                    value,
                    width: if ty.bits() == 64 { Width::X } else { Width::W },
                    to: RegClass::of(ty).unwrap(),
                },
                U::Reduce => return Err(Unsupported("cross-class reduction".into())),
                U::SignExtend | U::ZeroExtend => {
                    let extend = match (from, op) {
                        (types::I8, U::ZeroExtend) => Extend::Uxtb,
                        (types::I16, U::ZeroExtend) => Extend::Uxth,
                        (types::I32, U::ZeroExtend) => Extend::Uxtw,
                        (types::I8, U::SignExtend) => Extend::Sxtb,
                        (types::I16, U::SignExtend) => Extend::Sxth,
                        (types::I32, U::SignExtend) => Extend::Sxtw,
                        _ => return Err(Unsupported(format!("extension from {from}"))),
                    };
                    a::Op::Extend {
                        value,
                        width: width(ty),
                        extend,
                    }
                }
                U::FloatFromSigned | U::FloatFromUnsigned => {
                    if !matches!(from, types::I32 | types::I64) {
                        return Err(Unsupported(format!("float conversion from {from}")));
                    }
                    a::Op::IntToFloat {
                        signed: matches!(op, U::FloatFromSigned),
                        width: width(from),
                        ty: float(ty),
                        value,
                    }
                }
                U::SignedSaturatingFromFloat => {
                    if !matches!(ty, types::I32 | types::I64) {
                        return Err(Unsupported(format!("integer conversion to {ty}")));
                    }
                    a::Op::FloatToInt {
                        signed: true,
                        width: width(ty),
                        ty: float(from),
                        value,
                    }
                }
                U::FloatPromote | U::FloatDemote => a::Op::FloatConvert {
                    promote: matches!(op, U::FloatPromote),
                    value,
                },
                U::FloatNeg | U::FloatAbs | U::Ceil | U::Floor | U::Trunc | U::Sqrt => {
                    a::Op::FloatUnary {
                        op: match op {
                            U::FloatNeg => FloatUnary::Neg,
                            U::FloatAbs => FloatUnary::Abs,
                            U::Ceil => FloatUnary::Ceil,
                            U::Floor => FloatUnary::Floor,
                            U::Trunc => FloatUnary::Trunc,
                            _ => FloatUnary::Sqrt,
                        },
                        ty: float(ty),
                        value,
                    }
                }
            }
        }
        L::IntCompare { .. } | L::FloatCompare { .. } | L::InRange { .. } | L::TestMask { .. } => {
            a::Op::Compare(select.predicate(&Op::LowLevel(op))?)
        }
        L::Select { condition, yes, no } => {
            let condition = match patterns.select_compare(select.b.graph, inst) {
                Some(def) => select.predicate(&select.b.graph.insts[def].op.clone())?,
                None => select.truth(condition),
            };
            let ty = ty.unwrap();
            a::Op::Select {
                condition,
                yes,
                no,
                ty: if RegClass::of(ty) == Some(RegClass::Float) {
                    a::SelectType::Float(float(ty))
                } else {
                    a::SelectType::Int(width(ty))
                },
            }
        }
        L::Load {
            ty,
            address,
            offset,
            flags,
        } => {
            if flags.trap_code().is_some() {
                return Err(Unsupported("trapping load".into()));
            }
            let op = load(ty)?;
            let address = patterns
                .address(select.b.graph, inst)
                .unwrap_or(Address::Offset {
                    base: address,
                    offset: i64::from(offset),
                });
            a::Op::Load {
                op,
                address: select.address(op, address),
            }
        }
        L::Store {
            value,
            address,
            offset,
            flags,
        } => {
            if flags.trap_code().is_some() {
                return Err(Unsupported("trapping store".into()));
            }
            let op = store(select.ty(value))?;
            let value = (constant(select.b.graph, value) != Some(0)).then_some(value);
            let address = patterns
                .address(select.b.graph, inst)
                .unwrap_or(Address::Offset {
                    base: address,
                    offset: i64::from(offset),
                });
            a::Op::Store {
                op,
                address: select.address(op, address),
                value,
            }
        }
        L::AtomicLoad { ty, address, flags } => {
            if flags.trap_code().is_some() || RegClass::of(ty) != Some(RegClass::Int) {
                return Err(Unsupported("atomic load".into()));
            }
            a::Op::LoadAcquire {
                address,
                bytes: ty.bytes(),
            }
        }
        L::StackAddress { slot, offset, .. } => a::Op::StackAddress { slot, offset },
        L::StackPointer { .. } => a::Op::StackPointer,
        L::ExtractBits {
            value,
            shift,
            width: bits,
        } => a::Op::ExtractBits {
            value,
            width: width(select.ty(value)),
            shift,
            bits,
        },
        L::Vector(vector) => a::Op::Vector(select.vector(vector)?),
        _ => return Err(Unsupported("unlowered operation".into())),
    })
}

struct Selector<'g> {
    b: Builder<'g>,
}

impl Selector<'_> {
    fn ty(&self, value: Value) -> Type {
        self.b.graph.repr(value).machine()
    }

    fn constant(&mut self, ty: Type, bits: u64) -> Value {
        let bits = bits & (u64::MAX >> (64 - ty.bits()));
        self.b.semantic(
            Op::Aarch64(a::Op::Constant {
                bits,
                class: RegClass::Int,
            }),
            Repr::from_machine(ty),
        )
    }

    fn convert(&mut self, value: Value, to: Repr) -> Result<a::Op, Unsupported> {
        let integer = |repr| if repr == Repr::Word { Repr::I64 } else { repr };
        let from = integer(self.b.graph.repr(value));
        let to = integer(to);
        if from == to {
            return Ok(a::Op::Copy { value });
        }
        if matches!(from, Repr::I32 | Repr::I64)
            && let Some(bits) = constant(self.b.graph, value)
        {
            let number = if from == Repr::I32 {
                i64::from(bits as i32)
            } else {
                bits as i64
            };
            let folded = match to {
                Repr::I32 => Some((number as u32 as u64, RegClass::Int)),
                Repr::I64 => Some((number as u64, RegClass::Int)),
                Repr::F64 => Some(((number as f64).to_bits(), RegClass::Float)),
                _ => None,
            };
            if let Some((bits, class)) = folded {
                return Ok(a::Op::Constant { bits, class });
            }
        }
        Ok(match (from, to) {
            (Repr::I32 | Repr::I64, Repr::F64) => a::Op::IntToFloat {
                signed: true,
                width: width(from.machine()),
                ty: Float::D,
                value,
            },
            (Repr::Boolean, Repr::I32) => a::Op::Extend {
                value,
                width: Width::W,
                extend: Extend::Uxtb,
            },
            (Repr::Boolean, Repr::F64) => {
                let wide = self.b.semantic(
                    Op::Aarch64(a::Op::Extend {
                        value,
                        width: Width::W,
                        extend: Extend::Uxtb,
                    }),
                    Repr::I32,
                );
                a::Op::IntToFloat {
                    signed: false,
                    width: Width::W,
                    ty: Float::D,
                    value: wide,
                }
            }
            (Repr::I32, Repr::I64) => a::Op::Extend {
                value,
                width: Width::X,
                extend: Extend::Sxtw,
            },
            (Repr::I64, Repr::I32) => a::Op::Copy { value },
            _ => {
                return Err(Unsupported(format!(
                    "unlowered conversion from {from:?} to {to:?}"
                )));
            }
        })
    }

    fn compare(&self, cond: IntCC, left: Value, right: Value) -> Condition {
        comparison(self.b.graph, cond, left, right)
    }

    fn truth(&self, value: Value) -> Condition {
        let ty = self.ty(value);
        if matches!(ty, types::I8 | types::I16) {
            return Condition::Mask {
                width: Width::W,
                value,
                mask: Mask::Immediate((1u64 << ty.bits()) - 1),
            };
        }
        let extend = match ty {
            types::I8 => Some(Extend::Uxtb),
            types::I16 => Some(Extend::Uxth),
            _ => None,
        };
        Condition::Int {
            cond: Cond::Ne,
            width: width(ty),
            extend,
            left: value,
            right: CompareOperand::Subs(0, false),
        }
    }

    fn predicate(&mut self, op: &Op) -> Result<Condition, Unsupported> {
        use low_level::Op as L;
        Ok(match *op {
            Op::LowLevel(L::IntCompare {
                condition,
                left,
                right,
            }) => self.compare(condition, left, right),
            Op::LowLevel(L::FloatCompare {
                condition,
                left,
                right,
            }) => {
                let (first, second) = float_conds(condition);
                Condition::Float {
                    ty: float(self.ty(left)),
                    first,
                    second,
                    left,
                    right,
                }
            }
            Op::LowLevel(L::TestMask { value, mask }) => {
                let ty = self.ty(value);
                let mask = mask & (u64::MAX >> (64 - ty.bits()));
                let mask = if e::logical_imm(Logical::Ands, width(ty), 0, 0, mask).is_some() {
                    Mask::Immediate(mask)
                } else {
                    Mask::Reg(self.constant(ty, mask))
                };
                Condition::Mask {
                    width: width(ty),
                    value,
                    mask,
                }
            }
            Op::LowLevel(L::InRange {
                value,
                low,
                high,
                inside,
            }) => {
                let ty = self.ty(value);
                let mask = u64::MAX >> (64 - ty.bits());
                let amount = low & mask;
                let mut offset = value;
                if amount != 0 && low != high {
                    let immediate = e::arith_imm(amount as i64)
                        .map(|(imm, shift)| a::Immediate::Sub(imm, shift))
                        .or_else(|| {
                            (amount as i64)
                                .checked_neg()
                                .and_then(e::arith_imm)
                                .map(|(imm, shift)| a::Immediate::Add(imm, shift))
                        });
                    let op = if let Some(immediate) = immediate {
                        a::Op::AluImmediate {
                            width: width(ty),
                            value,
                            immediate,
                        }
                    } else {
                        let right = self.constant(ty, amount);
                        a::Op::Alu {
                            op: Alu::Sub,
                            width: width(ty),
                            left: value,
                            right,
                        }
                    };
                    offset = self.b.semantic(Op::Aarch64(op), Repr::from_machine(ty));
                }
                let (cond, limit) = if low == high {
                    (
                        if inside {
                            IntCC::Equal
                        } else {
                            IntCC::NotEqual
                        },
                        low,
                    )
                } else {
                    (
                        if inside {
                            IntCC::UnsignedLessThanOrEqual
                        } else {
                            IntCC::UnsignedGreaterThan
                        },
                        high - low,
                    )
                };
                let limit = self.constant(ty, limit);
                self.compare(cond, offset, limit)
            }
            _ => return Err(Unsupported("comparison".into())),
        })
    }

    fn vector(&mut self, vector: low_level::Vector) -> Result<a::Vector, Unsupported> {
        use low_level::{Binary as B, Vector as V};
        Ok(match vector {
            V::Load {
                base,
                index,
                format,
                ..
            } => a::Vector::Load {
                address: self.lanes(base, index, format),
                bytes: format.bytes(),
                signed: matches!(format, MemoryFormat::I8 | MemoryFormat::I16),
            },
            V::Store {
                base,
                index,
                value,
                format,
            } => a::Vector::Store {
                address: self.lanes(base, index, format),
                bytes: format.bytes(),
                value,
            },
            V::Splat { value, .. } => a::Vector::Splat { value },
            V::Binary {
                op: op @ (B::ShiftLeft | B::ShiftRightSigned | B::ShiftRightUnsigned),
                left,
                right,
            } => {
                let amount = constant(self.b.graph, right)
                    .ok_or_else(|| Unsupported("vector shift amount".into()))?
                    as u32
                    & 31;
                a::Vector::Shift {
                    op: match op {
                        _ if amount == 0 => Shift::Lsl,
                        B::ShiftLeft => Shift::Lsl,
                        B::ShiftRightSigned => Shift::Asr,
                        _ => Shift::Lsr,
                    },
                    value: left,
                    amount,
                }
            }
            V::Binary { op, left, right } => a::Vector::Binary {
                op: match op {
                    B::Add => VectorOp::Add,
                    B::Subtract => VectorOp::Sub,
                    B::Multiply => VectorOp::Mul,
                    B::And => VectorOp::And,
                    B::Or => VectorOp::Orr,
                    B::Xor => VectorOp::Eor,
                    _ => return Err(Unsupported(format!("vector {op:?}"))),
                },
                left,
                right,
            },
        })
    }

    fn lanes(&mut self, base: Value, index: Value, format: MemoryFormat) -> a::Address {
        let shift = format.bytes().trailing_zeros();
        if shift == 0 {
            return a::Address::Indexed {
                base,
                index,
                kind: Index::Lsl,
                scaled: false,
            };
        }
        let base = self.b.semantic(
            Op::Aarch64(a::Op::AddShifted {
                sub: false,
                width: Width::X,
                left: base,
                right: index,
                kind: Index::Lsl,
                shift,
            }),
            Repr::I64,
        );
        a::Address::Scaled { base, offset: 0 }
    }

    fn address(&mut self, op: Mem, address: Address) -> a::Address {
        match address {
            Address::Indexed {
                base,
                index,
                kind,
                scaled,
            } => a::Address::Indexed {
                base,
                index,
                kind,
                scaled,
            },
            Address::Offset { base, offset } => {
                if let Ok(offset) = u32::try_from(offset)
                    && e::mem_scaled(op, 0, 0, offset).is_some()
                {
                    return a::Address::Scaled { base, offset };
                }
                if let Ok(offset) = i32::try_from(offset)
                    && e::mem_unscaled(op, 0, 0, offset).is_some()
                {
                    return a::Address::Unscaled { base, offset };
                }
                let index = self.constant(types::I64, offset as u64);
                a::Address::Indexed {
                    base,
                    index,
                    kind: Index::Lsl,
                    scaled: false,
                }
            }
        }
    }
}

fn format_store(format: MemoryFormat) -> Mem {
    match format {
        MemoryFormat::U8 | MemoryFormat::I8 => Mem::Strb,
        MemoryFormat::U16 | MemoryFormat::I16 => Mem::Strh,
        MemoryFormat::U32 | MemoryFormat::I32 => Mem::StrW,
        MemoryFormat::F32 => Mem::StrS,
        MemoryFormat::F64 => Mem::StrD,
    }
}

fn format_mem(format: MemoryFormat) -> Mem {
    match format {
        MemoryFormat::U8 | MemoryFormat::I8 => Mem::Ldrb,
        MemoryFormat::U16 | MemoryFormat::I16 => Mem::Ldrh,
        MemoryFormat::U32 | MemoryFormat::I32 => Mem::LdrW,
        MemoryFormat::F32 => Mem::LdrS,
        MemoryFormat::F64 => Mem::LdrD,
    }
}

fn remove_unused(graph: &mut Graph) {
    let mut uses = graph.use_counts();
    let mut work = Vec::new();
    let mut removed = SecondaryMap::<Inst, bool>::new();
    for block in graph.blocks.values() {
        for &inst in &block.insts {
            if let Op::Aarch64(op) = &graph.insts[inst].op
                && op.pure()
                && graph.insts[inst]
                    .results
                    .iter()
                    .all(|&value| uses[value] == 0)
            {
                work.push(inst);
            }
        }
    }
    while let Some(inst) = work.pop() {
        if removed[inst] {
            continue;
        }
        removed[inst] = true;
        for value in graph.insts[inst].op.operands_iter() {
            uses[value] -= 1;
            if uses[value] == 0
                && let ValueDef::Result(def, _) = graph.values[value]
                && let Op::Aarch64(op) = &graph.insts[def].op
                && op.pure()
                && graph.insts[def]
                    .results
                    .iter()
                    .all(|&value| uses[value] == 0)
            {
                work.push(def);
            }
        }
    }
    for block in graph.blocks.values_mut() {
        block.insts.retain(|&inst| !removed[inst]);
    }
}

fn check_function(graph: &Graph, order: &[Block]) -> Result<(), Unsupported> {
    let signature = graph
        .low_level_signature
        .as_ref()
        .ok_or_else(|| Unsupported("missing signature".into()))?;
    if let Some(param) = signature
        .returns
        .iter()
        .find(|p| p.extension != ArgumentExtension::None)
    {
        return Err(Unsupported(format!("extended return {param}")));
    }
    for &block in order {
        for &value in &graph.blocks[block].params {
            let ty = graph.reprs[value]
                .map(Repr::machine)
                .ok_or_else(|| Unsupported(format!("missing type for {value}")))?;
            if RegClass::of(ty).is_none() {
                return Err(Unsupported(format!("value type {ty}")));
            }
        }
        for &inst in &graph.blocks[block].insts {
            if matches!(graph.insts[inst].op, Op::LowLevel(low_level::Op::Vector(_))) {
                continue;
            }
            for &value in &graph.insts[inst].results {
                let ty = graph.reprs[value]
                    .map(Repr::machine)
                    .ok_or_else(|| Unsupported(format!("missing type for {value}")))?;
                if RegClass::of(ty).is_none() {
                    return Err(Unsupported(format!("value type {ty}")));
                }
            }
        }
    }
    Ok(())
}

fn constant(graph: &Graph, value: Value) -> Option<u64> {
    let op = graph.defining_op(value)?;
    let bits = match *op {
        Op::LowLevel(low_level::Op::Integer { bits, .. })
        | Op::Aarch64(a::Op::Constant {
            bits,
            class: RegClass::Int,
        }) => bits,
        Op::Javascript(JsOp::Constant(Constant::Number(number)))
            if matches!(graph.repr(value), Repr::I32 | Repr::I64) =>
        {
            if graph.repr(value) == Repr::I32 {
                number as i32 as u32 as u64
            } else {
                number as i64 as u64
            }
        }
        Op::Javascript(JsOp::Constant(Constant::Boolean(value))) => u64::from(value),
        Op::Convert(input, to) => {
            let bits = constant(graph, input)?;
            match (graph.repr(input), to) {
                (Repr::I32, Repr::I64 | Repr::Word) => bits as i32 as i64 as u64,
                (Repr::I64 | Repr::Word, Repr::I32) => bits as u32 as u64,
                (from, to)
                    if from == to
                        || matches!(
                            (from, to),
                            (Repr::Word, Repr::I64)
                                | (Repr::I64, Repr::Word)
                                | (Repr::Boolean, Repr::I32)
                        ) =>
                {
                    bits
                }
                _ => return None,
            }
        }
        _ => return None,
    };
    Some(bits & (u64::MAX >> (64 - graph.repr(value).machine().bits())))
}

fn extended_index(graph: &Graph, inst: Inst) -> Option<(Value, Index)> {
    let (value, kind) = match graph.insts[inst].op {
        Op::LowLevel(low_level::Op::Unary {
            op: low_level::Unary::ZeroExtend,
            value,
            ..
        }) => (value, Index::Uxtw),
        Op::LowLevel(low_level::Op::Unary {
            op: low_level::Unary::SignExtend,
            value,
            ..
        }) => (value, Index::Sxtw),
        Op::Convert(value, Repr::I64 | Repr::Word) => (value, Index::Sxtw),
        _ => return None,
    };
    (graph.repr(value).machine() == types::I32).then_some((value, kind))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::semantic::low_level::Builder;
    use crate::jit::semantic::variables::Variables;
    use cranelift_codegen::ir::{AbiParam, Signature};
    use cranelift_codegen::isa::CallConv;
    use oxc_span::Span;

    #[test]
    fn selected_shift_and_compare_keep_their_inputs_live() {
        let mut signature = Signature::new(CallConv::SystemV);
        signature
            .params
            .extend([AbiParam::new(types::I64), AbiParam::new(types::I32)]);
        let mut graph = Graph::new_low_level("test".into(), Span::new(0, 0), signature);
        let entry = graph.entry;
        let mut builder = Builder::new(&mut graph);
        builder.switch_to_block(entry);
        builder.append_block_params_for_function_params(entry);
        let base = builder.block_params(entry)[0];
        let index = builder.block_params(entry)[1];
        let wide = builder.ins().uextend(types::I64, index);
        let shift = builder.ins().ishl_imm_s(wide, 2);
        let sum = builder.ins().iadd(base, shift);
        let compare = builder.ins().icmp_imm_s(IntCC::NotEqual, sum, 0);
        let yes = builder.create_block();
        let no = builder.create_block();
        builder.ins().brif(compare, yes, &[], no, &[]);
        builder.switch_to_block(yes);
        builder.ins().return_(&[]);
        builder.switch_to_block(no);
        builder.ins().return_(&[]);
        let wide_inst = builder.defining_inst(wide);
        let shift_inst = builder.defining_inst(shift);
        let sum_inst = builder.defining_inst(sum);
        let compare_inst = builder.defining_inst(compare);
        drop(builder);
        run(&mut graph).unwrap();
        let insts = &graph.blocks[entry].insts;
        assert!(!insts.contains(&wide_inst));
        assert!(!insts.contains(&shift_inst));
        assert!(!insts.contains(&compare_inst));
        assert!(
            matches!(graph.insts[sum_inst].op, Op::Aarch64(a::Op::AddShifted { right, kind: Index::Uxtw, shift: 2, .. }) if right == index)
        );
        assert_eq!(graph.insts[sum_inst].op.operands(), vec![base, index]);
        assert!(
            graph.blocks[entry]
                .terminator
                .as_ref()
                .unwrap()
                .kind
                .operands()
                .contains(&sum)
        );
    }

    #[test]
    fn shared_addresses_fold_without_discarding_other_uses() {
        for returned in 0..4 {
            let mut signature = Signature::new(CallConv::SystemV);
            signature
                .params
                .extend([AbiParam::new(types::I64), AbiParam::new(types::I32)]);
            signature.returns.push(AbiParam::new(types::I64));
            let mut graph =
                Graph::new_low_level("shared_address".into(), Span::new(0, 0), signature);
            let entry = graph.entry;
            let mut builder = Builder::new(&mut graph);
            builder.switch_to_block(entry);
            builder.append_block_params_for_function_params(entry);
            let base = builder.block_params(entry)[0];
            let index = builder.block_params(entry)[1];
            let wide = builder.ins().uextend(types::I64, index);
            let shift = builder.ins().ishl_imm_s(wide, 3);
            let address = builder.ins().iadd(base, shift);
            let flags = cranelift_codegen::ir::MemFlagsData::trusted();
            let loaded = builder.ins().load(types::I64, flags, address, 0);
            let store = builder.ins().store(flags, loaded, address, 0);
            builder
                .ins()
                .return_(&[[loaded, address, shift, wide][returned]]);
            let load = builder.defining_inst(loaded);
            let address_inst = builder.defining_inst(address);
            let shift_inst = builder.defining_inst(shift);
            let wide_inst = builder.defining_inst(wide);
            Variables::default().finish(&mut graph).unwrap();
            graph.verify().unwrap();
            run(&mut graph).unwrap();
            for inst in [load, store] {
                let address = match graph.insts[inst].op {
                    Op::Aarch64(a::Op::Load { address, .. } | a::Op::Store { address, .. }) => {
                        address
                    }
                    _ => panic!("unselected memory operation"),
                };
                assert!(
                    matches!(address, a::Address::Indexed { base: b, index: i, kind: Index::Uxtw, scaled: true } if b == base && i == index)
                );
            }
            let insts = &graph.blocks[entry].insts;
            assert_eq!(!insts.contains(&address_inst), returned != 1);
            assert_eq!(!insts.contains(&shift_inst), returned != 2);
            assert_eq!(!insts.contains(&wide_inst), returned < 2);
            assert_eq!(graph.insts[load].op.operands(), vec![base, index]);
            assert_eq!(graph.insts[store].op.operands(), vec![loaded, base, index]);
        }
    }

    #[test]
    fn fused_select_exposes_and_rewrites_all_four_inputs() {
        let mut signature = Signature::new(CallConv::AppleAarch64);
        signature.params.extend([AbiParam::new(types::I64); 4]);
        signature.returns.push(AbiParam::new(types::I64));
        let mut graph = Graph::new_low_level("select".into(), Span::new(0, 0), signature);
        let entry = graph.entry;
        let mut builder = Builder::new(&mut graph);
        builder.switch_to_block(entry);
        builder.append_block_params_for_function_params(entry);
        let inputs = builder.block_params(entry).to_vec();
        let condition = builder
            .ins()
            .icmp(IntCC::SignedLessThan, inputs[0], inputs[1]);
        let result = builder.ins().select(condition, inputs[2], inputs[3]);
        let inst = builder.defining_inst(result);
        builder.ins().return_(&[result]);
        run(&mut graph).unwrap();
        assert_eq!(graph.insts[inst].op.operands(), inputs);
        let mut aliases = crate::FxHashMap::default();
        aliases.insert(inputs[0], inputs[3]);
        graph.replace_uses(&aliases);
        graph.verify_machine().unwrap();
        assert_eq!(
            graph.insts[inst].op.operands(),
            [inputs[3], inputs[1], inputs[2], inputs[3]]
        );
        let prepared = super::super::prepare::run(&graph).unwrap();
        assert!(prepared.input.calls.iter().all(|(_, &call)| !call));
    }

    #[test]
    fn indexed_loads_select_raw_format_and_scale() {
        let mut signature = Signature::new(CallConv::SystemV);
        signature
            .params
            .extend([AbiParam::new(types::I64), AbiParam::new(types::I64)]);
        let mut graph = Graph::new_low_level("test".into(), Span::new(0, 0), signature);
        let entry = graph.entry;
        let mut builder = Builder::new(&mut graph);
        builder.switch_to_block(entry);
        builder.append_block_params_for_function_params(entry);
        let base = builder.block_params(entry)[0];
        let index = builder.block_params(entry)[1];
        for (format, ty, expected) in [
            (MemoryFormat::U8, types::I8, Mem::Ldrb),
            (MemoryFormat::I8, types::I8, Mem::Ldrb),
            (MemoryFormat::U16, types::I16, Mem::Ldrh),
            (MemoryFormat::I16, types::I16, Mem::Ldrh),
            (MemoryFormat::U32, types::I32, Mem::LdrW),
            (MemoryFormat::I32, types::I32, Mem::LdrW),
            (MemoryFormat::F32, types::F32, Mem::LdrS),
            (MemoryFormat::F64, types::F64, Mem::LdrD),
        ] {
            let value = builder.indexed_load(base, index, format, ty);
            let inst = builder.defining_inst(value);
            let (address, mem) = indexed_load(builder.graph, inst).unwrap();
            assert_eq!(mem, expected);
            assert!(
                matches!(address, Address::Indexed { base: b, index: i, kind: Index::Lsl, scaled } if b == base && i == index && scaled == (format.bytes() > 1))
            );
        }
    }
}

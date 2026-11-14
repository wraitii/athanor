use cranelift_codegen::ir::condcodes::IntCC;

use super::cursor::{Cursor, Insert};
use super::low_level::{Binary, Op as LowOp};
use super::{Graph, Op, Repr, Value};

pub(in crate::jit) fn expand(graph: &mut Graph) {
    for block in graph.blocks.keys().collect::<Vec<_>>() {
        for inst in graph.blocks[block].insts.clone() {
            let Op::LowLevel(op) = &graph.insts[inst].op else {
                continue;
            };
            let op = op.clone();
            let mut at = Cursor::before(graph, inst);
            let replacement = match op {
                LowOp::ExtractBits {
                    value,
                    shift,
                    width,
                } => extract(&mut at, value, shift, width),
                LowOp::InRange {
                    value,
                    low,
                    high,
                    inside,
                } => range(&mut at, value, low, high, inside),
                LowOp::TestMask { value, mask } => {
                    let masked = at.ins().band_imm_u(value, mask as i64);
                    compare(&mut at, IntCC::NotEqual, masked, 0)
                }
                _ => continue,
            };
            graph.insts[inst].op = Op::LowLevel(replacement);
        }
    }
}

fn extract(at: &mut Cursor, value: Value, shift: u8, width: u8) -> LowOp {
    let bits = at.graph().reprs[value].map(Repr::machine).unwrap().bits();
    let mask = (1u64 << width) - 1;
    if u32::from(shift) + u32::from(width) == bits {
        let amount = constant(at, value, u64::from(shift));
        return LowOp::Binary {
            op: Binary::ShiftRightUnsigned,
            left: value,
            right: amount,
        };
    }
    let shifted = if shift == 0 {
        value
    } else {
        at.ins().ushr_imm_s(value, i64::from(shift))
    };
    let mask = constant(at, shifted, mask);
    LowOp::Binary {
        op: Binary::And,
        left: shifted,
        right: mask,
    }
}

fn range(at: &mut Cursor, value: Value, low: u64, high: u64, inside: bool) -> LowOp {
    if low == high {
        let condition = if inside { IntCC::Equal } else { IntCC::NotEqual };
        return compare(at, condition, value, low);
    }
    let offset = if low == 0 {
        value
    } else {
        at.ins().iadd_imm_s(value, low.wrapping_neg() as i64)
    };
    let condition = if inside {
        IntCC::UnsignedLessThanOrEqual
    } else {
        IntCC::UnsignedGreaterThan
    };
    compare(at, condition, offset, high - low)
}

fn compare(at: &mut Cursor, condition: IntCC, left: Value, right: u64) -> LowOp {
    let right = constant(at, left, right);
    LowOp::IntCompare {
        condition,
        left,
        right,
    }
}

fn constant(at: &mut Cursor, like: Value, bits: u64) -> Value {
    let ty = at.graph().reprs[like].map(Repr::machine).unwrap();
    at.ins().iconst(ty, bits as i64)
}

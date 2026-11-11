use cranelift_codegen::entity::SecondaryMap;
use cranelift_codegen::ir::{Function, Inst, InstructionData, Opcode, Value, ValueDef, types};

use super::encode::Index;
use super::lower::{float_conds, load_op, store_op};
use crate::jit::alloc;

#[derive(Clone, Copy, Debug)]
pub(super) enum Address {
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

#[derive(Clone, Copy, Debug)]
pub(super) struct Operand {
    pub position: usize,
    pub value: Value,
    pub kind: Index,
    pub shift: u32,
}

pub(super) struct Selection {
    address: SecondaryMap<Inst, Option<Address>>,
    operand: SecondaryMap<Inst, Option<Operand>>,
    compare: SecondaryMap<Inst, Option<Inst>>,
    skip: SecondaryMap<Inst, bool>,
}

fn defined_by(func: &Function, value: Value, opcode: Opcode) -> Option<Inst> {
    let ValueDef::Result(inst, 0) = func.dfg.value_def(value) else {
        return None;
    };
    (func.dfg.insts[inst].opcode() == opcode).then_some(inst)
}

fn extension(func: &Function, value: Value) -> Option<(Value, Index)> {
    for (opcode, kind) in [
        (Opcode::Uextend, Index::Uxtw),
        (Opcode::Sextend, Index::Sxtw),
    ] {
        if let Some(inst) = defined_by(func, value, opcode) {
            let arg = func.dfg.inst_args(inst)[0];
            if func.dfg.value_type(arg) == types::I32 {
                return Some((arg, kind));
            }
        }
    }
    None
}

fn shift_amount(func: &Function, value: Value) -> Option<(Value, u32)> {
    let inst = defined_by(func, value, Opcode::Ishl)?;
    let args = func.dfg.inst_args(inst);
    let bits = func.dfg.value_type(value).bits();
    let amount = alloc::constant(func, args[1])? as u32 & (bits - 1);
    Some((args[0], amount))
}

fn index(func: &Function, value: Value, size: u32) -> (Value, Index, bool) {
    if size > 1
        && let Some((shifted, amount)) = shift_amount(func, value)
        && amount == size.trailing_zeros()
    {
        let (leaf, kind) = extension(func, shifted).unwrap_or((shifted, Index::Lsl));
        return (leaf, kind, true);
    }
    let (leaf, kind) = extension(func, value).unwrap_or((value, Index::Lsl));
    (leaf, kind, false)
}

fn address(func: &Function, value: Value, offset: i64, size: u32) -> Option<Address> {
    let inst = defined_by(func, value, Opcode::Iadd)?;
    if func.dfg.value_type(value) != types::I64 {
        return None;
    }
    let [a, b] = [func.dfg.inst_args(inst)[0], func.dfg.inst_args(inst)[1]];
    if let Some(constant) = alloc::constant(func, b) {
        return Some(Address::Offset {
            base: a,
            offset: offset.wrapping_add(constant),
        });
    }
    if let Some(constant) = alloc::constant(func, a) {
        return Some(Address::Offset {
            base: b,
            offset: offset.wrapping_add(constant),
        });
    }
    if offset != 0 {
        return None;
    }
    let composite = |v: Value| shift_amount(func, v).is_some() || extension(func, v).is_some();
    let (base, index_value) = if !composite(b) && composite(a) {
        (b, a)
    } else {
        (a, b)
    };
    let (index, kind, scaled) = index(func, index_value, size);
    Some(Address::Indexed {
        base,
        index,
        kind,
        scaled,
    })
}

fn operand(func: &Function, value: Value, position: usize) -> Option<Operand> {
    let wide = func.dfg.value_type(value) == types::I64;
    if let Some((shifted, amount)) = shift_amount(func, value) {
        if wide
            && amount <= 4
            && let Some((leaf, kind)) = extension(func, shifted)
        {
            return Some(Operand {
                position,
                value: leaf,
                kind,
                shift: amount,
            });
        }
        return Some(Operand {
            position,
            value: shifted,
            kind: Index::Lsl,
            shift: amount,
        });
    }
    let (leaf, kind) = extension(func, value).filter(|_| wide)?;
    Some(Operand {
        position,
        value: leaf,
        kind,
        shift: 0,
    })
}

fn compare(func: &Function, value: Value, uses: &SecondaryMap<Value, u32>) -> Option<Inst> {
    if uses[value] != 1 {
        return None;
    }
    let ValueDef::Result(inst, 0) = func.dfg.value_def(value) else {
        return None;
    };
    match func.dfg.insts[inst] {
        InstructionData::IntCompare { .. } => Some(inst),
        InstructionData::FloatCompare { cond, .. } if float_conds(cond).1.is_none() => Some(inst),
        _ => None,
    }
}

fn absorbable(func: &Function, value: Value) -> Option<Inst> {
    let ValueDef::Result(inst, 0) = func.dfg.value_def(value) else {
        return None;
    };
    if func.layout.inst_block(inst).is_none() {
        return None;
    }
    matches!(
        func.dfg.insts[inst].opcode(),
        Opcode::Iadd
            | Opcode::Ishl
            | Opcode::Uextend
            | Opcode::Sextend
            | Opcode::Icmp
            | Opcode::Fcmp
    )
    .then_some(inst)
}

impl Selection {
    pub(super) fn new(func: &Function) -> Self {
        let dfg = &func.dfg;
        let mut uses = SecondaryMap::<Value, u32>::new();
        for block in func.layout.blocks() {
            for inst in func.layout.block_insts(block) {
                for value in dfg.inst_values(inst) {
                    uses[value] += 1;
                }
            }
        }

        let mut selection = Self {
            address: SecondaryMap::new(),
            operand: SecondaryMap::new(),
            compare: SecondaryMap::new(),
            skip: SecondaryMap::new(),
        };
        for block in func.layout.blocks() {
            for inst in func.layout.block_insts(block) {
                let data = &dfg.insts[inst];
                let opcode = data.opcode();
                let args = dfg.inst_args(inst);
                let offset = data.load_store_offset().map(i64::from);
                if let Some(ty) = dfg.inst_results(inst).first().map(|&r| dfg.value_type(r))
                    && let Some(op) = load_op(opcode, ty)
                    && data.memflags_trap_code(dfg).is_none()
                {
                    selection.address[inst] = address(func, args[0], offset.unwrap(), op.size());
                } else if let Some(op) = args
                    .first()
                    .and_then(|&v| store_op(opcode, dfg.value_type(v)))
                    && data.memflags_trap_code(dfg).is_none()
                {
                    selection.address[inst] = address(func, args[1], offset.unwrap(), op.size());
                } else if matches!(opcode, Opcode::Iadd | Opcode::Isub)
                    && alloc::constant(func, args[0]).is_none()
                    && alloc::constant(func, args[1]).is_none()
                {
                    selection.operand[inst] = operand(func, args[1], 1).or_else(|| {
                        (opcode == Opcode::Iadd)
                            .then(|| operand(func, args[0], 0))
                            .flatten()
                    });
                } else if matches!(opcode, Opcode::Brif | Opcode::Select) {
                    selection.compare[inst] = compare(func, args[0], &uses);
                }
            }
        }

        let mut references = SecondaryMap::<Value, u32>::new();
        let mut operands = Vec::new();
        for block in func.layout.blocks() {
            for inst in func.layout.block_insts(block) {
                selection.operands(func, inst, &mut operands);
                for &value in &operands {
                    references[value] += 1;
                }
            }
        }
        let mut work: Vec<Inst> = dfg
            .values()
            .filter(|&value| references[value] == 0)
            .filter_map(|value| absorbable(func, value))
            .collect();
        while let Some(inst) = work.pop() {
            if selection.skip[inst] || references[dfg.first_result(inst)] != 0 {
                continue;
            }
            selection.operands(func, inst, &mut operands);
            selection.skip[inst] = true;
            for &value in &operands {
                references[value] -= 1;
                if references[value] == 0
                    && let Some(def) = absorbable(func, value)
                {
                    work.push(def);
                }
            }
        }
        selection
    }

    pub(super) fn operands(&self, func: &Function, inst: Inst, out: &mut Vec<Value>) {
        out.clear();
        if self.skip[inst] {
            return;
        }
        let dfg = &func.dfg;
        out.extend(dfg.inst_values(inst));
        if let Some(address) = self.address[inst] {
            let position = if dfg.insts[inst].opcode().can_load() {
                0
            } else {
                1
            };
            out.swap_remove(position);
            match address {
                Address::Offset { base, .. } => out.push(base),
                Address::Indexed { base, index, .. } => out.extend([base, index]),
            }
        } else if let Some(operand) = self.operand[inst] {
            out.swap_remove(operand.position);
            out.push(operand.value);
        } else if let Some(compare) = self.compare[inst] {
            out.swap_remove(0);
            out.extend_from_slice(dfg.inst_args(compare));
        }
    }

    pub(super) fn skips(&self, inst: Inst) -> bool {
        self.skip[inst]
    }

    pub(super) fn address(&self, inst: Inst) -> Option<Address> {
        self.address[inst]
    }

    pub(super) fn operand(&self, inst: Inst) -> Option<Operand> {
        self.operand[inst]
    }

    pub(super) fn compare(&self, inst: Inst) -> Option<Inst> {
        self.compare[inst]
    }
}

use cranelift_codegen::entity::SecondaryMap;
use cranelift_codegen::ir::condcodes::{CondCode, IntCC};
use cranelift_codegen::ir::types;

use super::encode::{self as e, Index, Logical, Mem, Width};
use super::lower::{float_conds, width};
use crate::jit::alloc::{RegClass, graph as alloc};
use crate::jit::semantic::Repr;
use crate::jit::semantic::{
    Block, Graph, Inst, MemoryFormat, MemoryOp, Op, TerminatorKind, Value, ValueDef, low_level,
};

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

#[derive(Clone, Copy)]
pub(super) struct Operand {
    pub position: usize,
    pub value: Value,
    pub kind: Index,
    pub shift: u32,
}

#[derive(Clone, Copy)]
pub(super) enum Immediate {
    Add(u32, bool),
    Sub(u32, bool),
    Logical(Logical, u64),
}

pub(super) struct Binary {
    pub x: Value,
    pub immediate: Immediate,
}

#[derive(Clone, Copy)]
pub(super) enum CompareImmediate {
    Subs(u32, bool),
    Adds(u32, bool),
}

pub(super) struct Comparison {
    pub cond: IntCC,
    pub x: Value,
    pub y: Value,
    pub immediate: Option<CompareImmediate>,
}

pub(super) fn signed_constant(graph: &Graph, value: Value) -> Option<i64> {
    let bits = alloc::constant(graph, value)?;
    let width = graph.reprs[value].map(Repr::machine)?.bits();
    Some(if width >= 64 {
        bits as i64
    } else {
        ((bits << (64 - width)) as i64) >> (64 - width)
    })
}

pub(super) fn binary(graph: &Graph, inst: Inst) -> Option<Binary> {
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
            Immediate::Add(imm, shift)
        } else {
            let (imm, shift) = amount.checked_neg().and_then(e::arith_imm)?;
            Immediate::Sub(imm, shift)
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
        Immediate::Logical(logical, bits)
    };
    Some(Binary { x, immediate })
}

pub(super) fn comparison(graph: &Graph, cond: IntCC, x: Value, y: Value) -> Comparison {
    let is_constant = |value| alloc::constant(graph, value).is_some();
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
    let unsigned = alloc::constant(graph, y).map(|bits| bits as i64);
    let sign_extended = signed_constant(graph, y);
    let negatable = signed || (!extended && matches!(cond, IntCC::Equal | IntCC::NotEqual));
    let immediate = (if signed { sign_extended } else { unsigned })
        .and_then(e::arith_imm)
        .map(|(imm, shift)| CompareImmediate::Subs(imm, shift))
        .or_else(|| {
            sign_extended
                .filter(|_| negatable)
                .and_then(i64::checked_neg)
                .and_then(e::arith_imm)
                .map(|(imm, shift)| CompareImmediate::Adds(imm, shift))
        });
    Comparison {
        cond,
        x,
        y,
        immediate,
    }
}

pub(super) fn remat_bits(graph: &Graph, value: Value) -> Option<u64> {
    if let Some(bits) = alloc::constant(graph, value) {
        return Some(bits);
    }
    let ValueDef::Result(inst, _) = graph.values[value] else {
        return None;
    };
    match graph.insts[inst].op {
        Op::LowLevel(low_level::Op::Float64(bits)) => Some(bits),
        _ => None,
    }
}

pub(super) fn remat_cost(graph: &Graph, value: Value) -> u32 {
    let Some(bits) = remat_bits(graph, value) else {
        return 0;
    };
    let class = graph.reprs[value].map(Repr::machine).and_then(RegClass::of);
    if class == Some(RegClass::Float) && (bits == 0 || e::float_imm8(bits).is_some()) {
        return 1;
    }
    let mut count = 0;
    e::load_imm(0, bits, |_| count += 1);
    if class == Some(RegClass::Float) {
        count += 1;
    }
    count
}

pub(super) fn indexed_load(graph: &Graph, inst: Inst) -> Option<(Address, Mem)> {
    let Op::Memory(MemoryOp::IndexedLoad {
        base,
        index,
        format,
    }) = graph.insts[inst].op
    else {
        return None;
    };
    let mem = match format {
        MemoryFormat::U8 | MemoryFormat::I8 => Mem::Ldrb,
        MemoryFormat::U16 | MemoryFormat::I16 => Mem::Ldrh,
        MemoryFormat::U32 | MemoryFormat::I32 => Mem::LdrW,
        MemoryFormat::F32 => Mem::LdrS,
        MemoryFormat::F64 => Mem::LdrD,
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

pub(super) struct Selection {
    select_compare: SecondaryMap<Inst, Option<Inst>>,
    branch_compare: SecondaryMap<Block, Option<Inst>>,
    skip: SecondaryMap<Inst, bool>,
    address: SecondaryMap<Inst, Option<Address>>,
    operand: SecondaryMap<Inst, Option<Operand>>,
}

fn fused_compare(graph: &Graph, value: Value, uses: &SecondaryMap<Value, u32>) -> Option<Inst> {
    if uses[value] != 1 {
        return None;
    }
    let ValueDef::Result(inst, 0) = graph.values[value] else {
        return None;
    };
    match graph.insts[inst].op {
        Op::LowLevel(low_level::Op::IntCompare { .. }) => Some(inst),
        Op::LowLevel(low_level::Op::FloatCompare { condition, .. })
            if float_conds(condition).1.is_none() =>
        {
            Some(inst)
        }
        _ => None,
    }
}

fn memory_address(
    graph: &Graph,
    value: Value,
    offset: i32,
    size: u32,
) -> Option<(Address, Vec<Inst>)> {
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
    if let Some(constant) = alloc::constant(graph, right) {
        return Some((
            Address::Offset {
                base: left,
                offset: i64::from(offset).wrapping_add(constant as i64),
            },
            vec![add],
        ));
    }
    if let Some(constant) = alloc::constant(graph, left) {
        return Some((
            Address::Offset {
                base: right,
                offset: i64::from(offset).wrapping_add(constant as i64),
            },
            vec![add],
        ));
    }
    if offset != 0 {
        return None;
    }
    let mut index = right;
    let mut skip = vec![add];
    let mut scaled = false;
    if let ValueDef::Result(inst, 0) = graph.values[index]
        && let Op::LowLevel(low_level::Op::Binary {
            op: low_level::Binary::ShiftLeft,
            left,
            right,
        }) = graph.insts[inst].op
        && alloc::constant(graph, right) == Some(u64::from(size.trailing_zeros()))
        && size > 1
    {
        index = left;
        skip.push(inst);
        scaled = true;
    }
    let mut kind = Index::Lsl;
    if let ValueDef::Result(inst, 0) = graph.values[index]
        && let Op::LowLevel(low_level::Op::Unary { op, value, .. }) = graph.insts[inst].op
        && graph.reprs[value].map(Repr::machine) == Some(types::I32)
    {
        kind = match op {
            low_level::Unary::ZeroExtend => Index::Uxtw,
            low_level::Unary::SignExtend => Index::Sxtw,
            _ => Index::Lsl,
        };
        if !matches!(kind, Index::Lsl) {
            index = value;
            skip.push(inst);
        }
    }
    Some((
        Address::Indexed {
            base: left,
            index,
            kind,
            scaled,
        },
        skip,
    ))
}

fn shifted_operand(
    graph: &Graph,
    value: Value,
    position: usize,
    uses: &SecondaryMap<Value, u32>,
) -> Option<(Operand, Vec<Inst>)> {
    if uses[value] != 1 {
        return None;
    }
    let wide = graph.reprs[value].map(Repr::machine) == Some(types::I64);
    let ValueDef::Result(inst, 0) = graph.values[value] else {
        return None;
    };
    let (mut leaf, shift, mut skip) = match graph.insts[inst].op {
        Op::LowLevel(low_level::Op::Binary {
            op: low_level::Binary::ShiftLeft,
            left,
            right,
        }) => {
            let bits = graph.reprs[value].map(Repr::machine)?.bits();
            (
                left,
                alloc::constant(graph, right)? as u32 & (bits - 1),
                vec![inst],
            )
        }
        _ => (value, 0, Vec::new()),
    };
    let mut kind = Index::Lsl;
    if wide && shift <= 4 && uses[leaf] == 1 {
        if let ValueDef::Result(def, 0) = graph.values[leaf] {
            if let Op::LowLevel(low_level::Op::Unary { op, value, .. }) = graph.insts[def].op {
                if graph.reprs[value].map(Repr::machine) == Some(types::I32) {
                    kind = match op {
                        low_level::Unary::ZeroExtend => Index::Uxtw,
                        low_level::Unary::SignExtend => Index::Sxtw,
                        _ => Index::Lsl,
                    };
                    if !matches!(kind, Index::Lsl) {
                        leaf = value;
                        skip.push(def);
                    }
                }
            }
        }
    }
    if skip.is_empty() {
        return None;
    }
    Some((
        Operand {
            position,
            value: leaf,
            kind,
            shift,
        },
        skip,
    ))
}

impl Selection {
    pub(super) fn new(graph: &Graph, order: &[Block]) -> Self {
        let mut uses = SecondaryMap::<Value, u32>::new();
        for &block in order {
            for &inst in &graph.blocks[block].insts {
                for value in graph.insts[inst].op.operands_iter() {
                    uses[value] += 1;
                }
            }
            if let Some(term) = &graph.blocks[block].terminator {
                for value in term.kind.operands_iter() {
                    uses[value] += 1;
                }
            }
        }
        let mut selection = Self {
            select_compare: SecondaryMap::new(),
            branch_compare: SecondaryMap::new(),
            skip: SecondaryMap::new(),
            address: SecondaryMap::new(),
            operand: SecondaryMap::new(),
        };
        let mut candidates = SecondaryMap::<Inst, bool>::new();
        for &block in order {
            for &inst in &graph.blocks[block].insts {
                let memory = match &graph.insts[inst].op {
                    Op::LowLevel(low_level::Op::Load {
                        ty,
                        address,
                        offset,
                        flags,
                    }) if flags.trap_code().is_none() => Some((*address, *offset, ty.bytes())),
                    Op::LowLevel(low_level::Op::Store {
                        value,
                        address,
                        offset,
                        flags,
                    }) if flags.trap_code().is_none() => graph.reprs[*value]
                        .map(Repr::machine)
                        .map(|ty| (*address, *offset, ty.bytes())),
                    _ => None,
                };
                if let Some((address, offset, size)) = memory {
                    if let Some((address, skip)) = memory_address(graph, address, offset, size) {
                        selection.address[inst] = Some(address);
                        for def in skip {
                            candidates[def] = true;
                        }
                    }
                }
                if let Op::LowLevel(low_level::Op::Binary {
                    op: operation @ (low_level::Binary::Add | low_level::Binary::Subtract),
                    left,
                    right,
                }) = graph.insts[inst].op
                {
                    if alloc::constant(graph, left).is_none()
                        && alloc::constant(graph, right).is_none()
                    {
                        let selected = shifted_operand(graph, right, 1, &uses).or_else(|| {
                            if matches!(operation, low_level::Binary::Add) {
                                shifted_operand(graph, left, 0, &uses)
                            } else {
                                None
                            }
                        });
                        if let Some((operand, skip)) = selected {
                            selection.operand[inst] = Some(operand);
                            for def in skip {
                                candidates[def] = true;
                            }
                        }
                    }
                }
                if let Op::LowLevel(low_level::Op::Select { condition, .. }) = graph.insts[inst].op
                {
                    if let Some(def) = fused_compare(graph, condition, &uses) {
                        selection.select_compare[inst] = Some(def);
                        candidates[def] = true;
                    }
                }
            }
            if let Some(term) = &graph.blocks[block].terminator {
                if let TerminatorKind::Branch(condition, ..) = term.kind {
                    if let Some(def) = fused_compare(graph, condition, &uses) {
                        selection.branch_compare[block] = Some(def);
                        candidates[def] = true;
                    }
                }
            }
        }
        let mut references = SecondaryMap::<Value, u32>::new();
        let mut operands = Vec::new();
        for &block in order {
            for &inst in &graph.blocks[block].insts {
                selection.operands(graph, inst, &mut operands);
                for &value in &operands {
                    references[value] += 1;
                }
            }
            selection.terminator_operands(graph, block, &mut operands);
            for &value in &operands {
                references[value] += 1;
            }
        }
        let mut work: Vec<_> = candidates
            .iter()
            .filter_map(|(inst, &candidate)| {
                (candidate && references[graph.insts[inst].results[0]] == 0).then_some(inst)
            })
            .collect();
        while let Some(inst) = work.pop() {
            if selection.skip[inst] {
                continue;
            }
            selection.operands(graph, inst, &mut operands);
            selection.skip[inst] = true;
            for &value in &operands {
                references[value] -= 1;
                if references[value] == 0
                    && let ValueDef::Result(def, 0) = graph.values[value]
                    && candidates[def]
                {
                    work.push(def);
                }
            }
        }
        selection
    }

    pub(super) fn operands(&self, graph: &Graph, inst: Inst, out: &mut Vec<Value>) {
        out.clear();
        if self.skip[inst] {
            return;
        }
        out.extend(graph.insts[inst].op.operands_iter());
        if let Some(address) = self.address[inst] {
            let position = if matches!(
                graph.insts[inst].op,
                Op::LowLevel(low_level::Op::Load { .. })
            ) {
                0
            } else {
                1
            };
            out.remove(position);
            match address {
                Address::Offset { base, .. } => out.push(base),
                Address::Indexed { base, index, .. } => out.extend([base, index]),
            }
        }
        if let Some(operand) = self.operand[inst] {
            out.remove(operand.position);
            out.push(operand.value);
        }
        if let Some(def) = self.select_compare[inst] {
            out.remove(0);
            out.extend(graph.insts[def].op.operands_iter());
        }
    }

    pub(super) fn terminator_operands(&self, graph: &Graph, block: Block, out: &mut Vec<Value>) {
        out.clear();
        let Some(term) = &graph.blocks[block].terminator else {
            return;
        };
        out.extend(term.kind.operands_iter());
        if let Some(def) = self.branch_compare[block] {
            out.remove(0);
            out.extend(graph.insts[def].op.operands_iter());
        }
    }

    pub(super) fn operand(&self, inst: Inst) -> Option<Operand> {
        self.operand[inst]
    }

    pub(super) fn address(&self, inst: Inst) -> Option<Address> {
        self.address[inst]
    }

    pub(super) fn skips(&self, inst: Inst) -> bool {
        self.skip[inst]
    }
    pub(super) fn select_compare(&self, inst: Inst) -> Option<Inst> {
        self.select_compare[inst]
    }
    pub(super) fn branch_compare(&self, block: Block) -> Option<Inst> {
        self.branch_compare[block]
    }
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
        let selection = Selection::new(&graph, &crate::jit::semantic::Cfg::new(&graph).rpo);
        assert!(selection.skips(wide_inst));
        assert!(selection.skips(shift_inst));
        assert!(selection.skips(compare_inst));
        let operand = selection.operand(sum_inst).unwrap();
        assert_eq!(operand.value, index);
        assert_eq!(operand.kind, Index::Uxtw);
        assert_eq!(operand.shift, 2);
        let mut operands = Vec::new();
        selection.operands(&graph, sum_inst, &mut operands);
        assert_eq!(operands, vec![base, index]);
        selection.terminator_operands(&graph, entry, &mut operands);
        assert!(operands.contains(&sum));
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
            let selection = Selection::new(&graph, &crate::jit::semantic::Cfg::new(&graph).rpo);
            for inst in [load, store] {
                assert!(matches!(selection.address(inst), Some(Address::Indexed {
                    base: b, index: i, kind: Index::Uxtw, scaled: true
                }) if b == base && i == index));
            }
            assert_eq!(selection.skips(address_inst), returned != 1);
            assert_eq!(selection.skips(shift_inst), returned != 2);
            assert_eq!(selection.skips(wide_inst), returned < 2);
            let mut operands = Vec::new();
            selection.operands(&graph, load, &mut operands);
            assert_eq!(operands, vec![base, index]);
            selection.operands(&graph, store, &mut operands);
            assert_eq!(operands, vec![loaded, base, index]);
        }
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

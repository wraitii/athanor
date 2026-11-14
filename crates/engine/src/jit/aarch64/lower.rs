use cranelift_codegen::binemit::Reloc;
use cranelift_codegen::entity::SecondaryMap;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{ArgumentExtension, Opcode, Signature, Type, types};
use cranelift_codegen::isa::CallConv;
use cranelift_module::{ModuleReloc, ModuleRelocTarget};

use super::Unsupported;
use super::abi::{self, Location};
use super::alloc::TARGET;
use super::asm::{Assembler, Label};
use super::encode::{
    self as e, Alu, Cond, Extend, Float, FloatOp, FloatUnary, Index, Mem, Shift, Width,
};
use super::select::{
    self as select, Address, Binary, CompareImmediate, Comparison, Immediate, Selection, remat_bits,
};
use crate::jit::alloc::moves::{self, Loc, Move};
use crate::jit::alloc::{
    RegClass,
    graph::{self as alloc, Allocation, Frame, Hint, Place},
};
use crate::jit::machine::{MachineCode, SourceRange};
use crate::jit::semantic::Repr;
use crate::jit::semantic::{Block, Cfg, Graph, Inst, Op, Target, TerminatorKind, Value, low_level};

const T0: u8 = 9;
const T1: u8 = 10;
const T2: u8 = 11;
const CALLEE: u8 = 16;
const ADDRESS: u8 = 17;
const F0: u8 = 16;
const F1: u8 = 17;

pub(super) fn lower(func: &Graph) -> Result<MachineCode, Unsupported> {
    let cfg = Cfg::new(func);
    check_function(func, &cfg.rpo)?;
    let order = block_order(func, &cfg.rpo);
    let hints = hints(func, &cfg.rpo)?;
    let selection = Selection::new(func, &cfg.rpo);
    let operands = |inst, out: &mut Vec<Value>| selection.operands(func, inst, out);
    let terminator_operands =
        |block, out: &mut Vec<Value>| selection.terminator_operands(func, block, out);
    let mut remat = SecondaryMap::new();
    for value in func.values.keys() {
        remat[value] = select::remat_cost(func, value);
    }
    let allocation = alloc::allocate(
        func,
        &cfg,
        &order,
        &TARGET,
        &hints,
        &operands,
        &terminator_operands,
        &remat,
        &|_| true,
    )
    .map_err(Unsupported)?;
    match lower_with(func, &order, &allocation, &selection, false)? {
        Some(machine) => Ok(machine),
        None => lower_with(func, &order, &allocation, &selection, true)?
            .ok_or_else(|| Unsupported("conditional branch out of range".into())),
    }
}

fn lower_with(
    func: &Graph,
    order: &[Block],
    allocation: &Allocation,
    selection: &Selection,
    far: bool,
) -> Result<Option<MachineCode>, Unsupported> {
    let conv = func.low_level_signature.as_ref().unwrap().call_conv;
    let incoming =
        abi::assign(conv, &func.low_level_signature.as_ref().unwrap().params)?.stack_size;
    let (outgoing, tail_calls) = call_stack_sizes(func, order)?;
    if tail_calls > 0 && conv != CallConv::Tail {
        return Err(Unsupported(format!("tail call from {conv}")));
    }
    let tail_args = incoming.max(tail_calls);
    let frame = Frame::new(func, allocation, outgoing);
    if frame.size >= 1 << 24 || tail_args >= 1 << 24 {
        return Err(Unsupported("frame too large".into()));
    }
    let entry = func.entry;
    let mut lower = Lower {
        func,
        allocation,
        selection,
        stubs: Vec::new(),
        block: entry,
        frame,
        incoming,
        tail_args,
        pop: if conv == CallConv::Tail { tail_args } else { 0 },
        asm: Assembler::new(far),
        labels: SecondaryMap::new(),
        relocs: Vec::new(),
        locations: Vec::new(),
        block_starts: Vec::new(),
    };
    for block in func.blocks.keys() {
        lower.labels[block] = Some(lower.asm.label());
    }
    lower.prologue()?;
    for (index, &block) in order.iter().enumerate() {
        let next = order.get(index + 1).copied();
        lower.block(block, next)?;
    }
    for (label, moves, target) in std::mem::take(&mut lower.stubs) {
        lower.asm.bind(label);
        lower.parallel(moves);
        let target = lower.label(target);
        lower.asm.jump(e::B, target);
    }
    let Lower {
        asm,
        relocs,
        locations,
        block_starts,
        ..
    } = lower;
    if asm.conditional_overflow() {
        return Ok(None);
    }
    Ok(Some(MachineCode {
        bytes: asm.finish()?,
        alignment: 4,
        relocs,
        traps: Vec::new(),
        locations,
        block_starts,
    }))
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

fn call_stack_sizes(graph: &Graph, order: &[Block]) -> Result<(u32, u32), Unsupported> {
    let (mut outgoing, mut tail) = (0, 0);
    for &block in order {
        for &inst in &graph.blocks[block].insts {
            if let Op::LowLevel(low_level::Op::Call { signature, .. }) = &graph.insts[inst].op {
                outgoing =
                    outgoing.max(abi::assign(signature.call_conv, &signature.params)?.stack_size);
            }
        }
        if let Some(term) = &graph.blocks[block].terminator {
            if let TerminatorKind::TailCall { signature, .. } = &term.kind {
                tail = tail.max(abi::assign(signature.call_conv, &signature.params)?.stack_size);
            }
        }
    }
    Ok((outgoing, tail))
}

fn hints(graph: &Graph, order: &[Block]) -> Result<Vec<Hint>, Unsupported> {
    let mut hints = Vec::new();
    let mut add = |values: &[Value], locations: &[Location], block: Block| {
        for (&value, location) in values.iter().zip(locations) {
            if let Location::Reg { class, reg } = *location {
                hints.push(Hint {
                    value,
                    class,
                    reg,
                    block,
                });
            }
        }
    };
    let signature = graph.low_level_signature.as_ref().unwrap();
    let locations = abi::assign(signature.call_conv, &signature.params)?.locations;
    add(&graph.blocks[graph.entry].params, &locations, graph.entry);
    for &block in order {
        for &inst in &graph.blocks[block].insts {
            if let Op::LowLevel(low_level::Op::Call {
                signature, args, ..
            }) = &graph.insts[inst].op
            {
                let locations = abi::assign(signature.call_conv, &signature.params)?.locations;
                add(args, &locations, block);
                let locations = abi::assign_returns(signature.call_conv, &signature.returns)?;
                add(&graph.insts[inst].results, &locations, block);
            }
        }
        if let Some(term) = &graph.blocks[block].terminator {
            match &term.kind {
                TerminatorKind::ReturnValues(values) => {
                    let locations = abi::assign_returns(signature.call_conv, &signature.returns)?;
                    add(values, &locations, block);
                }
                TerminatorKind::TailCall {
                    signature, args, ..
                } => {
                    let locations = abi::assign(signature.call_conv, &signature.params)?.locations;
                    add(args, &locations, block);
                }
                _ => {}
            }
        }
    }
    Ok(hints)
}

fn block_order(graph: &Graph, order: &[Block]) -> Vec<Block> {
    let (cold, mut hot): (Vec<_>, Vec<_>) = order
        .iter()
        .copied()
        .partition(|&block| graph.cold.contains(&block) && block != graph.entry);
    hot.extend(cold);
    hot
}

pub(super) fn width(ty: Type) -> Width {
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

pub(super) fn float_conds(cond: FloatCC) -> (Cond, Option<Cond>) {
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

pub(super) fn load_op(opcode: Opcode, ty: Type) -> Option<Mem> {
    Some(match opcode {
        Opcode::Load => match ty {
            types::I8 => Mem::Ldrb,
            types::I16 => Mem::Ldrh,
            types::I32 => Mem::LdrW,
            types::I64 => Mem::LdrX,
            types::F32 => Mem::LdrS,
            types::F64 => Mem::LdrD,
            _ => return None,
        },
        Opcode::Uload8 => Mem::Ldrb,
        Opcode::Sload8 => Mem::Ldrsb,
        Opcode::Uload16 => Mem::Ldrh,
        Opcode::Sload16 => Mem::Ldrsh,
        Opcode::Uload32 => Mem::LdrW,
        Opcode::Sload32 => Mem::Ldrsw,
        _ => return None,
    })
}

pub(super) fn store_op(opcode: Opcode, ty: Type) -> Option<Mem> {
    Some(match opcode {
        Opcode::Store => match ty {
            types::I8 => Mem::Strb,
            types::I16 => Mem::Strh,
            types::I32 => Mem::StrW,
            types::I64 => Mem::StrX,
            types::F32 => Mem::StrS,
            types::F64 => Mem::StrD,
            _ => return None,
        },
        Opcode::Istore8 => Mem::Strb,
        Opcode::Istore16 => Mem::Strh,
        Opcode::Istore32 => Mem::StrW,
        _ => return None,
    })
}

#[derive(Clone, Copy)]
enum Test {
    Flags(Cond),
    Zero {
        nonzero: bool,
        width: Width,
        reg: u8,
    },
}

impl Test {
    fn invert(self) -> Self {
        match self {
            Self::Flags(cond) => Self::Flags(cond.invert()),
            Self::Zero {
                nonzero,
                width,
                reg,
            } => Self::Zero {
                nonzero: !nonzero,
                width,
                reg,
            },
        }
    }
}

struct Lower<'a> {
    func: &'a Graph,
    allocation: &'a Allocation,
    selection: &'a Selection,
    stubs: Vec<(Label, Vec<Move>, Block)>,
    block: Block,
    frame: Frame,
    incoming: u32,
    tail_args: u32,
    pop: u32,
    asm: Assembler,
    labels: SecondaryMap<Block, Option<Label>>,
    relocs: Vec<ModuleReloc>,
    locations: Vec<SourceRange>,
    block_starts: Vec<u32>,
}

impl Lower<'_> {
    fn label(&self, block: Block) -> Label {
        self.labels[block].expect("every block has a label")
    }

    fn ty(&self, value: Value) -> Type {
        self.func.reprs[value].map(Repr::machine).unwrap()
    }

    fn class(&self, value: Value) -> RegClass {
        RegClass::of(self.ty(value)).expect("value types are checked")
    }

    fn constant(&self, value: Value) -> Option<u64> {
        alloc::constant(self.func, value)
    }

    fn loc_in(&self, value: Value, block: Block) -> Loc {
        match self.allocation.place(value, block) {
            Place::Reg(reg) => Loc::Reg(self.class(value), reg),
            Place::Slot(index) => Loc::Slot(self.frame.slot(index)),
            Place::Remat => {
                Loc::Imm(remat_bits(self.func, value).expect("rematerialized values are constants"))
            }
        }
    }

    fn loc(&self, value: Value) -> Loc {
        self.loc_in(value, self.block)
    }

    fn mov_imm(&mut self, width: Width, rd: u8, value: u64) {
        let asm = &mut self.asm;
        e::mov_imm(width, rd, value, |word| asm.emit(word));
    }

    fn load_imm(&mut self, rd: u8, value: u64) {
        let asm = &mut self.asm;
        e::load_imm(rd, value, |word| asm.emit(word));
    }

    fn materialize(&mut self, class: RegClass, rd: u8, bits: u64) {
        match class {
            RegClass::Int => self.load_imm(rd, bits),
            RegClass::Float if bits == 0 => self.asm.emit(e::move_to_float(Width::X, rd, e::ZR)),
            RegClass::Float => match e::float_imm8(bits) {
                Some(imm8) => self.asm.emit(e::fmov_imm(rd, imm8)),
                None => {
                    self.load_imm(ADDRESS, bits);
                    self.asm.emit(e::move_to_float(Width::X, rd, ADDRESS));
                }
            },
        }
    }

    fn binary_imm(&mut self, inst: Inst, result: Value) -> bool {
        let Some(Binary { x, immediate, .. }) = select::binary(self.func, inst) else {
            return false;
        };
        let width = width(self.ty(result));
        let a = self.use_reg(x, T0);
        let d = self.def_reg(result, T0);
        let word = match immediate {
            Immediate::Add(imm, shift) => e::add_imm(width, d, a, imm, shift),
            Immediate::Sub(imm, shift) => e::sub_imm(width, d, a, imm, shift),
            Immediate::Logical(op, bits) => {
                e::logical_imm(op, width, d, a, bits).expect("logical immediates are validated")
            }
        };
        self.asm.emit(word);
        self.commit(result, d);
        true
    }

    fn compare(&mut self, cond: IntCC, x: Value, y: Value) -> Cond {
        let Comparison {
            cond,
            x,
            y,
            immediate,
        } = select::comparison(self.func, cond, x, y);
        let ty = self.ty(x);
        let signed = matches!(
            cond,
            IntCC::SignedLessThan
                | IntCC::SignedLessThanOrEqual
                | IntCC::SignedGreaterThan
                | IntCC::SignedGreaterThanOrEqual
        );
        let extend = match (ty, signed) {
            (types::I8, false) => Some(Extend::Uxtb),
            (types::I8, true) => Some(Extend::Sxtb),
            (types::I16, false) => Some(Extend::Uxth),
            (types::I16, true) => Some(Extend::Sxth),
            _ => None,
        };
        let width = width(ty);
        let mut a = self.use_reg(x, T0);
        if let Some(extend) = extend {
            self.asm.emit(e::extend(extend, Width::W, T0, a));
            a = T0;
        }
        match immediate {
            Some(CompareImmediate::Subs(imm, shift)) => {
                self.asm.emit(e::subs_imm(width, e::ZR, a, imm, shift));
            }
            Some(CompareImmediate::Adds(imm, shift)) => {
                self.asm.emit(e::adds_imm(width, e::ZR, a, imm, shift));
            }
            None => {
                let mut b = self.use_reg(y, T1);
                if let Some(extend) = extend {
                    self.asm.emit(e::extend(extend, Width::W, T1, b));
                    b = T1;
                }
                self.asm.emit(e::alu(Alu::Subs, width, e::ZR, a, b));
            }
        }
        int_cond(cond)
    }

    fn access(&mut self, inst: Inst, op: Mem, rt: u8, base: Value, offset: i64) {
        match self.selection.address(inst) {
            Some(Address::Offset { base, offset }) => {
                let base = self.use_reg(base, T1);
                self.memory(op, rt, base, offset);
            }
            Some(Address::Indexed {
                base,
                index,
                kind,
                scaled,
            }) => {
                let base = self.use_reg(base, T1);
                let index = self.use_reg(index, T2);
                self.asm
                    .emit(e::mem_indexed(op, rt, base, index, kind, scaled));
            }
            None => {
                let base = self.use_reg(base, T1);
                self.memory(op, rt, base, offset);
            }
        }
    }

    fn memory(&mut self, op: Mem, rt: u8, base: u8, offset: i64) {
        if let Ok(unsigned) = u32::try_from(offset)
            && let Some(word) = e::mem_scaled(op, rt, base, unsigned)
        {
            self.asm.emit(word);
            return;
        }
        if let Ok(signed) = i32::try_from(offset)
            && let Some(word) = e::mem_unscaled(op, rt, base, signed)
        {
            self.asm.emit(word);
            return;
        }
        self.mov_imm(Width::X, ADDRESS, offset as u64);
        self.asm.emit(e::mem_register(op, rt, base, ADDRESS));
    }

    fn spill_op(class: RegClass, load: bool) -> Mem {
        match (class, load) {
            (RegClass::Int, true) => Mem::LdrX,
            (RegClass::Int, false) => Mem::StrX,
            (RegClass::Float, true) => Mem::LdrD,
            (RegClass::Float, false) => Mem::StrD,
        }
    }

    fn use_reg(&mut self, value: Value, scratch: u8) -> u8 {
        match self.loc(value) {
            Loc::Reg(_, reg) => reg,
            Loc::Slot(offset) => {
                let op = Self::spill_op(self.class(value), true);
                self.memory(op, scratch, e::SP, i64::from(offset));
                scratch
            }
            Loc::Imm(bits) => {
                self.materialize(self.class(value), scratch, bits);
                scratch
            }
        }
    }

    fn def_reg(&self, value: Value, scratch: u8) -> u8 {
        match self.loc(value) {
            Loc::Reg(_, reg) => reg,
            Loc::Slot(_) | Loc::Imm(_) => scratch,
        }
    }

    fn commit(&mut self, value: Value, reg: u8) {
        if let Loc::Slot(offset) = self.loc(value) {
            let op = Self::spill_op(self.class(value), false);
            self.memory(op, reg, e::SP, i64::from(offset));
        }
    }

    fn emit_move(&mut self, m: Move) {
        match (m.from, m.to) {
            (Loc::Reg(_, from), Loc::Reg(_, to)) => {
                let word = match m.class {
                    RegClass::Int => e::mov(Width::X, to, from),
                    RegClass::Float => e::float_unary(FloatUnary::Mov, Float::D, to, from),
                };
                self.asm.emit(word);
            }
            (Loc::Reg(_, from), Loc::Slot(offset)) => {
                self.memory(
                    Self::spill_op(m.class, false),
                    from,
                    e::SP,
                    i64::from(offset),
                );
            }
            (Loc::Slot(offset), Loc::Reg(_, to)) => {
                self.memory(Self::spill_op(m.class, true), to, e::SP, i64::from(offset));
            }
            (Loc::Imm(bits), Loc::Reg(_, to)) => self.materialize(m.class, to, bits),
            (Loc::Imm(bits), Loc::Slot(offset)) => {
                self.load_imm(T0, bits);
                self.memory(Mem::StrX, T0, e::SP, i64::from(offset));
            }
            (_, Loc::Imm(_)) => unreachable!("constants are never move destinations"),
            (Loc::Slot(from), Loc::Slot(to)) => {
                let temporary = match m.class {
                    RegClass::Int => T0,
                    RegClass::Float => F0,
                };
                self.memory(
                    Self::spill_op(m.class, true),
                    temporary,
                    e::SP,
                    i64::from(from),
                );
                self.memory(
                    Self::spill_op(m.class, false),
                    temporary,
                    e::SP,
                    i64::from(to),
                );
            }
        }
    }

    fn parallel(&mut self, pending: Vec<Move>) {
        let ordered = moves::sequence(pending, |class| match class {
            RegClass::Int => Loc::Reg(class, T1),
            RegClass::Float => Loc::Reg(class, F1),
        });
        for m in ordered {
            self.emit_move(m);
        }
    }

    fn incoming_area(&self) -> i64 {
        i64::from(16 + self.tail_args - self.incoming)
    }

    fn adjust_sp(&mut self, size: u32, sub: bool) {
        let encode = if sub { e::sub_imm } else { e::add_imm };
        if size >> 12 != 0 {
            self.asm
                .emit(encode(Width::X, e::SP, e::SP, size >> 12, true));
        }
        if size & 0xFFF != 0 {
            self.asm
                .emit(encode(Width::X, e::SP, e::SP, size & 0xFFF, false));
        }
    }

    fn callee_saves(&mut self, load: bool) {
        let saved = &self.allocation.callee_saved;
        let mut index = 0;
        while index < saved.len() {
            let (class, reg) = saved[index];
            let offset = self.frame.saves + 8 * index as u32;
            if let Some(&(next_class, next)) = saved.get(index + 1)
                && next_class == class
                && let Some(word) =
                    e::pair(load, class == RegClass::Float, reg, next, e::SP, offset)
            {
                self.asm.emit(word);
                index += 2;
                continue;
            }
            self.memory(Self::spill_op(class, load), reg, e::SP, i64::from(offset));
            index += 1;
        }
    }

    fn prologue(&mut self) -> Result<(), Unsupported> {
        let func = self.func;
        let signature = &func.low_level_signature.as_ref().unwrap();
        let locations = abi::assign(signature.call_conv, &signature.params)?.locations;
        abi::assign_returns(signature.call_conv, &signature.returns)?;
        self.adjust_sp(self.tail_args - self.incoming, true);
        self.asm.emit(e::STP_FP_LR_PRE);
        self.asm.emit(e::mov_sp(e::FP, e::SP));
        self.adjust_sp(self.frame.size, true);
        self.callee_saves(false);
        let params = &func.blocks[self.block].params;
        let moves = params
            .iter()
            .zip(&locations)
            .filter_map(|(&param, location)| match *location {
                Location::Reg { class, reg } => Some(Move {
                    class,
                    from: Loc::Reg(class, reg),
                    to: self.loc(param),
                }),
                Location::Stack { .. } => None,
            })
            .collect();
        self.parallel(moves);
        let area = self.incoming_area();
        for (&param, location) in params.iter().zip(locations) {
            let Location::Stack { offset, ty } = location else {
                continue;
            };
            let op = load_op(Opcode::Load, ty).expect("argument types are checked");
            let scratch = match self.class(param) {
                RegClass::Int => T0,
                RegClass::Float => F0,
            };
            let reg = self.def_reg(param, scratch);
            self.memory(op, reg, e::FP, area + i64::from(offset));
            self.commit(param, reg);
        }
        Ok(())
    }

    fn epilogue(&mut self, pop: u32) {
        self.callee_saves(true);
        self.asm.emit(e::mov_sp(e::SP, e::FP));
        self.asm.emit(e::LDP_FP_LR_POST);
        self.adjust_sp(pop, false);
    }

    fn block(&mut self, block: Block, next: Option<Block>) -> Result<(), Unsupported> {
        self.block = block;
        self.block_starts.push(self.asm.offset());
        let label = self.label(block);
        self.asm.bind(label);
        for &inst in &self.func.blocks[block].insts {
            let start = self.asm.offset();
            self.inst(inst)?;
            let end = self.asm.offset();
            if end > start {
                self.locations.push(SourceRange {
                    start,
                    end,
                    location: self.func.insts[inst].emission_location,
                });
            }
        }
        let terminator = self.func.blocks[block]
            .terminator
            .as_ref()
            .ok_or_else(|| Unsupported(format!("missing terminator for {block}")))?;
        let start = self.asm.offset();
        self.terminator(&terminator.kind, next)?;
        let end = self.asm.offset();
        if end > start {
            self.locations.push(SourceRange {
                start,
                end,
                location: terminator.emission_location,
            });
        }
        Ok(())
    }

    fn edge_moves(&self, target: &Target) -> Vec<Move> {
        let mut pending = Vec::new();
        for (&arg, &param) in target
            .args
            .iter()
            .zip(&self.func.blocks[target.block].params)
        {
            pending.push(Move {
                class: self.class(param),
                from: self.loc(arg),
                to: self.loc_in(param, target.block),
            });
        }
        if self.allocation.is_cold(self.block) != self.allocation.is_cold(target.block) {
            for &value in self.allocation.boundary(target.block) {
                let to = self.loc_in(value, target.block);
                if !matches!(to, Loc::Imm(_)) {
                    pending.push(Move {
                        class: self.class(value),
                        from: self.loc(value),
                        to,
                    });
                }
            }
        }
        pending.retain(|m| m.from != m.to);
        pending
    }

    fn edge(&mut self, target: &Target, next: Option<Block>) {
        let moves = self.edge_moves(target);
        self.parallel(moves);
        if next != Some(target.block) {
            let label = self.label(target.block);
            self.asm.jump(e::B, label);
        }
    }

    fn edge_target(&mut self, target: &Target) -> Label {
        let moves = self.edge_moves(target);
        if moves.is_empty() {
            return self.label(target.block);
        }
        let label = self.asm.label();
        self.stubs.push((label, moves, target.block));
        label
    }

    fn condition(&mut self, value: Value) -> Cond {
        let ty = self.ty(value);
        let reg = self.use_reg(value, T0);
        match ty {
            types::I8 => self.asm.emit(e::tst_low_byte(reg)),
            types::I16 => {
                self.asm.emit(e::extend(Extend::Uxth, Width::W, T0, reg));
                self.asm.emit(e::alu(Alu::Subs, Width::W, e::ZR, T0, e::ZR));
            }
            _ => self
                .asm
                .emit(e::alu(Alu::Subs, width(ty), e::ZR, reg, e::ZR)),
        }
        Cond::Ne
    }

    fn fused_flags(&mut self, inst: Inst) -> Cond {
        match self.func.insts[inst].op {
            Op::LowLevel(low_level::Op::IntCompare {
                condition,
                left,
                right,
            }) => self.compare(condition, left, right),
            Op::LowLevel(low_level::Op::FloatCompare {
                condition,
                left,
                right,
            }) => {
                let ty = float(self.ty(left));
                let a = self.use_reg(left, F0);
                let b = self.use_reg(right, F1);
                self.asm.emit(e::fcmp(ty, a, b));
                float_conds(condition).0
            }
            _ => unreachable!(),
        }
    }

    fn fused_branch_test(&mut self, inst: Inst) -> Test {
        if let Op::LowLevel(low_level::Op::IntCompare {
            condition: cond @ (IntCC::Equal | IntCC::NotEqual),
            left,
            right,
        }) = self.func.insts[inst].op
        {
            let value = if self.constant(right) == Some(0) {
                Some(left)
            } else if self.constant(left) == Some(0) {
                Some(right)
            } else {
                None
            };
            if let Some(value) = value {
                let ty = self.ty(value);
                if matches!(ty, types::I32 | types::I64) {
                    let reg = self.use_reg(value, T0);
                    return Test::Zero {
                        nonzero: cond == IntCC::NotEqual,
                        width: width(ty),
                        reg,
                    };
                }
            }
        }
        Test::Flags(self.fused_flags(inst))
    }

    fn branch_test(&mut self, value: Value) -> Test {
        let ty = self.ty(value);
        if matches!(ty, types::I32 | types::I64) {
            let reg = self.use_reg(value, T0);
            Test::Zero {
                nonzero: true,
                width: width(ty),
                reg,
            }
        } else {
            Test::Flags(self.condition(value))
        }
    }

    fn branch_if(&mut self, test: Test, target: Label) {
        match test {
            Test::Flags(cond) => self.asm.branch(cond, target),
            Test::Zero {
                nonzero,
                width,
                reg,
            } => self.asm.conditional(
                e::compare_branch(nonzero, width, reg),
                e::compare_branch(!nonzero, width, reg),
                target,
            ),
        }
    }

    fn call(
        &mut self,
        signature: &Signature,
        args: &[Value],
        results: &[Value],
        callee: Option<Value>,
        tail: bool,
        emit: impl FnOnce(&mut Self),
    ) -> Result<(), Unsupported> {
        let conv = signature.call_conv;
        let assignment = abi::assign(conv, &signature.params)?;
        let (base, area) = if tail {
            (
                e::FP,
                i64::from(16 + self.tail_args - assignment.stack_size),
            )
        } else {
            (e::SP, 0)
        };
        let mut pending = Vec::with_capacity(args.len() + 1);
        for (&arg, location) in args.iter().zip(&assignment.locations) {
            match *location {
                Location::Reg { class, reg } => pending.push(Move {
                    class,
                    from: self.loc(arg),
                    to: Loc::Reg(class, reg),
                }),
                Location::Stack { offset, ty } => {
                    let op = store_op(Opcode::Store, ty).expect("argument types are checked");
                    let scratch = match self.class(arg) {
                        RegClass::Int => T0,
                        RegClass::Float => F0,
                    };
                    let reg = self.use_reg(arg, scratch);
                    self.memory(op, reg, base, area + i64::from(offset));
                }
            }
        }
        if let Some(callee) = callee {
            pending.push(Move {
                class: RegClass::Int,
                from: self.loc(callee),
                to: Loc::Reg(RegClass::Int, CALLEE),
            });
        }
        self.parallel(pending);
        if conv == CallConv::AppleAarch64 {
            for (param, location) in signature.params.iter().zip(&assignment.locations) {
                let Location::Reg { reg, .. } = *location else {
                    continue;
                };
                let extend = match (param.extension, param.value_type) {
                    (ArgumentExtension::Uext, types::I8) => Extend::Uxtb,
                    (ArgumentExtension::Uext, types::I16) => Extend::Uxth,
                    (ArgumentExtension::Sext, types::I8) => Extend::Sxtb,
                    (ArgumentExtension::Sext, types::I16) => Extend::Sxth,
                    _ => continue,
                };
                self.asm.emit(e::extend(extend, Width::W, reg, reg));
            }
        }
        emit(self);
        if tail {
            return Ok(());
        }
        if conv == CallConv::Tail {
            self.adjust_sp(assignment.stack_size, true);
        }
        let locations = abi::assign_returns(conv, &signature.returns)?;
        let mut pending = Vec::new();
        for (&result, location) in results.iter().zip(locations) {
            if let Location::Reg { class, reg } = location {
                pending.push(Move {
                    class,
                    from: Loc::Reg(class, reg),
                    to: self.loc(result),
                });
            }
        }
        self.parallel(pending);
        Ok(())
    }

    fn emit_call(&mut self, word: u32, target: ModuleRelocTarget) {
        self.relocs.push(ModuleReloc {
            offset: self.asm.offset(),
            kind: Reloc::Arm64Call,
            name: target,
            addend: 0,
        });
        self.asm.emit(word);
    }

    fn terminator(
        &mut self,
        kind: &TerminatorKind,
        next: Option<Block>,
    ) -> Result<(), Unsupported> {
        match kind {
            TerminatorKind::Jump(target) => self.edge(target, next),
            TerminatorKind::Branch(value, yes, no) => {
                let test = match self.selection.branch_compare(self.block) {
                    Some(def) => self.fused_branch_test(def),
                    None => self.branch_test(*value),
                };
                let then_block = yes.block;
                let then_moves = self.edge_moves(yes);
                if next == Some(then_block) && then_moves.is_empty() {
                    let target = self.edge_target(no);
                    self.branch_if(test.invert(), target);
                } else {
                    let target = self.edge_target(yes);
                    self.branch_if(test, target);
                    self.edge(no, next);
                }
            }
            TerminatorKind::Switch(value, cases, default) => {
                let ty = self.ty(*value);
                let mut reg = self.use_reg(*value, T0);
                let extension = match ty {
                    types::I8 => Some(Extend::Uxtb),
                    types::I16 => Some(Extend::Uxth),
                    _ => None,
                };
                if let Some(extension) = extension {
                    self.asm.emit(e::extend(extension, Width::W, T0, reg));
                    reg = T0;
                }
                for (key, target) in cases {
                    if *key >= (1u128 << ty.bits()) {
                        continue;
                    }
                    if let Some((imm, shift)) = e::arith_imm(*key as i64) {
                        self.asm
                            .emit(e::subs_imm(width(ty), e::ZR, reg, imm, shift));
                    } else {
                        self.mov_imm(Width::X, T1, *key as u64);
                        self.asm.emit(e::alu(Alu::Subs, width(ty), e::ZR, reg, T1));
                    }
                    let label = self.edge_target(target);
                    self.asm.branch(Cond::Eq, label);
                }
                self.edge(default, next);
            }
            TerminatorKind::ReturnValues(values) => {
                let signature = self.func.low_level_signature.as_ref().unwrap();
                let locations = abi::assign_returns(signature.call_conv, &signature.returns)?;
                let mut pending = Vec::with_capacity(values.len());
                for (&value, location) in values.iter().zip(locations) {
                    if let Location::Reg { class, reg } = location {
                        pending.push(Move {
                            class,
                            from: self.loc(value),
                            to: Loc::Reg(class, reg),
                        });
                    }
                }
                self.parallel(pending);
                self.epilogue(self.pop);
                self.asm.emit(e::RET);
            }
            TerminatorKind::TailCall {
                target,
                signature,
                args,
            } => match target {
                low_level::CallTarget::Direct(id) => {
                    self.call(signature, args, &[], None, true, |_| {})?;
                    let size = abi::assign(signature.call_conv, &signature.params)?.stack_size;
                    self.epilogue(self.tail_args - size);
                    self.emit_call(e::B, ModuleRelocTarget::user(0, id.as_u32()));
                }
                low_level::CallTarget::Indirect(value) => {
                    self.call(signature, args, &[], Some(*value), true, |_| {})?;
                    let size = abi::assign(signature.call_conv, &signature.params)?.stack_size;
                    self.epilogue(self.tail_args - size);
                    self.asm.emit(e::br(CALLEE));
                }
            },
            _ => return Err(Unsupported("non-low-level terminator".into())),
        }
        Ok(())
    }

    fn inst(&mut self, inst: Inst) -> Result<(), Unsupported> {
        if self.selection.skips(inst) {
            return Ok(());
        }
        let data = &self.func.insts[inst];
        let result = data.results.first().copied();
        let result_ty = result.map(|result| self.ty(result));
        if let Op::Memory(low_level_memory) = &data.op {
            use crate::jit::semantic::{MemoryFormat as F, MemoryOp as M};
            let result = result.ok_or_else(|| Unsupported("memory result".into()))?;
            match *low_level_memory {
                M::IndexedLoad { .. } => {
                    let (address, op) = select::indexed_load(self.func, inst)
                        .ok_or_else(|| Unsupported("indexed load".into()))?;
                    let Address::Indexed {
                        base,
                        index,
                        kind,
                        scaled,
                    } = address
                    else {
                        unreachable!()
                    };
                    let base = self.use_reg(base, T1);
                    let index = self.use_reg(index, T2);
                    let scratch = if matches!(op, Mem::LdrS | Mem::LdrD) {
                        F0
                    } else {
                        T0
                    };
                    let d = self.def_reg(result, scratch);
                    self.asm
                        .emit(e::mem_indexed(op, d, base, index, kind, scaled));
                    self.commit(result, d);
                }
                M::Address {
                    base,
                    index,
                    format,
                } => {
                    if self.ty(index) != types::I64 {
                        return Err(Unsupported("memory address index type".into()));
                    }
                    let base = self.use_reg(base, T0);
                    let index = self.use_reg(index, T1);
                    let d = self.def_reg(result, T0);
                    self.asm.emit(e::add_lsl(
                        Width::X,
                        d,
                        base,
                        index,
                        format.bytes().trailing_zeros(),
                    ));
                    self.commit(result, d);
                }
                M::Load { address, format } => {
                    let op = match format {
                        F::U8 | F::I8 => Mem::Ldrb,
                        F::U16 | F::I16 => Mem::Ldrh,
                        F::U32 | F::I32 => Mem::LdrW,
                        F::F32 => Mem::LdrS,
                        F::F64 => Mem::LdrD,
                    };
                    let scratch = if matches!(op, Mem::LdrS | Mem::LdrD) {
                        F0
                    } else {
                        T0
                    };
                    let d = self.def_reg(result, scratch);
                    self.access(inst, op, d, address, 0);
                    self.commit(result, d);
                }
            }
            return Ok(());
        }
        let Op::LowLevel(op) = &data.op else {
            return Err(Unsupported("non-low-level operation".into()));
        };
        if let low_level::Op::Call {
            target,
            signature,
            args,
            ..
        } = op
        {
            match target {
                low_level::CallTarget::Direct(id) => {
                    let target = ModuleRelocTarget::user(0, id.as_u32());
                    self.call(signature, args, &data.results, None, false, |this| {
                        this.emit_call(e::BL, target)
                    })?;
                }
                low_level::CallTarget::Indirect(callee) => {
                    self.call(
                        signature,
                        args,
                        &data.results,
                        Some(*callee),
                        false,
                        |this| this.asm.emit(e::blr(CALLEE)),
                    )?;
                }
            }
            return Ok(());
        }
        let args = data.op.operands();
        use low_level::{Binary as B, Op as L, Unary as U};
        let opcode = match op {
            L::Integer { .. } => Opcode::Iconst,
            L::Float64(_) => Opcode::F64const,
            L::Unary { op, .. } => match op {
                U::BitNot => Opcode::Bnot,
                U::FloatNeg => Opcode::Fneg,
                U::FloatAbs => Opcode::Fabs,
                U::Ceil => Opcode::Ceil,
                U::Floor => Opcode::Floor,
                U::Trunc => Opcode::Trunc,
                U::Sqrt => Opcode::Sqrt,
                U::Reduce => Opcode::Ireduce,
                U::SignExtend => Opcode::Sextend,
                U::ZeroExtend => Opcode::Uextend,
                U::Bitcast => Opcode::Bitcast,
                U::FloatPromote => Opcode::Fpromote,
                U::FloatDemote => Opcode::Fdemote,
                U::FloatFromSigned => Opcode::FcvtFromSint,
                U::FloatFromUnsigned => Opcode::FcvtFromUint,
                U::SignedSaturatingFromFloat => Opcode::FcvtToSintSat,
            },
            L::Binary { op, .. } => match op {
                B::Add => Opcode::Iadd,
                B::Subtract => Opcode::Isub,
                B::Multiply => Opcode::Imul,
                B::And => Opcode::Band,
                B::Or => Opcode::Bor,
                B::Xor => Opcode::Bxor,
                B::ShiftLeft => Opcode::Ishl,
                B::ShiftRightSigned => Opcode::Sshr,
                B::ShiftRightUnsigned => Opcode::Ushr,
                B::FloatAdd => Opcode::Fadd,
                B::FloatSubtract => Opcode::Fsub,
                B::FloatMultiply => Opcode::Fmul,
                B::FloatDivide => Opcode::Fdiv,
                B::FloatMin => Opcode::Fmin,
                B::FloatMax => Opcode::Fmax,
                B::BooleanAnd | B::BooleanOr | B::BooleanXor | B::BooleanEqual => {
                    return Err(Unsupported("unexpanded boolean operation".into()));
                }
            },
            L::IntCompare { .. } => Opcode::Icmp,
            L::FloatCompare { .. } => Opcode::Fcmp,
            L::Select { .. } => Opcode::Select,
            L::Load { .. } => Opcode::Load,
            L::AtomicLoad { .. } => Opcode::AtomicLoad,
            L::Store { .. } => Opcode::Store,
            L::StackAddress { .. } => Opcode::StackAddr,
            L::StackPointer { .. } => Opcode::GetStackPointer,
            L::Cleanup { .. } | L::Scratch { .. } | L::Reference { .. } | L::FrameStore { .. } => {
                return Err(Unsupported("unlowered runtime operation".into()));
            }
            L::VariableRead => return Err(Unsupported("unresolved variable".into())),
            L::ExtractBits { .. } | L::InRange { .. } | L::TestMask { .. } => {
                return Err(Unsupported("unexpanded bit operation".into()));
            }
            L::Call { .. } => unreachable!(),
        };
        match opcode {
            Opcode::Iconst | Opcode::F64const | Opcode::F32const => {
                let result = result.unwrap();
                if let Loc::Reg(class, reg) = self.loc(result) {
                    let bits = remat_bits(self.func, result).expect("constants have bits");
                    self.materialize(class, reg, bits);
                }
            }
            Opcode::Iadd
            | Opcode::Isub
            | Opcode::Imul
            | Opcode::Band
            | Opcode::Bor
            | Opcode::Bxor => {
                let result = result.unwrap();
                if self.binary_imm(inst, result) {
                    return Ok(());
                }
                let width = width(result_ty.unwrap());
                if let Some(operand) = self.selection.operand(inst) {
                    let a = self.use_reg(args[1 - operand.position], T0);
                    let b = self.use_reg(operand.value, T1);
                    let d = self.def_reg(result, T0);
                    let sub = opcode == Opcode::Isub;
                    let word = match operand.kind {
                        Index::Lsl => e::alu_shifted(sub, width, d, a, b, operand.shift),
                        kind => e::alu_extended(sub, d, a, b, kind, operand.shift),
                    };
                    self.asm.emit(word);
                    self.commit(result, d);
                    return Ok(());
                }
                let a = self.use_reg(args[0], T0);
                let b = self.use_reg(args[1], T1);
                let d = self.def_reg(result, T0);
                let word = match opcode {
                    Opcode::Iadd => e::alu(Alu::Add, width, d, a, b),
                    Opcode::Isub => e::alu(Alu::Sub, width, d, a, b),
                    Opcode::Imul => e::mul(width, d, a, b),
                    Opcode::Band => e::alu(Alu::And, width, d, a, b),
                    Opcode::Bor => e::alu(Alu::Orr, width, d, a, b),
                    _ => e::alu(Alu::Eor, width, d, a, b),
                };
                self.asm.emit(word);
                self.commit(result, d);
            }
            Opcode::Bnot | Opcode::Ineg => {
                let result = result.unwrap();
                let width = width(result_ty.unwrap());
                let a = self.use_reg(args[0], T0);
                let d = self.def_reg(result, T0);
                let word = if opcode == Opcode::Bnot {
                    e::alu(Alu::Orn, width, d, e::ZR, a)
                } else {
                    e::alu(Alu::Sub, width, d, e::ZR, a)
                };
                self.asm.emit(word);
                self.commit(result, d);
            }
            Opcode::Ishl | Opcode::Ushr | Opcode::Sshr => {
                let result = result.unwrap();
                let ty = result_ty.unwrap();
                if !matches!(ty, types::I32 | types::I64) {
                    return Err(Unsupported(format!("{opcode}.{ty}")));
                }
                let op = match opcode {
                    Opcode::Ishl => Shift::Lsl,
                    Opcode::Ushr => Shift::Lsr,
                    _ => Shift::Asr,
                };
                let a = self.use_reg(args[0], T0);
                if let Some(amount) = self.constant(args[1]) {
                    let d = self.def_reg(result, T0);
                    self.asm
                        .emit(e::shift_imm(op, width(ty), d, a, amount as u32));
                    self.commit(result, d);
                    return Ok(());
                }
                let b = self.use_reg(args[1], T1);
                let d = self.def_reg(result, T0);
                self.asm.emit(e::shift(op, width(ty), d, a, b));
                self.commit(result, d);
            }
            Opcode::Icmp => {
                let result = result.unwrap();
                let cond = self.compare(
                    match op {
                        L::IntCompare { condition, .. } => *condition,
                        _ => unreachable!(),
                    },
                    args[0],
                    args[1],
                );
                let d = self.def_reg(result, T0);
                self.asm.emit(e::cset(d, cond));
                self.commit(result, d);
            }
            Opcode::Fcmp => {
                let result = result.unwrap();
                let (first, second) = float_conds(match op {
                    L::FloatCompare { condition, .. } => *condition,
                    _ => unreachable!(),
                });
                let ty = float(self.ty(args[0]));
                let a = self.use_reg(args[0], F0);
                let b = self.use_reg(args[1], F1);
                self.asm.emit(e::fcmp(ty, a, b));
                let d = self.def_reg(result, T0);
                self.asm.emit(e::cset(d, first));
                if let Some(second) = second {
                    self.asm.emit(e::cset(T1, second));
                    self.asm.emit(e::alu(Alu::Orr, Width::W, d, d, T1));
                }
                self.commit(result, d);
            }
            Opcode::Select => {
                let result = result.unwrap();
                let ty = result_ty.unwrap();
                let cond = match self.selection.select_compare(inst) {
                    Some(def) => self.fused_flags(def),
                    None => self.condition(args[0]),
                };
                if self.class(result) == RegClass::Float {
                    let a = self.use_reg(args[1], F0);
                    let b = self.use_reg(args[2], F1);
                    let d = self.def_reg(result, F0);
                    self.asm.emit(e::fcsel(float(ty), d, a, b, cond));
                    self.commit(result, d);
                } else {
                    let a = self.use_reg(args[1], T1);
                    let b = self.use_reg(args[2], T2);
                    let d = self.def_reg(result, T0);
                    self.asm.emit(e::csel(width(ty), d, a, b, cond));
                    self.commit(result, d);
                }
            }
            Opcode::Bitcast | Opcode::Ireduce => {
                let result = result.unwrap();
                let from = self.class(args[0]);
                let to = self.class(result);
                let width = if self.ty(result).bits() == 64 {
                    Width::X
                } else {
                    Width::W
                };
                match (from, to) {
                    (RegClass::Int, RegClass::Float) => {
                        let a = self.use_reg(args[0], T0);
                        let d = self.def_reg(result, F0);
                        self.asm.emit(e::move_to_float(width, d, a));
                        self.commit(result, d);
                    }
                    (RegClass::Float, RegClass::Int) => {
                        let a = self.use_reg(args[0], F0);
                        let d = self.def_reg(result, T0);
                        self.asm.emit(e::move_to_int(width, d, a));
                        self.commit(result, d);
                    }
                    _ => {
                        let m = Move {
                            class: to,
                            from: self.loc(args[0]),
                            to: self.loc(result),
                        };
                        self.parallel(vec![m]);
                    }
                }
            }
            Opcode::Uextend | Opcode::Sextend => {
                let result = result.unwrap();
                let from = self.ty(args[0]);
                let signed = opcode == Opcode::Sextend;
                let extend = match (from, signed) {
                    (types::I8, false) => Extend::Uxtb,
                    (types::I16, false) => Extend::Uxth,
                    (types::I32, false) => Extend::Uxtw,
                    (types::I8, true) => Extend::Sxtb,
                    (types::I16, true) => Extend::Sxth,
                    (types::I32, true) => Extend::Sxtw,
                    _ => return Err(Unsupported(format!("{opcode} from {from}"))),
                };
                let a = self.use_reg(args[0], T0);
                let d = self.def_reg(result, T0);
                self.asm
                    .emit(e::extend(extend, width(result_ty.unwrap()), d, a));
                self.commit(result, d);
            }
            Opcode::Load
            | Opcode::Uload8
            | Opcode::Sload8
            | Opcode::Uload16
            | Opcode::Sload16
            | Opcode::Uload32
            | Opcode::Sload32 => {
                if match op {
                    L::Load { flags, .. }
                    | L::AtomicLoad { flags, .. }
                    | L::Store { flags, .. } => flags.trap_code().is_some(),
                    _ => false,
                } {
                    return Err(Unsupported("trapping load".into()));
                }
                let result = result.unwrap();
                let ty = result_ty.unwrap();
                let op =
                    load_op(opcode, ty).ok_or_else(|| Unsupported(format!("{opcode}.{ty}")))?;
                let offset = match &data.op {
                    Op::LowLevel(L::Load { offset, .. } | L::Store { offset, .. }) => {
                        i64::from(*offset)
                    }
                    _ => 0,
                };
                let scratch = if matches!(op, Mem::LdrS | Mem::LdrD) {
                    F0
                } else {
                    T0
                };
                let d = self.def_reg(result, scratch);
                self.access(inst, op, d, args[0], offset);
                self.commit(result, d);
            }
            Opcode::AtomicLoad => {
                if match op {
                    L::Load { flags, .. }
                    | L::AtomicLoad { flags, .. }
                    | L::Store { flags, .. } => flags.trap_code().is_some(),
                    _ => false,
                } {
                    return Err(Unsupported("trapping atomic load".into()));
                }
                let result = result.unwrap();
                let ty = result_ty.unwrap();
                if RegClass::of(ty) != Some(RegClass::Int) {
                    return Err(Unsupported(format!("{opcode}.{ty}")));
                }
                let base = self.use_reg(args[0], T1);
                let d = self.def_reg(result, T0);
                self.asm.emit(e::load_acquire(ty.bytes(), d, base));
                self.commit(result, d);
            }
            Opcode::Store | Opcode::Istore8 | Opcode::Istore16 | Opcode::Istore32 => {
                if match op {
                    L::Load { flags, .. }
                    | L::AtomicLoad { flags, .. }
                    | L::Store { flags, .. } => flags.trap_code().is_some(),
                    _ => false,
                } {
                    return Err(Unsupported("trapping store".into()));
                }
                let ty = self.ty(args[0]);
                let op =
                    store_op(opcode, ty).ok_or_else(|| Unsupported(format!("{opcode}.{ty}")))?;
                let offset = match &data.op {
                    Op::LowLevel(L::Load { offset, .. } | L::Store { offset, .. }) => {
                        i64::from(*offset)
                    }
                    _ => 0,
                };
                let scratch = if matches!(op, Mem::StrS | Mem::StrD) {
                    F0
                } else {
                    T0
                };
                let value = if self.constant(args[0]) == Some(0) {
                    e::ZR
                } else {
                    self.use_reg(args[0], scratch)
                };
                self.access(inst, op, value, args[1], offset);
            }
            Opcode::StackAddr => {
                let result = result.unwrap();
                let L::StackAddress { slot, offset, .. } = op else {
                    unreachable!()
                };
                let offset = i64::from(self.frame.stack_slot(*slot)) + i64::from(*offset);
                let d = self.def_reg(result, T0);
                if (0..4096).contains(&offset) {
                    self.asm
                        .emit(e::add_imm(Width::X, d, e::SP, offset as u32, false));
                } else {
                    self.mov_imm(Width::X, T0, offset as u64);
                    self.asm.emit(e::mov_sp(T1, e::SP));
                    self.asm.emit(e::alu(Alu::Add, Width::X, d, T1, T0));
                }
                self.commit(result, d);
            }
            Opcode::GetStackPointer => {
                let result = result.unwrap();
                let d = self.def_reg(result, T0);
                self.asm.emit(e::mov_sp(d, e::SP));
                self.commit(result, d);
            }
            Opcode::Fadd
            | Opcode::Fsub
            | Opcode::Fmul
            | Opcode::Fdiv
            | Opcode::Fmin
            | Opcode::Fmax => {
                let result = result.unwrap();
                let op = match opcode {
                    Opcode::Fadd => FloatOp::Add,
                    Opcode::Fsub => FloatOp::Sub,
                    Opcode::Fmul => FloatOp::Mul,
                    Opcode::Fdiv => FloatOp::Div,
                    Opcode::Fmin => FloatOp::Min,
                    _ => FloatOp::Max,
                };
                let a = self.use_reg(args[0], F0);
                let b = self.use_reg(args[1], F1);
                let d = self.def_reg(result, F0);
                self.asm
                    .emit(e::float_binary(op, float(result_ty.unwrap()), d, a, b));
                self.commit(result, d);
            }
            Opcode::Fneg
            | Opcode::Fabs
            | Opcode::Sqrt
            | Opcode::Ceil
            | Opcode::Floor
            | Opcode::Trunc
            | Opcode::Nearest => {
                let result = result.unwrap();
                let op = match opcode {
                    Opcode::Fneg => FloatUnary::Neg,
                    Opcode::Fabs => FloatUnary::Abs,
                    Opcode::Sqrt => FloatUnary::Sqrt,
                    Opcode::Ceil => FloatUnary::Ceil,
                    Opcode::Floor => FloatUnary::Floor,
                    Opcode::Trunc => FloatUnary::Trunc,
                    _ => FloatUnary::Nearest,
                };
                let a = self.use_reg(args[0], F0);
                let d = self.def_reg(result, F0);
                self.asm
                    .emit(e::float_unary(op, float(result_ty.unwrap()), d, a));
                self.commit(result, d);
            }
            Opcode::FcvtFromSint | Opcode::FcvtFromUint => {
                let result = result.unwrap();
                let from = self.ty(args[0]);
                if !matches!(from, types::I32 | types::I64) {
                    return Err(Unsupported(format!("{opcode} from {from}")));
                }
                let a = self.use_reg(args[0], T0);
                let d = self.def_reg(result, F0);
                self.asm.emit(e::int_to_float(
                    opcode == Opcode::FcvtFromSint,
                    width(from),
                    float(result_ty.unwrap()),
                    d,
                    a,
                ));
                self.commit(result, d);
            }
            Opcode::FcvtToSintSat | Opcode::FcvtToUintSat => {
                let result = result.unwrap();
                let ty = result_ty.unwrap();
                if !matches!(ty, types::I32 | types::I64) {
                    return Err(Unsupported(format!("{opcode}.{ty}")));
                }
                let a = self.use_reg(args[0], F0);
                let d = self.def_reg(result, T0);
                self.asm.emit(e::float_to_int(
                    opcode == Opcode::FcvtToSintSat,
                    width(ty),
                    float(self.ty(args[0])),
                    d,
                    a,
                ));
                self.commit(result, d);
            }
            Opcode::Fdemote | Opcode::Fpromote => {
                let result = result.unwrap();
                let a = self.use_reg(args[0], F0);
                let d = self.def_reg(result, F0);
                let word = if opcode == Opcode::Fdemote {
                    e::demote(d, a)
                } else {
                    e::promote(d, a)
                };
                self.asm.emit(word);
                self.commit(result, d);
            }
            _ => return Err(Unsupported(format!("opcode {opcode}"))),
        }
        Ok(())
    }
}

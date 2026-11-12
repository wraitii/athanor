use cranelift_codegen::binemit::Reloc;
use cranelift_codegen::entity::SecondaryMap;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{
    ArgumentExtension, Block, BlockCall, ExternalName, FuncRef, Function, Inst, InstructionData,
    Opcode, SigRef, Type, Value, types,
};
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
    self, Address, Binary, CompareImmediate, Comparison, Immediate, Selection, remat_bits,
};
use crate::jit::alloc::moves::{self, Loc, Move};
use crate::jit::alloc::{self, Allocation, Frame, Hint, Place, RegClass};
use crate::jit::machine::{MachineCode, SourceRange};

const T0: u8 = 9;
const T1: u8 = 10;
const T2: u8 = 11;
const CALLEE: u8 = 16;
const ADDRESS: u8 = 17;
const F0: u8 = 16;
const F1: u8 = 17;

pub(super) fn lower(func: &Function) -> Result<MachineCode, Unsupported> {
    check_function(func)?;
    let order = block_order(func);
    let hints = hints(func)?;
    let selection = Selection::new(func);
    let operands = |inst, out: &mut Vec<Value>| selection.operands(func, inst, out);
    let mut remat = SecondaryMap::new();
    for value in func.dfg.values() {
        remat[value] = select::remat_cost(func, value);
    }
    let allocation = alloc::allocate(func, &order, &TARGET, &hints, &operands, &remat, &|_| true)
        .map_err(Unsupported)?;
    match lower_with(func, &order, &allocation, &selection, false)? {
        Some(machine) => Ok(machine),
        None => lower_with(func, &order, &allocation, &selection, true)?
            .ok_or_else(|| Unsupported("conditional branch out of range".into())),
    }
}

fn lower_with(
    func: &Function,
    order: &[Block],
    allocation: &Allocation,
    selection: &Selection,
    far: bool,
) -> Result<Option<MachineCode>, Unsupported> {
    let conv = func.signature.call_conv;
    let incoming = abi::assign(conv, &func.signature.params)?.stack_size;
    let (outgoing, tail_calls) = call_stack_sizes(func)?;
    if tail_calls > 0 && conv != CallConv::Tail {
        return Err(Unsupported(format!("tail call from {conv}")));
    }
    let tail_args = incoming.max(tail_calls);
    let frame = Frame::new(func, allocation, outgoing);
    if frame.size >= 1 << 24 || tail_args >= 1 << 24 {
        return Err(Unsupported("frame too large".into()));
    }
    let entry = func
        .layout
        .entry_block()
        .ok_or_else(|| Unsupported("no entry block".into()))?;
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
    for block in func.layout.blocks() {
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

fn check_function(func: &Function) -> Result<(), Unsupported> {
    if func.stack_limit.is_some() {
        return Err(Unsupported("stack limit".into()));
    }
    if !func.dynamic_stack_slots.is_empty() {
        return Err(Unsupported("dynamic stack slots".into()));
    }
    if let Some(param) = func
        .signature
        .returns
        .iter()
        .find(|param| param.extension != ArgumentExtension::None)
    {
        return Err(Unsupported(format!("extended return {param}")));
    }
    for value in func.dfg.values() {
        let ty = func.dfg.value_type(value);
        if RegClass::of(ty).is_none() {
            return Err(Unsupported(format!("value type {ty}")));
        }
    }
    Ok(())
}

fn call_stack_sizes(func: &Function) -> Result<(u32, u32), Unsupported> {
    let mut outgoing = 0;
    let mut tail = 0;
    for block in func.layout.blocks() {
        for inst in func.layout.block_insts(block) {
            let Some(sig_ref) = func.dfg.call_signature(inst) else {
                continue;
            };
            let signature = &func.dfg.signatures[sig_ref];
            let size = abi::assign(signature.call_conv, &signature.params)?.stack_size;
            let target = if func.dfg.insts[inst].opcode().is_return() {
                &mut tail
            } else {
                &mut outgoing
            };
            *target = size.max(*target);
        }
    }
    Ok((outgoing, tail))
}

fn hints(func: &Function) -> Result<Vec<Hint>, Unsupported> {
    let dfg = &func.dfg;
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
    let signature = &func.signature;
    if let Some(entry) = func.layout.entry_block() {
        let locations = abi::assign(signature.call_conv, &signature.params)?.locations;
        add(dfg.block_params(entry), &locations, entry);
    }
    for block in func.layout.blocks() {
        for inst in func.layout.block_insts(block) {
            let opcode = dfg.insts[inst].opcode();
            let args = dfg.inst_args(inst);
            if opcode == Opcode::Return {
                let locations = abi::assign_returns(signature.call_conv, &signature.returns)?;
                add(args, &locations, block);
                continue;
            }
            let Some(sig_ref) = dfg.call_signature(inst) else {
                continue;
            };
            let callee = &dfg.signatures[sig_ref];
            let args = match opcode {
                Opcode::CallIndirect | Opcode::ReturnCallIndirect => &args[1..],
                _ => args,
            };
            let locations = abi::assign(callee.call_conv, &callee.params)?.locations;
            add(args, &locations, block);
            let locations = abi::assign_returns(callee.call_conv, &callee.returns)?;
            add(dfg.inst_results(inst), &locations, block);
        }
    }
    Ok(hints)
}

fn block_order(func: &Function) -> Vec<Block> {
    let (cold, mut hot): (Vec<_>, Vec<_>) = func
        .layout
        .blocks()
        .partition(|&block| func.layout.is_cold(block) && Some(block) != func.layout.entry_block());
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
    func: &'a Function,
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
        self.func.dfg.value_type(value)
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
        let signature = &func.signature;
        let locations = abi::assign(signature.call_conv, &signature.params)?.locations;
        abi::assign_returns(signature.call_conv, &signature.returns)?;
        self.adjust_sp(self.tail_args - self.incoming, true);
        self.asm.emit(e::STP_FP_LR_PRE);
        self.asm.emit(e::mov_sp(e::FP, e::SP));
        self.adjust_sp(self.frame.size, true);
        self.callee_saves(false);
        let params = func.dfg.block_params(self.block);
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
        for inst in self.func.layout.block_insts(block) {
            let start = self.asm.offset();
            self.inst(inst, next)?;
            let end = self.asm.offset();
            let location = self.func.srcloc(inst);
            if end > start {
                self.locations.push(SourceRange {
                    start,
                    end,
                    location: (!location.is_default()).then(|| location.bits()),
                });
            }
        }
        Ok(())
    }

    fn edge_moves(&self, call: BlockCall) -> Result<Vec<Move>, Unsupported> {
        let dfg = &self.func.dfg;
        let target = call.block(&dfg.value_lists);
        let params = dfg.block_params(target);
        let mut pending = Vec::with_capacity(params.len());
        for (arg, &param) in call.args(&dfg.value_lists).zip(params) {
            let arg = arg
                .as_value()
                .ok_or_else(|| Unsupported("non-value block argument".into()))?;
            pending.push(Move {
                class: self.class(param),
                from: self.loc(arg),
                to: self.loc_in(param, target),
            });
        }
        if self.allocation.is_cold(self.block) != self.allocation.is_cold(target) {
            for &value in self.allocation.boundary(target) {
                let to = self.loc_in(value, target);
                if matches!(to, Loc::Imm(_)) {
                    continue;
                }
                pending.push(Move {
                    class: self.class(value),
                    from: self.loc(value),
                    to,
                });
            }
        }
        pending.retain(|m| m.from != m.to);
        Ok(pending)
    }

    fn edge(&mut self, call: BlockCall, next: Option<Block>) -> Result<(), Unsupported> {
        let moves = self.edge_moves(call)?;
        self.parallel(moves);
        let target = call.block(&self.func.dfg.value_lists);
        if next != Some(target) {
            let label = self.label(target);
            self.asm.jump(e::B, label);
        }
        Ok(())
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

    fn fused_flags(&mut self, def: Inst) -> Cond {
        let dfg = &self.func.dfg;
        let args = dfg.inst_args(def);
        match dfg.insts[def] {
            InstructionData::IntCompare { cond, .. } => self.compare(cond, args[0], args[1]),
            InstructionData::FloatCompare { cond, .. } => {
                let ty = float(self.ty(args[0]));
                let a = self.use_reg(args[0], F0);
                let b = self.use_reg(args[1], F1);
                self.asm.emit(e::fcmp(ty, a, b));
                float_conds(cond).0
            }
            _ => unreachable!("only compares are fused"),
        }
    }

    fn select_condition(&mut self, value: Value, user: Inst) -> Cond {
        match self.selection.compare(user) {
            Some(def) => self.fused_flags(def),
            None => self.condition(value),
        }
    }

    fn zero_test(&mut self, value: Value, nonzero: bool) -> Option<Test> {
        let ty = self.ty(value);
        if !matches!(ty, types::I32 | types::I64) {
            return None;
        }
        let reg = self.use_reg(value, T0);
        Some(Test::Zero {
            nonzero,
            width: width(ty),
            reg,
        })
    }

    fn branch_test(&mut self, value: Value, user: Inst) -> Test {
        let dfg = &self.func.dfg;
        let Some(def) = self.selection.compare(user) else {
            return self
                .zero_test(value, true)
                .unwrap_or_else(|| Test::Flags(self.condition(value)));
        };
        if let InstructionData::IntCompare {
            cond: cond @ (IntCC::Equal | IntCC::NotEqual),
            ..
        } = dfg.insts[def]
        {
            let args = dfg.inst_args(def);
            let operand = if self.constant(args[1]) == Some(0) {
                Some(args[0])
            } else if self.constant(args[0]) == Some(0) {
                Some(args[1])
            } else {
                None
            };
            if let Some(test) =
                operand.and_then(|operand| self.zero_test(operand, cond == IntCC::NotEqual))
            {
                return test;
            }
        }
        Test::Flags(self.fused_flags(def))
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
        inst: Inst,
        sig_ref: SigRef,
        args: &[Value],
        callee: Option<Value>,
        emit: impl FnOnce(&mut Self),
    ) -> Result<(), Unsupported> {
        let func = self.func;
        let signature = &func.dfg.signatures[sig_ref];
        let conv = signature.call_conv;
        let assignment = abi::assign(conv, &signature.params)?;
        let tail = func.dfg.insts[inst].opcode().is_return();
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
        let results = func.dfg.inst_results(inst);
        let mut pending = Vec::with_capacity(results.len());
        for (&result, location) in results.iter().zip(locations) {
            let Location::Reg { class, reg } = location else {
                continue;
            };
            pending.push(Move {
                class,
                from: Loc::Reg(class, reg),
                to: self.loc(result),
            });
        }
        self.parallel(pending);
        Ok(())
    }

    fn direct_target(&self, func_ref: FuncRef) -> Result<ModuleRelocTarget, Unsupported> {
        let external = &self.func.dfg.ext_funcs[func_ref];
        if !external.colocated {
            return Err(Unsupported("non-colocated call".into()));
        }
        let ExternalName::User(name) = &external.name else {
            return Err(Unsupported("non-user call target".into()));
        };
        let name = &self.func.params.user_named_funcs()[*name];
        Ok(ModuleRelocTarget::user(name.namespace, name.index))
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

    fn tail_jump(&mut self, sig_ref: SigRef) -> Result<(), Unsupported> {
        let signature = &self.func.dfg.signatures[sig_ref];
        let stack_size = abi::assign(signature.call_conv, &signature.params)?.stack_size;
        self.epilogue(self.tail_args - stack_size);
        Ok(())
    }

    fn edge_target(&mut self, call: BlockCall) -> Result<Label, Unsupported> {
        let moves = self.edge_moves(call)?;
        Ok(self.stub(call, moves))
    }

    fn stub(&mut self, call: BlockCall, moves: Vec<Move>) -> Label {
        let target = call.block(&self.func.dfg.value_lists);
        if moves.is_empty() {
            return self.label(target);
        }
        let label = self.asm.label();
        self.stubs.push((label, moves, target));
        label
    }

    fn inst(&mut self, inst: Inst, next: Option<Block>) -> Result<(), Unsupported> {
        if self.selection.skips(inst) {
            return Ok(());
        }
        let func = self.func;
        let dfg = &func.dfg;
        let data = &dfg.insts[inst];
        let opcode = data.opcode();
        let args = dfg.inst_args(inst);
        let result = dfg.inst_results(inst).first().copied();
        let result_ty = result.map(|result| self.ty(result));
        match (opcode, data) {
            (Opcode::Iconst | Opcode::F64const | Opcode::F32const, _) => {
                let result = result.unwrap();
                if let Loc::Reg(class, reg) = self.loc(result) {
                    let bits = remat_bits(func, result).expect("constants have bits");
                    self.materialize(class, reg, bits);
                }
            }
            (
                Opcode::Iadd
                | Opcode::Isub
                | Opcode::Imul
                | Opcode::Band
                | Opcode::Bor
                | Opcode::Bxor,
                _,
            ) => {
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
            (Opcode::Bnot | Opcode::Ineg, _) => {
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
            (Opcode::Ishl | Opcode::Ushr | Opcode::Sshr, _) => {
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
            (Opcode::Icmp, _) => {
                let result = result.unwrap();
                let cond = self.compare(data.cond_code().unwrap(), args[0], args[1]);
                let d = self.def_reg(result, T0);
                self.asm.emit(e::cset(d, cond));
                self.commit(result, d);
            }
            (Opcode::Fcmp, _) => {
                let result = result.unwrap();
                let (first, second) = float_conds(data.fp_cond_code().unwrap());
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
            (Opcode::Select, _) => {
                let result = result.unwrap();
                let ty = result_ty.unwrap();
                let cond = self.select_condition(args[0], inst);
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
            (Opcode::Bitcast | Opcode::Ireduce, _) => {
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
            (Opcode::Uextend | Opcode::Sextend, _) => {
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
            (
                Opcode::Load
                | Opcode::Uload8
                | Opcode::Sload8
                | Opcode::Uload16
                | Opcode::Sload16
                | Opcode::Uload32
                | Opcode::Sload32,
                _,
            ) => {
                if data.memflags_trap_code(dfg).is_some() {
                    return Err(Unsupported("trapping load".into()));
                }
                let result = result.unwrap();
                let ty = result_ty.unwrap();
                let op =
                    load_op(opcode, ty).ok_or_else(|| Unsupported(format!("{opcode}.{ty}")))?;
                let offset = i64::from(data.load_store_offset().unwrap());
                let scratch = if matches!(op, Mem::LdrS | Mem::LdrD) {
                    F0
                } else {
                    T0
                };
                let d = self.def_reg(result, scratch);
                self.access(inst, op, d, args[0], offset);
                self.commit(result, d);
            }
            (Opcode::AtomicLoad, _) => {
                if data.memflags_trap_code(dfg).is_some() {
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
            (Opcode::Store | Opcode::Istore8 | Opcode::Istore16 | Opcode::Istore32, _) => {
                if data.memflags_trap_code(dfg).is_some() {
                    return Err(Unsupported("trapping store".into()));
                }
                let ty = self.ty(args[0]);
                let op =
                    store_op(opcode, ty).ok_or_else(|| Unsupported(format!("{opcode}.{ty}")))?;
                let offset = i64::from(data.load_store_offset().unwrap());
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
            (
                Opcode::StackAddr,
                InstructionData::StackAddr {
                    stack_slot, offset, ..
                },
            ) => {
                let result = result.unwrap();
                let offset =
                    i64::from(self.frame.stack_slot(*stack_slot)) + i64::from(i32::from(*offset));
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
            (Opcode::GetStackPointer, _) => {
                let result = result.unwrap();
                let d = self.def_reg(result, T0);
                self.asm.emit(e::mov_sp(d, e::SP));
                self.commit(result, d);
            }
            (
                Opcode::Fadd
                | Opcode::Fsub
                | Opcode::Fmul
                | Opcode::Fdiv
                | Opcode::Fmin
                | Opcode::Fmax,
                _,
            ) => {
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
            (
                Opcode::Fneg
                | Opcode::Fabs
                | Opcode::Sqrt
                | Opcode::Ceil
                | Opcode::Floor
                | Opcode::Trunc
                | Opcode::Nearest,
                _,
            ) => {
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
            (Opcode::FcvtFromSint | Opcode::FcvtFromUint, _) => {
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
            (Opcode::FcvtToSintSat | Opcode::FcvtToUintSat, _) => {
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
            (Opcode::Fdemote | Opcode::Fpromote, _) => {
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
            (Opcode::CallIndirect, InstructionData::CallIndirect { sig_ref, .. }) => {
                self.call(inst, *sig_ref, &args[1..], Some(args[0]), |this| {
                    this.asm.emit(e::blr(CALLEE));
                })?;
            }
            (Opcode::ReturnCallIndirect, InstructionData::CallIndirect { sig_ref, .. }) => {
                self.call(inst, *sig_ref, &args[1..], Some(args[0]), |_| {})?;
                self.tail_jump(*sig_ref)?;
                self.asm.emit(e::br(CALLEE));
            }
            (Opcode::Call, InstructionData::Call { func_ref, .. }) => {
                let target = self.direct_target(*func_ref)?;
                let sig_ref = dfg.ext_funcs[*func_ref].signature;
                self.call(inst, sig_ref, args, None, |this| {
                    this.emit_call(e::BL, target)
                })?;
            }
            (Opcode::ReturnCall, InstructionData::Call { func_ref, .. }) => {
                let target = self.direct_target(*func_ref)?;
                let sig_ref = dfg.ext_funcs[*func_ref].signature;
                self.call(inst, sig_ref, args, None, |_| {})?;
                self.tail_jump(sig_ref)?;
                self.emit_call(e::B, target);
            }
            (Opcode::BrTable, InstructionData::BranchTable { arg, table, .. }) => {
                let table = &dfg.jump_tables[*table];
                let entries = table.as_slice();
                let reg = self.use_reg(*arg, T0);
                let index = match self.ty(*arg) {
                    types::I8 => {
                        self.asm.emit(e::extend(Extend::Uxtb, Width::W, T0, reg));
                        T0
                    }
                    types::I16 => {
                        self.asm.emit(e::extend(Extend::Uxth, Width::W, T0, reg));
                        T0
                    }
                    types::I32 => {
                        self.asm.emit(e::extend(Extend::Uxtw, Width::X, T0, reg));
                        T0
                    }
                    _ => reg,
                };
                self.mov_imm(Width::X, T1, entries.len() as u64);
                self.asm.emit(e::alu(Alu::Subs, Width::X, e::ZR, index, T1));
                let default = self.edge_target(table.default_block())?;
                self.asm.branch(Cond::Hs, default);
                let start = self.asm.label();
                self.asm.address(e::adr(T1), start);
                self.asm.emit(e::add_lsl(Width::X, T1, T1, index, 2));
                self.asm.emit(e::br(T1));
                self.asm.bind(start);
                for &entry in entries {
                    let target = self.edge_target(entry)?;
                    self.asm.jump(e::B, target);
                }
            }
            (Opcode::Jump, InstructionData::Jump { destination, .. }) => {
                self.edge(*destination, next)?;
            }
            (Opcode::Brif, InstructionData::Brif { arg, blocks, .. }) => {
                let [then_call, else_call] = *blocks;
                let test = self.branch_test(*arg, inst);
                let then_block = then_call.block(&dfg.value_lists);
                let then_moves = self.edge_moves(then_call)?;
                if next == Some(then_block) && then_moves.is_empty() {
                    let target = self.edge_target(else_call)?;
                    self.branch_if(test.invert(), target);
                } else {
                    let target = self.stub(then_call, then_moves);
                    self.branch_if(test, target);
                    self.edge(else_call, next)?;
                }
            }
            (Opcode::Return, _) => {
                let signature = &func.signature;
                let locations = abi::assign_returns(signature.call_conv, &signature.returns)?;
                let mut pending = Vec::with_capacity(args.len());
                for (&value, location) in args.iter().zip(locations) {
                    let Location::Reg { class, reg } = location else {
                        continue;
                    };
                    pending.push(Move {
                        class,
                        from: self.loc(value),
                        to: Loc::Reg(class, reg),
                    });
                }
                self.parallel(pending);
                self.epilogue(self.pop);
                self.asm.emit(e::RET);
            }
            _ => return Err(Unsupported(format!("opcode {opcode}"))),
        }
        Ok(())
    }
}

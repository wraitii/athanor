use cranelift_codegen::binemit::Reloc;
use cranelift_codegen::entity::SecondaryMap;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{
    AbiParam, ArgumentExtension, Block, BlockCall, ExternalName, FuncRef, Function, Inst,
    InstructionData, Opcode, SigRef, Type, Value, types,
};
use cranelift_codegen::isa::CallConv;
use cranelift_module::{ModuleReloc, ModuleRelocTarget};

use super::Unsupported;
use super::abi::{self, Location};
use super::asm::{Assembler, Label};
use super::encode::{self as e, Alu, Cond, Extend, Float, FloatOp, FloatUnary, Mem, Shift, Width};
use crate::jit::alloc::{Frame, RegClass};
use crate::jit::machine::{MachineCode, SourceRange};

const T0: u8 = 9;
const T1: u8 = 10;
const T2: u8 = 11;
const CALLEE: u8 = 16;
const ADDRESS: u8 = 17;
const F0: u8 = 16;
const F1: u8 = 17;

pub(super) fn lower(func: &Function) -> Result<MachineCode, Unsupported> {
    check_types(func)?;
    match lower_with(func, false)? {
        Some(machine) => Ok(machine),
        None => lower_with(func, true)?
            .ok_or_else(|| Unsupported("conditional branch out of range".into())),
    }
}

fn lower_with(func: &Function, far: bool) -> Result<Option<MachineCode>, Unsupported> {
    let conv = func.signature.call_conv;
    let incoming = abi::assign(conv, &func.signature.params)?.stack_size;
    let (outgoing, tail_calls) = call_stack_sizes(func)?;
    if tail_calls > 0 && conv != CallConv::Tail {
        return Err(Unsupported(format!("tail call from {conv}")));
    }
    let tail_args = incoming.max(tail_calls);
    let frame = Frame::spill_all(func, max_edge_args(func), outgoing);
    if frame.size >= 1 << 24 || tail_args >= 1 << 24 {
        return Err(Unsupported("frame too large".into()));
    }
    let mut lower = Lower {
        func,
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
    let order = block_order(func);
    for (index, &block) in order.iter().enumerate() {
        let next = order.get(index + 1).copied();
        lower.block(block, next)?;
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

fn check_types(func: &Function) -> Result<(), Unsupported> {
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

fn max_edge_args(func: &Function) -> u32 {
    func.layout
        .blocks()
        .map(|block| func.dfg.block_params(block).len() as u32)
        .max()
        .unwrap_or(0)
}

fn block_order(func: &Function) -> Vec<Block> {
    let (cold, mut hot): (Vec<_>, Vec<_>) = func
        .layout
        .blocks()
        .partition(|&block| func.layout.is_cold(block) && Some(block) != func.layout.entry_block());
    hot.extend(cold);
    hot
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

fn load_op(opcode: Opcode, ty: Type) -> Option<Mem> {
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

fn store_op(opcode: Opcode, ty: Type) -> Option<Mem> {
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

struct Lower<'a> {
    func: &'a Function,
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

    fn mov_imm(&mut self, width: Width, rd: u8, value: u64) {
        let count = if width == Width::X { 4 } else { 2 };
        let chunks: Vec<u16> = (0..count).map(|i| (value >> (16 * i)) as u16).collect();
        let ones = chunks.iter().filter(|&&chunk| chunk == 0xFFFF).count();
        let zeros = chunks.iter().filter(|&&chunk| chunk == 0).count();
        let (skip, first) = if ones > zeros {
            (0xFFFF, e::Wide::Movn)
        } else {
            (0, e::Wide::Movz)
        };
        let mut started = false;
        for (hw, &chunk) in chunks.iter().enumerate() {
            if chunk == skip {
                continue;
            }
            let word = if started {
                e::mov_wide(e::Wide::Movk, width, rd, chunk, hw as u8)
            } else if first == e::Wide::Movn {
                e::mov_wide(first, width, rd, !chunk, hw as u8)
            } else {
                e::mov_wide(first, width, rd, chunk, hw as u8)
            };
            self.asm.emit(word);
            started = true;
        }
        if !started {
            self.asm.emit(e::mov_wide(first, width, rd, 0, 0));
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

    fn slot(&self, value: Value) -> i64 {
        i64::from(self.frame.value(value))
    }

    fn load_int(&mut self, value: Value, reg: u8) {
        self.memory(Mem::LdrX, reg, e::SP, self.slot(value));
    }

    fn store_int(&mut self, value: Value, reg: u8) {
        self.memory(Mem::StrX, reg, e::SP, self.slot(value));
    }

    fn load_float(&mut self, value: Value, reg: u8) {
        let op = if self.ty(value) == types::F32 {
            Mem::LdrS
        } else {
            Mem::LdrD
        };
        self.memory(op, reg, e::SP, self.slot(value));
    }

    fn store_float(&mut self, value: Value, reg: u8) {
        let op = if self.ty(value) == types::F32 {
            Mem::StrS
        } else {
            Mem::StrD
        };
        self.memory(op, reg, e::SP, self.slot(value));
    }

    fn load_location(&mut self, value: Value, location: Location, base: u8, area: i64) {
        match location {
            Location::Reg {
                class: RegClass::Int,
                reg,
            } => self.load_int(value, reg),
            Location::Reg {
                class: RegClass::Float,
                reg,
            } => self.load_float(value, reg),
            Location::Stack { offset, ty } => {
                let op = store_op(Opcode::Store, ty).expect("argument types are checked");
                let reg = self.load_scratch(value);
                self.memory(op, reg, base, area + i64::from(offset));
            }
        }
    }

    fn store_location(&mut self, value: Value, location: Location, base: u8, area: i64) {
        match location {
            Location::Reg {
                class: RegClass::Int,
                reg,
            } => self.store_int(value, reg),
            Location::Reg {
                class: RegClass::Float,
                reg,
            } => self.store_float(value, reg),
            Location::Stack { offset, ty } => {
                let op = load_op(Opcode::Load, ty).expect("argument types are checked");
                let reg = if matches!(op, Mem::LdrS | Mem::LdrD) {
                    F0
                } else {
                    T0
                };
                self.memory(op, reg, base, area + i64::from(offset));
                self.store_scratch(value, reg);
            }
        }
    }

    fn load_scratch(&mut self, value: Value) -> u8 {
        if RegClass::of(self.ty(value)) == Some(RegClass::Float) {
            self.load_float(value, F0);
            F0
        } else {
            self.load_int(value, T0);
            T0
        }
    }

    fn store_scratch(&mut self, value: Value, reg: u8) {
        if RegClass::of(self.ty(value)) == Some(RegClass::Float) {
            self.store_float(value, reg);
        } else {
            self.store_int(value, reg);
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

    fn prologue(&mut self) -> Result<(), Unsupported> {
        let func = self.func;
        let signature = &func.signature;
        let locations = abi::assign(signature.call_conv, &signature.params)?.locations;
        abi::assign_returns(signature.call_conv, &signature.returns)?;
        self.adjust_sp(self.tail_args - self.incoming, true);
        self.asm.emit(e::STP_FP_LR_PRE);
        self.asm.emit(e::mov_sp(e::FP, e::SP));
        self.adjust_sp(self.frame.size, true);
        let entry = func
            .layout
            .entry_block()
            .ok_or_else(|| Unsupported("no entry block".into()))?;
        let area = self.incoming_area();
        for (&param, location) in func.dfg.block_params(entry).iter().zip(locations) {
            self.store_location(param, location, e::FP, area);
        }
        Ok(())
    }

    fn epilogue(&mut self, pop: u32) {
        self.asm.emit(e::mov_sp(e::SP, e::FP));
        self.asm.emit(e::LDP_FP_LR_POST);
        self.adjust_sp(pop, false);
    }

    fn block(&mut self, block: Block, next: Option<Block>) -> Result<(), Unsupported> {
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

    fn edge(&mut self, call: BlockCall, next: Option<Block>) -> Result<(), Unsupported> {
        let dfg = &self.func.dfg;
        let target = call.block(&dfg.value_lists);
        let params = dfg.block_params(target);
        let args = call
            .args(&dfg.value_lists)
            .map(|arg| {
                arg.as_value()
                    .ok_or_else(|| Unsupported("non-value block argument".into()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let moves: Vec<(Value, Value)> = args
            .iter()
            .zip(params)
            .filter(|(arg, param)| arg != param)
            .map(|(&arg, &param)| (arg, param))
            .collect();
        let overlapping = moves.iter().any(|(arg, _)| params.contains(arg));
        if overlapping {
            let scratch = i64::from(self.frame.scratch);
            for (index, &(arg, _)) in moves.iter().enumerate() {
                self.load_int(arg, T0);
                self.memory(Mem::StrX, T0, e::SP, scratch + 8 * index as i64);
            }
            for (index, &(_, param)) in moves.iter().enumerate() {
                self.memory(Mem::LdrX, T0, e::SP, scratch + 8 * index as i64);
                self.store_int(param, T0);
            }
        } else {
            for &(arg, param) in &moves {
                self.load_int(arg, T0);
                self.store_int(param, T0);
            }
        }
        if next != Some(target) {
            let label = self.label(target);
            self.asm.jump(e::B, label);
        }
        Ok(())
    }

    fn has_args(&self, call: BlockCall) -> bool {
        call.len(&self.func.dfg.value_lists) > 0
    }

    fn condition(&mut self, value: Value) -> Cond {
        let ty = self.ty(value);
        self.load_int(value, T0);
        match ty {
            types::I8 => self.asm.emit(e::tst_low_byte(T0)),
            types::I16 => {
                self.asm.emit(e::extend(Extend::Uxth, Width::W, T0, T0));
                self.asm.emit(e::alu(Alu::Subs, Width::W, e::ZR, T0, e::ZR));
            }
            _ => self
                .asm
                .emit(e::alu(Alu::Subs, width(ty), e::ZR, T0, e::ZR)),
        }
        Cond::Ne
    }

    fn call_args(
        &mut self,
        conv: CallConv,
        params: &[AbiParam],
        args: &[Value],
        base: u8,
        area: i64,
    ) -> Result<(), Unsupported> {
        let locations = abi::assign(conv, params)?.locations;
        for ((&arg, location), param) in args.iter().zip(locations).zip(params) {
            self.load_location(arg, location, base, area);
            let Location::Reg { reg, .. } = location else {
                continue;
            };
            if conv != CallConv::AppleAarch64 {
                continue;
            }
            let extend = match (param.extension, param.value_type) {
                (ArgumentExtension::Uext, types::I8) => Extend::Uxtb,
                (ArgumentExtension::Uext, types::I16) => Extend::Uxth,
                (ArgumentExtension::Sext, types::I8) => Extend::Sxtb,
                (ArgumentExtension::Sext, types::I16) => Extend::Sxth,
                _ => continue,
            };
            self.asm.emit(e::extend(extend, Width::W, reg, reg));
        }
        Ok(())
    }

    fn call(
        &mut self,
        inst: Inst,
        sig_ref: SigRef,
        args: &[Value],
        emit: impl FnOnce(&mut Self),
    ) -> Result<(), Unsupported> {
        let func = self.func;
        let signature = &func.dfg.signatures[sig_ref];
        let conv = signature.call_conv;
        let stack_size = abi::assign(conv, &signature.params)?.stack_size;
        if func.dfg.insts[inst].opcode().is_return() {
            let area = i64::from(16 + self.tail_args - stack_size);
            self.call_args(conv, &signature.params, args, e::FP, area)?;
            emit(self);
            return Ok(());
        }
        self.call_args(conv, &signature.params, args, e::SP, 0)?;
        emit(self);
        if conv == CallConv::Tail {
            self.adjust_sp(stack_size, true);
        }
        let locations = abi::assign_returns(conv, &signature.returns)?;
        let results = func.dfg.inst_results(inst);
        for (&result, location) in results.iter().zip(locations) {
            self.store_location(result, location, e::SP, 0);
        }
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

    fn table_target(&mut self, call: BlockCall, stubs: &mut Vec<(Label, BlockCall)>) -> Label {
        if !self.has_args(call) {
            return self.label(call.block(&self.func.dfg.value_lists));
        }
        let label = self.asm.label();
        stubs.push((label, call));
        label
    }

    fn inst(&mut self, inst: Inst, next: Option<Block>) -> Result<(), Unsupported> {
        let func = self.func;
        let dfg = &func.dfg;
        let data = &dfg.insts[inst];
        let opcode = data.opcode();
        let args = dfg.inst_args(inst);
        let result = dfg.inst_results(inst).first().copied();
        let result_ty = result.map(|result| self.ty(result));
        match (opcode, data) {
            (Opcode::Iconst, InstructionData::UnaryImm { imm, .. }) => {
                let ty = result_ty.unwrap();
                self.mov_imm(width(ty), T0, imm.bits() as u64);
                self.store_int(result.unwrap(), T0);
            }
            (Opcode::F64const, InstructionData::UnaryIeee64 { imm, .. }) => {
                self.mov_imm(Width::X, T0, imm.bits());
                self.store_int(result.unwrap(), T0);
            }
            (Opcode::F32const, InstructionData::UnaryIeee32 { imm, .. }) => {
                self.mov_imm(Width::W, T0, u64::from(imm.bits()));
                self.store_int(result.unwrap(), T0);
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
                let width = width(result_ty.unwrap());
                self.load_int(args[0], T0);
                self.load_int(args[1], T1);
                let word = match opcode {
                    Opcode::Iadd => e::alu(Alu::Add, width, T0, T0, T1),
                    Opcode::Isub => e::alu(Alu::Sub, width, T0, T0, T1),
                    Opcode::Imul => e::mul(width, T0, T0, T1),
                    Opcode::Band => e::alu(Alu::And, width, T0, T0, T1),
                    Opcode::Bor => e::alu(Alu::Orr, width, T0, T0, T1),
                    _ => e::alu(Alu::Eor, width, T0, T0, T1),
                };
                self.asm.emit(word);
                self.store_int(result.unwrap(), T0);
            }
            (Opcode::Bnot | Opcode::Ineg, _) => {
                let width = width(result_ty.unwrap());
                self.load_int(args[0], T0);
                let word = if opcode == Opcode::Bnot {
                    e::alu(Alu::Orn, width, T0, e::ZR, T0)
                } else {
                    e::alu(Alu::Sub, width, T0, e::ZR, T0)
                };
                self.asm.emit(word);
                self.store_int(result.unwrap(), T0);
            }
            (Opcode::Ishl | Opcode::Ushr | Opcode::Sshr, _) => {
                let ty = result_ty.unwrap();
                if !matches!(ty, types::I32 | types::I64) {
                    return Err(Unsupported(format!("{opcode}.{ty}")));
                }
                let op = match opcode {
                    Opcode::Ishl => Shift::Lsl,
                    Opcode::Ushr => Shift::Lsr,
                    _ => Shift::Asr,
                };
                self.load_int(args[0], T0);
                self.load_int(args[1], T1);
                self.asm.emit(e::shift(op, width(ty), T0, T0, T1));
                self.store_int(result.unwrap(), T0);
            }
            (Opcode::Icmp, _) => {
                let cond = data.cond_code().unwrap();
                let ty = self.ty(args[0]);
                self.load_int(args[0], T0);
                self.load_int(args[1], T1);
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
                if let Some(extend) = extend {
                    self.asm.emit(e::extend(extend, Width::W, T0, T0));
                    self.asm.emit(e::extend(extend, Width::W, T1, T1));
                }
                self.asm.emit(e::alu(Alu::Subs, width(ty), e::ZR, T0, T1));
                self.asm.emit(e::cset(T0, int_cond(cond)));
                self.store_int(result.unwrap(), T0);
            }
            (Opcode::Fcmp, _) => {
                let (first, second) = float_conds(data.fp_cond_code().unwrap());
                let ty = float(self.ty(args[0]));
                self.load_float(args[0], F0);
                self.load_float(args[1], F1);
                self.asm.emit(e::fcmp(ty, F0, F1));
                self.asm.emit(e::cset(T0, first));
                if let Some(second) = second {
                    self.asm.emit(e::cset(T1, second));
                    self.asm.emit(e::alu(Alu::Orr, Width::W, T0, T0, T1));
                }
                self.store_int(result.unwrap(), T0);
            }
            (Opcode::Select, _) => {
                let ty = result_ty.unwrap();
                let cond = self.condition(args[0]);
                self.load_int(args[1], T1);
                self.load_int(args[2], T2);
                let width = if matches!(ty, types::I64 | types::F64) {
                    Width::X
                } else {
                    Width::W
                };
                self.asm.emit(e::csel(width, T0, T1, T2, cond));
                self.store_int(result.unwrap(), T0);
            }
            (Opcode::Bitcast | Opcode::Ireduce, _) => {
                self.load_int(args[0], T0);
                self.store_int(result.unwrap(), T0);
            }
            (Opcode::Uextend | Opcode::Sextend, _) => {
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
                self.load_int(args[0], T0);
                self.asm
                    .emit(e::extend(extend, width(result_ty.unwrap()), T0, T0));
                self.store_int(result.unwrap(), T0);
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
                let ty = result_ty.unwrap();
                let op =
                    load_op(opcode, ty).ok_or_else(|| Unsupported(format!("{opcode}.{ty}")))?;
                let offset = i64::from(data.load_store_offset().unwrap());
                self.load_int(args[0], T1);
                if matches!(op, Mem::LdrS | Mem::LdrD) {
                    self.memory(op, F0, T1, offset);
                    self.store_float(result.unwrap(), F0);
                } else {
                    self.memory(op, T0, T1, offset);
                    self.store_int(result.unwrap(), T0);
                }
            }
            (Opcode::AtomicLoad, _) => {
                let ty = result_ty.unwrap();
                if RegClass::of(ty) != Some(RegClass::Int) {
                    return Err(Unsupported(format!("{opcode}.{ty}")));
                }
                self.load_int(args[0], T1);
                self.asm.emit(e::load_acquire(ty.bytes(), T0, T1));
                self.store_int(result.unwrap(), T0);
            }
            (Opcode::Store | Opcode::Istore8 | Opcode::Istore16 | Opcode::Istore32, _) => {
                if data.memflags_trap_code(dfg).is_some() {
                    return Err(Unsupported("trapping store".into()));
                }
                let ty = self.ty(args[0]);
                let op =
                    store_op(opcode, ty).ok_or_else(|| Unsupported(format!("{opcode}.{ty}")))?;
                let offset = i64::from(data.load_store_offset().unwrap());
                self.load_int(args[1], T1);
                if matches!(op, Mem::StrS | Mem::StrD) {
                    self.load_float(args[0], F0);
                    self.memory(op, F0, T1, offset);
                } else {
                    self.load_int(args[0], T0);
                    self.memory(op, T0, T1, offset);
                }
            }
            (
                Opcode::StackAddr,
                InstructionData::StackAddr {
                    stack_slot, offset, ..
                },
            ) => {
                let offset =
                    i64::from(self.frame.stack_slot(*stack_slot)) + i64::from(i32::from(*offset));
                if offset < 4096 {
                    self.asm
                        .emit(e::add_imm(Width::X, T0, e::SP, offset as u32, false));
                } else {
                    self.mov_imm(Width::X, T0, offset as u64);
                    self.asm.emit(e::mov_sp(T1, e::SP));
                    self.asm.emit(e::alu(Alu::Add, Width::X, T0, T1, T0));
                }
                self.store_int(result.unwrap(), T0);
            }
            (Opcode::GetStackPointer, _) => {
                self.asm.emit(e::mov_sp(T0, e::SP));
                self.store_int(result.unwrap(), T0);
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
                let op = match opcode {
                    Opcode::Fadd => FloatOp::Add,
                    Opcode::Fsub => FloatOp::Sub,
                    Opcode::Fmul => FloatOp::Mul,
                    Opcode::Fdiv => FloatOp::Div,
                    Opcode::Fmin => FloatOp::Min,
                    _ => FloatOp::Max,
                };
                self.load_float(args[0], F0);
                self.load_float(args[1], F1);
                self.asm
                    .emit(e::float_binary(op, float(result_ty.unwrap()), F0, F0, F1));
                self.store_float(result.unwrap(), F0);
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
                let op = match opcode {
                    Opcode::Fneg => FloatUnary::Neg,
                    Opcode::Fabs => FloatUnary::Abs,
                    Opcode::Sqrt => FloatUnary::Sqrt,
                    Opcode::Ceil => FloatUnary::Ceil,
                    Opcode::Floor => FloatUnary::Floor,
                    Opcode::Trunc => FloatUnary::Trunc,
                    _ => FloatUnary::Nearest,
                };
                self.load_float(args[0], F0);
                self.asm
                    .emit(e::float_unary(op, float(result_ty.unwrap()), F0, F0));
                self.store_float(result.unwrap(), F0);
            }
            (Opcode::FcvtFromSint | Opcode::FcvtFromUint, _) => {
                let from = self.ty(args[0]);
                if !matches!(from, types::I32 | types::I64) {
                    return Err(Unsupported(format!("{opcode} from {from}")));
                }
                self.load_int(args[0], T0);
                self.asm.emit(e::int_to_float(
                    opcode == Opcode::FcvtFromSint,
                    width(from),
                    float(result_ty.unwrap()),
                    F0,
                    T0,
                ));
                self.store_float(result.unwrap(), F0);
            }
            (Opcode::FcvtToSintSat | Opcode::FcvtToUintSat, _) => {
                let ty = result_ty.unwrap();
                if !matches!(ty, types::I32 | types::I64) {
                    return Err(Unsupported(format!("{opcode}.{ty}")));
                }
                self.load_float(args[0], F0);
                self.asm.emit(e::float_to_int(
                    opcode == Opcode::FcvtToSintSat,
                    width(ty),
                    float(self.ty(args[0])),
                    T0,
                    F0,
                ));
                self.store_int(result.unwrap(), T0);
            }
            (Opcode::Fdemote | Opcode::Fpromote, _) => {
                self.load_float(args[0], F0);
                let word = if opcode == Opcode::Fdemote {
                    e::demote(F0, F0)
                } else {
                    e::promote(F0, F0)
                };
                self.asm.emit(word);
                self.store_float(result.unwrap(), F0);
            }
            (Opcode::CallIndirect, InstructionData::CallIndirect { sig_ref, .. }) => {
                let callee = args[0];
                self.call(inst, *sig_ref, &args[1..], |this| {
                    this.load_int(callee, CALLEE);
                    this.asm.emit(e::blr(CALLEE));
                })?;
            }
            (Opcode::ReturnCallIndirect, InstructionData::CallIndirect { sig_ref, .. }) => {
                let callee = args[0];
                let sig_ref = *sig_ref;
                self.call(inst, sig_ref, &args[1..], |this| {
                    this.load_int(callee, CALLEE)
                })?;
                self.tail_jump(sig_ref)?;
                self.asm.emit(e::br(CALLEE));
            }
            (Opcode::Call, InstructionData::Call { func_ref, .. }) => {
                let target = self.direct_target(*func_ref)?;
                let sig_ref = dfg.ext_funcs[*func_ref].signature;
                self.call(inst, sig_ref, args, |this| this.emit_call(e::BL, target))?;
            }
            (Opcode::ReturnCall, InstructionData::Call { func_ref, .. }) => {
                let target = self.direct_target(*func_ref)?;
                let sig_ref = dfg.ext_funcs[*func_ref].signature;
                self.call(inst, sig_ref, args, |_| {})?;
                self.tail_jump(sig_ref)?;
                self.emit_call(e::B, target);
            }
            (Opcode::BrTable, InstructionData::BranchTable { arg, table, .. }) => {
                let table = &dfg.jump_tables[*table];
                let entries = table.as_slice();
                match self.ty(*arg) {
                    types::I8 => {
                        self.load_int(*arg, T0);
                        self.asm.emit(e::extend(Extend::Uxtb, Width::W, T0, T0));
                    }
                    types::I16 => {
                        self.load_int(*arg, T0);
                        self.asm.emit(e::extend(Extend::Uxth, Width::W, T0, T0));
                    }
                    types::I32 => {
                        self.load_int(*arg, T0);
                        self.asm.emit(e::extend(Extend::Uxtw, Width::X, T0, T0));
                    }
                    _ => self.load_int(*arg, T0),
                }
                let mut stubs = Vec::new();
                self.mov_imm(Width::X, T1, entries.len() as u64);
                self.asm.emit(e::alu(Alu::Subs, Width::X, e::ZR, T0, T1));
                let default = self.table_target(table.default_block(), &mut stubs);
                self.asm.branch(Cond::Hs, default);
                let start = self.asm.label();
                self.asm.address(e::adr(T1), start);
                self.asm.emit(e::add_lsl(Width::X, T1, T1, T0, 2));
                self.asm.emit(e::br(T1));
                self.asm.bind(start);
                for &entry in entries {
                    let target = self.table_target(entry, &mut stubs);
                    self.asm.jump(e::B, target);
                }
                for (label, call) in stubs {
                    self.asm.bind(label);
                    self.edge(call, None)?;
                }
            }
            (Opcode::Jump, InstructionData::Jump { destination, .. }) => {
                self.edge(*destination, next)?;
            }
            (Opcode::Brif, InstructionData::Brif { arg, blocks, .. }) => {
                let [then_call, else_call] = *blocks;
                let cond = self.condition(*arg);
                let then_label = if self.has_args(then_call) {
                    self.asm.label()
                } else {
                    self.label(then_call.block(&dfg.value_lists))
                };
                self.asm.branch(cond, then_label);
                let then_stub = self.has_args(then_call);
                self.edge(else_call, if then_stub { None } else { next })?;
                if then_stub {
                    self.asm.bind(then_label);
                    self.edge(then_call, next)?;
                }
            }
            (Opcode::Return, _) => {
                let signature = &func.signature;
                let locations = abi::assign_returns(signature.call_conv, &signature.returns)?;
                for (&value, location) in args.iter().zip(locations) {
                    self.load_location(value, location, e::SP, 0);
                }
                self.epilogue(self.pop);
                self.asm.emit(e::RET);
            }
            _ => return Err(Unsupported(format!("opcode {opcode}"))),
        }
        Ok(())
    }
}

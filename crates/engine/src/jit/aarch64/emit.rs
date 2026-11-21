use cranelift_codegen::binemit::Reloc;
use cranelift_codegen::entity::SecondaryMap;
use cranelift_module::{ModuleReloc, ModuleRelocTarget};

use super::Unsupported;
use super::abi::{CALLEE, F0, F1, Location, T0, T1, T2};
use super::asm::{Assembler, Label};
use super::encode::{self as e, Alu, Cond, Index, Mem, Width};
use super::inst::{self as a, Address, CompareOperand, Condition, Mask};
use super::prepare::{Call, Prepared};
use crate::jit::alloc::moves::{Loc, Move};
use crate::jit::alloc::{
    RegClass,
    graph::{Allocation, Frame, Place},
};
use crate::jit::machine::{MachineCode, SourceRange};
use crate::jit::semantic::{Block, Graph, Inst, Op, Target, TerminatorKind, Value, low_level};

pub(super) fn emit(
    func: &Graph,
    prepared: &Prepared,
    allocation: &Allocation,
) -> Result<MachineCode, Unsupported> {
    match emit_with(func, prepared, allocation, false)? {
        Some(machine) => Ok(machine),
        None => emit_with(func, prepared, allocation, true)?
            .ok_or_else(|| Unsupported("conditional branch out of range".into())),
    }
}

fn emit_with(
    func: &Graph,
    prepared: &Prepared,
    allocation: &Allocation,
    far: bool,
) -> Result<Option<MachineCode>, Unsupported> {
    let order = &prepared.order;
    let frame = Frame::new(func, allocation, prepared.outgoing);
    if frame.size >= 1 << 24 || prepared.tail_args >= 1 << 24 {
        return Err(Unsupported("frame too large".into()));
    }
    let entry = func.entry;
    let mut emitter = Emitter {
        func,
        allocation,
        prepared,
        stubs: Vec::new(),
        block: entry,
        frame,
        incoming: prepared.incoming,
        tail_args: prepared.tail_args,
        pop: prepared.pop,
        asm: Assembler::new(far),
        labels: SecondaryMap::new(),
        relocs: Vec::new(),
        locations: Vec::new(),
    };
    for block in func.blocks.keys() {
        emitter.labels[block] = Some(emitter.asm.label());
    }
    emitter.prologue();
    for (index, &block) in order.iter().enumerate() {
        let next = order.get(index + 1).copied();
        emitter.block(block, next)?;
    }
    for (label, moves, target) in std::mem::take(&mut emitter.stubs) {
        emitter.asm.bind(label);
        emitter.asm.parallel(moves);
        let target = emitter.label(target);
        emitter.asm.jump(e::B, target);
    }
    let Emitter {
        asm,
        relocs,
        locations,
        ..
    } = emitter;
    if asm.conditional_overflow() {
        return Ok(None);
    }
    Ok(Some(MachineCode {
        bytes: asm.finish()?,
        alignment: 4,
        relocs,
        locations,
    }))
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

struct Emitter<'a> {
    func: &'a Graph,
    allocation: &'a Allocation,
    prepared: &'a Prepared,
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
}

impl Emitter<'_> {
    fn label(&self, block: Block) -> Label {
        self.labels[block].expect("every block has a label")
    }

    fn class(&self, value: Value) -> RegClass {
        self.prepared.input.classes[value].expect("prepared register class")
    }

    fn loc_in(&self, value: Value, block: Block) -> Loc {
        match self.allocation.place(value, block) {
            Place::Reg(reg) => Loc::Reg(self.class(value), reg),
            Place::Slot(index) => Loc::Slot(self.frame.slot(index)),
            Place::Remat => {
                Loc::Imm(self.prepared.constants[value].expect("rematerialized constant"))
            }
        }
    }

    fn loc(&self, value: Value) -> Loc {
        self.loc_in(value, self.block)
    }

    fn access(&mut self, op: Mem, address: Address, rt: u8) {
        match address {
            Address::Scaled { base, offset } => {
                let base = self.use_reg(base, T1);
                self.asm
                    .emit(e::mem_scaled(op, rt, base, offset).expect("selected scaled address"));
            }
            Address::Unscaled { base, offset } => {
                let base = self.use_reg(base, T1);
                self.asm.emit(
                    e::mem_unscaled(op, rt, base, offset).expect("selected unscaled address"),
                );
            }
            Address::Indexed {
                base,
                index,
                kind,
                scaled,
            } => {
                let base = self.use_reg(base, T1);
                let index = self.use_reg(index, T2);
                self.asm
                    .emit(e::mem_indexed(op, rt, base, index, kind, scaled));
            }
        }
    }

    fn use_reg(&mut self, value: Value, scratch: u8) -> u8 {
        match self.loc(value) {
            Loc::Reg(_, reg) => reg,
            Loc::Slot(offset) => {
                let op = Assembler::spill_op(self.class(value), true);
                self.asm.memory(op, scratch, e::SP, i64::from(offset));
                scratch
            }
            Loc::Imm(bits) => {
                self.asm.materialize(self.class(value), scratch, bits);
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
            let op = Assembler::spill_op(self.class(value), false);
            self.asm.memory(op, reg, e::SP, i64::from(offset));
        }
    }

    fn incoming_area(&self) -> i64 {
        i64::from(16 + self.tail_args - self.incoming)
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
            self.asm.memory(
                Assembler::spill_op(class, load),
                reg,
                e::SP,
                i64::from(offset),
            );
            index += 1;
        }
    }

    fn prologue(&mut self) {
        let func = self.func;
        let locations = &self.prepared.entry;
        self.asm.adjust_sp(self.tail_args - self.incoming, true);
        self.asm.emit(e::STP_FP_LR_PRE);
        self.asm.emit(e::mov_sp(e::FP, e::SP));
        self.asm.adjust_sp(self.frame.size, true);
        self.callee_saves(false);
        let params = &func.blocks[self.block].params;
        let moves = params
            .iter()
            .zip(locations)
            .filter_map(|(&param, location)| match *location {
                Location::Reg { class, reg } => Some(Move {
                    class,
                    from: Loc::Reg(class, reg),
                    to: self.loc(param),
                }),
                Location::Stack { .. } => None,
            })
            .collect();
        self.asm.parallel(moves);
        let area = self.incoming_area();
        for (&param, location) in params.iter().zip(locations) {
            let Location::Stack { offset, load, .. } = *location else {
                continue;
            };
            let scratch = match self.class(param) {
                RegClass::Int => T0,
                RegClass::Float | RegClass::Vector => F0,
            };
            let reg = self.def_reg(param, scratch);
            self.asm.memory(load, reg, e::FP, area + i64::from(offset));
            self.commit(param, reg);
        }
    }

    fn epilogue(&mut self, pop: u32) {
        self.callee_saves(true);
        self.asm.emit(e::mov_sp(e::SP, e::FP));
        self.asm.emit(e::LDP_FP_LR_POST);
        self.asm.adjust_sp(pop, false);
    }

    fn block(&mut self, block: Block, next: Option<Block>) -> Result<(), Unsupported> {
        self.block = block;
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
        self.asm.parallel(moves);
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

    fn flags(&mut self, condition: Condition) -> (Cond, Option<Cond>) {
        match condition {
            Condition::Int {
                cond,
                width,
                extend,
                left,
                right,
            } => {
                let mut a = self.use_reg(left, T0);
                if let Some(extend) = extend {
                    self.asm.emit(e::extend(extend, Width::W, T0, a));
                    a = T0;
                }
                match right {
                    CompareOperand::Subs(imm, shift) => {
                        self.asm.emit(e::subs_imm(width, e::ZR, a, imm, shift))
                    }
                    CompareOperand::Adds(imm, shift) => {
                        self.asm.emit(e::adds_imm(width, e::ZR, a, imm, shift))
                    }
                    CompareOperand::Reg(value) => {
                        let mut b = self.use_reg(value, T1);
                        if let Some(extend) = extend {
                            self.asm.emit(e::extend(extend, Width::W, T1, b));
                            b = T1;
                        }
                        self.asm.emit(e::alu(Alu::Subs, width, e::ZR, a, b));
                    }
                }
                (cond, None)
            }
            Condition::Float {
                ty,
                first,
                second,
                left,
                right,
            } => {
                let a = self.use_reg(left, F0);
                let b = self.use_reg(right, F1);
                self.asm.emit(e::fcmp(ty, a, b));
                (first, second)
            }
            Condition::Mask { width, value, mask } => {
                let a = self.use_reg(value, T0);
                match mask {
                    Mask::Immediate(bits) => self.asm.emit(
                        e::logical_imm(e::Logical::Ands, width, e::ZR, a, bits)
                            .expect("selected test mask"),
                    ),
                    Mask::Reg(value) => {
                        let b = self.use_reg(value, T1);
                        self.asm.emit(e::alu(Alu::Ands, width, e::ZR, a, b));
                    }
                }
                (Cond::Ne, None)
            }
            Condition::Zero {
                width,
                value,
                nonzero,
            } => {
                let reg = self.use_reg(value, T0);
                self.asm.emit(e::alu(Alu::Subs, width, e::ZR, reg, e::ZR));
                (if nonzero { Cond::Ne } else { Cond::Eq }, None)
            }
        }
    }

    fn branch_test(&mut self, condition: Condition) -> Test {
        match condition {
            Condition::Zero {
                width,
                value,
                nonzero,
            } => Test::Zero {
                width,
                reg: self.use_reg(value, T0),
                nonzero,
            },
            _ => Test::Flags(self.flags(condition).0),
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
        call: &Call,
        args: &[Value],
        results: &[Value],
        callee: Option<Value>,
        tail: bool,
        emit: impl FnOnce(&mut Self),
    ) {
        let assignment = &call.args;
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
                Location::Stack {
                    offset, store: op, ..
                } => {
                    let scratch = match self.class(arg) {
                        RegClass::Int => T0,
                        RegClass::Float | RegClass::Vector => F0,
                    };
                    let reg = self.use_reg(arg, scratch);
                    self.asm.memory(op, reg, base, area + i64::from(offset));
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
        self.asm.parallel(pending);
        for &(reg, extend) in &call.extensions {
            self.asm.emit(e::extend(extend, Width::W, reg, reg));
        }
        emit(self);
        if tail {
            return;
        }
        if call.callee_pops {
            self.asm.adjust_sp(assignment.stack_size, true);
        }
        let locations = &call.returns;
        let mut pending = Vec::new();
        for (&result, location) in results.iter().zip(locations) {
            if let Location::Reg { class, reg } = *location {
                pending.push(Move {
                    class,
                    from: Loc::Reg(class, reg),
                    to: self.loc(result),
                });
            }
        }
        self.asm.parallel(pending);
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
            TerminatorKind::Aarch64Branch(condition, yes, no) => {
                let test = self.branch_test(*condition);
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
            TerminatorKind::ReturnValues(values) => {
                let locations = &self.prepared.returns;
                let mut pending = Vec::with_capacity(values.len());
                for (&value, location) in values.iter().zip(locations) {
                    if let Location::Reg { class, reg } = *location {
                        pending.push(Move {
                            class,
                            from: self.loc(value),
                            to: Loc::Reg(class, reg),
                        });
                    }
                }
                self.asm.parallel(pending);
                self.epilogue(self.pop);
                self.asm.emit(e::RET);
            }
            TerminatorKind::TailCall { target, args, .. } => {
                let prepared = self.prepared;
                let call = prepared.tails[self.block].as_ref().unwrap();
                match target {
                    low_level::CallTarget::Direct(id) => {
                        self.call(call, args, &[], None, true, |_| {});
                        let size = call.args.stack_size;
                        self.epilogue(self.tail_args - size);
                        self.emit_call(e::B, ModuleRelocTarget::user(0, id.as_u32()));
                    }
                    low_level::CallTarget::Indirect(value) => {
                        self.call(call, args, &[], Some(*value), true, |_| {});
                        let size = call.args.stack_size;
                        self.epilogue(self.tail_args - size);
                        self.asm.emit(e::br(CALLEE));
                    }
                }
            }
            _ => return Err(Unsupported("unselected terminator".into())),
        }
        Ok(())
    }

    fn inst(&mut self, inst: Inst) -> Result<(), Unsupported> {
        let data = &self.func.insts[inst];
        let Op::Aarch64(op) = &data.op else {
            return Err(Unsupported("unselected instruction".into()));
        };
        let result = data.results.first().copied();
        use a::Op as A;
        match *op {
            A::Constant { bits, class } => {
                if let Loc::Reg(_, reg) = self.loc(result.unwrap()) {
                    self.asm.materialize(class, reg, bits);
                }
            }
            A::Alu {
                op,
                width,
                left,
                right,
            } => {
                let a = self.use_reg(left, T0);
                let b = self.use_reg(right, T1);
                let result = result.unwrap();
                let d = self.def_reg(result, T0);
                self.asm.emit(e::alu(op, width, d, a, b));
                self.commit(result, d);
            }
            A::AluImmediate {
                width,
                value,
                immediate,
            } => {
                let a = self.use_reg(value, T0);
                let result = result.unwrap();
                let d = self.def_reg(result, T0);
                let word = match immediate {
                    a::Immediate::Add(imm, shift) => e::add_imm(width, d, a, imm, shift),
                    a::Immediate::Sub(imm, shift) => e::sub_imm(width, d, a, imm, shift),
                    a::Immediate::Logical(op, bits) => {
                        e::logical_imm(op, width, d, a, bits).expect("selected logical immediate")
                    }
                };
                self.asm.emit(word);
                self.commit(result, d);
            }
            A::AddShifted {
                sub,
                width,
                left,
                right,
                kind,
                shift,
            } => {
                let a = self.use_reg(left, T0);
                let b = self.use_reg(right, T1);
                let result = result.unwrap();
                let d = self.def_reg(result, T0);
                self.asm.emit(match kind {
                    Index::Lsl => e::alu_shifted(sub, width, d, a, b, shift),
                    _ => e::alu_extended(sub, d, a, b, kind, shift),
                });
                self.commit(result, d);
            }
            A::Multiply { width, left, right } => {
                let a = self.use_reg(left, T0);
                let b = self.use_reg(right, T1);
                let result = result.unwrap();
                let d = self.def_reg(result, T0);
                self.asm.emit(e::mul(width, d, a, b));
                self.commit(result, d);
            }
            A::BitNot { width, value } => {
                let a = self.use_reg(value, T0);
                let result = result.unwrap();
                let d = self.def_reg(result, T0);
                self.asm.emit(e::alu(Alu::Orn, width, d, e::ZR, a));
                self.commit(result, d);
            }
            A::Shift {
                op,
                width,
                value,
                amount,
            } => {
                let a = self.use_reg(value, T0);
                let result = result.unwrap();
                let word = match amount {
                    a::ShiftAmount::Immediate(amount) => {
                        e::shift_imm(op, width, self.def_reg(result, T0), a, amount)
                    }
                    a::ShiftAmount::Reg(value) => {
                        let b = self.use_reg(value, T1);
                        e::shift(op, width, self.def_reg(result, T0), a, b)
                    }
                };
                self.asm.emit(word);
                self.commit(result, self.def_reg(result, T0));
            }
            A::Compare(condition) => {
                let (first, second) = self.flags(condition);
                let result = result.unwrap();
                let d = self.def_reg(result, T0);
                self.asm.emit(e::cset(d, first));
                if let Some(second) = second {
                    self.asm.emit(e::cset(T1, second));
                    self.asm.emit(e::alu(Alu::Orr, Width::W, d, d, T1));
                }
                self.commit(result, d);
            }
            A::Select {
                condition,
                yes,
                no,
                ty,
            } => {
                let cond = self.flags(condition).0;
                let result = result.unwrap();
                let (word, d) = match ty {
                    a::SelectType::Float(ty) => {
                        let a = self.use_reg(yes, F0);
                        let b = self.use_reg(no, F1);
                        let d = self.def_reg(result, F0);
                        (e::fcsel(ty, d, a, b, cond), d)
                    }
                    a::SelectType::Int(width) => {
                        let a = self.use_reg(yes, T1);
                        let b = self.use_reg(no, T2);
                        let d = self.def_reg(result, T0);
                        (e::csel(width, d, a, b, cond), d)
                    }
                };
                self.asm.emit(word);
                self.commit(result, d);
            }
            A::Copy { value } => {
                let result = result.unwrap();
                self.asm.parallel(vec![Move {
                    class: self.class(result),
                    from: self.loc(value),
                    to: self.loc(result),
                }]);
            }
            A::Bitcast { value, width, to } => {
                let result = result.unwrap();
                let (a, d) = if to == RegClass::Float {
                    (self.use_reg(value, T0), self.def_reg(result, F0))
                } else {
                    (self.use_reg(value, F0), self.def_reg(result, T0))
                };
                self.asm.emit(if to == RegClass::Float {
                    e::move_to_float(width, d, a)
                } else {
                    e::move_to_int(width, d, a)
                });
                self.commit(result, d);
            }
            A::Extend {
                value,
                width,
                extend,
            } => {
                let a = self.use_reg(value, T0);
                let result = result.unwrap();
                let d = self.def_reg(result, T0);
                self.asm.emit(e::extend(extend, width, d, a));
                self.commit(result, d);
            }
            A::Load { op, address } => {
                let result = result.unwrap();
                let scratch = if self.class(result) == RegClass::Float {
                    F0
                } else {
                    T0
                };
                let d = self.def_reg(result, scratch);
                self.access(op, address, d);
                self.commit(result, d);
            }
            A::Store { op, address, value } => {
                let reg = match value {
                    Some(value) => self.use_reg(
                        value,
                        if self.class(value) == RegClass::Float {
                            F0
                        } else {
                            T0
                        },
                    ),
                    None => e::ZR,
                };
                self.access(op, address, reg);
            }
            A::LoadAcquire { address, bytes } => {
                let base = self.use_reg(address, T1);
                let result = result.unwrap();
                let d = self.def_reg(result, T0);
                self.asm.emit(e::load_acquire(bytes, d, base));
                self.commit(result, d);
            }
            A::StackAddress { slot, offset } => {
                let offset = i64::from(self.frame.stack_slot(slot)) + i64::from(offset);
                let result = result.unwrap();
                let d = self.def_reg(result, T0);
                if (0..4096).contains(&offset) {
                    self.asm
                        .emit(e::add_imm(Width::X, d, e::SP, offset as u32, false));
                } else {
                    self.asm.mov_imm(Width::X, T0, offset as u64);
                    self.asm.emit(e::mov_sp(T1, e::SP));
                    self.asm.emit(e::alu(Alu::Add, Width::X, d, T1, T0));
                }
                self.commit(result, d);
            }
            A::StackPointer => {
                let result = result.unwrap();
                let d = self.def_reg(result, T0);
                self.asm.emit(e::mov_sp(d, e::SP));
                self.commit(result, d);
            }
            A::ExtractBits {
                value,
                width,
                shift,
                bits,
            } => {
                let a = self.use_reg(value, T0);
                let result = result.unwrap();
                let d = self.def_reg(result, T0);
                self.asm.emit(e::extract_bits(width, d, a, shift, bits));
                self.commit(result, d);
            }
            A::FloatBinary {
                op,
                ty,
                left,
                right,
            } => {
                let a = self.use_reg(left, F0);
                let b = self.use_reg(right, F1);
                let result = result.unwrap();
                let d = self.def_reg(result, F0);
                self.asm.emit(e::float_binary(op, ty, d, a, b));
                self.commit(result, d);
            }
            A::FloatUnary { op, ty, value } => {
                let a = self.use_reg(value, F0);
                let result = result.unwrap();
                let d = self.def_reg(result, F0);
                self.asm.emit(e::float_unary(op, ty, d, a));
                self.commit(result, d);
            }
            A::IntToFloat {
                signed,
                width,
                ty,
                value,
            } => {
                let a = self.use_reg(value, T0);
                let result = result.unwrap();
                let d = self.def_reg(result, F0);
                self.asm.emit(e::int_to_float(signed, width, ty, d, a));
                self.commit(result, d);
            }
            A::FloatToInt {
                signed,
                width,
                ty,
                value,
            } => {
                let a = self.use_reg(value, F0);
                let result = result.unwrap();
                let d = self.def_reg(result, T0);
                self.asm.emit(e::float_to_int(signed, width, ty, d, a));
                self.commit(result, d);
            }
            A::FloatConvert { promote, value } => {
                let a = self.use_reg(value, F0);
                let result = result.unwrap();
                let d = self.def_reg(result, F0);
                self.asm.emit(if promote {
                    e::promote(d, a)
                } else {
                    e::demote(d, a)
                });
                self.commit(result, d);
            }
            A::Call {
                ref target,
                ref args,
                ..
            } => {
                let prepared = self.prepared;
                let call = prepared.calls[inst].as_ref().unwrap();
                match target {
                    low_level::CallTarget::Direct(id) => {
                        self.call(call, args, &data.results, None, false, |this| {
                            this.emit_call(e::BL, ModuleRelocTarget::user(0, id.as_u32()))
                        })
                    }
                    low_level::CallTarget::Indirect(callee) => {
                        self.call(call, args, &data.results, Some(*callee), false, |this| {
                            this.asm.emit(e::blr(CALLEE))
                        })
                    }
                }
            }
            A::Vector(vector) => self.vector(vector, result),
        }
        Ok(())
    }

    fn vector(&mut self, vector: a::Vector, result: Option<Value>) {
        match vector {
            a::Vector::Load {
                address,
                bytes,
                signed,
            } => {
                let result = result.unwrap();
                let d = self.def_reg(result, F0);
                self.access(e::vector_memory(bytes, true), address, d);
                let mut bits = bytes * 8;
                while bits < 32 {
                    self.asm.emit(e::vector_widen(signed, bits, d, d));
                    bits *= 2;
                }
                self.commit(result, d);
            }
            a::Vector::Store {
                address,
                bytes,
                value,
            } => {
                let mut reg = self.use_reg(value, F0);
                let mut bits = 32;
                while bits > bytes * 8 {
                    self.asm.emit(e::vector_narrow(bits, F0, reg));
                    reg = F0;
                    bits /= 2;
                }
                self.access(e::vector_memory(bytes, false), address, reg);
            }
            a::Vector::Splat { value } => {
                let a = self.use_reg(value, T0);
                let result = result.unwrap();
                let d = self.def_reg(result, F0);
                self.asm.emit(e::vector_dup(d, a));
                self.commit(result, d);
            }
            a::Vector::Binary { op, left, right } => {
                let a = self.use_reg(left, F0);
                let b = self.use_reg(right, F1);
                let result = result.unwrap();
                let d = self.def_reg(result, F0);
                self.asm.emit(e::vector_binary(op, d, a, b));
                self.commit(result, d);
            }
            a::Vector::Shift { op, value, amount } => {
                let a = self.use_reg(value, F0);
                let result = result.unwrap();
                let d = self.def_reg(result, F0);
                self.asm.emit(e::vector_shift(op, d, a, amount));
                self.commit(result, d);
            }
        }
    }
}

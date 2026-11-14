use super::Unsupported;
use super::abi::{ADDRESS, F0, F1, T0, T1};
use super::encode::{self as e, Cond, Float, FloatUnary, Mem, Width};
use crate::jit::alloc::{
    RegClass,
    moves::{self, Loc, Move},
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Label(u32);

#[derive(Clone, Copy)]
enum Fixup {
    Imm26,
    Imm19,
}

#[derive(Default)]
pub(super) struct Assembler {
    far: bool,
    words: Vec<u32>,
    labels: Vec<Option<u32>>,
    fixups: Vec<(u32, Label, Fixup)>,
}

impl Assembler {
    pub(super) fn new(far: bool) -> Self {
        Self {
            far,
            ..Self::default()
        }
    }

    pub(super) fn emit(&mut self, word: u32) {
        self.words.push(word);
    }

    pub(super) fn offset(&self) -> u32 {
        (self.words.len() * 4) as u32
    }

    pub(super) fn label(&mut self) -> Label {
        self.labels.push(None);
        Label(self.labels.len() as u32 - 1)
    }

    pub(super) fn bind(&mut self, label: Label) {
        self.labels[label.0 as usize] = Some(self.words.len() as u32);
    }

    pub(super) fn jump(&mut self, word: u32, target: Label) {
        self.fixups
            .push((self.words.len() as u32, target, Fixup::Imm26));
        self.emit(word);
    }

    pub(super) fn branch(&mut self, cond: Cond, target: Label) {
        self.conditional(e::b_cond(cond), e::b_cond(cond.invert()), target);
    }

    pub(super) fn conditional(&mut self, word: u32, inverted: u32, target: Label) {
        if self.far {
            self.emit(inverted | (2 << 5));
            self.jump(e::B, target);
            return;
        }
        self.fixups
            .push((self.words.len() as u32, target, Fixup::Imm19));
        self.emit(word);
    }

    pub(super) fn conditional_overflow(&self) -> bool {
        self.fixups.iter().any(|&(at, target, fixup)| {
            let Some(position) = self.labels[target.0 as usize] else {
                return false;
            };
            let distance = i64::from(position) - i64::from(at);
            matches!(fixup, Fixup::Imm19) && !(-(1 << 18)..(1 << 18)).contains(&distance)
        })
    }

    pub(super) fn finish(mut self) -> Result<Vec<u8>, Unsupported> {
        for &(at, target, fixup) in &self.fixups {
            let Some(position) = self.labels[target.0 as usize] else {
                return Err(Unsupported("unbound label".into()));
            };
            let distance = i64::from(position) - i64::from(at);
            let word = &mut self.words[at as usize];
            match fixup {
                Fixup::Imm26 => {
                    if !(-(1 << 25)..(1 << 25)).contains(&distance) {
                        return Err(Unsupported("branch out of range".into()));
                    }
                    *word |= (distance as u32) & 0x03FF_FFFF;
                }
                Fixup::Imm19 => {
                    if !(-(1 << 18)..(1 << 18)).contains(&distance) {
                        return Err(Unsupported("conditional branch out of range".into()));
                    }
                    *word |= ((distance as u32) & 0x7_FFFF) << 5;
                }
            }
        }
        Ok(self
            .words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect())
    }
    pub(super) fn mov_imm(&mut self, width: Width, rd: u8, value: u64) {
        e::mov_imm(width, rd, value, |word| self.emit(word));
    }
    pub(super) fn load_imm(&mut self, rd: u8, value: u64) {
        e::load_imm(rd, value, |word| self.emit(word));
    }
    pub(super) fn materialize(&mut self, class: RegClass, rd: u8, bits: u64) {
        match class {
            RegClass::Int => self.load_imm(rd, bits),
            RegClass::Float if bits == 0 => self.emit(e::move_to_float(Width::X, rd, e::ZR)),
            RegClass::Float => match e::float_imm8(bits) {
                Some(imm8) => self.emit(e::fmov_imm(rd, imm8)),
                None => {
                    self.load_imm(ADDRESS, bits);
                    self.emit(e::move_to_float(Width::X, rd, ADDRESS));
                }
            },
        }
    }
    pub(super) fn memory(&mut self, op: Mem, rt: u8, base: u8, offset: i64) {
        if let Ok(unsigned) = u32::try_from(offset)
            && let Some(word) = e::mem_scaled(op, rt, base, unsigned)
        {
            self.emit(word);
            return;
        }
        if let Ok(signed) = i32::try_from(offset)
            && let Some(word) = e::mem_unscaled(op, rt, base, signed)
        {
            self.emit(word);
            return;
        }
        self.mov_imm(Width::X, ADDRESS, offset as u64);
        self.emit(e::mem_register(op, rt, base, ADDRESS));
    }
    pub(super) fn spill_op(class: RegClass, load: bool) -> Mem {
        match (class, load) {
            (RegClass::Int, true) => Mem::LdrX,
            (RegClass::Int, false) => Mem::StrX,
            (RegClass::Float, true) => Mem::LdrD,
            (RegClass::Float, false) => Mem::StrD,
        }
    }
    pub(super) fn emit_move(&mut self, m: Move) {
        match (m.from, m.to) {
            (Loc::Reg(_, from), Loc::Reg(_, to)) => {
                let word = match m.class {
                    RegClass::Int => e::mov(Width::X, to, from),
                    RegClass::Float => e::float_unary(FloatUnary::Mov, Float::D, to, from),
                };
                self.emit(word);
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
    pub(super) fn parallel(&mut self, pending: Vec<Move>) {
        let ordered = moves::sequence(pending, |class| match class {
            RegClass::Int => Loc::Reg(class, T1),
            RegClass::Float => Loc::Reg(class, F1),
        });
        for m in ordered {
            self.emit_move(m);
        }
    }
    pub(super) fn adjust_sp(&mut self, size: u32, sub: bool) {
        let encode = if sub { e::sub_imm } else { e::add_imm };
        if size >> 12 != 0 {
            self.emit(encode(Width::X, e::SP, e::SP, size >> 12, true));
        }
        if size & 0xFFF != 0 {
            self.emit(encode(Width::X, e::SP, e::SP, size & 0xFFF, false));
        }
    }
}

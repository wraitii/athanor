use super::Unsupported;
use super::encode::{self as e, Cond};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Label(u32);

#[derive(Clone, Copy)]
enum Fixup {
    Imm26,
    Imm19,
    Adr,
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
        if self.far {
            self.emit(e::b_cond(cond.invert()) | (2 << 5));
            self.jump(e::B, target);
            return;
        }
        self.fixups
            .push((self.words.len() as u32, target, Fixup::Imm19));
        self.emit(e::b_cond(cond));
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

    pub(super) fn address(&mut self, word: u32, target: Label) {
        self.fixups
            .push((self.words.len() as u32, target, Fixup::Adr));
        self.emit(word);
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
                Fixup::Adr => {
                    let bytes = distance * 4;
                    if !(-(1 << 20)..(1 << 20)).contains(&bytes) {
                        return Err(Unsupported("address out of range".into()));
                    }
                    let bytes = bytes as u32;
                    *word |= ((bytes & 3) << 29) | (((bytes >> 2) & 0x7_FFFF) << 5);
                }
            }
        }
        Ok(self
            .words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect())
    }
}

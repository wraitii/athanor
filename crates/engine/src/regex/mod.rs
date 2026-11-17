mod dfa;
mod nfa;
mod parse;
mod set;
#[cfg(test)]
pub(crate) mod tests;

pub(crate) use dfa::{DEAD, Dfa};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Unsupported(pub(crate) &'static str);

#[derive(Clone, Copy, Debug)]
pub(crate) struct Flags {
    pub(crate) icase: bool,
    pub(crate) multiline: bool,
    pub(crate) dot_all: bool,
    pub(crate) unicode: bool,
}

impl Flags {
    pub(crate) fn from_bits(bits: u8) -> Self {
        Self {
            icase: bits & 1 != 0,
            multiline: bits & 2 != 0,
            dot_all: bits & 4 != 0,
            unicode: bits & 8 != 0,
        }
    }
}

pub(crate) struct Program {
    pub(crate) dfa: Dfa,
    pub(crate) groups: usize,
}

pub(crate) fn compile(pattern: &str, flags: Flags) -> Result<Program, Unsupported> {
    let (node, groups) = parse::parse(pattern, flags)?;
    let nfa = nfa::Nfa::new(&node)?;
    Ok(Program {
        dfa: dfa::determinize(&nfa)?,
        groups,
    })
}

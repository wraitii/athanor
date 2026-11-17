mod dfa;
mod nfa;
mod parse;
mod set;
#[cfg(test)]
pub(crate) mod tests;

use std::cell::Cell;
use std::ops::Range;

pub(crate) use dfa::{DEAD, Dfa};

pub(crate) type Search = extern "C" fn(*const u8, u64, u64) -> u64;

pub(crate) const NO_MATCH: u64 = u64::MAX;

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

#[derive(Clone, Copy)]
enum Native {
    Untried,
    Unavailable,
    Ready { search: Search, groups: bool },
}

pub(crate) struct Captures {
    pub(crate) range: Range<usize>,
    pub(crate) groups: Vec<Option<Range<usize>>>,
}

pub(crate) struct Pattern {
    source: String,
    flags: u8,
    backtracking: regress::Regex,
    native: Cell<Native>,
}

impl Pattern {
    pub(crate) fn new(source: &str, flags: u8) -> Result<Self, String> {
        let backtracking = regress::Regex::with_flags(source, crate::heap::regexp_flags(flags))
            .map_err(|error| error.to_string())?;
        Ok(Self {
            source: source.to_owned(),
            flags,
            backtracking,
            native: Cell::new(Native::Untried),
        })
    }

    pub(crate) fn needs_native(&self) -> bool {
        matches!(self.native.get(), Native::Untried)
    }

    pub(crate) fn program(&self) -> Result<Program, Unsupported> {
        compile(&self.source, Flags::from_bits(self.flags))
    }

    pub(crate) fn install(&self, native: Option<(Search, usize)>) {
        self.native.set(match native {
            Some((search, groups)) => Native::Ready {
                search,
                groups: groups != 0,
            },
            None => Native::Unavailable,
        });
    }

    fn native_search(&self, text: &str) -> Option<(Search, bool)> {
        match self.native.get() {
            Native::Ready { search, groups } if text.len() <= u32::MAX as usize => {
                Some((search, groups))
            }
            _ => None,
        }
    }

    pub(crate) fn find(&self, text: &str, from: usize) -> Option<Range<usize>> {
        let Some((search, _)) = self.native_search(text) else {
            return self.backtracking.find_from(text, from).next().map(|found| found.range);
        };
        let found = search(text.as_ptr(), text.len() as u64, from as u64);
        (found != NO_MATCH).then(|| (found >> 32) as usize..found as u32 as usize)
    }

    pub(crate) fn captures(&self, text: &str, from: usize) -> Option<Captures> {
        if let Some((_, false)) = self.native_search(text) {
            let range = self.find(text, from)?;
            return Some(Captures {
                range,
                groups: Vec::new(),
            });
        }
        let found = self.backtracking.find_from(text, from).next()?;
        Some(Captures {
            range: found.range,
            groups: found.captures,
        })
    }
}

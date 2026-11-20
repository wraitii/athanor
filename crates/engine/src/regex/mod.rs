mod captures;
mod dfa;
mod nfa;
mod parse;
mod set;
#[cfg(test)]
pub(crate) mod tests;

use std::cell::{Cell, RefCell};
use std::ops::Range;
use std::sync::{LazyLock, Mutex, OnceLock};

use crate::FxHashMap;

pub(crate) use dfa::{DEAD, Dfa, is_word};

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
    pub(crate) global: bool,
}

impl Flags {
    pub(crate) fn from_bits(bits: u8) -> Self {
        Self {
            icase: bits & 1 != 0,
            multiline: bits & 2 != 0,
            dot_all: bits & 4 != 0,
            unicode: bits & 8 != 0,
            global: bits & 16 != 0,
        }
    }
}

pub(crate) struct Program {
    pub(crate) dfa: Dfa,
    pub(crate) groups: usize,
    nfa: nfa::Nfa,
}

impl Program {
    fn captures(
        &self,
        text: &str,
        range: Range<usize>,
        scratch: &mut captures::Scratch,
    ) -> Option<Vec<Option<Range<usize>>>> {
        captures::captures(&self.nfa, self.groups, text.as_bytes(), range, scratch)
    }
}

pub(crate) fn compile(pattern: &str, flags: Flags) -> Result<Program, Unsupported> {
    let (node, groups) = parse::parse(pattern, flags)?;
    let nfa = nfa::Nfa::new(&node)?;
    Ok(Program {
        dfa: dfa::determinize(&nfa)?,
        groups,
        nfa,
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
    definition: &'static Definition,
    backtracking: regress::Regex,
    native: Cell<Native>,
    scratch: RefCell<captures::Scratch>,
}

pub(crate) struct Definition {
    source: String,
    flags: u8,
    program: OnceLock<Result<Program, Unsupported>>,
}

impl Definition {
    pub(crate) fn program(&self) -> Result<&Program, Unsupported> {
        self.program
            .get_or_init(|| compile(&self.source, Flags::from_bits(self.flags)))
            .as_ref()
            .map_err(|error| *error)
    }

    pub(crate) fn global(&self) -> bool {
        Flags::from_bits(self.flags).global
    }

    // Compiled code embeds definition addresses and background compilation can outlive a heap,
    // so definitions are interned for the whole process and never freed.
    fn intern(source: &str, flags: u8) -> &'static Self {
        static DEFINITIONS: LazyLock<Mutex<FxHashMap<(String, u8), &'static Definition>>> =
            LazyLock::new(Default::default);
        let mut definitions = DEFINITIONS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        definitions
            .entry((source.to_owned(), flags))
            .or_insert_with(|| {
                Box::leak(Box::new(Definition {
                    source: source.to_owned(),
                    flags,
                    program: OnceLock::new(),
                }))
            })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PatternRef(&'static Definition);

impl PatternRef {
    pub(crate) fn address(self) -> usize {
        std::ptr::from_ref(self.0) as usize
    }

    pub(crate) fn get(self) -> &'static Definition {
        self.0
    }
}

impl PartialEq for Definition {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl Eq for Definition {}

impl std::hash::Hash for Definition {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(self, state);
    }
}

impl std::fmt::Debug for Definition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "/{}/{}", self.source, self.flags)
    }
}

impl std::fmt::Display for PatternRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl Pattern {
    pub(crate) fn new(source: &str, flags: u8) -> Result<Self, String> {
        let backtracking = regress::Regex::with_flags(source, crate::heap::regexp_flags(flags))
            .map_err(|error| error.to_string())?;
        Ok(Self {
            definition: Definition::intern(source, flags),
            backtracking,
            native: Cell::new(Native::Untried),
            scratch: RefCell::default(),
        })
    }

    pub(crate) fn needs_native(&self) -> bool {
        matches!(self.native.get(), Native::Untried)
    }

    pub(crate) fn reference(&self) -> PatternRef {
        PatternRef(self.definition)
    }

    pub(crate) fn program(&self) -> Result<&'static Program, Unsupported> {
        self.definition.program()
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
            return self
                .backtracking
                .find_from(text, from)
                .next()
                .map(|found| found.range);
        };
        let found = search(text.as_ptr(), text.len() as u64, from as u64);
        (found != NO_MATCH).then(|| (found >> 32) as usize..found as u32 as usize)
    }

    pub(crate) fn captures(&self, text: &str, from: usize) -> Option<Captures> {
        if self.native_search(text).is_none() {
            return self.backtrack(text, from);
        }
        let range = self.find(text, from)?;
        self.captures_at(text, range)
    }

    pub(crate) fn captures_at(&self, text: &str, range: Range<usize>) -> Option<Captures> {
        match self.native_search(text) {
            None => return self.backtrack(text, range.start),
            Some((_, false)) => {
                return Some(Captures {
                    range,
                    groups: Vec::new(),
                });
            }
            Some((_, true)) => {}
        }
        let groups = self.program().ok().and_then(|program| {
            program.captures(text, range.clone(), &mut self.scratch.borrow_mut())
        });
        match groups {
            Some(groups) => Some(Captures { range, groups }),
            None => self.backtrack(text, range.start),
        }
    }

    fn backtrack(&self, text: &str, from: usize) -> Option<Captures> {
        let found = self.backtracking.find_from(text, from).next()?;
        Some(Captures {
            range: found.range,
            groups: found.captures,
        })
    }
}

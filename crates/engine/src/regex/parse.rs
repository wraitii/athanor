use oxc_regular_expression::ast::{
    Alternative, BoundaryAssertionKind, CharacterClass, CharacterClassContents,
    CharacterClassContentsKind, CharacterClassEscapeKind, Disjunction, Term,
};

use super::set::{self, Set};
use super::{Flags, Unsupported};

const REPEAT_LIMIT: u64 = 1000;

pub(super) enum Node {
    Empty,
    Set(Set),
    Concat(Vec<Node>),
    Alternation(Vec<Node>),
    Repeat {
        node: Box<Node>,
        min: u32,
        max: Option<u32>,
        greedy: bool,
    },
    Group(usize, Box<Node>),
    WordBoundary(bool),
    Start,
    End,
}

impl Node {
    fn nullable(&self) -> bool {
        match self {
            Self::Empty | Self::Start | Self::End | Self::WordBoundary(_) => true,
            Self::Set(_) => false,
            Self::Concat(nodes) => nodes.iter().all(Self::nullable),
            Self::Alternation(nodes) => nodes.iter().any(Self::nullable),
            Self::Repeat { node, min, .. } => *min == 0 || node.nullable(),
            Self::Group(_, node) => node.nullable(),
        }
    }

    pub(super) fn groups(&self) -> Option<(usize, usize)> {
        let children = match self {
            Self::Empty | Self::Start | Self::End | Self::Set(_) | Self::WordBoundary(_) => {
                return None;
            }
            Self::Concat(nodes) | Self::Alternation(nodes) => nodes.iter().collect(),
            Self::Repeat { node, .. } => vec![node.as_ref()],
            Self::Group(index, node) => {
                let last = node.groups().map_or(*index, |(_, last)| last);
                return Some((*index, last));
            }
        };
        children
            .into_iter()
            .filter_map(Self::groups)
            .reduce(|(first, _), (_, last)| (first, last))
    }
}

pub(super) fn parse(pattern: &str, flags: Flags) -> Result<(Node, usize), Unsupported> {
    let allocator = oxc_allocator::Allocator::default();
    let flags_text = if flags.unicode { "u" } else { "" };
    let parsed = oxc_regular_expression::LiteralParser::new(
        &allocator,
        pattern,
        Some(flags_text),
        oxc_regular_expression::Options::default(),
    )
    .parse()
    .map_err(|_| Unsupported("syntax"))?;
    let lowering = Lowering {
        flags,
        groups: std::cell::Cell::new(0),
    };
    let node = lowering.disjunction(&parsed.body)?;
    Ok((node, lowering.groups.get()))
}

struct Lowering {
    flags: Flags,
    groups: std::cell::Cell<usize>,
}

impl Lowering {
    fn disjunction(&self, disjunction: &Disjunction<'_>) -> Result<Node, Unsupported> {
        let mut nodes = disjunction
            .body
            .iter()
            .map(|alternative| self.alternative(alternative))
            .collect::<Result<Vec<_>, _>>()?;
        if nodes.len() == 1 {
            return Ok(nodes.pop().unwrap());
        }
        Ok(Node::Alternation(nodes))
    }

    fn alternative(&self, alternative: &Alternative<'_>) -> Result<Node, Unsupported> {
        let mut nodes = alternative
            .body
            .iter()
            .map(|term| self.term(term))
            .collect::<Result<Vec<_>, _>>()?;
        match nodes.len() {
            0 => Ok(Node::Empty),
            1 => Ok(nodes.pop().unwrap()),
            _ => Ok(Node::Concat(nodes)),
        }
    }

    fn term(&self, term: &Term<'_>) -> Result<Node, Unsupported> {
        match term {
            Term::BoundaryAssertion(assertion) => match assertion.kind {
                BoundaryAssertionKind::Start | BoundaryAssertionKind::End
                    if self.flags.multiline =>
                {
                    Err(Unsupported("multiline anchor"))
                }
                BoundaryAssertionKind::Start => Ok(Node::Start),
                BoundaryAssertionKind::End => Ok(Node::End),
                _ if self.flags.icase && self.flags.unicode => {
                    Err(Unsupported("unicode case-insensitive word boundary"))
                }
                BoundaryAssertionKind::Boundary => Ok(Node::WordBoundary(false)),
                BoundaryAssertionKind::NegativeBoundary => Ok(Node::WordBoundary(true)),
                _ => Err(Unsupported("buffer anchor")),
            },
            Term::LookAroundAssertion(_) => Err(Unsupported("lookaround")),
            Term::Quantifier(quantifier) => {
                let node = self.term(&quantifier.body)?;
                if quantifier.min > REPEAT_LIMIT
                    || quantifier.max.is_some_and(|max| max > REPEAT_LIMIT)
                {
                    return Err(Unsupported("large counted repetition"));
                }
                if node.nullable() && quantifier.max.is_none_or(|max| max > 1) {
                    return Err(Unsupported("repetition of an empty-matching body"));
                }
                Ok(Node::Repeat {
                    node: Box::new(node),
                    min: quantifier.min as u32,
                    max: quantifier.max.map(|max| max as u32),
                    greedy: quantifier.greedy,
                })
            }
            Term::Character(character) => {
                self.atom(Set::range(character.value, character.value), character.value)
            }
            Term::Dot(_) => {
                let any = Set::range(0, set::MAX);
                if self.flags.dot_all {
                    return self.set(any);
                }
                self.set(any.minus(&set::line_terminators()))
            }
            Term::CharacterClassEscape(escape) => self.set(escape_set(escape.kind)),
            Term::UnicodePropertyEscape(_) => Err(Unsupported("unicode property escape")),
            Term::CharacterClass(class) => {
                let positive = self.class(class)?;
                let positive = self.fold(positive)?;
                Ok(Node::Set(if class.negative {
                    positive.complement().without_surrogates()
                } else {
                    positive
                }))
            }
            Term::CapturingGroup(group) => {
                let index = self.groups.get() + 1;
                self.groups.set(index);
                Ok(Node::Group(index, Box::new(self.disjunction(&group.body)?)))
            }
            Term::IgnoreGroup(group) => {
                if group.modifiers.is_some() {
                    return Err(Unsupported("group modifiers"));
                }
                self.disjunction(&group.body)
            }
            Term::IndexedReference(_) | Term::NamedReference(_) => {
                Err(Unsupported("backreference"))
            }
        }
    }

    fn class(&self, class: &CharacterClass<'_>) -> Result<Set, Unsupported> {
        if class.kind != CharacterClassContentsKind::Union || class.strings {
            return Err(Unsupported("set operations"));
        }
        let mut result = Set::default();
        for contents in &class.body {
            let part = match contents {
                CharacterClassContents::CharacterClassRange(range) => {
                    let (start, end) = (range.min.value, range.max.value);
                    if set::is_surrogate(start) || set::is_surrogate(end) {
                        return Err(Unsupported("surrogate code unit"));
                    }
                    Set::range(start, end)
                }
                CharacterClassContents::CharacterClassEscape(escape) => escape_set(escape.kind),
                CharacterClassContents::Character(character) => {
                    if set::is_surrogate(character.value) {
                        return Err(Unsupported("surrogate code unit"));
                    }
                    Set::range(character.value, character.value)
                }
                CharacterClassContents::NestedCharacterClass(_)
                | CharacterClassContents::ClassStringDisjunction(_)
                | CharacterClassContents::UnicodePropertyEscape(_) => {
                    return Err(Unsupported("set operations"));
                }
            };
            result = result.union(&part);
        }
        Ok(result)
    }

    fn atom(&self, set: Set, value: u32) -> Result<Node, Unsupported> {
        if set::is_surrogate(value) {
            return Err(Unsupported("surrogate code unit"));
        }
        self.set(set)
    }

    fn set(&self, set: Set) -> Result<Node, Unsupported> {
        Ok(Node::Set(self.fold(set)?.without_surrogates()))
    }

    fn fold(&self, set: Set) -> Result<Set, Unsupported> {
        if !self.flags.icase {
            return Ok(set);
        }
        if self.flags.unicode {
            return Err(Unsupported("unicode case folding"));
        }
        let non_ascii = Set::range(0x80, set::MAX);
        let uncased = set::whitespace().intersect(&non_ascii);
        let present = set.intersect(&non_ascii);
        let absent = non_ascii.minus(&present);
        if !present.minus(&uncased).is_empty() && !absent.minus(&uncased).is_empty() {
            return Err(Unsupported("non-ASCII case folding"));
        }
        Ok(set.ascii_case_closure())
    }
}

fn escape_set(kind: CharacterClassEscapeKind) -> Set {
    match kind {
        CharacterClassEscapeKind::D => set::digits(),
        CharacterClassEscapeKind::NegativeD => set::digits().complement(),
        CharacterClassEscapeKind::S => set::whitespace(),
        CharacterClassEscapeKind::NegativeS => set::whitespace().complement(),
        CharacterClassEscapeKind::W => set::word(),
        CharacterClassEscapeKind::NegativeW => set::word().complement(),
    }
}

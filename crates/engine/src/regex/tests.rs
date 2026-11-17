use oxc_ast::ast::{RegExpFlags, RegExpLiteral, StringLiteral};
use oxc_ast_visit::Visit;

use super::dfa::is_word;
use super::{DEAD, Dfa, Flags, Unsupported, compile};

fn find(dfa: &Dfa, text: &str, from: usize) -> Option<(usize, usize)> {
    let bytes = text.as_bytes();
    for start in from..=bytes.len() {
        if !text.is_char_boundary(start) {
            continue;
        }
        let mut state = if start == 0 {
            dfa.start_at_beginning
        } else if is_word(bytes[start - 1]) {
            dfa.start_after_word
        } else {
            dfa.start
        };
        let mut last = None;
        let mut position = start;
        loop {
            if position == bytes.len() {
                if dfa.matching_at_end[state as usize] {
                    last = Some(position);
                }
                break;
            }
            if dfa.matches_before(state, bytes[position]) {
                last = Some(position);
            }
            state = dfa.next(state, bytes[position]);
            position += 1;
            if state == DEAD {
                break;
            }
        }
        if let Some(end) = last {
            return Some((start, end));
        }
    }
    None
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }
}

fn boundaries(text: &str) -> Vec<usize> {
    let mut offsets: Vec<usize> = [0, 1, text.len() / 2, text.len()]
        .into_iter()
        .filter(|&offset| offset <= text.len())
        .map(|mut offset| {
            while !text.is_char_boundary(offset) {
                offset += 1;
            }
            offset
        })
        .collect();
    offsets.dedup();
    offsets
}

fn inputs(pattern: &str, pool: &[String], rng: &mut Rng) -> Vec<String> {
    let mut alphabet: Vec<char> = pattern.chars().filter(|c| !"\\[]()|*+?{}^$".contains(*c)).collect();
    alphabet.extend(['a', 'B', '0', '_', ' ', '\t', '\n', '=', ';', '/', 'é', '€', '😀', '\u{a0}', '\u{2028}']);
    let mut inputs: Vec<String> = (0..64)
        .map(|_| {
            let length = rng.below(24);
            (0..length).map(|_| alphabet[rng.below(alphabet.len())]).collect()
        })
        .collect();
    inputs.push(String::new());
    for _ in 0..24 {
        inputs.push(pool[rng.below(pool.len())].clone());
    }
    inputs
}

pub(crate) type Search<'a> = Box<dyn Fn(&str, usize) -> Option<(usize, usize)> + 'a>;

fn check<'a>(
    pattern: &str,
    bits: u8,
    pool: &[String],
    rng: &mut Rng,
    matcher: &mut dyn FnMut(&Dfa) -> Search<'a>,
) -> Result<(), Unsupported> {
    let program = compile(pattern, Flags::from_bits(bits))?;
    let search = matcher(&program.dfa);
    let mut scratch = Default::default();
    let oracle = regress::Regex::with_flags(pattern, crate::heap::regexp_flags(bits)).unwrap();
    for text in inputs(pattern, pool, rng) {
        for from in boundaries(&text) {
            let found = oracle.find_from(&text, from).next();
            let expected = found
                .as_ref()
                .map(|found| (found.range.start, found.range.end));
            assert_eq!(
                search(&text, from),
                expected,
                "/{pattern}/ (flags {bits}) on {text:?} from {from}"
            );
            let Some(found) = found else {
                continue;
            };
            assert_eq!(
                program.captures(&text, found.range.clone(), &mut scratch),
                Some(found.captures),
                "groups of /{pattern}/ (flags {bits}) on {text:?} from {from}"
            );
        }
    }
    Ok(())
}

#[derive(Default)]
struct Literals {
    patterns: Vec<(String, u8)>,
    strings: Vec<String>,
}

impl<'a> Visit<'a> for Literals {
    fn visit_reg_exp_literal(&mut self, literal: &RegExpLiteral<'a>) {
        let flags = literal.regex.flags;
        if flags.intersects(RegExpFlags::Y | RegExpFlags::D | RegExpFlags::V) {
            return;
        }
        let bits = u8::from(flags.contains(RegExpFlags::I))
            | (u8::from(flags.contains(RegExpFlags::M)) << 1)
            | (u8::from(flags.contains(RegExpFlags::S)) << 2)
            | (u8::from(flags.contains(RegExpFlags::U)) << 3);
        self.patterns
            .push((literal.regex.pattern.text.to_string(), bits));
    }

    fn visit_string_literal(&mut self, literal: &StringLiteral<'a>) {
        if let Some(value) = literal.value.as_str()
            && value.len() < 600
        {
            self.strings.push(value.to_owned());
        }
    }
}

#[test]
fn automata_match_regress() {
    differential(&mut |dfa| {
        let dfa = std::rc::Rc::new(dfa.clone());
        Box::new(move |text, from| find(&dfa, text, from))
    });
}

pub(crate) fn differential<'a>(matcher: &mut dyn FnMut(&Dfa) -> Search<'a>) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/v8-v7/regexp.js");
    let source = std::fs::read_to_string(path).unwrap();
    let allocator = oxc_allocator::Allocator::default();
    let program = crate::parse::parse(&allocator, &source, "regexp.js").unwrap();
    let mut literals = Literals::default();
    literals.visit_program(&program);
    let corpus = [
        (r"\s+", 0),
        (r"\{civ\}", 0),
        (r"^", 2),
        (r"\{(civ|native)\}", 0),
        (r"\s*\.\.\s*", 0),
        (r"\[", 0),
        (r"^[A-Z]+$", 0),
        (r"\s*,\s*", 0),
        (r"[+\s]+", 0),
        (r"hotkey.([a-z0-9_.]+)", 0),
        (r"\.0$", 0),
        (r"^(?=.*[a-z])[a-z0-9._-]{3,20}$", 1),
        (r"a|ab", 0),
        (r"ab|a", 0),
        (r"a*?b", 0),
        (r"(a|b)*?c|a+", 0),
        (r"x{2,3}?y?", 0),
        (r"^$", 0),
        (r"$^", 0),
        (r"[^a-c]+", 1),
        (r"[\S]", 1),
        (r"\s", 1),
        (r"é+", 0),
        (r"[à-ÿ€]+", 0),
        (r".+", 0),
        (r".+", 4),
        (r"[^]", 0),
        (r"\u{1F600}|é", 8),
        (r"(a)|(b)", 0),
        (r"(a|(b))+", 0),
        (r"(?:(a)|b)*c", 0),
        (r"(\d+)-(\d+)?", 0),
        (r"^(?:(x)|(y))+$", 0),
        (r"((a)|(b)){2,3}?(c)?", 0),
        (r"(a*?)(a*)", 0),
        (r"(a+|b+)*?(c)", 0),
        (r"\b", 0),
        (r"\B", 0),
        (r"\ba", 0),
        (r"a\b", 0),
        (r"\Ba\B", 0),
        (r"\b\w+\b", 0),
        (r"(\$a)|(\$a\b)", 0),
        (r"é\b|\bé", 0),
        (r"\bB", 1),
        (r"x\b|y\B|z", 0),
        (r"(?:\b(a)|a\B)+", 0),
        (r"a$\b", 0),
        (r"a$\B", 0),
        (r"b|a$\b", 0),
        (r"\w$\b|\W$\B", 0),
    ];
    let patterns = corpus
        .iter()
        .map(|&(pattern, bits)| (pattern.to_owned(), bits))
        .chain(literals.patterns);
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut supported = 0;
    let mut unsupported: Vec<(String, &'static str)> = Vec::new();
    for (pattern, bits) in patterns {
        match check(&pattern, bits, &literals.strings, &mut rng, matcher) {
            Ok(()) => supported += 1,
            Err(Unsupported(reason)) => unsupported.push((pattern, reason)),
        }
    }
    eprintln!("{supported} supported, {} unsupported", unsupported.len());
    for (pattern, reason) in &unsupported {
        eprintln!("  {reason}: /{pattern}/");
    }
    assert!(supported > 0);
}

use std::collections::{HashMap, HashSet};
use std::fmt;

use cranelift_codegen::entity::SecondaryMap;
use oxc_syntax::operator::BinaryOperator;

use super::{Block, Cfg, Constant, Graph, Inst, Op, TerminatorKind, UnaryOp, Value};
use crate::heap::TypedKind;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::jit) struct Types(u16);

impl Types {
    pub(in crate::jit) const INT32: Self = Self(1);
    pub(in crate::jit) const DOUBLE: Self = Self(2);
    pub(in crate::jit) const INT64: Self = Self(2048);
    pub(in crate::jit) const INTEGER: Self = Self(1 | 2048);
    pub(in crate::jit) const BOOLEAN: Self = Self(4);
    pub(in crate::jit) const UNDEFINED: Self = Self(8);
    pub(in crate::jit) const NULL: Self = Self(16);
    pub(in crate::jit) const STRING: Self = Self(32);
    pub(in crate::jit) const OBJECT: Self = Self(64);
    pub(in crate::jit) const OTHER: Self = Self(128);
    pub(in crate::jit) const UNINITIALIZED: Self = Self(256);
    pub(in crate::jit) const PACK: Self = Self(1024);
    pub(in crate::jit) const VIEW: Self = Self(512);
    pub(in crate::jit) const NUMBER: Self = Self(1 | 2 | 2048);
    pub(in crate::jit) const PRIMITIVE: Self = Self(1 | 2 | 4 | 8 | 16 | 32 | 2048);
    pub(in crate::jit) const VALUE: Self = Self((511 & !256) | 2048);
    pub(in crate::jit) const ALL: Self = Self(511 | 2048);

    pub(in crate::jit) const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub(in crate::jit) fn within(self, other: Self) -> bool {
        self.0 & !other.0 == 0
    }

    pub(in crate::jit) const fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    pub(in crate::jit) fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    pub(in crate::jit) fn is_informative(self) -> bool {
        !Self::VALUE.within(self)
    }
}

impl fmt::Display for Types {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut names = Vec::new();
        if Self::NUMBER.within(*self) {
            names.push("number");
        } else {
            if self.intersects(Self::INT32) {
                names.push("int32");
            }
            if self.intersects(Self::INT64) {
                names.push("int64");
            }
            if self.intersects(Self::DOUBLE) {
                names.push("double");
            }
        }
        for (types, name) in [
            (Self::BOOLEAN, "bool"),
            (Self::UNDEFINED, "undefined"),
            (Self::NULL, "null"),
            (Self::STRING, "string"),
            (Self::OBJECT, "object"),
            (Self::OTHER, "other"),
            (Self::UNINITIALIZED, "uninit"),
            (Self::VIEW, "view"),
            (Self::PACK, "pack"),
        ] {
            if self.intersects(types) {
                names.push(name);
            }
        }
        if names.is_empty() {
            names.push("none");
        }
        f.write_str(&names.join("|"))
    }
}

pub(in crate::jit) struct Facts {
    types: SecondaryMap<Value, Types>,
    indexed: HashSet<Value>,
    known: bool,
}

impl Facts {
    pub(super) fn indexed(&self, value: Value) -> bool {
        self.indexed.contains(&value)
    }

    pub(in crate::jit) fn of(&self, value: Value) -> Types {
        if self.known {
            self.types[value]
        } else {
            Types::ALL
        }
    }

    pub(in crate::jit) fn compute(graph: &Graph) -> Self {
        Self::assuming(graph, &HashMap::new())
    }

    pub(super) fn assuming(graph: &Graph, assumptions: &HashMap<Value, Types>) -> Self {
        let cfg = Cfg::new(graph);
        let steps = steps(graph, &cfg);
        let (indexed, reads) = super::string_split::result_facts(graph);
        let mut facts = Self {
            types: SecondaryMap::new(),
            indexed,
            known: true,
        };
        for (&value, &types) in assumptions {
            facts.types[value] = types;
        }
        let mut changed = true;
        while changed {
            changed = false;
            for &block in &cfg.rpo {
                for &inst in &graph.blocks[block].insts {
                    let data = &graph.insts[inst];
                    if let Some(result) = data.result {
                        if assumptions.contains_key(&result) {
                            continue;
                        }
                        let types = if let Some(&types) = reads.get(&inst) {
                            types
                        } else {
                            match steps.get(&inst) {
                                Some(bounds) if facts.bounded(&data.op, bounds) => Types::INT32,
                                _ => result_types(&data.op, &facts),
                            }
                        };
                        changed |= facts.widen(result, types);
                    }
                }
                let Some(terminator) = &graph.blocks[block].terminator else {
                    continue;
                };
                let refinement = refined_edge(graph, &terminator.kind);
                for target in terminator.kind.targets() {
                    for (&param, &arg) in graph.blocks[target.block].params.iter().zip(&target.args)
                    {
                        if assumptions.contains_key(&param) {
                            continue;
                        }
                        let mut types = facts.of(arg);
                        if let Some((yes, value, mask)) = refinement
                            && value == arg
                        {
                            types = if target.block == yes {
                                Types(types.0 & mask.0)
                            } else {
                                types.without(mask)
                            };
                        }
                        changed |= facts.widen(param, types);
                    }
                }
            }
        }
        facts
    }

    fn bounded(&self, op: &Op, bounds: &[Value]) -> bool {
        step(op).is_some_and(|(operand, _)| self.of(operand).within(Types::INT32))
            && bounds
                .iter()
                .any(|&bound| self.of(bound).within(Types::INT32))
    }

    fn widen(&mut self, value: Value, types: Types) -> bool {
        let widened = self.types[value].union(types);
        if widened == self.types[value] {
            return false;
        }
        self.types[value] = widened;
        true
    }
}

fn refined_edge(graph: &Graph, kind: &TerminatorKind) -> Option<(Block, Value, Types)> {
    let TerminatorKind::Branch(condition, yes, no) = kind else {
        return None;
    };
    if yes.block == no.block {
        return None;
    }
    let (value, mask) = match *graph.defining_op(*condition)? {
        Op::IsUninitialized(value) => (value, Types::UNINITIALIZED),
        Op::IsUndefined(value) => (value, Types::UNDEFINED),
        Op::IsNullish(value) => (value, Types::UNDEFINED.union(Types::NULL)),
        Op::IsString(value) => (value, Types::STRING),
        Op::IsNumber(value) => (value, Types::NUMBER),
        Op::IsInteger(value) => (value, Types::INTEGER),
        _ => return None,
    };
    Some((yes.block, value, mask))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Relation {
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

impl Relation {
    fn of(operator: BinaryOperator) -> Option<Self> {
        Some(match operator {
            BinaryOperator::LessThan => Self::Less,
            BinaryOperator::LessEqualThan => Self::LessEqual,
            BinaryOperator::GreaterThan => Self::Greater,
            BinaryOperator::GreaterEqualThan => Self::GreaterEqual,
            _ => return None,
        })
    }

    fn negate(self) -> Self {
        match self {
            Self::Less => Self::GreaterEqual,
            Self::LessEqual => Self::Greater,
            Self::Greater => Self::LessEqual,
            Self::GreaterEqual => Self::Less,
        }
    }

    fn swap(self) -> Self {
        match self {
            Self::Less => Self::Greater,
            Self::LessEqual => Self::GreaterEqual,
            Self::Greater => Self::Less,
            Self::GreaterEqual => Self::LessEqual,
        }
    }
}

fn one(graph: &Graph, value: Value) -> bool {
    matches!(graph.defining_op(value), Some(Op::Constant(Constant::Number(n))) if *n == 1.0)
}

fn step(op: &Op) -> Option<(Value, bool)> {
    match *op {
        Op::Unary(UnaryOp::Increment, operand) => Some((operand, true)),
        Op::Unary(UnaryOp::Decrement, operand) => Some((operand, false)),
        Op::Binary(BinaryOperator::Addition, left, _) => Some((left, true)),
        Op::Binary(BinaryOperator::Subtraction, left, _) => Some((left, false)),
        _ => None,
    }
}

fn steps(graph: &Graph, cfg: &Cfg) -> HashMap<Inst, Vec<Value>> {
    let mut incoming: HashMap<Value, Vec<Value>> = HashMap::new();
    let mut relations: Vec<(Block, Value, Relation, Value)> = Vec::new();
    for &block in &cfg.rpo {
        let Some(terminator) = &graph.blocks[block].terminator else {
            continue;
        };
        for target in terminator.kind.targets() {
            for (&param, &arg) in graph.blocks[target.block].params.iter().zip(&target.args) {
                incoming.entry(param).or_default().push(arg);
            }
        }
        let TerminatorKind::Branch(condition, yes, no) = &terminator.kind else {
            continue;
        };
        let Some(
            &(Op::Binary(operator, left, right)
            | Op::GuardedCompare {
                operator,
                left,
                right,
                ..
            }),
        ) = graph.defining_op(*condition)
        else {
            continue;
        };
        let Some(relation) = Relation::of(operator) else {
            continue;
        };
        if yes.block == no.block {
            continue;
        }
        for (target, relation) in [(yes.block, relation), (no.block, relation.negate())] {
            if cfg.preds[target].len() == 1 {
                relations.push((target, left, relation, right));
                relations.push((target, right, relation.swap(), left));
            }
        }
    }
    let mut steps = HashMap::new();
    for &block in &cfg.rpo {
        for &inst in &graph.blocks[block].insts {
            let op = &graph.insts[inst].op;
            let Some((operand, up)) = step(op) else {
                continue;
            };
            if let Op::Binary(_, left, right) = *op
                && (left != operand || !one(graph, right))
            {
                continue;
            }
            let bounds: Vec<Value> = relations
                .iter()
                .filter(|(target, value, ..)| *value == operand && cfg.dominates(*target, block))
                .filter(|(.., relation, bound)| {
                    let (low, high) = range(graph, &incoming, *bound, 0);
                    match (up, relation) {
                        (true, Relation::Less) | (false, Relation::Greater) => true,
                        (true, Relation::LessEqual) => high < i32::MAX as i64,
                        (false, Relation::GreaterEqual) => low > i32::MIN as i64,
                        _ => false,
                    }
                })
                .map(|(.., bound)| *bound)
                .collect();
            if !bounds.is_empty() {
                steps.insert(inst, bounds);
            }
        }
    }
    steps
}

fn range(
    graph: &Graph,
    incoming: &HashMap<Value, Vec<Value>>,
    value: Value,
    depth: usize,
) -> (i64, i64) {
    const FULL: (i64, i64) = (i32::MIN as i64, i32::MAX as i64);
    let constant = |value: Value| match graph.defining_op(value) {
        Some(Op::Constant(Constant::Number(n)))
            if n.fract() == 0.0 && *n >= i32::MIN as f64 && *n <= i32::MAX as f64 =>
        {
            Some(*n as i64)
        }
        _ => None,
    };
    if depth > 8 {
        return FULL;
    }
    if let Some(n) = constant(value) {
        return (n, n);
    }
    match graph.defining_op(value) {
        Some(&Op::Binary(BinaryOperator::BitwiseAnd, left, right)) => {
            match constant(left).or(constant(right)) {
                Some(mask) if mask >= 0 => (0, mask),
                _ => FULL,
            }
        }
        Some(&Op::Binary(BinaryOperator::ShiftRight, _, count)) => match constant(count) {
            Some(count) => (FULL.0 >> (count & 31), FULL.1 >> (count & 31)),
            None => FULL,
        },
        Some(_) => FULL,
        None => incoming.get(&value).map_or(FULL, |args| {
            args.iter()
                .map(|&arg| range(graph, incoming, arg, depth + 1))
                .reduce(|(low, high), (l, h)| (low.min(l), high.max(h)))
                .unwrap_or(FULL)
        }),
    }
}

fn number(value: f64) -> Types {
    let integral = value.fract() == 0.0 && value >= i32::MIN as f64 && value <= i32::MAX as f64;
    if integral && !(value == 0.0 && value.is_sign_negative()) {
        Types::INT32
    } else {
        Types::DOUBLE
    }
}

fn result_types(op: &Op, facts: &Facts) -> Types {
    match op {
        Op::OsrStart(..) => Types::INT32,
        Op::Continuation(body, _) => body.result,
        Op::ContinuationOutput(..) => Types::ALL,
        Op::ContinuationStore(..) | Op::Observe(..) | Op::ReadHint(..) => Types::default(),
        Op::Constant(constant) => match constant {
            Constant::Undefined => Types::UNDEFINED,
            Constant::Null => Types::NULL,
            Constant::Boolean(_) => Types::BOOLEAN,
            Constant::Number(value) => number(*value),
            Constant::String(_) => Types::STRING,
            Constant::Uninitialized => Types::UNINITIALIZED,
        },
        Op::NewObject
        | Op::NewInstance(_)
        | Op::ConstructorResult(..)
        | Op::NewArray(_)
        | Op::NewHoleyArray(_)
        | Op::IteratorNew(_)
        | Op::EnumerateNew(_) => Types::OBJECT,
        Op::MapElement(..) => Types::ALL,
        Op::IsArrayIteration(..) => Types::BOOLEAN,
        Op::IteratorNext(_) | Op::EnumerateNext(_) => Types::ALL,
        Op::IsUninitialized(_) | Op::IsPropertyCached(..) => Types::BOOLEAN,
        Op::ProbeProperty(..)
        | Op::ProbeIndex(..)
        | Op::ConstructorPrototype(_)
        | Op::CachedProperty(..)
        | Op::FillPropertyCache(..) => Types::ALL,
        Op::View(_) => Types::VIEW,
        Op::TypedIndex(.., read) if read.bounded => match read.element {
            Some(TypedKind::Float32 | TypedKind::Float64 | TypedKind::Uint32) | None => {
                Types::NUMBER
            }
            Some(_) => Types::INT32,
        },
        Op::TypedIndex(..) => Types::NUMBER.union(Types::UNDEFINED),
        Op::IsTyped(..) | Op::IsIndexed(_) | Op::IsArrayIndex(..) => Types::BOOLEAN,
        Op::IndexedLength(_) => Types::INT64,
        Op::IsNumber(_)
        | Op::IsInteger(_)
        | Op::IsString(_)
        | Op::IsStringSplit(_)
        | Op::IsTypeof(..) => Types::BOOLEAN,
        Op::NewPack | Op::Spread(_) | Op::StringSplitParts(..) => Types::PACK,
        Op::StringSplit(..) => Types::OBJECT,
        Op::ArrayPosition(_) => Types::INT64,
        Op::ToInt32(_) => Types::INT32,
        Op::AsInteger(_) => Types::INT64,
        Op::AsObject(_) => Types::OBJECT,
        Op::AsNumber(value) | Op::ToNumeric(value) | Op::Unary(UnaryOp::Plus, value)
            if facts.of(*value).within(Types::NUMBER) =>
        {
            facts.of(*value)
        }
        Op::AsNumber(_) | Op::ToNumeric(_) => Types::NUMBER,
        Op::Unary(UnaryOp::Increment | UnaryOp::Decrement, value)
            if facts.of(*value).within(Types::INTEGER) =>
        {
            Types::INT64
        }
        Op::Unary(operator, _) => match operator {
            UnaryOp::Not => Types::BOOLEAN,
            UnaryOp::Typeof => Types::STRING,
            UnaryOp::Plus => Types::NUMBER,
            UnaryOp::BitwiseNot => Types::INT32,
            UnaryOp::Negate | UnaryOp::Increment | UnaryOp::Decrement => Types::NUMBER,
        },
        Op::Binary(BinaryOperator::Addition | BinaryOperator::Subtraction, left, right)
            if facts.of(*left).within(Types::INTEGER)
                && facts.of(*right).within(Types::INTEGER) =>
        {
            Types::INT64
        }
        Op::Binary(operator, left, right) => match operator {
            BinaryOperator::Addition => {
                let (left, right) = (facts.of(*left), facts.of(*right));
                let nonstring = Types::NUMBER
                    .union(Types::BOOLEAN)
                    .union(Types::UNDEFINED)
                    .union(Types::NULL);
                if left.within(nonstring) && right.within(nonstring) {
                    Types::NUMBER
                } else if left.within(Types::STRING) || right.within(Types::STRING) {
                    Types::STRING
                } else {
                    Types::NUMBER.union(Types::STRING)
                }
            }
            BinaryOperator::Subtraction
            | BinaryOperator::Multiplication
            | BinaryOperator::Division
            | BinaryOperator::Remainder
            | BinaryOperator::Exponential => Types::NUMBER,
            BinaryOperator::ShiftLeft
            | BinaryOperator::ShiftRight
            | BinaryOperator::BitwiseOR
            | BinaryOperator::BitwiseXOR
            | BinaryOperator::BitwiseAnd => Types::INT32,
            BinaryOperator::ShiftRightZeroFill => Types::NUMBER,
            _ => Types::BOOLEAN,
        },
        Op::GuardedCompare { .. } => Types::BOOLEAN,
        Op::CallNative(..) => Types::NUMBER,
        Op::IsUndefined(_)
        | Op::IsNullish(_)
        | Op::IsNative(..)
        | Op::IsCode(..)
        | Op::GuardValid(_) => Types::BOOLEAN,
        _ => Types::VALUE,
    }
}

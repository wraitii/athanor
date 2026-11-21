use crate::{FxHashMap, FxHashSet};
use oxc_syntax::operator::BinaryOperator;

use super::facts::Relation;
use super::guards;
use super::low_level::{Binary, Op as LowOp};
use super::{
    Block, Constant, Facts, Graph, Inst, JsOp, Op, TerminatorKind, UnaryOp, Value, ValueDef,
    ValueType,
};

mod endpoints;
pub(super) use endpoints::LoopEndpoints;

const QUERY_LIMIT: usize = 512;
const DEPTH_LIMIT: usize = 24;
pub(super) const MAX_SAFE_INTEGER: i64 = (1 << 53) - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct IntRange {
    pub(super) min: i64,
    pub(super) max: i64,
}

impl IntRange {
    pub(super) const I32: Self = Self::new(i32::MIN as i64, i32::MAX as i64);
    pub(super) const SAFE: Self = Self::new(-MAX_SAFE_INTEGER, MAX_SAFE_INTEGER);
    const I64: Self = Self::new(i64::MIN, i64::MAX);

    pub(super) const fn new(min: i64, max: i64) -> Self {
        Self { min, max }
    }

    fn point(value: i64) -> Self {
        Self::new(value, value)
    }

    pub(super) fn within(self, other: Self) -> bool {
        self.min >= other.min && self.max <= other.max
    }

    fn join(self, other: Self) -> Self {
        Self::new(self.min.min(other.min), self.max.max(other.max))
    }

    fn intersect(self, other: Self) -> Option<Self> {
        let min = self.min.max(other.min);
        let max = self.max.min(other.max);
        (min <= max).then_some(Self::new(min, max))
    }

    pub(super) fn arithmetic(self, op: Binary, right: Self) -> Option<Self> {
        Some(match op {
            Binary::Add => Self::new(
                self.min.checked_add(right.min)?,
                self.max.checked_add(right.max)?,
            ),
            Binary::Subtract => Self::new(
                self.min.checked_sub(right.max)?,
                self.max.checked_sub(right.min)?,
            ),
            Binary::Multiply => {
                let products = [
                    self.min.checked_mul(right.min)?,
                    self.min.checked_mul(right.max)?,
                    self.max.checked_mul(right.min)?,
                    self.max.checked_mul(right.max)?,
                ];
                Self::new(*products.iter().min()?, *products.iter().max()?)
            }
            _ => return None,
        })
    }
}

#[derive(Clone, Copy)]
enum Constraint {
    Range(IntRange),
    Compare(Relation, Value),
}

pub(super) struct Ranges<'a> {
    graph: &'a Graph,
    facts: &'a Facts,
    assumed: &'a [(Value, IntRange)],
    constraints: Option<FxHashMap<Value, Vec<(Block, Constraint)>>>,
    cache: FxHashMap<(Value, Block), Option<IntRange>>,
    visiting: FxHashSet<(Value, Block)>,
    budget: usize,
}

impl<'a> Ranges<'a> {
    pub(super) fn new(
        graph: &'a Graph,
        facts: &'a Facts,
        assumed: &'a [(Value, IntRange)],
    ) -> Self {
        Self {
            graph,
            facts,
            assumed,
            constraints: None,
            cache: FxHashMap::default(),
            visiting: FxHashSet::default(),
            budget: QUERY_LIMIT,
        }
    }

    pub(super) fn integer_at(&mut self, value: Value, block: Block) -> Option<IntRange> {
        self.query(value, block, 0)
    }

    pub(super) fn prove_arithmetic(&mut self, value: Value, proved: &mut FxHashSet<Inst>) -> bool {
        self.prove_at_definition(value, proved, 0)
    }

    fn prove_at_definition(
        &mut self,
        value: Value,
        proved: &mut FxHashSet<Inst>,
        depth: usize,
    ) -> bool {
        let types = self.facts.of(value);
        if types != ValueType::default() && types.within(ValueType::INTEGER) {
            return true;
        }
        let ValueDef::Result(inst, _) = self.graph.values[value] else {
            return false;
        };
        if proved.contains(&inst) {
            return true;
        }
        let Op::Javascript(JsOp::Binary(op, left, right)) = self.graph.insts[inst].op else {
            return false;
        };
        if depth >= DEPTH_LIMIT
            || super::js::facts::integer_arithmetic(op).is_none()
            || self
                .integer_at(value, self.graph.insts[inst].block)
                .is_none()
            || !self.prove_at_definition(left, proved, depth + 1)
            || !self.prove_at_definition(right, proved, depth + 1)
        {
            return false;
        }
        proved.insert(inst);
        true
    }

    fn query(&mut self, value: Value, block: Block, depth: usize) -> Option<IntRange> {
        if let Some(value) = integer_literal(self.graph, value) {
            return Some(IntRange::point(value));
        }
        if let Some(range) = self.cache.get(&(value, block)) {
            return *range;
        }
        if depth >= DEPTH_LIMIT || self.budget == 0 || !self.visiting.insert((value, block)) {
            return None;
        }
        self.budget -= 1;
        let range = self.compute(value, block, depth + 1);
        self.visiting.remove(&(value, block));
        self.cache.insert((value, block), range);
        range
    }

    fn compute(&mut self, value: Value, block: Block, depth: usize) -> Option<IntRange> {
        let constraints = self.dominating(value, block);
        let assumed = self
            .assumed
            .iter()
            .find(|&&(input, _)| self.source(input) == self.source(value));
        let mut range = assumed.map(|&(_, range)| range);
        for &(_, constraint) in &constraints {
            if let Constraint::Range(proven) = constraint {
                range = Some(match range {
                    Some(range) => range.intersect(proven)?,
                    None => proven,
                });
            }
        }
        let mut range = match range {
            Some(range) => range,
            None => self.definition(value, block, depth)?,
        };
        for (_, constraint) in constraints {
            if let Constraint::Compare(relation, right) = constraint
                && let Some(bound) = self.query(right, block, depth)
            {
                range = range.intersect(match relation {
                    Relation::Less => IntRange::new(i64::MIN, bound.max.checked_sub(1)?),
                    Relation::LessEqual => IntRange::new(i64::MIN, bound.max),
                    Relation::Greater => IntRange::new(bound.min.checked_add(1)?, i64::MAX),
                    Relation::GreaterEqual => IntRange::new(bound.min, i64::MAX),
                })?;
            }
        }
        Some(range)
    }

    fn definition(&mut self, value: Value, block: Block, depth: usize) -> Option<IntRange> {
        match self.graph.defining_op(value) {
            Some(Op::Javascript(JsOp::AsInteger(inner))) => {
                return Some(self.query(*inner, block, depth).unwrap_or(IntRange::I64));
            }
            Some(Op::Javascript(JsOp::AsNumber(inner))) => {
                if let Some(range) = self.query(*inner, block, depth)
                    && range.within(IntRange::SAFE)
                {
                    return Some(range);
                }
            }
            Some(Op::Javascript(JsOp::Binary(operator, left, right))) => {
                let op = super::js::facts::integer_arithmetic(*operator);
                if let Some(op) = op
                    && let Some(left) = self.query(*left, block, depth)
                    && let Some(right) = self.query(*right, block, depth)
                {
                    if let Some(range) = super::js::facts::exact_integer_arithmetic(op, left, right)
                    {
                        return Some(range);
                    }
                }
            }
            Some(Op::LowLevel(LowOp::Binary { op, left, right })) if op.is_integer_arithmetic() => {
                let width = repr_range(self.graph, value)?;
                let range = self
                    .query(*left, block, depth)?
                    .arithmetic(*op, self.query(*right, block, depth)?)?;
                return range.within(width).then_some(range);
            }
            None if let ValueDef::Param(header, index) = self.graph.values[value] => {
                if let Some(range) = self.parameter(value, header, index, depth) {
                    return Some(range);
                }
            }
            _ => {}
        }
        let types = self.facts.of(value);
        if types == ValueType::default() {
            None
        } else if types.within(ValueType::INT32) {
            Some(IntRange::I32)
        } else if types.within(ValueType::INTEGER) {
            Some(IntRange::I64)
        } else {
            None
        }
    }

    fn parameter(
        &mut self,
        value: Value,
        header: Block,
        index: usize,
        depth: usize,
    ) -> Option<IntRange> {
        let mut initial: Option<IntRange> = None;
        let mut backedges = Vec::new();
        for &pred in &self.facts.cfg.preds[header] {
            let terminator = self.graph.blocks[pred].terminator.as_ref()?;
            for target in terminator
                .kind
                .targets()
                .filter(|target| target.block == header)
            {
                let arg = target.args[index];
                if self.facts.cfg.dominates(header, pred) {
                    backedges.push((pred, arg));
                } else {
                    let range = self.query(arg, pred, depth)?;
                    initial = Some(initial.map_or(range, |start| start.join(range)));
                }
            }
        }
        let mut range = initial?;
        for (pred, arg) in backedges {
            let (operand, step) = induction_step(self.graph, arg)?;
            if operand != value {
                return None;
            }
            let mut limit = None;
            for (_, constraint) in self.dominating(value, self.graph.def_block(arg)) {
                let Constraint::Compare(relation, other) = constraint else {
                    continue;
                };
                if !matches!(
                    (step, relation),
                    (1, Relation::Less | Relation::LessEqual)
                        | (-1, Relation::Greater | Relation::GreaterEqual)
                ) {
                    continue;
                }
                let Some(bound) = self.query(other, pred, depth) else {
                    continue;
                };
                let edge = match relation {
                    Relation::Less => bound.max,
                    Relation::LessEqual => bound.max.checked_add(1)?,
                    Relation::Greater => bound.min,
                    Relation::GreaterEqual => bound.min.checked_sub(1)?,
                };
                limit = Some(limit.map_or(edge, |old: i64| {
                    if step > 0 {
                        old.min(edge)
                    } else {
                        old.max(edge)
                    }
                }));
            }
            let limit = limit?;
            let bound = if step > 0 {
                IntRange::new(range.min, range.max.max(limit))
            } else {
                IntRange::new(range.min.min(limit), range.max)
            };
            let width = match self.graph.defining_op(arg) {
                Some(Op::LowLevel(_)) => {
                    repr_range(self.graph, arg)?.intersect(repr_range(self.graph, value)?)?
                }
                _ => IntRange::I32,
            };
            if !bound.within(IntRange::I32) || !bound.within(width) {
                return None;
            }
            range = bound;
        }
        Some(range)
    }

    fn source(&self, mut value: Value) -> Value {
        while let Some(Op::Javascript(JsOp::AsInteger(inner) | JsOp::AsNumber(inner))) =
            self.graph.defining_op(value)
        {
            value = *inner;
        }
        value
    }

    fn dominating(&mut self, value: Value, at: Block) -> Vec<(Block, Constraint)> {
        if self.constraints.is_none() {
            self.constraints = Some(self.branch_constraints());
        }
        let constraints = self.constraints.as_ref().unwrap();
        let source = self.source(value);
        constraints
            .get(&value)
            .into_iter()
            .flatten()
            .chain(
                constraints
                    .get(&source)
                    .filter(|_| source != value)
                    .into_iter()
                    .flatten()
                    .filter(|(_, constraint)| matches!(constraint, Constraint::Range(_))),
            )
            .copied()
            .filter(|&(block, _)| self.facts.cfg.dominates(block, at))
            .collect()
    }

    fn branch_constraints(&self) -> FxHashMap<Value, Vec<(Block, Constraint)>> {
        let mut constraints = FxHashMap::<_, Vec<_>>::default();
        let cfg = &self.facts.cfg;
        for &block in &cfg.rpo {
            let Some(TerminatorKind::Branch(condition, yes, no)) = self.graph.blocks[block]
                .terminator
                .as_ref()
                .map(|term| &term.kind)
            else {
                continue;
            };
            if yes.block == no.block {
                continue;
            }
            let (condition, yes, no) = match guards::negated(self.graph, *condition) {
                Some(value) => (value, no.block, yes.block),
                None => (*condition, yes.block, no.block),
            };
            let Some(op) = self.graph.defining_op(condition) else {
                continue;
            };
            if let Some((value, guards::Fact::IntegerRange(min, max))) = guards::implies(op)
                && yes != self.graph.entry
                && cfg.preds[yes].len() == 1
            {
                constraints.entry(self.source(value)).or_default().push((
                    yes,
                    Constraint::Range(IntRange::new(min.into(), max.into())),
                ));
            }
            if let Some((relation, left, right)) = Relation::comparison(op) {
                for (target, relation) in [(yes, relation), (no, relation.negate())] {
                    if target != self.graph.entry && cfg.preds[target].len() == 1 {
                        constraints
                            .entry(left)
                            .or_default()
                            .push((target, Constraint::Compare(relation, right)));
                        constraints
                            .entry(right)
                            .or_default()
                            .push((target, Constraint::Compare(relation.swap(), left)));
                    }
                }
            }
        }
        constraints
    }
}

pub(super) fn integer_literal(graph: &Graph, value: Value) -> Option<i64> {
    match graph.defining_op(value)? {
        Op::Javascript(JsOp::Constant(Constant::Number(n)))
            if n.fract() == 0.0
                && n.abs() <= MAX_SAFE_INTEGER as f64
                && (*n != 0.0 || !n.is_sign_negative()) =>
        {
            Some(*n as i64)
        }
        Op::LowLevel(LowOp::Integer { ty, bits }) if ty.is_int() => {
            let shift = 64 - ty.bits();
            Some(((*bits << shift) as i64) >> shift)
        }
        _ => None,
    }
}

fn repr_range(graph: &Graph, value: Value) -> Option<IntRange> {
    use super::Repr;
    match graph.reprs[value] {
        Some(Repr::I8) => Some(IntRange::new(i8::MIN.into(), i8::MAX.into())),
        Some(Repr::I16) => Some(IntRange::new(i16::MIN.into(), i16::MAX.into())),
        Some(Repr::I32) => Some(IntRange::I32),
        None | Some(Repr::I64 | Repr::INTEGER) => Some(IntRange::I64),
        _ => None,
    }
}

pub(super) fn induction_step(graph: &Graph, value: Value) -> Option<(Value, i64)> {
    Some(match *graph.defining_op(value)? {
        Op::Javascript(JsOp::Unary(UnaryOp::Increment, operand)) => (operand, 1),
        Op::Javascript(JsOp::Unary(UnaryOp::Decrement, operand)) => (operand, -1),
        Op::Javascript(JsOp::Binary(BinaryOperator::Addition, operand, one))
        | Op::LowLevel(LowOp::Binary {
            op: Binary::Add,
            left: operand,
            right: one,
        }) if integer_literal(graph, one) == Some(1) => (operand, 1),
        Op::Javascript(JsOp::Binary(BinaryOperator::Subtraction, operand, one))
        | Op::LowLevel(LowOp::Binary {
            op: Binary::Subtract,
            left: operand,
            right: one,
        }) if integer_literal(graph, one) == Some(1) => (operand, -1),
        _ => return None,
    })
}

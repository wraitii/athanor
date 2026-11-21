use super::super::super::low_level::Binary;
use super::super::super::ranges::IntRange;
use super::*;
use crate::jit::semantic::JsOp;
use cranelift_codegen::entity::packed_option::ReservedValue;

const DEPTH: usize = 12;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::jit::semantic) enum Interval {
    Constant(i32),
    Invariant(Value, Option<IntRange>),
    Counter {
        start: Box<Interval>,
        bound: Box<Interval>,
        inclusive: bool,
    },
    Add(Box<Interval>, Box<Interval>),
    Subtract(Box<Interval>, Box<Interval>),
    Scale(Box<Interval>, Box<Interval>),
}

pub(super) struct Counters(FxHashMap<Value, CountedLoop>);

impl Counters {
    pub(super) fn discover(graph: &Graph, cfg: &Cfg) -> Self {
        Self(
            counted_loops(graph, cfg)
                .into_iter()
                .map(|counted| (counted.counter, counted))
                .collect(),
        )
    }
}

impl Interval {
    pub(super) fn of(
        graph: &Graph,
        cfg: &Cfg,
        counters: &Counters,
        block: Block,
        value: Value,
        invariant: &mut dyn FnMut(Value) -> Option<Option<IntRange>>,
    ) -> Option<Self> {
        Self::at(graph, cfg, counters, block, value, invariant, 0)
    }

    fn at(
        graph: &Graph,
        cfg: &Cfg,
        counters: &Counters,
        block: Block,
        value: Value,
        invariant: &mut dyn FnMut(Value) -> Option<Option<IntRange>>,
        depth: usize,
    ) -> Option<Self> {
        if depth > DEPTH {
            return None;
        }
        let mut of = |value| Self::at(graph, cfg, counters, block, value, invariant, depth + 1);
        if let Some(counted) = counters.0.get(&value)
            && cfg.dominates(counted.inside, block)
        {
            return Some(Self::Counter {
                start: Box::new(of(counted.start)?),
                bound: Box::new(of(counted.bound)?),
                inclusive: counted.inclusive,
            });
        }
        match graph.defining_op(value) {
            Some(Op::Javascript(JsOp::Constant(Constant::Number(number))))
                if number.fract() == 0.0 && number.abs() <= f64::from(i32::MAX) =>
            {
                return Some(Self::Constant(*number as i32));
            }
            Some(&Op::Javascript(JsOp::Binary(operator, left, right))) => {
                let combined = match operator {
                    BinaryOperator::Addition => of(left)
                        .zip(of(right))
                        .map(|(left, right)| Self::Add(Box::new(left), Box::new(right))),
                    BinaryOperator::Subtraction => of(left)
                        .zip(of(right))
                        .map(|(left, right)| Self::Subtract(Box::new(left), Box::new(right))),
                    BinaryOperator::Multiplication => match (of(left), of(right)) {
                        (Some(varying), Some(point)) | (Some(point), Some(varying))
                            if point.is_point() =>
                        {
                            Some(Self::Scale(Box::new(varying), Box::new(point)))
                        }
                        _ => None,
                    },
                    _ => None,
                };
                if combined.is_some() {
                    return combined;
                }
            }
            _ => {}
        }
        invariant(value).map(|range| Self::Invariant(value, range))
    }

    fn is_point(&self) -> bool {
        match self {
            Self::Constant(_) | Self::Invariant(..) => true,
            Self::Counter { .. } => false,
            Self::Add(left, right) | Self::Subtract(left, right) | Self::Scale(left, right) => {
                left.is_point() && right.is_point()
            }
        }
    }

    /// The range of the smallest value, as far as it is known without evaluating the expression.
    pub(super) fn low(&self) -> IntRange {
        let values = FxHashMap::default();
        let mut emitter = Emitter {
            target: None,
            values: &values,
            checks: Vec::new(),
        };
        emitter.emit(self).0.1
    }

    /// Appends a check to `block` that every value the expression takes is an int32 integer
    /// and the smallest one is non-negative, branching to `fallback` otherwise, and returns the
    /// continuation block and one more than the largest value.
    pub(super) fn guard(
        &self,
        graph: &mut Graph,
        block: Block,
        values: &FxHashMap<Value, Value>,
        fallback: &Target,
    ) -> (Block, Value) {
        let mut emitter = Emitter {
            target: Some((graph, block)),
            values,
            checks: Vec::new(),
        };
        let (low, high) = emitter.emit(self);
        if low.1.min < 0 {
            emitter.check(low.0, 0);
        }
        let one = emitter.constant(1);
        let (limit, _) = emitter.binary(Binary::Add, high, one);
        let Emitter {
            target: Some((graph, block)),
            checks,
            ..
        } = emitter
        else {
            unreachable!()
        };
        let span = graph.span;
        let Some(all) = checks
            .into_iter()
            .reduce(|all, check| graph.append_value(block, span, Op::BooleanAnd(all, check)))
        else {
            return (block, limit);
        };
        let next = graph.blocks.push(BlockData::default());
        let kind = TerminatorKind::Branch(all, Target::new(next), fallback.clone());
        graph.terminate(block, span, kind);
        (next, limit)
    }
}

type Bound = (Value, IntRange);

struct Emitter<'a> {
    target: Option<(&'a mut Graph, Block)>,
    values: &'a FxHashMap<Value, Value>,
    checks: Vec<Value>,
}

impl Emitter<'_> {
    fn append(&mut self, op: JsOp) -> Value {
        let Some((graph, block)) = &mut self.target else {
            return Value::reserved_value();
        };
        let span = graph.span;
        graph.append_value(*block, span, Op::Javascript(op))
    }

    fn constant(&mut self, value: i32) -> Bound {
        let constant = self.append(JsOp::Constant(Constant::Number(f64::from(value))));
        (constant, IntRange::new(value.into(), value.into()))
    }

    fn check(&mut self, value: Value, min: i32) {
        let check = self.append(JsOp::IsIntegerInRange(value, min, i32::MAX));
        self.checks.push(check);
    }

    // Keeping every endpoint within int32 keeps all values the loop computes there too, so
    // its f64 arithmetic on them is exact and the endpoints bound it.
    fn within_int32(&mut self, value: Value, range: Option<IntRange>) -> Bound {
        if let Some(range) = range
            && range.within(IntRange::I32)
        {
            return (value, range);
        }
        self.check(value, i32::MIN);
        (value, IntRange::I32)
    }

    fn binary(
        &mut self,
        op: Binary,
        (left, left_range): Bound,
        (right, right_range): Bound,
    ) -> Bound {
        let operator = match op {
            Binary::Add => BinaryOperator::Addition,
            Binary::Subtract => BinaryOperator::Subtraction,
            _ => BinaryOperator::Multiplication,
        };
        let value = self.append(JsOp::Binary(operator, left, right));
        self.within_int32(value, left_range.arithmetic(op, right_range))
    }

    fn emit(&mut self, interval: &Interval) -> (Bound, Bound) {
        match interval {
            &Interval::Constant(value) => {
                let value = self.constant(value);
                (value, value)
            }
            &Interval::Invariant(value, range) => {
                let value = self.values.get(&value).copied().unwrap_or(value);
                let value = self.append(JsOp::AsNumber(value));
                let value = self.within_int32(value, range);
                (value, value)
            }
            Interval::Counter {
                start,
                bound,
                inclusive,
            } => {
                let (low, _) = self.emit(start);
                let (_, bound) = self.emit(bound);
                if *inclusive {
                    return (low, bound);
                }
                let one = self.constant(1);
                let high = self.binary(Binary::Subtract, bound, one);
                (low, high)
            }
            Interval::Add(left, right) => {
                let (left_low, left_high) = self.emit(left);
                let (right_low, right_high) = self.emit(right);
                (
                    self.binary(Binary::Add, left_low, right_low),
                    self.binary(Binary::Add, left_high, right_high),
                )
            }
            Interval::Subtract(left, right) => {
                let (left_low, left_high) = self.emit(left);
                let (right_low, right_high) = self.emit(right);
                (
                    self.binary(Binary::Subtract, left_low, right_high),
                    self.binary(Binary::Subtract, left_high, right_low),
                )
            }
            Interval::Scale(varying, point) => {
                let (low, high) = self.emit(varying);
                let (factor, _) = self.emit(point);
                let negative = factor.1.max < 0;
                if !negative && factor.1.min < 0 {
                    self.check(factor.0, 0);
                }
                let low = self.binary(Binary::Multiply, low, factor);
                let high = self.binary(Binary::Multiply, high, factor);
                if negative { (high, low) } else { (low, high) }
            }
        }
    }
}

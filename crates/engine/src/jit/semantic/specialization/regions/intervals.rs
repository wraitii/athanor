use super::*;
use crate::jit::semantic::JsOp;

const DEPTH: usize = 12;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::jit::semantic) enum Interval {
    Constant(i32),
    Invariant(Value),
    Counter { start: i32, bound: Box<Interval> },
    Add(Box<Interval>, Box<Interval>),
    Subtract(Box<Interval>, Box<Interval>),
    Scale(Box<Interval>, Box<Interval>),
}

pub(super) struct Counters(FxHashMap<Value, (Block, i32, Value)>);

impl Counters {
    pub(super) fn discover(graph: &Graph, cfg: &Cfg) -> Self {
        Self(
            counted_loops(graph, cfg)
                .into_iter()
                .map(|counted| {
                    (
                        counted.counter,
                        (counted.inside, counted.start, counted.bound),
                    )
                })
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
        invariant: &dyn Fn(Value) -> bool,
    ) -> Option<Self> {
        Self::at(graph, cfg, counters, block, value, invariant, 0)
    }

    fn at(
        graph: &Graph,
        cfg: &Cfg,
        counters: &Counters,
        block: Block,
        value: Value,
        invariant: &dyn Fn(Value) -> bool,
        depth: usize,
    ) -> Option<Self> {
        if depth > DEPTH {
            return None;
        }
        let of = |value| Self::at(graph, cfg, counters, block, value, invariant, depth + 1);
        if let Some(&(inside, start, bound)) = counters.0.get(&value)
            && cfg.dominates(inside, block)
        {
            let bound = Box::new(of(bound)?);
            return Some(Self::Counter { start, bound });
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
        invariant(value).then_some(Self::Invariant(value))
    }

    fn is_point(&self) -> bool {
        match self {
            Self::Constant(_) | Self::Invariant(_) => true,
            Self::Counter { .. } => false,
            Self::Add(left, right) | Self::Subtract(left, right) | Self::Scale(left, right) => {
                left.is_point() && right.is_point()
            }
        }
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
            graph,
            block,
            values,
            checks: Vec::new(),
        };
        let (low, high) = emitter.emit(self);
        emitter.check(low, 0);
        let one = emitter.constant(1);
        let limit = emitter.binary(BinaryOperator::Addition, high, one);
        let Emitter {
            graph,
            block,
            checks,
            ..
        } = emitter;
        let span = graph.span;
        let all = checks
            .into_iter()
            .reduce(|all, check| graph.append_value(block, span, Op::BooleanAnd(all, check)))
            .unwrap();
        let next = graph.blocks.push(BlockData::default());
        let kind = TerminatorKind::Branch(all, Target::new(next), fallback.clone());
        graph.terminate(block, span, kind);
        (next, limit)
    }
}

struct Emitter<'a> {
    graph: &'a mut Graph,
    block: Block,
    values: &'a FxHashMap<Value, Value>,
    checks: Vec<Value>,
}

impl Emitter<'_> {
    fn append(&mut self, op: JsOp) -> Value {
        let span = self.graph.span;
        self.graph
            .append_value(self.block, span, Op::Javascript(op))
    }

    fn constant(&mut self, value: i32) -> Value {
        self.append(JsOp::Constant(Constant::Number(f64::from(value))))
    }

    fn check(&mut self, value: Value, min: i32) {
        let check = self.append(JsOp::IsIntegerInRange(value, min, i32::MAX));
        self.checks.push(check);
    }

    // Checking every endpoint keeps all values the loop computes within int32, so its f64
    // arithmetic on them is exact and the endpoints bound it.
    fn binary(&mut self, operator: BinaryOperator, left: Value, right: Value) -> Value {
        let value = self.append(JsOp::Binary(operator, left, right));
        self.check(value, i32::MIN);
        value
    }

    fn emit(&mut self, interval: &Interval) -> (Value, Value) {
        match interval {
            &Interval::Constant(value) => {
                let value = self.constant(value);
                (value, value)
            }
            &Interval::Invariant(value) => {
                let value = self.values.get(&value).copied().unwrap_or(value);
                let value = self.append(JsOp::AsNumber(value));
                self.check(value, i32::MIN);
                (value, value)
            }
            Interval::Counter { start, bound } => {
                let low = self.constant(*start);
                let (_, bound) = self.emit(bound);
                let one = self.constant(1);
                let high = self.binary(BinaryOperator::Subtraction, bound, one);
                (low, high)
            }
            Interval::Add(left, right) => {
                let (left_low, left_high) = self.emit(left);
                let (right_low, right_high) = self.emit(right);
                (
                    self.binary(BinaryOperator::Addition, left_low, right_low),
                    self.binary(BinaryOperator::Addition, left_high, right_high),
                )
            }
            Interval::Subtract(left, right) => {
                let (left_low, left_high) = self.emit(left);
                let (right_low, right_high) = self.emit(right);
                (
                    self.binary(BinaryOperator::Subtraction, left_low, right_high),
                    self.binary(BinaryOperator::Subtraction, left_high, right_low),
                )
            }
            Interval::Scale(varying, point) => {
                let (low, high) = self.emit(varying);
                let (factor, _) = self.emit(point);
                let negative = match **point {
                    Interval::Constant(factor) => factor < 0,
                    _ => {
                        self.check(factor, 0);
                        false
                    }
                };
                let low = self.binary(BinaryOperator::Multiplication, low, factor);
                let high = self.binary(BinaryOperator::Multiplication, high, factor);
                if negative { (high, low) } else { (low, high) }
            }
        }
    }
}

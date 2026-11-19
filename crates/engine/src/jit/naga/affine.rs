use std::collections::BTreeMap;

use oxc_syntax::operator::BinaryOperator;

use super::kernel::{Scale, StrideCheck};
use crate::jit::semantic::low_level;
use crate::jit::semantic::{Constant, Graph, JsOp, Op, Value};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Atom {
    One,
    Arg(usize),
    Axis(usize),
    ScaledAxis(usize, usize),
}

#[derive(Clone, Copy)]
enum Arith {
    Add,
    Subtract,
    Multiply,
}

#[derive(Clone, PartialEq, Eq)]
pub(super) struct Affine(BTreeMap<Atom, i64>);

impl Affine {
    fn atom(atom: Atom, factor: i64) -> Self {
        Self(BTreeMap::from([(atom, factor)]))
    }

    pub(super) fn of(graph: &Graph, value: Value, axes: &[Value]) -> Option<Self> {
        if let Some(axis) = axes.iter().position(|&iv| iv == value) {
            return Some(Self::atom(Atom::Axis(axis), 1));
        }
        let (arith, left, right) = match graph.defining_op(value)? {
            Op::Argument(index) => return Some(Self::atom(Atom::Arg(*index), 1)),
            Op::Javascript(JsOp::Constant(Constant::Number(n)))
                if n.fract() == 0.0 && n.abs() <= f64::from(i32::MAX) =>
            {
                return Some(Self::atom(Atom::One, *n as i64));
            }
            Op::Javascript(JsOp::AsInteger(inner)) => return Self::of(graph, *inner, axes),
            Op::Javascript(JsOp::Binary(op, left, right)) => {
                let arith = match op {
                    BinaryOperator::Addition => Arith::Add,
                    BinaryOperator::Subtraction => Arith::Subtract,
                    BinaryOperator::Multiplication => Arith::Multiply,
                    _ => return None,
                };
                (arith, *left, *right)
            }
            Op::LowLevel(low_level::Op::Binary { op, left, right }) => {
                let arith = match op {
                    low_level::Binary::Add => Arith::Add,
                    low_level::Binary::Subtract => Arith::Subtract,
                    low_level::Binary::Multiply => Arith::Multiply,
                    _ => return None,
                };
                (arith, *left, *right)
            }
            _ => return None,
        };
        let left = Self::of(graph, left, axes)?;
        let right = Self::of(graph, right, axes)?;
        match arith {
            Arith::Add => left.add(&right, 1),
            Arith::Subtract => left.add(&right, -1),
            Arith::Multiply => left.multiply(&right),
        }
    }

    fn add(mut self, other: &Self, sign: i64) -> Option<Self> {
        for (&atom, &factor) in &other.0 {
            let sum = self.0.get(&atom).copied().unwrap_or(0).checked_add(factor.checked_mul(sign)?)?;
            if sum == 0 {
                self.0.remove(&atom);
            } else {
                self.0.insert(atom, sum);
            }
        }
        Some(self)
    }

    fn multiply(&self, other: &Self) -> Option<Self> {
        let mut product = Self(BTreeMap::new());
        for (&a, &fa) in &self.0 {
            for (&b, &fb) in &other.0 {
                let atom = match (a, b) {
                    (Atom::One, atom) | (atom, Atom::One) => atom,
                    (Atom::Arg(arg), Atom::Axis(axis)) | (Atom::Axis(axis), Atom::Arg(arg)) => {
                        Atom::ScaledAxis(axis, arg)
                    }
                    _ => return None,
                };
                product = product.add(&Self::atom(atom, fa.checked_mul(fb)?), 1)?;
            }
        }
        Some(product)
    }

    fn scale(&self, axis: usize) -> Result<Scale, String> {
        let mut terms = self.0.iter().filter(|(atom, _)| match atom {
            Atom::Axis(a) | Atom::ScaledAxis(a, _) => *a == axis,
            _ => false,
        });
        let scale = match terms.next() {
            None => return Err("store index does not cover the grid".into()),
            Some((Atom::ScaledAxis(_, arg), &factor)) => Scale::Arg(*arg, factor),
            Some((_, &factor)) => Scale::Const(factor),
        };
        if terms.next().is_some() {
            return Err("store index stride is not a single term".into());
        }
        Ok(scale)
    }

    pub(super) fn injectivity(&self, axes: usize) -> Result<Option<StrideCheck>, String> {
        if axes == 1 {
            return match self.scale(0)? {
                Scale::Const(_) => Ok(None),
                Scale::Arg(..) => Err("1D store index has a runtime stride".into()),
            };
        }
        let (inner_axis, inner, stride) = match (self.scale(0)?, self.scale(1)?) {
            (Scale::Const(x), Scale::Const(y)) if y.unsigned_abs() < x.unsigned_abs() => (1, y, Scale::Const(x)),
            (Scale::Const(x), stride) => (0, x, stride),
            (stride, Scale::Const(y)) => (1, y, stride),
            _ => return Err("store index has no unit axis".into()),
        };
        Ok(Some(StrideCheck {
            stride,
            inner: inner.unsigned_abs(),
            inner_axis,
        }))
    }
}

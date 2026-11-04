use std::fmt;

use cranelift_codegen::entity::SecondaryMap;
use oxc_syntax::operator::BinaryOperator;

use super::{Cfg, Constant, Graph, Op, UnaryOp, Value};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::jit) struct Types(u16);

impl Types {
    pub(in crate::jit) const INT32: Self = Self(1);
    pub(in crate::jit) const DOUBLE: Self = Self(2);
    pub(in crate::jit) const BOOLEAN: Self = Self(4);
    pub(in crate::jit) const UNDEFINED: Self = Self(8);
    pub(in crate::jit) const NULL: Self = Self(16);
    pub(in crate::jit) const STRING: Self = Self(32);
    pub(in crate::jit) const OBJECT: Self = Self(64);
    pub(in crate::jit) const OTHER: Self = Self(128);
    pub(in crate::jit) const UNINITIALIZED: Self = Self(256);
    pub(in crate::jit) const NUMBER: Self = Self(1 | 2);
    pub(in crate::jit) const PRIMITIVE: Self = Self(1 | 2 | 4 | 8 | 16 | 32);
    pub(in crate::jit) const VALUE: Self = Self(511 & !256);
    pub(in crate::jit) const ALL: Self = Self(511);

    pub(in crate::jit) const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub(in crate::jit) fn within(self, other: Self) -> bool {
        self.0 & !other.0 == 0
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
        } else if self.intersects(Self::INT32) {
            names.push("int32");
        } else if self.intersects(Self::DOUBLE) {
            names.push("double");
        }
        for (types, name) in [
            (Self::BOOLEAN, "bool"),
            (Self::UNDEFINED, "undefined"),
            (Self::NULL, "null"),
            (Self::STRING, "string"),
            (Self::OBJECT, "object"),
            (Self::OTHER, "other"),
            (Self::UNINITIALIZED, "uninit"),
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
    known: bool,
}

impl Facts {
    pub(in crate::jit) fn of(&self, value: Value) -> Types {
        if self.known {
            self.types[value]
        } else {
            Types::ALL
        }
    }

    pub(in crate::jit) fn compute(graph: &Graph) -> Self {
        let cfg = Cfg::new(graph);
        let mut facts = Self {
            types: SecondaryMap::new(),
            known: true,
        };
        let mut changed = true;
        while changed {
            changed = false;
            for &block in &cfg.rpo {
                for &inst in &graph.blocks[block].insts {
                    let data = &graph.insts[inst];
                    if let Some(result) = data.result {
                        changed |= facts.widen(result, result_types(&data.op, &facts));
                    }
                }
                let Some(terminator) = &graph.blocks[block].terminator else {
                    continue;
                };
                for target in terminator.kind.targets() {
                    for (&param, &arg) in graph.blocks[target.block].params.iter().zip(&target.args)
                    {
                        changed |= facts.widen(param, facts.of(arg));
                    }
                }
            }
        }
        facts
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
        Op::Constant(constant) => match constant {
            Constant::Undefined => Types::UNDEFINED,
            Constant::Null => Types::NULL,
            Constant::Boolean(_) => Types::BOOLEAN,
            Constant::Number(value) => number(*value),
            Constant::String(_) => Types::STRING,
            Constant::Uninitialized => Types::UNINITIALIZED,
        },
        Op::NewObject | Op::NewArray(_) | Op::IteratorNew(_) | Op::EnumerateNew(_) => Types::OBJECT,
        Op::IteratorNext(_) | Op::EnumerateNext(_) => Types::ALL,
        Op::IsUninitialized(_) => Types::BOOLEAN,
        Op::ToNumeric(value) | Op::Unary(UnaryOp::Plus, value)
            if facts.of(*value).within(Types::NUMBER) =>
        {
            facts.of(*value)
        }
        Op::ToNumeric(_) => Types::NUMBER,
        Op::Unary(operator, _) => match operator {
            UnaryOp::Not => Types::BOOLEAN,
            UnaryOp::Typeof => Types::STRING,
            UnaryOp::Plus => Types::NUMBER,
            UnaryOp::BitwiseNot => Types::INT32,
            UnaryOp::Negate | UnaryOp::Increment | UnaryOp::Decrement => Types::NUMBER,
        },
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
        Op::CallNative(..) => Types::NUMBER,
        Op::IsUndefined(_) | Op::IsNullish(_) | Op::IsNative(..) => Types::BOOLEAN,
        _ => Types::VALUE,
    }
}

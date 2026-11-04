mod build;
pub(in crate::jit) mod facts;
mod optimize;
mod promote;

use std::collections::{HashMap, HashSet};
use std::fmt;

use cranelift_codegen::entity::{EntityRef, PrimaryMap, SecondaryMap};
use oxc_ast::ast::{Expression, FormalParameters, Statement};
use oxc_semantic::Scoping;
use oxc_span::Span;
use oxc_syntax::operator::BinaryOperator;
use oxc_syntax::symbol::SymbolId;

use super::effects::{
    ALLOCATION, Assumption, ERROR, GLOBAL_READ, GLOBAL_WRITE, HEAP_READ, HEAP_WRITE, INTERNAL,
    Reentry, UNKNOWN,
};
use super::math::METHODS;

pub(super) use facts::{Facts, Types};
pub(super) use ids::{Binding, Block, Inst, Scope, Value};

#[allow(deprecated)]
mod ids {
    use cranelift_codegen::entity::entity_impl;

    #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub(in crate::jit) struct Value(u32);
    entity_impl!(Value, "v");

    #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub(in crate::jit) struct Block(u32);
    entity_impl!(Block, "block");

    #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub(in crate::jit) struct Inst(u32);
    entity_impl!(Inst, "inst");

    #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub(in crate::jit) struct Binding(u32);
    entity_impl!(Binding, "b");

    #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub(in crate::jit) struct Scope(u32);
    entity_impl!(Scope, "s");
}

#[derive(Clone)]
pub(super) struct Graph {
    pub(super) name: String,
    pub(super) span: Span,
    pub(super) entry: Block,
    pub(super) has_handlers: bool,
    pub(in crate::jit) guards: usize,
    pub(in crate::jit) views: HashMap<Inst, Value>,
    pub(in crate::jit) cold: HashSet<Block>,
    pub(super) generator: bool,
    pub(super) blocks: PrimaryMap<Block, BlockData>,
    pub(super) insts: PrimaryMap<Inst, InstData>,
    pub(super) values: PrimaryMap<Value, ValueDef>,
    pub(super) bindings: PrimaryMap<Binding, BindingData>,
    pub(super) scopes: PrimaryMap<Scope, ScopeData>,
}

#[derive(Clone, Default)]
pub(super) struct BlockData {
    pub(super) params: Vec<Value>,
    pub(super) insts: Vec<Inst>,
    pub(super) terminator: Option<Terminator>,
}

#[derive(Clone)]
pub(super) struct InstData {
    pub(super) span: Span,
    pub(super) block: Block,
    pub(super) result: Option<Value>,
    pub(super) op: Op,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ValueDef {
    Result(Inst),
    Param(Block, usize),
}

#[derive(Clone)]
pub(super) struct BindingData {
    pub(super) name: String,
    pub(super) kind: BindingKind,
    pub(super) scope: Scope,
    pub(super) captured: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BindingKind {
    Parameter,
    Var,
    Let,
    Const,
}

#[derive(Clone)]
pub(super) struct ScopeData {
    pub(super) kind: ScopeKind,
    pub(super) parent: Option<Scope>,
    pub(super) bindings: Vec<Binding>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ScopeKind {
    Captures,
    Parameters,
    Function,
    Body,
    Block,
    Loop,
}

#[derive(Clone, Debug)]
pub(super) enum Constant {
    Undefined,
    Null,
    Boolean(bool),
    Number(f64),
    String(String),
    Uninitialized,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum UnaryOp {
    Negate,
    Plus,
    Not,
    BitwiseNot,
    Typeof,
    Increment,
    Decrement,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Native(usize);

impl Native {
    pub(in crate::jit) fn index(self) -> usize {
        self.0
    }

    pub(super) fn lookup(global: &str, property: &str) -> Option<Self> {
        if global != "Math" {
            return None;
        }
        METHODS
            .iter()
            .position(|(name, _)| *name == property)
            .map(Self)
    }
}

impl fmt::Display for Native {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Math.{}", METHODS[self.0].0)
    }
}

#[derive(Clone, Debug)]
pub(super) enum Op {
    Argument(usize),
    Arguments,
    Rest(usize),
    This,
    Throw(Value),
    Catch,
    Closure(Span, bool, bool, Vec<Binding>),
    Constant(Constant),
    GlobalRead(String),
    GlobalReadOptional(String),
    RegExp(String, u8),
    PropertyKey(Value),
    GlobalWrite(String, Value),
    GlobalInitialize(String, Value),
    Read(Binding),
    Initialize(Binding, Value),
    Write(Binding, Value),
    CheckInitialized(Value),
    NewObject,
    NewArray(Vec<Value>),
    ArrayAppend(Value, Value, bool),
    CallSpread(Value, Value, Value),
    ConstructSpread(Value, Value),
    IteratorNew(Value),
    IteratorNext(Value),
    EnumerateNew(Value),
    EnumerateNext(Value),
    IsUninitialized(Value),
    Define(Value, String, Value),
    DefineHidden(Value, String, Value),
    MarkClass(Value),
    DisableConstructor(Value),
    Get(Value, String),
    GetIndex(Value, Value),
    Set(Value, String, Value),
    SetIndex(Value, Value, Value),
    ToNumeric(Value),
    Unary(UnaryOp, Value),
    Binary(BinaryOperator, Value, Value),
    Call {
        callee: Value,
        receiver: Option<Value>,
        args: Vec<Value>,
    },
    Construct {
        callee: Value,
        args: Vec<Value>,
    },
    CallNative(Native, Vec<Value>),
    IsUndefined(Value),
    IsNullish(Value),
    ProbeGlobal(String),
    ProbeProperty(Value, String),
    ProbeIndex(Value, Value),
    IsNative(Value, Native),
    GuardValid(usize),
    GuardCommit(usize),
    View(Value),
    IsTyped(Value),
    TypedIndex(Value, Value),
    IsNumber(Value),
    AsNumber(Value),
    EnterScope(Scope),
    LeaveScope(Scope),
    NextIteration(Scope),
    FreshIteration(Scope),
}

#[derive(Clone)]
pub(super) struct Target {
    pub(super) block: Block,
    pub(super) args: Vec<Value>,
}

#[derive(Clone)]
pub(super) enum TerminatorKind {
    Jump(Target),
    CheckException(Target, Target),
    Yield(Span, Value, Target),
    Branch(Value, Target, Target),
    Return(Value),
}

#[derive(Clone)]
pub(super) struct Terminator {
    pub(super) span: Span,
    pub(super) kind: TerminatorKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Ownership {
    None,
    Unowned,
    Borrowed,
    Owned,
    MaybeOwned,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Contract {
    pub(super) reentry: Reentry,
    pub(super) flags: u16,
    pub(super) assumptions: &'static [Assumption],
    pub(super) result: Ownership,
}

impl Contract {
    pub(super) fn may_throw(&self) -> bool {
        self.flags & ERROR != 0
    }

    pub(super) fn is_pure(&self) -> bool {
        self.reentry == Reentry::Closed && self.flags == 0
    }
}

impl Target {
    pub(super) fn new(block: Block) -> Self {
        Self {
            block,
            args: Vec::new(),
        }
    }
}

impl Op {
    pub(super) fn operands(&self) -> Vec<Value> {
        let mut op = self.clone();
        op.operands_mut().into_iter().map(|value| *value).collect()
    }

    pub(super) fn operands_mut(&mut self) -> Vec<&mut Value> {
        match self {
            Op::Argument(_)
            | Op::Arguments
            | Op::Rest(_)
            | Op::This
            | Op::Catch
            | Op::Closure(..)
            | Op::Constant(_)
            | Op::GlobalRead(_)
            | Op::GlobalReadOptional(_)
            | Op::RegExp(_, _)
            | Op::Read(_)
            | Op::NewObject
            | Op::ProbeGlobal(_)
            | Op::GuardValid(_)
            | Op::GuardCommit(_)
            | Op::EnterScope(_)
            | Op::LeaveScope(_)
            | Op::NextIteration(_)
            | Op::FreshIteration(_) => Vec::new(),
            Op::Throw(value)
            | Op::GlobalInitialize(_, value)
            | Op::GlobalWrite(_, value)
            | Op::Initialize(_, value)
            | Op::Write(_, value)
            | Op::CheckInitialized(value)
            | Op::IteratorNew(value)
            | Op::IteratorNext(value)
            | Op::EnumerateNew(value)
            | Op::EnumerateNext(value)
            | Op::IsUninitialized(value)
            | Op::Get(value, _)
            | Op::MarkClass(value)
            | Op::DisableConstructor(value)
            | Op::PropertyKey(value)
            | Op::ToNumeric(value)
            | Op::Unary(_, value)
            | Op::IsUndefined(value)
            | Op::IsNullish(value)
            | Op::ProbeProperty(value, _)
            | Op::IsNative(value, _)
            | Op::View(value)
            | Op::IsTyped(value)
            | Op::IsNumber(value)
            | Op::AsNumber(value) => vec![value],
            Op::Define(object, _, value)
            | Op::DefineHidden(object, _, value)
            | Op::Set(object, _, value) => vec![object, value],
            Op::GetIndex(object, key)
            | Op::TypedIndex(object, key)
            | Op::ProbeIndex(object, key)
            | Op::Binary(_, object, key)
            | Op::ArrayAppend(object, key, _)
            | Op::ConstructSpread(object, key) => vec![object, key],
            Op::CallSpread(callee, receiver, args) => vec![callee, receiver, args],
            Op::SetIndex(object, key, value) => vec![object, key, value],
            Op::NewArray(values) | Op::CallNative(_, values) => values.iter_mut().collect(),
            Op::Construct { callee, args } => {
                std::iter::once(callee).chain(args.iter_mut()).collect()
            }
            Op::Call {
                callee,
                receiver,
                args,
            } => std::iter::once(callee)
                .chain(receiver.as_mut())
                .chain(args.iter_mut())
                .collect(),
        }
    }

    pub(super) fn write_target(&self) -> Option<Value> {
        match self {
            Op::Define(object, ..)
            | Op::Set(object, ..)
            | Op::SetIndex(object, ..)
            | Op::IteratorNext(object)
            | Op::EnumerateNext(object) => Some(*object),
            _ => None,
        }
    }
}

impl TerminatorKind {
    pub(super) fn operands_mut(&mut self) -> Vec<&mut Value> {
        match self {
            TerminatorKind::Jump(target) => target.args.iter_mut().collect(),
            TerminatorKind::CheckException(normal, error) => normal
                .args
                .iter_mut()
                .chain(error.args.iter_mut())
                .collect(),
            TerminatorKind::Branch(condition, then, otherwise) => std::iter::once(condition)
                .chain(then.args.iter_mut())
                .chain(otherwise.args.iter_mut())
                .collect(),
            TerminatorKind::Yield(_, value, target) => std::iter::once(value)
                .chain(target.args.iter_mut())
                .collect(),
            TerminatorKind::Return(value) => vec![value],
        }
    }

    pub(super) fn operands(&self) -> Vec<Value> {
        let mut kind = self.clone();
        kind.operands_mut()
            .into_iter()
            .map(|value| *value)
            .collect()
    }

    pub(super) fn targets(&self) -> Vec<&Target> {
        match self {
            TerminatorKind::Jump(target) => vec![target],
            TerminatorKind::CheckException(normal, error) => vec![normal, error],
            TerminatorKind::Branch(_, then, otherwise) => vec![then, otherwise],
            TerminatorKind::Yield(_, _, target) => vec![target],
            TerminatorKind::Return(_) => Vec::new(),
        }
    }

    pub(super) fn targets_mut(&mut self) -> Vec<&mut Target> {
        match self {
            TerminatorKind::Jump(target) => vec![target],
            TerminatorKind::CheckException(normal, error) => vec![normal, error],
            TerminatorKind::Branch(_, then, otherwise) => vec![then, otherwise],
            TerminatorKind::Yield(_, _, target) => vec![target],
            TerminatorKind::Return(_) => Vec::new(),
        }
    }
}

pub(super) fn resolve(aliases: &HashMap<Value, Value>, mut value: Value) -> Value {
    while let Some(&next) = aliases.get(&value) {
        if next == value {
            break;
        }
        value = next;
    }
    value
}

pub(super) struct Cfg {
    pub(super) rpo: Vec<Block>,
    pub(super) preds: SecondaryMap<Block, Vec<Block>>,
    dominators: SecondaryMap<Block, Option<Block>>,
}

impl Cfg {
    pub(super) fn new(graph: &Graph) -> Self {
        let mut visited = HashSet::from([graph.entry]);
        let mut postorder = Vec::new();
        let mut stack = vec![(graph.entry, 0)];
        while let Some((block, index)) = stack.pop() {
            let successors = graph.successors(block);
            if let Some(&next) = successors.get(index) {
                stack.push((block, index + 1));
                if visited.insert(next) {
                    stack.push((next, 0));
                }
            } else {
                postorder.push(block);
            }
        }
        let rpo: Vec<Block> = postorder.into_iter().rev().collect();
        let mut preds: SecondaryMap<Block, Vec<Block>> = SecondaryMap::new();
        for &block in &rpo {
            for successor in graph.successors(block) {
                preds[successor].push(block);
            }
        }
        let mut ranks = SecondaryMap::<Block, usize>::new();
        for (index, &block) in rpo.iter().enumerate() {
            ranks[block] = index;
        }
        let mut dominators = SecondaryMap::<Block, Option<Block>>::new();
        dominators[graph.entry] = Some(graph.entry);
        let mut changed = true;
        while changed {
            changed = false;
            for &block in rpo.iter().skip(1) {
                let mut incoming = preds[block]
                    .iter()
                    .copied()
                    .filter(|&pred| dominators[pred].is_some());
                let Some(mut next) = incoming.next() else {
                    continue;
                };
                for mut pred in incoming {
                    while next != pred {
                        if ranks[next] > ranks[pred] {
                            next = dominators[next].unwrap();
                        } else {
                            pred = dominators[pred].unwrap();
                        }
                    }
                }
                if dominators[block] != Some(next) {
                    dominators[block] = Some(next);
                    changed = true;
                }
            }
        }
        Self {
            rpo,
            preds,
            dominators,
        }
    }

    pub(super) fn dominates(&self, dominator: Block, block: Block) -> bool {
        let mut current = block;
        while let Some(parent) = self.dominators[current] {
            if current == dominator {
                return true;
            }
            if parent == current {
                break;
            }
            current = parent;
        }
        false
    }
}

impl Graph {
    pub(super) fn successors(&self, block: Block) -> Vec<Block> {
        let mut successors = Vec::new();
        if let Some(terminator) = &self.blocks[block].terminator {
            for target in terminator.kind.targets() {
                if !successors.contains(&target.block) {
                    successors.push(target.block);
                }
            }
        }
        successors
    }

    pub(super) fn def_block(&self, value: Value) -> Block {
        match self.values[value] {
            ValueDef::Result(inst) => self.insts[inst].block,
            ValueDef::Param(block, _) => block,
        }
    }

    pub(super) fn defining_op(&self, value: Value) -> Option<&Op> {
        match self.values[value] {
            ValueDef::Result(inst) => Some(&self.insts[inst].op),
            ValueDef::Param(..) => None,
        }
    }

    pub(super) fn add_param(&mut self, block: Block) -> Value {
        let index = self.blocks[block].params.len();
        let value = self.values.push(ValueDef::Param(block, index));
        self.blocks[block].params.push(value);
        value
    }

    pub(super) fn push_edge_arg(&mut self, from: Block, to: Block, value: Value) {
        let terminator = self.blocks[from].terminator.as_mut().unwrap();
        for target in terminator.kind.targets_mut() {
            if target.block == to {
                target.args.push(value);
            }
        }
    }

    pub(super) fn insert(
        &mut self,
        block: Block,
        position: usize,
        span: Span,
        op: Op,
        has_result: bool,
    ) -> (Inst, Option<Value>) {
        let inst = self.insts.next_key();
        let result = has_result.then(|| self.values.push(ValueDef::Result(inst)));
        self.insts.push(InstData {
            span,
            block,
            result,
            op,
        });
        self.blocks[block].insts.insert(position, inst);
        (inst, result)
    }

    pub(super) fn append(
        &mut self,
        block: Block,
        span: Span,
        op: Op,
        has_result: bool,
    ) -> (Inst, Option<Value>) {
        let position = self.blocks[block].insts.len();
        self.insert(block, position, span, op, has_result)
    }

    pub(super) fn remove_inst(&mut self, inst: Inst) {
        let block = self.insts[inst].block;
        self.blocks[block].insts.retain(|&i| i != inst);
    }

    pub(super) fn move_inst(&mut self, inst: Inst, block: Block) {
        self.remove_inst(inst);
        self.blocks[block].insts.push(inst);
        self.insts[inst].block = block;
    }

    pub(super) fn replace_uses(&mut self, aliases: &HashMap<Value, Value>) {
        if aliases.is_empty() {
            return;
        }
        for data in self.insts.values_mut() {
            for value in data.op.operands_mut() {
                *value = resolve(aliases, *value);
            }
        }
        for data in self.blocks.values_mut() {
            if let Some(terminator) = &mut data.terminator {
                for value in terminator.kind.operands_mut() {
                    *value = resolve(aliases, *value);
                }
            }
        }
    }

    pub(super) fn use_counts(&self) -> SecondaryMap<Value, u32> {
        let mut counts = SecondaryMap::new();
        for data in self.blocks.values() {
            for &inst in &data.insts {
                for value in self.insts[inst].op.operands() {
                    counts[value] += 1;
                }
            }
            if let Some(terminator) = &data.terminator {
                for value in terminator.kind.operands() {
                    counts[value] += 1;
                }
            }
        }
        counts
    }

    pub(super) fn materialized(&self, scope: Scope) -> bool {
        self.scopes.iter().any(|(candidate, data)| {
            if !data
                .bindings
                .iter()
                .any(|&binding| self.bindings[binding].captured)
            {
                return false;
            }
            let mut current = Some(candidate);
            while let Some(ancestor) = current {
                if ancestor == scope {
                    return true;
                }
                current = self.scopes[ancestor].parent;
            }
            false
        })
    }

    pub(super) fn contract(&self, op: &Op, facts: &Facts) -> Contract {
        use Ownership::*;
        let closed = |flags, assumptions, result| Contract {
            reentry: Reentry::Closed,
            flags,
            assumptions,
            result,
        };
        const PROPERTY: &[Assumption] = &[Assumption::NoAccessors, Assumption::NoCoercionCallbacks];
        const COERCION: &[Assumption] = &[Assumption::NoCoercionCallbacks];
        const COERCING: u16 = HEAP_READ | ALLOCATION | INTERNAL | ERROR;
        let operands = op
            .operands()
            .into_iter()
            .fold(Types::default(), |types, value| {
                types.union(facts.of(value))
            });
        let coercing = |result| {
            if operands.within(Types::NUMBER) {
                closed(0, &[], Unowned)
            } else if operands.within(Types::PRIMITIVE) {
                closed(COERCING, &[], result)
            } else {
                closed(COERCING, COERCION, result)
            }
        };
        match op {
            Op::Rest(_) | Op::Arguments => closed(ALLOCATION | INTERNAL | ERROR, &[], Owned),
            Op::Throw(_) => closed(INTERNAL | ERROR, &[], None),
            Op::Catch => closed(INTERNAL, &[], Owned),
            Op::Argument(_) | Op::This => closed(0, &[], Borrowed),
            Op::Constant(_) => closed(0, &[], Unowned),
            Op::GlobalRead(_) | Op::GlobalReadOptional(_) => {
                closed(GLOBAL_READ | INTERNAL | ERROR, &[], Borrowed)
            }
            Op::RegExp(..) => closed(ALLOCATION | INTERNAL | ERROR, &[], Owned),
            Op::PropertyKey(_) => closed(HEAP_READ | ALLOCATION | INTERNAL | ERROR, &[], Owned),
            Op::GlobalWrite(..) | Op::GlobalInitialize(..) => {
                closed(GLOBAL_WRITE | INTERNAL | ERROR, &[], None)
            }
            Op::Read(binding) => closed(
                if self.bindings[*binding].captured {
                    HEAP_READ | INTERNAL | ERROR
                } else {
                    ERROR
                },
                &[],
                Borrowed,
            ),
            Op::Initialize(binding, _) | Op::Write(binding, _)
                if self.bindings[*binding].captured =>
            {
                closed(HEAP_WRITE | INTERNAL | ERROR, &[], None)
            }
            Op::Initialize(..) => closed(INTERNAL, &[], None),
            Op::Write(..) => closed(INTERNAL | ERROR, &[], None),
            Op::CheckInitialized(_) => {
                if operands.intersects(Types::UNINITIALIZED) {
                    closed(ERROR, &[], None)
                } else {
                    closed(0, &[], None)
                }
            }
            Op::Closure(..)
            | Op::NewObject
            | Op::NewArray(_)
            | Op::IteratorNew(_)
            | Op::EnumerateNew(_) => closed(ALLOCATION | INTERNAL | ERROR, &[], Owned),
            Op::IteratorNext(_) | Op::EnumerateNext(_) => closed(
                HEAP_READ | HEAP_WRITE | ALLOCATION | INTERNAL | ERROR,
                &[Assumption::DirectIteration],
                MaybeOwned,
            ),
            Op::IsUninitialized(_) => closed(0, &[], Unowned),
            Op::Define(..) => closed(HEAP_READ | ALLOCATION | INTERNAL | ERROR, PROPERTY, None),
            Op::DefineHidden(..) | Op::MarkClass(_) | Op::DisableConstructor(_) => {
                closed(HEAP_WRITE | ALLOCATION | INTERNAL | ERROR, &[], None)
            }
            Op::ArrayAppend(..) => closed(
                HEAP_READ | HEAP_WRITE | ALLOCATION | INTERNAL | ERROR,
                &[],
                None,
            ),
            Op::Get(..) | Op::GetIndex(..) => closed(
                HEAP_READ | GLOBAL_READ | ALLOCATION | INTERNAL | ERROR,
                PROPERTY,
                Borrowed,
            ),
            Op::Set(..) | Op::SetIndex(..) => closed(
                HEAP_READ | HEAP_WRITE | ALLOCATION | INTERNAL | ERROR,
                PROPERTY,
                None,
            ),
            Op::Unary(UnaryOp::Not, _) => closed(HEAP_READ, &[], Unowned),
            Op::Unary(UnaryOp::Typeof, _) => {
                closed(HEAP_READ | ALLOCATION | ERROR, &[], MaybeOwned)
            }
            Op::Unary(UnaryOp::Increment | UnaryOp::Decrement, _) => {
                if operands.within(Types::NUMBER) {
                    closed(0, &[], Unowned)
                } else {
                    closed(ALLOCATION | INTERNAL | ERROR, &[], MaybeOwned)
                }
            }
            Op::ToNumeric(_) | Op::Unary(..) => coercing(MaybeOwned),
            Op::Binary(operator, ..) => match operator {
                BinaryOperator::StrictEquality | BinaryOperator::StrictInequality => {
                    closed(HEAP_READ, &[], Unowned)
                }
                BinaryOperator::Equality
                | BinaryOperator::Inequality
                | BinaryOperator::LessThan
                | BinaryOperator::LessEqualThan
                | BinaryOperator::GreaterThan
                | BinaryOperator::GreaterEqualThan => coercing(Unowned),
                BinaryOperator::In | BinaryOperator::Instanceof => Contract {
                    reentry: Reentry::Unknown,
                    flags: UNKNOWN,
                    assumptions: &[],
                    result: Unowned,
                },
                _ => coercing(MaybeOwned),
            },
            Op::Call { .. }
            | Op::Construct { .. }
            | Op::CallSpread(..)
            | Op::ConstructSpread(..) => Contract {
                reentry: Reentry::Arbitrary,
                flags: UNKNOWN,
                assumptions: &[],
                result: Owned,
            },
            Op::CallNative(..) => coercing(Unowned),
            Op::IsUndefined(_) | Op::IsNullish(_) | Op::IsNative(..) => closed(0, &[], Unowned),
            Op::ProbeGlobal(_) => closed(GLOBAL_READ, &[Assumption::NoAccessors], Borrowed),
            Op::ProbeProperty(..) | Op::ProbeIndex(..) => {
                closed(HEAP_READ, &[Assumption::NoAccessors], Borrowed)
            }
            Op::GuardValid(_) => closed(GLOBAL_READ, &[], Unowned),
            Op::GuardCommit(_) => closed(INTERNAL, &[], None),
            Op::View(_) => closed(HEAP_READ, &[Assumption::FixedTypedStorage], Unowned),
            Op::IsTyped(_) => closed(0, &[], Unowned),
            Op::TypedIndex(..) => closed(HEAP_READ, &[Assumption::FixedTypedStorage], Unowned),
            Op::IsNumber(_) => closed(0, &[], Unowned),
            Op::AsNumber(_) => closed(INTERNAL, &[], Unowned),
            Op::EnterScope(scope) | Op::NextIteration(scope) | Op::FreshIteration(scope) => {
                if self.materialized(*scope) {
                    closed(ALLOCATION | INTERNAL | ERROR, &[], None)
                } else {
                    closed(0, &[], None)
                }
            }
            Op::LeaveScope(scope) => {
                if self.materialized(*scope) {
                    closed(INTERNAL, &[], None)
                } else {
                    closed(0, &[], None)
                }
            }
        }
    }

    pub(super) fn verify(&self) -> Result<()> {
        let mut stacks: SecondaryMap<Block, Option<Vec<Scope>>> = SecondaryMap::new();
        stacks[self.entry] = Some(Vec::new());
        let mut worklist = vec![self.entry];
        while let Some(block) = worklist.pop() {
            let mut stack = stacks[block].clone().unwrap();
            for &inst in &self.blocks[block].insts {
                if self.insts[inst].block != block {
                    return Err(format!("{inst} is listed in the wrong block"));
                }
                match self.insts[inst].op {
                    Op::EnterScope(scope) => {
                        if self.scopes[scope].parent != stack.last().copied() {
                            return Err(format!("{inst} enters {scope} outside its parent"));
                        }
                        stack.push(scope);
                    }
                    Op::LeaveScope(scope) => {
                        if stack.pop() != Some(scope) {
                            return Err(format!("{inst} leaves {scope} out of order"));
                        }
                    }
                    Op::NextIteration(scope) | Op::FreshIteration(scope) => {
                        if stack.last() != Some(&scope) {
                            return Err(format!("{inst} steps {scope} while it is not innermost"));
                        }
                    }
                    _ => {}
                }
            }
            let Some(terminator) = &self.blocks[block].terminator else {
                return Err(format!("{block} has no terminator"));
            };
            if matches!(terminator.kind, TerminatorKind::Return(_)) && !stack.is_empty() {
                return Err(format!("{block} returns with open scopes"));
            }
            for target in terminator.kind.targets() {
                if target.args.len() != self.blocks[target.block].params.len() {
                    return Err(format!("{block} passes wrong arity to {}", target.block));
                }
                match &stacks[target.block] {
                    Some(existing) if *existing != stack => {
                        return Err(format!(
                            "{block} reaches {} with a different scope stack",
                            target.block
                        ));
                    }
                    Some(_) => {}
                    None => {
                        stacks[target.block] = Some(stack.clone());
                        worklist.push(target.block);
                    }
                }
            }
        }
        let cfg = Cfg::new(self);
        for &block in &cfg.rpo {
            for (index, &param) in self.blocks[block].params.iter().enumerate() {
                if self.values[param] != ValueDef::Param(block, index) {
                    return Err(format!("{param} is not parameter {index} of {block}"));
                }
            }
            let check = |value: Value, position: usize| -> Result<()> {
                let available = match self.values[value] {
                    ValueDef::Param(definer, index) => {
                        self.blocks[definer].params.get(index) == Some(&value)
                            && cfg.dominates(definer, block)
                    }
                    ValueDef::Result(inst) => {
                        let definer = self.insts[inst].block;
                        let index = self.blocks[definer].insts.iter().position(|&i| i == inst);
                        self.insts[inst].result == Some(value)
                            && index.is_some_and(|index| {
                                if definer == block {
                                    index < position
                                } else {
                                    cfg.dominates(definer, block)
                                }
                            })
                    }
                };
                if available {
                    Ok(())
                } else {
                    Err(format!(
                        "{value} is used in {block} without a dominating definition"
                    ))
                }
            };
            let data = &self.blocks[block];
            for (index, &inst) in data.insts.iter().enumerate() {
                for value in self.insts[inst].op.operands() {
                    check(value, index)?;
                }
            }
            for value in data.terminator.as_ref().unwrap().kind.operands() {
                check(value, data.insts.len())?;
            }
        }
        Ok(())
    }

    fn binding_name(&self, binding: Binding) -> String {
        format!("{}#{}", self.bindings[binding].name, binding.index())
    }

    pub(super) fn op_text(&self, op: &Op) -> String {
        let list = |values: &[Value]| {
            values
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        };
        match op {
            Op::Argument(index) => format!("argument {index}"),
            Op::Arguments => "arguments".into(),
            Op::Constant(Constant::Undefined) => "undefined".into(),
            Op::Constant(Constant::Null) => "null".into(),
            Op::Rest(index) => format!("rest {index}"),
            Op::This => "this".into(),
            Op::Throw(value) => format!("throw {value}"),
            Op::Catch => "catch".into(),
            Op::Closure(span, arrow, generator, captures) => format!(
                "closure {}..{} arrow={arrow} generator={generator} captures={captures:?}",
                span.start, span.end
            ),
            Op::Constant(Constant::Boolean(value)) => value.to_string(),
            Op::Constant(Constant::Number(value)) => format!("number {value}"),
            Op::Constant(Constant::String(value)) => format!("string {value:?}"),
            Op::Constant(Constant::Uninitialized) => "uninitialized".into(),
            Op::GlobalRead(name) => format!("global {name}"),
            Op::GlobalReadOptional(name) => format!("global_optional {name}"),
            Op::RegExp(pattern, flags) => format!("regexp {pattern:?} {flags}"),
            Op::PropertyKey(value) => format!("property_key {value}"),
            Op::GlobalWrite(name, value) => format!("global_write {name}, {value}"),
            Op::GlobalInitialize(name, value) => format!("global_initialize {name}, {value}"),
            Op::Read(binding) => format!("read {}", self.binding_name(*binding)),
            Op::Initialize(binding, value) => {
                format!("initialize {}, {value}", self.binding_name(*binding))
            }
            Op::Write(binding, value) => format!("write {}, {value}", self.binding_name(*binding)),
            Op::CheckInitialized(value) => format!("check_initialized {value}"),
            Op::NewObject => "new_object".into(),
            Op::NewArray(values) => format!("new_array [{}]", list(values)),
            Op::ArrayAppend(array, value, spread) => {
                format!("array_append {array}, {value}, {spread}")
            }
            Op::CallSpread(callee, receiver, args) => {
                format!("call_spread {callee}, {receiver}, {args}")
            }
            Op::ConstructSpread(callee, args) => format!("construct_spread {callee}, {args}"),
            Op::IteratorNew(value) => format!("iterator_new {value}"),
            Op::IteratorNext(value) => format!("iterator_next {value}"),
            Op::EnumerateNew(value) => format!("enumerate_new {value}"),
            Op::EnumerateNext(value) => format!("enumerate_next {value}"),
            Op::IsUninitialized(value) => format!("is_uninitialized {value}"),
            Op::Define(object, name, value) => format!("define {object}.{name}, {value}"),
            Op::DefineHidden(object, name, value) => {
                format!("define_hidden {object}.{name}, {value}")
            }
            Op::MarkClass(value) => format!("mark_class {value}"),
            Op::DisableConstructor(value) => format!("disable_constructor {value}"),
            Op::Get(object, name) => format!("get {object}.{name}"),
            Op::GetIndex(object, key) => format!("get {object}[{key}]"),
            Op::Set(object, name, value) => format!("set {object}.{name}, {value}"),
            Op::SetIndex(object, key, value) => format!("set {object}[{key}], {value}"),
            Op::ToNumeric(value) => format!("to_numeric {value}"),
            Op::Unary(operator, value) => format!("{operator:?} {value}").to_lowercase(),
            Op::Binary(operator, left, right) => {
                format!("{left} {} {right}", operator.as_str())
            }
            Op::Call {
                callee,
                receiver,
                args,
            } => match receiver {
                Some(receiver) => format!("call {callee}({}) this={receiver}", list(args)),
                None => format!("call {callee}({})", list(args)),
            },
            Op::Construct { callee, args } => format!("construct {callee}({})", list(args)),
            Op::CallNative(native, args) => format!("call_native {native}({})", list(args)),
            Op::IsUndefined(value) => format!("is_undefined {value}"),
            Op::IsNullish(value) => format!("is_nullish {value}"),
            Op::ProbeGlobal(name) => format!("probe_global {name}"),
            Op::ProbeProperty(object, name) => format!("probe {object}.{name}"),
            Op::ProbeIndex(object, key) => format!("probe {object}[{key}]"),
            Op::IsNative(value, native) => format!("is_native {value}, {native}"),
            Op::GuardValid(guard) => format!("guard_valid {guard}"),
            Op::GuardCommit(guard) => format!("guard_commit {guard}"),
            Op::View(value) => format!("view {value}"),
            Op::IsTyped(value) => format!("is_typed {value}"),
            Op::TypedIndex(object, key) => format!("get_typed {object}[{key}]"),
            Op::IsNumber(value) => format!("is_number {value}"),
            Op::AsNumber(value) => format!("as_number {value}"),
            Op::EnterScope(scope) => format!("enter {scope}"),
            Op::LeaveScope(scope) => format!("leave {scope}"),
            Op::NextIteration(scope) => format!("next_iteration {scope}"),
            Op::FreshIteration(scope) => format!("fresh_iteration {scope}"),
        }
    }

    pub(super) fn listing<'a>(&'a self, facts: &'a Facts) -> Listing<'a> {
        Listing { graph: self, facts }
    }
}

pub(super) struct Listing<'a> {
    graph: &'a Graph,
    facts: &'a Facts,
}

impl Listing<'_> {
    fn value(&self, value: Value) -> String {
        let types = self.facts.of(value);
        if types.is_informative() {
            format!("{value}:{types}")
        } else {
            value.to_string()
        }
    }
}

impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.block)?;
        if !self.args.is_empty() {
            let args: Vec<_> = self.args.iter().map(ToString::to_string).collect();
            write!(f, "({})", args.join(", "))?;
        }
        Ok(())
    }
}

impl Listing<'_> {
    fn lines(&self) -> Vec<(u32, Option<Span>, String)> {
        let graph = self.graph;
        let mut lines = vec![(0, None, format!("function {}", graph.name))];
        for (scope, data) in &graph.scopes {
            debug_assert!(
                data.bindings
                    .iter()
                    .all(|&binding| graph.bindings[binding].scope == scope)
            );
            let parent = data
                .parent
                .map_or(String::new(), |parent| format!(" in {parent}"));
            let mut text = format!("  {scope} {:?}{parent}:", data.kind);
            for &binding in &data.bindings {
                let binding_data = &graph.bindings[binding];
                text.push_str(&format!(
                    " {} {:?}",
                    graph.binding_name(binding),
                    binding_data.kind
                ));
                if binding_data.captured {
                    text.push_str(" captured");
                }
                text.push(';');
            }
            lines.push((0, None, text));
        }
        for block in Cfg::new(graph).rpo {
            let data = &graph.blocks[block];
            let header = if data.params.is_empty() {
                format!("{block}:")
            } else {
                let params: Vec<_> = data.params.iter().map(|&p| self.value(p)).collect();
                format!("{block}({}):", params.join(", "))
            };
            lines.push((0, None, header));
            for &inst in &data.insts {
                let inst_data = &graph.insts[inst];
                let mut text = graph.op_text(&inst_data.op);
                if let Some(result) = inst_data.result {
                    text = format!("{} = {text}", self.value(result));
                }
                let contract = graph.contract(&inst_data.op, self.facts);
                let note = match (contract.reentry, contract.may_throw()) {
                    (Reentry::Closed, true) => " throws",
                    (Reentry::Closed, false) => "",
                    (_, _) => " reenters, throws",
                };
                let span = inst_data.span;
                lines.push((
                    graph.inst_location(inst),
                    Some(span),
                    format!("  {text:<48} ; {}..{}{note}", span.start, span.end),
                ));
            }
            if let Some(terminator) = &data.terminator {
                let text = match &terminator.kind {
                    TerminatorKind::Jump(target) => format!("jump {target}"),
                    TerminatorKind::CheckException(normal, error) => {
                        format!("check_exception {normal}, {error}")
                    }
                    TerminatorKind::Branch(condition, then, otherwise) => {
                        format!("branch {condition}, {then}, {otherwise}")
                    }
                    TerminatorKind::Yield(_, value, target) => format!("yield {value}, {target}"),
                    TerminatorKind::Return(value) => format!("return {value}"),
                };
                let span = terminator.span;
                lines.push((
                    graph.terminator_location(block),
                    Some(span),
                    format!("  {text:<48} ; {}..{}", span.start, span.end),
                ));
            }
        }
        lines
    }
}

impl fmt::Display for Listing<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (_, _, line) in self.lines() {
            writeln!(f, "{line}")?;
        }
        Ok(())
    }
}

impl Graph {
    pub(super) fn inst_location(&self, inst: Inst) -> u32 {
        u32::try_from(inst.index() + 1).unwrap()
    }

    pub(super) fn terminator_location(&self, block: Block) -> u32 {
        u32::try_from(self.insts.len() + block.index() + 1).unwrap()
    }
}

type Result<T> = std::result::Result<T, String>;

fn unsupported<T>(what: &str, span: Span) -> Result<T> {
    Err(format!(
        "unsupported {what} at {}..{}",
        span.start, span.end
    ))
}

pub(super) struct Analysis {
    pub(super) graph: Graph,
    facts: Facts,
    notes: Vec<String>,
}

impl Analysis {
    pub(super) fn listing_lines(&self) -> Vec<(u32, Option<Span>, String)> {
        self.graph.listing(&self.facts).lines()
    }
}

impl fmt::Display for Analysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.graph.listing(&self.facts))?;
        for note in &self.notes {
            writeln!(f, "  note: {note}")?;
        }
        Ok(())
    }
}

pub(super) fn lower_global(
    statements: &[Statement<'_>],
    span: Span,
    scoping: &Scoping,
    captured: &HashSet<SymbolId>,
    closures: &std::collections::HashMap<u32, (cranelift_module::FuncId, Vec<SymbolId>, bool)>,
    arguments_references: &std::collections::HashMap<u32, SymbolId>,
    ssa_opt: bool,
) -> Result<Analysis> {
    let graph = build::lower_global(
        statements,
        span,
        scoping,
        captured,
        closures,
        arguments_references,
    )?;
    optimize_graph(graph, ssa_opt)
}

pub(super) fn lower_for_emission(
    params: &FormalParameters<'_>,
    statements: &[Statement<'_>],
    expression: Option<&Expression<'_>>,
    span: Span,
    scoping: &Scoping,
    captured: &HashSet<SymbolId>,
    captures: &[SymbolId],
    closures: &std::collections::HashMap<u32, (cranelift_module::FuncId, Vec<SymbolId>, bool)>,
    arguments_binding: Option<SymbolId>,
    arguments_references: &std::collections::HashMap<u32, SymbolId>,
    generator: bool,
    ssa_opt: bool,
) -> Result<Analysis> {
    let graph = build::lower_parts(
        params,
        statements,
        expression,
        span,
        scoping,
        captured,
        captures,
        closures,
        arguments_binding,
        arguments_references,
        generator,
    )?;
    optimize_graph(graph, ssa_opt)
}

#[cfg(test)]
pub(super) fn analyze(
    function: &oxc_ast::ast::Function<'_>,
    scoping: &Scoping,
    captured: &HashSet<SymbolId>,
) -> Result<Analysis> {
    optimize_graph(build::lower(function, scoping, captured)?, true)
}

fn optimize_graph(mut graph: Graph, ssa_opt: bool) -> Result<Analysis> {
    let mut notes = Vec::new();
    if !ssa_opt || graph.has_handlers || graph.generator {
        let facts = Facts::compute(&graph);
        return Ok(Analysis {
            graph,
            facts,
            notes,
        });
    }
    promote::promote(&mut graph, &mut notes);
    graph.verify()?;
    optimize::simplify(&mut graph);
    let mut regions = optimize::version(&mut graph, &mut notes)?;
    graph.verify()?;
    for region in &regions {
        optimize::reuse_loads(&mut graph, region, &mut notes);
        graph.verify()?;
        optimize::hoist(&mut graph, region, &mut notes);
        graph.verify()?;
    }
    for copy in optimize::check_loops(&mut graph, &mut regions, &mut notes)? {
        optimize::reuse_loads(&mut graph, &copy, &mut notes);
        graph.verify()?;
        optimize::hoist(&mut graph, &copy, &mut notes);
        graph.verify()?;
    }
    optimize::views(&mut graph, &mut notes);
    graph.verify()?;
    optimize::simplify(&mut graph);
    graph.verify()?;
    let facts = Facts::compute(&graph);
    Ok(Analysis {
        graph,
        facts,
        notes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxc_allocator::Allocator;
    use oxc_ast::ast::Statement;
    use oxc_parser::Parser;
    use oxc_span::SourceType;

    fn with_function<T>(
        source: &str,
        f: impl FnOnce(&oxc_ast::ast::Function<'_>, &Scoping) -> Result<T>,
    ) -> Result<T> {
        let allocator = Allocator::default();
        let parsed = Parser::new(&allocator, source, SourceType::script()).parse();
        assert!(!parsed.diagnostics.has_errors());
        let Statement::FunctionDeclaration(function) = &parsed.program.body[0] else {
            panic!()
        };
        let semantic = oxc_semantic::SemanticBuilder::new_compiler()
            .build(&parsed.program)
            .semantic;
        f(function, semantic.scoping())
    }

    fn lower_source(source: &str) -> Result<Graph> {
        with_function(source, |function, scoping| {
            build::lower(function, scoping, &HashSet::new())
        })
    }

    fn analyze_source(source: &str) -> Analysis {
        with_function(source, |function, scoping| {
            analyze(function, scoping, &HashSet::new())
        })
        .unwrap()
    }

    fn ops(graph: &Graph) -> Vec<&Op> {
        Cfg::new(graph)
            .rpo
            .into_iter()
            .flat_map(|block| &graph.blocks[block].insts)
            .map(|&inst| &graph.insts[inst].op)
            .collect()
    }

    const CONTROL_FLOW: &str = "function f(a, b = g.h) {
        const r = { min: Infinity, list: [a, 'x', null] };
        let i = 0;
        while (i < a.length) {
            let v = a[i++];
            if (v === undefined) continue;
            if (v > 10) { const w = v; break; }
            r.min = Math.min(r.min, v || b ? v : -v);
            a[i] += 1;
        }
        for (let x = 0; x < 3; ++x) {
            if (x == 2) return r;
            else { let y = x; }
        }
        return b ?? r;
    }";

    #[test]
    fn lowers_control_flow_into_verified_ssa() {
        let graph = lower_source(CONTROL_FLOW).unwrap();
        assert!(
            ops(&graph)
                .iter()
                .any(|op| matches!(op, Op::Constant(Constant::Number(n)) if n.is_infinite()))
        );
        assert!(graph.blocks.values().any(|block| !block.params.is_empty()));
        analyze_source(CONTROL_FLOW);
    }

    #[test]
    fn postfix_update_yields_numeric_old_value() {
        let graph = lower_source("function f(x) { return x++; }").unwrap();
        let block = &graph.blocks[graph.entry];
        let TerminatorKind::Return(value) = block.terminator.as_ref().unwrap().kind else {
            panic!()
        };
        assert!(matches!(graph.defining_op(value), Some(Op::ToNumeric(_))));
    }

    #[test]
    fn keeps_shadowed_bindings_and_call_order() {
        let graph = lower_source("function f(x) { { let x = 1; obj.m(x); } return x; }").unwrap();
        assert_eq!(graph.bindings.len(), 2);
        let ops = ops(&graph);
        let get = ops
            .iter()
            .position(|op| matches!(op, Op::Get(_, name) if name == "m"))
            .unwrap();
        let read = ops
            .iter()
            .position(|op| matches!(op, Op::Read(binding) if binding.index() == 1))
            .unwrap();
        let call = ops
            .iter()
            .position(|op| {
                matches!(
                    op,
                    Op::Call {
                        receiver: Some(_),
                        ..
                    }
                )
            })
            .unwrap();
        assert!(get < read && read < call);
    }

    #[test]
    fn rejects_unsupported_syntax() {
        let Err(error) = lower_source("function f(a) { return delete a.x; }") else {
            panic!()
        };
        assert!(error.contains("unsupported"));
    }

    #[test]
    fn promotes_bindings_and_infers_numbers() {
        let analysis = analyze_source(
            "function f(a) { let s = 0; for (let i = 0; i < a; ++i) s = s + i; return s; }",
        );
        let graph = &analysis.graph;
        assert!(!ops(graph).iter().any(|op| matches!(
            op,
            Op::Read(_) | Op::Write(..) | Op::Initialize(..) | Op::CheckInitialized(_)
        )));
        let returned = graph
            .blocks
            .values()
            .find_map(|block| match block.terminator.as_ref()?.kind {
                TerminatorKind::Return(value) => Some(value),
                _ => None,
            })
            .unwrap();
        assert!(analysis.facts.of(returned).within(Types::NUMBER));
    }

    #[test]
    fn keeps_initialization_check_before_declaration() {
        let analysis = analyze_source("function f() { let x = y; const y = 2; return x; }");
        assert!(
            ops(&analysis.graph)
                .iter()
                .any(|op| matches!(op, Op::CheckInitialized(_)))
        );
    }

    #[test]
    fn resets_promoted_bindings_on_scope_entry() {
        for source in [
            "function f() { for (let i = 0; i < 2; i++) { if (i) return x; let x = 1; } }",
            "function f() { for (let i = 0; i < 2; i++) { if (i) x; let x = 1; } }",
        ] {
            let analysis = analyze_source(source);
            let graph = &analysis.graph;
            let checks: Vec<_> = ops(graph)
                .into_iter()
                .filter_map(|op| match op {
                    Op::CheckInitialized(value) => Some(*value),
                    _ => None,
                })
                .collect();
            assert!(!checks.is_empty());
            for value in checks {
                assert!(matches!(
                    graph.defining_op(value),
                    Some(Op::Constant(Constant::Uninitialized))
                ));
            }
        }
    }

    #[test]
    fn global_writes_prevent_versioning_and_load_reuse() {
        let analysis = analyze_source(
            "function f() { let before = shared; Math = shared; shared = Math.min(3, 4); return before + shared; }",
        );
        let ops = ops(&analysis.graph);
        assert!(ops.iter().any(|op| matches!(op, Op::GlobalWrite(..))));
        assert_eq!(
            ops.iter()
                .filter(|op| matches!(op, Op::GlobalRead(name) if name == "shared"))
                .count(),
            3
        );
        assert!(!ops.iter().any(|op| matches!(op, Op::CallNative(..))));
    }

    #[test]
    fn versions_on_native_calls_only_when_closed() {
        let source = "function f(m) {
            const r = { min: Infinity };
            for (let x = 0; x < m.length; ++x)
                for (let y = 0; y < m[x].length; ++y)
                    r.min = Math.min(r.min, m[x][y]);
            return r;
        }";
        let analysis = analyze_source(source);
        let versioned = ops(&analysis.graph);
        assert!(versioned.iter().any(|op| matches!(op, Op::CallNative(..))));
        assert!(versioned.iter().any(|op| matches!(op, Op::Call { .. })));
        let unknown = analyze_source(
            "function f(m) { let s = 0; for (let x = 0; x < m.length; ++x) s = Math.min(s, g(x)); return s; }",
        );
        assert!(
            !ops(&unknown.graph)
                .iter()
                .any(|op| matches!(op, Op::CallNative(..)))
        );
    }
}

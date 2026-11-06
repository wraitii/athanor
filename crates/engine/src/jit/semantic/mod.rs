mod arrays;
mod borrow;
mod build;
mod comparisons;
mod construction;
pub(in crate::jit) mod facts;
mod inlining;
mod listing;
mod optimize;
mod promote;
pub(super) mod reads;
pub(super) mod specialize;
mod string_split;

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
use crate::heap::TypedKind;

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
    pub(in crate::jit) borrowed: HashSet<Value>,
    closed_calls: HashMap<Inst, (Value, u16)>,
    pub(in crate::jit) cold: HashSet<Block>,
    pub(super) lazy_captures: HashSet<Binding>,
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
    pub(super) source_name: Option<(usize, usize)>,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ArrayMethod {
    Map,
    Some,
    Every,
    Filter,
}

#[derive(Clone, Debug)]
pub(super) enum Op {
    Argument(usize),
    Arguments,
    Rest(usize),
    This,
    ClosureThis(Value),
    Throw(Value),
    Catch,
    Closure(
        Span,
        cranelift_module::FuncId,
        bool,
        bool,
        bool,
        Vec<Binding>,
        Option<Value>,
    ),
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
    NewHoleyArray(Value),
    MapElement(Value, Value),
    IsArrayIteration(Value, Value, ArrayMethod),
    ArrayAppend(Value, Value, bool),
    CallSpread(Value, Value, Value),
    ConstructSpread(Value, Value),
    // UNINITIALIZED leaves native constructors and lazy prototypes to generic construction.
    ConstructorPrototype(Value),
    ConstructSpreadResolved(Value, Value, Value),
    ConstructResolved {
        callee: Value,
        prototype: Value,
        args: Vec<Value>,
    },
    NewInstance(Value),
    ConstructorResult(Value, Value),
    // Calls a validated constructor with its allocated receiver; class constructors are allowed.
    ConstructorCall {
        callee: Value,
        receiver: Value,
        args: Vec<Value>,
    },
    ConstructorCallSpread(Value, Value, Value),
    IteratorNew(Value),
    IteratorNext(Value),
    IsIndexed(Value),
    IndexedLength(Value),
    IndexedGet(Value, Value),
    // Dominated by an indexed guard; results stay rooted across argument initialization.
    SpreadGet(Value, Value),
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
    // Pure integer coercion of a value proven to be a number or undefined.
    ToInt32(Value),
    Unary(UnaryOp, Value),
    Binary(BinaryOperator, Value, Value),
    GuardedCompare {
        operator: BinaryOperator,
        left: Value,
        right: Value,
        numbers: [Value; 2],
    },
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
    IsCode(Value, cranelift_module::FuncId),
    GuardValid(usize),
    GuardCommit(usize),
    View(Value),
    IsTyped(Value, Option<TypedKind>),
    TypedIndex(Value, Value, TypedRead),
    IsNumber(Value),
    IsString(Value),
    IsStringSplit(Value),
    StringSplit(Value, Vec<Value>),
    IsTypeof(Value, u8),
    AsNumber(Value),
    EnterScope(Scope),
    EnterCaptures(Scope, Value),
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
        self.operands_iter().collect()
    }

    pub(super) fn operands_iter(&self) -> impl Iterator<Item = Value> + '_ {
        let (fixed, rest): ([Option<Value>; 3], &[Value]) = match self {
            Op::Argument(_)
            | Op::Arguments
            | Op::Rest(_)
            | Op::This
            | Op::Catch
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
            | Op::FreshIteration(_) => ([None; 3], &[]),
            Op::Closure(_, _, _, _, _, _, receiver) => ([*receiver, None, None], &[]),
            Op::ClosureThis(value)
            | Op::NewHoleyArray(value)
            | Op::EnterCaptures(_, value)
            | Op::Throw(value)
            | Op::GlobalInitialize(_, value)
            | Op::GlobalWrite(_, value)
            | Op::Initialize(_, value)
            | Op::Write(_, value)
            | Op::CheckInitialized(value)
            | Op::IteratorNew(value)
            | Op::IteratorNext(value)
            | Op::IsIndexed(value)
            | Op::IndexedLength(value)
            | Op::EnumerateNew(value)
            | Op::EnumerateNext(value)
            | Op::IsUninitialized(value)
            | Op::Get(value, _)
            | Op::MarkClass(value)
            | Op::DisableConstructor(value)
            | Op::PropertyKey(value)
            | Op::ToNumeric(value)
            | Op::ToInt32(value)
            | Op::Unary(_, value)
            | Op::IsUndefined(value)
            | Op::IsNullish(value)
            | Op::ProbeProperty(value, _)
            | Op::IsNative(value, _)
            | Op::NewInstance(value)
            | Op::ConstructorPrototype(value)
            | Op::IsCode(value, _)
            | Op::View(value)
            | Op::IsTyped(value, _)
            | Op::IsNumber(value)
            | Op::IsString(value)
            | Op::IsStringSplit(value)
            | Op::IsTypeof(value, _)
            | Op::AsNumber(value) => ([Some(*value), None, None], &[]),
            Op::Define(object, _, value)
            | Op::DefineHidden(object, _, value)
            | Op::Set(object, _, value) => ([Some(*object), Some(*value), None], &[]),
            Op::MapElement(object, key)
            | Op::IsArrayIteration(object, key, _)
            | Op::GetIndex(object, key)
            | Op::TypedIndex(object, key, _)
            | Op::IndexedGet(object, key)
            | Op::SpreadGet(object, key)
            | Op::ProbeIndex(object, key)
            | Op::Binary(_, object, key)
            | Op::ArrayAppend(object, key, _)
            | Op::ConstructorResult(object, key)
            | Op::ConstructSpread(object, key) => ([Some(*object), Some(*key), None], &[]),
            Op::CallSpread(callee, receiver, args)
            | Op::ConstructorCallSpread(callee, receiver, args)
            | Op::ConstructSpreadResolved(callee, receiver, args) => {
                ([Some(*callee), Some(*receiver), Some(*args)], &[])
            }
            Op::ConstructorCall {
                callee,
                receiver,
                args,
            } => ([Some(*callee), Some(*receiver), None], args),
            Op::ConstructResolved {
                callee,
                prototype,
                args,
            } => ([Some(*callee), Some(*prototype), None], args),
            Op::GuardedCompare {
                left,
                right,
                numbers,
                ..
            } => ([Some(*left), Some(*right), None], numbers),
            Op::SetIndex(object, key, value) => ([Some(*object), Some(*key), Some(*value)], &[]),
            Op::NewArray(values) | Op::CallNative(_, values) => ([None; 3], values),
            Op::StringSplit(receiver, args) => ([Some(*receiver), None, None], args),
            Op::Construct { callee, args } => ([Some(*callee), None, None], args),
            Op::Call {
                callee,
                receiver,
                args,
            } => ([Some(*callee), *receiver, None], args),
        };
        fixed.into_iter().flatten().chain(rest.iter().copied())
    }

    pub(super) fn operands_mut(&mut self) -> impl Iterator<Item = &mut Value> + '_ {
        let (fixed, rest): ([Option<&mut Value>; 3], &mut [Value]) = match self {
            Op::Argument(_)
            | Op::Arguments
            | Op::Rest(_)
            | Op::This
            | Op::Catch
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
            | Op::FreshIteration(_) => ([None, None, None], &mut []),
            Op::Closure(_, _, _, _, _, _, receiver) => ([receiver.as_mut(), None, None], &mut []),
            Op::ClosureThis(value)
            | Op::NewHoleyArray(value)
            | Op::EnterCaptures(_, value)
            | Op::Throw(value)
            | Op::GlobalInitialize(_, value)
            | Op::GlobalWrite(_, value)
            | Op::Initialize(_, value)
            | Op::Write(_, value)
            | Op::CheckInitialized(value)
            | Op::IteratorNew(value)
            | Op::IteratorNext(value)
            | Op::IsIndexed(value)
            | Op::IndexedLength(value)
            | Op::EnumerateNew(value)
            | Op::EnumerateNext(value)
            | Op::IsUninitialized(value)
            | Op::Get(value, _)
            | Op::MarkClass(value)
            | Op::DisableConstructor(value)
            | Op::PropertyKey(value)
            | Op::ToNumeric(value)
            | Op::ToInt32(value)
            | Op::Unary(_, value)
            | Op::IsUndefined(value)
            | Op::IsNullish(value)
            | Op::ProbeProperty(value, _)
            | Op::IsNative(value, _)
            | Op::NewInstance(value)
            | Op::ConstructorPrototype(value)
            | Op::IsCode(value, _)
            | Op::View(value)
            | Op::IsTyped(value, _)
            | Op::IsNumber(value)
            | Op::IsString(value)
            | Op::IsStringSplit(value)
            | Op::IsTypeof(value, _)
            | Op::AsNumber(value) => ([Some(value), None, None], &mut []),
            Op::Define(object, _, value)
            | Op::DefineHidden(object, _, value)
            | Op::Set(object, _, value) => ([Some(object), Some(value), None], &mut []),
            Op::MapElement(object, key)
            | Op::IsArrayIteration(object, key, _)
            | Op::GetIndex(object, key)
            | Op::TypedIndex(object, key, _)
            | Op::IndexedGet(object, key)
            | Op::SpreadGet(object, key)
            | Op::ProbeIndex(object, key)
            | Op::Binary(_, object, key)
            | Op::ArrayAppend(object, key, _)
            | Op::ConstructorResult(object, key)
            | Op::ConstructSpread(object, key) => ([Some(object), Some(key), None], &mut []),
            Op::CallSpread(callee, receiver, args)
            | Op::ConstructorCallSpread(callee, receiver, args)
            | Op::ConstructSpreadResolved(callee, receiver, args) => {
                ([Some(callee), Some(receiver), Some(args)], &mut [])
            }
            Op::ConstructorCall {
                callee,
                receiver,
                args,
            } => ([Some(callee), Some(receiver), None], args),
            Op::ConstructResolved {
                callee,
                prototype,
                args,
            } => ([Some(callee), Some(prototype), None], args),
            Op::GuardedCompare {
                left,
                right,
                numbers,
                ..
            } => ([Some(left), Some(right), None], numbers),
            Op::SetIndex(object, key, value) => ([Some(object), Some(key), Some(value)], &mut []),
            Op::NewArray(values) | Op::CallNative(_, values) => ([None, None, None], values),
            Op::StringSplit(receiver, args) => ([Some(receiver), None, None], args),
            Op::Construct { callee, args } => ([Some(callee), None, None], args),
            Op::Call {
                callee,
                receiver,
                args,
            } => ([Some(callee), receiver.as_mut(), None], args),
        };
        fixed.into_iter().flatten().chain(rest.iter_mut())
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
    pub(super) fn operands_mut(&mut self) -> impl Iterator<Item = &mut Value> + '_ {
        let (first, rest, tail): (Option<&mut Value>, &mut [Value], &mut [Value]) = match self {
            TerminatorKind::Jump(target) => (None, &mut target.args, &mut []),
            TerminatorKind::CheckException(normal, error) => {
                (None, &mut normal.args, &mut error.args)
            }
            TerminatorKind::Branch(condition, then, otherwise) => {
                (Some(condition), &mut then.args, &mut otherwise.args)
            }
            TerminatorKind::Yield(_, value, target) => (Some(value), &mut target.args, &mut []),
            TerminatorKind::Return(value) => (Some(value), &mut [], &mut []),
        };
        first
            .into_iter()
            .chain(rest.iter_mut())
            .chain(tail.iter_mut())
    }

    pub(super) fn operands(&self) -> Vec<Value> {
        self.operands_iter().collect()
    }

    pub(super) fn operands_iter(&self) -> impl Iterator<Item = Value> + '_ {
        let (first, rest, tail): (Option<Value>, &[Value], &[Value]) = match self {
            TerminatorKind::Jump(target) => (None, &target.args, &[]),
            TerminatorKind::CheckException(normal, error) => (None, &normal.args, &error.args),
            TerminatorKind::Branch(condition, then, otherwise) => {
                (Some(*condition), &then.args, &otherwise.args)
            }
            TerminatorKind::Yield(_, value, target) => (Some(*value), &target.args, &[]),
            TerminatorKind::Return(value) => (Some(*value), &[], &[]),
        };
        first
            .into_iter()
            .chain(rest.iter().copied())
            .chain(tail.iter().copied())
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

/// What a typed-array read may assume: a fixed element kind, and a key proven to be an
/// integral index below the length.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::jit) struct TypedRead {
    pub(in crate::jit) element: Option<TypedKind>,
    pub(in crate::jit) bounded: bool,
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
        let source_name = self.blocks[block]
            .insts
            .get(position)
            .and_then(|&next| self.insts[next].source_name);
        self.insts.push(InstData {
            source_name,
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
                for value in self.insts[inst].op.operands_iter() {
                    counts[value] += 1;
                }
            }
            if let Some(terminator) = &data.terminator {
                for value in terminator.kind.operands_iter() {
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

    fn inst_contract(&self, inst: Inst, facts: &Facts) -> Contract {
        let op = &self.insts[inst].op;
        let mut contract = self.contract(op, facts);
        if let (Op::Call { callee, .. }, Some((expected, flags))) =
            (op, self.closed_calls.get(&inst))
            && callee == expected
        {
            contract.reentry = Reentry::Closed;
            contract.flags = *flags;
        }
        contract
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
        let operands = op.operands_iter().fold(Types::default(), |types, value| {
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
            Op::NewInstance(_) => closed(ALLOCATION | INTERNAL | ERROR, &[], Owned),
            Op::ConstructorResult(..) => closed(0, &[], Borrowed),
            Op::Rest(_) | Op::Arguments => closed(ALLOCATION | INTERNAL | ERROR, &[], Owned),
            Op::Throw(_) => closed(INTERNAL | ERROR, &[], None),
            Op::Catch => closed(INTERNAL, &[], Owned),
            Op::Argument(_) | Op::This => closed(0, &[], Borrowed),
            Op::ClosureThis(_) => closed(HEAP_READ, &[], Borrowed),
            Op::NewHoleyArray(_) => closed(ALLOCATION | INTERNAL | ERROR, &[], Owned),
            Op::MapElement(..) => closed(HEAP_READ | INTERNAL, &[], Borrowed),
            Op::IsArrayIteration(..) | Op::IsStringSplit(_) => closed(HEAP_READ, &[], Unowned),
            Op::StringSplit(..) => closed(HEAP_READ | ALLOCATION | INTERNAL | ERROR, &[], Owned),
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
            Op::ToInt32(_) => closed(0, &[], Unowned),
            Op::ToNumeric(_) | Op::Unary(..) => coercing(MaybeOwned),
            Op::GuardedCompare {
                operator,
                left,
                right,
                ..
            } => self.contract(&Op::Binary(*operator, *left, *right), facts),
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
            | Op::ConstructSpread(..)
            | Op::ConstructorCall { .. }
            | Op::ConstructorCallSpread(..)
            | Op::ConstructResolved { .. }
            | Op::ConstructSpreadResolved(..) => Contract {
                reentry: Reentry::Arbitrary,
                flags: UNKNOWN,
                assumptions: &[],
                result: Owned,
            },
            Op::CallNative(native, _)
                if METHODS[native.index()].1 == crate::heap::MathFunction::Random =>
            {
                closed(INTERNAL, &[], Unowned)
            }
            Op::CallNative(..) => coercing(Unowned),
            Op::IsUndefined(_) | Op::IsNullish(_) | Op::IsNative(..) => closed(0, &[], Unowned),
            Op::ConstructorPrototype(_) => closed(HEAP_READ, &[Assumption::NoAccessors], Borrowed),
            Op::IsCode(..) => closed(HEAP_READ, &[], Unowned),
            Op::ProbeGlobal(_) => closed(GLOBAL_READ, &[Assumption::NoAccessors], Borrowed),
            Op::ProbeProperty(..) | Op::ProbeIndex(..) => {
                closed(HEAP_READ, &[Assumption::NoAccessors], Borrowed)
            }
            Op::GuardValid(_) => closed(GLOBAL_READ, &[], Unowned),
            Op::GuardCommit(_) => closed(INTERNAL, &[], None),
            Op::View(_) => closed(HEAP_READ, &[Assumption::FixedTypedStorage], Unowned),
            Op::IsTyped(..) | Op::IsIndexed(_) => closed(0, &[], Unowned),
            Op::IndexedLength(_) => closed(HEAP_READ, &[], Unowned),
            Op::IndexedGet(..) => closed(HEAP_READ | INTERNAL | ERROR, &[], Borrowed),
            Op::SpreadGet(..) => closed(HEAP_READ | INTERNAL, &[], Borrowed),
            Op::TypedIndex(..) => closed(HEAP_READ, &[Assumption::FixedTypedStorage], Unowned),
            Op::IsNumber(_) | Op::IsString(_) => closed(0, &[], Unowned),
            Op::IsTypeof(..) => closed(HEAP_READ, &[], Unowned),
            Op::AsNumber(_) => closed(INTERNAL, &[], Unowned),
            Op::EnterCaptures(..) => closed(HEAP_READ | INTERNAL | ERROR, &[], None),
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
                    Op::EnterScope(scope) | Op::EnterCaptures(scope, _) => {
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
        let mut positions: SecondaryMap<Inst, Option<usize>> = SecondaryMap::new();
        for (block, data) in &self.blocks {
            for (index, &inst) in data.insts.iter().enumerate() {
                if self.insts[inst].block == block && positions[inst].is_none() {
                    positions[inst] = Some(index);
                }
            }
        }
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
                        let index = positions[inst];
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
                for value in self.insts[inst].op.operands_iter() {
                    check(value, index)?;
                }
            }
            for value in data.terminator.as_ref().unwrap().kind.operands_iter() {
                check(value, data.insts.len())?;
            }
        }
        Ok(())
    }
}

type Result<T> = std::result::Result<T, String>;

fn unsupported<T>(what: &str, span: Span) -> Result<T> {
    Err(format!(
        "unsupported {what} at {}..{}",
        span.start, span.end
    ))
}

pub(super) struct Planned {
    pub(super) inlines: Vec<specialize::Inline>,
    pub(super) reads: Vec<reads::Read>,
    graph: Graph,
    notes: Vec<String>,
    optimize: bool,
    pub(super) conditional: specialize::ConditionalPlan,
}

pub(super) struct Analysis {
    pub(super) conditional: specialize::ConditionalPlan,
    pub(super) inlines: Vec<specialize::Inline>,
    pub(super) reads: Vec<reads::Read>,
    pub(super) graph: Graph,
    facts: Facts,
    notes: Vec<String>,
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

pub(super) fn canonical_for_emission(
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
) -> Result<Canonical> {
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
    canonicalize(graph, ssa_opt)
}

#[cfg(test)]
pub(super) fn analyze(
    function: &oxc_ast::ast::Function<'_>,
    scoping: &Scoping,
    captured: &HashSet<SymbolId>,
) -> Result<Analysis> {
    optimize_graph(build::lower(function, scoping, captured)?, true)
}

#[derive(Clone)]
pub(super) struct Canonical {
    pub(super) graph: Graph,
    notes: Vec<String>,
}

fn canonicalize(mut graph: Graph, ssa_opt: bool) -> Result<Canonical> {
    let mut notes = Vec::new();
    if ssa_opt && !graph.has_handlers && !graph.generator {
        promote::promote(&mut graph, &mut notes);
        graph.verify()?;
        optimize::typeof_comparisons(&mut graph);
        optimize::simplify(&mut graph);
        graph.verify()?;
    }
    Ok(Canonical { graph, notes })
}

fn optimize_graph(graph: Graph, ssa_opt: bool) -> Result<Analysis> {
    optimize_canonical(canonicalize(graph, ssa_opt)?, ssa_opt, None)
}

pub(super) fn optimize_canonical(
    canonical: Canonical,
    ssa_opt: bool,
    specialization: Option<(&specialize::Repository, usize)>,
) -> Result<Analysis> {
    optimize_planned(plan_canonical(canonical, ssa_opt, specialization)?)
}

#[derive(Default)]
struct Expansions {
    arrays: HashSet<Inst>,
    constructors: HashSet<Inst>,
    splits: HashSet<Inst>,
}

impl Expansions {
    fn calls(
        &mut self,
        graph: &mut Graph,
        bodies: &specialize::Repository,
        source: usize,
        notes: &mut Vec<String>,
    ) -> Result<()> {
        arrays::expand(graph, bodies, source, notes, &mut self.arrays)?;
        string_split::expand(graph, &mut self.splits);
        Ok(())
    }

    fn constructors(&mut self, graph: &mut Graph) {
        construction::resolve(graph);
        construction::expand(graph, &mut self.constructors);
    }
}

pub(super) fn plan_canonical(
    canonical: Canonical,
    ssa_opt: bool,
    specialization: Option<(&specialize::Repository, usize)>,
) -> Result<Planned> {
    let Canonical {
        mut graph,
        mut notes,
    } = canonical;
    let optimize = ssa_opt && !graph.has_handlers && !graph.generator;
    if !optimize {
        return Ok(Planned {
            inlines: Vec::new(),
            reads: Vec::new(),
            graph,
            notes,
            optimize,
            conditional: specialize::ConditionalPlan::default(),
        });
    }
    let mut inline_budget = specialize::Budget::default();
    let mut expansions = Expansions::default();
    if let Some((bodies, source)) = specialization {
        expansions.calls(&mut graph, bodies, source, &mut notes)?;
        let requests = specialize::requests(&graph, bodies, source);
        let plan = specialize::plan_round(&graph, bodies, source, &requests, &mut inline_budget);
        specialize::apply(&mut graph, bodies, source, plan, &mut notes)?;
        optimize::simplify(&mut graph);
        expansions.calls(&mut graph, bodies, source, &mut notes)?;
        graph.verify()?;
    }
    expansions.constructors(&mut graph);
    if let Some((bodies, source)) = specialization {
        for _ in 0..3 {
            let requests = specialize::requests(&graph, bodies, source);
            let plan =
                specialize::plan_round(&graph, bodies, source, &requests, &mut inline_budget);
            if plan.is_empty() {
                break;
            }
            specialize::apply(&mut graph, bodies, source, plan, &mut notes)?;
            optimize::simplify(&mut graph);
            expansions.calls(&mut graph, bodies, source, &mut notes)?;
            expansions.constructors(&mut graph);
            graph.verify()?;
        }
    }
    if let Some((bodies, _)) = specialization {
        specialize::prove_closed_calls(&mut graph, bodies, &mut notes);
    }
    let reads = specialization.map_or_else(Vec::new, |(bodies, source)| {
        reads::plan(&graph, bodies, source)
    });
    let conditional = specialization
        .map_or_else(specialize::ConditionalPlan::default, |(bodies, _)| {
            specialize::conditional_plan(&graph, bodies, &mut notes)
        });
    Ok(Planned {
        conditional,
        inlines: inline_budget.inlines,
        reads,
        graph,
        notes,
        optimize,
    })
}

pub(super) fn optimize_planned(planned: Planned) -> Result<Analysis> {
    let Planned {
        inlines,
        reads,
        mut graph,
        mut notes,
        optimize,
        conditional,
    } = planned;
    if !optimize {
        let facts = Facts::compute(&graph);
        return Ok(Analysis {
            conditional,
            inlines,
            reads,
            graph,
            facts,
            notes,
        });
    }
    reads::apply(&mut graph, &reads, &mut notes)?;
    comparisons::expose(&mut graph);
    optimize::sink_closures(&mut graph, &mut notes);
    if optimize::lazy_captures(&mut graph, &mut notes) {
        promote::promote(&mut graph, &mut notes);
        optimize::simplify(&mut graph);
    }
    graph.verify()?;
    let mut regions = optimize::version(&mut graph, &mut notes)?;
    if regions.is_empty()
        && let Some(region) = optimize::version_conditional(&mut graph, &conditional, &mut notes)?
    {
        regions.push(region);
    }
    graph.verify()?;
    if regions.is_empty() {
        regions.extend(optimize::version_string_leaves(&mut graph, &mut notes)?);
    }
    graph.verify()?;
    optimize::simplify(&mut graph);
    graph.verify()?;
    for region in &regions {
        optimize::reuse_loads(&mut graph, region, &mut notes);
        graph.verify()?;
        optimize::hoist(&mut graph, region, &mut notes);
        graph.verify()?;
    }
    let copies = optimize::check_loops(&mut graph, &mut regions, &mut notes)?;
    optimize::coercions(&mut graph);
    optimize::bit_tests(&mut graph);
    graph.verify()?;
    for region in regions.iter().chain(&copies) {
        optimize::reuse_loads(&mut graph, region, &mut notes);
        graph.verify()?;
        optimize::hoist(&mut graph, region, &mut notes);
        graph.verify()?;
    }
    optimize::views(&mut graph, &mut notes);
    comparisons::expose(&mut graph);
    graph.verify()?;
    optimize::simplify(&mut graph);
    graph.verify()?;
    optimize::sink_closures(&mut graph, &mut notes);
    optimize::lazy_captures(&mut graph, &mut notes);
    graph.verify()?;
    let facts = Facts::compute(&graph);
    borrow::borrows(&mut graph, &facts, &mut notes);
    Ok(Analysis {
        conditional,
        inlines,
        reads,
        graph,
        facts,
        notes,
    })
}

#[cfg(test)]
mod tests;

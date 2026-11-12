mod access;
mod arrays;
mod available;
mod borrow;
pub(in crate::jit) mod boundary;
mod build;
mod caches;
pub(in crate::jit) mod call_variants;
mod comparisons;
mod construction;
pub(in crate::jit) mod continuation;
mod dispatch;
pub(in crate::jit) mod facts;
mod fresh;
mod guards;
mod inlining;
mod js;
mod listing;
pub(in crate::jit) mod low_level;
mod memory_lowering;
pub(in crate::jit) use memory_lowering::lower_memory_for_cranelift;
mod numeric_lowering;
mod observations;
mod optimize;
pub(super) mod osr;
pub(super) mod reads;
mod shapes;
pub(super) mod specialize;
mod string_split;
pub(in crate::jit) mod suspension;
mod transforms;
pub(in crate::jit) use transforms::canonicalize_parameters;
mod typed_lowering;

use std::collections::{HashMap, HashSet};
use std::fmt;

use cranelift_codegen::entity::{EntityRef, PrimaryMap, SecondaryMap};
use oxc_ast::ast::{Expression, FormalParameters, Statement};
use oxc_semantic::Scoping;
use oxc_span::Span;
use oxc_syntax::operator::BinaryOperator;
use oxc_syntax::symbol::SymbolId;

use super::effects::{
    ALLOCATION, CODE_READ, CODE_WRITE, Control, ERROR, Effects, GLOBAL_READ, GLOBAL_WRITE,
    HEAP_READ, HEAP_WRITE, INTERNAL, Memory,
};
use super::math::METHODS;
use crate::heap::TypedKind;

pub(super) use borrow::receiver as borrowed_receiver;
pub(in crate::jit) use facts::{Facts, Identity, Types};
pub(super) use ids::{Binding, Block, Inst, RegionId, Scope, Value};
pub(in crate::jit) use listing::ListingLine;
#[cfg(test)]
pub(in crate::jit) use listing::SourceLocation;

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
    pub(in crate::jit) struct RegionId(u32);
    entity_impl!(RegionId, "region");

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
pub(in crate::jit) struct CallSummary {
    pub(super) memory: Memory,
    pub(super) control: Control,
    pub(super) requires: Vec<(String, cranelift_module::FuncId)>,
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
    pub(in crate::jit) summaries: std::sync::Arc<HashMap<cranelift_module::FuncId, CallSummary>>,
    pub(in crate::jit) cold: HashSet<Block>,
    pub(in crate::jit) megamorphic_gets: HashSet<super::profile::Site>,
    pub(super) lazy_captures: HashSet<Binding>,
    pub(super) generator: bool,
    pub(super) has_assumes: bool,
    // Input checks reported by the shared guard emitters for call specialization.
    pub(super) entry_guards: crate::jit::variants::Requirements,
    /// Parameter count of a function whose entry takes its arguments in registers; `None` when
    /// it reads `arguments` or a rest parameter, or is a generator.
    pub(in crate::jit) arity: Option<u16>,
    pub(super) regions: PrimaryMap<RegionId, Region>,
    pub(super) blocks: PrimaryMap<Block, BlockData>,
    pub(super) insts: PrimaryMap<Inst, InstData>,
    pub(super) values: PrimaryMap<Value, ValueDef>,
    pub(super) low_level_types: SecondaryMap<Value, Option<cranelift_codegen::ir::Type>>,
    pub(super) low_level_slots: Vec<low_level::StackSlotData>,
    pub(super) low_level_signature: Option<cranelift_codegen::ir::Signature>,
    pub(super) bindings: PrimaryMap<Binding, BindingData>,
    pub(super) scopes: PrimaryMap<Scope, ScopeData>,
}

#[derive(Clone)]
pub(super) struct Region {
    pub(super) blocks: HashSet<Block>,
    pub(super) fallback: Option<RegionFallback>,
}

#[derive(Clone, Copy)]
pub(super) enum RegionFallback {
    Region(RegionId),
    CompiledEntry(usize),
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
    pub(super) emission_location: Option<u32>,
    pub(super) span: Span,
    pub(super) block: Block,
    pub(super) results: Vec<Value>,
    pub(super) op: Op,
}

impl InstData {
    pub(super) fn result_at(&self, index: usize) -> Option<Value> {
        self.results.get(index).copied()
    }
    pub(super) fn result(&self) -> Option<Value> {
        match self.results.as_slice() {
            [] => None,
            [value] => Some(*value),
            _ => panic!("expected at most one result"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ValueDef {
    Result(Inst, usize),
    Param(Block, usize),
}

#[derive(Clone)]
pub(super) struct BindingData {
    pub(super) name: String,
    pub(super) kind: BindingKind,
    pub(super) scope: Scope,
    pub(super) captured: bool,
    // Region boundaries preserve this binding's representation, including capture cells.
    pub(super) persistent: bool,
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
    ForEach,
    Find,
    Push,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::jit) struct DirectTarget {
    pub(in crate::jit) variant: crate::jit::entries::VariantId,
    pub(in crate::jit) code: cranelift_module::FuncId,
    pub(in crate::jit) arity: u16,
    pub(in crate::jit) lexical_this: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in crate::jit) enum MemoryFormat {
    U8,
    U16,
    U32,
    I8,
    I16,
    I32,
    F32,
    F64,
}

impl MemoryFormat {
    pub(in crate::jit) fn bytes(self) -> u32 {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::jit) enum MemoryOp {
    IndexedLoad {
        base: Value,
        index: Value,
        format: MemoryFormat,
    },
    Address {
        base: Value,
        index: Value,
        format: MemoryFormat,
    },
    Load {
        address: Value,
        format: MemoryFormat,
    },
}

#[derive(Clone, Debug)]
pub(super) enum Op {
    LowLevel(low_level::Op),
    Observe(Value, super::profile::Observation, super::profile::Site),
    ReadHint(
        Value,
        super::profile::ReadKind,
        super::profile::ReadType,
        super::profile::Site,
    ),
    Assume(Value, guards::Fact, guards::Origin),
    // A caller-established fact, unlike the speculative proposals in Assume.
    EntryAssume(Value, guards::Fact),
    OsrStart(std::sync::Arc<osr::Plan>, usize, Vec<Value>),
    Continuation(std::sync::Arc<continuation::Body>, Vec<Value>),
    ContinuationOutput(Value, usize),
    ContinuationStore(Vec<Value>),
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
    BindingLoad(Binding),
    BindingRestore(Binding, Value),
    Initialize(Binding, Value),
    Write(Binding, Value),
    CheckInitialized(Value),
    NewObject,
    NewArray(Vec<Value>),
    NewHoleyArray(Value),
    MapElement(Value, Value),
    IsArrayIteration(Value, Value, ArrayMethod),
    ArrayAppend(Value, Value, bool),
    NewPack,
    Spread(Value),
    PackRelease(Value),
    PackAppend(Value, Value, bool),
    CallPack(Value, Value, Value),
    ConstructPack(Value, Value),
    // UNINITIALIZED leaves native constructors and lazy prototypes to generic construction.
    ConstructorPrototype(Value),
    ConstructPackResolved(Value, Value, Value),
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
    ConstructorCallPack(Value, Value, Value),
    IteratorNew(Value),
    IteratorNext(Value),
    IsIndexed(Value),
    // Saturating numeric conversion; validity is checked separately by IsArrayIndex or IsExactPosition.
    ArrayPosition(Value),
    // Compare a numeric key with its integer position converted back to a number.
    IsExactPosition(Value, Value),
    // Compare integer position and length as unsigned values.
    IsInBounds(Value, Value),
    IsArrayIndex(Value, Value, Value, bool),
    IsArray(Value, bool),
    // Non-short-circuit conjunction of boolean operands.
    BooleanAnd(Value, Value),
    // Requires matching layout, exact position and bounds checks with no intervening array mutation.
    ArrayLoad(Value, Value, bool),
    IndexedLength(Value),
    ArrayKind(Value),
    IndexedGet(Value, Value),
    // Extracted values stay rooted independently of the pack across argument initialization.
    PackTake(Value, Value),
    EnumerateNew(Value),
    EnumerateNext(Value),
    IsUninitialized(Value),
    Define(Value, String, Value),
    DefineHidden(Value, String, Value),
    MarkClass(Value),
    DisableConstructor(Value),
    Get(Value, String),
    IsPropertyCached(Value, Inst, String),
    CachedProperty(Value, Inst),
    IsShape(Value, u32),
    ShapedGet(Value, String, u32),
    ShapedSet(Value, String, u32, Value),
    FillPropertyCache(Value, Inst, String),
    PrimePropertyCache(Value, Inst, String),
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
        target: Option<DirectTarget>,
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
    ProbeShapeProperty(Value, String, u32, u32),
    ProbeIndex(Value, Value),
    IsNative(Value, Native),
    IsCode(Value, cranelift_module::FuncId),
    GuardValid(usize),
    GuardCommit(usize),
    View(Value),
    IsTyped(Value, Option<TypedKind>),
    TypedIndex(Value, Value, TypedRead),
    TypedData(Value),
    Memory(MemoryOp),
    MemoryNumber(Value, MemoryFormat),
    KeepAlive(Value),
    IsNumber(Value),
    IsInteger(Value),
    AsInteger(Value),
    NumericFloat(Value),
    SaturatingInteger(Value),
    IntegerCheck(Value, Value, Value),
    IsString(Value),
    IsStringSplit(Value),
    StringSplit(Value, Vec<Value>),
    StringSplitParts(Value, Vec<Value>),
    IsTypeof(Value, u8),
    AsNumber(Value),
    AsObject(Value),
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
    Switch(Value, Vec<(u128, Target)>, Target),
    Return(Value),
    ReturnValues(Vec<Value>),
    TailCall {
        target: low_level::CallTarget,
        signature: cranelift_codegen::ir::Signature,
        args: Vec<Value>,
    },
    Fallback(usize),
}

#[derive(Clone)]
pub(super) struct Terminator {
    pub(super) source_name: Option<(usize, usize)>,
    pub(super) emission_location: Option<u32>,
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
    pub(super) effects: Effects,
    pub(super) result: Ownership,
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
            Op::LowLevel(low_level::Op::Call { target, args, .. }) => (
                [
                    match target {
                        low_level::CallTarget::Indirect(value) => Some(*value),
                        _ => None,
                    },
                    None,
                    None,
                ],
                args,
            ),
            Op::LowLevel(
                low_level::Op::Unary { value, .. }
                | low_level::Op::Load { address: value, .. }
                | low_level::Op::AtomicLoad { address: value, .. },
            ) => ([Some(*value), None, None], &[]),
            Op::LowLevel(
                low_level::Op::Binary { left, right, .. }
                | low_level::Op::IntCompare { left, right, .. }
                | low_level::Op::FloatCompare { left, right, .. }
                | low_level::Op::Store {
                    value: left,
                    address: right,
                    ..
                },
            ) => ([Some(*left), Some(*right), None], &[]),
            Op::LowLevel(low_level::Op::Select { condition, yes, no }) => {
                ([Some(*condition), Some(*yes), Some(*no)], &[])
            }
            Op::LowLevel(
                low_level::Op::Integer { .. }
                | low_level::Op::Float64(_)
                | low_level::Op::StackAddress { .. }
                | low_level::Op::StackPointer { .. },
            ) => ([None; 3], &[]),
            Op::LowLevel(low_level::Op::VariableRead) => ([None; 3], &[]),
            Op::OsrStart(_, _, values)
            | Op::Continuation(_, values)
            | Op::ContinuationStore(values) => ([None, None, None], values),
            Op::ContinuationOutput(value, _)
            | Op::Observe(value, ..)
            | Op::ReadHint(value, ..)
            | Op::Assume(value, ..)
            | Op::EntryAssume(value, _) => ([Some(*value), None, None], &[]),
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
            | Op::BindingLoad(_)
            | Op::NewPack
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
            | Op::BindingRestore(_, value)
            | Op::Write(_, value)
            | Op::CheckInitialized(value)
            | Op::PackRelease(value)
            | Op::Spread(value)
            | Op::IteratorNew(value)
            | Op::IteratorNext(value)
            | Op::IsIndexed(value)
            | Op::ArrayKind(value)
            | Op::IndexedLength(value)
            | Op::EnumerateNew(value)
            | Op::EnumerateNext(value)
            | Op::IsUninitialized(value)
            | Op::Get(value, _)
            | Op::IsPropertyCached(value, _, _)
            | Op::CachedProperty(value, _)
            | Op::IsShape(value, _)
            | Op::ShapedGet(value, _, _)
            | Op::FillPropertyCache(value, _, _)
            | Op::PrimePropertyCache(value, _, _)
            | Op::MarkClass(value)
            | Op::DisableConstructor(value)
            | Op::PropertyKey(value)
            | Op::ToNumeric(value)
            | Op::ToInt32(value)
            | Op::ArrayPosition(value)
            | Op::Unary(_, value)
            | Op::IsUndefined(value)
            | Op::IsNullish(value)
            | Op::ProbeProperty(value, _)
            | Op::ProbeShapeProperty(value, _, _, _)
            | Op::IsNative(value, _)
            | Op::NewInstance(value)
            | Op::ConstructorPrototype(value)
            | Op::IsCode(value, _)
            | Op::View(value)
            | Op::TypedData(value)
            | Op::MemoryNumber(value, _)
            | Op::KeepAlive(value)
            | Op::Memory(MemoryOp::Load { address: value, .. })
            | Op::IsArray(value, _)
            | Op::IsTyped(value, _)
            | Op::IsNumber(value)
            | Op::IsInteger(value)
            | Op::AsInteger(value)
            | Op::NumericFloat(value)
            | Op::SaturatingInteger(value)
            | Op::IsString(value)
            | Op::IsStringSplit(value)
            | Op::IsTypeof(value, _)
            | Op::AsObject(value)
            | Op::AsNumber(value) => ([Some(*value), None, None], &[]),
            Op::Define(object, _, value)
            | Op::DefineHidden(object, _, value)
            | Op::ShapedSet(object, _, _, value)
            | Op::Set(object, _, value) => ([Some(*object), Some(*value), None], &[]),
            Op::Memory(MemoryOp::IndexedLoad {
                base: object,
                index: key,
                ..
            })
            | Op::Memory(MemoryOp::Address {
                base: object,
                index: key,
                ..
            })
            | Op::MapElement(object, key)
            | Op::IsArrayIteration(object, key, _)
            | Op::GetIndex(object, key)
            | Op::TypedIndex(object, key, _)
            | Op::IndexedGet(object, key)
            | Op::ArrayLoad(object, key, _)
            | Op::PackTake(object, key)
            | Op::ProbeIndex(object, key)
            | Op::IsExactPosition(object, key)
            | Op::IsInBounds(object, key)
            | Op::BooleanAnd(object, key)
            | Op::Binary(_, object, key)
            | Op::PackAppend(object, key, _)
            | Op::ArrayAppend(object, key, _)
            | Op::ConstructorResult(object, key)
            | Op::ConstructPack(object, key) => ([Some(*object), Some(*key), None], &[]),
            Op::IntegerCheck(object, key, position)
            | Op::IsArrayIndex(object, key, position, _) => {
                ([Some(*object), Some(*key), Some(*position)], &[])
            }
            Op::CallPack(callee, receiver, args)
            | Op::ConstructorCallPack(callee, receiver, args)
            | Op::ConstructPackResolved(callee, receiver, args) => {
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
            Op::StringSplit(receiver, args) | Op::StringSplitParts(receiver, args) => {
                ([Some(*receiver), None, None], args)
            }
            Op::Construct { callee, args } => ([Some(*callee), None, None], args),
            Op::Call {
                callee,
                receiver,
                args,
                ..
            } => ([Some(*callee), *receiver, None], args),
        };
        fixed.into_iter().flatten().chain(rest.iter().copied())
    }

    pub(super) fn operands_mut(&mut self) -> impl Iterator<Item = &mut Value> + '_ {
        let (fixed, rest): ([Option<&mut Value>; 3], &mut [Value]) = match self {
            Op::LowLevel(low_level::Op::Call { target, args, .. }) => (
                [
                    match target {
                        low_level::CallTarget::Indirect(value) => Some(value),
                        _ => None,
                    },
                    None,
                    None,
                ],
                args,
            ),
            Op::LowLevel(
                low_level::Op::Unary { value, .. }
                | low_level::Op::Load { address: value, .. }
                | low_level::Op::AtomicLoad { address: value, .. },
            ) => ([Some(value), None, None], &mut []),
            Op::LowLevel(
                low_level::Op::Binary { left, right, .. }
                | low_level::Op::IntCompare { left, right, .. }
                | low_level::Op::FloatCompare { left, right, .. }
                | low_level::Op::Store {
                    value: left,
                    address: right,
                    ..
                },
            ) => ([Some(left), Some(right), None], &mut []),
            Op::LowLevel(low_level::Op::Select { condition, yes, no }) => {
                ([Some(condition), Some(yes), Some(no)], &mut [])
            }
            Op::LowLevel(
                low_level::Op::Integer { .. }
                | low_level::Op::Float64(_)
                | low_level::Op::StackAddress { .. }
                | low_level::Op::StackPointer { .. },
            ) => ([None, None, None], &mut []),
            Op::LowLevel(low_level::Op::VariableRead) => ([None, None, None], &mut []),
            Op::OsrStart(_, _, values)
            | Op::Continuation(_, values)
            | Op::ContinuationStore(values) => ([None, None, None], values),
            Op::ContinuationOutput(value, _)
            | Op::Observe(value, ..)
            | Op::ReadHint(value, ..)
            | Op::Assume(value, ..)
            | Op::EntryAssume(value, _) => ([Some(value), None, None], &mut []),
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
            | Op::BindingLoad(_)
            | Op::NewPack
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
            | Op::BindingRestore(_, value)
            | Op::Write(_, value)
            | Op::CheckInitialized(value)
            | Op::PackRelease(value)
            | Op::Spread(value)
            | Op::IteratorNew(value)
            | Op::IteratorNext(value)
            | Op::IsIndexed(value)
            | Op::ArrayKind(value)
            | Op::IndexedLength(value)
            | Op::EnumerateNew(value)
            | Op::EnumerateNext(value)
            | Op::IsUninitialized(value)
            | Op::Get(value, _)
            | Op::IsPropertyCached(value, _, _)
            | Op::CachedProperty(value, _)
            | Op::IsShape(value, _)
            | Op::ShapedGet(value, _, _)
            | Op::FillPropertyCache(value, _, _)
            | Op::PrimePropertyCache(value, _, _)
            | Op::MarkClass(value)
            | Op::DisableConstructor(value)
            | Op::PropertyKey(value)
            | Op::ToNumeric(value)
            | Op::ToInt32(value)
            | Op::ArrayPosition(value)
            | Op::Unary(_, value)
            | Op::IsUndefined(value)
            | Op::IsNullish(value)
            | Op::ProbeProperty(value, _)
            | Op::ProbeShapeProperty(value, _, _, _)
            | Op::IsNative(value, _)
            | Op::NewInstance(value)
            | Op::ConstructorPrototype(value)
            | Op::IsCode(value, _)
            | Op::View(value)
            | Op::TypedData(value)
            | Op::MemoryNumber(value, _)
            | Op::KeepAlive(value)
            | Op::Memory(MemoryOp::Load { address: value, .. })
            | Op::IsArray(value, _)
            | Op::IsTyped(value, _)
            | Op::IsNumber(value)
            | Op::IsInteger(value)
            | Op::AsInteger(value)
            | Op::NumericFloat(value)
            | Op::SaturatingInteger(value)
            | Op::IsString(value)
            | Op::IsStringSplit(value)
            | Op::IsTypeof(value, _)
            | Op::AsObject(value)
            | Op::AsNumber(value) => ([Some(value), None, None], &mut []),
            Op::Define(object, _, value)
            | Op::DefineHidden(object, _, value)
            | Op::ShapedSet(object, _, _, value)
            | Op::Set(object, _, value) => ([Some(object), Some(value), None], &mut []),
            Op::Memory(MemoryOp::IndexedLoad {
                base: object,
                index: key,
                ..
            })
            | Op::Memory(MemoryOp::Address {
                base: object,
                index: key,
                ..
            })
            | Op::MapElement(object, key)
            | Op::IsArrayIteration(object, key, _)
            | Op::GetIndex(object, key)
            | Op::TypedIndex(object, key, _)
            | Op::IndexedGet(object, key)
            | Op::ArrayLoad(object, key, _)
            | Op::PackTake(object, key)
            | Op::ProbeIndex(object, key)
            | Op::IsExactPosition(object, key)
            | Op::IsInBounds(object, key)
            | Op::BooleanAnd(object, key)
            | Op::Binary(_, object, key)
            | Op::PackAppend(object, key, _)
            | Op::ArrayAppend(object, key, _)
            | Op::ConstructorResult(object, key)
            | Op::ConstructPack(object, key) => ([Some(object), Some(key), None], &mut []),
            Op::IntegerCheck(object, key, position)
            | Op::IsArrayIndex(object, key, position, _) => {
                ([Some(object), Some(key), Some(position)], &mut [])
            }
            Op::CallPack(callee, receiver, args)
            | Op::ConstructorCallPack(callee, receiver, args)
            | Op::ConstructPackResolved(callee, receiver, args) => {
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
            Op::StringSplit(receiver, args) | Op::StringSplitParts(receiver, args) => {
                ([Some(receiver), None, None], args)
            }
            Op::Construct { callee, args } => ([Some(callee), None, None], args),
            Op::Call {
                callee,
                receiver,
                args,
                ..
            } => ([Some(callee), receiver.as_mut(), None], args),
        };
        fixed.into_iter().flatten().chain(rest.iter_mut())
    }

    pub(super) fn write_target(&self) -> Option<Value> {
        match self {
            Op::Define(object, ..)
            | Op::Set(object, ..)
            | Op::ShapedSet(object, ..)
            | Op::SetIndex(object, ..)
            | Op::ArrayAppend(object, ..)
            | Op::IteratorNext(object)
            | Op::EnumerateNext(object) => Some(*object),
            _ => None,
        }
    }
}

impl TerminatorKind {
    pub(super) fn operands_mut(&mut self) -> impl Iterator<Item = &mut Value> + '_ {
        let (first, rest, tail, cases): (
            Option<&mut Value>,
            &mut [Value],
            &mut [Value],
            Option<&mut Vec<(u128, Target)>>,
        ) = match self {
            TerminatorKind::Jump(target) => (None, &mut target.args, &mut [], None),
            TerminatorKind::CheckException(normal, error) => {
                (None, &mut normal.args, &mut error.args, None)
            }
            TerminatorKind::Branch(condition, yes, no) => {
                (Some(condition), &mut yes.args, &mut no.args, None)
            }
            TerminatorKind::Switch(value, cases, default) => {
                (Some(value), &mut default.args, &mut [], Some(cases))
            }
            TerminatorKind::Yield(_, value, target) => {
                (Some(value), &mut target.args, &mut [], None)
            }
            TerminatorKind::Return(value) => (Some(value), &mut [], &mut [], None),
            TerminatorKind::ReturnValues(values) => (None, values, &mut [], None),
            TerminatorKind::TailCall { target, args, .. } => (
                match target {
                    low_level::CallTarget::Indirect(value) => Some(value),
                    _ => None,
                },
                args,
                &mut [],
                None,
            ),
            TerminatorKind::Fallback(_) => (None, &mut [], &mut [], None),
        };
        first
            .into_iter()
            .chain(rest.iter_mut())
            .chain(tail.iter_mut())
            .chain(cases.into_iter().flat_map(|cases| {
                cases
                    .iter_mut()
                    .flat_map(|(_, target)| target.args.iter_mut())
            }))
    }

    pub(super) fn operands(&self) -> Vec<Value> {
        self.operands_iter().collect()
    }

    pub(super) fn operands_iter(&self) -> impl Iterator<Item = Value> + '_ {
        let (first, rest, tail, cases): (
            Option<Value>,
            &[Value],
            &[Value],
            Option<&[(u128, Target)]>,
        ) = match self {
            TerminatorKind::Jump(target) => (None, &target.args, &[], None),
            TerminatorKind::CheckException(normal, error) => {
                (None, &normal.args, &error.args, None)
            }
            TerminatorKind::Branch(condition, yes, no) => {
                (Some(*condition), &yes.args, &no.args, None)
            }
            TerminatorKind::Switch(value, cases, default) => {
                (Some(*value), &default.args, &[], Some(cases))
            }
            TerminatorKind::Yield(_, value, target) => (Some(*value), &target.args, &[], None),
            TerminatorKind::Return(value) => (Some(*value), &[], &[], None),
            TerminatorKind::ReturnValues(values) => (None, values, &[], None),
            TerminatorKind::TailCall { target, args, .. } => (
                match target {
                    low_level::CallTarget::Indirect(value) => Some(*value),
                    _ => None,
                },
                args,
                &[],
                None,
            ),
            TerminatorKind::Fallback(_) => (None, &[], &[], None),
        };
        first
            .into_iter()
            .chain(rest.iter().copied())
            .chain(tail.iter().copied())
            .chain(cases.into_iter().flat_map(|cases| {
                cases
                    .iter()
                    .flat_map(|(_, target)| target.args.iter().copied())
            }))
    }

    pub(super) fn targets(&self) -> impl Iterator<Item = &Target> {
        let (first, second, cases): (Option<&Target>, Option<&Target>, Option<&[(u128, Target)]>) =
            match self {
                TerminatorKind::Jump(target) => (Some(target), None, None),
                TerminatorKind::CheckException(normal, error) => (Some(normal), Some(error), None),
                TerminatorKind::Branch(_, yes, no) => (Some(yes), Some(no), None),
                TerminatorKind::Switch(_, cases, default) => (Some(default), None, Some(cases)),
                TerminatorKind::Yield(_, _, target) => (Some(target), None, None),
                TerminatorKind::Return(_)
                | TerminatorKind::ReturnValues(_)
                | TerminatorKind::TailCall { .. }
                | TerminatorKind::Fallback(_) => (None, None, None),
            };
        first.into_iter().chain(second).chain(
            cases
                .into_iter()
                .flat_map(|cases| cases.iter().map(|(_, target)| target)),
        )
    }

    pub(super) fn targets_mut(&mut self) -> Vec<&mut Target> {
        match self {
            TerminatorKind::Jump(target) => vec![target],
            TerminatorKind::CheckException(normal, error) => vec![normal, error],
            TerminatorKind::Branch(_, yes, no) => vec![yes, no],
            TerminatorKind::Switch(_, cases, default) => std::iter::once(default)
                .chain(cases.iter_mut().map(|(_, target)| target))
                .collect(),
            TerminatorKind::Yield(_, _, target) => vec![target],
            TerminatorKind::Return(_)
            | TerminatorKind::ReturnValues(_)
            | TerminatorKind::TailCall { .. }
            | TerminatorKind::Fallback(_) => Vec::new(),
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

#[derive(Clone, Default)]
pub(super) struct Cfg {
    pub(super) rpo: Vec<Block>,
    pub(super) preds: SecondaryMap<Block, Vec<Block>>,
    dominators: SecondaryMap<Block, Option<Block>>,
}

impl Cfg {
    pub(super) fn new(graph: &Graph) -> Self {
        Self::from_entry(graph, graph.entry)
    }

    pub(super) fn from_entry(graph: &Graph, entry: Block) -> Self {
        let mut visited = vec![false; graph.blocks.len()];
        visited[entry.index()] = true;
        let mut postorder = Vec::new();
        let mut stack = vec![(entry, 0)];
        while let Some((block, index)) = stack.pop() {
            if let Some(next) = graph.successors(block).nth(index) {
                stack.push((block, index + 1));
                if !visited[next.index()] {
                    visited[next.index()] = true;
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
        dominators[entry] = Some(entry);
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
    pub(super) fn new_low_level(
        name: String,
        span: Span,
        signature: cranelift_codegen::ir::Signature,
    ) -> Self {
        let mut blocks = PrimaryMap::new();
        let entry = blocks.push(BlockData::default());
        Self {
            name,
            span,
            entry,
            has_handlers: false,
            guards: 0,
            views: HashMap::new(),
            borrowed: HashSet::new(),
            summaries: Default::default(),
            cold: HashSet::new(),
            megamorphic_gets: HashSet::new(),
            lazy_captures: HashSet::new(),
            generator: false,
            has_assumes: false,
            entry_guards: Vec::new(),
            arity: None,
            regions: PrimaryMap::new(),
            blocks,
            insts: PrimaryMap::new(),
            values: PrimaryMap::new(),
            low_level_types: SecondaryMap::new(),
            low_level_slots: Vec::new(),
            low_level_signature: Some(signature),
            bindings: PrimaryMap::new(),
            scopes: PrimaryMap::new(),
        }
    }

    pub(super) fn successors(&self, block: Block) -> impl Iterator<Item = Block> + use<> {
        let mut targets = self.blocks[block]
            .terminator
            .iter()
            .flat_map(|terminator| terminator.kind.targets());
        let first = targets.next().map(|target| target.block);
        let second = targets.next().map(|target| target.block);
        let mut rest = Vec::new();
        for target in targets {
            let block = target.block;
            if Some(block) != first && Some(block) != second && !rest.contains(&block) {
                rest.push(block);
            }
        }
        first
            .into_iter()
            .chain(second.filter(|block| Some(*block) != first))
            .chain(rest)
    }

    pub(super) fn def_block(&self, value: Value) -> Block {
        match self.values[value] {
            ValueDef::Result(inst, _) => self.insts[inst].block,
            ValueDef::Param(block, _) => block,
        }
    }

    pub(super) fn defining_op(&self, value: Value) -> Option<&Op> {
        match self.values[value] {
            ValueDef::Result(inst, _) => Some(&self.insts[inst].op),
            ValueDef::Param(..) => None,
        }
    }

    pub(in crate::jit) fn callees(&self) -> HashSet<Value> {
        self.insts
            .values()
            .filter_map(|data| match data.op {
                Op::Call { callee, .. } | Op::CallPack(callee, ..) => Some(callee),
                _ => None,
            })
            .collect()
    }

    pub(super) fn caller_rooted(&self, value: Value) -> bool {
        match self.defining_op(value) {
            Some(Op::This | Op::Argument(_)) => true,
            Some(Op::AsObject(source)) => self.caller_rooted(*source),
            _ => false,
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
        let result = has_result.then(|| self.values.push(ValueDef::Result(inst, 0)));
        let source_name = self.blocks[block]
            .insts
            .get(position)
            .and_then(|&next| self.insts[next].source_name);
        self.has_assumes |= matches!(op, Op::Assume(..));
        self.insts.push(InstData {
            source_name,
            emission_location: None,
            span,
            block,
            results: result.into_iter().collect(),
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

    pub(super) fn insert_value(
        &mut self,
        block: Block,
        position: usize,
        span: Span,
        op: Op,
    ) -> Value {
        self.insert(block, position, span, op, true).1.unwrap()
    }

    pub(super) fn append_results(
        &mut self,
        block: Block,
        span: Span,
        op: Op,
        count: usize,
    ) -> (Inst, Vec<Value>) {
        if count == 0 {
            return (self.append(block, span, op, false).0, Vec::new());
        }
        let (inst, first) = self.append(block, span, op, true);
        let mut results = vec![first.unwrap()];
        for index in 1..count {
            let value = self.values.push(ValueDef::Result(inst, index));
            self.insts[inst].results.push(value);
            results.push(value);
        }
        (inst, results)
    }

    pub(super) fn append_value(&mut self, block: Block, span: Span, op: Op) -> Value {
        self.append(block, span, op, true).1.unwrap()
    }

    pub(super) fn terminate(&mut self, block: Block, span: Span, kind: TerminatorKind) {
        self.blocks[block].terminator = Some(Terminator {
            source_name: None,
            emission_location: None,
            span,
            kind,
        });
    }

    pub(super) fn remove_inst(&mut self, inst: Inst) {
        self.detach_inst(inst);
        if !self.has_assumes {
            return;
        }
        let results = self.insts[inst].results.clone();
        for block in self.blocks.keys().collect::<Vec<_>>() {
            let insts = std::mem::take(&mut self.blocks[block].insts);
            self.blocks[block].insts = insts
                .into_iter()
                .filter(|&other| {
                    !matches!(self.insts[other].op, Op::Assume(value, ..) if results.contains(&value))
                })
                .collect();
        }
    }

    pub(super) fn strip_assumes(&mut self) {
        if !self.has_assumes {
            return;
        }
        for block in self.blocks.keys().collect::<Vec<_>>() {
            let insts = std::mem::take(&mut self.blocks[block].insts);
            self.blocks[block].insts = insts
                .into_iter()
                .filter(|&inst| !matches!(self.insts[inst].op, Op::Assume(..)))
                .collect();
        }
        self.has_assumes = false;
    }

    pub(super) fn detach_inst(&mut self, inst: Inst) {
        let block = self.insts[inst].block;
        self.blocks[block].insts.retain(|&i| i != inst);
        self.views.remove(&inst);
    }

    pub(super) fn move_inst(&mut self, inst: Inst, block: Block) {
        let view = self.views.get(&inst).copied();
        self.detach_inst(inst);
        if let Some(view) = view {
            self.views.insert(inst, view);
        }
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
        for view in self.views.values_mut() {
            *view = resolve(aliases, *view);
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
                if matches!(self.insts[inst].op, Op::Assume(..)) {
                    continue;
                }
                for value in self.insts[inst].op.operands_iter() {
                    counts[value] += 1;
                }
                if let Some(&view) = self.views.get(&inst) {
                    counts[view] += 1;
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
            if !data.bindings.iter().any(|&binding| {
                self.bindings[binding].captured || self.bindings[binding].persistent
            }) {
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
        if let Op::Call { callee, .. } = op
            && let Some(summary) = self.call_summary(inst, *callee, facts)
        {
            contract.effects.memory = summary.memory;
            contract.effects.control = summary.control;
        }
        contract
    }

    fn call_summary(&self, inst: Inst, callee: Value, facts: &Facts) -> Option<&CallSummary> {
        if self.summaries.is_empty() {
            return None;
        }
        let block = self.insts[inst].block;
        let Identity::Code(code) = facts.identity_at(callee, block) else {
            return None;
        };
        let summary = self.summaries.get(&code)?;
        let guarded = facts.assumes(callee)
            || summary
                .requires
                .iter()
                .all(|(name, target)| facts.guarded_global(name, *target, inst, block));
        guarded.then_some(summary)
    }

    pub(super) fn contract(&self, op: &Op, facts: &Facts) -> Contract {
        use Ownership::*;
        let closed = |flags, result| Contract {
            effects: Effects::of(flags),
            result,
        };
        const COERCING: u16 = HEAP_READ | ALLOCATION | INTERNAL | ERROR;
        let operands = op.operands_iter().fold(Types::default(), |types, value| {
            types.union(facts.of(value))
        });
        let coercing = |result| {
            if operands.within(Types::NUMBER) {
                closed(0, Unowned)
            } else {
                closed(COERCING, result)
            }
        };
        match op {
            Op::LowLevel(low_level::Op::Call { .. }) => Contract {
                effects: Effects::UNKNOWN,
                result: Unowned,
            },
            Op::LowLevel(low_level::Op::Load { .. } | low_level::Op::AtomicLoad { .. }) => {
                Contract {
                    effects: Effects::UNKNOWN,
                    result: Unowned,
                }
            }
            Op::LowLevel(low_level::Op::Store { .. }) => Contract {
                effects: Effects::UNKNOWN,
                result: None,
            },
            Op::LowLevel(_) => closed(0, Unowned),
            Op::OsrStart(..) => Contract {
                effects: Effects::UNKNOWN,
                result: Unowned,
            },
            Op::Continuation(..) => Contract {
                effects: Effects::UNKNOWN,
                result: Owned,
            },
            Op::ContinuationOutput(..) => closed(INTERNAL, Owned),
            Op::ContinuationStore(..) | Op::Observe(..) | Op::ReadHint(..) => {
                closed(INTERNAL, None)
            }
            Op::Assume(..) => closed(0, None),
            Op::EntryAssume(..) => closed(0, Unowned),
            Op::NewInstance(_) => closed(ALLOCATION | INTERNAL | ERROR, Owned),
            Op::ConstructorResult(..) => closed(0, Borrowed),
            Op::Rest(_) | Op::Arguments => closed(ALLOCATION | INTERNAL | ERROR, Owned),
            Op::Throw(_) => closed(INTERNAL | ERROR, None),
            Op::Catch => closed(INTERNAL, Owned),
            Op::Argument(_) | Op::This => closed(0, Borrowed),
            Op::ClosureThis(_) => closed(HEAP_READ, Borrowed),
            Op::NewHoleyArray(_) => closed(ALLOCATION | INTERNAL | ERROR, Owned),
            Op::MapElement(..) => closed(HEAP_READ | INTERNAL, Borrowed),
            Op::IsArrayIteration(..) | Op::IsStringSplit(_) => closed(HEAP_READ, Unowned),
            Op::StringSplit(..) | Op::StringSplitParts(..) => {
                closed(HEAP_READ | ALLOCATION | INTERNAL | ERROR, Owned)
            }
            Op::Constant(_) => closed(0, Unowned),
            Op::GlobalRead(_) | Op::GlobalReadOptional(_) => {
                closed(GLOBAL_READ | INTERNAL | ERROR, Borrowed)
            }
            Op::RegExp(..) => closed(ALLOCATION | INTERNAL | ERROR, Owned),
            Op::PropertyKey(_) => closed(HEAP_READ | ALLOCATION | INTERNAL | ERROR, Owned),
            Op::GlobalWrite(..) | Op::GlobalInitialize(..) => {
                closed(GLOBAL_WRITE | INTERNAL | ERROR, None)
            }
            Op::BindingLoad(_) => closed(INTERNAL, Borrowed),
            Op::BindingRestore(..) => closed(INTERNAL, None),
            Op::Read(binding) => closed(
                if self.bindings[*binding].captured {
                    HEAP_READ | INTERNAL | ERROR
                } else {
                    ERROR
                },
                Borrowed,
            ),
            Op::Initialize(binding, _) | Op::Write(binding, _)
                if self.bindings[*binding].captured =>
            {
                closed(HEAP_WRITE | INTERNAL | ERROR, None)
            }
            Op::Initialize(..) => closed(INTERNAL, None),
            Op::Write(..) => closed(INTERNAL | ERROR, None),
            Op::CheckInitialized(_) => {
                if operands.intersects(Types::UNINITIALIZED) {
                    closed(ERROR, None)
                } else {
                    closed(0, None)
                }
            }
            Op::Closure(..)
            | Op::NewPack
            | Op::NewObject
            | Op::NewArray(_)
            | Op::IteratorNew(_)
            | Op::EnumerateNew(_) => closed(ALLOCATION | INTERNAL | ERROR, Owned),
            Op::PackRelease(_) => closed(HEAP_WRITE | INTERNAL, None),
            Op::Spread(_) => closed(HEAP_READ | ALLOCATION | INTERNAL | ERROR, Owned),
            Op::IteratorNext(_) | Op::EnumerateNext(_) => closed(
                HEAP_READ | HEAP_WRITE | ALLOCATION | INTERNAL | ERROR,
                MaybeOwned,
            ),
            Op::IsUninitialized(_) => closed(0, Unowned),
            // Defines only target the literal being built, which nothing has read yet, so the
            // HEAP write is left out to keep borrows and loads alive across object construction.
            Op::Define(..) => closed(HEAP_READ | CODE_WRITE | ALLOCATION | INTERNAL | ERROR, None),
            Op::DefineHidden(..) | Op::MarkClass(_) | Op::DisableConstructor(_) => closed(
                HEAP_WRITE | CODE_WRITE | ALLOCATION | INTERNAL | ERROR,
                None,
            ),
            Op::PackAppend(..) | Op::ArrayAppend(..) => {
                closed(HEAP_READ | HEAP_WRITE | ALLOCATION | INTERNAL | ERROR, None)
            }
            Op::Get(..) | Op::GetIndex(..) | Op::FillPropertyCache(..) => closed(
                HEAP_READ | GLOBAL_READ | ALLOCATION | INTERNAL | ERROR,
                Borrowed,
            ),
            Op::IsPropertyCached(..) | Op::IsShape(..) => closed(CODE_READ, Unowned),
            Op::ShapedGet(..) => closed(HEAP_READ | INTERNAL | ERROR, Borrowed),
            Op::ShapedSet(..) => {
                closed(HEAP_READ | HEAP_WRITE | ALLOCATION | INTERNAL | ERROR, None)
            }
            Op::PrimePropertyCache(..) => closed(HEAP_READ | CODE_READ | INTERNAL, None),
            Op::CachedProperty(..) => closed(HEAP_READ | INTERNAL | ERROR, Borrowed),
            Op::Set(..) | Op::SetIndex(..) => closed(
                HEAP_READ | HEAP_WRITE | CODE_WRITE | ALLOCATION | INTERNAL | ERROR,
                None,
            ),
            Op::Unary(UnaryOp::Not, _) => closed(HEAP_READ, Unowned),
            Op::Unary(UnaryOp::Typeof, _) => closed(HEAP_READ | ALLOCATION | ERROR, MaybeOwned),
            Op::Unary(UnaryOp::Increment | UnaryOp::Decrement, _) => {
                if operands.within(Types::NUMBER) {
                    closed(0, Unowned)
                } else {
                    closed(ALLOCATION | INTERNAL | ERROR, MaybeOwned)
                }
            }
            Op::ToInt32(_) => closed(0, Unowned),
            Op::ToNumeric(_) | Op::Unary(..) => coercing(MaybeOwned),
            Op::GuardedCompare {
                operator,
                left,
                right,
                ..
            } => self.contract(&Op::Binary(*operator, *left, *right), facts),
            Op::Binary(operator, ..) => match operator {
                BinaryOperator::StrictEquality | BinaryOperator::StrictInequality => closed(
                    if operands.within(Types::NUMBER) {
                        0
                    } else {
                        HEAP_READ
                    },
                    Unowned,
                ),
                BinaryOperator::Equality
                | BinaryOperator::Inequality
                | BinaryOperator::LessThan
                | BinaryOperator::LessEqualThan
                | BinaryOperator::GreaterThan
                | BinaryOperator::GreaterEqualThan => coercing(Unowned),
                BinaryOperator::In | BinaryOperator::Instanceof => Contract {
                    effects: Effects::UNKNOWN,
                    result: Unowned,
                },
                _ => coercing(MaybeOwned),
            },
            Op::Call { .. }
            | Op::Construct { .. }
            | Op::CallPack(..)
            | Op::ConstructPack(..)
            | Op::ConstructorCall { .. }
            | Op::ConstructorCallPack(..)
            | Op::ConstructResolved { .. }
            | Op::ConstructPackResolved(..) => Contract {
                effects: Effects::UNKNOWN,
                result: Owned,
            },
            Op::CallNative(native, _)
                if METHODS[native.index()].1 == crate::heap::MathFunction::Random =>
            {
                closed(INTERNAL, Unowned)
            }
            Op::CallNative(..) => coercing(Unowned),
            Op::IsUndefined(_) | Op::IsNullish(_) | Op::IsNative(..) => closed(0, Unowned),
            Op::ConstructorPrototype(_) => closed(HEAP_READ, Borrowed),
            Op::IsCode(..) => closed(HEAP_READ, Unowned),
            Op::ProbeGlobal(_) => closed(GLOBAL_READ, Borrowed),
            Op::ProbeProperty(..) | Op::ProbeIndex(..) => closed(HEAP_READ, Borrowed),
            Op::ProbeShapeProperty(..) => closed(HEAP_READ | CODE_READ, Borrowed),
            Op::GuardValid(_) => closed(GLOBAL_READ, Unowned),
            Op::GuardCommit(_) => closed(INTERNAL, None),
            Op::View(_)
            | Op::TypedData(_)
            | Op::Memory(MemoryOp::IndexedLoad { .. } | MemoryOp::Load { .. }) => {
                closed(HEAP_READ, Unowned)
            }
            Op::Memory(MemoryOp::Address { .. }) | Op::MemoryNumber(..) => closed(0, Unowned),
            Op::KeepAlive(_) => closed(INTERNAL, None),
            Op::IsTyped(..) | Op::IsIndexed(_) => closed(0, Unowned),
            Op::IsArrayIndex(..) | Op::IsArray(..) => closed(HEAP_READ, Unowned),
            Op::ArrayPosition(_) => closed(INTERNAL, Unowned),
            Op::IsExactPosition(..) | Op::IsInBounds(..) | Op::BooleanAnd(..) => closed(0, Unowned),
            Op::ArrayKind(_) | Op::IndexedLength(_) => closed(HEAP_READ, Unowned),
            Op::IndexedGet(..) | Op::ArrayLoad(..) => {
                closed(HEAP_READ | INTERNAL | ERROR, Borrowed)
            }
            Op::PackTake(..) => closed(HEAP_READ | HEAP_WRITE | INTERNAL, Owned),
            Op::TypedIndex(..) => closed(HEAP_READ, Unowned),
            Op::IsNumber(_)
            | Op::IsInteger(_)
            | Op::AsInteger(_)
            | Op::NumericFloat(_)
            | Op::SaturatingInteger(_)
            | Op::IntegerCheck(..)
            | Op::IsString(_) => closed(0, Unowned),
            Op::IsTypeof(..) => closed(HEAP_READ, Unowned),
            Op::AsNumber(_) => closed(INTERNAL, Unowned),
            Op::AsObject(_) => closed(0, Unowned),
            Op::EnterCaptures(..) => closed(HEAP_READ | INTERNAL | ERROR, None),
            Op::EnterScope(scope) | Op::NextIteration(scope) | Op::FreshIteration(scope) => {
                if self.materialized(*scope) {
                    closed(ALLOCATION | INTERNAL | ERROR, None)
                } else {
                    closed(0, None)
                }
            }
            Op::LeaveScope(scope) => {
                if self.materialized(*scope) {
                    closed(INTERNAL, None)
                } else {
                    closed(0, None)
                }
            }
        }
    }

    fn lowered_memory_scalar(&self, value: Value) -> bool {
        if self.low_level_signature.is_none() {
            return false;
        }
        let Some(Op::Memory(MemoryOp::IndexedLoad { format, .. } | MemoryOp::Load { format, .. })) =
            self.defining_op(value)
        else {
            return false;
        };
        let ty = match format {
            MemoryFormat::U8 | MemoryFormat::I8 => cranelift_codegen::ir::types::I8,
            MemoryFormat::U16 | MemoryFormat::I16 => cranelift_codegen::ir::types::I16,
            MemoryFormat::U32 | MemoryFormat::I32 => cranelift_codegen::ir::types::I32,
            MemoryFormat::F32 => cranelift_codegen::ir::types::F32,
            MemoryFormat::F64 => cranelift_codegen::ir::types::F64,
        };
        self.low_level_types[value] == Some(ty)
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
                    Op::NextIteration(scope) | Op::FreshIteration(scope)
                        if stack.last() != Some(&scope) =>
                    {
                        return Err(format!("{inst} steps {scope} while it is not innermost"));
                    }
                    _ => {}
                }
            }
            let Some(terminator) = &self.blocks[block].terminator else {
                return Err(format!("{block} has no terminator"));
            };
            if matches!(
                terminator.kind,
                TerminatorKind::Return(_)
                    | TerminatorKind::ReturnValues(_)
                    | TerminatorKind::TailCall { .. }
                    | TerminatorKind::Fallback(_)
            ) && !stack.is_empty()
            {
                return Err(format!("{block} returns with open scopes"));
            }
            if let TerminatorKind::ReturnValues(values) = &terminator.kind {
                let Some(signature) = &self.low_level_signature else {
                    return Err(format!("{block} has no low-level signature"));
                };
                if values.len() != signature.returns.len()
                    || values
                        .iter()
                        .zip(&signature.returns)
                        .any(|(&value, param)| {
                            self.low_level_types[value] != Some(param.value_type)
                        })
                {
                    return Err(format!("{block} has incompatible return values"));
                }
            }
            if let TerminatorKind::TailCall {
                signature, args, ..
            } = &terminator.kind
            {
                if args.len() != signature.params.len()
                    || args.iter().zip(&signature.params).any(|(&value, param)| {
                        self.low_level_types[value] != Some(param.value_type)
                    })
                {
                    return Err(format!("{block} has incompatible tail-call arguments"));
                }
            }
            for value in terminator.kind.operands() {
                if matches!(
                    self.defining_op(value),
                    Some(Op::TypedData(_) | Op::Memory(_))
                ) && !self.lowered_memory_scalar(value)
                {
                    return Err(format!("raw value {value} escapes {block}"));
                }
            }
            for target in terminator.kind.targets() {
                if target.args.len() != self.blocks[target.block].params.len() {
                    return Err(format!("{block} passes wrong arity to {}", target.block));
                }
                for (&arg, &param) in target.args.iter().zip(&self.blocks[target.block].params) {
                    if self.low_level_types[arg] != self.low_level_types[param] {
                        return Err(format!(
                            "{block} passes incompatible scalar type to {}",
                            target.block
                        ));
                    }
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
                let instruction = &self.insts[inst];
                for (result_index, &value) in instruction.results.iter().enumerate() {
                    if self.values[value] != ValueDef::Result(inst, result_index) {
                        return Err(format!("{value} is not result {result_index} of {inst}"));
                    }
                }
                if let Op::LowLevel(low_level::Op::Call {
                    signature, args, ..
                }) = &instruction.op
                {
                    if signature.params.len() != args.len()
                        || signature.returns.len() != instruction.results.len()
                    {
                        return Err(format!("{inst} has an invalid call signature"));
                    }
                    for (&value, ty) in instruction.results.iter().zip(&signature.returns) {
                        if self.low_level_types[value] != Some(ty.value_type) {
                            return Err(format!("{inst} has an invalid call result type"));
                        }
                    }
                }
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
                    ValueDef::Result(inst, result_index) => {
                        let definer = self.insts[inst].block;
                        let index = positions[inst];
                        self.insts[inst].result_at(result_index) == Some(value)
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
                    let Some(def) = self.defining_op(value) else {
                        continue;
                    };
                    let raw = matches!(def, Op::TypedData(_) | Op::Memory(_));
                    if !raw {
                        continue;
                    }
                    if self.lowered_memory_scalar(value)
                        && matches!(self.insts[inst].op, Op::LowLevel(_))
                    {
                        continue;
                    }
                    if self.def_block(value) != block {
                        return Err(format!("{value} crosses a block boundary"));
                    }
                    let valid = match (&self.insts[inst].op, def) {
                        (
                            Op::Memory(
                                MemoryOp::IndexedLoad { base, .. } | MemoryOp::Address { base, .. },
                            ),
                            Op::TypedData(_) | Op::Memory(MemoryOp::Address { .. }),
                        ) => *base == value,
                        (
                            Op::Memory(MemoryOp::Load { address, .. }),
                            Op::Memory(MemoryOp::Address { .. }),
                        ) => *address == value,
                        (
                            Op::MemoryNumber(input, format),
                            Op::Memory(
                                MemoryOp::IndexedLoad { format: source, .. }
                                | MemoryOp::Load { format: source, .. },
                            ),
                        ) => *input == value && format == source,
                        _ => false,
                    };
                    if !valid {
                        return Err(format!(
                            "{inst} {:?} has invalid raw operand {value} from {def:?}",
                            self.insts[inst].op
                        ));
                    }
                }
                if let Some(&view) = self.views.get(&inst) {
                    check(view, index)?;
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

#[derive(Clone, Default, PartialEq, Eq)]
pub(in crate::jit) struct SpecializationPlan {
    pub(in crate::jit) call_variants: Vec<(super::profile::Site, super::entries::VariantId)>,
    pub(in crate::jit) inlines: Vec<specialize::Inline>,
    pub(in crate::jit) reads: Vec<reads::Read>,
    pub(in crate::jit) shapes: shapes::Plan,
    pub(in crate::jit) numbers: Vec<(super::profile::Site, super::profile::ReadType)>,
    pub(in crate::jit) conditional: specialize::ConditionalPlan,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum FactsLevel {
    None,
    Cfg,
    Values,
    Effects,
}

pub(super) struct FunctionState {
    pub(in crate::jit) plan: SpecializationPlan,
    pub(super) graph: Graph,
    pub(super) osr: Option<std::sync::Arc<osr::Plan>>,
    pub(in crate::jit) facts: Facts,
    notes: Vec<String>,
    optimize: bool,
    // Graph mutations invalidate this level and all dependent analyses.
    facts_level: FactsLevel,
}

impl FunctionState {
    pub(in crate::jit) fn from_graph(graph: Graph) -> Self {
        Self::new(graph, Vec::new(), true)
    }

    fn new(graph: Graph, notes: Vec<String>, optimize: bool) -> Self {
        let optimize = optimize && !graph.has_handlers && !graph.generator;
        Self {
            graph,
            notes,
            optimize,
            facts: Facts::default(),
            facts_level: FactsLevel::None,
            plan: SpecializationPlan::default(),
            osr: None,
        }
    }

    fn ensure_cfg(&mut self) {
        if self.facts_level < FactsLevel::Cfg {
            self.facts.cfg = Cfg::new(&self.graph);
            self.facts_level = FactsLevel::Cfg;
        }
    }

    fn ensure_values(&mut self) {
        self.ensure_cfg();
        if self.facts_level < FactsLevel::Values {
            let cfg = std::mem::take(&mut self.facts.cfg);
            self.facts = Facts::values_with_cfg(&self.graph, cfg);
            self.facts_level = FactsLevel::Values;
        }
    }

    fn ensure_facts(&mut self) {
        self.ensure_values();
        if self.facts_level < FactsLevel::Effects {
            self.facts.compute_effects(&self.graph);
            self.facts_level = FactsLevel::Effects;
        }
    }

    fn invalidate_cfg(&mut self) {
        self.facts_level = FactsLevel::None;
    }

    fn invalidate_values(&mut self) {
        self.facts_level = self.facts_level.min(FactsLevel::Cfg);
    }

    fn invalidate_effects(&mut self) {
        self.facts_level = self.facts_level.min(FactsLevel::Values);
    }

    pub(in crate::jit) fn lower_memory(&mut self) -> Result<()> {
        if typed_lowering::lower_memory(self) {
            self.graph.verify()?;
            self.ensure_facts();
            borrow::borrows(&mut self.graph, &self.facts, &mut self.notes);
        }
        Ok(())
    }

    pub(in crate::jit) fn prepare_osr(&mut self, source: usize) -> Result<()> {
        self.osr = osr::Plan::discover(&self.graph, source).map(std::sync::Arc::new);
        if let Some(plan) = self.osr.clone() {
            self.invalidate_cfg();
            plan.instrument(&mut self.graph)?;
            self.ensure_facts();
            borrow::borrows(&mut self.graph, &self.facts, &mut self.notes);
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
    continuations: bool,
    report_regions: bool,
) -> Result<FunctionState> {
    let graph = build::lower_global(
        statements,
        span,
        scoping,
        captured,
        closures,
        arguments_references,
    )?;
    let mut function = FunctionState::new(graph, Vec::new(), ssa_opt);
    canonicalize(&mut function)?;
    let planned = plan_canonical(function, ssa_opt, None)?;
    let versioning = Versioning::Speculate {
        fallback: None,
        continuations,
    };
    optimize_planned(planned, versioning, report_regions)
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
) -> Result<FunctionState> {
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
    let mut function = FunctionState::new(graph, Vec::new(), ssa_opt);
    canonicalize(&mut function)?;
    Ok(function)
}

#[cfg(test)]
pub(super) fn analyze(
    function: &oxc_ast::ast::Function<'_>,
    scoping: &Scoping,
    captured: &HashSet<SymbolId>,
) -> Result<FunctionState> {
    optimize_graph(build::lower(function, scoping, captured)?, true)
}

fn canonicalize(function: &mut FunctionState) -> Result<()> {
    if !function.optimize {
        return Ok(());
    }
    function.invalidate_cfg();
    transforms::promote(&mut function.graph, &mut function.notes);
    function.graph.verify()?;
    js::typeof_comparisons(&mut function.graph);
    js::simplify(function);
    function.graph.verify()
}

pub(in crate::jit) fn canonicalize_graph(graph: Graph, ssa_opt: bool) -> Result<FunctionState> {
    let mut function = FunctionState::new(graph, Vec::new(), ssa_opt);
    canonicalize(&mut function)?;
    Ok(function)
}

#[cfg(test)]
fn optimize_graph(graph: Graph, ssa_opt: bool) -> Result<FunctionState> {
    optimize_canonical(canonicalize_graph(graph, ssa_opt)?, ssa_opt, None)
}

#[cfg(test)]
pub(super) fn optimize_canonical(
    function: FunctionState,
    ssa_opt: bool,
    specialization: Option<(&specialize::Repository, usize)>,
) -> Result<FunctionState> {
    optimize_planned(
        plan_canonical(function, ssa_opt, specialization)?,
        Versioning::Speculate {
            fallback: None,
            continuations: true,
        },
        false,
    )
}

#[derive(Default)]
struct Expansions {
    arrays: HashSet<Inst>,
    constructors: HashSet<Inst>,
    splits: HashSet<Inst>,
}

impl Expansions {
    fn calls(&mut self, graph: &mut Graph, source: usize, notes: &mut Vec<String>) -> Result<()> {
        arrays::expand(graph, source, notes, &mut self.arrays)?;
        string_split::expand(graph, &mut self.splits);
        Ok(())
    }

    fn constructors(&mut self, graph: &mut Graph) {
        construction::resolve(graph);
        construction::expand(graph, &mut self.constructors);
    }
}

pub(super) fn plan_canonical(
    function: FunctionState,
    ssa_opt: bool,
    specialization: Option<(&specialize::Repository, usize)>,
) -> Result<FunctionState> {
    plan_canonical_with_budget(function, ssa_opt, specialization, 1)
}

pub(super) fn plan_canonical_with_budget(
    function: FunctionState,
    ssa_opt: bool,
    specialization: Option<(&specialize::Repository, usize)>,
    inline_multiplier: usize,
) -> Result<FunctionState> {
    plan_canonical_inner(function, ssa_opt, specialization, inline_multiplier, true)
}

fn plan_canonical_inner(
    mut function: FunctionState,
    ssa_opt: bool,
    specialization: Option<(&specialize::Repository, usize)>,
    inline_multiplier: usize,
    select_call_variants: bool,
) -> Result<FunctionState> {
    function.optimize = ssa_opt && !function.graph.has_handlers && !function.graph.generator;
    if !function.optimize {
        return Ok(function);
    }
    let mut inline_budget = specialize::Budget::new(inline_multiplier);
    let mut expansions = Expansions::default();
    if let Some((bodies, source)) = specialization {
        function.invalidate_cfg();
        expansions.calls(&mut function.graph, source, &mut function.notes)?;
        let requests = specialize::requests(&function.graph, bodies, source, &inline_budget);
        let plan = specialize::plan_round(
            &function.graph,
            bodies,
            source,
            &requests,
            &mut inline_budget,
        );
        specialize::apply(
            &mut function.graph,
            bodies,
            source,
            plan,
            &mut function.notes,
        )?;
        js::simplify(&mut function);
        function.invalidate_cfg();
        expansions.calls(&mut function.graph, source, &mut function.notes)?;
        function.graph.verify()?;
    }
    function.invalidate_cfg();
    expansions.constructors(&mut function.graph);
    if let Some((bodies, source)) = specialization {
        for _ in 0..specialize::SPECIALIZATION_ROUNDS {
            let requests = specialize::requests(&function.graph, bodies, source, &inline_budget);
            let plan = specialize::plan_round(
                &function.graph,
                bodies,
                source,
                &requests,
                &mut inline_budget,
            );
            if plan.is_empty() {
                break;
            }
            function.invalidate_cfg();
            specialize::apply(
                &mut function.graph,
                bodies,
                source,
                plan,
                &mut function.notes,
            )?;
            js::simplify(&mut function);
            function.invalidate_cfg();
            expansions.calls(&mut function.graph, source, &mut function.notes)?;
            expansions.constructors(&mut function.graph);
            function.graph.verify()?;
        }
    }
    function.invalidate_effects();
    if let Some((bodies, _)) = specialization {
        specialize::install_summaries(&mut function.graph, bodies, &mut function.notes);
    }
    function.invalidate_values();
    let mut call_variants = Vec::new();
    if let Some((bodies, source)) = specialization {
        specialize::direct_calls(&mut function.graph, bodies, source);
        if select_call_variants {
            call_variants = call_variants::apply(&mut function.graph, bodies, source)?;
        }
    }
    let numbers = observations::plan(&mut function.graph, specialization);
    let reads = specialization.map_or_else(Vec::new, |(bodies, source)| {
        reads::plan(&function.graph, bodies, source)
    });
    let shapes = specialization.map_or_else(shapes::Plan::default, |(bodies, source)| {
        shapes::plan(&function.graph, bodies, source)
    });
    let mut conditional = specialization
        .map_or_else(specialize::ConditionalPlan::default, |(bodies, _)| {
            specialize::conditional_plan(&function.graph, bodies, &mut function.notes)
        });
    if let Some((bodies, source)) = specialization {
        conditional.native_calls =
            arrays::push_calls(&mut function.graph, bodies, source, &mut function.notes)?;
    }
    function.plan = SpecializationPlan {
        call_variants,
        conditional,
        inlines: inline_budget.inlines,
        reads,
        shapes,
        numbers,
    };
    Ok(function)
}

#[derive(Clone, Copy)]
pub(super) enum Versioning {
    Generic,
    Speculate {
        fallback: Option<usize>,
        continuations: bool,
    },
}

pub(super) fn optimize_planned(
    mut function: FunctionState,
    versioning: Versioning,
    report_regions: bool,
) -> Result<FunctionState> {
    if !function.optimize {
        function.ensure_facts();
        return Ok(function);
    }
    function.graph.entry_guards.clear();
    function.invalidate_cfg();
    function.graph.megamorphic_gets = function
        .plan
        .shapes
        .megamorphic_sites()
        .iter()
        .copied()
        .collect();
    caches::apply(&mut function.graph, &function.plan.shapes)?;
    reads::apply(
        &mut function.graph,
        &function.plan.reads,
        &mut function.notes,
    )?;
    comparisons::expose(&mut function);
    optimize::sink_closures(&mut function);
    if optimize::lazy_captures(&mut function) {
        function.invalidate_cfg();
        transforms::promote(&mut function.graph, &mut function.notes);
        js::simplify(&mut function);
    }
    function.graph.verify()?;
    optimize::forward_captured_parameters(&mut function);
    function.graph.verify()?;
    function.invalidate_cfg();
    let regions = match versioning {
        Versioning::Generic => Vec::new(),
        Versioning::Speculate {
            fallback,
            continuations,
        } => {
            let mut regions = optimize::version_with_report(
                &mut function.graph,
                &function.plan.conditional,
                fallback,
                continuations,
                &mut function.notes,
                report_regions,
                &function.plan.shapes,
            )?;
            function.graph.verify()?;
            if regions.is_empty() {
                regions.extend(optimize::version_string_leaves(
                    &mut function.graph,
                    continuations,
                    &mut function.notes,
                )?);
            }
            regions
        }
    };
    if report_regions {
        for (id, region) in &function.graph.regions {
            let mut blocks: Vec<_> = region.blocks.iter().copied().collect();
            blocks.sort();
            let blocks = blocks
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            let fallback = match region.fallback {
                Some(RegionFallback::Region(id)) => id.to_string(),
                Some(RegionFallback::CompiledEntry(address)) => {
                    format!("compiled entry {address:#x}")
                }
                None => "none".into(),
            };
            function
                .notes
                .push(format!("{id}: blocks [{blocks}], fallback {fallback}"));
        }
    }
    optimize::normalise_regions(&mut function, &regions)?;
    js::coercions(&mut function);
    js::bit_tests(&mut function);
    function.graph.verify()?;
    optimize::reuse_loads(&mut function);
    function.graph.verify()?;
    for region in &regions {
        optimize::hoist(&mut function, *region);
        function.graph.verify()?;
    }
    if typed_lowering::lower(&mut function)? {
        js::simplify(&mut function);
        optimize::reuse_loads(&mut function);
    }
    function.invalidate_cfg();
    shapes::apply(
        &mut function.graph,
        &function.plan.shapes,
        &mut function.notes,
    )?;
    if !function.plan.shapes.polymorphic_reads().is_empty()
        || function
            .graph
            .insts
            .values()
            .any(|data| matches!(data.op, Op::EntryAssume(_, guards::Fact::Shape(_))))
    {
        optimize::reuse_loads(&mut function);
    }
    if typed_lowering::lower_array_kinds(&mut function) {
        optimize::reuse_loads(&mut function);
    }
    optimize::views(&mut function);
    comparisons::expose(&mut function);
    function.graph.verify()?;
    js::simplify(&mut function);
    function.graph.verify()?;
    optimize::sink_closures(&mut function);
    optimize::lazy_captures(&mut function);
    function.graph.verify()?;
    function.invalidate_cfg();
    string_split::forward_packs(&mut function.graph);
    transforms::trim(&mut function.graph);
    transforms::remove_trivial_params(&mut function.graph);
    function.graph.verify()?;
    for region in &regions {
        optimize::hoist(&mut function, *region);
        function.graph.verify()?;
    }
    if numeric_lowering::lower(&mut function) {
        optimize::reuse_loads(&mut function);
        for region in &regions {
            optimize::hoist(&mut function, *region);
            function.graph.verify()?;
        }
    }
    function.ensure_facts();
    borrow::borrows(&mut function.graph, &function.facts, &mut function.notes);
    Ok(function)
}

#[cfg(test)]
mod tests;

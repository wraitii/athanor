use crate::jit::layout::ReadLayout;
mod available;
pub(in crate::jit) mod bits;
mod borrow;
pub(in crate::jit) mod boundary;
mod boxing;
mod build;
pub(in crate::jit) mod continuation;
pub(in crate::jit) mod contracts;
pub(in crate::jit) mod cursor;
mod dispatch;
pub(in crate::jit) mod facts;
mod guards;
mod js;
pub(in crate::jit) mod legalize;
mod listing;
mod liveness;
pub(in crate::jit) mod low_level;
mod memory;
pub(super) mod osr;
pub(in crate::jit) mod ownership;
mod pipeline;
pub(in crate::jit) mod regex;
pub(in crate::jit) mod repr;
pub(in crate::jit) mod runtime;
pub(in crate::jit) mod specialization;
pub(in crate::jit) mod suspension;
mod transforms;
pub(super) use transforms::loops::natural_loops;
pub(in crate::jit) mod variables;
pub(in crate::jit) mod widths;

fn verification_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("JS_JIT_VERIFY").as_deref() == Ok("1"));
    cfg!(any(debug_assertions, test)) || *ENABLED
}

pub(super) use pipeline::{
    LOW_LEVEL_PASSES, RUNTIME_PASSES, SPECIALIZATION_PASSES, Versioning, build_canonical,
    canonicalize_graph, inline_and_plan_specialization, lower_global, lower_low_level, specialize,
    specialize_observed,
};
#[cfg(test)]
use pipeline::{analyze, lower_planned, optimize_canonical, plan_specialization};

use crate::{FxHashMap, FxHashSet};
use std::fmt;

use cranelift_codegen::entity::{EntityRef, PrimaryMap, SecondaryMap};
use oxc_span::Span;
use oxc_syntax::operator::BinaryOperator;

use super::effects::{
    ALLOCATION, CODE_READ, CODE_WRITE, Control, ERROR, Effects, GLOBAL_READ, GLOBAL_WRITE,
    HEAP_READ, HEAP_WRITE, INTERNAL, Memory,
};
use super::math::METHODS;
use crate::heap::TypedKind;

pub(in crate::jit) use facts::{Facts, Identity, ValueType};
pub(super) use ids::{Binding, Block, Inst, RegionId, Scope, Value};
pub(in crate::jit) use listing::ListingLine;
#[cfg(test)]
pub(in crate::jit) use listing::SourceLocation;
pub(in crate::jit) use repr::Repr;

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
    pub(in crate::jit) views: FxHashMap<Inst, Value>,
    property_accesses: FxHashMap<Inst, specialization::properties::Access>,
    pub(in crate::jit) borrowed: FxHashSet<Value>,
    pub(in crate::jit) summaries: std::sync::Arc<FxHashMap<cranelift_module::FuncId, CallSummary>>,
    pub(in crate::jit) cold: FxHashSet<Block>,
    pub(in crate::jit) megamorphic_gets: FxHashSet<super::profile::Site>,
    pub(super) lazy_captures: FxHashSet<Binding>,
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
    pub(in crate::jit) reprs: SecondaryMap<Value, Option<Repr>>,
    pub(in crate::jit) ownership: Option<ownership::Plan>,
    pub(in crate::jit) low_level_slots: Vec<low_level::StackSlotData>,
    pub(super) low_level_signature: Option<cranelift_codegen::ir::Signature>,
    pub(super) bindings: PrimaryMap<Binding, BindingData>,
    pub(super) scopes: PrimaryMap<Scope, ScopeData>,
}

#[derive(Clone)]
pub(super) struct Region {
    pub(super) blocks: FxHashSet<Block>,
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
    // Property-shape feedback is local to the inline; call and numeric feedback stay shared.
    pub(super) inline_context: u64,
    pub(super) source_name: Option<(usize, usize)>,
    pub(super) emission_location: Option<u32>,
    pub(super) span: Span,
    pub(super) block: Block,
    pub(super) results: smallvec::SmallVec<[Value; 1]>,
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
pub(in crate::jit) enum ArrayLayout {
    Packed,
    Holey,
}

impl ArrayLayout {
    pub(in crate::jit) fn kind(self) -> u32 {
        crate::heap::kind::ARRAY
            | match self {
                Self::Packed => 0,
                Self::Holey => crate::heap::kind::HOLEY,
            }
    }
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

#[derive(Clone, Copy, Debug)]
pub(super) enum PropertyWrite {
    Set,
    Define,
}

#[derive(Clone, Debug)]
pub(super) enum Op {
    LowLevel(low_level::Op),
    Regex(regex::Op),
    Aarch64(crate::jit::aarch64::inst::Op),
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
    Restored(usize),
    Arguments,
    ArgumentCount,
    Rest(usize),
    This,
    Callee,
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
    ClosureCells(
        Span,
        cranelift_module::FuncId,
        bool,
        bool,
        bool,
        Vec<Value>,
        Option<Value>,
    ),
    CellNew(Value),
    CellGet(Value),
    CellSet(Value, Value),
    CaptureGet(Value, usize),
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
    // Borrows the current frame's original arguments, which inlining substitutes.
    CallForward(Value, Value),
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
    IsOrdinaryArray(Value),
    // Saturating numeric conversion; validity is checked separately by IsArrayIndex or IsExactPosition.
    ArrayPosition(Value),
    // Compare a numeric key with its integer position converted back to a number.
    IsExactPosition(Value, Value),
    // Compare integer position and length as unsigned values.
    IsInBounds(Value, Value),
    IsArrayIndex(Value, Value, Value, ArrayLayout),
    IsArray(Value, ArrayLayout),
    // Non-short-circuit conjunction of boolean operands.
    BooleanAnd(Value, Value),
    // Requires matching layout, exact position and bounds checks with no intervening array mutation.
    // The array holds the borrowed result; holes produce undefined.
    ArrayLoad(Value, Value, ArrayLayout),
    IndexedLength(Value),
    ArrayKind(Value),
    // Array or typed-array receiver; retains generic key and bounds handling.
    ArrayGet(Value, Value),
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
    PropertyCache(Inst, String),
    GlobalCache(String),
    PropertyLoad(Value, Value, js::memory::Field),
    RelatedPropertyLoad(Value, Value, js::memory::Field),
    HelperCall {
        helper: super::helpers::Helper,
        args: Vec<Value>,
        uses: Vec<Repr>,
        produces: Option<Repr>,
        result_type: ValueType,
        effects: Effects,
    },
    IsPropertyCached(Value, Value, bool),
    CachedProperty(Value, Value, bool),
    IsShape(Value, u32),
    IsReadLayout(Value, crate::jit::layout::LayoutGuard),
    LayoutGet(Value, String, ReadLayout),
    ShapedGet(Value, String, u32),
    ShapedSet(Value, String, u32, Value),
    FillPropertyCache(Value, Value, String),
    PrimePropertyCache(Value, Value),
    GetIndex(Value, Value),
    Set(Value, String, Value),
    SetIndex(Value, Value, Value),
    // Bounded packed-array replacement; storage retains the incoming value before releasing the old one.
    ArrayStore(Value, Value, Value),
    PropertyStore(Value, Value, Value, js::memory::Field, PropertyWrite),
    PropertyAdd(Value, Value, Value, PropertyWrite),
    // Native slot replacement must go through the runtime setter.
    CanOverwrite(Value, Value),
    FillPropertyStore(Value, Value, String, Value, PropertyWrite),
    ToNumeric(Value),
    // Pure integer coercion of a value proven to be a number or undefined.
    ToInt32(Value),
    Unary(UnaryOp, Value),
    Binary(BinaryOperator, Value, Value),
    StringConcat(Vec<Value>),
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
    ProbeShapeProperty(Value, String, ReadLayout),
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
    Truthy(Value),
    IsStringSplit(Value),
    StringSplit(Value, Vec<Value>),
    StringSplitParts(Value, Vec<Value>),
    IsTypeof(Value, u8),
    AsNumber(Value),
    AsString(Value),
    Convert(Value, Repr),
    Box(Value),
    AsBoxed(Value),
    AsWord(Value),
    Unbox(Value, Repr),
    Release(Value),
    Retain(Value),
    SlotStore(Value),
    SlotClear(Value),
    ParamStore(Value, Value),
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
    Aarch64Branch(crate::jit::aarch64::inst::Condition, Target, Target),
    Return(Value),
    ReturnValues(Vec<Value>),
    TailCall {
        target: low_level::CallTarget,
        signature: cranelift_codegen::ir::Signature,
        args: Vec<Value>,
    },
    Bailout {
        // The target belongs to the generic graph; only inputs are SSA operands.
        target: Target,
        inputs: Vec<Value>,
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

pub(in crate::jit) struct Operands<'a> {
    fixed: std::array::IntoIter<Option<Value>, 4>,
    rest: std::slice::Iter<'a, Value>,
}

impl<'a> Operands<'a> {
    pub(in crate::jit) fn new(fixed: [Option<Value>; 4], rest: &'a [Value]) -> Self {
        Self {
            fixed: fixed.into_iter(),
            rest: rest.iter(),
        }
    }
}

impl Iterator for Operands<'_> {
    type Item = Value;

    fn next(&mut self) -> Option<Value> {
        self.fixed
            .by_ref()
            .flatten()
            .next()
            .or_else(|| self.rest.next().copied())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self
            .fixed
            .as_slice()
            .iter()
            .filter(|value| value.is_some())
            .count()
            + self.rest.len();
        (size, Some(size))
    }
}

impl ExactSizeIterator for Operands<'_> {}

pub(in crate::jit) struct OperandsMut<'a> {
    fixed: std::array::IntoIter<Option<&'a mut Value>, 4>,
    rest: std::slice::IterMut<'a, Value>,
}

impl<'a> OperandsMut<'a> {
    pub(in crate::jit) fn new(fixed: [Option<&'a mut Value>; 4], rest: &'a mut [Value]) -> Self {
        Self {
            fixed: fixed.into_iter(),
            rest: rest.iter_mut(),
        }
    }
}

impl<'a> Iterator for OperandsMut<'a> {
    type Item = &'a mut Value;

    fn next(&mut self) -> Option<Self::Item> {
        self.fixed
            .by_ref()
            .flatten()
            .next()
            .or_else(|| self.rest.next())
    }
}

impl Op {
    pub(super) fn operands(&self) -> Vec<Value> {
        self.operands_iter().collect()
    }

    pub(super) fn operands_iter(&self) -> Operands<'_> {
        let (fixed, rest): ([Option<Value>; 3], &[Value]) = match self {
            Op::Aarch64(op) => return op.operands(),
            Op::Regex(op) => return op.operands(),
            Op::LowLevel(low_level::Op::Cleanup { values, .. }) => ([None; 3], values),
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
                | low_level::Op::Reference { value, .. }
                | low_level::Op::ExtractBits { value, .. }
                | low_level::Op::InRange { value, .. }
                | low_level::Op::TestMask { value, .. }
                | low_level::Op::Load { address: value, .. }
                | low_level::Op::AtomicLoad { address: value, .. },
            ) => ([Some(*value), None, None], &[]),
            Op::LowLevel(
                low_level::Op::FrameStore {
                    ctx: left,
                    value: right,
                    ..
                }
                | low_level::Op::Binary { left, right, .. }
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
            | Op::EntryAssume(value, _) => ([Some(*value), None, None], &[]),
            Op::Assume(
                value,
                guards::Fact::Cached(cache, _) | guards::Fact::Property(cache),
                _,
            ) => ([Some(*value), Some(*cache), None], &[]),
            Op::Assume(value, ..) => ([Some(*value), None, None], &[]),
            Op::PropertyCache(..)
            | Op::GlobalCache(..)
            | Op::Argument(_)
            | Op::Restored(_)
            | Op::Arguments
            | Op::ArgumentCount
            | Op::Rest(_)
            | Op::This
            | Op::Callee
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
            Op::ClosureCells(_, _, _, _, _, cells, receiver) => {
                ([*receiver, None, None], cells.as_slice())
            }
            Op::CellSet(cell, value) => ([Some(*cell), Some(*value), None], &[]),
            Op::CellNew(value) | Op::CellGet(value) | Op::CaptureGet(value, _) => {
                ([Some(*value), None, None], &[])
            }
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
            | Op::IsOrdinaryArray(value)
            | Op::ArrayKind(value)
            | Op::IndexedLength(value)
            | Op::EnumerateNew(value)
            | Op::EnumerateNext(value)
            | Op::IsUninitialized(value)
            | Op::Get(value, _)
            | Op::IsShape(value, _)
            | Op::IsReadLayout(value, _)
            | Op::LayoutGet(value, _, _)
            | Op::ShapedGet(value, _, _)
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
            | Op::ProbeShapeProperty(value, _, _)
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
            | Op::Truthy(value)
            | Op::IsStringSplit(value)
            | Op::IsTypeof(value, _)
            | Op::AsString(value)
            | Op::AsObject(value)
            | Op::AsNumber(value)
            | Op::Convert(value, _)
            | Op::Box(value)
            | Op::AsBoxed(value)
            | Op::AsWord(value)
            | Op::Unbox(value, _)
            | Op::Release(value)
            | Op::Retain(value)
            | Op::SlotStore(value)
            | Op::SlotClear(value)
            | Op::ParamStore(_, value) => ([Some(*value), None, None], &[]),
            Op::PropertyLoad(object, cache, _)
            | Op::RelatedPropertyLoad(object, cache, _)
            | Op::IsPropertyCached(object, cache, _)
            | Op::CachedProperty(object, cache, _)
            | Op::FillPropertyCache(object, cache, _)
            | Op::PrimePropertyCache(object, cache) => ([Some(*object), Some(*cache), None], &[]),
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
            | Op::CanOverwrite(object, key)
            | Op::GetIndex(object, key)
            | Op::TypedIndex(object, key, _)
            | Op::ArrayGet(object, key)
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
            Op::CallForward(callee, receiver) => ([Some(*callee), Some(*receiver), None], &[]),
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
            Op::ArrayStore(object, key, value)
            | Op::PropertyStore(object, key, value, _, _)
            | Op::PropertyAdd(object, key, value, _)
            | Op::FillPropertyStore(object, key, _, value, _)
            | Op::SetIndex(object, key, value) => ([Some(*object), Some(*key), Some(*value)], &[]),
            Op::HelperCall { args: values, .. }
            | Op::StringConcat(values)
            | Op::NewArray(values)
            | Op::CallNative(_, values) => ([None; 3], values),
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
        let [a, b, c] = fixed;
        Operands::new([a, b, c, None], rest)
    }

    pub(super) fn operands_mut(&mut self) -> OperandsMut<'_> {
        let (fixed, rest): ([Option<&mut Value>; 3], &mut [Value]) = match self {
            Op::Aarch64(op) => return op.operands_mut(),
            Op::Regex(op) => return op.operands_mut(),
            Op::LowLevel(low_level::Op::Cleanup { values, .. }) => ([None, None, None], values),
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
                | low_level::Op::Reference { value, .. }
                | low_level::Op::ExtractBits { value, .. }
                | low_level::Op::InRange { value, .. }
                | low_level::Op::TestMask { value, .. }
                | low_level::Op::Load { address: value, .. }
                | low_level::Op::AtomicLoad { address: value, .. },
            ) => ([Some(value), None, None], &mut []),
            Op::LowLevel(
                low_level::Op::FrameStore {
                    ctx: left,
                    value: right,
                    ..
                }
                | low_level::Op::Binary { left, right, .. }
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
            | Op::EntryAssume(value, _) => ([Some(value), None, None], &mut []),
            Op::Assume(
                value,
                guards::Fact::Cached(cache, _) | guards::Fact::Property(cache),
                _,
            ) => ([Some(value), Some(cache), None], &mut []),
            Op::Assume(value, ..) => ([Some(value), None, None], &mut []),
            Op::PropertyCache(..)
            | Op::GlobalCache(..)
            | Op::Argument(_)
            | Op::Restored(_)
            | Op::Arguments
            | Op::ArgumentCount
            | Op::Rest(_)
            | Op::This
            | Op::Callee
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
            Op::ClosureCells(_, _, _, _, _, cells, receiver) => {
                ([receiver.as_mut(), None, None], cells.as_mut_slice())
            }
            Op::CellSet(cell, value) => ([Some(cell), Some(value), None], &mut []),
            Op::CellNew(value) | Op::CellGet(value) | Op::CaptureGet(value, _) => {
                ([Some(value), None, None], &mut [])
            }
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
            | Op::IsOrdinaryArray(value)
            | Op::ArrayKind(value)
            | Op::IndexedLength(value)
            | Op::EnumerateNew(value)
            | Op::EnumerateNext(value)
            | Op::IsUninitialized(value)
            | Op::Get(value, _)
            | Op::IsShape(value, _)
            | Op::IsReadLayout(value, _)
            | Op::LayoutGet(value, _, _)
            | Op::ShapedGet(value, _, _)
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
            | Op::ProbeShapeProperty(value, _, _)
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
            | Op::Truthy(value)
            | Op::IsStringSplit(value)
            | Op::IsTypeof(value, _)
            | Op::AsString(value)
            | Op::AsObject(value)
            | Op::AsNumber(value)
            | Op::Convert(value, _)
            | Op::Box(value)
            | Op::AsBoxed(value)
            | Op::AsWord(value)
            | Op::Unbox(value, _)
            | Op::Release(value)
            | Op::Retain(value)
            | Op::SlotStore(value)
            | Op::SlotClear(value)
            | Op::ParamStore(_, value) => ([Some(value), None, None], &mut []),
            Op::PropertyLoad(object, cache, _)
            | Op::RelatedPropertyLoad(object, cache, _)
            | Op::IsPropertyCached(object, cache, _)
            | Op::CachedProperty(object, cache, _)
            | Op::FillPropertyCache(object, cache, _)
            | Op::PrimePropertyCache(object, cache) => ([Some(object), Some(cache), None], &mut []),
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
            | Op::CanOverwrite(object, key)
            | Op::GetIndex(object, key)
            | Op::TypedIndex(object, key, _)
            | Op::ArrayGet(object, key)
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
            Op::CallForward(callee, receiver) => ([Some(callee), Some(receiver), None], &mut []),
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
            Op::ArrayStore(object, key, value)
            | Op::PropertyStore(object, key, value, _, _)
            | Op::PropertyAdd(object, key, value, _)
            | Op::FillPropertyStore(object, key, _, value, _)
            | Op::SetIndex(object, key, value) => ([Some(object), Some(key), Some(value)], &mut []),
            Op::HelperCall { args: values, .. }
            | Op::StringConcat(values)
            | Op::NewArray(values)
            | Op::CallNative(_, values) => ([None, None, None], values),
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
        let [a, b, c] = fixed;
        OperandsMut::new([a, b, c, None], rest)
    }
}

impl TerminatorKind {
    pub(super) fn operands_mut(&mut self) -> impl Iterator<Item = &mut Value> + '_ {
        let mut machine = None;
        let (first, rest, tail): (Option<&mut Value>, &mut [Value], &mut [Value]) = match self {
            TerminatorKind::Jump(target) => (None, &mut target.args, &mut []),
            TerminatorKind::CheckException(normal, error) => {
                (None, &mut normal.args, &mut error.args)
            }
            TerminatorKind::Aarch64Branch(condition, yes, no) => {
                machine = Some(condition.operands_mut());
                (None, &mut yes.args, &mut no.args)
            }
            TerminatorKind::Branch(condition, yes, no) => {
                (Some(condition), &mut yes.args, &mut no.args)
            }
            TerminatorKind::Yield(_, value, target) => (Some(value), &mut target.args, &mut []),
            TerminatorKind::Return(value) => (Some(value), &mut [], &mut []),
            TerminatorKind::ReturnValues(values)
            | TerminatorKind::Bailout { inputs: values, .. } => (None, values, &mut []),
            TerminatorKind::TailCall { target, args, .. } => (
                match target {
                    low_level::CallTarget::Indirect(value) => Some(value),
                    _ => None,
                },
                args,
                &mut [],
            ),
            TerminatorKind::Fallback(_) => (None, &mut [], &mut []),
        };
        first
            .into_iter()
            .chain(rest.iter_mut())
            .chain(tail.iter_mut())
            .chain(machine.into_iter().flatten())
    }

    pub(super) fn operands(&self) -> Vec<Value> {
        self.operands_iter().collect()
    }

    pub(super) fn operands_iter(&self) -> impl Iterator<Item = Value> + '_ {
        let mut machine = None;
        let (first, rest, tail): (Option<Value>, &[Value], &[Value]) = match self {
            TerminatorKind::Jump(target) => (None, &target.args, &[]),
            TerminatorKind::CheckException(normal, error) => (None, &normal.args, &error.args),
            TerminatorKind::Aarch64Branch(condition, yes, no) => {
                machine = Some(condition.operands());
                (None, &yes.args, &no.args)
            }
            TerminatorKind::Branch(condition, yes, no) => (Some(*condition), &yes.args, &no.args),
            TerminatorKind::Yield(_, value, target) => (Some(*value), &target.args, &[]),
            TerminatorKind::Return(value) => (Some(*value), &[], &[]),
            TerminatorKind::ReturnValues(values)
            | TerminatorKind::Bailout { inputs: values, .. } => (None, values, &[]),
            TerminatorKind::TailCall { target, args, .. } => (
                match target {
                    low_level::CallTarget::Indirect(value) => Some(*value),
                    _ => None,
                },
                args,
                &[],
            ),
            TerminatorKind::Fallback(_) => (None, &[], &[]),
        };
        first
            .into_iter()
            .chain(rest.iter().copied())
            .chain(tail.iter().copied())
            .chain(machine.into_iter().flatten())
    }

    pub(super) fn targets(&self) -> impl Iterator<Item = &Target> {
        let (first, second) = match self {
            TerminatorKind::Jump(target) | TerminatorKind::Yield(_, _, target) => {
                (Some(target), None)
            }
            TerminatorKind::CheckException(normal, error) => (Some(normal), Some(error)),
            TerminatorKind::Branch(_, yes, no) | TerminatorKind::Aarch64Branch(_, yes, no) => {
                (Some(yes), Some(no))
            }
            TerminatorKind::Return(_)
            | TerminatorKind::ReturnValues(_)
            | TerminatorKind::TailCall { .. }
            | TerminatorKind::Bailout { .. }
            | TerminatorKind::Fallback(_) => (None, None),
        };
        first.into_iter().chain(second)
    }

    pub(super) fn targets_mut(&mut self) -> impl Iterator<Item = &mut Target> {
        let (first, second) = match self {
            TerminatorKind::Jump(target) | TerminatorKind::Yield(_, _, target) => {
                (Some(target), None)
            }
            TerminatorKind::CheckException(normal, error) => (Some(normal), Some(error)),
            TerminatorKind::Branch(_, yes, no) | TerminatorKind::Aarch64Branch(_, yes, no) => {
                (Some(yes), Some(no))
            }
            TerminatorKind::Return(_)
            | TerminatorKind::ReturnValues(_)
            | TerminatorKind::TailCall { .. }
            | TerminatorKind::Bailout { .. }
            | TerminatorKind::Fallback(_) => (None, None),
        };
        first.into_iter().chain(second)
    }
}

pub(super) trait Aliases {
    fn alias(&self, value: Value) -> Option<Value>;
    fn is_empty(&self) -> bool;
}

impl Aliases for FxHashMap<Value, Value> {
    fn alias(&self, value: Value) -> Option<Value> {
        self.get(&value).copied()
    }

    fn is_empty(&self) -> bool {
        FxHashMap::is_empty(self)
    }
}

impl Aliases for SecondaryMap<Value, Option<Value>> {
    fn alias(&self, value: Value) -> Option<Value> {
        self[value]
    }

    fn is_empty(&self) -> bool {
        false
    }
}

pub(super) fn resolve(aliases: &(impl Aliases + ?Sized), mut value: Value) -> Value {
    while let Some(next) = aliases.alias(value) {
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
pub(super) struct Preds {
    ranges: SecondaryMap<Block, (u32, u32)>,
    list: Vec<Block>,
}

impl std::ops::Index<Block> for Preds {
    type Output = [Block];

    fn index(&self, block: Block) -> &[Block] {
        let (start, end) = self.ranges[block];
        &self.list[start as usize..end as usize]
    }
}

impl std::ops::IndexMut<Block> for Preds {
    fn index_mut(&mut self, block: Block) -> &mut [Block] {
        let (start, end) = self.ranges[block];
        &mut self.list[start as usize..end as usize]
    }
}

#[derive(Clone, Default)]
pub(super) struct Cfg {
    pub(super) rpo: Vec<Block>,
    pub(super) preds: Preds,
    dominance: SecondaryMap<Block, (u32, u32)>,
}

impl Cfg {
    pub(super) fn new(graph: &Graph) -> Self {
        Self::from_entry(graph, graph.entry)
    }

    pub(super) fn reverse_postorder(graph: &Graph) -> Vec<Block> {
        Self::reverse_postorder_from(graph, graph.entry)
    }

    fn reverse_postorder_from(graph: &Graph, entry: Block) -> Vec<Block> {
        let mut visited = vec![false; graph.blocks.len()];
        visited[entry.index()] = true;
        let mut postorder = Vec::new();
        let mut stack = vec![(entry, graph.successors(entry))];
        while let Some((block, successors)) = stack.last_mut() {
            if let Some(next) = successors.next() {
                if !visited[next.index()] {
                    visited[next.index()] = true;
                    stack.push((next, graph.successors(next)));
                }
            } else {
                postorder.push(*block);
                stack.pop();
            }
        }
        postorder.reverse();
        postorder
    }

    pub(super) fn from_entry(graph: &Graph, entry: Block) -> Self {
        let rpo = Self::reverse_postorder_from(graph, entry);
        let mut ranks = SecondaryMap::<Block, u32>::with_capacity(graph.blocks.len());
        for (index, &block) in rpo.iter().enumerate() {
            ranks[block] = index as u32;
        }
        let mut edges = Vec::new();
        let mut offsets = vec![0u32; rpo.len() + 1];
        for &block in &rpo {
            for successor in graph.successors(block) {
                edges.push((ranks[successor], block));
                offsets[ranks[successor] as usize + 1] += 1;
            }
        }
        for index in 1..offsets.len() {
            offsets[index] += offsets[index - 1];
        }
        let mut ranges = SecondaryMap::<Block, (u32, u32)>::with_capacity(graph.blocks.len());
        for (index, &block) in rpo.iter().enumerate() {
            ranges[block] = (offsets[index], offsets[index + 1]);
        }
        let mut list = vec![entry; edges.len()];
        let mut incoming = vec![0u32; edges.len()];
        let mut fill = offsets.clone();
        for &(successor, pred) in &edges {
            let at = &mut fill[successor as usize];
            list[*at as usize] = pred;
            incoming[*at as usize] = ranks[pred];
            *at += 1;
        }
        let mut dominators = vec![u32::MAX; rpo.len()];
        if !rpo.is_empty() {
            dominators[0] = 0;
        }
        let mut changed = true;
        while changed {
            changed = false;
            for index in 1..rpo.len() {
                let preds = &incoming[offsets[index] as usize..offsets[index + 1] as usize];
                let mut processed = preds
                    .iter()
                    .copied()
                    .filter(|&pred| dominators[pred as usize] != u32::MAX);
                let Some(mut next) = processed.next() else {
                    continue;
                };
                for mut pred in processed {
                    while next != pred {
                        if next > pred {
                            next = dominators[next as usize];
                        } else {
                            pred = dominators[pred as usize];
                        }
                    }
                }
                if dominators[index] != next {
                    dominators[index] = next;
                    changed = true;
                }
            }
        }
        let mut size = vec![1u32; rpo.len()];
        for index in (1..rpo.len()).rev() {
            size[dominators[index] as usize] += size[index];
        }
        let mut dominance = SecondaryMap::with_default((u32::MAX, 0));
        let mut next = vec![0u32; rpo.len()];
        if !rpo.is_empty() {
            dominance[entry] = (0, size[0]);
            next[0] = 1;
        }
        for index in 1..rpo.len() {
            let parent = dominators[index] as usize;
            let start = next[parent];
            next[parent] += size[index];
            next[index] = start + 1;
            dominance[rpo[index]] = (start, size[index]);
        }
        Self {
            rpo,
            preds: Preds { ranges, list },
            dominance,
        }
    }

    pub(super) fn dominator_preorder(&self) -> Vec<Block> {
        let mut blocks = self.rpo.clone();
        for &block in &self.rpo {
            blocks[self.dominance[block].0 as usize] = block;
        }
        blocks
    }

    pub(super) fn dominates(&self, dominator: Block, block: Block) -> bool {
        let (position, _) = self.dominance[block];
        let (start, size) = self.dominance[dominator];
        position != u32::MAX && start <= position && position - start < size
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
            views: FxHashMap::default(),
            property_accesses: FxHashMap::default(),
            borrowed: FxHashSet::default(),
            summaries: Default::default(),
            cold: FxHashSet::default(),
            megamorphic_gets: FxHashSet::default(),
            lazy_captures: FxHashSet::default(),
            generator: false,
            has_assumes: false,
            entry_guards: Vec::new(),
            arity: None,
            regions: PrimaryMap::new(),
            blocks,
            insts: PrimaryMap::new(),
            values: PrimaryMap::new(),
            reprs: SecondaryMap::new(),
            ownership: None,
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

    pub(in crate::jit) fn repr(&self, value: Value) -> Repr {
        self.reprs[value].expect("value has no representation")
    }

    pub(super) fn defining_op(&self, value: Value) -> Option<&Op> {
        match self.values[value] {
            ValueDef::Result(inst, _) => Some(&self.insts[inst].op),
            ValueDef::Param(..) => None,
        }
    }

    pub(in crate::jit) fn callees(&self) -> FxHashSet<Value> {
        self.insts
            .values()
            .filter_map(|data| match data.op {
                Op::Call { callee, .. } | Op::CallPack(callee, ..) | Op::CallForward(callee, _) => {
                    Some(callee)
                }
                _ => None,
            })
            .collect()
    }

    pub(super) fn caller_rooted(&self, value: Value) -> bool {
        match self.defining_op(value) {
            Some(Op::This | Op::Callee | Op::Argument(_) | Op::Restored(_)) => true,
            Some(Op::AsObject(source) | Op::AsString(source)) => self.caller_rooted(*source),
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

    pub(in crate::jit) fn set_op(&mut self, inst: Inst, op: Op) {
        let kept = match (&op, self.insts[inst].result()) {
            (Op::Constant(constant), Some(value)) => {
                self.reprs[value].filter(|&repr| repr == Repr::Boxed || repr::fits(constant, repr))
            }
            _ => None,
        };
        let repr = kept.or_else(|| repr::intrinsic(&op));
        for &value in &self.insts[inst].results {
            self.reprs[value] = repr;
        }
        self.insts[inst].op = op;
    }

    pub(in crate::jit) fn insert(
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
        let inline_context = self.blocks[block]
            .insts
            .get(position)
            .map_or(0, |&next| self.insts[next].inline_context);
        self.has_assumes |= matches!(op, Op::Assume(..));
        if let Some(result) = result {
            self.reprs[result] = repr::intrinsic(&op);
        }
        self.insts.push(InstData {
            inline_context,
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
        self.property_accesses.remove(&inst);
        if !self.has_assumes {
            return;
        }
        let results = self.insts[inst].results.clone();
        for block in self.blocks.keys().collect::<Vec<_>>() {
            let insts = std::mem::take(&mut self.blocks[block].insts);
            self.blocks[block].insts = insts
                .into_iter()
                .filter(|&other| {
                    !(matches!(self.insts[other].op, Op::Assume(..))
                        && self.insts[other]
                            .op
                            .operands_iter()
                            .any(|value| results.contains(&value)))
                })
                .collect();
        }
    }

    pub(super) fn remove_insts(&mut self, insts: impl IntoIterator<Item = Inst>) {
        let mut removed = SecondaryMap::<Inst, bool>::new();
        let mut results = SecondaryMap::<Value, bool>::new();
        for inst in insts {
            removed[inst] = true;
            self.views.remove(&inst);
            self.property_accesses.remove(&inst);
            if self.has_assumes {
                for &value in &self.insts[inst].results {
                    results[value] = true;
                }
            }
        }
        if removed.is_empty() {
            return;
        }
        for block in self.blocks.values_mut() {
            block.insts.retain(|&inst| {
                !removed[inst]
                    && !(self.has_assumes
                        && matches!(self.insts[inst].op, Op::Assume(..))
                        && self.insts[inst]
                            .op
                            .operands_iter()
                            .any(|value| results[value]))
            });
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

    pub(super) fn strip_hints(&mut self) {
        for block in self.blocks.keys().collect::<Vec<_>>() {
            let insts = std::mem::take(&mut self.blocks[block].insts);
            self.blocks[block].insts = insts
                .into_iter()
                .filter(|&inst| !matches!(self.insts[inst].op, Op::ReadHint(..)))
                .collect();
        }
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

    pub(super) fn replace_uses(&mut self, aliases: &impl Aliases) {
        if aliases.is_empty() {
            return;
        }
        for data in self.insts.values_mut() {
            for value in data.op.operands_mut() {
                *value = resolve(aliases, *value);
            }
        }
        for read in self.property_accesses.values_mut() {
            if let Some(cache) = &mut read.cache {
                *cache = resolve(aliases, *cache);
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
                if let Some(cache) = self
                    .property_accesses
                    .get(&inst)
                    .and_then(|read| read.cache)
                {
                    counts[cache] += 1;
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
        let operands = || {
            op.operands_iter()
                .fold(ValueType::default(), |types, value| {
                    types.union(facts.of(value))
                })
        };
        let coercing = |result| {
            if operands().within(ValueType::NUMBER) {
                closed(0, Unowned)
            } else {
                closed(COERCING, result)
            }
        };
        match op {
            Op::Aarch64(op) => op.contract(),
            Op::Regex(op) => op.contract(),
            Op::LowLevel(low_level::Op::Reference { .. } | low_level::Op::Cleanup { .. }) => {
                closed(INTERNAL | HEAP_WRITE, None)
            }
            Op::LowLevel(low_level::Op::FrameStore { .. }) => Contract {
                effects: Effects::UNKNOWN,
                result: Unowned,
            },
            Op::LowLevel(low_level::Op::Call { .. }) => Contract {
                effects: Effects::UNKNOWN,
                result: Unowned,
            },
            Op::LowLevel(low_level::Op::Load { .. }) => closed(HEAP_READ, Unowned),
            Op::LowLevel(low_level::Op::AtomicLoad { .. }) => Contract {
                effects: Effects::UNKNOWN,
                result: Unowned,
            },
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
            Op::ArgumentCount => closed(0, Unowned),
            Op::Argument(_) | Op::Restored(_) | Op::This | Op::Callee => closed(0, Borrowed),
            Op::ClosureThis(_) => closed(HEAP_READ, Borrowed),
            Op::NewHoleyArray(_) => closed(ALLOCATION | INTERNAL | ERROR, Owned),
            Op::MapElement(..) => closed(HEAP_READ | INTERNAL, Owned),
            Op::IsArrayIteration(..) | Op::IsStringSplit(_) => closed(HEAP_READ, Unowned),
            Op::StringSplit(..) | Op::StringSplitParts(..) => {
                closed(HEAP_READ | ALLOCATION | INTERNAL | ERROR, Owned)
            }
            Op::Constant(_) => closed(0, Unowned),
            Op::GlobalRead(_) | Op::GlobalReadOptional(_) => {
                closed(GLOBAL_READ | INTERNAL | ERROR, Borrowed)
            }
            Op::RegExp(..) => closed(ALLOCATION | INTERNAL | ERROR, Owned),
            Op::StringConcat(_) => closed(HEAP_READ | ALLOCATION | INTERNAL | ERROR, Owned),
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
                if self.bindings[*binding].captured || self.lazy_captures.contains(binding) {
                    Owned
                } else {
                    Borrowed
                },
            ),
            Op::Initialize(binding, _) | Op::Write(binding, _)
                if self.bindings[*binding].captured =>
            {
                closed(HEAP_WRITE | INTERNAL | ERROR, None)
            }
            Op::Initialize(..) => closed(INTERNAL, None),
            Op::Write(..) => closed(INTERNAL | ERROR, None),
            Op::CellGet(_) | Op::CaptureGet(..) => closed(HEAP_READ | INTERNAL | ERROR, Owned),
            Op::CellSet(..) => closed(HEAP_WRITE | INTERNAL | ERROR, None),
            Op::CheckInitialized(_) => {
                if operands().intersects(ValueType::UNINITIALIZED) {
                    closed(ERROR, None)
                } else {
                    closed(0, None)
                }
            }
            Op::Closure(..)
            | Op::ClosureCells(..)
            | Op::CellNew(_)
            | Op::NewPack
            | Op::NewObject
            | Op::NewArray(_)
            | Op::IteratorNew(_)
            | Op::EnumerateNew(_) => closed(ALLOCATION | INTERNAL | ERROR, Owned),
            Op::PackRelease(_) => closed(HEAP_WRITE | INTERNAL, None),
            Op::Spread(_) => closed(HEAP_READ | ALLOCATION | INTERNAL | ERROR, Owned),
            Op::IteratorNext(_) | Op::EnumerateNext(_) => closed(
                HEAP_READ | HEAP_WRITE | ALLOCATION | INTERNAL | ERROR,
                Owned,
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
            Op::HelperCall {
                helper,
                produces,
                effects,
                ..
            } => Contract {
                effects: *effects,
                result: if produces.is_some() {
                    helper.result_ownership()
                } else {
                    None
                },
            },
            Op::PropertyLoad(..) | Op::RelatedPropertyLoad(..) => closed(HEAP_READ, Borrowed),
            Op::PropertyCache(..) | Op::GlobalCache(..) => closed(0, Unowned),
            Op::IsPropertyCached(..) | Op::IsShape(..) | Op::IsReadLayout(..) => {
                closed(CODE_READ, Unowned)
            }
            Op::ShapedGet(..) | Op::LayoutGet(..) => closed(HEAP_READ | INTERNAL | ERROR, Borrowed),
            Op::ShapedSet(..) => {
                closed(HEAP_READ | HEAP_WRITE | ALLOCATION | INTERNAL | ERROR, None)
            }
            Op::PrimePropertyCache(..) => closed(HEAP_READ | CODE_READ | INTERNAL, None),
            Op::CachedProperty(..) => closed(HEAP_READ | INTERNAL | ERROR, Borrowed),
            Op::FillPropertyStore(.., PropertyWrite::Define) => {
                closed(HEAP_READ | CODE_WRITE | ALLOCATION | INTERNAL | ERROR, None)
            }
            Op::PropertyStore(.., PropertyWrite::Define)
            | Op::PropertyAdd(.., PropertyWrite::Define) => {
                closed(HEAP_READ | CODE_WRITE | INTERNAL, None)
            }
            Op::Set(..) | Op::SetIndex(..) | Op::FillPropertyStore(..) => closed(
                HEAP_READ | HEAP_WRITE | CODE_WRITE | ALLOCATION | INTERNAL | ERROR,
                None,
            ),
            Op::Unary(UnaryOp::Not, _) => closed(HEAP_READ, Unowned),
            Op::Unary(UnaryOp::Typeof, _) => closed(HEAP_READ | ALLOCATION | ERROR, Owned),
            Op::Unary(UnaryOp::Increment | UnaryOp::Decrement, _) => {
                if operands().within(ValueType::NUMBER) {
                    closed(0, Unowned)
                } else {
                    closed(ALLOCATION | INTERNAL | ERROR, Owned)
                }
            }
            Op::ToInt32(_) => closed(0, Unowned),
            Op::ToNumeric(_) | Op::Unary(..) => coercing(Owned),
            Op::GuardedCompare {
                operator,
                left,
                right,
                ..
            } => self.contract(&Op::Binary(*operator, *left, *right), facts),
            Op::Binary(operator, ..) => match operator {
                BinaryOperator::StrictEquality | BinaryOperator::StrictInequality => closed(
                    if operands().within(ValueType::NUMBER) {
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
                _ => coercing(Owned),
            },
            Op::Call { .. }
            | Op::Construct { .. }
            | Op::CallPack(..)
            | Op::CallForward(..)
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
            Op::IsTyped(..) | Op::IsIndexed(_) | Op::IsOrdinaryArray(_) => closed(0, Unowned),
            Op::IsArrayIndex(..) | Op::IsArray(..) => closed(HEAP_READ, Unowned),
            Op::ArrayPosition(_) => closed(INTERNAL, Unowned),
            Op::IsExactPosition(..) | Op::IsInBounds(..) | Op::BooleanAnd(..) => closed(0, Unowned),
            Op::ArrayKind(_) | Op::IndexedLength(_) => closed(HEAP_READ, Unowned),
            Op::ArrayGet(..) => closed(HEAP_READ | INTERNAL | ERROR, Borrowed),
            Op::ArrayLoad(..) => closed(HEAP_READ, Borrowed),
            Op::ArrayStore(..) | Op::PropertyStore(..) => {
                closed(HEAP_READ | HEAP_WRITE | INTERNAL, None)
            }
            Op::PropertyAdd(..) => closed(HEAP_READ | HEAP_WRITE | CODE_WRITE | INTERNAL, None),
            Op::CanOverwrite(..) => closed(HEAP_READ, Unowned),
            Op::PackTake(..) => closed(HEAP_READ | HEAP_WRITE | INTERNAL, Owned),
            Op::TypedIndex(..) => closed(HEAP_READ, Unowned),
            Op::IsNumber(_)
            | Op::IsInteger(_)
            | Op::AsInteger(_)
            | Op::NumericFloat(_)
            | Op::SaturatingInteger(_)
            | Op::IntegerCheck(..)
            | Op::IsString(_) => closed(0, Unowned),
            Op::IsTypeof(..) | Op::Truthy(_) => closed(HEAP_READ, Unowned),
            Op::AsNumber(_) => closed(INTERNAL, Unowned),
            Op::Convert(..) | Op::Box(_) | Op::AsBoxed(_) | Op::AsWord(_) | Op::Unbox(..) => {
                closed(0, Unowned)
            }
            Op::Release(_)
            | Op::Retain(_)
            | Op::SlotStore(_)
            | Op::SlotClear(_)
            | Op::ParamStore(..) => closed(INTERNAL, None),
            Op::AsString(_) => closed(0, Borrowed),
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
        self.reprs[value] == Some(format.repr())
    }

    pub(super) fn verify(&self) -> Result<()> {
        self.verify_representations(false)
    }

    pub(super) fn verify_machine(&self) -> Result<()> {
        self.verify_representations(true)
    }

    fn verify_representations(&self, machine: bool) -> Result<()> {
        if !verification_enabled() {
            return Ok(());
        }
        let mut stacks: SecondaryMap<Block, Option<Vec<Scope>>> = SecondaryMap::new();
        stacks[self.entry] = Some(Vec::new());
        let mut worklist = vec![self.entry];
        while let Some(block) = worklist.pop() {
            let mut stack = stacks[block].clone().unwrap();
            for &inst in &self.blocks[block].insts {
                if self.insts[inst].block != block {
                    return Err(format!("{inst} is listed in the wrong block"));
                }
                if let Op::LowLevel(op) = &self.insts[inst].op
                    && !matches!(op, low_level::Op::FrameStore { .. })
                    && self.insts[inst]
                        .results
                        .iter()
                        .any(|&value| self.reprs[value] == Some(Repr::Boxed))
                {
                    return Err(format!("{inst} ({op:?}) produces a boxed word"));
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
            if machine
                && let TerminatorKind::Branch(value, ..) = terminator.kind
                && self.repr(value) != Repr::Boolean
            {
                return Err(format!("{block} branches on a non-boolean scalar"));
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
                            !self.reprs[value].is_some_and(|repr| repr.fits(param.value_type))
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
                        !self.reprs[value].is_some_and(|repr| repr.fits(param.value_type))
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
                    let compatible = match (self.reprs[arg], self.reprs[param]) {
                        (_, None) => true,
                        (Some(arg), Some(param)) => {
                            arg.compatible(param)
                                || (!machine && arg != Repr::Boxed && param != Repr::Boxed)
                        }
                        (None, Some(_)) => false,
                    };
                    if !compatible {
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
                    if machine
                        && args
                            .iter()
                            .zip(&signature.params)
                            .any(|(&value, param)| !self.repr(value).fits(param.value_type))
                    {
                        return Err(format!("{inst} has an invalid call argument type"));
                    }
                    for (&value, ty) in instruction.results.iter().zip(&signature.returns) {
                        if !self.reprs[value].is_some_and(|repr| repr.fits(ty.value_type)) {
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
                        && matches!(
                            self.insts[inst].op,
                            Op::LowLevel(_)
                                | Op::Box(_)
                                | Op::Convert(..)
                                | Op::NumericFloat(_)
                                | Op::SaturatingInteger(_)
                                | Op::AsInteger(_)
                                | Op::ArrayPosition(_)
                                | Op::ToInt32(_)
                                | Op::AsNumber(_)
                        )
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
                if let Some(cache) = self
                    .property_accesses
                    .get(&inst)
                    .and_then(|read| read.cache)
                {
                    check(cache, index)?;
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
    pub(in crate::jit) inlines: Vec<specialization::calls::Inline>,
    pub(in crate::jit) reads: Vec<specialization::reads::Read>,
    stores: Vec<specialization::reads::Read>,
    pub(in crate::jit) shapes: specialization::shapes::Plan,
    pub(in crate::jit) numbers: Vec<(super::profile::Site, super::profile::ReadType)>,
    pub(in crate::jit) conditional: specialization::calls::ConditionalPlan,
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
    // Consumed by JS lowering; continuation graphs are already prepared.
    specialized_regions: Option<Vec<RegionId>>,
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
            specialized_regions: None,
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
}

#[cfg(test)]
mod tests;

pub(in crate::jit) fn lower_intrinsics(function: &mut FunctionState) -> crate::Result<()> {
    js::intrinsics::lower(&mut function.graph).map_err(crate::Error::Unsupported)?;
    function.invalidate_cfg();
    Ok(())
}

use super::super::{Binding, Inst, MemoryFormat, Operands, OperandsMut, Scope, Value};
use super::memory::Field;
use crate::heap::TypedKind;
use crate::jit::layout::ReadLayout;
use crate::jit::math::METHODS;
use oxc_span::Span;
use oxc_syntax::operator::BinaryOperator;
use std::fmt;

#[derive(Clone, Debug)]
pub(in crate::jit) enum Constant {
    Undefined,
    Null,
    Boolean(bool),
    Number(f64),
    String(String),
    Uninitialized,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::jit) enum UnaryOp {
    Negate,
    Plus,
    Not,
    BitwiseNot,
    Typeof,
    Increment,
    Decrement,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in crate::jit) struct Native(usize);

impl Native {
    pub(in crate::jit) fn index(self) -> usize {
        self.0
    }

    pub(in crate::jit) fn lookup(global: &str, property: &str) -> Option<Self> {
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
pub(in crate::jit) enum ArrayMethod {
    Map,
    Some,
    Every,
    Filter,
    ForEach,
    Find,
    Push,
    Reduce,
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

#[derive(Clone, Copy, Debug)]
pub(in crate::jit) enum PropertyWrite {
    Set,
    Define,
}

/// What a typed-array read may assume: a fixed element kind, and a key proven to be an
/// integral index below the length.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::jit) struct TypedRead {
    pub(in crate::jit) element: Option<TypedKind>,
    pub(in crate::jit) bounded: bool,
}

#[derive(Clone, Debug)]
pub(in crate::jit) enum Op {
    Arguments,
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
    PropertyLoad(Value, Value, Field),
    RelatedPropertyLoad(Value, Value, Field),
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
    Delete(Value, Value),
    Set(Value, String, Value),
    SetIndex(Value, Value, Value),
    // Bounded packed-array replacement; storage retains the incoming value before releasing the old one.
    ArrayStore(Value, Value, Value),
    PropertyStore(Value, Value, Value, Field, PropertyWrite),
    PropertyAdd(Value, Value, Value, PropertyWrite),
    // Native slot replacement must go through the runtime setter.
    CanOverwrite(Value, Value, Field),
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
    // Typed-array store whose key is an exact position within the length.
    TypedSet(Value, Value, Value, TypedKind),
    TypedData(Value),
    MemoryNumber(Value, MemoryFormat),
    IsNumber(Value),
    IsInteger(Value),
    IsIntegerInRange(Value, i32, i32),
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
    AsObject(Value),
    EnterScope(Scope),
    EnterCaptures(Scope, Value),
    LeaveScope(Scope),
    NextIteration(Scope),
    FreshIteration(Scope),
}

impl Op {
    pub(in crate::jit::semantic) fn operands_iter(&self) -> Operands<'_> {
        let (fixed, rest): ([Option<Value>; 3], &[Value]) = match self {
            Op::PropertyCache(..)
            | Op::GlobalCache(..)
            | Op::Arguments
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
            | Op::IsArray(value, _)
            | Op::IsTyped(value, _)
            | Op::IsNumber(value)
            | Op::IsInteger(value)
            | Op::IsIntegerInRange(value, ..)
            | Op::AsInteger(value)
            | Op::NumericFloat(value)
            | Op::SaturatingInteger(value)
            | Op::IsString(value)
            | Op::Truthy(value)
            | Op::IsStringSplit(value)
            | Op::IsTypeof(value, _)
            | Op::AsString(value)
            | Op::AsObject(value)
            | Op::AsNumber(value) => ([Some(*value), None, None], &[]),
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
            Op::MapElement(object, key)
            | Op::IsArrayIteration(object, key, _)
            | Op::CanOverwrite(object, key, _)
            | Op::Delete(object, key)
            | Op::GetIndex(object, key)
            | Op::TypedIndex(object, key, _)
            | Op::ArrayGet(object, key)
            | Op::ArrayLoad(object, key, _)
            | Op::PackTake(object, key)
            | Op::ProbeIndex(object, key)
            | Op::IsExactPosition(object, key)
            | Op::IsInBounds(object, key)
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
            | Op::SetIndex(object, key, value)
            | Op::TypedSet(object, key, value, _) => {
                ([Some(*object), Some(*key), Some(*value)], &[])
            }
            Op::StringConcat(values) | Op::NewArray(values) | Op::CallNative(_, values) => {
                ([None; 3], values)
            }
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

    pub(in crate::jit::semantic) fn operands_mut(&mut self) -> OperandsMut<'_> {
        let (fixed, rest): ([Option<&mut Value>; 3], &mut [Value]) = match self {
            Op::PropertyCache(..)
            | Op::GlobalCache(..)
            | Op::Arguments
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
            | Op::IsArray(value, _)
            | Op::IsTyped(value, _)
            | Op::IsNumber(value)
            | Op::IsInteger(value)
            | Op::IsIntegerInRange(value, ..)
            | Op::AsInteger(value)
            | Op::NumericFloat(value)
            | Op::SaturatingInteger(value)
            | Op::IsString(value)
            | Op::Truthy(value)
            | Op::IsStringSplit(value)
            | Op::IsTypeof(value, _)
            | Op::AsString(value)
            | Op::AsObject(value)
            | Op::AsNumber(value) => ([Some(value), None, None], &mut []),
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
            Op::MapElement(object, key)
            | Op::IsArrayIteration(object, key, _)
            | Op::CanOverwrite(object, key, _)
            | Op::Delete(object, key)
            | Op::GetIndex(object, key)
            | Op::TypedIndex(object, key, _)
            | Op::ArrayGet(object, key)
            | Op::ArrayLoad(object, key, _)
            | Op::PackTake(object, key)
            | Op::ProbeIndex(object, key)
            | Op::IsExactPosition(object, key)
            | Op::IsInBounds(object, key)
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
            | Op::SetIndex(object, key, value)
            | Op::TypedSet(object, key, value, _) => {
                ([Some(object), Some(key), Some(value)], &mut [])
            }
            Op::StringConcat(values) | Op::NewArray(values) | Op::CallNative(_, values) => {
                ([None, None, None], values)
            }
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

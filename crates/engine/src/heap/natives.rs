use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NativeFunction {
    DateNow,
    ObjectKeys,
    ObjectValues,
    ObjectGetOwnPropertyNames,
    ObjectFreeze,
    ObjectIsFrozen,
    Array,
    ArrayFrom,
    ArrayIsArray,
    TypedConstructor(TypedKind),
    TypedFrom(TypedKind),
    TypedFill,
    Math(MathFunction),
    JsonParse,
    JsonStringify,
    RegExpTest,
    RegExpExec,
    StringMatch,
    HostSeedRandom,
    HostReadFile,
    Print,
    ConsoleLog,
    HostListFiles,
    HostEvaluateSource,
    Error,
    Set,
    SetAdd,
    SetHas,
    SetDelete,
    SetClear,
    GeneratorNext,
    ArrayFilter,
    ArrayMap,
    ArrayEvery,
    ArraySome,
    ArrayFind,
    ArrayForEach,
    ArrayAt,
    ArraySort,
    ArrayJoin,
    ArrayPop,
    ArrayIndexOf,
    ArrayConcat,
    ArrayPush,
    ArrayIncludes,
    ArrayReverse,
    ArraySlice,
    ArrayFlat,
    ArrayFlatMap,
    ArrayFill,
    NumberToFixed,
    NumberToString,
    StringSlice,
    StringSplit,
    StringStartsWith,
    StringReplace,
    StringPadStart,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MathFunction {
    Random,
    Abs,
    Ceil,
    Floor,
    Round,
    Trunc,
    Sqrt,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Atan2,
    Exp,
    Log,
    Pow,
    Min,
    Max,
    Imul,
    Sign,
}

impl Heap {
    pub(crate) fn native_function(&self, function: NativeFunction) -> Option<Value> {
        let mut data = self.0.borrow_mut();
        let id = if let Some((_, id)) = data.natives.iter().find(|(kind, _)| *kind == function) {
            *id
        } else {
            let id = data.reserve()?;
            data.store(id, Payload::Native(function));
            data.natives.push((function, id));
            id
        };
        data.slot_mut(id).references += 1;
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn native(&self, value: Value) -> Option<NativeFunction> {
        let data = self.0.borrow();
        let slot = data.live(heap_id(value)?)?;
        match &slot.payload {
            Payload::Native(function) => Some(*function),
            _ => None,
        }
    }
}

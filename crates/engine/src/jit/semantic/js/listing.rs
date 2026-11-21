use super::super::{Graph, Value};
use super::{ArrayLayout, Constant, Op};

pub(in crate::jit::semantic) fn op_text(graph: &Graph, op: &Op) -> String {
    let list = |values: &[Value]| {
        values
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    };
    match op {
        Op::Arguments => "arguments".into(),
        Op::CallForward(callee, receiver) => format!("call_forward {callee}, this={receiver}"),
        Op::Constant(Constant::Undefined) => "undefined".into(),
        Op::Constant(Constant::Null) => "null".into(),
        Op::Rest(index) => format!("rest {index}"),
        Op::This => "this".into(),
        Op::Callee => "callee".into(),
        Op::ClosureThis(value) => format!("closure_this {value}"),
        Op::Throw(value) => format!("throw {value}"),
        Op::Catch => "catch".into(),
        Op::Closure(span, _, arrow, generator, _, captures, _) => format!(
            "closure {}..{} arrow={arrow} generator={generator} captures={captures:?}",
            span.start, span.end
        ),
        Op::ClosureCells(span, _, arrow, generator, _, cells, _) => format!(
            "closure_cells {}..{} arrow={arrow} generator={generator} cells={cells:?}",
            span.start, span.end
        ),
        Op::CellNew(value) => format!("cell_new {value}"),
        Op::CellGet(cell) => format!("cell_get {cell}"),
        Op::CellSet(cell, value) => format!("cell_set {cell}, {value}"),
        Op::CaptureGet(closure, index) => format!("capture_get {closure}, {index}"),
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
        Op::BindingLoad(binding) => format!("binding_load {}", graph.binding_name(*binding)),
        Op::BindingRestore(binding, value) => {
            format!("binding_restore {}, {value}", graph.binding_name(*binding))
        }
        Op::Read(binding) => format!("read {}", graph.binding_name(*binding)),
        Op::Initialize(binding, value) => {
            format!("initialize {}, {value}", graph.binding_name(*binding))
        }
        Op::Write(binding, value) => format!("write {}, {value}", graph.binding_name(*binding)),
        Op::CheckInitialized(value) => format!("check_initialized {value}"),
        Op::NewPack => "new_pack".into(),
        Op::PackRelease(value) => format!("pack_release {value}"),
        Op::Spread(value) => format!("spread {value}"),
        Op::PackAppend(pack, value, spread) => format!("pack_append {pack}, {value}, {spread}"),
        Op::NewObject => "new_object".into(),
        Op::StringConcat(values) => format!("string_concat [{}]", list(values)),
        Op::NewArray(values) => format!("new_array [{}]", list(values)),
        Op::NewHoleyArray(length) => format!("new_holey_array {length}"),
        Op::MapElement(object, key) => format!("map_element {object}[{key}]"),
        Op::IsArrayIteration(callee, receiver, method) => {
            format!("is_array_{} {callee}, {receiver}", method.name())
        }
        Op::ArrayAppend(array, value, spread) => {
            format!("array_append {array}, {value}, {spread}")
        }
        Op::CallPack(callee, receiver, args) => {
            format!("call_pack {callee}, {receiver}, {args}")
        }
        Op::NewInstance(prototype) => format!("new_instance {prototype}"),
        Op::ConstructorResult(result, instance) => {
            format!("constructor_result {result}, {instance}")
        }
        Op::ConstructorCall {
            callee,
            receiver,
            args,
        } => format!("constructor_call {callee}, {receiver}({})", list(args)),
        Op::ConstructorCallPack(callee, receiver, args) => {
            format!("constructor_call_pack {callee}, {receiver}, {args}")
        }
        Op::ConstructorPrototype(callee, prototype) => {
            format!("constructor_prototype {callee}, {prototype}")
        }
        Op::ConstructResolved {
            callee,
            prototype,
            args,
        } => format!("construct_resolved {callee}, {prototype}({})", list(args)),
        Op::ConstructPackResolved(callee, prototype, args) => {
            format!("construct_pack_resolved {callee}, {prototype}, {args}")
        }
        Op::ConstructPack(callee, args) => format!("construct_pack {callee}, {args}"),
        Op::IteratorNew(value) => format!("iterator_new {value}"),
        Op::IteratorNext(value) => format!("iterator_next {value}"),
        Op::IsIndexed(value) => format!("is_indexed {value}"),
        Op::IsOrdinaryArray(value) => format!("is_array {value}"),
        Op::IsExactPosition(key, position) => format!("is_exact_position {key}, {position}"),
        Op::IsInBounds(position, length) => format!("is_in_bounds {position}, {length}"),
        Op::ArrayPosition(key) => format!("array_position {key}"),
        Op::IsArrayIndex(object, key, position, layout) => format!(
            "is_array_index {object}[{key}] position={position} {}",
            match layout {
                ArrayLayout::Packed => "packed",
                ArrayLayout::Holey => "holey",
            }
        ),
        Op::IsArray(object, layout) => format!(
            "is_array {object} {}",
            match layout {
                ArrayLayout::Packed => "packed",
                ArrayLayout::Holey => "holey",
            }
        ),
        Op::ArrayLoad(object, key, layout) => format!(
            "load_array {object}[{key}] {}",
            match layout {
                ArrayLayout::Packed => "packed",
                ArrayLayout::Holey => "holey",
            }
        ),
        Op::ArrayKind(value) => format!("array_kind {value}"),
        Op::IndexedLength(value) => format!("indexed_length {value}"),
        Op::ArrayGet(object, key) => format!("array_get {object}[{key}]"),
        Op::PackTake(object, key) => format!("pack_take {object}[{key}]"),
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
        Op::IsPropertyCached(object, cache, _) => {
            format!("is_property_cached {object} {cache}")
        }
        Op::CachedProperty(object, cache, _) => format!("cached_property {object} {cache}"),
        Op::IsShape(object, shape) => format!("is_shape {object}, {shape}"),
        Op::IsReadLayout(object, layout) => format!("is_read_layout {object} {layout:?}"),
        Op::LayoutGet(object, name, layout) => format!("layout_get {object}.{name} {layout:?}"),
        Op::ShapedGet(object, name, index) => {
            format!("shaped_get {object}.{name} @{index}")
        }
        Op::ShapedSet(object, name, index, value) => {
            format!("shaped_set {object}.{name} @{index}, {value}")
        }
        Op::FillPropertyCache(object, cache, name) => {
            format!("fill_property_cache {object}.{name} {cache}")
        }
        Op::PrimePropertyCache(object, cache) => {
            format!("prime_property_cache {object} {cache}")
        }
        Op::Delete(object, key) => format!("delete {object}[{key}]"),
        Op::GetIndex(object, key) => format!("get {object}[{key}]"),
        Op::Set(object, name, value) => format!("set {object}.{name}, {value}"),
        Op::ArrayStore(object, key, value) => format!("store_array {object}[{key}], {value}"),
        Op::PropertyStore(object, key, value, field, _) => {
            format!("store_property {object}[{key}], {value} field={field:?}")
        }
        Op::PropertyAdd(object, cache, value, _) => {
            format!("add_property {object}, {cache}, {value}")
        }
        Op::CanOverwrite(object, key, field) => {
            format!("can_overwrite {object}[{key}] {field:?}")
        }
        Op::RelatedPropertyLoad(object, address, field) => {
            format!("related_property_load {object} {address} {field:?}")
        }
        Op::PropertyLoad(object, address, field) => {
            format!("property_load {object} {address} field={field:?}")
        }
        Op::GlobalCache(name) => format!("global_cache {name}"),
        Op::PropertyCache(site, name) => format!("property_cache {site} {name}"),
        Op::FillPropertyStore(object, cache, name, value, _) => {
            format!("fill_property_store {object}.{name}, {value} {cache}")
        }
        Op::SetIndex(object, key, value) => format!("set {object}[{key}], {value}"),
        Op::TypedSet(object, key, value, element) => {
            format!("typed_set {object}[{key}], {value} {element:?}")
        }
        Op::ToNumeric(value) => format!("to_numeric {value}"),
        Op::ToInt32(value) => format!("to_int32 {value}"),
        Op::Unary(operator, value) => format!("{operator:?} {value}").to_lowercase(),
        Op::GuardedCompare {
            operator,
            left,
            right,
            numbers,
        } => format!(
            "compare {left} {} {right} numbers=[{}, {}]",
            operator.as_str(),
            numbers[0],
            numbers[1]
        ),
        Op::Binary(operator, left, right) => {
            format!("{left} {} {right}", operator.as_str())
        }
        Op::Call {
            callee,
            receiver,
            args,
            target,
        } => {
            let call = match receiver {
                Some(receiver) => format!("call {callee}({}) this={receiver}", list(args)),
                None => format!("call {callee}({})", list(args)),
            };
            match target {
                Some(target) => match target.variant {
                    crate::jit::entries::VariantId::Ordinary(_) => {
                        format!("{call} direct={}", target.code)
                    }
                    crate::jit::entries::VariantId::Specialized(_) => {
                        format!("{call} direct={} specialized", target.code)
                    }
                },
                None => call,
            }
        }
        Op::Construct { callee, args } => format!("construct {callee}({})", list(args)),
        Op::CallNative(native, args) => format!("call_native {native}({})", list(args)),
        Op::IsUndefined(value) => format!("is_undefined {value}"),
        Op::IsNullish(value) => format!("is_nullish {value}"),
        Op::ProbeGlobal(name) => format!("probe_global {name}"),
        Op::ProbeProperty(object, name) => format!("probe {object}.{name}"),
        Op::ProbeShapeProperty(object, name, layout) => {
            format!("probe_shape_property {object}.{name} {layout:?}")
        }
        Op::ProbeIndex(object, key) => format!("probe {object}[{key}]"),
        Op::IsNative(value, native) => format!("is_native {value}, {native}"),
        Op::IsCode(value, code) => format!("is_code {value}, {code}"),
        Op::GuardValid(guard) => format!("guard_valid {guard}"),
        Op::GuardCommit(guard) => format!("guard_commit {guard}"),
        Op::View(value) => format!("view {value}"),
        Op::TypedData(value) => format!("typed_data {value}"),
        Op::MemoryNumber(value, format) => format!("memory_number {value} {format:?}"),
        Op::IsTyped(value, None) => format!("is_typed {value}"),
        Op::IsTyped(value, Some(element)) => format!("is_typed {value}, {element:?}"),
        Op::TypedIndex(object, key, read) => {
            let element = read
                .element
                .map_or(String::new(), |element| format!(" {element:?}"));
            let bounded = if read.bounded { " bounded" } else { "" };
            format!("get_typed {object}[{key}]{element}{bounded}")
        }
        Op::IsInteger(value) => format!("is_integer {value}"),
        Op::IsIntegerInRange(value, min, max) => {
            format!("is_integer_in_range {value}, {min}, {max}")
        }
        Op::AsInteger(value) => format!("as_integer {value}"),
        Op::NumericFloat(value) => format!("numeric_float {value}"),
        Op::SaturatingInteger(value) => format!("saturating_integer {value}"),
        Op::IntegerCheck(value, float, exact) => {
            format!("integer_check {value}, {float}, {exact}")
        }
        Op::IsNumber(value) => format!("is_number {value}"),
        Op::IsString(value) => format!("is_string {value}"),
        Op::Truthy(value) => format!("truthy {value}"),
        Op::IsStringSplit(value) => format!("is_string_split {value}"),
        Op::StringSplitParts(receiver, args) => format!(
            "split_parts {receiver}({})",
            args.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Op::StringSplit(receiver, args) => format!("string_split {receiver}({})", list(args)),
        Op::IsTypeof(value, kind) => format!("is_typeof {value} {kind}"),
        Op::AsNumber(value) => format!("as_number {value}"),
        Op::AsString(value) => format!("as_string {value}"),
        Op::AsObject(value) => format!("as_object {value}"),
        Op::EnterScope(scope) => format!("enter {scope}"),
        Op::EnterCaptures(scope, closure) => format!("enter_captures {scope}, {closure}"),
        Op::LeaveScope(scope) => format!("leave {scope}"),
        Op::NextIteration(scope) => format!("next_iteration {scope}"),
        Op::FreshIteration(scope) => format!("fresh_iteration {scope}"),
    }
}

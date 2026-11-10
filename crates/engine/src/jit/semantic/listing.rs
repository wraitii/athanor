use super::*;
use std::fmt;

#[derive(Clone, Copy)]
pub(in crate::jit) struct SourceLocation {
    pub(in crate::jit) source: Option<usize>,
    pub(in crate::jit) span: Span,
}

pub(in crate::jit) type ListingLine = (u32, Option<SourceLocation>, String);

impl Graph {
    fn binding_name(&self, binding: Binding) -> String {
        format!("{}#{}", self.bindings[binding].name, binding.index())
    }

    pub(in crate::jit) fn entering_guard(&self, inst: Inst) -> String {
        let block = self.insts[inst].block;
        self.blocks
            .values()
            .find_map(|data| match &data.terminator.as_ref()?.kind {
                TerminatorKind::Branch(condition, then, otherwise)
                    if then.block == block || otherwise.block == block =>
                {
                    Some(self.defining_op(*condition).map_or_else(
                        || format!("branch {condition}"),
                        |op| self.op_text(op),
                    ))
                }
                _ => None,
            })
            .unwrap_or_default()
    }

    pub(in crate::jit) fn op_text(&self, op: &Op) -> String {
        let list = |values: &[Value]| {
            values
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        };
        match op {
            Op::Observe(value, kind, site) => {
                format!("observe {kind:?} {value} site={site}")
            }
            Op::OsrStart(_, index, values) => format!("osr_start {index} [{}]", list(values)),
            Op::Continuation(body, values) => {
                format!("continuation {} [{}]", body.graph.name, list(values))
            }
            Op::ContinuationOutput(value, index) => {
                format!("continuation_output {value}, {index}")
            }
            Op::ContinuationStore(values) => format!("continuation_store [{}]", list(values)),
            Op::Argument(index) => format!("argument {index}"),
            Op::Arguments => "arguments".into(),
            Op::Constant(Constant::Undefined) => "undefined".into(),
            Op::Constant(Constant::Null) => "null".into(),
            Op::Rest(index) => format!("rest {index}"),
            Op::This => "this".into(),
            Op::ClosureThis(value) => format!("closure_this {value}"),
            Op::Throw(value) => format!("throw {value}"),
            Op::Catch => "catch".into(),
            Op::Closure(span, _, arrow, generator, _, captures, _) => format!(
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
            Op::NewPack => "new_pack".into(),
            Op::PackRelease(value) => format!("pack_release {value}"),
            Op::Spread(value) => format!("spread {value}"),
            Op::PackAppend(pack, value, spread) => format!("pack_append {pack}, {value}, {spread}"),
            Op::NewObject => "new_object".into(),
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
            Op::ConstructorPrototype(callee) => format!("constructor_prototype {callee}"),
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
            Op::IsExactPosition(key, position) => format!("is_exact_position {key}, {position}"),
            Op::IsInBounds(position, length) => format!("is_in_bounds {position}, {length}"),
            Op::ArrayPosition(key) => format!("array_position {key}"),
            Op::IsArrayIndex(object, key, position, holey) => format!(
                "is_array_index {object}[{key}] position={position} {}",
                if *holey { "holey" } else { "packed" }
            ),
            Op::ArrayLoad(object, key, holey) => format!(
                "load_array {object}[{key}] {}",
                if *holey { "holey" } else { "packed" }
            ),
            Op::IndexedLength(value) => format!("indexed_length {value}"),
            Op::IndexedGet(object, key) => format!("get_indexed {object}[{key}]"),
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
            Op::IsPropertyCached(object, _, name) => {
                format!("property_cached {object}.{name}")
            }
            Op::CachedProperty(object, _) => format!("cached_property {object}"),
            Op::IsShape(object, shape) => format!("is_shape {object}, {shape}"),
            Op::ShapedGet(object, name, index) => {
                format!("shaped_get {object}.{name} @{index}")
            }
            Op::ShapedSet(object, name, index, value) => {
                format!("shaped_set {object}.{name} @{index}, {value}")
            }
            Op::FillPropertyCache(object, _, name) => {
                format!("fill_property_cache {object}.{name}")
            }
            Op::PrimePropertyCache(object, _, name) => {
                format!("prime_property_cache {object}.{name}")
            }
            Op::GetIndex(object, key) => format!("get {object}[{key}]"),
            Op::Set(object, name, value) => format!("set {object}.{name}, {value}"),
            Op::SetIndex(object, key, value) => format!("set {object}[{key}], {value}"),
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
                    Some(target) => format!("{call} direct={}", target.code),
                    None => call,
                }
            }
            Op::Construct { callee, args } => format!("construct {callee}({})", list(args)),
            Op::CallNative(native, args) => format!("call_native {native}({})", list(args)),
            Op::IsUndefined(value) => format!("is_undefined {value}"),
            Op::IsNullish(value) => format!("is_nullish {value}"),
            Op::ProbeGlobal(name) => format!("probe_global {name}"),
            Op::ProbeProperty(object, name) => format!("probe {object}.{name}"),
            Op::ProbeIndex(object, key) => format!("probe {object}[{key}]"),
            Op::IsNative(value, native) => format!("is_native {value}, {native}"),
            Op::IsCode(value, code) => format!("is_code {value}, {code}"),
            Op::GuardValid(guard) => format!("guard_valid {guard}"),
            Op::GuardCommit(guard) => format!("guard_commit {guard}"),
            Op::View(value) => format!("view {value}"),
            Op::IsTyped(value, None) => format!("is_typed {value}"),
            Op::IsTyped(value, Some(element)) => format!("is_typed {value}, {element:?}"),
            Op::TypedIndex(object, key, read) => {
                let element = read
                    .element
                    .map_or(String::new(), |element| format!(" {element:?}"));
                let bounded = if read.bounded { " bounded" } else { "" };
                format!("get_typed {object}[{key}]{element}{bounded}")
            }
            Op::ReadHint(value, kind, observed, site) => {
                format!("read_hint {kind:?} {value} {observed:?} site={site}")
            }
            Op::IsInteger(value) => format!("is_integer {value}"),
            Op::AsInteger(value) => format!("as_integer {value}"),
            Op::IsNumber(value) => format!("is_number {value}"),
            Op::IsString(value) => format!("is_string {value}"),
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
            Op::AsObject(value) => format!("as_object {value}"),
            Op::EnterScope(scope) => format!("enter {scope}"),
            Op::EnterCaptures(scope, closure) => format!("enter_captures {scope}, {closure}"),
            Op::LeaveScope(scope) => format!("leave {scope}"),
            Op::NextIteration(scope) => format!("next_iteration {scope}"),
            Op::FreshIteration(scope) => format!("fresh_iteration {scope}"),
        }
    }

    pub(in crate::jit) fn listing<'a>(&'a self, facts: &'a Facts) -> Listing<'a> {
        Listing { graph: self, facts }
    }
}

pub(in crate::jit) struct Listing<'a> {
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
    pub(in crate::jit) fn lines(&self) -> Vec<ListingLine> {
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
                let effects = graph.contract(&inst_data.op, self.facts).effects;
                let note = match (effects.memory.reenters(), effects.control.throws) {
                    (false, true) => " throws",
                    (false, false) => "",
                    (true, _) => " reenters, throws",
                };
                let span = inst_data.span;
                lines.push((
                    graph.inst_location(inst),
                    Some(SourceLocation {
                        source: inst_data.source_name.map(|name| name.0),
                        span,
                    }),
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
                    TerminatorKind::Fallback(entry) => format!("fallback {entry:#x}"),
                };
                let span = terminator.span;
                lines.push((
                    graph.terminator_location(block),
                    Some(SourceLocation {
                        source: terminator.source_name.map(|name| name.0),
                        span,
                    }),
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
    pub(in crate::jit) fn inst_location(&self, inst: Inst) -> u32 {
        u32::try_from(inst.index() + 1).unwrap()
    }

    pub(in crate::jit) fn terminator_location(&self, block: Block) -> u32 {
        u32::try_from(self.insts.len() + block.index() + 1).unwrap()
    }
}

impl Analysis {
    pub(in crate::jit) fn listing_lines(&self) -> Vec<ListingLine> {
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

impl Analysis {
    pub(in crate::jit) fn plan_report(
        &self,
        repository: &super::specialize::Repository,
        source: usize,
        inline_multiplier: usize,
    ) -> String {
        use std::fmt::Write;

        let mut out = String::new();
        writeln!(
            out,
            "inline budget: {} instructions, {} per callee (multiplier {inline_multiplier})",
            super::specialize::SPECIALIZATION_BUDGET.saturating_mul(inline_multiplier),
            super::specialize::SPECIALIZATION_BODY_INSTS.saturating_mul(inline_multiplier)
        )
        .unwrap();
        let relevant = |site: &crate::jit::profile::Site| {
            site.source == source
                && self.graph.span.start <= site.start
                && site.end <= self.graph.span.end
        };
        let mut feedback = Vec::new();
        for (site, target) in &repository.feedback {
            if relevant(site) {
                feedback.push(format!(
                    "  call {site} -> {target} ({})",
                    repository.name(*target).unwrap_or("unknown")
                ));
            }
        }
        for (site, native) in &repository.native_feedback {
            if relevant(site) {
                feedback.push(format!("  call {site} -> native {native:?}"));
            }
        }
        for ((site, kind), observed) in &repository.read_feedback {
            if relevant(site) {
                feedback.push(format!("  read {site} {kind:?} -> {observed:?}"));
            }
        }
        feedback.sort();
        writeln!(out, "feedback: {}", feedback.len()).unwrap();
        for entry in feedback {
            writeln!(out, "{entry}").unwrap();
        }
        writeln!(out, "inlines: {}", self.plan.inlines.len()).unwrap();
        for inline in &self.plan.inlines {
            writeln!(
                out,
                "  {} -> {} ({}) guard_code={} arrow={}",
                inline.call,
                inline.target,
                repository.name(inline.target).unwrap_or("unknown"),
                inline.guard_code,
                inline.arrow
            )
            .unwrap();
        }
        writeln!(out, "numeric observations: {}", self.plan.numbers.len()).unwrap();
        for (site, observed) in &self.plan.numbers {
            writeln!(out, "  {site}: {observed:?}").unwrap();
        }
        writeln!(out, "read refinements: {}", self.plan.reads.len()).unwrap();
        for read in &self.plan.reads {
            writeln!(
                out,
                "  {} typed={} element={:?} guard_number={}",
                read.inst, read.typed, read.element, read.guard_number
            )
            .unwrap();
        }
        writeln!(
            out,
            "conditional calls: {}",
            self.plan.conditional.calls.len()
        )
        .unwrap();
        for call in &self.plan.conditional.calls {
            writeln!(
                out,
                "  {} {} callee={} {} {}",
                call.inst, call.name, call.callee, call.memory, call.control
            )
            .unwrap();
            for (name, target) in &call.identities {
                writeln!(out, "    identity {name}={target}").unwrap();
            }
        }
        if let Some(osr) = &self.osr {
            writeln!(out, "{}", osr.report(&self.graph.name)).unwrap();
        }
        writeln!(out, "regions:").unwrap();
        let mut reported = false;
        for note in &self.notes {
            if note.starts_with("region plan before versioning:")
                || note.starts_with("region emission:")
            {
                writeln!(out, "{note}").unwrap();
                reported = true;
            }
        }
        if !reported {
            writeln!(out, "  none").unwrap();
        }
        for note in &self.notes {
            if !note.starts_with("region plan before versioning:")
                && !note.starts_with("region emission:")
            {
                writeln!(out, "  note: {note}").unwrap();
            }
        }
        out
    }
}

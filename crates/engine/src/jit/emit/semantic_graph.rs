use std::cell::RefCell;

use cranelift_codegen::entity::{EntityRef, SecondaryMap};
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{BlockArg, InstBuilder, MemFlagsData, Value as ClifValue, types};
use cranelift_frontend::Variable;

use crate::error::Error;
use crate::heap::kind;
use crate::jit::helpers::Helper;
use crate::jit::semantic::{
    Constant, Facts, Graph, Op, ScopeKind, TerminatorKind, TypedRead, Types, UnaryOp,
    Value as GraphValue,
};
use crate::value::{TAG_SHIFT, TAG_STRING, Value};

use super::FunctionCompiler;
use super::ops::{self, BinaryOp, UnaryOp as EmitUnaryOp};
use super::repr::{Operand, Repr};
use super::slots::{self, Slots};

const HEAP: Types = Types::STRING.union(Types::OBJECT).union(Types::OTHER);
const NUMERIC: Types = Types::NUMBER.union(Types::UNDEFINED);

struct Context<'a> {
    osr_buffers: RefCell<std::collections::HashMap<GraphValue, (ClifValue, usize)>>,
    values: &'a [Variable],
    direct: Option<RefCell<SecondaryMap<GraphValue, Option<ClifValue>>>>,
    reprs: &'a SecondaryMap<GraphValue, Repr>,
    facts: &'a Facts,
    slots: Slots,
}

impl Context<'_> {
    fn define(&self, value: GraphValue, clif: ClifValue) {
        if let Some(direct) = &self.direct {
            direct.borrow_mut()[value] = Some(clif);
        }
    }

    fn numeric(&self, value: GraphValue) -> bool {
        self.facts.of(value).within(NUMERIC)
    }

    fn number(&self, value: GraphValue) -> bool {
        self.facts.of(value).within(Types::NUMBER)
    }
}

impl FunctionCompiler<'_, '_> {
    fn observe(&mut self, value: ClifValue, span: oxc_span::Span, helper: Helper) {
        let Some(feedback) = self.feedback else {
            return;
        };
        if !feedback.get_refinement && matches!(helper, Helper::ObserveGet | Helper::ObserveIndex) {
            return;
        }
        let address = self.function_data(crate::jit::planner::ObservationCache {
            recorded: 0,
            owner: feedback.owner,
            site: crate::jit::planner::Site {
                source: self.source_name.0,
                start: span.start,
                end: span.end,
            },
        });
        let recorded = self
            .b
            .ins()
            .load(types::I64, MemFlagsData::trusted(), address, 0);
        let observe = self.b.create_block();
        let next = self.b.create_block();
        self.b.ins().brif(recorded, next, &[], observe, &[]);
        self.b.switch_to_block(observe);
        self.call_helper(helper, &[value, address]);
        self.b.ins().jump(next, &[]);
        self.b.switch_to_block(next);
    }

    fn epoch(&mut self) -> ClifValue {
        let address = self.b.ins().load(
            self.pointer,
            MemFlagsData::trusted(),
            self.ctx,
            crate::jit::session::EPOCH_OFFSET,
        );
        self.b
            .ins()
            .load(types::I64, MemFlagsData::trusted(), address, 0)
    }

    fn lazy_capture_read(&mut self, local: Variable, cell: Variable) -> ClifValue {
        let raw = self.b.use_var(local);
        let cell = self.b.use_var(cell);
        let boxed = self
            .b
            .ins()
            .icmp_imm_s(IntCC::NotEqual, cell, Value::UNDEFINED.bits() as i64);
        let shared = self.b.create_block();
        let direct = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param(done, types::I64);
        self.brif(boxed, shared, &[], direct, &[]);
        self.switch(shared);
        let value = self.call_helper(Helper::CellGet, &[cell]);
        self.check_error();
        self.jump(done, &[value.into()]);
        self.switch(direct);
        let value = self.checked_temporary(raw, true);
        self.jump(done, &[value.into()]);
        self.switch(done);
        self.b.block_params(done)[0]
    }

    fn lazy_capture_write(&mut self, local: Variable, cell: Variable, value: ClifValue) {
        let cell = self.b.use_var(cell);
        let boxed = self
            .b
            .ins()
            .icmp_imm_s(IntCC::NotEqual, cell, Value::UNDEFINED.bits() as i64);
        let shared = self.b.create_block();
        let direct = self.b.create_block();
        let done = self.b.create_block();
        self.brif(boxed, shared, &[], direct, &[]);
        self.switch(shared);
        self.call_helper(Helper::CellSet, &[cell, value]);
        self.check_error();
        self.jump(done, &[]);
        self.switch(direct);
        self.initialize_local(local, value);
        self.jump(done, &[]);
        self.switch(done);
    }

    fn materialize_capture(&mut self, local: Variable, cell: Variable) -> ClifValue {
        let current = self.b.use_var(cell);
        let boxed =
            self.b
                .ins()
                .icmp_imm_s(IntCC::NotEqual, current, Value::UNDEFINED.bits() as i64);
        let create = self.b.create_block();
        let done = self.b.create_block();
        self.brif(boxed, done, &[], create, &[]);
        self.switch(create);
        let value = self.b.use_var(local);
        let created = self.call_helper(Helper::CellNew, &[value]);
        self.check_error();
        self.initialize_local(cell, created);
        // The cell now owns the binding value, including updates made by escaped callbacks.
        let undefined = self.constant(Value::UNDEFINED);
        self.initialize_local(local, undefined);
        self.jump(done, &[]);
        self.switch(done);
        self.b.use_var(cell)
    }

    pub(super) fn emit_semantic_graph(
        &mut self,
        graph: &Graph,
        closure: ClifValue,
    ) -> Result<(), Error> {
        let source_name = self.source_name;
        let blocks: Vec<_> = graph
            .blocks
            .values()
            .map(|_| self.b.create_block())
            .collect();
        let facts = Facts::compute(graph);
        let suspends = self.generator.is_some();
        let mut reprs: SecondaryMap<GraphValue, Repr> = SecondaryMap::with_default(Repr::Boxed);
        if !suspends {
            for value in graph.values.keys() {
                reprs[value] = Repr::of(facts.of(value));
            }
        }
        let values: Vec<Variable> = graph
            .values
            .keys()
            .map(|value| self.b.declare_var(reprs[value].clif()))
            .collect();
        let bindings: Vec<Variable> = graph
            .bindings
            .values()
            .map(|_| self.b.declare_var(types::I64))
            .collect();
        let lazy_cells: Vec<_> = graph
            .bindings
            .keys()
            .map(|binding| {
                graph
                    .lazy_captures
                    .contains(&binding)
                    .then(|| self.b.declare_var(types::I64))
            })
            .collect();
        let cells: Vec<i64> = (0..graph.guards)
            .map(|_| {
                let mut cell = Box::new(u64::MAX);
                let address = &raw mut *cell as i64;
                self.cells.push(cell);
                address
            })
            .collect();
        let rooted =
            |value: crate::jit::semantic::Value| suspends || facts.of(value).intersects(HEAP);
        let cfg = crate::jit::semantic::Cfg::new(graph);
        let reachable = cfg.rpo.clone();
        let cyclic = cyclic_blocks(graph, &cfg);
        let repeated = |block: crate::jit::semantic::Block| {
            suspends || graph.has_handlers || cyclic.contains(&block)
        };
        let mut pending: SecondaryMap<crate::jit::semantic::Block, usize> = SecondaryMap::new();
        for &block in &reachable {
            pending[block] = cfg.preds[block].len();
        }
        for &block in &reachable {
            let data = &graph.blocks[block];
            for &param in &data.params {
                self.b
                    .append_block_param(blocks[block.index()], reprs[param].clif());
            }
        }
        let undefined = self.constant(Value::UNDEFINED);
        for &variable in bindings.iter().chain(lazy_cells.iter().flatten()) {
            self.b.def_var(variable, undefined);
        }
        if suspends {
            for value in graph.values.keys() {
                let zero = self.zero(reprs[value]);
                self.b.def_var(values[value.index()], zero);
            }
        }
        let cx = Context {
            osr_buffers: RefCell::new(std::collections::HashMap::new()),
            values: &values,
            direct: (!suspends).then(|| RefCell::new(SecondaryMap::new())),
            reprs: &reprs,
            facts: &facts,
            slots: if suspends || self.frame_size.is_none() {
                Slots::none()
            } else {
                slots::assign(
                    graph,
                    &cfg,
                    |value| {
                        facts.of(value).intersects(HEAP)
                            && !graph.borrowed.contains(&value)
                            && !matches!(graph.defining_op(value), Some(Op::This | Op::Argument(_)))
                    },
                    |value| match graph.defining_op(value) {
                        Some(Op::Get(_, name)) => name != "length",
                        Some(Op::Read(binding)) => {
                            !graph.bindings[*binding].captured
                                && !graph.lazy_captures.contains(binding)
                        }
                        Some(
                            Op::GetIndex(..)
                            | Op::TypedIndex(..)
                            | Op::IndexedGet(..)
                            | Op::SpreadGet(..)
                            | Op::Call {
                                target: Some(_), ..
                            }
                            | Op::GlobalRead(_)
                            | Op::GlobalReadOptional(_)
                            | Op::NewObject
                            | Op::NewArray(_)
                            | Op::NewHoleyArray(_)
                            | Op::NewInstance(_)
                            | Op::StringSplit(..),
                        ) => true,
                        _ => false,
                    },
                )
            },
        };
        for (value, slot) in cx.slots.assigned() {
            self.slots.insert(values[value.index()].as_u32(), slot);
        }
        self.slot_count = cx.slots.count();
        self.jump(blocks[graph.entry.index()], &[]);
        if pending[graph.entry] == 0 {
            self.b.seal_block(blocks[graph.entry.index()]);
        }
        for &block in &reachable {
            let data = &graph.blocks[block];
            if self.annotations.semantic_locations {
                self.b
                    .set_srcloc(cranelift_codegen::ir::SourceLoc::default());
            }
            self.switch(blocks[block.index()]);
            let cold = graph.cold.contains(&block);
            let created = self.b.func.dfg.num_blocks();
            let params = self.b.block_params(blocks[block.index()]).to_vec();
            let undefined = self.constant(Value::UNDEFINED);
            let slotted = self.frame_size.is_some();
            let mut retained_any = false;
            for (&value, &param) in data.params.iter().zip(&params) {
                if !rooted(value) {
                    continue;
                }
                if slotted {
                    if !cx.slots.forwarded(value) {
                        self.increment(param);
                    }
                    continue;
                }
                let uninitialized = self.b.ins().icmp_imm_s(
                    cranelift_codegen::ir::condcodes::IntCC::Equal,
                    param,
                    Value::UNINITIALIZED.bits() as i64,
                );
                let retained = self.b.ins().select(uninitialized, undefined, param);
                let heap = self.is_heap(retained);
                let retain = self.b.create_block();
                let done = self.b.create_block();
                self.brif(heap, retain, &[], done, &[]);
                self.switch(retain);
                self.retain_temporary(retained);
                self.jump(done, &[]);
                self.switch(done);
                retained_any = true;
            }
            for (&value, param) in data.params.iter().zip(params) {
                if rooted(value) && slotted {
                    if cx.slots.forwarded(value) {
                        self.b.def_var(values[value.index()], param);
                    } else {
                        let repeated = repeated(block) || cx.slots.shared(value);
                        self.replace_local(values[value.index()], param, repeated);
                    }
                } else if rooted(value) {
                    self.initialize_local(values[value.index()], param);
                } else if suspends {
                    self.b.def_var(values[value.index()], param);
                }
                cx.define(value, param);
            }
            if retained_any {
                self.clear_temporaries();
            }
            let calls = self.annotations.helpers.len();
            for &inst in &data.insts {
                let handler = match &data.terminator.as_ref().unwrap().kind {
                    TerminatorKind::CheckException(_, error) => Some(blocks[error.block.index()]),
                    _ => None,
                };
                if let Some(handler) = handler {
                    self.handlers.push(handler);
                }
                let data = &graph.insts[inst];
                self.source_span = data.span;
                self.source_name = data.source_name.unwrap_or(source_name);
                let location = if self.annotations.semantic_locations {
                    cranelift_codegen::ir::SourceLoc::new(graph.inst_location(inst))
                } else if self.source_name == source_name {
                    cranelift_codegen::ir::SourceLoc::new(data.span.start)
                } else {
                    cranelift_codegen::ir::SourceLoc::default()
                };
                self.b.set_srcloc(location);
                let get = |this: &mut Self, value: GraphValue| {
                    let operand = this.operand(&cx, value);
                    this.convert(operand, Repr::Boxed)
                };
                if let Op::GetIndex(object, _)
                | Op::TypedIndex(object, _, TypedRead { element: None, .. }) = data.op
                {
                    let object = get(self, object);
                    self.observe(object, data.span, Helper::ObserveIndex);
                }
                let typed = self.typed_op(&cx, graph, inst)?;
                let mut owned = false;
                let result = if let Some(typed) = typed {
                    typed
                } else {
                    let result = match &data.op {
                        Op::OsrStart(plan, candidate, inputs) => {
                            let transfer = plan.loops[*candidate].transfer.as_ref().unwrap();
                            let words = inputs.len() + transfer.outputs.len();
                            let slot = self.b.create_sized_stack_slot(
                                cranelift_codegen::ir::StackSlotData::new(
                                    cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                                    (words * 8) as u32,
                                    3,
                                ),
                            );
                            let address = self.b.ins().stack_addr(self.pointer, slot, 0);
                            cx.osr_buffers
                                .borrow_mut()
                                .insert(data.result.unwrap(), (address, inputs.len()));
                            if let Some(feedback) = self.feedback {
                                let site = self.data.pin(crate::jit::osr::Site::new(
                                    feedback.owner,
                                    plan.clone(),
                                    *candidate,
                                ));
                                let site = self.b.ins().iconst(self.pointer, site as i64);
                                let count = self.b.ins().load(
                                    types::I64,
                                    MemFlagsData::trusted(),
                                    site,
                                    crate::jit::osr::ENTRIES_OFFSET,
                                );
                                let count = self.b.ins().iadd_imm_s(count, 1);
                                self.b.ins().store(
                                    MemFlagsData::trusted(),
                                    count,
                                    site,
                                    crate::jit::osr::ENTRIES_OFFSET,
                                );
                                let tick = self.b.ins().band_imm_s(count, 31);
                                let tick = self.b.ins().icmp_imm_s(IntCC::Equal, tick, 0);
                                let entry = self.b.ins().load(
                                    self.pointer,
                                    MemFlagsData::trusted(),
                                    site,
                                    crate::jit::osr::ENTRY_OFFSET,
                                );
                                let ready = self.b.ins().icmp_imm_s(IntCC::NotEqual, entry, 0);
                                let poll = self.b.ins().bor(tick, ready);
                                let slow = self.b.create_block();
                                let done = self.b.create_block();
                                self.b.append_block_param(done, types::I64);
                                let no = self.constant(Value::FALSE);
                                self.brif(poll, slow, &[], done, &[no.into()]);
                                self.switch(slow);
                                for (index, &value) in inputs.iter().enumerate() {
                                    let value = get(self, value);
                                    self.b.ins().store(
                                        MemFlagsData::trusted(),
                                        value,
                                        address,
                                        (index * 8) as i32,
                                    );
                                }
                                let result = self.call_helper(Helper::OsrPoll, &[site, address]);
                                self.check_error();
                                self.jump(done, &[result.into()]);
                                self.switch(done);
                                Some(self.b.block_params(done)[0])
                            } else {
                                Some(self.constant(Value::FALSE))
                            }
                        }
                        Op::OsrOutput(start, index) => {
                            let (address, inputs) = cx.osr_buffers.borrow()[start];
                            Some(self.b.ins().load(
                                types::I64,
                                MemFlagsData::trusted(),
                                address,
                                ((inputs + index) * 8) as i32,
                            ))
                        }
                        Op::OsrStore(offset, values) => {
                            let super::Arguments::Vector { argv, .. } = self.arguments else {
                                return Err(Error::Compile(
                                    "OSR fragment requires vector arguments".into(),
                                ));
                            };
                            for (index, &value) in values.iter().enumerate() {
                                let value = get(self, value);
                                self.b.ins().store(
                                    MemFlagsData::trusted(),
                                    value,
                                    argv,
                                    ((offset + index) * 8) as i32,
                                );
                            }
                            None
                        }
                        Op::Argument(index) => Some(self.load_arg(*index)),
                        Op::Arguments => {
                            let (argv, argc) = self.argument_vector()?;
                            let value = self.call_helper(Helper::ArrayNew, &[argv, argc]);
                            self.check_error();
                            Some(value)
                        }
                        Op::Rest(index) => {
                            let index = self.b.ins().iconst(types::I64, *index as i64);
                            let (argv, argc) = self.argument_vector()?;
                            let value = self.call_helper(Helper::Rest, &[argv, argc, index]);
                            self.check_error();
                            Some(value)
                        }
                        Op::This => Some(self.receiver),
                        Op::ClosureThis(value) => {
                            let value = get(self, *value);
                            Some(self.call_helper(Helper::ClosureThis, &[value]))
                        }
                        Op::IsArrayIteration(callee, receiver, method) => {
                            let callee = get(self, *callee);
                            let receiver = get(self, *receiver);
                            let method = self.b.ins().iconst(types::I64, *method as i64);
                            Some(self.call_helper(
                                Helper::ArrayIterationGuard,
                                &[callee, receiver, method],
                            ))
                        }
                        Op::NewHoleyArray(length) => {
                            let length = get(self, *length);
                            owned = data
                                .result
                                .is_some_and(|value| self.slot_owned(graph, &cx, value));
                            let helper = if owned {
                                Helper::ArrayMapNewOwned
                            } else {
                                Helper::ArrayMapNew
                            };
                            let value = self.call_helper(helper, &[length]);
                            self.check_error();
                            Some(value)
                        }
                        Op::MapElement(receiver, index) => {
                            let receiver = get(self, *receiver);
                            let index = get(self, *index);
                            Some(self.call_helper(Helper::ArrayMapElement, &[receiver, index]))
                        }
                        Op::Throw(value) => {
                            let value = get(self, *value);
                            self.call_helper(Helper::Throw, &[value]);
                            self.check_error();
                            None
                        }
                        Op::Catch => {
                            let dummy = self.constant(Value::UNDEFINED);
                            Some(self.call_helper(Helper::Catch, &[dummy]))
                        }
                        Op::Closure(
                            span,
                            id,
                            arrow,
                            generator,
                            lexical_this,
                            captures,
                            lexical_receiver,
                        ) => {
                            let cells: Vec<_> = captures
                                .iter()
                                .map(|binding| {
                                    let local = bindings[binding.index()];
                                    match lazy_cells[binding.index()] {
                                        Some(cell) => self.materialize_capture(local, cell),
                                        None => self.b.use_var(local),
                                    }
                                })
                                .collect();
                            let argv = self.stack_values(&cells, *span)?;
                            let count = self.b.ins().iconst(types::I64, cells.len() as i64);
                            let code = self.b.ins().iconst(types::I64, id.as_u32() as i64);
                            let result = self.call_helper(Helper::ClosureNew, &[code, argv, count]);
                            self.check_error();
                            if *arrow {
                                let receiver = if *lexical_this {
                                    lexical_receiver
                                        .map(|value| get(self, value))
                                        .unwrap_or(self.receiver)
                                } else {
                                    self.constant(Value::UNDEFINED)
                                };
                                self.call_helper(Helper::BindThis, &[result, receiver]);
                                self.check_error();
                            } else if *generator {
                                self.call_helper(Helper::DisableConstructor, &[result]);
                                self.check_error();
                                self.check_error();
                            }
                            Some(result)
                        }
                        Op::Constant(c) => Some(match c {
                            Constant::Undefined => self.constant(Value::UNDEFINED),
                            Constant::Null => self.constant(Value::NULL),
                            Constant::Boolean(value) => self.constant(Value::boolean(*value)),
                            Constant::Number(value) => self.constant(Value::number(*value)),
                            Constant::Uninitialized => self.constant(Value::UNINITIALIZED),
                            Constant::String(value) => {
                                let text: Box<str> = value.as_str().into();
                                let ptr = self.b.ins().iconst(self.pointer, text.as_ptr() as i64);
                                let len = self.b.ins().iconst(types::I64, text.len() as i64);
                                self.names.push(text);
                                let result = self.call_helper(Helper::StringNew, &[ptr, len]);
                                self.check_error();
                                result
                            }
                        }),
                        Op::GlobalRead(name) => {
                            let retain = data
                                .result
                                .is_none_or(|value| !self.slot_owned(graph, &cx, value));
                            Some(self.cached_global_get(name, false, retain))
                        }
                        Op::GlobalReadOptional(name) => {
                            let retain = data
                                .result
                                .is_none_or(|value| !self.slot_owned(graph, &cx, value));
                            Some(self.global_get_optional(name, retain))
                        }
                        Op::PropertyKey(value) => {
                            let value = get(self, *value);
                            let result = self.call_helper(Helper::PropertyKey, &[value]);
                            self.check_error();
                            Some(result)
                        }
                        Op::RegExp(pattern, flags) => {
                            let pattern: Box<str> = pattern.as_str().into();
                            let ptr = self.b.ins().iconst(self.pointer, pattern.as_ptr() as i64);
                            let len = self.b.ins().iconst(types::I64, pattern.len() as i64);
                            let flags = self.b.ins().iconst(types::I64, *flags as i64);
                            self.names.push(pattern);
                            let result = self.call_helper(Helper::RegExpNew, &[flags, ptr, len]);
                            self.check_error();
                            Some(result)
                        }
                        Op::GlobalInitialize(name, value) => {
                            let value = get(self, *value);
                            self.global_initialize(name, value);
                            None
                        }
                        Op::GlobalWrite(name, value) => {
                            let value = get(self, *value);
                            self.global_access(name, Some(value));
                            None
                        }
                        Op::Read(binding) if lazy_cells[binding.index()].is_some() => {
                            Some(self.lazy_capture_read(
                                bindings[binding.index()],
                                lazy_cells[binding.index()].unwrap(),
                            ))
                        }
                        Op::Write(binding, value) | Op::Initialize(binding, value)
                            if lazy_cells[binding.index()].is_some() =>
                        {
                            let local = bindings[binding.index()];
                            let cell = lazy_cells[binding.index()].unwrap();
                            if matches!(data.op, Op::Write(..)) {
                                self.lazy_capture_read(local, cell);
                            }
                            let value = get(self, *value);
                            self.lazy_capture_write(local, cell, value);
                            None
                        }
                        Op::Read(binding) => {
                            let value = self.b.use_var(bindings[binding.index()]);
                            let value = if graph.bindings[*binding].captured {
                                let value = self.call_helper(Helper::CellGet, &[value]);
                                self.check_error();
                                value
                            } else {
                                let retain = data
                                    .result
                                    .is_none_or(|value| !self.slot_owned(graph, &cx, value));
                                self.checked_temporary(value, retain)
                            };
                            Some(value)
                        }
                        Op::Write(binding, value) => {
                            let old = self.b.use_var(bindings[binding.index()]);
                            let helper = if graph.bindings[*binding].captured {
                                Helper::CellGet
                            } else {
                                Helper::TempRetain
                            };
                            self.call_helper(helper, &[old]);
                            self.check_error();
                            let value = get(self, *value);
                            if graph.bindings[*binding].captured {
                                self.call_helper(Helper::CellSet, &[old, value]);
                                self.check_error();
                            } else {
                                self.initialize_local(bindings[binding.index()], value);
                            }
                            None
                        }
                        Op::Initialize(binding, value) => {
                            let value = get(self, *value);
                            if graph.bindings[*binding].captured {
                                let cell = self.b.use_var(bindings[binding.index()]);
                                self.call_helper(Helper::CellSet, &[cell, value]);
                                self.check_error();
                            } else {
                                self.initialize_local(bindings[binding.index()], value);
                            }
                            None
                        }
                        Op::CheckInitialized(value) => {
                            let value = get(self, *value);
                            let value = self.checked_temporary(value, true);
                            Some(value)
                        }
                        Op::IteratorNew(value)
                        | Op::IteratorNext(value)
                        | Op::EnumerateNew(value)
                        | Op::EnumerateNext(value) => {
                            let value = get(self, *value);
                            let helper = match data.op {
                                Op::IteratorNew(_) => Helper::IterateNew,
                                Op::IteratorNext(_) => Helper::IterateNext,
                                Op::EnumerateNew(_) => Helper::EnumerateNew,
                                Op::EnumerateNext(_) => Helper::EnumerateNext,
                                _ => unreachable!(),
                            };
                            let result = self.call_helper(helper, &[value]);
                            self.check_error();
                            self.check_error();
                            Some(result)
                        }
                        Op::IsUninitialized(value) => {
                            let value = get(self, *value);
                            let condition = self.b.ins().icmp_imm_s(
                                cranelift_codegen::ir::condcodes::IntCC::Equal,
                                value,
                                Value::UNINITIALIZED.bits() as i64,
                            );
                            let yes = self.constant(Value::TRUE);
                            let no = self.constant(Value::FALSE);
                            Some(self.b.ins().select(condition, yes, no))
                        }
                        Op::GuardedCompare {
                            operator,
                            left,
                            right,
                            numbers,
                        } => {
                            let left = get(self, *left);
                            let right = get(self, *right);
                            let first = self.operand(&cx, numbers[0]);
                            let first = self.convert(first, Repr::Bool);
                            let second = self.operand(&cx, numbers[1]);
                            let second = self.convert(second, Repr::Bool);
                            let value = self.numeric_comparison(
                                ops::binary_operator(*operator),
                                left,
                                right,
                                Some((first, second)),
                            );
                            self.check_error();
                            Some(value)
                        }
                        Op::NewObject => {
                            let dummy = self.constant(Value::UNDEFINED);
                            owned = data
                                .result
                                .is_some_and(|value| self.slot_owned(graph, &cx, value));
                            let helper = if owned {
                                Helper::ObjectNewOwned
                            } else {
                                Helper::ObjectNew
                            };
                            let value = self.call_helper(helper, &[dummy]);
                            self.check_error();
                            Some(value)
                        }
                        Op::NewArray(args) => {
                            let args: Vec<_> = args.iter().map(|&v| get(self, v)).collect();
                            let ptr = self.stack_values(&args, data.span)?;
                            let len = self.b.ins().iconst(types::I64, args.len() as i64);
                            owned = data
                                .result
                                .is_some_and(|value| self.slot_owned(graph, &cx, value));
                            let helper = if owned {
                                Helper::ArrayNewOwned
                            } else {
                                Helper::ArrayNew
                            };
                            let value = self.call_helper(helper, &[ptr, len]);
                            self.check_error();
                            Some(value)
                        }
                        Op::DefineHidden(object, name, value) => {
                            let object = get(self, *object);
                            let value = get(self, *value);
                            self.define_hidden_property(object, name, value);
                            None
                        }
                        Op::MarkClass(value) | Op::DisableConstructor(value) => {
                            let value = get(self, *value);
                            let helper = if matches!(data.op, Op::MarkClass(_)) {
                                Helper::MarkClass
                            } else {
                                Helper::DisableConstructor
                            };
                            self.call_helper(helper, &[value]);
                            self.check_error();
                            None
                        }
                        Op::Define(object, name, value) | Op::Set(object, name, value) => {
                            let object = get(self, *object);
                            let value = get(self, *value);
                            self.property_access(object, name, Some(value));
                            None
                        }
                        Op::Get(object, name) if name == "length" => {
                            let object = get(self, *object);
                            Some(self.length_access(object))
                        }
                        Op::Get(object, name) => {
                            let object = get(self, *object);
                            let retain = data.result.is_none_or(|value| {
                                !graph.borrowed.contains(&value)
                                    && !self.slot_owned(graph, &cx, value)
                            });
                            Some(self.cached_property_get(object, name, retain))
                        }
                        Op::GetIndex(object, index)
                        | Op::TypedIndex(object, index, _)
                        | Op::IndexedGet(object, index)
                        | Op::SpreadGet(object, index) => {
                            let object = get(self, *object);
                            let index = get(self, *index);
                            let retain = data.result.is_none_or(|value| {
                                !graph.borrowed.contains(&value)
                                    && !self.slot_owned(graph, &cx, value)
                            });
                            Some(self.element_read(
                                object,
                                Operand::new(index, Repr::Boxed),
                                retain,
                            ))
                        }
                        Op::SetIndex(object, index, value) => {
                            let object = get(self, *object);
                            let index = get(self, *index);
                            let value = get(self, *value);
                            self.index_access(
                                object,
                                Operand::new(index, Repr::Boxed),
                                Some(value),
                            );
                            None
                        }
                        Op::View(object) => {
                            let object = get(self, *object);
                            Some(self.view(object))
                        }
                        Op::IsNumber(value) => {
                            let value = get(self, *value);
                            let condition = self.is_number(value);
                            let yes = self.constant(Value::TRUE);
                            let no = self.constant(Value::FALSE);
                            Some(self.b.ins().select(condition, yes, no))
                        }
                        Op::IsTypeof(value, kind) => {
                            let value = get(self, *value);
                            let kind = self.constant(Value::number(*kind as f64));
                            Some(self.call_helper(Helper::IsTypeof, &[value, kind]))
                        }
                        Op::IsString(value) => {
                            let value = get(self, *value);
                            let tag = self.b.ins().ushr_imm_s(value, TAG_SHIFT as i64);
                            let condition = self.b.ins().icmp_imm_s(
                                IntCC::Equal,
                                tag,
                                (TAG_STRING >> TAG_SHIFT) as i64,
                            );
                            let yes = self.constant(Value::TRUE);
                            let no = self.constant(Value::FALSE);
                            Some(self.b.ins().select(condition, yes, no))
                        }
                        Op::IsTyped(..) | Op::IsCode(..) => unreachable!(),
                        Op::IsIndexed(value) => {
                            let value = get(self, *value);
                            let view = self.view(value);
                            let condition = self.has_kind(view, kind::ARRAY | kind::TYPED);
                            let yes = self.constant(Value::TRUE);
                            let no = self.constant(Value::FALSE);
                            Some(self.b.ins().select(condition, yes, no))
                        }
                        Op::IndexedLength(value) => {
                            let value = get(self, *value);
                            let view = self.view(value);
                            let length = self.view_length(view);
                            Some(
                                self.b
                                    .ins()
                                    .bitcast(types::I64, MemFlagsData::new(), length),
                            )
                        }
                        Op::AsNumber(value) => Some(get(self, *value)),
                        Op::ToInt32(_) => unreachable!(),
                        Op::ToNumeric(value) => {
                            let value = get(self, *value);
                            Some(ops::to_number(self, value))
                        }
                        Op::Unary(op, value) => {
                            let value = get(self, *value);
                            Some(match op {
                                UnaryOp::Increment | UnaryOp::Decrement => {
                                    let one = self.constant(Value::number(1.0));
                                    self.binary(
                                        if matches!(op, UnaryOp::Increment) {
                                            BinaryOp::Add
                                        } else {
                                            BinaryOp::Sub
                                        },
                                        value,
                                        one,
                                    )
                                }
                                _ => self.unary_helper(
                                    match op {
                                        UnaryOp::Negate => EmitUnaryOp::Negation,
                                        UnaryOp::Plus => EmitUnaryOp::Plus,
                                        UnaryOp::Not => EmitUnaryOp::Not,
                                        UnaryOp::BitwiseNot => EmitUnaryOp::BitwiseNot,
                                        UnaryOp::Typeof => EmitUnaryOp::Typeof,
                                        _ => unreachable!(),
                                    },
                                    value,
                                ),
                            })
                        }
                        Op::Binary(op, left, right) => {
                            let left = get(self, *left);
                            let right = get(self, *right);
                            Some(self.binary(ops::binary_operator(*op), left, right))
                        }
                        Op::ArrayAppend(array, value, spread) => {
                            let array = get(self, *array);
                            let value = get(self, *value);
                            self.call_helper(
                                if *spread {
                                    Helper::ArrayExtend
                                } else {
                                    Helper::ArrayAppend
                                },
                                &[array, value],
                            );
                            self.check_error();
                            None
                        }
                        Op::CallSpread(callee, receiver, args) => {
                            let callee = get(self, *callee);
                            let receiver = get(self, *receiver);
                            let args = get(self, *args);
                            let value =
                                self.call_helper(Helper::CallSpread, &[callee, receiver, args]);
                            self.check_error();
                            Some(value)
                        }
                        Op::ConstructSpread(callee, args) => {
                            let callee = get(self, *callee);
                            let args = get(self, *args);
                            let value = self.call_helper(Helper::ConstructSpread, &[callee, args]);
                            self.check_error();
                            Some(value)
                        }
                        Op::Call {
                            callee,
                            receiver,
                            args,
                            target,
                        } => {
                            let closure = graph.defining_op(*callee);
                            let callee = get(self, *callee);
                            let receiver = receiver
                                .map(|v| get(self, v))
                                .unwrap_or_else(|| self.constant(Value::UNDEFINED));
                            let args: Vec<_> = args.iter().map(|&v| get(self, v)).collect();
                            self.observe(callee, data.span, Helper::ObserveCall);
                            let value = match target {
                                Some(target) => {
                                    let receiver = match closure {
                                        _ if !target.lexical_this => receiver,
                                        Some(Op::Closure(_, _, true, _, true, _, lexical)) => {
                                            lexical.map_or(self.receiver, |v| get(self, v))
                                        }
                                        _ => self.call_helper(
                                            Helper::LexicalReceiver,
                                            &[callee, receiver],
                                        ),
                                    };
                                    owned = data
                                        .result
                                        .is_some_and(|value| self.slot_owned(graph, &cx, value));
                                    self.direct_call(
                                        *target, callee, receiver, &args, data.span, owned,
                                    )?
                                }
                                None => self.generic_call(callee, receiver, &args, data.span)?,
                            };
                            Some(value)
                        }
                        Op::NewInstance(prototype) => {
                            let prototype = get(self, *prototype);
                            owned = data
                                .result
                                .is_some_and(|value| self.slot_owned(graph, &cx, value));
                            let helper = if owned {
                                Helper::NewInstanceOwned
                            } else {
                                Helper::NewInstance
                            };
                            let result = self.call_helper(helper, &[prototype]);
                            self.check_error();
                            Some(result)
                        }
                        Op::ConstructorResult(result, instance) => {
                            let result = get(self, *result);
                            let instance = get(self, *instance);
                            let tag = self
                                .b
                                .ins()
                                .band_imm_s(result, 0xffff_0000_0000_0000u64 as i64);
                            let object = self.b.ins().icmp_imm_s(
                                IntCC::Equal,
                                tag,
                                crate::value::TAG_OBJECT as i64,
                            );
                            Some(self.b.ins().select(object, result, instance))
                        }
                        Op::ConstructorCall {
                            callee,
                            receiver,
                            args,
                        } => {
                            let callee = get(self, *callee);
                            let receiver = get(self, *receiver);
                            let args: Vec<_> = args.iter().map(|&v| get(self, v)).collect();
                            self.observe(callee, data.span, Helper::ObserveCall);
                            let argv = self.stack_values(&args, data.span)?;
                            let argc = self.b.ins().iconst(types::I64, args.len() as i64);
                            let result = self.call_helper(
                                Helper::ConstructorCall,
                                &[callee, receiver, argv, argc],
                            );
                            self.check_error();
                            Some(result)
                        }
                        Op::ConstructorCallSpread(callee, receiver, args) => {
                            let callee = get(self, *callee);
                            let receiver = get(self, *receiver);
                            let args = get(self, *args);
                            let result = self.call_helper(
                                Helper::ConstructorCallSpread,
                                &[callee, receiver, args],
                            );
                            self.check_error();
                            Some(result)
                        }
                        Op::ConstructorPrototype(callee) => {
                            let callee = get(self, *callee);
                            Some(self.call_helper(Helper::ConstructorPrototype, &[callee]))
                        }
                        Op::ConstructResolved {
                            callee,
                            prototype,
                            args,
                        } => {
                            let callee = get(self, *callee);
                            let prototype = get(self, *prototype);
                            let args: Vec<_> = args.iter().map(|&v| get(self, v)).collect();
                            let argv = self.stack_values(&args, data.span)?;
                            let argc = self.b.ins().iconst(types::I64, args.len() as i64);
                            let value = self.call_helper(
                                Helper::ConstructResolved,
                                &[callee, prototype, argv, argc],
                            );
                            self.check_error();
                            Some(value)
                        }
                        Op::ConstructSpreadResolved(callee, prototype, args) => {
                            let callee = get(self, *callee);
                            let prototype = get(self, *prototype);
                            let args = get(self, *args);
                            let value = self.call_helper(
                                Helper::ConstructSpreadResolved,
                                &[callee, prototype, args],
                            );
                            self.check_error();
                            Some(value)
                        }
                        Op::Construct { callee, args } => {
                            let callee = get(self, *callee);
                            let args: Vec<_> = args.iter().map(|&v| get(self, v)).collect();
                            let argv = self.stack_values(&args, data.span)?;
                            let argc = self.b.ins().iconst(types::I64, args.len() as i64);
                            let value = self.call_helper(Helper::Construct, &[callee, argv, argc]);
                            self.check_error();
                            Some(value)
                        }
                        Op::ProbeGlobal(name) => {
                            let name: Box<str> = name.as_str().into();
                            let ptr = self.b.ins().iconst(self.pointer, name.as_ptr() as i64);
                            let len = self.b.ins().iconst(types::I64, name.len() as i64);
                            self.names.push(name);
                            Some(self.call_helper(Helper::SemanticGlobalProbe, &[ptr, len]))
                        }
                        Op::ProbeProperty(object, name) => {
                            let object = get(self, *object);
                            Some(self.cached_property_probe(object, name))
                        }
                        Op::ProbeIndex(object, key) => {
                            let object = get(self, *object);
                            let key = get(self, *key);
                            Some(self.call_helper(Helper::SemanticIndexProbe, &[object, key]))
                        }
                        Op::IsStringSplit(value) => {
                            let value = get(self, *value);
                            Some(self.call_helper(Helper::IsStringSplit, &[value]))
                        }
                        Op::StringSplit(receiver, args) => {
                            let receiver = get(self, *receiver);
                            let args: Vec<_> = args.iter().map(|&v| get(self, v)).collect();
                            let argv = self.stack_values(&args, data.span)?;
                            let operation = self.b.ins().iconst(types::I64, 0);
                            let argc = self.b.ins().iconst(types::I64, args.len() as i64);
                            owned = data
                                .result
                                .is_some_and(|value| self.slot_owned(graph, &cx, value));
                            let helper = if owned {
                                Helper::StringSplitOwned
                            } else {
                                Helper::StringSplit
                            };
                            let result =
                                self.call_helper(helper, &[operation, receiver, argv, argc]);
                            self.check_error();
                            Some(result)
                        }
                        Op::IsNative(value, native) => {
                            let value = get(self, *value);
                            let operation = self.b.ins().iconst(types::I64, native.index() as i64);
                            Some(self.call_helper(Helper::SemanticMathNative, &[value, operation]))
                        }
                        Op::GuardValid(guard) => {
                            let epoch = self.epoch();
                            let cell = self.b.ins().iconst(self.pointer, cells[*guard]);
                            let validated =
                                self.b
                                    .ins()
                                    .load(types::I64, MemFlagsData::trusted(), cell, 0);
                            let condition = self.b.ins().icmp(IntCC::Equal, epoch, validated);
                            let yes = self.constant(Value::TRUE);
                            let no = self.constant(Value::FALSE);
                            Some(self.b.ins().select(condition, yes, no))
                        }
                        Op::GuardCommit(guard) => {
                            let epoch = self.epoch();
                            let cell = self.b.ins().iconst(self.pointer, cells[*guard]);
                            self.b.ins().store(MemFlagsData::trusted(), epoch, cell, 0);
                            None
                        }
                        Op::CallNative(native, args) => {
                            let args: Vec<_> = args.iter().map(|&v| get(self, v)).collect();
                            let argv = self.stack_values(&args, data.span)?;
                            let function = crate::jit::math::METHODS[native.index()].1;
                            Some(self.native_math(function, &args, |this| {
                                let argc = this.b.ins().iconst(types::I64, args.len() as i64);
                                let operation =
                                    this.b.ins().iconst(types::I64, native.index() as i64);
                                let receiver = this.constant(Value::UNDEFINED);
                                let value = this.call_helper(
                                    Helper::MathDirect,
                                    &[operation, receiver, argv, argc],
                                );
                                this.check_error();
                                value
                            }))
                        }
                        Op::IsUndefined(value) => {
                            let value = get(self, *value);
                            let condition = self.b.ins().icmp_imm_s(
                                cranelift_codegen::ir::condcodes::IntCC::Equal,
                                value,
                                Value::UNDEFINED.bits() as i64,
                            );
                            let yes = self.constant(Value::TRUE);
                            let no = self.constant(Value::FALSE);
                            Some(self.b.ins().select(condition, yes, no))
                        }
                        Op::IsNullish(value) => {
                            let value = get(self, *value);
                            let condition = self.is_nullish(value);
                            let yes = self.constant(Value::TRUE);
                            let no = self.constant(Value::FALSE);
                            Some(self.b.ins().select(condition, yes, no))
                        }
                        Op::EnterScope(scope) | Op::EnterCaptures(scope, _) => {
                            let closure = match data.op {
                                Op::EnterCaptures(_, value) => get(self, value),
                                _ => closure,
                            };
                            for (index, &binding) in
                                graph.scopes[*scope].bindings.iter().enumerate()
                            {
                                let value = if graph.scopes[*scope].kind == ScopeKind::Captures {
                                    let index = self.b.ins().iconst(types::I64, index as i64);
                                    let cell =
                                        self.call_helper(Helper::CaptureGet, &[closure, index]);
                                    self.check_error();
                                    cell
                                } else if graph.bindings[binding].captured
                                    && lazy_cells[binding.index()].is_none()
                                {
                                    let initial = self.constant(Value::UNINITIALIZED);
                                    let cell = self.call_helper(Helper::CellNew, &[initial]);
                                    self.check_error();
                                    cell
                                } else {
                                    self.constant(Value::UNINITIALIZED)
                                };
                                if let Some(cell) = lazy_cells[binding.index()] {
                                    let undefined = self.constant(Value::UNDEFINED);
                                    self.initialize_local(cell, undefined);
                                }
                                self.initialize_local(bindings[binding.index()], value);
                            }
                            None
                        }
                        Op::LeaveScope(scope) => {
                            for &binding in &graph.scopes[*scope].bindings {
                                let value = self.constant(Value::UNDEFINED);
                                self.initialize_local(bindings[binding.index()], value);
                                if let Some(cell) = lazy_cells[binding.index()] {
                                    self.initialize_local(cell, value);
                                }
                            }
                            None
                        }
                        Op::NextIteration(scope) | Op::FreshIteration(scope) => {
                            for &binding in &graph.scopes[*scope].bindings {
                                if !graph.bindings[binding].captured {
                                    continue;
                                }
                                let initial = if matches!(data.op, Op::FreshIteration(_)) {
                                    self.constant(Value::UNINITIALIZED)
                                } else {
                                    if let Some(cell) = lazy_cells[binding.index()] {
                                        self.lazy_capture_read(bindings[binding.index()], cell)
                                    } else {
                                        let cell = self.b.use_var(bindings[binding.index()]);
                                        let value = self.call_helper(Helper::CellGet, &[cell]);
                                        self.check_error();
                                        value
                                    }
                                };
                                if let Some(cell) = lazy_cells[binding.index()] {
                                    let undefined = self.constant(Value::UNDEFINED);
                                    self.initialize_local(cell, undefined);
                                    self.initialize_local(bindings[binding.index()], initial);
                                    continue;
                                }
                                let cell = self.call_helper(Helper::CellNew, &[initial]);
                                self.check_error();
                                self.initialize_local(bindings[binding.index()], cell);
                            }
                            None
                        }
                    };
                    result.map(|value| Operand::new(value, Repr::Boxed))
                };
                if let (Some(value), Some(result)) = (data.result, result) {
                    if matches!(data.op, Op::Get(..) | Op::GetIndex(..) | Op::IndexedGet(..))
                        && facts.of(value).intersects(Types::OBJECT)
                    {
                        let boxed = self.convert(result, Repr::Boxed);
                        self.observe(boxed, data.span, Helper::ObserveGet);
                    }
                    let result = self.convert(result, reprs[value]);
                    let borrowed = (matches!(data.op, Op::This | Op::Argument(_))
                        || graph.borrowed.contains(&value))
                        && !suspends;
                    if owned {
                        let repeated = repeated(block) || cx.slots.shared(value);
                        self.replace_local(values[value.index()], result, repeated);
                    } else if !borrowed && self.slot_owned(graph, &cx, value) {
                        let repeated = repeated(block) || cx.slots.shared(value);
                        self.own_local(values[value.index()], result, repeated);
                    } else if rooted(value) && !borrowed && !slotted {
                        self.initialize_local(values[value.index()], result);
                    } else if suspends {
                        self.b.def_var(values[value.index()], result);
                    }
                    cx.define(value, result);
                }
                if handler.is_some() {
                    self.handlers.pop();
                }
            }
            if self.annotations.helpers.len() != calls {
                self.clear_temporaries();
            }
            let terminator = data.terminator.as_ref().unwrap();
            self.source_span = terminator.span;
            self.source_name = terminator.source_name.unwrap_or(source_name);
            let location = if self.annotations.semantic_locations {
                cranelift_codegen::ir::SourceLoc::new(graph.terminator_location(block))
            } else if self.source_name == source_name {
                cranelift_codegen::ir::SourceLoc::new(terminator.span.start)
            } else {
                cranelift_codegen::ir::SourceLoc::default()
            };
            self.b.set_srcloc(location);
            let target = |this: &mut Self, target: &crate::jit::semantic::Target| {
                let args: Vec<_> = target
                    .args
                    .iter()
                    .zip(&graph.blocks[target.block].params)
                    .map(|(&value, &param)| {
                        let operand = this.operand(&cx, value);
                        BlockArg::Value(this.convert(operand, reprs[param]))
                    })
                    .collect();
                (blocks[target.block.index()], args)
            };
            match &terminator.kind {
                TerminatorKind::CheckException(to, _) | TerminatorKind::Jump(to) => {
                    let (block, args) = target(self, to);
                    self.jump(block, &args);
                }
                TerminatorKind::Branch(value, yes, no) => {
                    let operand = self.operand(&cx, *value);
                    let condition = self.condition(operand, cx.numeric(*value));
                    let (yes, yes_args) = target(self, yes);
                    let (no, no_args) = target(self, no);
                    self.brif(condition, yes, &yes_args, no, &no_args);
                }
                TerminatorKind::Yield(span, value, target_) => {
                    let generator = self.generator.unwrap();
                    let operand = self.operand(&cx, *value);
                    let value = self.convert(operand, Repr::Boxed);
                    let (state, resume) = self.suspensions[&span.start];
                    let state = self.b.ins().iconst(types::I64, state as i64);
                    let result =
                        self.call_helper(Helper::GeneratorSuspend, &[generator, value, state]);
                    self.b.ins().return_(&[result]);
                    self.resume_blocks.push(resume);
                    self.switch(resume);
                    for &variable in values.iter().chain(&bindings) {
                        let slot = self.b.ins().iconst(types::I64, variable.as_u32() as i64);
                        let value = self.call_helper(Helper::LocalGet, &[slot]);
                        self.b.def_var(variable, value);
                    }
                    self.clear_temporaries();
                    let (block, args) = target(self, target_);
                    self.jump(block, &args);
                }
                TerminatorKind::Fallback(previous) => {
                    let mut args = vec![self.ctx, closure, self.receiver];
                    match &self.arguments {
                        super::Arguments::Fixed(values) => args.extend(values),
                        super::Arguments::Vector { argv, argc } => args.extend([*argv, *argc]),
                    }
                    let undefined = self.constant(Value::UNDEFINED);
                    self.exit_frame(undefined);
                    let signature = self.b.import_signature(self.b.func.signature.clone());
                    let address = self.b.ins().iconst(self.pointer, *previous as i64);
                    if matches!(self.arguments, super::Arguments::Fixed(_)) {
                        self.b.ins().return_call_indirect(signature, address, &args);
                    } else {
                        let call = self.b.ins().call_indirect(signature, address, &args);
                        let results = self.b.inst_results(call).to_vec();
                        self.b.ins().return_(&results);
                    }
                    self.terminated = true;
                }
                TerminatorKind::Return(value) => {
                    let operand = self.operand(&cx, *value);
                    let value = self.convert(operand, Repr::Boxed);
                    let args = self.return_args(value, false);
                    self.jump(self.return_block, &args);
                }
            }
            for successor in graph.successors(block) {
                pending[successor] -= 1;
                if pending[successor] == 0 {
                    self.b.seal_block(blocks[successor.index()]);
                }
            }
            if cold {
                self.b.set_cold_block(blocks[block.index()]);
                for index in created..self.b.func.dfg.num_blocks() {
                    self.b
                        .set_cold_block(cranelift_codegen::ir::Block::from_u32(index as u32));
                }
            }
        }
        if self.annotations.semantic_locations {
            self.b
                .set_srcloc(cranelift_codegen::ir::SourceLoc::default());
        }
        self.finish_failed_exit();
        self.switch(self.return_block);
        let result = self.b.block_params(self.return_block)[0];
        self.return_value(result);
        Ok(())
    }
}

impl FunctionCompiler<'_, '_> {
    /// Whether a result is stored straight into its frame slot, which takes its own reference.
    /// Values that die within their block stay owned by their temporary instead.
    fn slot_owned(&self, graph: &Graph, cx: &Context<'_>, value: GraphValue) -> bool {
        self.frame_size.is_some()
            && self.generator.is_none()
            && cx.facts.of(value).intersects(HEAP)
            && !graph.borrowed.contains(&value)
            && cx.slots.of(value).is_some()
    }

    fn operand(&mut self, cx: &Context<'_>, value: GraphValue) -> Operand {
        let clif = match &cx.direct {
            Some(direct) => direct.borrow()[value].expect("semantic value used before definition"),
            None => self.b.use_var(cx.values[value.index()]),
        };
        Operand::new(clif, cx.reprs[value])
    }

    fn typed_op(
        &mut self,
        cx: &Context<'_>,
        graph: &Graph,
        inst: crate::jit::semantic::Inst,
    ) -> Result<Option<Option<Operand>>, Error> {
        let data = &graph.insts[inst];
        let target = data.result.map(|value| cx.reprs[value]);
        let boolean = |condition: ClifValue| Some(Some(Operand::new(condition, Repr::Bool)));
        Ok(match &data.op {
            Op::Constant(Constant::Number(number)) => match target {
                Some(Repr::Float) => Some(Some(Operand::new(
                    self.b.ins().f64const(*number),
                    Repr::Float,
                ))),
                Some(Repr::Int32) => Some(Some(Operand::new(
                    self.b.ins().iconst(types::I32, *number as i32 as i64),
                    Repr::Int32,
                ))),
                _ => None,
            },
            Op::Constant(Constant::Boolean(value)) if target == Some(Repr::Bool) => {
                let value = self.b.ins().iconst(types::I8, i64::from(*value));
                boolean(value)
            }
            Op::Binary(operator, left, right)
            | Op::GuardedCompare {
                operator,
                left,
                right,
                ..
            } => {
                let op = ops::binary_operator(*operator);
                let applicable = match op {
                    BinaryOp::LooseEqual
                    | BinaryOp::StrictEqual
                    | BinaryOp::LooseNotEqual
                    | BinaryOp::StrictNotEqual => cx.number(*left) && cx.number(*right),
                    BinaryOp::Instanceof | BinaryOp::In | BinaryOp::Rem | BinaryOp::Pow => false,
                    _ => cx.numeric(*left) && cx.numeric(*right),
                };
                let (left, right) = (self.operand(cx, *left), self.operand(cx, *right));
                if left.repr == Repr::Bool
                    && right.repr == Repr::Bool
                    && matches!(op, BinaryOp::StrictEqual | BinaryOp::StrictNotEqual)
                {
                    let condition = if op == BinaryOp::StrictEqual {
                        IntCC::Equal
                    } else {
                        IntCC::NotEqual
                    };
                    let value = self.b.ins().icmp(condition, left.value, right.value);
                    return Ok(boolean(value));
                }
                if !applicable {
                    return Ok(None);
                }
                self.typed_binary(op, left, right).map(Some)
            }
            Op::Unary(UnaryOp::Increment | UnaryOp::Decrement, value) if cx.numeric(*value) => {
                let operand = self.operand(cx, *value);
                let delta = if matches!(data.op, Op::Unary(UnaryOp::Increment, _)) {
                    1
                } else {
                    -1
                };
                Some(Some(if target == Some(Repr::Int32) {
                    let integer = self.convert(operand, Repr::Int32);
                    Operand::new(self.b.ins().iadd_imm_s(integer, delta), Repr::Int32)
                } else {
                    let float = self.convert(operand, Repr::Float);
                    let one = self.b.ins().f64const(delta as f64);
                    Operand::new(self.b.ins().fadd(float, one), Repr::Float)
                }))
            }
            Op::Unary(UnaryOp::Negate, value) if cx.numeric(*value) => {
                let operand = self.operand(cx, *value);
                let float = self.convert(operand, Repr::Float);
                Some(Some(Operand::new(self.b.ins().fneg(float), Repr::Float)))
            }
            Op::ToInt32(value) => {
                debug_assert!(cx.numeric(*value));
                let operand = self.operand(cx, *value);
                let integer = self.convert(operand, Repr::Int32);
                Some(Some(Operand::new(integer, Repr::Int32)))
            }
            Op::Unary(UnaryOp::BitwiseNot, value) if cx.numeric(*value) => {
                let operand = self.operand(cx, *value);
                let integer = self.convert(operand, Repr::Int32);
                Some(Some(Operand::new(self.b.ins().bnot(integer), Repr::Int32)))
            }
            Op::Unary(UnaryOp::Plus, value) | Op::ToNumeric(value) | Op::AsNumber(value)
                if cx.numeric(*value) || matches!(data.op, Op::AsNumber(_)) =>
            {
                let operand = self.operand(cx, *value);
                Some(Some(match operand.repr {
                    Repr::Int32 | Repr::Float => operand,
                    _ => Operand::new(self.convert(operand, Repr::Float), Repr::Float),
                }))
            }
            Op::Unary(UnaryOp::Not, value)
                if cx.numeric(*value) || cx.reprs[*value] != Repr::Boxed =>
            {
                let operand = self.operand(cx, *value);
                let truth = self.condition(operand, true);
                let value = self.b.ins().icmp_imm_s(IntCC::Equal, truth, 0);
                boolean(value)
            }
            Op::IsCode(value, code) => {
                let operand = self.operand(cx, *value);
                let value = self.convert(operand, Repr::Boxed);
                boolean(self.is_code(value, *code))
            }
            Op::IsNumber(value) => {
                let operand = self.operand(cx, *value);
                let value = match operand.repr {
                    Repr::Boxed => self.is_number(operand.value),
                    Repr::Bool => self.b.ins().iconst(types::I8, 0),
                    _ => self.b.ins().iconst(types::I8, 1),
                };
                boolean(value)
            }
            Op::IsUninitialized(value) | Op::IsUndefined(value) | Op::IsNullish(value) => {
                let operand = self.operand(cx, *value);
                let value = if operand.repr != Repr::Boxed {
                    self.b.ins().iconst(types::I8, 0)
                } else {
                    match data.op {
                        Op::IsNullish(_) => self.is_nullish(operand.value),
                        Op::IsUndefined(_) => self.b.ins().icmp_imm_s(
                            IntCC::Equal,
                            operand.value,
                            Value::UNDEFINED.bits() as i64,
                        ),
                        _ => self.b.ins().icmp_imm_s(
                            IntCC::Equal,
                            operand.value,
                            Value::UNINITIALIZED.bits() as i64,
                        ),
                    }
                };
                boolean(value)
            }
            Op::CallNative(native, args) if args.iter().all(|&arg| cx.numeric(arg)) => {
                let function = crate::jit::math::METHODS[native.index()].1;
                let operands: Vec<_> = args.iter().map(|&arg| self.operand(cx, arg)).collect();
                self.typed_math(function, &operands).map(Some)
            }
            Op::IsTyped(object, element) => {
                let operand = self.operand(cx, *object);
                let object = self.convert(operand, Repr::Boxed);
                let view = self.view(object);
                boolean(match element {
                    Some(element) => self.typed_kind_is(view, *element),
                    None => self.has_kind(view, kind::TYPED),
                })
            }
            Op::IsIndexed(object) => {
                let operand = self.operand(cx, *object);
                let object = self.convert(operand, Repr::Boxed);
                let view = self.view(object);
                boolean(self.has_kind(view, kind::ARRAY | kind::TYPED))
            }
            Op::IndexedLength(object) => {
                let operand = self.operand(cx, *object);
                let object = self.convert(operand, Repr::Boxed);
                let view = self.view(object);
                let length = self.view_length(view);
                Some(Some(Operand::new(length, Repr::Float)))
            }
            Op::TypedIndex(
                object,
                key,
                TypedRead {
                    element: Some(element),
                    bounded,
                },
            ) => {
                let view = match graph.views.get(&inst) {
                    Some(&view) => self.operand(cx, view).value,
                    None => {
                        let object = self.operand(cx, *object);
                        let object = self.convert(object, Repr::Boxed);
                        self.view(object)
                    }
                };
                let key = self.operand(cx, *key);
                Some(Some(if *bounded {
                    self.bounded_element_read(view, key, *element)
                } else {
                    self.checked_element_read(view, key, *element)
                }))
            }
            Op::GetIndex(object, key)
            | Op::TypedIndex(object, key, _)
            | Op::IndexedGet(object, key)
            | Op::SpreadGet(object, key) => {
                let fallback = !matches!(data.op, Op::TypedIndex(..) | Op::SpreadGet(..));
                let object = self.operand(cx, *object);
                let object = self.convert(object, Repr::Boxed);
                let view = match graph.views.get(&inst) {
                    Some(&view) => self.operand(cx, view).value,
                    None => self.view(object),
                };
                let key = match self.operand(cx, *key) {
                    key if key.repr == Repr::Bool => {
                        Operand::new(self.convert(key, Repr::Boxed), Repr::Boxed)
                    }
                    key => key,
                };
                let retain = data.result.is_none_or(|value| {
                    !graph.borrowed.contains(&value) && !self.slot_owned(graph, cx, value)
                });
                let result = self.viewed_index_access(view, object, key, None, fallback, retain);
                Some(Some(Operand::new(result, Repr::Boxed)))
            }
            Op::SetIndex(object, key, value) => {
                let Some(&view) = graph.views.get(&inst) else {
                    return Ok(None);
                };
                let view = self.operand(cx, view).value;
                let object = self.operand(cx, *object);
                let object = self.convert(object, Repr::Boxed);
                let key = self.operand(cx, *key);
                let value = self.operand(cx, *value);
                self.viewed_index_access(view, object, key, Some(value), true, true);
                Some(None)
            }
            _ => None,
        })
    }
}

/// Blocks on a cycle of the control flow graph, which may run more than once per call.
fn cyclic_blocks(
    graph: &Graph,
    cfg: &crate::jit::semantic::Cfg,
) -> std::collections::HashSet<crate::jit::semantic::Block> {
    let mut finished = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for &root in &cfg.rpo {
        if !seen.insert(root) {
            continue;
        }
        let mut stack = vec![(root, graph.successors(root), 0)];
        while let Some((block, successors, next)) = stack.last_mut() {
            if let Some(&successor) = successors.get(*next) {
                *next += 1;
                if seen.insert(successor) {
                    stack.push((successor, graph.successors(successor), 0));
                }
            } else {
                finished.push(*block);
                stack.pop();
            }
        }
    }
    let mut cyclic = std::collections::HashSet::new();
    let mut assigned = std::collections::HashSet::new();
    for &root in finished.iter().rev() {
        if !assigned.insert(root) {
            continue;
        }
        let mut component = vec![root];
        let mut stack = vec![root];
        while let Some(block) = stack.pop() {
            for &pred in &cfg.preds[block] {
                if assigned.insert(pred) {
                    component.push(pred);
                    stack.push(pred);
                }
            }
        }
        if component.len() > 1 || graph.successors(root).contains(&root) {
            cyclic.extend(component);
        }
    }
    cyclic
}

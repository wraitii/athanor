use crate::jit::semantic::JsOp;
use crate::{FxHashMap, FxHashSet};
use std::cell::RefCell;

use super::{BlockArg, LoweredValue};
use cranelift_codegen::entity::{EntityRef, SecondaryMap};
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{MemFlagsData, types};
use cranelift_module::{FuncId, Module};
use oxc_span::Span;

use crate::error::Error;
use crate::jit::helpers::Helper;
use crate::jit::semantic::{
    Block, Constant, Facts, Graph, Inst, InstData, MemoryFormat, MemoryOp, Op, TerminatorKind,
    TypedRead, UnaryOp, Value as GraphValue,
};
use crate::value::Value;

use super::FunctionCompiler;
use super::ops::{self, BinaryOp, UnaryOp as EmitUnaryOp};
use super::repr::Operand;
use crate::jit::semantic::Repr;
use crate::jit::semantic::ValueType;
use crate::jit::semantic::contracts::ResultReference;
use crate::jit::semantic::ownership::{Holder, Plan};
use crate::jit::semantic::regex;
use crate::jit::semantic::runtime::layout;

const UNMANAGED: ValueType = ValueType::NUMBER
    .union(ValueType::BOOLEAN)
    .union(ValueType::UNDEFINED)
    .union(ValueType::NULL);

pub(super) struct Context<'a> {
    entry: Block,
    plan: Plan,
    cyclic: FxHashSet<Block>,
    entering_guards: FxHashMap<Inst, String>,
    terminator_base: usize,
    osr_buffers: RefCell<FxHashMap<GraphValue, (LoweredValue, usize)>>,
    direct: RefCell<SecondaryMap<GraphValue, Option<LoweredValue>>>,
    facts: &'a Facts,
}

impl<'a> Context<'a> {
    pub(super) fn new(graph: &mut Graph, facts: &'a Facts) -> Self {
        let rpo = &facts.cfg.rpo;
        let entering_guards = if crate::jit::bailouts::enabled() {
            rpo.iter()
                .flat_map(|&block| &graph.blocks[block].insts)
                .filter(|&&inst| matches!(graph.insts[inst].op, Op::Continuation(..)))
                .map(|&inst| (inst, graph.entering_guard(inst)))
                .collect()
        } else {
            FxHashMap::default()
        };
        let reachable: FxHashSet<_> = rpo.iter().copied().collect();
        for (block, data) in graph.blocks.iter_mut() {
            if !reachable.contains(&block) {
                data.insts.clear();
                data.terminator = None;
            }
        }
        Self {
            entry: graph.entry,
            plan: graph
                .ownership
                .take()
                .expect("ownership is planned before emission"),
            cyclic: cyclic_blocks(graph, &facts.cfg),
            entering_guards,
            terminator_base: graph.insts.len(),
            osr_buffers: RefCell::new(FxHashMap::default()),
            direct: RefCell::new(SecondaryMap::new()),
            facts,
        }
    }

    fn terminator_location(&self, block: Block) -> u32 {
        u32::try_from(self.terminator_base + block.index() + 1).unwrap()
    }

    fn define(&self, value: GraphValue, clif: LoweredValue) {
        self.direct.borrow_mut()[value] = Some(clif);
    }
}

impl FunctionCompiler<'_, '_> {
    fn observe(
        &mut self,
        value: LoweredValue,
        mut site: crate::jit::profile::Site,
        kind: crate::jit::profile::Observation,
    ) {
        let Some(feedback) = self.feedback else {
            return;
        };
        use crate::jit::profile::Observation;
        if !feedback.observations
            && !matches!(
                kind,
                Observation::NumberFailure | Observation::PrimitiveFailure
            )
        {
            return;
        }
        if !feedback.get_refinement && kind != Observation::Call {
            return;
        }
        if site.source == 0 {
            site.source = self.source_name.0;
        }
        let helper = match kind {
            Observation::Call => Helper::ObserveCall,
            Observation::Read(crate::jit::profile::ReadKind::Result) => Helper::ObserveGet,
            Observation::Read(crate::jit::profile::ReadKind::Receiver) => Helper::ObserveIndex,
            Observation::Read(crate::jit::profile::ReadKind::Number) => Helper::ObserveNumber,
            Observation::Read(crate::jit::profile::ReadKind::Pattern) => Helper::ObservePattern,
            Observation::Read(crate::jit::profile::ReadKind::Primitive) => Helper::ObservePrimitive,
            Observation::NumberFailure => Helper::ObserveNumberFailure,
            Observation::PrimitiveFailure => Helper::ObservePrimitiveFailure,
        };
        let address = self.function_data(crate::jit::profile::ObservationCache {
            complete: 0,
            last_callee: 0,
            failures: 0,
            last_shape: 0,
            owner: feedback.owner,
            site,
        });
        let complete = self
            .b
            .ins()
            .load(types::I64, MemFlagsData::trusted(), address, 0);
        let observe = self.b.create_block();
        let next = self.b.create_block();
        let complete = if kind == Observation::Call {
            let last = self.b.ins().load(
                types::I64,
                MemFlagsData::trusted(),
                address,
                std::mem::offset_of!(crate::jit::profile::ObservationCache, last_callee) as i32,
            );
            let word = self.b.as_word(value);
            let same = self.b.ins().icmp(IntCC::Equal, word, last);
            let complete = self.b.ins().icmp_imm_s(IntCC::NotEqual, complete, 0);
            self.b.ins().boolean_or(complete, same)
        } else {
            complete
        };
        self.b.ins().brif(complete, next, &[], observe, &[]);
        self.b.switch_to_block(observe);
        self.call_helper(helper, &[value, address]);
        self.b.ins().jump(next, &[]);
        self.b.switch_to_block(next);
    }

    fn property_observer(
        &self,
        data: &crate::jit::semantic::InstData,
    ) -> Option<(u64, crate::jit::profile::Site)> {
        let feedback = self
            .feedback
            .filter(|feedback| feedback.observations && feedback.get_refinement)?;
        Some((
            feedback.owner,
            crate::jit::profile::Site {
                inline_context: data.inline_context,
                source: data.source_name.map_or(self.source_name.0, |name| name.0),
                start: data.span.start,
                end: data.span.end,
            },
        ))
    }

    fn observe_method(
        &mut self,
        callee: LoweredValue,
        receiver: LoweredValue,
        mut site: crate::jit::profile::Site,
    ) {
        let Some(feedback) = self.feedback.filter(|feedback| feedback.observations) else {
            return;
        };
        use crate::jit::profile::ObservationCache;
        if site.source == 0 {
            site.source = self.source_name.0;
        }
        let address = self.function_data(ObservationCache {
            complete: 0,
            last_callee: 0,
            failures: 0,
            last_shape: 0,
            owner: feedback.owner,
            site,
        });
        let flags = MemFlagsData::trusted();
        let complete = self.b.ins().load(types::I64, flags, address, 0);
        let inspect = self.b.create_block();
        let done = self.b.create_block();
        self.b.ins().brif(complete, done, &[], inspect, &[]);
        self.b.switch_to_block(inspect);
        let receiver = self.b.as_word(receiver);
        let object = layout::is_object(self.b, receiver);
        let object_path = self.b.create_block();
        let check = self.b.create_block();
        self.b.append_block_param(check, types::I64);
        let no_shape = self
            .b
            .ins()
            .iconst(types::I64, i64::from(crate::heap::NO_SHAPE));
        self.b.ins().brif(
            object,
            object_path,
            &[],
            check,
            &[BlockArg::Value(no_shape)],
        );
        self.b.switch_to_block(object_path);
        let view = layout::heap_view(self.b, receiver);
        let view = if self.pointer == types::I64 {
            view
        } else {
            self.b.ins().ireduce(self.pointer, view)
        };
        let shape = self
            .b
            .ins()
            .load(types::I32, flags, view, crate::heap::SHAPE_OFFSET);
        let shape = self.b.ins().uextend(types::I64, shape);
        self.b.ins().jump(check, &[BlockArg::Value(shape)]);
        self.b.switch_to_block(check);
        let shape = self.b.block_params(check)[0];
        let last_shape = self.b.ins().load(
            types::I64,
            flags,
            address,
            std::mem::offset_of!(ObservationCache, last_shape) as i32,
        );
        let last_callee = self.b.ins().load(
            types::I64,
            flags,
            address,
            std::mem::offset_of!(ObservationCache, last_callee) as i32,
        );
        let word = self.b.as_word(callee);
        let same_callee = self.b.ins().icmp(IntCC::Equal, word, last_callee);
        let same_shape = self.b.ins().icmp(IntCC::Equal, shape, last_shape);
        let same = self.b.ins().boolean_and(same_callee, same_shape);
        let observe = self.b.create_block();
        self.b.ins().brif(same, done, &[], observe, &[]);
        self.b.switch_to_block(observe);
        self.call_helper(Helper::ObserveMethod, &[callee, shape, address]);
        self.b.ins().jump(done, &[]);
        self.b.switch_to_block(done);
    }

    fn epoch(&mut self) -> LoweredValue {
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

    fn closure_object(
        &mut self,
        span: Span,
        id: FuncId,
        cells: &[LoweredValue],
        (arrow, generator, lexical_this): (bool, bool, bool),
        lexical_receiver: Option<LoweredValue>,
    ) -> Result<LoweredValue, Error> {
        let argv = self.stack_values(cells, span)?;
        let count = self.b.ins().iconst(types::I64, cells.len() as i64);
        let code = self.b.ins().iconst(types::I64, id.as_u32() as i64);
        let result = self.call_helper(Helper::ClosureNew, &[code, argv, count]);
        self.check_error();
        if arrow {
            let receiver = if lexical_this {
                lexical_receiver.unwrap_or(self.receiver)
            } else {
                self.constant(Value::UNDEFINED)
            };
            self.call_helper(Helper::BindThis, &[result, receiver]);
            self.check_error();
        } else if generator {
            self.call_helper(Helper::DisableConstructor, &[result]);
            self.check_error();
            self.check_error();
        }
        Ok(result)
    }

    pub(super) fn finalize_runtime_lowering(
        &mut self,
        cx: Context<'_>,
        closure: LoweredValue,
        continuations: &FxHashMap<usize, FuncId>,
    ) -> Result<(), Error> {
        let source_name = self.source_name;
        let cells: Vec<i64> = (0..self.b.graph.guards)
            .map(|_| {
                let mut cell = Box::new(u64::MAX);
                let address = &raw mut *cell as i64;
                self.cells.push(cell);
                address
            })
            .collect();
        let reachable = &cx.facts.cfg.rpo;
        let packs: Vec<Inst> = reachable
            .iter()
            .flat_map(|&block| self.b.graph.blocks[block].insts.iter())
            .copied()
            .filter(|&inst| {
                matches!(
                    self.b.graph.insts[inst].op,
                    Op::Javascript(JsOp::NewPack | JsOp::Spread(_) | JsOp::StringSplitParts(..))
                )
            })
            .collect();
        let mut pack_slots = SecondaryMap::new();
        let mut pack_cleanup = Vec::new();
        for inst in packs {
            let slot = self
                .b
                .create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                    cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                    std::mem::size_of::<crate::jit::pack::StackPack>() as u32,
                    std::mem::align_of::<crate::jit::pack::StackPack>().trailing_zeros() as u8,
                ));
            let address = self.b.ins().stack_addr(self.pointer, slot, 0);
            let address = self.b.as_boxed(address);
            self.call_helper(Helper::PackInit, &[address]);
            pack_cleanup.push(address);
            pack_slots[inst] = Some(address);
        }

        let continuation_words = reachable
            .iter()
            .flat_map(|&block| &self.b.graph.blocks[block].insts)
            .filter_map(|&inst| match &self.b.graph.insts[inst].op {
                Op::Continuation(_, inputs) => Some(inputs.len()),
                _ => None,
            })
            .max();
        let continuation_buffer = continuation_words.map(|words| {
            let slot = self
                .b
                .create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                    cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                    (words.max(1) * 8) as u32,
                    3,
                ));
            self.b.ins().stack_addr(self.pointer, slot, 0)
        });

        let has_handlers = self.b.graph.has_handlers;
        let repeated = |block: Block| has_handlers || cx.cyclic.contains(&block);
        for (value, slot) in cx.plan.slots.assigned() {
            self.slots.insert(value, slot);
        }
        self.slot_count = cx.plan.slots.count();
        self.jump(cx.entry, &[]);
        for &block in reachable {
            let insts = std::mem::take(&mut self.b.graph.blocks[block].insts);
            let terminator = self.b.graph.blocks[block]
                .terminator
                .take()
                .expect("reachable block has a terminator");
            if self.semantic_locations {
                self.b
                    .set_srcloc(cranelift_codegen::ir::SourceLoc::default());
            }
            self.switch(block);
            let cold = self.b.graph.cold.contains(&block);
            let created = self.b.num_blocks();
            for value in self.b.block_params(block).to_vec() {
                cx.define(value, value);
            }
            for &inst in &insts {
                self.references.begin();
                let handler = match &terminator.kind {
                    TerminatorKind::CheckException(_, error) => Some(error.block),
                    _ => None,
                };
                if let Some(handler) = handler {
                    self.handlers.push(handler);
                }
                let data = self.b.graph.insts[inst].clone();
                let data = &data;
                self.source_span = data.span;
                self.b.set_span(data.span);
                self.source_name = data.source_name.unwrap_or(source_name);
                self.b.set_source_name(Some(self.source_name));
                let location = if self.semantic_locations {
                    cranelift_codegen::ir::SourceLoc::new(self.b.graph.inst_location(inst))
                } else if self.source_name == source_name {
                    cranelift_codegen::ir::SourceLoc::new(data.span.start)
                } else {
                    cranelift_codegen::ir::SourceLoc::default()
                };
                self.b.set_srcloc(location);
                let get = |this: &mut Self, value: GraphValue| {
                    let operand = this.operand(&cx, value);
                    debug_assert_eq!(operand.repr, Repr::Boxed, "{inst} uses unboxed {value}");
                    operand.value
                };
                let result = if deferred(&data.op) {
                    self.adopt(&cx, inst);
                    data.result()
                        .map(|value| Operand::new(value, self.b.graph.repr(value)))
                } else {
                    let target = data.result().map(|value| self.b.graph.repr(value));
                    'lowered: {
                        let result = match &data.op {
                            Op::Aarch64(_) => unreachable!("target operation before selection"),
                            Op::Regex(regex::Op::Class(..)) => {
                                unreachable!("regex class before regex lowering")
                            }
                            Op::Regex(_) => {
                                break 'lowered self.typed_op(&cx, inst, data, target)?;
                            }
                            Op::LowLevel(_) => {
                                return Err(Error::Compile(
                                    "low-level operation reached runtime lowering finalization"
                                        .into(),
                                ));
                            }
                            Op::Continuation(target, inputs) => {
                                if crate::jit::bailouts::enabled() {
                                    let site = crate::jit::bailouts::register(format!(
                                        "{} => {} (source {}..{}) guard: {}",
                                        self.b.graph.name,
                                        target.graph.name,
                                        data.span.start,
                                        data.span.end,
                                        Some(crate::jit::bailouts::guard_text(data.span))
                                            .filter(|text| !text.is_empty())
                                            .unwrap_or_else(|| cx.entering_guards[&inst].clone())
                                    ));
                                    let site = self.b.ins().iconst(types::I64, site as i64);
                                    self.call_helper(Helper::Bailout, &[site]);
                                }
                                let id = continuations[&(std::sync::Arc::as_ptr(target) as usize)];
                                let signature = self
                                    .module
                                    .declarations()
                                    .get_function_decl(id)
                                    .signature
                                    .clone();
                                let function = self.b.import_function(id, signature);
                                let argv = if target.outputs == 0 {
                                    continuation_buffer.unwrap()
                                } else {
                                    let words = inputs.len().max(target.outputs);
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
                                        .insert(data.result().unwrap(), (address, 0));
                                    address
                                };
                                for (index, &input) in inputs.iter().enumerate() {
                                    let value = get(self, input);
                                    if cx.plan.holder(input) == Holder::Borrowed {
                                        let heap = self.is_heap(value);
                                        let retain = self.b.create_block();
                                        let stored = self.b.create_block();
                                        self.brif(heap, retain, &[], stored, &[]);
                                        self.switch(retain);
                                        self.retain_temporary(value);
                                        self.jump(stored, &[]);
                                        self.switch(stored);
                                    }
                                    let word = self.b.as_word(value);
                                    self.b.ins().store(
                                        MemFlagsData::trusted(),
                                        word,
                                        argv,
                                        (index * 8) as i32,
                                    );
                                }
                                let argc = self.b.ins().iconst(types::I64, inputs.len() as i64);
                                let args = [self.ctx, closure, self.receiver, argv, argc];
                                let call = self.b.ins().call(function, &args);
                                let value = self.b.inst_results(call)[0];
                                let value = self.b.as_boxed(value);
                                self.check_error();
                                Some(self.own_temporary(value))
                            }
                            Op::OsrStart(plan, candidate, inputs) => {
                                let transfer = plan.loops[*candidate].transfer.as_ref().unwrap();
                                let words = transfer.words();
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
                                    .insert(data.result().unwrap(), (address, 0));
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
                                    let tick = self
                                        .b
                                        .ins()
                                        .band_imm_s(count, crate::jit::osr::TICK as i64 - 1);
                                    let tick = self.b.ins().icmp_imm_s(IntCC::Equal, tick, 0);
                                    let entry = self.b.ins().load(
                                        self.pointer,
                                        MemFlagsData::trusted(),
                                        site,
                                        crate::jit::osr::ENTRY_OFFSET,
                                    );
                                    let ready = self.b.ins().icmp_imm_s(IntCC::NotEqual, entry, 0);
                                    let poll = self.b.ins().boolean_or(tick, ready);
                                    let slow = self.b.create_block();
                                    let done = self.b.create_block();
                                    self.b.append_block_param_repr(done, Repr::Boxed);
                                    let declined = self.constant(Value::number(-1.0));
                                    self.brif(poll, slow, &[], done, &[declined.into()]);
                                    self.switch(slow);
                                    for (index, &input) in inputs.iter().enumerate() {
                                        let operand = self.operand(&cx, input);
                                        let boxed = operand.repr == Repr::Boxed;
                                        let value = self.convert(operand, Repr::Boxed);
                                        if boxed && cx.plan.holder(input) == Holder::Borrowed {
                                            let heap = self.is_heap(value);
                                            let retain = self.b.create_block();
                                            let stored = self.b.create_block();
                                            self.brif(heap, retain, &[], stored, &[]);
                                            self.switch(retain);
                                            self.retain_temporary(value);
                                            self.jump(stored, &[]);
                                            self.switch(stored);
                                        }
                                        let word = self.b.as_word(value);
                                        self.b.ins().store(
                                            MemFlagsData::trusted(),
                                            word,
                                            address,
                                            (index * 8) as i32,
                                        );
                                    }
                                    let result = self.call_helper(
                                        Helper::OsrPoll,
                                        &[self.receiver, address, site],
                                    );
                                    self.check_error();
                                    self.jump(done, &[result.into()]);
                                    self.switch(done);
                                    Some(self.b.block_params(done)[0])
                                } else {
                                    Some(self.constant(Value::number(-1.0)))
                                }
                            }
                            Op::EntryAssume(..) => Some(self.constant(Value::boolean(true))),
                            Op::Assume(..) => None,
                            Op::Javascript(JsOp::StringConcat(..))
                            | Op::ReadHint(..)
                            | Op::KeepAlive(..)
                            | Op::Javascript(JsOp::View(..))
                            | Op::BooleanAnd(..)
                            | Op::Javascript(
                                JsOp::IsInBounds(..)
                                | JsOp::IsExactPosition(..)
                                | JsOp::IntegerCheck(..)
                                | JsOp::IsInteger(..)
                                | JsOp::IsIntegerInRange(..)
                                | JsOp::IsCode(..)
                                | JsOp::IsShape(..)
                                | JsOp::IsNumber(..)
                                | JsOp::IsUninitialized(..)
                                | JsOp::IsUndefined(..)
                                | JsOp::IsNullish(..)
                                | JsOp::IsTyped(..)
                                | JsOp::TypedSet(..)
                                | JsOp::IsString(..)
                                | JsOp::IsIndexed(..)
                                | JsOp::IsOrdinaryArray(..)
                                | JsOp::CellNew(..)
                                | JsOp::CellGet(..)
                                | JsOp::CellSet(..)
                                | JsOp::CaptureGet(..)
                                | JsOp::Closure(..)
                                | JsOp::Read(_)
                                | JsOp::Write(..)
                                | JsOp::Initialize(..)
                                | JsOp::BindingLoad(_)
                                | JsOp::BindingRestore(..),
                            ) => {
                                unreachable!("{:?} must be lowered before emission", data.op)
                            }
                            Op::Javascript(JsOp::Constant(
                                Constant::Undefined
                                | Constant::Null
                                | Constant::Boolean(_)
                                | Constant::Number(_)
                                | Constant::Uninitialized,
                            ))
                            | Op::Convert(..)
                            | Op::Box(_)
                            | Op::AsBoxed(_)
                            | Op::AsWord(_)
                            | Op::Unbox(..)
                            | Op::Javascript(
                                JsOp::NumericFloat(_)
                                | JsOp::SaturatingInteger(_)
                                | JsOp::AsInteger(_)
                                | JsOp::ArrayPosition(_)
                                | JsOp::ToInt32(_)
                                | JsOp::AsNumber(_),
                            ) => {
                                unreachable!("{:?} is adopted", data.op)
                            }
                            Op::Retain(graph_value)
                                if cx.plan.holder(*graph_value) == Holder::Temporary =>
                            {
                                let value = get(self, *graph_value);
                                let heap = self.is_heap(value);
                                let retain = self.b.create_block();
                                let done = self.b.create_block();
                                self.brif(heap, retain, &[], done, &[]);
                                self.switch(retain);
                                self.retain_temporary(value);
                                self.jump(done, &[]);
                                self.switch(done);
                                self.references
                                    .bind(self.b, graph_value.index() as u32, value, true, true)
                                    .map_err(Error::Compile)?;
                                None
                            }
                            Op::Retain(value) => {
                                let value = get(self, *value);
                                self.increment(value);
                                None
                            }
                            Op::SlotStore(value) => {
                                self.references.move_owner(self.b, value.index() as u32);
                                let repeated = repeated(block) || cx.plan.slots.shared(*value);
                                let slot = *value;
                                let value = get(self, slot);
                                self.replace_local(slot, value, repeated);
                                None
                            }
                            Op::ParamStore(param, value) => {
                                let value = get(self, *value);
                                self.replace_local(*param, value, true);
                                None
                            }
                            Op::SlotClear(value) => {
                                self.clear_local(*value);
                                None
                            }
                            Op::Release(value) => {
                                self.references.release_owner(self.b, value.index() as u32);
                                None
                            }
                            Op::Observe(value, kind, site) => {
                                let value = get(self, *value);
                                self.observe(value, *site, *kind);
                                None
                            }
                            Op::ObserveMethod(callee, receiver, site) => {
                                let callee = get(self, *callee);
                                let receiver = get(self, *receiver);
                                self.observe_method(callee, receiver, *site);
                                None
                            }
                            Op::ContinuationOutput(call, index) => {
                                let (address, _) = cx.osr_buffers.borrow()[call];
                                let value = self.b.ins().load(
                                    types::I64,
                                    MemFlagsData::trusted(),
                                    address,
                                    (index * 8) as i32,
                                );
                                let value = self.b.as_boxed(value);
                                Some(self.own_temporary(value))
                            }
                            Op::ContinuationStore(values) => {
                                let super::Arguments::Vector { argv, .. } = self.arguments else {
                                    return Err(Error::Compile(
                                        "continuation exit requires vector arguments".into(),
                                    ));
                                };
                                for (index, &value) in values.iter().enumerate() {
                                    let value = get(self, value);
                                    self.increment(value);
                                    let word = self.b.as_word(value);
                                    self.b.ins().store(
                                        MemFlagsData::trusted(),
                                        word,
                                        argv,
                                        (index * 8) as i32,
                                    );
                                }
                                None
                            }
                            Op::Argument(index) | Op::Restored(index) => {
                                Some(self.load_arg(*index))
                            }
                            Op::Javascript(JsOp::Arguments) => {
                                let (argv, argc) = self.argument_vector()?;
                                let value = self.call_helper(Helper::ArrayNew, &[argv, argc]);
                                self.check_error();
                                Some(value)
                            }
                            Op::Javascript(JsOp::CallForward(callee, receiver)) => {
                                let callee = get(self, *callee);
                                let receiver = get(self, *receiver);
                                let (argv, argc) = self.argument_vector()?;
                                let value =
                                    self.call_helper(Helper::Call, &[callee, receiver, argv, argc]);
                                self.check_error();
                                Some(value)
                            }
                            Op::ArgumentCount => {
                                let (_, argc) = self.argument_vector()?;
                                Some(self.b.ins().fcvt_from_uint(types::F64, argc))
                            }
                            Op::Javascript(JsOp::Rest(index)) => {
                                let index = self.b.ins().iconst(types::I64, *index as i64);
                                let (argv, argc) = self.argument_vector()?;
                                let value = self.call_helper(Helper::Rest, &[argv, argc, index]);
                                self.check_error();
                                Some(value)
                            }
                            Op::Javascript(JsOp::This) => Some(self.receiver),
                            Op::Javascript(JsOp::Callee) => Some(closure),
                            Op::Javascript(
                                JsOp::ClosureThis(..)
                                | JsOp::IsArrayIteration(..)
                                | JsOp::NewHoleyArray(..)
                                | JsOp::MapElement(..)
                                | JsOp::Throw(..)
                                | JsOp::PropertyKey(..)
                                | JsOp::IteratorNew(..)
                                | JsOp::IteratorNext(..)
                                | JsOp::EnumerateNew(..)
                                | JsOp::EnumerateNext(..)
                                | JsOp::NewObject
                                | JsOp::MarkClass(..)
                                | JsOp::DisableConstructor(..)
                                | JsOp::CachedProperty(..)
                                | JsOp::ShapedGet(..)
                                | JsOp::FillPropertyCache(..)
                                | JsOp::PrimePropertyCache(..)
                                | JsOp::PropertyLoad(..)
                                | JsOp::RelatedPropertyLoad(..),
                            )
                            | Op::HelperCall { .. }
                            | Op::Javascript(
                                JsOp::ArrayStore(..)
                                | JsOp::PropertyStore(..)
                                | JsOp::PropertyAdd(..)
                                | JsOp::CanOverwrite(..)
                                | JsOp::FillPropertyStore(..)
                                | JsOp::PackAppend(..)
                                | JsOp::PackRelease(..)
                                | JsOp::PackTake(..)
                                | JsOp::TypedData(..),
                            )
                            | Op::Memory(..)
                            | Op::Javascript(
                                JsOp::MemoryNumber(..)
                                | JsOp::IsTypeof(..)
                                | JsOp::IsArrayIndex(..)
                                | JsOp::IsArray(..)
                                | JsOp::ArrayKind(..)
                                | JsOp::ArrayLoad(..)
                                | JsOp::GlobalCache(..)
                                | JsOp::PropertyCache(..)
                                | JsOp::IsPropertyCached(..)
                                | JsOp::ArrayAppend(..)
                                | JsOp::CallPack(..)
                                | JsOp::ConstructPack(..)
                                | JsOp::NewInstance(..)
                                | JsOp::ConstructorResult(..)
                                | JsOp::IndexedLength(..)
                                | JsOp::NewArray(..)
                                | JsOp::Construct { .. }
                                | JsOp::ConstructResolved { .. }
                                | JsOp::ConstructorCall { .. }
                                | JsOp::StringSplit(..)
                                | JsOp::ConstructorCallPack(..)
                                | JsOp::ConstructorPrototype(..)
                                | JsOp::ConstructPackResolved(..)
                                | JsOp::ProbeIndex(..)
                                | JsOp::IsStringSplit(..)
                                | JsOp::IsNative(..)
                                | JsOp::GetIndex(..)
                                | JsOp::Delete(..)
                                | JsOp::ArrayGet(..)
                                | JsOp::SetIndex(..)
                                | JsOp::Truthy(..)
                                | JsOp::TypedIndex(..),
                            ) => {
                                break 'lowered self.typed_op(&cx, inst, data, target)?;
                            }
                            Op::Javascript(JsOp::Catch) => {
                                // A caught throw abandons argument evaluation in this frame.
                                for &address in &pack_cleanup {
                                    self.call_helper(Helper::PackRelease, &[address]);
                                }
                                let dummy = self.constant(Value::UNDEFINED);
                                Some(self.call_helper(Helper::Catch, &[dummy]))
                            }
                            Op::Javascript(JsOp::ClosureCells(
                                span,
                                id,
                                arrow,
                                generator,
                                lexical_this,
                                cells,
                                lexical_receiver,
                            )) => {
                                let cells: Vec<_> =
                                    cells.iter().map(|&cell| get(self, cell)).collect();
                                let lexical_receiver =
                                    lexical_receiver.map(|value| get(self, value));
                                Some(self.closure_object(
                                    *span,
                                    *id,
                                    &cells,
                                    (*arrow, *generator, *lexical_this),
                                    lexical_receiver,
                                )?)
                            }
                            Op::Javascript(JsOp::Constant(Constant::String(value))) => {
                                Some(self.string_constant(value))
                            }
                            Op::Javascript(JsOp::GlobalRead(..) | JsOp::GlobalReadOptional(..)) => {
                                unreachable!("global reads must be lowered before ownership")
                            }
                            Op::Javascript(JsOp::RegExp(pattern, flags)) => {
                                let pattern: Box<str> = pattern.as_str().into();
                                let ptr =
                                    self.b.ins().iconst(self.pointer, pattern.as_ptr() as i64);
                                let len = self.b.ins().iconst(types::I64, pattern.len() as i64);
                                let flags = self.b.ins().iconst(types::I64, *flags as i64);
                                self.names.push(pattern);
                                let result =
                                    self.call_helper(Helper::RegExpNew, &[flags, ptr, len]);
                                self.check_error();
                                Some(result)
                            }
                            Op::Javascript(JsOp::GlobalInitialize(name, value)) => {
                                let value = get(self, *value);
                                self.global_initialize(name, value);
                                None
                            }
                            Op::Javascript(JsOp::GlobalWrite(name, value)) => {
                                let value = get(self, *value);
                                self.global_set(name, value);
                                None
                            }
                            Op::Javascript(JsOp::CheckInitialized(value)) => {
                                let value = get(self, *value);
                                let value = self.checked_temporary(value, data.result().is_some());
                                Some(value)
                            }
                            Op::Javascript(JsOp::GuardedCompare {
                                operator,
                                left,
                                right,
                                ..
                            }) => {
                                let left = get(self, *left);
                                let right = get(self, *right);
                                Some(self.binary(ops::binary_operator(*operator), left, right))
                            }
                            Op::Javascript(JsOp::DefineHidden(object, name, value)) => {
                                let object = get(self, *object);
                                let value = get(self, *value);
                                self.define_hidden_property(object, name, value);
                                None
                            }
                            Op::Javascript(
                                JsOp::Set(..)
                                | JsOp::Define(..)
                                | JsOp::ShapedSet(..)
                                | JsOp::Get(..),
                            ) => {
                                unreachable!("property accesses must be lowered before ownership")
                            }
                            Op::Javascript(JsOp::NewPack) => {
                                let destination = pack_slots[inst]
                                    .unwrap_or_else(|| self.constant(Value::UNDEFINED));
                                let value = self.call_helper(Helper::NewPack, &[destination]);
                                self.check_error();
                                Some(value)
                            }
                            Op::Javascript(JsOp::Spread(value)) => {
                                let value = get(self, *value);
                                let destination = pack_slots[inst]
                                    .unwrap_or_else(|| self.constant(Value::UNDEFINED));
                                let pack = self.call_helper(Helper::Spread, &[destination, value]);
                                self.check_error();
                                Some(pack)
                            }
                            Op::Javascript(JsOp::AsObject(value) | JsOp::AsString(value)) => {
                                Some(get(self, *value))
                            }
                            Op::Javascript(JsOp::ToNumeric(value)) => {
                                let value = get(self, *value);
                                Some(ops::to_number(self, value))
                            }
                            Op::Javascript(JsOp::Unary(op, value)) => {
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
                            Op::Javascript(JsOp::Binary(op, left, right)) => {
                                let left = get(self, *left);
                                let right = get(self, *right);
                                Some(self.binary(ops::binary_operator(*op), left, right))
                            }
                            Op::Javascript(JsOp::Call {
                                callee,
                                receiver,
                                args,
                                target,
                            }) => {
                                let closure = self.b.graph.defining_op(*callee).cloned();
                                let identity = cx.facts.identity_at(*callee, data.block);
                                let callee = get(self, *callee);
                                let receiver = receiver
                                    .map(|v| get(self, v))
                                    .unwrap_or_else(|| self.constant(Value::UNDEFINED));
                                let args: Vec<_> = args.iter().map(|&v| get(self, v)).collect();
                                let value = match target {
                                    Some(target) => {
                                        let receiver = match closure {
                                            _ if !target.lexical_this => receiver,
                                            Some(Op::Javascript(JsOp::Closure(
                                                _,
                                                _,
                                                true,
                                                _,
                                                true,
                                                _,
                                                lexical,
                                            ))) => lexical.map_or(self.receiver, |v| get(self, v)),
                                            _ => self.call_helper(
                                                Helper::LexicalReceiver,
                                                &[callee, receiver],
                                            ),
                                        };
                                        let owned = cx.plan.results[inst] == ResultReference::Slot;
                                        self.direct_call(
                                            *target,
                                            callee,
                                            receiver,
                                            &args,
                                            data.span,
                                            owned,
                                            identity
                                                == crate::jit::semantic::Identity::Code(
                                                    target.code,
                                                ),
                                        )?
                                    }
                                    None => {
                                        self.generic_call(callee, receiver, &args, data.span)?
                                    }
                                };
                                Some(value)
                            }
                            Op::Javascript(JsOp::ProbeGlobal(..) | JsOp::ProbeProperty(..)) => {
                                unreachable!("cache probes must be lowered before ownership")
                            }
                            Op::Javascript(
                                JsOp::ProbeShapeProperty(..)
                                | JsOp::LayoutGet(..)
                                | JsOp::IsReadLayout(..),
                            ) => {
                                unreachable!("layout reads must be lowered before ownership")
                            }
                            Op::Javascript(JsOp::StringSplitParts(receiver, args)) => {
                                let receiver = get(self, *receiver);
                                let args: Vec<_> = args.iter().map(|&v| get(self, v)).collect();
                                let argv = self.stack_values(&args, data.span)?;
                                let operation = pack_slots[inst]
                                    .unwrap_or_else(|| self.constant(Value::UNDEFINED));
                                let argc = self.b.ins().iconst(types::I64, args.len() as i64);
                                let track = cx.plan.results[inst] != ResultReference::Slot;
                                let result = self.call_helper_tracked(
                                    Helper::StringSplitParts,
                                    &[operation, receiver, argv, argc],
                                    track,
                                );
                                self.check_error();
                                Some(result)
                            }
                            Op::Javascript(JsOp::GuardValid(guard)) => {
                                let epoch = self.epoch();
                                let cell = self.b.ins().iconst(self.pointer, cells[*guard]);
                                let validated =
                                    self.b
                                        .ins()
                                        .load(types::I64, MemFlagsData::trusted(), cell, 0);
                                let condition = self.b.ins().icmp(IntCC::Equal, epoch, validated);
                                let yes =
                                    self.b.ins().iconst(types::I64, Value::TRUE.bits() as i64);
                                let no =
                                    self.b.ins().iconst(types::I64, Value::FALSE.bits() as i64);
                                let word = self.b.ins().select(condition, yes, no);
                                Some(self.b.as_boxed(word))
                            }
                            Op::Javascript(JsOp::GuardCommit(guard)) => {
                                let epoch = self.epoch();
                                let cell = self.b.ins().iconst(self.pointer, cells[*guard]);
                                self.b.ins().store(MemFlagsData::trusted(), epoch, cell, 0);
                                None
                            }
                            Op::Javascript(JsOp::CallNative(native, args)) => {
                                let args: Vec<_> = args.iter().map(|&v| get(self, v)).collect();
                                let argv = self.stack_values(&args, data.span)?;
                                let argc = self.b.ins().iconst(types::I64, args.len() as i64);
                                let operation =
                                    self.b.ins().iconst(types::I64, native.index() as i64);
                                let receiver = self.constant(Value::UNDEFINED);
                                let value = self.call_helper(
                                    Helper::MathDirect,
                                    &[operation, receiver, argv, argc],
                                );
                                self.check_error();
                                Some(value)
                            }
                            Op::Javascript(
                                JsOp::EnterScope(_)
                                | JsOp::EnterCaptures(..)
                                | JsOp::LeaveScope(_)
                                | JsOp::NextIteration(_)
                                | JsOp::FreshIteration(_),
                            ) => None,
                        };
                        result.map(|value| Operand::new(value, Repr::Boxed))
                    }
                };
                if let (Some(value), Some(result)) = (data.result(), result) {
                    debug_assert_eq!(
                        result.repr,
                        self.b.graph.repr(value),
                        "{inst} produced {:?}",
                        result.repr
                    );
                    let result = result.value;
                    cx.define(value, result);
                    let required = cx.plan.results[inst] == ResultReference::Move;
                    if required || cx.plan.holder(value) == Holder::Temporary {
                        self.references
                            .bind(self.b, value.index() as u32, result, required, false)
                            .map_err(Error::Compile)?;
                    }
                }
                if handler.is_some() {
                    self.handlers.pop();
                }
            }
            self.release_owned();
            self.source_span = terminator.span;
            self.b.set_span(terminator.span);
            self.source_name = terminator.source_name.unwrap_or(source_name);
            self.b.set_source_name(Some(self.source_name));
            let location = if self.semantic_locations {
                cranelift_codegen::ir::SourceLoc::new(cx.terminator_location(block))
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
                    .map(|&value| BlockArg::Value(this.operand(&cx, value).value))
                    .collect();
                (target.block, args)
            };
            match &terminator.kind {
                TerminatorKind::Bailout { .. } => {
                    return Err(Error::Compile("unresolved specialization bailout".into()));
                }
                TerminatorKind::Aarch64Branch(..) => unreachable!("target branch before selection"),
                TerminatorKind::CheckException(to, _) | TerminatorKind::Jump(to) => {
                    let (block, args) = target(self, to);
                    self.jump(block, &args);
                }
                TerminatorKind::Branch(value, yes, no) => {
                    let condition = self.operand(&cx, *value);
                    let condition = condition.value;
                    let (yes_block, yes_args) = target(self, yes);
                    let (no_block, no_args) = target(self, no);
                    self.brif(condition, yes_block, &yes_args, no_block, &no_args);
                }
                TerminatorKind::ReturnValues(..) | TerminatorKind::TailCall { .. } => {
                    return Err(Error::Compile(
                        "low-level terminator reached runtime lowering finalization".into(),
                    ));
                }
                TerminatorKind::Yield(..) => {
                    return Err(Error::Compile("suspension reached native emission".into()));
                }
                TerminatorKind::Fallback(previous) => {
                    let mut args = vec![self.ctx, closure, self.receiver];
                    match &self.arguments {
                        super::Arguments::Fixed(values) => args.extend(values),
                        super::Arguments::Vector { argv, argc } => args.extend([*argv, *argc]),
                    }
                    self.exit_frame();
                    let signature = self.b.import_signature(
                        self.b.graph.low_level_signature.as_ref().unwrap().clone(),
                    );
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
                    debug_assert_eq!(operand.repr, Repr::Boxed, "{block} returns unboxed {value}");
                    if !cx.facts.of(*value).within(UNMANAGED) {
                        self.increment(operand.value);
                    }
                    let args = self.return_args(operand.value, false);
                    self.jump(self.return_block, &args);
                }
            }
            if cold {
                self.b.set_cold_block(block);
                for index in created..self.b.num_blocks() {
                    self.b
                        .set_cold_block(crate::jit::semantic::Block::from_u32(index as u32));
                }
            }
        }
        if self.semantic_locations {
            self.b
                .set_srcloc(cranelift_codegen::ir::SourceLoc::default());
        }
        self.finish_failed_exit();
        self.switch(self.return_block);
        let result = self.b.block_params(self.return_block)[0];
        for address in pack_cleanup {
            self.call_helper(Helper::PackRelease, &[address]);
        }
        self.return_value(result);
        Ok(())
    }
}

impl FunctionCompiler<'_, '_> {
    fn adopt(&mut self, cx: &Context<'_>, inst: Inst) {
        let direct = cx.direct.borrow();
        for operand in self.b.graph.insts[inst].op.operands_mut() {
            *operand = direct[*operand].expect("semantic value used before definition");
        }
        drop(direct);
        self.b.adopt(inst);
    }

    fn operand(&mut self, cx: &Context<'_>, value: GraphValue) -> Operand {
        let clif = cx.direct.borrow()[value].expect("semantic value used before definition");
        Operand::new(clif, self.b.graph.repr(clif))
    }

    fn regex_search(
        &mut self,
        subject: LoweredValue,
        pattern: crate::regex::PatternRef,
        mode: regex::Mode,
    ) -> LoweredValue {
        let word = self.b.as_word(subject);
        // Speculative hoisting can run the search ahead of its string guard, so non-strings read
        // as an empty slot.
        let address = layout::string_view(self.b, word);
        let flags = MemFlagsData::trusted();
        let bytes = self
            .b
            .ins()
            .load(self.pointer, flags, address, crate::heap::DATA_OFFSET);
        let length = self
            .b
            .ins()
            .load(types::I64, flags, address, crate::heap::LENGTH_OFFSET);
        let from = self.b.ins().iconst(types::I64, 0);
        let program = pattern
            .get()
            .program()
            .expect("regex expansion requires a compiled program");
        regex::emit(self.b, &program.dfa, mode, bytes, length, from)
    }

    fn typed_op(
        &mut self,
        cx: &Context<'_>,
        inst: Inst,
        data: &InstData,
        target: Option<Repr>,
    ) -> Result<Option<Operand>, Error> {
        let boolean = |condition: LoweredValue| Some(Operand::new(condition, Repr::Boolean));
        let result = match &data.op {
            Op::Javascript(JsOp::GlobalCache(name)) => {
                let address = self.function_data(crate::jit::helpers::GlobalCache::new(name));
                Some(Operand::new(address, Repr::Word))
            }
            Op::Javascript(JsOp::PropertyCache(site, name)) => {
                let observer = self.property_observer(data);
                let address = self.property_cache(*site, name, observer);
                Some(Operand::new(address, Repr::Word))
            }
            Op::HelperCall {
                helper,
                args,
                produces,
                effects,
                ..
            } => {
                let builtins = if *helper == Helper::SemanticPropertyProbeCached {
                    match self.b.graph.defining_op(args[1]) {
                        Some(Op::Javascript(JsOp::PropertyCache(_, name))) => {
                            crate::builtins::builtin_method(name)
                        }
                        _ => unreachable!("property probe requires a cache"),
                    }
                } else {
                    false
                };
                let args: Vec<_> = args
                    .iter()
                    .map(|&value| self.operand(cx, value).value)
                    .collect();
                if matches!(
                    helper,
                    Helper::GlobalGetBorrowed
                        | Helper::GlobalGetOptionalBorrowed
                        | Helper::SemanticGlobalProbe
                ) {
                    let value = self.cached_global(args[0], *helper);
                    return Ok(produces.map(|repr| Operand::new(value, repr)));
                }
                if *helper == Helper::SemanticPropertyProbeCached {
                    let value = self.cached_property_probe(args[0], args[1], builtins);
                    return Ok(produces.map(|repr| Operand::new(value, repr)));
                }
                let track = helper.result_ownership() == crate::jit::semantic::Ownership::Owned
                    && cx.plan.results[inst] != ResultReference::Slot;
                let value = self.call_helper_tracked(*helper, &args, track);
                if effects.control.throws {
                    self.check_error();
                }
                produces.map(|repr| Operand::new(value, repr))
            }
            Op::Javascript(JsOp::Truthy(value)) => {
                let types = cx.facts.of(*value);
                let value = self.operand(cx, *value).value;
                boolean(self.truthy(value, types))
            }
            Op::Javascript(JsOp::TypedData(..)) => {
                unreachable!("data pointer must be lowered before emission")
            }
            Op::Memory(MemoryOp::Address {
                base,
                index,
                format,
            }) => {
                let base = self.operand(cx, *base).value;
                let index = self.operand(cx, *index).value;
                let offset = self.b.ins().imul_imm_s(index, format.bytes() as i64);
                let address = self.b.ins().iadd(base, offset);
                Some(Operand::new(address, Repr::Word))
            }
            Op::Memory(MemoryOp::Load { address, .. }) => {
                let address = self.operand(cx, *address).value;
                let repr = target.unwrap();
                let value = self
                    .b
                    .ins()
                    .load(repr.machine(), MemFlagsData::trusted(), address, 0);
                Some(Operand::new(value, repr))
            }
            Op::Memory(MemoryOp::IndexedLoad {
                base,
                index,
                format,
            }) => {
                let base = self.operand(cx, *base).value;
                let index = self.operand(cx, *index).value;
                let repr = target.unwrap();
                let value = self.b.indexed_load(base, index, *format, repr.machine());
                Some(Operand::new(value, repr))
            }
            Op::Memory(MemoryOp::IndexedStore {
                base,
                index,
                value,
                format,
            }) => {
                let base = self.operand(cx, *base).value;
                let index = self.operand(cx, *index).value;
                let value = self.operand(cx, *value).value;
                self.b.indexed_store(base, index, value, *format);
                None
            }
            Op::Javascript(JsOp::MemoryNumber(value, format)) => {
                let raw = self.operand(cx, *value).value;
                let (value, repr) = match format {
                    MemoryFormat::U8 | MemoryFormat::U16 => {
                        (self.b.ins().uextend(types::I32, raw), Repr::I32)
                    }
                    MemoryFormat::I8 | MemoryFormat::I16 => {
                        (self.b.ins().sextend(types::I32, raw), Repr::I32)
                    }
                    MemoryFormat::I32 => (raw, Repr::I32),
                    MemoryFormat::U32 => (self.b.ins().fcvt_from_uint(types::F64, raw), Repr::F64),
                    MemoryFormat::F32 => (self.b.ins().fpromote(types::F64, raw), Repr::F64),
                    MemoryFormat::F64 => (raw, Repr::F64),
                };
                Some(Operand::new(value, repr))
            }
            Op::Javascript(JsOp::IsArrayIndex(..)) => unreachable!(),
            Op::Javascript(JsOp::ArrayKind(..) | JsOp::IndexedLength(..) | JsOp::ArrayLoad(..)) => {
                unreachable!("array storage must be lowered before emission")
            }
            Op::Regex(regex::Op::IsPattern(value, pattern)) => {
                let object = self.operand(cx, *value).value;
                let view = self.view(object);
                let data = self.b.ins().load(
                    self.pointer,
                    MemFlagsData::trusted(),
                    view,
                    crate::heap::DATA_OFFSET,
                );
                boolean(
                    self.b
                        .ins()
                        .icmp_imm_s(IntCC::Equal, data, pattern.address() as i64),
                )
            }
            Op::Regex(regex::Op::Test(subject, pattern)) => {
                let subject = self.operand(cx, *subject).value;
                boolean(self.regex_search(subject, *pattern, regex::Mode::Test))
            }
            Op::Regex(regex::Op::Exec(regexp, subject, pattern)) => {
                let regexp = self.operand(cx, *regexp).value;
                let subject = self.operand(cx, *subject).value;
                let found = self.regex_search(subject, *pattern, regex::Mode::Search);
                let missed =
                    self.b
                        .ins()
                        .icmp_imm_s(IntCC::Equal, found, crate::regex::NO_MATCH as i64);
                let matched = self.b.create_block();
                let done = self.b.create_block();
                self.b.append_block_param_repr(done, Repr::Boxed);
                let null = self.constant(Value::NULL);
                self.brif(missed, done, &[null.into()], matched, &[]);
                self.switch(matched);
                let shift = self.b.ins().iconst(types::I64, 32);
                let start = self.b.ins().binary(
                    crate::jit::semantic::low_level::Binary::ShiftRightUnsigned,
                    found,
                    shift,
                );
                let mask = self.b.ins().iconst(types::I64, u32::MAX as i64);
                let end =
                    self.b
                        .ins()
                        .binary(crate::jit::semantic::low_level::Binary::And, found, mask);
                let result =
                    self.call_helper(Helper::RegExpExecResult, &[regexp, subject, start, end]);
                self.check_error();
                self.jump(done, &[result.into()]);
                self.switch(done);
                Some(Operand::new(self.b.block_params(done)[0], Repr::Boxed))
            }
            Op::Regex(regex::Op::Class(..)) => unreachable!(),
            Op::Javascript(JsOp::IsArray(object, layout)) => {
                let view = match self.b.graph.views.get(&inst).copied() {
                    Some(view) => self.operand(cx, view).value,
                    None => {
                        let object = self.operand(cx, *object).value;
                        self.view(object)
                    }
                };
                let kind = self.view_kind(view);
                let expected = layout.kind();
                boolean(self.b.ins().icmp_imm_s(IntCC::Equal, kind, expected as i64))
            }
            Op::Javascript(JsOp::TypedIndex(
                object,
                key,
                TypedRead {
                    element: Some(element),
                    bounded,
                },
            )) => {
                let view = match self.b.graph.views.get(&inst).copied() {
                    Some(view) => self.operand(cx, view).value,
                    None => {
                        let object = self.operand(cx, *object).value;
                        self.view(object)
                    }
                };
                let key = self.operand(cx, *key);
                Some(if *bounded {
                    self.bounded_element_read(view, key, *element)
                } else {
                    self.checked_element_read(view, key, *element)
                })
            }
            _ => {
                return Err(Error::Compile(format!(
                    "{inst}: unsupported selected lowering"
                )));
            }
        };
        if result.as_ref().map(|operand| operand.repr) != target {
            return Err(Error::Compile(format!(
                "{inst}: selected lowering produced the wrong representation"
            )));
        }
        Ok(result)
    }
}

fn deferred(op: &Op) -> bool {
    match op {
        Op::Javascript(JsOp::Constant(constant)) => !matches!(constant, Constant::String(_)),
        Op::LowLevel(_)
        | Op::Convert(..)
        | Op::Box(_)
        | Op::AsBoxed(_)
        | Op::AsWord(_)
        | Op::Unbox(..)
        | Op::Javascript(
            JsOp::NumericFloat(_)
            | JsOp::SaturatingInteger(_)
            | JsOp::AsInteger(_)
            | JsOp::ArrayPosition(_)
            | JsOp::ToInt32(_)
            | JsOp::AsNumber(_),
        ) => true,
        _ => false,
    }
}

/// Blocks on a cycle of the control flow graph, which may run more than once per call.
fn cyclic_blocks(
    graph: &Graph,
    cfg: &crate::jit::semantic::Cfg,
) -> FxHashSet<crate::jit::semantic::Block> {
    let mut finished = Vec::new();
    let mut seen = FxHashSet::default();
    for &root in &cfg.rpo {
        if !seen.insert(root) {
            continue;
        }
        let mut stack = vec![(root, graph.successors(root))];
        while let Some((block, successors)) = stack.last_mut() {
            if let Some(successor) = successors.next() {
                if seen.insert(successor) {
                    stack.push((successor, graph.successors(successor)));
                }
            } else {
                finished.push(*block);
                stack.pop();
            }
        }
    }
    let mut cyclic = FxHashSet::default();
    let mut assigned = FxHashSet::default();
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
        if component.len() > 1 || graph.successors(root).any(|block| block == root) {
            cyclic.extend(component);
        }
    }
    cyclic
}

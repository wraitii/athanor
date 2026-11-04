use std::cell::RefCell;

use cranelift_codegen::entity::{EntityRef, SecondaryMap};
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{BlockArg, InstBuilder, MemFlags, Value as ClifValue, types};
use cranelift_frontend::Variable;

use crate::error::{Error, compile_at};
use crate::heap::kind;
use crate::jit::helpers::Helper;
use crate::jit::semantic::{
    Constant, Facts, Graph, Op, ScopeKind, TerminatorKind, Types, UnaryOp, Value as GraphValue,
};
use crate::value::Value;

use super::FunctionCompiler;
use super::ops::{self, BinaryOp, UnaryOp as EmitUnaryOp};
use super::repr::{Operand, Repr};

const HEAP: Types = Types::STRING.union(Types::OBJECT).union(Types::OTHER);
const NUMERIC: Types = Types::NUMBER.union(Types::UNDEFINED);

struct Context<'a> {
    values: &'a [Variable],
    direct: Option<RefCell<SecondaryMap<GraphValue, Option<ClifValue>>>>,
    reprs: &'a SecondaryMap<GraphValue, Repr>,
    facts: &'a Facts,
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
    fn epoch(&mut self) -> ClifValue {
        let address = self.b.ins().load(
            self.pointer,
            MemFlags::trusted(),
            self.ctx,
            crate::jit::session::EPOCH_OFFSET,
        );
        self.b
            .ins()
            .load(types::I64, MemFlags::trusted(), address, 0)
    }

    pub(super) fn emit_semantic_graph(
        &mut self,
        graph: &Graph,
        closure: ClifValue,
    ) -> Result<(), Error> {
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
        for &variable in &bindings {
            self.b.def_var(variable, undefined);
        }
        if suspends {
            for value in graph.values.keys() {
                let zero = self.zero(reprs[value]);
                self.b.def_var(values[value.index()], zero);
            }
        }
        let cx = Context {
            values: &values,
            direct: (!suspends).then(|| RefCell::new(SecondaryMap::new())),
            reprs: &reprs,
            facts: &facts,
        };
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
            let calls = self.annotations.helpers.len();
            let mut retained_any = false;
            for (&value, &param) in data.params.iter().zip(&params) {
                if !rooted(value) {
                    continue;
                }
                let uninitialized = self.b.ins().icmp_imm(
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
                self.call_helper(Helper::TempRetain, &[retained]);
                self.check_error();
                self.jump(done, &[]);
                self.switch(done);
                retained_any = true;
            }
            for (&value, param) in data.params.iter().zip(params) {
                if rooted(value) {
                    self.initialize_local(values[value.index()], param);
                } else if suspends {
                    self.b.def_var(values[value.index()], param);
                }
                cx.define(value, param);
            }
            if retained_any {
                self.clear_temporaries();
            }
            let calls = if retained_any {
                self.annotations.helpers.len()
            } else {
                calls
            };
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
                self.b.set_srcloc(cranelift_codegen::ir::SourceLoc::new(
                    if self.annotations.semantic_locations {
                        graph.inst_location(inst)
                    } else {
                        data.span.start
                    },
                ));
                let get = |this: &mut Self, value: GraphValue| {
                    let operand = this.operand(&cx, value);
                    this.convert(operand, Repr::Boxed)
                };
                let typed = self.typed_op(&cx, graph, inst)?;
                let result = if let Some(typed) = typed {
                    typed
                } else {
                    let result = match &data.op {
                    Op::Argument(index) => Some(self.load_arg(*index)),
                    Op::Arguments => {
                        let value = self.call_helper(Helper::ArrayNew, &[self.argv, self.argc]);
                        self.check_error();
                        Some(value)
                    }
                    Op::Rest(index) => {
                        let index = self.b.ins().iconst(types::I64, *index as i64);
                        let value = self.call_helper(Helper::Rest, &[self.argv, self.argc, index]);
                        self.check_error();
                        Some(value)
                    }
                    Op::This => Some(self.receiver),
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
                    Op::Closure(span, arrow, generator, captures) => {
                        let (id, _, lexical_this) = self
                            .closures
                            .get(&span.start)
                            .cloned()
                            .ok_or_else(|| compile_at(*span, "missing closure code"))?;
                        let cells: Vec<_> = captures
                            .iter()
                            .map(|binding| self.b.use_var(bindings[binding.index()]))
                            .collect();
                        let argv = self.stack_values(&cells, *span)?;
                        let count = self.b.ins().iconst(types::I64, cells.len() as i64);
                        let code = self.b.ins().iconst(types::I64, id.as_u32() as i64);
                        let result = self.call_helper(Helper::ClosureNew, &[code, argv, count]);
                        self.check_error();
                        if *arrow {
                            let receiver = if lexical_this {
                                self.receiver
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
                    Op::GlobalRead(name) => Some(self.global_access(name, None)),
                    Op::GlobalReadOptional(name) => Some(self.global_get_optional(name)),
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
                    Op::Read(binding) => {
                        let value = self.b.use_var(bindings[binding.index()]);
                        let helper = if graph.bindings[*binding].captured {
                            Helper::CellGet
                        } else {
                            Helper::TempRetain
                        };
                        let value = self.call_helper(helper, &[value]);
                        self.check_error();
                        self.check_error();
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
                        let value = self.call_helper(Helper::TempRetain, &[value]);
                        self.check_error();
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
                        let condition = self.b.ins().icmp_imm(
                            cranelift_codegen::ir::condcodes::IntCC::Equal,
                            value,
                            Value::UNINITIALIZED.bits() as i64,
                        );
                        let yes = self.constant(Value::TRUE);
                        let no = self.constant(Value::FALSE);
                        Some(self.b.ins().select(condition, yes, no))
                    }
                    Op::NewObject => {
                        let dummy = self.constant(Value::UNDEFINED);
                        let value = self.call_helper(Helper::ObjectNew, &[dummy]);
                        self.check_error();
                        Some(value)
                    }
                    Op::NewArray(args) => {
                        let args: Vec<_> = args.iter().map(|&v| get(self, v)).collect();
                        let ptr = self.stack_values(&args, data.span)?;
                        let len = self.b.ins().iconst(types::I64, args.len() as i64);
                        let value = self.call_helper(Helper::ArrayNew, &[ptr, len]);
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
                        Some(self.property_access(object, name, None))
                    }
                    Op::GetIndex(object, index)
                    | Op::TypedIndex(object, index)
                    | Op::IndexedGet(object, index) => {
                        let object = get(self, *object);
                        let index = get(self, *index);
                        Some(self.element_read(object, Operand::new(index, Repr::Boxed)))
                    }
                    Op::SetIndex(object, index, value) => {
                        let object = get(self, *object);
                        let index = get(self, *index);
                        let value = get(self, *value);
                        self.index_access(object, index, Some(value));
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
                    Op::IsTyped(_) => unreachable!(),
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
                        Some(self.b.ins().bitcast(types::I64, MemFlags::new(), length))
                    }
                    Op::AsNumber(value) => Some(get(self, *value)),
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
                        let value = self.call_helper(Helper::CallSpread, &[callee, receiver, args]);
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
                    } => {
                        let callee = get(self, *callee);
                        let receiver = receiver
                            .map(|v| get(self, v))
                            .unwrap_or_else(|| self.constant(Value::UNDEFINED));
                        let args: Vec<_> = args.iter().map(|&v| get(self, v)).collect();
                        let argv = self.stack_values(&args, data.span)?;
                        let argc = self.b.ins().iconst(types::I64, args.len() as i64);
                        let value = self.call_helper(Helper::Call, &[callee, receiver, argv, argc]);
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
                        let name: Box<str> = name.as_str().into();
                        let ptr = self.b.ins().iconst(self.pointer, name.as_ptr() as i64);
                        let len = self.b.ins().iconst(types::I64, name.len() as i64);
                        self.names.push(name);
                        Some(self.call_helper(Helper::SemanticPropertyProbe, &[object, ptr, len]))
                    }
                    Op::ProbeIndex(object, key) => {
                        let object = get(self, *object);
                        let key = get(self, *key);
                        Some(self.call_helper(Helper::SemanticIndexProbe, &[object, key]))
                    }
                    Op::IsNative(value, native) => {
                        let value = get(self, *value);
                        let operation = self.b.ins().iconst(types::I64, native.index() as i64);
                        Some(self.call_helper(Helper::SemanticMathNative, &[value, operation]))
                    }
                    Op::GuardValid(guard) => {
                        let epoch = self.epoch();
                        let cell = self.b.ins().iconst(self.pointer, cells[*guard]);
                        let validated = self.b.ins().load(types::I64, MemFlags::trusted(), cell, 0);
                        let condition = self.b.ins().icmp(IntCC::Equal, epoch, validated);
                        let yes = self.constant(Value::TRUE);
                        let no = self.constant(Value::FALSE);
                        Some(self.b.ins().select(condition, yes, no))
                    }
                    Op::GuardCommit(guard) => {
                        let epoch = self.epoch();
                        let cell = self.b.ins().iconst(self.pointer, cells[*guard]);
                        self.b.ins().store(MemFlags::trusted(), epoch, cell, 0);
                        None
                    }
                    Op::CallNative(native, args) => {
                        let args: Vec<_> = args.iter().map(|&v| get(self, v)).collect();
                        let argv = self.stack_values(&args, data.span)?;
                        let function = crate::jit::math::METHODS[native.index()].1;
                        Some(self.native_math(function, &args, |this| {
                            let argc = this.b.ins().iconst(types::I64, args.len() as i64);
                            let operation = this.b.ins().iconst(types::I64, native.index() as i64);
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
                        let condition = self.b.ins().icmp_imm(
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
                    Op::EnterScope(scope) => {
                        for (index, &binding) in graph.scopes[*scope].bindings.iter().enumerate() {
                            let value = if graph.scopes[*scope].kind == ScopeKind::Captures {
                                let index = self.b.ins().iconst(types::I64, index as i64);
                                let cell = self.call_helper(Helper::CaptureGet, &[closure, index]);
                                self.check_error();
                                cell
                            } else if graph.bindings[binding].captured {
                                let initial = self.constant(Value::UNINITIALIZED);
                                let cell = self.call_helper(Helper::CellNew, &[initial]);
                                self.check_error();
                                cell
                            } else {
                                self.constant(Value::UNINITIALIZED)
                            };
                            self.initialize_local(bindings[binding.index()], value);
                        }
                        None
                    }
                    Op::LeaveScope(scope) => {
                        for &binding in &graph.scopes[*scope].bindings {
                            let value = self.constant(Value::UNDEFINED);
                            self.initialize_local(bindings[binding.index()], value);
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
                                let cell = self.b.use_var(bindings[binding.index()]);
                                let value = self.call_helper(Helper::CellGet, &[cell]);
                                self.check_error();
                                value
                            };
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
                    let result = self.convert(result, reprs[value]);
                    let borrowed = matches!(data.op, Op::This | Op::Argument(_)) && !suspends;
                    if rooted(value) && !borrowed {
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
            self.b.set_srcloc(cranelift_codegen::ir::SourceLoc::new(
                if self.annotations.semantic_locations {
                    graph.terminator_location(block)
                } else {
                    terminator.span.start
                },
            ));
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
                TerminatorKind::Return(value) => {
                    let operand = self.operand(&cx, *value);
                    let value = self.convert(operand, Repr::Boxed);
                    self.jump(self.return_block, &[BlockArg::Value(value)]);
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
        self.switch(self.return_block);
        let result = self.b.block_params(self.return_block)[0];
        self.return_value(result);
        Ok(())
    }
}

impl FunctionCompiler<'_, '_> {
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
            Op::Binary(operator, left, right) => {
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
                    Operand::new(self.b.ins().iadd_imm(integer, delta), Repr::Int32)
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
                let value = self.b.ins().icmp_imm(IntCC::Equal, truth, 0);
                boolean(value)
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
                        Op::IsUndefined(_) => self.b.ins().icmp_imm(
                            IntCC::Equal,
                            operand.value,
                            Value::UNDEFINED.bits() as i64,
                        ),
                        _ => self.b.ins().icmp_imm(
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
            Op::IsTyped(object) => {
                let operand = self.operand(cx, *object);
                let object = self.convert(operand, Repr::Boxed);
                let view = self.view(object);
                boolean(self.has_kind(view, kind::TYPED))
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
            Op::GetIndex(object, key)
            | Op::TypedIndex(object, key)
            | Op::IndexedGet(object, key) => {
                let fallback = !matches!(data.op, Op::TypedIndex(..));
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
                let result = self.viewed_index_access(view, object, key, None, fallback);
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
                self.viewed_index_access(view, object, key, Some(value), true);
                Some(None)
            }
            _ => None,
        })
    }
}

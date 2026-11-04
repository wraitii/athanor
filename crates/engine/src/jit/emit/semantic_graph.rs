use cranelift_codegen::entity::EntityRef;
use cranelift_codegen::ir::{BlockArg, InstBuilder, Value as ClifValue, types};
use cranelift_frontend::Variable;

use crate::error::{Error, compile_at};
use crate::jit::helpers::Helper;
use crate::jit::semantic::{Constant, Facts, Graph, Op, ScopeKind, TerminatorKind, Types, UnaryOp};
use crate::value::Value;

use super::FunctionCompiler;
use super::ops::{self, BinaryOp, UnaryOp as EmitUnaryOp};

const HEAP: Types = Types::STRING.union(Types::OBJECT).union(Types::OTHER);

impl FunctionCompiler<'_, '_> {
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
        let values: Vec<Variable> = graph
            .values
            .values()
            .map(|_| self.b.declare_var(types::I64))
            .collect();
        let bindings: Vec<Variable> = graph
            .bindings
            .values()
            .map(|_| self.b.declare_var(types::I64))
            .collect();
        let facts = Facts::compute(graph);
        let suspends = self.generator.is_some();
        let rooted =
            |value: crate::jit::semantic::Value| suspends || facts.of(value).intersects(HEAP);
        let reachable = crate::jit::semantic::Cfg::new(graph).rpo;
        for &block in &reachable {
            let data = &graph.blocks[block];
            for _ in &data.params {
                self.b.append_block_param(blocks[block.index()], types::I64);
            }
        }
        let undefined = self.constant(Value::UNDEFINED);
        for &variable in values.iter().chain(&bindings) {
            self.b.def_var(variable, undefined);
        }
        self.jump(blocks[graph.entry.index()], &[]);
        for &block in &reachable {
            let data = &graph.blocks[block];
            if self.annotations.semantic_locations {
                self.b
                    .set_srcloc(cranelift_codegen::ir::SourceLoc::default());
            }
            self.switch(blocks[block.index()]);
            let params = self.b.block_params(blocks[block.index()]).to_vec();
            let undefined = self.constant(Value::UNDEFINED);
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
                self.call_helper(Helper::TempRetain, &[retained]);
                self.check_error();
                retained_any = true;
            }
            for (&value, param) in data.params.iter().zip(params) {
                if rooted(value) {
                    self.initialize_local(values[value.index()], param);
                } else {
                    self.b.def_var(values[value.index()], param);
                }
            }
            if retained_any {
                self.clear_temporaries();
            }
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
                let get = |this: &mut Self, value: crate::jit::semantic::Value| {
                    this.b.use_var(values[value.index()])
                };
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
                    Op::Get(object, name) => {
                        let object = get(self, *object);
                        Some(self.property_access(object, name, None))
                    }
                    Op::GetIndex(object, index) => {
                        let object = get(self, *object);
                        let index = get(self, *index);
                        Some(self.index_access(object, index, None))
                    }
                    Op::SetIndex(object, index, value) => {
                        let object = get(self, *object);
                        let index = get(self, *index);
                        let value = get(self, *value);
                        self.index_access(object, index, Some(value));
                        None
                    }
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
                    Op::IsNative(value, native) => {
                        let value = get(self, *value);
                        let operation = self.b.ins().iconst(types::I64, native.index() as i64);
                        Some(self.call_helper(Helper::SemanticMathNative, &[value, operation]))
                    }
                    Op::CallNative(native, args) => {
                        let args: Vec<_> = args.iter().map(|&v| get(self, v)).collect();
                        let argv = self.stack_values(&args, data.span)?;
                        let argc = self.b.ins().iconst(types::I64, args.len() as i64);
                        let operation = self.b.ins().iconst(types::I64, native.index() as i64);
                        let receiver = self.constant(Value::UNDEFINED);
                        let value = self
                            .call_helper(Helper::MathDirect, &[operation, receiver, argv, argc]);
                        self.check_error();
                        Some(value)
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
                if let (Some(value), Some(result)) = (data.result, result) {
                    let borrowed = matches!(data.op, Op::This | Op::Argument(_)) && !suspends;
                    if rooted(value) && !borrowed {
                        self.initialize_local(values[value.index()], result);
                    } else {
                        self.b.def_var(values[value.index()], result);
                    }
                }
                if handler.is_some() {
                    self.handlers.pop();
                }
            }
            if !data.insts.is_empty() {
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
                    .map(|value| BlockArg::Value(this.b.use_var(values[value.index()])))
                    .collect();
                (blocks[target.block.index()], args)
            };
            match &terminator.kind {
                TerminatorKind::CheckException(to, _) | TerminatorKind::Jump(to) => {
                    let (block, args) = target(self, to);
                    self.jump(block, &args);
                }
                TerminatorKind::Branch(value, yes, no) => {
                    let value = self.b.use_var(values[value.index()]);
                    let condition = self.truthy(value);
                    let (yes, yes_args) = target(self, yes);
                    let (no, no_args) = target(self, no);
                    self.brif(condition, yes, &yes_args, no, &no_args);
                }
                TerminatorKind::Yield(span, value, target_) => {
                    let generator = self.generator.unwrap();
                    let value = self.b.use_var(values[value.index()]);
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
                    let value = self.b.use_var(values[value.index()]);
                    self.jump(self.return_block, &[BlockArg::Value(value)]);
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

use cranelift_codegen::entity::{EntityRef, SecondaryMap};
use oxc_syntax::operator::BinaryOperator;

use super::asm::Assembler;
use super::encode::{self as e, Alu, Cond, Mem, Shift, Width};
use crate::jit::helpers::{HELPERS, Helper};
use crate::jit::machine::{MachineCode, SourceRange};
use crate::jit::semantic::Value as SemanticValue;
use crate::jit::semantic::{Constant, Graph, Op, ScopeKind, TerminatorKind, UnaryOp, ValueDef};
use crate::jit::session::JitCtx;
use crate::value::Value;

pub(super) fn compile(
    graph: &Graph,
    source_name: (usize, usize),
    semantic_locations: bool,
    names: &mut Vec<Box<str>>,
    function_data: &mut crate::jit::function_data::FunctionData,
) -> Result<MachineCode, super::Unsupported> {
    let mut reason = String::new();
    emit(
        graph,
        source_name,
        semantic_locations,
        names,
        function_data,
        &mut reason,
    )
    .ok_or_else(|| {
        super::Unsupported(if reason.is_empty() {
            "encoding or graph layout limit".into()
        } else {
            reason
        })
    })
}

fn refuse<T>(reason: &mut String, message: impl Into<String>) -> Option<T> {
    *reason = message.into();
    None
}

fn emit(
    graph: &Graph,
    source_name: (usize, usize),
    semantic_locations: bool,
    names: &mut Vec<Box<str>>,
    function_data: &mut crate::jit::function_data::FunctionData,
    reason: &mut String,
) -> Option<MachineCode> {
    let mut literals = Vec::new();
    let fixed = graph.arity.is_some();
    let arity = usize::from(graph.arity.unwrap_or(0));
    if arity > 32 {
        return refuse(reason, "more than 32 parameters");
    }
    if graph.generator {
        return refuse(reason, "generator");
    }
    if graph.has_handlers {
        return refuse(reason, "exception handlers");
    }
    let values_size = graph.values.len().checked_mul(8)?;
    let temporary_offset = values_size.checked_add(graph.bindings.len().checked_mul(8)?)?;
    let edge_slots = graph
        .blocks
        .values()
        .map(|block| block.params.len())
        .max()
        .unwrap_or(0);
    let context_offset = temporary_offset.checked_add(edge_slots.checked_mul(8)?)?;
    let slot_count = context_offset / 8;
    let receiver_offset = context_offset.checked_add(8)?;
    let closure_offset = receiver_offset.checked_add(8)?;
    let span_offset = closure_offset.checked_add(8)?;
    let argv_offset = span_offset.checked_add(8)?;
    let argc_offset = argv_offset.checked_add(8)?;
    let arguments_offset = argc_offset.checked_add(8)?;
    let call_offset = arguments_offset.checked_add(arity.checked_mul(8)?)?;
    let max_call_args = graph
        .insts
        .values()
        .filter_map(|data| match &data.op {
            Op::Call { args, .. } | Op::Construct { args, .. } | Op::NewArray(args) => {
                Some(args.len())
            }
            Op::Closure(_, _, _, _, _, captures, _) => Some(captures.len()),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    let size = call_offset
        .checked_add(max_call_args.checked_mul(8)?)?
        .checked_add(15)?
        & !15;
    let incoming = arity.saturating_sub(3).checked_mul(8)?.checked_add(15)? & !15;
    if size.checked_add(16)?.checked_add(incoming)? > 32760 {
        return refuse(reason, "stack frame exceeds scaled addressing range");
    }
    let mut asm = Assembler::new(true);
    let failed = asm.label();
    let mut labels = SecondaryMap::new();
    for (block, _) in graph.blocks.iter() {
        labels[block] = Some(asm.label());
    }
    asm.emit(e::STP_FP_LR_PRE);
    if !fixed {
        asm.emit(e::mov(Width::X, 11, 3));
        asm.emit(e::mov(Width::X, 12, 4));
        asm.emit(e::mov(Width::X, 4, 2));
        asm.emit(e::mov(Width::X, 3, 1));
        asm.emit(e::mov(Width::X, 2, 0));
    }
    let stack_ok = asm.label();
    asm.emit(e::mov_sp(9, e::SP));
    e::mov_imm(Width::X, 10, size as u64, |word| asm.emit(word));
    asm.emit(e::alu(Alu::Sub, Width::X, 9, 9, 10));
    asm.emit(e::mem_scaled(
        Mem::LdrX,
        10,
        2,
        crate::jit::session::STACK_LIMIT_OFFSET as u32,
    )?);
    asm.emit(e::alu(Alu::Subs, Width::X, e::ZR, 9, 10));
    asm.branch(Cond::Hs, stack_ok);
    asm.emit(e::mov(Width::X, 0, 2));
    call(&mut asm, Helper::FrameOverflow);
    e::mov_imm(
        Width::X,
        if fixed { 2 } else { 0 },
        Value::UNDEFINED.bits(),
        |word| asm.emit(word),
    );
    e::mov_imm(Width::W, 3, 1, |word| asm.emit(word));
    asm.emit(e::LDP_FP_LR_POST);
    stack_address(&mut asm, e::SP, incoming as u32);
    asm.emit(e::RET);
    asm.bind(stack_ok);
    // Touch each page after checking the full frame against the runtime stack limit.
    let mut remaining = size;
    while remaining != 0 {
        let chunk = remaining.min(4080);
        asm.emit(e::sub_imm(Width::X, e::SP, e::SP, chunk as u32, false));
        asm.emit(e::mem_scaled(Mem::StrX, e::ZR, e::SP, 0)?);
        remaining -= chunk;
    }
    asm.emit(e::mem_scaled(Mem::StrX, 3, e::SP, closure_offset as u32)?);
    asm.emit(e::mem_scaled(Mem::StrX, 2, e::SP, context_offset as u32)?);
    asm.emit(e::mem_scaled(Mem::StrX, 4, e::SP, receiver_offset as u32)?);
    if !fixed {
        asm.emit(e::mem_scaled(Mem::StrX, 11, e::SP, argv_offset as u32)?);
        asm.emit(e::mem_scaled(Mem::StrX, 12, e::SP, argc_offset as u32)?);
    }
    for index in 0..arity {
        let reg = if index < 3 {
            5 + index as u8
        } else {
            let offset = size
                .checked_add(16)?
                .checked_add((index - 3).checked_mul(8)?)?;
            asm.emit(e::mem_scaled(Mem::LdrX, 9, e::SP, offset as u32)?);
            9
        };
        asm.emit(e::mem_scaled(
            Mem::StrX,
            reg,
            e::SP,
            (arguments_offset + index * 8) as u32,
        )?);
    }
    if values_size != 0 || !graph.bindings.is_empty() {
        e::mov_imm(Width::X, 9, Value::UNDEFINED.bits(), |word| asm.emit(word));
        for index in 0..slot_count {
            let offset = (index * 8) as u32;
            asm.emit(e::mem_scaled(Mem::StrX, 9, e::SP, offset)?);
        }
    }
    asm.jump(e::B, labels[graph.entry]?);
    let error_source = function_data.pin(source_name);
    let mut block_starts = Vec::new();
    let mut locations = Vec::new();
    for (block, data) in graph.blocks.iter() {
        block_starts.push(asm.offset());
        asm.bind(labels[block]?);
        for &inst in &data.insts {
            let data = &graph.insts[inst];
            let start = asm.offset();
            let span = u64::from(data.span.start) | (u64::from(data.span.end) << 32);
            e::mov_imm(Width::X, 9, span, |word| asm.emit(word));
            asm.emit(e::mem_scaled(Mem::StrX, 9, e::SP, span_offset as u32)?);
            match &data.op {
                Op::This => {
                    asm.emit(e::mem_scaled(Mem::LdrX, 9, e::SP, receiver_offset as u32)?);
                    store(
                        &mut asm,
                        context_offset as u32,
                        (data.result()?.index() * 8) as u32,
                        9,
                        false,
                        failed,
                    )?;
                }
                Op::RegExp(pattern, flags) => {
                    asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                    e::mov_imm(Width::X, 1, u64::from(*flags), |word| asm.emit(word));
                    literal(&mut asm, &mut literals, pattern, 2, 3);
                    call(&mut asm, Helper::RegExpNew);
                    store(
                        &mut asm,
                        context_offset as u32,
                        (data.result()?.index() * 8) as u32,
                        0,
                        true,
                        failed,
                    )?;
                }
                Op::Closure(_, id, arrow, generator, lexical_this, captures, lexical_receiver) => {
                    for (index, binding) in captures.iter().enumerate() {
                        asm.emit(e::mem_scaled(
                            Mem::LdrX,
                            9,
                            e::SP,
                            (values_size + binding.index() * 8) as u32,
                        )?);
                        asm.emit(e::mem_scaled(
                            Mem::StrX,
                            9,
                            e::SP,
                            (call_offset + index * 8) as u32,
                        )?);
                    }
                    asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                    e::mov_imm(Width::X, 1, u64::from(id.as_u32()), |word| asm.emit(word));
                    stack_address(&mut asm, 2, call_offset as u32);
                    e::mov_imm(Width::X, 3, captures.len() as u64, |word| asm.emit(word));
                    call(&mut asm, Helper::ClosureNew);
                    let dest = (data.result()?.index() * 8) as u32;
                    store(&mut asm, context_offset as u32, dest, 0, true, failed)?;
                    if *arrow || *generator {
                        asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                        asm.emit(e::mem_scaled(Mem::LdrX, 1, e::SP, dest)?);
                        if *arrow {
                            if *lexical_this {
                                let offset = lexical_receiver
                                    .map_or(receiver_offset, |value| value.index() * 8);
                                asm.emit(e::mem_scaled(Mem::LdrX, 2, e::SP, offset as u32)?);
                            } else {
                                e::mov_imm(Width::X, 2, Value::UNDEFINED.bits(), |word| {
                                    asm.emit(word)
                                });
                            }
                            call(&mut asm, Helper::BindThis);
                        } else {
                            call(&mut asm, Helper::DisableConstructor);
                        }
                        check_error(&mut asm, context_offset as u32, failed)?;
                    }
                }
                Op::Initialize(binding, value) | Op::Write(binding, value)
                    if graph.bindings[*binding].captured =>
                {
                    asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                    asm.emit(e::mem_scaled(
                        Mem::LdrX,
                        1,
                        e::SP,
                        (values_size + binding.index() * 8) as u32,
                    )?);
                    asm.emit(e::mem_scaled(
                        Mem::LdrX,
                        2,
                        e::SP,
                        (value.index() * 8) as u32,
                    )?);
                    e::mov_imm(
                        Width::X,
                        3,
                        u64::from(matches!(data.op, Op::Initialize(..))),
                        |word| asm.emit(word),
                    );
                    call_address(&mut asm, write_cell as *const () as u64);
                    check_error(&mut asm, context_offset as u32, failed)?;
                }
                Op::Read(binding) if graph.bindings[*binding].captured => {
                    asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                    asm.emit(e::mem_scaled(
                        Mem::LdrX,
                        1,
                        e::SP,
                        (values_size + binding.index() * 8) as u32,
                    )?);
                    call(&mut asm, Helper::CellGet);
                    store(
                        &mut asm,
                        context_offset as u32,
                        (data.result()?.index() * 8) as u32,
                        0,
                        true,
                        failed,
                    )?;
                }
                Op::MarkClass(value) | Op::DisableConstructor(value) => {
                    asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                    asm.emit(e::mem_scaled(
                        Mem::LdrX,
                        1,
                        e::SP,
                        (value.index() * 8) as u32,
                    )?);
                    call(
                        &mut asm,
                        if matches!(data.op, Op::MarkClass(_)) {
                            Helper::MarkClass
                        } else {
                            Helper::DisableConstructor
                        },
                    );
                    check_error(&mut asm, context_offset as u32, failed)?;
                }
                Op::Throw(value) => {
                    asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                    asm.emit(e::mem_scaled(
                        Mem::LdrX,
                        1,
                        e::SP,
                        (value.index() * 8) as u32,
                    )?);
                    call(&mut asm, Helper::Throw);
                    check_error(&mut asm, context_offset as u32, failed)?;
                }
                Op::NewObject => {
                    asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                    e::mov_imm(Width::X, 1, Value::UNDEFINED.bits(), |word| asm.emit(word));
                    call(&mut asm, Helper::ObjectNew);
                    store(
                        &mut asm,
                        context_offset as u32,
                        (data.result()?.index() * 8) as u32,
                        0,
                        true,
                        failed,
                    )?;
                }
                Op::NewArray(args) => {
                    for (index, value) in args.iter().enumerate() {
                        asm.emit(e::mem_scaled(
                            Mem::LdrX,
                            9,
                            e::SP,
                            (value.index() * 8) as u32,
                        )?);
                        asm.emit(e::mem_scaled(
                            Mem::StrX,
                            9,
                            e::SP,
                            (call_offset + index * 8) as u32,
                        )?);
                    }
                    asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                    stack_address(&mut asm, 1, call_offset as u32);
                    e::mov_imm(Width::X, 2, args.len() as u64, |word| asm.emit(word));
                    call(&mut asm, Helper::ArrayNew);
                    store(
                        &mut asm,
                        context_offset as u32,
                        (data.result()?.index() * 8) as u32,
                        0,
                        true,
                        failed,
                    )?;
                }
                Op::IsIndexed(value) | Op::IndexedLength(value) => {
                    asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                    asm.emit(e::mem_scaled(
                        Mem::LdrX,
                        1,
                        e::SP,
                        (value.index() * 8) as u32,
                    )?);
                    e::mov_imm(
                        Width::X,
                        2,
                        u64::from(matches!(data.op, Op::IndexedLength(_))),
                        |word| asm.emit(word),
                    );
                    call_address(&mut asm, indexed as *const () as u64);
                    store(
                        &mut asm,
                        context_offset as u32,
                        (data.result()?.index() * 8) as u32,
                        0,
                        true,
                        failed,
                    )?;
                }
                Op::IteratorNew(value)
                | Op::IteratorNext(value)
                | Op::EnumerateNew(value)
                | Op::EnumerateNext(value) => {
                    asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                    asm.emit(e::mem_scaled(
                        Mem::LdrX,
                        1,
                        e::SP,
                        (value.index() * 8) as u32,
                    )?);
                    let helper = match data.op {
                        Op::IteratorNew(_) => Helper::IterateNew,
                        Op::IteratorNext(_) => Helper::IterateNext,
                        Op::EnumerateNew(_) => Helper::EnumerateNew,
                        Op::EnumerateNext(_) => Helper::EnumerateNext,
                        _ => unreachable!(),
                    };
                    call(&mut asm, helper);
                    store(
                        &mut asm,
                        context_offset as u32,
                        (data.result()?.index() * 8) as u32,
                        0,
                        true,
                        failed,
                    )?;
                }
                Op::Construct { callee, args } => {
                    for (index, value) in args.iter().enumerate() {
                        asm.emit(e::mem_scaled(
                            Mem::LdrX,
                            9,
                            e::SP,
                            (value.index() * 8) as u32,
                        )?);
                        asm.emit(e::mem_scaled(
                            Mem::StrX,
                            9,
                            e::SP,
                            (call_offset + index * 8) as u32,
                        )?);
                    }
                    asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                    asm.emit(e::mem_scaled(
                        Mem::LdrX,
                        1,
                        e::SP,
                        (callee.index() * 8) as u32,
                    )?);
                    stack_address(&mut asm, 2, call_offset as u32);
                    e::mov_imm(Width::X, 3, args.len() as u64, |word| asm.emit(word));
                    call(&mut asm, Helper::Construct);
                    store(
                        &mut asm,
                        context_offset as u32,
                        (data.result()?.index() * 8) as u32,
                        0,
                        true,
                        failed,
                    )?;
                }
                Op::Call {
                    callee,
                    receiver,
                    args,
                    ..
                } => {
                    for (index, value) in args.iter().enumerate() {
                        asm.emit(e::mem_scaled(
                            Mem::LdrX,
                            9,
                            e::SP,
                            (value.index() * 8) as u32,
                        )?);
                        asm.emit(e::mem_scaled(
                            Mem::StrX,
                            9,
                            e::SP,
                            (call_offset + index * 8) as u32,
                        )?);
                    }
                    asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                    asm.emit(e::mem_scaled(
                        Mem::LdrX,
                        1,
                        e::SP,
                        (callee.index() * 8) as u32,
                    )?);
                    if let Some(receiver) = receiver {
                        asm.emit(e::mem_scaled(
                            Mem::LdrX,
                            2,
                            e::SP,
                            (receiver.index() * 8) as u32,
                        )?);
                    } else {
                        e::mov_imm(Width::X, 2, Value::UNDEFINED.bits(), |word| asm.emit(word));
                    }
                    stack_address(&mut asm, 3, call_offset as u32);
                    e::mov_imm(Width::X, 4, args.len() as u64, |word| asm.emit(word));
                    call(&mut asm, Helper::Call);
                    store(
                        &mut asm,
                        context_offset as u32,
                        (data.result()?.index() * 8) as u32,
                        0,
                        true,
                        failed,
                    )?;
                }
                Op::GlobalWrite(name, value) | Op::GlobalInitialize(name, value) => {
                    asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                    literal(&mut asm, &mut literals, name, 1, 2);
                    asm.emit(e::mem_scaled(
                        Mem::LdrX,
                        3,
                        e::SP,
                        (value.index() * 8) as u32,
                    )?);
                    call(
                        &mut asm,
                        if matches!(data.op, Op::GlobalInitialize(..)) {
                            Helper::GlobalInitialize
                        } else {
                            Helper::GlobalSet
                        },
                    );
                    check_error(&mut asm, context_offset as u32, failed)?;
                }
                Op::GlobalRead(name) | Op::GlobalReadOptional(name) => {
                    asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                    literal(&mut asm, &mut literals, name, 1, 2);
                    e::mov_imm(
                        Width::X,
                        3,
                        u64::from(matches!(data.op, Op::GlobalReadOptional(_))),
                        |word| asm.emit(word),
                    );
                    call_address(&mut asm, global as *const () as u64);
                    store(
                        &mut asm,
                        context_offset as u32,
                        (data.result()?.index() * 8) as u32,
                        0,
                        true,
                        failed,
                    )?;
                }
                Op::IsNullish(value) | Op::IsUndefined(value) | Op::IsUninitialized(value) => {
                    asm.emit(e::mem_scaled(
                        Mem::LdrX,
                        0,
                        e::SP,
                        (value.index() * 8) as u32,
                    )?);
                    e::mov_imm(
                        Width::X,
                        1,
                        match data.op {
                            Op::IsNullish(_) => 1,
                            Op::IsUninitialized(_) => 2,
                            _ => 0,
                        },
                        |word| asm.emit(word),
                    );
                    call_address(&mut asm, nullish as *const () as u64);
                    store(
                        &mut asm,
                        context_offset as u32,
                        (data.result()?.index() * 8) as u32,
                        0,
                        false,
                        failed,
                    )?;
                }
                Op::Unary(
                    UnaryOp::Not
                    | UnaryOp::BitwiseNot
                    | UnaryOp::Negate
                    | UnaryOp::Plus
                    | UnaryOp::Typeof,
                    value,
                ) => {
                    asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                    asm.emit(e::mem_scaled(
                        Mem::LdrX,
                        1,
                        e::SP,
                        (value.index() * 8) as u32,
                    )?);
                    let helper = match data.op {
                        Op::Unary(UnaryOp::Not, _) => Helper::LogicalNot,
                        Op::Unary(UnaryOp::BitwiseNot, _) => Helper::BitwiseNot,
                        Op::Unary(UnaryOp::Negate, _) => Helper::UnaryNegation,
                        Op::Unary(UnaryOp::Typeof, _) => Helper::Typeof,
                        _ => Helper::UnaryPlus,
                    };
                    call(&mut asm, helper);
                    store(
                        &mut asm,
                        context_offset as u32,
                        (data.result()?.index() * 8) as u32,
                        0,
                        true,
                        failed,
                    )?;
                }
                Op::Get(object, name) => {
                    asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                    asm.emit(e::mem_scaled(
                        Mem::LdrX,
                        1,
                        e::SP,
                        (object.index() * 8) as u32,
                    )?);
                    let cache = function_data
                        .pin(crate::jit::helpers::PropertyCache::new(name, None, false));
                    e::mov_imm(Width::X, 2, cache as u64, |word| asm.emit(word));
                    call(&mut asm, Helper::PropertyGetBaseline);
                    store(
                        &mut asm,
                        context_offset as u32,
                        (data.result()?.index() * 8) as u32,
                        0,
                        true,
                        failed,
                    )?;
                }
                Op::Set(object, name, value)
                | Op::Define(object, name, value)
                | Op::DefineHidden(object, name, value) => {
                    asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                    asm.emit(e::mem_scaled(
                        Mem::LdrX,
                        1,
                        e::SP,
                        (object.index() * 8) as u32,
                    )?);
                    literal(&mut asm, &mut literals, name, 2, 3);
                    asm.emit(e::mem_scaled(
                        Mem::LdrX,
                        4,
                        e::SP,
                        (value.index() * 8) as u32,
                    )?);
                    call(
                        &mut asm,
                        if matches!(data.op, Op::DefineHidden(..)) {
                            Helper::PropertyDefineHidden
                        } else {
                            Helper::PropertySet
                        },
                    );
                    check_error(&mut asm, context_offset as u32, failed)?;
                }
                Op::Argument(index) | Op::Restored(index) if !fixed => {
                    let missing = asm.label();
                    let loaded = asm.label();
                    asm.emit(e::mem_scaled(Mem::LdrX, 10, e::SP, argc_offset as u32)?);
                    e::mov_imm(Width::X, 9, *index as u64, |word| asm.emit(word));
                    asm.emit(e::alu(Alu::Subs, Width::X, e::ZR, 10, 9));
                    asm.branch(Cond::Ls, missing);
                    asm.emit(e::mem_scaled(Mem::LdrX, 10, e::SP, argv_offset as u32)?);
                    asm.emit(e::mem_scaled(
                        Mem::LdrX,
                        9,
                        10,
                        index.checked_mul(8)? as u32,
                    )?);
                    asm.jump(e::B, loaded);
                    asm.bind(missing);
                    e::mov_imm(Width::X, 9, Value::UNDEFINED.bits(), |word| asm.emit(word));
                    asm.bind(loaded);
                    store(
                        &mut asm,
                        context_offset as u32,
                        (data.result()?.index() * 8) as u32,
                        9,
                        false,
                        failed,
                    )?;
                }
                Op::BindingLoad(binding) => {
                    asm.emit(e::mem_scaled(
                        Mem::LdrX,
                        9,
                        e::SP,
                        (values_size + binding.index() * 8) as u32,
                    )?);
                    store(
                        &mut asm,
                        context_offset as u32,
                        (data.result()?.index() * 8) as u32,
                        9,
                        false,
                        failed,
                    )?;
                }
                Op::BindingRestore(binding, value) => {
                    asm.emit(e::mem_scaled(
                        Mem::LdrX,
                        9,
                        e::SP,
                        (value.index() * 8) as u32,
                    )?);
                    store(
                        &mut asm,
                        context_offset as u32,
                        (values_size + binding.index() * 8) as u32,
                        9,
                        false,
                        failed,
                    )?;
                }
                Op::ContinuationStore(values) if !fixed => {
                    for (index, value) in values.iter().enumerate() {
                        asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                        asm.emit(e::mem_scaled(Mem::LdrX, 1, e::SP, argv_offset as u32)?);
                        e::mov_imm(Width::X, 2, index as u64, |word| asm.emit(word));
                        asm.emit(e::mem_scaled(
                            Mem::LdrX,
                            3,
                            e::SP,
                            (value.index() * 8) as u32,
                        )?);
                        call_address(&mut asm, output as *const () as u64);
                    }
                }
                Op::Argument(index) | Op::Restored(index) if *index < arity => {
                    let source = (arguments_offset + index * 8) as u32;
                    let dest = data.result()?.index().checked_mul(8)? as u32;
                    asm.emit(e::mem_scaled(Mem::LdrX, 9, e::SP, source)?);
                    store(&mut asm, context_offset as u32, dest, 9, false, failed)?;
                }
                Op::Constant(constant) => {
                    if let Constant::String(text) = constant {
                        asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                        literal(&mut asm, &mut literals, text, 1, 2);
                        call_address(&mut asm, string as *const () as u64);
                        store(
                            &mut asm,
                            context_offset as u32,
                            (data.result()?.index() * 8) as u32,
                            0,
                            true,
                            failed,
                        )?;
                        let end = asm.offset();
                        locations.push(SourceRange {
                            start,
                            end,
                            location: Some(if semantic_locations {
                                graph.inst_location(inst)
                            } else {
                                data.span.start
                            }),
                        });
                        continue;
                    }
                    let value = match constant {
                        Constant::Undefined => Value::UNDEFINED,
                        Constant::Null => Value::NULL,
                        Constant::Boolean(value) => Value::boolean(*value),
                        Constant::Number(value) => Value::number(*value),
                        Constant::Uninitialized => Value::UNINITIALIZED,
                        Constant::String(_) => {
                            return refuse(reason, "unsupported constant");
                        }
                    };
                    let offset = data.result()?.index().checked_mul(8)? as u32;
                    e::mov_imm(Width::X, 9, value.bits(), |word| asm.emit(word));
                    store(&mut asm, context_offset as u32, offset, 9, false, failed)?;
                }
                Op::Initialize(binding, value) | Op::Write(binding, value) => {
                    let source = value.index().checked_mul(8)? as u32;
                    let dest = (values_size + binding.index() * 8) as u32;
                    if matches!(data.op, Op::Write(..)) {
                        check_initialized(&mut asm, context_offset as u32, dest, failed)?;
                    }
                    asm.emit(e::mem_scaled(Mem::LdrX, 9, e::SP, source)?);
                    store(&mut asm, context_offset as u32, dest, 9, false, failed)?;
                }
                Op::Read(binding) => {
                    let source = (values_size + binding.index() * 8) as u32;
                    let dest = data.result()?.index().checked_mul(8)? as u32;
                    check_initialized(&mut asm, context_offset as u32, source, failed)?;
                    asm.emit(e::mem_scaled(Mem::LdrX, 9, e::SP, source)?);
                    store(&mut asm, context_offset as u32, dest, 9, false, failed)?;
                }
                Op::Binary(operator, left, right) => {
                    let helper = match operator {
                        BinaryOperator::In => Helper::In,
                        BinaryOperator::Instanceof => Helper::Instanceof,
                        BinaryOperator::Exponential => Helper::Pow,
                        BinaryOperator::Addition => Helper::Add,
                        BinaryOperator::Subtraction => Helper::Sub,
                        BinaryOperator::Multiplication => Helper::Mul,
                        BinaryOperator::Division => Helper::Div,
                        BinaryOperator::Remainder => Helper::Rem,
                        BinaryOperator::LessThan => Helper::Less,
                        BinaryOperator::LessEqualThan => Helper::LessEqual,
                        BinaryOperator::GreaterThan => Helper::Greater,
                        BinaryOperator::GreaterEqualThan => Helper::GreaterEqual,
                        BinaryOperator::Equality => Helper::LooseEqual,
                        BinaryOperator::Inequality => Helper::LooseNotEqual,
                        BinaryOperator::StrictEquality => Helper::StrictEqual,
                        BinaryOperator::StrictInequality => Helper::StrictNotEqual,
                        BinaryOperator::BitwiseAnd => Helper::BitwiseAnd,
                        BinaryOperator::BitwiseOR => Helper::BitwiseOr,
                        BinaryOperator::BitwiseXOR => Helper::BitwiseXor,
                        BinaryOperator::ShiftLeft => Helper::ShiftLeft,
                        BinaryOperator::ShiftRight => Helper::ShiftRight,
                        BinaryOperator::ShiftRightZeroFill => Helper::ShiftRightUnsigned,
                    };
                    asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                    asm.emit(e::mem_scaled(
                        Mem::LdrX,
                        1,
                        e::SP,
                        (left.index() * 8) as u32,
                    )?);
                    asm.emit(e::mem_scaled(
                        Mem::LdrX,
                        2,
                        e::SP,
                        (right.index() * 8) as u32,
                    )?);
                    e::mov_imm(
                        Width::X,
                        16,
                        HELPERS[helper as usize].pointer as u64,
                        |word| asm.emit(word),
                    );
                    asm.emit(e::blr(16));
                    store(
                        &mut asm,
                        context_offset as u32,
                        (data.result()?.index() * 8) as u32,
                        0,
                        *operator == BinaryOperator::Addition,
                        failed,
                    )?;
                }
                Op::ToNumeric(value) => {
                    asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                    asm.emit(e::mem_scaled(
                        Mem::LdrX,
                        1,
                        e::SP,
                        (value.index() * 8) as u32,
                    )?);
                    e::mov_imm(
                        Width::X,
                        16,
                        HELPERS[Helper::UnaryPlus as usize].pointer as u64,
                        |word| asm.emit(word),
                    );
                    asm.emit(e::blr(16));
                    store(
                        &mut asm,
                        context_offset as u32,
                        (data.result()?.index() * 8) as u32,
                        0,
                        false,
                        failed,
                    )?;
                }
                Op::Unary(UnaryOp::Increment | UnaryOp::Decrement, value)
                    if numeric(graph, *value, 0) =>
                {
                    asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                    asm.emit(e::mem_scaled(
                        Mem::LdrX,
                        1,
                        e::SP,
                        (value.index() * 8) as u32,
                    )?);
                    e::mov_imm(Width::X, 2, Value::number(1.0).bits(), |word| {
                        asm.emit(word)
                    });
                    let helper = if matches!(data.op, Op::Unary(UnaryOp::Increment, _)) {
                        Helper::Add
                    } else {
                        Helper::Sub
                    };
                    e::mov_imm(
                        Width::X,
                        16,
                        HELPERS[helper as usize].pointer as u64,
                        |word| asm.emit(word),
                    );
                    asm.emit(e::blr(16));
                    check_error(&mut asm, context_offset as u32, failed)?;
                    store(
                        &mut asm,
                        context_offset as u32,
                        (data.result()?.index() * 8) as u32,
                        0,
                        true,
                        failed,
                    )?;
                }
                Op::GetIndex(array, index) | Op::ArrayGet(array, index) => {
                    asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                    asm.emit(e::mem_scaled(
                        Mem::LdrX,
                        1,
                        e::SP,
                        (array.index() * 8) as u32,
                    )?);
                    asm.emit(e::mem_scaled(
                        Mem::LdrX,
                        2,
                        e::SP,
                        (index.index() * 8) as u32,
                    )?);
                    e::mov_imm(
                        Width::X,
                        16,
                        HELPERS[Helper::IndexGet as usize].pointer as u64,
                        |word| asm.emit(word),
                    );
                    asm.emit(e::blr(16));
                    store(
                        &mut asm,
                        context_offset as u32,
                        (data.result()?.index() * 8) as u32,
                        0,
                        true,
                        failed,
                    )?;
                }
                Op::SetIndex(array, index, value) => {
                    asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                    asm.emit(e::mem_scaled(
                        Mem::LdrX,
                        1,
                        e::SP,
                        (array.index() * 8) as u32,
                    )?);
                    asm.emit(e::mem_scaled(
                        Mem::LdrX,
                        2,
                        e::SP,
                        (index.index() * 8) as u32,
                    )?);
                    asm.emit(e::mem_scaled(
                        Mem::LdrX,
                        3,
                        e::SP,
                        (value.index() * 8) as u32,
                    )?);
                    e::mov_imm(
                        Width::X,
                        16,
                        HELPERS[Helper::IndexSet as usize].pointer as u64,
                        |word| asm.emit(word),
                    );
                    asm.emit(e::blr(16));
                    check_error(&mut asm, context_offset as u32, failed)?;
                }
                Op::EnterScope(scope) | Op::EnterCaptures(scope, _) => {
                    for (index, binding) in graph.scopes[*scope].bindings.iter().enumerate() {
                        let dest = (values_size + binding.index() * 8) as u32;
                        if graph.scopes[*scope].kind == ScopeKind::Captures {
                            let source = match data.op {
                                Op::EnterCaptures(_, value) => value.index() * 8,
                                _ => closure_offset,
                            };
                            asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                            asm.emit(e::mem_scaled(Mem::LdrX, 1, e::SP, source as u32)?);
                            e::mov_imm(Width::X, 2, index as u64, |word| asm.emit(word));
                            call(&mut asm, Helper::CaptureGet);
                            store(&mut asm, context_offset as u32, dest, 0, true, failed)?;
                        } else if graph.bindings[*binding].captured {
                            asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                            e::mov_imm(Width::X, 1, Value::UNINITIALIZED.bits(), |word| {
                                asm.emit(word)
                            });
                            call(&mut asm, Helper::CellNew);
                            store(&mut asm, context_offset as u32, dest, 0, true, failed)?;
                        } else {
                            e::mov_imm(Width::X, 9, Value::UNINITIALIZED.bits(), |word| {
                                asm.emit(word)
                            });
                            store(&mut asm, context_offset as u32, dest, 9, true, failed)?;
                        }
                    }
                }
                Op::LeaveScope(scope) => {
                    for binding in &graph.scopes[*scope].bindings {
                        e::mov_imm(Width::X, 9, Value::UNDEFINED.bits(), |word| asm.emit(word));
                        store(
                            &mut asm,
                            context_offset as u32,
                            (values_size + binding.index() * 8) as u32,
                            9,
                            true,
                            failed,
                        )?;
                    }
                }
                Op::NextIteration(scope) | Op::FreshIteration(scope) => {
                    for binding in &graph.scopes[*scope].bindings {
                        if !graph.bindings[*binding].captured {
                            continue;
                        }
                        let dest = (values_size + binding.index() * 8) as u32;
                        asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                        asm.emit(e::mem_scaled(Mem::LdrX, 1, e::SP, dest)?);
                        e::mov_imm(
                            Width::X,
                            2,
                            u64::from(matches!(data.op, Op::FreshIteration(_))),
                            |word| asm.emit(word),
                        );
                        call_address(&mut asm, next_cell as *const () as u64);
                        store(&mut asm, context_offset as u32, dest, 0, true, failed)?;
                    }
                }
                op => {
                    return refuse(
                        reason,
                        format!(
                            "operation {}",
                            format!("{op:?}").split(['(', '{']).next().unwrap().trim()
                        ),
                    );
                }
            }
            let end = asm.offset();
            if start != end {
                locations.push(SourceRange {
                    start,
                    end,
                    location: Some(if semantic_locations {
                        graph.inst_location(inst)
                    } else {
                        data.span.start
                    }),
                });
            }
        }
        let start = asm.offset();
        match &data.terminator.as_ref()?.kind {
            TerminatorKind::Jump(target) => {
                edge(
                    &mut asm,
                    graph,
                    target,
                    temporary_offset as u32,
                    context_offset as u32,
                    failed,
                    labels[target.block]?,
                )?;
            }
            TerminatorKind::Branch(value, yes, no) => {
                let offset = value.index().checked_mul(8)? as u32;
                asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                asm.emit(e::mem_scaled(Mem::LdrX, 1, e::SP, offset)?);
                call(&mut asm, Helper::Truthy);
                e::mov_imm(Width::X, 10, Value::boolean(false).bits(), |word| {
                    asm.emit(word)
                });
                asm.emit(e::alu(Alu::Subs, Width::X, e::ZR, 0, 10));
                let false_edge = asm.label();
                asm.branch(Cond::Eq, false_edge);
                edge(
                    &mut asm,
                    graph,
                    yes,
                    temporary_offset as u32,
                    context_offset as u32,
                    failed,
                    labels[yes.block]?,
                )?;
                asm.bind(false_edge);
                edge(
                    &mut asm,
                    graph,
                    no,
                    temporary_offset as u32,
                    context_offset as u32,
                    failed,
                    labels[no.block]?,
                )?;
            }
            TerminatorKind::Return(value) => {
                let offset = value.index().checked_mul(8)? as u32;
                asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
                asm.emit(e::mem_scaled(Mem::LdrX, 3, e::SP, offset)?);
                asm.emit(e::mov_sp(1, e::SP));
                e::mov_imm(Width::X, 2, slot_count as u64, |word| asm.emit(word));
                e::mov_imm(Width::X, 16, finish as *const () as u64, |word| {
                    asm.emit(word)
                });
                asm.emit(e::blr(16));
                asm.emit(e::mov(Width::X, 2, 0));
                asm.emit(e::mem_scaled(Mem::LdrX, 9, e::SP, context_offset as u32)?);
                asm.emit(e::mem_scaled(
                    Mem::LdrX,
                    9,
                    9,
                    crate::jit::session::ERROR_OFFSET as u32,
                )?);
                asm.emit(e::alu(Alu::Subs, Width::X, e::ZR, 9, e::ZR));
                asm.emit(e::cset(3, Cond::Ne));
                stack_address(&mut asm, e::SP, size as u32);
                asm.emit(e::LDP_FP_LR_POST);
                if incoming != 0 {
                    asm.emit(e::add_imm(Width::X, e::SP, e::SP, incoming as u32, false));
                }
                asm.emit(e::RET);
            }
            TerminatorKind::CheckException(..) => {
                return refuse(reason, "exception check terminator");
            }
            TerminatorKind::Yield(..) => return refuse(reason, "yield terminator"),
            TerminatorKind::Aarch64Branch(..)
            | TerminatorKind::ReturnValues(..)
            | TerminatorKind::TailCall { .. } => {
                return refuse(reason, "low-level terminator");
            }
            TerminatorKind::Bailout { .. } => return refuse(reason, "unresolved bailout"),
            TerminatorKind::Fallback(..) => return refuse(reason, "fallback terminator"),
        }
        let end = asm.offset();
        if start != end {
            let terminator = data.terminator.as_ref()?;
            locations.push(SourceRange {
                start,
                end,
                location: Some(if semantic_locations {
                    graph.terminator_location(block)
                } else {
                    terminator.span.start
                }),
            });
        }
    }
    asm.bind(failed);
    asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
    e::mov_imm(Width::X, 1, error_source as u64, |word| asm.emit(word));
    asm.emit(e::mem_scaled(Mem::LdrX, 2, e::SP, span_offset as u32)?);
    asm.emit(e::shift_imm(Shift::Lsr, Width::X, 3, 2, 32));
    asm.emit(e::mov(Width::W, 2, 2));
    call(&mut asm, Helper::ErrorOrigin);
    asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context_offset as u32)?);
    asm.emit(e::mov_sp(1, e::SP));
    e::mov_imm(Width::X, 2, slot_count as u64, |word| asm.emit(word));
    e::mov_imm(Width::X, 3, Value::UNDEFINED.bits(), |word| asm.emit(word));
    e::mov_imm(Width::X, 16, finish as *const () as u64, |word| {
        asm.emit(word)
    });
    asm.emit(e::blr(16));
    asm.emit(e::mov(Width::X, 2, 0));
    e::mov_imm(Width::W, 3, 1, |word| asm.emit(word));
    stack_address(&mut asm, e::SP, size as u32);
    asm.emit(e::LDP_FP_LR_POST);
    if incoming != 0 {
        asm.emit(e::add_imm(Width::X, e::SP, e::SP, incoming as u32, false));
    }
    asm.emit(e::RET);
    let bytes = match asm.finish() {
        Ok(bytes) => bytes,
        Err(error) => return refuse(reason, format!("assembly: {error:?}")),
    };
    names.extend(literals);
    Some(MachineCode {
        bytes,
        alignment: 4,
        relocs: Vec::new(),
        traps: Vec::new(),
        locations,
        block_starts,
    })
}

fn stack_address(asm: &mut Assembler, dest: u8, offset: u32) {
    asm.emit(e::add_imm(Width::X, dest, e::SP, offset >> 12, true));
    asm.emit(e::add_imm(Width::X, dest, dest, offset & 4095, false));
}

extern "C" fn write_cell(ctx: *mut JitCtx, cell: Value, value: Value, initialize: u64) {
    let ctx = unsafe { &mut *ctx };
    if initialize == 0 && ctx.global.heap().cell_get(cell) == Some(Value::UNINITIALIZED) {
        ctx.error = Some(Box::new(crate::Error::Runtime(
            "binding accessed before initialization".into(),
        )));
        return;
    }
    if !ctx.global.heap().cell_set(cell, value) {
        ctx.error = Some(Box::new(crate::Error::Runtime(
            "invalid capture cell or value".into(),
        )));
    }
}

extern "C" fn next_cell(ctx: *mut JitCtx, cell: Value, fresh: u64) -> Value {
    let ctx = unsafe { &mut *ctx };
    let value = if fresh != 0 {
        Value::UNINITIALIZED
    } else {
        match ctx.global.heap().cell_get(cell) {
            Some(value) if value != Value::UNINITIALIZED => value,
            _ => {
                ctx.error = Some(Box::new(crate::Error::Runtime(
                    "binding accessed before initialization".into(),
                )));
                return Value::UNDEFINED;
            }
        }
    };
    // The old cell still roots value while the new cell retains it.
    match ctx.global.heap().cell(value) {
        Some(cell) => cell,
        None => {
            ctx.error = Some(Box::new(crate::Error::Runtime(
                "invalid cell value or heap exhausted".into(),
            )));
            Value::UNDEFINED
        }
    }
}

fn call_address(asm: &mut Assembler, address: u64) {
    e::mov_imm(Width::X, 16, address, |word| asm.emit(word));
    asm.emit(e::blr(16));
}

fn call(asm: &mut Assembler, helper: Helper) {
    call_address(asm, HELPERS[helper as usize].pointer as u64);
}

fn literal(asm: &mut Assembler, names: &mut Vec<Box<str>>, text: &str, ptr: u8, len: u8) {
    let name: Box<str> = text.into();
    e::mov_imm(Width::X, ptr, name.as_ptr() as u64, |word| asm.emit(word));
    e::mov_imm(Width::X, len, name.len() as u64, |word| asm.emit(word));
    names.push(name);
}

fn edge(
    asm: &mut Assembler,
    graph: &Graph,
    target: &crate::jit::semantic::Target,
    temporary: u32,
    context: u32,
    failed: super::asm::Label,
    label: super::asm::Label,
) -> Option<()> {
    let params = &graph.blocks[target.block].params;
    if params.len() != target.args.len() {
        return None;
    }
    for (index, value) in target.args.iter().enumerate() {
        asm.emit(e::mem_scaled(
            Mem::LdrX,
            9,
            e::SP,
            (value.index() * 8) as u32,
        )?);
        store(asm, context, temporary + index as u32 * 8, 9, false, failed)?;
    }
    for (index, param) in params.iter().enumerate() {
        asm.emit(e::mem_scaled(
            Mem::LdrX,
            9,
            e::SP,
            temporary + index as u32 * 8,
        )?);
        store(asm, context, (param.index() * 8) as u32, 9, false, failed)?;
    }
    asm.jump(e::B, label);
    Some(())
}

extern "C" fn global(ctx: *mut JitCtx, ptr: *const u8, len: usize, optional: u64) -> Value {
    let ctx = unsafe { &mut *ctx };
    let name = unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len)) };
    match ctx.global.get_raw(name) {
        Some(Value::UNINITIALIZED) => {
            ctx.error = Some(Box::new(crate::Error::Runtime(format!(
                "binding `{name}` accessed before initialization"
            ))));
            Value::UNDEFINED
        }
        Some(value) => ctx.retain(value),
        None if optional != 0 => Value::UNDEFINED,
        None => {
            ctx.error = Some(Box::new(crate::Error::Runtime(format!(
                "unknown variable `{name}`"
            ))));
            Value::UNDEFINED
        }
    }
}

extern "C" fn string(ctx: *mut JitCtx, ptr: *const u8, len: usize) -> Value {
    let ctx = unsafe { &mut *ctx };
    let text = unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len)) };
    match ctx.global.heap().string(text) {
        Some(value) => value,
        None => {
            ctx.error = Some(Box::new(crate::Error::Runtime(
                "string heap exhausted".into(),
            )));
            Value::UNDEFINED
        }
    }
}

extern "C" fn nullish(value: Value, kind: u64) -> Value {
    Value::boolean(match kind {
        1 => value.is_nullish(),
        2 => value == Value::UNINITIALIZED,
        _ => value.is_undefined(),
    })
}

extern "C" fn indexed(ctx: *mut JitCtx, value: Value, length: u64) -> Value {
    let heap = unsafe { &*ctx }.global.heap();
    if length != 0 {
        return Value::number(heap.len(value).unwrap_or(0) as f64);
    }
    Value::boolean(heap.is_array(value) || heap.typed_kind(value).is_some())
}

extern "C" fn replace(ctx: *mut JitCtx, slot: *mut Value, value: Value, owned: u64) {
    let ctx = unsafe { &mut *ctx };
    let value = if owned == 0 { ctx.retain(value) } else { value };
    let old = unsafe { std::ptr::replace(slot, value) };
    ctx.global.heap().release(old);
}

extern "C" fn finish(ctx: *mut JitCtx, slots: *mut Value, count: usize, value: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    let result = if ctx.error.is_none() {
        ctx.retain(value)
    } else {
        Value::UNDEFINED
    };
    for value in unsafe { std::slice::from_raw_parts(slots, count) } {
        ctx.global.heap().release(*value);
    }
    result
}

fn store(
    asm: &mut Assembler,
    context: u32,
    offset: u32,
    value_reg: u8,
    owned: bool,
    failed: super::asm::Label,
) -> Option<()> {
    let slow = asm.label();
    let done = asm.label();
    asm.emit(e::mov(Width::X, 11, value_reg));
    asm.emit(e::mem_scaled(Mem::LdrX, 12, e::SP, offset)?);
    e::mov_imm(
        Width::X,
        14,
        crate::value::TAG_OBJECT >> crate::value::TAG_SHIFT,
        |word| asm.emit(word),
    );
    asm.emit(e::shift_imm(
        Shift::Lsr,
        Width::X,
        13,
        12,
        crate::value::TAG_SHIFT,
    ));
    asm.emit(e::alu(Alu::Sub, Width::X, 13, 13, 14));
    asm.emit(e::subs_imm(Width::X, e::ZR, 13, 2, false));
    asm.branch(Cond::Lo, slow);
    if !owned {
        asm.emit(e::shift_imm(
            Shift::Lsr,
            Width::X,
            13,
            11,
            crate::value::TAG_SHIFT,
        ));
        asm.emit(e::alu(Alu::Sub, Width::X, 13, 13, 14));
        asm.emit(e::subs_imm(Width::X, e::ZR, 13, 2, false));
        asm.branch(Cond::Lo, slow);
    }
    asm.emit(e::mem_scaled(Mem::StrX, 11, e::SP, offset)?);
    asm.jump(e::B, done);
    asm.bind(slow);
    asm.emit(e::mov(Width::X, 2, 11));
    asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context)?);
    stack_address(asm, 1, offset);
    e::mov_imm(Width::X, 3, u64::from(owned), |word| asm.emit(word));
    e::mov_imm(Width::X, 16, replace as *const () as u64, |word| {
        asm.emit(word)
    });
    asm.emit(e::blr(16));
    asm.bind(done);
    check_error(asm, context, failed)
}

fn numeric(graph: &Graph, value: SemanticValue, depth: usize) -> bool {
    if depth > 128 {
        return false;
    }
    let ValueDef::Result(inst, _) = graph.values[value] else {
        return false;
    };
    match graph.insts[inst].op {
        Op::Constant(Constant::Number(_)) => true,
        Op::ToNumeric(_) => true,
        Op::Binary(
            BinaryOperator::Addition
            | BinaryOperator::Subtraction
            | BinaryOperator::Multiplication
            | BinaryOperator::Division
            | BinaryOperator::Remainder,
            left,
            right,
        ) => numeric(graph, left, depth + 1) && numeric(graph, right, depth + 1),
        _ => false,
    }
}

fn check_initialized(
    asm: &mut Assembler,
    context: u32,
    offset: u32,
    failed: super::asm::Label,
) -> Option<()> {
    let ready = asm.label();
    asm.emit(e::mem_scaled(Mem::LdrX, 9, e::SP, offset)?);
    e::mov_imm(Width::X, 10, Value::UNINITIALIZED.bits(), |word| {
        asm.emit(word)
    });
    asm.emit(e::alu(Alu::Subs, Width::X, e::ZR, 9, 10));
    asm.branch(Cond::Ne, ready);
    asm.emit(e::mov(Width::X, 1, 9));
    asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context)?);
    call(asm, Helper::Retain);
    asm.jump(e::B, failed);
    asm.bind(ready);
    Some(())
}

fn check_error(asm: &mut Assembler, context_offset: u32, failed: super::asm::Label) -> Option<()> {
    asm.emit(e::mem_scaled(Mem::LdrX, 9, e::SP, context_offset)?);
    asm.emit(e::mem_scaled(
        Mem::LdrX,
        9,
        9,
        crate::jit::session::ERROR_OFFSET as u32,
    )?);
    asm.emit(e::alu(Alu::Subs, Width::X, e::ZR, 9, e::ZR));
    asm.branch(Cond::Ne, failed);
    asm.emit(e::mem_scaled(Mem::LdrX, 9, e::SP, context_offset)?);
    asm.emit(e::mem_scaled(
        Mem::LdrX,
        9,
        9,
        crate::jit::session::EXCEPTION_OFFSET as u32,
    )?);
    e::mov_imm(Width::X, 10, Value::UNINITIALIZED.bits(), |word| {
        asm.emit(word)
    });
    asm.emit(e::alu(Alu::Subs, Width::X, e::ZR, 9, 10));
    asm.branch(Cond::Ne, failed);
    Some(())
}

extern "C" fn output(ctx: *mut JitCtx, buffer: *mut Value, index: usize, value: Value) {
    let ctx = unsafe { &mut *ctx };
    ctx.global.heap().retain(value);
    unsafe { buffer.add(index).write(value) };
}

mod ops;
mod poll;
mod references;
mod repr;
mod scope;
mod semantic_graph;

use crate::FxHashMap;

use super::semantic::low_level::{BlockArg, Builder as GraphBuilder, SignatureRef as SigRef};
use super::semantic::{Block, Inst, Value as LoweredValue};
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{MemFlagsData, Type, types};
use cranelift_frontend::FunctionBuilder;
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Module};
use oxc_span::Span;

use crate::error::Error;
use crate::jit::clif::Annotations;
use crate::jit::cranelift::stubs::Stubs;
use crate::jit::function_data::{FUNCTION_DATA_BASE_OFFSET, FunctionData};
use crate::jit::helpers::Helper;
use crate::value::Value;

pub(super) enum FunctionKind {
    Ordinary,
    Factory(std::sync::Arc<crate::continuation::Plan>),
}

#[derive(Clone, Copy)]
pub(super) struct Feedback {
    pub(super) function_return: bool,
    pub(super) observations: bool,
    pub(super) get_refinement: bool,
    pub(super) owner: u64,
}

pub(super) struct FunctionCompiler<'a, 'f> {
    feedback: Option<Feedback>,
    names: &'a mut Vec<Box<str>>,
    cells: &'a mut Vec<Box<u64>>,
    data: &'a mut FunctionData,
    data_base: LoweredValue,
    source_name: (usize, usize),
    source_span: Span,
    b: &'a mut GraphBuilder<'f>,
    module: &'a mut JITModule,
    helper_signatures: FxHashMap<Helper, SigRef>,
    source_names: FxHashMap<usize, i64>,
    error_sites: FxHashMap<(usize, u32, u32), i64>,
    error_dispatch: FxHashMap<Block, Block>,
    stubs: Stubs,
    annotations: &'a mut Annotations,
    pointer: Type,
    ctx: LoweredValue,
    receiver: LoweredValue,
    arguments: Arguments,
    return_block: Block,
    failed_exit: Option<Block>,
    error_origin: Option<Block>,
    handlers: Vec<Block>,
    terminated: bool,
    frame_size: Option<Inst>,
    frame: Option<CallerFrame>,
    fixed_signatures: FxHashMap<u16, SigRef>,
    property_caches: FxHashMap<super::semantic::Inst, i64>,
    slots: FxHashMap<LoweredValue, u32>,
    slot_count: u32,
    unchecked: Option<references::Token>,
    references: references::References,
}

enum Arguments {
    Vector {
        argv: LoweredValue,
        argc: LoweredValue,
    },
    Fixed(Vec<LoweredValue>),
}

#[derive(Clone, Copy)]
struct CallerFrame {
    start: LoweredValue,
    base: LoweredValue,
}

const INLINE_FRAME_RELEASES: u32 = 4;

fn data_base(b: &mut GraphBuilder<'_>, pointer: Type, data: &FunctionData) -> LoweredValue {
    let holder = b.ins().iconst(pointer, &raw const *data as i64);
    b.ins().load(
        pointer,
        MemFlagsData::trusted(),
        holder,
        FUNCTION_DATA_BASE_OFFSET,
    )
}

pub(super) struct Output<'a, 'f> {
    pub(super) b: &'a mut FunctionBuilder<'f>,
    pub(super) clif: bool,
    pub(super) native: bool,
    pub(super) lowered: &'a mut Option<super::semantic::Graph>,
    pub(super) module: &'a mut JITModule,
    pub(super) stubs: Stubs,
    pub(super) annotations: &'a mut Annotations,
    pub(super) names: &'a mut Vec<Box<str>>,
    pub(super) cells: &'a mut Vec<Box<u64>>,
    pub(super) data: &'a mut FunctionData,
    pub(super) source_name: (usize, usize),
    pub(super) after: &'a mut dyn FnMut(&str, &super::semantic::Graph),
}

pub(super) fn build_function(
    out: Output<'_, '_>,
    params: Span,
    graph: super::semantic::Graph,
    facts: &super::semantic::Facts,
    continuations: &FxHashMap<usize, FuncId>,
    feedback: Option<Feedback>,
) -> Result<(), Error> {
    build_callable(
        out,
        params,
        CallableBody::Graph(graph, facts),
        continuations,
        feedback,
    )
}

pub(super) fn build_factory(
    out: Output<'_, '_>,
    params: Span,
    resume: std::sync::Arc<crate::continuation::Plan>,
) -> Result<(), Error> {
    build_callable(
        out,
        params,
        CallableBody::Factory(resume),
        &FxHashMap::default(),
        None,
    )
}

pub(super) fn build_global(
    out: Output<'_, '_>,
    graph: super::semantic::Graph,
    facts: &super::semantic::Facts,
    continuations: &FxHashMap<usize, FuncId>,
) -> Result<(), Error> {
    debug_assert!(graph.arity.is_none());
    build_callable(
        out,
        Span::new(0, 0),
        CallableBody::Graph(graph, facts),
        continuations,
        None,
    )
}

enum CallableBody<'g> {
    Graph(super::semantic::Graph, &'g super::semantic::Facts),
    Factory(std::sync::Arc<crate::continuation::Plan>),
}

enum Body<'g> {
    Graph(semantic_graph::Context<'g>),
    Factory(std::sync::Arc<crate::continuation::Plan>),
}

fn build_callable(
    out: Output<'_, '_>,
    params: Span,
    body: CallableBody<'_>,
    continuations: &FxHashMap<usize, FuncId>,
    feedback: Option<Feedback>,
) -> Result<(), Error> {
    let Output {
        b: clif_builder,
        clif,
        native,
        lowered: output_graph,
        module,
        stubs,
        annotations,
        names,
        cells,
        data,
        source_name,
        after,
    } = out;
    let mut lowered = lower_callable(
        params,
        body,
        continuations,
        feedback,
        clif_builder.func.signature.clone(),
        module,
        stubs,
        annotations,
        names,
        cells,
        data,
        source_name,
        after,
    )?;
    if !clif {
        *output_graph = Some(lowered);
        return Ok(());
    }
    if native {
        *output_graph = Some(lowered.clone());
    }
    super::cranelift::prepare(&mut lowered, after).map_err(Error::Compile)?;
    super::cranelift::emit(clif_builder, module, &lowered, Some(annotations))?;
    Ok(())
}

fn lower_callable(
    params: Span,
    body: CallableBody<'_>,
    continuations: &FxHashMap<usize, FuncId>,
    feedback: Option<Feedback>,
    signature: cranelift_codegen::ir::Signature,
    module: &mut JITModule,
    stubs: Stubs,
    annotations: &mut Annotations,
    names: &mut Vec<Box<str>>,
    cells: &mut Vec<Box<u64>>,
    data: &mut FunctionData,
    source_name: (usize, usize),
    after: &mut dyn FnMut(&str, &super::semantic::Graph),
) -> Result<super::semantic::Graph, Error> {
    let (mut lowered, body) = match body {
        CallableBody::Graph(mut graph, facts) => {
            let context = semantic_graph::Context::new(&mut graph, facts);
            graph.span = params;
            graph.low_level_signature = Some(signature);
            (graph, Body::Graph(context))
        }
        CallableBody::Factory(resume) => (
            super::semantic::Graph::new_low_level("<lowered>".into(), params, signature),
            Body::Factory(resume),
        ),
    };
    let fixed = lowered.arity.is_some();
    let mut graph_builder = GraphBuilder::new(&mut lowered);
    let b = &mut graph_builder;
    b.set_source_name(Some(source_name));
    if matches!(body, Body::Graph(_)) {
        b.graph.entry = b.create_block();
    }
    let entry = b.graph.entry;
    b.switch_to_block(entry);
    b.append_block_params_for_function_params(entry);
    let values = b.block_params(entry).to_vec();
    let (ctx, closure, receiver) = (values[0], values[1], values[2]);
    let words = if fixed { &values[1..] } else { &values[1..3] };
    for &word in words {
        b.graph.reprs[word] = Some(super::semantic::Repr::Boxed);
    }
    let arguments = if fixed {
        Arguments::Fixed(values[3..].to_vec())
    } else {
        Arguments::Vector {
            argv: values[3],
            argc: values[4],
        }
    };
    let return_block = b.create_block();
    b.append_block_param_repr(return_block, super::semantic::Repr::Boxed);
    if matches!(arguments, Arguments::Fixed(_)) {
        b.append_block_param(return_block, types::I8);
    }

    let pointer = module.target_config().pointer_type();
    let data_base = data_base(b, pointer, data);
    let empty_owned = b.ins().iconst(types::I64, 0);
    let empty_owned = b.as_boxed(empty_owned);
    let mut compiler = FunctionCompiler {
        feedback,
        names,
        cells,
        data,
        data_base,
        source_name,
        source_span: params,
        b,
        module,
        helper_signatures: FxHashMap::default(),
        source_names: FxHashMap::default(),
        error_sites: FxHashMap::default(),
        error_dispatch: FxHashMap::default(),
        stubs,
        annotations,
        pointer,
        ctx,
        receiver,
        arguments,
        return_block,
        failed_exit: None,
        error_origin: None,
        handlers: Vec::new(),
        terminated: false,
        frame_size: None,
        frame: None,
        fixed_signatures: FxHashMap::default(),
        property_caches: FxHashMap::default(),
        slots: FxHashMap::default(),
        slot_count: 0,
        unchecked: None,
        references: references::References::new(empty_owned),
    };

    compiler.enter_frame();
    if let Body::Factory(resume) = body {
        let (argv, argc) = compiler.argument_vector()?;
        let args = compiler.call_helper(Helper::ArrayNew, &[argv, argc]);
        let arguments = compiler.unchecked.expect("argument array reference");
        compiler.check_error();
        let plan = compiler.data.pin(resume);
        let code = compiler.b.ins().iconst(types::I64, plan as i64);
        let generator =
            compiler.call_helper(Helper::GeneratorNew, &[closure, receiver, args, code]);
        compiler.release_tracked(&[arguments])?;
        compiler.check_error();
        let args = compiler.return_args(generator, false);
        compiler.jump(return_block, &args);
        compiler.finish_failed_exit();
        compiler.switch(return_block);
        let result = compiler.b.block_params(return_block)[0];
        compiler.return_value(result);
    } else if let Body::Graph(context) = body {
        compiler.finalize_runtime_lowering(context, closure, continuations)?;
        compiler.patch_frame_size();
    }
    compiler
        .references
        .finish(compiler.b.graph)
        .map_err(Error::Compile)?;
    after("finalize-runtime", graph_builder.graph);
    super::semantic::lower_low_level(
        &mut lowered,
        ctx,
        module.target_config().default_call_conv,
        pointer,
        after,
    )
    .map_err(Error::Compile)?;
    Ok(lowered)
}

impl FunctionCompiler<'_, '_> {
    pub(super) fn switch(&mut self, block: Block) {
        self.b.switch_to_block(block);
        self.terminated = false;
    }

    pub(super) fn jump(&mut self, block: Block, args: &[BlockArg]) {
        self.forward_references(block, args);
        self.b.ins().jump(block, args);
        self.terminated = true;
    }

    pub(super) fn brif(
        &mut self,
        condition: LoweredValue,
        then_block: Block,
        then_args: &[BlockArg],
        else_block: Block,
        else_args: &[BlockArg],
    ) {
        self.forward_references(then_block, then_args);
        self.forward_references(else_block, else_args);
        self.b
            .ins()
            .brif(condition, then_block, then_args, else_block, else_args);
        self.terminated = true;
    }

    /// Fixed-arity code also returns whether it left with an error or exception pending,
    /// sparing direct callers from checking.
    pub(super) fn return_args(&mut self, value: LoweredValue, failed: bool) -> Vec<BlockArg> {
        let mut args = vec![BlockArg::Value(value)];
        if matches!(self.arguments, Arguments::Fixed(_)) {
            let status = self.b.ins().iconst(types::I8, i64::from(failed));
            args.push(BlockArg::Value(status));
        }
        args
    }

    pub(super) fn failed_exit(&mut self) -> Block {
        if let Some(block) = self.failed_exit {
            return block;
        }
        let block = self.b.create_block();
        self.b.set_cold_block(block);
        self.failed_exit = Some(block);
        block
    }

    pub(super) fn finish_failed_exit(&mut self) {
        if let Some(block) = self.error_origin {
            for (target, dispatch) in std::mem::take(&mut self.error_dispatch) {
                self.switch(dispatch);
                let site = self.b.block_params(dispatch)[0];
                let error = self.load_ctx(crate::jit::session::ERROR_OFFSET);
                self.brif(error, block, &[site.into()], target, &[]);
            }
            self.switch(block);
            let site = self.b.block_params(block)[0];
            let flags = MemFlagsData::trusted();
            let offset = self.b.ins().load(types::I64, flags, site, 0);
            let source = self.b.ins().iadd(self.data_base, offset);
            let span = self.b.ins().load(types::I64, flags, site, 8);
            let start = self.b.ins().band_imm_u(span, u32::MAX as i64);
            let end = self.b.ins().ushr_imm_s(span, 32);
            self.call_helper(Helper::ErrorOrigin, &[source, start, end]);
            let exit = self.failed_exit();
            self.jump(exit, &[]);
        }
        let Some(block) = self.failed_exit else {
            return;
        };
        self.switch(block);
        let undefined = self.constant(Value::UNDEFINED);
        let args = self.return_args(undefined, true);
        self.jump(self.return_block, &args);
    }

    pub(super) fn return_value(&mut self, value: LoweredValue) {
        if let Some(feedback) = self.feedback.filter(|feedback| feedback.function_return)
            && self.stubs.compilation_state != 0
            && feedback.owner >= crate::jit::entries::DIRECT_ENTRIES as u64
        {
            let owner = self.b.ins().iconst(types::I64, feedback.owner as i64);
            self.call_helper(Helper::CompileLeave, &[owner]);
        } else if let Some(feedback) = self.feedback.filter(|feedback| feedback.function_return)
            && self.stubs.compilation_state != 0
        {
            let address = self.stubs.direct_entries as usize
                + feedback.owner as usize * crate::jit::entries::DIRECT_ENTRY_SIZE
                + crate::jit::entries::DIRECT_CALLS_OFFSET;
            let address = self.b.ins().iconst(self.pointer, address as i64);
            let count = self
                .b
                .ins()
                .load(self.pointer, MemFlagsData::trusted(), address, 0);
            let calls = self
                .b
                .ins()
                .band_imm_u(count, (crate::jit::entries::RECOMPILE_PENDING - 1) as i64);
            let next = self.b.ins().iadd_imm_u(count, 1);
            let cold = self.b.ins().icmp_imm_u(
                IntCC::UnsignedLessThan,
                calls,
                crate::jit::entries::RECOMPILE_WARMUP as i64,
            );
            let count = self.b.ins().select(cold, next, count);
            self.b
                .ins()
                .store(MemFlagsData::trusted(), count, address, 0);
            let warm = self.b.ins().icmp_imm_u(
                IntCC::UnsignedGreaterThanOrEqual,
                count,
                (crate::jit::entries::RECOMPILE_PENDING + crate::jit::entries::RECOMPILE_WARMUP)
                    as i64,
            );
            let poll = self.b.create_block();
            let done = self.b.create_block();
            self.b.ins().brif(warm, poll, &[], done, &[]);
            self.b.switch_to_block(poll);
            let owner = self.b.ins().iconst(types::I64, feedback.owner as i64);
            if let Some(call) = poll::emit(
                self.b,
                self.module,
                self.stubs.compilation_state,
                self.ctx,
                Some(owner),
            ) {
                self.b.mark_helper(call, Helper::CompileLeave);
            }
            self.b.ins().jump(done, &[]);
            self.b.switch_to_block(done);
        }
        self.exit_frame(value);
        let mut results = vec![value];
        if matches!(self.arguments, Arguments::Fixed(_)) {
            results.push(self.b.block_params(self.return_block)[1]);
        }
        self.b.ins().return_(&results);
        self.terminated = true;
    }

    fn exit_frame(&mut self, value: LoweredValue) {
        let caller = self.frame.unwrap();
        self.increment(value);
        if self.slot_count > 0 {
            if self.slot_count <= INLINE_FRAME_RELEASES {
                for slot in 0..self.slot_count {
                    self.release_slot(slot);
                }
            } else {
                self.call_helper(Helper::FrameRelease, &[]);
            }
        }
        self.release_owned();
        let start = self.load_ctx(crate::jit::session::LOCALS_START_OFFSET);
        self.store_ctx(start, crate::jit::session::LOCALS_LEN_OFFSET);
        self.store_ctx(caller.start, crate::jit::session::LOCALS_START_OFFSET);
        self.store_ctx(caller.base, crate::jit::session::LOCALS_OFFSET);
    }

    fn release_slot(&mut self, slot: u32) {
        let empty = self.constant(Value::UNDEFINED);
        let value = self.b.store_frame(self.ctx, empty, slot);
        let heap = self.is_heap(value);
        let release = self.b.create_block();
        let done = self.b.create_block();
        self.brif(heap, release, &[], done, &[]);
        self.switch(release);
        self.release_inline(value, done, true);
        self.switch(done);
    }

    /// Slots above the stack's length are always empty, so nothing needs initializing.
    fn enter_frame(&mut self) {
        use crate::jit::session::{
            LOCALS_LEN_OFFSET, LOCALS_LIMIT_OFFSET, LOCALS_OFFSET, LOCALS_START_OFFSET,
            STACK_LIMIT_OFFSET,
        };
        let caller = CallerFrame {
            start: self.load_ctx(LOCALS_START_OFFSET),
            base: self.load_ctx(LOCALS_OFFSET),
        };
        let start = self.load_ctx(LOCALS_LEN_OFFSET);
        let size = self.b.ins().iconst(types::I64, 0);
        self.frame_size = Some(self.b.defining_inst(size));
        let len = self.b.ins().iadd(start, size);
        let limit = self.load_ctx(LOCALS_LIMIT_OFFSET);
        let full = self.b.ins().icmp(IntCC::UnsignedGreaterThan, len, limit);
        let stack_pointer = self.b.ins().get_stack_pointer(self.pointer);
        let stack_limit = self.load_ctx(STACK_LIMIT_OFFSET);
        let deep = self
            .b
            .ins()
            .icmp(IntCC::UnsignedLessThan, stack_pointer, stack_limit);
        let overflow = self.b.ins().boolean_or(full, deep);
        let fail = self.b.create_block();
        let enter = self.b.create_block();
        self.b.set_cold_block(fail);
        self.brif(overflow, fail, &[], enter, &[]);
        self.switch(fail);
        self.call_helper(Helper::FrameOverflow, &[]);
        let undefined = self.constant(Value::UNDEFINED);
        let mut results = vec![undefined];
        if matches!(self.arguments, Arguments::Fixed(_)) {
            results.push(self.b.ins().iconst(types::I8, 1));
        }
        self.b.ins().return_(&results);
        self.switch(enter);
        let caller_slots = self.b.ins().isub(start, caller.start);
        let offset = self.b.ins().ishl_imm_s(caller_slots, 3);
        let base = self.b.ins().iadd(caller.base, offset);
        self.store_ctx(base, LOCALS_OFFSET);
        self.store_ctx(start, LOCALS_START_OFFSET);
        self.store_ctx(len, LOCALS_LEN_OFFSET);
        self.frame = Some(caller);
    }

    fn store_ctx(&mut self, value: LoweredValue, offset: i32) -> Inst {
        let store = self
            .b
            .ins()
            .store(MemFlagsData::trusted(), value, self.ctx, offset);
        store
    }

    fn patch_frame_size(&mut self) {
        if let Some(inst) = self.frame_size {
            self.b.patch_integer(inst, self.slot_count as u64);
        }
    }

    pub(super) fn argument_vector(&self) -> Result<(LoweredValue, LoweredValue), Error> {
        match self.arguments {
            Arguments::Vector { argv, argc } => Ok((argv, argc)),
            Arguments::Fixed(_) => Err(Error::Compile(
                "argument vector read in a fixed-arity function".into(),
            )),
        }
    }

    pub(super) fn load_arg(&mut self, index: usize) -> LoweredValue {
        let (argv, argc) = match &self.arguments {
            Arguments::Fixed(params) => return params[index],
            Arguments::Vector { argv, argc } => (*argv, *argc),
        };
        let present = self
            .b
            .ins()
            .icmp_imm_s(IntCC::UnsignedGreaterThan, argc, index as i64);
        let load_block = self.b.create_block();
        let missing_block = self.b.create_block();
        let done = self.b.create_block();
        self.b
            .append_block_param_repr(done, crate::jit::semantic::Repr::Boxed);
        self.brif(present, load_block, &[], missing_block, &[]);

        self.switch(load_block);
        let word = self.b.ins().load(
            types::I64,
            MemFlagsData::trusted(),
            argv,
            (index * 8) as i32,
        );
        let value = self.b.as_boxed(word);
        self.jump(done, &[BlockArg::Value(value)]);

        self.switch(missing_block);
        let undefined = self.constant(Value::UNDEFINED);
        self.jump(done, &[BlockArg::Value(undefined)]);

        self.switch(done);
        self.b.block_params(done)[0]
    }
}

pub(super) mod class;
pub(in crate::jit) mod expression;
mod ir_helpers;
mod ops;
mod repr;
pub(in crate::jit) mod scope;
mod semantic_graph;
mod slots;
mod statement;
pub(super) mod stubs;

use std::collections::HashMap;

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{
    Block, BlockArg, Inst, InstBuilder, MemFlagsData, SigRef, Type, Value as ClifValue, types,
};
use cranelift_frontend::{FunctionBuilder, Switch, Variable};
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Module};
use oxc_span::Span;

use crate::error::Error;
use crate::jit::clif::Annotations;
use crate::jit::function_data::{FUNCTION_DATA_BASE_OFFSET, FunctionData};
use crate::jit::helpers::Helper;
use crate::value::Value;

pub(crate) use statement::statement_kind;
pub(super) use stubs::Stubs;

pub(super) enum FunctionKind {
    Ordinary,
    Generator(super::generator::ResumePlan),
    Factory(FuncId),
}

#[derive(Clone, Copy)]
pub(super) struct Feedback {
    pub(super) function_return: bool,
    pub(super) get_refinement: bool,
    pub(super) owner: u64,
}

pub(super) struct FunctionCompiler<'a, 'f> {
    feedback: Option<Feedback>,
    names: &'a mut Vec<Box<str>>,
    cells: &'a mut Vec<Box<u64>>,
    data: &'a mut FunctionData,
    data_base: ClifValue,
    source_name: (usize, usize),
    source_span: Span,
    b: &'a mut FunctionBuilder<'f>,
    module: &'a mut JITModule,
    helper_signatures: HashMap<Helper, SigRef>,
    source_names: HashMap<usize, i64>,
    stubs: Stubs,
    annotations: &'a mut Annotations,
    pointer: Type,
    ctx: ClifValue,
    receiver: ClifValue,
    arguments: Arguments,
    return_block: Block,
    failed_exit: Option<Block>,
    error_origin: Option<Block>,
    handlers: Vec<Block>,
    generator: Option<ClifValue>,
    resume_blocks: Vec<Block>,
    suspensions: HashMap<u32, (usize, Block)>,
    terminated: bool,
    frame_size: Option<Inst>,
    frame_roots: Option<Variable>,
    frame: Option<CallerFrame>,
    frame_code: bool,
    fixed_signatures: HashMap<u16, SigRef>,
    property_caches: HashMap<super::semantic::Inst, i64>,
    slots: HashMap<u32, u32>,
    slot_count: u32,
}

enum Arguments {
    Vector { argv: ClifValue, argc: ClifValue },
    Fixed(Vec<ClifValue>),
}

#[derive(Clone, Copy)]
struct CallerFrame {
    start: ClifValue,
    base: ClifValue,
    floor: ClifValue,
}

const INLINE_FRAME_RELEASES: u32 = 4;

fn data_base(b: &mut FunctionBuilder<'_>, pointer: Type, data: &FunctionData) -> ClifValue {
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
    pub(super) module: &'a mut JITModule,
    pub(super) stubs: Stubs,
    pub(super) annotations: &'a mut Annotations,
    pub(super) names: &'a mut Vec<Box<str>>,
    pub(super) cells: &'a mut Vec<Box<u64>>,
    pub(super) data: &'a mut FunctionData,
    pub(super) source_name: (usize, usize),
}

pub(super) fn build_function(
    out: Output<'_, '_>,
    params: Span,
    kind: FunctionKind,
    graph: &super::semantic::Graph,
    facts: &super::semantic::Facts,
    continuations: &HashMap<usize, FuncId>,
    feedback: Option<Feedback>,
) -> Result<(), Error> {
    build_callable(
        out,
        params,
        kind,
        CallableBody::Graph(graph, facts),
        continuations,
        feedback,
    )
}

pub(super) fn build_factory(
    out: Output<'_, '_>,
    params: Span,
    resume: FuncId,
) -> Result<(), Error> {
    build_callable(
        out,
        params,
        FunctionKind::Factory(resume),
        CallableBody::Factory(resume),
        &HashMap::new(),
        None,
    )
}

pub(super) fn build_global(
    out: Output<'_, '_>,
    graph: &super::semantic::Graph,
    facts: &super::semantic::Facts,
    continuations: &HashMap<usize, FuncId>,
) -> Result<(), Error> {
    debug_assert!(graph.arity.is_none());
    build_callable(
        out,
        Span::new(0, 0),
        FunctionKind::Ordinary,
        CallableBody::Graph(graph, facts),
        continuations,
        None,
    )
}

enum CallableBody<'g> {
    Graph(&'g super::semantic::Graph, &'g super::semantic::Facts),
    Factory(FuncId),
}

fn build_callable(
    out: Output<'_, '_>,
    params: Span,
    kind: FunctionKind,
    body: CallableBody<'_>,
    continuations: &HashMap<usize, FuncId>,
    feedback: Option<Feedback>,
) -> Result<(), Error> {
    let Output {
        b,
        module,
        stubs,
        annotations,
        names,
        cells,
        data,
        source_name,
    } = out;
    let entry = b.create_block();
    b.switch_to_block(entry);
    b.append_block_params_for_function_params(entry);
    let values = b.block_params(entry).to_vec();
    let (ctx, closure, receiver) = (values[0], values[1], values[2]);
    let arguments = match body {
        CallableBody::Graph(graph, _) if graph.arity.is_some() => {
            Arguments::Fixed(values[3..].to_vec())
        }
        _ => Arguments::Vector {
            argv: values[3],
            argc: values[4],
        },
    };
    let return_block = b.create_block();
    b.append_block_param(return_block, types::I64);
    if matches!(arguments, Arguments::Fixed(_)) {
        b.append_block_param(return_block, types::I8);
    }

    let pointer = module.target_config().pointer_type();
    let data_base = data_base(b, pointer, data);
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
        helper_signatures: HashMap::new(),
        source_names: HashMap::new(),
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
        generator: None,
        resume_blocks: Vec::new(),
        suspensions: HashMap::new(),
        terminated: false,
        frame_size: None,
        frame_roots: None,
        frame: None,
        frame_code: false,
        fixed_signatures: HashMap::new(),
        property_caches: HashMap::new(),
        slots: HashMap::new(),
        slot_count: 0,
    };

    let dispatch = if let FunctionKind::Generator(plan) = &kind {
        compiler.generator = Some(closure);
        let dispatch = compiler.b.create_block();
        let start = compiler.b.create_block();
        compiler.jump(dispatch, &[]);
        compiler.resume_blocks.push(start);
        for (index, span) in plan.suspensions.iter().enumerate() {
            let block = compiler.b.create_block();
            compiler.suspensions.insert(*span, (index + 1, block));
        }
        compiler.switch(start);
        Some(dispatch)
    } else {
        compiler.enter_frame();
        None
    };
    if let CallableBody::Factory(resume) = body {
        let (argv, argc) = compiler.argument_vector()?;
        let args = compiler.call_helper(Helper::ArrayNew, &[argv, argc]);
        compiler.check_error();
        let code = compiler.b.ins().iconst(types::I64, resume.as_u32() as i64);
        let generator =
            compiler.call_helper(Helper::GeneratorNew, &[closure, receiver, args, code]);
        compiler.check_error();
        let args = compiler.return_args(generator, false);
        compiler.jump(return_block, &args);
        compiler.finish_failed_exit();
        compiler.switch(return_block);
        let result = compiler.b.block_params(return_block)[0];
        compiler.return_value(result);
        return Ok(());
    }
    if let CallableBody::Graph(graph, facts) = body {
        let graph_closure = if compiler.generator.is_some() {
            compiler.call_helper(Helper::GeneratorClosure, &[closure])
        } else {
            closure
        };
        compiler.emit_semantic_graph(graph, facts, graph_closure, continuations)?;
        compiler.patch_frame_size();
        if let Some(dispatch) = dispatch {
            compiler.switch(dispatch);
            let state = compiler.call_helper(Helper::GeneratorState, &[]);
            let mut switch = Switch::new();
            for (state, block) in compiler.resume_blocks.iter().enumerate() {
                switch.set_entry(state as u128, *block);
            }
            let invalid = compiler.b.create_block();
            switch.emit(compiler.b, state, invalid);
            compiler.switch(invalid);
            let undefined = compiler.constant(Value::UNDEFINED);
            let args = compiler.return_args(undefined, false);
            compiler.jump(return_block, &args);
        }
        return Ok(());
    }
    Ok(())
}

impl FunctionCompiler<'_, '_> {
    pub(super) fn switch(&mut self, block: Block) {
        self.b.switch_to_block(block);
        self.terminated = false;
    }

    pub(super) fn jump(&mut self, block: Block, args: &[BlockArg]) {
        self.b.ins().jump(block, args);
        self.terminated = true;
    }

    pub(super) fn brif(
        &mut self,
        condition: ClifValue,
        then_block: Block,
        then_args: &[BlockArg],
        else_block: Block,
        else_args: &[BlockArg],
    ) {
        self.b
            .ins()
            .brif(condition, then_block, then_args, else_block, else_args);
        self.terminated = true;
    }

    /// Fixed-arity code also returns whether it left with an error or exception pending,
    /// sparing direct callers from checking.
    pub(super) fn return_args(&mut self, value: ClifValue, failed: bool) -> Vec<BlockArg> {
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
            self.switch(block);
            let args = self.b.block_params(block).to_vec();
            self.call_helper(Helper::ErrorOrigin, &args);
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

    pub(super) fn return_value(&mut self, value: ClifValue) {
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
            if let Some(call) = stubs::poll(
                self.b,
                self.module,
                self.stubs.compilation_state,
                self.ctx,
                Some(owner),
            ) {
                self.annotations.helpers.insert(call, Helper::CompileLeave);
            }
            self.b.ins().jump(done, &[]);
            self.b.switch_to_block(done);
        }
        let value = match self.generator {
            Some(generator) => self.call_helper(Helper::GeneratorComplete, &[generator, value]),
            None => {
                self.exit_frame(value);
                value
            }
        };
        let mut results = vec![value];
        if matches!(self.arguments, Arguments::Fixed(_)) {
            results.push(self.b.block_params(self.return_block)[1]);
        }
        self.b.ins().return_(&results);
        self.terminated = true;
    }

    fn exit_frame(&mut self, value: ClifValue) {
        let caller = self.frame.unwrap();
        self.frame_code = true;
        self.increment(value);
        if self.slot_count > 0 {
            let release = self.b.create_block();
            let done = self.b.create_block();
            let roots = self.b.use_var(self.frame_roots.unwrap());
            self.brif(roots, release, &[], done, &[]);
            self.switch(release);
            if self.slot_count <= INLINE_FRAME_RELEASES {
                for slot in 0..self.slot_count {
                    self.release_slot(slot);
                }
            } else {
                self.call_helper(Helper::FrameRelease, &[]);
            }
            self.jump(done, &[]);
            self.switch(done);
        }
        self.clear_temporaries();
        let root = self.load_ctx(crate::jit::session::INDEX_ROOT_OFFSET);
        let rooted = self
            .b
            .ins()
            .icmp_imm_s(IntCC::NotEqual, root, Value::UNDEFINED.bits() as i64);
        let unroot = self.b.create_block();
        let done = self.b.create_block();
        self.b.set_cold_block(unroot);
        self.brif(rooted, unroot, &[], done, &[]);
        self.switch(unroot);
        self.call_helper(Helper::IndexUnroot, &[]);
        self.jump(done, &[]);
        self.switch(done);
        let start = self.load_ctx(crate::jit::session::LOCALS_START_OFFSET);
        self.store_ctx(start, crate::jit::session::LOCALS_LEN_OFFSET);
        self.store_ctx(caller.start, crate::jit::session::LOCALS_START_OFFSET);
        self.store_ctx(caller.base, crate::jit::session::LOCALS_OFFSET);
        self.store_ctx(caller.floor, crate::jit::session::TEMPS_FLOOR_OFFSET);
        self.frame_code = false;
    }

    fn release_slot(&mut self, slot: u32) {
        let flags = MemFlagsData::trusted();
        let base = self.load_ctx(crate::jit::session::LOCALS_OFFSET);
        let offset = (slot * 8) as i32;
        let value = self.b.ins().load(types::I64, flags, base, offset);
        let empty = self.constant(Value::UNDEFINED);
        let store = self.b.ins().store(flags, empty, base, offset);
        self.mark_frame_store(store);
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
            STACK_LIMIT_OFFSET, TEMPS_FLOOR_OFFSET, TEMPS_LEN_OFFSET,
        };
        let roots = self.b.declare_var(types::I8);
        let clean = self.b.ins().iconst(types::I8, 0);
        self.b.def_var(roots, clean);
        self.frame_roots = Some(roots);
        self.frame_code = true;
        let caller = CallerFrame {
            start: self.load_ctx(LOCALS_START_OFFSET),
            base: self.load_ctx(LOCALS_OFFSET),
            floor: self.load_ctx(TEMPS_FLOOR_OFFSET),
        };
        let start = self.load_ctx(LOCALS_LEN_OFFSET);
        let size = self.b.ins().iconst(types::I64, 0);
        self.frame_size = Some(self.b.func.dfg.value_def(size).unwrap_inst());
        let len = self.b.ins().iadd(start, size);
        let limit = self.load_ctx(LOCALS_LIMIT_OFFSET);
        let full = self.b.ins().icmp(IntCC::UnsignedGreaterThan, len, limit);
        let stack_pointer = self.b.ins().get_stack_pointer(self.pointer);
        let stack_limit = self.load_ctx(STACK_LIMIT_OFFSET);
        let deep = self
            .b
            .ins()
            .icmp(IntCC::UnsignedLessThan, stack_pointer, stack_limit);
        let overflow = self.b.ins().bor(full, deep);
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
        let temps = self.load_ctx(TEMPS_LEN_OFFSET);
        let floor = self.store_ctx(temps, TEMPS_FLOOR_OFFSET);
        self.annotations.temporary_resets.insert(floor);
        self.frame = Some(caller);
        self.frame_code = false;
    }

    fn store_ctx(&mut self, value: ClifValue, offset: i32) -> Inst {
        let store = self
            .b
            .ins()
            .store(MemFlagsData::trusted(), value, self.ctx, offset);
        self.mark_frame_store(store);
        store
    }

    pub(super) fn mark_frame_store(&mut self, store: Inst) {
        if self.frame_code {
            self.annotations.frame_stores.insert(store);
        }
    }

    fn patch_frame_size(&mut self) {
        if let Some(inst) = self.frame_size {
            self.b
                .func
                .replace(inst)
                .iconst(types::I64, self.slot_count as i64);
        }
    }

    pub(super) fn clear_temporaries(&mut self) {
        let flags = MemFlagsData::trusted();
        let len = self.b.ins().load(
            types::I64,
            flags,
            self.ctx,
            crate::jit::session::TEMPS_LEN_OFFSET,
        );
        let floor = self.b.ins().load(
            types::I64,
            flags,
            self.ctx,
            crate::jit::session::TEMPS_FLOOR_OFFSET,
        );
        let pending = self.b.ins().icmp(IntCC::NotEqual, len, floor);
        let clear = self.b.create_block();
        let done = self.b.create_block();
        self.brif(pending, clear, &[], done, &[]);
        let branch = self
            .b
            .func
            .layout
            .last_inst(self.b.current_block().unwrap())
            .unwrap();
        self.annotations.temporary_clears.insert(branch, done);
        self.switch(clear);
        let dummy = self.constant(Value::UNDEFINED);
        self.call_helper(Helper::TempClear, &[dummy]);
        self.jump(done, &[]);
        self.switch(done);
    }

    pub(super) fn argument_vector(&self) -> Result<(ClifValue, ClifValue), Error> {
        match self.arguments {
            Arguments::Vector { argv, argc } => Ok((argv, argc)),
            Arguments::Fixed(_) => Err(Error::Compile(
                "argument vector read in a fixed-arity function".into(),
            )),
        }
    }

    pub(super) fn load_arg(&mut self, index: usize) -> ClifValue {
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
        self.b.append_block_param(done, types::I64);
        self.brif(present, load_block, &[], missing_block, &[]);

        self.switch(load_block);
        let value = self.b.ins().load(
            types::I64,
            MemFlagsData::trusted(),
            argv,
            (index * 8) as i32,
        );
        self.jump(done, &[BlockArg::Value(value)]);

        self.switch(missing_block);
        let undefined = self.constant(Value::UNDEFINED);
        self.jump(done, &[BlockArg::Value(undefined)]);

        self.switch(done);
        self.b.block_params(done)[0]
    }
}

pub(super) mod class;
pub(in crate::jit) mod expression;
mod ir_helpers;
mod ops;
mod repr;
pub(in crate::jit) mod scope;
mod semantic_graph;
mod slots;
mod statement;

use std::collections::HashMap;

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{
    Block, BlockArg, Inst, InstBuilder, MemFlagsData, Type, Value as ClifValue, types,
};
use cranelift_frontend::{FunctionBuilder, Switch, Variable};
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Module};
use oxc_span::Span;
use oxc_syntax::symbol::SymbolId;

use crate::error::Error;
use crate::jit::clif::Annotations;
use crate::jit::helpers::{GlobalCache, Helper, PropertyCache};
use crate::value::Value;

pub(crate) use statement::statement_kind;

pub(super) enum FunctionKind {
    Ordinary,
    Generator(super::generator::ResumePlan),
    Factory(FuncId),
}

pub(super) struct Feedback<'a> {
    pub(super) get_refinement: bool,
    pub(super) owner: u64,
    pub(super) caches: &'a mut Vec<Box<super::planner::ObservationCache>>,
}

pub(super) struct FunctionCompiler<'a, 'f> {
    feedback: Option<Feedback<'a>>,
    names: &'a mut Vec<Box<str>>,
    cells: &'a mut Vec<Box<u64>>,
    property_caches: &'a mut Vec<Box<PropertyCache>>,
    global_caches: &'a mut Vec<Box<GlobalCache>>,
    source_name: (usize, usize),
    source_span: Span,
    b: &'a mut FunctionBuilder<'f>,
    module: &'a mut JITModule,
    helpers: &'a [FuncId],
    annotations: &'a mut Annotations,
    pointer: Type,
    ctx: ClifValue,
    receiver: ClifValue,
    argv: ClifValue,
    argc: ClifValue,
    return_block: Block,
    handlers: Vec<Block>,
    generator: Option<ClifValue>,
    resume_blocks: Vec<Block>,
    suspensions: HashMap<u32, (usize, Block)>,
    terminated: bool,
    frame_size: Option<Inst>,
    frame_roots: Option<Variable>,
    slots: HashMap<u32, u32>,
    slot_count: u32,
}

pub(super) fn build_function(
    b: &mut FunctionBuilder<'_>,
    module: &mut JITModule,
    helpers: &[FuncId],
    annotations: &mut Annotations,
    names: &mut Vec<Box<str>>,
    cells: &mut Vec<Box<u64>>,
    property_caches: &mut Vec<Box<PropertyCache>>,
    global_caches: &mut Vec<Box<GlobalCache>>,
    source_name: (usize, usize),
    params: Span,
    _closures: &HashMap<u32, (FuncId, Vec<SymbolId>, bool)>,
    kind: FunctionKind,
    graph: &super::semantic::Graph,
    feedback: Option<Feedback<'_>>,
) -> Result<(), Error> {
    build_callable(
        b,
        module,
        helpers,
        annotations,
        names,
        cells,
        property_caches,
        global_caches,
        source_name,
        params,
        kind,
        CallableBody::Graph(graph),
        feedback,
    )
}

pub(super) fn build_factory(
    b: &mut FunctionBuilder<'_>,
    module: &mut JITModule,
    helpers: &[FuncId],
    annotations: &mut Annotations,
    names: &mut Vec<Box<str>>,
    cells: &mut Vec<Box<u64>>,
    property_caches: &mut Vec<Box<PropertyCache>>,
    global_caches: &mut Vec<Box<GlobalCache>>,
    source_name: (usize, usize),
    params: Span,
    _closures: &HashMap<u32, (FuncId, Vec<SymbolId>, bool)>,
    resume: FuncId,
) -> Result<(), Error> {
    build_callable(
        b,
        module,
        helpers,
        annotations,
        names,
        cells,
        property_caches,
        global_caches,
        source_name,
        params,
        FunctionKind::Factory(resume),
        CallableBody::Factory(resume),
        None,
    )
}

enum CallableBody<'g> {
    Graph(&'g super::semantic::Graph),
    Factory(FuncId),
}

fn build_callable(
    b: &mut FunctionBuilder<'_>,
    module: &mut JITModule,
    helpers: &[FuncId],
    annotations: &mut Annotations,
    names: &mut Vec<Box<str>>,
    cells: &mut Vec<Box<u64>>,
    property_caches: &mut Vec<Box<PropertyCache>>,
    global_caches: &mut Vec<Box<GlobalCache>>,
    source_name: (usize, usize),
    params: Span,
    kind: FunctionKind,
    body: CallableBody<'_>,
    feedback: Option<Feedback<'_>>,
) -> Result<(), Error> {
    let entry = b.create_block();
    b.switch_to_block(entry);
    b.append_block_params_for_function_params(entry);
    let [ctx, closure, receiver, argv, argc] = b.block_params(entry).try_into().unwrap();
    let return_block = b.create_block();
    b.append_block_param(return_block, types::I64);

    let pointer = module.target_config().pointer_type();
    let mut compiler = FunctionCompiler {
        feedback,
        names,
        cells,
        property_caches,
        global_caches,
        source_name,
        source_span: params,
        b,
        module,
        helpers,
        annotations,
        pointer,
        ctx,
        receiver,
        argv,
        argc,
        return_block,
        handlers: Vec::new(),
        generator: None,
        resume_blocks: Vec::new(),
        suspensions: HashMap::new(),
        terminated: false,
        frame_size: None,
        frame_roots: None,
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
        let args = compiler.call_helper(Helper::ArrayNew, &[argv, argc]);
        compiler.check_error();
        let code = compiler.b.ins().iconst(types::I64, resume.as_u32() as i64);
        let generator =
            compiler.call_helper(Helper::GeneratorNew, &[closure, receiver, args, code]);
        compiler.check_error();
        compiler.jump(return_block, &[BlockArg::Value(generator)]);
        compiler.switch(return_block);
        let result = compiler.b.block_params(return_block)[0];
        compiler.return_value(result);
        return Ok(());
    }
    if let CallableBody::Graph(graph) = body {
        let graph_closure = if compiler.generator.is_some() {
            compiler.call_helper(Helper::GeneratorClosure, &[closure])
        } else {
            closure
        };
        compiler.emit_semantic_graph(graph, graph_closure)?;
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
            compiler.jump(return_block, &[BlockArg::Value(undefined)]);
        }
        return Ok(());
    }
    Ok(())
}

pub(super) fn build_global(
    b: &mut FunctionBuilder<'_>,
    module: &mut JITModule,
    helpers: &[FuncId],
    annotations: &mut Annotations,
    names: &mut Vec<Box<str>>,
    cells: &mut Vec<Box<u64>>,
    property_caches: &mut Vec<Box<PropertyCache>>,
    global_caches: &mut Vec<Box<GlobalCache>>,
    source_name: (usize, usize),
    _closures: &HashMap<u32, (FuncId, Vec<SymbolId>, bool)>,
    graph: &super::semantic::Graph,
) -> Result<(), Error> {
    let entry = b.create_block();
    b.switch_to_block(entry);
    b.append_block_params_for_function_params(entry);
    let [ctx, closure, receiver, argv, argc] = b.block_params(entry).try_into().unwrap();
    let return_block = b.create_block();
    b.append_block_param(return_block, types::I64);
    let pointer = module.target_config().pointer_type();
    let mut compiler = FunctionCompiler {
        feedback: None,
        names,
        cells,
        property_caches,
        global_caches,
        source_name,
        source_span: Span::new(0, 0),
        b,
        module,
        helpers,
        annotations,
        pointer,
        ctx,
        receiver,
        argv,
        argc,
        return_block,
        handlers: Vec::new(),
        generator: None,
        resume_blocks: Vec::new(),
        suspensions: HashMap::new(),
        terminated: false,
        frame_size: None,
        frame_roots: None,
        slots: HashMap::new(),
        slot_count: 0,
    };
    compiler.enter_frame();
    compiler.emit_semantic_graph(graph, closure)?;
    compiler.patch_frame_size();
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

    pub(super) fn return_value(&mut self, value: ClifValue) {
        let value = match self.generator {
            Some(generator) => self.call_helper(Helper::GeneratorComplete, &[generator, value]),
            None => match self.frame_roots {
                Some(roots) => {
                    let rooted = self.b.create_block();
                    let unrooted = self.b.create_block();
                    let done = self.b.create_block();
                    self.b.append_block_param(done, types::I64);
                    let roots = self.b.use_var(roots);
                    self.brif(roots, rooted, &[], unrooted, &[]);
                    self.switch(rooted);
                    let result = self.call_helper(Helper::FrameExit, &[value]);
                    self.jump(done, &[BlockArg::Value(result)]);
                    self.switch(unrooted);
                    let result = self.call_helper(Helper::FrameExitUnrooted, &[value]);
                    self.jump(done, &[BlockArg::Value(result)]);
                    self.switch(done);
                    self.b.block_params(done)[0]
                }
                None => self.call_helper(Helper::FrameExit, &[value]),
            },
        };
        self.b.ins().return_(&[value]);
        self.terminated = true;
    }

    fn enter_frame(&mut self) {
        let roots = self.b.declare_var(types::I8);
        let clean = self.b.ins().iconst(types::I8, 0);
        self.b.def_var(roots, clean);
        self.frame_roots = Some(roots);
        let size = self.b.ins().iconst(types::I64, 0);
        self.frame_size = Some(self.b.func.dfg.value_def(size).unwrap_inst());
        self.call_helper(Helper::FrameEnter, &[size]);
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

    pub(super) fn load_arg(&mut self, index: usize) -> ClifValue {
        let present = self
            .b
            .ins()
            .icmp_imm_s(IntCC::UnsignedGreaterThan, self.argc, index as i64);
        let load_block = self.b.create_block();
        let missing_block = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param(done, types::I64);
        self.brif(present, load_block, &[], missing_block, &[]);

        self.switch(load_block);
        let value = self.b.ins().load(
            types::I64,
            MemFlagsData::trusted(),
            self.argv,
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

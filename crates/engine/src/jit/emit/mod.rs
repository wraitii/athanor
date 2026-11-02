mod expression;
mod loops;
mod ops;
mod scope;
mod statement;

use std::collections::{HashMap, HashSet};

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{
    Block, BlockArg, InstBuilder, MemFlags, Type, Value as ClifValue, types,
};
use cranelift_frontend::{FunctionBuilder, Switch};
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Module};
use oxc_ast::ast::{Expression, FormalParameters, Statement};
use oxc_semantic::Scoping;
use oxc_syntax::symbol::SymbolId;

use crate::error::Error;
use crate::jit::helpers::Helper;
use crate::value::Value;

use scope::ScopeStack;

pub(crate) use statement::statement_kind;

pub(super) enum FunctionKind {
    Ordinary,
    Generator(super::generator::ResumePlan),
    Factory(FuncId),
}

pub(super) struct FunctionCompiler<'a, 'f> {
    names: &'a mut Vec<Box<str>>,
    b: &'a mut FunctionBuilder<'f>,
    module: &'a mut JITModule,
    helpers: &'a [FuncId],
    scoping: &'a Scoping,
    pointer: Type,
    ctx: ClifValue,
    receiver: ClifValue,
    argv: ClifValue,
    argc: ClifValue,
    captured: &'a HashSet<SymbolId>,
    closures: &'a HashMap<u32, (FuncId, Vec<SymbolId>, bool)>,
    scopes: ScopeStack,
    return_block: Block,
    handlers: Vec<Block>,
    loops: Vec<(Block, Block)>,
    generator: Option<ClifValue>,
    resume_blocks: Vec<Block>,
    suspensions: HashMap<u32, (usize, Block)>,
    terminated: bool,
}

pub(super) fn build_function(
    b: &mut FunctionBuilder<'_>,
    module: &mut JITModule,
    scoping: &Scoping,
    helpers: &[FuncId],
    names: &mut Vec<Box<str>>,
    params: &FormalParameters<'_>,
    statements: &[Statement<'_>],
    expression: Option<&Expression<'_>>,
    captures: &[SymbolId],
    captured: &HashSet<SymbolId>,
    closures: &HashMap<u32, (FuncId, Vec<SymbolId>, bool)>,
    kind: FunctionKind,
) -> Result<(), Error> {
    let entry = b.create_block();
    b.switch_to_block(entry);
    b.append_block_params_for_function_params(entry);
    let [ctx, closure, receiver, argv, argc] = b.block_params(entry).try_into().unwrap();
    let return_block = b.create_block();
    b.append_block_param(return_block, types::I64);

    let pointer = module.target_config().pointer_type();
    let mut compiler = FunctionCompiler {
        names,
        b,
        module,
        helpers,
        scoping,
        pointer,
        ctx,
        receiver,
        argv,
        argc,
        captured,
        closures,
        scopes: ScopeStack::new(),
        return_block,
        handlers: Vec::new(),
        loops: Vec::new(),
        generator: None,
        resume_blocks: Vec::new(),
        suspensions: HashMap::new(),
        terminated: false,
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
    if let FunctionKind::Factory(resume) = kind {
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
    compiler.initialize_function(params, statements)?;
    let closure = if compiler.generator.is_some() {
        compiler.call_helper(Helper::GeneratorClosure, &[closure])
    } else {
        closure
    };
    for (index, symbol) in captures.iter().enumerate() {
        let index = compiler.b.ins().iconst(types::I64, index as i64);
        let cell = compiler.call_helper(Helper::CaptureGet, &[closure, index]);
        compiler.check_error();
        compiler.install_capture(*symbol, cell);
    }
    compiler.clear_temporaries();
    if let Some(expression) = expression {
        let value = compiler.emit_expression(expression)?;
        compiler.jump(return_block, &[BlockArg::Value(value)]);
    } else {
        compiler.emit_block(statements)?;
    }
    if !compiler.terminated {
        let undefined = compiler.constant(Value::UNDEFINED);
        compiler.jump(return_block, &[BlockArg::Value(undefined)]);
    }
    compiler.switch(return_block);
    let result = compiler.b.block_params(return_block)[0];
    compiler.return_value(result);
    if let Some(dispatch) = dispatch {
        compiler.switch(dispatch);
        let dummy = compiler.constant(Value::UNDEFINED);
        let state = compiler.call_helper(Helper::GeneratorState, &[dummy]);
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
    Ok(())
}

pub(super) fn build_global(
    b: &mut FunctionBuilder<'_>,
    module: &mut JITModule,
    scoping: &Scoping,
    helpers: &[FuncId],
    names: &mut Vec<Box<str>>,
    statements: &[oxc_ast::ast::Statement<'_>],
    captured: &HashSet<SymbolId>,
    closures: &HashMap<u32, (FuncId, Vec<SymbolId>, bool)>,
) -> Result<(), Error> {
    let entry = b.create_block();
    b.switch_to_block(entry);
    b.append_block_params_for_function_params(entry);
    let [ctx, _closure, receiver, argv, argc] = b.block_params(entry).try_into().unwrap();
    let return_block = b.create_block();
    b.append_block_param(return_block, types::I64);
    let pointer = module.target_config().pointer_type();
    let mut compiler = FunctionCompiler {
        names,
        b,
        module,
        helpers,
        scoping,
        pointer,
        ctx,
        receiver,
        argv,
        argc,
        captured,
        closures,
        scopes: ScopeStack::new(),
        return_block,
        handlers: Vec::new(),
        loops: Vec::new(),
        generator: None,
        resume_blocks: Vec::new(),
        suspensions: HashMap::new(),
        terminated: false,
    };
    compiler.enter_frame();
    for statement in statements {
        if let oxc_ast::ast::Statement::ExpressionStatement(expression) = statement {
            compiler.emit_expression(&expression.expression)?;
            compiler.clear_temporaries();
        }
        if let oxc_ast::ast::Statement::VariableDeclaration(declaration) = statement {
            for declarator in &declaration.declarations {
                if let Some(initializer) = &declarator.init {
                    let oxc_ast::ast::BindingPattern::BindingIdentifier(identifier) =
                        &declarator.id
                    else {
                        unreachable!()
                    };
                    let value = compiler.emit_expression(initializer)?;
                    compiler.global_access(identifier.name.as_str(), Some(value));
                    compiler.clear_temporaries();
                }
            }
        }
    }
    let undefined = compiler.constant(Value::UNDEFINED);
    compiler.jump(return_block, &[BlockArg::Value(undefined)]);
    compiler.switch(return_block);
    let result = compiler.b.block_params(return_block)[0];
    compiler.return_value(result);
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
            None => self.call_helper(Helper::FrameExit, &[value]),
        };
        self.b.ins().return_(&[value]);
        self.terminated = true;
    }

    fn enter_frame(&mut self) {
        let dummy = self.constant(Value::UNDEFINED);
        self.call_helper(Helper::FrameEnter, &[dummy]);
    }

    pub(super) fn clear_temporaries(&mut self) {
        let dummy = self.constant(Value::UNDEFINED);
        self.call_helper(Helper::TempClear, &[dummy]);
    }

    pub(super) fn load_arg(&mut self, index: usize) -> ClifValue {
        let present = self
            .b
            .ins()
            .icmp_imm(IntCC::UnsignedGreaterThan, self.argc, index as i64);
        let load_block = self.b.create_block();
        let missing_block = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param(done, types::I64);
        self.brif(present, load_block, &[], missing_block, &[]);

        self.switch(load_block);
        let value = self.b.ins().load(
            types::I64,
            MemFlags::trusted(),
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

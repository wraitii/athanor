mod expression;
mod ops;
mod scope;
mod statement;

use std::collections::HashMap;

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{
    Block, BlockArg, InstBuilder, MemFlags, Type, Value as ClifValue, types,
};
use cranelift_frontend::FunctionBuilder;
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Module};
use oxc_ast::ast::Function;
use oxc_semantic::Scoping;
use oxc_syntax::symbol::SymbolId;

use crate::error::Error;
use crate::jit::helpers::Helper;
use crate::value::Value;

use scope::ScopeStack;

pub(crate) use statement::statement_kind;

pub(super) struct FunctionCompiler<'a, 'f> {
    names: &'a mut Vec<Box<str>>,
    b: &'a mut FunctionBuilder<'f>,
    module: &'a mut JITModule,
    functions: &'a HashMap<String, FuncId>,
    helpers: &'a [FuncId],
    scoping: &'a Scoping,
    function_symbols: &'a HashMap<SymbolId, FuncId>,
    pointer: Type,
    ctx: ClifValue,
    argv: ClifValue,
    argc: ClifValue,
    scopes: ScopeStack,
    return_block: Block,
    terminated: bool,
}

pub(super) fn build_function(
    b: &mut FunctionBuilder<'_>,
    module: &mut JITModule,
    functions: &HashMap<String, FuncId>,
    scoping: &Scoping,
    function_symbols: &HashMap<SymbolId, FuncId>,
    helpers: &[FuncId],
    names: &mut Vec<Box<str>>,
    function: &Function<'_>,
) -> Result<(), Error> {
    let entry = b.create_block();
    b.switch_to_block(entry);
    b.append_block_params_for_function_params(entry);
    let [ctx, argv, argc] = b.block_params(entry).try_into().unwrap();
    let return_block = b.create_block();
    b.append_block_param(return_block, types::I64);

    let pointer = module.target_config().pointer_type();
    let mut compiler = FunctionCompiler {
        names,
        b,
        module,
        functions,
        helpers,
        scoping,
        function_symbols,
        pointer,
        ctx,
        argv,
        argc,
        scopes: ScopeStack::new(),
        return_block,
        terminated: false,
    };

    let body = function.body.as_ref().unwrap();
    compiler.enter_frame();
    compiler.initialize_function(function)?;
    compiler.clear_temporaries();
    compiler.emit_block(&body.statements)?;
    if !compiler.terminated {
        let undefined = compiler.constant(Value::UNDEFINED);
        compiler.jump(return_block, &[BlockArg::Value(undefined)]);
    }
    compiler.switch(return_block);
    let result = compiler.b.block_params(return_block)[0];
    compiler.return_value(result);
    Ok(())
}

pub(super) fn build_global(
    b: &mut FunctionBuilder<'_>,
    module: &mut JITModule,
    functions: &HashMap<String, FuncId>,
    scoping: &Scoping,
    function_symbols: &HashMap<SymbolId, FuncId>,
    helpers: &[FuncId],
    names: &mut Vec<Box<str>>,
    statements: &[oxc_ast::ast::Statement<'_>],
) -> Result<(), Error> {
    let entry = b.create_block();
    b.switch_to_block(entry);
    b.append_block_params_for_function_params(entry);
    let [ctx, argv, argc] = b.block_params(entry).try_into().unwrap();
    let return_block = b.create_block();
    b.append_block_param(return_block, types::I64);
    let pointer = module.target_config().pointer_type();
    let mut compiler = FunctionCompiler {
        names,
        b,
        module,
        functions,
        helpers,
        scoping,
        function_symbols,
        pointer,
        ctx,
        argv,
        argc,
        scopes: ScopeStack::new(),
        return_block,
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
        let value = self.call_helper(Helper::FrameExit, &[value]);
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

use std::collections::HashMap;

use cranelift_codegen::Context;
use cranelift_codegen::ir::{AbiParam, Signature, types};
use cranelift_frontend::FunctionBuilderContext;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module, default_libcall_names};
use oxc_ast::ast::{BindingPattern, Function, Program, Statement, VariableDeclarationKind};
use oxc_semantic::Scoping;
use oxc_span::GetSpan;
use oxc_syntax::symbol::SymbolId;

use crate::EngineOptions;
use crate::error::{Error, compile_at, unsupported_at};

use super::{emit, helpers};

pub(crate) struct CompiledFunction {
    pub(crate) name: String,
    pub(crate) entry: usize,
    pub(crate) arity: usize,
}

pub(crate) struct Compiler {
    names: Vec<Box<str>>,
    module: JITModule,
    signature: Signature,
    helpers: Vec<FuncId>,
    functions: HashMap<String, FuncId>,
    next_id: u64,
    context: Context,
    builder_context: FunctionBuilderContext,
}

impl Compiler {
    pub(crate) fn new(options: EngineOptions) -> Result<Self, Error> {
        let flags = [("opt_level", options.opt_level.as_flag())];
        let mut builder = JITBuilder::with_flags(&flags, default_libcall_names())
            .map_err(|error| Error::Compile(error.to_string()))?;
        builder.symbols(helpers::symbols());
        let mut module = JITModule::new(builder);
        let target = module.target_config();
        let pointer = target.pointer_type();
        let call_conv = target.default_call_conv;
        let mut signature = Signature::new(call_conv);
        signature.params.push(AbiParam::new(pointer));
        signature.params.push(AbiParam::new(pointer));
        signature.params.push(AbiParam::new(types::I64));
        signature.returns.push(AbiParam::new(types::I64));

        let mut helper_ids = Vec::with_capacity(helpers::HELPERS.len());
        for spec in helpers::HELPERS {
            let signature = helpers::signature(spec.shape, call_conv, pointer);
            let id = module
                .declare_function(spec.name, Linkage::Import, &signature)
                .map_err(|error| Error::Compile(error.to_string()))?;
            helper_ids.push(id);
        }

        let context = module.make_context();
        Ok(Self {
            names: Vec::new(),
            module,
            signature,
            helpers: helper_ids,
            functions: HashMap::new(),
            next_id: 0,
            context,
            builder_context: FunctionBuilderContext::new(),
        })
    }

    pub(crate) fn compile_program<'a>(
        &mut self,
        program: &'a Program<'a>,
        scoping: &Scoping,
    ) -> Result<(Vec<CompiledFunction>, Vec<String>, Option<usize>), Error> {
        let result = self.compile_program_inner(program, scoping);
        if result.is_err() {
            self.context.clear();
            self.builder_context = FunctionBuilderContext::new();
        }
        result
    }

    fn compile_program_inner<'a>(
        &mut self,
        program: &'a Program<'a>,
        scoping: &Scoping,
    ) -> Result<(Vec<CompiledFunction>, Vec<String>, Option<usize>), Error> {
        let mut globals = Vec::new();
        let mut declarations = Vec::new();
        for statement in &program.body {
            match statement {
                Statement::FunctionDeclaration(function) => {
                    let name = validate_function(function)?;
                    declarations.push((name, function));
                }
                Statement::EmptyStatement(_) | Statement::ExpressionStatement(_) => {}
                Statement::VariableDeclaration(declaration)
                    if declaration.kind == VariableDeclarationKind::Var =>
                {
                    for declarator in &declaration.declarations {
                        let BindingPattern::BindingIdentifier(identifier) = &declarator.id else {
                            return Err(unsupported_at(
                                declarator.span,
                                "destructuring declaration",
                            ));
                        };
                        globals.push(identifier.name.to_string());
                    }
                }
                other => {
                    return Err(unsupported_at(
                        other.span(),
                        format!("{} statement at top level", emit::statement_kind(other)),
                    ));
                }
            }
        }

        let mut ids = self.functions.clone();
        let mut ordered = Vec::new();
        let mut function_symbols = HashMap::<SymbolId, FuncId>::new();
        for (name, function) in declarations {
            if ids.contains_key(&name) {
                return Err(compile_at(
                    function.span,
                    format!("duplicate function `{name}`"),
                ));
            }
            let internal = format!("jsfn_{}_{}", self.next_id, name);
            self.next_id += 1;
            let id = self
                .module
                .declare_function(&internal, Linkage::Local, &self.signature)
                .map_err(|error| Error::Compile(error.to_string()))?;
            ids.insert(name.clone(), id);
            function_symbols.insert(function.id.as_ref().unwrap().symbol_id.get().unwrap(), id);
            ordered.push((name, function));
        }

        let mut pending = Vec::new();
        for (name, function) in &ordered {
            let code = self.compile_function(function, &ids, scoping, &function_symbols)?;
            pending.push((ids[name], code));
        }

        let initializer = if !program.body.iter().any(|s| {
            matches!(
                s,
                Statement::VariableDeclaration(_) | Statement::ExpressionStatement(_)
            )
        }) {
            None
        } else {
            let internal = format!("jsinit_{}", self.next_id);
            self.next_id += 1;
            let id = self
                .module
                .declare_function(&internal, Linkage::Local, &self.signature)
                .map_err(|error| Error::Compile(error.to_string()))?;
            self.context.clear();
            self.context.func.signature = self.signature.clone();
            {
                let mut builder = cranelift_frontend::FunctionBuilder::new(
                    &mut self.context.func,
                    &mut self.builder_context,
                );
                emit::build_global(
                    &mut builder,
                    &mut self.module,
                    &ids,
                    scoping,
                    &function_symbols,
                    &self.helpers,
                    &mut self.names,
                    &program.body,
                )?;
                builder.seal_all_blocks();
                builder.finalize();
            }
            pending.push((
                id,
                std::mem::replace(
                    &mut self.context.func,
                    cranelift_codegen::ir::Function::new(),
                ),
            ));
            Some(id)
        };

        for (id, code) in pending {
            self.context.clear();
            self.context.func = code;
            self.module
                .define_function(id, &mut self.context)
                .map_err(|error| Error::Compile(error.to_string()))?;
        }
        self.module
            .finalize_definitions()
            .map_err(|error| Error::Compile(error.to_string()))?;

        let mut functions = Vec::with_capacity(ordered.len());
        for (name, function) in ordered {
            let entry = self.module.get_finalized_function(ids[&name]);
            functions.push(CompiledFunction {
                name,
                entry: entry as usize,
                arity: function.params.items.len(),
            });
        }
        self.functions = ids;
        Ok((
            functions,
            globals,
            initializer.map(|id| self.module.get_finalized_function(id) as usize),
        ))
    }

    fn compile_function(
        &mut self,
        function: &Function<'_>,
        functions: &HashMap<String, FuncId>,
        scoping: &Scoping,
        function_symbols: &HashMap<SymbolId, FuncId>,
    ) -> Result<cranelift_codegen::ir::Function, Error> {
        self.context.clear();
        self.context.func.signature = self.signature.clone();
        {
            let mut builder = cranelift_frontend::FunctionBuilder::new(
                &mut self.context.func,
                &mut self.builder_context,
            );
            emit::build_function(
                &mut builder,
                &mut self.module,
                functions,
                scoping,
                function_symbols,
                &self.helpers,
                &mut self.names,
                function,
            )?;
            builder.seal_all_blocks();
            builder.finalize();
        }
        Ok(std::mem::replace(
            &mut self.context.func,
            cranelift_codegen::ir::Function::new(),
        ))
    }
}

fn validate_function(function: &Function<'_>) -> Result<String, Error> {
    let name = function
        .id
        .as_ref()
        .map(|identifier| identifier.name.to_string())
        .ok_or_else(|| compile_at(function.span, "function declaration has no name"))?;
    if function.generator || function.r#async {
        return Err(unsupported_at(
            function.span,
            format!("generator and async function `{name}`"),
        ));
    }
    if function.body.is_none() {
        return Err(unsupported_at(
            function.span,
            format!("function `{name}` has no body"),
        ));
    }
    if let Some(rest) = &function.params.rest
        && !matches!(rest.rest.argument, BindingPattern::BindingIdentifier(_))
    {
        return Err(unsupported_at(
            rest.span,
            format!("rest parameter pattern in function `{name}`"),
        ));
    }
    for parameter in &function.params.items {
        if parameter.initializer.is_some() {
            return Err(unsupported_at(
                parameter.span,
                format!("default parameter in function `{name}`"),
            ));
        }
        if !matches!(parameter.pattern, BindingPattern::BindingIdentifier(_)) {
            return Err(unsupported_at(
                parameter.span,
                format!("parameter pattern in function `{name}`"),
            ));
        }
    }
    Ok(name)
}

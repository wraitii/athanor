use std::collections::{HashMap, HashSet};

use cranelift_codegen::Context;
use cranelift_codegen::ir::{AbiParam, Signature, types};
use cranelift_frontend::FunctionBuilderContext;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module, default_libcall_names};
use oxc_ast::AstKind;
use oxc_ast::ast::{
    BindingPattern, Expression, FormalParameters, Function, Program, Statement,
    VariableDeclarationKind,
};
use oxc_semantic::{AstNodes, Scoping};
use oxc_span::GetSpan;
use oxc_syntax::symbol::SymbolId;

use crate::EngineOptions;
use crate::error::{Error, compile_at, unsupported_at};

use super::{analysis::CaptureAnalysis, emit, helpers};

pub(crate) struct CompiledFunction {
    pub(crate) name: String,
    pub(crate) arity: usize,
    pub(crate) code: u64,
    pub(crate) generator: bool,
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
        signature.params.push(AbiParam::new(types::I64));
        signature.params.push(AbiParam::new(types::I64));
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
        nodes: &AstNodes<'a>,
    ) -> Result<
        (
            Vec<CompiledFunction>,
            Vec<String>,
            Option<usize>,
            Vec<(u64, usize)>,
        ),
        Error,
    > {
        let result = self.compile_program_inner(program, scoping, nodes);
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
        nodes: &AstNodes<'a>,
    ) -> Result<
        (
            Vec<CompiledFunction>,
            Vec<String>,
            Option<usize>,
            Vec<(u64, usize)>,
        ),
        Error,
    > {
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
            ordered.push((name, function));
        }

        let analysis = CaptureAnalysis::analyze(program, scoping);
        let mut closures = HashMap::new();
        let mut nested = Vec::new();
        for node in nodes.iter() {
            let (span, params, statements, expression, generator) = match node.kind() {
                AstKind::ArrowFunctionExpression(function) => {
                    if function.r#async {
                        return Err(unsupported_at(function.span, "async arrow"));
                    }
                    let (statements, expression) = match &function.body {
                        oxc_ast::ast::ArrowFunctionBody::FunctionBody(body) => {
                            (body.statements.as_slice(), None)
                        }
                        body => (&[][..], body.as_expression()),
                    };
                    (
                        function.span,
                        &*function.params,
                        statements,
                        expression,
                        false,
                    )
                }
                AstKind::Function(function)
                    if !ordered.iter().any(|(_, top)| top.span == function.span) =>
                {
                    if function.id.is_some() || function.r#async || function.body.is_none() {
                        return Err(unsupported_at(
                            function.span,
                            "named, async or bodyless nested function",
                        ));
                    }
                    (
                        function.span,
                        &*function.params,
                        function.body.as_ref().unwrap().statements.as_slice(),
                        None,
                        function.generator,
                    )
                }
                _ => continue,
            };
            validate_params(params, span)?;
            let internal = format!("jsclosure_{}", self.next_id);
            self.next_id += 1;
            let id = self
                .module
                .declare_function(&internal, Linkage::Local, &self.signature)
                .map_err(|error| Error::Compile(error.to_string()))?;
            let captures = analysis
                .functions
                .get(&span.start)
                .map(|f| f.captures.clone())
                .unwrap_or_default();
            let lexical_this = analysis
                .functions
                .get(&span.start)
                .is_some_and(|function| function.lexical_this);
            closures.insert(span.start, (id, captures, lexical_this));
            nested.push((id, span, params, statements, expression, generator));
        }
        let mut pending = Vec::new();
        for (name, function) in &ordered {
            let (code, resume) = self.compile_callable(
                &function.params,
                &function.body.as_ref().unwrap().statements,
                None,
                &[],
                &analysis.captured,
                &closures,
                scoping,
                function.generator,
            )?;
            if let Some(resume) = resume {
                pending.push(resume);
            }
            pending.push((ids[name], code));
        }
        for (id, span, params, statements, expression, generator) in nested {
            let captures = &closures[&span.start].1;
            let (code, resume) = self.compile_callable(
                params,
                statements,
                expression,
                captures,
                &analysis.captured,
                &closures,
                scoping,
                generator,
            )?;
            if let Some(resume) = resume {
                pending.push(resume);
            }
            pending.push((id, code));
        }
        let compiled_ids: Vec<_> = pending.iter().map(|(id, _)| *id).collect();

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
                    scoping,
                    &self.helpers,
                    &mut self.names,
                    &program.body,
                    &analysis.captured,
                    &closures,
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
            functions.push(CompiledFunction {
                name: name.clone(),
                arity: function.params.items.len(),
                code: ids[&name].as_u32() as u64,
                generator: function.generator,
            });
        }
        self.functions = ids;
        Ok((
            functions,
            globals,
            initializer.map(|id| self.module.get_finalized_function(id) as usize),
            compiled_ids
                .into_iter()
                .map(|id| {
                    (
                        id.as_u32() as u64,
                        self.module.get_finalized_function(id) as usize,
                    )
                })
                .collect(),
        ))
    }

    fn compile_callable(
        &mut self,
        params: &FormalParameters<'_>,
        statements: &[Statement<'_>],
        expression: Option<&Expression<'_>>,
        captures: &[SymbolId],
        captured: &HashSet<SymbolId>,
        closures: &HashMap<u32, (FuncId, Vec<SymbolId>, bool)>,
        scoping: &Scoping,
        generator: bool,
    ) -> Result<
        (
            cranelift_codegen::ir::Function,
            Option<(FuncId, cranelift_codegen::ir::Function)>,
        ),
        Error,
    > {
        if !generator {
            return Ok((
                self.compile_function(
                    params,
                    statements,
                    expression,
                    captures,
                    captured,
                    closures,
                    scoping,
                    emit::FunctionKind::Ordinary,
                )?,
                None,
            ));
        }
        let plan = super::generator::ResumePlan::lower(statements)?;
        let internal = format!("jsresume_{}", self.next_id);
        self.next_id += 1;
        let id = self
            .module
            .declare_function(&internal, Linkage::Local, &self.signature)
            .map_err(|error| Error::Compile(error.to_string()))?;
        let resume = self.compile_function(
            params,
            statements,
            None,
            captures,
            captured,
            closures,
            scoping,
            emit::FunctionKind::Generator(plan),
        )?;
        let factory = self.compile_function(
            params,
            &[],
            None,
            &[],
            captured,
            closures,
            scoping,
            emit::FunctionKind::Factory(id),
        )?;
        Ok((factory, Some((id, resume))))
    }

    fn compile_function(
        &mut self,
        params: &FormalParameters<'_>,
        statements: &[Statement<'_>],
        expression: Option<&Expression<'_>>,
        captures: &[SymbolId],
        captured: &HashSet<SymbolId>,
        closures: &HashMap<u32, (FuncId, Vec<SymbolId>, bool)>,
        scoping: &Scoping,
        kind: emit::FunctionKind,
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
                scoping,
                &self.helpers,
                &mut self.names,
                params,
                statements,
                expression,
                captures,
                captured,
                closures,
                kind,
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
    if function.r#async {
        return Err(unsupported_at(
            function.span,
            format!("async function `{name}`"),
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

fn validate_params(params: &FormalParameters<'_>, span: oxc_span::Span) -> Result<(), Error> {
    if params.items.iter().any(|p| {
        p.initializer.is_some() || !matches!(p.pattern, BindingPattern::BindingIdentifier(_))
    }) || params
        .rest
        .as_ref()
        .is_some_and(|p| !matches!(p.rest.argument, BindingPattern::BindingIdentifier(_)))
    {
        return Err(unsupported_at(
            span,
            "function parameter pattern or default",
        ));
    }
    Ok(())
}

use std::collections::{BTreeMap, HashMap, HashSet};

use cranelift_codegen::Context;
use cranelift_codegen::ir::{AbiParam, Signature, SourceLoc, types};
use cranelift_frontend::FunctionBuilderContext;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module, default_libcall_names};
use oxc_ast::AstKind;
use oxc_ast::ast::{
    AssignmentTarget, BindingPattern, Expression, FormalParameters, Function, Program, PropertyKey,
    Statement, VariableDeclarationKind,
};
use oxc_semantic::{AstNodes, Scoping};
use oxc_span::GetSpan;
use oxc_syntax::symbol::SymbolId;

use crate::EngineOptions;
use crate::error::{Error, compile_at, unsupported_at};

use super::numeric::{Annotations, FunctionBody};
use super::{analysis::CaptureAnalysis, emit, helpers, jitdump, numeric};

pub(crate) struct CompiledFunction {
    pub(crate) name: String,
    pub(crate) arity: usize,
    pub(crate) code: u64,
    pub(crate) generator: bool,
}

pub(crate) struct Compiler {
    names: Vec<Box<str>>,
    source_name: (usize, usize),
    module: JITModule,
    signature: Signature,
    helpers: Vec<FuncId>,
    functions: HashMap<String, FuncId>,
    function_names: HashMap<FuncId, String>,
    dump: Option<jitdump::Writer>,
    dump_ir: bool,
    ssa_opt: bool,
    ssa_dump: bool,
    compile_dump: bool,
    compile_ir: bool,
    compile_ast: bool,
    compile_asm: bool,
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
        let dump = options.jitdump.then(jitdump::Writer::new).flatten();
        let dump_ir = dump.is_some() && options.jitdump_ir;
        Ok(Self {
            names: Vec::new(),
            source_name: (0, 0),
            module,
            signature,
            helpers: helper_ids,
            functions: HashMap::new(),
            function_names: HashMap::new(),
            dump,
            dump_ir,
            ssa_opt: options.ssa_opt,
            ssa_dump: options.ssa_dump,
            compile_dump: options.compile_dump,
            compile_ir: options.compile_ir,
            compile_ast: options.compile_ast,
            compile_asm: options.compile_asm,
            next_id: 0,
            context,
            builder_context: FunctionBuilderContext::new(),
        })
    }

    fn declare(&mut self, name: &str) -> Result<FuncId, Error> {
        let id = self
            .module
            .declare_function(name, Linkage::Local, &self.signature)
            .map_err(|error| Error::Compile(error.to_string()))?;
        self.function_names.insert(id, name.to_string());
        Ok(id)
    }

    fn declare_js(&mut self, label: &str, line: u32) -> Result<FuncId, Error> {
        let id = self.next_id;
        self.next_id += 1;
        self.declare(&format!("{label}@{line}#{id}"))
    }

    pub(crate) fn compile_program<'a>(
        &mut self,
        program: &'a Program<'a>,
        scoping: &Scoping,
        nodes: &AstNodes<'a>,
        source: &str,
        filename: &str,
    ) -> Result<
        (
            Vec<CompiledFunction>,
            Vec<String>,
            Option<usize>,
            Vec<(u64, usize)>,
        ),
        Error,
    > {
        if self.compile_dump && self.compile_ast {
            println!("[{filename}] analyzed AST\n{program:#?}");
        }
        let source_path = filename;
        let filename: Box<str> = filename.into();
        self.source_name = (filename.as_ptr() as usize, filename.len());
        self.names.push(filename);
        let result = self.compile_program_inner(program, scoping, nodes, source, source_path);
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
        source: &str,
        source_path: &str,
    ) -> Result<
        (
            Vec<CompiledFunction>,
            Vec<String>,
            Option<usize>,
            Vec<(u64, usize)>,
        ),
        Error,
    > {
        let lines = if self.dump.is_some() || self.compile_dump {
            line_starts(source)
        } else {
            Vec::new()
        };
        let mut globals = Vec::new();
        let mut declarations = Vec::new();
        for statement in &program.body {
            match statement {
                Statement::FunctionDeclaration(function) => {
                    let name = validate_function(function)?;
                    declarations.push((name, function));
                }
                Statement::ClassDeclaration(class) => {
                    emit::class::validate_class(class)?;
                    globals.push(class.id.as_ref().unwrap().name.to_string());
                }
                Statement::EmptyStatement(_) | Statement::ExpressionStatement(_) => {}
                Statement::VariableDeclaration(declaration) => {
                    for declarator in &declaration.declarations {
                        let BindingPattern::BindingIdentifier(identifier) = &declarator.id else {
                            return Err(unsupported_at(
                                declarator.span,
                                "destructuring declaration",
                            ));
                        };
                        if declaration.kind == VariableDeclarationKind::Var {
                            globals.push(identifier.name.to_string());
                        }
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
            let line = line_at(&lines, function.span.start);
            let id = self.declare_js(&name, line)?;
            ids.insert(name.clone(), id);
            ordered.push((name, function));
        }

        let analysis = CaptureAnalysis::analyze(program, scoping);
        if let Some(span) = analysis.class_name_writes.first() {
            return Err(unsupported_at(*span, "assignment to inner class name"));
        }
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
                    if (function.id.is_some()
                        && function.r#type != oxc_ast::ast::FunctionType::FunctionDeclaration)
                        || function.r#async
                        || function.body.is_none()
                    {
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
            validate_params(params, span, generator)?;
            let self_name = match node.kind() {
                AstKind::Function(function) => function.id.as_ref().map(|id| id.name.as_str()),
                _ => None,
            };
            let label = function_label(nodes, node.id(), self_name);
            let line = line_at(&lines, span.start);
            let id = self.declare_js(&label, line)?;
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
            nested.push((
                id, span, params, statements, expression, generator, label, line,
            ));
        }
        let mut pending = Vec::new();
        for node in nodes.iter() {
            if let AstKind::Class(class) = node.kind() {
                emit::class::validate_class(class)?;
                if class.body.body.iter().any(|element| {
                    matches!(element,
                    oxc_ast::ast::ClassElement::MethodDefinition(method)
                        if method.kind == oxc_ast::ast::MethodDefinitionKind::Constructor)
                }) {
                    continue;
                }
                let allocator = oxc_allocator::Allocator::default();
                let builder = oxc_ast::builder::AstBuilder::new(&allocator);
                let params = FormalParameters::new(
                    class.span,
                    oxc_ast::ast::FormalParameterKind::FormalParameter,
                    oxc_allocator::Vec::new_in(&builder),
                    None,
                    &builder,
                );
                let label = format!("{}.constructor", class.id.as_ref().unwrap().name.as_str());
                let line = line_at(&lines, class.span.start);
                let id = self.declare_js(&label, line)?;
                let (code, _) = self.compile_callable(
                    &params,
                    &[],
                    None,
                    None,
                    &analysis.arguments_references,
                    &[],
                    &analysis.captured,
                    &closures,
                    scoping,
                    false,
                    &label,
                    line,
                )?;
                closures.insert(class.span.start, (id, Vec::new(), false));
                pending.push((id, code));
            }
        }

        for (name, function) in &ordered {
            let line = line_at(&lines, function.span.start);
            let (code, resume) = self.compile_callable(
                &function.params,
                &function.body.as_ref().unwrap().statements,
                None,
                analysis
                    .arguments_bindings
                    .get(&function.span.start)
                    .copied(),
                &analysis.arguments_references,
                &[],
                &analysis.captured,
                &closures,
                scoping,
                function.generator,
                name,
                line,
            )?;
            if let Some(resume) = resume {
                pending.push(resume);
            }
            pending.push((ids[name], code));
        }
        for (id, span, params, statements, expression, generator, label, line) in nested {
            let captures = &closures[&span.start].1;
            let (code, resume) = self.compile_callable(
                params,
                statements,
                expression,
                analysis.arguments_bindings.get(&span.start).copied(),
                &analysis.arguments_references,
                captures,
                &analysis.captured,
                &closures,
                scoping,
                generator,
                &label,
                line,
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
                Statement::VariableDeclaration(_)
                    | Statement::ExpressionStatement(_)
                    | Statement::ClassDeclaration(_)
            )
        }) {
            None
        } else {
            let line = line_at(&lines, program.span.start);
            let module = std::path::Path::new(source_path)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("module");
            let label = format!("{module}.module");
            let id = self.declare_js(&label, line)?;
            self.context.clear();
            self.context.func.signature = self.signature.clone();
            let mut annotations = Annotations::default();
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
                    &mut annotations,
                    &mut self.names,
                    self.source_name,
                    &program.body,
                    &analysis.captured,
                    &closures,
                    &analysis.arguments_references,
                )?;
                builder.seal_all_blocks();
                builder.finalize();
            }
            pending.push((
                id,
                FunctionBody {
                    function: std::mem::replace(
                        &mut self.context.func,
                        cranelift_codegen::ir::Function::new(),
                    ),
                    annotations,
                },
            ));
            Some(id)
        };

        let mut dumped = Vec::new();
        let mut next_ir = 1;
        let mut complete_ir = Vec::new();
        for (id, code) in pending {
            self.context.clear();
            self.context.func = code.function;
            let name = self
                .function_names
                .get(&id)
                .cloned()
                .unwrap_or_else(|| format!("jsfn_{}", id.as_u32()));
            if self.ssa_opt || self.ssa_dump {
                let analysis = numeric::Analysis::compute(&self.context.func, &code.annotations);
                if self.ssa_dump {
                    analysis.dump(&self.context.func, &name, &code.annotations);
                }
                if self.ssa_opt {
                    analysis.simplify(&mut self.context.func);
                    let numeric_get = self.module.declare_func_in_func(
                        self.helpers[helpers::Helper::IndexGetNumeric as usize],
                        &mut self.context.func,
                    );
                    analysis.specialize_index_gets(&mut self.context.func, numeric_get);
                    self.context
                        .verify(self.module.isa())
                        .map_err(|error| Error::Compile(error.to_string()))?;
                }
            }
            if self.dump_ir {
                complete_ir.push((0, format!("[{name}]")));
                complete_ir.extend(ir_lines(&mut self.context.func, &lines, &mut next_ir));
            }
            if self.compile_dump {
                if self.compile_ir {
                    println!("[{name}] IR\n{}", self.context.func.display());
                }
                self.context.set_disasm(self.compile_asm);
            }
            self.module
                .define_function(id, &mut self.context)
                .map_err(|error| Error::Compile(error.to_string()))?;
            if self.compile_dump && self.compile_asm {
                let compiled = self
                    .context
                    .compiled_code()
                    .ok_or_else(|| Error::Compile("missing compiled code".into()))?;
                println!(
                    "[{name}] machine code\n{}",
                    compiled.vcode.as_deref().unwrap_or("unavailable")
                );
            }
            if self.dump.is_some() {
                let Some(compiled) = self.context.compiled_code() else {
                    return Err(Error::Compile("missing compiled code".into()));
                };
                let size = compiled.code_buffer().len();
                let mut locations = BTreeMap::new();
                let mut default_line = None;
                for location in compiled.buffer.get_srclocs_sorted() {
                    if location.loc.is_default() {
                        continue;
                    }
                    let line = if self.dump_ir {
                        location.loc.bits()
                    } else {
                        line_at(&lines, location.loc.bits())
                    };
                    default_line.get_or_insert(line);
                    locations.insert(location.start, line);
                }
                locations.insert(0, default_line.unwrap_or(1));
                dumped.push((id, name, size, locations.into_iter().collect::<Vec<_>>()));
            }
        }
        self.module
            .finalize_definitions()
            .map_err(|error| Error::Compile(error.to_string()))?;
        let listing = if self.dump_ir {
            if let Some(dump) = self.dump.as_mut() {
                match dump.listing(source, &complete_ir) {
                    Ok(listing) => Some(listing),
                    Err(error) => {
                        eprintln!("js jit: source listing write failed: {error}");
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };
        if let Some(dump) = self.dump.as_mut() {
            for (id, name, size, locations) in dumped {
                let code = self.module.get_finalized_function(id);
                let bytes = unsafe { std::slice::from_raw_parts(code, size) };
                if let Some((path, mapping, ir_mapping)) = &listing {
                    let filename = path.to_str().unwrap_or(source_path);
                    let locations = locations
                        .iter()
                        .filter_map(|&(offset, line)| {
                            let line = if self.dump_ir {
                                ir_mapping.get(&line).copied()
                            } else {
                                mapping.get(line as usize - 1).copied()
                            };
                            line.map(|line| (offset, line))
                        })
                        .collect::<Vec<_>>();
                    dump.record(&name, filename, &locations, bytes);
                } else if self.dump_ir {
                    dump.record(&name, source_path, &[], bytes);
                } else {
                    dump.record(&name, source_path, &locations, bytes);
                }
            }
        }

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
        arguments_binding: Option<SymbolId>,
        arguments_references: &HashMap<u32, SymbolId>,
        captures: &[SymbolId],
        captured: &HashSet<SymbolId>,
        closures: &HashMap<u32, (FuncId, Vec<SymbolId>, bool)>,
        scoping: &Scoping,
        generator: bool,
        label: &str,
        line: u32,
    ) -> Result<(FunctionBody, Option<(FuncId, FunctionBody)>), Error> {
        if !generator {
            return Ok((
                self.compile_function(
                    params,
                    statements,
                    expression,
                    arguments_binding,
                    arguments_references,
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
        if self.compile_dump && self.compile_ast {
            println!("[{label}@{line}] generator resume plan\n{plan:#?}");
        }
        let internal = format!("{label}.resume");
        let id = self.declare_js(&internal, line)?;
        let resume = self.compile_function(
            params,
            statements,
            None,
            arguments_binding,
            arguments_references,
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
            None,
            arguments_references,
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
        arguments_binding: Option<SymbolId>,
        arguments_references: &HashMap<u32, SymbolId>,
        captures: &[SymbolId],
        captured: &HashSet<SymbolId>,
        closures: &HashMap<u32, (FuncId, Vec<SymbolId>, bool)>,
        scoping: &Scoping,
        kind: emit::FunctionKind,
    ) -> Result<FunctionBody, Error> {
        let mut annotations = Annotations::default();
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
                &mut annotations,
                &mut self.names,
                self.source_name,
                params,
                statements,
                expression,
                arguments_binding,
                arguments_references,
                captures,
                captured,
                closures,
                kind,
            )?;
            builder.seal_all_blocks();
            builder.finalize();
        }
        Ok(FunctionBody {
            function: std::mem::replace(
                &mut self.context.func,
                cranelift_codegen::ir::Function::new(),
            ),
            annotations,
        })
    }
}

fn line_starts(source: &str) -> Vec<u32> {
    let mut starts = vec![0];
    for (index, byte) in source.bytes().enumerate() {
        if byte == b'\n' {
            starts.push(index as u32 + 1);
        }
    }
    starts
}

fn function_label(
    nodes: &AstNodes<'_>,
    node_id: oxc_semantic::NodeId,
    name: Option<&str>,
) -> String {
    let mut parts = Vec::new();
    if let Some(name) = name {
        parts.push(name.to_string());
    }
    for ancestor in nodes.ancestor_kinds(node_id) {
        match ancestor {
            AstKind::VariableDeclarator(declarator) if parts.is_empty() => {
                if let BindingPattern::BindingIdentifier(identifier) = &declarator.id {
                    parts.push(identifier.name.to_string());
                }
            }
            AstKind::AssignmentExpression(assignment) if parts.is_empty() => {
                if let Some(name) = assignment_target_label(&assignment.left) {
                    parts.push(name);
                }
            }
            AstKind::ObjectProperty(property) if parts.is_empty() => {
                if let PropertyKey::StaticIdentifier(key) = &property.key {
                    parts.push(key.name.to_string());
                }
            }
            AstKind::MethodDefinition(method) => {
                if let PropertyKey::StaticIdentifier(key) = &method.key {
                    parts.push(key.name.to_string());
                }
            }
            AstKind::Function(function) => {
                if let Some(identifier) = &function.id {
                    parts.push(identifier.name.to_string());
                }
            }
            AstKind::Class(class) => {
                if let Some(identifier) = &class.id {
                    parts.push(identifier.name.to_string());
                }
            }
            _ => {}
        }
    }
    parts.reverse();
    if parts.is_empty() {
        parts.push("anonymous".into());
    }
    parts.join(".")
}

fn expression_label(expression: &Expression<'_>) -> Option<String> {
    match expression {
        Expression::Identifier(identifier) => Some(identifier.name.to_string()),
        Expression::ThisExpression(_) => Some("this".into()),
        Expression::StaticMemberExpression(member) => Some(format!(
            "{}.{}",
            expression_label(&member.object)?,
            member.property.name
        )),
        _ => None,
    }
}

fn assignment_target_label(target: &AssignmentTarget<'_>) -> Option<String> {
    match target {
        AssignmentTarget::AssignmentTargetIdentifier(identifier) => {
            Some(identifier.name.to_string())
        }
        AssignmentTarget::StaticMemberExpression(member) => Some(format!(
            "{}.{}",
            expression_label(&member.object)?,
            member.property.name
        )),
        _ => None,
    }
}

fn line_at(starts: &[u32], offset: u32) -> u32 {
    starts.partition_point(|&start| start <= offset) as u32
}

fn ir_lines(
    function: &mut cranelift_codegen::ir::Function,
    starts: &[u32],
    next_ir: &mut u32,
) -> Vec<(u32, String)> {
    let mut result = Vec::new();
    let blocks: Vec<_> = function.layout.blocks().collect();
    let display = function.display().to_string();
    if let Some(first) = blocks.first() {
        let header = format!("{first}(");
        for line in display
            .lines()
            .take_while(|line| !line.trim_start().starts_with(&header))
        {
            result.push((0, line.to_owned()));
        }
    }
    for block in blocks {
        let params = function
            .dfg
            .block_params(block)
            .iter()
            .map(|value| format!("{value}: {}", function.dfg.value_type(*value)))
            .collect::<Vec<_>>()
            .join(", ");
        result.push((0, format!("{block}({params}):")));
        let header = format!("{block}(");
        let plain_header = format!("{block}:");
        let mut in_block = false;
        for line in display.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with(&header) || trimmed == plain_header {
                in_block = true;
                continue;
            }
            if in_block && trimmed.starts_with("block") && trimmed.ends_with(':') {
                break;
            }
            if in_block && trimmed.contains(" -> ") {
                result.push((0, format!("    {trimmed}")));
            }
        }
        let insts: Vec<_> = function.layout.block_insts(block).collect();
        for inst in insts {
            let location = function.srcloc(inst);
            let source = if location.is_default() {
                String::new()
            } else {
                format!("  ; source line {}", line_at(starts, location.bits()))
            };
            let text = format!("    {}{source}", function.dfg.display_inst(inst));
            let index = *next_ir;
            *next_ir += 1;
            function.set_srcloc(inst, SourceLoc::new(index));
            result.push((index, text));
        }
    }
    result.push((0, "}".into()));
    result
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
    validate_params(&function.params, function.span, function.generator)?;
    Ok(name)
}

fn validate_params(
    params: &FormalParameters<'_>,
    span: oxc_span::Span,
    generator: bool,
) -> Result<(), Error> {
    if params
        .rest
        .as_ref()
        .is_some_and(|p| !matches!(p.rest.argument, BindingPattern::BindingIdentifier(_)))
    {
        return Err(unsupported_at(span, "rest parameter pattern"));
    }
    for parameter in &params.items {
        if generator
            && (parameter.initializer.is_some()
                || !matches!(parameter.pattern, BindingPattern::BindingIdentifier(_)))
        {
            return Err(unsupported_at(
                parameter.span,
                "generator default or destructured parameter",
            ));
        }
        match &parameter.pattern {
            BindingPattern::BindingIdentifier(_) => {}
            BindingPattern::ArrayPattern(pattern)
                if pattern.rest.is_none()
                    && pattern
                        .elements
                        .iter()
                        .flatten()
                        .all(|element| matches!(element, BindingPattern::BindingIdentifier(_))) => {
            }
            _ => return Err(unsupported_at(parameter.span, "function parameter pattern")),
        }
    }
    Ok(())
}

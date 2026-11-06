use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use cranelift_codegen::Context;
use cranelift_codegen::ir::{AbiParam, Signature, types};
use cranelift_codegen::isa::CallConv;
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

use super::clif::{Annotations, FunctionBody};
use super::entries::Entry;
use super::function_data::FunctionData;
use super::{analysis::CaptureAnalysis, emit, jitdump, temporaries};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompilationMode {
    Synchronous,
    Background,
}

pub(crate) struct CompiledFunction {
    pub(crate) name: String,
    pub(crate) arity: usize,
    pub(crate) code: u64,
    pub(crate) generator: bool,
}

#[derive(Clone)]
struct Emission {
    callees: Vec<FuncId>,
    conditional: super::semantic::specialize::ConditionalPlan,
    inlines: Vec<super::semantic::specialize::Inline>,
    reads: Vec<super::semantic::reads::Read>,
    osr: Option<Arc<super::semantic::osr::Plan>>,
    source_name: (usize, usize),
    params: oxc_span::Span,
    closures: HashMap<u32, (FuncId, Vec<SymbolId>, bool)>,
}

pub(crate) struct RecompileJob {
    pub(crate) code: FuncId,
    initial: bool,
    previous: Entry,
    emission: Emission,
    canonical: Arc<super::semantic::specialize::Repository>,
    sources: Arc<HashMap<usize, Arc<(String, String)>>>,
    pub(super) label: String,
    pub(super) queued_at: Option<Instant>,
    region: Option<OsrRequest>,
}

struct OsrRequest {
    site: usize,
    plan: Arc<super::semantic::osr::Plan>,
    candidate: usize,
}

impl RecompileJob {
    pub(crate) fn key(&self) -> super::recompile::Key {
        self.region.as_ref().map_or(
            super::recompile::Key::Function(self.code.as_u32() as u64),
            |region| super::recompile::Key::Loop(region.site),
        )
    }
}

pub(crate) struct Replacement {
    code: FuncId,
    initial: bool,
    entry: Entry,
    emission: Emission,
    region: bool,
}

pub(crate) struct Unpublished {
    replacement: Replacement,
    names: Vec<Box<str>>,
    cells: Vec<Box<u64>>,
    data: Vec<Box<FunctionData>>,
}

impl Unpublished {
    pub(crate) fn initial(&self) -> bool {
        self.replacement.initial
    }
}

// These caches have never executed: global cache owners and runtime pointers are empty.
// The worker relinquishes every cache before the execution thread publishes the entry.
unsafe impl Send for Unpublished {}

#[derive(Default)]
struct CompileStats {
    functions: usize,
    blocks: usize,
    insts: usize,
    bytes: usize,
}

impl Drop for Compiler {
    fn drop(&mut self) {
        if let Some(stats) = &self.stats {
            eprintln!(
                "js jit stats: {} functions, {} blocks, {} instructions, {} code bytes",
                stats.functions, stats.blocks, stats.insts, stats.bytes
            );
        }
    }
}

pub(crate) struct Compiler {
    log: super::recompile::Log,
    emissions: HashMap<FuncId, Emission>,
    lazy: HashMap<FuncId, (Emission, Arc<super::semantic::specialize::Repository>)>,
    sources: Arc<HashMap<usize, Arc<(String, String)>>>,
    names: Vec<Box<str>>,
    cells: Vec<Box<u64>>,
    data: Vec<Box<FunctionData>>,
    source_name: (usize, usize),
    canonical: Arc<super::semantic::specialize::Repository>,
    module: JITModule,
    signature: Signature,
    fixed: HashMap<FuncId, u16>,
    adapters: HashMap<u16, usize>,
    stats: Option<CompileStats>,
    stubs: emit::Stubs,
    functions: HashMap<String, FuncId>,
    function_names: HashMap<FuncId, String>,
    dump: Option<jitdump::Writer>,
    dump_ir: bool,
    ssa_opt: bool,
    recompile_inline_multiplier: usize,
    osr: bool,
    get_refinement: bool,
    ssa_dump: bool,
    compile_dump: bool,
    compile_ir: bool,
    compile_ast: bool,
    compile_semantic_ir: bool,
    compile_asm: bool,
    next_id: u64,
    context: Context,
    builder_context: FunctionBuilderContext,
}

impl Compiler {
    #[cfg(test)]
    pub(crate) fn osr_transfers(&self) -> u64 {
        self.data
            .iter()
            .flat_map(|data| data.pinned::<super::osr::Site>())
            .map(|site| site.transfers.get())
            .sum()
    }

    #[cfg(test)]
    pub(crate) fn osr_read_site(
        &self,
        owner: u64,
    ) -> Option<(
        *const super::osr::Site,
        super::planner::Site,
        super::planner::ReadKind,
    )> {
        self.data
            .iter()
            .flat_map(|data| data.pinned::<super::osr::Site>())
            .find_map(|site| {
                if site.owner != owner {
                    return None;
                }
                site.plan.loops[site.candidate]
                    .opportunities
                    .iter()
                    .find_map(|opportunity| {
                        let super::semantic::osr::Observation::Read(kind) = opportunity.kind else {
                            return None;
                        };
                        Some((site as *const _, opportunity.site, kind))
                    })
            })
    }

    pub(crate) fn new(
        options: EngineOptions,
        mode: CompilationMode,
        direct_entries: *const usize,
        compilation_state: usize,
        log: super::recompile::Log,
    ) -> Result<Self, Error> {
        let opt_level = if mode == CompilationMode::Synchronous {
            options.sync_opt_level.unwrap_or(options.opt_level)
        } else {
            options.opt_level
        };
        let flags = [
            ("opt_level", opt_level.as_flag()),
            (
                "regalloc_algorithm",
                if mode == CompilationMode::Synchronous && options.sync_single_pass {
                    "single_pass"
                } else {
                    "backtracking"
                },
            ),
            (
                "enable_verifier",
                if options.osr { "true" } else { "false" },
            ),
        ];
        let builder = JITBuilder::with_flags(&flags, default_libcall_names())
            .map_err(|error| Error::Compile(error.to_string()))?;
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

        let mut dump = options.jitdump.then(jitdump::Writer::new).flatten();
        let stubs = emit::Stubs::new(
            &mut module,
            &signature,
            direct_entries,
            compilation_state,
            dump.as_mut(),
        )?;
        let context = module.make_context();
        let dump_ir = dump.is_some() && options.jitdump_ir;
        Ok(Self {
            log,
            osr: options.osr,
            emissions: HashMap::new(),
            lazy: HashMap::new(),
            sources: Arc::default(),
            names: Vec::new(),
            cells: Vec::new(),
            data: Vec::new(),
            source_name: (0, 0),
            canonical: Arc::default(),
            module,
            signature,
            fixed: HashMap::new(),
            adapters: HashMap::new(),
            stats: std::env::var_os("JS_JIT_STATS").map(|_| CompileStats::default()),
            stubs,
            functions: HashMap::new(),
            function_names: HashMap::new(),
            dump,
            dump_ir,
            ssa_opt: options.ssa_opt,
            recompile_inline_multiplier: options.recompile_inline_multiplier,
            get_refinement: options.get_refinement,
            ssa_dump: options.ssa_dump,
            compile_dump: options.compile_dump,
            compile_ir: options.compile_ir,
            compile_ast: options.compile_ast,
            compile_semantic_ir: options.compile_semantic_ir,
            compile_asm: options.compile_asm,
            next_id: 0,
            context,
            builder_context: FunctionBuilderContext::new(),
        })
    }

    fn fixed_signature(&self, arity: u16) -> Signature {
        emit::stubs::fixed_signature(self.module.target_config().pointer_type(), arity)
    }

    fn entry(&mut self, id: FuncId) -> Result<Entry, Error> {
        let code = self.module.get_finalized_function(id) as usize;
        let arity = self.fixed.get(&id).copied();
        let adapter = match arity {
            Some(arity) => Some(self.adapter(arity)?),
            None => None,
        };
        Ok(Entry {
            code,
            adapter,
            arity: arity.unwrap_or(0),
            lexical_this: self.canonical.lexical_this(id),
        })
    }

    fn adapter(&mut self, arity: u16) -> Result<usize, Error> {
        if let Some(adapter) = self.stubs.adapter(arity) {
            return Ok(adapter);
        }
        if let Some(&adapter) = self.adapters.get(&arity) {
            return Ok(adapter);
        }
        let stub = emit::stubs::build_adapter(&mut self.module, &self.signature, arity)?;
        self.module
            .finalize_definitions()
            .map_err(|error| Error::Compile(error.to_string()))?;
        let adapter = stub.finalize(
            &self.module,
            self.dump.as_mut(),
            &format!("adapter/{arity}"),
        );
        self.adapters.insert(arity, adapter);
        Ok(adapter)
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
            Vec<(u64, Entry)>,
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
        Arc::make_mut(&mut self.sources).insert(
            self.source_name.0,
            Arc::new((source.to_owned(), source_path.to_owned())),
        );
        Arc::make_mut(&mut self.canonical)
            .sources
            .insert(self.source_name.0, self.source_name.1);
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
            Vec<(u64, Entry)>,
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
        if let Some(span) = analysis.class_name_write {
            return Err(unsupported_at(span, "assignment to inner class name"));
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
                    Some(class.span),
                )?;
                closures.insert(class.span.start, (id, Vec::new(), false));
                pending.push((id, code));
            }
        }

        for (name, function) in &ordered {
            if function.generator {
                continue;
            }
            let canonical = super::semantic::canonical_for_emission(
                &function.params,
                &function.body.as_ref().unwrap().statements,
                None,
                function.span,
                scoping,
                &analysis.captured,
                &[],
                &closures,
                analysis
                    .arguments_bindings
                    .get(&function.span.start)
                    .copied(),
                &analysis.arguments_references,
                false,
                self.ssa_opt,
            )
            .map_err(Error::Unsupported)?;
            Arc::make_mut(&mut self.canonical).insert(
                self.source_name.0,
                name,
                ids[name],
                canonical,
            );
        }

        let arrows: HashSet<u32> = nodes
            .iter()
            .filter_map(|node| match node.kind() {
                AstKind::ArrowFunctionExpression(function) => Some(function.span.start),
                _ => None,
            })
            .collect();
        for (id, span, params, statements, expression, generator, _, _) in &nested {
            if *generator {
                continue;
            }
            let canonical = super::semantic::canonical_for_emission(
                params,
                statements,
                *expression,
                *span,
                scoping,
                &analysis.captured,
                &closures[&span.start].1,
                &closures,
                analysis.arguments_bindings.get(&span.start).copied(),
                &analysis.arguments_references,
                false,
                self.ssa_opt,
            )
            .map_err(Error::Unsupported)?;
            Arc::make_mut(&mut self.canonical).insert_closure(
                self.source_name.0,
                *id,
                canonical,
                arrows.contains(&span.start),
            );
        }

        if self.compile_dump && self.compile_semantic_ir {
            for (name, function) in &ordered {
                let graph = super::semantic::canonical_for_emission(
                    &function.params,
                    &function.body.as_ref().unwrap().statements,
                    None,
                    function.span,
                    scoping,
                    &analysis.captured,
                    &[],
                    &closures,
                    analysis
                        .arguments_bindings
                        .get(&function.span.start)
                        .copied(),
                    &analysis.arguments_references,
                    function.generator,
                    self.ssa_opt,
                );
                let graph = graph.and_then(|canonical| {
                    super::semantic::optimize_canonical(
                        canonical,
                        self.ssa_opt,
                        Some((&self.canonical, self.source_name.0)),
                    )
                });
                match graph {
                    Ok(graph) => println!("[{source_path}] semantic IR {name}\n{}", graph),
                    Err(error) => println!("[{source_path}] semantic IR {name}: {error}"),
                }
            }
            for (_, span, params, statements, expression, generator, label, _) in &nested {
                let graph = super::semantic::canonical_for_emission(
                    params,
                    statements,
                    *expression,
                    *span,
                    scoping,
                    &analysis.captured,
                    &closures[&span.start].1,
                    &closures,
                    analysis.arguments_bindings.get(&span.start).copied(),
                    &analysis.arguments_references,
                    *generator,
                    self.ssa_opt,
                );
                let graph = graph.and_then(|canonical| {
                    super::semantic::optimize_canonical(
                        canonical,
                        self.ssa_opt,
                        Some((&self.canonical, self.source_name.0)),
                    )
                });
                match graph {
                    Ok(graph) => println!("[{source_path}] semantic IR {label}\n{}", graph),
                    Err(error) => println!("[{source_path}] semantic IR {label}: {error}"),
                }
            }
        }

        for (name, function) in &ordered {
            let line = line_at(&lines, function.span.start);
            if !function.generator && !self.compile_dump {
                self.lazy.insert(
                    ids[name],
                    (
                        Emission {
                            callees: Vec::new(),
                            conditional: Default::default(),
                            inlines: Vec::new(),
                            reads: Vec::new(),
                            osr: None,
                            source_name: self.source_name,
                            params: function.params.span,
                            closures: closures.clone(),
                        },
                        self.canonical.clone(),
                    ),
                );
                continue;
            }
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
                Some(function.span),
            )?;
            if let Some(resume) = resume {
                pending.push(resume);
            }
            pending.push((ids[name], code));
        }
        for (id, span, params, statements, expression, generator, label, line) in nested {
            let captures = &closures[&span.start].1;
            if !generator && !self.compile_dump {
                self.lazy.insert(
                    id,
                    (
                        Emission {
                            callees: Vec::new(),
                            conditional: Default::default(),
                            inlines: Vec::new(),
                            reads: Vec::new(),
                            osr: None,
                            source_name: self.source_name,
                            params: params.span,
                            closures: closures.clone(),
                        },
                        self.canonical.clone(),
                    ),
                );
                continue;
            }
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
                Some(span),
            )?;
            if let Some(resume) = resume {
                pending.push(resume);
            }
            pending.push((id, code));
        }
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
            let mut annotations = Annotations {
                semantic_locations: self.dump_ir,
                ..Annotations::default()
            };
            let graph = super::semantic::lower_global(
                &program.body,
                program.span,
                scoping,
                &analysis.captured,
                &closures,
                &analysis.arguments_references,
                self.ssa_opt,
            )
            .map_err(Error::Unsupported)?;
            {
                let mut data = FunctionData::new();
                let mut builder = cranelift_frontend::FunctionBuilder::new(
                    &mut self.context.func,
                    &mut self.builder_context,
                );
                emit::build_global(
                    &mut builder,
                    &mut self.module,
                    self.stubs,
                    &mut annotations,
                    &mut self.names,
                    &mut self.cells,
                    &mut data,
                    self.source_name,
                    &closures,
                    &graph.graph,
                )?;
                builder.seal_all_blocks();
                builder.finalize(self.module.target_config());
                data.seal();
                self.data.push(data);
            }
            pending.push((
                id,
                FunctionBody {
                    function: std::mem::replace(
                        &mut self.context.func,
                        cranelift_codegen::ir::Function::new(),
                    ),
                    annotations,
                    listing: self.dump_ir.then(|| graph.listing_lines()),
                },
            ));
            Some(id)
        };

        let compiled_ids: Vec<_> = pending
            .iter()
            .map(|(id, _)| *id)
            .filter(|id| Some(*id) != initializer)
            .collect();

        self.finish_functions(pending, source, source_path, &lines)?;

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
                .map(|id| Ok((id.as_u32() as u64, self.entry(id)?)))
                .collect::<Result<_, Error>>()?,
        ))
    }

    fn finish_functions(
        &mut self,
        pending: Vec<(FuncId, FunctionBody)>,
        source: &str,
        source_path: &str,
        lines: &[u32],
    ) -> Result<(), Error> {
        let mut dumped = Vec::new();
        for (id, mut code) in pending {
            let mut listing = None;
            self.context.clear();
            self.context.func = code.function;
            let name = self
                .function_names
                .get(&id)
                .cloned()
                .unwrap_or_else(|| format!("jsfn_{}", id.as_u32()));
            if self.ssa_opt {
                super::read_scope::specialize(
                    &mut self.context,
                    &mut code.annotations,
                    &mut self.module,
                );
                let removed =
                    temporaries::remove_empty_clears(&mut self.context.func, &code.annotations);
                if self.ssa_dump {
                    eprintln!("js ssa [{name}]: removed {removed} empty temporary clears");
                }
            }
            if let Some(semantic) = &code.listing {
                let ir = ir_lines(semantic, source, source_path, &lines, &self.sources);
                if let Some(dump) = self.dump.as_mut() {
                    match dump.listing(&name, &ir) {
                        Ok(written) => listing = Some(written),
                        Err(error) => eprintln!("js jit: listing write failed: {error}"),
                    }
                }
            }
            if self.compile_dump {
                if self.compile_ir {
                    println!("[{name}] IR\n{}", self.context.func.display());
                }
            }
            if let Some(stats) = &mut self.stats {
                stats.functions += 1;
                stats.blocks += self.context.func.layout.blocks().count();
                stats.insts += self
                    .context
                    .func
                    .layout
                    .blocks()
                    .map(|block| self.context.func.layout.block_insts(block).count())
                    .sum::<usize>();
            }
            if self.context.func.signature.call_conv == CallConv::Tail {
                let arity = self.context.func.signature.params.len() - 3;
                self.fixed.insert(id, arity as u16);
            } else {
                self.fixed.remove(&id);
            }
            self.module
                .define_function(id, &mut self.context)
                .map_err(|error| Error::Compile(format!("{error:?}")))?;
            if let Some(stats) = &mut self.stats {
                stats.bytes += self
                    .context
                    .compiled_code()
                    .map_or(0, |compiled| compiled.code_buffer().len());
            }
            if self.compile_dump && self.compile_asm {
                let compiled = self
                    .context
                    .compiled_code()
                    .ok_or_else(|| Error::Compile("missing compiled code".into()))?;
                let disassembler = self
                    .module
                    .isa()
                    .to_capstone()
                    .map_err(|error| Error::Compile(error.to_string()))?;
                let asm = compiled
                    .disassemble(Some(&self.context.func.params), &disassembler)
                    .map_err(|error| Error::Compile(error.to_string()))?;
                println!("[{name}] machine code\n{asm}");
            }
            if self.dump.is_some() {
                let Some(compiled) = self.context.compiled_code() else {
                    return Err(Error::Compile("missing compiled code".into()));
                };
                let size = compiled.code_buffer().len();
                let mut locations = BTreeMap::new();
                let ranges = compiled.buffer.get_srclocs_sorted();
                if self.dump_ir {
                    locations.insert(0, 0);
                    for location in ranges {
                        if (location.end as usize) < size {
                            locations.entry(location.end).or_insert(0);
                        }
                    }
                    for location in ranges {
                        locations.insert(
                            location.start,
                            if location.loc.as_abs().is_default() {
                                0
                            } else {
                                location.loc.as_abs().bits()
                            },
                        );
                    }
                } else {
                    let mut default_line = None;
                    for location in ranges {
                        if location.loc.as_abs().is_default() {
                            continue;
                        }
                        let line = line_at(&lines, location.loc.as_abs().bits());
                        default_line.get_or_insert(line);
                        locations.insert(location.start, line);
                    }
                    locations.insert(0, default_line.unwrap_or(1));
                }
                dumped.push((
                    id,
                    name,
                    size,
                    locations.into_iter().collect::<Vec<_>>(),
                    listing,
                ));
            }
        }
        self.module
            .finalize_definitions()
            .map_err(|error| Error::Compile(error.to_string()))?;
        if let Some(dump) = self.dump.as_mut() {
            for (id, name, size, locations, listing) in dumped {
                let code = self.module.get_finalized_function(id);
                let bytes = unsafe { std::slice::from_raw_parts(code, size) };
                if let Some((path, ir_mapping)) = &listing {
                    let filename = path.to_str().unwrap_or(source_path);
                    let locations = locations
                        .iter()
                        .map(|&(offset, id)| (offset, ir_mapping.get(&id).copied().unwrap_or(1)))
                        .collect::<Vec<_>>();
                    dump.record(&name, filename, &locations, bytes);
                } else if self.dump_ir {
                    dump.record(&name, source_path, &[], bytes);
                } else {
                    dump.record(&name, source_path, &locations, bytes);
                }
            }
        }

        Ok(())
    }

    pub(crate) fn update_feedback(&mut self, planner: &super::planner::Planner) {
        if planner.targets().all(|(site, target)| {
            u32::try_from(target).map_or(true, |target| {
                self.canonical.feedback.get(&site) == Some(&FuncId::from_u32(target))
            })
        }) && (!self.get_refinement
            || planner
                .reads()
                .all(|(site, observed)| self.canonical.read_feedback.get(&site) == Some(&observed)))
        {
            return;
        }
        let canonical = Arc::make_mut(&mut self.canonical);
        for (site, target) in planner.targets() {
            if let Ok(target) = u32::try_from(target) {
                canonical.feedback.insert(site, FuncId::from_u32(target));
            }
        }
        if self.get_refinement {
            for (site, element) in planner.reads() {
                canonical.read_feedback.insert(site, element);
            }
        }
    }

    pub(crate) fn call_profile(&self, planner: &super::Planner) -> String {
        let mut sites = planner.calls();
        sites.sort_by_key(|(_, profile)| std::cmp::Reverse(profile.calls()));
        let total: u64 = sites.iter().map(|(_, profile)| profile.calls()).sum();
        let guarded: u64 = sites.iter().map(|(_, profile)| profile.guarded).sum();
        let native: u64 = sites.iter().map(|(_, profile)| profile.native).sum();
        let mut report = format!(
            "js call profile: {total} stub calls at {} sites \
             ({guarded} from guarded sites, {native} to natives)\n",
            sites.len()
        );
        for (label, range) in [
            ("1", 1..2),
            ("2", 2..3),
            ("3-4", 3..5),
            ("5+", 5..usize::MAX),
        ] {
            let (count, calls) = sites
                .iter()
                .filter(|(_, profile)| range.contains(&profile.target_count()))
                .fold((0, 0), |(count, calls), (_, profile)| {
                    (count + 1, calls + profile.calls())
                });
            report += &format!("  {label} targets: {count} sites, {calls} calls\n");
        }
        for (site, profile) in sites.iter().take(20) {
            let location = self.sources.get(&site.source).map_or_else(
                || format!("?:{}", site.start),
                |source| {
                    let before = &source.0[..site.start as usize];
                    let line = before.matches('\n').count() + 1;
                    let column = before.len() - before.rfind('\n').map_or(0, |index| index + 1);
                    format!("{}:{line}:{}", source.1, column + 1)
                },
            );
            let mut targets = profile.targets.clone();
            targets.sort_by_key(|&(_, count)| std::cmp::Reverse(count));
            let mut targets: Vec<_> = targets
                .iter()
                .take(4)
                .map(|&(code, count)| format!("{}={count}", self.function_name(code)))
                .collect();
            if profile.native != 0 {
                targets.push(format!("<native>={}", profile.native));
            }
            report += &format!(
                "  {location} generic={} guarded={} [{}]\n",
                profile.generic,
                profile.guarded,
                targets.join(", ")
            );
        }
        report
    }

    pub(crate) fn function_name(&self, code: u64) -> &str {
        u32::try_from(code)
            .ok()
            .and_then(|code| self.function_names.get(&FuncId::from_u32(code)))
            .map_or("<unknown>", String::as_str)
    }

    pub(crate) fn recompilation(&self, code: u64, previous: Entry) -> Option<RecompileJob> {
        let code = FuncId::from_u32(u32::try_from(code).ok()?);
        Some(RecompileJob {
            region: None,
            code,
            initial: false,
            previous,
            emission: self.emissions.get(&code)?.clone(),
            canonical: self.canonical.clone(),
            sources: self.sources.clone(),
            label: self.function_names.get(&code)?.clone(),
            queued_at: None,
        })
    }

    pub(crate) fn osr_compilation(&self, site: &super::osr::Site) -> Option<RecompileJob> {
        let mut job = self.recompilation(site.owner, Entry::generic(0))?;
        let has_feedback =
            site.plan.loops[site.candidate]
                .opportunities
                .iter()
                .any(|opportunity| {
                    let mut location = opportunity.site;
                    if location.source == 0 {
                        location.source = job.emission.source_name.0;
                    }
                    match opportunity.kind {
                        super::semantic::osr::Observation::Call => {
                            self.canonical.feedback.contains_key(&location)
                        }
                        super::semantic::osr::Observation::Read(kind) => {
                            self.canonical.read_feedback.contains_key(&(location, kind))
                        }
                    }
                });
        if !has_feedback {
            return None;
        }
        job.label += &format!(".loop{}", site.plan.loops[site.candidate].header);
        job.region = Some(OsrRequest {
            site: site as *const _ as usize,
            plan: site.plan.clone(),
            candidate: site.candidate,
        });
        Some(job)
    }

    pub(crate) fn recompile(&mut self, job: RecompileJob) -> Result<Option<Replacement>, Error> {
        let trace = self
            .log
            .enabled()
            .then(|| (job.code.as_u32(), job.label.clone(), Instant::now()));
        if let Some((code, label, _)) = &trace {
            self.log
                .event(format_args!("compiling code={code} function={label}"));
        }
        let result = self.recompile_inner(job, false);
        if let Some((code, label, started)) = trace {
            let outcome = match &result {
                Ok(Some(_)) => "compiled",
                Ok(None) => "unchanged",
                Err(_) => "failed",
            };
            self.log.event(format_args!(
                "{outcome} code={code} function={label} compile_ms={:.3}",
                started.elapsed().as_secs_f64() * 1000.0
            ));
        }
        if result.is_err() {
            self.context.clear();
            self.builder_context = FunctionBuilderContext::new();
        }
        result
    }

    fn recompile_inner(
        &mut self,
        job: RecompileJob,
        declared: bool,
    ) -> Result<Option<Replacement>, Error> {
        let RecompileJob {
            code,
            initial,
            previous,
            mut emission,
            canonical,
            sources,
            label,
            queued_at: _,
            region,
        } = job;
        self.canonical = canonical;
        self.sources = sources;
        let (source, canonical) = if let Some(region) = &region {
            (
                emission.source_name.0,
                super::semantic::Canonical::from_graph(
                    region
                        .plan
                        .extract(region.candidate)
                        .map_err(Error::Unsupported)?,
                ),
            )
        } else {
            let Some(canonical) = self.canonical.emission(code) else {
                return Ok(None);
            };
            canonical
        };
        let inline_multiplier = if initial {
            1
        } else {
            self.recompile_inline_multiplier
        };
        let planned = super::semantic::plan_canonical_with_budget(
            canonical,
            self.ssa_opt,
            Some((&self.canonical, source)),
            inline_multiplier,
        )
        .map_err(Error::Unsupported)?;
        emission.callees = self.canonical.graph_callees(&planned.graph, source);
        if region.is_some()
            && planned.inlines.is_empty()
            && planned.reads.is_empty()
            && planned.conditional == Default::default()
        {
            return Ok(None);
        }
        if self.log.enabled() {
            self.log.event(format_args!(
                "plan code={} function={label} inlines={}->{} reads={}->{} conditional_changed={} inline_multiplier={inline_multiplier}",
                code.as_u32(),
                emission.inlines.len(),
                planned.inlines.len(),
                emission.reads.len(),
                planned.reads.len(),
                emission.conditional != planned.conditional,
            ));
        }
        if !initial
            && region.is_none()
            && planned.inlines == emission.inlines
            && planned.reads == emission.reads
            && planned.conditional == emission.conditional
        {
            return Ok(None);
        }
        let mut graph = super::semantic::optimize_planned(
            planned,
            (!initial && region.is_none()).then_some(previous.code),
        )
        .map_err(Error::Unsupported)?;
        if self.osr && region.is_none() {
            graph.prepare_osr(source).map_err(Error::Unsupported)?;
        }
        if self.compile_dump && self.compile_semantic_ir {
            println!("[recompile {}] semantic IR\n{}", label, graph);
        }
        let mut annotations = Annotations {
            semantic_locations: self.dump_ir,
            ..Annotations::default()
        };
        self.context.clear();
        self.context.func.signature = match graph.graph.arity {
            Some(arity) => self.fixed_signature(arity),
            None => self.signature.clone(),
        };
        {
            let mut data = FunctionData::new();
            let mut builder = cranelift_frontend::FunctionBuilder::new(
                &mut self.context.func,
                &mut self.builder_context,
            );
            emit::build_function(
                &mut builder,
                &mut self.module,
                self.stubs,
                &mut annotations,
                &mut self.names,
                &mut self.cells,
                &mut data,
                emission.source_name,
                emission.params,
                &emission.closures,
                emit::FunctionKind::Ordinary,
                &graph.graph,
                self.ssa_opt.then_some(emit::Feedback {
                    function_return: region.is_none(),
                    get_refinement: self.get_refinement,
                    owner: code.as_u32() as u64,
                }),
            )?;
            builder.seal_all_blocks();
            builder.finalize(self.module.target_config());
            data.seal();
            self.data.push(data);
        }
        let body = FunctionBody {
            function: std::mem::replace(
                &mut self.context.func,
                cranelift_codegen::ir::Function::new(),
            ),
            annotations,
            listing: self.dump_ir.then(|| graph.listing_lines()),
        };
        let id = if declared {
            code
        } else {
            let suffix = if initial { "prefetch" } else { "feedback" };
            self.declare_js(&format!("{label}.{suffix}"), 1)?
        };
        let (source, filename) = self.sources[&source].as_ref().clone();
        let lines = line_starts(&source);
        self.finish_functions(vec![(id, body)], &source, &filename, &lines)?;
        emission.conditional = graph.conditional;
        emission.inlines = graph.inlines;
        emission.reads = graph.reads;
        emission.osr = graph.osr;
        if self.log.enabled()
            && let Some(plan) = &emission.osr
        {
            self.log.event(format_args!("{}", plan.report(&label)));
        }
        let mut entry = self.entry(id)?;
        if region.is_none() {
            entry.lexical_this = if initial {
                self.canonical.lexical_this(code)
            } else {
                previous.lexical_this
            };
        }
        Ok(Some(Replacement {
            code,
            initial,
            entry,
            emission,
            region: region.is_some(),
        }))
    }

    pub(crate) fn has_lazy(&self, code: u64) -> bool {
        u32::try_from(code)
            .ok()
            .is_some_and(|id| self.lazy.contains_key(&FuncId::from_u32(id)))
    }

    pub(crate) fn lazy_compilation(&self, code: u64) -> Option<RecompileJob> {
        let id = FuncId::from_u32(u32::try_from(code).ok()?);
        let (emission, canonical) = self.lazy.get(&id)?.clone();
        Some(RecompileJob {
            region: None,
            code: id,
            initial: true,
            previous: Entry::generic(0),
            emission,
            canonical,
            sources: self.sources.clone(),
            label: self.function_names[&id].clone(),
            queued_at: None,
        })
    }

    pub(crate) fn lazy_callees(&self, code: u64) -> Vec<RecompileJob> {
        let Ok(index) = u32::try_from(code) else {
            return Vec::new();
        };
        let id = FuncId::from_u32(index);
        let mut callees = self.canonical.callees(id);
        if let Some(emission) = self.emissions.get(&id) {
            for &callee in &emission.callees {
                if !callees.contains(&callee) {
                    callees.push(callee);
                }
            }
        }
        callees
            .into_iter()
            .filter(|&callee| callee != id)
            .filter_map(|callee| self.lazy_compilation(u64::from(callee.as_u32())))
            .collect()
    }

    pub(crate) fn compile_lazy(&mut self, code: u64) -> Result<Option<Entry>, Error> {
        let Some(job) = self.lazy_compilation(code) else {
            return Ok(None);
        };
        // Initial emission uses the repository as it existed when the source was loaded,
        // before later sources and runtime feedback change its specialization decisions.
        let current = self.canonical.clone();
        let started = self.log.enabled().then(Instant::now);
        let label = started.map(|_| job.label.clone());
        let result = self.recompile_inner(job, true);
        if let (Some(started), Some(label)) = (started, label) {
            self.log.event(format_args!(
                "lazy-compiled code={code} function={label} compile_ms={:.3}",
                started.elapsed().as_secs_f64() * 1000.0
            ));
        }
        self.canonical = current;
        if result.is_err() {
            self.context.clear();
            self.builder_context = FunctionBuilderContext::new();
        }
        let Some(replacement) = result? else {
            return Ok(None);
        };
        Ok(Some(self.install(replacement).1))
    }

    pub(crate) fn install(&mut self, replacement: Replacement) -> (u64, Entry) {
        if replacement.initial {
            self.lazy.remove(&replacement.code);
        }
        if !replacement.region {
            self.emissions
                .insert(replacement.code, replacement.emission);
        }
        (replacement.code.as_u32() as u64, replacement.entry)
    }

    pub(crate) fn take_unpublished(&mut self, replacement: Replacement) -> Unpublished {
        Unpublished {
            replacement,
            names: std::mem::take(&mut self.names),
            cells: std::mem::take(&mut self.cells),
            data: std::mem::take(&mut self.data),
        }
    }

    pub(crate) fn install_unpublished(&mut self, compiled: Unpublished) -> Option<(u64, Entry)> {
        // Dispatch may have compiled this lazy body while the worker was running.
        if compiled.replacement.initial && !self.lazy.contains_key(&compiled.replacement.code) {
            return None;
        }
        self.names.extend(compiled.names);
        self.cells.extend(compiled.cells);
        self.data.extend(compiled.data);
        Some(self.install(compiled.replacement))
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
        semantic_function: Option<oxc_span::Span>,
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
                    semantic_function,
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
            semantic_function,
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
            None,
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
        semantic_function: Option<oxc_span::Span>,
    ) -> Result<FunctionBody, Error> {
        let mut graph = semantic_function
            .map(|span| {
                let canonical = match self.canonical.canonical(self.source_name.0, span.start) {
                    Some(canonical) => Ok(canonical.clone()),
                    None => super::semantic::canonical_for_emission(
                        params,
                        statements,
                        expression,
                        span,
                        scoping,
                        captured,
                        captures,
                        closures,
                        arguments_binding,
                        arguments_references,
                        matches!(kind, emit::FunctionKind::Generator(_)),
                        self.ssa_opt,
                    ),
                }?;
                super::semantic::optimize_canonical(
                    canonical,
                    self.ssa_opt,
                    Some((&self.canonical, self.source_name.0)),
                )
            })
            .transpose()
            .map_err(Error::Unsupported)?;
        if self.osr
            && let Some(graph) = &mut graph
        {
            graph
                .prepare_osr(self.source_name.0)
                .map_err(Error::Unsupported)?;
        }
        let feedback = if self.ssa_opt && matches!(kind, emit::FunctionKind::Ordinary) {
            semantic_function.and_then(|span| self.canonical.code(self.source_name.0, span.start))
        } else {
            None
        };
        if let Some(code) = feedback {
            self.emissions.insert(
                code,
                Emission {
                    callees: self
                        .canonical
                        .graph_callees(&graph.as_ref().unwrap().graph, self.source_name.0),
                    conditional: graph.as_ref().unwrap().conditional.clone(),
                    inlines: graph.as_ref().unwrap().inlines.clone(),
                    reads: graph.as_ref().unwrap().reads.clone(),
                    osr: graph.as_ref().unwrap().osr.clone(),
                    source_name: self.source_name,
                    params: params.span,
                    closures: closures.clone(),
                },
            );
        }
        let mut annotations = Annotations {
            semantic_locations: self.dump_ir,
            ..Annotations::default()
        };
        self.context.clear();
        self.context.func.signature = match graph.as_ref().and_then(|graph| graph.graph.arity) {
            Some(arity) => self.fixed_signature(arity),
            None => self.signature.clone(),
        };
        {
            let mut data = FunctionData::new();
            let mut builder = cranelift_frontend::FunctionBuilder::new(
                &mut self.context.func,
                &mut self.builder_context,
            );
            if let Some(graph) = &graph {
                emit::build_function(
                    &mut builder,
                    &mut self.module,
                    self.stubs,
                    &mut annotations,
                    &mut self.names,
                    &mut self.cells,
                    &mut data,
                    self.source_name,
                    params.span,
                    closures,
                    kind,
                    &graph.graph,
                    feedback.map(|owner| emit::Feedback {
                        function_return: true,
                        get_refinement: self.get_refinement,
                        owner: owner.as_u32() as u64,
                    }),
                )?;
            } else if let emit::FunctionKind::Factory(resume) = kind {
                emit::build_factory(
                    &mut builder,
                    &mut self.module,
                    self.stubs,
                    &mut annotations,
                    &mut self.names,
                    &mut self.cells,
                    &mut data,
                    self.source_name,
                    params.span,
                    closures,
                    resume,
                )?;
            }
            builder.seal_all_blocks();
            builder.finalize(self.module.target_config());
            data.seal();
            self.data.push(data);
        }
        Ok(FunctionBody {
            function: std::mem::replace(
                &mut self.context.func,
                cranelift_codegen::ir::Function::new(),
            ),
            annotations,
            listing: if self.dump_ir {
                graph.as_ref().map(|graph| graph.listing_lines())
            } else {
                None
            },
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
                    if parts.is_empty() {
                        parts.push("closure".into());
                    }
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
    semantic: &[super::semantic::ListingLine],
    source: &str,
    filename: &str,
    starts: &[u32],
    sources: &HashMap<usize, Arc<(String, String)>>,
) -> Vec<(u32, String)> {
    let mut result = Vec::new();
    let mut last_location = None;
    let mut source_lines = HashMap::new();
    for (id, location, text) in semantic {
        if let Some(location) = location {
            let origin = match location.source {
                Some(key) => sources.get(&key).map(|source| {
                    let (source, filename) = source.as_ref();
                    let starts = source_lines
                        .entry(key)
                        .or_insert_with(|| line_starts(source));
                    (source.as_str(), filename.as_str(), starts.as_slice())
                }),
                None => Some((source, filename, starts)),
            };
            if let Some((source, filename, starts)) = origin {
                let line = line_at(starts, location.span.start);
                let current = (location.source, line);
                if last_location != Some(current) {
                    result.push((0, String::new()));
                    if last_location.is_none_or(|(source, _)| source != location.source) {
                        result.push((0, format!("  ;;; SOURCE {filename}")));
                    }
                    last_location = Some(current);
                    let start = starts[line as usize - 1] as usize;
                    let end = starts
                        .get(line as usize)
                        .map_or(source.len(), |&end| end as usize);
                    result.push((
                        0,
                        format!("  ;;; JS {line:>5} | {}", source[start..end].trim()),
                    ));
                }
            }
        }
        result.push((*id, text.clone()));
    }
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

#[cfg(test)]
mod listing_tests {
    use super::{ir_lines, line_starts};
    use crate::jit::semantic::SourceLocation;
    use oxc_span::Span;
    use std::collections::HashMap;

    #[test]
    fn semantic_listing_interleaves_source_and_keeps_ids() {
        let source = "let x = 1;\nreturn x;\n";
        let semantic = vec![
            (0, None, "block0:".into()),
            (
                1,
                Some(SourceLocation {
                    source: None,
                    span: Span::new(8, 9),
                }),
                "constant".into(),
            ),
            (
                2,
                Some(SourceLocation {
                    source: None,
                    span: Span::new(8, 9),
                }),
                "store".into(),
            ),
            (
                3,
                Some(SourceLocation {
                    source: None,
                    span: Span::new(11, 20),
                }),
                "return".into(),
            ),
        ];
        let lines = ir_lines(
            &semantic,
            source,
            "caller.js",
            &line_starts(source),
            &HashMap::new(),
        );
        assert_eq!(
            lines
                .iter()
                .filter(|(_, line)| line.contains(";;; JS"))
                .count(),
            2
        );
        assert_eq!(
            lines
                .iter()
                .filter(|(id, _)| *id != 0)
                .map(|(id, _)| *id)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert!(
            lines
                .iter()
                .any(|(_, line)| line.contains("JS     2 | return x;"))
        );
    }
}

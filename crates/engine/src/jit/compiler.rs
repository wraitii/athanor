use crate::{FxHashMap, FxHashSet};
use std::collections::BTreeMap;
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
use super::entries::{Entry, VariantId};
use super::function_data::FunctionData;
use super::machine::MachineCode;
use super::{analysis::CaptureAnalysis, cranelift, finalize, jitdump, syntax};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum CompilationTier {
    Baseline,
    Optimized,
}

impl CompilationTier {
    fn options(self, mut options: EngineOptions) -> EngineOptions {
        if self == Self::Baseline {
            options.opt_level = crate::OptLevel::None;
        }
        options
    }
}

struct Callable<'a> {
    id: FuncId,
    declaration: Option<String>,
    span: oxc_span::Span,
    params: &'a FormalParameters<'a>,
    statements: &'a [Statement<'a>],
    expression: Option<&'a Expression<'a>>,
    generator: bool,
    arrow: bool,
    label: String,
    line: u32,
    captures: Vec<SymbolId>,
}

pub(crate) struct CompiledFunction {
    pub(crate) name: String,
    pub(crate) arity: usize,
    pub(crate) code: u64,
    pub(crate) generator: bool,
}

#[derive(Clone)]
struct Emission {
    tier: CompilationTier,
    specialized: bool,
    callees: Vec<FuncId>,
    plan: super::semantic::SpecializationPlan,
    specialization: Option<Arc<BTreeMap<String, String>>>,
    osr: Option<Arc<super::semantic::osr::Plan>>,
    source_name: (usize, usize),
    params: oxc_span::Span,
}

pub(crate) struct RecompileJob {
    pub(crate) tier: CompilationTier,
    pub(crate) code: FuncId,
    variant: VariantId,
    initial: bool,
    previous: Entry,
    emission: Emission,
    canonical: Arc<super::semantic::specialization::calls::Repository>,
    sources: Arc<FxHashMap<usize, Arc<(String, String)>>>,
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
            super::recompile::Key::Variant {
                function: self.code.as_u32() as u64,
                variant: self.variant,
            },
            |region| super::recompile::Key::Loop(region.site),
        )
    }
}

pub(crate) struct Replacement {
    code: FuncId,
    variant: VariantId,
    initial: bool,
    entry: Entry,
    emission: Emission,
    region: bool,
}

pub(crate) struct Installed {
    pub(crate) function: u64,
    pub(crate) variant: VariantId,
    pub(crate) entry: Entry,
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
    baseline: usize,
    baseline_refusals: BTreeMap<String, usize>,
    blocks: usize,
    insts: usize,
    bytes: usize,
    native: usize,
}

impl Drop for Compiler {
    fn drop(&mut self) {
        if let Some(stats) = &self.stats {
            eprintln!(
                "js jit stats: {} functions, {} blocks, {} instructions, {} code bytes",
                stats.functions, stats.blocks, stats.insts, stats.bytes
            );
            if self.backend != crate::Backend::Cranelift {
                eprintln!("js jit stats: {} native functions", stats.native);
            }
            if std::env::var_os("JS_JIT_BASELINE").is_some() {
                eprintln!(
                    "js jit stats: {} direct baseline functions, {} other functions",
                    stats.baseline,
                    stats.functions - stats.baseline
                );
                for (reason, count) in &stats.baseline_refusals {
                    eprintln!("js jit baseline refusal: {count} functions: {reason}");
                }
            }
        }
    }
}

pub(crate) struct Compiler {
    continuations: Vec<(FuncId, FunctionBody)>,
    log: super::recompile::Log,
    emissions: FxHashMap<VariantId, Emission>,
    lazy: FxHashMap<
        FuncId,
        (
            Emission,
            Arc<super::semantic::specialization::calls::Repository>,
        ),
    >,
    sources: Arc<FxHashMap<usize, Arc<(String, String)>>>,
    names: Vec<Box<str>>,
    cells: Vec<Box<u64>>,
    data: Vec<Box<FunctionData>>,
    source_name: (usize, usize),
    canonical: Arc<super::semantic::specialization::calls::Repository>,
    module: JITModule,
    signature: Signature,
    fixed: FxHashMap<FuncId, u16>,
    adapters: FxHashMap<u16, usize>,
    stats: Option<CompileStats>,
    stubs: cranelift::stubs::Stubs,
    functions: FxHashMap<String, FuncId>,
    function_names: FxHashMap<FuncId, String>,
    dump: Option<jitdump::Writer>,
    dump_ir: bool,
    ssa_opt: bool,
    inlining: bool,
    recompile_inline_multiplier: usize,
    osr: bool,
    speculative_continuations: bool,
    generic_first: bool,
    get_refinement: bool,
    specialization_log: Option<String>,
    backend: crate::Backend,
    inspection: Option<Arc<super::inspect::CompileOptions>>,
    next_id: u64,
    context: Context,
    builder_context: FunctionBuilderContext,
}

impl Compiler {
    #[cfg(test)]
    pub(super) fn specialization_plan(
        &self,
        variant: VariantId,
    ) -> Option<&super::semantic::SpecializationPlan> {
        self.emissions.get(&variant).map(|emission| &emission.plan)
    }

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
        super::profile::Site,
        super::profile::ReadKind,
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
                        let super::profile::Observation::Read(kind) = opportunity.kind else {
                            return None;
                        };
                        Some((site as *const _, opportunity.site, kind))
                    })
            })
    }

    pub(crate) fn new(
        options: EngineOptions,
        tier: CompilationTier,
        direct_entries: *const usize,
        compilation_state: usize,
        log: super::recompile::Log,
    ) -> Result<Self, Error> {
        let options = tier.options(options);
        let flags = [
            ("opt_level", options.opt_level.as_flag()),
            ("regalloc_algorithm", "backtracking"),
            ("enable_verifier", "false"),
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
        let stubs = cranelift::stubs::Stubs::new(
            &mut module,
            &signature,
            direct_entries,
            compilation_state,
            options.call_profile,
            dump.as_mut(),
        )?;
        let context = module.make_context();
        let dump_ir = dump.is_some() && options.jitdump_ir;
        Ok(Self {
            log,
            osr: options.osr,
            speculative_continuations: options.continuations,
            generic_first: options.generic_first && options.background_compile,
            emissions: FxHashMap::default(),
            lazy: FxHashMap::default(),
            sources: Arc::default(),
            names: Vec::new(),
            cells: Vec::new(),
            data: Vec::new(),
            continuations: Vec::new(),
            source_name: (0, 0),
            canonical: Arc::new(
                super::semantic::specialization::calls::Repository::with_call_options(
                    options.call_variants,
                    options.call_specialization,
                ),
            ),
            module,
            signature,
            fixed: FxHashMap::default(),
            adapters: FxHashMap::default(),
            stats: std::env::var_os("JS_JIT_STATS").map(|_| CompileStats::default()),
            stubs,
            functions: FxHashMap::default(),
            function_names: FxHashMap::default(),
            dump,
            dump_ir,
            ssa_opt: options.ssa_opt,
            inlining: options.inlining,
            recompile_inline_multiplier: options.recompile_inline_multiplier,
            get_refinement: options.get_refinement,
            specialization_log: options.specialization_log,
            backend: options.backend,
            inspection: options.compile,
            next_id: 0,
            context,
            builder_context: FunctionBuilderContext::new(),
        })
    }

    fn fixed_signature(&self, arity: u16) -> Signature {
        cranelift::stubs::fixed_signature(self.module.target_config().pointer_type(), arity)
    }

    fn graph_signature(&self, graph: Option<&super::semantic::Graph>) -> Signature {
        match graph.and_then(|graph| graph.arity) {
            Some(arity) => self.fixed_signature(arity),
            None => self.signature.clone(),
        }
    }

    fn listing(&self) -> bool {
        self.dump_ir || (self.dump.is_some() && self.inspection.is_some())
    }

    fn report_regions(&self) -> bool {
        self.inspection
            .as_ref()
            .is_some_and(|options| options.planner)
    }

    fn feedback(&self, owner: FuncId, function_return: bool) -> finalize::Feedback {
        finalize::Feedback {
            function_return,
            get_refinement: self.get_refinement,
            owner: owner.as_u32() as u64,
        }
    }

    fn emit_body(
        &mut self,
        signature: Signature,
        source_name: (usize, usize),
        build: impl FnOnce(finalize::Output<'_, '_>) -> Result<(), Error>,
    ) -> Result<
        (
            cranelift_codegen::ir::Function,
            Annotations,
            Option<super::semantic::Graph>,
            Option<(String, String)>,
        ),
        Error,
    > {
        let pass = self
            .inspection
            .as_deref()
            .filter(|options| options.semantic_ir)
            .and_then(|options| options.after.clone());
        let mut snapshot = None;
        let mut after = |name: &str, graph: &super::semantic::Graph| {
            if pass.as_deref() == Some(name) {
                snapshot = Some((name.to_owned(), super::inspect::graph_text(graph)));
            }
        };
        let mut annotations = Annotations {
            semantic_locations: self.dump_ir || self.inspection.is_some(),
            ..Annotations::default()
        };
        self.context.clear();
        self.context.func.signature = signature;
        let mut data = FunctionData::new();
        let mut builder = cranelift_frontend::FunctionBuilder::new(
            &mut self.context.func,
            &mut self.builder_context,
        );
        let mut lowered = None;
        let native =
            self.backend != crate::Backend::Cranelift && self.module.isa().name() == "aarch64";
        if pass.as_deref() == Some("aarch64-select") && !native {
            return Err(Error::Compile(
                "aarch64-select requires the native AArch64 backend".into(),
            ));
        }
        let clif = !native
            || self.inspection.as_ref().is_some_and(|options| {
                options.clif
                    || matches!(
                        options.after.as_deref(),
                        Some("legalize-convert" | "bits" | "cranelift-memory")
                    )
            });
        build(finalize::Output {
            b: &mut builder,
            clif,
            native,
            lowered: &mut lowered,
            module: &mut self.module,
            stubs: self.stubs,
            annotations: &mut annotations,
            names: &mut self.names,
            cells: &mut self.cells,
            data: &mut data,
            source_name,
            after: &mut after,
        })?;
        builder.seal_all_blocks();
        builder.finalize(self.module.target_config());
        if pass.as_deref() == Some("aarch64-select") {
            let mut selected = lowered.as_ref().unwrap().clone();
            super::aarch64::select_graph(&mut selected)
                .map_err(|reason| Error::Compile(reason.0))?;
            snapshot = Some((
                "aarch64-select".into(),
                super::inspect::graph_text(&selected),
            ));
        }
        data.seal();
        self.data.push(data);
        let function = std::mem::replace(
            &mut self.context.func,
            cranelift_codegen::ir::Function::new(),
        );
        Ok((function, annotations, lowered, snapshot))
    }

    fn machine_code(
        &mut self,
        id: FuncId,
        name: &str,
        annotations: &mut Annotations,
        graph: Option<&super::semantic::Graph>,
    ) -> Result<MachineCode, Error> {
        let native =
            self.backend != crate::Backend::Cranelift && self.module.isa().name() == "aarch64";
        if !native {
            return MachineCode::from_cranelift(&mut self.context, &self.module, id);
        }
        let Some(graph) = graph else {
            return MachineCode::from_cranelift(&mut self.context, &self.module, id);
        };
        let result = super::aarch64::compile_graph(graph);
        let reason = match result {
            Ok(machine) => {
                if let Some(stats) = &mut self.stats {
                    stats.native += 1;
                }
                return Ok(machine);
            }
            Err(super::aarch64::Unsupported(reason)) => reason,
        };
        if self.backend == crate::Backend::NativeOnly {
            return Err(Error::Compile(format!(
                "native backend cannot compile {name}: {reason}"
            )));
        }
        if self.inspection.as_deref().is_some_and(|o| o.matches(name)) {
            println!("[{name}] native backend fallback: {reason}");
        }
        if self.context.func.layout.entry_block().is_none() {
            let mut graph = graph.clone();
            cranelift::prepare(&mut graph, &mut |_, _| {}).map_err(Error::Compile)?;
            let mut builder = cranelift_frontend::FunctionBuilder::new(
                &mut self.context.func,
                &mut self.builder_context,
            );
            cranelift::emit(&mut builder, &mut self.module, &graph, Some(annotations))?;
            builder.seal_all_blocks();
            builder.finalize(self.module.target_config());
        }
        MachineCode::from_cranelift(&mut self.context, &self.module, id)
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
        let stub = cranelift::stubs::build_adapter(&mut self.module, &self.signature, arity)?;
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
            self.continuations.clear();
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
        let lines = if self.dump.is_some() || self.inspection.is_some() {
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
                    syntax::validate_class(class)?;
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
                Statement::ImportDeclaration(_)
                | Statement::ExportAllDeclaration(_)
                | Statement::ExportDefaultDeclaration(_)
                | Statement::ExportNamedDeclaration(_) => {
                    return Err(unsupported_at(
                        statement.span(),
                        format!(
                            "{} statement at top level",
                            syntax::statement_kind(statement)
                        ),
                    ));
                }
                _ => {}
            }
        }
        for node in nodes.iter() {
            let AstKind::VariableDeclaration(declaration) = node.kind() else {
                continue;
            };
            if declaration.kind != VariableDeclarationKind::Var {
                continue;
            }
            for declarator in &declaration.declarations {
                let BindingPattern::BindingIdentifier(identifier) = &declarator.id else {
                    continue;
                };
                let Some(symbol) = identifier.symbol_id.get() else {
                    continue;
                };
                if scoping.symbol_scope_id(symbol) == scoping.root_scope_id()
                    && !globals.iter().any(|name| name == identifier.name.as_str())
                {
                    globals.push(identifier.name.to_string());
                }
            }
        }

        let mut ids = self.functions.clone();
        let mut callables = Vec::new();
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
            callables.push(Callable {
                id,
                span: function.span,
                params: &function.params,
                statements: &function.body.as_ref().unwrap().statements,
                expression: None,
                generator: function.generator,
                arrow: false,
                label: name.clone(),
                line,
                captures: Vec::new(),
                declaration: Some(name),
            });
        }
        let declared = callables.len();

        let analysis = CaptureAnalysis::analyze(program, scoping);
        if let Some(span) = analysis.class_name_write {
            return Err(unsupported_at(span, "assignment to inner class name"));
        }
        let mut closures = FxHashMap::default();
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
                    if !callables[..declared]
                        .iter()
                        .any(|top| top.span == function.span) =>
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
            let function = analysis.functions.get(&span.start);
            let captures = function.map(|f| f.captures.clone()).unwrap_or_default();
            let lexical_this = function.is_some_and(|function| function.lexical_this);
            closures.insert(span.start, (id, captures.clone(), lexical_this));
            callables.push(Callable {
                id,
                span,
                params,
                statements,
                expression,
                generator,
                arrow: matches!(node.kind(), AstKind::ArrowFunctionExpression(_)),
                label,
                line,
                captures,
                declaration: None,
            });
        }
        let mut pending = Vec::new();
        for node in nodes.iter() {
            if let AstKind::Class(class) = node.kind() {
                syntax::validate_class(class)?;
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
                let code = self.compile_callable(
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

        for callable in &callables {
            if callable.generator {
                continue;
            }
            let canonical = super::semantic::build_canonical(
                callable.params,
                callable.statements,
                callable.expression,
                callable.span,
                scoping,
                &analysis.captured,
                &callable.captures,
                &closures,
                analysis
                    .arguments_bindings
                    .get(&callable.span.start)
                    .copied(),
                &analysis.arguments_references,
                false,
                self.ssa_opt,
            )
            .map_err(Error::Unsupported)?;
            let repository = Arc::make_mut(&mut self.canonical);
            match &callable.declaration {
                Some(name) => repository.insert(self.source_name.0, name, callable.id, canonical),
                None => repository.insert_closure(
                    self.source_name.0,
                    callable.id,
                    canonical,
                    callable.arrow,
                ),
            }
        }

        for callable in &callables {
            if !callable.generator && self.inspection.is_none() {
                self.lazy.insert(
                    callable.id,
                    (
                        Emission {
                            tier: CompilationTier::Optimized,
                            specialized: false,
                            callees: Vec::new(),
                            plan: Default::default(),
                            specialization: None,
                            osr: None,
                            source_name: self.source_name,
                            params: callable.params.span,
                        },
                        self.canonical.clone(),
                    ),
                );
                continue;
            }
            let code = self.compile_callable(
                callable.params,
                callable.statements,
                callable.expression,
                analysis
                    .arguments_bindings
                    .get(&callable.span.start)
                    .copied(),
                &analysis.arguments_references,
                &callable.captures,
                &analysis.captured,
                &closures,
                scoping,
                callable.generator,
                &callable.label,
                callable.line,
                Some(callable.span),
            )?;
            pending.push((callable.id, code));
        }
        let initializer = if program.body.iter().all(|s| {
            matches!(
                s,
                Statement::FunctionDeclaration(_) | Statement::EmptyStatement(_)
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
            let mut graph = super::semantic::lower_global(
                &program.body,
                program.span,
                scoping,
                &analysis.captured,
                &closures,
                &analysis.arguments_references,
                self.ssa_opt,
                self.speculative_continuations,
                self.report_regions(),
            )
            .map_err(Error::Unsupported)?;
            if self.ssa_opt {
                graph.lower_typed_storage().map_err(Error::Unsupported)?;
            }
            let continuations = self.compile_continuations(
                &graph.graph,
                self.source_name,
                program.span,
                (&label, line),
                None,
            )?;
            let mut baseline_data = super::function_data::FunctionData::new();
            let baseline = if !self.ssa_opt
                && self.module.isa().name() == "aarch64"
                && std::env::var_os("JS_JIT_BASELINE").is_some()
            {
                super::aarch64::compile_baseline(
                    &graph.graph,
                    self.source_name,
                    self.dump_ir || self.inspection.is_some(),
                    &mut self.names,
                    &mut baseline_data,
                )
                .map_err(|reason| {
                    if let Some(stats) = &mut self.stats {
                        *stats.baseline_refusals.entry(reason.0).or_default() += 1;
                    }
                })
                .ok()
                .map(Arc::new)
            } else {
                None
            };
            let mut after = None;
            if baseline.is_none() {
                after = super::inspect::lower_runtime(self.inspection.as_deref(), &mut graph)
                    .map_err(Error::Unsupported)?;
            }
            let inspection = super::inspect::Inspection::new(
                self.inspection.as_deref(),
                &graph,
                &self.canonical,
                self.source_name.0,
                1,
            )
            .after(after)
            .ast(self.inspection.as_deref(), &program);
            let listing = self.listing().then(|| graph.listing_lines());
            let (function, annotations, lowered, snapshot) = if baseline.is_some() {
                self.data.push(baseline_data);
                let mut function = cranelift_codegen::ir::Function::new();
                function.signature = self.signature.clone();
                (
                    function,
                    Annotations {
                        semantic_locations: self.dump_ir || self.inspection.is_some(),
                        ..Annotations::default()
                    },
                    None,
                    None,
                )
            } else {
                self.emit_body(self.signature.clone(), self.source_name, |out| {
                    finalize::build_global(out, graph.graph, &graph.facts, &continuations)
                })?
            };
            let inspection = inspection.lowered(snapshot);
            pending.push((
                id,
                FunctionBody {
                    function,
                    lowered,
                    baseline,
                    annotations,
                    inspection,
                    listing,
                },
            ));
            Some(id)
        };

        let compiled_ids: Vec<_> = pending
            .iter()
            .chain(&self.continuations)
            .map(|(id, _)| *id)
            .filter(|id| Some(*id) != initializer)
            .collect();

        if let Some(options) = self.inspection.as_deref()
            && let Some(filter) = &options.filter
            && !pending.iter().chain(&self.continuations).any(|(id, _)| {
                self.function_names
                    .get(id)
                    .is_some_and(|name| options.matches(name))
            })
        {
            return Err(Error::Compile(format!(
                "no functions match filter `{filter}`"
            )));
        }
        self.finish_functions(pending, source, source_path, &lines)?;

        let functions = callables
            .iter()
            .filter_map(|callable| {
                Some(CompiledFunction {
                    name: callable.declaration.clone()?,
                    arity: callable.params.items.len(),
                    code: callable.id.as_u32() as u64,
                    generator: callable.generator,
                })
            })
            .collect();
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
        mut pending: Vec<(FuncId, FunctionBody)>,
        source: &str,
        source_path: &str,
        lines: &[u32],
    ) -> Result<(), Error> {
        let mut continuations = std::mem::take(&mut self.continuations);
        continuations.append(&mut pending);
        let mut dumped = Vec::new();
        for (id, mut code) in continuations {
            let direct = code.baseline.is_some();
            let mut listing = None;
            self.context.clear();
            self.context.func = code.function;
            let name = self
                .function_names
                .get(&id)
                .cloned()
                .unwrap_or_else(|| format!("jsfn_{}", id.as_u32()));
            if self.dump_ir
                && let Some(semantic) = &code.listing
            {
                let ir = ir_lines(semantic, source, source_path, lines, &self.sources);
                if let Some(dump) = self.dump.as_mut() {
                    match dump.listing(&name, &ir) {
                        Ok(written) => listing = Some(written),
                        Err(error) => eprintln!("js jit: listing write failed: {error}"),
                    }
                }
            }
            if let Some(options) = self.inspection.as_deref().filter(|o| o.matches(&name)) {
                options.print_body(
                    &name,
                    &code.inspection,
                    &self.context.func,
                    &code.annotations,
                    direct,
                );
            }
            if let Some(stats) = &mut self.stats {
                stats.functions += 1;
                stats.baseline += usize::from(direct);
                if let Some(graph) = code.lowered.as_ref().filter(|_| {
                    self.backend != crate::Backend::Cranelift
                        && self.module.isa().name() == "aarch64"
                }) {
                    let blocks = super::semantic::Cfg::new(graph).rpo;
                    stats.blocks += blocks.len();
                    stats.insts += blocks
                        .iter()
                        .map(|&block| {
                            graph.blocks[block].insts.len()
                                + usize::from(graph.blocks[block].terminator.is_some())
                        })
                        .sum::<usize>();
                } else {
                    stats.blocks += self.context.func.layout.blocks().count();
                    stats.insts += self
                        .context
                        .func
                        .layout
                        .blocks()
                        .map(|block| self.context.func.layout.block_insts(block).count())
                        .sum::<usize>();
                }
            }
            if self.context.func.signature.call_conv == CallConv::Tail {
                let arity = self.context.func.signature.params.len() - 3;
                self.fixed.insert(id, arity as u16);
            } else {
                self.fixed.remove(&id);
            }
            let machine = match code.baseline {
                Some(machine) => machine,
                None => Arc::new(self.machine_code(
                    id,
                    &name,
                    &mut code.annotations,
                    code.lowered.as_ref(),
                )?),
            };
            machine.define(&mut self.module, id)?;
            if let Some(stats) = &mut self.stats {
                stats.bytes += machine.bytes.len();
            }
            if let Some(options) = self.inspection.as_deref().filter(|o| o.matches(&name)) {
                options.print_compiled(&name, &self.context, &machine, &self.module, direct)?;
            }
            if self.dump.is_some() {
                let size = machine.bytes.len();
                let mut locations = BTreeMap::new();
                let ranges = &machine.locations;
                if self.dump_ir {
                    locations.insert(0, 0);
                    for range in ranges {
                        if (range.end as usize) < size {
                            locations.entry(range.end).or_insert(0);
                        }
                    }
                    for range in ranges {
                        locations.insert(range.start, range.location.unwrap_or(0));
                    }
                } else {
                    let source_locations: FxHashMap<_, _> = code
                        .listing
                        .iter()
                        .flatten()
                        .filter_map(|(id, location, _)| {
                            let location = location.as_ref()?;
                            location
                                .source
                                .is_none_or(|source| source == self.source_name.0)
                                .then_some((*id, location.span.start))
                        })
                        .collect();
                    let mut default_line = None;
                    for range in ranges {
                        let Some(offset) = range.location else {
                            continue;
                        };
                        let offset = if code.annotations.semantic_locations {
                            let Some(offset) = source_locations.get(&offset) else {
                                continue;
                            };
                            *offset
                        } else {
                            offset
                        };
                        let line = line_at(lines, offset);
                        default_line.get_or_insert(line);
                        locations.insert(range.start, line);
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

    pub(crate) fn update_feedback(&mut self, profile: &super::Profile) {
        if profile.targets().all(|(site, target)| {
            self.canonical.feedback.get(&site).is_some_and(|known| {
                known
                    .iter()
                    .map(|target| u64::from(target.as_u32()))
                    .eq(target.iter().copied())
            })
        }) && profile
            .natives()
            .all(|(site, target)| self.canonical.native_feedback.get(&site) == Some(&target))
            && (!self.get_refinement
                || profile.reads().all(|(site, observed)| {
                    self.canonical.read_feedback.get(&site) == Some(&observed)
                }) && profile.shapes().all(|(site, access)| {
                    self.canonical.shape_feedback.get(&site) == Some(&access)
                }) && profile.number_first().all(|code| {
                    u32::try_from(code).map_or(true, |code| {
                        self.canonical
                            .number_first
                            .contains(&FuncId::from_u32(code))
                    })
                }))
        {
            return;
        }
        let canonical = Arc::make_mut(&mut self.canonical);
        for (site, targets) in profile.targets() {
            canonical.feedback.insert(
                site,
                targets
                    .iter()
                    .filter_map(|&target| u32::try_from(target).ok().map(FuncId::from_u32))
                    .collect(),
            );
        }
        for (site, target) in profile.natives() {
            canonical.native_feedback.insert(site, target);
        }
        if self.get_refinement {
            for code in profile.number_first() {
                if let Ok(code) = u32::try_from(code) {
                    canonical.number_first.insert(FuncId::from_u32(code));
                }
            }
            for (site, element) in profile.reads() {
                canonical.read_feedback.insert(site, element);
            }
            for (site, access) in profile.shapes() {
                canonical.shape_feedback.insert(site, access);
            }
        }
    }

    pub(crate) fn call_profile(&self, profile: &super::Profile) -> String {
        let mut sites = profile.calls();
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

    fn specialization_entries(
        &self,
        label: &str,
        graph: &super::semantic::Graph,
        plan: &super::semantic::SpecializationPlan,
        source: usize,
    ) -> Option<Arc<BTreeMap<String, String>>> {
        let filter = self.specialization_log.as_deref()?;
        if !matches!(filter, "1" | "*") && !label.contains(filter) {
            return None;
        }
        Some(Arc::new(graph.specialization_entries(
            plan,
            &self.canonical,
            source,
            |mut site| {
                if site.source == 0 {
                    site.source = source;
                }
                let filename = self
                    .sources
                    .get(&site.source)
                    .map_or("<unknown>", |source| source.1.as_str());
                format!("{filename}:{}..{}", site.start, site.end)
            },
        )))
    }

    fn log_specialization(
        &self,
        code: FuncId,
        label: &str,
        action: &str,
        previous: Option<&BTreeMap<String, String>>,
        current: Option<&BTreeMap<String, String>>,
    ) {
        use std::fmt::Write;

        let Some(current) = current else {
            return;
        };
        let empty = BTreeMap::new();
        let previous = previous.unwrap_or(&empty);
        let mut changes = String::new();
        for (key, before) in previous {
            match current.get(key) {
                None => writeln!(changes, "  - {key} = {before}").unwrap(),
                Some(after) if before != after => {
                    writeln!(changes, "  ~ {key}: {before} -> {after}").unwrap();
                }
                _ => {}
            }
        }
        for (key, after) in current {
            if !previous.contains_key(key) {
                writeln!(changes, "  + {key} = {after}").unwrap();
            }
        }
        if changes.is_empty() {
            changes.push_str("  no changes\n");
        }
        eprintln!(
            "[js jit specialization] code={} function={label} action={action}\n{changes}",
            code.as_u32()
        );
    }

    pub(crate) fn function_name(&self, code: u64) -> &str {
        u32::try_from(code)
            .ok()
            .and_then(|code| self.function_names.get(&FuncId::from_u32(code)))
            .map_or("<unknown>", String::as_str)
    }

    pub(crate) fn recompilation(&self, code: u64, previous: Entry) -> Option<RecompileJob> {
        let variant = VariantId::ordinary(code);
        let code = FuncId::from_u32(u32::try_from(code).ok()?);
        Some(RecompileJob {
            tier: CompilationTier::Optimized,
            region: None,
            code,
            variant,
            initial: false,
            previous,
            emission: self.emissions.get(&variant)?.clone(),
            canonical: self.canonical.clone(),
            sources: self.sources.clone(),
            label: self.function_names.get(&code)?.clone(),
            queued_at: None,
        })
    }

    pub(crate) fn variants_for_function(
        &self,
        function: u64,
    ) -> Vec<Arc<super::variants::Variant>> {
        self.canonical.variants.for_function(function)
    }

    pub(crate) fn variant(&self, id: VariantId) -> Option<Arc<super::variants::Variant>> {
        self.canonical.variants.get(id)
    }

    pub(crate) fn variant_compilation(
        &self,
        id: VariantId,
        previous: Entry,
    ) -> Option<RecompileJob> {
        let variant = self.variant(id)?;
        let mut job = self.recompilation(variant.function, previous)?;
        job.variant = id;
        if let Some(emission) = self.emissions.get(&id) {
            job.emission = emission.clone();
        } else {
            job.emission.specialized = false;
        }
        job.label += &format!(".variant{:?}", variant.requirements);
        Some(job)
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
                    self.canonical.observed(location, opportunity.kind)
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
            self.log.event(format_args!(
                "compiling code={code} function={label} tier={:?}",
                job.tier
            ));
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
            self.continuations.clear();
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
            tier,
            code,
            variant,
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
        let (source, mut canonical) = if let Some(region) = &region {
            (
                emission.source_name.0,
                super::semantic::FunctionState::from_graph(
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
        if region.is_none()
            && let Some(variant) = self.variant(variant)
        {
            super::semantic::specialization::call_variants::assume(
                &mut canonical.graph,
                &variant.requirements,
            );
        }
        let inline_multiplier = if !self.inlining {
            0
        } else if initial {
            1
        } else {
            self.recompile_inline_multiplier
        };
        let planned = super::semantic::specialize_with_budget(
            canonical,
            self.ssa_opt,
            Some((&self.canonical, source)),
            inline_multiplier,
        )
        .map_err(Error::Unsupported)?;
        emission.callees = self.canonical.graph_callees(&planned.graph, source);
        let specialization =
            self.specialization_entries(&label, &planned.graph, &planned.plan, source);
        let previous_specialization = if region.is_some() {
            None
        } else {
            emission.specialization.as_deref()
        };
        if region.is_some() && planned.plan == Default::default() {
            self.log_specialization(
                code,
                &label,
                "skip-osr-empty",
                previous_specialization,
                specialization.as_deref(),
            );
            return Ok(None);
        }
        if self.log.enabled() {
            self.log.event(format_args!(
                "plan code={} function={label} inlines={}->{} reads={}->{} numeric_changed={} conditional_changed={} inline_multiplier={inline_multiplier}",
                code.as_u32(),
                emission.plan.inlines.len(),
                planned.plan.inlines.len(),
                emission.plan.reads.len(),
                planned.plan.reads.len(),
                emission.plan.numbers != planned.plan.numbers,
                emission.plan.conditional != planned.plan.conditional,
            ));
        }
        if !initial
            && emission.specialized
            && emission.tier == tier
            && region.is_none()
            && planned.plan == emission.plan
        {
            self.log_specialization(
                code,
                &label,
                "skip-unchanged",
                previous_specialization,
                specialization.as_deref(),
            );
            return Ok(None);
        }
        let versioning = if initial && region.is_none() && self.generic_first {
            super::semantic::Versioning::Generic
        } else {
            super::semantic::Versioning::Speculate {
                fallback: (!initial && region.is_none()).then_some(previous.code),
                continuations: self.speculative_continuations,
            }
        };
        self.log_specialization(
            code,
            &label,
            match versioning {
                super::semantic::Versioning::Generic => "compile-generic",
                _ if region.is_some() => "compile-osr",
                _ => "compile-specialized",
            },
            previous_specialization,
            specialization.as_deref(),
        );
        emission.specialization = specialization;
        emission.tier = tier;
        emission.specialized = matches!(versioning, super::semantic::Versioning::Speculate { .. });
        let mut graph =
            super::semantic::optimize_specialized(planned, versioning, self.report_regions())
                .map_err(Error::Unsupported)?;
        if self.osr && region.is_none() && !self.canonical.is_region(code) {
            graph.prepare_osr(source).map_err(Error::Unsupported)?;
        }
        if self.ssa_opt {
            graph.lower_typed_storage().map_err(Error::Unsupported)?;
        }
        let continuations = self.compile_continuations(
            &graph.graph,
            emission.source_name,
            emission.params,
            (&label, 1),
            self.ssa_opt.then(|| self.feedback(code, false)),
        )?;
        let signature = self.graph_signature(Some(&graph.graph));
        let feedback = (self.ssa_opt && !self.canonical.is_region(code))
            .then(|| self.feedback(code, region.is_none()));
        let mut baseline_data = super::function_data::FunctionData::new();
        let baseline = if initial
            && region.is_none()
            && !self.ssa_opt
            && self.module.isa().name() == "aarch64"
            && std::env::var_os("JS_JIT_BASELINE").is_some()
        {
            super::aarch64::compile_baseline(
                &graph.graph,
                emission.source_name,
                self.dump_ir || self.inspection.is_some(),
                &mut self.names,
                &mut baseline_data,
            )
            .map_err(|reason| {
                if let Some(stats) = &mut self.stats {
                    *stats.baseline_refusals.entry(reason.0).or_default() += 1;
                }
            })
            .ok()
            .map(Arc::new)
        } else {
            None
        };
        let mut after = None;
        if baseline.is_some() {
            self.data.push(baseline_data);
        } else {
            after = super::inspect::lower_runtime(self.inspection.as_deref(), &mut graph)
                .map_err(Error::Unsupported)?;
        }
        let listing = self.listing().then(|| graph.listing_lines());
        let inspection = super::inspect::Inspection::new(
            self.inspection.as_deref(),
            &graph,
            &self.canonical,
            source,
            inline_multiplier,
        )
        .after(after);
        let (function, annotations, lowered, snapshot) = if baseline.is_some() {
            let mut function = cranelift_codegen::ir::Function::new();
            function.signature = signature;
            (
                function,
                Annotations {
                    semantic_locations: self.dump_ir || self.inspection.is_some(),
                    ..Annotations::default()
                },
                None,
                None,
            )
        } else {
            self.emit_body(signature, emission.source_name, |out| {
                finalize::build_function(
                    out,
                    emission.params,
                    graph.graph,
                    &graph.facts,
                    &continuations,
                    feedback,
                )
            })?
        };
        let inspection = inspection.lowered(snapshot);
        let body = FunctionBody {
            function,
            lowered,
            baseline,
            annotations,
            listing,
            inspection,
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
        emission.plan = graph.plan;
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
            variant,
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

    pub(crate) fn lazy_codes(&self) -> Vec<u64> {
        let mut codes: Vec<_> = self.lazy.keys().map(|id| u64::from(id.as_u32())).collect();
        codes.sort_unstable();
        codes
    }

    pub(crate) fn lazy_compilation(&self, code: u64) -> Option<RecompileJob> {
        let id = FuncId::from_u32(u32::try_from(code).ok()?);
        let (emission, canonical) = self.lazy.get(&id)?.clone();
        Some(RecompileJob {
            tier: CompilationTier::Optimized,
            region: None,
            code: id,
            variant: VariantId::ordinary(code),
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
        if let Some(emission) = self.emissions.get(&VariantId::ordinary(code)) {
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
            self.continuations.clear();
            self.context.clear();
            self.builder_context = FunctionBuilderContext::new();
        }
        let Some(replacement) = result? else {
            return Ok(None);
        };
        Ok(Some(self.install(replacement).entry))
    }

    pub(crate) fn install(&mut self, replacement: Replacement) -> Installed {
        if replacement.initial {
            self.lazy.remove(&replacement.code);
        }
        if !replacement.region {
            self.emissions
                .insert(replacement.variant, replacement.emission);
        }
        Installed {
            function: replacement.code.as_u32() as u64,
            variant: replacement.variant,
            entry: replacement.entry,
        }
    }

    pub(crate) fn take_unpublished(&mut self, replacement: Replacement) -> Unpublished {
        Unpublished {
            replacement,
            names: std::mem::take(&mut self.names),
            cells: std::mem::take(&mut self.cells),
            data: std::mem::take(&mut self.data),
        }
    }

    pub(crate) fn install_unpublished(&mut self, compiled: Unpublished) -> Option<Installed> {
        // Dispatch may have compiled this lazy body while the worker was running.
        if compiled.replacement.initial && !self.lazy.contains_key(&compiled.replacement.code) {
            return None;
        }
        self.names.extend(compiled.names);
        self.cells.extend(compiled.cells);
        self.data.extend(compiled.data);
        Some(self.install(compiled.replacement))
    }

    fn compile_continuations(
        &mut self,
        graph: &super::semantic::Graph,
        source_name: (usize, usize),
        params: oxc_span::Span,
        name: (&str, u32),
        feedback: Option<finalize::Feedback>,
    ) -> Result<FxHashMap<usize, FuncId>, Error> {
        let mut ids = FxHashMap::default();
        self.compile_continuations_inner(graph, source_name, params, name, feedback, &mut ids)?;
        Ok(ids)
    }

    fn compile_continuations_inner(
        &mut self,
        graph: &super::semantic::Graph,
        source_name: (usize, usize),
        params: oxc_span::Span,
        name: (&str, u32),
        feedback: Option<finalize::Feedback>,
        ids: &mut FxHashMap<usize, FuncId>,
    ) -> Result<(), Error> {
        let cfg = super::semantic::Cfg::new(graph);
        for &inst in cfg.rpo.iter().flat_map(|&block| &graph.blocks[block].insts) {
            let super::semantic::Op::Continuation(target, _) = &graph.insts[inst].op else {
                continue;
            };
            let key = Arc::as_ptr(target) as usize;
            if ids.contains_key(&key) {
                continue;
            }
            let id = self.declare_js(&format!("{}.continuation", name.0), name.1)?;
            ids.insert(key, id);
            self.compile_continuations_inner(
                &target.graph,
                source_name,
                params,
                name,
                feedback,
                ids,
            )?;
            let mut body =
                self.compile_region(&target.graph, source_name, params, ids, feedback)?;
            body.inspection = body
                .inspection
                .named(&format!("continuation of {}", name.0));
            self.continuations.push((id, body));
        }
        Ok(())
    }

    fn compile_region(
        &mut self,
        graph: &super::semantic::Graph,
        source_name: (usize, usize),
        params: oxc_span::Span,
        continuations: &FxHashMap<usize, FuncId>,
        feedback: Option<finalize::Feedback>,
    ) -> Result<FunctionBody, Error> {
        let mut state = super::semantic::FunctionState::from_graph(graph.clone());
        if self.ssa_opt {
            state.lower_typed_storage().map_err(Error::Unsupported)?;
        }
        let mut data = FunctionData::new();
        let baseline = if !self.ssa_opt
            && self.module.isa().name() == "aarch64"
            && std::env::var_os("JS_JIT_BASELINE").is_some()
        {
            super::aarch64::compile_baseline(
                &state.graph,
                source_name,
                self.dump_ir || self.inspection.is_some(),
                &mut self.names,
                &mut data,
            )
            .map_err(|reason| {
                if let Some(stats) = &mut self.stats {
                    *stats.baseline_refusals.entry(reason.0).or_default() += 1;
                }
            })
            .ok()
            .map(Arc::new)
        } else {
            None
        };
        let mut after = None;
        if baseline.is_some() {
            self.data.push(data);
        } else {
            after = super::inspect::lower_runtime(self.inspection.as_deref(), &mut state)
                .map_err(Error::Unsupported)?;
        }
        let inspection =
            super::inspect::Inspection::from_graph(self.inspection.as_deref(), &state.graph)
                .after(after);
        let listing = self
            .listing()
            .then(|| state.graph.listing(&state.facts).lines());
        let (function, annotations, lowered, snapshot) = if baseline.is_some() {
            let mut function = cranelift_codegen::ir::Function::new();
            function.signature = self.signature.clone();
            (
                function,
                Annotations {
                    semantic_locations: self.dump_ir || self.inspection.is_some(),
                    ..Annotations::default()
                },
                None,
                None,
            )
        } else {
            self.emit_body(self.signature.clone(), source_name, |out| {
                finalize::build_function(
                    out,
                    params,
                    state.graph,
                    &state.facts,
                    continuations,
                    feedback,
                )
            })?
        };
        let inspection = inspection.lowered(snapshot);
        Ok(FunctionBody {
            function,
            lowered,
            annotations,
            baseline,
            inspection,
            listing,
        })
    }

    fn compile_callable(
        &mut self,
        params: &FormalParameters<'_>,
        statements: &[Statement<'_>],
        expression: Option<&Expression<'_>>,
        arguments_binding: Option<SymbolId>,
        arguments_references: &FxHashMap<u32, SymbolId>,
        captures: &[SymbolId],
        captured: &FxHashSet<SymbolId>,
        closures: &FxHashMap<u32, (FuncId, Vec<SymbolId>, bool)>,
        scoping: &Scoping,
        generator: bool,
        label: &str,
        line: u32,
        semantic_function: Option<oxc_span::Span>,
    ) -> Result<FunctionBody, Error> {
        if !generator {
            let mut body = self.compile_function(
                params,
                statements,
                expression,
                arguments_binding,
                arguments_references,
                captures,
                captured,
                closures,
                scoping,
                finalize::FunctionKind::Ordinary,
                (label, line),
                semantic_function,
            )?;
            body.inspection = body.inspection.named(label).ast(
                self.inspection.as_deref(),
                &(params, statements, expression),
            );
            return Ok(body);
        }
        super::generator::validate(statements)?;
        let canonical = super::semantic::build_canonical(
            params,
            statements,
            expression,
            semantic_function.unwrap(),
            scoping,
            captured,
            captures,
            closures,
            arguments_binding,
            arguments_references,
            true,
            false,
        )
        .map_err(Error::Unsupported)?;
        let plan = super::semantic::suspension::Plan::lower(&canonical.graph)
            .map_err(Error::Unsupported)?;
        let mut ids = Vec::new();
        for (index, _) in plan.regions.iter().enumerate() {
            ids.push(self.declare_js(&format!("{label}.resume.{index}"), line)?);
        }
        let exits: Vec<_> = std::iter::once(crate::continuation::Exit::Complete)
            .chain((1..ids.len()).map(crate::continuation::Exit::Suspend))
            .collect();
        let layout = Arc::new(crate::continuation::Plan {
            entries: plan
                .regions
                .iter()
                .zip(&ids)
                .map(|(region, id)| crate::continuation::Entry {
                    code: crate::heap::CodeId(u64::from(id.as_u32())),
                    inputs: region.inputs,
                    outputs: region.outputs,
                    exits: exits.clone().into_boxed_slice(),
                })
                .collect(),
        });
        for (region, &id) in plan.regions.into_iter().zip(&ids) {
            let region = super::semantic::canonicalize_graph(region.graph, self.ssa_opt)
                .map_err(Error::Unsupported)?;
            Arc::make_mut(&mut self.canonical).insert_region(self.source_name.0, id, region.graph);
        }
        for &id in &ids {
            if self.inspection.is_none() {
                self.lazy.insert(
                    id,
                    (
                        Emission {
                            tier: CompilationTier::Optimized,
                            specialized: false,
                            callees: Vec::new(),
                            plan: Default::default(),
                            specialization: None,
                            osr: None,
                            source_name: self.source_name,
                            params: params.span,
                        },
                        self.canonical.clone(),
                    ),
                );
                continue;
            }
            let (_, region) = self.canonical.emission(id).unwrap();
            let body = self.compile_region(
                &region.graph,
                self.source_name,
                params.span,
                &FxHashMap::default(),
                None,
            )?;
            self.continuations.push((id, body));
        }
        let mut factory = self.compile_function(
            params,
            &[],
            None,
            None,
            arguments_references,
            &[],
            captured,
            closures,
            scoping,
            finalize::FunctionKind::Factory(layout),
            (label, line),
            None,
        )?;
        factory.inspection = factory
            .inspection
            .ast(self.inspection.as_deref(), &(params, statements));
        Ok(factory)
    }

    fn compile_function(
        &mut self,
        params: &FormalParameters<'_>,
        statements: &[Statement<'_>],
        expression: Option<&Expression<'_>>,
        arguments_binding: Option<SymbolId>,
        arguments_references: &FxHashMap<u32, SymbolId>,
        captures: &[SymbolId],
        captured: &FxHashSet<SymbolId>,
        closures: &FxHashMap<u32, (FuncId, Vec<SymbolId>, bool)>,
        scoping: &Scoping,
        kind: finalize::FunctionKind,
        name: (&str, u32),
        semantic_function: Option<oxc_span::Span>,
    ) -> Result<FunctionBody, Error> {
        let mut specialization = None;
        let mut graph = semantic_function
            .map(|span| {
                let canonical = match self.canonical.function(self.source_name.0, span.start) {
                    Some(function) => Ok(function),
                    None => super::semantic::build_canonical(
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
                        false,
                        self.ssa_opt,
                    ),
                }?;
                let planned = super::semantic::specialize_with_budget(
                    canonical,
                    self.ssa_opt,
                    Some((&self.canonical, self.source_name.0)),
                    usize::from(self.inlining),
                )?;
                specialization = self.specialization_entries(
                    name.0,
                    &planned.graph,
                    &planned.plan,
                    self.source_name.0,
                );
                super::semantic::optimize_specialized(
                    planned,
                    super::semantic::Versioning::Speculate {
                        fallback: None,
                        continuations: self.speculative_continuations,
                    },
                    self.report_regions(),
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
        if self.ssa_opt
            && let Some(graph) = &mut graph
        {
            graph.lower_typed_storage().map_err(Error::Unsupported)?;
        }
        let feedback = if self.ssa_opt && matches!(kind, finalize::FunctionKind::Ordinary) {
            semantic_function.and_then(|span| self.canonical.code(self.source_name.0, span.start))
        } else {
            None
        };
        if let Some(code) = feedback {
            let graph = graph.as_ref().unwrap();
            self.log_specialization(
                code,
                name.0,
                "compile-initial",
                None,
                specialization.as_deref(),
            );
            self.emissions.insert(
                VariantId::ordinary(code.as_u32() as u64),
                Emission {
                    tier: CompilationTier::Optimized,
                    specialized: true,
                    callees: self
                        .canonical
                        .graph_callees(&graph.graph, self.source_name.0),
                    plan: graph.plan.clone(),
                    specialization,
                    osr: graph.osr.clone(),
                    source_name: self.source_name,
                    params: params.span,
                },
            );
        }
        let mut baseline_data = super::function_data::FunctionData::new();
        let baseline = if std::env::var_os("JS_JIT_BASELINE").is_some()
            && self.module.isa().name() == "aarch64"
            && !self.ssa_opt
            && matches!(kind, finalize::FunctionKind::Ordinary)
        {
            graph
                .as_ref()
                .and_then(|graph| {
                    super::aarch64::compile_baseline(
                        &graph.graph,
                        self.source_name,
                        self.dump_ir || self.inspection.is_some(),
                        &mut self.names,
                        &mut baseline_data,
                    )
                    .map_err(|reason| {
                        if let Some(stats) = &mut self.stats {
                            *stats.baseline_refusals.entry(reason.0).or_default() += 1;
                        }
                    })
                    .ok()
                })
                .map(Arc::new)
        } else {
            None
        };
        if baseline.is_some() {
            self.data.push(baseline_data);
        }
        let continuations = match &graph {
            Some(graph) => self.compile_continuations(
                &graph.graph,
                self.source_name,
                params.span,
                name,
                feedback.map(|owner| self.feedback(owner, false)),
            )?,
            None => FxHashMap::default(),
        };
        let signature = self.graph_signature(graph.as_ref().map(|graph| &graph.graph));
        let feedback = feedback.map(|owner| self.feedback(owner, true));
        let mut after = None;
        if baseline.is_none()
            && let Some(graph) = &mut graph
        {
            after = super::inspect::lower_runtime(self.inspection.as_deref(), graph)
                .map_err(Error::Unsupported)?;
        }
        let inspection = graph.as_ref().map_or_else(Default::default, |graph| {
            super::inspect::Inspection::new(
                self.inspection.as_deref(),
                graph,
                &self.canonical,
                self.source_name.0,
                usize::from(self.inlining),
            )
            .after(after)
        });
        let listing = self
            .listing()
            .then(|| graph.as_ref().map(|graph| graph.listing_lines()))
            .flatten();
        let (function, annotations, lowered, snapshot) = if baseline.is_some() {
            let mut function = cranelift_codegen::ir::Function::new();
            function.signature = signature;
            (
                function,
                Annotations {
                    semantic_locations: self.dump_ir || self.inspection.is_some(),
                    ..Annotations::default()
                },
                None,
                None,
            )
        } else {
            self.emit_body(signature, self.source_name, |out| match graph {
                Some(graph) => finalize::build_function(
                    out,
                    params.span,
                    graph.graph,
                    &graph.facts,
                    &continuations,
                    feedback,
                ),
                None => match &kind {
                    finalize::FunctionKind::Factory(resume) => {
                        finalize::build_factory(out, params.span, resume.clone())
                    }
                    _ => Ok(()),
                },
            })?
        };
        let inspection = inspection.lowered(snapshot);
        Ok(FunctionBody {
            function,
            lowered,
            baseline,
            annotations,
            inspection,
            listing,
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
    sources: &FxHashMap<usize, Arc<(String, String)>>,
) -> Vec<(u32, String)> {
    let mut result = Vec::new();
    let mut last_location = None;
    let mut source_lines = FxHashMap::default();
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
    use crate::FxHashMap;
    use crate::jit::semantic::SourceLocation;
    use oxc_span::Span;

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
            &FxHashMap::default(),
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

#[cfg(all(test, target_arch = "aarch64"))]
mod native_fallback_tests {
    use super::super::semantic::{Graph, Op, Repr, low_level::Builder, variables::Variables};
    use super::*;
    use cranelift_codegen::ir::{AbiParam, MemFlagsData, Signature, TrapCode, types};
    use cranelift_module::{Linkage, Module};

    #[test]
    fn compound_bits_match_cranelift() {
        let mut reference = Vec::new();
        for backend in [crate::Backend::Cranelift, crate::Backend::NativeOnly] {
            let mut compiler = Compiler::new(
                EngineOptions {
                    backend,
                    ..Default::default()
                },
                CompilationTier::Optimized,
                std::ptr::null(),
                0,
                super::super::recompile::Log::new(false),
            )
            .unwrap();
            let mut results = Vec::new();
            for ty in [types::I8, types::I16, types::I32, types::I64] {
                for operation in 0..8 {
                    let mut signature = compiler.module.make_signature();
                    signature.params.push(AbiParam::new(types::I64));
                    signature.returns.push(AbiParam::new(types::I64));
                    let name = format!("bits_{ty}_{operation}");
                    let id = compiler
                        .module
                        .declare_function(&name, Linkage::Local, &signature)
                        .unwrap();
                    let mut graph = Graph::new_low_level(
                        name.clone(),
                        oxc_span::Span::new(0, 0),
                        signature.clone(),
                    );
                    let entry = graph.entry;
                    let mut builder = Builder::new(&mut graph);
                    builder.switch_to_block(entry);
                    builder.append_block_params_for_function_params(entry);
                    let input = builder.block_params(entry)[0];
                    let value = if ty == types::I64 {
                        input
                    } else {
                        builder.ins().ireduce(ty, input)
                    };
                    let result = match operation {
                        0 => builder.ins().extract_bits(value, 3, 5),
                        1 => builder.ins().in_range(value, 17, 200),
                        2 => builder.ins().outside_range(value, 0xf0, 0xff),
                        3 => builder.ins().test_mask(value, u64::MAX),
                        4 => builder.ins().test_mask(value, 0x8100_0180_0000_0001),
                        5 => builder.ins().in_range(value, 0, 0),
                        6 => builder.ins().in_range(value, u64::MAX - 100, u64::MAX),
                        _ => {
                            let a = builder.ins().in_range(value, 17, 200);
                            let b = builder.ins().test_mask(value, 0x80);
                            let different = builder.ins().binary(
                                super::super::semantic::low_level::Binary::BooleanXor,
                                a,
                                b,
                            );
                            let same = builder.ins().boolean_equal(a, b);
                            let either = builder.ins().boolean_or(different, same);
                            builder.ins().boolean_and(either, a)
                        }
                    };
                    let result = if operation == 0 && ty == types::I64 {
                        result
                    } else {
                        builder.ins().uextend(types::I64, result)
                    };
                    builder.ins().return_(&[result]);
                    Variables::default().finish(&mut graph).unwrap();
                    compiler.context.clear();
                    compiler.context.func.signature = signature;
                    if backend == crate::Backend::Cranelift {
                        let mut prepared = graph.clone();
                        cranelift::prepare(&mut prepared, &mut |_, _| {}).unwrap();
                        let mut builder = cranelift_frontend::FunctionBuilder::new(
                            &mut compiler.context.func,
                            &mut compiler.builder_context,
                        );
                        cranelift::emit(&mut builder, &mut compiler.module, &prepared, None)
                            .unwrap();
                        builder.seal_all_blocks();
                        builder.finalize(compiler.module.target_config());
                    }
                    let machine = compiler
                        .machine_code(id, &name, &mut Annotations::default(), Some(&graph))
                        .unwrap();
                    machine.define(&mut compiler.module, id).unwrap();
                    compiler.module.finalize_definitions().unwrap();
                    let run: extern "C" fn(u64) -> u64 =
                        unsafe { std::mem::transmute(compiler.module.get_finalized_function(id)) };
                    for input in [
                        0,
                        1,
                        16,
                        17,
                        199,
                        200,
                        201,
                        0xef,
                        0xf0,
                        0xff,
                        0x100,
                        u32::MAX as u64,
                        i64::MAX as u64,
                        u64::MAX - 100,
                        u64::MAX,
                    ] {
                        results.push(run(input));
                    }
                }
            }
            if backend == crate::Backend::Cranelift {
                reference = results;
            } else {
                assert_eq!(results, reference);
            }
        }
    }

    #[test]
    fn unsupported_graph_operation_falls_back_to_cranelift() {
        let mut compiler = Compiler::new(
            EngineOptions {
                backend: crate::Backend::Native,
                ..Default::default()
            },
            CompilationTier::Optimized,
            std::ptr::null(),
            0,
            super::super::recompile::Log::new(false),
        )
        .unwrap();
        let mut signature = Signature::new(compiler.module.target_config().default_call_conv);
        signature.params.push(AbiParam::new(types::I64));
        signature.returns.push(AbiParam::new(types::I64));
        let id = compiler
            .module
            .declare_function("fallback_test", Linkage::Local, &signature)
            .unwrap();
        let mut graph =
            Graph::new_low_level("test".into(), oxc_span::Span::new(0, 0), signature.clone());
        let entry = graph.entry;
        let mut builder = Builder::new(&mut graph);
        builder.switch_to_block(entry);
        builder.append_block_params_for_function_params(entry);
        let pointer = builder.block_params(entry)[0];
        let flags = MemFlagsData::new().with_trap_code(Some(TrapCode::HEAP_OUT_OF_BOUNDS));
        let value = builder.ins().load(types::I32, flags, pointer, 0);
        let wide = builder.semantic(Op::Convert(value, Repr::I64), Repr::I64);
        let extracted = builder.ins().extract_bits(wide, 4, 8);
        builder.ins().return_(&[extracted]);
        Variables::default().finish(&mut graph).unwrap();
        assert!(super::super::aarch64::compile_graph(&graph).is_err());
        compiler.context.func.signature = signature;
        let machine = compiler
            .machine_code(
                id,
                "fallback_test",
                &mut Annotations::default(),
                Some(&graph),
            )
            .unwrap();
        assert!(compiler.context.compiled_code().is_some());
        machine.define(&mut compiler.module, id).unwrap();
        compiler.module.finalize_definitions().unwrap();
        let load: extern "C" fn(*const u32) -> u64 =
            unsafe { std::mem::transmute(compiler.module.get_finalized_function(id)) };
        let value = 0x1234;
        assert_eq!(load(&value), 0x23);
        assert!(matches!(graph.defining_op(wide), Some(Op::Convert(..))));
    }
}

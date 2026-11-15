use std::fmt::{Debug, Write};

use cranelift_codegen::{Context, ir::Function};
use cranelift_jit::JITModule;
use cranelift_module::Module;

use crate::error::Error;

use super::{clif::Annotations, machine::MachineCode, semantic};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompileOptions {
    pub filter: Option<String>,
    pub clif: bool,
    pub ast: bool,
    pub semantic_ir: bool,
    pub after: Option<String>,
    pub planner: bool,
    pub asm: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            filter: None,
            clif: false,
            ast: false,
            semantic_ir: true,
            after: None,
            planner: false,
            asm: false,
        }
    }
}

#[derive(Clone, Default)]
pub(super) struct Inspection {
    label: Option<String>,
    ast: Option<String>,
    semantic: Option<String>,
    after: Option<(String, String)>,
    planner: Option<String>,
}

pub(super) fn graph_text(graph: &semantic::Graph) -> String {
    semantic_text(graph.listing(&semantic::Facts::compute(graph)).lines())
}

fn semantic_text(lines: Vec<semantic::ListingLine>) -> String {
    let mut text = String::new();
    for (id, _, line) in lines {
        if id == 0 {
            writeln!(text, "{line}").unwrap();
        } else {
            writeln!(text, "{line}  @{id:04x}").unwrap();
        }
    }
    text
}

pub(super) struct PassInspection {
    pass: Option<String>,
    pub(super) snapshot: Option<(String, String)>,
}

impl PassInspection {
    pub(super) fn new(options: Option<&CompileOptions>) -> Self {
        Self {
            pass: options
                .filter(|options| options.semantic_ir)
                .and_then(|options| options.after.clone()),
            snapshot: None,
        }
    }

    pub(super) fn observe(&mut self, name: &str, state: &semantic::FunctionState) {
        if self.pass.as_deref() == Some(name) {
            self.snapshot = Some((name.to_owned(), semantic_text(state.listing_lines())));
        }
    }

    pub(super) fn lower_runtime(
        &mut self,
        state: &mut semantic::FunctionState,
    ) -> Result<(), String> {
        state.lower_runtime_observed(&mut |name, state| self.observe(name, state))?;
        let Some(pass) = self.pass.as_deref() else {
            return Ok(());
        };
        let passes = semantic::SPECIALIZATION_PASSES
            .iter()
            .chain(semantic::RUNTIME_PASSES)
            .chain(semantic::LOW_LEVEL_PASSES)
            .copied()
            .collect::<Vec<_>>();
        if !passes.contains(&pass) {
            return Err(format!(
                "unknown pass {pass}; available passes: {}",
                passes.join(", ")
            ));
        }
        Ok(())
    }
}

impl Inspection {
    pub(super) fn new(
        options: Option<&CompileOptions>,
        graph: &semantic::FunctionState,
        repository: &semantic::specialization::calls::Repository,
        source: usize,
        inline_multiplier: usize,
    ) -> Self {
        let Some(options) = options else {
            return Self::default();
        };
        Self {
            semantic: options
                .semantic_ir
                .then(|| semantic_text(graph.listing_lines())),
            planner: options
                .planner
                .then(|| graph.plan_report(repository, source, inline_multiplier)),
            ..Self::default()
        }
    }

    pub(super) fn from_graph(options: Option<&CompileOptions>, graph: &semantic::Graph) -> Self {
        let Some(options) = options else {
            return Self::default();
        };
        Self {
            semantic: options.semantic_ir.then(|| graph_text(graph)),
            ..Self::default()
        }
    }

    pub(super) fn named(mut self, label: &str) -> Self {
        self.label = Some(label.to_owned());
        self
    }

    pub(super) fn after(mut self, snapshot: Option<(String, String)>) -> Self {
        self.after = snapshot;
        self
    }

    pub(super) fn lowered(mut self, snapshot: Option<(String, String)>) -> Self {
        if snapshot.is_some() {
            self.after = snapshot;
        }
        self
    }

    pub(super) fn ast(mut self, options: Option<&CompileOptions>, ast: &impl Debug) -> Self {
        if options.is_some_and(|options| options.ast) {
            self.ast = Some(format!("{ast:#?}"));
        }
        self
    }
}

impl CompileOptions {
    pub(super) fn matches(&self, name: &str) -> bool {
        self.filter
            .as_ref()
            .is_none_or(|filter| name.contains(filter))
    }

    pub(super) fn print_body(
        &self,
        name: &str,
        body: &Inspection,
        function: &Function,
        annotations: &Annotations,
        direct: bool,
    ) {
        let label = body.label.as_deref().unwrap_or(name);
        if let Some(ast) = &body.ast {
            println!("[{name}] analyzed AST {label}\n{ast}");
        }
        match (&body.after, &body.semantic) {
            (Some((pass, semantic)), _) => {
                println!("[{name}] semantic IR {label} after {pass}\n{semantic}")
            }
            (None, Some(semantic)) => println!("[{name}] semantic IR {label}\n{semantic}"),
            (None, None) => {}
        }
        if let Some(planner) = &body.planner {
            println!("[{name}] planner {label}\n{planner}");
        }
        if self.clif {
            if direct {
                println!("[{name}] Cranelift IR unavailable for direct baseline code");
                return;
            }
            println!(
                "[{name}] Cranelift IR before optimization\n{}",
                function.display()
            );
            let mut helpers: Vec<_> = annotations.helpers.iter().collect();
            helpers.sort_by_key(|(inst, _)| inst.as_u32());
            for (inst, helper) in helpers {
                println!("  helper {inst}: {helper:?}");
            }
        }
    }

    pub(super) fn print_compiled(
        &self,
        name: &str,
        context: &Context,
        machine: &MachineCode,
        module: &JITModule,
        direct: bool,
    ) -> Result<(), Error> {
        if self.clif && !direct {
            println!(
                "[{name}] Cranelift IR after optimization\n{}",
                context.func.display()
            );
        }
        if !self.asm {
            return Ok(());
        }
        let cs = module
            .isa()
            .to_capstone()
            .map_err(|error| Error::Compile(error.to_string()))?;
        let code = &machine.bytes;
        let mut starts = vec![0];
        starts.extend(
            machine
                .block_starts
                .iter()
                .copied()
                .filter(|&start| start != 0),
        );
        starts.push(code.len() as u32);
        let mut out = String::new();
        writeln!(out, "[{name}] machine code ({} bytes)", code.len()).unwrap();
        for (block, bounds) in starts.windows(2).enumerate() {
            let start = bounds[0] as usize;
            let end = bounds[1] as usize;
            writeln!(out, "block{block}:").unwrap();
            let instructions = cs
                .disasm_all(&code[start..end], start as u64)
                .map_err(|error| Error::Compile(error.to_string()))?;
            let mut decoded = start;
            for inst in instructions.iter() {
                let offset = inst.address() as u32;
                let end = offset + inst.bytes().len() as u32;
                write!(
                    out,
                    "  +{offset:06x}  {:8} {}",
                    inst.mnemonic().unwrap_or(""),
                    inst.op_str().unwrap_or("")
                )
                .unwrap();
                match machine.location_at(offset) {
                    Some(location) => write!(out, " ; @{location:04x}").unwrap(),
                    None => out.push_str(" ; unattributed"),
                }
                for reloc in machine
                    .relocs
                    .iter()
                    .filter(|reloc| offset <= reloc.offset && reloc.offset < end)
                {
                    write!(
                        out,
                        " ; reloc {} {} {:+}",
                        reloc.kind, reloc.name, reloc.addend
                    )
                    .unwrap();
                }
                for (_, code) in machine
                    .traps
                    .iter()
                    .filter(|(trap, _)| offset <= *trap && *trap < end)
                {
                    write!(out, " ; trap {code}").unwrap();
                }
                out.push('\n');
                decoded = end as usize;
            }
            for (offset, byte) in code.iter().enumerate().take(end).skip(decoded) {
                writeln!(out, "  +{offset:06x}  .byte 0x{byte:02x}").unwrap();
            }
        }
        print!("{out}");
        Ok(())
    }
}

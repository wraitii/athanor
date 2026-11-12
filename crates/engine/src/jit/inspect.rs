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
    planner: Option<String>,
}

impl Inspection {
    pub(super) fn new(
        options: Option<&CompileOptions>,
        graph: &semantic::FunctionState,
        repository: &semantic::specialize::Repository,
        source: usize,
        inline_multiplier: usize,
    ) -> Self {
        let Some(options) = options else {
            return Self::default();
        };
        Self {
            semantic: options.semantic_ir.then(|| {
                let mut text = String::new();
                for (id, _, line) in graph.listing_lines() {
                    if id == 0 {
                        writeln!(text, "{line}").unwrap();
                    } else {
                        writeln!(text, "{line}  @{id:04x}").unwrap();
                    }
                }
                text
            }),
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
            semantic: options.semantic_ir.then(|| {
                let mut text = String::new();
                for (id, _, line) in graph.listing(&semantic::Facts::compute(graph)).lines() {
                    if id == 0 {
                        writeln!(text, "{line}").unwrap();
                    } else {
                        writeln!(text, "{line}  @{id:04x}").unwrap();
                    }
                }
                text
            }),
            ..Self::default()
        }
    }

    pub(super) fn named(mut self, label: &str) -> Self {
        self.label = Some(label.to_owned());
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
        if let Some(semantic) = &body.semantic {
            println!("[{name}] semantic IR {label}\n{semantic}");
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

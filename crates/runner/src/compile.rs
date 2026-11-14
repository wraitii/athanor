use std::sync::Arc;

use js_engine::{CompileOptions, Engine, EngineOptions};

#[derive(Default)]
pub(super) struct Options {
    pub enabled: bool,
    output: bool,
    execute: Option<String>,
    dump: CompileOptions,
}

impl Options {
    pub fn parse(
        &mut self,
        argument: &str,
        arguments: &mut impl Iterator<Item = String>,
    ) -> Result<bool, String> {
        match argument {
            "--compile" => {
                self.enabled = true;
                return Ok(true);
            }
            "--execute" => {
                self.execute = Some(
                    arguments
                        .next()
                        .ok_or("--execute expects JavaScript source")?,
                );
            }
            "--clif" => self.dump.clif = true,
            "--no-ir" | "--no-semantic-ir" => self.dump.semantic_ir = false,
            "--ast" => self.dump.ast = true,
            "--semantic-ir" => self.dump.semantic_ir = true,
            "--planner" => self.dump.planner = true,
            "--after" => {
                self.dump.semantic_ir = true;
                self.dump.after = Some(arguments.next().ok_or("--after expects a pass name")?)
            }
            "--asm" => self.dump.asm = true,
            "--filter" => {
                self.dump.filter = Some(arguments.next().ok_or("--filter expects a function name")?)
            }
            _ => return Ok(false),
        }
        self.output = true;
        Ok(true)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.output && !self.enabled {
            return Err("--clif, --no-ir, --no-semantic-ir, --ast, --semantic-ir, --after, --planner, --asm, --filter and --execute require --compile".into());
        }
        Ok(())
    }

    pub fn run(self, source: &str, filename: &str) -> Result<(), js_engine::Error> {
        let mut engine = Engine::with_options(EngineOptions {
            compile: Some(Arc::new(self.dump)),
            background_compile: false,
            ..Default::default()
        })?;
        match self.execute {
            Some(execute) => {
                engine.install_host(super::host::FileHost)?;
                engine.compile(
                    &format!("{source}\n;(() => {{\n{execute}\n}})();"),
                    filename,
                )
            }
            None => engine.compile_only(source, filename),
        }
    }
}

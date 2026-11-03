use cranelift_codegen::ir::types;
use oxc_ast::ast::SwitchStatement;

use super::FunctionCompiler;
use super::scope::collect_lexical_symbols;
use crate::error::Error;
use crate::jit::helpers::Helper;
use crate::value::Value;

impl FunctionCompiler<'_, '_> {
    pub(super) fn emit_switch(&mut self, statement: &SwitchStatement<'_>) -> Result<(), Error> {
        let discriminant = self.emit_expression(&statement.discriminant)?;
        let saved = self.b.declare_var(types::I64);
        self.store_local(saved, discriminant);
        self.clear_temporaries();

        self.scopes.push();
        let mut symbols = Vec::new();
        for case in &statement.cases {
            collect_lexical_symbols(&case.consequent, &mut symbols);
        }
        for symbol in symbols {
            self.declare_lexical_binding(symbol);
        }
        for case in &statement.cases {
            self.declare_block_functions(&case.consequent);
        }
        for case in &statement.cases {
            self.initialize_function_declarations(&case.consequent)?;
        }

        let done = self.b.create_block();
        let bodies: Vec<_> = statement
            .cases
            .iter()
            .map(|_| self.b.create_block())
            .collect();
        let mut fallback = done;
        for (case, body) in statement.cases.iter().zip(&bodies) {
            let Some(test) = &case.test else {
                fallback = *body;
                continue;
            };
            let candidate = self.emit_expression(test)?;
            let discriminant = self.b.use_var(saved);
            let equal = self.call_helper(Helper::StrictEqual, &[discriminant, candidate]);
            let condition = self.truthy(equal);
            self.clear_temporaries();
            let next = self.b.create_block();
            self.brif(condition, *body, &[], next, &[]);
            self.switch(next);
        }
        self.jump(fallback, &[]);

        self.control_targets.push((done, None));
        for (index, case) in statement.cases.iter().enumerate() {
            self.switch(bodies[index]);
            self.emit_statements(&case.consequent)?;
            if !self.terminated {
                self.jump(bodies.get(index + 1).copied().unwrap_or(done), &[]);
            }
        }
        self.control_targets.pop();
        self.switch(done);
        self.leave_block();
        let undefined = self.constant(Value::UNDEFINED);
        self.store_local(saved, undefined);
        Ok(())
    }
}

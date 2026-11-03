use cranelift_codegen::ir::{Block, BlockArg, Value as ClifValue, types};
use oxc_ast::ast::{
    CallExpression, ChainElement, ChainExpression, ComputedMemberExpression, Expression,
    StaticMemberExpression,
};

use super::FunctionCompiler;
use crate::error::{Error, unsupported_at};
use crate::value::Value;

impl FunctionCompiler<'_, '_> {
    pub(super) fn emit_optional_chain(
        &mut self,
        chain: &ChainExpression<'_>,
    ) -> Result<(ClifValue, ClifValue), Error> {
        let short = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param(done, types::I64);
        self.b.append_block_param(done, types::I64);
        let (value, receiver) = match &chain.expression {
            ChainElement::StaticMemberExpression(member) => {
                self.emit_chain_static(member, short)?
            }
            ChainElement::ComputedMemberExpression(member) => {
                self.emit_chain_computed(member, short)?
            }
            ChainElement::CallExpression(call) => self.emit_chain_call(call, short)?,
            _ => return Err(unsupported_at(chain.span, "optional chain element")),
        };
        self.jump(done, &[BlockArg::Value(value), BlockArg::Value(receiver)]);
        self.switch(short);
        let undefined = self.constant(Value::UNDEFINED);
        self.jump(
            done,
            &[BlockArg::Value(undefined), BlockArg::Value(undefined)],
        );
        self.switch(done);
        Ok((self.b.block_params(done)[0], self.b.block_params(done)[1]))
    }

    fn chain_guard(&mut self, value: ClifValue, short: Block) {
        let nullish = self.is_nullish(value);
        let present = self.b.create_block();
        self.brif(nullish, short, &[], present, &[]);
        self.switch(present);
    }

    fn emit_chain_reference(
        &mut self,
        expression: &Expression<'_>,
        short: Block,
    ) -> Result<(ClifValue, ClifValue), Error> {
        match expression {
            Expression::StaticMemberExpression(member) => self.emit_chain_static(member, short),
            Expression::ComputedMemberExpression(member) => self.emit_chain_computed(member, short),
            Expression::CallExpression(call) => self.emit_chain_call(call, short),
            _ => self.emit_call_reference(expression),
        }
    }

    fn emit_chain_static(
        &mut self,
        member: &StaticMemberExpression<'_>,
        short: Block,
    ) -> Result<(ClifValue, ClifValue), Error> {
        let (receiver, _) = self.emit_chain_reference(&member.object, short)?;
        if member.optional {
            self.chain_guard(receiver, short);
        }
        let value = self.property_access(receiver, member.property.name.as_str(), None);
        Ok((value, receiver))
    }

    fn emit_chain_computed(
        &mut self,
        member: &ComputedMemberExpression<'_>,
        short: Block,
    ) -> Result<(ClifValue, ClifValue), Error> {
        let (receiver, _) = self.emit_chain_reference(&member.object, short)?;
        if member.optional {
            self.chain_guard(receiver, short);
        }
        let key = self.emit_expression(&member.expression)?;
        let value = self.index_access(receiver, key, None);
        Ok((value, receiver))
    }

    fn emit_chain_call(
        &mut self,
        call: &CallExpression<'_>,
        short: Block,
    ) -> Result<(ClifValue, ClifValue), Error> {
        let (callable, receiver) = self.emit_chain_reference(&call.callee, short)?;
        if call.optional {
            self.chain_guard(callable, short);
        }
        let value = self.emit_call_arguments(call, callable, receiver)?;
        Ok((value, self.constant(Value::UNDEFINED)))
    }
}

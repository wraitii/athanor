use super::{FunctionCompiler, LoweredValue};
use crate::error::{Error, compile_at};
use cranelift_codegen::ir::{MemFlagsData, StackSlotData, StackSlotKind};
use oxc_ast::ast::Expression;
use oxc_span::Span;

impl FunctionCompiler<'_, '_> {
    pub(super) fn stack_values(
        &mut self,
        values: &[LoweredValue],
        span: Span,
    ) -> Result<LoweredValue, Error> {
        if values.is_empty() {
            return Ok(self.b.ins().iconst(self.pointer, 0));
        }
        let size = values
            .len()
            .checked_mul(8)
            .filter(|size| *size <= i32::MAX as usize)
            .ok_or_else(|| compile_at(span, "too many stack values"))?;
        let slot = self.b.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            size as u32,
            3,
        ));
        for (index, value) in values.iter().enumerate() {
            let address = self
                .b
                .ins()
                .stack_addr(self.pointer, slot, (index * 8) as i32);
            self.b
                .ins()
                .store(MemFlagsData::trusted(), *value, address, 0);
        }
        Ok(self.b.ins().stack_addr(self.pointer, slot, 0))
    }
}

pub(in crate::jit) fn has_named_capture(pattern: &str) -> bool {
    let bytes = pattern.as_bytes();
    let mut index = 0;
    let mut class = false;
    while index < bytes.len() {
        match bytes[index] {
            b'\\' => index += 2,
            b'[' if !class => {
                class = true;
                index += 1;
            }
            b']' if class => {
                class = false;
                index += 1;
            }
            b'(' if !class
                && bytes.get(index + 1) == Some(&b'?')
                && bytes.get(index + 2) == Some(&b'<') =>
            {
                if !matches!(bytes.get(index + 3), Some(b'=') | Some(b'!')) {
                    return true;
                }
                index += 1;
            }
            _ => index += 1,
        }
    }
    false
}

pub(in crate::jit) fn expression_kind(expression: &Expression<'_>) -> &'static str {
    match expression {
        Expression::BooleanLiteral(_) => "boolean literal",
        Expression::NullLiteral(_) => "null literal",
        Expression::NumericLiteral(_) => "numeric literal",
        Expression::BigIntLiteral(_) => "bigint literal",
        Expression::RegExpLiteral(_) => "regexp literal",
        Expression::StringLiteral(_) => "string literal",
        Expression::TemplateLiteral(_) => "template literal",
        Expression::Identifier(_) => "identifier",
        Expression::Super(_) => "super",
        Expression::ArrayExpression(_) => "array",
        Expression::ArrowFunctionExpression(_) => "arrow function",
        Expression::AssignmentExpression(_) => "assignment",
        Expression::AwaitExpression(_) => "await",
        Expression::BinaryExpression(_) => "binary",
        Expression::CallExpression(_) => "call",
        Expression::ChainExpression(_) => "chain",
        Expression::ClassExpression(_) => "class",
        Expression::ConditionalExpression(_) => "conditional",
        Expression::FunctionExpression(_) => "function",
        Expression::ImportExpression(_) => "import",
        Expression::LogicalExpression(_) => "logical",
        Expression::NewExpression(_) => "new",
        Expression::ObjectExpression(_) => "object",
        Expression::ParenthesizedExpression(_) => "parenthesized",
        Expression::SequenceExpression(_) => "sequence",
        Expression::TaggedTemplateExpression(_) => "tagged template",
        Expression::ThisExpression(_) => "this",
        Expression::UnaryExpression(_) => "unary",
        Expression::UpdateExpression(_) => "update",
        Expression::YieldExpression(_) => "yield",
        Expression::PrivateInExpression(_) => "private-in",
        Expression::ComputedMemberExpression(_) => "computed member",
        Expression::StaticMemberExpression(_) => "static member",
        Expression::PrivateFieldExpression(_) => "private field",
        _ => "expression",
    }
}

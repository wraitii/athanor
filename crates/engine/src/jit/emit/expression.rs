use cranelift_codegen::ir::{
    BlockArg, InstBuilder, MemFlags, SourceLoc, StackSlotData, StackSlotKind, Value as ClifValue,
    types,
};
use oxc_ast::ast::{
    Argument, ArrayExpressionElement, AssignmentTarget, CallExpression, Expression,
    ObjectPropertyKind, PropertyKind, SimpleAssignmentTarget,
};
use oxc_span::{GetSpan, Span};
use oxc_syntax::operator::{AssignmentOperator, LogicalOperator, UnaryOperator, UpdateOperator};

use crate::error::{Error, compile_at, unsupported_at};
use crate::jit::helpers::Helper;
use crate::value::Value;

use super::FunctionCompiler;
use super::ops::{BinaryOp, UnaryOp, binary_operator, compound_operator, to_number};

impl FunctionCompiler<'_, '_> {
    pub(super) fn emit_expression(
        &mut self,
        expression: &Expression<'_>,
    ) -> Result<ClifValue, Error> {
        let previous = std::mem::replace(&mut self.source_span, expression.span());
        self.b.set_srcloc(SourceLoc::new(self.source_span.start));
        let result = self.emit_expression_inner(expression);
        self.b.set_srcloc(SourceLoc::new(previous.start));
        self.source_span = previous;
        result
    }

    fn emit_expression_inner(&mut self, expression: &Expression<'_>) -> Result<ClifValue, Error> {
        match expression {
            Expression::StringLiteral(literal) => {
                let text: Box<str> = literal
                    .value
                    .as_str()
                    .ok_or_else(|| unsupported_at(literal.span, "lone surrogate in string"))?
                    .into();
                let ptr = self.b.ins().iconst(self.pointer, text.as_ptr() as i64);
                let len = self.b.ins().iconst(types::I64, text.len() as i64);
                self.names.push(text);
                let result = self.call_helper(Helper::StringNew, &[ptr, len]);
                self.check_error();
                Ok(result)
            }
            Expression::RegExpLiteral(literal) => {
                use oxc_ast::ast::RegExpFlags;
                let flags = literal.regex.flags;
                if flags.intersects(RegExpFlags::Y | RegExpFlags::D | RegExpFlags::V) {
                    return Err(unsupported_at(
                        literal.span,
                        "regular expression flags y, d, and v",
                    ));
                }
                let pattern = literal.regex.pattern.text.as_str();
                if has_named_capture(pattern) {
                    return Err(unsupported_at(
                        literal.span,
                        "named regular expression capture",
                    ));
                }
                let bits = u8::from(flags.contains(RegExpFlags::I))
                    | (u8::from(flags.contains(RegExpFlags::M)) << 1)
                    | (u8::from(flags.contains(RegExpFlags::S)) << 2)
                    | (u8::from(flags.contains(RegExpFlags::U)) << 3)
                    | (u8::from(flags.contains(RegExpFlags::G)) << 4);
                regress::Regex::with_flags(pattern, crate::heap::regexp_flags(bits)).map_err(
                    |error| {
                        compile_at(literal.span, format!("invalid regular expression: {error}"))
                    },
                )?;
                let pattern: Box<str> = pattern.into();
                let ptr = self.b.ins().iconst(self.pointer, pattern.as_ptr() as i64);
                let len = self.b.ins().iconst(types::I64, pattern.len() as i64);
                let flags = self.b.ins().iconst(types::I64, bits as i64);
                self.names.push(pattern);
                let result = self.call_helper(Helper::RegExpNew, &[flags, ptr, len]);
                self.check_error();
                Ok(result)
            }
            Expression::ChainExpression(chain) => Ok(self.emit_optional_chain(chain)?.0),
            Expression::ThisExpression(_) => Ok(self.receiver),
            Expression::NumericLiteral(literal) => Ok(self.constant(Value::number(literal.value))),
            Expression::BooleanLiteral(literal) => Ok(self.constant(Value::boolean(literal.value))),
            Expression::NullLiteral(_) => Ok(self.constant(Value::NULL)),
            Expression::ObjectExpression(expression) => {
                let dummy = self.constant(Value::UNDEFINED);
                let object = self.call_helper(Helper::ObjectNew, &[dummy]);
                self.check_error();
                for property in &expression.properties {
                    let ObjectPropertyKind::ObjectProperty(property) = property else {
                        return Err(unsupported_at(property.span(), "object spread"));
                    };
                    if property.kind != PropertyKind::Init || property.method {
                        return Err(unsupported_at(property.span, "object accessor or method"));
                    }
                    if property.computed {
                        let key = property
                            .key
                            .as_expression()
                            .ok_or_else(|| unsupported_at(property.span, "property key"))?;
                        let key = self.emit_expression(key)?;
                        let key = self.call_helper(Helper::PropertyKey, &[key]);
                        self.check_error();
                        let value = self.emit_expression(&property.value)?;
                        self.index_access(object, key, Some(value));
                    } else {
                        let name = property
                            .key
                            .static_name()
                            .ok_or_else(|| unsupported_at(property.span, "property key"))?;
                        let name = name.as_str().ok_or_else(|| {
                            unsupported_at(property.span, "lone surrogate in property key")
                        })?;
                        if name == "__proto__" && !property.shorthand {
                            return Err(unsupported_at(property.span, "object prototype setter"));
                        }
                        let value = self.emit_expression(&property.value)?;
                        self.property_access(object, name, Some(value));
                    }
                }
                Ok(object)
            }
            Expression::ArrayExpression(array) => {
                if array
                    .elements
                    .iter()
                    .any(|element| matches!(element, ArrayExpressionElement::SpreadElement(_)))
                {
                    if array
                        .elements
                        .iter()
                        .any(|element| matches!(element, ArrayExpressionElement::Elision(_)))
                    {
                        return Err(unsupported_at(array.span, "array literal holes"));
                    }
                    let ptr = self.b.ins().iconst(self.pointer, 0);
                    let zero = self.b.ins().iconst(types::I64, 0);
                    let result = self.call_helper(Helper::ArrayNew, &[ptr, zero]);
                    self.check_error();
                    for element in &array.elements {
                        let (expression, helper) = match element {
                            ArrayExpressionElement::SpreadElement(spread) => {
                                (&spread.argument, Helper::ArrayExtend)
                            }
                            _ => (element.as_expression().unwrap(), Helper::ArrayAppend),
                        };
                        let value = self.emit_expression(expression)?;
                        self.call_helper(helper, &[result, value]);
                        self.check_error();
                    }
                    return Ok(result);
                }
                let mut elements = Vec::with_capacity(array.elements.len());
                for element in &array.elements {
                    let expression = element
                        .as_expression()
                        .ok_or_else(|| unsupported_at(element.span(), "array literal holes"))?;
                    elements.push(self.emit_expression(expression)?);
                }
                let argv = self.stack_values(&elements, array.span)?;
                let argc = self.b.ins().iconst(types::I64, elements.len() as i64);
                let array = self.call_helper(Helper::ArrayNew, &[argv, argc]);
                self.check_error();
                Ok(array)
            }
            Expression::Identifier(identifier) => {
                let symbol = identifier
                    .reference_id
                    .get()
                    .and_then(|id| self.scoping.get_reference(id).symbol_id());
                let symbol = self
                    .arguments_references
                    .get(&identifier.span.start)
                    .copied()
                    .or(symbol);
                if let Some(variable) = symbol.and_then(|id| self.local(id)) {
                    return Ok(self.read_local(variable));
                }
                if identifier.name == "undefined" && symbol.is_none() {
                    return Ok(self.constant(Value::UNDEFINED));
                }
                Ok(self.global_access(identifier.name.as_str(), None))
            }
            Expression::ParenthesizedExpression(expression) => {
                self.emit_expression(&expression.expression)
            }
            Expression::SequenceExpression(expression) => {
                let mut value = self.constant(Value::UNDEFINED);
                for expression in &expression.expressions {
                    value = self.emit_expression(expression)?;
                }
                Ok(value)
            }
            Expression::UnaryExpression(expression) => {
                let mut argument = &expression.argument;
                while let Expression::ParenthesizedExpression(parenthesized) = argument {
                    argument = &parenthesized.expression;
                }
                let value = if expression.operator == UnaryOperator::Typeof {
                    if let Expression::Identifier(identifier) = argument {
                        let symbol = identifier
                            .reference_id
                            .get()
                            .and_then(|id| self.scoping.get_reference(id).symbol_id());
                        if symbol.is_none()
                            && !self
                                .arguments_references
                                .contains_key(&identifier.span.start)
                        {
                            self.global_get_optional(identifier.name.as_str())
                        } else {
                            self.emit_expression(argument)?
                        }
                    } else {
                        self.emit_expression(argument)?
                    }
                } else {
                    self.emit_expression(argument)?
                };
                let op = match expression.operator {
                    UnaryOperator::Typeof => UnaryOp::Typeof,
                    UnaryOperator::UnaryPlus => UnaryOp::Plus,
                    UnaryOperator::UnaryNegation => UnaryOp::Negation,
                    UnaryOperator::LogicalNot => UnaryOp::Not,
                    UnaryOperator::BitwiseNot => UnaryOp::BitwiseNot,
                    UnaryOperator::Void => return Ok(self.constant(Value::UNDEFINED)),
                    _ => {
                        return Err(unsupported_at(
                            expression.span,
                            "unary operator in unary expression",
                        ));
                    }
                };
                Ok(self.unary_helper(op, value))
            }
            Expression::UpdateExpression(expression) => {
                if let SimpleAssignmentTarget::StaticMemberExpression(member) = &expression.argument
                {
                    let object = self.emit_expression(&member.object)?;
                    let name = member.property.name.as_str();
                    let previous = self.property_access(object, name, None);
                    let number = to_number(self, previous);
                    let one = self.constant(Value::number(1.0));
                    let op = match expression.operator {
                        UpdateOperator::Increment => BinaryOp::Add,
                        UpdateOperator::Decrement => BinaryOp::Sub,
                    };
                    let value = self.binary(op, number, one);
                    self.property_access(object, name, Some(value));
                    return Ok(if expression.prefix { value } else { number });
                }
                if let SimpleAssignmentTarget::ComputedMemberExpression(member) =
                    &expression.argument
                {
                    let object = self.emit_expression(&member.object)?;
                    let index = self.emit_expression(&member.expression)?;
                    let previous = self.index_access(object, index, None);
                    let number = to_number(self, previous);
                    let one = self.constant(Value::number(1.0));
                    let op = match expression.operator {
                        UpdateOperator::Increment => BinaryOp::Add,
                        UpdateOperator::Decrement => BinaryOp::Sub,
                    };
                    let value = self.binary(op, number, one);
                    self.index_access(object, index, Some(value));
                    return Ok(if expression.prefix { value } else { number });
                }
                let (name, symbol) = match &expression.argument {
                    SimpleAssignmentTarget::AssignmentTargetIdentifier(identifier) => (
                        identifier.name.as_str(),
                        identifier
                            .reference_id
                            .get()
                            .and_then(|id| self.scoping.get_reference(id).symbol_id()),
                    ),
                    other => {
                        return Err(unsupported_at(other.span(), "this update target"));
                    }
                };
                if !self
                    .arguments_references
                    .contains_key(&expression.argument.span().start)
                    && symbol.is_some_and(|id| self.scoping.symbol_flags(id).is_const_variable())
                {
                    return Err(unsupported_at(expression.span, "assignment to const"));
                }
                let symbol = self
                    .arguments_references
                    .get(&expression.argument.span().start)
                    .copied()
                    .or(symbol);
                let variable = symbol.and_then(|id| self.local(id));
                let previous = match variable {
                    Some(variable) => self.read_local(variable),
                    None => self.global_access(name, None),
                };
                let number = to_number(self, previous);
                let one = self.constant(Value::number(1.0));
                let value = match expression.operator {
                    UpdateOperator::Increment => self.binary(BinaryOp::Add, number, one),
                    UpdateOperator::Decrement => self.binary(BinaryOp::Sub, number, one),
                };
                if let Some(variable) = variable {
                    self.store_local(variable, value);
                } else {
                    self.global_access(name, Some(value));
                }
                Ok(if expression.prefix { value } else { number })
            }
            Expression::BinaryExpression(expression) => {
                let op = binary_operator(expression.operator);
                let left = self.emit_expression(&expression.left)?;
                let right = self.emit_expression(&expression.right)?;
                Ok(self.binary(op, left, right))
            }
            Expression::LogicalExpression(expression) => {
                let left = self.emit_expression(&expression.left)?;
                let right_block = self.b.create_block();
                let done = self.b.create_block();
                self.b.append_block_param(done, types::I64);
                match expression.operator {
                    LogicalOperator::And => {
                        let condition = self.truthy(left);
                        self.brif(condition, right_block, &[], done, &[BlockArg::Value(left)]);
                    }
                    LogicalOperator::Or => {
                        let condition = self.truthy(left);
                        self.brif(condition, done, &[BlockArg::Value(left)], right_block, &[]);
                    }
                    LogicalOperator::Coalesce => {
                        let nullish = self.is_nullish(left);
                        self.brif(nullish, right_block, &[], done, &[BlockArg::Value(left)]);
                    }
                }
                self.switch(right_block);
                let right = self.emit_expression(&expression.right)?;
                self.jump(done, &[BlockArg::Value(right)]);
                self.switch(done);
                Ok(self.b.block_params(done)[0])
            }
            Expression::ConditionalExpression(expression) => {
                let test = self.emit_expression(&expression.test)?;
                let condition = self.truthy(test);
                let consequent = self.b.create_block();
                let alternate = self.b.create_block();
                let done = self.b.create_block();
                self.b.append_block_param(done, types::I64);
                self.brif(condition, consequent, &[], alternate, &[]);

                self.switch(consequent);
                let value = self.emit_expression(&expression.consequent)?;
                self.jump(done, &[BlockArg::Value(value)]);

                self.switch(alternate);
                let value = self.emit_expression(&expression.alternate)?;
                self.jump(done, &[BlockArg::Value(value)]);

                self.switch(done);
                Ok(self.b.block_params(done)[0])
            }
            Expression::StaticMemberExpression(member) => {
                let object = self.emit_expression(&member.object)?;
                Ok(self.property_access(object, member.property.name.as_str(), None))
            }
            Expression::ComputedMemberExpression(member) => {
                let object = self.emit_expression(&member.object)?;
                let index = self.emit_expression(&member.expression)?;
                Ok(self.index_access(object, index, None))
            }
            Expression::AssignmentExpression(expression) => {
                if let AssignmentTarget::ComputedMemberExpression(member) = &expression.left {
                    let object = self.emit_expression(&member.object)?;
                    let index = self.emit_expression(&member.expression)?;
                    let compound = match expression.operator {
                        AssignmentOperator::Assign => None,
                        operator => Some(compound_operator(operator).ok_or_else(|| {
                            unsupported_at(expression.span, "assignment operator")
                        })?),
                    };
                    let left = compound.map(|_| self.index_access(object, index, None));
                    let right = self.emit_expression(&expression.right)?;
                    let value = match compound {
                        Some(op) => self.binary(op, left.unwrap(), right),
                        None => right,
                    };
                    return Ok(self.index_access(object, index, Some(value)));
                }
                if let AssignmentTarget::StaticMemberExpression(member) = &expression.left {
                    let object = self.emit_expression(&member.object)?;
                    let compound = match expression.operator {
                        AssignmentOperator::Assign => None,
                        operator => Some(compound_operator(operator).ok_or_else(|| {
                            unsupported_at(expression.span, "assignment operator")
                        })?),
                    };
                    let left = compound
                        .map(|_| self.property_access(object, member.property.name.as_str(), None));
                    let right = self.emit_expression(&expression.right)?;
                    let value = match compound {
                        Some(op) => self.binary(op, left.unwrap(), right),
                        None => right,
                    };
                    return Ok(self.property_access(
                        object,
                        member.property.name.as_str(),
                        Some(value),
                    ));
                }
                let (name, symbol) = match &expression.left {
                    AssignmentTarget::AssignmentTargetIdentifier(identifier) => (
                        identifier.name.as_str(),
                        identifier
                            .reference_id
                            .get()
                            .and_then(|id| self.scoping.get_reference(id).symbol_id()),
                    ),
                    other => {
                        return Err(unsupported_at(other.span(), "this assignment target"));
                    }
                };
                if !self
                    .arguments_references
                    .contains_key(&expression.left.span().start)
                    && symbol.is_some_and(|id| self.scoping.symbol_flags(id).is_const_variable())
                {
                    return Err(unsupported_at(expression.span, "assignment to const"));
                }
                let symbol = self
                    .arguments_references
                    .get(&expression.left.span().start)
                    .copied()
                    .or(symbol);
                let variable = symbol.and_then(|id| self.local(id));
                let compound = match expression.operator {
                    AssignmentOperator::Assign => None,
                    operator => Some(compound_operator(operator).ok_or_else(|| {
                        unsupported_at(
                            expression.span,
                            "assignment operator in assignment expression",
                        )
                    })?),
                };
                let left = compound.map(|_| match variable {
                    Some(variable) => self.read_local(variable),
                    None => self.global_access(name, None),
                });
                let right = self.emit_expression(&expression.right)?;
                let value = match compound {
                    Some(op) => self.binary(op, left.unwrap(), right),
                    None => right,
                };
                if let Some(variable) = variable {
                    self.store_local(variable, value);
                } else {
                    self.global_access(name, Some(value));
                }
                Ok(value)
            }
            Expression::ArrowFunctionExpression(function) => {
                self.make_closure(function.span, true, false)
            }
            Expression::FunctionExpression(function) => {
                self.make_closure(function.span, false, function.generator)
            }
            Expression::CallExpression(expression) => self.emit_call(expression),
            Expression::NewExpression(expression) => {
                let callable = self.emit_expression(&expression.callee)?;
                if let Some(arguments) = self.emit_spread_arguments(&expression.arguments)? {
                    let result = self.call_helper(Helper::ConstructSpread, &[callable, arguments]);
                    self.check_error();
                    return Ok(result);
                }
                let mut args = Vec::new();
                for argument in &expression.arguments {
                    let argument = argument
                        .as_expression()
                        .ok_or_else(|| unsupported_at(expression.span, "spread argument"))?;
                    args.push(self.emit_expression(argument)?);
                }
                let argv = self.stack_values(&args, expression.span)?;
                let argc = self.b.ins().iconst(types::I64, args.len() as i64);
                let result = self.call_helper(Helper::Construct, &[callable, argv, argc]);
                self.check_error();
                Ok(result)
            }
            other => Err(unsupported_at(
                other.span(),
                format!("{} expression", expression_kind(other)),
            )),
        }
    }

    fn emit_call(&mut self, expression: &CallExpression<'_>) -> Result<ClifValue, Error> {
        let (callable, receiver) = self.emit_call_reference(&expression.callee)?;
        self.emit_call_arguments(expression, callable, receiver)
    }

    pub(super) fn emit_call_reference(
        &mut self,
        expression: &Expression<'_>,
    ) -> Result<(ClifValue, ClifValue), Error> {
        let mut callee = expression;
        while let Expression::ParenthesizedExpression(expression) = callee {
            callee = &expression.expression;
        }
        let result = match callee {
            Expression::StaticMemberExpression(member) => {
                let receiver = self.emit_expression(&member.object)?;
                let callable = self.property_access(receiver, member.property.name.as_str(), None);
                (callable, receiver)
            }
            Expression::ComputedMemberExpression(member) => {
                let receiver = self.emit_expression(&member.object)?;
                let key = self.emit_expression(&member.expression)?;
                let callable = self.index_access(receiver, key, None);
                (callable, receiver)
            }
            Expression::ChainExpression(chain) => return self.emit_optional_chain(chain),
            _ => {
                let callable = self.emit_expression(callee)?;
                let receiver = self.constant(Value::UNDEFINED);
                (callable, receiver)
            }
        };
        Ok(result)
    }

    pub(super) fn emit_call_arguments(
        &mut self,
        expression: &CallExpression<'_>,
        callable: ClifValue,
        receiver: ClifValue,
    ) -> Result<ClifValue, Error> {
        if let Some(arguments) = self.emit_spread_arguments(&expression.arguments)? {
            let result = self.call_helper(Helper::CallSpread, &[callable, receiver, arguments]);
            self.check_error();
            return Ok(result);
        }
        let mut args = Vec::with_capacity(expression.arguments.len());
        for argument in &expression.arguments {
            let argument: &Expression<'_> = argument
                .as_expression()
                .ok_or_else(|| unsupported_at(expression.span, "spread argument"))?;
            args.push(self.emit_expression(argument)?);
        }
        let argc = args.len();
        let argv = self.stack_values(&args, expression.span)?;
        let argc = self.b.ins().iconst(types::I64, argc as i64);
        let result = self.call_helper(Helper::Call, &[callable, receiver, argv, argc]);
        self.check_error();
        Ok(result)
    }

    fn emit_spread_arguments(
        &mut self,
        arguments: &[Argument<'_>],
    ) -> Result<Option<ClifValue>, Error> {
        if !arguments
            .iter()
            .any(|argument| matches!(argument, Argument::SpreadElement(_)))
        {
            return Ok(None);
        }
        let ptr = self.b.ins().iconst(self.pointer, 0);
        let zero = self.b.ins().iconst(types::I64, 0);
        let values = self.call_helper(Helper::ArrayNew, &[ptr, zero]);
        self.check_error();
        for argument in arguments {
            let (expression, helper) = match argument {
                Argument::SpreadElement(spread) => (&spread.argument, Helper::ArrayExtend),
                _ => (argument.as_expression().unwrap(), Helper::ArrayAppend),
            };
            let value = self.emit_expression(expression)?;
            self.call_helper(helper, &[values, value]);
            self.check_error();
        }
        Ok(Some(values))
    }

    pub(super) fn make_closure(
        &mut self,
        span: Span,
        arrow: bool,
        generator: bool,
    ) -> Result<ClifValue, Error> {
        let (id, captures, lexical_this) = self
            .closures
            .get(&span.start)
            .cloned()
            .ok_or_else(|| compile_at(span, "missing closure code"))?;
        let mut cells = Vec::with_capacity(captures.len());
        for symbol in captures {
            cells.push(
                self.capture_cell(symbol)
                    .ok_or_else(|| compile_at(span, "missing captured binding"))?,
            );
        }
        let argv = self.stack_values(&cells, span)?;
        let count = self.b.ins().iconst(types::I64, cells.len() as i64);
        let code = self.b.ins().iconst(types::I64, id.as_u32() as i64);
        let result = self.call_helper(Helper::ClosureNew, &[code, argv, count]);
        self.check_error();
        if arrow {
            let receiver = if lexical_this {
                self.receiver
            } else {
                self.constant(Value::UNDEFINED)
            };
            self.call_helper(Helper::BindThis, &[result, receiver]);
            self.check_error();
        } else if generator {
            self.call_helper(Helper::DisableConstructor, &[result]);
        }
        Ok(result)
    }

    fn stack_values(&mut self, values: &[ClifValue], span: Span) -> Result<ClifValue, Error> {
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
            self.b.ins().store(MemFlags::trusted(), *value, address, 0);
        }
        Ok(self.b.ins().stack_addr(self.pointer, slot, 0))
    }
}

fn has_named_capture(pattern: &str) -> bool {
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

fn expression_kind(expression: &Expression<'_>) -> &'static str {
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

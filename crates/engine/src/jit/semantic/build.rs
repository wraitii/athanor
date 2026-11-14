use crate::{FxHashMap, FxHashSet};

use cranelift_codegen::entity::PrimaryMap;
use oxc_ast::ast::{
    Argument, ArrayExpressionElement, AssignmentExpression, BindingIdentifier, BindingPattern,
    CallExpression, ChainElement, ChainExpression, DoWhileStatement, Expression, ForInStatement,
    ForOfStatement, ForStatement, ForStatementInit, ForStatementLeft, FormalParameters,
    IfStatement, ObjectExpression, ObjectPropertyKind, PropertyKind, SimpleAssignmentTarget,
    Statement, SwitchStatement, UpdateExpression, VariableDeclaration, VariableDeclarationKind,
    WhileStatement,
};
use oxc_semantic::Scoping;
use oxc_span::{GetSpan, Span};
use oxc_syntax::operator::{
    AssignmentOperator, BinaryOperator, LogicalOperator, UnaryOperator, UpdateOperator,
};
use oxc_syntax::{reference::ReferenceId, symbol::SymbolId};

use super::{
    Binding, BindingData, BindingKind, Block, BlockData, Constant, Graph, InstData, Op, Result,
    Scope, ScopeData, ScopeKind, Target, TerminatorKind, UnaryOp, Value, ValueDef, unsupported,
};

enum Reference {
    Local(Binding),
    Global(String),
    Constant(Constant),
}

enum Place {
    Binding(Binding),
    Global(String),
    Property(Value, String),
    Index(Value, Value),
}

struct Loop {
    continue_target: Option<Block>,
    break_target: Block,
    depth: usize,
    is_loop: bool,
}

struct Builder<'a> {
    scoping: &'a Scoping,
    captured: &'a FxHashSet<SymbolId>,
    closures: &'a FxHashMap<u32, (cranelift_module::FuncId, Vec<SymbolId>, bool)>,
    arguments_references: &'a FxHashMap<u32, SymbolId>,
    graph: Graph,
    current: Option<Block>,
    symbols: FxHashMap<SymbolId, Binding>,
    open: Vec<Scope>,
    loops: Vec<Loop>,
    handlers: Vec<(Block, usize)>,
}

fn literal_elements<'s, 'a>(init: &'s Expression<'a>) -> Option<Vec<&'s Expression<'a>>> {
    let Expression::ArrayExpression(array) = init.without_parentheses() else {
        return None;
    };
    array
        .elements
        .iter()
        .map(ArrayExpressionElement::as_expression)
        .collect()
}

fn pattern_bindings<'s, 'a>(
    pattern: &'s BindingPattern<'a>,
) -> Result<Vec<&'s BindingIdentifier<'a>>> {
    match pattern {
        BindingPattern::BindingIdentifier(id) => Ok(vec![id]),
        BindingPattern::ArrayPattern(pattern) if pattern.rest.is_none() => pattern
            .elements
            .iter()
            .flatten()
            .map(|element| match element {
                BindingPattern::BindingIdentifier(id) => Ok(&**id),
                _ => unsupported("array binding pattern", element.span()),
            })
            .collect(),
        BindingPattern::ObjectPattern(pattern) if pattern.rest.is_none() => pattern
            .properties
            .iter()
            .map(|property| {
                if property.computed
                    || property
                        .key
                        .static_name()
                        .and_then(|name| name.as_str().map(str::to_owned))
                        .is_none()
                {
                    return unsupported("object binding key", property.span);
                }
                match &property.value {
                    BindingPattern::BindingIdentifier(id) => Ok(&**id),
                    _ => unsupported("object binding pattern", property.span),
                }
            })
            .collect(),
        _ => unsupported("binding pattern", pattern.span()),
    }
}

fn declaration_bindings<'s, 'a>(
    declaration: &'s VariableDeclaration<'a>,
) -> Result<Vec<(&'s BindingIdentifier<'a>, BindingKind)>> {
    let kind = match declaration.kind {
        VariableDeclarationKind::Let => BindingKind::Let,
        VariableDeclarationKind::Const => BindingKind::Const,
        _ => return unsupported("non-lexical declaration", declaration.span),
    };
    let mut bindings = Vec::new();
    for item in &declaration.declarations {
        bindings.extend(pattern_bindings(&item.id)?.into_iter().map(|id| (id, kind)));
    }
    Ok(bindings)
}

fn lexical_bindings<'s, 'a>(
    statements: &'s [Statement<'a>],
) -> Result<Vec<(&'s BindingIdentifier<'a>, BindingKind)>> {
    let mut bindings = Vec::new();
    for statement in statements {
        match statement {
            Statement::VariableDeclaration(declaration)
                if declaration.kind != VariableDeclarationKind::Var =>
            {
                bindings.extend(declaration_bindings(declaration)?);
            }
            Statement::FunctionDeclaration(function) => {
                let Some(id) = &function.id else {
                    return unsupported("anonymous function declaration", function.span);
                };
                bindings.push((id, BindingKind::Let));
            }
            Statement::ClassDeclaration(class) => {
                let Some(id) = &class.id else {
                    return unsupported("anonymous class declaration", class.span);
                };
                bindings.push((id, BindingKind::Let));
            }
            _ => {}
        }
    }
    Ok(bindings)
}

impl<'a> Builder<'a> {
    fn block(&mut self) -> Block {
        self.graph.blocks.push(BlockData::default())
    }

    fn param(&mut self, block: Block) -> Value {
        let index = self.graph.blocks[block].params.len();
        let value = self.graph.values.push(ValueDef::Param(block, index));
        self.graph.blocks[block].params.push(value);
        value
    }

    fn switch(&mut self, block: Block) {
        self.current = Some(block);
    }

    fn push(&mut self, span: Span, op: Op, has_result: bool) -> Option<Value> {
        let block = self.current.expect("emission into unreachable code");
        let inst = self.graph.insts.next_key();
        let result = has_result.then(|| self.graph.values.push(ValueDef::Result(inst, 0)));
        self.graph.insts.push(InstData {
            source_name: None,
            emission_location: None,
            span,
            block,
            results: result.into_iter().collect(),
            op,
        });
        self.graph.blocks[block].insts.push(inst);
        if let Some(&(handler, depth)) = self.handlers.last()
            && !matches!(
                self.graph.insts[inst].op,
                Op::EnterScope(_) | Op::LeaveScope(_) | Op::Catch
            )
        {
            let normal = self.block();
            let error = self.block();
            self.terminate(
                span,
                TerminatorKind::CheckException(Target::new(normal), Target::new(error)),
            );
            self.switch(error);
            for index in (depth..self.open.len()).rev() {
                let scope = self.open[index];
                let block = self.current.unwrap();
                let inst = self.graph.insts.push(InstData {
                    source_name: None,
                    emission_location: None,
                    span,
                    block,
                    results: Vec::new(),
                    op: Op::LeaveScope(scope),
                });
                self.graph.blocks[block].insts.push(inst);
            }
            self.jump(span, handler, Vec::new());
            self.switch(normal);
        }
        result
    }

    fn emit(&mut self, span: Span, op: Op) -> Value {
        self.push(span, op, true).unwrap()
    }

    fn effect(&mut self, span: Span, op: Op) {
        self.push(span, op, false);
    }

    fn constant(&mut self, span: Span, constant: Constant) -> Value {
        self.emit(span, Op::Constant(constant))
    }

    fn terminate(&mut self, span: Span, kind: TerminatorKind) {
        let block = self
            .current
            .take()
            .expect("termination of unreachable code");
        self.graph.terminate(block, span, kind);
    }

    fn jump(&mut self, span: Span, block: Block, args: Vec<Value>) {
        self.terminate(span, TerminatorKind::Jump(Target { block, args }));
    }

    fn branch(&mut self, span: Span, condition: Value, then: Target, otherwise: Target) {
        self.terminate(span, TerminatorKind::Branch(condition, then, otherwise));
    }

    fn open_scope(
        &mut self,
        span: Span,
        kind: ScopeKind,
        bindings: Vec<(&BindingIdentifier<'_>, BindingKind)>,
    ) -> Result<Option<Scope>> {
        let bindings: Vec<_> = bindings
            .into_iter()
            .filter(|(id, _)| {
                kind != ScopeKind::Body
                    || !id
                        .symbol_id
                        .get()
                        .is_some_and(|symbol| self.symbols.contains_key(&symbol))
            })
            .collect();
        if bindings.is_empty() {
            return Ok(None);
        }
        let scope = self.graph.scopes.push(ScopeData {
            kind,
            parent: self.open.last().copied(),
            bindings: Vec::new(),
        });
        for (id, binding_kind) in bindings {
            let Some(symbol) = id.symbol_id.get() else {
                return unsupported("unresolved binding", id.span);
            };
            let binding = self.graph.bindings.push(BindingData {
                persistent: false,
                name: id.name.to_string(),
                kind: binding_kind,
                scope,
                captured: self.captured.contains(&symbol),
            });
            self.graph.scopes[scope].bindings.push(binding);
            self.symbols.insert(symbol, binding);
        }
        self.effect(span, Op::EnterScope(scope));
        self.open.push(scope);
        Ok(Some(scope))
    }

    fn close_scope(&mut self, span: Span, scope: Option<Scope>) {
        let Some(scope) = scope else {
            return;
        };
        self.open.pop();
        if self.current.is_some() {
            self.effect(span, Op::LeaveScope(scope));
        }
    }

    fn unwind(&mut self, span: Span, depth: usize) {
        for index in (depth..self.open.len()).rev() {
            self.effect(span, Op::LeaveScope(self.open[index]));
        }
    }

    fn resolve(&self, name: &str, reference: Option<ReferenceId>, span: Span) -> Result<Reference> {
        let symbol = self
            .arguments_references
            .get(&span.start)
            .copied()
            .or_else(|| reference.and_then(|id| self.scoping.get_reference(id).symbol_id()));
        let Some(symbol) = symbol else {
            return Ok(match name {
                "arguments" => return unsupported("arguments object", span),
                "undefined" => Reference::Constant(Constant::Undefined),
                "Infinity" => Reference::Constant(Constant::Number(f64::INFINITY)),
                "NaN" => Reference::Constant(Constant::Number(f64::NAN)),
                _ => Reference::Global(name.to_owned()),
            });
        };
        if let Some(&binding) = self.symbols.get(&symbol) {
            return Ok(Reference::Local(binding));
        }
        if self.scoping.symbol_scope_id(symbol) == self.scoping.root_scope_id() {
            return Ok(Reference::Global(name.to_owned()));
        }
        unsupported("binding", span)
    }

    fn nullish_literal(&mut self, expression: &Expression<'_>) -> Result<bool> {
        Ok(match expression.without_parentheses() {
            Expression::NullLiteral(_) => true,
            Expression::Identifier(id) => matches!(
                self.resolve(&id.name, id.reference_id.get(), id.span)?,
                Reference::Constant(Constant::Undefined)
            ),
            _ => false,
        })
    }

    fn place(&mut self, target: &SimpleAssignmentTarget<'_>) -> Result<Place> {
        match target {
            SimpleAssignmentTarget::AssignmentTargetIdentifier(id) => {
                let symbol = id
                    .reference_id
                    .get()
                    .and_then(|id| self.scoping.get_reference(id).symbol_id());
                if symbol.is_some_and(|id| self.scoping.symbol_flags(id).is_const_variable()) {
                    return unsupported("assignment to const", id.span);
                }
                match self.resolve(&id.name, id.reference_id.get(), id.span)? {
                    Reference::Local(binding)
                        if self.graph.bindings[binding].kind != BindingKind::Const =>
                    {
                        Ok(Place::Binding(binding))
                    }
                    Reference::Local(_) => unsupported("assignment to const", id.span),
                    Reference::Global(_) | Reference::Constant(_) => {
                        Ok(Place::Global(id.name.to_string()))
                    }
                }
            }
            SimpleAssignmentTarget::StaticMemberExpression(member) => {
                let object = self.expression(&member.object)?;
                Ok(Place::Property(object, member.property.name.to_string()))
            }
            SimpleAssignmentTarget::ComputedMemberExpression(member) => {
                let object = self.expression(&member.object)?;
                let key = self.expression(&member.expression)?;
                Ok(Place::Index(object, key))
            }
            _ => unsupported("assignment target", target.span()),
        }
    }

    fn load(&mut self, span: Span, place: &Place) -> Value {
        let op = match place {
            Place::Binding(binding) => Op::Read(*binding),
            Place::Global(name) => Op::GlobalRead(name.clone()),
            Place::Property(object, name) => Op::Get(*object, name.clone()),
            Place::Index(object, key) => Op::GetIndex(*object, *key),
        };
        self.emit(span, op)
    }

    fn store(&mut self, span: Span, place: Place, value: Value) {
        let op = match place {
            Place::Binding(binding) => Op::Write(binding, value),
            Place::Global(name) => Op::GlobalWrite(name, value),
            Place::Property(object, name) => Op::Set(object, name, value),
            Place::Index(object, key) => Op::SetIndex(object, key, value),
        };
        self.effect(span, op);
    }

    fn closure(&self, span: Span, arrow: bool, generator: bool) -> Result<Op> {
        let Some((code, symbols, lexical_this)) = self.closures.get(&span.start) else {
            return unsupported("closure capture metadata", span);
        };
        let captures = symbols
            .iter()
            .map(|symbol| {
                self.symbols.get(symbol).copied().ok_or_else(|| {
                    format!(
                        "unsupported closure capture at {}..{}",
                        span.start, span.end
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Op::Closure(
            span,
            *code,
            arrow,
            generator,
            *lexical_this,
            captures,
            None,
        ))
    }

    fn expression(&mut self, expr: &Expression<'_>) -> Result<Value> {
        let span = expr.span();
        let op = match expr {
            Expression::ChainExpression(chain) => return Ok(self.chain(chain)?.0),
            Expression::RegExpLiteral(literal) => {
                use oxc_ast::ast::RegExpFlags;
                let flags = literal.regex.flags;
                let pattern = literal.regex.pattern.text.as_str();
                if flags.intersects(RegExpFlags::Y | RegExpFlags::D | RegExpFlags::V) {
                    return unsupported("regular expression flags y, d, and v", span);
                }
                if crate::jit::emit::expression::has_named_capture(pattern) {
                    return unsupported("named regular expression capture", span);
                }
                let bits = u8::from(flags.contains(RegExpFlags::I))
                    | (u8::from(flags.contains(RegExpFlags::M)) << 1)
                    | (u8::from(flags.contains(RegExpFlags::S)) << 2)
                    | (u8::from(flags.contains(RegExpFlags::U)) << 3)
                    | (u8::from(flags.contains(RegExpFlags::G)) << 4);
                regress::Regex::with_flags(pattern, crate::heap::regexp_flags(bits))
                    .map_err(|error| format!("invalid regular expression: {error}"))?;
                Op::RegExp(pattern.to_owned(), bits)
            }
            Expression::ThisExpression(_) => Op::This,
            Expression::FunctionExpression(function) => {
                self.closure(function.span, false, function.generator)?
            }
            Expression::ArrowFunctionExpression(function) => {
                self.closure(function.span, true, false)?
            }
            Expression::NumericLiteral(x) => Op::Constant(Constant::Number(x.value)),
            Expression::StringLiteral(x) => match x.value.as_str() {
                Some(value) => Op::Constant(Constant::String(value.to_owned())),
                None => return unsupported("string literal", span),
            },
            Expression::BooleanLiteral(x) => Op::Constant(Constant::Boolean(x.value)),
            Expression::NullLiteral(_) => Op::Constant(Constant::Null),
            Expression::Identifier(x) => match self.resolve(&x.name, x.reference_id.get(), span)? {
                Reference::Local(binding) => Op::Read(binding),
                Reference::Global(name) => Op::GlobalRead(name),
                Reference::Constant(constant) => Op::Constant(constant),
            },
            Expression::ParenthesizedExpression(x) => return self.expression(&x.expression),
            Expression::SequenceExpression(x) => {
                let mut value = None;
                for expression in &x.expressions {
                    value = Some(self.expression(expression)?);
                }
                return value.map_or_else(|| unsupported("empty sequence", span), Ok);
            }
            Expression::StaticMemberExpression(x) => {
                let object = self.expression(&x.object)?;
                Op::Get(object, x.property.name.to_string())
            }
            Expression::ComputedMemberExpression(x) => {
                let object = self.expression(&x.object)?;
                let key = self.expression(&x.expression)?;
                Op::GetIndex(object, key)
            }
            Expression::ArrayExpression(x) => {
                if x.elements
                    .iter()
                    .any(|element| matches!(element, ArrayExpressionElement::SpreadElement(_)))
                {
                    if x.elements
                        .iter()
                        .any(|element| matches!(element, ArrayExpressionElement::Elision(_)))
                    {
                        return unsupported("array literal holes", span);
                    }
                    let array = self.emit(span, Op::NewArray(Vec::new()));
                    for element in &x.elements {
                        let (expression, spread) = match element {
                            ArrayExpressionElement::SpreadElement(spread) => {
                                (&spread.argument, true)
                            }
                            _ => (element.as_expression().unwrap(), false),
                        };
                        let value = self.expression(expression)?;
                        let value = if spread {
                            self.emit(element.span(), Op::Spread(value))
                        } else {
                            value
                        };
                        self.effect(element.span(), Op::ArrayAppend(array, value, spread));
                    }
                    return Ok(array);
                }
                let mut values = Vec::new();
                for element in &x.elements {
                    let Some(element) = element.as_expression() else {
                        return unsupported("array element", element.span());
                    };
                    values.push(self.expression(element)?);
                }
                Op::NewArray(values)
            }
            Expression::ObjectExpression(x) => return self.object(x),
            Expression::UnaryExpression(x) => {
                let operator = match x.operator {
                    UnaryOperator::UnaryNegation => UnaryOp::Negate,
                    UnaryOperator::UnaryPlus => UnaryOp::Plus,
                    UnaryOperator::LogicalNot => UnaryOp::Not,
                    UnaryOperator::BitwiseNot => UnaryOp::BitwiseNot,
                    UnaryOperator::Typeof => {
                        if let Expression::Identifier(id) = x.argument.without_parentheses()
                            && let Reference::Global(_) =
                                self.resolve(&id.name, id.reference_id.get(), id.span)?
                        {
                            let value =
                                self.emit(id.span, Op::GlobalReadOptional(id.name.to_string()));
                            return Ok(self.emit(span, Op::Unary(UnaryOp::Typeof, value)));
                        }
                        UnaryOp::Typeof
                    }
                    UnaryOperator::Void => {
                        self.expression(&x.argument)?;
                        return Ok(self.constant(span, Constant::Undefined));
                    }
                    UnaryOperator::Delete => {
                        return unsupported("unary operator in unary expression", span);
                    }
                };
                let value = self.expression(&x.argument)?;
                Op::Unary(operator, value)
            }
            Expression::BinaryExpression(x) => {
                if matches!(
                    x.operator,
                    BinaryOperator::Equality | BinaryOperator::Inequality
                ) {
                    let other = if self.nullish_literal(&x.right)? {
                        Some(&x.left)
                    } else if self.nullish_literal(&x.left)? {
                        Some(&x.right)
                    } else {
                        None
                    };
                    if let Some(other) = other {
                        let value = self.expression(other)?;
                        let nullish = self.emit(span, Op::IsNullish(value));
                        if x.operator == BinaryOperator::Equality {
                            return Ok(nullish);
                        }
                        return Ok(self.emit(span, Op::Unary(UnaryOp::Not, nullish)));
                    }
                }
                let left = self.expression(&x.left)?;
                let right = self.expression(&x.right)?;
                Op::Binary(x.operator, left, right)
            }
            Expression::LogicalExpression(x) => {
                return self.logical(span, x.operator, &x.left, &x.right);
            }
            Expression::ConditionalExpression(x) => {
                let condition = self.expression(&x.test)?;
                let then = self.block();
                let otherwise = self.block();
                let join = self.block();
                let result = self.param(join);
                self.branch(span, condition, Target::new(then), Target::new(otherwise));
                self.switch(then);
                let value = self.expression(&x.consequent)?;
                self.jump(span, join, vec![value]);
                self.switch(otherwise);
                let value = self.expression(&x.alternate)?;
                self.jump(span, join, vec![value]);
                self.switch(join);
                return Ok(result);
            }
            Expression::CallExpression(x) => return self.call(x),
            Expression::NewExpression(x) => {
                let callee = self.expression(&x.callee)?;
                if let Some(args) = self.spread_arguments(x.span, &x.arguments)? {
                    Op::ConstructPack(callee, args)
                } else {
                    let mut args = Vec::new();
                    for arg in &x.arguments {
                        args.push(self.expression(arg.as_expression().unwrap())?);
                    }
                    Op::Construct { callee, args }
                }
            }

            Expression::AssignmentExpression(x) => return self.assignment(x),
            Expression::UpdateExpression(x) => return self.update(x),
            _ => {
                return unsupported(
                    &format!(
                        "{} expression",
                        crate::jit::emit::expression::expression_kind(expr)
                    ),
                    span,
                );
            }
        };
        Ok(self.emit(span, op))
    }

    fn object(&mut self, x: &ObjectExpression<'_>) -> Result<Value> {
        let object = self.emit(x.span, Op::NewObject);
        for property in &x.properties {
            let ObjectPropertyKind::ObjectProperty(property) = property else {
                return unsupported("object spread", property.span());
            };
            if property.kind != PropertyKind::Init || property.method {
                return unsupported("object property", property.span);
            }
            if property.computed {
                let Some(key) = property.key.as_expression() else {
                    return unsupported("property key", property.span);
                };
                let key = self.expression(key)?;
                let key = self.emit(property.span, Op::PropertyKey(key));
                let value = self.expression(&property.value)?;
                self.effect(property.span, Op::SetIndex(object, key, value));
                continue;
            }
            let Some(name) = property
                .key
                .static_name()
                .and_then(|name| name.as_str().map(str::to_owned))
            else {
                return unsupported("property key", property.span);
            };
            if name == "__proto__" && !property.shorthand {
                return unsupported("prototype setter", property.span);
            }
            let value = self.expression(&property.value)?;
            self.effect(property.span, Op::Define(object, name, value));
        }
        Ok(object)
    }

    fn logical(
        &mut self,
        span: Span,
        operator: LogicalOperator,
        left: &Expression<'_>,
        right: &Expression<'_>,
    ) -> Result<Value> {
        let left = self.expression(left)?;
        let rest = self.block();
        let join = self.block();
        let result = self.param(join);
        let done = Target {
            block: join,
            args: vec![left],
        };
        match operator {
            LogicalOperator::Or => self.branch(span, left, done, Target::new(rest)),
            LogicalOperator::And => self.branch(span, left, Target::new(rest), done),
            LogicalOperator::Coalesce => {
                let nullish = self.emit(span, Op::IsNullish(left));
                self.branch(span, nullish, Target::new(rest), done);
            }
        }
        self.switch(rest);
        let value = self.expression(right)?;
        self.jump(span, join, vec![value]);
        self.switch(join);
        Ok(result)
    }

    fn chain(&mut self, chain: &ChainExpression<'_>) -> Result<(Value, Value)> {
        let short = self.block();
        let done = self.block();
        let result = self.param(done);
        let receiver = self.param(done);
        let (value, this) = match &chain.expression {
            ChainElement::StaticMemberExpression(member) => self.chain_static(member, short)?,
            ChainElement::ComputedMemberExpression(member) => self.chain_computed(member, short)?,
            ChainElement::CallExpression(call) => self.chain_call(call, short)?,
            _ => return unsupported("optional chain element", chain.span),
        };
        self.jump(chain.span, done, vec![value, this]);
        self.switch(short);
        let undefined = self.constant(chain.span, Constant::Undefined);
        self.jump(chain.span, done, vec![undefined, undefined]);
        self.switch(done);
        Ok((result, receiver))
    }

    fn chain_guard(&mut self, span: Span, value: Value, short: Block) {
        let nullish = self.emit(span, Op::IsNullish(value));
        let present = self.block();
        self.branch(span, nullish, Target::new(short), Target::new(present));
        self.switch(present);
    }

    fn reference(
        &mut self,
        expression: &Expression<'_>,
        short: Option<Block>,
    ) -> Result<(Value, Value)> {
        match expression {
            Expression::ParenthesizedExpression(expr) => self.reference(&expr.expression, None),
            Expression::ChainExpression(chain) => self.chain(chain),
            Expression::StaticMemberExpression(member) if short.is_some() => {
                self.chain_static(member, short.unwrap())
            }
            Expression::ComputedMemberExpression(member) if short.is_some() => {
                self.chain_computed(member, short.unwrap())
            }
            Expression::StaticMemberExpression(member) => {
                let receiver = self.expression(&member.object)?;
                let value = self.emit(
                    member.span,
                    Op::Get(receiver, member.property.name.to_string()),
                );
                Ok((value, receiver))
            }
            Expression::ComputedMemberExpression(member) => {
                let receiver = self.expression(&member.object)?;
                let key = self.expression(&member.expression)?;
                let value = self.emit(member.span, Op::GetIndex(receiver, key));
                Ok((value, receiver))
            }
            Expression::CallExpression(call) if short.is_some() => {
                self.chain_call(call, short.unwrap())
            }
            _ => {
                let value = self.expression(expression)?;
                let receiver = self.constant(expression.span(), Constant::Undefined);
                Ok((value, receiver))
            }
        }
    }

    fn chain_static(
        &mut self,
        member: &oxc_ast::ast::StaticMemberExpression<'_>,
        short: Block,
    ) -> Result<(Value, Value)> {
        let (receiver, _) = self.reference(&member.object, Some(short))?;
        if member.optional {
            self.chain_guard(member.span, receiver, short);
        }
        let value = self.emit(
            member.span,
            Op::Get(receiver, member.property.name.to_string()),
        );
        Ok((value, receiver))
    }

    fn chain_computed(
        &mut self,
        member: &oxc_ast::ast::ComputedMemberExpression<'_>,
        short: Block,
    ) -> Result<(Value, Value)> {
        let (receiver, _) = self.reference(&member.object, Some(short))?;
        if member.optional {
            self.chain_guard(member.span, receiver, short);
        }
        let key = self.expression(&member.expression)?;
        let value = self.emit(member.span, Op::GetIndex(receiver, key));
        Ok((value, receiver))
    }

    fn chain_call(&mut self, call: &CallExpression<'_>, short: Block) -> Result<(Value, Value)> {
        let (callee, receiver) = self.reference(&call.callee, Some(short))?;
        if call.optional {
            self.chain_guard(call.span, callee, short);
        }
        let value = self.call_arguments(call, callee, receiver)?;
        let undefined = self.constant(call.span, Constant::Undefined);
        Ok((value, undefined))
    }

    fn spread_arguments(
        &mut self,
        span: Span,
        arguments: &[Argument<'_>],
    ) -> Result<Option<Value>> {
        if let [Argument::SpreadElement(spread)] = arguments {
            let value = self.expression(&spread.argument)?;
            return Ok(Some(self.emit(span, Op::Spread(value))));
        }
        if !arguments
            .iter()
            .any(|arg| matches!(arg, Argument::SpreadElement(_)))
        {
            return Ok(None);
        }
        let array = self.emit(span, Op::NewPack);
        for argument in arguments {
            let (expression, spread) = match argument {
                Argument::SpreadElement(spread) => (&spread.argument, true),
                _ => (argument.as_expression().unwrap(), false),
            };
            let value = self.expression(expression)?;
            let value = if spread {
                self.emit(argument.span(), Op::Spread(value))
            } else {
                value
            };
            self.effect(argument.span(), Op::PackAppend(array, value, spread));
        }
        Ok(Some(array))
    }

    fn call_arguments(
        &mut self,
        x: &CallExpression<'_>,
        callee: Value,
        receiver: Value,
    ) -> Result<Value> {
        if let Some(args) = self.spread_arguments(x.span, &x.arguments)? {
            return Ok(self.emit(x.span, Op::CallPack(callee, receiver, args)));
        }
        let mut args = Vec::new();
        for arg in &x.arguments {
            args.push(self.expression(arg.as_expression().unwrap())?);
        }
        Ok(self.emit(
            x.span,
            Op::Call {
                callee,
                receiver: Some(receiver),
                args,
                target: None,
            },
        ))
    }

    fn call(&mut self, x: &CallExpression<'_>) -> Result<Value> {
        if x.optional {
            return unsupported("optional call", x.span);
        }
        let (callee, receiver) = self.reference(&x.callee, None)?;
        self.call_arguments(x, callee, receiver)
    }

    fn assignment(&mut self, x: &AssignmentExpression<'_>) -> Result<Value> {
        let operator = match x.operator {
            AssignmentOperator::Assign => None,
            operator => match operator.to_binary_operator() {
                Some(operator) => Some(operator),
                None => return unsupported("logical assignment", x.span),
            },
        };
        let Some(target) = x.left.as_simple_assignment_target() else {
            return unsupported("destructuring assignment", x.span);
        };
        let place = self.place(target)?;
        let value = match operator {
            Some(operator) => {
                let old = self.load(x.span, &place);
                let right = self.expression(&x.right)?;
                self.emit(x.span, Op::Binary(operator, old, right))
            }
            None => self.expression(&x.right)?,
        };
        self.store(x.span, place, value);
        Ok(value)
    }

    fn update(&mut self, x: &UpdateExpression<'_>) -> Result<Value> {
        let operator = match x.operator {
            UpdateOperator::Increment => UnaryOp::Increment,
            UpdateOperator::Decrement => UnaryOp::Decrement,
        };
        let place = self.place(&x.argument)?;
        let old = self.load(x.span, &place);
        let numeric = self.emit(x.span, Op::ToNumeric(old));
        let value = self.emit(x.span, Op::Unary(operator, numeric));
        self.store(x.span, place, value);
        Ok(if x.prefix { value } else { numeric })
    }

    fn bind_pattern(&mut self, pattern: &BindingPattern<'_>, value: Value) -> Result<()> {
        match pattern {
            BindingPattern::BindingIdentifier(id) => {
                let symbol = id.symbol_id.get().unwrap();
                let Some(&binding) = self.symbols.get(&symbol) else {
                    if self.scoping.symbol_scope_id(symbol) != self.scoping.root_scope_id() {
                        return unsupported("binding", id.span);
                    }
                    self.effect(id.span, Op::GlobalWrite(id.name.to_string(), value));
                    return Ok(());
                };
                self.effect(id.span, Op::Initialize(binding, value));
            }
            BindingPattern::ArrayPattern(pattern) => {
                let iterator = self.emit(pattern.span, Op::IteratorNew(value));
                for element in &pattern.elements {
                    let next = self.emit(pattern.span, Op::IteratorNext(iterator));
                    if let Some(element) = element {
                        let missing = self.emit(element.span(), Op::IsUninitialized(next));
                        let absent = self.block();
                        let done = self.block();
                        let result = self.param(done);
                        self.branch(
                            element.span(),
                            missing,
                            Target::new(absent),
                            Target {
                                block: done,
                                args: vec![next],
                            },
                        );
                        self.switch(absent);
                        let undefined = self.constant(element.span(), Constant::Undefined);
                        self.jump(element.span(), done, vec![undefined]);
                        self.switch(done);
                        self.bind_pattern(element, result)?;
                    }
                }
            }
            BindingPattern::ObjectPattern(pattern) => {
                for property in &pattern.properties {
                    let Some(name) = property
                        .key
                        .static_name()
                        .and_then(|name| name.as_str().map(str::to_owned))
                    else {
                        return unsupported("object binding key", property.span);
                    };
                    if property.computed {
                        return unsupported("object binding key", property.span);
                    }
                    let value = self.emit(property.span, Op::Get(value, name));
                    self.bind_pattern(&property.value, value)?;
                }
            }
            _ => return unsupported("binding pattern", pattern.span()),
        }
        Ok(())
    }

    fn initialize(&mut self, declaration: &VariableDeclaration<'_>) -> Result<()> {
        for item in &declaration.declarations {
            pattern_bindings(&item.id)?;
            if let BindingPattern::ArrayPattern(pattern) = &item.id
                && let Some(elements) = item.init.as_ref().and_then(literal_elements)
            {
                let mut values = Vec::new();
                for element in elements {
                    values.push(self.expression(element)?);
                }
                for (index, element) in pattern.elements.iter().enumerate() {
                    let Some(element) = element else {
                        continue;
                    };
                    let value = match values.get(index) {
                        Some(&value) => value,
                        None => self.constant(element.span(), Constant::Undefined),
                    };
                    self.bind_pattern(element, value)?;
                }
                continue;
            }
            let value = match &item.init {
                Some(init) => self.expression(init)?,
                None if declaration.kind == VariableDeclarationKind::Var => continue,
                None if !matches!(&item.id, BindingPattern::BindingIdentifier(_)) => {
                    return unsupported("destructuring without initializer", item.span);
                }
                None => self.constant(item.span, Constant::Undefined),
            };
            self.bind_pattern(&item.id, value)?;
        }
        Ok(())
    }

    fn statements(&mut self, statements: &[Statement<'_>]) -> Result<()> {
        for statement in statements {
            if let Statement::FunctionDeclaration(function) = statement {
                let symbol = function.id.as_ref().unwrap().symbol_id.get().unwrap();
                let binding = self.symbols[&symbol];
                let closure = self.closure(function.span, false, function.generator)?;
                let value = self.emit(function.span, closure);
                self.effect(function.span, Op::Initialize(binding, value));
            }
        }
        for statement in statements {
            if self.current.is_none() {
                break;
            }
            if !matches!(statement, Statement::FunctionDeclaration(_)) {
                self.statement(statement)?;
            }
        }
        Ok(())
    }

    fn statement(&mut self, statement: &Statement<'_>) -> Result<()> {
        match statement {
            Statement::BlockStatement(block) => {
                let scope =
                    self.open_scope(block.span, ScopeKind::Block, lexical_bindings(&block.body)?)?;
                self.statements(&block.body)?;
                self.close_scope(block.span, scope);
            }
            Statement::EmptyStatement(_) | Statement::DebuggerStatement(_) => {}
            Statement::VariableDeclaration(declaration) => {
                self.initialize(declaration)?;
            }
            Statement::ExpressionStatement(statement) => {
                if let Expression::YieldExpression(yield_) = &statement.expression {
                    if !self.graph.generator {
                        return unsupported("yield outside generator", yield_.span);
                    }
                    let value = match &yield_.argument {
                        Some(argument) => self.expression(argument)?,
                        None => self.constant(yield_.span, Constant::Undefined),
                    };
                    let resume = self.block();
                    self.terminate(
                        yield_.span,
                        TerminatorKind::Yield(yield_.span, value, Target::new(resume)),
                    );
                    self.switch(resume);
                } else {
                    self.expression(&statement.expression)?;
                }
            }
            Statement::ThrowStatement(statement) => {
                let value = self.expression(&statement.argument)?;
                self.effect(statement.span, Op::Throw(value));
                let undefined = self.constant(statement.span, Constant::Undefined);
                self.unwind(statement.span, 0);
                self.terminate(statement.span, TerminatorKind::Return(undefined));
            }
            Statement::TryStatement(statement) => self.try_statement(statement)?,
            Statement::ClassDeclaration(class) => self.class_declaration(class)?,
            Statement::ReturnStatement(statement) => {
                let value = match &statement.argument {
                    Some(expression) => self.expression(expression)?,
                    None => self.constant(statement.span, Constant::Undefined),
                };
                self.unwind(statement.span, 0);
                self.terminate(statement.span, TerminatorKind::Return(value));
            }
            Statement::IfStatement(statement) => self.if_statement(statement)?,
            Statement::ForStatement(statement) => self.for_statement(statement)?,
            Statement::ForOfStatement(statement) => self.for_of_statement(statement)?,
            Statement::WhileStatement(statement) => self.while_statement(statement)?,
            Statement::DoWhileStatement(statement) => self.do_while_statement(statement)?,
            Statement::ForInStatement(statement) => self.for_in_statement(statement)?,
            Statement::SwitchStatement(statement) => self.switch_statement(statement)?,
            Statement::BreakStatement(statement) => {
                if statement.label.is_some() {
                    return unsupported("labelled break", statement.span);
                }
                let Some(target) = self.loops.last() else {
                    return unsupported("break outside loop", statement.span);
                };
                let (block, depth) = (target.break_target, target.depth);
                self.unwind(statement.span, depth);
                self.jump(statement.span, block, Vec::new());
            }
            Statement::ContinueStatement(statement) => {
                if statement.label.is_some() {
                    return unsupported("labelled continue", statement.span);
                }
                let Some(index) = self.loops.iter().rposition(|target| target.is_loop) else {
                    return unsupported("continue outside loop", statement.span);
                };
                let depth = self.loops[index].depth;
                let block = self.continue_target(index);
                self.unwind(statement.span, depth);
                self.jump(statement.span, block, Vec::new());
            }
            _ => return unsupported("statement", statement.span()),
        }
        Ok(())
    }

    fn continue_target(&mut self, index: usize) -> Block {
        if let Some(block) = self.loops[index].continue_target {
            return block;
        }
        let block = self.block();
        self.loops[index].continue_target = Some(block);
        block
    }

    fn if_statement(&mut self, statement: &IfStatement<'_>) -> Result<()> {
        let span = statement.span;
        let condition = self.expression(&statement.test)?;
        let then = self.block();
        let Some(alternate) = &statement.alternate else {
            let join = self.block();
            self.branch(span, condition, Target::new(then), Target::new(join));
            self.switch(then);
            self.statement(&statement.consequent)?;
            if self.current.is_some() {
                self.jump(span, join, Vec::new());
            }
            self.switch(join);
            return Ok(());
        };
        let otherwise = self.block();
        self.branch(span, condition, Target::new(then), Target::new(otherwise));
        self.switch(then);
        self.statement(&statement.consequent)?;
        let then_end = self.current.take();
        self.switch(otherwise);
        self.statement(alternate)?;
        let otherwise_end = self.current.take();
        if then_end.is_none() && otherwise_end.is_none() {
            return Ok(());
        }
        let join = self.block();
        for end in [then_end, otherwise_end].into_iter().flatten() {
            self.switch(end);
            self.jump(span, join, Vec::new());
        }
        self.switch(join);
        Ok(())
    }

    fn loop_body(&mut self, body: &Statement<'_>, loop_: Loop) -> Result<Option<Block>> {
        self.loops.push(loop_);
        let result = self.statement(body);
        if result.is_ok() && self.current.is_some() {
            let target = self.continue_target(self.loops.len() - 1);
            self.jump(body.span(), target, Vec::new());
        }
        let loop_ = self.loops.pop().unwrap();
        result.map(|()| loop_.continue_target)
    }

    fn for_statement(&mut self, statement: &ForStatement<'_>) -> Result<()> {
        let span = statement.span;
        let scope = match &statement.init {
            Some(ForStatementInit::VariableDeclaration(declaration)) => {
                let scope = if declaration.kind == VariableDeclarationKind::Var {
                    None
                } else {
                    self.open_scope(span, ScopeKind::Loop, declaration_bindings(declaration)?)?
                };
                self.initialize(declaration)?;
                scope
            }
            Some(init) => {
                let Some(init) = init.as_expression() else {
                    return unsupported("for initializer", span);
                };
                self.expression(init)?;
                None
            }
            None => None,
        };
        let head = self.block();
        let body = self.block();
        let done = self.block();
        if let Some(scope) = scope {
            self.effect(span, Op::NextIteration(scope));
        }
        self.jump(span, head, Vec::new());
        self.switch(head);
        if let Some(test) = &statement.test {
            let condition = self.expression(test)?;
            self.branch(span, condition, Target::new(body), Target::new(done));
        } else {
            self.jump(span, body, Vec::new());
        }
        self.switch(body);
        let update = self.loop_body(
            &statement.body,
            Loop {
                continue_target: None,
                break_target: done,
                depth: self.open.len(),
                is_loop: true,
            },
        )?;
        if let Some(update) = update {
            self.switch(update);
            if let Some(scope) = scope {
                self.effect(span, Op::NextIteration(scope));
            }
            if let Some(expression) = &statement.update {
                self.expression(expression)?;
            }
            self.jump(span, head, Vec::new());
        }
        self.switch(done);
        self.close_scope(span, scope);
        Ok(())
    }

    fn for_of_statement(&mut self, statement: &ForOfStatement<'_>) -> Result<()> {
        let span = statement.span;
        if statement.r#await {
            return unsupported("for await", span);
        }
        let ForStatementLeft::VariableDeclaration(declaration) = &statement.left else {
            return unsupported("for-of assignment target", span);
        };
        let [declarator] = declaration.declarations.as_slice() else {
            return unsupported("for-of declaration", span);
        };
        if matches!(&declarator.id, BindingPattern::ObjectPattern(_)) {
            return unsupported("for-of binding pattern", declarator.span);
        }
        pattern_bindings(&declarator.id)?;
        let (state, position) = self.hidden_binding(span, "%position");
        let scope = if declaration.kind == VariableDeclarationKind::Var {
            None
        } else {
            self.open_scope(span, ScopeKind::Loop, declaration_bindings(declaration)?)?
        };
        let iterable = self.expression(&statement.right)?;
        // Arrays and typed arrays walk their elements by index; other iterables keep the
        // iterator protocol. Both paths share one copy of the body.
        let indexed = self.emit(span, Op::IsIndexed(iterable));
        let zero = self.constant(span, Constant::Number(0.0));
        self.effect(span, Op::Initialize(position, zero));
        let direct = self.block();
        let generic = self.block();
        let entry = self.block();
        let iterator = self.param(entry);
        let head = self.block();
        self.branch(span, indexed, Target::new(direct), Target::new(generic));
        self.switch(direct);
        let undefined = self.constant(span, Constant::Undefined);
        self.jump(span, entry, vec![undefined]);
        self.switch(generic);
        let created = self.emit(span, Op::IteratorNew(iterable));
        self.jump(span, entry, vec![created]);
        self.switch(entry);
        self.jump(span, head, Vec::new());

        self.switch(head);
        let direct = self.block();
        let generic = self.block();
        let read = self.block();
        let body = self.block();
        let done = self.block();
        let item = self.param(body);
        self.branch(span, indexed, Target::new(direct), Target::new(generic));
        self.switch(direct);
        let index = self.emit(span, Op::Read(position));
        let length = self.emit(span, Op::IndexedLength(iterable));
        let inside = self.emit(span, Op::Binary(BinaryOperator::LessThan, index, length));
        self.branch(span, inside, Target::new(read), Target::new(done));
        self.switch(read);
        let element = self.emit(span, Op::ArrayGet(iterable, index));
        let one = self.constant(span, Constant::Number(1.0));
        let next = self.emit(span, Op::Binary(BinaryOperator::Addition, index, one));
        self.effect(span, Op::Write(position, next));
        self.jump(span, body, vec![element]);
        self.switch(generic);
        let next = self.emit(span, Op::IteratorNext(iterator));
        let finished = self.emit(span, Op::IsUninitialized(next));
        self.branch(
            span,
            finished,
            Target::new(done),
            Target {
                block: body,
                args: vec![next],
            },
        );
        self.switch(body);
        if let Some(scope) = scope {
            self.effect(span, Op::FreshIteration(scope));
        }
        self.bind_pattern(&declarator.id, item)?;
        self.loop_body(
            &statement.body,
            Loop {
                continue_target: Some(head),
                break_target: done,
                depth: self.open.len(),
                is_loop: true,
            },
        )?;
        self.switch(done);
        self.close_scope(span, scope);
        self.close_scope(span, Some(state));
        Ok(())
    }

    fn hidden_binding(&mut self, span: Span, name: &str) -> (Scope, Binding) {
        let scope = self.graph.scopes.push(ScopeData {
            kind: ScopeKind::Block,
            parent: self.open.last().copied(),
            bindings: Vec::new(),
        });
        let binding = self.graph.bindings.push(BindingData {
            persistent: false,
            name: name.to_owned(),
            kind: BindingKind::Let,
            scope,
            captured: false,
        });
        self.graph.scopes[scope].bindings.push(binding);
        self.effect(span, Op::EnterScope(scope));
        self.open.push(scope);
        (scope, binding)
    }

    fn class_declaration(&mut self, class: &oxc_ast::ast::Class<'_>) -> Result<()> {
        use oxc_ast::ast::{ClassElement, MethodDefinitionKind, PropertyKey};
        let Some(id) = &class.id else {
            return unsupported("anonymous class declaration", class.span);
        };
        let symbol = id.symbol_id.get().unwrap();
        let outer = self.symbols.get(&symbol).copied();
        let scope = self.open_scope(class.span, ScopeKind::Block, vec![(id, BindingKind::Let)])?;
        let inner = self.symbols[&symbol];
        let constructor = class
            .body
            .body
            .iter()
            .find_map(|element| match element {
                ClassElement::MethodDefinition(method)
                    if method.kind == MethodDefinitionKind::Constructor =>
                {
                    Some(method.value.span)
                }
                _ => None,
            })
            .unwrap_or(class.span);
        let closure = self.closure(constructor, false, false)?;
        let constructor = self.emit(class.span, closure);
        self.effect(class.span, Op::Initialize(inner, constructor));
        self.effect(class.span, Op::MarkClass(constructor));
        let prototype = self.emit(class.span, Op::Get(constructor, "prototype".into()));
        self.effect(
            class.span,
            Op::DefineHidden(prototype, "constructor".into(), constructor),
        );
        for element in &class.body.body {
            let ClassElement::MethodDefinition(method) = element else {
                return unsupported("class element", element.span());
            };
            if method.kind == MethodDefinitionKind::Constructor {
                continue;
            }
            let PropertyKey::StaticIdentifier(key) = &method.key else {
                return unsupported("class method key", method.span);
            };
            let closure = self.closure(method.value.span, false, false)?;
            let function = self.emit(method.span, closure);
            self.effect(method.span, Op::DisableConstructor(function));
            self.effect(
                method.span,
                Op::DefineHidden(prototype, key.name.to_string(), function),
            );
        }
        self.close_scope(class.span, scope);
        if let Some(outer) = outer {
            self.symbols.insert(symbol, outer);
            self.effect(class.span, Op::Initialize(outer, constructor));
        } else {
            self.symbols.remove(&symbol);
            self.effect(
                class.span,
                Op::GlobalWrite(id.name.to_string(), constructor),
            );
        }
        Ok(())
    }

    fn try_statement(&mut self, statement: &oxc_ast::ast::TryStatement<'_>) -> Result<()> {
        let span = statement.span;
        if statement.finalizer.is_some() {
            return unsupported("finally", span);
        }
        let Some(handler) = &statement.handler else {
            return unsupported("try without catch", span);
        };
        if handler
            .param
            .as_ref()
            .is_some_and(|param| !matches!(param.pattern, BindingPattern::BindingIdentifier(_)))
        {
            return unsupported("destructuring catch binding", handler.span);
        }
        self.graph.has_handlers = true;
        let catch = self.block();
        let done = self.block();
        let body = self.block();
        self.jump(span, body, Vec::new());
        self.switch(body);
        self.handlers.push((catch, self.open.len()));
        let scope = self.open_scope(
            statement.block.span,
            ScopeKind::Block,
            lexical_bindings(&statement.block.body)?,
        )?;
        self.statements(&statement.block.body)?;
        self.close_scope(statement.block.span, scope);
        self.handlers.pop();
        if self.current.is_some() {
            self.jump(span, done, Vec::new());
        }
        self.switch(catch);
        let exception = self.emit(handler.span, Op::Catch);
        let bindings = match &handler.param {
            Some(param) => pattern_bindings(&param.pattern)?
                .into_iter()
                .map(|id| (id, BindingKind::Let))
                .collect(),
            None => Vec::new(),
        };
        let scope = self.open_scope(handler.span, ScopeKind::Block, bindings)?;
        if let Some(param) = &handler.param {
            self.bind_pattern(&param.pattern, exception)?;
        }
        let body_scope = self.open_scope(
            handler.body.span,
            ScopeKind::Block,
            lexical_bindings(&handler.body.body)?,
        )?;
        self.statements(&handler.body.body)?;
        self.close_scope(handler.body.span, body_scope);
        self.close_scope(handler.span, scope);
        if self.current.is_some() {
            self.jump(span, done, Vec::new());
        }
        self.switch(done);
        Ok(())
    }

    fn switch_statement(&mut self, statement: &SwitchStatement<'_>) -> Result<()> {
        let span = statement.span;
        let discriminant = self.expression(&statement.discriminant)?;
        let mut bindings = Vec::new();
        for case in &statement.cases {
            bindings.extend(lexical_bindings(&case.consequent)?);
        }
        let scope = self.open_scope(span, ScopeKind::Block, bindings)?;
        for case in &statement.cases {
            for statement in &case.consequent {
                if let Statement::FunctionDeclaration(function) = statement {
                    let symbol = function.id.as_ref().unwrap().symbol_id.get().unwrap();
                    let binding = self.symbols[&symbol];
                    let closure = self.closure(function.span, false, function.generator)?;
                    let value = self.emit(function.span, closure);
                    self.effect(function.span, Op::Initialize(binding, value));
                }
            }
        }
        let done = self.block();
        let bodies: Vec<_> = statement.cases.iter().map(|_| self.block()).collect();
        let mut fallback = done;
        for (case, &body) in statement.cases.iter().zip(&bodies) {
            let Some(test) = &case.test else {
                fallback = body;
                continue;
            };
            let candidate = self.expression(test)?;
            let equal = self.emit(
                case.span,
                Op::Binary(
                    oxc_syntax::operator::BinaryOperator::StrictEquality,
                    discriminant,
                    candidate,
                ),
            );
            let next = self.block();
            self.branch(case.span, equal, Target::new(body), Target::new(next));
            self.switch(next);
        }
        self.jump(span, fallback, Vec::new());
        self.loops.push(Loop {
            continue_target: None,
            break_target: done,
            depth: self.open.len(),
            is_loop: false,
        });
        for (index, case) in statement.cases.iter().enumerate() {
            self.switch(bodies[index]);
            for statement in &case.consequent {
                if self.current.is_none() {
                    break;
                }
                if !matches!(statement, Statement::FunctionDeclaration(_)) {
                    self.statement(statement)?;
                }
            }
            if self.current.is_some() {
                self.jump(
                    case.span,
                    bodies.get(index + 1).copied().unwrap_or(done),
                    Vec::new(),
                );
            }
        }
        self.loops.pop();
        self.switch(done);
        self.close_scope(span, scope);
        Ok(())
    }

    fn do_while_statement(&mut self, statement: &DoWhileStatement<'_>) -> Result<()> {
        let span = statement.span;
        let body = self.block();
        let done = self.block();
        self.jump(span, body, Vec::new());
        self.switch(body);
        let test = self.loop_body(
            &statement.body,
            Loop {
                continue_target: None,
                break_target: done,
                depth: self.open.len(),
                is_loop: true,
            },
        )?;
        if let Some(test) = test {
            self.switch(test);
            let condition = self.expression(&statement.test)?;
            self.branch(span, condition, Target::new(body), Target::new(done));
        }
        self.switch(done);
        Ok(())
    }

    fn for_in_statement(&mut self, statement: &ForInStatement<'_>) -> Result<()> {
        let span = statement.span;
        let ForStatementLeft::VariableDeclaration(declaration) = &statement.left else {
            return unsupported("for-in assignment target", span);
        };
        let [declarator] = declaration.declarations.as_slice() else {
            return unsupported("for-in declaration", span);
        };
        if !matches!(&declarator.id, BindingPattern::BindingIdentifier(_)) {
            return unsupported("for-in destructuring", declarator.span);
        }
        let scope = if declaration.kind == VariableDeclarationKind::Var {
            None
        } else {
            self.open_scope(span, ScopeKind::Loop, declaration_bindings(declaration)?)?
        };
        let iterable = self.expression(&statement.right)?;
        let iterator = self.emit(span, Op::EnumerateNew(iterable));
        let head = self.block();
        let body = self.block();
        let done = self.block();
        self.jump(span, head, Vec::new());
        self.switch(head);
        let item = self.emit(span, Op::EnumerateNext(iterator));
        let finished = self.emit(span, Op::IsUninitialized(item));
        self.branch(span, finished, Target::new(done), Target::new(body));
        self.switch(body);
        if let Some(scope) = scope {
            self.effect(span, Op::FreshIteration(scope));
        }
        self.bind_pattern(&declarator.id, item)?;
        self.loop_body(
            &statement.body,
            Loop {
                continue_target: Some(head),
                break_target: done,
                depth: self.open.len(),
                is_loop: true,
            },
        )?;
        self.switch(done);
        self.close_scope(span, scope);
        Ok(())
    }

    fn while_statement(&mut self, statement: &WhileStatement<'_>) -> Result<()> {
        let span = statement.span;
        let head = self.block();
        let body = self.block();
        let done = self.block();
        self.jump(span, head, Vec::new());
        self.switch(head);
        let condition = self.expression(&statement.test)?;
        self.branch(span, condition, Target::new(body), Target::new(done));
        self.switch(body);
        self.loop_body(
            &statement.body,
            Loop {
                continue_target: Some(head),
                break_target: done,
                depth: self.open.len(),
                is_loop: true,
            },
        )?;
        self.switch(done);
        Ok(())
    }
}

#[cfg(test)]
pub(super) fn lower(
    function: &oxc_ast::ast::Function<'_>,
    scoping: &Scoping,
    captured: &FxHashSet<SymbolId>,
) -> Result<Graph> {
    if function.generator || function.r#async {
        return unsupported("function kind", function.span);
    }
    let Some(body) = &function.body else {
        return unsupported("function without body", function.span);
    };
    let mut graph = lower_parts(
        &function.params,
        &body.statements,
        None,
        function.span,
        scoping,
        captured,
        &[],
        &FxHashMap::default(),
        None,
        &FxHashMap::default(),
        false,
    )?;
    graph.name = function
        .id
        .as_ref()
        .map_or("<anonymous>".into(), |id| id.name.to_string());
    Ok(graph)
}

pub(super) fn lower_global(
    statements: &[Statement<'_>],
    span: Span,
    scoping: &Scoping,
    captured: &FxHashSet<SymbolId>,
    closures: &FxHashMap<u32, (cranelift_module::FuncId, Vec<SymbolId>, bool)>,
    arguments_references: &FxHashMap<u32, SymbolId>,
) -> Result<Graph> {
    let mut builder = Builder {
        scoping,
        captured,
        closures,
        arguments_references,
        graph: Graph {
            name: "<module>".into(),
            span,
            entry: Block::from_u32(0),
            has_handlers: false,
            guards: 0,
            views: FxHashMap::default(),
            property_accesses: FxHashMap::default(),
            borrowed: FxHashSet::default(),
            summaries: Default::default(),
            lazy_captures: FxHashSet::default(),
            cold: FxHashSet::default(),
            generator: false,
            has_assumes: false,
            entry_guards: Vec::new(),
            megamorphic_gets: Default::default(),
            arity: None,
            regions: PrimaryMap::new(),
            blocks: PrimaryMap::new(),
            insts: PrimaryMap::new(),
            values: PrimaryMap::new(),
            reprs: Default::default(),
            ownership: None,
            low_level_slots: Vec::new(),
            low_level_signature: None,
            bindings: PrimaryMap::new(),
            scopes: PrimaryMap::new(),
        },
        current: None,
        symbols: FxHashMap::default(),
        open: Vec::new(),
        loops: Vec::new(),
        handlers: Vec::new(),
    };
    let entry = builder.block();
    builder.switch(entry);
    for statement in statements {
        if let Statement::ClassDeclaration(class) = statement {
            let value = builder.constant(class.span, Constant::Uninitialized);
            builder.effect(
                class.span,
                Op::GlobalWrite(class.id.as_ref().unwrap().name.to_string(), value),
            );
        }
    }
    for statement in statements {
        match statement {
            Statement::ClassDeclaration(class) => builder.class_declaration(class)?,
            Statement::ExpressionStatement(expression) => {
                builder.expression(&expression.expression)?;
            }
            Statement::VariableDeclaration(declaration) => {
                for declarator in &declaration.declarations {
                    let BindingPattern::BindingIdentifier(id) = &declarator.id else {
                        return unsupported("destructuring declaration", declarator.span);
                    };
                    let value = match &declarator.init {
                        Some(expression) => builder.expression(expression)?,
                        None if declaration.kind != VariableDeclarationKind::Var => {
                            builder.constant(declarator.span, Constant::Undefined)
                        }
                        None => continue,
                    };
                    let op = if declaration.kind == VariableDeclarationKind::Var {
                        Op::GlobalWrite(id.name.to_string(), value)
                    } else {
                        Op::GlobalInitialize(id.name.to_string(), value)
                    };
                    builder.effect(declarator.span, op);
                }
            }
            Statement::FunctionDeclaration(_) | Statement::EmptyStatement(_) => {}
            other => builder.statement(other)?,
        }
    }
    let value = builder.constant(span, Constant::Undefined);
    builder.terminate(span, TerminatorKind::Return(value));
    builder.graph.verify()?;
    Ok(builder.graph)
}

pub(super) fn lower_parts(
    params: &FormalParameters<'_>,
    statements: &[Statement<'_>],
    expression: Option<&Expression<'_>>,
    span: Span,
    scoping: &Scoping,
    captured: &FxHashSet<SymbolId>,
    captures: &[SymbolId],
    closures: &FxHashMap<u32, (cranelift_module::FuncId, Vec<SymbolId>, bool)>,
    arguments_binding: Option<SymbolId>,
    arguments_references: &FxHashMap<u32, SymbolId>,
    generator: bool,
) -> Result<Graph> {
    let mut builder = Builder {
        scoping,
        captured,
        closures,
        arguments_references,
        graph: Graph {
            name: "<function>".into(),
            span,
            entry: Block::from_u32(0),
            has_handlers: false,
            guards: 0,
            views: FxHashMap::default(),
            property_accesses: FxHashMap::default(),
            borrowed: FxHashSet::default(),
            summaries: Default::default(),
            lazy_captures: FxHashSet::default(),
            cold: FxHashSet::default(),
            generator,
            has_assumes: false,
            entry_guards: Vec::new(),
            megamorphic_gets: Default::default(),
            arity: None,
            regions: PrimaryMap::new(),
            blocks: PrimaryMap::new(),
            insts: PrimaryMap::new(),
            values: PrimaryMap::new(),
            reprs: Default::default(),
            ownership: None,
            low_level_slots: Vec::new(),
            low_level_signature: None,
            bindings: PrimaryMap::new(),
            scopes: PrimaryMap::new(),
        },
        current: None,
        symbols: FxHashMap::default(),
        open: Vec::new(),
        loops: Vec::new(),
        handlers: Vec::new(),
    };
    let entry = builder.block();
    builder.switch(entry);
    let capture_scope = if captures.is_empty() {
        None
    } else {
        let scope = builder.graph.scopes.push(ScopeData {
            kind: ScopeKind::Captures,
            parent: None,
            bindings: Vec::new(),
        });
        for &symbol in captures {
            let binding = builder.graph.bindings.push(BindingData {
                persistent: false,
                name: if symbol.index() >= scoping.symbols_len() {
                    "arguments".into()
                } else {
                    scoping.symbol_name(symbol).to_string()
                },
                kind: if symbol.index() < scoping.symbols_len()
                    && scoping.symbol_flags(symbol).is_const_variable()
                {
                    BindingKind::Const
                } else {
                    BindingKind::Let
                },
                scope,
                captured: true,
            });
            builder.graph.scopes[scope].bindings.push(binding);
            builder.symbols.insert(symbol, binding);
        }
        builder.effect(span, Op::EnterScope(scope));
        builder.open.push(scope);
        Some(scope)
    };
    let arguments_scope = if let Some(symbol) = arguments_binding {
        let scope = builder.graph.scopes.push(ScopeData {
            kind: ScopeKind::Function,
            parent: builder.open.last().copied(),
            bindings: Vec::new(),
        });
        let binding = builder.graph.bindings.push(BindingData {
            persistent: false,
            name: "arguments".into(),
            kind: BindingKind::Var,
            scope,
            captured: captured.contains(&symbol),
        });
        builder.graph.scopes[scope].bindings.push(binding);
        builder.symbols.insert(symbol, binding);
        builder.effect(span, Op::EnterScope(scope));
        builder.open.push(scope);
        let value = builder.emit(span, Op::Arguments);
        builder.effect(span, Op::Initialize(binding, value));
        Some(scope)
    } else {
        None
    };
    let mut parameters = Vec::new();
    for parameter in &params.items {
        parameters.extend(
            pattern_bindings(&parameter.pattern)?
                .into_iter()
                .map(|id| (id, BindingKind::Parameter)),
        );
    }
    if let Some(rest) = &params.rest {
        parameters.extend(
            pattern_bindings(&rest.rest.argument)?
                .into_iter()
                .map(|id| (id, BindingKind::Parameter)),
        );
    }
    let parameter_scope = builder.open_scope(params.span, ScopeKind::Parameters, parameters)?;
    for (index, parameter) in params.items.iter().enumerate() {
        let span = parameter.span;
        let argument = builder.emit(span, Op::Argument(index));
        let value = match &parameter.initializer {
            Some(default) => {
                let missing = builder.emit(span, Op::IsUndefined(argument));
                let compute = builder.block();
                let join = builder.block();
                let value = builder.param(join);
                builder.branch(
                    span,
                    missing,
                    Target::new(compute),
                    Target {
                        block: join,
                        args: vec![argument],
                    },
                );
                builder.switch(compute);
                let computed = builder.expression(default)?;
                builder.jump(span, join, vec![computed]);
                builder.switch(join);
                value
            }
            None => argument,
        };
        builder.bind_pattern(&parameter.pattern, value)?;
    }
    if let Some(rest) = &params.rest {
        let value = builder.emit(rest.span, Op::Rest(params.items.len()));
        builder.bind_pattern(&rest.rest.argument, value)?;
    }
    let mut var_symbols = Vec::new();
    super::super::emit::scope::collect_var_symbols(statements, &mut var_symbols);
    var_symbols.sort_unstable();
    var_symbols.dedup();
    var_symbols.retain(|symbol| !builder.symbols.contains_key(symbol));
    let var_scope = if var_symbols.is_empty() {
        None
    } else {
        let scope = builder.graph.scopes.push(ScopeData {
            kind: ScopeKind::Function,
            parent: builder.open.last().copied(),
            bindings: Vec::new(),
        });
        for symbol in var_symbols {
            let binding = builder.graph.bindings.push(BindingData {
                persistent: false,
                name: scoping.symbol_name(symbol).to_string(),
                kind: BindingKind::Var,
                scope,
                captured: captured.contains(&symbol),
            });
            builder.graph.scopes[scope].bindings.push(binding);
            builder.symbols.insert(symbol, binding);
        }
        builder.effect(span, Op::EnterScope(scope));
        builder.open.push(scope);
        let undefined = builder.constant(span, Constant::Undefined);
        for binding in builder.graph.scopes[scope].bindings.clone() {
            builder.effect(span, Op::Initialize(binding, undefined));
        }
        Some(scope)
    };
    let body_scope = builder.open_scope(span, ScopeKind::Body, lexical_bindings(statements)?)?;
    if let Some(expression) = expression {
        let value = builder.expression(expression)?;
        builder.unwind(span, 0);
        builder.terminate(span, TerminatorKind::Return(value));
    } else {
        builder.statements(statements)?;
    }
    builder.close_scope(span, body_scope);
    builder.close_scope(span, var_scope);
    builder.close_scope(span, parameter_scope);
    builder.close_scope(span, arguments_scope);
    builder.close_scope(span, capture_scope);
    if builder.current.is_some() {
        let value = builder.constant(span, Constant::Undefined);
        builder.terminate(span, TerminatorKind::Return(value));
    }
    let dynamic = builder
        .graph
        .insts
        .values()
        .any(|data| matches!(data.op, Op::Arguments | Op::Rest(_)));
    if !generator && !dynamic {
        builder.graph.arity = u16::try_from(params.items.len()).ok();
    }
    builder.graph.verify()?;
    Ok(builder.graph)
}

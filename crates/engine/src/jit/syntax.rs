use oxc_ast::ast::{
    BindingPattern, Class, ClassElement, Expression, ForStatementInit, ForStatementLeft,
    MethodDefinitionKind, PropertyKey, Statement, VariableDeclaration, VariableDeclarationKind,
};
use oxc_span::GetSpan;
use oxc_syntax::symbol::SymbolId;

use crate::error::{Error, unsupported_at};

pub(in crate::jit) fn validate_class(class: &Class<'_>) -> Result<(), Error> {
    if class.heritage.is_some() || !class.decorators.is_empty() {
        return Err(unsupported_at(
            class.span,
            "class inheritance or decorators",
        ));
    }
    for element in &class.body.body {
        let ClassElement::MethodDefinition(method) = element else {
            return Err(unsupported_at(
                element.span(),
                "class fields or static blocks",
            ));
        };
        if method.computed
            || method.r#static
            || !method.decorators.is_empty()
            || !matches!(method.key, PropertyKey::StaticIdentifier(_))
            || !matches!(
                method.kind,
                MethodDefinitionKind::Constructor | MethodDefinitionKind::Method
            )
            || method.value.generator
            || method.value.r#async
        {
            return Err(unsupported_at(method.span, "unsupported class method"));
        }
    }
    Ok(())
}

pub(in crate::jit) fn statement_kind(statement: &Statement<'_>) -> &'static str {
    match statement {
        Statement::BlockStatement(_) => "block",
        Statement::BreakStatement(_) => "break",
        Statement::ContinueStatement(_) => "continue",
        Statement::DebuggerStatement(_) => "debugger",
        Statement::DoWhileStatement(_) => "do-while",
        Statement::EmptyStatement(_) => "empty",
        Statement::ExpressionStatement(_) => "expression",
        Statement::ForInStatement(_) => "for-in",
        Statement::ForOfStatement(_) => "for-of",
        Statement::ForStatement(_) => "for",
        Statement::IfStatement(_) => "if",
        Statement::LabeledStatement(_) => "labeled",
        Statement::ReturnStatement(_) => "return",
        Statement::SwitchStatement(_) => "switch",
        Statement::ThrowStatement(_) => "throw",
        Statement::TryStatement(_) => "try",
        Statement::WhileStatement(_) => "while",
        Statement::WithStatement(_) => "with",
        Statement::VariableDeclaration(_) => "variable declaration",
        Statement::FunctionDeclaration(_) => "function declaration",
        Statement::ClassDeclaration(_) => "class declaration",
        _ => "statement",
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

pub(in crate::jit) fn collect_var_symbols(
    statements: &[Statement<'_>],
    symbols: &mut Vec<SymbolId>,
) {
    for statement in statements {
        match statement {
            Statement::VariableDeclaration(declaration)
                if declaration.kind == VariableDeclarationKind::Var =>
            {
                collect_declaration_symbols(declaration, symbols);
            }
            Statement::BlockStatement(block) => collect_var_symbols(&block.body, symbols),
            Statement::SwitchStatement(statement) => {
                for case in &statement.cases {
                    collect_var_symbols(&case.consequent, symbols);
                }
            }
            Statement::IfStatement(statement) => {
                collect_var_symbols(std::slice::from_ref(&statement.consequent), symbols);
                if let Some(alternate) = &statement.alternate {
                    collect_var_symbols(std::slice::from_ref(alternate), symbols);
                }
            }
            Statement::DoWhileStatement(statement) => {
                collect_var_symbols(std::slice::from_ref(&statement.body), symbols);
            }
            Statement::WhileStatement(statement) => {
                collect_var_symbols(std::slice::from_ref(&statement.body), symbols);
            }
            Statement::TryStatement(statement) => {
                collect_var_symbols(&statement.block.body, symbols);
                if let Some(finalizer) = &statement.finalizer {
                    collect_var_symbols(&finalizer.body, symbols);
                }
                if let Some(handler) = &statement.handler {
                    collect_var_symbols(&handler.body.body, symbols);
                }
            }
            Statement::ForInStatement(statement) => {
                if let ForStatementLeft::VariableDeclaration(declaration) = &statement.left
                    && declaration.kind == VariableDeclarationKind::Var
                {
                    collect_declaration_symbols(declaration, symbols);
                }
                collect_var_symbols(std::slice::from_ref(&statement.body), symbols);
            }
            Statement::ForOfStatement(statement) => {
                if let ForStatementLeft::VariableDeclaration(declaration) = &statement.left
                    && declaration.kind == VariableDeclarationKind::Var
                {
                    collect_declaration_symbols(declaration, symbols);
                }
                collect_var_symbols(std::slice::from_ref(&statement.body), symbols);
            }
            Statement::ForStatement(statement) => {
                if let Some(ForStatementInit::VariableDeclaration(declaration)) = &statement.init
                    && declaration.kind == VariableDeclarationKind::Var
                {
                    collect_declaration_symbols(declaration, symbols);
                }
                collect_var_symbols(std::slice::from_ref(&statement.body), symbols);
            }
            _ => {}
        }
    }
}

fn collect_declaration_symbols(declaration: &VariableDeclaration<'_>, symbols: &mut Vec<SymbolId>) {
    for declarator in &declaration.declarations {
        match &declarator.id {
            BindingPattern::BindingIdentifier(identifier) => {
                symbols.push(identifier.symbol_id.get().unwrap())
            }
            BindingPattern::ArrayPattern(pattern) => {
                for element in pattern.elements.iter().flatten() {
                    if let BindingPattern::BindingIdentifier(identifier) = element {
                        symbols.push(identifier.symbol_id.get().unwrap());
                    }
                }
            }
            BindingPattern::ObjectPattern(pattern) => {
                for property in &pattern.properties {
                    if let BindingPattern::BindingIdentifier(identifier) = &property.value {
                        symbols.push(identifier.symbol_id.get().unwrap());
                    }
                }
            }
            _ => {}
        }
    }
}

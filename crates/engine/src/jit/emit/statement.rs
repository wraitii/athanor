use oxc_ast::ast::Statement;

pub(crate) fn statement_kind(statement: &Statement<'_>) -> &'static str {
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

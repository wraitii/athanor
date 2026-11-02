use oxc_allocator::Allocator;
use oxc_ast::ast::Program;
use oxc_parser::Parser;
use oxc_span::SourceType;

use crate::error::Error;

pub(crate) fn parse<'a>(
    allocator: &'a Allocator,
    source: &'a str,
    filename: &str,
) -> Result<Program<'a>, Error> {
    let source_type = if filename.ends_with(".mjs") {
        SourceType::mjs()
    } else {
        SourceType::script()
    };
    let parsed = Parser::new(allocator, source, source_type).parse();
    if parsed.diagnostics.has_errors() {
        let message = parsed
            .diagnostics
            .errors()
            .map(|error| format!("{error:?}"))
            .collect::<Vec<_>>()
            .join("\n");
        return Err(Error::Parse(message));
    }
    Ok(parsed.program)
}

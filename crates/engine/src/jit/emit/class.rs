use oxc_ast::ast::{Class, ClassElement, MethodDefinitionKind, PropertyKey};
use oxc_span::GetSpan;

use crate::error::{Error, unsupported_at};

pub(crate) fn validate_class(class: &Class<'_>) -> Result<(), Error> {
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

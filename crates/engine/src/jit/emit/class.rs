use cranelift_codegen::ir::Value as ClifValue;
use oxc_ast::ast::{Class, ClassElement, MethodDefinitionKind, PropertyKey};
use oxc_span::GetSpan;

use crate::error::{Error, unsupported_at};
use crate::jit::helpers::Helper;

use super::FunctionCompiler;

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

impl FunctionCompiler<'_, '_> {
    pub(super) fn emit_class(&mut self, class: &Class<'_>) -> Result<ClifValue, Error> {
        validate_class(class)?;
        self.scopes.push();
        let symbol = class.id.as_ref().unwrap().symbol_id.get().unwrap();
        let binding = self.declare_lexical_binding(symbol);
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
        let constructor = self.make_closure(constructor, false, false)?;
        self.initialize_local(binding, constructor);
        self.call_helper(Helper::MarkClass, &[constructor]);
        let prototype = self.property_access(constructor, "prototype", None);
        self.define_hidden_property(prototype, "constructor", constructor);
        for element in &class.body.body {
            let ClassElement::MethodDefinition(method) = element else {
                unreachable!()
            };
            if method.kind == MethodDefinitionKind::Constructor {
                continue;
            }
            let PropertyKey::StaticIdentifier(key) = &method.key else {
                unreachable!()
            };
            let function = self.make_closure(method.value.span, false, false)?;
            self.call_helper(Helper::DisableConstructor, &[function]);
            self.define_hidden_property(prototype, key.name.as_str(), function);
        }
        self.leave_block();
        Ok(constructor)
    }

    pub(super) fn emit_class_declaration(&mut self, class: &Class<'_>) -> Result<(), Error> {
        let identifier = class.id.as_ref().unwrap();
        let value = self.emit_class(class)?;
        if let Some(variable) = self.local(identifier.symbol_id.get().unwrap()) {
            self.initialize_local(variable, value);
        } else {
            self.global_access(identifier.name.as_str(), Some(value));
        }
        self.clear_temporaries();
        Ok(())
    }
}

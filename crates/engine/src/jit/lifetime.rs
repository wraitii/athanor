use cranelift_codegen::ir::{Function, InstructionData, Opcode, Value, ValueDef};

use super::numeric::Annotations;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ReadLifetime {
    Invocation,
    Frame,
}

pub(super) fn read_lifetime(
    function: &Function,
    annotations: &Annotations,
    value: Value,
) -> ReadLifetime {
    let mut value = value;
    for _ in 0..32 {
        value = function.dfg.resolve_aliases(value);
        if annotations.invocation_roots.contains(&value) {
            return ReadLifetime::Invocation;
        }
        let ValueDef::Result(inst, _) = function.dfg.value_def(value) else {
            return ReadLifetime::Frame;
        };
        match function.dfg.insts[inst] {
            InstructionData::BinaryImm64 {
                opcode: Opcode::IaddImm,
                arg,
                imm,
            } if imm.bits() == 0 => value = arg,
            _ => return ReadLifetime::Frame,
        }
    }
    ReadLifetime::Frame
}

#[cfg(test)]
mod tests {
    use super::*;
    use cranelift_codegen::ir::{AbiParam, InstBuilder, types};
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};

    #[test]
    fn invocation_roots_require_a_seed_and_identity_copies() {
        let mut function = Function::new();
        function
            .signature
            .params
            .extend([AbiParam::new(types::I64), AbiParam::new(types::I64)]);
        let mut context = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut function, &mut context);
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        let rooted = builder.block_params(entry)[0];
        let unknown = builder.block_params(entry)[1];
        let copy = builder.ins().iadd_imm(rooted, 0);
        let changed = builder.ins().iadd_imm(rooted, 1);
        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize();
        let mut annotations = Annotations::default();
        annotations.invocation_roots.insert(rooted);
        assert_eq!(
            read_lifetime(&function, &annotations, rooted),
            ReadLifetime::Invocation
        );
        assert_eq!(
            read_lifetime(&function, &annotations, copy),
            ReadLifetime::Invocation
        );
        assert_eq!(
            read_lifetime(&function, &annotations, changed),
            ReadLifetime::Frame
        );
        assert_eq!(
            read_lifetime(&function, &annotations, unknown),
            ReadLifetime::Frame
        );
    }
}

use cranelift_codegen::entity::SecondaryMap;
use cranelift_codegen::ir::{Block, Function, Inst, Value};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use super::helpers::Helper;
use super::phis::{self, Input};

pub(super) fn canonicalize_parameters(func: &mut Function) {
    let func = &mut func.stencil;
    func.dfg.resolve_all_aliases();
    let mut parameters = Vec::new();
    let mut indices = SecondaryMap::<Value, Option<usize>>::new();
    for block in func.layout.blocks() {
        if Some(block) == func.layout.entry_block() {
            continue;
        }
        for &value in func.dfg.block_params(block) {
            indices[value] = Some(parameters.len());
            parameters.push(value);
        }
    }
    if parameters.is_empty() {
        return;
    }
    let mut incoming = vec![Vec::new(); parameters.len()];
    for block in func.layout.blocks() {
        for inst in func.layout.block_insts(block) {
            for call in func.dfg.insts[inst]
                .branch_destination(&func.dfg.jump_tables, &func.dfg.exception_tables)
            {
                let destination = call.block(&func.dfg.value_lists);
                for (&parameter, argument) in func
                    .dfg
                    .block_params(destination)
                    .iter()
                    .zip(call.args(&func.dfg.value_lists))
                {
                    let Some(index) = indices[parameter] else {
                        continue;
                    };
                    let input = match argument.as_value() {
                        Some(value) => match indices[value] {
                            Some(source) => Input::Parameter(source),
                            None => Input::Value(value),
                        },
                        None => Input::Varying,
                    };
                    incoming[index].push(input);
                }
            }
        }
    }
    let replacements = phis::solve(&incoming);
    let mut removed = SecondaryMap::<Block, Vec<usize>>::new();
    for block in func.layout.blocks() {
        let params = func.dfg.block_params(block).to_vec();
        for (position, &value) in params.iter().enumerate().rev() {
            let Some(replacement) = indices[value].and_then(|index| replacements[index]) else {
                continue;
            };
            func.dfg.remove_block_param(value);
            func.dfg.change_to_alias(value, replacement);
            removed[block].push(position);
        }
    }
    for block in func.layout.blocks() {
        for inst in func.layout.block_insts(block) {
            for call in func.dfg.insts[inst]
                .branch_destination_mut(&mut func.dfg.jump_tables, &mut func.dfg.exception_tables)
            {
                let destination = call.block(&func.dfg.value_lists);
                for &position in &removed[destination] {
                    call.remove(position, &mut func.dfg.value_lists);
                }
            }
        }
    }
    func.dfg.resolve_all_aliases();
}

#[derive(Clone, Default)]
pub(super) struct Annotations {
    pub semantic_locations: bool,
    pub helpers: HashMap<Inst, Helper>,
    pub frame_stores: HashSet<Inst>,
}

#[derive(Clone)]
pub(super) struct FunctionBody {
    pub function: Function,
    pub baseline: Option<std::sync::Arc<super::machine::MachineCode>>,
    pub annotations: Annotations,
    pub listing: Option<Vec<super::semantic::ListingLine>>,
    pub inspection: super::inspect::Inspection,
}

#[cfg(test)]
mod tests {
    use cranelift_codegen::ir::{AbiParam, InstBuilder, types};
    use cranelift_codegen::{settings, verify_function};
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};

    use super::*;

    #[test]
    fn parameter_removal_preserves_loop_arguments_and_entry_signature() {
        let mut func = Function::new();
        func.signature.params.extend([
            AbiParam::new(types::I64),
            AbiParam::new(types::I64),
            AbiParam::new(types::I8),
        ]);
        func.signature.returns.push(AbiParam::new(types::I64));
        let mut context = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut func, &mut context);
        let entry = builder.create_block();
        let header = builder.create_block();
        let exit = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        for block in [header, exit] {
            for _ in 0..3 {
                builder.append_block_param(block, types::I64);
            }
        }
        builder.switch_to_block(entry);
        let inputs = builder.block_params(entry).to_vec();
        let zero = builder.ins().iconst(types::I64, 0);
        builder
            .ins()
            .jump(header, &[inputs[0].into(), zero.into(), inputs[1].into()]);
        builder.switch_to_block(header);
        let params = builder.block_params(header).to_vec();
        let next = builder.ins().iadd_imm_s(params[1], 1);
        builder.ins().brif(
            inputs[2],
            header,
            &[params[0].into(), next.into(), params[2].into()],
            exit,
            &[params[0].into(), params[1].into(), params[2].into()],
        );
        builder.switch_to_block(exit);
        let outputs = builder.block_params(exit).to_vec();
        let sum = builder.ins().iadd(outputs[0], outputs[1]);
        let first_sum = builder.func.layout.last_inst(exit).unwrap();
        let sum = builder.ins().iadd(sum, outputs[2]);
        let second_sum = builder.func.layout.last_inst(exit).unwrap();
        builder.ins().return_(&[sum]);
        builder.seal_all_blocks();
        drop(builder);
        let flags = settings::Flags::new(settings::builder());
        verify_function(&func, &flags).unwrap();

        canonicalize_parameters(&mut func);

        verify_function(&func, &flags).unwrap();
        assert_eq!(func.dfg.block_params(entry), inputs);
        assert_eq!(func.dfg.block_params(header), &[params[1]]);
        assert_eq!(func.dfg.block_params(exit), &[outputs[1]]);
        assert_eq!(func.dfg.inst_args(first_sum), &[inputs[0], outputs[1]]);
        assert_eq!(func.dfg.inst_args(second_sum)[1], inputs[1]);
        let before = func.display().to_string();
        canonicalize_parameters(&mut func);
        assert_eq!(func.display().to_string(), before);
    }
}

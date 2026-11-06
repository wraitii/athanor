use std::collections::VecDeque;

use cranelift_codegen::ir::{BlockArg, Function, Inst, InstBuilder};

use super::clif::Annotations;
use super::effects::{self, TemporaryEffect};

fn effect(function: &Function, annotations: &Annotations, inst: Inst) -> Option<bool> {
    if let Some(&helper) = annotations.helpers.get(&inst) {
        return match effects::temporary_effect(helper) {
            TemporaryEffect::Enter | TemporaryEffect::Clear => Some(false),
            TemporaryEffect::MayRegister => Some(true),
            TemporaryEffect::None => None,
        };
    }
    // Emission records inline pushes; other stores do not register temporaries.
    (annotations.temporary_pushes.contains(&inst) || function.dfg.insts[inst].opcode().is_call())
        .then_some(true)
}

pub(super) fn remove_empty_clears(function: &mut Function, annotations: &Annotations) -> usize {
    if annotations.temporary_clears.is_empty() {
        return 0;
    }
    let Some(entry) = function.layout.entry_block() else {
        return 0;
    };
    let mut summaries = vec![None; function.dfg.num_blocks()];
    for block in function.layout.blocks() {
        for inst in function.layout.block_insts(block) {
            if let Some(dirty) = effect(function, annotations, inst) {
                summaries[block.as_u32() as usize] = Some(dirty);
            }
        }
    }
    let mut entries = vec![None; summaries.len()];
    entries[entry.as_u32() as usize] = Some(true);
    let mut queue = VecDeque::from([entry]);
    let mut queued = vec![false; entries.len()];
    queued[entry.as_u32() as usize] = true;
    while let Some(block) = queue.pop_front() {
        let index = block.as_u32() as usize;
        queued[index] = false;
        let dirty = summaries[index].unwrap_or(entries[index].unwrap());
        let Some(terminator) = function.layout.last_inst(block) else {
            continue;
        };
        for destination in function.dfg.insts[terminator]
            .branch_destination(&function.dfg.jump_tables, &function.dfg.exception_tables)
        {
            let successor = destination.block(&function.dfg.value_lists);
            let index = successor.as_u32() as usize;
            // Bypassing the runtime clear proves the temporary stack is already empty.
            let incoming =
                dirty && annotations.temporary_clears.get(&terminator) != Some(&successor);
            let merged = entries[index].unwrap_or(false) || incoming;
            if entries[index] != Some(merged) {
                entries[index] = Some(merged);
                if !queued[index] {
                    queue.push_back(successor);
                    queued[index] = true;
                }
            }
        }
    }
    let mut remove = Vec::new();
    for block in function.layout.blocks() {
        let Some(mut dirty) = entries[block.as_u32() as usize] else {
            continue;
        };
        for inst in function.layout.block_insts(block) {
            if !dirty && let Some(&done) = annotations.temporary_clears.get(&inst) {
                remove.push((inst, done));
            }
            if let Some(next) = effect(function, annotations, inst) {
                dirty = next;
            }
        }
    }
    for &(inst, done) in &remove {
        // SSA construction may still pass variables to the continuation.
        let dfg = &function.dfg;
        let args: Vec<BlockArg> = dfg.insts[inst]
            .branch_destination(&dfg.jump_tables, &dfg.exception_tables)
            .iter()
            .find(|destination| destination.block(&dfg.value_lists) == done)
            .map(|destination| destination.args(&dfg.value_lists).collect())
            .unwrap_or_default();
        function.replace(inst).jump(done, &args);
    }
    remove.len()
}

#[cfg(test)]
mod tests {
    use cranelift_codegen::ir::{
        AbiParam, Block, ExtFuncData, ExternalName, FuncRef, MemFlagsData, Signature, types,
    };
    use cranelift_codegen::isa::CallConv;
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};

    use super::*;
    use crate::jit::helpers::Helper;

    fn call(b: &mut FunctionBuilder<'_>, annotations: &mut Annotations, helper: Option<Helper>) {
        let signature = b.import_signature(Signature::new(CallConv::SystemV));
        let callee: FuncRef = b.import_function(ExtFuncData {
            name: ExternalName::testcase("effect"),
            signature,
            colocated: false,
            patchable: false,
        });
        let inst = b.ins().call(callee, &[]);
        if let Some(helper) = helper {
            annotations.helpers.insert(inst, helper);
        }
    }

    fn clear(b: &mut FunctionBuilder<'_>, annotations: &mut Annotations) {
        let condition = b.block_params(b.func.layout.entry_block().unwrap())[0];
        let clear = b.create_block();
        let done = b.create_block();
        let branch = b.ins().brif(condition, clear, &[], done, &[]);
        annotations.temporary_clears.insert(branch, done);
        b.switch_to_block(clear);
        call(b, annotations, Some(Helper::TempClear));
        b.ins().jump(done, &[]);
        b.switch_to_block(done);
    }

    fn check(
        expected: usize,
        emit: impl FnOnce(&mut FunctionBuilder<'_>, &mut Annotations, Block),
    ) {
        let mut function = Function::new();
        function.signature.params = vec![AbiParam::new(types::I64)];
        let mut context = FunctionBuilderContext::new();
        let mut b = FunctionBuilder::new(&mut function, &mut context);
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        let mut annotations = Annotations::default();
        call(&mut b, &mut annotations, Some(Helper::FrameEnter));
        emit(&mut b, &mut annotations, entry);
        b.ins().return_(&[]);
        b.seal_all_blocks();
        let module = cranelift_jit::JITModule::new(
            cranelift_jit::JITBuilder::new(cranelift_module::default_libcall_names()).unwrap(),
        );
        b.finalize(cranelift_module::Module::target_config(&module));
        assert_eq!(remove_empty_clears(&mut function, &annotations), expected);
        cranelift_codegen::verify_function(
            &function,
            &cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder()),
        )
        .unwrap();
    }

    #[test]
    fn tracks_inline_pushes_and_calls_without_treating_other_stores_as_pushes() {
        for push in [false, true] {
            check(usize::from(!push), |b, annotations, entry| {
                let address = b.block_params(entry)[0];
                let value = b.ins().iconst(types::I64, 0);
                let store = b.ins().store(MemFlagsData::trusted(), value, address, 0);
                if push {
                    annotations.temporary_pushes.insert(store);
                }
                clear(b, annotations);
            });
        }
        for helper in [
            None,
            Some(Helper::Call),
            Some(Helper::TempRetain),
            Some(Helper::Add),
        ] {
            check(2, |b, annotations, _| {
                clear(b, annotations);
                call(b, annotations, helper);
                clear(b, annotations);
                clear(b, annotations);
            });
        }
    }

    #[test]
    fn dirty_join_preserves_clear() {
        check(0, |b, annotations, entry| {
            let condition = b.block_params(entry)[0];
            let dirty = b.create_block();
            let done = b.create_block();
            b.ins().brif(condition, dirty, &[], done, &[]);
            b.switch_to_block(dirty);
            call(b, annotations, Some(Helper::ObjectNew));
            b.ins().jump(done, &[]);
            b.switch_to_block(done);
            clear(b, annotations);
        });
    }

    #[test]
    fn dirty_backedge_preserves_clear_at_loop_header() {
        check(0, |b, annotations, entry| {
            let condition = b.block_params(entry)[0];
            let header = b.create_block();
            let body = b.create_block();
            let done = b.create_block();
            b.ins().jump(header, &[]);
            b.switch_to_block(header);
            clear(b, annotations);
            b.ins().brif(condition, body, &[], done, &[]);
            b.switch_to_block(body);
            call(b, annotations, None);
            b.ins().jump(header, &[]);
            b.switch_to_block(done);
        });
    }
}

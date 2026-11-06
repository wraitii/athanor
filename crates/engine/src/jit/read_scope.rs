use cranelift_codegen::Context;
use cranelift_codegen::cursor::{Cursor, FuncCursor};
use cranelift_codegen::ir::{
    BlockArg, InstBuilder, InstructionData, Opcode, condcodes::IntCC, types,
};
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Module};

use super::clif::{Annotations, IndexedRead};
use super::effects::{self, GLOBAL_WRITE, HEAP_WRITE, Reentry};
use super::helpers::Helper;
use crate::value::Value;

fn admitted(context: &Context, annotations: &Annotations) -> bool {
    let function = &context.func;
    for block in function.layout.blocks() {
        for inst in function.layout.block_insts(block) {
            if let Some(&helper) = annotations.helpers.get(&inst) {
                if matches!(
                    helper,
                    Helper::Catch
                        | Helper::Throw
                        | Helper::GeneratorNew
                        | Helper::GeneratorState
                        | Helper::GeneratorSuspend
                        | Helper::GeneratorComplete
                        | Helper::LocalGet
                ) {
                    return false;
                }
                let effect = effects::effects(helper);
                if effect.reentry != Reentry::Closed
                    || effect.flags & (HEAP_WRITE | GLOBAL_WRITE) != 0
                {
                    return false;
                }
            } else if function.dfg.insts[inst].opcode().is_call() {
                return false;
            } else if annotations.frame_stores.contains(&inst) {
                continue;
            } else if function.dfg.insts[inst].opcode().can_store() {
                match function.dfg.insts[inst] {
                    InstructionData::Store { args, .. } => {
                        let address = function.dfg.resolve_aliases(args[1]);
                        let cranelift_codegen::ir::ValueDef::Result(inst, _) =
                            function.dfg.value_def(address)
                        else {
                            return false;
                        };
                        if function.dfg.insts[inst].opcode() != Opcode::StackAddr {
                            return false;
                        }
                    }
                    _ => return false,
                }
            }
        }
    }
    true
}

pub(super) fn specialize(
    context: &mut Context,
    annotations: &mut Annotations,
    module: &mut JITModule,
    root_id: FuncId,
) {
    if !admitted(context, annotations) || annotations.indexed_reads.len() < 2 {
        return;
    }
    context.flowgraph();
    context.compute_loop_analysis();
    let function = &context.func;
    let mut pair: Option<(IndexedRead, IndexedRead)> = None;
    for &target in &annotations.indexed_reads {
        let Some(block) = function.layout.inst_block(target.call) else {
            continue;
        };
        let Some(lp) = context.loop_analysis.innermost_loop(block) else {
            continue;
        };
        for &source in &annotations.indexed_reads {
            if source.call == target.call || context.loop_analysis.is_in_loop(source.success, lp) {
                continue;
            }
            if context
                .domtree
                .dominates(source.success, block, &function.layout)
            {
                pair = Some((source, target));
                break;
            }
        }
        if pair.is_some() {
            break;
        }
    }
    let Some((source, target)) = pair else {
        return;
    };
    let function = &mut context.func;
    let Some(first) = function.layout.first_inst(source.success) else {
        return;
    };
    let stable = {
        let root = module.declare_func_in_func(root_id, function);
        let ctx = function.dfg.inst_args(source.call)[0];
        let mut cursor = FuncCursor::new(function).at_inst(first);
        cursor.set_srcloc(cranelift_codegen::ir::SourceLoc::new(source.source.0));
        let call = cursor
            .ins()
            .call(root, &[ctx, source.receiver, source.key, source.result]);
        annotations.helpers.insert(call, Helper::IndexRoot);
        cursor.func.dfg.first_result(call)
    };

    let target_block = function.layout.inst_block(target.call).unwrap();
    let next = function.layout.next_inst(target.call).unwrap();
    let slow = function.dfg.make_block();
    function.layout.split_block(slow, target.call);
    let done = function.dfg.make_block();
    function.layout.split_block(done, next);
    let merged = function.dfg.append_block_param(done, types::I64);
    let InstructionData::Call { func_ref, .. } = function.dfg.insts[target.call] else {
        unreachable!()
    };
    let args = function.dfg.inst_args(target.call).to_vec();
    let result = function.dfg.first_result(target.call);
    let loc = function.srcloc(target.call);
    function.dfg.clear_results(target.call);
    function.dfg.change_to_alias(result, merged);
    function.layout.remove_inst(target.call);
    annotations.helpers.remove(&target.call);
    annotations
        .indexed_reads
        .retain(|read| read.call != target.call);
    let mut cursor = FuncCursor::new(function).at_bottom(slow);
    cursor.set_srcloc(loc);
    let slow_call = cursor.ins().call(func_ref, &args);
    annotations.helpers.insert(slow_call, Helper::IndexGet);
    let slow_result = cursor.func.dfg.first_result(slow_call);
    cursor.ins().jump(done, &[BlockArg::Value(slow_result)]);
    let mut cursor = FuncCursor::new(cursor.func).at_bottom(target_block);
    cursor.set_srcloc(loc);
    let ready = cursor
        .ins()
        .icmp_imm_s(IntCC::Equal, stable, Value::TRUE.bits() as i64);
    let receiver = cursor
        .ins()
        .icmp(IntCC::Equal, source.receiver, target.receiver);
    let key = cursor.ins().icmp(IntCC::Equal, source.key, target.key);
    let same = cursor.ins().band(receiver, key);
    let reuse = cursor.ins().band(ready, same);
    cursor
        .ins()
        .brif(reuse, done, &[BlockArg::Value(source.result)], slow, &[]);
    context.flowgraph();
}

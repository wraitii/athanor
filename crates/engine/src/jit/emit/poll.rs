use cranelift_codegen::ir::{MemFlagsData, types};
use cranelift_jit::JITModule;
use cranelift_module::Module;

use crate::jit::helpers::{self, HELPERS, Helper};
use crate::jit::semantic::{Inst, Value, low_level::Builder};
use crate::jit::session::HELPERS_OFFSET;

pub(super) fn emit(
    b: &mut Builder<'_>,
    module: &JITModule,
    state: usize,
    ctx: Value,
    owner: Option<Value>,
) -> Option<Inst> {
    if state == 0 {
        return None;
    }
    let target = module.target_config();
    let pointer = target.pointer_type();
    let address = b.ins().iconst(pointer, state as i64);
    let flags = b
        .ins()
        .atomic_load(types::I8, MemFlagsData::trusted(), address);
    let mask = if owner.is_some() {
        crate::jit::recompile::REQUESTED
    } else {
        crate::jit::recompile::READY
    };
    let pending = b.ins().band_imm_u(flags, i64::from(mask));
    let slow = b.create_block();
    let done = b.create_block();
    b.set_cold_block(slow);
    b.ins().brif(pending, slow, &[], done, &[]);
    b.switch_to_block(slow);
    let helper = if owner.is_some() {
        Helper::CompileLeave
    } else {
        Helper::CompilePoll
    };
    let signature = b.import_signature(helpers::signature(
        HELPERS[helper as usize].shape,
        target.default_call_conv,
        pointer,
    ));
    let offset = HELPERS_OFFSET + helper as i32 * std::mem::size_of::<usize>() as i32;
    let address = b.ins().load(pointer, MemFlagsData::trusted(), ctx, offset);
    let args: Vec<_> = std::iter::once(ctx).chain(owner).collect();
    let call = b.ins().call_indirect(signature, address, &args);
    b.mark_helper(call, helper);
    b.ins().jump(done, &[]);
    b.switch_to_block(done);
    Some(call)
}

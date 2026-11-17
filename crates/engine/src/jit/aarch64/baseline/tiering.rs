use super::{
    Alu, Assembler, Cond, Helper, Mem, Width, call, call_address, check_error, e, stack_address,
};
use crate::jit::aarch64::asm::Label;
use crate::jit::baseline_tiering::Site;
use crate::jit::function_data::FunctionData;

pub(super) fn warmup(
    asm: &mut Assembler,
    function_data: &mut FunctionData,
    owner: u64,
    warmup: u64,
    context: u32,
) -> Option<()> {
    let counter = function_data.pin(std::cell::Cell::new(0u64));
    let done = asm.label();
    e::mov_imm(Width::X, 10, counter as u64, |word| asm.emit(word));
    asm.emit(e::mem_scaled(Mem::LdrX, 9, 10, 0)?);
    e::mov_imm(Width::X, 11, warmup, |word| asm.emit(word));
    asm.emit(e::alu(Alu::Subs, Width::X, e::ZR, 9, 11));
    asm.branch(Cond::Hs, done);
    asm.emit(e::add_imm(Width::X, 9, 9, 1, false));
    asm.emit(e::mem_scaled(Mem::StrX, 9, 10, 0)?);
    asm.emit(e::alu(Alu::Subs, Width::X, e::ZR, 9, 11));
    asm.branch(Cond::Ne, done);
    asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context)?);
    e::mov_imm(Width::X, 1, owner, |word| asm.emit(word));
    call(asm, Helper::BaselineWarm);
    asm.bind(done);
    Some(())
}

pub(super) fn poll(
    asm: &mut Assembler,
    site: *const Site,
    context: u32,
    receiver: u32,
    result: u32,
    failed: Label,
    resumed: Label,
) -> Option<()> {
    let slow = asm.label();
    let done = asm.label();
    e::mov_imm(Width::X, 10, site as u64, |word| asm.emit(word));
    asm.emit(e::mem_scaled(
        Mem::LdrX,
        9,
        10,
        crate::jit::baseline_tiering::ENTRIES_OFFSET as u32,
    )?);
    asm.emit(e::add_imm(Width::X, 9, 9, 1, false));
    asm.emit(e::mem_scaled(
        Mem::StrX,
        9,
        10,
        crate::jit::baseline_tiering::ENTRIES_OFFSET as u32,
    )?);
    asm.emit(e::mem_scaled(
        Mem::LdrX,
        11,
        10,
        crate::jit::baseline_tiering::ENTRY_OFFSET as u32,
    )?);
    asm.emit(e::alu(Alu::Subs, Width::X, e::ZR, 11, e::ZR));
    asm.branch(Cond::Ne, slow);
    asm.emit(e::logical_imm(
        e::Logical::Ands,
        Width::X,
        e::ZR,
        9,
        crate::jit::osr::TICK - 1,
    )?);
    asm.branch(Cond::Ne, done);
    asm.bind(slow);
    asm.emit(e::mem_scaled(Mem::LdrX, 0, e::SP, context)?);
    asm.emit(e::mov(Width::X, 1, 10));
    asm.emit(e::mov_sp(2, e::SP));
    asm.emit(e::mem_scaled(Mem::LdrX, 3, e::SP, receiver)?);
    stack_address(asm, 4, result);
    call_address(asm, crate::jit::baseline_tiering::poll as *const () as u64);
    check_error(asm, context, failed)?;
    asm.emit(e::alu(Alu::Subs, Width::X, e::ZR, 0, e::ZR));
    asm.branch(Cond::Ne, resumed);
    asm.bind(done);
    Some(())
}

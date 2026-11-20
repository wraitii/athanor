use crate::jit::module::CodeModule;
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{
    AbiParam, InstBuilder, MemFlagsData, Signature, StackSlotData, StackSlotKind, Type, types,
};
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{FuncId, Module};

use crate::error::Error;
use crate::heap::CODE_OFFSET;
use crate::jit::entries::{
    DIRECT_ADAPTER_OFFSET, DIRECT_ARITY_OFFSET, DIRECT_ENTRIES, DIRECT_ENTRY_SIZE,
};
use crate::jit::helpers::{self, Helper};
use crate::jit::jitdump::Writer;
use crate::jit::machine::MachineCode;
use crate::jit::session::{
    CALL_SAMPLE_CODE_OFFSET, CALL_SAMPLE_SIZE, CALL_SAMPLES, CALL_SAMPLES_LEN_OFFSET,
    CALL_SAMPLES_OFFSET, CALL_SITE_OFFSET, ERROR_OFFSET, EXCEPTION_OFFSET,
};
use crate::value::{TAG_OBJECT, TAG_SHIFT, Value};

pub(in crate::jit) const STUB_ARITIES: u16 = 8;

fn heap_address(
    b: &mut FunctionBuilder<'_>,
    value: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    let tag = b.ins().ushr_imm_s(value, TAG_SHIFT as i64);
    let offset = b.ins().iadd_imm_s(tag, -((TAG_OBJECT >> TAG_SHIFT) as i64));
    let heap = b.ins().icmp_imm_s(IntCC::UnsignedLessThan, offset, 2);
    let address = b.ins().band_imm_s(value, (1 << TAG_SHIFT) - 1);
    let zero = b.ins().iconst(types::I64, 0);
    b.ins().select(heap, address, zero)
}

#[derive(Clone, Copy)]
pub(in crate::jit) struct Stubs {
    pub(in crate::jit) direct_entries: *const usize,
    pub(in crate::jit) compilation_state: usize,
    misses: [usize; STUB_ARITIES as usize + 1],
    adapters: [usize; STUB_ARITIES as usize + 1],
}

impl Stubs {
    pub(in crate::jit) fn new(
        module: &mut CodeModule,
        generic: &Signature,
        direct_entries: *const usize,
        compilation_state: usize,
        call_profile: bool,
        mut dump: Option<&mut Writer>,
    ) -> Result<Self, Error> {
        let mut misses = Vec::new();
        let mut adapters = Vec::new();
        for arity in 0..=STUB_ARITIES {
            misses.push(build_miss(
                module,
                generic,
                direct_entries,
                compilation_state,
                call_profile,
                arity,
            )?);
            adapters.push(build_adapter(module, generic, arity)?);
        }
        module
            .finalize_definitions()
            .map_err(|error| Error::Compile(error.to_string()))?;
        let mut finalized = |kind: &str, stubs: Vec<Stub>| {
            let mut addresses = [0; STUB_ARITIES as usize + 1];
            for (arity, (address, stub)) in addresses.iter_mut().zip(stubs).enumerate() {
                *address = stub.finalize(module, dump.as_deref_mut(), &format!("{kind}/{arity}"));
            }
            addresses
        };
        Ok(Self {
            direct_entries,
            compilation_state,
            misses: finalized("miss", misses),
            adapters: finalized("adapter", adapters),
        })
    }

    pub(in crate::jit) fn miss(&self, arity: u16) -> Option<usize> {
        self.misses.get(arity as usize).copied()
    }

    pub(in crate::jit) fn adapter(&self, arity: u16) -> Option<usize> {
        self.adapters.get(arity as usize).copied()
    }
}

pub(in crate::jit) fn fixed_signature(pointer: Type, arity: u16) -> Signature {
    let mut signature = Signature::new(CallConv::Tail);
    signature.params.push(AbiParam::new(pointer));
    for _ in 0..arity + 2 {
        signature.params.push(AbiParam::new(types::I64));
    }
    signature.returns.push(AbiParam::new(types::I64));
    signature.returns.push(AbiParam::new(types::I8));
    signature
}

pub(super) fn poll(
    b: &mut FunctionBuilder<'_>,
    module: &CodeModule,
    state: usize,
    ctx: cranelift_codegen::ir::Value,
    owner: Option<cranelift_codegen::ir::Value>,
) -> Option<cranelift_codegen::ir::Inst> {
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
    let args: Vec<_> = owner.into_iter().collect();
    let call = helpers::call(b, target.default_call_conv, pointer, ctx, helper, &args);
    b.ins().jump(done, &[]);
    b.switch_to_block(done);
    Some(call)
}

pub(in crate::jit) struct Stub {
    id: FuncId,
    size: usize,
}

impl Stub {
    pub(in crate::jit) fn finalize(
        self,
        module: &CodeModule,
        dump: Option<&mut Writer>,
        name: &str,
    ) -> usize {
        let code = module.get_finalized_function(self.id);
        if let Some(dump) = dump {
            let bytes = unsafe { std::slice::from_raw_parts(code, self.size) };
            dump.record(&format!("stub {name}"), "", &[], bytes);
        }
        code as usize
    }
}

fn define(
    module: &mut CodeModule,
    signature: Signature,
    body: impl FnOnce(&mut FunctionBuilder<'_>, &mut CodeModule),
) -> Result<Stub, Error> {
    let id = module
        .declare_anonymous_function(&signature)
        .map_err(|error| Error::Compile(error.to_string()))?;
    let mut context = module.make_context();
    context.func.signature = signature;
    let mut builder_context = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut context.func, &mut builder_context);
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        body(&mut b, module);
        b.seal_all_blocks();
        b.finalize(module.target_config());
    }
    let machine = MachineCode::from_cranelift(&mut context, module, id)?;
    machine.define(module, id)?;
    Ok(Stub {
        id,
        size: machine.bytes.len(),
    })
}

/// Tail calls published code of this arity, and otherwise dispatches through published
/// adapters before trying the runtime. Returns an owned result, like fixed-arity code.
fn build_miss(
    module: &mut CodeModule,
    generic: &Signature,
    direct_entries: *const usize,
    compilation_state: usize,
    call_profile: bool,
    arity: u16,
) -> Result<Stub, Error> {
    let target = module.target_config();
    let pointer = target.pointer_type();
    define(module, fixed_signature(pointer, arity), |b, module| {
        let flags = MemFlagsData::trusted();
        let entry = b.current_block().unwrap();
        let params = b.block_params(entry).to_vec();
        let (ctx, callee, receiver) = (params[0], params[1], params[2]);
        poll(b, module, compilation_state, ctx, None);
        let inspect = b.create_block();
        let lookup = b.create_block();
        let direct = b.create_block();
        let dispatch = b.create_block();
        let fast = b.create_block();
        let slow = b.create_block();
        let done = b.create_block();
        b.append_block_param(dispatch, pointer);
        b.append_block_param(dispatch, pointer);
        b.append_block_param(done, types::I64);
        let none = b.ins().iconst(pointer, 0);
        let address = heap_address(b, callee);
        b.ins()
            .brif(address, inspect, &[], dispatch, &[none.into(), none.into()]);
        b.switch_to_block(inspect);
        let code = b.ins().load(types::I32, flags, address, CODE_OFFSET);
        let code = b.ins().uextend(pointer, code);
        if call_profile {
            record_call(b, target.default_call_conv, pointer, ctx, code);
        }
        let index = b.ins().iadd_imm_s(code, -1);
        let valid = b
            .ins()
            .icmp_imm_u(IntCC::UnsignedLessThan, index, DIRECT_ENTRIES as i64);
        b.ins()
            .brif(valid, lookup, &[], dispatch, &[none.into(), none.into()]);
        b.switch_to_block(lookup);
        let offset = b.ins().imul_imm_s(index, DIRECT_ENTRY_SIZE as i64);
        let table = b.ins().iconst(pointer, direct_entries as i64);
        let slot = b.ins().iadd(table, offset);
        let code = b.ins().load(pointer, flags, slot, 0);
        let adapter = b
            .ins()
            .load(pointer, flags, slot, DIRECT_ADAPTER_OFFSET as i32);
        let published = b
            .ins()
            .load(pointer, flags, slot, DIRECT_ARITY_OFFSET as i32);
        let matches = b
            .ins()
            .icmp_imm_s(IntCC::Equal, published, i64::from(arity) + 1);
        b.ins().brif(
            matches,
            direct,
            &[],
            dispatch,
            &[code.into(), adapter.into()],
        );
        b.switch_to_block(direct);
        let fixed = b.import_signature(fixed_signature(pointer, arity));
        b.ins().return_call_indirect(fixed, code, &params);
        b.switch_to_block(dispatch);
        let [code, adapter] = b.block_params(dispatch).try_into().unwrap();
        let argv = if arity == 0 {
            b.ins().iconst(pointer, 0)
        } else {
            let slot = b.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                u32::from(arity) * 8,
                3,
            ));
            for (index, &value) in params[3..].iter().enumerate() {
                let address = b.ins().stack_addr(pointer, slot, (index * 8) as i32);
                b.ins().store(flags, value, address, 0);
            }
            b.ins().stack_addr(pointer, slot, 0)
        };
        let argc = b.ins().iconst(types::I64, i64::from(arity));
        b.ins().brif(adapter, fast, &[], slow, &[]);
        b.switch_to_block(fast);
        let mut signature = generic.clone();
        signature.params.push(AbiParam::new(pointer));
        let signature = b.import_signature(signature);
        let adapted = b.ins().call_indirect(
            signature,
            adapter,
            &[ctx, callee, receiver, argv, argc, code],
        );
        let result = b.inst_results(adapted)[0];
        b.ins().jump(done, &[result.into()]);
        b.switch_to_block(slow);
        let call = helpers::call(
            b,
            target.default_call_conv,
            pointer,
            ctx,
            Helper::Call,
            &[callee, receiver, argv, argc],
        );
        let result = b.inst_results(call)[0];
        b.ins().jump(done, &[result.into()]);
        b.switch_to_block(done);
        let result = b.block_params(done)[0];
        let error = b.ins().load(types::I64, flags, ctx, ERROR_OFFSET);
        let error = b.ins().icmp_imm_s(IntCC::NotEqual, error, 0);
        let exception = b.ins().load(types::I64, flags, ctx, EXCEPTION_OFFSET);
        let exception = b.ins().icmp_imm_s(
            IntCC::NotEqual,
            exception,
            Value::UNINITIALIZED.bits() as i64,
        );
        let status = b.ins().bor(error, exception);
        b.ins().return_(&[result, status]);
    })
}

/// Appends the caller's site, stored in ctx before entering a miss stub, and the callee's
/// code word to the sample buffer, draining it into the planner when full.
fn record_call(
    b: &mut FunctionBuilder<'_>,
    call_conv: CallConv,
    pointer: Type,
    ctx: cranelift_codegen::ir::Value,
    code: cranelift_codegen::ir::Value,
) {
    let flags = MemFlagsData::trusted();
    let site = b.ins().load(pointer, flags, ctx, CALL_SITE_OFFSET);
    let len = b.ins().load(pointer, flags, ctx, CALL_SAMPLES_LEN_OFFSET);
    let offset = b.ins().imul_imm_s(len, CALL_SAMPLE_SIZE as i64);
    let sample = b.ins().iadd(ctx, offset);
    b.ins().store(flags, site, sample, CALL_SAMPLES_OFFSET);
    b.ins().store(
        flags,
        code,
        sample,
        CALL_SAMPLES_OFFSET + CALL_SAMPLE_CODE_OFFSET,
    );
    let len = b.ins().iadd_imm_s(len, 1);
    b.ins().store(flags, len, ctx, CALL_SAMPLES_LEN_OFFSET);
    let full = b.ins().icmp_imm_u(IntCC::Equal, len, CALL_SAMPLES as i64);
    let flush = b.create_block();
    let done = b.create_block();
    b.set_cold_block(flush);
    b.ins().brif(full, flush, &[], done, &[]);
    b.switch_to_block(flush);
    helpers::call(b, call_conv, pointer, ctx, Helper::CallProfileFlush, &[]);
    b.ins().jump(done, &[]);
    b.switch_to_block(done);
}

/// Lets generic callers enter fixed-arity code, passed as the last argument.
pub(in crate::jit) fn build_adapter(
    module: &mut CodeModule,
    generic: &Signature,
    arity: u16,
) -> Result<Stub, Error> {
    static UNDEFINED: u64 = Value::UNDEFINED.bits();
    let pointer = module.target_config().pointer_type();
    let mut signature = generic.clone();
    signature.params.push(AbiParam::new(pointer));
    define(module, signature, |b, _| {
        let entry = b.current_block().unwrap();
        let [ctx, closure, receiver, argv, argc, code] = b.block_params(entry).try_into().unwrap();
        let undefined = b.ins().iconst(pointer, &raw const UNDEFINED as i64);
        let mut args = vec![ctx, closure, receiver];
        for index in 0..arity {
            let present = b
                .ins()
                .icmp_imm_u(IntCC::UnsignedGreaterThan, argc, i64::from(index));
            let address = b.ins().iadd_imm_s(argv, i64::from(index) * 8);
            let address = b.ins().select(present, address, undefined);
            args.push(
                b.ins()
                    .load(types::I64, MemFlagsData::trusted(), address, 0),
            );
        }
        let fixed = b.import_signature(fixed_signature(pointer, arity));
        let call = b.ins().call_indirect(fixed, code, &args);
        let result = b.inst_results(call)[0];
        b.ins().return_(&[result]);
    })
}

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{
    AbiParam, InstBuilder, MemFlagsData, Signature, StackSlotData, StackSlotKind, Type, types,
};
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Module};

use super::scope::{heap_address, increment_references};
use crate::error::Error;
use crate::heap::CODE_OFFSET;
use crate::jit::entries::{DIRECT_ENTRIES, DIRECT_ENTRY_SIZE};
use crate::jit::helpers::Helper;
use crate::jit::jitdump::Writer;
use crate::jit::session::{ERROR_OFFSET, EXCEPTION_OFFSET};
use crate::value::Value;

pub(in crate::jit) const STUB_ARITIES: u16 = 8;

#[derive(Clone, Copy)]
pub(in crate::jit) struct Stubs {
    pub(in crate::jit) direct_entries: *const usize,
    misses: [usize; STUB_ARITIES as usize + 1],
    adapters: [usize; STUB_ARITIES as usize + 1],
}

impl Stubs {
    pub(in crate::jit) fn new(
        module: &mut JITModule,
        helpers: &[FuncId],
        generic: &Signature,
        direct_entries: *const usize,
        mut dump: Option<&mut Writer>,
    ) -> Result<Self, Error> {
        let mut misses = Vec::new();
        let mut adapters = Vec::new();
        for arity in 0..=STUB_ARITIES {
            misses.push(build_miss(module, helpers, generic, direct_entries, arity)?);
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

pub(in crate::jit) struct Stub {
    id: FuncId,
    size: usize,
}

impl Stub {
    pub(in crate::jit) fn finalize(
        self,
        module: &JITModule,
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
    module: &mut JITModule,
    signature: Signature,
    body: impl FnOnce(&mut FunctionBuilder<'_>, &mut JITModule),
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
    module
        .define_function(id, &mut context)
        .map_err(|error| Error::Compile(error.to_string()))?;
    let size = context.compiled_code().unwrap().code_buffer().len();
    Ok(Stub { id, size })
}

/// Dispatches unexpected targets through their published adapters before trying the runtime.
/// Returns an owned result, like fixed-arity code.
fn build_miss(
    module: &mut JITModule,
    helpers: &[FuncId],
    generic: &Signature,
    direct_entries: *const usize,
    arity: u16,
) -> Result<Stub, Error> {
    let pointer = module.target_config().pointer_type();
    define(module, fixed_signature(pointer, arity), |b, module| {
        let flags = MemFlagsData::trusted();
        let call = module.declare_func_in_func(helpers[Helper::Call as usize], b.func);
        let entry = b.current_block().unwrap();
        let params = b.block_params(entry).to_vec();
        let (ctx, callee, receiver) = (params[0], params[1], params[2]);
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
        let inspect = b.create_block();
        let lookup = b.create_block();
        let fast = b.create_block();
        let slow = b.create_block();
        let done = b.create_block();
        b.append_block_param(done, types::I64);
        let address = heap_address(b, callee);
        b.ins().brif(address, inspect, &[], slow, &[]);
        b.switch_to_block(inspect);
        let code = b.ins().load(types::I32, flags, address, CODE_OFFSET);
        let code = b.ins().uextend(pointer, code);
        let index = b.ins().iadd_imm_s(code, -1);
        let valid = b
            .ins()
            .icmp_imm_u(IntCC::UnsignedLessThan, index, DIRECT_ENTRIES as i64);
        b.ins().brif(valid, lookup, &[], slow, &[]);
        b.switch_to_block(lookup);
        let offset = b.ins().imul_imm_s(index, DIRECT_ENTRY_SIZE as i64);
        let table = b.ins().iconst(pointer, direct_entries as i64);
        let slot = b.ins().iadd(table, offset);
        let code = b.ins().load(pointer, flags, slot, 0);
        let adapter = b.ins().load(pointer, flags, slot, pointer.bytes() as i32);
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
        let call = b.ins().call(call, &[ctx, callee, receiver, argv, argc]);
        let result = b.inst_results(call)[0];
        let address = heap_address(b, result);
        let increment = b.create_block();
        b.ins()
            .brif(address, increment, &[], done, &[result.into()]);
        b.switch_to_block(increment);
        increment_references(b, address);
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

/// Lets generic callers enter fixed-arity code, passed as the last argument.
pub(in crate::jit) fn build_adapter(
    module: &mut JITModule,
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
            args.push(b.ins().load(types::I64, MemFlagsData::trusted(), address, 0));
        }
        let fixed = b.import_signature(fixed_signature(pointer, arity));
        let call = b.ins().call_indirect(fixed, code, &args);
        let result = b.inst_results(call)[0];
        b.ins().return_(&[result]);
    })
}

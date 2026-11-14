use super::{Block, BlockArg, GraphBuilder as FunctionBuilder, Inst};
use super::{FunctionCompiler, LoweredValue};
use crate::jit::helpers::Helper;
use crate::jit::semantic::Repr;
use crate::jit::semantic::low_level::{Reference, Scratch};
use crate::jit::semantic::runtime::layout;
use crate::value::Value;
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{MemFlagsData, types};

pub(super) fn is_heap(b: &mut FunctionBuilder<'_>, value: LoweredValue) -> LoweredValue {
    let value = b.as_word(value);
    layout::is_heap(b, value)
}

impl FunctionCompiler<'_, '_> {
    pub(super) fn increment(&mut self, value: LoweredValue) {
        self.b.reference(Reference::Retain, value);
    }

    pub(super) fn checked_temporary(&mut self, value: LoweredValue, retain: bool) -> LoweredValue {
        let word = self.b.as_word(value);
        let missing =
            self.b
                .ins()
                .icmp_imm_s(IntCC::Equal, word, Value::UNINITIALIZED.bits() as i64);
        let error = self.b.create_block();
        let valid = self.b.create_block();
        self.b.set_cold_block(error);
        self.brif(missing, error, &[], valid, &[]);
        self.switch(error);
        self.call_helper(Helper::Retain, &[value]);
        self.check_error();
        self.jump(valid, &[]);
        self.switch(valid);
        if !retain {
            return value;
        }
        let heap = self.is_heap(value);
        let push = self.b.create_block();
        let done = self.b.create_block();
        self.brif(heap, push, &[], done, &[]);
        self.switch(push);
        self.retain_temporary(value);
        self.jump(done, &[]);
        self.switch(done);
        value
    }

    /// Takes a reference to a value the block keeps until its end.
    pub(super) fn retain_temporary(&mut self, value: LoweredValue) -> LoweredValue {
        self.increment(value);
        self.track(value)
    }

    /// Keeps a reference the generated code already holds until the end of the block.
    pub(super) fn own_temporary(&mut self, value: LoweredValue) -> LoweredValue {
        self.track(value)
    }

    pub(super) fn track(&mut self, value: LoweredValue) -> LoweredValue {
        self.b.scratch(Scratch::Own, &[value]);
        value
    }

    pub(super) fn forward_references(&mut self, block: Block, args: &[BlockArg]) {
        if args.is_empty() {
            return;
        }
        let values: Vec<_> = args.iter().map(|&BlockArg::Value(value)| value).collect();
        self.b
            .scratch_owned(Scratch::Forward { target: block }, values);
    }

    pub(super) fn move_result_reference(&mut self, value: LoweredValue) {
        self.b.scratch(Scratch::MoveValue, &[value]);
    }

    pub(super) fn release_owned(&mut self) {
        self.b.scratch(Scratch::End, &[]);
        self.unchecked = None;
    }

    pub(super) fn release_tracked(&mut self, tokens: &[Inst]) {
        for &token in tokens {
            self.b.scratch(Scratch::ReleaseToken { token }, &[]);
        }
        self.unchecked = self.unchecked.filter(|token| !tokens.contains(token));
    }

    pub(super) fn unwind(&mut self, target: Block, args: &[BlockArg]) {
        self.b.scratch(
            Scratch::Unwind {
                exclude: self.unchecked,
            },
            &[],
        );
        self.b.ins().jump(target, args);
        self.terminated = true;
    }

    /// Loads a string literal from its cache in the function's data, creating it on the first
    /// execution. The cache owns the reference, so the value is never tracked.
    pub(super) fn string_constant(&mut self, value: &str) -> LoweredValue {
        let offset = self.data.push(0u64);
        let cache = self.data_address(offset);
        let cached = self
            .b
            .ins()
            .load(types::I64, MemFlagsData::trusted(), cache, 0);
        let create = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param_repr(done, Repr::Boxed);
        self.b.set_cold_block(create);
        let hit = self.b.as_boxed(cached);
        self.brif(cached, done, &[BlockArg::Value(hit)], create, &[]);
        self.switch(create);
        let text: Box<str> = value.into();
        let ptr = self.b.ins().iconst(self.pointer, text.as_ptr() as i64);
        let len = self.b.ins().iconst(types::I64, text.len() as i64);
        self.names.push(text);
        let created = self.call_helper(Helper::StringConstant, &[cache, ptr, len]);
        self.check_error();
        self.jump(done, &[BlockArg::Value(created)]);
        self.switch(done);
        self.b.block_params(done)[0]
    }

    pub(super) fn load_ctx(&mut self, offset: i32) -> LoweredValue {
        self.b
            .ins()
            .load(types::I64, MemFlagsData::trusted(), self.ctx, offset)
    }

    /// Stores a value whose reference the slot already owns, releasing the value it replaces
    /// when the slot may already hold one.
    pub(super) fn replace_local(
        &mut self,
        local: LoweredValue,
        value: LoweredValue,
        repeated: bool,
    ) {
        let (_, old) = self.store_local(local, value);
        if !repeated {
            return;
        }
        let heap = self.is_heap(old);
        let release = self.b.create_block();
        let done = self.b.create_block();
        self.brif(heap, release, &[], done, &[]);
        self.switch(release);
        self.release_inline(old, done, true);
        self.switch(done);
    }

    /// Empties a slot whose reference has moved elsewhere.
    pub(super) fn clear_local(&mut self, local: LoweredValue) {
        let undefined = self.constant(Value::UNDEFINED);
        self.store_local(local, undefined);
    }

    fn store_local(&mut self, local: LoweredValue, value: LoweredValue) -> (u32, LoweredValue) {
        let index = *self.slots.entry(local).or_insert_with(|| {
            self.slot_count += 1;
            self.slot_count - 1
        });
        let old = self.b.store_frame(self.ctx, value, index);
        (index, old)
    }

    /// Releases a heap-tagged value, calling the runtime only when its count would reach zero.
    pub(super) fn release_inline(&mut self, value: LoweredValue, done: Block, cold_drop: bool) {
        self.b.reference(
            Reference::Release {
                heap: true,
                cold: cold_drop,
            },
            value,
        );
        self.jump(done, &[]);
    }

    pub(super) fn is_heap(&mut self, value: LoweredValue) -> LoweredValue {
        is_heap(self.b, value)
    }
}

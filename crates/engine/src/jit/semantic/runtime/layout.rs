use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types;

use super::super::Value;
use super::super::cursor::Insert;
use crate::value::{TAG_FIRST, TAG_LAST, TAG_OBJECT, TAG_SHIFT, TAG_STRING};

const ADDRESS_BITS: u8 = TAG_SHIFT as u8;
const TAG_BITS: u8 = 64 - ADDRESS_BITS;
const OBJECT: u64 = TAG_OBJECT >> TAG_SHIFT;
const STRING: u64 = TAG_STRING >> TAG_SHIFT;
const _: () = assert!(TAG_STRING - TAG_OBJECT == 1 << TAG_SHIFT);

fn tag<I: Insert>(at: &mut I, word: Value) -> Value {
    at.ins().extract_bits(word, ADDRESS_BITS, TAG_BITS)
}

pub(in crate::jit) fn is_number<I: Insert>(at: &mut I, word: Value) -> Value {
    let tag = tag(at, word);
    at.ins().outside_range(tag, TAG_FIRST, TAG_LAST)
}

pub(in crate::jit) fn is_heap<I: Insert>(at: &mut I, word: Value) -> Value {
    let tag = tag(at, word);
    at.ins().in_range(tag, OBJECT, STRING)
}

pub(in crate::jit) fn is_object<I: Insert>(at: &mut I, word: Value) -> Value {
    let tag = tag(at, word);
    at.ins().in_range(tag, OBJECT, OBJECT)
}

pub(in crate::jit) fn is_string<I: Insert>(at: &mut I, word: Value) -> Value {
    let tag = tag(at, word);
    at.ins().in_range(tag, STRING, STRING)
}

pub(in crate::jit) fn address<I: Insert>(at: &mut I, word: Value) -> Value {
    at.ins().extract_bits(word, 0, ADDRESS_BITS)
}

pub(in crate::jit) fn heap_address<I: Insert>(at: &mut I, word: Value) -> Value {
    let heap = is_heap(at, word);
    let address = address(at, word);
    let zero = at.ins().iconst(types::I64, 0);
    at.ins().select(heap, address, zero)
}

pub(in crate::jit) fn object_view<I: Insert>(at: &mut I, word: Value) -> Value {
    let object = is_object(at, word);
    view(at, word, object)
}

pub(in crate::jit) fn heap_view<I: Insert>(at: &mut I, word: Value) -> Value {
    let heap = is_heap(at, word);
    view(at, word, heap)
}

fn view<I: Insert>(at: &mut I, word: Value, tagged: Value) -> Value {
    let address = address(at, word);
    let present = at.ins().icmp_imm_s(IntCC::NotEqual, address, 0);
    let valid = at.ins().band(tagged, present);
    let empty = at
        .ins()
        .iconst(types::I64, crate::heap::NO_ELEMENTS.as_ptr() as i64);
    at.ins().select(valid, address, empty)
}

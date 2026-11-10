use super::session::JitCtx;
use crate::heap::Heap;
use crate::value::Value;

// Generated code initializes one descriptor per producer and empties it on every exit.
// Only these internal operations accept its untagged address; it never reaches JS.
pub(super) struct StackPack {
    values: Vec<Value>,
}

// The owning JIT context must stay alive and at the same address until this guard drops.
// No pool borrow is held across calls: nested invocations check out distinct buffers.
pub(super) struct OwnedValues<'a> {
    context: *mut JitCtx<'a>,
    pub(super) values: Vec<Value>,
}

impl Drop for OwnedValues<'_> {
    fn drop(&mut self) {
        let context = unsafe { &mut *self.context };
        context.global.heap().release_all(&self.values);
        self.values.clear();
        if self.values.capacity() != 0 {
            context.pack_buffers.push(std::mem::take(&mut self.values));
        }
    }
}

pub(super) unsafe fn initialize(address: Value) {
    unsafe {
        (address.bits() as *mut StackPack).write(StackPack { values: Vec::new() });
    }
}

pub(super) unsafe fn store(ctx: *mut JitCtx, address: Value, values: Vec<Value>) -> Option<Value> {
    let heap = unsafe { &*ctx }.global.heap();
    if address == Value::UNDEFINED {
        return heap.array_from_owned(values);
    }
    let old =
        unsafe { std::mem::replace(&mut (*(address.bits() as *mut StackPack)).values, values) };
    drop(unsafe { owned(ctx, old) });
    Some(address)
}

pub(super) unsafe fn take<'a>(ctx: *mut JitCtx<'a>, pack: Value) -> OwnedValues<'a> {
    let heap = unsafe { &*ctx }.global.heap();
    let values = if pack.is_object() {
        heap.take_array_values(pack)
    } else {
        unsafe { std::mem::take(&mut (*(pack.bits() as *mut StackPack)).values) }
    };
    unsafe { owned(ctx, values) }
}

pub(super) unsafe fn buffer<'a>(ctx: *mut JitCtx<'a>) -> OwnedValues<'a> {
    let values = unsafe { &mut *ctx }.pack_buffers.pop().unwrap_or_default();
    unsafe { owned(ctx, values) }
}

unsafe fn owned<'a>(ctx: *mut JitCtx<'a>, values: Vec<Value>) -> OwnedValues<'a> {
    OwnedValues {
        context: ctx,
        values,
    }
}

pub(super) unsafe fn take_element(heap: &Heap, pack: Value, index: usize) -> Value {
    if pack.is_object() {
        return heap.take_array_element(pack, index);
    }
    let pack = unsafe { &mut *(pack.bits() as *mut StackPack) };
    pack.values
        .get_mut(index)
        .map(|value| std::mem::replace(value, Value::UNDEFINED))
        .unwrap_or(Value::UNDEFINED)
}

pub(super) unsafe fn append(heap: &Heap, pack: Value, values: &mut Vec<Value>) {
    if pack.is_object() {
        heap.append_array_owned(pack, values);
        return;
    }
    unsafe {
        (*(pack.bits() as *mut StackPack)).values.append(values);
    }
}

pub(super) unsafe fn push(heap: &Heap, pack: Value, value: Value) {
    if pack.is_object() {
        heap.push_array_owned(pack, value);
        return;
    }
    unsafe {
        (*(pack.bits() as *mut StackPack)).values.push(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stack_pack_transfers_ownership_and_releases_extracted_values_independently() {
        let engine = crate::Engine::new().unwrap();
        let mut ctx = JitCtx::new(&engine);
        let heap = ctx.global.heap().clone();
        let mut storage = std::mem::MaybeUninit::<StackPack>::uninit();
        let pack = Value::from_bits(storage.as_mut_ptr() as u64);
        unsafe {
            initialize(pack);
        }
        let child = heap.string("owned by pack").unwrap();
        let unused = heap.string("unused argument").unwrap();
        assert_eq!(
            unsafe { store(&mut ctx, pack, vec![child, unused]) },
            Some(pack)
        );
        assert!(!pack.is_object());
        let extracted = unsafe { take_element(&heap, pack, 0) };
        drop(unsafe { take(&mut ctx, pack) });
        heap.drain(1000);
        assert!(heap.valid(extracted));
        assert!(!heap.valid(unused));
        assert_eq!(unsafe { take_element(&heap, pack, 0) }, Value::UNDEFINED);
        heap.release(extracted);
        heap.drain(1000);
        assert!(!heap.valid(child));
        drop(unsafe { take(&mut ctx, pack) });
    }
}

#[cfg(test)]
mod pool_tests {
    use super::*;

    #[test]
    fn buffers_are_reused_only_after_the_owner_drops() {
        let engine = crate::Engine::new().unwrap();
        let mut context = JitCtx::new(&engine);
        let mut first = unsafe { buffer(&mut context) };
        first.values.reserve(4);
        let address = first.values.as_ptr();
        let child = context.global.heap().string("temporary argument").unwrap();
        first.values.push(child);
        let mut nested = unsafe { buffer(&mut context) };
        nested.values.reserve(4);
        assert_ne!(nested.values.as_ptr(), address);
        drop(first);
        context.global.heap().drain(1000);
        assert!(!context.global.heap().valid(child));
        let reused = unsafe { buffer(&mut context) };
        assert_eq!(reused.values.as_ptr(), address);
        assert!(reused.values.is_empty());
        drop(reused);
        drop(nested);
        assert_eq!(context.pack_buffers.len(), 2);
    }
}

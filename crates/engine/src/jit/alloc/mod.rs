use cranelift_codegen::entity::SecondaryMap;
use cranelift_codegen::ir::{Function, StackSlot, Type, Value, types};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum RegClass {
    Int,
    Float,
}

impl RegClass {
    pub(super) fn of(ty: Type) -> Option<Self> {
        match ty {
            types::I8 | types::I16 | types::I32 | types::I64 => Some(Self::Int),
            types::F32 | types::F64 => Some(Self::Float),
            _ => None,
        }
    }
}

const WORD: u32 = 8;

pub(super) struct Frame {
    values: SecondaryMap<Value, u32>,
    stack_slots: SecondaryMap<StackSlot, u32>,
    pub scratch: u32,
    pub size: u32,
}

impl Frame {
    pub(super) fn spill_all(func: &Function, scratch_words: u32, outgoing: u32) -> Self {
        let mut values = SecondaryMap::new();
        let mut size = outgoing.next_multiple_of(16);
        for block in func.layout.blocks() {
            for &param in func.dfg.block_params(block) {
                values[param] = size;
                size += WORD;
            }
            for inst in func.layout.block_insts(block) {
                for &result in func.dfg.inst_results(inst) {
                    values[result] = size;
                    size += WORD;
                }
            }
        }
        let mut stack_slots = SecondaryMap::new();
        for (slot, data) in func.sized_stack_slots.iter() {
            let align = 1u32 << data.align_shift;
            size = size.next_multiple_of(align.max(WORD));
            stack_slots[slot] = size;
            size += data.size;
        }
        size = size.next_multiple_of(WORD);
        let scratch = size;
        size += scratch_words * WORD;
        Self {
            values,
            stack_slots,
            scratch,
            size: size.next_multiple_of(16),
        }
    }

    pub(super) fn value(&self, value: Value) -> u32 {
        self.values[value]
    }

    pub(super) fn stack_slot(&self, slot: StackSlot) -> u32 {
        self.stack_slots[slot]
    }
}

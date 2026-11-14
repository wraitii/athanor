use cranelift_codegen::ir::{MemFlagsData, Type, condcodes::IntCC, types};
use cranelift_codegen::isa::CallConv;

use super::super::low_level::{Builder, Reference, SignatureRef};
use super::super::{Block, Cfg, Graph, Op, Value, low_level};
use super::layout;
use crate::FxHashMap;
use crate::heap::REFERENCES_OFFSET;
use crate::jit::helpers::{self, Helper};
use crate::jit::session::{HELPERS_OFFSET, LOCALS_OFFSET};

struct Lowering<'a, 'g> {
    b: &'a mut Builder<'g>,
    ctx: Value,
    pointer: Type,
    convention: CallConv,
    signatures: FxHashMap<Helper, SignatureRef>,
}

impl Lowering<'_, '_> {
    fn helper(&mut self, helper: Helper, value: Value) {
        let signature = *self.signatures.entry(helper).or_insert_with(|| {
            self.b
                .import_signature(helpers::signature(helper, self.convention, self.pointer))
        });
        let offset = HELPERS_OFFSET + helper as i32 * std::mem::size_of::<usize>() as i32;
        let callee = self
            .b
            .ins()
            .load(self.pointer, MemFlagsData::trusted(), self.ctx, offset);
        let call = self
            .b
            .ins()
            .call_indirect(signature, callee, &[self.ctx, value]);
        self.b.mark_helper(call, helper);
    }

    fn store_frame(&mut self, ctx: Value, value: Value, slot: u32) -> Value {
        let flags = MemFlagsData::trusted();
        let base = self.b.ins().load(types::I64, flags, ctx, LOCALS_OFFSET);
        let offset = slot as i32 * 8;
        let old = self.b.ins().load(types::I64, flags, base, offset);
        self.b.ins().store(flags, value, base, offset);
        old
    }

    fn heap(&mut self, value: Value) -> Value {
        layout::is_heap(self.b, value)
    }

    fn retain_address(&mut self, address: Value) {
        if crate::heap::strict_references() {
            self.helper(Helper::RetainAddress, address);
            return;
        }
        let flags = MemFlagsData::trusted();
        let references = self
            .b
            .ins()
            .load(types::I32, flags, address, REFERENCES_OFFSET);
        let references = self.b.ins().iadd_imm_s(references, 1);
        self.b
            .ins()
            .store(flags, references, address, REFERENCES_OFFSET);
    }

    fn retain(&mut self, value: Value) -> Block {
        let address = layout::heap_address(self.b, value);
        let increment = self.b.create_block();
        let done = self.b.create_block();
        self.b.ins().brif(address, increment, &[], done, &[]);
        self.b.switch_to_block(increment);
        self.retain_address(address);
        self.b.ins().jump(done, &[]);
        self.b.switch_to_block(done);
        done
    }

    fn release(&mut self, value: Value, heap: bool, cold: bool) -> Block {
        let done = self.b.create_block();
        if !heap {
            let is_heap = self.heap(value);
            let release = self.b.create_block();
            self.b.ins().brif(is_heap, release, &[], done, &[]);
            self.b.switch_to_block(release);
        }
        if crate::heap::strict_references() {
            self.helper(Helper::Release, value);
            self.b.ins().jump(done, &[]);
            self.b.switch_to_block(done);
            return done;
        }
        let address = layout::address(self.b, value);
        let load = self.b.create_block();
        let decrement = self.b.create_block();
        let last = self.b.create_block();
        self.b.ins().brif(address, load, &[], done, &[]);
        self.b.switch_to_block(load);
        let flags = MemFlagsData::trusted();
        let references = self
            .b
            .ins()
            .load(types::I32, flags, address, REFERENCES_OFFSET);
        let shared = self
            .b
            .ins()
            .icmp_imm_s(IntCC::UnsignedGreaterThan, references, 1);
        self.b.ins().brif(shared, decrement, &[], last, &[]);
        self.b.switch_to_block(decrement);
        let references = self.b.ins().iadd_imm_s(references, -1);
        self.b
            .ins()
            .store(flags, references, address, REFERENCES_OFFSET);
        self.b.ins().jump(done, &[]);
        self.b.switch_to_block(last);
        if cold {
            self.b.set_cold_block(last);
        }
        self.helper(Helper::Release, value);
        self.b.ins().jump(done, &[]);
        self.b.switch_to_block(done);
        done
    }
}

fn pure(op: &Op) -> bool {
    matches!(
        op,
        Op::LowLevel(
            low_level::Op::Integer { .. }
                | low_level::Op::Float64(_)
                | low_level::Op::Unary { .. }
                | low_level::Op::Binary { .. }
                | low_level::Op::IntCompare { .. }
                | low_level::Op::FloatCompare { .. }
                | low_level::Op::Select { .. }
                | low_level::Op::StackAddress { .. }
                | low_level::Op::StackPointer { .. }
        )
    )
}

fn cancel_pairs(graph: &mut Graph, blocks: &[Block]) {
    if crate::heap::strict_references() {
        return;
    }
    for &block in blocks {
        let insts = std::mem::take(&mut graph.blocks[block].insts);
        let mut retained = FxHashMap::<Value, Vec<usize>>::default();
        let mut kept = Vec::with_capacity(insts.len());
        for inst in insts {
            match graph.insts[inst].op {
                Op::LowLevel(low_level::Op::Reference {
                    op: Reference::Retain,
                    value,
                }) => {
                    retained.entry(value).or_default().push(kept.len());
                }
                Op::LowLevel(low_level::Op::Reference {
                    op: Reference::Release { .. },
                    value,
                }) => {
                    if let Some(positions) = retained.get_mut(&value)
                        && let Some(position) = positions.pop()
                    {
                        kept[position] = None;
                        continue;
                    }
                    retained.clear();
                }
                ref op if pure(op) => {}
                _ => retained.clear(),
            }
            kept.push(Some(inst));
        }
        graph.blocks[block].insts = kept.into_iter().flatten().collect();
    }
}

pub(in crate::jit) fn lower(graph: &mut Graph, ctx: Value, convention: CallConv, pointer: Type) {
    let blocks = Cfg::new(graph).rpo;
    cancel_pairs(graph, &blocks);
    let blocks: Vec<_> = blocks
        .into_iter()
        .filter(|&block| {
            graph.blocks[block].insts.iter().any(|&inst| {
                matches!(
                    graph.insts[inst].op,
                    Op::LowLevel(
                        low_level::Op::Reference { .. } | low_level::Op::FrameStore { .. }
                    )
                )
            })
        })
        .collect();
    let mut builder = Builder::new(graph);
    let mut lowering = Lowering {
        b: &mut builder,
        ctx,
        pointer,
        convention,
        signatures: FxHashMap::default(),
    };
    let mut aliases = FxHashMap::default();
    for block in blocks {
        let insts = std::mem::take(&mut lowering.b.graph.blocks[block].insts);
        let terminator = lowering.b.graph.blocks[block].terminator.take();
        let cold = lowering.b.graph.cold.contains(&block);
        let mut current = block;
        lowering.b.switch_to_block(current);
        for inst in insts {
            let data = &lowering.b.graph.insts[inst];
            let op = match &data.op {
                Op::LowLevel(
                    op @ (low_level::Op::Reference { .. } | low_level::Op::FrameStore { .. }),
                ) => op.clone(),
                _ => {
                    lowering.b.graph.insts[inst].block = current;
                    lowering.b.graph.blocks[current].insts.push(inst);
                    continue;
                }
            };
            let (span, source_name, location, result) = (
                data.span,
                data.source_name,
                data.emission_location,
                data.result(),
            );
            let created = lowering.b.num_blocks();
            lowering.b.set_span(span);
            lowering.b.set_source_name(source_name);
            lowering.b.set_location(location);
            match op {
                low_level::Op::Reference { op, value } => match op {
                    Reference::Retain => current = lowering.retain(value),
                    Reference::Release { heap, cold } => {
                        current = lowering.release(value, heap, cold)
                    }
                },
                low_level::Op::FrameStore { ctx, value, slot } => {
                    let old = lowering.store_frame(ctx, value, slot);
                    aliases.insert(result.unwrap(), old);
                }
                _ => unreachable!(),
            }
            if cold {
                for index in created..lowering.b.num_blocks() {
                    lowering.b.set_cold_block(Block::from_u32(index as u32));
                }
            }
        }
        lowering.b.graph.blocks[current].terminator = terminator;
    }
    lowering.b.graph.replace_uses(&aliases);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::semantic::variables::Variables;
    use cranelift_codegen::ir::{AbiParam, Signature};
    use oxc_span::Span;

    #[test]
    fn refcount_pairs_cross_only_pure_operations() {
        for barrier in 0..5 {
            let mut signature = Signature::new(CallConv::SystemV);
            signature.params.push(AbiParam::new(types::I64));
            let mut graph = Graph::new_low_level("rc".into(), Span::new(0, 0), signature);
            let entry = graph.entry;
            let mut b = Builder::new(&mut graph);
            b.switch_to_block(entry);
            b.append_block_params_for_function_params(entry);
            let value = b.block_params(entry)[0];
            b.reference(Reference::Retain, value);
            let one = b.ins().iconst(types::I64, 1);
            b.ins().iadd(value, one);
            match barrier {
                0 => {}
                1 => {
                    b.ins().load(types::I64, MemFlagsData::trusted(), value, 0);
                }
                2 => {
                    b.ins().store(MemFlagsData::trusted(), one, value, 0);
                }
                3 => {
                    b.store_frame(value, one, 0);
                }
                4 => {
                    let mut signature = Signature::new(CallConv::SystemV);
                    signature.params.push(AbiParam::new(types::I64));
                    let signature = b.import_signature(signature);
                    b.ins().call_indirect(signature, value, &[one]);
                }
                _ => unreachable!(),
            }
            b.reference(
                Reference::Release {
                    heap: false,
                    cold: true,
                },
                value,
            );
            b.ins().return_(&[]);
            Variables::default().finish(&mut graph).unwrap();
            cancel_pairs(&mut graph, &[entry]);
            let count = graph.blocks[entry]
                .insts
                .iter()
                .filter(|&&inst| {
                    matches!(
                        graph.insts[inst].op,
                        Op::LowLevel(low_level::Op::Reference { .. })
                    )
                })
                .count();
            assert_eq!(
                count,
                if barrier == 0 && !crate::heap::strict_references() {
                    0
                } else {
                    2
                }
            );
            graph.verify().unwrap();
        }
    }
}

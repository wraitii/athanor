use super::{Graph, Op};

pub(in crate::jit) fn lower_memory_for_cranelift(graph: &mut Graph) {
    use super::MemoryOp;

    for block in super::Cfg::new(graph).rpo {
        for inst in graph.blocks[block].insts.clone() {
            let data = graph.insts[inst].clone();
            if data.block != block {
                continue;
            }
            let Op::Memory(MemoryOp::IndexedLoad {
                base,
                index,
                format,
            }) = data.op
            else {
                continue;
            };
            let at = graph.blocks[block]
                .insts
                .iter()
                .position(|&i| i == inst)
                .unwrap();
            let (address_inst, address) = graph.insert(
                block,
                at,
                data.span,
                Op::Memory(MemoryOp::Address {
                    base,
                    index,
                    format,
                }),
                true,
            );
            graph.insts[address_inst].source_name = data.source_name;
            graph.insts[inst].op = Op::Memory(MemoryOp::Load {
                address: address.unwrap(),
                format,
            });
        }
    }
}

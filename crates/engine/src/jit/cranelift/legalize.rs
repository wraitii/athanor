use crate::jit::semantic::{Cfg, Graph, MemoryOp, Op, lower_machine};

pub(in crate::jit) fn legalize(graph: &mut Graph) -> Result<(), String> {
    lower_machine(graph)?;
    split_indexed_loads(graph);
    Ok(())
}

pub(in crate::jit) fn split_indexed_loads(graph: &mut Graph) {
    for block in Cfg::new(graph).rpo {
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
            graph.insts[address_inst].emission_location = data.emission_location;
            graph.reprs[address.unwrap()] = graph.reprs[base];
            graph.insts[inst].op = Op::Memory(MemoryOp::Load {
                address: address.unwrap(),
                format,
            });
        }
    }
}

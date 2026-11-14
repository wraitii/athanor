use crate::jit::semantic::{Cfg, Graph, MemoryOp, Op};

pub(in crate::jit) fn prepare(
    graph: &mut Graph,
    after: &mut dyn FnMut(&str, &Graph),
) -> Result<(), String> {
    crate::jit::semantic::legalize::convert(graph)?;
    after("legalize-convert", graph);
    crate::jit::semantic::bits::expand(graph);
    after("bits", graph);
    split_indexed_loads(graph);
    after("cranelift-memory", graph);
    graph.verify_machine()
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

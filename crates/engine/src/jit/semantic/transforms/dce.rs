use cranelift_codegen::entity::SecondaryMap;

use super::super::{Facts, Graph, Inst, Op, ValueDef};

pub(super) fn eliminate(graph: &mut Graph, facts: &Facts) {
    let mut uses = graph.use_counts();
    let dead = |inst: Inst, uses: &SecondaryMap<_, u32>| {
        facts.removable(inst)
            && graph.insts[inst]
                .results
                .iter()
                .all(|&value| uses[value] == 0)
    };
    let mut work: Vec<_> = facts
        .cfg
        .rpo
        .iter()
        .flat_map(|&block| graph.blocks[block].insts.iter().copied())
        .filter(|&inst| dead(inst, &uses))
        .collect();
    let mut removed = SecondaryMap::<Inst, bool>::new();
    let mut insts = Vec::new();
    while let Some(inst) = work.pop() {
        if removed[inst] {
            continue;
        }
        removed[inst] = true;
        insts.push(inst);
        let data = &graph.insts[inst];
        let operands = data
            .op
            .operands_iter()
            .filter(|_| !matches!(data.op, Op::Assume(..)))
            .chain(graph.views.get(&inst).copied())
            .chain(
                graph
                    .property_accesses
                    .get(&inst)
                    .and_then(|read| read.cache),
            );
        for value in operands {
            uses[value] -= 1;
            if uses[value] == 0
                && let ValueDef::Result(definition, _) = graph.values[value]
                && !removed[definition]
                && dead(definition, &uses)
            {
                work.push(definition);
            }
        }
    }
    graph.remove_insts(insts);
}

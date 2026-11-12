use crate::FxHashSet;

use super::{BlockData, Cfg, Graph, Inst, Op, Target, Terminator, TerminatorKind};

pub(super) fn assume(graph: &mut Graph, shapes: &super::shapes::Plan) -> Vec<Inst> {
    let guarded: FxHashSet<_> = graph
        .insts
        .values()
        .filter_map(|data| match data.op {
            Op::IsCode(value, _) | Op::IsNative(value, _) => Some(value),
            _ => None,
        })
        .collect();
    let reads: Vec<_> = Cfg::new(graph)
        .rpo
        .into_iter()
        .flat_map(|block| graph.blocks[block].insts.clone())
        .filter(|&inst| {
            matches!(
                graph.insts[inst].op,
                Op::Get(_, ref name)
                    if name != "length"
                        && !shapes.megamorphic(graph, inst)
                        && shapes.alternatives(graph, inst).is_none()
                        && graph.insts[inst]
                            .result()
                            .is_some_and(|value| guarded.contains(&value))
            )
        })
        .collect();
    for &inst in &reads {
        let data = graph.insts[inst].clone();
        let Op::Get(object, _) = data.op else {
            unreachable!();
        };
        let at = graph.blocks[data.block]
            .insts
            .iter()
            .position(|&i| i == inst)
            .unwrap();
        graph.insert(
            data.block,
            at,
            data.span,
            Op::Assume(
                object,
                super::guards::Fact::Cached(inst),
                super::guards::Origin {
                    site: None,
                    kind: super::guards::Kind::Receiver,
                    gain: 1,
                },
            ),
            false,
        );
    }
    reads
}

pub(super) fn lower(graph: &mut Graph, planned: &[Inst]) -> super::Result<()> {
    let sites: Vec<_> = planned
        .iter()
        .map(|&inst| {
            let data = &graph.insts[inst];
            (data.span, data.source_name, inst)
        })
        .collect();
    let reads: Vec<_> = Cfg::new(graph)
        .rpo
        .into_iter()
        .flat_map(|block| graph.blocks[block].insts.clone())
        .filter_map(|inst| {
            let data = &graph.insts[inst];
            if !matches!(data.op, Op::Get(..)) {
                return None;
            }
            sites
                .iter()
                .find(|&&(span, source, _)| span == data.span && source == data.source_name)
                .map(|&(_, _, site)| (inst, site))
        })
        .collect();
    for (inst, site) in reads {
        let data = graph.insts[inst].clone();
        let Op::Get(object, name) = data.op else {
            unreachable!();
        };
        let cfg = Cfg::new(graph);
        if super::guards::holds(
            graph,
            &cfg,
            data.block,
            object,
            super::guards::Fact::Cached(site),
        ) {
            graph.insts[inst].op =
                Op::CachedProperty(object, site, crate::heap::builtin_method(&name));
            continue;
        }
        let first_block = graph.blocks.len();
        let (block, continuation, fallback) = super::dispatch::split(graph, inst);
        graph.cold.remove(&fallback);
        graph.insts[inst].op = Op::FillPropertyCache(object, site, name.clone());
        let fast = graph.blocks.push(BlockData::default());
        let (check_inst, check) = graph.append(
            block,
            data.span,
            Op::IsPropertyCached(object, site, name.clone()),
            true,
        );
        graph.insts[check_inst].source_name = data.source_name;
        graph.blocks[block].terminator = Some(Terminator {
            source_name: data.source_name,
            emission_location: None,
            span: data.span,
            kind: TerminatorKind::Branch(check.unwrap(), Target::new(fast), Target::new(fallback)),
        });
        let (read, value) = graph.append(
            fast,
            data.span,
            Op::CachedProperty(object, site, crate::heap::builtin_method(&name)),
            true,
        );
        graph.insts[read].source_name = data.source_name;
        graph.blocks[fast].terminator = Some(Terminator {
            source_name: data.source_name,
            emission_location: None,
            span: data.span,
            kind: TerminatorKind::Jump(Target {
                block: continuation,
                args: vec![value.unwrap()],
            }),
        });
        let added: Vec<_> = graph.blocks.keys().skip(first_block).collect();
        for region in graph.regions.values_mut() {
            if region.blocks.contains(&block) {
                region.blocks.extend(added.iter().copied());
            }
        }
    }
    graph.verify()
}

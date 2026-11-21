use super::*;

pub(super) struct RecoveryPoint {
    pub(super) before: Option<Block>,
    pub(super) target: Target,
    pub(super) body: FxHashSet<Block>,
}

pub(super) struct AssumeCfg {
    pub(super) start: Block,
    pub(super) exit: Block,
    pub(super) fallback: Block,
    pub(super) recovery: Block,
    pub(super) before: Option<Block>,
    pub(super) blocks: FxHashSet<Block>,
    pub(super) scope: FxHashSet<Block>,
    pub(super) dependencies: Vec<Access>,
}

pub(super) trait Motion {
    fn dependencies(
        &self,
        graph: &Graph,
        facts: &Facts,
        group: &AssumeCfg,
        point: &RecoveryPoint,
    ) -> Option<Vec<Inst>>;
}

pub(super) fn hoist(
    graph: &mut Graph,
    groups: &mut [AssumeCfg],
    points: &[RecoveryPoint],
    bailouts: &super::versioning::Bailouts<'_>,
    values: &FxHashMap<Value, Value>,
    policy: &impl Motion,
    notes: &mut Vec<String>,
) -> bool {
    let mut changed = false;
    let mut facts: Option<Facts> = None;
    loop {
        let mut moved = false;
        for index in 0..groups.len() {
            for point in points {
                let group = &mut groups[index];
                // Rewriting can shrink a scope without moving its recovery boundary.
                if point.target.block == group.recovery
                    || point.body.len() <= group.scope.len()
                    || !point.body.is_superset(&group.scope)
                {
                    continue;
                }
                let Some(source) = group.before else {
                    continue;
                };
                if !matches!(&graph.blocks[source].terminator.as_ref().unwrap().kind,
                    TerminatorKind::Jump(target) if target.block == group.start)
                {
                    continue;
                }
                let destination = match point.before {
                    Some(before) => {
                        let TerminatorKind::Jump(target) =
                            &graph.blocks[before].terminator.as_ref().unwrap().kind
                        else {
                            continue;
                        };
                        target.clone()
                    }
                    None => Target::new(graph.entry),
                };
                if group.blocks.contains(&destination.block) {
                    continue;
                }
                let TerminatorKind::Jump(good) = graph.blocks[group.exit]
                    .terminator
                    .as_ref()
                    .unwrap()
                    .kind
                    .clone()
                else {
                    continue;
                };
                let known = facts.get_or_insert_with(|| Facts::compute(graph));
                if !known.cfg.rpo.contains(&source) {
                    continue;
                }
                let Some(dependencies) = policy.dependencies(graph, known, group, point) else {
                    continue;
                };
                let Some(recovery) =
                    bailouts.available(graph, &point.target, values, &known.cfg, point.before)
                else {
                    continue;
                };
                let failure = bailouts.target(graph, point.target.clone(), &recovery);
                for &block in &group.blocks {
                    for target in graph.blocks[block]
                        .terminator
                        .as_mut()
                        .unwrap()
                        .kind
                        .targets_mut()
                    {
                        if target.block == group.fallback {
                            *target = failure.clone();
                        }
                    }
                }
                graph.terminate(source, graph.span, TerminatorKind::Jump(good.clone()));
                graph.terminate(
                    group.exit,
                    graph.span,
                    TerminatorKind::Jump(destination.clone()),
                );
                match point.before {
                    Some(before) => graph.terminate(
                        before,
                        graph.span,
                        TerminatorKind::Jump(Target::new(group.start)),
                    ),
                    None => graph.entry = group.start,
                }
                for inst in dependencies.into_iter().rev() {
                    graph.detach_inst(inst);
                    graph.blocks[group.start].insts.insert(0, inst);
                    graph.insts[inst].block = group.start;
                }
                notes.push(format!(
                    "assumption CFG {} hoisted to generic {}",
                    group.start, point.target.block
                ));
                group.before = point.before;
                group.fallback = failure.block;
                group.recovery = point.target.block;
                group.scope = point.body.clone();
                let exit = group.exit;
                for other in groups.iter_mut() {
                    if other.before == Some(exit) && other.start == good.block {
                        other.before = Some(source);
                    } else if other.before == point.before && other.start == destination.block {
                        other.before = Some(exit);
                    }
                }
                facts = None;
                moved = true;
                changed = true;
                break;
            }
        }
        if !moved {
            return changed;
        }
    }
}

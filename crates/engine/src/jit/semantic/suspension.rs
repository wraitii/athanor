use super::{Binding, Block, Cfg, Graph, Op, Scope, Target, TerminatorKind, boundary};
use crate::jit::semantic::JsOp;
use crate::{FxHashMap, FxHashSet};

pub(in crate::jit) struct Plan {
    pub(in crate::jit) regions: Vec<Region>,
}

pub(in crate::jit) struct Region {
    pub(in crate::jit) graph: Graph,
    pub(in crate::jit) inputs: usize,
    pub(in crate::jit) outputs: usize,
}

struct Entry {
    target: Target,
    values: Vec<super::Value>,
    bindings: Vec<Binding>,
    scopes: Vec<Scope>,
}

impl Plan {
    pub(in crate::jit) fn lower(graph: &Graph) -> super::Result<Self> {
        graph.verify()?;
        let cfg = Cfg::new(graph);
        let live = boundary::liveness(graph, &cfg.rpo);
        let scopes = boundary::scopes(graph);
        let mut entries = vec![Entry {
            target: Target::new(graph.entry),
            values: Vec::new(),
            bindings: Vec::new(),
            scopes: Vec::new(),
        }];
        let mut exits = FxHashMap::default();
        for &block in &cfg.rpo {
            if let TerminatorKind::Yield(_, _, target) =
                &graph.blocks[block].terminator.as_ref().unwrap().kind
            {
                let stack = scopes[target.block].clone();
                let bindings = stack
                    .iter()
                    .flat_map(|&scope| graph.scopes[scope].bindings.iter().copied())
                    .collect();
                exits.insert(block, entries.len());
                entries.push(Entry {
                    target: target.clone(),
                    values: boundary::inputs(target, &live),
                    bindings,
                    scopes: stack,
                });
            }
        }
        let mut regions = Vec::new();
        for (index, entry) in entries.iter().enumerate() {
            let mut region = graph.clone();
            region.name = format!("{}.resume.{index}", graph.name);
            region.generator = false;
            for entry in &entries {
                for &binding in &entry.bindings {
                    region.bindings[binding].persistent = true;
                }
            }
            region.borrowed.clear();
            region.lazy_captures.clear();
            region.regions.clear();
            for &block in &cfg.rpo {
                match graph.blocks[block]
                    .terminator
                    .as_ref()
                    .unwrap()
                    .kind
                    .clone()
                {
                    TerminatorKind::Return(value) => {
                        boundary::store_exit(&mut region, block, vec![value], 0)
                    }
                    TerminatorKind::Yield(_, value, _) => {
                        let target = exits[&block];
                        let next = &entries[target];
                        let mut outputs = vec![value];
                        outputs.extend(&next.values);
                        for &binding in &next.bindings {
                            outputs.push(region.append_value(
                                block,
                                graph.span,
                                Op::Javascript(JsOp::BindingLoad(binding)),
                            ));
                        }
                        boundary::store_exit(&mut region, block, outputs, target);
                        for &scope in next.scopes.iter().rev() {
                            region.append(
                                block,
                                graph.span,
                                Op::Javascript(JsOp::LeaveScope(scope)),
                                false,
                            );
                        }
                    }
                    _ => {}
                }
            }
            if index != 0 {
                boundary::enter(
                    &mut region,
                    &entry.target,
                    &entry.values,
                    &entry.bindings,
                    &entry.scopes,
                    None,
                );
            } else {
                region.arity = None;
                let reachable: FxHashSet<Block> = Cfg::new(&region).rpo.into_iter().collect();
                for (block, data) in region.blocks.iter_mut() {
                    if !reachable.contains(&block) {
                        *data = Default::default();
                    }
                }
            }
            boundary::compact(&mut region);
            region.verify()?;
            let outputs = Cfg::new(&region)
                .rpo
                .iter()
                .flat_map(|&block| &region.blocks[block].insts)
                .filter_map(|&inst| match &region.insts[inst].op {
                    Op::ContinuationStore(values) => Some(values.len()),
                    _ => None,
                })
                .max()
                .unwrap_or(0);
            regions.push(Region {
                graph: region,
                inputs: entry.values.len() + entry.bindings.len(),
                outputs,
            });
        }
        Ok(Self { regions })
    }
}

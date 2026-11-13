use super::super::super::fresh::Fresh;
use super::*;
use crate::{FxHashMap, FxHashSet};

pub(in crate::jit::semantic) struct Writes<F = Facts> {
    writes: Vec<(Inst, Access)>,
    fresh: std::rc::Rc<Fresh>,
    pub(in crate::jit::semantic) facts: F,
}

impl Writes {
    #[cfg(test)]
    pub(in crate::jit::semantic) fn collect(graph: &Graph, blocks: &FxHashSet<Block>) -> Self {
        Self::collect_assuming(graph, blocks, &FxHashMap::default())
    }

    pub(in crate::jit::semantic) fn collect_assuming(
        graph: &Graph,
        blocks: &FxHashSet<Block>,
        assumed: &FxHashMap<Value, cranelift_module::FuncId>,
    ) -> Self {
        let facts = Facts::assuming(graph, &guards::assumed(&FxHashMap::default(), assumed));
        Self::with_facts(graph, blocks, facts)
    }
}

impl<F: std::borrow::Borrow<Facts>> Writes<F> {
    pub(in crate::jit::semantic) fn with_facts(
        graph: &Graph,
        blocks: &FxHashSet<Block>,
        facts: F,
    ) -> Self {
        let known = facts.borrow();
        let mut writes = Vec::new();
        let fresh = known.fresh(graph);
        for &block in blocks {
            for &inst in &graph.blocks[block].insts {
                let memory = known.memory(inst);
                if memory.writes_state() {
                    writes.push((
                        inst,
                        Access::written(&graph.insts[inst].op, known, memory.writes),
                    ));
                }
            }
        }
        Self {
            writes,
            fresh,
            facts,
        }
    }

    pub(in crate::jit::semantic) fn kills(
        &self,
        inst: Inst,
        write: &Access,
        read: &Access,
    ) -> bool {
        write.overlaps(read)
            && self.fresh.may_alias(inst, write, read)
            && write.field.conflicts(&read.field)
    }

    pub(in crate::jit::semantic) fn killer(&self, read: &Access) -> Option<Inst> {
        self.writes
            .iter()
            .find(|(inst, write)| self.kills(*inst, write, read))
            .map(|(inst, _)| *inst)
    }

    pub(in crate::jit::semantic) fn clobbers(&self, read: &Access) -> bool {
        self.writes
            .iter()
            .any(|(inst, write)| self.kills(*inst, write, read))
    }

    pub(in crate::jit::semantic) fn preserves(&self, graph: &Graph, inst: Inst) -> bool {
        let facts = self.facts.borrow();
        if let Some(read) = Access::read(&graph.insts[inst].op, facts) {
            return !self.clobbers(&read);
        }
        if !facts.removable(inst) || facts.memory(inst).reenters() {
            return false;
        }
        let read = Access {
            location: facts.memory(inst).reads,
            base: None,
            field: Field::Any,
        };
        !self.writes.iter().any(|(_, write)| write.overlaps(&read))
    }
}

pub(in crate::jit::semantic) fn guarded_accesses(
    requirements: &[Requirement],
    identities: &[(String, cranelift_module::FuncId)],
) -> Vec<Access> {
    let global = |name: &String| Access {
        location: Locations::HEAP.union(Locations::GLOBAL),
        base: None,
        field: Field::Named(name.clone()),
    };
    let property = |name: &String| Access {
        location: Locations::HEAP,
        base: None,
        field: Field::Named(name.clone()),
    };
    let mut guards: Vec<Access> = requirements
        .iter()
        .filter(|r| r.captured.is_none() && r.object.is_none())
        .map(|r| global(&r.global))
        .collect();
    guards.extend(identities.iter().map(|(name, _)| global(name)));
    guards.extend(
        requirements
            .iter()
            .filter(|r| r.captured.is_none())
            .map(|r| property(&r.property)),
    );
    guards
}

pub(in crate::jit::semantic) fn closure_failure(
    graph: &Graph,
    blocks: &FxHashSet<Block>,
    guards: &[Access],
    assumed: &FxHashMap<Value, cranelift_module::FuncId>,
) -> Option<String> {
    let facts = Facts::assuming(graph, &guards::assumed(&FxHashMap::default(), assumed));
    closure_failure_with(graph, blocks, guards, &facts)
}

pub(in crate::jit::semantic) fn closure_failure_with(
    graph: &Graph,
    blocks: &FxHashSet<Block>,
    guards: &[Access],
    facts: &Facts,
) -> Option<String> {
    let writes = Writes::with_facts(graph, blocks, facts);
    for &block in facts.cfg.rpo.iter().filter(|block| blocks.contains(block)) {
        for &inst in &graph.blocks[block].insts {
            let op = &graph.insts[inst].op;
            let memory = facts.memory(inst);
            if memory.reenters() {
                return Some(format!("reenters at {inst} ({})", graph.op_text(op)));
            }
            if !memory.writes_state() || guards.is_empty() {
                continue;
            }
            let write = Access::written(op, facts, memory.writes);
            if guards.iter().any(|guard| {
                (write.field == Field::Any || write.location.intersects(guard.location))
                    && writes.kills(inst, &write, guard)
            }) {
                return Some(format!(
                    "writes guarded state at {inst} ({})",
                    graph.op_text(op)
                ));
            }
        }
    }
    None
}

pub(in crate::jit::semantic) struct Conflicts {
    pub(in crate::jit::semantic) reentry: Option<String>,
    writes: Writes,
}

impl Conflicts {
    pub(in crate::jit::semantic) fn invalidates(&self, guards: &[Access]) -> bool {
        self.reentry.is_some()
            || self.writes.writes.iter().any(|(inst, write)| {
                guards.iter().any(|guard| {
                    (write.field == Field::Any || write.location.intersects(guard.location))
                        && self.writes.kills(*inst, write, guard)
                })
            })
    }
}

pub(in crate::jit::semantic) fn conflicts(
    graph: &Graph,
    blocks: &FxHashSet<Block>,
    assumed: &FxHashMap<Value, cranelift_module::FuncId>,
) -> Conflicts {
    let writes = Writes::collect_assuming(graph, blocks, assumed);
    let reentry = writes
        .facts
        .cfg
        .rpo
        .iter()
        .filter(|block| blocks.contains(block))
        .flat_map(|&block| graph.blocks[block].insts.iter())
        .find(|&&inst| writes.facts.memory(inst).reenters())
        .map(|&inst| {
            format!(
                "reenters at {inst} ({})",
                graph.op_text(&graph.insts[inst].op)
            )
        });
    Conflicts { reentry, writes }
}

pub(in crate::jit::semantic) struct Requirement {
    pub(in crate::jit::semantic) call: Inst,
    pub(in crate::jit::semantic) global: String,
    pub(in crate::jit::semantic) property: String,
    pub(in crate::jit::semantic) native: Native,
    pub(in crate::jit::semantic) captured: Option<Value>,
    pub(in crate::jit::semantic) object: Option<Value>,
}

pub(in crate::jit::semantic) fn requirements(graph: &Graph, blocks: &[Block]) -> Vec<Requirement> {
    let mut requirements = Vec::new();
    let inside: FxHashSet<Block> = blocks.iter().copied().collect();
    for &block in blocks {
        for &inst in &graph.blocks[block].insts {
            let Op::Call { callee, .. } = graph.insts[inst].op else {
                continue;
            };
            let Some(Op::Get(object, property)) = graph.defining_op(callee) else {
                continue;
            };
            let Some(Op::GlobalRead(global)) = graph.defining_op(*object) else {
                continue;
            };
            if let Some(native) = Native::lookup(global, property) {
                requirements.push(Requirement {
                    call: inst,
                    global: global.clone(),
                    property: property.clone(),
                    native,
                    captured: (!inside.contains(&graph.def_block(callee))).then_some(callee),
                    object: (!inside.contains(&graph.def_block(*object))).then_some(*object),
                });
            }
        }
    }
    requirements
}

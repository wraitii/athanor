use super::*;
use super::super::fresh::Fresh;

pub(super) struct Writes {
    writes: Vec<(Inst, Access)>,
    fresh: std::rc::Rc<Fresh>,
    pub(super) facts: Facts,
}

impl Writes {
    pub(super) fn collect(graph: &Graph, blocks: &HashSet<Block>) -> Self {
        Self::collect_assuming(graph, blocks, &HashMap::new())
    }

    pub(super) fn collect_assuming(
        graph: &Graph,
        blocks: &HashSet<Block>,
        assumed: &HashMap<Value, cranelift_module::FuncId>,
    ) -> Self {
        let facts = Facts::assuming(graph, &guards::assumed(&HashMap::new(), assumed));
        let mut writes = Vec::new();
        let fresh = facts.fresh(graph);
        for &block in blocks {
            for &inst in &graph.blocks[block].insts {
                let memory = facts.memory(inst);
                if memory.writes_data() {
                    writes.push((
                        inst,
                        Access::written(&graph.insts[inst].op, &facts, memory.writes),
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

    pub(super) fn kills(&self, inst: Inst, write: &Access, read: &Access) -> bool {
        self.fresh.may_alias(inst, write, read) && write.field.conflicts(&read.field)
    }

    pub(super) fn killer(&self, read: &Access) -> Option<Inst> {
        self.writes
            .iter()
            .find(|(inst, write)| self.kills(*inst, write, read))
            .map(|(inst, _)| *inst)
    }

    pub(super) fn clobbers(&self, read: &Access) -> bool {
        self.writes
            .iter()
            .any(|(inst, write)| self.kills(*inst, write, read))
    }

    pub(super) fn preserves(&self, op: &Op) -> bool {
        Access::read(op, &self.facts).is_some_and(|read| !self.clobbers(&read))
    }
}

pub(super) fn guarded_accesses(
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

pub(super) fn closure_failure(
    graph: &Graph,
    blocks: &HashSet<Block>,
    guards: &[Access],
    assumed: &HashMap<Value, cranelift_module::FuncId>,
) -> Option<String> {
    let writes = Writes::collect_assuming(graph, blocks, assumed);
    let facts = &writes.facts;
    let cfg = Cfg::new(graph);
    for &block in cfg.rpo.iter().filter(|block| blocks.contains(block)) {
        for &inst in &graph.blocks[block].insts {
            let op = &graph.insts[inst].op;
            let memory = facts.memory(inst);
            if memory.reenters() {
                return Some(format!("reenters at {inst} ({})", graph.op_text(op)));
            }
            if !memory.writes_data() || guards.is_empty() {
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

pub(super) struct Conflicts {
    pub(super) reentry: Option<String>,
    writes: Writes,
}

impl Conflicts {
    pub(super) fn invalidates(&self, guards: &[Access]) -> bool {
        self.reentry.is_some()
            || self.writes.writes.iter().any(|(inst, write)| {
                guards.iter().any(|guard| {
                    (write.field == Field::Any || write.location.intersects(guard.location))
                        && self.writes.kills(*inst, write, guard)
                })
            })
    }
}

pub(super) fn conflicts(
    graph: &Graph,
    blocks: &HashSet<Block>,
    assumed: &HashMap<Value, cranelift_module::FuncId>,
) -> Conflicts {
    let writes = Writes::collect_assuming(graph, blocks, assumed);
    let cfg = Cfg::new(graph);
    let reentry = cfg
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

pub(super) struct Requirement {
    pub(super) call: Inst,
    pub(super) global: String,
    pub(super) property: String,
    pub(super) native: Native,
    pub(super) captured: Option<Value>,
    pub(super) object: Option<Value>,
}

pub(super) fn requirements(graph: &Graph, blocks: &[Block]) -> Vec<Requirement> {
    let mut requirements = Vec::new();
    let inside: HashSet<Block> = blocks.iter().copied().collect();
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

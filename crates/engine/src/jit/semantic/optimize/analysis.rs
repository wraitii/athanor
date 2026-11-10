use super::*;

fn contained_use(op: &Op, index: usize) -> bool {
    match op {
        Op::Observe(..) | Op::ReadHint(..) => true,
        Op::Get(..)
        | Op::ShapedGet(..)
        | Op::ShapedSet(..)
        | Op::GetIndex(..)
        | Op::TypedIndex(..)
        | Op::IndexedGet(..)
        | Op::ArrayLoad(..)
        | Op::Set(..)
        | Op::SetIndex(..)
        | Op::Define(..)
        | Op::ArrayAppend(..) => index == 0,
        Op::IsUndefined(_)
        | Op::IsNullish(_)
        | Op::CheckInitialized(_)
        | Op::IsIndexed(_)
        | Op::IndexedLength(_) => true,
        Op::Binary(BinaryOperator::StrictEquality | BinaryOperator::StrictInequality, ..) => true,
        Op::IteratorNew(_) | Op::IteratorNext(_) | Op::EnumerateNew(_) | Op::EnumerateNext(_) => {
            true
        }
        _ => false,
    }
}

fn terminal_result(graph: &Graph, scope: &HashSet<Block>, value: Value) -> bool {
    let mut pending = vec![value];
    let mut visited = HashSet::new();
    while let Some(value) = pending.pop() {
        if !visited.insert(value) {
            continue;
        }
        for &block in scope {
            let data = &graph.blocks[block];
            if data.insts.iter().any(|&inst| {
                graph.insts[inst]
                    .op
                    .operands_iter()
                    .any(|operand| operand == value)
            }) {
                return false;
            }
            let Some(terminator) = &data.terminator else {
                continue;
            };
            match &terminator.kind {
                TerminatorKind::Branch(condition, ..) if *condition == value => return false,
                TerminatorKind::Yield(_, yielded, _) if *yielded == value => return false,
                _ => {}
            }
            for target in terminator.kind.targets() {
                for (index, &arg) in target.args.iter().enumerate() {
                    if arg != value {
                        continue;
                    }
                    let param = graph.blocks[target.block].params[index];
                    if scope.contains(&target.block) {
                        pending.push(param);
                    } else if !graph.blocks[target.block].insts.is_empty()
                        || !matches!(&graph.blocks[target.block].terminator, Some(Terminator { kind: TerminatorKind::Return(result), .. }) if *result == param)
                    {
                        return false;
                    }
                }
            }
        }
    }
    true
}

pub(super) fn unescaped(graph: &Graph, scope: &HashSet<Block>, cfg: &Cfg) -> HashMap<Value, Value> {
    let blocks: Vec<Block> = cfg
        .rpo
        .iter()
        .copied()
        .filter(|block| scope.contains(block))
        .collect();
    let mut allocations = HashSet::new();
    let mut aliases: HashMap<Value, Vec<Value>> = HashMap::new();
    let mut sources: HashMap<Value, Vec<Value>> = HashMap::new();
    for &block in &blocks {
        // Start optimistically so loop-carried aliases can survive; reject nonprivate inputs below.
        if block != graph.entry && cfg.preds[block].iter().all(|pred| scope.contains(pred)) {
            allocations.extend(graph.blocks[block].params.iter().copied());
        }
        for &inst in &graph.blocks[block].insts {
            let data = &graph.insts[inst];
            match data.op {
                Op::NewObject | Op::NewInstance(_) | Op::NewArray(_) | Op::NewHoleyArray(_) => {
                    allocations.extend(data.result)
                }
                Op::IteratorNew(source) | Op::EnumerateNew(source) => {
                    allocations.extend(data.result);
                    if let Some(iterator) = data.result {
                        sources.entry(iterator).or_default().push(source);
                    }
                }
                _ => {}
            }
        }
    }
    let mut escaped = Vec::new();
    for &block in &blocks {
        for &inst in &graph.blocks[block].insts {
            let op = &graph.insts[inst].op;
            for (index, value) in op.operands_iter().enumerate() {
                let terminal_constructor = index == 1
                    && allocations.contains(&value)
                    && matches!(op, Op::ConstructorResult(..))
                    && graph.insts[inst]
                        .result
                        .is_some_and(|result| terminal_result(graph, scope, result));
                if !contained_use(op, index) && !terminal_constructor {
                    escaped.push(value);
                }
            }
        }
        if let Some(terminator) = &graph.blocks[block].terminator {
            for target in terminator.kind.targets() {
                if scope.contains(&target.block) {
                    for (&value, &param) in
                        target.args.iter().zip(&graph.blocks[target.block].params)
                    {
                        aliases.entry(value).or_default().push(param);
                        aliases.entry(param).or_default().push(value);
                        if !allocations.contains(&value) {
                            escaped.push(param);
                        }
                        if !allocations.contains(&param) {
                            escaped.push(value);
                        }
                    }
                }
            }
        }
    }
    let mut visited = HashSet::new();
    while let Some(value) = escaped.pop() {
        if !visited.insert(value) {
            continue;
        }
        allocations.remove(&value);
        escaped.extend(aliases.get(&value).into_iter().flatten().copied());
        escaped.extend(sources.get(&value).into_iter().flatten().copied());
    }
    // Shared representatives keep writes through one alias visible to reads through another.
    let mut private = HashMap::new();
    for &value in &allocations {
        let mut pending = vec![value];
        while let Some(alias) = pending.pop() {
            if private.contains_key(&alias) {
                continue;
            }
            private.insert(alias, value);
            pending.extend(aliases.get(&alias).into_iter().flatten().copied());
        }
    }
    private
}

pub(super) struct Writes {
    writes: Vec<Access>,
    fresh: HashMap<Value, Value>,
    pub(super) facts: Facts,
}

impl Writes {
    pub(super) fn collect(graph: &Graph, blocks: &HashSet<Block>) -> Self {
        let facts = Facts::compute(graph);
        let mut writes = Vec::new();
        let fresh = unescaped(graph, blocks, &Cfg::new(graph));
        for &block in blocks {
            for &inst in &graph.blocks[block].insts {
                let memory = facts.memory(inst);
                if memory.writes_data() {
                    writes.push(Access::written(
                        &graph.insts[inst].op,
                        &facts,
                        memory.writes,
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

    fn aliases(&self, write: &Access, read: &Access) -> bool {
        match write.base {
            Some(target) if self.fresh.contains_key(&target) => {
                self.fresh.get(&target) == read.base.and_then(|base| self.fresh.get(&base))
            }
            _ => true,
        }
    }

    pub(super) fn clobbers(&self, read: &Access) -> bool {
        self.writes
            .iter()
            .any(|write| self.aliases(write, read) && write.field.conflicts(&read.field))
    }

    pub(super) fn preserves(&self, op: &Op) -> bool {
        Access::read(op, &self.facts).is_some_and(|read| !self.clobbers(&read))
    }
}

#[derive(Default)]
pub(super) struct Guarded {
    globals: HashSet<String>,
    properties: HashSet<String>,
}

impl Guarded {
    pub(super) fn new(requirements: &[Requirement]) -> Self {
        Self {
            globals: requirements
                .iter()
                .filter(|r| r.captured.is_none() && r.object.is_none())
                .map(|r| r.global.clone())
                .collect(),
            properties: requirements
                .iter()
                .filter(|r| r.captured.is_none())
                .map(|r| r.property.clone())
                .collect(),
        }
    }

    pub(super) fn with_code(mut self, identities: &[(String, cranelift_module::FuncId)]) -> Self {
        self.globals
            .extend(identities.iter().map(|(name, _)| name.clone()));
        self
    }

    fn is_empty(&self) -> bool {
        self.globals.is_empty() && self.properties.is_empty()
    }

    fn breaks(&self, write: &Access) -> bool {
        match &write.field {
            Field::Named(name) => {
                self.globals.contains(name)
                    || write.location.intersects(Locations::HEAP) && self.properties.contains(name)
            }
            Field::Cells | Field::Packs | Field::Elements | Field::Iteration => false,
            Field::Length | Field::Any => !self.is_empty(),
        }
    }
}

pub(super) fn closure_failure(
    graph: &Graph,
    blocks: &HashSet<Block>,
    guarded: &Guarded,
) -> Option<String> {
    let facts = Facts::compute(graph);
    let cfg = Cfg::new(graph);
    let fresh = unescaped(graph, blocks, &cfg);
    for &block in cfg.rpo.iter().filter(|block| blocks.contains(block)) {
        for &inst in &graph.blocks[block].insts {
            let op = &graph.insts[inst].op;
            let memory = facts.memory(inst);
            if memory.reenters() {
                return Some(format!("reenters at {inst} ({})", graph.op_text(op)));
            }
            let private = op
                .write_target()
                .is_some_and(|target| fresh.contains_key(&target));
            if memory.writes_data()
                && !private
                && guarded.breaks(&Access::written(op, &facts, memory.writes))
            {
                return Some(format!(
                    "writes guarded state at {inst} ({})",
                    graph.op_text(op)
                ));
            }
        }
    }
    None
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

use super::*;

fn contained_use(op: &Op, index: usize) -> bool {
    match op {
        Op::Get(..)
        | Op::GetIndex(..)
        | Op::TypedIndex(..)
        | Op::IndexedGet(..)
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

#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) enum Access {
    Named(String),
    // Cells can alias other captures, but cannot alias JavaScript properties.
    Cells,
    Packs,
    Length,
    Elements,
    Iteration,
    Any,
}

impl Access {
    fn named(name: &str) -> Self {
        match name {
            "length" => Self::Length,
            _ if name.starts_with(|c: char| c.is_ascii_digit() || c == '-')
                || matches!(name, "NaN" | "Infinity") =>
            {
                Self::Any
            }
            _ => Self::Named(name.to_owned()),
        }
    }

    pub(super) fn of(op: &Op, facts: &Facts) -> Self {
        match op {
            Op::Read(_) | Op::Initialize(..) | Op::Write(..) => Self::Cells,
            Op::Get(_, name) | Op::Set(_, name, _) | Op::Define(_, name, _) => Self::named(name),
            Op::GetIndex(_, key)
            | Op::TypedIndex(_, key, _)
            | Op::IndexedGet(_, key)
            | Op::SetIndex(_, key, _)
                if facts.of(*key).within(Types::NUMBER) =>
            {
                Self::Elements
            }
            Op::ArrayAppend(..) | Op::MapElement(..) => Self::Elements,
            Op::PackAppend(..) | Op::PackRelease(_) => Self::Packs,
            Op::IteratorNext(_) | Op::EnumerateNext(_) => Self::Iteration,
            _ => Self::Any,
        }
    }

    fn written(op: &Op, facts: &Facts) -> Self {
        match Self::of(op, facts) {
            Self::Length => Self::Any,
            access => access,
        }
    }

    fn conflicts(&self, read: &Self) -> bool {
        match (self, read) {
            (Self::Iteration, _) | (_, Self::Iteration) => false,
            (Self::Any, _) | (_, Self::Any) => true,
            (Self::Cells, Self::Cells) | (Self::Packs, Self::Packs) => true,
            (Self::Named(write), Self::Named(read)) => write == read,
            (Self::Elements, Self::Elements | Self::Length) => true,
            _ => false,
        }
    }
}

pub(super) struct Writes {
    targets: Vec<(Option<Value>, Access)>,
    globals: Vec<Access>,
    fresh: HashMap<Value, Value>,
    pub(super) facts: Facts,
}

impl Writes {
    pub(super) fn collect(graph: &Graph, blocks: &HashSet<Block>) -> Self {
        let facts = Facts::compute(graph);
        let mut targets = Vec::new();
        let mut globals = Vec::new();
        let fresh = unescaped(graph, blocks, &Cfg::new(graph));
        for &block in blocks {
            for &inst in &graph.blocks[block].insts {
                let op = &graph.insts[inst].op;
                let contract = graph.inst_contract(inst, &facts);
                if contract.flags & HEAP_WRITE != 0 {
                    targets.push((op.write_target(), Access::written(op, &facts)));
                }
                if contract.flags & GLOBAL_WRITE != 0 {
                    let access = match op {
                        Op::GlobalWrite(name, _) | Op::GlobalInitialize(name, _) => {
                            Access::named(name)
                        }
                        _ => Access::Any,
                    };
                    targets.push((None, access.clone()));
                    globals.push(access);
                } else if contract.flags & HEAP_WRITE != 0
                    && !op
                        .write_target()
                        .is_some_and(|target| fresh.contains_key(&target))
                {
                    globals.push(Access::written(op, &facts));
                }
            }
        }
        Self {
            targets,
            globals,
            fresh,
            facts,
        }
    }

    pub(super) fn clobbers(&self, receiver: Value, read: &Access) -> bool {
        self.targets.iter().any(|(target, write)| {
            let aliases = match target {
                Some(target) if self.fresh.contains_key(target) => {
                    self.fresh.get(target) == self.fresh.get(&receiver)
                }
                _ => true,
            };
            aliases && write.conflicts(read)
        })
    }

    pub(super) fn clobbers_global(&self, name: &str) -> bool {
        self.globals
            .iter()
            .any(|write| write.conflicts(&Access::named(name)))
    }

    pub(super) fn preserves(&self, op: &Op) -> bool {
        match op {
            Op::Get(object, _)
            | Op::GetIndex(object, _)
            | Op::TypedIndex(object, ..)
            | Op::IndexedGet(object, _)
            | Op::ConstructorPrototype(object) => {
                !self.clobbers(*object, &Access::of(op, &self.facts))
            }
            Op::GlobalRead(name) => !self.clobbers_global(name),
            _ => false,
        }
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

    fn breaks(&self, op: &Op, facts: &Facts) -> bool {
        match (op, Access::written(op, facts)) {
            (Op::GlobalWrite(name, _) | Op::GlobalInitialize(name, _), _) => {
                self.globals.contains(name)
            }
            (_, Access::Named(name)) => {
                self.properties.contains(&name) || self.globals.contains(&name)
            }
            (_, Access::Cells | Access::Packs | Access::Elements | Access::Iteration) => false,
            _ => !self.is_empty(),
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
            let contract = graph.inst_contract(inst, &facts);
            if contract.reentry != Reentry::Closed {
                return Some(format!("reenters at {inst} ({})", graph.op_text(op)));
            }
            let private = op
                .write_target()
                .is_some_and(|target| fresh.contains_key(&target));
            let writes = contract.flags & (GLOBAL_WRITE | HEAP_WRITE) != 0;
            if writes && !private && guarded.breaks(op, &facts) {
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

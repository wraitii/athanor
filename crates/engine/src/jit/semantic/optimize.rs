use std::collections::{HashMap, HashSet};

use oxc_syntax::operator::BinaryOperator;

use super::super::effects::{GLOBAL_WRITE, HEAP_WRITE, Reentry};
use super::{
    Block, BlockData, Cfg, Facts, Graph, Inst, Native, Op, Result, Target, Terminator,
    TerminatorKind, Types, UnaryOp, Value, resolve,
};

pub(super) fn simplify(graph: &mut Graph) {
    let facts = Facts::compute(graph);
    let cfg = Cfg::new(graph);
    let mut uses = graph.use_counts();
    loop {
        let mut aliases = HashMap::new();
        let mut dead = Vec::new();
        for &block in &cfg.rpo {
            for &inst in &graph.blocks[block].insts {
                let data = &graph.insts[inst];
                if let (Some(result), Op::ToNumeric(value) | Op::Unary(UnaryOp::Plus, value)) =
                    (data.result, &data.op)
                    && facts.of(*value).within(Types::NUMBER)
                {
                    aliases.insert(result, *value);
                    dead.push(inst);
                } else if graph.contract(&data.op, &facts).is_pure()
                    && data.result.is_none_or(|result| uses[result] == 0)
                {
                    dead.push(inst);
                }
            }
        }
        if dead.is_empty() {
            return;
        }
        for inst in dead {
            for operand in graph.insts[inst].op.operands() {
                uses[operand] -= 1;
            }
            graph.remove_inst(inst);
        }
        if !aliases.is_empty() {
            graph.replace_uses(&aliases);
            uses = graph.use_counts();
        }
    }
}

fn contained_use(op: &Op, index: usize) -> bool {
    match op {
        Op::Get(..) | Op::GetIndex(..) | Op::Set(..) | Op::SetIndex(..) | Op::Define(..) => {
            index == 0
        }
        Op::IsUndefined(_) | Op::IsNullish(_) | Op::CheckInitialized(_) => true,
        Op::Binary(BinaryOperator::StrictEquality | BinaryOperator::StrictInequality, ..) => true,
        Op::IteratorNew(_) | Op::IteratorNext(_) | Op::EnumerateNew(_) | Op::EnumerateNext(_) => {
            true
        }
        _ => false,
    }
}

fn unescaped(graph: &Graph) -> HashSet<Value> {
    let cfg = Cfg::new(graph);
    let mut allocations = HashSet::new();
    let mut sources = Vec::new();
    for &block in &cfg.rpo {
        for &inst in &graph.blocks[block].insts {
            let data = &graph.insts[inst];
            match data.op {
                Op::NewObject | Op::NewArray(_) => allocations.extend(data.result),
                Op::IteratorNew(source) | Op::EnumerateNew(source) => {
                    allocations.extend(data.result);
                    sources.extend(data.result.map(|iterator| (iterator, source)));
                }
                _ => {}
            }
        }
    }
    for &block in &cfg.rpo {
        for &inst in &graph.blocks[block].insts {
            let op = &graph.insts[inst].op;
            for (index, value) in op.operands().into_iter().enumerate() {
                if !contained_use(op, index) {
                    allocations.remove(&value);
                }
            }
        }
        if let Some(terminator) = &graph.blocks[block].terminator {
            for target in terminator.kind.targets() {
                for value in &target.args {
                    allocations.remove(value);
                }
            }
        }
    }
    for (iterator, source) in sources {
        if !allocations.contains(&iterator) {
            allocations.remove(&source);
        }
    }
    allocations
}

struct Writes {
    targets: Vec<Option<Value>>,
    globals: bool,
    fresh: HashSet<Value>,
}

impl Writes {
    fn collect(graph: &Graph, blocks: &HashSet<Block>) -> Self {
        let facts = Facts::compute(graph);
        let mut writes = Self {
            targets: Vec::new(),
            globals: false,
            fresh: unescaped(graph),
        };
        for &block in blocks {
            for &inst in &graph.blocks[block].insts {
                let op = &graph.insts[inst].op;
                let contract = graph.contract(op, &facts);
                if contract.flags & HEAP_WRITE != 0 {
                    writes.targets.push(op.write_target());
                }
                writes.globals |= contract.flags & GLOBAL_WRITE != 0;
            }
        }
        writes
    }

    fn clobbers(&self, receiver: Value) -> bool {
        self.targets.iter().any(|target| match target {
            Some(target) if self.fresh.contains(target) => *target == receiver,
            _ => true,
        })
    }

    fn preserves(&self, op: &Op) -> bool {
        match op {
            Op::Get(object, _) | Op::GetIndex(object, _) => !self.clobbers(*object),
            Op::GlobalRead(_) => !self.globals,
            _ => false,
        }
    }
}

fn closure_failure(graph: &Graph, blocks: &HashSet<Block>) -> Option<String> {
    let facts = Facts::compute(graph);
    let fresh = unescaped(graph);
    let cfg = Cfg::new(graph);
    for &block in cfg.rpo.iter().filter(|block| blocks.contains(block)) {
        for &inst in &graph.blocks[block].insts {
            let op = &graph.insts[inst].op;
            let contract = graph.contract(op, &facts);
            if contract.reentry != Reentry::Closed {
                return Some(format!("reenters at {inst} ({})", graph.op_text(op)));
            }
            if contract.flags & GLOBAL_WRITE != 0 {
                return Some(format!("writes globals at {inst}"));
            }
            if contract.flags & HEAP_WRITE != 0
                && !op
                    .write_target()
                    .is_some_and(|target| fresh.contains(&target))
            {
                return Some(format!(
                    "writes shared heap at {inst} ({})",
                    graph.op_text(op)
                ));
            }
        }
    }
    None
}

struct Requirement {
    call: Inst,
    global: String,
    property: String,
    native: Native,
}

fn requirements(graph: &Graph, cfg: &Cfg) -> Vec<Requirement> {
    let mut requirements = Vec::new();
    for &block in &cfg.rpo {
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
                });
            }
        }
    }
    requirements
}

pub(super) fn version(
    graph: &mut Graph,
    notes: &mut Vec<String>,
) -> Result<Option<HashSet<Block>>> {
    let cfg = Cfg::new(graph);
    let requirements = requirements(graph, &cfg);
    let all: HashSet<Block> = cfg.rpo.iter().copied().collect();
    if requirements.is_empty() {
        return Ok(match closure_failure(graph, &all) {
            None => {
                notes.push("closed without guards".into());
                Some(all)
            }
            Some(reason) => {
                notes.push(format!("not optimized: {reason}"));
                None
            }
        });
    }
    for requirement in &requirements {
        let span = graph.insts[requirement.call].span;
        notes.push(format!(
            "call at {}..{} reenters unless its callee is {}",
            span.start, span.end, requirement.native
        ));
    }
    let mut candidate = graph.clone();
    let fast = apply_version(&mut candidate, &cfg, &requirements);
    candidate.verify()?;
    if let Some(reason) = closure_failure(&candidate, &fast) {
        notes.push(format!("not versioned: guarded copy {reason}"));
        return Ok(None);
    }
    notes.push(format!(
        "versioned: guards at {} select the closed copy, {} keeps the original",
        candidate.entry, graph.entry
    ));
    *graph = candidate;
    Ok(Some(fast))
}

fn apply_version(graph: &mut Graph, cfg: &Cfg, requirements: &[Requirement]) -> HashSet<Block> {
    let original = graph.entry;
    let mut blocks = HashMap::new();
    let mut values = HashMap::new();
    let mut insts = HashMap::new();
    for &block in &cfg.rpo {
        let copy = graph.blocks.push(BlockData::default());
        blocks.insert(block, copy);
        for param in graph.blocks[block].params.clone() {
            values.insert(param, graph.add_param(copy));
        }
    }
    for &block in &cfg.rpo {
        for inst in graph.blocks[block].insts.clone() {
            let data = graph.insts[inst].clone();
            let mut op = data.op;
            for value in op.operands_mut() {
                *value = values[value];
            }
            let (copy, result) = graph.append(blocks[&block], data.span, op, data.result.is_some());
            if let (Some(old), Some(new)) = (data.result, result) {
                values.insert(old, new);
            }
            insts.insert(inst, copy);
        }
        let mut terminator = graph.blocks[block].terminator.clone().unwrap();
        for value in terminator.kind.operands_mut() {
            *value = values[value];
        }
        for target in terminator.kind.targets_mut() {
            target.block = blocks[&target.block];
        }
        graph.blocks[blocks[&block]].terminator = Some(terminator);
    }

    let guard = graph.blocks.push(BlockData::default());
    let span = graph.span;
    let mut globals: HashMap<String, Value> = HashMap::new();
    let mut methods: HashMap<(String, String), Value> = HashMap::new();
    let mut checks = Vec::new();
    for requirement in requirements {
        let global = match globals.get(&requirement.global) {
            Some(&value) => value,
            None => {
                let op = Op::ProbeGlobal(requirement.global.clone());
                let value = graph.append(guard, span, op, true).1.unwrap();
                globals.insert(requirement.global.clone(), value);
                value
            }
        };
        let key = (requirement.global.clone(), requirement.property.clone());
        if methods.contains_key(&key) {
            continue;
        }
        let op = Op::ProbeProperty(global, requirement.property.clone());
        let method = graph.append(guard, span, op, true).1.unwrap();
        methods.insert(key, method);
        let op = Op::IsNative(method, requirement.native);
        checks.push(graph.append(guard, span, op, true).1.unwrap());
    }
    let mut current = guard;
    for (index, &check) in checks.iter().enumerate() {
        let next = if index + 1 == checks.len() {
            blocks[&original]
        } else {
            graph.blocks.push(BlockData::default())
        };
        graph.blocks[current].terminator = Some(Terminator {
            span,
            kind: TerminatorKind::Branch(check, Target::new(next), Target::new(original)),
        });
        current = next;
    }
    graph.entry = guard;

    let mut aliases = HashMap::new();
    let mut replaced = Vec::new();
    for (&inst, &copy) in &insts {
        let Some(result) = graph.insts[copy].result else {
            continue;
        };
        let target = match &graph.insts[inst].op {
            Op::GlobalRead(global) => globals.get(global).copied(),
            Op::Get(object, property) => match graph.defining_op(*object) {
                Some(Op::GlobalRead(global)) => {
                    methods.get(&(global.clone(), property.clone())).copied()
                }
                _ => None,
            },
            _ => None,
        };
        if let Some(target) = target {
            aliases.insert(result, target);
            replaced.push(copy);
        }
    }
    for requirement in requirements {
        let copy = insts[&requirement.call];
        if let Op::Call { args, .. } = &graph.insts[copy].op {
            graph.insts[copy].op = Op::CallNative(requirement.native, args.clone());
        }
    }
    for inst in replaced {
        graph.remove_inst(inst);
    }
    graph.replace_uses(&aliases);
    blocks.into_values().collect()
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum LoadKey {
    Property(Value, String),
    Index(Value, Value),
    Global(String),
    This,
    Argument(usize),
}

pub(super) fn reuse_loads(graph: &mut Graph, region: &HashSet<Block>, notes: &mut Vec<String>) {
    let cfg = Cfg::new(graph);
    let writes = Writes::collect(graph, region);
    let mut aliases = HashMap::new();
    let mut available: HashMap<LoadKey, Vec<(Block, usize, Value)>> = HashMap::new();
    let mut reused = Vec::new();
    for &block in cfg.rpo.iter().filter(|block| region.contains(block)) {
        for (index, &inst) in graph.blocks[block].insts.iter().enumerate() {
            let data = &graph.insts[inst];
            let Some(result) = data.result else {
                continue;
            };
            let key = match &data.op {
                Op::Get(object, name) => {
                    LoadKey::Property(resolve(&aliases, *object), name.clone())
                }
                Op::GetIndex(object, key) => {
                    LoadKey::Index(resolve(&aliases, *object), resolve(&aliases, *key))
                }
                Op::GlobalRead(name) => LoadKey::Global(name.clone()),
                Op::This => LoadKey::This,
                Op::Argument(index) => LoadKey::Argument(*index),
                _ => continue,
            };
            let preserved = match &key {
                LoadKey::Property(object, _) | LoadKey::Index(object, _) => {
                    !writes.clobbers(*object)
                }
                LoadKey::Global(_) => !writes.globals,
                LoadKey::This | LoadKey::Argument(_) => true,
            };
            if !preserved {
                continue;
            }
            let earlier = available.get(&key).and_then(|defs| {
                defs.iter().find(|(def_block, def_index, _)| {
                    if *def_block == block {
                        *def_index < index
                    } else {
                        cfg.dominates(*def_block, block)
                    }
                })
            });
            match earlier {
                Some(&(_, _, value)) => {
                    aliases.insert(result, value);
                    reused.push((inst, value));
                }
                None => available
                    .entry(key)
                    .or_default()
                    .push((block, index, result)),
            }
        }
    }
    graph.replace_uses(&aliases);
    for (inst, value) in reused {
        notes.push(format!(
            "{} reuses {value}",
            graph.op_text(&graph.insts[inst].op)
        ));
        graph.remove_inst(inst);
    }
}

fn natural_loops(graph: &Graph, cfg: &Cfg) -> Vec<(Block, HashSet<Block>)> {
    let mut loops: HashMap<Block, HashSet<Block>> = HashMap::new();
    for &block in &cfg.rpo {
        for header in graph.successors(block) {
            if !cfg.dominates(header, block) {
                continue;
            }
            let body = loops
                .entry(header)
                .or_insert_with(|| HashSet::from([header]));
            let mut stack = vec![block];
            while let Some(node) = stack.pop() {
                if body.insert(node) {
                    stack.extend(cfg.preds[node].iter().copied());
                }
            }
        }
    }
    let mut loops: Vec<_> = loops.into_iter().collect();
    loops.sort_by_key(|(header, body)| (body.len(), *header));
    loops
}

pub(super) fn hoist(graph: &mut Graph, region: &HashSet<Block>, notes: &mut Vec<String>) {
    let cfg = Cfg::new(graph);
    let facts = Facts::compute(graph);
    for (header, body) in natural_loops(graph, &cfg) {
        if !body.is_subset(region) {
            continue;
        }
        let outside: Vec<Block> = cfg.preds[header]
            .iter()
            .copied()
            .filter(|pred| !body.contains(pred))
            .collect();
        let [preheader] = outside[..] else {
            continue;
        };
        let jumps = matches!(
            graph.blocks[preheader].terminator,
            Some(Terminator {
                kind: TerminatorKind::Jump(_),
                ..
            })
        );
        if !jumps {
            continue;
        }
        let writes = Writes::collect(graph, &body);
        for inst in graph.blocks[header].insts.clone() {
            let data = &graph.insts[inst];
            let pure = graph.contract(&data.op, &facts).is_pure();
            let invariant = data
                .op
                .operands()
                .into_iter()
                .all(|value| !body.contains(&graph.def_block(value)));
            if data.result.is_some() && invariant && (pure || writes.preserves(&data.op)) {
                let result = data.result.unwrap();
                notes.push(format!(
                    "{result} = {} hoisted from {header} to {preheader}",
                    graph.op_text(&data.op)
                ));
                graph.move_inst(inst, preheader);
            } else if !pure {
                break;
            }
        }
    }
}

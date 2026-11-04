use std::collections::{HashMap, HashSet};

use oxc_syntax::operator::BinaryOperator;

use super::super::effects::{GLOBAL_WRITE, HEAP_WRITE, Reentry};
use super::{
    Block, BlockData, Cfg, Facts, Graph, Inst, Native, Op, Result, Target, Terminator,
    TerminatorKind, Types, UnaryOp, Value, ValueDef, resolve,
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
        Op::Get(..)
        | Op::GetIndex(..)
        | Op::TypedIndex(..)
        | Op::Set(..)
        | Op::SetIndex(..)
        | Op::Define(..) => index == 0,
        Op::IsUndefined(_) | Op::IsNullish(_) | Op::CheckInitialized(_) => true,
        Op::Binary(BinaryOperator::StrictEquality | BinaryOperator::StrictInequality, ..) => true,
        Op::IteratorNew(_) | Op::IteratorNext(_) | Op::EnumerateNew(_) | Op::EnumerateNext(_) => {
            true
        }
        _ => false,
    }
}

fn unescaped(graph: &Graph, scope: &HashSet<Block>) -> HashSet<Value> {
    let cfg = Cfg::new(graph);
    let blocks: Vec<Block> = cfg
        .rpo
        .iter()
        .copied()
        .filter(|block| scope.contains(block))
        .collect();
    let mut allocations = HashSet::new();
    let mut sources = Vec::new();
    for &block in &blocks {
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
    for &block in &blocks {
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
                if scope.contains(&target.block) {
                    for value in &target.args {
                        allocations.remove(value);
                    }
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

#[derive(Clone, PartialEq, Eq, Hash)]
enum Access {
    Named(String),
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

    fn of(op: &Op, facts: &Facts) -> Self {
        match op {
            Op::Get(_, name) | Op::Set(_, name, _) | Op::Define(_, name, _) => Self::named(name),
            Op::GetIndex(_, key) | Op::TypedIndex(_, key) | Op::SetIndex(_, key, _)
                if facts.of(*key).within(Types::NUMBER) =>
            {
                Self::Elements
            }
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
            (Self::Named(write), Self::Named(read)) => write == read,
            (Self::Elements, Self::Elements | Self::Length) => true,
            _ => false,
        }
    }
}

struct Writes {
    targets: Vec<(Option<Value>, Access)>,
    globals: Vec<Access>,
    fresh: HashSet<Value>,
    facts: Facts,
}

impl Writes {
    fn collect(graph: &Graph, blocks: &HashSet<Block>) -> Self {
        let facts = Facts::compute(graph);
        let mut targets = Vec::new();
        let mut globals = Vec::new();
        let fresh = unescaped(graph, blocks);
        for &block in blocks {
            for &inst in &graph.blocks[block].insts {
                let op = &graph.insts[inst].op;
                let contract = graph.contract(op, &facts);
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
                        .is_some_and(|target| fresh.contains(&target))
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

    fn clobbers(&self, receiver: Value, read: &Access) -> bool {
        self.targets.iter().any(|(target, write)| {
            let aliases = match target {
                Some(target) if self.fresh.contains(target) => *target == receiver,
                _ => true,
            };
            aliases && write.conflicts(read)
        })
    }

    fn clobbers_global(&self, name: &str) -> bool {
        self.globals
            .iter()
            .any(|write| write.conflicts(&Access::named(name)))
    }

    fn preserves(&self, op: &Op) -> bool {
        match op {
            Op::Get(object, _) | Op::GetIndex(object, _) | Op::TypedIndex(object, _) => {
                !self.clobbers(*object, &Access::of(op, &self.facts))
            }
            Op::GlobalRead(name) => !self.clobbers_global(name),
            _ => false,
        }
    }
}

#[derive(Default)]
struct Guarded {
    globals: HashSet<String>,
    properties: HashSet<String>,
}

impl Guarded {
    fn new(requirements: &[Requirement]) -> Self {
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
            (_, Access::Elements | Access::Iteration) => false,
            _ => !self.is_empty(),
        }
    }
}

fn closure_failure(graph: &Graph, blocks: &HashSet<Block>, guarded: &Guarded) -> Option<String> {
    let facts = Facts::compute(graph);
    let fresh = unescaped(graph, blocks);
    let cfg = Cfg::new(graph);
    for &block in cfg.rpo.iter().filter(|block| blocks.contains(block)) {
        for &inst in &graph.blocks[block].insts {
            let op = &graph.insts[inst].op;
            let contract = graph.contract(op, &facts);
            if contract.reentry != Reentry::Closed {
                return Some(format!("reenters at {inst} ({})", graph.op_text(op)));
            }
            let private = op
                .write_target()
                .is_some_and(|target| fresh.contains(&target));
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

struct Requirement {
    call: Inst,
    global: String,
    property: String,
    native: Native,
    captured: Option<Value>,
    object: Option<Value>,
}

fn requirements(graph: &Graph, blocks: &[Block]) -> Vec<Requirement> {
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

pub(super) fn version(graph: &mut Graph, notes: &mut Vec<String>) -> Result<Vec<HashSet<Block>>> {
    let blocks = Cfg::new(graph).rpo;
    if let Some(region) = region(graph, &blocks, None, "", notes)? {
        return Ok(vec![region]);
    }
    let mut regions = Vec::new();
    let mut covered = HashSet::new();
    let mut tried = HashSet::new();
    loop {
        let cfg = Cfg::new(graph);
        let Some((header, body)) = natural_loops(graph, &cfg)
            .into_iter()
            .rev()
            .find(|(header, body)| !tried.contains(header) && body.is_disjoint(&covered))
        else {
            return Ok(regions);
        };
        tried.insert(header);
        let Some(preheader) = preheader(graph, &cfg, header, &body) else {
            continue;
        };
        let blocks: Vec<Block> = cfg
            .rpo
            .iter()
            .copied()
            .filter(|block| body.contains(block))
            .collect();
        let label = format!("loop at {header}: ");
        if let Some(region) = region(graph, &blocks, Some((preheader, header)), &label, notes)? {
            covered.extend(body);
            covered.extend(region.iter().copied());
            regions.push(region);
        }
    }
}

fn region(
    graph: &mut Graph,
    blocks: &[Block],
    entry: Option<(Block, Block)>,
    label: &str,
    notes: &mut Vec<String>,
) -> Result<Option<HashSet<Block>>> {
    let all: HashSet<Block> = blocks.iter().copied().collect();
    let requirements = requirements(graph, blocks);
    if requirements.is_empty() {
        if let Some(reason) = closure_failure(graph, &all, &Guarded::default()) {
            notes.push(format!("{label}not optimized: {reason}"));
            return Ok(None);
        }
    }
    if requirements.is_empty() {
        notes.push(format!("{label}closed without guards"));
        return Ok(Some(all));
    }
    let mut candidate = graph.clone();
    let (fast, _) = apply_version(&mut candidate, blocks, entry, &requirements, &[], &[], &[]);
    candidate.verify()?;
    if let Some(reason) = closure_failure(&candidate, &fast, &Guarded::new(&requirements)) {
        notes.push(format!("{label}not versioned: guarded copy {reason}"));
        return Ok(None);
    }
    for requirement in &requirements {
        let span = graph.insts[requirement.call].span;
        notes.push(format!(
            "{label}call at {}..{} reenters unless its callee is {}",
            span.start, span.end, requirement.native
        ));
    }
    let head = entry.map_or(graph.entry, |(_, header)| header);
    notes.push(format!(
        "{label}versioned: entry guards select the closed copy, {head} keeps the original"
    ));
    *graph = candidate;
    Ok(Some(fast))
}

pub(super) fn check_loops(
    graph: &mut Graph,
    regions: &mut [HashSet<Block>],
    notes: &mut Vec<String>,
) -> Result<Vec<HashSet<Block>>> {
    let mut copies = Vec::new();
    let mut fallbacks = HashSet::new();
    let mut tried = HashSet::new();
    loop {
        let cfg = Cfg::new(graph);
        let Some((header, body)) =
            natural_loops(graph, &cfg)
                .into_iter()
                .rev()
                .find(|(header, body)| {
                    !tried.contains(header)
                        && body.is_disjoint(&fallbacks)
                        && regions.iter().any(|region| body.is_subset(region))
                })
        else {
            return Ok(copies);
        };
        tried.insert(header);
        let Some(preheader) = preheader(graph, &cfg, header, &body) else {
            continue;
        };
        let blocks: Vec<Block> = cfg
            .rpo
            .iter()
            .copied()
            .filter(|block| body.contains(block))
            .collect();
        let probes = probe_reads(graph, &blocks);
        let checks = entry_checks(graph, &blocks, &probes);
        let mut candidate = graph.clone();
        let mut numbered = None;
        if !checks.is_empty() || !probes.is_empty() {
            let (copy, copy_header) = apply_version(
                &mut candidate,
                &blocks,
                Some((preheader, header)),
                &[],
                &checks,
                &probes,
                &[],
            );
            candidate.verify()?;
            if closure_failure(&candidate, &copy, &Guarded::default()).is_none() {
                let list: Vec<String> = checks.iter().map(ToString::to_string).collect();
                notes.push(format!(
                    "loop at {header}: number checks on [{}] and {} read probes select the typed copy",
                    list.join(", "),
                    probes.len()
                ));
                numbered = Some((copy, copy_header));
            } else {
                candidate = graph.clone();
            }
        }
        let (layer_body, layer_header) = match &numbered {
            Some((copy, copy_header)) => (copy.clone(), *copy_header),
            None => (body.clone(), header),
        };
        let typed = typed_version(&mut candidate, layer_header, &layer_body, notes)?;
        if numbered.is_none() && typed.is_none() {
            continue;
        }
        *graph = candidate;
        fallbacks.extend(body.iter().copied());
        let mut added: Vec<HashSet<Block>> = Vec::new();
        if let Some((copy, _)) = numbered {
            if typed.is_some() {
                fallbacks.extend(copy.iter().copied());
            }
            added.push(copy);
        }
        added.extend(typed);
        for copy in added {
            for region in regions.iter_mut() {
                if body.is_subset(region) {
                    region.extend(copy.iter().copied());
                }
            }
            copies.push(copy);
        }
    }
}

const NUMERIC: Types = Types::NUMBER.union(Types::UNDEFINED);

fn receiver_use(op: &Op, index: usize) -> bool {
    match op {
        Op::Get(..)
        | Op::GetIndex(..)
        | Op::TypedIndex(..)
        | Op::Set(..)
        | Op::SetIndex(..)
        | Op::Define(..)
        | Op::Construct { .. }
        | Op::IteratorNew(_)
        | Op::EnumerateNew(_) => index == 0,
        Op::Call { receiver, .. } => index == 0 || (index == 1 && receiver.is_some()),
        _ => false,
    }
}

fn typed_objects(graph: &Graph, body: &HashSet<Block>) -> Vec<Value> {
    let facts = Facts::compute(graph);
    let mut reads = HashMap::new();
    for &block in body {
        for &inst in &graph.blocks[block].insts {
            let data = &graph.insts[inst];
            if let (Op::GetIndex(object, key), Some(result)) = (&data.op, data.result)
                && !body.contains(&graph.def_block(*object))
                && facts.of(*object).intersects(Types::OBJECT)
                && facts.of(*key).within(NUMERIC)
            {
                reads.insert(result, *object);
            }
        }
    }
    let mut excluded = HashSet::new();
    for &block in body {
        for &inst in &graph.blocks[block].insts {
            let op = &graph.insts[inst].op;
            for (index, value) in op.operands().into_iter().enumerate() {
                if let Some(&object) = reads.get(&value)
                    && receiver_use(op, index)
                {
                    excluded.insert(object);
                }
            }
        }
    }
    let objects: HashSet<Value> = reads.into_values().collect();
    let mut objects: Vec<Value> = objects.difference(&excluded).copied().collect();
    objects.sort();
    objects
}

fn typed_version(
    graph: &mut Graph,
    header: Block,
    body: &HashSet<Block>,
    notes: &mut Vec<String>,
) -> Result<Option<HashSet<Block>>> {
    let cfg = Cfg::new(graph);
    let Some(preheader) = preheader(graph, &cfg, header, body) else {
        return Ok(None);
    };
    let objects = typed_objects(graph, body);
    if objects.is_empty() {
        return Ok(None);
    }
    let blocks: Vec<Block> = cfg
        .rpo
        .iter()
        .copied()
        .filter(|block| body.contains(block))
        .collect();
    let mut candidate = graph.clone();
    let (copy, _) = apply_version(
        &mut candidate,
        &blocks,
        Some((preheader, header)),
        &[],
        &[],
        &[],
        &objects,
    );
    let facts = Facts::compute(&candidate);
    for &block in &copy {
        for inst in candidate.blocks[block].insts.clone() {
            if let Op::GetIndex(object, key) = candidate.insts[inst].op
                && objects.contains(&object)
                && facts.of(key).within(NUMERIC)
            {
                candidate.insts[inst].op = Op::TypedIndex(object, key);
            }
        }
    }
    candidate.verify()?;
    if closure_failure(&candidate, &copy, &Guarded::default()).is_some() {
        return Ok(None);
    }
    let list: Vec<String> = objects.iter().map(ToString::to_string).collect();
    notes.push(format!(
        "loop at {header}: typed array checks on [{}] select the typed copy",
        list.join(", ")
    ));
    *graph = candidate;
    Ok(Some(copy))
}

fn preheader(graph: &Graph, cfg: &Cfg, header: Block, body: &HashSet<Block>) -> Option<Block> {
    let outside: Vec<Block> = cfg.preds[header]
        .iter()
        .copied()
        .filter(|pred| !body.contains(pred))
        .collect();
    let [preheader] = outside[..] else {
        return None;
    };
    matches!(
        graph.blocks[preheader].terminator,
        Some(Terminator {
            kind: TerminatorKind::Jump(_),
            ..
        })
    )
    .then_some(preheader)
}

fn numeric_use(op: &Op, index: usize) -> bool {
    match op {
        Op::Binary(operator, ..) => !matches!(
            operator,
            BinaryOperator::In
                | BinaryOperator::Instanceof
                | BinaryOperator::Equality
                | BinaryOperator::Inequality
                | BinaryOperator::StrictEquality
                | BinaryOperator::StrictInequality
        ),
        Op::Unary(operator, _) => !matches!(operator, UnaryOp::Not | UnaryOp::Typeof),
        Op::ToNumeric(_) | Op::CallNative(..) => true,
        Op::GetIndex(..) | Op::TypedIndex(..) | Op::SetIndex(..) => index == 1,
        _ => false,
    }
}

fn entry_checks(graph: &Graph, region: &[Block], probes: &[Inst]) -> Vec<Value> {
    let facts = Facts::compute(graph);
    let inside: HashSet<Block> = region.iter().copied().collect();
    let probed: HashSet<Value> = probes
        .iter()
        .filter_map(|inst| graph.insts[*inst].result)
        .collect();
    let mut checks = HashSet::new();
    for &block in region {
        for &inst in &graph.blocks[block].insts {
            let op = &graph.insts[inst].op;
            for (index, value) in op.operands().into_iter().enumerate() {
                if numeric_use(op, index)
                    && !facts.of(value).within(NUMERIC)
                    && (!inside.contains(&graph.def_block(value)) || probed.contains(&value))
                    && !matches!(graph.defining_op(value), Some(Op::Constant(_)))
                {
                    checks.insert(value);
                }
            }
        }
    }
    let mut checks: Vec<Value> = checks.into_iter().collect();
    checks.sort();
    checks
}

fn retarget(graph: &mut Graph, blocks: &HashSet<Block>, from: Value, to: Value) {
    let aliases = HashMap::from([(from, to)]);
    for &block in blocks {
        for inst in graph.blocks[block].insts.clone() {
            if !matches!(graph.insts[inst].op, Op::IsNumber(_) | Op::AsNumber(_)) {
                for value in graph.insts[inst].op.operands_mut() {
                    *value = resolve(&aliases, *value);
                }
            }
        }
        if let Some(terminator) = &mut graph.blocks[block].terminator {
            for value in terminator.kind.operands_mut() {
                *value = resolve(&aliases, *value);
            }
        }
    }
}

fn apply_version(
    graph: &mut Graph,
    region: &[Block],
    entry: Option<(Block, Block)>,
    requirements: &[Requirement],
    checks: &[Value],
    reads: &[Inst],
    typed: &[Value],
) -> (HashSet<Block>, Block) {
    let head = entry.map_or(graph.entry, |(_, header)| header);
    let mut blocks = HashMap::new();
    let mut values = HashMap::new();
    let mut insts = HashMap::new();
    for &block in region {
        let copy = graph.blocks.push(BlockData::default());
        blocks.insert(block, copy);
        for param in graph.blocks[block].params.clone() {
            values.insert(param, graph.add_param(copy));
        }
    }
    for &block in region {
        for inst in graph.blocks[block].insts.clone() {
            let data = graph.insts[inst].clone();
            let mut op = data.op;
            for value in op.operands_mut() {
                *value = values.get(value).copied().unwrap_or(*value);
            }
            let (copy, result) = graph.append(blocks[&block], data.span, op, data.result.is_some());
            if let (Some(old), Some(new)) = (data.result, result) {
                values.insert(old, new);
            }
            insts.insert(inst, copy);
        }
        let mut terminator = graph.blocks[block].terminator.clone().unwrap();
        for value in terminator.kind.operands_mut() {
            *value = values.get(value).copied().unwrap_or(*value);
        }
        for target in terminator.kind.targets_mut() {
            if let Some(&copy) = blocks.get(&target.block) {
                target.block = copy;
            }
        }
        graph.blocks[blocks[&block]].terminator = Some(terminator);
    }
    let copies: HashSet<Block> = blocks.values().copied().collect();
    graph.cold.extend(region.iter().copied());

    let span = graph.span;
    let mut args = match entry {
        Some((preheader, _)) => match &graph.blocks[preheader].terminator {
            Some(Terminator {
                kind: TerminatorKind::Jump(target),
                ..
            }) => target.args.clone(),
            _ => unreachable!(),
        },
        None => Vec::new(),
    };
    let original = Target {
        block: head,
        args: args.clone(),
    };
    let start = graph.blocks.push(BlockData::default());
    let mut current = start;
    if requirements
        .iter()
        .any(|r| r.captured.is_none() && r.object.is_none())
    {
        let guard = graph.guards;
        graph.guards += 1;
        let probes = graph.blocks.push(BlockData::default());
        let commit = graph.blocks.push(BlockData::default());
        graph.cold.extend([probes, commit]);
        let validated = graph.blocks.push(BlockData::default());
        let valid = graph
            .append(start, span, Op::GuardValid(guard), true)
            .1
            .unwrap();
        graph.blocks[start].terminator = Some(Terminator {
            span,
            kind: TerminatorKind::Branch(valid, Target::new(validated), Target::new(probes)),
        });
        let mut globals: HashMap<String, Value> = HashMap::new();
        let mut methods = HashSet::new();
        let mut probed = Vec::new();
        for requirement in requirements
            .iter()
            .filter(|r| r.captured.is_none() && r.object.is_none())
        {
            let global = match globals.get(&requirement.global) {
                Some(&value) => value,
                None => {
                    let op = Op::ProbeGlobal(requirement.global.clone());
                    let value = graph.append(probes, span, op, true).1.unwrap();
                    globals.insert(requirement.global.clone(), value);
                    value
                }
            };
            if !methods.insert((requirement.global.clone(), requirement.property.clone())) {
                continue;
            }
            let op = Op::ProbeProperty(global, requirement.property.clone());
            let method = graph.append(probes, span, op, true).1.unwrap();
            let op = Op::IsNative(method, requirement.native);
            probed.push(graph.append(probes, span, op, true).1.unwrap());
        }
        let mut chain = probes;
        for (index, &check) in probed.iter().enumerate() {
            let next = if index + 1 == probed.len() {
                commit
            } else {
                let next = graph.blocks.push(BlockData::default());
                graph.cold.insert(next);
                next
            };
            graph.blocks[chain].terminator = Some(Terminator {
                span,
                kind: TerminatorKind::Branch(check, Target::new(next), original.clone()),
            });
            chain = next;
        }
        graph.append(commit, span, Op::GuardCommit(guard), false);
        graph.blocks[commit].terminator = Some(Terminator {
            span,
            kind: TerminatorKind::Jump(Target::new(validated)),
        });
        current = validated;
    }
    for requirement in requirements
        .iter()
        .filter(|r| r.captured.is_some() || r.object.is_some())
    {
        let method = match requirement.captured {
            Some(value) => value,
            None => graph
                .append(
                    current,
                    span,
                    Op::ProbeProperty(requirement.object.unwrap(), requirement.property.clone()),
                    true,
                )
                .1
                .unwrap(),
        };
        let check = graph
            .append(
                current,
                span,
                Op::IsNative(method, requirement.native),
                true,
            )
            .1
            .unwrap();
        let next = graph.blocks.push(BlockData::default());
        graph.blocks[current].terminator = Some(Terminator {
            span,
            kind: TerminatorKind::Branch(check, Target::new(next), original.clone()),
        });
        current = next;
    }
    for &inst in reads {
        let data = graph.insts[inst].clone();
        let op = match data.op {
            Op::Get(object, name) => Op::ProbeProperty(object, name),
            Op::GetIndex(object, key) => Op::ProbeIndex(object, key),
            _ => unreachable!(),
        };
        let value = graph.append(current, data.span, op, true).1.unwrap();
        let failed = graph
            .append(current, data.span, Op::IsUninitialized(value), true)
            .1
            .unwrap();
        let next = graph.blocks.push(BlockData::default());
        graph.blocks[current].terminator = Some(Terminator {
            span: data.span,
            kind: TerminatorKind::Branch(failed, original.clone(), Target::new(next)),
        });
        current = next;
        let result = data.result.unwrap();
        retarget(graph, &copies, values[&result], value);
        graph.remove_inst(insts[&inst]);
        values.insert(result, value);
    }
    for &object in typed {
        let check = graph
            .append(current, span, Op::IsTyped(object), true)
            .1
            .unwrap();
        let next = graph.blocks.push(BlockData::default());
        graph.blocks[current].terminator = Some(Terminator {
            span,
            kind: TerminatorKind::Branch(check, Target::new(next), original.clone()),
        });
        current = next;
    }
    for &value in checks {
        let checked = values.get(&value).copied().unwrap_or(value);
        let check = graph
            .append(current, span, Op::IsNumber(checked), true)
            .1
            .unwrap();
        let next = graph.blocks.push(BlockData::default());
        graph.blocks[current].terminator = Some(Terminator {
            span,
            kind: TerminatorKind::Branch(check, Target::new(next), original.clone()),
        });
        current = next;
    }
    for &value in checks {
        let checked = values.get(&value).copied().unwrap_or(value);
        let number = graph
            .append(current, span, Op::AsNumber(checked), true)
            .1
            .unwrap();
        retarget(graph, &copies, checked, number);
        for arg in &mut args {
            if *arg == checked {
                *arg = number;
            }
        }
        if values.contains_key(&value) {
            values.insert(value, number);
        }
    }
    graph.blocks[current].terminator = Some(Terminator {
        span,
        kind: TerminatorKind::Jump(Target {
            block: blocks[&head],
            args,
        }),
    });
    match entry {
        Some((preheader, _)) => {
            graph.blocks[preheader].terminator = Some(Terminator {
                span,
                kind: TerminatorKind::Jump(Target::new(start)),
            });
        }
        None => graph.entry = start,
    }

    let mut callees = Vec::new();
    for requirement in requirements {
        let copy = insts[&requirement.call];
        if let Op::Call { callee, args, .. } = &graph.insts[copy].op {
            callees.push(*callee);
            graph.insts[copy].op = Op::CallNative(requirement.native, args.clone());
        }
    }
    let inside: HashSet<Block> = region.iter().chain(&copies).copied().collect();
    let mut outside = HashSet::new();
    for (block, data) in graph.blocks.iter() {
        if inside.contains(&block) {
            continue;
        }
        for &inst in &data.insts {
            outside.extend(graph.insts[inst].op.operands());
        }
        if let Some(terminator) = &data.terminator {
            outside.extend(terminator.kind.operands());
        }
    }
    let pinned: HashSet<Value> = values
        .iter()
        .filter(|(original, _)| outside.contains(*original))
        .map(|(_, &copy)| copy)
        .collect();
    let mut uses = graph.use_counts();
    while let Some(value) = callees.pop() {
        if uses[value] != 0 || pinned.contains(&value) {
            continue;
        }
        let ValueDef::Result(inst) = graph.values[value] else {
            continue;
        };
        match graph.insts[inst].op {
            Op::Get(object, _) => {
                uses[object] -= 1;
                callees.push(object);
            }
            Op::GlobalRead(_) => {}
            _ => continue,
        }
        uses[value] = u32::MAX;
        graph.remove_inst(inst);
    }
    let definitions: HashMap<Value, Vec<Value>> = values
        .iter()
        .map(|(&original, &copy)| {
            let live = uses[copy] != u32::MAX;
            (original, if live { vec![copy] } else { Vec::new() })
        })
        .collect();
    repair(graph, &definitions);
    (copies, blocks[&head])
}

fn repair(graph: &mut Graph, values: &HashMap<Value, Vec<Value>>) {
    let cfg = Cfg::new(graph);
    let mut repair = Repair {
        cfg: &cfg,
        values,
        reaching: HashMap::new(),
    };
    for &block in &cfg.rpo {
        for inst in graph.blocks[block].insts.clone() {
            let mut op = graph.insts[inst].op.clone();
            for value in op.operands_mut() {
                if values.contains_key(value) && !cfg.dominates(graph.def_block(*value), block) {
                    *value = repair.entering(graph, block, *value);
                }
            }
            graph.insts[inst].op = op;
        }
        let mut reaching = HashMap::new();
        for value in graph.blocks[block]
            .terminator
            .as_ref()
            .unwrap()
            .kind
            .operands()
        {
            if values.contains_key(&value) && !cfg.dominates(graph.def_block(value), block) {
                let replacement = repair.entering(graph, block, value);
                reaching.insert(value, replacement);
            }
        }
        if let Some(terminator) = &mut graph.blocks[block].terminator {
            for value in terminator.kind.operands_mut() {
                if let Some(&replacement) = reaching.get(value) {
                    *value = replacement;
                }
            }
        }
    }
}

struct Repair<'a> {
    cfg: &'a Cfg,
    values: &'a HashMap<Value, Vec<Value>>,
    reaching: HashMap<(Block, Value), Value>,
}

impl Repair<'_> {
    fn leaving(&mut self, graph: &mut Graph, block: Block, value: Value) -> Value {
        if self.cfg.dominates(graph.def_block(value), block) {
            return value;
        }
        let copy = self.values[&value]
            .iter()
            .copied()
            .find(|&copy| self.cfg.dominates(graph.def_block(copy), block));
        match copy {
            Some(copy) => copy,
            None => self.entering(graph, block, value),
        }
    }

    fn entering(&mut self, graph: &mut Graph, block: Block, value: Value) -> Value {
        if let Some(&reaching) = self.reaching.get(&(block, value)) {
            return reaching;
        }
        let preds = self.cfg.preds[block].clone();
        if let [pred] = preds[..] {
            let reaching = self.leaving(graph, pred, value);
            self.reaching.insert((block, value), reaching);
            return reaching;
        }
        let param = graph.add_param(block);
        self.reaching.insert((block, value), param);
        for pred in preds {
            let arg = self.leaving(graph, pred, value);
            graph.push_edge_arg(pred, block, arg);
        }
        param
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum LoadKey {
    Property(Value, String),
    Index(Value, Value),
    Global(String),
    This,
    Argument(usize),
    Binary(BinaryOperator, Value, Value),
}

pub(super) fn reuse_loads(graph: &mut Graph, region: &HashSet<Block>, notes: &mut Vec<String>) {
    let cfg = Cfg::new(graph);
    let writes = Writes::collect(graph, region);
    let facts = &writes.facts;
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
                Op::GetIndex(object, key) | Op::TypedIndex(object, key) => {
                    LoadKey::Index(resolve(&aliases, *object), resolve(&aliases, *key))
                }
                Op::GlobalRead(name) => LoadKey::Global(name.clone()),
                Op::This => LoadKey::This,
                Op::Argument(index) => LoadKey::Argument(*index),
                Op::Binary(operator, left, right) if graph.contract(&data.op, facts).is_pure() => {
                    LoadKey::Binary(
                        *operator,
                        resolve(&aliases, *left),
                        resolve(&aliases, *right),
                    )
                }
                _ => continue,
            };
            let preserved = match &key {
                LoadKey::Property(object, _) | LoadKey::Index(object, _) => {
                    !writes.clobbers(*object, &Access::of(&data.op, facts))
                }
                LoadKey::Global(name) => !writes.clobbers_global(name),
                LoadKey::This | LoadKey::Argument(_) | LoadKey::Binary(..) => true,
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
        let Some(preheader) = preheader(graph, &cfg, header, &body) else {
            continue;
        };
        let writes = Writes::collect(graph, &body);
        let invariant = |graph: &Graph, op: &Op| {
            op.operands()
                .into_iter()
                .all(|value| !body.contains(&graph.def_block(value)))
        };
        let mut moved = Vec::new();
        for inst in graph.blocks[header].insts.clone() {
            let data = &graph.insts[inst];
            let pure = graph.contract(&data.op, &facts).is_pure();
            if data.result.is_some()
                && invariant(graph, &data.op)
                && (pure || writes.preserves(&data.op))
            {
                moved.push(inst);
                graph.move_inst(inst, preheader);
            } else if !pure {
                break;
            }
        }
        for &block in cfg.rpo.iter().filter(|block| body.contains(block)) {
            for inst in graph.blocks[block].insts.clone() {
                let data = &graph.insts[inst];
                if data.result.is_some()
                    && !matches!(data.op, Op::Constant(_))
                    && graph.contract(&data.op, &facts).is_pure()
                    && invariant(graph, &data.op)
                {
                    moved.push(inst);
                    graph.move_inst(inst, preheader);
                }
            }
        }
        for inst in moved {
            let data = &graph.insts[inst];
            notes.push(format!(
                "{} = {} hoisted from {header} to {preheader}",
                data.result.unwrap(),
                graph.op_text(&data.op)
            ));
        }
    }
}

fn probe_reads(graph: &Graph, blocks: &[Block]) -> Vec<Inst> {
    let inside: HashSet<Block> = blocks.iter().copied().collect();
    let writes = Writes::collect(graph, &inside);
    let mut reads = Vec::new();
    for &block in blocks {
        for &inst in &graph.blocks[block].insts {
            let op = &graph.insts[inst].op;
            if matches!(op, Op::Get(..) | Op::GetIndex(..))
                && op
                    .operands()
                    .iter()
                    .all(|value| !inside.contains(&graph.def_block(*value)))
                && writes.preserves(op)
            {
                reads.push(inst);
            }
        }
    }
    reads
}

pub(super) fn views(graph: &mut Graph, notes: &mut Vec<String>) {
    let cfg = Cfg::new(graph);
    let facts = Facts::compute(graph);
    let mut loops = natural_loops(graph, &cfg);
    loops.reverse();
    let mut placed: HashMap<(Block, Value), Value> = HashMap::new();
    for &block in &cfg.rpo {
        for inst in graph.blocks[block].insts.clone() {
            let (Op::GetIndex(object, key)
            | Op::TypedIndex(object, key)
            | Op::SetIndex(object, key, _)) = graph.insts[inst].op
            else {
                continue;
            };
            if !facts.of(key).intersects(Types::NUMBER)
                || !facts.of(object).intersects(Types::OBJECT)
            {
                continue;
            }
            let Some(preheader) = loops.iter().find_map(|(header, body)| {
                (body.contains(&block) && !body.contains(&graph.def_block(object)))
                    .then(|| preheader(graph, &cfg, *header, body))
                    .flatten()
            }) else {
                continue;
            };
            let view = match placed.get(&(preheader, object)) {
                Some(&view) => view,
                None => {
                    let span = graph.insts[inst].span;
                    let view = graph
                        .append(preheader, span, Op::View(object), true)
                        .1
                        .unwrap();
                    notes.push(format!("{view} = view {object} in {preheader}"));
                    placed.insert((preheader, object), view);
                    view
                }
            };
            graph.views.insert(inst, view);
        }
    }
}

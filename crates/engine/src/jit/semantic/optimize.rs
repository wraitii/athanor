use std::collections::{HashMap, HashSet};

use oxc_syntax::operator::BinaryOperator;

use super::super::effects::{GLOBAL_WRITE, HEAP_WRITE, Reentry};
use super::{
    Block, BlockData, Cfg, Constant, Facts, Graph, Inst, Native, Op, Result, Target, Terminator,
    TerminatorKind, TypedRead, Types, UnaryOp, Value, ValueDef, resolve,
};
use crate::heap::TypedKind;

pub(super) fn typeof_comparisons(graph: &mut Graph) {
    let uses = graph.use_counts();
    for block in Cfg::new(graph).rpo {
        for inst in graph.blocks[block].insts.clone() {
            let Op::Binary(operator, left, right) = graph.insts[inst].op else {
                continue;
            };
            if !matches!(
                operator,
                BinaryOperator::StrictEquality | BinaryOperator::StrictInequality
            ) {
                continue;
            }
            let pair = [(left, right), (right, left)]
                .into_iter()
                .find_map(|(value, literal)| {
                    let Op::Unary(UnaryOp::Typeof, operand) = graph.defining_op(value)? else {
                        return None;
                    };
                    let Op::Constant(Constant::String(text)) = graph.defining_op(literal)? else {
                        return None;
                    };
                    let kind = [
                        "string",
                        "number",
                        "boolean",
                        "undefined",
                        "object",
                        "function",
                    ]
                    .iter()
                    .position(|name| text == name)?;
                    (uses[value] == 1).then_some((value, literal, *operand, kind as u8))
                });
            let Some((value, literal, operand, kind)) = pair else {
                continue;
            };
            let ValueDef::Result(typeof_inst) = graph.values[value] else {
                unreachable!();
            };
            // Keep TDZ errors at the original typeof evaluation point.
            graph.insts[typeof_inst].op = Op::CheckInitialized(operand);
            graph.insts[typeof_inst].result = None;
            if uses[literal] == 1 {
                let ValueDef::Result(literal_inst) = graph.values[literal] else {
                    unreachable!();
                };
                graph.remove_inst(literal_inst);
            }
            let test = if kind == 0 {
                Op::IsString(operand)
            } else if kind == 1 {
                Op::IsNumber(operand)
            } else {
                Op::IsTypeof(operand, kind)
            };
            graph.insts[inst].op = if operator == BinaryOperator::StrictEquality {
                test
            } else {
                let position = graph.blocks[block]
                    .insts
                    .iter()
                    .position(|&i| i == inst)
                    .unwrap();
                let span = graph.insts[inst].span;
                let (_, result) = graph.insert(block, position, span, test, true);
                Op::Unary(UnaryOp::Not, result.unwrap())
            };
        }
    }
}

pub(super) fn coercions(graph: &mut Graph) {
    let facts = Facts::compute(graph);
    for block in Cfg::new(graph).rpo {
        for inst in graph.blocks[block].insts.clone() {
            let op = &graph.insts[inst].op;
            let bitwise = matches!(
                op,
                Op::Unary(UnaryOp::BitwiseNot, _)
                    | Op::Binary(
                        BinaryOperator::BitwiseAnd
                            | BinaryOperator::BitwiseOR
                            | BinaryOperator::BitwiseXOR
                            | BinaryOperator::ShiftLeft
                            | BinaryOperator::ShiftRight
                            | BinaryOperator::ShiftRightZeroFill,
                        ..
                    )
            );
            if !bitwise {
                continue;
            }
            let operands = op.operands();
            if !operands
                .iter()
                .all(|&value| facts.of(value).within(NUMERIC))
                || operands
                    .iter()
                    .all(|&value| facts.of(value).within(Types::INT32))
            {
                continue;
            }
            let mut op = op.clone();
            let mut position = graph.blocks[block]
                .insts
                .iter()
                .position(|&candidate| candidate == inst)
                .unwrap();
            let span = graph.insts[inst].span;
            for operand in op.operands_mut() {
                if facts.of(*operand).within(Types::INT32) {
                    continue;
                }
                *operand = graph
                    .insert(block, position, span, Op::ToInt32(*operand), true)
                    .1
                    .unwrap();
                position += 1;
            }
            graph.insts[inst].op = op;
        }
    }
}

pub(super) fn bit_tests(graph: &mut Graph) {
    let facts = Facts::compute(graph);
    let uses = graph.use_counts();
    for block in Cfg::new(graph).rpo {
        let terminator = graph.blocks[block].terminator.as_ref().unwrap();
        let TerminatorKind::Branch(condition, _, _) = terminator.kind else {
            continue;
        };
        // The sign-bit mask may be negative; this rewrite preserves only truthiness.
        if uses[condition] != 1 {
            continue;
        }
        let Some(&Op::Binary(BinaryOperator::BitwiseAnd, left, right)) =
            graph.defining_op(condition)
        else {
            continue;
        };
        let test = [(left, right), (right, left)]
            .into_iter()
            .find_map(|(word, mask)| {
                let &Op::Binary(BinaryOperator::ShiftLeft, one, index) = graph.defining_op(mask)?
                else {
                    return None;
                };
                (matches!(
                    graph.defining_op(one),
                    Some(Op::Constant(Constant::Number(1.0)))
                ) && facts.of(word).within(Types::INT32)
                    && facts.of(index).within(Types::INT32))
                .then_some((word, index, one))
            });
        let Some((word, index, one)) = test else {
            continue;
        };
        let ValueDef::Result(inst) = graph.values[condition] else {
            unreachable!();
        };
        let data = &graph.insts[inst];
        let span = data.span;
        let source_name = data.source_name;
        let block = data.block;
        let position = graph.blocks[block]
            .insts
            .iter()
            .position(|&candidate| candidate == inst)
            .unwrap();
        let (shift, shifted) = graph.insert(
            block,
            position,
            span,
            Op::Binary(BinaryOperator::ShiftRight, word, index),
            true,
        );
        graph.insts[shift].source_name = source_name;
        graph.insts[inst].op = Op::Binary(BinaryOperator::BitwiseAnd, shifted.unwrap(), one);
    }
}

pub(super) fn simplify(graph: &mut Graph) {
    loop {
        let facts = Facts::compute(graph);
        let cfg = Cfg::new(graph);
        let uses = graph.use_counts();
        let mut aliases = HashMap::new();
        let mut dead = Vec::new();
        let mut changed = false;
        for &block in &cfg.rpo {
            for &inst in &graph.blocks[block].insts {
                let data = &mut graph.insts[inst];
                match data.op {
                    Op::GuardedCompare {
                        operator,
                        left,
                        right,
                        ..
                    } if [left, right]
                        .iter()
                        .all(|&value| facts.of(value).within(NUMERIC)) =>
                    {
                        data.op = Op::Binary(operator, left, right);
                        changed = true;
                    }
                    Op::IsNumber(value) if facts.of(value).within(Types::NUMBER) => {
                        data.op = Op::Constant(super::Constant::Boolean(true));
                        changed = true;
                    }
                    Op::IsNumber(value) if !facts.of(value).intersects(Types::NUMBER) => {
                        data.op = Op::Constant(super::Constant::Boolean(false));
                        changed = true;
                    }
                    _ => {}
                }
                let data = &graph.insts[inst];
                if let (
                    Some(result),
                    Op::ToNumeric(value) | Op::Unary(UnaryOp::Plus, value) | Op::AsNumber(value),
                ) = (data.result, &data.op)
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
        graph.replace_uses(&aliases);
        for inst in &dead {
            graph.remove_inst(*inst);
        }
        for &block in &cfg.rpo {
            let Some(Terminator {
                kind: TerminatorKind::Branch(condition, yes, no),
                ..
            }) = &graph.blocks[block].terminator
            else {
                continue;
            };
            let Some(Op::Constant(super::Constant::Boolean(taken))) = graph.defining_op(*condition)
            else {
                continue;
            };
            let target = if *taken { yes } else { no }.clone();
            graph.blocks[block].terminator.as_mut().unwrap().kind = TerminatorKind::Jump(target);
            changed = true;
        }
        if !changed && dead.is_empty() {
            return;
        }
    }
}

fn contained_use(op: &Op, index: usize) -> bool {
    match op {
        Op::Get(..)
        | Op::GetIndex(..)
        | Op::TypedIndex(..)
        | Op::IndexedGet(..)
        | Op::Set(..)
        | Op::SetIndex(..)
        | Op::Define(..) => index == 0,
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
                Op::NewObject | Op::NewInstance(_) | Op::NewArray(_) => {
                    allocations.extend(data.result)
                }
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
    // Cells can alias other captures, but cannot alias JavaScript properties.
    Cells,
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
            (Self::Cells, Self::Cells) => true,
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
            (_, Access::Cells | Access::Elements | Access::Iteration) => false,
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
    let mut reasons = Vec::new();
    if let Some(region) = region(graph, &blocks, None, "", &mut reasons)? {
        notes.extend(reasons);
        return Ok(vec![region]);
    }
    let mut checked = Vec::new();
    if let Some((copy, _)) =
        checked_region(graph, &blocks, None, &blocks, "function: ", &mut checked)?
    {
        notes.extend(checked);
        return Ok(vec![copy]);
    }
    notes.extend(reasons);
    notes.extend(checked);
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
    let speculation = Speculation {
        requirements: &requirements,
        ..Default::default()
    };
    let (fast, _) = apply_version(&mut candidate, blocks, entry, &speculation);
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

const LIFT: usize = 2;

/// Blocks of loop `index` whose guards may be checked on entry to it: those at most `LIFT` loops
/// deep, where each nested loop on the way is entered on every iteration of its parent.
fn placed_blocks(graph: &Graph, loops: &[(Block, HashSet<Block>)], index: usize) -> HashSet<Block> {
    let (_, body) = &loops[index];
    let mut entered = HashMap::new();
    body.iter()
        .copied()
        .filter(|&block| {
            let chain: Vec<usize> = (0..=index)
                .filter(|&inner| loops[inner].1.contains(&block))
                .collect();
            chain.len() <= LIFT
                && chain.windows(2).all(|pair| {
                    *entered
                        .entry((pair[1], pair[0]))
                        .or_insert_with(|| enters(graph, &loops[pair[1]], loops[pair[0]].0))
                })
        })
        .collect()
}

fn enters(graph: &Graph, (header, body): &(Block, HashSet<Block>), inner: Block) -> bool {
    let mut visited = HashSet::new();
    let mut pending = graph.successors(*header);
    while let Some(block) = pending.pop() {
        if block == *header {
            return false;
        }
        if block != inner && body.contains(&block) && visited.insert(block) {
            pending.extend(graph.successors(block));
        }
    }
    true
}

fn checked_region(
    graph: &mut Graph,
    blocks: &[Block],
    entry: Option<(Block, Block)>,
    scan: &[Block],
    label: &str,
    notes: &mut Vec<String>,
) -> Result<Option<(HashSet<Block>, Block)>> {
    let body: HashSet<_> = blocks.iter().copied().collect();
    let branches: Vec<_> = blocks
        .iter()
        .copied()
        .filter(|&block| guard_fallback(graph, block).is_some())
        .collect();
    let fallbacks: HashSet<_> = branches
        .iter()
        .filter_map(|&block| guard_fallback(graph, block))
        .collect();
    let written: HashSet<_> = body.difference(&fallbacks).copied().collect();
    let requirements = requirements(graph, blocks);
    let guarded = Guarded::new(&requirements);
    let mut planning = graph.clone();
    // These effects apply only after the native identity checks at region entry.
    for requirement in &requirements {
        let Op::Call { args, .. } = &graph.insts[requirement.call].op else {
            unreachable!();
        };
        planning.insts[requirement.call].op = Op::CallNative(requirement.native, args.clone());
    }
    if closure_failure(&planning, &written, &guarded).is_some() {
        return Ok(None);
    }
    let probes = probe_reads(&planning, &body, &written, scan, entry.is_none());
    let checks = entry_checks(&planning, &body, scan, &probes);
    let guards = invariant_guards(&planning, &body, scan, &probes);
    if requirements.is_empty() && checks.is_empty() && probes.is_empty() && guards.is_empty() {
        return Ok(None);
    }
    let speculation = Speculation {
        requirements: &requirements,
        checks: &checks,
        reads: &probes,
        guards: &guards,
        exits: &branches,
        ..Default::default()
    };
    let mut candidate = graph.clone();
    let (copy, header) = apply_version(&mut candidate, blocks, entry, &speculation);
    candidate.verify()?;
    if let Some(reason) = closure_failure(&candidate, &copy, &guarded) {
        notes.push(format!("{label}not versioned: guarded copy {reason}"));
        return Ok(None);
    }
    let list: Vec<_> = checks.iter().map(ToString::to_string).collect();
    notes.push(format!(
        "{label}versioned: number checks on [{}], {} read probes and {} hoisted guards select the closed copy",
        list.join(", "), probes.len(), guards.len()
    ));
    *graph = candidate;
    Ok(Some((copy, header)))
}

pub(super) fn check_loops(
    graph: &mut Graph,
    regions: &mut Vec<HashSet<Block>>,
    notes: &mut Vec<String>,
) -> Result<Vec<HashSet<Block>>> {
    let mut copies = Vec::new();
    let mut tried = HashSet::new();
    loop {
        let cfg = Cfg::new(graph);
        let loops = natural_loops(graph, &cfg);
        let Some(index) = (0..loops.len()).rev().find(|&index| {
            let header = loops[index].0;
            !tried.contains(&header) && !graph.cold.contains(&header)
        }) else {
            return Ok(copies);
        };
        let (header, body) = loops[index].clone();
        tried.insert(header);
        let Some(preheader) = preheader(graph, &cfg, header, &body) else {
            continue;
        };
        let placed = placed_blocks(graph, &loops, index);
        let blocks: Vec<Block> = cfg
            .rpo
            .iter()
            .copied()
            .filter(|block| body.contains(block))
            .collect();
        let scan: Vec<Block> = blocks
            .iter()
            .copied()
            .filter(|block| placed.contains(block))
            .collect();
        let closed = regions.iter().any(|region| body.is_subset(region));
        let mut candidate = graph.clone();
        let label = format!("loop at {header}: ");
        let numbered = checked_region(
            &mut candidate,
            &blocks,
            Some((preheader, header)),
            &scan,
            &label,
            notes,
        )?;
        let (layer_body, layer_header) = match &numbered {
            Some((copy, copy_header)) => (copy.clone(), *copy_header),
            None => (body.clone(), header),
        };
        if !closed && numbered.is_none() {
            continue;
        }
        let typed = typed_version(&mut candidate, layer_header, &layer_body, notes)?;
        if numbered.is_none() && typed.is_none() {
            continue;
        }
        *graph = candidate;
        let mut added: Vec<HashSet<Block>> = Vec::new();
        if let Some((copy, copy_header)) = numbered {
            tried.insert(copy_header);
            added.push(copy);
        }
        if let Some((copy, copy_header)) = typed {
            tried.insert(copy_header);
            added.push(copy);
        }
        if !closed {
            regions.push(HashSet::new());
        }
        for copy in added {
            reuse_loads(graph, &copy, notes);
            for region in regions.iter_mut() {
                if body.is_subset(region) || !closed && region.is_empty() {
                    region.extend(copy.iter().copied());
                }
            }
            copies.push(copy);
        }
    }
}

fn guard_fallback(graph: &Graph, block: Block) -> Option<Block> {
    let Some(Terminator {
        kind: TerminatorKind::Branch(condition, _, otherwise),
        ..
    }) = &graph.blocks[block].terminator
    else {
        return None;
    };
    let guard = matches!(
        graph.defining_op(*condition),
        Some(
            Op::IsCode(..)
                | Op::IsNumber(_)
                | Op::IsTyped(..)
                | Op::IsIndexed(_)
                | Op::IsArrayIteration(..)
        )
    );
    (guard && graph.cold.contains(&otherwise.block)).then_some(otherwise.block)
}

fn invariant_guards(
    graph: &Graph,
    inside: &HashSet<Block>,
    scan: &[Block],
    probes: &[Inst],
) -> Vec<Block> {
    let probed: HashSet<Value> = probes
        .iter()
        .filter_map(|inst| graph.insts[*inst].result)
        .collect();
    let invariant =
        |value: Value| !inside.contains(&graph.def_block(value)) || probed.contains(&value);
    scan.iter()
        .copied()
        .filter(|&block| {
            guard_fallback(graph, block).is_some()
                && matches!(
                    &graph.blocks[block].terminator,
                    Some(Terminator { kind: TerminatorKind::Branch(condition, ..), .. })
                        if invariant(*condition)
                            || graph.defining_op(*condition).unwrap().operands().into_iter().all(invariant)
                )
        })
        .collect()
}

const NUMERIC: Types = Types::NUMBER.union(Types::UNDEFINED);

fn receiver_use(op: &Op, index: usize) -> bool {
    match op {
        Op::Get(..)
        | Op::GetIndex(..)
        | Op::TypedIndex(..)
        | Op::IndexedGet(..)
        | Op::Set(..)
        | Op::SetIndex(..)
        | Op::Define(..)
        | Op::Construct { .. }
        | Op::ConstructResolved { .. }
        | Op::IteratorNew(_)
        | Op::EnumerateNew(_) => index == 0,
        Op::Call { receiver, .. } => index == 0 || (index == 1 && receiver.is_some()),
        _ => false,
    }
}

fn typed_objects(graph: &Graph, body: &HashSet<Block>, scan: &HashSet<Block>) -> Vec<Value> {
    let facts = Facts::compute(graph);
    let mut reads = HashMap::new();
    for &block in scan {
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
) -> Result<Option<(HashSet<Block>, Block)>> {
    let cfg = Cfg::new(graph);
    let Some(preheader) = preheader(graph, &cfg, header, body) else {
        return Ok(None);
    };
    let loops = natural_loops(graph, &cfg);
    let Some(index) = loops.iter().position(|(candidate, _)| *candidate == header) else {
        return Ok(None);
    };
    let objects = typed_objects(graph, body, &placed_blocks(graph, &loops, index));
    if objects.is_empty() {
        return Ok(None);
    }
    let blocks: Vec<Block> = cfg
        .rpo
        .iter()
        .copied()
        .filter(|block| body.contains(block))
        .collect();
    let typed: Vec<(Value, Option<TypedKind>)> = objects
        .iter()
        .map(|&object| (object, element_guess(graph, object)))
        .collect();
    let mut bounds: Vec<(Value, Value)> = bounded_reads(graph, &cfg, body, &objects)
        .into_values()
        .collect();
    bounds.sort();
    bounds.dedup();
    let mut candidate = graph.clone();
    let speculation = Speculation {
        typed: &typed,
        bounds: &bounds,
        ..Default::default()
    };
    let (copy, copy_header) = apply_version(
        &mut candidate,
        &blocks,
        Some((preheader, header)),
        &speculation,
    );
    let facts = Facts::compute(&candidate);
    let bounded = bounded_reads(&candidate, &Cfg::new(&candidate), &copy, &objects);
    for &block in &copy {
        for inst in candidate.blocks[block].insts.clone() {
            if let Op::GetIndex(object, key) = candidate.insts[inst].op
                && let Some(&(_, element)) = typed.iter().find(|(typed, _)| *typed == object)
                && facts.of(key).within(NUMERIC)
            {
                let read = TypedRead {
                    element,
                    bounded: bounded.contains_key(&inst),
                };
                candidate.insts[inst].op = Op::TypedIndex(object, key, read);
            }
        }
    }
    candidate.verify()?;
    if closure_failure(&candidate, &copy, &Guarded::default()).is_some() {
        return Ok(None);
    }
    let list: Vec<String> = typed
        .iter()
        .map(|(object, element)| match element {
            Some(element) => format!("{object} as {element:?}"),
            None => object.to_string(),
        })
        .collect();
    notes.push(format!(
        "loop at {header}: typed array checks on [{}] and {} length checks select the typed copy",
        list.join(", "),
        bounds.len()
    ));
    *graph = candidate;
    Ok(Some((copy, copy_header)))
}

fn element_guess(graph: &Graph, object: Value) -> Option<TypedKind> {
    element_guess_at(graph, object, 0)
}

fn element_guess_at(graph: &Graph, object: Value, depth: usize) -> Option<TypedKind> {
    if depth > 4 {
        return None;
    }
    if let ValueDef::Param(block, index) = graph.values[object] {
        return graph
            .blocks
            .values()
            .filter_map(|data| data.terminator.as_ref())
            .flat_map(|term| term.kind.targets())
            .filter(|target| target.block == block)
            .find_map(|target| element_guess_at(graph, *target.args.get(index)?, depth + 1));
    }
    if let Some(Op::ConstructorResult(result, _)) = graph.defining_op(object) {
        return element_guess_at(graph, *result, depth + 1);
    }
    let constructor = match graph.defining_op(object)? {
        Op::Construct { callee, .. }
        | Op::ConstructResolved { callee, .. }
        | Op::ConstructorCall { callee, .. } => *callee,
        Op::Call { callee, .. } => match graph.defining_op(*callee)? {
            Op::Get(constructor, name) if name == "from" || name == "of" => *constructor,
            _ => return None,
        },
        _ => return None,
    };
    let Op::GlobalRead(name) = graph.defining_op(constructor)? else {
        return None;
    };
    Some(match name.as_str() {
        "Uint8Array" => TypedKind::Uint8,
        "Uint16Array" => TypedKind::Uint16,
        "Uint32Array" => TypedKind::Uint32,
        "Int8Array" => TypedKind::Int8,
        "Int16Array" => TypedKind::Int16,
        "Int32Array" => TypedKind::Int32,
        "Float32Array" => TypedKind::Float32,
        "Float64Array" => TypedKind::Float64,
        _ => return None,
    })
}

/// Index reads on `objects` inside `region` whose key is a loop counter starting at a
/// non-negative integer, stepping by one and tested against a bound defined before the region,
/// mapped to their receiver and that bound.
fn bounded_reads(
    graph: &Graph,
    cfg: &Cfg,
    region: &HashSet<Block>,
    objects: &[Value],
) -> HashMap<Inst, (Value, Value)> {
    let facts = Facts::compute(graph);
    let mut counters = HashMap::new();
    for (header, body) in natural_loops(graph, cfg) {
        if !region.contains(&header) {
            continue;
        }
        let Some(Terminator {
            kind: TerminatorKind::Branch(condition, inside, outside),
            ..
        }) = &graph.blocks[header].terminator
        else {
            continue;
        };
        let Some(
            &(Op::Binary(BinaryOperator::LessThan, counter, bound)
            | Op::GuardedCompare {
                operator: BinaryOperator::LessThan,
                left: counter,
                right: bound,
                ..
            }),
        ) = graph.defining_op(*condition)
        else {
            continue;
        };
        if !body.contains(&inside.block)
            || body.contains(&outside.block)
            || region.contains(&graph.def_block(bound))
            || !facts.of(bound).within(Types::NUMBER)
            || !counts_up(graph, cfg, header, &body, counter)
        {
            continue;
        }
        counters.insert(counter, (inside.block, bound));
    }
    let mut reads = HashMap::new();
    for &block in region {
        for &inst in &graph.blocks[block].insts {
            if let Op::GetIndex(object, key) | Op::TypedIndex(object, key, _) = graph.insts[inst].op
                && objects.contains(&object)
                && let Some(&(inside, bound)) = counters.get(&key)
                && cfg.dominates(inside, block)
            {
                reads.insert(inst, (object, bound));
            }
        }
    }
    reads
}

fn counts_up(
    graph: &Graph,
    cfg: &Cfg,
    header: Block,
    body: &HashSet<Block>,
    counter: Value,
) -> bool {
    let Some(index) = graph.blocks[header]
        .params
        .iter()
        .position(|&param| param == counter)
    else {
        return false;
    };
    cfg.preds[header].iter().all(|&pred| {
        let terminator = graph.blocks[pred].terminator.as_ref().unwrap();
        terminator
            .kind
            .targets()
            .iter()
            .filter(|target| target.block == header)
            .all(|target| {
                let arg = target.args[index];
                match graph.defining_op(arg) {
                    Some(Op::Constant(Constant::Number(start))) => {
                        !body.contains(&pred) && *start >= 0.0 && start.fract() == 0.0
                    }
                    Some(&Op::Unary(UnaryOp::Increment, operand)) => operand == counter,
                    Some(&Op::Binary(BinaryOperator::Addition, operand, step)) => {
                        operand == counter
                            && matches!(
                                graph.defining_op(step),
                                Some(Op::Constant(Constant::Number(step))) if *step == 1.0
                            )
                    }
                    _ => false,
                }
            })
    })
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
        Op::GuardedCompare { .. } => index < 2,
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
        Op::GetIndex(..) | Op::TypedIndex(..) | Op::IndexedGet(..) | Op::SetIndex(..) => index == 1,
        _ => false,
    }
}

fn entry_checks(
    graph: &Graph,
    inside: &HashSet<Block>,
    scan: &[Block],
    probes: &[Inst],
) -> Vec<Value> {
    let facts = Facts::compute(graph);
    let probed: HashSet<Value> = probes
        .iter()
        .filter_map(|inst| graph.insts[*inst].result)
        .collect();
    let mut checks = HashSet::new();
    for &block in scan {
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

/// Entry conditions for a versioned copy, checked in this order.
#[derive(Default)]
struct Speculation<'a> {
    requirements: &'a [Requirement],
    reads: &'a [Inst],
    guards: &'a [Block],
    exits: &'a [Block],
    typed: &'a [(Value, Option<TypedKind>)],
    bounds: &'a [(Value, Value)],
    checks: &'a [Value],
}

#[derive(PartialEq, Eq, Hash)]
enum EntryGuard {
    Code(Value, cranelift_module::FuncId),
    Number(Value),
    Value(Value),
}

fn apply_version(
    graph: &mut Graph,
    region: &[Block],
    entry: Option<(Block, Block)>,
    speculation: &Speculation<'_>,
) -> (HashSet<Block>, Block) {
    let Speculation {
        requirements,
        reads,
        guards,
        exits,
        typed,
        bounds,
        checks,
    } = *speculation;
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
            graph.insts[copy].source_name = data.source_name;
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
    // A varying guard leaves the copy before reentry; SSA repair carries its current values.
    for &block in exits {
        if guards.contains(&block) {
            continue;
        }
        let fallback = guard_fallback(graph, block).unwrap();
        let terminator = graph.blocks[blocks[&block]].terminator.as_mut().unwrap();
        let TerminatorKind::Branch(_, _, otherwise) = &mut terminator.kind else {
            unreachable!();
        };
        otherwise.block = fallback;
    }
    for &block in region {
        if graph.cold.contains(&block) {
            graph.cold.insert(blocks[&block]);
        }
    }
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
    let mut probes = HashMap::new();
    let mut available = HashMap::new();
    for &inst in reads {
        let data = graph.insts[inst].clone();
        let mut op = match data.op {
            Op::Get(object, name) => Op::ProbeProperty(object, name),
            Op::GetIndex(object, key) | Op::IndexedGet(object, key) => Op::ProbeIndex(object, key),
            Op::GlobalRead(name) => Op::ProbeGlobal(name),
            Op::Argument(index) => Op::Argument(index),
            Op::This => Op::This,
            _ => unreachable!(),
        };
        for value in op.operands_mut() {
            if let Some(&probe) = probes.get(value) {
                *value = probe;
            }
        }
        let key = match &op {
            Op::ProbeGlobal(name) => LoadKey::Global(name.clone()),
            Op::ProbeProperty(object, name) => LoadKey::Property(*object, name.clone()),
            Op::ProbeIndex(object, key) => LoadKey::Index(*object, *key),
            Op::Argument(index) => LoadKey::Argument(*index),
            Op::This => LoadKey::This,
            _ => unreachable!(),
        };
        let value = match available.get(&key) {
            Some(&value) => value,
            None => {
                let input = matches!(op, Op::Argument(_) | Op::This);
                let value = graph.append(current, data.span, op, true).1.unwrap();
                if !input {
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
                }
                available.insert(key, value);
                value
            }
        };
        let result = data.result.unwrap();
        graph.replace_uses(&HashMap::from([(values[&result], value)]));
        graph.remove_inst(insts[&inst]);
        values.insert(result, value);
        probes.insert(result, value);
    }
    let copied: HashSet<Block> = region.iter().copied().collect();
    let mut tested = HashSet::new();
    for &block in guards {
        let Some(Terminator {
            span,
            kind: TerminatorKind::Branch(condition, _, _),
        }) = graph.blocks[block].terminator.clone()
        else {
            unreachable!();
        };
        let copy = blocks[&block];
        let terminator = graph.blocks[copy].terminator.as_mut().unwrap();
        let TerminatorKind::Branch(_, taken, _) = &terminator.kind else {
            unreachable!();
        };
        terminator.kind = TerminatorKind::Jump(taken.clone());
        let (test, key) = if copied.contains(&graph.def_block(condition)) {
            let mut op = graph.defining_op(condition).unwrap().clone();
            for value in op.operands_mut() {
                if copied.contains(&graph.def_block(*value)) {
                    *value = values[value];
                }
            }
            let key = match op {
                Op::IsCode(value, code) => Some(EntryGuard::Code(value, code)),
                Op::IsNumber(value) => Some(EntryGuard::Number(value)),
                _ => None,
            };
            let ValueDef::Result(inst) = graph.values[condition] else {
                unreachable!();
            };
            graph.insts[insts[&inst]].op = Op::Constant(Constant::Boolean(true));
            (Some(op), key)
        } else {
            (None, Some(EntryGuard::Value(condition)))
        };
        if key.is_some_and(|key| !tested.insert(key)) {
            continue;
        }
        let condition = test.map_or(condition, |op| {
            graph.append(current, span, op, true).1.unwrap()
        });
        let next = graph.blocks.push(BlockData::default());
        graph.blocks[current].terminator = Some(Terminator {
            span,
            kind: TerminatorKind::Branch(condition, Target::new(next), original.clone()),
        });
        current = next;
    }
    for &(object, element) in typed {
        let check = graph
            .append(current, span, Op::IsTyped(object, element), true)
            .1
            .unwrap();
        let next = graph.blocks.push(BlockData::default());
        graph.blocks[current].terminator = Some(Terminator {
            span,
            kind: TerminatorKind::Branch(check, Target::new(next), original.clone()),
        });
        current = next;
    }
    for &(object, bound) in bounds {
        let length = graph
            .append(current, span, Op::IndexedLength(object), true)
            .1
            .unwrap();
        let fits = Op::Binary(BinaryOperator::LessEqualThan, bound, length);
        let fits = graph.append(current, span, fits, true).1.unwrap();
        let next = graph.blocks.push(BlockData::default());
        graph.blocks[current].terminator = Some(Terminator {
            span,
            kind: TerminatorKind::Branch(fits, Target::new(next), original.clone()),
        });
        current = next;
    }
    for &value in checks {
        let checked = values.get(&value).copied().unwrap_or(value);
        if !tested.insert(EntryGuard::Number(checked)) {
            continue;
        }
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
    // Disconnected fallbacks must not contribute writes to the closed copy.
    let reachable = Cfg::new(graph)
        .rpo
        .into_iter()
        .filter(|block| copies.contains(block))
        .collect();
    (reachable, blocks[&head])
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
    ToInt32(Value),
    Constructor(Value),
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
                Op::GetIndex(object, key)
                | Op::TypedIndex(object, key, _)
                | Op::IndexedGet(object, key) => {
                    LoadKey::Index(resolve(&aliases, *object), resolve(&aliases, *key))
                }
                Op::ConstructorPrototype(callee) => {
                    LoadKey::Constructor(resolve(&aliases, *callee))
                }
                Op::GlobalRead(name) => LoadKey::Global(name.clone()),
                Op::ToInt32(value) => LoadKey::ToInt32(resolve(&aliases, *value)),
                Op::This => LoadKey::This,
                Op::Argument(index) => LoadKey::Argument(*index),
                Op::Binary(operator, left, right)
                | Op::GuardedCompare {
                    operator,
                    left,
                    right,
                    ..
                } if graph.contract(&data.op, facts).is_pure() => LoadKey::Binary(
                    *operator,
                    resolve(&aliases, *left),
                    resolve(&aliases, *right),
                ),
                _ => continue,
            };
            let preserved = match &key {
                LoadKey::Property(object, _)
                | LoadKey::Index(object, _)
                | LoadKey::Constructor(object) => {
                    !writes.clobbers(*object, &Access::of(&data.op, facts))
                }
                LoadKey::Global(name) => !writes.clobbers_global(name),
                LoadKey::This
                | LoadKey::Argument(_)
                | LoadKey::Binary(..)
                | LoadKey::ToInt32(_) => true,
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

fn probe_reads(
    graph: &Graph,
    inside: &HashSet<Block>,
    written: &HashSet<Block>,
    scan: &[Block],
    inputs: bool,
) -> Vec<Inst> {
    let writes = Writes::collect(graph, written);
    let mut reads = Vec::new();
    let mut probed = HashSet::new();
    for &block in scan {
        for &inst in &graph.blocks[block].insts {
            let data = &graph.insts[inst];
            let input = inputs && matches!(data.op, Op::Argument(_) | Op::This);
            if (input
                || matches!(
                    data.op,
                    Op::Get(..) | Op::GetIndex(..) | Op::IndexedGet(..) | Op::GlobalRead(_)
                ))
                && data.op.operands().iter().all(|value| {
                    !inside.contains(&graph.def_block(*value)) || probed.contains(value)
                })
                && (input || writes.preserves(&data.op))
            {
                reads.push(inst);
                probed.extend(data.result);
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
            | Op::TypedIndex(object, key, _)
            | Op::IndexedGet(object, key)
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

pub(super) fn sink_closures(graph: &mut Graph, notes: &mut Vec<String>) {
    let cfg = Cfg::new(graph);
    let loops = natural_loops(graph, &cfg);
    let mut uses: HashMap<Value, HashSet<Block>> = HashMap::new();
    let mut closures = Vec::new();
    for &block in &cfg.rpo {
        for &inst in &graph.blocks[block].insts {
            let data = &graph.insts[inst];
            if matches!(&data.op, Op::Closure(_, _, false, _)) {
                closures.push(inst);
            }
            for value in data.op.operands() {
                uses.entry(value).or_default().insert(block);
            }
        }
        for value in graph.blocks[block]
            .terminator
            .as_ref()
            .unwrap()
            .kind
            .operands()
        {
            uses.entry(value).or_default().insert(block);
        }
    }
    for inst in closures {
        let value = graph.insts[inst].result.unwrap();
        let Some(users) = uses.get(&value) else {
            graph.remove_inst(inst);
            notes.push(format!("unused closure {value} removed after inlining"));
            continue;
        };
        if users.len() != 1 {
            continue;
        }
        let target = *users.iter().next().unwrap();
        let origin = graph.insts[inst].block;
        if target == origin
            || !graph.cold.contains(&target)
            || loops
                .iter()
                .any(|(_, body)| body.contains(&origin) != body.contains(&target))
        {
            continue;
        }
        if matches!(&graph.insts[inst].op, Op::Closure(_, _, _, captures) if !captures.is_empty()) {
            let user = graph.blocks[target]
                .insts
                .iter()
                .copied()
                .find(|&user| graph.insts[user].op.operands().contains(&value));
            if !user
                .is_some_and(|user| super::inlining::direct_captures(graph, user, value).is_some())
            {
                continue;
            }
        }
        graph.remove_inst(inst);
        let position = graph.blocks[target]
            .insts
            .iter()
            .position(|&user| graph.insts[user].op.operands().contains(&value))
            .unwrap_or(graph.blocks[target].insts.len());
        graph.blocks[target].insts.insert(position, inst);
        graph.insts[inst].block = target;
        notes.push(format!(
            "closure {value} materialized only in fallback {target}"
        ));
    }
}

pub(super) fn lazy_captures(graph: &mut Graph, notes: &mut Vec<String>) -> bool {
    let mut demoted = false;
    let mut candidates = HashSet::new();
    let mut eager = HashSet::new();
    for block in Cfg::new(graph).rpo {
        for &inst in &graph.blocks[block].insts {
            match &graph.insts[inst].op {
                Op::Closure(_, _, false, captures) if graph.cold.contains(&block) => {
                    candidates.extend(captures.iter().copied());
                }
                Op::Closure(_, _, _, captures) => eager.extend(captures.iter().copied()),
                _ => {}
            }
        }
    }
    for (binding, data) in &mut graph.bindings {
        if data.captured
            && graph.scopes[data.scope].kind != super::ScopeKind::Captures
            && !eager.contains(&binding)
        {
            if candidates.contains(&binding) {
                if graph.lazy_captures.insert(binding) {
                    notes.push(format!(
                        "binding {binding} materialized as a cell only on closure creation"
                    ));
                }
            } else {
                data.captured = false;
                graph.lazy_captures.remove(&binding);
                demoted = true;
                notes.push(format!("binding {binding} no longer needs a capture cell"));
            }
        }
    }
    demoted
}

use std::collections::{HashMap, HashSet};

use oxc_syntax::operator::BinaryOperator;

use super::access::Access;
use super::{Block, Cfg, Graph, Inst, Op, TerminatorKind, Value};

pub(super) struct Fresh {
    classes: HashMap<Value, Value>,
    positions: HashMap<Inst, (Block, usize)>,
    entering: HashMap<Block, HashSet<Value>>,
    first: HashMap<Block, HashMap<Value, usize>>,
}

fn contained_use(op: &Op, index: usize) -> bool {
    match op {
        Op::Observe(..) | Op::ReadHint(..) | Op::Assume(..) => true,
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

fn terminal_result(graph: &Graph, blocks: &[Block], value: Value) -> bool {
    let mut pending = vec![value];
    let mut visited = HashSet::new();
    while let Some(value) = pending.pop() {
        if !visited.insert(value) {
            continue;
        }
        for &block in blocks {
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
                    if arg == value {
                        pending.push(graph.blocks[target.block].params[index]);
                    }
                }
            }
        }
    }
    true
}

fn root(parents: &mut HashMap<Value, Value>, value: Value) -> Value {
    let mut current = value;
    while let Some(&parent) = parents.get(&current) {
        if parent == current {
            break;
        }
        current = parent;
    }
    let mut walk = value;
    while walk != current {
        let next = parents[&walk];
        parents.insert(walk, current);
        walk = next;
    }
    current
}

impl Fresh {
    pub(super) fn compute(graph: &Graph, cfg: &Cfg) -> Self {
        let reachable: HashSet<Block> = cfg.rpo.iter().copied().collect();
        let mut allocations = HashSet::new();
        let mut sources: HashMap<Value, Value> = HashMap::new();
        let mut edges = Vec::new();
        for &block in &cfg.rpo {
            if block != graph.entry && cfg.preds[block].iter().all(|pred| reachable.contains(pred))
            {
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
                            sources.insert(iterator, source);
                        }
                    }
                    _ => {}
                }
            }
            if let Some(terminator) = &graph.blocks[block].terminator {
                for target in terminator.kind.targets() {
                    for (&arg, &param) in target.args.iter().zip(&graph.blocks[target.block].params)
                    {
                        edges.push((block, arg, param));
                    }
                }
            }
        }
        let mut changed = true;
        while changed {
            changed = false;
            for &(_, arg, param) in &edges {
                if !allocations.contains(&arg) && allocations.remove(&param) {
                    changed = true;
                }
            }
        }
        let mut parents: HashMap<Value, Value> = allocations.iter().map(|&v| (v, v)).collect();
        for &(_, arg, param) in &edges {
            if allocations.contains(&arg) && allocations.contains(&param) {
                let (left, right) = (root(&mut parents, arg), root(&mut parents, param));
                parents.insert(left, right);
            }
        }
        let values: Vec<Value> = allocations.iter().copied().collect();
        let classes: HashMap<Value, Value> = values
            .into_iter()
            .map(|value| (value, root(&mut parents, value)))
            .collect();
        let mut implied: HashMap<Value, Vec<Value>> = HashMap::new();
        for (&iterator, &source) in &sources {
            if let (Some(&from), Some(&to)) = (classes.get(&iterator), classes.get(&source)) {
                implied.entry(from).or_default().push(to);
            }
        }
        let escape = |class: Value, into: &mut Vec<Value>| {
            let mut pending = vec![class];
            let mut seen = HashSet::new();
            while let Some(class) = pending.pop() {
                if !seen.insert(class) {
                    continue;
                }
                into.push(class);
                pending.extend(implied.get(&class).into_iter().flatten().copied());
            }
        };
        let mut positions = HashMap::new();
        let mut first: HashMap<Block, HashMap<Value, usize>> = HashMap::new();
        for &block in &cfg.rpo {
            let insts = &graph.blocks[block].insts;
            let events = first.entry(block).or_default();
            for (index, &inst) in insts.iter().enumerate() {
                positions.insert(inst, (block, index));
                let op = &graph.insts[inst].op;
                for (position, value) in op.operands_iter().enumerate() {
                    let Some(&class) = classes.get(&value) else {
                        continue;
                    };
                    let terminal_constructor = position == 1
                        && matches!(op, Op::ConstructorResult(..))
                        && graph.insts[inst]
                            .result
                            .is_some_and(|result| terminal_result(graph, &cfg.rpo, result));
                    if contained_use(op, position) || terminal_constructor {
                        continue;
                    }
                    let mut escaped = Vec::new();
                    escape(class, &mut escaped);
                    for class in escaped {
                        events.entry(class).or_insert(index);
                    }
                }
            }
        }
        for &(block, arg, param) in &edges {
            if allocations.contains(&param) {
                continue;
            }
            let Some(&class) = classes.get(&arg) else {
                continue;
            };
            let end = graph.blocks[block].insts.len();
            let mut escaped = Vec::new();
            escape(class, &mut escaped);
            let events = first.entry(block).or_default();
            for class in escaped {
                events.entry(class).or_insert(end);
            }
        }
        let mut entering: HashMap<Block, HashSet<Value>> = HashMap::new();
        let mut changed = true;
        while changed {
            changed = false;
            for &block in &cfg.rpo {
                let mut out = entering.get(&block).cloned().unwrap_or_default();
                out.extend(first[&block].keys().copied());
                let Some(terminator) = &graph.blocks[block].terminator else {
                    continue;
                };
                for target in terminator.kind.targets() {
                    let set = entering.entry(target.block).or_default();
                    let before = set.len();
                    set.extend(out.iter().copied());
                    changed |= set.len() != before;
                }
            }
        }
        Self {
            classes,
            positions,
            entering,
            first,
        }
    }

    pub(super) fn class(&self, value: Value) -> Option<Value> {
        self.classes.get(&value).copied()
    }

    pub(super) fn may_alias(&self, inst: Inst, write: &Access, read: &Access) -> bool {
        match write.base.and_then(|target| self.at(target, inst)) {
            Some(class) => Some(class) == read.base.and_then(|base| self.class(base)),
            None => true,
        }
    }

    pub(super) fn at(&self, value: Value, inst: Inst) -> Option<Value> {
        let class = self.class(value)?;
        let &(block, index) = self.positions.get(&inst)?;
        if self
            .entering
            .get(&block)
            .is_some_and(|set| set.contains(&class))
        {
            return None;
        }
        match self.first.get(&block).and_then(|events| events.get(&class)) {
            Some(&escaped) if escaped <= index => None,
            _ => Some(class),
        }
    }
}

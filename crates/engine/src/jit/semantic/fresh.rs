use cranelift_codegen::entity::SecondaryMap;
use oxc_syntax::operator::BinaryOperator;

use super::access::Access;
use super::{Block, Cfg, Graph, Inst, Op, TerminatorKind, Value};

fn contained_use(op: &Op, index: usize) -> bool {
    match op {
        Op::Observe(..) | Op::ReadHint(..) | Op::Assume(..) | Op::EntryAssume(..) => true,
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

const NONE: u32 = u32::MAX;

fn root(parents: &mut [u32], value: u32) -> u32 {
    let mut current = value;
    while parents[current as usize] != current {
        current = parents[current as usize];
    }
    let mut walk = value;
    while walk != current {
        let next = parents[walk as usize];
        parents[walk as usize] = current;
        walk = next;
    }
    current
}

struct Terminal<'a> {
    graph: &'a Graph,
    cfg: &'a Cfg,
    edges: &'a [(Block, Value, Value)],
    used: Option<SecondaryMap<Value, bool>>,
}

impl Terminal<'_> {
    fn used(&mut self) -> &SecondaryMap<Value, bool> {
        let (graph, cfg) = (self.graph, self.cfg);
        self.used.get_or_insert_with(|| {
            let mut used = SecondaryMap::new();
            for &block in &cfg.rpo {
                let data = &graph.blocks[block];
                for &inst in &data.insts {
                    for operand in graph.insts[inst].op.operands_iter() {
                        used[operand] = true;
                    }
                }
                match data.terminator.as_ref().map(|terminator| &terminator.kind) {
                    Some(TerminatorKind::Branch(condition, ..)) => used[*condition] = true,
                    Some(TerminatorKind::Yield(_, yielded, _)) => used[*yielded] = true,
                    _ => {}
                }
            }
            used
        })
    }

    fn result(&mut self, value: Value) -> bool {
        let mut pending = vec![value];
        let mut visited = Vec::new();
        while let Some(value) = pending.pop() {
            if visited.contains(&value) {
                continue;
            }
            visited.push(value);
            if self.used()[value] {
                return false;
            }
            pending.extend(
                self.edges
                    .iter()
                    .filter(|&&(_, arg, _)| arg == value)
                    .map(|&(_, _, param)| param),
            );
        }
        true
    }
}

pub(super) struct Fresh {
    classes: SecondaryMap<Value, u32>,
    positions: SecondaryMap<Inst, Option<(Block, u32)>>,
    words: usize,
    entering: SecondaryMap<Block, Vec<u64>>,
    first: SecondaryMap<Block, Vec<(u32, u32)>>,
}

impl Fresh {
    pub(super) fn compute(graph: &Graph, cfg: &Cfg) -> Self {
        let mut reachable = SecondaryMap::<Block, bool>::new();
        for &block in &cfg.rpo {
            reachable[block] = true;
        }
        let mut allocated = SecondaryMap::<Value, bool>::new();
        let mut allocations = Vec::new();
        let mut sources = Vec::new();
        let mut edges = Vec::new();
        for &block in &cfg.rpo {
            let data = &graph.blocks[block];
            if block != graph.entry && cfg.preds[block].iter().all(|&pred| reachable[pred]) {
                allocations.extend(data.params.iter().copied());
            }
            for &inst in &data.insts {
                let data = &graph.insts[inst];
                match data.op {
                    Op::NewObject | Op::NewInstance(_) | Op::NewArray(_) | Op::NewHoleyArray(_) => {
                        allocations.extend(data.result())
                    }
                    Op::IteratorNew(source) | Op::EnumerateNew(source) => {
                        if let Some(iterator) = data.result() {
                            allocations.push(iterator);
                            sources.push((iterator, source));
                        }
                    }
                    _ => {}
                }
            }
            if let Some(terminator) = &data.terminator {
                for target in terminator.kind.targets() {
                    for (&arg, &param) in target.args.iter().zip(&graph.blocks[target.block].params)
                    {
                        edges.push((block, arg, param));
                    }
                }
            }
        }
        for &value in &allocations {
            allocated[value] = true;
        }
        let mut changed = true;
        while changed {
            changed = false;
            for &(_, arg, param) in &edges {
                if !allocated[arg] && allocated[param] {
                    allocated[param] = false;
                    changed = true;
                }
            }
        }
        allocations.retain(|&value| allocated[value]);
        allocations.sort_unstable();
        allocations.dedup();

        let mut classes = SecondaryMap::with_default(NONE);
        if allocations.is_empty() {
            return Self {
                classes,
                positions: SecondaryMap::new(),
                words: 0,
                entering: SecondaryMap::new(),
                first: SecondaryMap::new(),
            };
        }
        let mut local = SecondaryMap::with_default(NONE);
        for (index, &value) in allocations.iter().enumerate() {
            local[value] = index as u32;
        }
        let mut parents: Vec<u32> = (0..allocations.len() as u32).collect();
        for &(_, arg, param) in &edges {
            if allocated[arg] && allocated[param] {
                let left = root(&mut parents, local[arg]);
                let right = root(&mut parents, local[param]);
                parents[left as usize] = right;
            }
        }
        let mut count = 0;
        let mut dense = vec![NONE; allocations.len()];
        for (index, &value) in allocations.iter().enumerate() {
            let root = root(&mut parents, index as u32) as usize;
            if dense[root] == NONE {
                dense[root] = count;
                count += 1;
            }
            classes[value] = dense[root];
        }
        let count = count as usize;
        let words = count.div_ceil(64);

        let mut implied = vec![Vec::new(); count];
        for &(iterator, source) in &sources {
            let (from, to) = (classes[iterator], classes[source]);
            if from != NONE && to != NONE {
                implied[from as usize].push(to);
            }
        }
        let mut closure: Vec<Vec<u32>> = (0..count as u32).map(|class| vec![class]).collect();
        if sources
            .iter()
            .any(|&(iterator, _)| classes[iterator] != NONE)
        {
            let mut seen = vec![false; count];
            for (class, reached) in closure.iter_mut().enumerate() {
                seen.fill(false);
                seen[class] = true;
                let mut index = 0;
                while index < reached.len() {
                    for &next in &implied[reached[index] as usize] {
                        if !std::mem::replace(&mut seen[next as usize], true) {
                            reached.push(next);
                        }
                    }
                    index += 1;
                }
            }
        }

        let mut terminal = Terminal {
            graph,
            cfg,
            edges: &edges,
            used: None,
        };
        let mut positions = SecondaryMap::new();
        let mut first = SecondaryMap::<Block, Vec<(u32, u32)>>::new();
        let mut entering = SecondaryMap::<Block, Vec<u64>>::new();
        let mut marked = vec![false; count];
        let mark = |marked: &mut [bool], events: &mut Vec<(u32, u32)>, class: u32, index: u32| {
            for &class in &closure[class as usize] {
                if !std::mem::replace(&mut marked[class as usize], true) {
                    events.push((class, index));
                }
            }
        };
        for &block in &cfg.rpo {
            let insts = &graph.blocks[block].insts;
            let mut events = Vec::new();
            for (index, &inst) in insts.iter().enumerate() {
                positions[inst] = Some((block, index as u32));
                let op = &graph.insts[inst].op;
                for (position, value) in op.operands_iter().enumerate() {
                    let class = classes[value];
                    if class == NONE || contained_use(op, position) {
                        continue;
                    }
                    let terminal_constructor = position == 1
                        && matches!(op, Op::ConstructorResult(..))
                        && graph.insts[inst]
                            .result()
                            .is_some_and(|result| terminal.result(result));
                    if terminal_constructor {
                        continue;
                    }
                    mark(&mut marked, &mut events, class, index as u32);
                }
            }
            if let Some(terminator) = &graph.blocks[block].terminator {
                let end = insts.len() as u32;
                for target in terminator.kind.targets() {
                    for (&arg, &param) in target.args.iter().zip(&graph.blocks[target.block].params)
                    {
                        let class = classes[arg];
                        if class != NONE && !allocated[param] {
                            mark(&mut marked, &mut events, class, end);
                        }
                    }
                }
            }
            for &(class, _) in &events {
                marked[class as usize] = false;
            }
            events.sort_unstable();
            first[block] = events;
            entering[block] = vec![0; words];
        }

        let mut out = vec![0u64; words];
        let mut changed = true;
        while changed {
            changed = false;
            for &block in &cfg.rpo {
                let Some(terminator) = &graph.blocks[block].terminator else {
                    continue;
                };
                out.copy_from_slice(&entering[block]);
                for &(class, _) in &first[block] {
                    out[class as usize / 64] |= 1 << (class % 64);
                }
                for target in terminator.kind.targets() {
                    let set = &mut entering[target.block];
                    if set.len() != words {
                        set.resize(words, 0);
                    }
                    for (word, &bits) in set.iter_mut().zip(&out) {
                        changed |= bits & !*word != 0;
                        *word |= bits;
                    }
                }
            }
        }
        Self {
            classes,
            positions,
            words,
            entering,
            first,
        }
    }

    fn class(&self, value: Value) -> Option<u32> {
        let class = self.classes[value];
        (class != NONE).then_some(class)
    }

    pub(super) fn may_alias(&self, inst: Inst, write: &Access, read: &Access) -> bool {
        match write.base.and_then(|target| self.at(target, inst)) {
            Some(class) => Some(class) == read.base.and_then(|base| self.class(base)),
            None => true,
        }
    }

    fn at(&self, value: Value, inst: Inst) -> Option<u32> {
        let class = self.class(value)?;
        let (block, index) = self.positions[inst]?;
        let entering = &self.entering[block];
        if entering.len() == self.words && entering[class as usize / 64] & (1 << (class % 64)) != 0
        {
            return None;
        }
        let events = &self.first[block];
        match events.binary_search_by_key(&class, |&(class, _)| class) {
            Ok(found) if events[found].1 <= index => None,
            _ => Some(class),
        }
    }
}

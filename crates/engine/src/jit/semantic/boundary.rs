use crate::{FxHashMap, FxHashSet};

use cranelift_codegen::entity::SecondaryMap;

use super::{
    Binding, Block, BlockData, Cfg, Constant, Graph, Op, Scope, Target, TerminatorKind, Value,
    ValueType,
};

pub(super) fn liveness(graph: &Graph, rpo: &[Block]) -> SecondaryMap<Block, FxHashSet<Value>> {
    let mut live_in: SecondaryMap<_, FxHashSet<Value>> = SecondaryMap::new();
    loop {
        let mut changed = false;
        for &block in rpo.iter().rev() {
            let data = &graph.blocks[block];
            let mut live: FxHashSet<_> = graph
                .successors(block)
                .into_iter()
                .flat_map(|next| live_in[next].iter().copied())
                .collect();
            live.extend(data.terminator.as_ref().unwrap().kind.operands_iter());
            for &inst in data.insts.iter().rev() {
                let data = &graph.insts[inst];
                for result in &data.results {
                    live.remove(result);
                }
                live.extend(data.op.operands_iter());
            }
            for param in &data.params {
                live.remove(param);
            }
            if live != live_in[block] {
                live_in[block] = live;
                changed = true;
            }
        }
        if !changed {
            return live_in;
        }
    }
}

pub(super) fn inputs(target: &Target, live: &SecondaryMap<Block, FxHashSet<Value>>) -> Vec<Value> {
    let mut values: Vec<_> = live[target.block]
        .iter()
        .copied()
        .chain(target.args.iter().copied())
        .collect();
    values.sort();
    values.dedup();
    values
}

pub(super) fn scopes(graph: &Graph) -> SecondaryMap<Block, Vec<Scope>> {
    let mut stacks: SecondaryMap<Block, Option<Vec<Scope>>> = SecondaryMap::new();
    stacks[graph.entry] = Some(Vec::new());
    let mut pending = vec![graph.entry];
    while let Some(block) = pending.pop() {
        let mut stack = stacks[block].clone().unwrap();
        for &inst in &graph.blocks[block].insts {
            match graph.insts[inst].op {
                Op::EnterScope(scope) | Op::EnterCaptures(scope, _) => stack.push(scope),
                Op::LeaveScope(_) => {
                    stack.pop();
                }
                _ => {}
            }
        }
        for target in graph.successors(block) {
            if stacks[target].is_none() {
                stacks[target] = Some(stack.clone());
                pending.push(target);
            }
        }
    }
    let mut result = SecondaryMap::new();
    for (block, stack) in stacks.iter() {
        if let Some(stack) = stack {
            result[block] = stack.clone();
        }
    }
    result
}

pub(super) fn store_exit(graph: &mut Graph, block: Block, values: Vec<Value>, exit: usize) {
    let span = graph.blocks[block]
        .terminator
        .as_ref()
        .map_or(graph.span, |terminator| match terminator.kind {
            TerminatorKind::Yield(span, ..) => span,
            _ => terminator.span,
        });
    graph.append(block, span, Op::ContinuationStore(values), false);
    let index = graph.append_value(block, span, Op::Constant(Constant::Number(exit as f64)));
    graph.terminate(block, span, TerminatorKind::Return(index));
}

pub(super) fn enter(
    graph: &mut Graph,
    target: &Target,
    values: &[Value],
    bindings: &[Binding],
    scopes: &[Scope],
    facts: Option<&super::Facts>,
) {
    let entry = graph.blocks.push(BlockData::default());
    graph.entry = entry;
    graph.arity = None;
    let mut definitions = FxHashMap::default();
    for (index, &value) in values.iter().enumerate() {
        let input = facts.map(|facts| facts.of(value));
        if input.is_some_and(|input| input.intersects(ValueType::UNINITIALIZED)) {
            let restored = graph.append_value(entry, graph.span, Op::Restored(index));
            definitions.insert(value, vec![restored]);
            continue;
        }
        let argument = graph.append_value(entry, graph.span, Op::Argument(index));
        let argument = input.map_or(argument, |input| unbox(graph, entry, argument, input));
        definitions.insert(value, vec![argument]);
    }
    for &scope in scopes {
        graph.append(entry, graph.span, Op::EnterScope(scope), false);
    }
    for (index, &binding) in bindings.iter().enumerate() {
        let argument = graph.append_value(entry, graph.span, Op::Restored(values.len() + index));
        graph.append(
            entry,
            graph.span,
            Op::BindingRestore(binding, argument),
            false,
        );
    }
    let mut target = target.clone();
    for value in &mut target.args {
        *value = definitions[value][0];
    }
    graph.terminate(entry, graph.span, TerminatorKind::Jump(target));
    super::transforms::repair(graph, &definitions);
    let reachable: FxHashSet<_> = Cfg::new(graph).rpo.into_iter().collect();
    for (block, data) in graph.blocks.iter_mut() {
        if !reachable.contains(&block) {
            *data = BlockData::default();
        }
    }
    graph.cold.retain(|block| reachable.contains(block));
}

pub(super) fn compact(graph: &mut Graph) {
    let reachable = Cfg::new(graph).rpo;
    let old_blocks = std::mem::take(&mut graph.blocks);
    let old_insts = std::mem::take(&mut graph.insts);
    graph.values.clear();
    let old_types = std::mem::take(&mut graph.reprs);
    let mut blocks = FxHashMap::default();
    let mut insts = FxHashMap::default();
    let mut values = FxHashMap::default();
    for &block in &reachable {
        let copy = graph.blocks.push(BlockData::default());
        blocks.insert(block, copy);
        for &value in &old_blocks[block].params {
            let new = graph.add_param(copy);
            graph.reprs[new] = old_types[value];
            values.insert(value, new);
        }
    }
    for &block in &reachable {
        let copy = blocks[&block];
        for &inst in &old_blocks[block].insts {
            let data = &old_insts[inst];
            let (new_inst, results) =
                graph.append_results(copy, data.span, data.op.clone(), data.results.len());
            graph.insts[new_inst].source_name = data.source_name;
            graph.insts[new_inst].inline_context = data.inline_context;
            graph.insts[new_inst].emission_location = data.emission_location;
            insts.insert(inst, new_inst);
            for (&old, &new) in data.results.iter().zip(&results) {
                graph.reprs[new] = old_types[old];
                values.insert(old, new);
            }
        }
        graph.blocks[copy].terminator = old_blocks[block].terminator.clone();
    }
    let mut caches = FxHashMap::default();
    for (inst, data) in graph.insts.iter_mut() {
        for value in data.op.operands_mut() {
            *value = values[value];
        }
        match &mut data.op {
            Op::PropertyCache(site, _) => *site = *caches.entry(*site).or_insert(inst),
            _ => {}
        }
    }
    for data in graph.blocks.values_mut() {
        let terminator = data.terminator.as_mut().unwrap();
        for value in terminator.kind.operands_mut() {
            *value = values[value];
        }
        for target in terminator.kind.targets_mut() {
            target.block = blocks[&target.block];
        }
    }
    graph.entry = blocks[&graph.entry];
    graph.cold = graph
        .cold
        .iter()
        .filter_map(|block| blocks.get(block).copied())
        .collect();
    graph.borrowed = graph
        .borrowed
        .iter()
        .filter_map(|value| values.get(value).copied())
        .collect();
    graph.property_accesses = graph
        .property_accesses
        .iter()
        .filter_map(|(inst, read)| {
            let inst = *insts.get(inst)?;
            let mut read = read.clone();
            read.cache = read.cache.map(|cache| values[&cache]);
            Some((inst, read))
        })
        .collect();
    graph.views = graph
        .views
        .iter()
        .filter_map(|(inst, value)| Some((*insts.get(inst)?, *values.get(value)?)))
        .collect();
    for region in graph.regions.values_mut() {
        region.blocks = region
            .blocks
            .iter()
            .filter_map(|block| blocks.get(block).copied())
            .collect();
    }
}

pub(super) fn unbox(
    graph: &mut Graph,
    block: Block,
    value: Value,
    types: super::ValueType,
) -> Value {
    let op = if types.intersects(super::ValueType::INTEGER)
        && types.within(super::ValueType::INTEGER)
    {
        Op::AsInteger(value)
    } else if types.intersects(super::ValueType::NUMBER) && types.within(super::ValueType::NUMBER) {
        Op::AsNumber(value)
    } else {
        return value;
    };
    graph.append_value(block, graph.span, op)
}

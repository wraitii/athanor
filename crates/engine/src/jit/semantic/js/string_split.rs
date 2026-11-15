use crate::{FxHashMap, FxHashSet};

use super::super::{
    Block, BlockData, Cfg, Constant, Graph, Inst, Op, Target, Terminator, TerminatorKind, Value,
    ValueType,
};

pub(in crate::jit::semantic) fn result_facts(
    graph: &Graph,
    cfg: &Cfg,
) -> (FxHashSet<Value>, FxHashMap<Inst, ValueType>) {
    let mut arrays = FxHashMap::default();
    for &block in &cfg.rpo {
        for &inst in &graph.blocks[block].insts {
            let data = &graph.insts[inst];
            let (Op::StringSplit(_, args) | Op::StringSplitParts(_, args)) = &data.op else {
                continue;
            };
            let nonempty = args.first().is_none_or(|separator| {
                matches!(graph.defining_op(*separator), Some(Op::Constant(Constant::Undefined)))
                    || matches!(graph.defining_op(*separator), Some(Op::Constant(Constant::String(text))) if !text.is_empty())
            });
            arrays.insert(data.result().unwrap(), nonempty);
        }
    }
    if arrays.is_empty() {
        return (FxHashSet::default(), FxHashMap::default());
    }
    let mut incoming: FxHashMap<Value, Vec<Value>> = FxHashMap::default();
    for &block in &cfg.rpo {
        if let Some(term) = &graph.blocks[block].terminator {
            for target in term.kind.targets() {
                for (&param, &arg) in graph.blocks[target.block].params.iter().zip(&target.args) {
                    incoming.entry(param).or_default().push(arg);
                }
            }
        }
    }
    for &block in &cfg.rpo {
        for &inst in &graph.blocks[block].insts {
            if let Op::Spread(source) = graph.insts[inst].op {
                incoming.insert(graph.insts[inst].result().unwrap(), vec![source]);
            }
        }
    }
    loop {
        let mut changed = false;
        for (&param, args) in &incoming {
            if arrays.contains_key(&param) || !args.iter().all(|arg| arrays.contains_key(arg)) {
                continue;
            }
            arrays.insert(param, args.iter().all(|arg| arrays[arg]));
            changed = true;
        }
        if !changed {
            break;
        }
    }
    let mut aliases: FxHashMap<Value, Vec<Value>> = FxHashMap::default();
    for (&param, args) in &incoming {
        for &arg in args {
            aliases.entry(param).or_default().push(arg);
            aliases.entry(arg).or_default().push(param);
        }
    }
    let mut hazards: FxHashMap<Value, Vec<Inst>> = FxHashMap::default();
    let mut positions = FxHashMap::default();
    for &block in &cfg.rpo {
        for (position, &inst) in graph.blocks[block].insts.iter().enumerate() {
            positions.insert(inst, position);
            let data = &graph.insts[inst];
            for (index, value) in data.op.operands_iter().enumerate() {
                let safe = match data.op {
                    Op::Get(..)
                    | Op::GetIndex(..)
                    | Op::ArrayGet(..)
                    | Op::ArrayLoad(..)
                    | Op::PackTake(..)
                    | Op::ProbeProperty(..)
                    | Op::ProbeIndex(..) => index == 0,
                    Op::Spread(_)
                    | Op::IsIndexed(_)
                    | Op::IndexedLength(_)
                    | Op::IsUndefined(_)
                    | Op::IsNullish(_)
                    | Op::IsNumber(_)
                    | Op::IsString(_)
                    | Op::IsTypeof(..)
                    | Op::IsUninitialized(_) => true,
                    Op::CheckInitialized(_) => data.result().is_none(),
                    // Spread passes elements, not the backing array, under direct iteration.
                    Op::ConstructPack(..) => index == 1,
                    Op::ConstructPackResolved(..)
                    | Op::ConstructorCallPack(..)
                    | Op::CallPack(..) => index == 2,
                    _ => false,
                };
                if !safe {
                    hazards.entry(value).or_default().push(inst);
                }
            }
        }
    }
    let mut reachable = FxHashMap::<Block, FxHashSet<Block>>::default();
    let mut array_hazards = FxHashMap::<Value, Vec<Inst>>::default();
    let mut reads = FxHashMap::default();
    for &block in &cfg.rpo {
        for &inst in &graph.blocks[block].insts {
            let (array, key) = match graph.insts[inst].op {
                Op::GetIndex(array, key)
                | Op::ArrayGet(array, key)
                | Op::ArrayLoad(array, key, _)
                | Op::PackTake(array, key) => (array, key),
                _ => continue,
            };
            let Some(&nonempty) = arrays.get(&array) else {
                continue;
            };
            let Some(Op::Constant(Constant::Number(index))) = graph.defining_op(key) else {
                continue;
            };
            if !index.is_finite() || *index < 0.0 || index.fract() != 0.0 {
                continue;
            }
            if !(nonempty && *index == 0.0) && !matches!(graph.insts[inst].op, Op::PackTake(..)) {
                continue;
            }
            let hazards = array_hazards.entry(array).or_insert_with(|| {
                let mut pending = vec![array];
                let mut visited = FxHashSet::default();
                let mut result = Vec::new();
                while let Some(value) = pending.pop() {
                    if !visited.insert(value) {
                        continue;
                    }
                    if let Some(uses) = hazards.get(&value) {
                        result.extend(uses.iter().copied());
                    }
                    if let Some(linked) = aliases.get(&value) {
                        pending.extend(linked.iter().copied());
                    }
                }
                result
            });
            // An escape on a disjoint fallback cannot change this read; backedges can.
            if hazards.iter().any(|&hazard| {
                let source = graph.insts[hazard].block;
                if source == block && positions[&hazard] < positions[&inst] {
                    return true;
                }
                reachable
                    .entry(source)
                    .or_insert_with(|| {
                        let mut seen = FxHashSet::default();
                        let mut pending: Vec<_> = graph.blocks[source]
                            .terminator
                            .as_ref()
                            .into_iter()
                            .flat_map(|term| term.kind.targets())
                            .map(|target| target.block)
                            .collect();
                        while let Some(block) = pending.pop() {
                            if !seen.insert(block) {
                                continue;
                            }
                            if let Some(term) = &graph.blocks[block].terminator {
                                pending.extend(
                                    term.kind.targets().into_iter().map(|target| target.block),
                                );
                            }
                        }
                        seen
                    })
                    .contains(&block)
            }) {
                continue;
            }
            reads.insert(
                inst,
                if nonempty && *index == 0.0 && !matches!(graph.insts[inst].op, Op::PackTake(..)) {
                    ValueType::STRING
                } else {
                    ValueType::STRING.union(ValueType::UNDEFINED)
                },
            );
        }
    }
    (arrays.into_keys().collect(), reads)
}

pub(in crate::jit::semantic) fn expand(graph: &mut Graph, expanded: &mut FxHashSet<Inst>) {
    let sites: Vec<_> = Cfg::new(graph)
        .rpo
        .into_iter()
        .flat_map(|block| graph.blocks[block].insts.clone())
        .filter(|&inst| {
            if expanded.contains(&inst) {
                return false;
            }
            let Op::Call {
                callee,
                receiver: Some(_),
                ..
            } = &graph.insts[inst].op
            else {
                return false;
            };
            matches!(graph.defining_op(*callee), Some(Op::Get(_, name)) if name == "split")
        })
        .collect();
    for inst in sites {
        expanded.insert(inst);
        let data = graph.insts[inst].clone();
        let Op::Call {
            callee,
            receiver: Some(receiver),
            args,
            ..
        } = data.op
        else {
            unreachable!()
        };
        let (block, continuation, fallback) = super::super::dispatch::split(graph, inst);
        let fast = graph.blocks.push(BlockData::default());
        let (guard, check) = graph.append(block, data.span, Op::IsStringSplit(callee), true);
        graph.insts[guard].source_name = data.source_name;
        graph.blocks[block].terminator = Some(Terminator {
            source_name: data.source_name,
            emission_location: None,
            span: data.span,
            kind: TerminatorKind::Branch(check.unwrap(), Target::new(fast), Target::new(fallback)),
        });
        let (call, result) = graph.append(fast, data.span, Op::StringSplit(receiver, args), true);
        graph.insts[call].source_name = data.source_name;
        graph.blocks[fast].terminator = Some(Terminator {
            source_name: data.source_name,
            emission_location: None,
            span: data.span,
            kind: TerminatorKind::Jump(Target {
                block: continuation,
                args: vec![result.unwrap()],
            }),
        });
    }
}

pub(in crate::jit::semantic) fn forward_packs(graph: &mut Graph) {
    let cfg = Cfg::new(graph);
    let mut uses = FxHashMap::<Value, usize>::default();
    for &block in &cfg.rpo {
        for &inst in &graph.blocks[block].insts {
            for value in graph.insts[inst].op.operands() {
                *uses.entry(value).or_default() += 1;
            }
        }
        if let Some(term) = &graph.blocks[block].terminator {
            for value in term.kind.operands() {
                *uses.entry(value).or_default() += 1;
            }
        }
    }
    let mut aliases = FxHashMap::default();
    for &block in &cfg.rpo {
        for inst in graph.blocks[block].insts.clone() {
            let Op::Spread(array) = graph.insts[inst].op else {
                continue;
            };
            if uses.get(&array) != Some(&1) {
                continue;
            }
            let mut source = array;
            let mut seen = FxHashSet::default();
            let producer = loop {
                if !seen.insert(source) || uses.get(&source) != Some(&1) {
                    break None;
                }
                match graph.values[source] {
                    super::super::ValueDef::Result(inst, _) => break Some(inst),
                    super::super::ValueDef::Param(block, index) => {
                        let mut incoming = cfg.preds[block]
                            .iter()
                            .flat_map(|&pred| {
                                graph.blocks[pred]
                                    .terminator
                                    .as_ref()
                                    .unwrap()
                                    .kind
                                    .targets()
                            })
                            .filter(|target| target.block == block)
                            .map(|target| target.args[index]);
                        let Some(value) = incoming.next() else {
                            break None;
                        };
                        if incoming.next().is_some() {
                            break None;
                        }
                        source = value;
                    }
                }
            };
            let Some(producer) = producer else {
                continue;
            };
            let Op::StringSplit(receiver, args) = graph.insts[producer].op.clone() else {
                continue;
            };
            graph.set_op(producer, Op::StringSplitParts(receiver, args));
            aliases.insert(graph.insts[inst].result().unwrap(), array);
            graph.remove_inst(inst);
        }
    }
    graph.replace_uses(&aliases);
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxc_allocator::Allocator;
    use oxc_ast::ast::Statement;
    use oxc_parser::Parser;
    use oxc_span::SourceType;

    fn native_graph(body: &str) -> Graph {
        let source = format!("function main(text, other, flag) {{ {body} }}");
        let allocator = Allocator::default();
        let parsed = Parser::new(&allocator, &source, SourceType::script()).parse();
        assert!(!parsed.diagnostics.has_errors());
        let Statement::FunctionDeclaration(function) = &parsed.program.body[0] else {
            panic!()
        };
        let semantic = oxc_semantic::SemanticBuilder::new_compiler()
            .build(&parsed.program)
            .semantic;
        let graph =
            super::super::super::build::lower(function, semantic.scoping(), &FxHashSet::default())
                .unwrap();
        let mut graph = super::super::super::canonicalize_graph(graph, true)
            .unwrap()
            .graph;
        for block in Cfg::new(&graph).rpo {
            for inst in graph.blocks[block].insts.clone() {
                if let Op::Call {
                    callee,
                    receiver: Some(receiver),
                    args,
                    ..
                } = graph.insts[inst].op.clone()
                    && matches!(graph.defining_op(callee), Some(Op::Get(_, name)) if name == "split")
                {
                    graph.set_op(inst, Op::StringSplit(receiver, args));
                }
            }
        }
        graph
    }

    #[test]
    fn split_spread_elements_retain_string_and_missing_argument_facts() {
        for (separator, expected) in [
            ("'|'", ValueType::STRING.union(ValueType::UNDEFINED)),
            ("undefined", ValueType::STRING.union(ValueType::UNDEFINED)),
            ("''", ValueType::STRING.union(ValueType::UNDEFINED)),
            ("other", ValueType::STRING.union(ValueType::UNDEFINED)),
        ] {
            let mut graph = native_graph(&format!(
                "const parts=text.split({separator}); return parts[0];"
            ));
            let read = graph
                .insts
                .iter()
                .find_map(|(inst, data)| matches!(data.op, Op::GetIndex(..)).then_some(inst))
                .unwrap();
            let Op::GetIndex(array, index) = graph.insts[read].op else {
                panic!()
            };
            graph.set_op(read, Op::PackTake(array, index));
            let facts = super::super::super::Facts::compute(&graph);
            assert!(facts.indexed(array));
            assert_eq!(facts.of(graph.insts[read].result().unwrap()), expected);
        }
    }

    #[test]
    fn split_result_facts_reject_mutation_escape_and_arbitrary_merges() {
        for body in [
            "const parts=text.split('|'); parts[0]=7; return parts[0];",
            "const parts=text.split('|'); other(parts); return parts[0];",
            "const parts=flag ? text.split('|') : other(); return parts[0];",
            "const parts=text.split('|'); const alias=flag ? parts : other(); alias[0]=7; return parts[0];",
        ] {
            let graph = native_graph(body);
            let (_, reads) = result_facts(&graph, &Cfg::new(&graph));
            assert!(reads.is_empty(), "{body}");
        }
    }

    #[test]
    fn split_results_merge_when_every_incoming_array_is_known() {
        let graph =
            native_graph("const parts=flag ? text.split('|') : text.split(','); return parts[0];");
        let (_, reads) = result_facts(&graph, &Cfg::new(&graph));
        assert_eq!(reads.len(), 1);
        assert!(reads.values().all(|types| *types == ValueType::STRING));
    }

    #[test]
    fn split_elements_remain_known_before_or_apart_from_escape() {
        for body in [
            "const parts=text.split('|'); const first=parts[0]; other(parts); return first;",
            "const parts=text.split('|'); if(flag) return other(parts); return parts[0];",
        ] {
            let graph = native_graph(body);
            let (_, reads) = result_facts(&graph, &Cfg::new(&graph));
            assert_eq!(reads.len(), 1, "{body}");
            assert!(reads.values().all(|types| *types == ValueType::STRING));
        }
    }

    #[test]
    fn split_elements_are_invalidated_across_loop_backedges() {
        let graph = native_graph(
            "const parts=text.split('|'); let result; for(let i=0;i<2;++i){result=parts[0];parts[0]=7;} return result;",
        );
        let (_, reads) = result_facts(&graph, &Cfg::new(&graph));
        assert!(reads.is_empty());
    }
}

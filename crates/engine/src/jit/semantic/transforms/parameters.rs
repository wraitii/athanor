use cranelift_codegen::entity::SecondaryMap;

use super::phis::{self, Input};
use crate::jit::semantic::{Block, Graph, Value, ValueDef, resolve};

pub(in crate::jit::semantic) fn simplify_parameters(
    graph: &mut Graph,
    blocks: &[Block],
    aliases: &mut SecondaryMap<Value, Option<Value>>,
    mut changed: bool,
) {
    let mut indices = SecondaryMap::<Value, Option<u32>>::new();
    let mut parameters = Vec::new();
    for &block in blocks {
        if block == graph.entry {
            continue;
        }
        for &value in &graph.blocks[block].params {
            indices[value] = Some(parameters.len() as u32);
            parameters.push(value);
        }
    }
    if !parameters.is_empty() {
        let mut edges = Vec::new();
        for &block in blocks {
            if let Some(terminator) = &graph.blocks[block].terminator {
                for target in terminator.kind.targets() {
                    for (&param, &arg) in graph.blocks[target.block].params.iter().zip(&target.args)
                    {
                        if let Some(index) = indices[param] {
                            edges.push((index, resolve(&*aliases, arg)));
                        }
                    }
                }
            }
        }
        edges.sort_by_key(|&(index, _)| index);
        let mut starts = Vec::with_capacity(parameters.len() + 1);
        let mut edge = 0;
        for index in 0..=parameters.len() as u32 {
            while edge < edges.len() && edges[edge].0 < index {
                edge += 1;
            }
            starts.push(edge);
        }
        let inputs: Vec<_> = edges
            .iter()
            .map(|&(_, arg)| match indices[arg] {
                Some(source) => Input::Parameter(source as usize),
                None => Input::Value(arg),
            })
            .collect();
        for (&param, replacement) in parameters.iter().zip(phis::solve(&starts, &inputs)) {
            if let Some(replacement) = replacement {
                aliases[param] = Some(replacement);
                changed = true;
            }
        }
        loop {
            let mut progress = false;
            for (index, &param) in parameters.iter().enumerate() {
                if aliases[param].is_some() {
                    continue;
                }
                let mut unique = None;
                let mut trivial = true;
                for &(_, arg) in &edges[starts[index]..starts[index + 1]] {
                    let value = resolve(&*aliases, arg);
                    if value == param || unique == Some(value) {
                        continue;
                    }
                    if unique.is_some() {
                        trivial = false;
                        break;
                    }
                    unique = Some(value);
                }
                if trivial && let Some(value) = unique {
                    aliases[param] = Some(value);
                    progress = true;
                }
            }
            if !progress {
                break;
            }
            changed = true;
        }
    }
    if !changed {
        return;
    }
    for value in graph.values.keys() {
        if aliases[value].is_some() {
            aliases[value] = Some(resolve(&*aliases, value));
        }
    }
    graph.replace_uses(&*aliases);
    let mut kept = SecondaryMap::<Block, Vec<bool>>::new();
    for &block in blocks {
        let params = &graph.blocks[block].params;
        if params.iter().any(|&param| aliases[param].is_some()) {
            kept[block] = params
                .iter()
                .map(|&param| aliases[param].is_none())
                .collect();
        }
    }
    for &block in blocks {
        if let Some(terminator) = &mut graph.blocks[block].terminator {
            for target in terminator.kind.targets_mut() {
                let mask = &kept[target.block];
                if mask.is_empty() {
                    continue;
                }
                let mut index = 0;
                target.args.retain(|_| {
                    index += 1;
                    mask.get(index - 1).copied().unwrap_or(true)
                });
            }
        }
    }
    for &block in blocks {
        if kept[block].is_empty() {
            continue;
        }
        let params = &mut graph.blocks[block].params;
        params.retain(|&param| aliases[param].is_none());
        for (index, &param) in params.iter().enumerate() {
            graph.values[param] = ValueDef::Param(block, index);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::semantic::{
        Target, TerminatorKind,
        low_level::{Builder, CallTarget},
    };
    use cranelift_codegen::ir::{AbiParam, Signature, types};
    use cranelift_codegen::isa::CallConv;
    use oxc_span::Span;

    fn cycle(conflicting: bool) -> (Graph, Block, Block, Value, Value) {
        let span = Span::new(0, 0);
        let mut signature = Signature::new(CallConv::SystemV);
        signature
            .params
            .extend([AbiParam::new(types::I64), AbiParam::new(types::I8)]);
        signature
            .returns
            .extend([AbiParam::new(types::I64), AbiParam::new(types::I64)]);
        let mut graph = Graph::new_low_level("cycle".into(), span, signature);
        let entry = graph.entry;
        let mut builder = Builder::new(&mut graph);
        builder.switch_to_block(entry);
        builder.append_block_params_for_function_params(entry);
        let address = builder.block_params(entry)[0];
        let condition = builder.block_params(entry)[1];
        let mut called = Signature::new(CallConv::SystemV);
        called
            .returns
            .extend([AbiParam::new(types::I64), AbiParam::new(types::I64)]);
        let results = builder.call(entry, span, CallTarget::Indirect(address), called, vec![]);
        let header = builder.create_block();
        let body = builder.create_block();
        let exit = builder.create_block();
        let x = builder.append_block_param(header, types::I64);
        let y = builder.append_block_param(header, types::I64);
        let a = builder.append_block_param(body, types::I64);
        let b = builder.append_block_param(body, types::I64);
        builder.terminate(
            entry,
            span,
            TerminatorKind::Jump(Target {
                block: header,
                args: vec![results[0], results[usize::from(conflicting)]],
            }),
        );
        builder.terminate(
            header,
            span,
            TerminatorKind::Branch(
                condition,
                Target {
                    block: body,
                    args: vec![x, y],
                },
                Target::new(exit),
            ),
        );
        builder.terminate(
            body,
            span,
            TerminatorKind::Jump(Target {
                block: header,
                args: vec![b, a],
            }),
        );
        builder.terminate(exit, span, TerminatorKind::ReturnValues(vec![x, y]));
        builder.finish().unwrap();
        (graph, header, entry, results[0], results[1])
    }

    #[test]
    fn removes_seeded_forwarding_cycle_and_preserves_entry_and_results() {
        let (graph, header, entry, first, second) = cycle(false);
        graph.verify().unwrap();
        assert_eq!(graph.blocks[entry].params.len(), 2);
        assert!(graph.blocks[header].params.is_empty());
        assert_eq!(graph.low_level_types[first], Some(types::I64));
        assert_eq!(graph.low_level_types[second], Some(types::I64));
        assert!(graph.blocks.iter().any(|(_, data)| data.terminator.as_ref().is_some_and(|term| matches!(&term.kind, TerminatorKind::ReturnValues(values) if values == &[first, first]))));
    }

    #[test]
    fn keeps_cycles_with_conflicting_origins() {
        let (graph, header, entry, _, _) = cycle(true);
        graph.verify().unwrap();
        assert_eq!(graph.blocks[entry].params.len(), 2);
        assert_eq!(graph.blocks[header].params.len(), 2);
    }

    #[test]
    fn forwards_a_parameter_with_multiple_origins() {
        let span = Span::new(0, 0);
        let mut signature = Signature::new(CallConv::SystemV);
        signature.params.extend([
            AbiParam::new(types::I64),
            AbiParam::new(types::I64),
            AbiParam::new(types::I8),
        ]);
        signature.returns.push(AbiParam::new(types::I64));
        let mut graph = Graph::new_low_level("diamond".into(), span, signature);
        let entry = graph.entry;
        let mut builder = Builder::new(&mut graph);
        builder.switch_to_block(entry);
        builder.append_block_params_for_function_params(entry);
        let args = builder.block_params(entry).to_vec();
        let left = builder.create_block();
        let right = builder.create_block();
        let join = builder.create_block();
        let exit = builder.create_block();
        let varying = builder.append_block_param(join, types::I64);
        let forwarded = builder.append_block_param(exit, types::I64);
        builder.terminate(
            entry,
            span,
            TerminatorKind::Branch(args[2], Target::new(left), Target::new(right)),
        );
        builder.terminate(
            left,
            span,
            TerminatorKind::Jump(Target {
                block: join,
                args: vec![args[0]],
            }),
        );
        builder.terminate(
            right,
            span,
            TerminatorKind::Jump(Target {
                block: join,
                args: vec![args[1]],
            }),
        );
        builder.terminate(
            join,
            span,
            TerminatorKind::Jump(Target {
                block: exit,
                args: vec![varying],
            }),
        );
        builder.terminate(exit, span, TerminatorKind::ReturnValues(vec![forwarded]));
        builder.finish().unwrap();
        graph.verify().unwrap();
        assert_eq!(graph.blocks[join].params, vec![varying]);
        assert!(graph.blocks[exit].params.is_empty());
        assert!(
            matches!(&graph.blocks[exit].terminator.as_ref().unwrap().kind, TerminatorKind::ReturnValues(values) if values == &[varying])
        );
    }

    #[test]
    fn ignores_incomplete_edges_from_unreachable_blocks() {
        let span = Span::new(0, 0);
        let mut signature = Signature::new(CallConv::SystemV);
        signature.params.push(AbiParam::new(types::I64));
        signature.returns.push(AbiParam::new(types::I64));
        let mut graph = Graph::new_low_level("unreachable".into(), span, signature);
        let entry = graph.entry;
        let mut builder = Builder::new(&mut graph);
        builder.switch_to_block(entry);
        builder.append_block_params_for_function_params(entry);
        let input = builder.block_params(entry)[0];
        let join = builder.create_block();
        let unreachable = builder.create_block();
        let param = builder.append_block_param(join, types::I64);
        builder.terminate(
            entry,
            span,
            TerminatorKind::Jump(Target {
                block: join,
                args: vec![input],
            }),
        );
        builder.terminate(unreachable, span, TerminatorKind::Jump(Target::new(join)));
        builder.terminate(join, span, TerminatorKind::ReturnValues(vec![param]));
        builder.finish().unwrap();
        graph.verify().unwrap();
        assert!(graph.blocks[join].params.is_empty());
        assert!(
            matches!(&graph.blocks[join].terminator.as_ref().unwrap().kind, TerminatorKind::ReturnValues(values) if values == &[input])
        );
    }

    #[test]
    fn leaves_unseeded_cycles_unresolved() {
        assert_eq!(
            phis::solve::<Value>(&[0, 1, 2], &[Input::Parameter(1), Input::Parameter(0)]),
            vec![None, None]
        );
    }
}

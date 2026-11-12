use std::collections::HashMap;

use cranelift_codegen::entity::SecondaryMap;

use super::phis::{self, Input};
use crate::jit::semantic::{Block, Graph, Value, ValueDef};

pub(in crate::jit::semantic) fn canonicalize_parameters(graph: &mut Graph, blocks: &[Block]) {
    let mut indices = SecondaryMap::<Value, Option<usize>>::new();
    let mut parameters = Vec::new();
    for &block in blocks {
        if block == graph.entry {
            continue;
        }
        for &value in &graph.blocks[block].params {
            indices[value] = Some(parameters.len());
            parameters.push(value);
        }
    }
    if parameters.is_empty() {
        return;
    }
    let mut incoming = vec![Vec::new(); parameters.len()];
    for &block in blocks {
        if let Some(terminator) = &graph.blocks[block].terminator {
            for target in terminator.kind.targets() {
                for (&param, &arg) in graph.blocks[target.block].params.iter().zip(&target.args) {
                    let Some(index) = indices[param] else {
                        continue;
                    };
                    incoming[index].push(match indices[arg] {
                        Some(source) => Input::Parameter(source),
                        None => Input::Value(arg),
                    });
                }
            }
        }
    }
    let replacements = phis::solve(&incoming);
    let aliases: HashMap<_, _> = parameters
        .into_iter()
        .zip(replacements)
        .filter_map(|(value, replacement)| replacement.map(|replacement| (value, replacement)))
        .collect();
    if aliases.is_empty() {
        return;
    }
    graph.replace_uses(&aliases);
    let mut removed = SecondaryMap::<Block, Vec<usize>>::new();
    for &block in blocks {
        for (index, param) in graph.blocks[block].params.iter().enumerate().rev() {
            if aliases.contains_key(param) {
                removed[block].push(index);
            }
        }
    }
    for &block in blocks {
        if let Some(terminator) = &mut graph.blocks[block].terminator {
            for target in terminator.kind.targets_mut() {
                for &index in &removed[target.block] {
                    target.args.remove(index);
                }
            }
        }
    }
    for &block in blocks {
        let params = &mut graph.blocks[block].params;
        params.retain(|param| !aliases.contains_key(param));
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
            phis::solve::<Value>(&[vec![Input::Parameter(1)], vec![Input::Parameter(0)]]),
            vec![None, None]
        );
    }
}

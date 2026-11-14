use super::super::low_level::{self, BlockArg, Builder, Reference};
use super::super::{Block, Cfg, Graph, Op, Repr, TerminatorKind, Value};
use crate::FxHashMap;

pub(in crate::jit) fn lower(graph: &mut Graph) {
    let blocks = Cfg::new(graph).rpo;
    let mut ladders = FxHashMap::<Block, Vec<Block>>::default();
    let mut b = Builder::new(graph);
    for block in blocks {
        let insts = std::mem::take(&mut b.graph.blocks[block].insts);
        b.switch_to_block(block);
        for inst in insts {
            let data = &b.graph.insts[inst];
            let Op::LowLevel(low_level::Op::Cleanup { values, unwind }) = &data.op else {
                b.graph.blocks[block].insts.push(inst);
                continue;
            };
            let unwind = *unwind;
            let values: Vec<_> = values
                .iter()
                .copied()
                .filter(|&value| !empty(b.graph, value))
                .collect();
            let (span, source_name, location) =
                (data.span, data.source_name, data.emission_location);
            b.set_span(span);
            b.set_source_name(source_name);
            b.set_location(location);
            if !unwind {
                for value in values.into_iter().rev() {
                    release(&mut b, value);
                }
                continue;
            }
            let terminator = b.graph.blocks[block].terminator.as_mut().unwrap();
            let TerminatorKind::Jump(target) = &mut terminator.kind else {
                panic!("unwind cleanup requires a jump");
            };
            let destination = target.block;
            let argument_reprs: Vec<_> = b.graph.blocks[destination]
                .params
                .iter()
                .map(|&param| b.graph.repr(param))
                .collect();
            let ladder = ladders.entry(destination).or_default();
            let built = ladder.len();
            while ladder.len() < values.len() {
                ladder.push(b.create_block());
            }
            let terminator = b.graph.blocks[block].terminator.as_mut().unwrap();
            let TerminatorKind::Jump(target) = &mut terminator.kind else {
                unreachable!()
            };
            if !values.is_empty() {
                target.block = ladder[values.len() - 1];
                target.args.extend(values.iter().copied());
            }
            for depth in built + 1..=values.len() {
                let exit = ladder[depth - 1];
                let next = if depth == 1 {
                    destination
                } else {
                    ladder[depth - 2]
                };
                for repr in argument_reprs
                    .iter()
                    .copied()
                    .chain(std::iter::repeat_n(Repr::Boxed, depth))
                {
                    b.append_block_param_repr(exit, repr);
                }
                b.switch_to_block(exit);
                let params = b.block_params(exit).to_vec();
                let (&released, kept) = params.split_last().unwrap();
                release(&mut b, released);
                let kept: Vec<_> = kept.iter().copied().map(BlockArg::Value).collect();
                b.ins().jump(next, &kept);
                b.set_cold_block(exit);
            }
            b.switch_to_block(block);
        }
    }
}

fn empty(graph: &Graph, value: Value) -> bool {
    match graph.defining_op(value) {
        Some(Op::AsBoxed(word)) => empty(graph, *word),
        Some(Op::LowLevel(low_level::Op::Integer { bits: 0, .. })) => true,
        _ => false,
    }
}

fn release(b: &mut Builder<'_>, value: Value) {
    b.reference(
        Reference::Release {
            heap: false,
            cold: true,
        },
        value,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::semantic::variables::Variables;
    use cranelift_codegen::ir::{AbiParam, Signature, types};
    use cranelift_codegen::isa::CallConv;
    use oxc_span::Span;

    #[test]
    fn cleanup_follows_rewritten_operands_and_typed_edges() {
        let mut signature = Signature::new(CallConv::SystemV);
        signature.params.extend([
            AbiParam::new(types::I64),
            AbiParam::new(types::I64),
            AbiParam::new(types::I8),
        ]);
        signature.returns.push(AbiParam::new(types::I8));
        let mut graph = Graph::new_low_level("cleanup".into(), Span::new(0, 0), signature);
        let entry = graph.entry;
        let mut b = Builder::new(&mut graph);
        b.switch_to_block(entry);
        b.append_block_params_for_function_params(entry);
        let args = b.block_params(entry).to_vec();
        let exit = b.create_block();
        let status = b.append_block_param(exit, types::I8);
        let left = b.create_block();
        let right = b.create_block();
        b.ins().brif(args[2], left, &[], right, &[]);
        b.switch_to_block(right);
        let zero = b.ins().iconst(types::I64, 0);
        b.effect(
            right,
            Span::new(0, 0),
            low_level::Op::Cleanup {
                values: vec![args[0], args[0], zero],
                unwind: true,
            },
        );
        let other_status = b.ins().iconst(types::I8, 0);
        b.ins().jump(exit, &[other_status.into()]);
        b.switch_to_block(left);
        b.effect(
            left,
            Span::new(0, 0),
            low_level::Op::Cleanup {
                values: vec![args[0]],
                unwind: true,
            },
        );
        b.ins().jump(exit, &[args[2].into()]);
        b.switch_to_block(exit);
        b.ins().return_(&[status]);
        Variables::default().finish(&mut graph).unwrap();
        graph.replace_uses(&FxHashMap::from_iter([(args[0], args[1])]));
        let moved = std::mem::take(&mut graph.blocks[left].insts);
        let terminator = graph.blocks[left].terminator.take();
        let suffix = graph.blocks.push(super::super::super::BlockData {
            insts: moved.clone(),
            terminator,
            ..Default::default()
        });
        for inst in moved {
            graph.insts[inst].block = suffix;
        }
        graph.blocks[left].terminator = Some(super::super::super::Terminator {
            span: Span::new(0, 0),
            source_name: None,
            emission_location: None,
            kind: TerminatorKind::Jump(super::super::super::Target::new(suffix)),
        });
        lower(&mut graph);
        graph.verify().unwrap();
        let target = |block| match &graph.blocks[block].terminator.as_ref().unwrap().kind {
            TerminatorKind::Jump(target) => target.clone(),
            _ => panic!(),
        };
        let first = target(suffix);
        let second = target(right);
        assert_eq!(first.args, vec![args[2], args[1]]);
        assert_eq!(second.args, vec![other_status, args[1], args[1]]);
        assert_eq!(target(second.block).block, first.block);
        assert_eq!(target(first.block).block, exit);
        assert!(graph.cold.contains(&first.block));
        assert!(graph.cold.contains(&second.block));
    }
}

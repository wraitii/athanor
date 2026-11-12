use crate::jit::semantic::{Block, Cfg, Graph, Inst, Op, Value, low_level};
use cranelift_codegen::entity::{EntityRef, SecondaryMap};

#[derive(Clone, Copy, Debug)]
pub(super) struct Segment {
    pub start: u32,
    pub end: u32,
    pub block: Block,
    pub cold: bool,
    pub live_in: bool,
}

pub(super) struct Liveness {
    segments: Vec<Segment>,
    ranges: SecondaryMap<Value, (u32, u32)>,
    pub calls: Vec<u32>,
}

impl Liveness {
    pub(super) fn of(&self, value: Value) -> &[Segment] {
        let (start, end) = self.ranges[value];
        &self.segments[start as usize..end as usize]
    }

    pub(super) fn crosses_call(&self, segment: &Segment) -> bool {
        let index = self.calls.partition_point(|&call| call < segment.start);
        self.calls
            .get(index)
            .is_some_and(|&call| call <= segment.end)
    }
}

fn use_position(index: u32) -> u32 {
    3 * index
}

fn call_position(index: u32) -> u32 {
    3 * index + 1
}

fn def_position(index: u32) -> u32 {
    3 * index + 2
}

struct Use {
    value: Value,
    block: Block,
    at: u32,
}

pub(super) fn compute(
    graph: &Graph,
    order: &[Block],
    cfg: &Cfg,
    cold: &SecondaryMap<Block, bool>,
    remat: &SecondaryMap<Value, u32>,
    operands: &dyn Fn(Inst, &mut Vec<Value>),
    terminator_operands: &dyn Fn(Block, &mut Vec<Value>),
    is_call: &dyn Fn(Inst) -> bool,
) -> Result<Liveness, String> {
    let mut header = SecondaryMap::<Block, u32>::new();
    let mut last = SecondaryMap::<Block, u32>::new();
    let mut definition = SecondaryMap::<Value, Option<(Block, u32)>>::new();
    let mut uses = Vec::new();
    let mut recent = SecondaryMap::<Value, Option<(Block, usize)>>::new();
    let mut calls = Vec::new();
    let mut values = Vec::new();
    let mut next = 0;
    for &block in order {
        header[block] = next;
        for &param in &graph.blocks[block].params {
            definition[param] = Some((block, def_position(next)));
        }
        next += 1;
        for inst in graph.blocks[block].insts.iter().copied() {
            let index = next;
            next += 1;
            values.clear();
            operands(inst, &mut values);
            for &value in &values {
                if remat[value] == 1 {
                    continue;
                }
                let at = use_position(index);
                match recent[value] {
                    Some((seen, position)) if seen == block => {
                        uses[position] = Use { value, block, at }
                    }
                    _ => {
                        recent[value] = Some((block, uses.len()));
                        uses.push(Use { value, block, at });
                    }
                }
            }
            if matches!(
                graph.insts[inst].op,
                Op::LowLevel(low_level::Op::Call { .. })
            ) && is_call(inst)
            {
                calls.push(call_position(index));
            }
            for &result in &graph.insts[inst].results {
                if remat[result] != 1 {
                    definition[result] = Some((block, def_position(index)));
                }
            }
        }
        let index = next;
        next += 1;
        {
            values.clear();
            terminator_operands(block, &mut values);
            for &value in &values {
                if remat[value] == 1 {
                    continue;
                }
                let at = use_position(index);
                match recent[value] {
                    Some((seen, position)) if seen == block => {
                        uses[position] = Use { value, block, at }
                    }
                    _ => {
                        recent[value] = Some((block, uses.len()));
                        uses.push(Use { value, block, at });
                    }
                }
            }
        }
        last[block] = index;
    }
    uses.sort_by_key(|u| u.value);

    let entry = graph.entry;
    let mut live_in = SecondaryMap::<Block, u32>::new();
    let mut live_out = SecondaryMap::<Block, u32>::new();
    let mut used_at = SecondaryMap::<Block, (u32, u32)>::new();
    let mut touched = Vec::new();
    let mut worklist = Vec::new();
    let mut segments = Vec::new();
    let mut ranges = SecondaryMap::<Value, (u32, u32)>::new();
    let mut cursor = 0;
    for index in 0..graph.values.len() {
        let value = Value::new(index);
        let stamp = index as u32 + 1;
        let first = cursor;
        while cursor < uses.len() && uses[cursor].value == value {
            cursor += 1;
        }
        let value_uses = &uses[first..cursor];
        let Some((def_block, def_at)) = definition[value] else {
            if !value_uses.is_empty() {
                return Err(format!("{value} used before definition"));
            }
            continue;
        };
        if remat[value] > 1 && value_uses.is_empty() {
            continue;
        }

        touched.clear();
        touched.push(def_block);
        for u in value_uses {
            used_at[u.block] = (stamp, u.at);
            if u.block != def_block && live_in[u.block] != stamp {
                live_in[u.block] = stamp;
                touched.push(u.block);
                worklist.push(u.block);
            }
        }
        while let Some(block) = worklist.pop() {
            for &pred in &cfg.preds[block] {
                live_out[pred] = stamp;
                if pred != def_block && live_in[pred] != stamp {
                    live_in[pred] = stamp;
                    touched.push(pred);
                    worklist.push(pred);
                }
            }
        }
        if live_in[entry] == stamp {
            return Err(format!("{value} used before definition"));
        }

        let first_segment = segments.len();
        for &block in &touched {
            let entering = live_in[block] == stamp;
            let start = if entering {
                use_position(header[block])
            } else {
                def_at
            };
            let end = if live_out[block] == stamp {
                def_position(last[block])
            } else {
                match used_at[block] {
                    (seen, at) if seen == stamp => at.max(start),
                    _ => start,
                }
            };
            segments.push(Segment {
                start,
                end,
                block,
                cold: cold[block],
                live_in: entering,
            });
        }
        segments[first_segment..].sort_unstable_by_key(|segment| segment.start);
        ranges[value] = (first_segment as u32, segments.len() as u32);
    }
    Ok(Liveness {
        segments,
        ranges,
        calls,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::semantic::{Target, TerminatorKind, low_level::Builder};
    use cranelift_codegen::ir::{AbiParam, Signature, types};
    use cranelift_codegen::isa::CallConv;
    use oxc_span::Span;

    #[test]
    fn tracks_edge_arguments_loops_and_multiple_call_results() {
        let span = Span::new(0, 0);
        let mut signature = Signature::new(CallConv::SystemV);
        signature.params.push(AbiParam::new(types::I64));
        let mut graph = Graph::new_low_level("test".into(), span, signature);
        let entry = graph.entry;
        let mut builder = Builder::new(&mut graph);
        builder.switch_to_block(entry);
        builder.append_block_params_for_function_params(entry);
        let input = builder.block_params(entry)[0];
        let loop_block = builder.create_block();
        let exit = builder.create_block();
        let carried = builder.append_block_param(loop_block, types::I64);
        let mut call_signature = Signature::new(CallConv::SystemV);
        call_signature.params.push(AbiParam::new(types::I64));
        call_signature
            .returns
            .extend([AbiParam::new(types::I64), AbiParam::new(types::I8)]);
        let results = builder.call(
            entry,
            span,
            low_level::CallTarget::Indirect(input),
            call_signature,
            vec![input],
        );
        builder.terminate(
            entry,
            span,
            TerminatorKind::Jump(Target {
                block: loop_block,
                args: vec![results[0]],
            }),
        );
        builder.terminate(
            loop_block,
            span,
            TerminatorKind::Branch(
                results[1],
                Target {
                    block: loop_block,
                    args: vec![carried],
                },
                Target::new(exit),
            ),
        );
        builder.terminate(exit, span, TerminatorKind::ReturnValues(vec![carried]));
        drop(builder);
        let cfg = Cfg::new(&graph);
        let cold = SecondaryMap::new();
        let remat = SecondaryMap::new();
        let operands = |inst, values: &mut Vec<Value>| {
            values.clear();
            values.extend(graph.insts[inst].op.operands_iter());
        };
        let terminators = |block: Block, out: &mut Vec<Value>| {
            out.extend(
                graph.blocks[block]
                    .terminator
                    .as_ref()
                    .unwrap()
                    .kind
                    .operands_iter(),
            );
        };
        let live = compute(
            &graph,
            &cfg.rpo,
            &cfg,
            &cold,
            &remat,
            &operands,
            &terminators,
            &|_| true,
        )
        .unwrap();
        assert_eq!(live.calls.len(), 1);
        assert!(live.of(results[0]).iter().any(|s| s.block == entry));
        assert!(
            live.of(results[1])
                .iter()
                .any(|s| s.block == loop_block && s.live_in)
        );
        assert!(
            live.of(carried)
                .iter()
                .any(|s| s.block == exit && s.live_in)
        );
        let target = crate::jit::alloc::Target {
            int: crate::jit::alloc::Registers {
                caller_saved: &[0, 1],
                callee_saved: &[2],
            },
            float: crate::jit::alloc::Registers {
                caller_saved: &[],
                callee_saved: &[],
            },
        };
        let allocation = super::super::graph::allocate(
            &graph,
            &cfg,
            &cfg.rpo,
            &target,
            &[],
            &operands,
            &terminators,
            &remat,
            &|_| true,
        )
        .unwrap();
        assert_ne!(
            allocation.place(carried, loop_block),
            allocation.place(results[1], loop_block)
        );
    }
}

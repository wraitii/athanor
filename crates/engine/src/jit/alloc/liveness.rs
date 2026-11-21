use crate::jit::semantic::{Block, Cfg, Graph, Value};
use cranelift_codegen::entity::{EntityRef, SecondaryMap};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Segment {
    pub start: u32,
    pub end: u32,
    pub cold: bool,
}

pub(super) struct Liveness {
    segments: Vec<Segment>,
    ranges: SecondaryMap<Value, (u32, u32)>,
    pub calls: Vec<u32>,
    pub boundary: Vec<(Value, Block)>,
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

struct Loop {
    header: Block,
    body: crate::FxHashSet<Block>,
    predecessors: Vec<Block>,
    segments: Vec<Segment>,
    boundary: Vec<Block>,
}

fn append_segment(segments: &mut Vec<Segment>, segment: Segment) {
    match segments.last_mut() {
        Some(previous) if previous.end + 1 == segment.start && previous.cold == segment.cold => {
            previous.end = segment.end;
        }
        _ => segments.push(segment),
    }
}

pub(super) fn compute(
    graph: &Graph,
    order: &[Block],
    cfg: &Cfg,
    cold: &SecondaryMap<Block, bool>,
    input: &super::graph::Input,
) -> Result<Liveness, String> {
    compute_with_loops(graph, order, cfg, cold, input, true)
}

fn compute_with_loops(
    graph: &Graph,
    order: &[Block],
    cfg: &Cfg,
    cold: &SecondaryMap<Block, bool>,
    input: &super::graph::Input,
    skip_loops: bool,
) -> Result<Liveness, String> {
    let remat = &input.remat;
    let mut header = SecondaryMap::<Block, u32>::new();
    let mut last = SecondaryMap::<Block, u32>::new();
    let mut definition = SecondaryMap::<Value, Option<(Block, u32)>>::new();
    let mut uses = Vec::new();
    let mut recent = SecondaryMap::<Value, Option<(Block, usize)>>::new();
    let mut calls = Vec::new();
    let mut position = SecondaryMap::<Block, u32>::new();
    let mut crossing = SecondaryMap::<Block, bool>::new();
    let mut next = 0;
    for (index, &block) in order.iter().enumerate() {
        position[block] = index as u32;
        crossing[block] = cfg.preds[block]
            .iter()
            .any(|&pred| cold[pred] != cold[block]);
        header[block] = next;
        for &param in &graph.blocks[block].params {
            definition[param] = Some((block, def_position(next)));
        }
        next += 1;
        for inst in graph.blocks[block].insts.iter().copied() {
            let index = next;
            next += 1;
            for value in graph.insts[inst].op.operands_iter() {
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
            if input.calls[inst] {
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
            for value in graph.blocks[block]
                .terminator
                .iter()
                .flat_map(|term| term.kind.operands_iter())
            {
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

    let mut loops = Vec::new();
    let mut containing = SecondaryMap::<Block, Vec<usize>>::new();
    if skip_loops {
        for (loop_header, body) in crate::jit::semantic::natural_loops(graph, cfg)
            .into_iter()
            .rev()
        {
            let mut blocks: Vec<_> = body.iter().copied().collect();
            blocks.sort_unstable_by_key(|&block| position[block]);
            let mut predecessors = Vec::new();
            let mut segments = Vec::new();
            let mut boundary = Vec::new();
            for block in blocks {
                containing[block].push(loops.len());
                predecessors.extend(
                    cfg.preds[block]
                        .iter()
                        .copied()
                        .filter(|pred| !body.contains(pred)),
                );
                append_segment(
                    &mut segments,
                    Segment {
                        start: use_position(header[block]),
                        end: def_position(last[block]),
                        cold: cold[block],
                    },
                );
                if crossing[block] {
                    boundary.push(block);
                }
            }
            predecessors.sort_unstable();
            predecessors.dedup();
            loops.push(Loop {
                header: loop_header,
                body,
                predecessors,
                segments,
                boundary,
            });
        }
    }
    let mut loop_live = vec![0u32; loops.len()];
    let mut active_loops = Vec::new();
    let mut value_segments = Vec::new();
    let entry = graph.entry;
    let mut live_in = SecondaryMap::<Block, u32>::new();
    let mut live_out = SecondaryMap::<Block, u32>::new();
    let mut used_at = SecondaryMap::<Block, (u32, u32)>::new();
    let mut touched = vec![0u64; order.len().div_ceil(64)];
    let mut worklist = Vec::new();
    let mut segments = Vec::<Segment>::new();
    let mut boundary = Vec::new();
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

        let mut low = position[def_block];
        let mut high = low;
        let mut touch = |block: Block| {
            let at = position[block];
            touched[at as usize / 64] |= 1 << (at % 64);
            low = low.min(at);
            high = high.max(at);
        };
        let loop_of = |block: Block| {
            containing[block]
                .iter()
                .copied()
                .find(|&index| !loops[index].body.contains(&def_block))
        };
        touch(def_block);
        for u in value_uses {
            used_at[u.block] = (stamp, u.at);
            if u.block != def_block && live_in[u.block] != stamp {
                live_in[u.block] = stamp;
                worklist.push(u.block);
            }
        }
        while let Some(block) = worklist.pop() {
            // A use of an external value remains reachable on the next iteration
            // from every block in a natural loop, so all its ranges are live.
            let predecessors = if let Some(index) = loop_of(block) {
                if loop_live[index] == stamp {
                    continue;
                }
                if loops[index].header == entry {
                    return Err(format!("{value} used before definition"));
                }
                loop_live[index] = stamp;
                active_loops.push(index);
                &loops[index].predecessors[..]
            } else {
                touch(block);
                &cfg.preds[block]
            };
            for &pred in predecessors {
                live_out[pred] = stamp;
                if pred != def_block && live_in[pred] != stamp {
                    live_in[pred] = stamp;
                    worklist.push(pred);
                }
            }
        }
        if live_in[entry] == stamp {
            return Err(format!("{value} used before definition"));
        }

        let first_segment = segments.len();
        let first_boundary = boundary.len();
        for index in active_loops.drain(..) {
            value_segments.extend_from_slice(&loops[index].segments);
            boundary.extend(loops[index].boundary.iter().map(|&block| (value, block)));
        }
        for word in low as usize / 64..=high as usize / 64 {
            let mut bits = std::mem::take(&mut touched[word]);
            while bits != 0 {
                let block = order[word * 64 + bits.trailing_zeros() as usize];
                bits &= bits - 1;
                let entering = live_in[block] == stamp;
                if entering && crossing[block] {
                    boundary.push((value, block));
                }
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
                let segment = Segment {
                    start,
                    end,
                    cold: cold[block],
                };
                value_segments.push(segment);
            }
        }
        value_segments.sort_unstable_by_key(|segment| segment.start);
        for segment in value_segments.drain(..) {
            if segments.len() == first_segment {
                segments.push(segment);
            } else {
                append_segment(&mut segments, segment);
            }
        }
        boundary[first_boundary..].sort_unstable_by_key(|&(_, block)| position[block]);
        ranges[value] = (first_segment as u32, segments.len() as u32);
    }
    Ok(Liveness {
        segments,
        ranges,
        calls,
        boundary,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::semantic::{
        Target, TerminatorKind,
        low_level::{self, Builder},
    };
    use cranelift_codegen::ir::{AbiParam, Signature, types};
    use cranelift_codegen::isa::CallConv;
    use oxc_span::Span;

    fn compare(graph: &Graph, order: &[Block], cfg: &Cfg, cold: &SecondaryMap<Block, bool>) {
        let input = super::super::graph::Input::default();
        let expected = compute_with_loops(graph, order, cfg, cold, &input, false).unwrap();
        let actual = compute(graph, order, cfg, cold, &input).unwrap();
        assert_eq!(actual.calls, expected.calls);
        assert_eq!(actual.boundary, expected.boundary);
        for value in graph.values.keys() {
            assert_eq!(actual.of(value), expected.of(value), "{value}");
        }
    }

    #[test]
    fn loop_summaries_match_predecessor_walk() {
        let span = Span::new(0, 0);
        for seed in 0..256u32 {
            let mut signature = Signature::new(CallConv::SystemV);
            signature.params.push(AbiParam::new(types::I64));
            let mut graph = Graph::new_low_level("loops".into(), span, signature);
            let entry = graph.entry;
            let mut builder = Builder::new(&mut graph);
            builder.switch_to_block(entry);
            builder.append_block_params_for_function_params(entry);
            let input = builder.block_params(entry)[0];
            let blocks: Vec<_> = (0..16).map(|_| builder.create_block()).collect();
            builder.terminate(entry, span, TerminatorKind::Jump(Target::new(blocks[0])));
            let mut state = seed + 1;
            for (index, &block) in blocks.iter().enumerate() {
                builder.switch_to_block(block);
                let local = builder.ins().iadd(input, input);
                if index + 1 == blocks.len() {
                    builder.terminate(block, span, TerminatorKind::ReturnValues(vec![local]));
                    continue;
                }
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let target = blocks[(state >> 16) as usize % blocks.len()];
                builder.terminate(
                    block,
                    span,
                    TerminatorKind::Branch(
                        local,
                        Target::new(blocks[index + 1]),
                        Target::new(target),
                    ),
                );
            }
            drop(builder);
            let cfg = Cfg::new(&graph);
            let values: Vec<_> = graph
                .values
                .keys()
                .map(|value| (value, graph.def_block(value)))
                .collect();
            let mut builder = Builder::new(&mut graph);
            for &block in &blocks {
                builder.switch_to_block(block);
                for &(value, definition) in &values {
                    if cfg.dominates(definition, block) {
                        builder.ins().iadd(value, value);
                    }
                }
            }
            drop(builder);
            let mut cold = SecondaryMap::new();
            for (index, &block) in blocks.iter().enumerate() {
                cold[block] = (index as u32 + seed) % 3 == 0;
            }
            compare(&graph, &cfg.rpo, &cfg, &cold);
            let mut order = cfg.rpo.clone();
            order.reverse();
            compare(&graph, &order, &cfg, &cold);
        }
    }

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
        let mut input = super::super::graph::Input::default();
        for value in graph.values.keys() {
            input.classes[value] =
                graph.reprs[value].and_then(|repr| super::super::RegClass::of(repr.machine()));
        }
        for (inst, data) in graph.insts.iter() {
            input.calls[inst] = matches!(
                data.op,
                crate::jit::semantic::Op::LowLevel(
                    crate::jit::semantic::low_level::Op::Call { .. }
                )
            );
        }
        compare(&graph, &cfg.rpo, &cfg, &cold);
        let live = compute(&graph, &cfg.rpo, &cfg, &cold, &input).unwrap();
        let mut headers = SecondaryMap::<Block, u32>::new();
        let mut next = 0;
        for &block in &cfg.rpo {
            headers[block] = next;
            next += graph.blocks[block].insts.len() as u32 + 2;
        }
        let live_into = |value: Value, block: Block| {
            let at = use_position(headers[block]);
            live.of(value).iter().any(|s| s.start <= at && at <= s.end)
        };
        assert_eq!(live.calls.len(), 1);
        assert_eq!(
            live.of(results[0]).first().map(|s| s.start),
            Some(def_position(headers[entry] + 1))
        );
        assert!(live_into(results[1], loop_block));
        assert!(live_into(carried, exit));
        assert!(!live_into(results[0], exit));
        let target = crate::jit::alloc::Target {
            int: crate::jit::alloc::Registers {
                caller_saved: &[0, 1],
                callee_saved: &[2],
            },
            float: crate::jit::alloc::Registers {
                caller_saved: &[],
                callee_saved: &[],
            },
            vector: crate::jit::alloc::Registers {
                caller_saved: &[],
                callee_saved: &[],
            },
        };
        let allocation =
            super::super::graph::allocate(&graph, &cfg, &cfg.rpo, &target, &input).unwrap();
        assert_ne!(
            allocation.place(carried, loop_block),
            allocation.place(results[1], loop_block)
        );
    }
}

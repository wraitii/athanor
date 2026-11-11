use cranelift_codegen::entity::{EntityRef, SecondaryMap};
use cranelift_codegen::flowgraph::ControlFlowGraph;
use cranelift_codegen::ir::{Block, Function, Value};

#[derive(Clone, Copy, Debug)]
pub(super) struct Segment {
    pub start: u32,
    pub end: u32,
    pub block: Block,
    pub cold: bool,
    pub live_in: bool,
}

pub(super) struct Liveness {
    pub segments: SecondaryMap<Value, Vec<Segment>>,
    pub calls: Vec<u32>,
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
    func: &Function,
    order: &[Block],
    cfg: &ControlFlowGraph,
    cold: &SecondaryMap<Block, bool>,
    constants: &SecondaryMap<Value, bool>,
    operands: super::Operands,
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
        for &param in func.dfg.block_params(block) {
            definition[param] = Some((block, def_position(next)));
        }
        next += 1;
        for inst in func.layout.block_insts(block) {
            let index = next;
            next += 1;
            operands(inst, &mut values);
            for &value in &values {
                if constants[value] {
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
            let opcode = func.dfg.insts[inst].opcode();
            if opcode.is_call() && !opcode.is_return() {
                calls.push(call_position(index));
            }
            for &result in func.dfg.inst_results(inst) {
                if !constants[result] {
                    definition[result] = Some((block, def_position(index)));
                }
            }
        }
        last[block] = next - 1;
    }
    uses.sort_by_key(|u| u.value);

    let entry = func.layout.entry_block();
    let mut live_in = SecondaryMap::<Block, u32>::new();
    let mut live_out = SecondaryMap::<Block, u32>::new();
    let mut used_at = SecondaryMap::<Block, (u32, u32)>::new();
    let mut touched = Vec::new();
    let mut worklist = Vec::new();
    let mut segments = SecondaryMap::<Value, Vec<Segment>>::new();
    let mut cursor = 0;
    for index in 0..func.dfg.num_values() {
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
            for pred in cfg.pred_iter(block) {
                let pred = pred.block;
                live_out[pred] = stamp;
                if pred != def_block && live_in[pred] != stamp {
                    live_in[pred] = stamp;
                    touched.push(pred);
                    worklist.push(pred);
                }
            }
        }
        if entry.is_some_and(|entry| live_in[entry] == stamp) {
            return Err(format!("{value} used before definition"));
        }

        let list = &mut segments[value];
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
            list.push(Segment {
                start,
                end,
                block,
                cold: cold[block],
                live_in: entering,
            });
        }
        list.sort_unstable_by_key(|segment| segment.start);
    }
    Ok(Liveness { segments, calls })
}

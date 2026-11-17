use cranelift_codegen::ir::{Signature, condcodes::IntCC, types};
use oxc_span::Span;

use super::cursor::Insert;
use super::low_level::{Binary, BlockArg, Builder};
use super::{
    Block, Contract, FxHashMap, Graph, MemoryFormat, Op as GraphOp, Operands, OperandsMut,
    Ownership, Value,
};
use crate::jit::effects::Effects;
use crate::regex::{DEAD, Dfa, NO_MATCH};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(in crate::jit) struct ByteSet([u64; 4]);

impl ByteSet {
    fn insert(&mut self, byte: u8) {
        self.0[byte as usize / 64] |= 1 << (byte % 64);
    }

    fn contains(&self, byte: u8) -> bool {
        self.0[byte as usize / 64] & (1 << (byte % 64)) != 0
    }

    fn is_empty(&self) -> bool {
        self.0 == [0; 4]
    }

    fn ranges(&self) -> Vec<(u8, u8)> {
        let mut ranges: Vec<(u8, u8)> = Vec::new();
        for byte in 0..=255u8 {
            if !self.contains(byte) {
                continue;
            }
            match ranges.last_mut() {
                Some(last) if last.1 as u16 + 1 == byte as u16 => last.1 = byte,
                _ => ranges.push((byte, byte)),
            }
        }
        ranges
    }
}

#[derive(Clone, Debug)]
pub(in crate::jit) enum Op {
    Class(Value, ByteSet),
}

impl Op {
    pub(in crate::jit) fn operands(&self) -> Operands<'_> {
        match self {
            Self::Class(byte, _) => Operands::new([Some(*byte), None, None, None], &[]),
        }
    }

    pub(in crate::jit) fn operands_mut(&mut self) -> OperandsMut<'_> {
        match self {
            Self::Class(byte, _) => OperandsMut::new([Some(byte), None, None, None], &mut []),
        }
    }

    pub(in crate::jit) fn contract(&self) -> Contract {
        Contract {
            effects: Effects::of(0),
            result: Ownership::Unowned,
        }
    }

    pub(in crate::jit) fn listing(&self) -> String {
        match self {
            Self::Class(byte, set) => {
                let ranges = set
                    .ranges()
                    .iter()
                    .map(|&(low, high)| {
                        if low == high {
                            format!("{low:#04x}")
                        } else {
                            format!("{low:#04x}-{high:#04x}")
                        }
                    })
                    .collect::<Vec<_>>();
                format!("regex.class {byte} [{}]", ranges.join(" "))
            }
        }
    }
}

pub(in crate::jit) fn lower(graph: &mut Graph) {
    let mut aliases = FxHashMap::default();
    let mut builder = Builder::new(graph);
    for block in builder.graph().blocks.keys().collect::<Vec<_>>() {
        let insts = std::mem::take(&mut builder.graph_mut().blocks[block].insts);
        builder.switch_to_block(block);
        for inst in insts {
            let GraphOp::Regex(Op::Class(byte, set)) = builder.graph().insts[inst].op else {
                builder.graph_mut().blocks[block].insts.push(inst);
                continue;
            };
            let result = builder.graph().insts[inst].result().unwrap();
            builder.set_span(builder.graph().insts[inst].span);
            let tests: Vec<Value> = set
                .ranges()
                .into_iter()
                .map(|(low, high)| builder.ins().in_range(byte, low as u64, high as u64))
                .collect();
            let lowered = match tests.split_first() {
                Some((&first, rest)) => rest
                    .iter()
                    .fold(first, |any, &test| builder.ins().boolean_or(any, test)),
                None => builder.ins().iconst(types::I8, 0),
            };
            aliases.insert(result, lowered);
        }
    }
    if !aliases.is_empty() {
        builder.graph_mut().replace_uses(&aliases);
    }
}

struct Matcher<'a, 'g> {
    dfa: &'a Dfa,
    builder: Builder<'g>,
    subject: Value,
    length: Value,
    states: FxHashMap<u32, Block>,
    pending: Vec<u32>,
    outer: Block,
    after: Block,
    found: Block,
    none: Block,
}

pub(in crate::jit) fn matcher(dfa: &Dfa, signature: Signature) -> Graph {
    let mut graph = Graph::new_low_level("regex".into(), Span::new(0, 0), signature);
    let entry = graph.entry;
    let mut builder = Builder::new(&mut graph);
    builder.switch_to_block(entry);
    builder.append_block_params_for_function_params(entry);
    let params = builder.block_params(entry).to_vec();
    let (subject, length, from) = (params[0], params[1], params[2]);
    let outer = builder.create_block();
    let after = builder.create_block();
    let found = builder.create_block();
    let none = builder.create_block();
    builder.ins().jump(outer, &[BlockArg::Value(from)]);
    let mut matcher = Matcher {
        dfa,
        builder,
        subject,
        length,
        states: FxHashMap::default(),
        pending: Vec::new(),
        outer,
        after,
        found,
        none,
    };
    matcher.outer();
    matcher.after();
    matcher.found();
    matcher.none();
    while let Some(state) = matcher.pending.pop() {
        matcher.state(state);
    }
    drop(matcher);
    graph
}

impl Matcher<'_, '_> {
    fn state_block(&mut self, state: u32) -> Block {
        if let Some(&block) = self.states.get(&state) {
            return block;
        }
        let block = self.builder.create_block();
        for _ in 0..3 {
            self.builder.append_block_param(block, types::I64);
        }
        self.states.insert(state, block);
        self.pending.push(state);
        block
    }

    fn first_bytes(&self) -> Option<ByteSet> {
        let starts = [self.dfa.start_at_beginning, self.dfa.start];
        if starts.iter().any(|&state| {
            state != DEAD
                && (self.dfa.matching[state as usize] || self.dfa.matching_at_end[state as usize])
        }) {
            return None;
        }
        let mut first = ByteSet::default();
        for state in starts.into_iter().filter(|&state| state != DEAD) {
            for byte in 0..=255u8 {
                if self.dfa.next(state, byte) != DEAD {
                    first.insert(byte);
                }
            }
        }
        Some(first)
    }

    fn outer(&mut self) {
        let outer = self.outer;
        let start = self.builder.append_block_param(outer, types::I64);
        self.builder.switch_to_block(outer);
        let dispatch = self.builder.create_block();
        let past_end = self.builder.ins().icmp(IntCC::UnsignedGreaterThan, start, self.length);
        let within = self.builder.create_block();
        self.builder
            .ins()
            .brif(past_end, self.none, &[], within, &[]);
        self.builder.switch_to_block(within);
        let next_start = self.builder.create_block();
        let at_end = self.builder.ins().icmp(IntCC::Equal, start, self.length);
        let inspect = self.builder.create_block();
        match self.first_bytes() {
            Some(first) => {
                self.builder.ins().brif(at_end, self.none, &[], inspect, &[]);
                self.builder.switch_to_block(inspect);
                let byte = self.load(start);
                let candidate = self.class(byte, first);
                self.builder
                    .ins()
                    .brif(candidate, dispatch, &[], next_start, &[]);
            }
            None => {
                self.builder.ins().brif(at_end, dispatch, &[], inspect, &[]);
                self.builder.switch_to_block(inspect);
                let byte = self.load(start);
                let continuation = self.builder.ins().in_range(byte, 0x80, 0xbf);
                self.builder
                    .ins()
                    .brif(continuation, next_start, &[], dispatch, &[]);
            }
        }
        self.builder.switch_to_block(next_start);
        let one = self.builder.ins().iconst(types::I64, 1);
        let advanced = self.builder.ins().iadd(start, one);
        self.builder.ins().jump(outer, &[BlockArg::Value(advanced)]);
        self.builder.switch_to_block(dispatch);
        let unset = self.builder.ins().iconst(types::I64, -1);
        let args = [start, start, unset].map(BlockArg::Value);
        let (beginning, elsewhere) = (self.dfa.start_at_beginning, self.dfa.start);
        if beginning == elsewhere {
            self.enter(beginning, &args);
            return;
        }
        let first = self.builder.create_block();
        let later = self.builder.create_block();
        let is_first = self.builder.ins().icmp_imm_u(IntCC::Equal, start, 0);
        self.builder.ins().brif(is_first, first, &[], later, &[]);
        self.builder.switch_to_block(first);
        self.enter(beginning, &args);
        self.builder.switch_to_block(later);
        if elsewhere == DEAD {
            self.builder.ins().jump(self.none, &[]);
            return;
        }
        self.enter(elsewhere, &args);
    }

    fn enter(&mut self, state: u32, args: &[BlockArg; 3]) {
        if state == DEAD {
            self.builder.ins().jump(self.after, &[args[0], args[2]]);
            return;
        }
        let block = self.state_block(state);
        self.builder.ins().jump(block, args);
    }

    fn state(&mut self, state: u32) {
        let block = self.states[&state];
        let params = self.builder.block_params(block).to_vec();
        let (start, position, last) = (params[0], params[1], params[2]);
        self.builder.switch_to_block(block);
        let last = if self.dfa.matching[state as usize] {
            position
        } else {
            last
        };
        let at_end = self.builder.ins().icmp(IntCC::Equal, position, self.length);
        let end = self.builder.create_block();
        let step = self.builder.create_block();
        self.builder.ins().brif(at_end, end, &[], step, &[]);
        self.builder.switch_to_block(end);
        if self.dfa.matching_at_end[state as usize] {
            self.builder
                .ins()
                .jump(self.found, &[BlockArg::Value(start), BlockArg::Value(position)]);
        } else {
            self.builder
                .ins()
                .jump(self.after, &[BlockArg::Value(start), BlockArg::Value(last)]);
        }
        self.builder.switch_to_block(step);
        let byte = self.load(position);
        let one = self.builder.ins().iconst(types::I64, 1);
        let next = self.builder.ins().iadd(position, one);
        let mut targets: Vec<(u32, ByteSet)> = Vec::new();
        for value in 0..=255u8 {
            let target = self.dfa.next(state, value);
            if target == DEAD {
                continue;
            }
            match targets.iter_mut().find(|(existing, _)| *existing == target) {
                Some((_, set)) => set.insert(value),
                None => {
                    let mut set = ByteSet::default();
                    set.insert(value);
                    targets.push((target, set));
                }
            }
        }
        let args = [start, next, last].map(BlockArg::Value);
        for (target, set) in targets {
            let block = self.state_block(target);
            let otherwise = self.builder.create_block();
            let test = self.class(byte, set);
            self.builder.ins().brif(test, block, &args, otherwise, &[]);
            self.builder.switch_to_block(otherwise);
        }
        self.builder
            .ins()
            .jump(self.after, &[BlockArg::Value(start), BlockArg::Value(last)]);
    }

    fn after(&mut self) {
        let after = self.after;
        let start = self.builder.append_block_param(after, types::I64);
        let last = self.builder.append_block_param(after, types::I64);
        self.builder.switch_to_block(after);
        let matched = self
            .builder
            .ins()
            .icmp_imm_s(IntCC::SignedGreaterThanOrEqual, last, 0);
        let retry = self.builder.create_block();
        self.builder.ins().brif(
            matched,
            self.found,
            &[BlockArg::Value(start), BlockArg::Value(last)],
            retry,
            &[],
        );
        self.builder.switch_to_block(retry);
        let one = self.builder.ins().iconst(types::I64, 1);
        let advanced = self.builder.ins().iadd(start, one);
        self.builder
            .ins()
            .jump(self.outer, &[BlockArg::Value(advanced)]);
    }

    fn found(&mut self) {
        let found = self.found;
        let start = self.builder.append_block_param(found, types::I64);
        let end = self.builder.append_block_param(found, types::I64);
        self.builder.switch_to_block(found);
        let shift = self.builder.ins().iconst(types::I64, 32);
        let high = self.builder.ins().binary(Binary::ShiftLeft, start, shift);
        let packed = self.builder.ins().binary(Binary::Or, high, end);
        self.builder.ins().return_(&[packed]);
    }

    fn none(&mut self) {
        self.builder.switch_to_block(self.none);
        let none = self.builder.ins().iconst(types::I64, NO_MATCH as i64);
        self.builder.ins().return_(&[none]);
    }

    fn load(&mut self, position: Value) -> Value {
        self.builder
            .indexed_load(self.subject, position, MemoryFormat::U8, types::I8)
    }

    fn class(&mut self, byte: Value, set: ByteSet) -> Value {
        if set.is_empty() {
            return self.builder.ins().iconst(types::I8, 0);
        }
        self.builder
            .semantic(GraphOp::Regex(Op::Class(byte, set)), super::Repr::Boolean)
    }
}

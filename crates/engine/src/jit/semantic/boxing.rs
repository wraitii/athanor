use crate::FxHashMap;
use oxc_span::Span;

use super::contracts::{Use, repr_contract as contract};
use super::{
    Block, BlockData, Cfg, Facts, Graph, Inst, InstData, Op, Repr, Target, Terminator,
    TerminatorKind, Value, ValueDef,
};

const NUMERIC: super::ValueType = super::ValueType::NUMBER.union(super::ValueType::UNDEFINED);

pub(super) fn conditions(graph: &mut Graph, facts: &mut Facts) {
    for block in facts.cfg.rpo.clone() {
        let terminator = graph.blocks[block].terminator.as_ref().unwrap();
        let TerminatorKind::Branch(value, ..) = terminator.kind else {
            continue;
        };
        let types = facts.of(value);
        if types.within(NUMERIC) || types.within(super::ValueType::BOOLEAN) {
            continue;
        }
        let (span, source_name) = (terminator.span, terminator.source_name);
        let end = graph.blocks[block].insts.len();
        let (inst, truth) = graph.insert(block, end, span, Op::Truthy(value), true);
        graph.insts[inst].source_name = source_name;
        let truth = truth.unwrap();
        facts.set(truth, super::ValueType::BOOLEAN);
        let Some(Terminator {
            kind: TerminatorKind::Branch(condition, ..),
            ..
        }) = &mut graph.blocks[block].terminator
        else {
            unreachable!()
        };
        *condition = truth;
    }
}

struct Boxing<'a> {
    graph: &'a mut Graph,
    facts: &'a mut Facts,
}

impl Boxing<'_> {
    fn convert(
        &mut self,
        block: Block,
        span: Span,
        source_name: Option<(usize, usize)>,
        value: Value,
        repr: Repr,
    ) -> (Inst, Value) {
        let op = match self.graph.defining_op(value) {
            Some(Op::Constant(constant)) if super::repr::fits(constant, repr) => {
                Op::Constant(constant.clone())
            }
            _ if repr == Repr::Boxed => Op::Box(value),
            _ => Op::Unbox(value, repr),
        };
        let inst = self.graph.insts.next_key();
        let result = self.graph.values.push(ValueDef::Result(inst, 0));
        self.graph.insts.push(InstData {
            source_name,
            emission_location: None,
            span,
            block,
            results: smallvec::smallvec![result],
            op,
        });
        self.graph.reprs[result] = Some(repr);
        self.facts.set(result, repr.coerce(self.facts.of(value)));
        (inst, result)
    }

    fn results(&mut self, rpo: &[Block]) {
        let mut produced = FxHashMap::default();
        for &block in rpo {
            for &inst in &self.graph.blocks[block].insts {
                let Some(value) = self.graph.insts[inst].result() else {
                    continue;
                };
                let Some(repr) = contract(self.graph, inst).produces else {
                    continue;
                };
                if boundary(self.graph.repr(value), repr) {
                    produced.insert(inst, repr);
                }
            }
        }
        if produced.is_empty() {
            return;
        }
        let mut aliases = FxHashMap::default();
        let mut converts = Vec::new();
        for &block in rpo {
            let insts = std::mem::take(&mut self.graph.blocks[block].insts);
            let mut rebuilt = Vec::with_capacity(insts.len());
            for inst in insts {
                rebuilt.push(inst);
                let Some(&repr) = produced.get(&inst) else {
                    continue;
                };
                let value = self.graph.insts[inst].result().unwrap();
                let assigned = self.graph.repr(value);
                self.graph.reprs[value] = Some(repr);
                let data = &self.graph.insts[inst];
                let (span, source_name) = (data.span, data.source_name);
                let (convert, converted) = self.convert(block, span, source_name, value, assigned);
                rebuilt.push(convert);
                converts.push((convert, value));
                aliases.insert(value, converted);
            }
            self.graph.blocks[block].insts = rebuilt;
        }
        self.graph.replace_uses(&aliases);
        let originals: FxHashMap<Value, Value> = aliases
            .iter()
            .map(|(&value, &converted)| (converted, value))
            .collect();
        for data in self.graph.insts.values_mut() {
            if let Op::ContinuationOutput(call, _) = &mut data.op
                && let Some(&original) = originals.get(call)
            {
                *call = original;
            }
        }
        for (convert, value) in converts {
            let repr = self.graph.repr(aliases[&value]);
            self.graph.insts[convert].op = if repr == Repr::Boxed {
                Op::Box(value)
            } else {
                Op::Unbox(value, repr)
            };
        }
    }

    fn operands(&mut self, block: Block) {
        let mut cache: FxHashMap<(Value, Repr), Value> = FxHashMap::default();
        let insts = std::mem::take(&mut self.graph.blocks[block].insts);
        let mut rebuilt = Vec::with_capacity(insts.len());
        for inst in insts {
            if let Op::Convert(value, repr) = self.graph.insts[inst].op
                && boundary(self.graph.repr(value), repr)
            {
                self.graph.insts[inst].op = if repr == Repr::Boxed {
                    Op::Box(value)
                } else {
                    Op::Unbox(value, repr)
                };
            }
            let uses = contract(self.graph, inst).uses;
            let operands: smallvec::SmallVec<[Value; 4]> =
                self.graph.insts[inst].op.operands_iter().collect();
            debug_assert_eq!(uses.len(), operands.len());
            let mut replaced = smallvec::SmallVec::<[Value; 4]>::with_capacity(operands.len());
            for (&value, use_) in operands.iter().zip(uses) {
                let Use::Exact(repr) = use_ else {
                    replaced.push(value);
                    continue;
                };
                if !boundary(self.graph.repr(value), repr) {
                    replaced.push(value);
                    continue;
                }
                let converted = match cache.get(&(value, repr)) {
                    Some(&converted) => converted,
                    None => {
                        let data = &self.graph.insts[inst];
                        let (span, source_name) = (data.span, data.source_name);
                        let (convert, converted) =
                            self.convert(block, span, source_name, value, repr);
                        rebuilt.push(convert);
                        cache.insert((value, repr), converted);
                        converted
                    }
                };
                replaced.push(converted);
            }
            for (operand, value) in self.graph.insts[inst].op.operands_mut().zip(replaced) {
                *operand = value;
            }
            rebuilt.push(inst);
        }
        self.graph.blocks[block].insts = rebuilt;
        self.terminator(block, &mut cache);
    }

    fn terminator(&mut self, block: Block, cache: &mut FxHashMap<(Value, Repr), Value>) {
        let terminator = self.graph.blocks[block].terminator.clone().unwrap();
        let (span, source_name) = (terminator.span, terminator.source_name);
        let location = terminator.emission_location;
        let mut tail = |this: &mut Self, value: Value, repr: Repr| {
            if !boundary(this.graph.repr(value), repr) {
                return value;
            }
            if let Some(&converted) = cache.get(&(value, repr)) {
                return converted;
            }
            let (convert, converted) = this.convert(block, span, source_name, value, repr);
            this.graph.blocks[block].insts.push(convert);
            cache.insert((value, repr), converted);
            converted
        };
        let kind = match terminator.kind {
            TerminatorKind::Jump(target) => TerminatorKind::Jump(self.target(target, &mut tail)),
            TerminatorKind::CheckException(target, error) => {
                TerminatorKind::CheckException(self.target(target, &mut tail), error)
            }
            TerminatorKind::Return(value) => TerminatorKind::Return(tail(self, value, Repr::Boxed)),
            TerminatorKind::Branch(value, yes, no) => {
                let value = if self.graph.repr(value) == Repr::Boxed
                    && self.facts.of(value).within(NUMERIC)
                {
                    tail(self, value, Repr::F64)
                } else {
                    value
                };
                let value = tail(self, value, Repr::Boolean);
                let yes = self.edge(block, (span, source_name, location), yes);
                let no = self.edge(block, (span, source_name, location), no);
                TerminatorKind::Branch(value, yes, no)
            }
            kind => kind,
        };
        self.graph.blocks[block].terminator.as_mut().unwrap().kind = kind;
    }

    fn target(
        &mut self,
        mut target: Target,
        tail: &mut impl FnMut(&mut Self, Value, Repr) -> Value,
    ) -> Target {
        for index in 0..target.args.len() {
            let repr = self
                .graph
                .repr(self.graph.blocks[target.block].params[index]);
            target.args[index] = tail(self, target.args[index], repr);
        }
        target
    }

    fn edge(
        &mut self,
        from: Block,
        (span, source_name, location): (Span, Option<(usize, usize)>, Option<u32>),
        target: Target,
    ) -> Target {
        let params = &self.graph.blocks[target.block].params;
        let mismatched = target
            .args
            .iter()
            .zip(params)
            .any(|(&arg, &param)| boundary(self.graph.repr(arg), self.graph.repr(param)));
        if !mismatched {
            return target;
        }
        let edge = self.graph.blocks.push(BlockData::default());
        if self.graph.cold.contains(&from) || self.graph.cold.contains(&target.block) {
            self.graph.cold.insert(edge);
        }
        let mut args = Vec::with_capacity(target.args.len());
        for index in 0..target.args.len() {
            let arg = target.args[index];
            let repr = self
                .graph
                .repr(self.graph.blocks[target.block].params[index]);
            if !boundary(self.graph.repr(arg), repr) {
                args.push(arg);
                continue;
            }
            let (convert, converted) = self.convert(edge, span, source_name, arg, repr);
            self.graph.blocks[edge].insts.push(convert);
            args.push(converted);
        }
        self.graph.blocks[edge].terminator = Some(Terminator {
            source_name,
            emission_location: location,
            span,
            kind: TerminatorKind::Jump(Target {
                block: target.block,
                args,
            }),
        });
        Target {
            block: edge,
            args: Vec::new(),
        }
    }
}

fn boundary(from: Repr, to: Repr) -> bool {
    (from == Repr::Boxed) != (to == Repr::Boxed)
}

pub(super) fn insert(graph: &mut Graph, facts: &mut Facts) {
    let rpo = facts.cfg.rpo.clone();
    let blocks = graph.blocks.len();
    let mut boxing = Boxing { graph, facts };
    boxing.results(&rpo);
    for &block in &rpo {
        boxing.operands(block);
    }
    if boxing.graph.blocks.len() != blocks {
        boxing.facts.cfg = Cfg::new(boxing.graph);
    }
}

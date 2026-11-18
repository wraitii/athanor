use std::ops::Range;

use super::super::transforms::loops::natural_loop;
use super::super::{Block, Cfg, Graph, JsOp, Op, Target, TerminatorKind, Value, ValueDef};
use super::{
    DEPTH_LIMIT, IntRange, LowOp, QUERY_LIMIT, Relation, guards, induction_step, integer_literal,
    repr_range,
};
use crate::FxHashMap;

pub(in crate::jit::semantic) struct LoopEndpoints<'a> {
    graph: &'a Graph,
    cfg: &'a Cfg,
    blocks: FxHashMap<Block, usize>,
    last: FxHashMap<Value, (Range<usize>, i64)>,
}

impl<'a> LoopEndpoints<'a> {
    pub(in crate::jit::semantic) fn discover(
        graph: &'a Graph,
        cfg: &'a Cfg,
        header: Block,
    ) -> Option<Self> {
        let mut endpoints = Self {
            graph,
            cfg,
            blocks: FxHashMap::default(),
            last: FxHashMap::default(),
        };
        endpoints.trace_loop(header, 0)?;
        Some(endpoints)
    }

    pub(in crate::jit::semantic) fn reaches(&self, value: Value, block: Block, end: i64) -> bool {
        let Some(&position) = self.blocks.get(&block) else {
            return false;
        };
        let mut budget = QUERY_LIMIT;
        self.evaluate(value, position, &mut budget, 0) == Some(end)
    }

    fn record(&mut self, block: Block) -> Option<()> {
        let position = self.blocks.len();
        (position < QUERY_LIMIT && self.blocks.insert(block, position).is_none()).then_some(())
    }

    fn trace_loop(&mut self, header: Block, depth: usize) -> Option<Block> {
        if depth >= DEPTH_LIMIT {
            return None;
        }
        let body = natural_loop(self.cfg, header)?;
        let [first, second] = self.cfg.preds[header][..] else {
            return None;
        };
        let (entry, latch) = if body.contains(&first) {
            (second, first)
        } else {
            (first, second)
        };
        if body.contains(&entry) || !body.contains(&latch) {
            return None;
        }
        let entry = self.jump_to(entry, header)?;
        let backedge = self.jump_to(latch, header)?;
        let mut test = header;
        loop {
            if !body.contains(&test) || test == latch {
                return None;
            }
            self.record(test)?;
            let Some(next) = self.linear_successor(test) else {
                break;
            };
            test = next;
        }
        let TerminatorKind::Branch(condition, yes, no) = self.terminator(test)? else {
            return None;
        };
        let (mut condition, mut entered, mut exit) = (*condition, yes.block, no.block);
        if let Some(positive) = guards::negated(self.graph, condition) {
            condition = positive;
            std::mem::swap(&mut entered, &mut exit);
        }
        let (mut relation, mut counter, mut limit) =
            Relation::comparison(self.graph.defining_op(condition)?)?;
        if !body.contains(&entered) {
            relation = relation.negate();
            std::mem::swap(&mut entered, &mut exit);
        }
        if body.contains(&exit) || !self.cfg.dominates(entered, latch) {
            return None;
        }
        if matches!(self.graph.values[limit], ValueDef::Param(block, _) if block == header) {
            relation = relation.swap();
            std::mem::swap(&mut counter, &mut limit);
        }
        let ValueDef::Param(block, index) = self.graph.values[counter] else {
            return None;
        };
        if block != header {
            return None;
        }
        let start = integer_literal(self.graph, entry.args[index])?;
        let limit = integer_literal(self.graph, limit)?;
        let update = backedge.args[index];
        let last = match (induction_step(self.graph, update)?, relation) {
            ((operand, 1), Relation::Less) if operand == counter => limit.checked_sub(1)?,
            ((operand, 1), Relation::LessEqual) if operand == counter => limit,
            _ => return None,
        };
        let iterations = IntRange::new(start, last.checked_add(1)?);
        if start > last || !iterations.within(IntRange::I32) {
            return None;
        }
        if matches!(self.graph.defining_op(update), Some(Op::LowLevel(_)))
            && (!iterations.within(repr_range(self.graph, counter)?)
                || !iterations.within(repr_range(self.graph, update)?))
        {
            return None;
        }
        let begin = self.blocks.len();
        let mut cursor = entered;
        // Every successful iteration follows this path. A nested loop contributes
        // its own last iteration before the path continues at its exit.
        while cursor != latch {
            if !body.contains(&cursor) || self.blocks.contains_key(&cursor) {
                return None;
            }
            if self.cfg.preds[cursor]
                .iter()
                .any(|&pred| self.cfg.dominates(cursor, pred))
            {
                cursor = self.trace_loop(cursor, depth + 1)?;
            } else {
                self.record(cursor)?;
                cursor = self.linear_successor(cursor)?;
            }
        }
        self.record(latch)?;
        self.last.insert(counter, (begin..self.blocks.len(), last));
        Some(exit)
    }

    fn terminator(&self, block: Block) -> Option<&TerminatorKind> {
        Some(&self.graph.blocks[block].terminator.as_ref()?.kind)
    }

    fn jump_to(&self, from: Block, to: Block) -> Option<&'a Target> {
        match &self.graph.blocks[from].terminator.as_ref()?.kind {
            TerminatorKind::Jump(target) if target.block == to => Some(target),
            _ => None,
        }
    }

    fn linear_successor(&self, block: Block) -> Option<Block> {
        match self.terminator(block)? {
            TerminatorKind::Jump(target) => Some(target.block),
            TerminatorKind::Branch(_, yes, no) if self.bails_out(no.block) => Some(yes.block),
            TerminatorKind::Branch(_, yes, no) if self.bails_out(yes.block) => Some(no.block),
            _ => None,
        }
    }

    fn bails_out(&self, mut block: Block) -> bool {
        for _ in 0..DEPTH_LIMIT {
            if !self.graph.cold.contains(&block) {
                return false;
            }
            match self.terminator(block) {
                Some(TerminatorKind::Bailout { .. }) => return true,
                Some(TerminatorKind::Jump(target)) => block = target.block,
                _ => return false,
            }
        }
        false
    }

    fn evaluate(
        &self,
        value: Value,
        position: usize,
        budget: &mut usize,
        depth: usize,
    ) -> Option<i64> {
        if depth >= DEPTH_LIMIT || *budget == 0 {
            return None;
        }
        *budget -= 1;
        if let Some(value) = integer_literal(self.graph, value) {
            return Some(value);
        }
        if let Some((blocks, last)) = self.last.get(&value) {
            return blocks.contains(&position).then_some(*last);
        }
        let (op, left, right, width) = match *self.graph.defining_op(value)? {
            Op::Javascript(JsOp::AsInteger(inner) | JsOp::AsNumber(inner)) => {
                return self.evaluate(inner, position, budget, depth + 1);
            }
            Op::Javascript(JsOp::Binary(op, left, right)) => (
                super::super::js::facts::integer_arithmetic(op)?,
                left,
                right,
                None,
            ),
            Op::LowLevel(LowOp::Binary { op, left, right }) if op.is_integer_arithmetic() => {
                (op, left, right, Some(repr_range(self.graph, value)?))
            }
            _ => return None,
        };
        let left = IntRange::point(self.evaluate(left, position, budget, depth + 1)?);
        let right = IntRange::point(self.evaluate(right, position, budget, depth + 1)?);
        let range = super::super::js::facts::exact_integer_arithmetic(op, left, right)?;
        width
            .is_none_or(|width| range.within(width))
            .then_some(range.max)
    }
}

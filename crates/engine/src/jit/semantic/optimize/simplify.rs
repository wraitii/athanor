use super::*;

pub(in super::super) fn typeof_comparisons(graph: &mut Graph) {
    let uses = graph.use_counts();
    for block in Cfg::new(graph).rpo {
        for inst in graph.blocks[block].insts.clone() {
            let Op::Binary(operator, left, right) = graph.insts[inst].op else {
                continue;
            };
            if !matches!(
                operator,
                BinaryOperator::StrictEquality | BinaryOperator::StrictInequality
            ) {
                continue;
            }
            let pair = [(left, right), (right, left)]
                .into_iter()
                .find_map(|(value, literal)| {
                    let Op::Unary(UnaryOp::Typeof, operand) = graph.defining_op(value)? else {
                        return None;
                    };
                    let Op::Constant(Constant::String(text)) = graph.defining_op(literal)? else {
                        return None;
                    };
                    let kind = [
                        "string",
                        "number",
                        "boolean",
                        "undefined",
                        "object",
                        "function",
                    ]
                    .iter()
                    .position(|name| text == name)?;
                    (uses[value] == 1).then_some((value, literal, *operand, kind as u8))
                });
            let Some((value, literal, operand, kind)) = pair else {
                continue;
            };
            let ValueDef::Result(typeof_inst) = graph.values[value] else {
                unreachable!();
            };
            // Keep TDZ errors at the original typeof evaluation point.
            graph.insts[typeof_inst].op = Op::CheckInitialized(operand);
            graph.insts[typeof_inst].result = None;
            if uses[literal] == 1 {
                let ValueDef::Result(literal_inst) = graph.values[literal] else {
                    unreachable!();
                };
                graph.remove_inst(literal_inst);
            }
            let test = if kind == 0 {
                Op::IsString(operand)
            } else if kind == 1 {
                Op::IsNumber(operand)
            } else {
                Op::IsTypeof(operand, kind)
            };
            graph.insts[inst].op = if operator == BinaryOperator::StrictEquality {
                test
            } else {
                let position = graph.blocks[block]
                    .insts
                    .iter()
                    .position(|&i| i == inst)
                    .unwrap();
                let span = graph.insts[inst].span;
                let (_, result) = graph.insert(block, position, span, test, true);
                Op::Unary(UnaryOp::Not, result.unwrap())
            };
        }
    }
}

pub(in super::super) fn coercions(graph: &mut Graph) {
    let facts = Facts::compute(graph);
    for block in Cfg::new(graph).rpo {
        for inst in graph.blocks[block].insts.clone() {
            let op = &graph.insts[inst].op;
            let bitwise = matches!(
                op,
                Op::Unary(UnaryOp::BitwiseNot, _)
                    | Op::Binary(
                        BinaryOperator::BitwiseAnd
                            | BinaryOperator::BitwiseOR
                            | BinaryOperator::BitwiseXOR
                            | BinaryOperator::ShiftLeft
                            | BinaryOperator::ShiftRight
                            | BinaryOperator::ShiftRightZeroFill,
                        ..
                    )
            );
            if !bitwise {
                continue;
            }
            if !op
                .operands_iter()
                .all(|value| facts.of(value).within(NUMERIC))
                || op
                    .operands_iter()
                    .all(|value| facts.of(value).within(Types::INT32))
            {
                continue;
            }
            let mut op = op.clone();
            let mut position = graph.blocks[block]
                .insts
                .iter()
                .position(|&candidate| candidate == inst)
                .unwrap();
            let span = graph.insts[inst].span;
            for operand in op.operands_mut() {
                if facts.of(*operand).within(Types::INT32) {
                    continue;
                }
                *operand = graph
                    .insert(block, position, span, Op::ToInt32(*operand), true)
                    .1
                    .unwrap();
                position += 1;
            }
            graph.insts[inst].op = op;
        }
    }
}

pub(in super::super) fn bit_tests(graph: &mut Graph) {
    let facts = Facts::compute(graph);
    let uses = graph.use_counts();
    for block in Cfg::new(graph).rpo {
        let terminator = graph.blocks[block].terminator.as_ref().unwrap();
        let TerminatorKind::Branch(condition, _, _) = terminator.kind else {
            continue;
        };
        // The sign-bit mask may be negative; this rewrite preserves only truthiness.
        if uses[condition] != 1 {
            continue;
        }
        let Some(&Op::Binary(BinaryOperator::BitwiseAnd, left, right)) =
            graph.defining_op(condition)
        else {
            continue;
        };
        let test = [(left, right), (right, left)]
            .into_iter()
            .find_map(|(word, mask)| {
                let &Op::Binary(BinaryOperator::ShiftLeft, one, index) = graph.defining_op(mask)?
                else {
                    return None;
                };
                (matches!(
                    graph.defining_op(one),
                    Some(Op::Constant(Constant::Number(1.0)))
                ) && facts.of(word).within(Types::INT32)
                    && facts.of(index).within(Types::INT32))
                .then_some((word, index, one))
            });
        let Some((word, index, one)) = test else {
            continue;
        };
        let ValueDef::Result(inst) = graph.values[condition] else {
            unreachable!();
        };
        let data = &graph.insts[inst];
        let span = data.span;
        let source_name = data.source_name;
        let block = data.block;
        let position = graph.blocks[block]
            .insts
            .iter()
            .position(|&candidate| candidate == inst)
            .unwrap();
        let (shift, shifted) = graph.insert(
            block,
            position,
            span,
            Op::Binary(BinaryOperator::ShiftRight, word, index),
            true,
        );
        graph.insts[shift].source_name = source_name;
        graph.insts[inst].op = Op::Binary(BinaryOperator::BitwiseAnd, shifted.unwrap(), one);
    }
}

fn refine_string_guards(graph: &mut Graph, cfg: &Cfg) -> bool {
    let mut checks: HashMap<(bool, Value), Vec<(Block, bool)>> = HashMap::new();
    for &block in &cfg.rpo {
        let Some(Terminator {
            kind: TerminatorKind::Branch(condition, yes, no),
            ..
        }) = &graph.blocks[block].terminator
        else {
            continue;
        };
        if yes.block == no.block {
            continue;
        }
        let key = match graph.defining_op(*condition) {
            Some(Op::IsString(value)) => (false, *value),
            Some(Op::IsStringSplit(value)) => (true, *value),
            _ => continue,
        };
        for (target, result) in [(yes.block, true), (no.block, false)] {
            if target != graph.entry
                && cfg.preds[target].len() == 1
                && cfg.preds[target][0] == block
            {
                checks.entry(key).or_default().push((target, result));
            }
        }
    }
    if checks.is_empty() {
        return false;
    }
    let mut changed = false;
    for &block in &cfg.rpo {
        for &inst in &graph.blocks[block].insts {
            let key = match graph.insts[inst].op {
                Op::IsString(value) => (false, value),
                Op::IsStringSplit(value) => (true, value),
                _ => continue,
            };
            let Some((_, result)) = checks
                .get(&key)
                .and_then(|edges| edges.iter().find(|(entry, _)| cfg.dominates(*entry, block)))
            else {
                continue;
            };
            graph.insts[inst].op = Op::Constant(Constant::Boolean(*result));
            changed = true;
        }
    }
    changed
}

pub(in super::super) fn simplify(graph: &mut Graph) {
    loop {
        let facts = Facts::compute(graph);
        let cfg = Cfg::new(graph);
        let uses = graph.use_counts();
        let mut aliases = HashMap::new();
        let mut dead = Vec::new();
        let mut changed = refine_string_guards(graph, &cfg);
        for &block in &cfg.rpo {
            for &inst in &graph.blocks[block].insts {
                let data = &mut graph.insts[inst];
                match data.op {
                    Op::GuardedCompare {
                        operator,
                        left,
                        right,
                        ..
                    } if [left, right]
                        .iter()
                        .all(|&value| facts.of(value).within(NUMERIC)) =>
                    {
                        data.op = Op::Binary(operator, left, right);
                        changed = true;
                    }
                    Op::IsIndexed(value) if facts.indexed(value) => {
                        data.op = Op::Constant(Constant::Boolean(true));
                        changed = true;
                    }
                    Op::IsString(value)
                        if facts.of(value).within(Types::STRING)
                            && facts.of(value).intersects(Types::STRING) =>
                    {
                        data.op = Op::Constant(Constant::Boolean(true));
                        changed = true;
                    }
                    Op::IsString(value)
                        if facts.of(value) != Types::default()
                            && !facts.of(value).intersects(Types::STRING) =>
                    {
                        data.op = Op::Constant(Constant::Boolean(false));
                        changed = true;
                    }
                    Op::IsUndefined(value)
                        if facts.of(value).within(Types::UNDEFINED)
                            && facts.of(value).intersects(Types::UNDEFINED) =>
                    {
                        data.op = Op::Constant(Constant::Boolean(true));
                        changed = true;
                    }
                    Op::IsUndefined(value)
                        if facts.of(value) != Types::default()
                            && !facts.of(value).intersects(Types::UNDEFINED) =>
                    {
                        data.op = Op::Constant(Constant::Boolean(false));
                        changed = true;
                    }
                    Op::Binary(
                        operator @ (BinaryOperator::StrictEquality
                        | BinaryOperator::StrictInequality),
                        left,
                        right,
                    ) => {
                        let left = facts.of(left);
                        let right = facts.of(right);
                        let compared =
                            if left.within(Types::UNDEFINED) && left.intersects(Types::UNDEFINED) {
                                Some(right)
                            } else if right.within(Types::UNDEFINED)
                                && right.intersects(Types::UNDEFINED)
                            {
                                Some(left)
                            } else {
                                None
                            };
                        let equal = compared.and_then(|types| {
                            if types.within(Types::UNDEFINED) && types.intersects(Types::UNDEFINED)
                            {
                                Some(true)
                            } else if types != Types::default()
                                && !types.intersects(Types::UNDEFINED)
                            {
                                Some(false)
                            } else {
                                None
                            }
                        });
                        if let Some(equal) = equal {
                            data.op = Op::Constant(Constant::Boolean(
                                equal == (operator == BinaryOperator::StrictEquality),
                            ));
                            changed = true;
                        }
                    }
                    Op::IsNumber(value) if facts.of(value).within(Types::NUMBER) => {
                        data.op = Op::Constant(super::Constant::Boolean(true));
                        changed = true;
                    }
                    Op::IsNumber(value) if !facts.of(value).intersects(Types::NUMBER) => {
                        data.op = Op::Constant(super::Constant::Boolean(false));
                        changed = true;
                    }
                    _ => {}
                }
                let data = &graph.insts[inst];
                if let (
                    Some(result),
                    Op::ToNumeric(value) | Op::Unary(UnaryOp::Plus, value) | Op::AsNumber(value),
                ) = (data.result, &data.op)
                    && facts.of(*value).within(Types::NUMBER)
                {
                    aliases.insert(result, *value);
                    dead.push(inst);
                } else if graph.contract(&data.op, &facts).is_pure()
                    && data.result.is_none_or(|result| uses[result] == 0)
                {
                    dead.push(inst);
                }
            }
        }
        graph.replace_uses(&aliases);
        for inst in &dead {
            graph.remove_inst(*inst);
        }
        for &block in &cfg.rpo {
            let Some(Terminator {
                kind: TerminatorKind::Branch(condition, yes, no),
                ..
            }) = &graph.blocks[block].terminator
            else {
                continue;
            };
            let Some(Op::Constant(super::Constant::Boolean(taken))) = graph.defining_op(*condition)
            else {
                continue;
            };
            let target = if *taken { yes } else { no }.clone();
            graph.blocks[block].terminator.as_mut().unwrap().kind = TerminatorKind::Jump(target);
            changed = true;
        }
        if !changed && dead.is_empty() {
            return;
        }
    }
}

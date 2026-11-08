use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::jit::planner::{ReadKind, Site};

use super::{
    Block, BlockData, Cfg, Constant, Facts, Graph, Inst, Op, Target, Terminator, TerminatorKind,
    Types, Value,
};

impl std::fmt::Debug for Plan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OsrPlan")
            .field("loops", &self.loops.len())
            .finish()
    }
}

pub(in crate::jit) struct Plan {
    pub(in crate::jit) graph: Graph,
    pub(in crate::jit) loops: Vec<Loop>,
}

pub(in crate::jit) struct Loop {
    pub(in crate::jit) header: Block,
    pub(in crate::jit) body: HashSet<Block>,
    pub(in crate::jit) opportunities: Vec<Opportunity>,
    pub(in crate::jit) transfer: Result<Transfer, String>,
}

pub(in crate::jit) struct Transfer {
    pub(in crate::jit) preheader: Block,
    pub(in crate::jit) entry: Target,
    pub(in crate::jit) inputs: Vec<Value>,
    pub(in crate::jit) exit: Block,
    pub(in crate::jit) outputs: Vec<Value>,
    pub(in crate::jit) backedges: Vec<Block>,
}

pub(in crate::jit) struct Opportunity {
    pub(in crate::jit) inst: Inst,
    pub(in crate::jit) site: Site,
    pub(in crate::jit) kind: Observation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::jit) enum Observation {
    Call,
    Read(ReadKind),
}

impl Plan {
    pub(super) fn discover(
        graph: &Graph,
        repository: Option<&super::specialize::Repository>,
        source: usize,
    ) -> Option<Self> {
        if graph.generator || graph.has_handlers {
            return None;
        }
        let cfg = Cfg::new(graph);
        let facts = Facts::compute(graph);
        let receivers: HashSet<_> = cfg
            .rpo
            .iter()
            .filter(|block| !graph.cold.contains(block))
            .flat_map(|&block| graph.blocks[block].insts.iter())
            .filter_map(|&inst| match graph.insts[inst].op {
                Op::GetIndex(object, _)
                | Op::TypedIndex(object, _, super::TypedRead { element: None, .. }) => Some(object),
                _ => None,
            })
            .collect();
        let mut opportunities = Vec::new();
        for &block in &cfg.rpo {
            if graph.cold.contains(&block) {
                continue;
            }
            for &inst in &graph.blocks[block].insts {
                let data = &graph.insts[inst];
                let site = Site {
                    source: data.source_name.map_or(source, |name| name.0),
                    start: data.span.start,
                    end: data.span.end,
                };
                let mut record = |kind| opportunities.push(Opportunity { inst, site, kind });
                match data.op {
                    Op::GetIndex(object, _)
                    | Op::TypedIndex(object, _, super::TypedRead { element: None, .. })
                        if super::optimize::element_guess(graph, object).is_none()
                            && repository.is_none_or(|repository| {
                                !repository
                                    .read_feedback
                                    .contains_key(&(site, ReadKind::Receiver))
                            }) =>
                    {
                        record(Observation::Read(ReadKind::Receiver));
                    }
                    Op::Call { target: None, .. } | Op::Construct { .. }
                        if repository
                            .is_none_or(|repository| !repository.feedback.contains_key(&site)) =>
                    {
                        record(Observation::Call);
                    }
                    _ => {}
                }
                if matches!(
                    data.op,
                    Op::Get(..) | Op::GetIndex(..) | Op::IndexedGet(..) | Op::ArrayIndex(..)
                ) && data.result.is_some_and(|value| {
                    receivers.contains(&value) && facts.of(value).intersects(Types::OBJECT)
                }) && repository.is_none_or(|repository| {
                    !repository
                        .read_feedback
                        .contains_key(&(site, ReadKind::Result))
                }) {
                    record(Observation::Read(ReadKind::Result));
                }
            }
        }
        let loops: Vec<_> = super::optimize::natural_loops(graph, &cfg)
            .into_iter()
            .filter_map(|(header, body)| {
                if graph.cold.contains(&header) {
                    return None;
                }
                let opportunities: Vec<_> = opportunities
                    .iter()
                    .filter(|opportunity| body.contains(&graph.insts[opportunity.inst].block))
                    .map(|opportunity| Opportunity {
                        inst: opportunity.inst,
                        site: opportunity.site,
                        kind: opportunity.kind,
                    })
                    .collect();
                if opportunities.is_empty() {
                    return None;
                }
                let transfer = transfer(graph, &cfg, header, &body);
                Some(Loop {
                    header,
                    body,
                    opportunities,
                    transfer,
                })
            })
            .collect();
        (!loops.is_empty()).then(|| Self {
            graph: graph.clone(),
            loops,
        })
    }

    pub(in crate::jit) fn instrument(self: &Arc<Self>, graph: &mut Graph) -> super::Result<()> {
        let facts = Facts::compute(&self.graph);
        for (index, candidate) in self.loops.iter().enumerate() {
            let Ok(transfer) = &candidate.transfer else {
                continue;
            };
            if transfer.outputs.is_empty()
                || transfer
                    .outputs
                    .iter()
                    .any(|&value| !facts.of(value).within(Types::NUMBER))
                || self.extract(index).is_err()
            {
                continue;
            }
            let span = graph.blocks[candidate.header]
                .terminator
                .as_ref()
                .unwrap()
                .span;
            let success = graph.blocks.push(BlockData::default());
            let start = graph
                .append(
                    transfer.preheader,
                    span,
                    Op::OsrStart(self.clone(), index, transfer.inputs.clone()),
                    true,
                )
                .1
                .unwrap();
            graph.blocks[transfer.preheader].terminator = Some(Terminator {
                source_name: None,
                span,
                kind: TerminatorKind::Branch(start, Target::new(success), transfer.entry.clone()),
            });
            let results: HashMap<_, _> = transfer
                .outputs
                .iter()
                .enumerate()
                .map(|(index, &value)| {
                    let result = graph
                        .append(success, span, Op::OsrOutput(start, index), true)
                        .1
                        .unwrap();
                    (value, result)
                })
                .collect();
            let original_params = graph.blocks[transfer.exit].params.clone();
            let mut args: Vec<_> = original_params.iter().map(|value| results[value]).collect();
            let mut aliases = HashMap::new();
            for &output in &transfer.outputs {
                if original_params.contains(&output) {
                    continue;
                }
                let param = graph.add_param(transfer.exit);
                aliases.insert(output, param);
                args.push(results[&output]);
                for &block in &candidate.body {
                    if graph.successors(block).contains(&transfer.exit) {
                        graph.push_edge_arg(block, transfer.exit, output);
                    }
                }
            }
            for (block, data) in graph.blocks.iter_mut() {
                if candidate.body.contains(&block) || block == success {
                    continue;
                }
                for &inst in &data.insts {
                    for value in graph.insts[inst].op.operands_mut() {
                        if let Some(&replacement) = aliases.get(value) {
                            *value = replacement;
                        }
                    }
                }
                if let Some(terminator) = &mut data.terminator {
                    for value in terminator.kind.operands_mut() {
                        if let Some(&replacement) = aliases.get(value) {
                            *value = replacement;
                        }
                    }
                }
            }
            graph.blocks[success].terminator = Some(Terminator {
                source_name: None,
                span,
                kind: TerminatorKind::Jump(Target {
                    block: transfer.exit,
                    args,
                }),
            });
            graph.verify()?;
            break;
        }
        Ok(())
    }

    pub(in crate::jit) fn extract(&self, index: usize) -> super::Result<Graph> {
        let candidate = &self.loops[index];
        let transfer = candidate.transfer.as_ref().map_err(Clone::clone)?;
        let mut graph = self.graph.clone();
        for (block, data) in graph.blocks.iter_mut() {
            if !candidate.body.contains(&block) {
                data.insts.clear();
                data.terminator = None;
                if block != transfer.exit {
                    data.params.clear();
                }
            }
        }
        for &block in &candidate.body {
            graph.blocks[block].insts.retain(|&inst| {
                !matches!(
                    graph.insts[inst].op,
                    Op::EnterScope(_)
                        | Op::LeaveScope(_)
                        | Op::NextIteration(_)
                        | Op::FreshIteration(_)
                        | Op::View(_)
                )
            });
        }
        graph.name += ".osr";
        graph.arity = None;
        graph.borrowed.clear();
        let entry = graph.blocks.push(BlockData::default());
        graph.entry = entry;
        let span = graph.span;
        let facts = Facts::compute(&self.graph);
        let aliases: HashMap<_, _> = transfer
            .inputs
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                let argument = if let Some(Op::Constant(constant)) = self.graph.defining_op(value) {
                    graph
                        .append(entry, span, Op::Constant(constant.clone()), true)
                        .1
                        .unwrap()
                } else {
                    let argument = graph
                        .append(entry, span, Op::Argument(index), true)
                        .1
                        .unwrap();
                    if facts.of(value).within(Types::NUMBER) {
                        graph
                            .append(entry, span, Op::AsNumber(argument), true)
                            .1
                            .unwrap()
                    } else {
                        argument
                    }
                };
                (value, argument)
            })
            .collect();
        graph.views.clear();
        for &block in &candidate.body {
            for &inst in &graph.blocks[block].insts {
                for value in graph.insts[inst].op.operands_mut() {
                    if let Some(&replacement) = aliases.get(value) {
                        *value = replacement;
                    }
                }
            }
            for value in graph.blocks[block]
                .terminator
                .as_mut()
                .unwrap()
                .kind
                .operands_mut()
            {
                if let Some(&replacement) = aliases.get(value) {
                    *value = replacement;
                }
            }
        }
        graph.blocks[entry].terminator = Some(Terminator {
            source_name: None,
            span,
            kind: TerminatorKind::Jump(Target {
                block: transfer.entry.block,
                args: transfer
                    .entry
                    .args
                    .iter()
                    .map(|value| aliases[value])
                    .collect(),
            }),
        });
        graph.append(
            transfer.exit,
            span,
            Op::OsrStore(transfer.inputs.len(), transfer.outputs.clone()),
            false,
        );
        let result = graph
            .append(transfer.exit, span, Op::Constant(Constant::Undefined), true)
            .1
            .unwrap();
        graph.blocks[transfer.exit].terminator = Some(Terminator {
            source_name: None,
            span,
            kind: TerminatorKind::Return(result),
        });
        graph.verify()?;
        Ok(graph)
    }

    pub(in crate::jit) fn report(&self, label: &str) -> String {
        let facts = Facts::compute(&self.graph);
        let mut report = String::new();
        for candidate in &self.loops {
            report += &format!(
                "[js jit osr] {label} {} blocks={} opportunities={} ",
                candidate.header,
                candidate.body.len(),
                candidate.opportunities.len()
            );
            match &candidate.transfer {
                Ok(transfer) => {
                    report += &format!(
                        "entry={} args={:?} exit={} outputs={:?} backedges={:?}\n",
                        transfer.preheader,
                        transfer.entry.args,
                        transfer.exit,
                        transfer.outputs,
                        transfer.backedges
                    );
                    for &value in &transfer.inputs {
                        report += &format!("  input {value}: {}", facts.of(value));
                        if let Some(op) = self.graph.defining_op(value) {
                            report += &format!(" = {}", self.graph.op_text(op));
                        }
                        report.push('\n');
                    }
                }
                Err(reason) => report += &format!("ineligible: {reason}\n"),
            }
            for opportunity in &candidate.opportunities {
                let data = &self.graph.insts[opportunity.inst];
                report += &format!(
                    "  {:?} source={} {} = {}\n",
                    opportunity.kind,
                    opportunity.site.source,
                    opportunity.site,
                    self.graph.op_text(&data.op)
                );
            }
        }
        report
    }
}

fn transfer(
    graph: &Graph,
    cfg: &Cfg,
    header: Block,
    body: &HashSet<Block>,
) -> Result<Transfer, String> {
    let preheader = super::optimize::preheader(graph, cfg, header, body)
        .ok_or("no unique unconditional preheader")?;
    let mut exits = HashSet::new();
    let mut inputs = HashSet::new();
    let mut backedges = Vec::new();
    for &block in &cfg.rpo {
        if !body.contains(&block) {
            continue;
        }
        for &inst in &graph.blocks[block].insts {
            match graph.insts[inst].op {
                Op::Read(_)
                | Op::Write(..)
                | Op::Initialize(..)
                | Op::Closure(..)
                | Op::Argument(_)
                | Op::This
                | Op::ClosureThis(_)
                | Op::EnterCaptures(..)
                | Op::Arguments
                | Op::Rest(_) => {
                    return Err("requires non-SSA frame state".into());
                }
                _ => {}
            }
            if let Op::EnterScope(scope)
            | Op::LeaveScope(scope)
            | Op::NextIteration(scope)
            | Op::FreshIteration(scope) = graph.insts[inst].op
                && graph.materialized(scope)
            {
                return Err("requires materialized scope state".into());
            }
            inputs.extend(
                graph.insts[inst]
                    .op
                    .operands_iter()
                    .filter(|&value| !body.contains(&graph.def_block(value))),
            );
            if let Some(&view) = graph.views.get(&inst)
                && !body.contains(&graph.def_block(view))
                && let Some(Op::View(object)) = graph.defining_op(view)
            {
                inputs.insert(*object);
            }
        }
        let terminator = &graph.blocks[block].terminator.as_ref().unwrap().kind;
        if !matches!(
            terminator,
            TerminatorKind::Jump(_) | TerminatorKind::Branch(..)
        ) {
            return Err("requires a nonlocal exit".into());
        }
        inputs.extend(
            terminator
                .operands_iter()
                .filter(|&value| !body.contains(&graph.def_block(value))),
        );
        for target in terminator.targets() {
            if target.block == header {
                backedges.push(block);
            } else if !body.contains(&target.block) {
                exits.insert(target.block);
            }
        }
    }
    if exits.len() != 1 {
        return Err(format!("requires one normal exit, found {}", exits.len()));
    }
    let exit = *exits.iter().next().unwrap();
    let TerminatorKind::Jump(entry) = &graph.blocks[preheader].terminator.as_ref().unwrap().kind
    else {
        unreachable!()
    };
    inputs.extend(entry.args.iter().copied());
    if inputs
        .iter()
        .any(|&value| !cfg.dominates(graph.def_block(value), preheader))
    {
        return Err("input does not dominate entry".into());
    }
    let mut outputs: HashSet<_> = graph.blocks[exit].params.iter().copied().collect();
    for &block in &cfg.rpo {
        if body.contains(&block) {
            continue;
        }
        for &inst in &graph.blocks[block].insts {
            outputs.extend(
                graph.insts[inst]
                    .op
                    .operands_iter()
                    .filter(|&value| body.contains(&graph.def_block(value))),
            );
        }
        outputs.extend(
            graph.blocks[block]
                .terminator
                .as_ref()
                .unwrap()
                .kind
                .operands_iter()
                .filter(|&value| body.contains(&graph.def_block(value))),
        );
    }
    if outputs.iter().any(|&value| {
        graph.def_block(value) != exit && !cfg.dominates(graph.def_block(value), exit)
    }) {
        return Err("output does not dominate exit".into());
    }
    let mut inputs: Vec<_> = inputs.into_iter().collect();
    let mut outputs: Vec<_> = outputs.into_iter().collect();
    inputs.sort();
    outputs.sort();
    Ok(Transfer {
        preheader,
        entry: entry.clone(),
        inputs,
        exit,
        outputs,
        backedges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxc_allocator::Allocator;
    use oxc_ast::ast::Statement;
    use oxc_parser::Parser;
    use oxc_span::SourceType;

    fn graph(source: &str) -> Graph {
        let allocator = Allocator::default();
        let parsed = Parser::new(&allocator, source, SourceType::script()).parse();
        assert!(parsed.diagnostics.is_empty());
        let Statement::FunctionDeclaration(function) = &parsed.program.body[0] else {
            panic!()
        };
        let semantic = oxc_semantic::SemanticBuilder::new_compiler()
            .build(&parsed.program)
            .semantic;
        let graph =
            super::super::build::lower(function, semantic.scoping(), &HashSet::new()).unwrap();
        super::super::canonicalize(graph, true).unwrap().graph
    }

    #[test]
    fn nested_loops_keep_their_own_transfer_state() {
        let graph = graph(
            "function f(rows,n){let total=0;for(let y=0;y<n;++y){let sum=0;for(let x=0;x<n;++x)sum+=rows[y][x];total+=sum;}return total;}",
        );
        let plan = Plan::discover(&graph, None, 0).unwrap();
        assert_eq!(plan.loops.len(), 2);
        let inner = &plan.loops[0];
        let transfer = inner.transfer.as_ref().unwrap();
        assert_eq!(plan.graph.blocks[inner.header].params.len(), 2);
        assert!(transfer.outputs.len() >= 1);
        assert_eq!(transfer.backedges.len(), 1);
        assert!(
            transfer
                .inputs
                .iter()
                .all(|&value| !inner.body.contains(&plan.graph.def_block(value)))
        );
        let mut optimized = graph;
        super::super::optimize::version(&mut optimized, &Default::default(), None, &mut Vec::new())
            .unwrap();
        plan.graph.verify().unwrap();
        assert_eq!(plan.loops.len(), 2);
    }

    #[test]
    fn arithmetic_only_loops_have_no_opportunities() {
        let graph = graph("function f(n){let sum=0;for(let i=0;i<n;++i)sum+=i;return sum;}");
        assert!(Plan::discover(&graph, None, 0).is_none());
    }

    #[test]
    fn multiple_normal_exits_are_not_transferable() {
        let graph = graph(
            "function f(xs,n){let sum=0;for(let i=0;i<n;++i){if(xs[i])return sum;sum+=xs[i];}return sum;}",
        );
        let plan = Plan::discover(&graph, None, 0).unwrap();
        assert!(plan.loops[0].transfer.is_err());
    }

    #[test]
    fn observed_reads_no_longer_request_discovery() {
        let graph = graph("function f(xs,n){let sum=0;for(let i=0;i<n;++i)sum+=xs[i];return sum;}");
        let initial = Plan::discover(&graph, None, 7).unwrap();
        let mut repository = super::super::specialize::Repository::default();
        for opportunity in &initial.loops[0].opportunities {
            let Observation::Read(kind) = opportunity.kind else {
                panic!()
            };
            repository.read_feedback.insert(
                (opportunity.site, kind),
                crate::jit::planner::ReadType::Array,
            );
        }
        assert!(Plan::discover(&graph, Some(&repository), 7).is_none());
    }
}

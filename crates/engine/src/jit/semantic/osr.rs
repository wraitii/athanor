use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::jit::planner::{Observation, Site};

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

impl Plan {
    pub(super) fn discover(graph: &Graph, source: usize) -> Option<Self> {
        if graph.generator || graph.has_handlers {
            return None;
        }
        let cfg = Cfg::new(graph);
        let opportunities: Vec<_> = cfg
            .rpo
            .iter()
            .filter(|block| !graph.cold.contains(block))
            .flat_map(|&block| graph.blocks[block].insts.iter())
            .filter_map(|&inst| {
                let Op::Observe(_, kind, mut site) = graph.insts[inst].op else {
                    return None;
                };
                if site.source == 0 {
                    site.source = source;
                }
                Some(Opportunity { inst, site, kind })
            })
            .collect();
        let loops: Vec<_> = super::optimize::natural_loops(graph, &cfg)
            .into_iter()
            .filter_map(|(header, body)| {
                if graph.cold.contains(&header) {
                    return None;
                }
                let transfer = transfer(graph, &cfg, header, &body);
                let opportunities: Vec<_> = opportunities
                    .iter()
                    .filter(|opportunity| {
                        let data = &graph.insts[opportunity.inst];
                        let Op::Observe(value, ..) = data.op else {
                            unreachable!()
                        };
                        body.contains(&data.block)
                            || cfg.dominates(data.block, header)
                                && transfer
                                    .as_ref()
                                    .is_ok_and(|transfer| transfer.inputs.contains(&value))
                    })
                    .map(|opportunity| Opportunity {
                        inst: opportunity.inst,
                        site: opportunity.site,
                        kind: opportunity.kind,
                    })
                    .collect();
                if opportunities.is_empty() {
                    return None;
                }
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
        for opportunity in &candidate.opportunities {
            let Op::Observe(value, kind, site) = self.graph.insts[opportunity.inst].op else {
                unreachable!()
            };
            if !candidate
                .body
                .contains(&self.graph.insts[opportunity.inst].block)
                && let Some(&argument) = aliases.get(&value)
            {
                graph.append(entry, span, Op::Observe(argument, kind, site), false);
            }
        }
        for (_, block) in self.graph.blocks.iter() {
            for &inst in &block.insts {
                if let Op::ReadHint(value, kind, observed, site) = self.graph.insts[inst].op
                    && let Some(&argument) = aliases.get(&value)
                {
                    graph.append(
                        entry,
                        span,
                        Op::ReadHint(argument, kind, observed, site),
                        false,
                    );
                }
            }
        }
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
        let mut graph = super::super::canonicalize(graph, true).unwrap().graph;
        super::super::observations::plan(&mut graph, None);
        graph
    }

    #[test]
    fn nested_loops_keep_their_own_transfer_state() {
        let graph = graph(
            "function f(rows,n){let total=0;for(let y=0;y<n;++y){let sum=0;for(let x=0;x<n;++x)sum+=rows[y][x];total+=sum;}return total;}",
        );
        let plan = Plan::discover(&graph, 0).unwrap();
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
        assert!(Plan::discover(&graph, 0).is_none());
    }

    #[test]
    fn multiple_normal_exits_are_not_transferable() {
        let graph = graph(
            "function f(xs,n){let sum=0;for(let i=0;i<n;++i){if(xs[i])return sum;sum+=xs[i];}return sum;}",
        );
        let plan = Plan::discover(&graph, 0).unwrap();
        assert!(plan.loops[0].transfer.is_err());
    }

    #[test]
    fn loop_inputs_keep_observations_before_the_loop() {
        let graph = graph(
            "function f(holder,n){const row=holder.row;let sum=0;for(let i=0;i<n;++i)sum+=row[i];return sum;}",
        );
        let plan = Plan::discover(&graph, 7).unwrap();
        let candidate = &plan.loops[0];
        let opportunity = candidate
            .opportunities
            .iter()
            .find(|opportunity| {
                opportunity.kind == Observation::Read(crate::jit::planner::ReadKind::Result)
                    && !candidate
                        .body
                        .contains(&graph.insts[opportunity.inst].block)
            })
            .unwrap();
        let Op::Observe(value, _, _) = graph.insts[opportunity.inst].op else {
            panic!()
        };
        assert!(candidate.transfer.as_ref().unwrap().inputs.contains(&value));
        assert_eq!(opportunity.site.source, 7);
        plan.extract(0).unwrap().verify().unwrap();
    }

    #[test]
    fn extracted_inputs_keep_known_storage_and_numeric_hints() {
        use crate::jit::planner::{ReadKind, ReadType};
        let mut graph = graph(
            "function f(holder,x,n){const row=holder.row;let sum=0;for(let i=0;i<n;++i)sum+=row[i]*x;return sum;}",
        );
        let cfg = Cfg::new(&graph);
        let (_, body) = super::super::optimize::natural_loops(&graph, &cfg).remove(0);
        let row = graph
            .blocks
            .iter()
            .filter(|(block, _)| !body.contains(block))
            .flat_map(|(_, block)| &block.insts)
            .find_map(|&inst| {
                matches!(&graph.insts[inst].op, Op::Get(_, name) if name == "row")
                    .then(|| graph.insts[inst].result.unwrap())
            })
            .unwrap();
        let x = graph
            .insts
            .values()
            .find_map(|data| matches!(data.op, Op::Argument(1)).then(|| data.result.unwrap()))
            .unwrap();
        let site = Site {
            source: 7,
            start: 1,
            end: 2,
        };
        for (value, kind, observed) in [
            (
                row,
                ReadKind::Result,
                ReadType::Typed(crate::heap::TypedKind::Float32),
            ),
            (x, ReadKind::Number, ReadType::Integer),
        ] {
            let block = graph.def_block(value);
            graph.append(
                block,
                graph.span,
                Op::ReadHint(value, kind, observed, site),
                false,
            );
        }
        let plan = Plan::discover(&graph, 7).unwrap();
        let mut extracted = plan.extract(0).unwrap();
        let mut repository = super::super::specialize::Repository::default();
        repository.read_feedback.insert(
            (site, ReadKind::Result),
            ReadType::Typed(crate::heap::TypedKind::Float32),
        );
        repository
            .read_feedback
            .insert((site, ReadKind::Number), ReadType::Integer);
        super::super::observations::plan(&mut extracted, Some((&repository, 7)));
        assert!(!super::super::observations::numeric_hints(&extracted).is_empty());
        let reads = super::super::reads::plan(&extracted, &repository, 7);
        assert!(
            reads
                .iter()
                .any(|read| read.element == Some(crate::heap::TypedKind::Float32))
        );
        extracted.verify().unwrap();
    }

    #[test]
    fn observations_survive_rejected_region_specialization() {
        let mut graph = graph(
            "function f(xs,n,callback){let sum=0;for(let i=0;i<n;++i)sum+=callback(xs[i]);return sum;}",
        );
        super::super::optimize::version(&mut graph, &Default::default(), None, &mut Vec::new())
            .unwrap();
        let plan = Plan::discover(&graph, 7).unwrap();
        let opportunities = &plan.loops[0].opportunities;
        assert!(
            opportunities
                .iter()
                .any(|opportunity| opportunity.kind == Observation::Call)
        );
        assert!(opportunities.iter().any(|opportunity| opportunity.kind
            == Observation::Read(crate::jit::planner::ReadKind::Receiver)));
        graph.verify().unwrap();
    }

    #[test]
    fn replanning_does_not_duplicate_observations() {
        let mut graph =
            graph("function f(xs,n){let sum=0;for(let i=0;i<n;++i)sum+=xs[i];return sum;}");
        let before: Vec<_> = graph
            .blocks
            .iter()
            .flat_map(|(_, block)| &block.insts)
            .filter(|&&inst| matches!(graph.insts[inst].op, Op::Observe(..)))
            .map(|&inst| graph.op_text(&graph.insts[inst].op))
            .collect();
        super::super::observations::plan(&mut graph, None);
        let after: Vec<_> = graph
            .blocks
            .iter()
            .flat_map(|(_, block)| &block.insts)
            .filter(|&&inst| matches!(graph.insts[inst].op, Op::Observe(..)))
            .map(|&inst| graph.op_text(&graph.insts[inst].op))
            .collect();
        assert!(!before.is_empty());
        assert_eq!(before, after);
        graph.verify().unwrap();
    }

    #[test]
    fn observed_reads_no_longer_request_discovery() {
        let graph = graph("function f(xs,n){let sum=0;for(let i=0;i<n;++i)sum+=xs[i];return sum;}");
        let initial = Plan::discover(&graph, 7).unwrap();
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
        let mut graph = graph;
        super::super::observations::plan(&mut graph, Some((&repository, 7)));
        assert!(Plan::discover(&graph, 7).is_none());
    }
}

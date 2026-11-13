use crate::{FxHashMap, FxHashSet};
use std::sync::Arc;

use cranelift_codegen::entity::SecondaryMap;
use oxc_syntax::operator::BinaryOperator;

use crate::jit::profile::{Observation, Site};

use super::continuation::{Shape, liveness, unbox};
use super::{
    Block, BlockData, Cfg, Constant, Facts, Graph, Inst, Op, Target, TerminatorKind, Value,
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
    pub(in crate::jit) body: FxHashSet<Block>,
    pub(in crate::jit) opportunities: Vec<Opportunity>,
    pub(in crate::jit) transfer: Result<Transfer, String>,
}

pub(in crate::jit) struct Transfer {
    pub(in crate::jit) inputs: Vec<Value>,
    pub(in crate::jit) params: Vec<Value>,
    pub(super) shape: Shape,
}

impl Transfer {
    pub(in crate::jit) fn arguments(&self) -> usize {
        self.inputs.len() + self.params.len()
    }

    pub(in crate::jit) fn words(&self) -> usize {
        self.arguments().max(self.shape.words())
    }
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
        if opportunities.is_empty() {
            return None;
        }
        let facts = Facts::compute(graph);
        let live = liveness(graph, &cfg.rpo);
        let loops: Vec<_> = super::optimize::natural_loops(graph, &cfg)
            .into_iter()
            .filter_map(|(header, body)| {
                if graph.cold.contains(&header) || header == graph.entry {
                    return None;
                }
                let transfer = transfer(graph, &cfg, &facts, &live, header, &body);
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
        let mut definitions = FxHashMap::default();
        for (index, candidate) in self.loops.iter().enumerate() {
            let Ok(transfer) = &candidate.transfer else {
                continue;
            };
            let header = candidate.header;
            let span = graph.blocks[header].terminator.as_ref().unwrap().span;
            let poll = graph.blocks.push(BlockData::default());
            let params: Vec<_> = transfer
                .params
                .iter()
                .map(|_| graph.add_param(poll))
                .collect();
            for data in graph.blocks.values_mut() {
                let Some(terminator) = &mut data.terminator else {
                    continue;
                };
                for target in terminator.kind.targets_mut() {
                    if target.block == header {
                        target.block = poll;
                    }
                }
            }
            let values = transfer.inputs.iter().chain(&params).copied().collect();
            let append = |graph: &mut Graph, op| graph.append_value(poll, span, op);
            let start = append(graph, Op::OsrStart(self.clone(), index, values));
            let declined = append(graph, Op::Constant(Constant::Number(-1.0)));
            let declined = append(
                graph,
                Op::Binary(BinaryOperator::StrictEquality, start, declined),
            );
            let dispatch = graph.blocks.push(BlockData::default());
            graph.terminate(
                poll,
                span,
                TerminatorKind::Branch(
                    declined,
                    Target {
                        block: header,
                        args: params,
                    },
                    Target::new(dispatch),
                ),
            );
            transfer
                .shape
                .dispatch(graph, dispatch, start, &mut definitions);
        }
        super::optimize::repair(graph, &definitions);
        graph.verify()
    }

    pub(in crate::jit) fn extract(&self, index: usize) -> super::Result<Graph> {
        let candidate = &self.loops[index];
        let transfer = candidate.transfer.as_ref().map_err(Clone::clone)?;
        let mut graph = self.graph.clone();
        for &block in &candidate.body {
            graph.blocks[block].insts.retain(|&inst| {
                !matches!(
                    graph.insts[inst].op,
                    Op::EnterScope(_)
                        | Op::LeaveScope(_)
                        | Op::NextIteration(_)
                        | Op::FreshIteration(_)
                )
            });
        }
        transfer
            .shape
            .store_exits(&self.graph, &mut graph, candidate.header);
        graph.name += ".osr";
        graph.arity = None;
        graph.borrowed.clear();
        graph.views.clear();
        let entry = graph.blocks.push(BlockData::default());
        graph.entry = entry;
        let span = graph.span;
        let facts = Facts::compute(&self.graph);
        let argument = |graph: &mut Graph, index: usize, value: Value| {
            if let Some(Op::Constant(constant)) = self.graph.defining_op(value) {
                return graph.append_value(entry, span, Op::Constant(constant.clone()));
            }
            let argument = graph.append_value(entry, span, Op::Argument(index));
            unbox(graph, entry, argument, facts.of(value))
        };
        let aliases: FxHashMap<_, _> = transfer
            .inputs
            .iter()
            .enumerate()
            .map(|(index, &value)| (value, argument(&mut graph, index, value)))
            .collect();
        let args = transfer
            .params
            .iter()
            .enumerate()
            .map(|(offset, &param)| argument(&mut graph, transfer.inputs.len() + offset, param))
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
                if let Op::Assume(value, fact, origin) = self.graph.insts[inst].op
                    && let Some(&argument) = aliases.get(&value)
                {
                    graph.append(entry, span, Op::Assume(argument, fact, origin), false);
                }
            }
        }
        graph.terminate(
            entry,
            span,
            TerminatorKind::Jump(Target {
                block: candidate.header,
                args,
            }),
        );
        let reachable: FxHashSet<_> = Cfg::new(&graph).rpo.into_iter().collect();
        for (block, data) in graph.blocks.iter_mut() {
            if !reachable.contains(&block) {
                data.insts.clear();
                data.params.clear();
                data.terminator = None;
                continue;
            }
            if block == entry {
                continue;
            }
            for &inst in &data.insts {
                for value in graph.insts[inst].op.operands_mut() {
                    if let Some(&replacement) = aliases.get(value) {
                        *value = replacement;
                    }
                }
            }
            for value in data.terminator.as_mut().unwrap().kind.operands_mut() {
                if let Some(&replacement) = aliases.get(value) {
                    *value = replacement;
                }
            }
        }
        graph.cold.retain(|block| reachable.contains(block));
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
                        "params={:?} exits={:?} returns={}\n",
                        transfer.params,
                        transfer
                            .shape
                            .exits
                            .iter()
                            .map(|exit| (exit.block, exit.outputs.len() + exit.params.len()))
                            .collect::<Vec<_>>(),
                        transfer.shape.returns.is_some()
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
    facts: &Facts,
    live: &SecondaryMap<Block, FxHashSet<Value>>,
    header: Block,
    body: &FxHashSet<Block>,
) -> Result<Transfer, String> {
    let mut inputs = FxHashSet::default();
    for &block in &cfg.rpo {
        let inside = body.contains(&block);
        for &inst in &graph.blocks[block].insts {
            if let Some(&view) = graph.views.get(&inst) {
                let defined = body.contains(&graph.def_block(view));
                if !inside && defined {
                    return Err("heap view escapes the loop".into());
                }
                if inside
                    && !defined
                    && let Some(Op::View(object)) = graph.defining_op(view)
                {
                    inputs.insert(*object);
                }
            }
            if !inside {
                continue;
            }
            match graph.insts[inst].op {
                Op::Read(_)
                | Op::Write(..)
                | Op::Initialize(..)
                | Op::Closure(..)
                | Op::Argument(_)
                | Op::EnterCaptures(..)
                | Op::Arguments
                | Op::Rest(_) => {
                    return Err(format!(
                        "requires non-SSA frame state: {}",
                        graph.op_text(&graph.insts[inst].op)
                    ));
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
        }
        if !inside {
            continue;
        }
        let terminator = &graph.blocks[block].terminator.as_ref().unwrap().kind;
        if !matches!(
            terminator,
            TerminatorKind::Jump(_) | TerminatorKind::Branch(..) | TerminatorKind::Return(_)
        ) {
            return Err("requires normal control flow".into());
        }
        inputs.extend(
            terminator
                .operands_iter()
                .filter(|&value| !body.contains(&graph.def_block(value))),
        );
    }
    if inputs
        .iter()
        .any(|&value| !cfg.dominates(graph.def_block(value), header))
    {
        return Err("input does not dominate the header".into());
    }
    let shape = Shape::new(graph, facts, live, body.clone());
    if shape.count() == 0 {
        return Err("requires an exit".into());
    }
    if shape.exits.iter().any(|exit| {
        exit.outputs
            .iter()
            .any(|&(value, _)| !cfg.dominates(graph.def_block(value), exit.block))
    }) {
        return Err("output does not dominate its exit".into());
    }
    let mut inputs: Vec<_> = inputs.into_iter().collect();
    inputs.sort();
    Ok(Transfer {
        inputs,
        params: graph.blocks[header].params.clone(),
        shape,
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
        let graph = super::super::build::lower(function, semantic.scoping(), &FxHashSet::default())
            .unwrap();
        let mut graph = super::super::canonicalize_graph(graph, true).unwrap().graph;
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
        assert_eq!(transfer.params.len(), 2);
        assert_eq!(transfer.shape.exits.len(), 1);
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
    fn osr_plan_keeps_typed_reads_before_memory_lowering() {
        use super::super::{FunctionState, MemoryOp, TypedRead};
        use crate::heap::TypedKind;

        let mut graph =
            graph("function f(xs,n){let sum=0;for(let i=0;i<n;++i)sum+=xs[i];return sum;}");
        let inst = graph
            .insts
            .iter()
            .find_map(|(inst, data)| matches!(data.op, Op::GetIndex(..)).then_some(inst))
            .unwrap();
        let Op::GetIndex(object, index) = graph.insts[inst].op else {
            unreachable!()
        };
        graph.insts[inst].op = Op::TypedIndex(
            object,
            index,
            TypedRead {
                element: Some(TypedKind::Uint16),
                bounded: true,
            },
        );
        let mut function = FunctionState::from_graph(graph);
        function.prepare_osr(0).unwrap();
        function.lower_memory().unwrap();
        let plan = function.osr.as_ref().unwrap();
        assert!(
            plan.graph
                .insts
                .values()
                .any(|data| matches!(data.op, Op::TypedIndex(..)))
        );
        assert!(
            !plan
                .graph
                .insts
                .values()
                .any(|data| matches!(data.op, Op::Memory(..)))
        );
        assert!(
            function
                .graph
                .insts
                .values()
                .any(|data| matches!(data.op, Op::Memory(MemoryOp::IndexedLoad { .. })))
        );
        plan.extract(0).unwrap().verify().unwrap();
    }

    #[test]
    fn arithmetic_only_loops_have_no_opportunities() {
        let graph = graph("function f(n){let sum=0;for(let i=0;i<n;++i)sum+=i;return sum;}");
        assert!(Plan::discover(&graph, 0).is_none());
    }

    #[test]
    fn early_returns_are_transferable() {
        let graph = graph(
            "function f(xs,n){let sum=0;for(let i=0;i<n;++i){if(xs[i])return sum;sum+=xs[i];}return sum;}",
        );
        let plan = Arc::new(Plan::discover(&graph, 0).unwrap());
        assert!(plan.loops[0].transfer.is_ok());
        plan.extract(0).unwrap().verify().unwrap();
        let mut instrumented = graph;
        plan.instrument(&mut instrumented).unwrap();
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
                opportunity.kind == Observation::Read(crate::jit::profile::ReadKind::Result)
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
        use crate::jit::profile::{ReadKind, ReadType};
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
                    .then(|| graph.insts[inst].result().unwrap())
            })
            .unwrap();
        let x = graph
            .insts
            .values()
            .find_map(|data| matches!(data.op, Op::Argument(1)).then(|| data.result().unwrap()))
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
            == Observation::Read(crate::jit::profile::ReadKind::Receiver)));
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
                crate::jit::profile::ReadType::Array,
            );
        }
        let mut graph = graph;
        super::super::observations::plan(&mut graph, Some((&repository, 7)));
        assert!(Plan::discover(&graph, 7).is_none());
    }
}

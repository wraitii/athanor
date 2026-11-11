use super::{
    Block, BlockData, Cfg, Constant, Facts, Graph, Op, Target, TerminatorKind, Types, Value,
};
use cranelift_codegen::entity::SecondaryMap;
use oxc_syntax::operator::BinaryOperator;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

pub(in crate::jit) struct Body {
    pub(in crate::jit) graph: Graph,
    pub(super) result: super::Types,
    pub(in crate::jit) outputs: usize,
}

impl Body {
    fn new(graph: Graph, outputs: usize) -> Self {
        let mut function = super::FunctionState::new(graph, Vec::new(), true);
        super::js::simplify(&mut function);
        super::js::coercions(&mut function);
        super::js::bit_tests(&mut function);
        if super::typed_lowering::lower_arrays(&mut function) {
            super::js::simplify(&mut function);
            super::optimize::reuse_loads(&mut function);
        }
        super::optimize::views(&mut function);
        super::js::simplify(&mut function);
        super::transforms::trim(&mut function.graph);
        super::transforms::remove_trivial_params(&mut function.graph);
        function.refresh_facts();
        let super::FunctionState {
            mut graph,
            facts,
            mut notes,
            ..
        } = function;
        super::borrow::borrows(&mut graph, &facts, &mut notes);
        let result =
            facts
                .cfg
                .rpo
                .iter()
                .fold(super::Types::default(), |types, &block| {
                    match graph.blocks[block].terminator.as_ref().unwrap().kind {
                        TerminatorKind::Return(value) => types.union(facts.of(value)),
                        _ => types,
                    }
                });
        Self {
            graph,
            result,
            outputs,
        }
    }
}

impl std::fmt::Debug for Body {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Continuation")
            .field("name", &self.graph.name)
            .finish()
    }
}

pub(super) fn liveness(graph: &Graph, rpo: &[Block]) -> SecondaryMap<Block, HashSet<Value>> {
    let mut live_in: SecondaryMap<_, HashSet<Value>> = SecondaryMap::new();
    loop {
        let mut changed = false;
        for &block in rpo.iter().rev() {
            let data = &graph.blocks[block];
            let mut live: HashSet<_> = graph
                .successors(block)
                .into_iter()
                .flat_map(|next| live_in[next].iter().copied())
                .collect();
            live.extend(data.terminator.as_ref().unwrap().kind.operands_iter());
            for &inst in data.insts.iter().rev() {
                let data = &graph.insts[inst];
                if let Some(result) = data.result {
                    live.remove(&result);
                }
                live.extend(data.op.operands_iter());
            }
            for param in &data.params {
                live.remove(param);
            }
            if live != live_in[block] {
                live_in[block] = live;
                changed = true;
            }
        }
        if !changed {
            return live_in;
        }
    }
}

fn append(graph: &mut Graph, block: Block, op: Op) -> Value {
    let span = graph.span;
    graph.append_value(block, span, op)
}

fn terminate(graph: &mut Graph, block: Block, kind: TerminatorKind) {
    graph.terminate(block, graph.span, kind);
}

pub(super) fn unbox(graph: &mut Graph, block: Block, value: Value, types: Types) -> Value {
    if types.intersects(Types::INTEGER) && types.within(Types::INTEGER) {
        append(graph, block, Op::AsInteger(value))
    } else if types.intersects(Types::NUMBER) && types.within(Types::NUMBER) {
        append(graph, block, Op::AsNumber(value))
    } else {
        value
    }
}

fn extract(graph: &Graph, target: Target, facts: &Facts) -> super::Result<(Graph, Vec<Value>)> {
    if graph.generator || graph.has_handlers {
        return Err("continuation requires a normal function without handlers".into());
    }
    if target.args.len() != graph.blocks[target.block].params.len() {
        return Err("continuation target has wrong arity".into());
    }
    let cfg = Cfg::from_entry(graph, target.block);
    let live_in = liveness(graph, &cfg.rpo);
    let mut inputs: Vec<_> = live_in[target.block]
        .iter()
        .copied()
        .chain(target.args.iter().copied())
        .collect();
    inputs.sort();
    inputs.dedup();
    let mut result = graph.clone();
    result.arity = None;
    result.borrowed.clear();
    let entry = result.blocks.push(BlockData::default());
    result.entry = entry;
    let mut aliases = HashMap::new();
    for (index, &value) in inputs.iter().enumerate() {
        let argument = append(&mut result, entry, Op::Argument(index));
        let argument = unbox(&mut result, entry, argument, facts.of(value));
        aliases.insert(value, argument);
    }
    let args = target.args.iter().map(|value| aliases[value]).collect();
    terminate(
        &mut result,
        entry,
        TerminatorKind::Jump(Target {
            block: target.block,
            args,
        }),
    );
    let reachable: HashSet<_> = Cfg::new(&result).rpo.into_iter().collect();
    for &block in &reachable {
        let data = &result.blocks[block];
        if matches!(
            data.terminator.as_ref().unwrap().kind,
            TerminatorKind::Yield(..) | TerminatorKind::Fallback(..)
        ) {
            return Err("continuation requires normal control flow".into());
        }
        for &inst in &data.insts {
            match result.insts[inst].op {
                Op::Read(_)
                | Op::Write(..)
                | Op::Initialize(..)
                | Op::Closure(..)
                | Op::Argument(_)
                | Op::ClosureThis(_)
                | Op::EnterCaptures(..)
                | Op::Arguments
                | Op::Rest(_)
                | Op::OsrStart(..)
                    if block != entry =>
                {
                    return Err("continuation requires SSA frame state".into());
                }
                Op::EnterScope(scope)
                | Op::LeaveScope(scope)
                | Op::NextIteration(scope)
                | Op::FreshIteration(scope)
                    if result.materialized(scope) =>
                {
                    return Err("continuation requires materialized scope state".into());
                }
                Op::View(_) => return Err("continuation requires a heap view".into()),
                _ => {}
            }
            if result.views.contains_key(&inst) {
                return Err("continuation requires a heap view".into());
            }
        }
    }
    for &block in &reachable {
        result.blocks[block].insts.retain(|&inst| {
            !matches!(
                result.insts[inst].op,
                Op::EnterScope(_)
                    | Op::LeaveScope(_)
                    | Op::NextIteration(_)
                    | Op::FreshIteration(_)
                    | Op::Assume(..)
            )
        });
    }
    result.has_assumes = false;
    result.views.clear();
    let definitions = aliases
        .into_iter()
        .map(|(value, alias)| (value, vec![alias]))
        .collect();
    super::optimize::repair(&mut result, &definitions);
    let reachable: HashSet<_> = Cfg::new(&result).rpo.into_iter().collect();
    for (block, data) in result.blocks.iter_mut() {
        if !reachable.contains(&block) {
            data.insts.clear();
            data.params.clear();
            data.terminator = None;
        }
    }
    result.cold.retain(|block| reachable.contains(block));
    result.verify()?;
    Ok((result, inputs))
}

#[derive(Clone)]
pub(super) struct Piece {
    pub(super) body: Arc<Body>,
    pub(super) inputs: Vec<Value>,
}

pub(super) struct Exit {
    pub(super) block: Block,
    pub(super) outputs: Vec<(Value, Types)>,
    pub(super) params: Vec<Types>,
}

impl Exit {
    fn words(&self) -> usize {
        self.outputs.len() + self.params.len()
    }
}

pub(super) struct Shape {
    pub(super) body: HashSet<Block>,
    pub(super) exits: Vec<Exit>,
    pub(super) returns: Option<Types>,
}

impl Shape {
    pub(super) fn new(
        graph: &Graph,
        facts: &Facts,
        live: &SecondaryMap<Block, HashSet<Value>>,
        body: HashSet<Block>,
    ) -> Self {
        let mut targets = BTreeSet::new();
        let mut returns: Option<Types> = None;
        for &block in &body {
            let kind = &graph.blocks[block].terminator.as_ref().unwrap().kind;
            if let TerminatorKind::Return(value) = kind {
                returns = Some(returns.unwrap_or_default().union(facts.of(*value)));
            }
            targets.extend(
                kind.targets()
                    .into_iter()
                    .map(|target| target.block)
                    .filter(|block| !body.contains(block)),
            );
        }
        let exits = targets
            .into_iter()
            .map(|block| {
                let mut outputs: Vec<_> = live[block]
                    .iter()
                    .copied()
                    .filter(|&value| body.contains(&graph.def_block(value)))
                    .collect();
                outputs.sort();
                Exit {
                    block,
                    outputs: outputs
                        .into_iter()
                        .map(|value| (value, facts.of(value)))
                        .collect(),
                    params: graph.blocks[block]
                        .params
                        .iter()
                        .map(|&param| facts.of(param))
                        .collect(),
                }
            })
            .collect();
        Self {
            body,
            exits,
            returns,
        }
    }

    pub(super) fn count(&self) -> usize {
        self.exits.len() + usize::from(self.returns.is_some())
    }

    pub(super) fn dispatch(
        &self,
        result: &mut Graph,
        block: Block,
        call: Value,
        definitions: &mut HashMap<Value, Vec<Value>>,
    ) {
        let count = self.count();
        let mut current = block;
        for index in 0..count {
            let case = result.blocks.push(BlockData::default());
            let kind = match self.exits.get(index) {
                Some(exit) => {
                    for (offset, &(value, types)) in exit.outputs.iter().enumerate() {
                        let output = append(result, case, Op::ContinuationOutput(call, offset));
                        let output = unbox(result, case, output, types);
                        definitions.entry(value).or_default().push(output);
                    }
                    let args = exit
                        .params
                        .iter()
                        .enumerate()
                        .map(|(offset, &types)| {
                            let offset = exit.outputs.len() + offset;
                            let output = append(result, case, Op::ContinuationOutput(call, offset));
                            unbox(result, case, output, types)
                        })
                        .collect();
                    TerminatorKind::Jump(Target {
                        block: exit.block,
                        args,
                    })
                }
                None => {
                    let output = append(result, case, Op::ContinuationOutput(call, 0));
                    let output = unbox(result, case, output, self.returns.unwrap());
                    TerminatorKind::Return(output)
                }
            };
            terminate(result, case, kind);
            if index + 1 == count {
                terminate(result, current, TerminatorKind::Jump(Target::new(case)));
                break;
            }
            let constant = append(
                result,
                current,
                Op::Constant(Constant::Number(index as f64)),
            );
            let test = append(
                result,
                current,
                Op::Binary(BinaryOperator::StrictEquality, call, constant),
            );
            let next = result.blocks.push(BlockData::default());
            terminate(
                result,
                current,
                TerminatorKind::Branch(test, Target::new(case), Target::new(next)),
            );
            current = next;
        }
    }

    pub(super) fn store_exits(&self, graph: &Graph, result: &mut Graph, entry: Block) -> usize {
        let outside = |block: Block| graph.blocks.is_valid(block) && !self.body.contains(&block);
        let index = |constant: usize| Op::Constant(Constant::Number(constant as f64));
        for block in Cfg::from_entry(result, entry).rpo {
            if outside(block) {
                continue;
            }
            let kind = &result.blocks[block].terminator.as_ref().unwrap().kind;
            if let TerminatorKind::Return(value) = *kind {
                result.append(block, graph.span, Op::ContinuationStore(vec![value]), false);
                let exit = append(result, block, index(self.exits.len()));
                terminate(result, block, TerminatorKind::Return(exit));
                continue;
            }
            let edges: Vec<_> = kind
                .targets()
                .into_iter()
                .enumerate()
                .filter(|(_, target)| outside(target.block))
                .map(|(edge, target)| (edge, target.clone()))
                .collect();
            for (edge, target) in edges {
                let position = self
                    .exits
                    .iter()
                    .position(|exit| exit.block == target.block)
                    .unwrap();
                let exit = &self.exits[position];
                let values = exit
                    .outputs
                    .iter()
                    .map(|&(value, _)| value)
                    .chain(target.args)
                    .collect();
                let stub = result.blocks.push(BlockData::default());
                result.append(stub, graph.span, Op::ContinuationStore(values), false);
                let value = append(result, stub, index(position));
                terminate(result, stub, TerminatorKind::Return(value));
                *result.blocks[block]
                    .terminator
                    .as_mut()
                    .unwrap()
                    .kind
                    .targets_mut()[edge] = Target::new(stub);
            }
        }
        self.words()
    }

    pub(super) fn words(&self) -> usize {
        self.exits
            .iter()
            .map(Exit::words)
            .chain(self.returns.map(|_| 1))
            .max()
            .unwrap_or(0)
    }
}

#[derive(Default)]
pub(super) struct Pieces {
    shapes: Option<HashMap<Block, Shape>>,
    facts: Option<Facts>,
    loops: HashMap<Block, super::Result<Piece>>,
    drivers: HashMap<Block, super::Result<Piece>>,
    resumes: HashMap<(Block, Vec<Value>), super::Result<Piece>>,
}

impl Pieces {
    pub(super) fn resume(&mut self, graph: &Graph, target: &Target) -> super::Result<Piece> {
        let key = (target.block, target.args.clone());
        if let Some(piece) = self.resumes.get(&key) {
            return piece.clone();
        }
        let name = format!("{}.resume.{}", graph.name, target.block);
        let piece = self.build(graph, target.clone(), None, None, name);
        self.resumes.insert(key, piece.clone());
        piece
    }

    pub(super) fn driver(&mut self, graph: &Graph, header: Block) -> super::Result<Piece> {
        if let Some(piece) = self.drivers.get(&header) {
            return piece.clone();
        }
        self.drivers
            .insert(header, Err("cyclic loop continuation".into()));
        let entry = Target {
            block: header,
            args: graph.blocks[header].params.clone(),
        };
        let name = format!("{}.next.{header}", graph.name);
        let piece = self.build(graph, entry, None, Some(header), name);
        self.drivers.insert(header, piece.clone());
        piece
    }

    pub(super) fn headers(&mut self, graph: &Graph) -> Vec<Block> {
        let shapes = self.shapes(graph);
        let mut headers: Vec<_> = shapes.keys().copied().collect();
        headers.sort_by_key(|header| (std::cmp::Reverse(shapes[header].body.len()), *header));
        headers
    }

    fn body(&mut self, graph: &Graph, header: Block) -> super::Result<Piece> {
        if let Some(piece) = self.loops.get(&header) {
            return piece.clone();
        }
        self.loops
            .insert(header, Err("cyclic loop continuation".into()));
        let entry = Target {
            block: header,
            args: graph.blocks[header].params.clone(),
        };
        let name = format!("{}.loop.{header}", graph.name);
        let piece = self.build(graph, entry, Some(header), None, name);
        self.loops.insert(header, piece.clone());
        piece
    }

    fn shapes(&mut self, graph: &Graph) -> &HashMap<Block, Shape> {
        let facts = self.facts.get_or_insert_with(|| Facts::compute(graph));
        self.shapes.get_or_insert_with(|| {
            let cfg = Cfg::new(graph);
            let live = liveness(graph, &cfg.rpo);
            super::optimize::natural_loops(graph, &cfg)
                .into_iter()
                .map(|(header, body)| (header, Shape::new(graph, facts, &live, body)))
                .collect()
        })
    }

    fn build(
        &mut self,
        graph: &Graph,
        entry: Target,
        own: Option<Block>,
        called: Option<Block>,
        name: String,
    ) -> super::Result<Piece> {
        if graph.generator || graph.has_handlers {
            return Err("continuation requires a normal function without handlers".into());
        }
        if graph
            .blocks
            .values()
            .flat_map(|block| &block.insts)
            .any(|&inst| matches!(graph.insts[inst].op, Op::ContinuationStore(_)))
        {
            return Err("continuation requires its own exit buffer".into());
        }
        self.shapes(graph);
        let mut entry = entry;
        let mut result = graph.clone();
        let mut definitions = HashMap::new();
        let mut seen = HashSet::new();
        let mut work = vec![entry.block];
        while let Some(block) = work.pop() {
            if !seen.insert(block) {
                continue;
            }
            let shapes = self.shapes.as_ref().unwrap();
            if own.is_some_and(|header| !shapes[&header].body.contains(&block)) {
                continue;
            }
            if !shapes.contains_key(&block) || own == Some(block) {
                work.extend(result.successors(block));
            } else if own.is_some() || called == Some(block) {
                let (call, exits) = self.call(graph, &mut result, block, &mut definitions)?;
                if entry.block == block {
                    entry.block = call;
                }
                work.extend(exits);
            } else {
                self.tail(graph, &mut result, block)?;
            }
        }
        let outputs = match own {
            Some(header) => self.exits(graph, &mut result, header, entry.block),
            None => 0,
        };
        let start = result.blocks.push(BlockData::default());
        terminate(&mut result, start, TerminatorKind::Jump(entry.clone()));
        result.entry = start;
        super::optimize::repair(&mut result, &definitions);
        let (mut extracted, inputs) = extract(&result, entry, self.facts.as_ref().unwrap())?;
        extracted.name = name;
        Ok(Piece {
            body: Arc::new(Body::new(extracted, outputs)),
            inputs,
        })
    }

    fn call(
        &mut self,
        graph: &Graph,
        result: &mut Graph,
        header: Block,
        definitions: &mut HashMap<Value, Vec<Value>>,
    ) -> super::Result<(Block, Vec<Block>)> {
        let piece = self.body(graph, header)?;
        let shape = &self.shapes.as_ref().unwrap()[&header];
        let block = result.blocks.push(BlockData::default());
        let renamed: HashMap<_, _> = graph.blocks[header]
            .params
            .iter()
            .map(|&param| (param, result.add_param(block)))
            .collect();
        for data in result.blocks.values_mut() {
            let Some(terminator) = &mut data.terminator else {
                continue;
            };
            for target in terminator.kind.targets_mut() {
                if target.block == header {
                    target.block = block;
                }
            }
        }
        let inputs = piece
            .inputs
            .iter()
            .map(|value| renamed.get(value).copied().unwrap_or(*value))
            .collect();
        let call = append(result, block, Op::Continuation(piece.body, inputs));
        if shape.count() == 0 {
            terminate(result, block, TerminatorKind::Return(call));
            return Ok((block, Vec::new()));
        }
        shape.dispatch(result, block, call, definitions);
        Ok((block, shape.exits.iter().map(|exit| exit.block).collect()))
    }

    fn tail(&mut self, graph: &Graph, result: &mut Graph, header: Block) -> super::Result<()> {
        let piece = self.driver(graph, header)?;
        result.blocks[header].insts.clear();
        let value = append(result, header, Op::Continuation(piece.body, piece.inputs));
        terminate(result, header, TerminatorKind::Return(value));
        Ok(())
    }

    fn exits(&self, graph: &Graph, result: &mut Graph, header: Block, entry: Block) -> usize {
        self.shapes.as_ref().unwrap()[&header].store_exits(graph, result, entry)
    }
}

#[cfg(test)]
pub(super) fn bodies(graph: &Graph) -> Vec<Arc<Body>> {
    fn direct(graph: &Graph) -> Vec<Arc<Body>> {
        Cfg::new(graph)
            .rpo
            .iter()
            .flat_map(|&block| &graph.blocks[block].insts)
            .filter_map(|&inst| match &graph.insts[inst].op {
                Op::Continuation(body, _) => Some(body.clone()),
                _ => None,
            })
            .collect()
    }
    let mut found = direct(graph);
    let mut index = 0;
    while index < found.len() {
        for body in direct(&found[index].graph) {
            if !found.iter().any(|other| Arc::ptr_eq(other, &body)) {
                found.push(body);
            }
        }
        index += 1;
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxc_allocator::Allocator;
    use oxc_ast::ast::Statement;
    use oxc_parser::Parser;
    use oxc_span::SourceType;

    #[test]
    fn resume_inside_loop_preserves_live_values() {
        let allocator = Allocator::default();
        let parsed = Parser::new(
            &allocator,
            "function f(n,x){let s=x;for(let i=0;i<n;++i){if(i>2)s+=this.x;s+=i;}return s+this.y;}",
            SourceType::script(),
        )
        .parse();
        let Statement::FunctionDeclaration(function) = &parsed.program.body[0] else {
            panic!()
        };
        let semantic = oxc_semantic::SemanticBuilder::new_compiler()
            .build(&parsed.program)
            .semantic;
        let graph =
            super::super::build::lower(function, semantic.scoping(), &HashSet::new()).unwrap();
        let graph = super::super::canonicalize_graph(graph, true).unwrap().graph;
        let cfg = Cfg::new(&graph);
        let (header, body) = super::super::optimize::natural_loops(&graph, &cfg).remove(0);
        let (pred, target) = body
            .iter()
            .find_map(|&block| {
                if block == header {
                    return None;
                }
                graph.blocks[block]
                    .terminator
                    .as_ref()
                    .unwrap()
                    .kind
                    .targets()
                    .into_iter()
                    .find(|target| body.contains(&target.block) && target.block != header)
                    .map(|target| (block, target.clone()))
            })
            .unwrap();
        let uses = |body: &Body, matches: &dyn Fn(&Op) -> bool| {
            Cfg::new(&body.graph).rpo.iter().any(|&block| {
                body.graph.blocks[block]
                    .insts
                    .iter()
                    .any(|&inst| matches(&body.graph.insts[inst].op))
            })
        };
        let mut pieces = Pieces::default();
        let driver = pieces.driver(&graph, header).unwrap();
        driver.body.graph.verify().unwrap();
        assert!(uses(
            &driver.body,
            &|op| matches!(op, Op::Continuation(body, _)
            if body.outputs > 0 && uses(body, &|op| matches!(op, Op::This)))
        ));
        let resume = pieces.resume(&graph, &target).unwrap();
        resume.body.graph.verify().unwrap();
        assert!(!resume.inputs.is_empty());
        assert!(
            resume
                .inputs
                .iter()
                .all(|&value| cfg.dominates(graph.def_block(value), pred))
        );
        assert!(uses(
            &resume.body,
            &|op| matches!(op, Op::Continuation(body, _)
            if Arc::ptr_eq(body, &driver.body))
        ));
    }
}

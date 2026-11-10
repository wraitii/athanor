use super::super::guards::{Fact, Origin};
use super::*;

#[derive(Clone)]
pub(super) struct Placed {
    pub(super) inst: Inst,
    pub(super) value: Value,
    pub(super) fact: Fact,
    pub(super) origin: Origin,
}

pub(super) struct Point {
    pub(super) block: Block,
    pub(super) header: Option<Block>,
    pub(super) body: HashSet<Block>,
    pub(super) assumes: Vec<Placed>,
}

pub(super) struct Reach {
    pub(super) value: Value,
    pub(super) fact: Fact,
    pub(super) use_block: Block,
    pub(super) header: Option<Block>,
    pub(super) global: Option<String>,
    pub(super) origin: Origin,
}

fn is_load(op: &Op) -> bool {
    matches!(
        op,
        Op::Get(..)
            | Op::ShapedGet(..)
            | Op::GetIndex(..)
            | Op::IndexedGet(..)
            | Op::GlobalRead(_)
            | Op::ProbeGlobal(_)
            | Op::ProbeProperty(..)
            | Op::ProbeShapeProperty(..)
            | Op::ProbeIndex(..)
    )
}

fn as_read(op: &Op) -> Op {
    match op {
        Op::ProbeGlobal(name) => Op::GlobalRead(name.clone()),
        Op::ProbeProperty(object, name) => Op::Get(*object, name.clone()),
        Op::ProbeShapeProperty(object, name, ..) => Op::Get(*object, name.clone()),
        Op::ProbeIndex(object, key) => Op::GetIndex(*object, *key),
        op => op.clone(),
    }
}

struct Hoister<'a> {
    graph: &'a Graph,
    cfg: &'a Cfg,
    assumed: &'a HashMap<Value, cranelift_module::FuncId>,
    writes: HashMap<usize, Writes>,
    memo: HashMap<(usize, Value), Result<()>>,
}

impl Hoister<'_> {
    fn hoistable(&mut self, points: &[Point], point: usize, value: Value) -> Result<()> {
        if let Some(known) = self.memo.get(&(point, value)) {
            return known.clone();
        }
        let result = self.compute(points, point, value);
        self.memo.insert((point, value), result.clone());
        result
    }

    fn compute(&mut self, points: &[Point], point: usize, value: Value) -> Result<()> {
        let target = &points[point];
        let definer = self.graph.def_block(value);
        if !target.body.contains(&definer) {
            return if self.cfg.dominates(definer, target.block) {
                Ok(())
            } else {
                Err(format!(
                    "{value} is defined where the point is not dominated"
                ))
            };
        }
        let ValueDef::Result(inst) = self.graph.values[value] else {
            return Err(format!("{value} is a block parameter inside the loop"));
        };
        let op = self.graph.insts[inst].op.clone();
        let input = matches!(op, Op::Argument(_) | Op::This);
        if !input && !is_load(&op) {
            return Err(format!(
                "{value} is not a load ({})",
                self.graph.op_text(&op)
            ));
        }
        for operand in op.operands_iter() {
            self.hoistable(points, point, operand)?;
        }
        if input {
            return Ok(());
        }
        let graph = self.graph;
        let assumed = self.assumed;
        let writes = self.writes.entry(point).or_insert_with(|| {
            let written: HashSet<Block> = target
                .body
                .iter()
                .copied()
                .filter(|block| !graph.cold.contains(block))
                .collect();
            Writes::collect_assuming(graph, &written, assumed)
        });
        let read = as_read(&op);
        let Some(access) = Access::read(&read, &writes.facts) else {
            return Err(format!("{value} has no access"));
        };
        match writes.killer(&access) {
            None => Ok(()),
            Some(killer) => Err(format!(
                "{value} is clobbered by {killer} ({})",
                self.graph.op_text(&self.graph.insts[killer].op)
            )),
        }
    }

    fn movable(&self, points: &[Point], point: usize, value: Value) -> bool {
        let target = &points[point];
        let definer = self.graph.def_block(value);
        if target.header.is_some() {
            return !target.body.contains(&definer);
        }
        definer == target.block
    }
}

pub(super) fn representative(
    graph: &Graph,
    reps: &mut HashMap<String, Value>,
    value: Value,
    depth: usize,
) -> Value {
    let ValueDef::Result(inst) = graph.values[value] else {
        return value;
    };
    let op = &graph.insts[inst].op;
    if depth > 8 || !(is_load(op) || matches!(op, Op::Argument(_) | Op::This)) {
        return value;
    }
    let mut op = op.clone();
    for operand in op.operands_mut() {
        *operand = representative(graph, reps, *operand, depth + 1);
    }
    *reps.entry(graph.op_text(&op)).or_insert(value)
}

fn merge(graph: &mut Graph, point: &mut Point) {
    let mut reps = HashMap::new();
    let mut kept: Vec<(Value, Fact, Placed)> = Vec::new();
    for placed in std::mem::take(&mut point.assumes) {
        let value = representative(graph, &mut reps, placed.value, 0);
        let fact = match placed.fact {
            Fact::Bound(bound) => Fact::Bound(representative(graph, &mut reps, bound, 0)),
            fact => fact,
        };
        if let Some(existing) = kept.iter_mut().find(|(other, known, _)| {
            *other == value && (known.implies(fact) || fact.implies(*known))
        }) {
            let gain = existing.2.origin.gain.saturating_add(placed.origin.gain);
            if fact.implies(existing.1) && fact != existing.1 {
                graph.detach_inst(existing.2.inst);
                *existing = (value, fact, placed);
            } else {
                graph.detach_inst(placed.inst);
            }
            existing.2.origin.gain = gain;
            if let Op::Assume(.., origin) = &mut graph.insts[existing.2.inst].op {
                origin.gain = gain;
            }
            continue;
        }
        kept.push((value, fact, placed));
    }
    point.assumes = kept.into_iter().map(|(_, _, placed)| placed).collect();
}

pub(super) fn place_assumes(
    graph: &mut Graph,
    assumed: &HashMap<Value, cranelift_module::FuncId>,
) -> Vec<Reach> {
    if !graph.has_assumes {
        return Vec::new();
    }
    let cfg = Cfg::new(graph);
    let mut points: Vec<Point> = natural_loops(graph, &cfg)
        .into_iter()
        .filter(|(header, _)| !graph.cold.contains(header))
        .filter_map(|(header, body)| {
            let block = preheader(graph, &cfg, header, &body)?;
            Some(Point {
                block,
                header: Some(header),
                body,
                assumes: Vec::new(),
            })
        })
        .collect();
    points.sort_by_key(|point| point.body.len());
    points.push(Point {
        block: graph.entry,
        header: None,
        body: cfg.rpo.iter().copied().collect(),
        assumes: Vec::new(),
    });
    let mut planning = graph.clone();
    for &block in &cfg.rpo {
        for &inst in &graph.blocks[block].insts {
            let Op::Assume(value, Fact::Native(native), _) = graph.insts[inst].op else {
                continue;
            };
            for call in graph.blocks.values().flat_map(|data| data.insts.iter()) {
                if let Op::Call { callee, args, .. } = &graph.insts[*call].op
                    && *callee == value
                {
                    planning.insts[*call].op = Op::CallNative(native, args.clone());
                }
            }
        }
    }
    let mut hoister = Hoister {
        graph: &planning,
        cfg: &cfg,
        assumed,
        writes: HashMap::new(),
        memo: HashMap::new(),
    };
    let mut moves = Vec::new();
    let mut reached = Vec::new();
    let mut placed: Vec<Vec<Placed>> = points.iter().map(|_| Vec::new()).collect();
    for &block in &cfg.rpo {
        for &inst in &graph.blocks[block].insts {
            let Op::Assume(value, fact, origin) = graph.insts[inst].op else {
                continue;
            };
            let operands: Vec<Value> = match fact {
                Fact::Bound(bound) => vec![value, bound],
                _ => vec![value],
            };
            let mut best = None;
            for index in 0..points.len() {
                if !points[index].body.contains(&block) {
                    continue;
                }
                let verdict = operands
                    .iter()
                    .try_for_each(|&operand| hoister.hoistable(&points, index, operand));
                if verdict.is_err() {
                    break;
                }
                best = Some(index);
            }
            let Some(index) = best else {
                continue;
            };
            if operands
                .iter()
                .all(|&operand| hoister.movable(&points, index, operand))
                && points[index].block != block
            {
                moves.push((inst, points[index].block));
            }
            let global = match (fact, graph.defining_op(value)) {
                (Fact::Code(_), Some(Op::ProbeGlobal(name) | Op::GlobalRead(name))) => {
                    Some(name.clone())
                }
                _ => None,
            };
            reached.push(Reach {
                value,
                fact,
                use_block: block,
                header: points[index].header,
                global,
                origin,
            });
            placed[index].push(Placed {
                inst,
                value,
                fact,
                origin,
            });
        }
    }
    for (inst, target) in moves {
        graph.detach_inst(inst);
        graph.blocks[target].insts.push(inst);
        graph.insts[inst].block = target;
    }
    for (point, assumes) in points.iter_mut().zip(placed) {
        point.assumes = assumes;
        merge(graph, point);
    }
    reached
}

#[cfg(test)]
mod tests {
    use super::super::super::guards::Kind;
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
            super::super::super::build::lower(function, semantic.scoping(), &HashSet::new())
                .unwrap();
        super::super::super::canonicalize(graph, true)
            .unwrap()
            .graph
    }

    fn origin() -> Origin {
        Origin {
            site: None,
            kind: Kind::Number,
            gain: 1,
        }
    }

    fn assume(graph: &mut Graph, value: Value, fact: Fact) -> Inst {
        let block = graph.def_block(value);
        graph
            .append(block, graph.span, Op::Assume(value, fact, origin()), false)
            .0
    }

    fn find(graph: &Graph, predicate: impl Fn(&Op) -> bool) -> Value {
        Cfg::new(graph)
            .rpo
            .iter()
            .flat_map(|&block| graph.blocks[block].insts.iter())
            .find_map(|&inst| {
                predicate(&graph.insts[inst].op)
                    .then_some(graph.insts[inst].result)
                    .flatten()
            })
            .unwrap()
    }

    fn number() -> Fact {
        Fact::Type(Types::NUMBER)
    }

    #[test]
    fn hoists_to_the_innermost_legal_preheader() {
        let mut graph = graph(
            "function f(rows,n){let s=0;for(let y=0;y<n;++y){const r=rows[y];for(let x=0;x<n;++x)s+=r.v;}return s;}",
        );
        let load = find(&graph, |op| matches!(op, Op::Get(_, name) if name == "v"));
        assume(&mut graph, load, number());
        let reached = place_assumes(&mut graph, &HashMap::new());
        let cfg = Cfg::new(&graph);
        let inner = natural_loops(&graph, &cfg)
            .into_iter()
            .min_by_key(|(_, body)| body.len())
            .unwrap();
        assert_eq!(reached.len(), 1);
        assert_eq!(reached[0].header, Some(inner.0));
        graph.verify().unwrap();
    }

    #[test]
    fn invariant_values_reach_the_function_entry() {
        let mut graph = graph("function f(a,n){let s=0;for(let i=0;i<n;++i)s+=a.x;return s;}");
        let limit = find(&graph, |op| matches!(op, Op::Argument(1)));
        let load = find(&graph, |op| matches!(op, Op::Get(_, name) if name == "x"));
        let moved = assume(&mut graph, limit, number());
        assume(&mut graph, load, number());
        let reached = place_assumes(&mut graph, &HashMap::new());
        assert_eq!(reached.len(), 2);
        assert!(reached.iter().all(|reach| reach.header.is_none()));
        assert_eq!(graph.insts[moved].block, graph.entry);
        graph.verify().unwrap();
    }

    #[test]
    fn conflicting_writes_stop_hoisting() {
        let mut graph =
            graph("function f(o,n){let s=0;for(let i=0;i<n;++i){o.x=i;s+=o.x;}return s;}");
        let load = find(&graph, |op| matches!(op, Op::Get(_, name) if name == "x"));
        assume(&mut graph, load, number());
        assert!(place_assumes(&mut graph, &HashMap::new()).is_empty());

        let mut graph =
            self::graph("function f(o,n){let s=0;for(let i=0;i<n;++i){o.z=i;s+=o.x;}return s;}");
        let load = find(&graph, |op| matches!(op, Op::Get(_, name) if name == "x"));
        assume(&mut graph, load, number());
        assert_eq!(place_assumes(&mut graph, &HashMap::new()).len(), 1);
    }

    #[test]
    fn merging_keeps_the_stronger_fact_and_sums_gain() {
        let mut graph = graph("function f(a,n){let s=0;for(let i=0;i<n;++i)s+=a;return s;}");
        let argument = find(&graph, |op| matches!(op, Op::Argument(0)));
        let weak = assume(&mut graph, argument, number());
        let strong = assume(&mut graph, argument, Fact::Type(Types::INTEGER));
        let again = assume(&mut graph, argument, number());
        place_assumes(&mut graph, &HashMap::new());
        let Op::Assume(_, _, origin) = graph.insts[strong].op else {
            panic!()
        };
        assert_eq!(origin.gain, 3);
        let entry = &graph.blocks[graph.entry].insts;
        assert!(entry.contains(&strong));
        assert!(!entry.contains(&weak) && !entry.contains(&again));
        graph.verify().unwrap();
    }

    #[test]
    fn assumes_do_not_count_as_uses_and_follow_their_definition() {
        let mut graph = graph("function f(a){return a.x;}");
        let load = find(&graph, |op| matches!(op, Op::Get(_, name) if name == "x"));
        let inst = assume(&mut graph, load, number());
        assert_eq!(graph.use_counts()[load], 1);
        let ValueDef::Result(def) = graph.values[load] else {
            panic!()
        };
        graph.remove_inst(def);
        assert!(
            !graph
                .blocks
                .values()
                .any(|block| block.insts.contains(&inst))
        );
    }
}

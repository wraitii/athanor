use std::collections::{HashMap, HashSet};

use cranelift_module::FuncId;
use oxc_syntax::operator::BinaryOperator;

use super::{Canonical, Cfg, Constant, Facts, Graph, Inst, Op, Result, TerminatorKind, Types};

#[derive(Default)]
pub(in crate::jit) struct Repository {
    bodies: HashMap<FuncId, Body>,
    pub(in crate::jit) feedback: HashMap<crate::jit::planner::Site, FuncId>,
    pub(in crate::jit) read_feedback: HashMap<
        (crate::jit::planner::Site, crate::jit::planner::ReadKind),
        crate::jit::planner::ReadType,
    >,
    locations: HashMap<(usize, u32), FuncId>,
    globals: HashMap<String, FuncId>,
    pub(in crate::jit) sources: HashMap<usize, usize>,
}

struct Body {
    source: usize,
    canonical: Canonical,
    arrow: bool,
}

impl Repository {
    pub(in crate::jit) fn insert(
        &mut self,
        source: usize,
        name: &str,
        code: FuncId,
        canonical: Canonical,
    ) {
        self.locations
            .insert((source, canonical.graph.span.start), code);
        self.globals.insert(name.to_owned(), code);
        self.bodies.insert(
            code,
            Body {
                source,
                canonical,
                arrow: false,
            },
        );
    }

    pub(in crate::jit) fn insert_closure(
        &mut self,
        source: usize,
        code: FuncId,
        canonical: Canonical,
        arrow: bool,
    ) {
        self.locations
            .insert((source, canonical.graph.span.start), code);
        self.bodies.insert(
            code,
            Body {
                source,
                canonical,
                arrow,
            },
        );
    }

    pub(in crate::jit) fn code(&self, source: usize, start: u32) -> Option<FuncId> {
        self.locations.get(&(source, start)).copied()
    }

    pub(in crate::jit) fn emission(&self, code: FuncId) -> Option<(usize, Canonical)> {
        let body = self.bodies.get(&code)?;
        Some((body.source, body.canonical.clone()))
    }

    pub(super) fn closure(&self, source: usize, start: u32) -> Option<FuncId> {
        self.locations.get(&(source, start)).copied()
    }

    pub(in crate::jit) fn canonical(&self, source: usize, start: u32) -> Option<&Canonical> {
        let code = self.locations.get(&(source, start))?;
        Some(&self.bodies.get(code)?.canonical)
    }

    pub(super) fn body(&self, code: FuncId) -> Option<&Graph> {
        let body = self.bodies.get(&code)?;
        Some(&body.canonical.graph)
    }
}

pub(super) struct Request {
    pub(super) call: Inst,
    pub(super) target: FuncId,
    pub(super) numeric_arguments: Vec<usize>,
}

pub(super) struct Plan {
    inlines: Vec<Inline>,
}

pub(super) struct Budget {
    remaining: usize,
    sites: HashSet<Inst>,
    pub(super) inlines: Vec<Inline>,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            remaining: 512,
            sites: HashSet::new(),
            inlines: Vec::new(),
        }
    }
}

impl Plan {
    pub(super) fn is_empty(&self) -> bool {
        self.inlines.is_empty()
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(in crate::jit) struct Inline {
    pub(super) call: Inst,
    pub(super) target: FuncId,
    pub(super) numbers: Vec<usize>,
    pub(super) arrow: bool,
    pub(super) guard_code: bool,
    pub(super) source_name: Option<(usize, usize)>,
}

fn numeric_body(graph: &Graph) -> bool {
    if graph.generator || graph.has_handlers {
        return false;
    }
    let cfg = Cfg::new(graph);
    let mut count = 0;
    for block in cfg.rpo {
        if !matches!(
            graph.blocks[block].terminator.as_ref().map(|t| &t.kind),
            Some(TerminatorKind::Jump(_) | TerminatorKind::Branch(..) | TerminatorKind::Return(_))
        ) {
            return false;
        }
        for &inst in &graph.blocks[block].insts {
            count += 1;
            if count > 128 {
                return false;
            }
            match graph.insts[inst].op {
                Op::Argument(_)
                | Op::Constant(
                    Constant::Number(_)
                    | Constant::Boolean(_)
                    | Constant::Null
                    | Constant::Undefined,
                )
                | Op::Unary(_, _)
                | Op::ToNumeric(_)
                | Op::IsUndefined(_)
                | Op::IsNullish(_) => {}
                Op::Binary(operator, ..)
                    if !matches!(operator, BinaryOperator::In | BinaryOperator::Instanceof) => {}
                _ => return false,
            }
        }
    }
    true
}

pub(super) fn eligible(graph: &Graph) -> bool {
    if graph.generator || graph.has_handlers {
        return false;
    }
    let mut count = 0;
    for block in Cfg::new(graph).rpo {
        if !matches!(
            graph.blocks[block].terminator.as_ref().map(|t| &t.kind),
            Some(TerminatorKind::Jump(_) | TerminatorKind::Branch(..) | TerminatorKind::Return(_))
        ) {
            return false;
        }
        for &inst in &graph.blocks[block].insts {
            count += 1;
            if count > 128
                || matches!(
                    graph.insts[inst].op,
                    Op::Closure(..) | Op::Arguments | Op::Rest(_) | Op::Catch
                )
            {
                return false;
            }
        }
    }
    true
}

pub(super) fn requests(graph: &Graph, bodies: &Repository, source: usize) -> Vec<Request> {
    let mut requests = Vec::new();
    for block in Cfg::new(graph).rpo {
        for &call in &graph.blocks[block].insts {
            let (callee, constructor) = match &graph.insts[call].op {
                Op::Call { callee, .. } => (callee, false),
                Op::ConstructorCall { callee, .. } => (callee, true),
                _ => continue,
            };
            let data = &graph.insts[call];
            let site = crate::jit::planner::Site {
                source: data.source_name.map_or(source, |name| name.0),
                start: data.span.start,
                end: data.span.end,
            };
            let (target, numeric) = if let Some(&target) = bodies.feedback.get(&site) {
                (target, false)
            } else {
                match graph.defining_op(*callee) {
                    Some(Op::GlobalRead(name)) => {
                        let Some(&target) = bodies.globals.get(name) else {
                            continue;
                        };
                        let Some(body) = bodies.body(target) else {
                            continue;
                        };
                        (target, !constructor && numeric_body(body))
                    }
                    Some(Op::Closure(span, _, false, _)) => {
                        let Some(target) = bodies.closure(source, span.start) else {
                            continue;
                        };
                        (target, false)
                    }
                    _ => continue,
                }
            };
            let Some(body) = bodies.body(target).filter(|body| eligible(body)) else {
                continue;
            };
            let mut numeric_arguments: Vec<usize> = Cfg::new(body)
                .rpo
                .iter()
                .flat_map(|&block| &body.blocks[block].insts)
                .filter_map(|&inst| match body.insts[inst].op {
                    Op::Argument(index) if numeric => Some(index),
                    _ => None,
                })
                .collect();
            numeric_arguments.sort_unstable();
            numeric_arguments.dedup();
            requests.push(Request {
                call,
                target,
                numeric_arguments,
            });
        }
    }
    requests
}

#[cfg(test)]
pub(super) fn plan(
    graph: &Graph,
    bodies: &Repository,
    source: usize,
    requests: &[Request],
) -> Plan {
    plan_round(graph, bodies, source, requests, &mut Budget::default())
}

pub(super) fn plan_round(
    graph: &Graph,
    bodies: &Repository,
    source: usize,
    requests: &[Request],
    budget: &mut Budget,
) -> Plan {
    let facts = Facts::compute(graph);
    let mut inlines = Vec::new();
    for request in requests {
        let Some(body) = bodies.body(request.target).filter(|body| eligible(body)) else {
            continue;
        };
        let (callee, args) = match &graph.insts[request.call].op {
            Op::Call { callee, args, .. } | Op::ConstructorCall { callee, args, .. } => {
                (callee, args)
            }
            _ => continue,
        };
        let cost: usize = Cfg::new(body)
            .rpo
            .iter()
            .map(|&block| body.blocks[block].insts.len())
            .sum();
        if cost > budget.remaining || budget.sites.contains(&request.call) {
            continue;
        }
        let mut numbers = Vec::new();
        let mut possible = true;
        for &index in &request.numeric_arguments {
            let Some(&value) = args.get(index) else {
                possible = false;
                break;
            };
            let types = facts.of(value);
            if !types.intersects(Types::NUMBER) {
                possible = false;
                break;
            }
            if !types.within(Types::NUMBER) && !numbers.iter().any(|&index| args[index] == value) {
                numbers.push(index);
            }
        }
        if !possible {
            continue;
        }
        budget.sites.insert(request.call);
        budget.remaining -= cost;
        let body_source = bodies.bodies[&request.target].source;
        let source_name = if body_source == source {
            None
        } else {
            bodies
                .sources
                .get(&body_source)
                .map(|&length| (body_source, length))
        };
        inlines.push(Inline {
            call: request.call,
            target: request.target,
            source_name,
            numbers,
            arrow: bodies.bodies[&request.target].arrow,
            guard_code: !matches!(graph.defining_op(*callee), Some(Op::Closure(span, _, false, _))
                if bodies.closure(source, span.start) == Some(request.target)
                && !graph.insts.values().any(|data| matches!(data.op, Op::MarkClass(value) if value == *callee))),
        });
    }
    budget.inlines.extend(inlines.iter().cloned());
    Plan { inlines }
}

pub(super) fn apply(
    graph: &mut Graph,
    bodies: &Repository,
    _source: usize,
    plan: Plan,
    notes: &mut Vec<String>,
) -> Result<()> {
    for inline in plan.inlines {
        let body = bodies.body(inline.target).unwrap();
        super::inlining::apply(graph, body, &inline);
        graph.verify()?;
        let span = graph.insts[inline.call].span;
        notes.push(format!(
            "call at {}..{}: inline {} with {} code check(s) and {} number check(s)",
            span.start,
            span.end,
            inline.target,
            usize::from(inline.guard_code),
            inline.numbers.len()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxc_allocator::Allocator;
    use oxc_ast::ast::Statement;
    use oxc_parser::Parser;
    use oxc_span::SourceType;

    fn fixture(source: &str) -> (Canonical, Repository) {
        let allocator = Allocator::default();
        let parsed = Parser::new(&allocator, source, SourceType::script()).parse();
        assert!(!parsed.diagnostics.has_errors());
        let semantic = oxc_semantic::SemanticBuilder::new_compiler()
            .build(&parsed.program)
            .semantic;
        let mut bodies = Repository::default();
        let mut caller = None;
        for (index, statement) in parsed.program.body.iter().enumerate() {
            let Statement::FunctionDeclaration(function) = statement else {
                panic!()
            };
            let graph =
                super::super::build::lower(function, semantic.scoping(), &HashSet::new()).unwrap();
            let canonical = super::super::canonicalize(graph, true).unwrap();
            let name = function.id.as_ref().unwrap().name.as_str();
            if name == "main" {
                caller = Some(canonical);
            } else {
                bodies.insert(0, name, FuncId::from_u32(index as u32), canonical);
            }
        }
        (caller.unwrap(), bodies)
    }

    fn operations(graph: &Graph) -> Vec<&Op> {
        Cfg::new(graph)
            .rpo
            .into_iter()
            .flat_map(|block| &graph.blocks[block].insts)
            .map(|&inst| &graph.insts[inst].op)
            .collect()
    }

    #[test]
    fn decision_records_include_nested_and_bounded_recursive_inlines() {
        for (source, count) in [
            (
                "function leaf(x){return x*x;} function middle(x){return leaf(x);} function main(x){return middle(x);}",
                2,
            ),
            (
                "function recurse(n){if(n)return recurse(n-1);return 0;} function main(n){return recurse(n);}",
                4,
            ),
        ] {
            let (canonical, bodies) = fixture(source);
            let first =
                super::super::plan_canonical(canonical.clone(), true, Some((&bodies, 0))).unwrap();
            let second = super::super::plan_canonical(canonical, true, Some((&bodies, 0))).unwrap();
            assert_eq!(first.inlines.len(), count);
            assert!(first.inlines == second.inlines);
            let analysis = super::super::optimize_planned(first).unwrap();
            assert!(analysis.inlines == second.inlines);
            analysis.graph.verify().unwrap();
        }
    }

    #[test]
    fn feedback_supplies_guarded_targets_for_property_calls() {
        let (canonical, mut bodies) = fixture(
            "function square(x){return x*x;} function main(x){return Math.square(x)+Math.square(x+1);}",
        );
        let mut graph = canonical.graph;
        assert!(requests(&graph, &bodies, 0).is_empty());
        for data in graph.insts.values() {
            if matches!(data.op, Op::Call { .. }) {
                bodies.feedback.insert(
                    crate::jit::planner::Site {
                        source: 0,
                        start: data.span.start,
                        end: data.span.end,
                    },
                    FuncId::from_u32(0),
                );
            }
        }
        let requests = requests(&graph, &bodies, 0);
        assert_eq!(requests.len(), 2);
        let plan = plan(&graph, &bodies, 0, &requests);
        assert!(plan.inlines.iter().all(|inline| inline.guard_code));
        let mut notes = Vec::new();
        apply(&mut graph, &bodies, 0, plan, &mut notes).unwrap();
        assert_eq!(notes.len(), 2);
        assert_eq!(
            operations(&graph)
                .iter()
                .filter(|op| matches!(op, Op::IsCode(..)))
                .count(),
            2
        );
        assert_eq!(
            operations(&graph)
                .iter()
                .filter(|op| matches!(op, Op::Binary(BinaryOperator::Multiplication, ..)))
                .count(),
            2
        );
    }

    #[test]
    fn varying_method_guards_exit_to_the_original_continuation() {
        let (canonical, mut bodies) = fixture(
            "function method(x){return Math.floor(x);} function main(objects,n){let sum=0;for(let i=0;i<n;++i)sum+=objects[i].method(i);return sum;}",
        );
        for data in canonical.graph.insts.values() {
            if let Op::Call { callee, .. } = data.op
                && matches!(canonical.graph.defining_op(callee), Some(Op::Get(_, name)) if name == "method")
            {
                bodies.feedback.insert(
                    crate::jit::planner::Site {
                        source: 0,
                        start: data.span.start,
                        end: data.span.end,
                    },
                    FuncId::from_u32(0),
                );
            }
        }
        let analysis =
            super::super::optimize_canonical(canonical, true, Some((&bodies, 0))).unwrap();
        let graph = &analysis.graph;
        graph.verify().unwrap();
        let mut fallbacks = Vec::new();
        for block in Cfg::new(graph).rpo {
            if let Some(super::super::Terminator {
                kind: TerminatorKind::Branch(condition, _, otherwise),
                ..
            }) = &graph.blocks[block].terminator
                && matches!(graph.defining_op(*condition), Some(Op::IsCode(_, code)) if *code == FuncId::from_u32(0))
            {
                fallbacks.push(otherwise.block);
            }
        }
        assert!(fallbacks.len() >= 2);
        assert!(fallbacks.iter().all(|block| *block == fallbacks[0]));
        assert!(
            operations(graph)
                .iter()
                .any(|op| matches!(op, Op::CallNative(..)))
        );
    }

    #[test]
    fn explicit_request_has_one_inline_body_and_one_fallback() {
        let (canonical, bodies) =
            fixture("function square(x) { return x * x; } function main(x) { return square(x); }");
        let request = requests(&canonical.graph, &bodies, 0).pop().unwrap();
        let mut graph = canonical.graph.clone();
        let before = canonical
            .graph
            .listing(&Facts::compute(&canonical.graph))
            .to_string();
        let plan = plan(&graph, &bodies, 0, &[request]);
        let mut notes = Vec::new();
        apply(&mut graph, &bodies, 0, plan, &mut notes).unwrap();
        let ops = operations(&graph);
        assert_eq!(
            ops.iter()
                .filter(|op| matches!(op, Op::Call { .. }))
                .count(),
            1
        );
        assert_eq!(
            ops.iter()
                .filter(|op| matches!(op, Op::Binary(BinaryOperator::Multiplication, ..)))
                .count(),
            1
        );
        assert_eq!(
            ops.iter().filter(|op| matches!(op, Op::IsCode(..))).count(),
            1
        );
        assert_eq!(
            ops.iter()
                .filter(|op| matches!(op, Op::IsNumber(_)))
                .count(),
            1
        );
        assert_eq!(
            canonical
                .graph
                .listing(&Facts::compute(&canonical.graph))
                .to_string(),
            before
        );
        let body = bodies.body(FuncId::from_u32(0)).unwrap();
        assert!(
            !operations(body)
                .iter()
                .any(|op| matches!(op, Op::IsNumber(_) | Op::IsCode(..)))
        );
    }

    #[test]
    fn planner_reuses_facts_and_deduplicates_argument_guards() {
        for (argument, checks) in [("x", 1), ("x * 2", 0), ("3", 0)] {
            let (canonical, bodies) = fixture(&format!(
                "function sum(a, b) {{ return a + b; }} function main(x) {{ let y = {argument}; return sum(y, y); }}",
            ));
            let mut graph = canonical.graph;
            let requests = requests(&graph, &bodies, 0);
            let plan = plan(&graph, &bodies, 0, &requests);
            apply(&mut graph, &bodies, 0, plan, &mut Vec::new()).unwrap();
            let ops = operations(&graph);
            assert_eq!(
                ops.iter()
                    .filter(|op| matches!(op, Op::IsNumber(_)))
                    .count(),
                checks
            );
            assert_eq!(
                ops.iter().filter(|op| matches!(op, Op::IsCode(..))).count(),
                1
            );
        }
    }

    #[test]
    fn planner_limits_expansion_and_accepts_general_bodies() {
        let (canonical, bodies) =
            fixture("function square(x) { return x * x; } function main(x) { return square(x); }");
        let request = requests(&canonical.graph, &bodies, 0).pop().unwrap();
        let duplicates = (0..100)
            .map(|_| Request {
                call: request.call,
                target: request.target,
                numeric_arguments: vec![0],
            })
            .collect::<Vec<_>>();
        assert_eq!(
            plan(&canonical.graph, &bodies, 0, &duplicates)
                .inlines
                .len(),
            1
        );
        assert_eq!(
            plan(&canonical.graph, &bodies, 1, &duplicates)
                .inlines
                .len(),
            1
        );
        let calls = "square(x);".repeat(Budget::default().remaining + 1);
        let (canonical, bodies) = fixture(&format!(
            "function square(x) {{ return x * x; }} function main(x) {{ {calls} return x; }}",
        ));
        let candidates = requests(&canonical.graph, &bodies, 0);
        let selected = plan(&canonical.graph, &bodies, 0, &candidates);
        assert!(!selected.inlines.is_empty());
        assert!(selected.inlines.len() < candidates.len());
        for body in [
            "return x.value;",
            "return other(x);",
            "globalThis.x = x; return x;",
        ] {
            let (canonical, bodies) = fixture(&format!(
                "function helper(x) {{ {body} }} function main(x) {{ return helper(x); }}",
            ));
            let requests = requests(&canonical.graph, &bodies, 0);
            assert_eq!(requests.len(), 1);
            assert!(requests[0].numeric_arguments.is_empty());
        }
    }
    #[test]
    fn cross_source_general_calls_and_constructors_are_inlined() {
        let (canonical, bodies) = fixture(
            "function C(x){this.value=x;} function touch(o){o.value+=2;return o;} function main(){C.prototype={};return touch(new C(7)).value;}",
        );
        let mut graph = canonical.graph;
        super::super::construction::resolve(&mut graph);
        super::super::construction::expand(&mut graph);
        let requests = requests(&graph, &bodies, 1);
        assert_eq!(requests.len(), 2);
        assert!(
            requests
                .iter()
                .all(|request| request.numeric_arguments.is_empty())
        );
        let plan = plan(&graph, &bodies, 1, &requests);
        assert_eq!(plan.inlines.len(), 2);
        apply(&mut graph, &bodies, 1, plan, &mut Vec::new()).unwrap();
        graph.verify().unwrap();
    }
}

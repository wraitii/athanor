use std::collections::{HashMap, HashSet};

use cranelift_module::FuncId;
use oxc_syntax::operator::BinaryOperator;

use super::{Canonical, Cfg, Constant, Facts, Graph, Inst, Op, Result, TerminatorKind, Types};

#[derive(Default)]
pub(in crate::jit) struct Repository {
    bodies: HashMap<FuncId, Body>,
    locations: HashMap<(usize, u32), FuncId>,
    globals: HashMap<(usize, String), FuncId>,
}

struct Body {
    source: usize,
    canonical: Canonical,
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
        self.globals.insert((source, name.to_owned()), code);
        self.bodies.insert(code, Body { source, canonical });
    }

    pub(in crate::jit) fn canonical(&self, source: usize, start: u32) -> Option<&Canonical> {
        let code = self.locations.get(&(source, start))?;
        Some(&self.bodies.get(code)?.canonical)
    }

    fn body(&self, source: usize, code: FuncId) -> Option<&Graph> {
        let body = self.bodies.get(&code)?;
        (body.source == source).then_some(&body.canonical.graph)
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

pub(super) struct Inline {
    pub(super) call: Inst,
    pub(super) target: FuncId,
    pub(super) numbers: Vec<usize>,
}

fn eligible(graph: &Graph) -> bool {
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
            if count > 32 {
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

pub(super) fn requests(graph: &Graph, bodies: &Repository, source: usize) -> Vec<Request> {
    let mut requests = Vec::new();
    for block in Cfg::new(graph).rpo {
        for &call in &graph.blocks[block].insts {
            let Op::Call { callee, .. } = &graph.insts[call].op else {
                continue;
            };
            let Some(Op::GlobalRead(name)) = graph.defining_op(*callee) else {
                continue;
            };
            let Some(&target) = bodies.globals.get(&(source, name.clone())) else {
                continue;
            };
            let Some(body) = bodies.body(source, target).filter(|body| eligible(body)) else {
                continue;
            };
            let mut numeric_arguments: Vec<usize> = Cfg::new(body)
                .rpo
                .iter()
                .flat_map(|&block| &body.blocks[block].insts)
                .filter_map(|&inst| match body.insts[inst].op {
                    Op::Argument(index) => Some(index),
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

pub(super) fn plan(
    graph: &Graph,
    bodies: &Repository,
    source: usize,
    requests: &[Request],
) -> Plan {
    let facts = Facts::compute(graph);
    let mut inlines = Vec::new();
    let mut sites = HashSet::new();
    let mut remaining = 128;
    for request in requests {
        let Some(body) = bodies
            .body(source, request.target)
            .filter(|body| eligible(body))
        else {
            continue;
        };
        let Op::Call { args, .. } = &graph.insts[request.call].op else {
            continue;
        };
        let cost: usize = Cfg::new(body)
            .rpo
            .iter()
            .map(|&block| body.blocks[block].insts.len())
            .sum();
        if cost > remaining || sites.contains(&request.call) {
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
        sites.insert(request.call);
        remaining -= cost;
        inlines.push(Inline {
            call: request.call,
            target: request.target,
            numbers,
        });
    }
    Plan { inlines }
}

pub(super) fn apply(
    graph: &mut Graph,
    bodies: &Repository,
    source: usize,
    plan: Plan,
    notes: &mut Vec<String>,
) -> Result<()> {
    for inline in plan.inlines {
        let body = bodies.body(source, inline.target).unwrap();
        super::inlining::apply(graph, body, &inline);
        graph.verify()?;
        let span = graph.insts[inline.call].span;
        notes.push(format!(
            "call at {}..{}: inline {} behind code identity and {} number check(s)",
            span.start,
            span.end,
            inline.target,
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
        let body = bodies.body(0, FuncId::from_u32(0)).unwrap();
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
    fn planner_limits_expansion_and_rejects_unsupported_bodies() {
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
        assert!(
            plan(&canonical.graph, &bodies, 1, &duplicates)
                .inlines
                .is_empty()
        );
        let calls = "square(x);".repeat(100);
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
            assert!(requests(&canonical.graph, &bodies, 0).is_empty());
        }
    }
}

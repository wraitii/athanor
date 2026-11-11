use super::*;
use oxc_allocator::Allocator;
use oxc_ast::ast::Statement;
use oxc_parser::Parser;
use oxc_span::SourceType;

fn with_function<T>(
    source: &str,
    f: impl FnOnce(&oxc_ast::ast::Function<'_>, &Scoping) -> Result<T>,
) -> Result<T> {
    let allocator = Allocator::default();
    let parsed = Parser::new(&allocator, source, SourceType::script()).parse();
    assert!(!parsed.diagnostics.has_errors());
    let Statement::FunctionDeclaration(function) = &parsed.program.body[0] else {
        panic!()
    };
    let semantic = oxc_semantic::SemanticBuilder::new_compiler()
        .build(&parsed.program)
        .semantic;
    f(function, semantic.scoping())
}

fn lower_source(source: &str) -> Result<Graph> {
    with_function(source, |function, scoping| {
        build::lower(function, scoping, &HashSet::new())
    })
}

fn analyze_source(source: &str) -> FunctionState {
    with_function(source, |function, scoping| {
        analyze(function, scoping, &HashSet::new())
    })
    .unwrap()
}

fn ops(graph: &Graph) -> Vec<&Op> {
    Cfg::new(graph)
        .rpo
        .into_iter()
        .flat_map(|block| &graph.blocks[block].insts)
        .map(|&inst| &graph.insts[inst].op)
        .collect()
}

#[test]
fn cfg_preserves_edge_order_and_deduplicates_successors() {
    let mut graph = lower_source("function f(a){return a;}").unwrap();
    let entry = graph.entry;
    let span = graph.span;
    let TerminatorKind::Return(value) = graph.blocks[entry].terminator.as_ref().unwrap().kind
    else {
        panic!()
    };
    let left = graph.blocks.push(BlockData::default());
    let right = graph.blocks.push(BlockData::default());
    let join = graph.blocks.push(BlockData::default());
    let unreachable = graph.blocks.push(BlockData::default());
    graph.terminate(
        entry,
        span,
        TerminatorKind::Branch(value, Target::new(left), Target::new(right)),
    );
    graph.terminate(
        left,
        span,
        TerminatorKind::Branch(value, Target::new(join), Target::new(join)),
    );
    graph.terminate(
        right,
        span,
        TerminatorKind::CheckException(Target::new(right), Target::new(join)),
    );
    graph.terminate(join, span, TerminatorKind::Return(value));
    graph.terminate(unreachable, span, TerminatorKind::Jump(Target::new(join)));

    assert_eq!(graph.successors(entry).collect::<Vec<_>>(), [left, right]);
    assert_eq!(graph.successors(left).collect::<Vec<_>>(), [join]);
    assert_eq!(
        graph.blocks[left]
            .terminator
            .as_ref()
            .unwrap()
            .kind
            .targets()
            .count(),
        2
    );
    assert_eq!(graph.successors(join).count(), 0);

    let cfg = Cfg::new(&graph);
    assert_eq!(cfg.rpo, [entry, right, left, join]);
    assert_eq!(cfg.preds[join], [right, left]);
    assert!(cfg.dominates(entry, join));
    assert!(!cfg.dominates(left, join));
    assert!(!cfg.dominates(right, join));

    let cfg = Cfg::from_entry(&graph, left);
    assert_eq!(cfg.rpo, [left, join]);
    assert_eq!(cfg.preds[join], [left]);
    assert!(cfg.dominates(left, join));
}

#[test]
fn constructor_result_facts_and_folding_preserve_object_returns() {
    for returned in [
        Op::Constant(Constant::Undefined),
        Op::Constant(Constant::Null),
        Op::Constant(Constant::Number(7.0)),
        Op::Constant(Constant::Boolean(true)),
        Op::Constant(Constant::String("ignored".into())),
        Op::NewObject,
        Op::Argument(0),
    ] {
        let mut graph = lower_source("function f(){return 0;}").unwrap();
        let block = graph.entry;
        let span = graph.span;
        let instance = graph.append_value(block, span, Op::NewObject);
        let object_return = matches!(returned, Op::NewObject);
        let unknown_return = matches!(returned, Op::Argument(_));
        let returned = graph.append_value(block, span, returned);
        let result = graph.append_value(block, span, Op::ConstructorResult(returned, instance));
        graph.terminate(block, span, TerminatorKind::Return(result));
        assert_eq!(Facts::compute(&graph).of(result), Types::OBJECT);
        let mut function = FunctionState::new(graph, Vec::new(), true);
        js::simplify(&mut function);
        let graph = function.graph;
        graph.verify().unwrap();
        let TerminatorKind::Return(value) = graph.blocks[block].terminator.as_ref().unwrap().kind
        else {
            panic!()
        };
        assert_eq!(
            value,
            if unknown_return {
                result
            } else if object_return {
                returned
            } else {
                instance
            }
        );
    }
}

#[test]
fn block_arguments_retain_type_test_facts_without_narrowing_unrelated_edges() {
    for kind in 0..5 {
        let mut graph = lower_source("function f(){return 0;}").unwrap();
        let entry = graph.entry;
        let span = graph.span;
        let value = graph.append_value(entry, span, Op::Argument(0));
        let (op, mask) = match kind {
            0 => (Op::IsUndefined(value), Types::UNDEFINED),
            1 => (Op::IsNullish(value), Types::UNDEFINED.union(Types::NULL)),
            2 => (Op::IsString(value), Types::STRING),
            3 => (Op::IsNumber(value), Types::NUMBER),
            _ => (Op::IsUninitialized(value), Types::UNINITIALIZED),
        };
        let condition = graph.append_value(entry, span, op);
        let yes = graph.blocks.push(BlockData::default());
        let no = graph.blocks.push(BlockData::default());
        let join = graph.blocks.push(BlockData::default());
        let positive = graph.add_param(yes);
        let negative = graph.add_param(no);
        let merged = graph.add_param(join);
        graph.terminate(
            entry,
            span,
            TerminatorKind::Branch(
                condition,
                Target {
                    block: yes,
                    args: vec![value],
                },
                Target {
                    block: no,
                    args: vec![value],
                },
            ),
        );
        for (block, arg) in [(yes, positive), (no, negative)] {
            graph.terminate(
                block,
                span,
                TerminatorKind::Jump(Target {
                    block: join,
                    args: vec![arg],
                }),
            );
        }
        graph.terminate(join, span, TerminatorKind::Return(merged));
        graph.verify().unwrap();
        let facts = Facts::compute(&graph);
        let original = facts.of(value);
        assert_eq!(
            facts.of(positive),
            if kind == 4 { Types::default() } else { mask }
        );
        assert_eq!(facts.of(negative), original.without(mask));
        assert_eq!(facts.of(merged), original);
        if let TerminatorKind::Branch(_, yes, no) =
            &mut graph.blocks[entry].terminator.as_mut().unwrap().kind
        {
            *no = yes.clone();
        }
        let facts = Facts::compute(&graph);
        assert_eq!(facts.of(positive), original);
        assert_eq!(facts.of(merged), original);
    }
}

#[test]
fn comparisons_share_number_checks_with_refined_reads_without_splitting() {
    let graph =
        lower_source("function f(height,x,limit){if(x<0||x>=limit)return 0;return height[x][0];}")
            .unwrap();
    let mut graph = canonicalize_graph(graph, true).unwrap().graph;
    let x = graph
        .insts
        .values()
        .find_map(|data| {
            matches!(data.op, Op::Argument(1))
                .then_some(data.result)
                .flatten()
        })
        .unwrap();
    let mut bodies = specialize::Repository::default();
    for data in graph.insts.values() {
        if matches!(data.op, Op::GetIndex(..)) {
            bodies.read_feedback.insert(
                (
                    crate::jit::profile::Site {
                        source: 0,
                        start: data.span.start,
                        end: data.span.end,
                    },
                    crate::jit::profile::ReadKind::Receiver,
                ),
                crate::jit::profile::ReadType::Array,
            );
        }
    }
    let reads = reads::plan(&graph, &bodies, 0);
    reads::apply(&mut graph, &reads, &mut Vec::new()).unwrap();
    let blocks = graph.blocks.len();
    let mut function = FunctionState::new(graph, Vec::new(), true);
    comparisons::expose(&mut function);
    let graph = function.graph;
    graph.verify().unwrap();
    assert_eq!(graph.blocks.len(), blocks);
    let checks: Vec<_> = ops(&graph)
        .into_iter()
        .filter(|op| matches!(op, Op::IsNumber(value) if *value == x))
        .collect();
    assert_eq!(checks.len(), 1);
    let guards: Vec<_> = ops(&graph)
        .into_iter()
        .filter_map(|op| match op {
            Op::GuardedCompare { left, numbers, .. } if *left == x => Some(numbers[0]),
            _ => None,
        })
        .collect();
    assert_eq!(guards.len(), 2);
    assert_eq!(guards[0], guards[1]);
    assert!(Cfg::new(&graph).rpo.iter().any(|&block| matches!(graph.blocks[block].terminator.as_ref().map(|term| &term.kind), Some(TerminatorKind::Branch(condition, ..)) if *condition == guards[0])));
}

#[test]
fn late_number_facts_remove_refinement_guards_and_assertions() {
    let graph = lower_source("function f(height,x,limit){const xx=x+1;if(xx<0||xx>=limit)return 0;return height[xx][0];}").unwrap();
    let mut graph = canonicalize_graph(graph, true).unwrap().graph;
    let sum = graph
        .insts
        .values()
        .find_map(|data| {
            matches!(data.op, Op::Binary(BinaryOperator::Addition, ..))
                .then_some(data.result)
                .flatten()
        })
        .unwrap();
    let mut bodies = specialize::Repository::default();
    for data in graph.insts.values() {
        if matches!(data.op, Op::GetIndex(..)) {
            bodies.read_feedback.insert(
                (
                    crate::jit::profile::Site {
                        source: 0,
                        start: data.span.start,
                        end: data.span.end,
                    },
                    crate::jit::profile::ReadKind::Receiver,
                ),
                crate::jit::profile::ReadType::Array,
            );
        }
    }
    let reads = reads::plan(&graph, &bodies, 0);
    reads::apply(&mut graph, &reads, &mut Vec::new()).unwrap();
    let mut function = FunctionState::new(graph, Vec::new(), true);
    comparisons::expose(&mut function);
    let mut graph = function.graph;
    assert!(
        ops(&graph)
            .iter()
            .any(|op| matches!(op, Op::IsNumber(value) if *value == sum))
    );
    assert!(
        ops(&graph)
            .iter()
            .any(|op| matches!(op, Op::AsNumber(value) if *value == sum))
    );
    let branches = Cfg::new(&graph)
        .rpo
        .iter()
        .filter(|&&block| {
            matches!(
                graph.blocks[block].terminator.as_ref().unwrap().kind,
                TerminatorKind::Branch(..)
            )
        })
        .count();
    for data in graph.insts.values_mut() {
        if matches!(data.op, Op::Argument(1)) {
            data.op = Op::Constant(Constant::Number(2.0));
        }
    }
    let mut function = FunctionState::new(graph, Vec::new(), true);
    js::simplify(&mut function);
    let graph = function.graph;
    graph.verify().unwrap();
    assert!(
        !ops(&graph)
            .iter()
            .any(|op| matches!(op, Op::IsNumber(value) | Op::AsNumber(value) if *value == sum))
    );
    assert!(
        Cfg::new(&graph)
            .rpo
            .iter()
            .filter(|&&block| matches!(
                graph.blocks[block].terminator.as_ref().unwrap().kind,
                TerminatorKind::Branch(..)
            ))
            .count()
            < branches
    );
}

#[test]
fn folding_guard_edges_recomputes_number_facts_at_joins() {
    let graph = lower_source(
        "function f(flag){let x;if(flag)x=1;else x='text';if(x<0)return 0;return +x;}",
    )
    .unwrap();
    let graph = canonicalize_graph(graph, true).unwrap().graph;
    let mut function = FunctionState::new(graph, Vec::new(), true);
    comparisons::expose(&mut function);
    let mut graph = function.graph;
    assert!(ops(&graph).iter().any(|op| matches!(op, Op::IsNumber(_))));
    for data in graph.insts.values_mut() {
        if matches!(data.op, Op::Argument(0)) {
            data.op = Op::Constant(Constant::Boolean(true));
        }
    }
    let mut function = FunctionState::new(graph, Vec::new(), true);
    js::simplify(&mut function);
    let graph = function.graph;
    graph.verify().unwrap();
    assert!(!ops(&graph).iter().any(|op| matches!(
        op,
        Op::IsNumber(_) | Op::AsNumber(_) | Op::ToNumeric(_) | Op::Unary(UnaryOp::Plus, _)
    )));
}

#[test]
fn loop_versioning_reuses_exposed_comparison_checks() {
    let analysis = analyze_source(
        "function f(limit){let sum=0;for(let i=0;i<2;++i){if(i<limit)sum+=i;}return sum;}",
    );
    let graph = &analysis.graph;
    assert_eq!(
        ops(graph)
            .into_iter()
            .filter(|op| matches!(op, Op::IsNumber(value) if matches!(graph.defining_op(*value), Some(Op::Argument(0)))))
            .count(),
        1
    );
}

#[test]
fn comparison_checks_do_not_cross_non_dominating_branches() {
    let graph = lower_source("function f(x,y,flag){if(flag)return x<y;return x>y;}").unwrap();
    let graph = canonicalize_graph(graph, true).unwrap().graph;
    let mut function = FunctionState::new(graph, Vec::new(), true);
    comparisons::expose(&mut function);
    let graph = function.graph;
    graph.verify().unwrap();
    assert_eq!(
        ops(&graph)
            .into_iter()
            .filter(|op| matches!(op, Op::IsNumber(_)))
            .count(),
        4
    );
}

#[test]
fn indexed_reads_are_reused_probed_and_get_hoisted_views() {
    let graph = lower_source("function f(array){const zero=0;let sum=0;for(let i=0;i<2;++i){sum+=array[zero]+array[zero];}return sum;}").unwrap();
    let mut canonical = canonicalize_graph(graph, true).unwrap();
    for data in canonical.graph.insts.values_mut() {
        if let Op::GetIndex(object, key) = data.op {
            data.op = Op::IndexedGet(object, key);
        }
    }
    let viewed = canonical.graph.clone();
    let mut function = FunctionState::new(viewed, Vec::new(), true);
    optimize::views(&mut function);
    let viewed = function.graph;
    assert!(
        viewed
            .views
            .keys()
            .any(|&inst| matches!(viewed.insts[inst].op, Op::IndexedGet(..)))
    );
    let analysis = optimize_canonical(canonical, true, None).unwrap();
    assert!(ops(&analysis.graph).iter().any(|op| matches!(
        op, Op::Binary(oxc_syntax::operator::BinaryOperator::Addition, left, right) if left == right
    )));
    assert!(
        analysis
            .notes
            .iter()
            .any(|note| note.contains("read probes"))
    );
    analysis.graph.verify().unwrap();
}

#[test]
fn indexed_rows_borrow_from_owned_receivers_through_guards() {
    for (body, borrowed) in [
        ("sum+=row[0];", true),
        ("change(height);sum+=row[0];", false),
        ("height[i&1]=null;sum+=row[0];", false),
    ] {
        let source = format!(
            "function f(owner,n){{const height=owner.height;let sum=0;for(let i=0;i<n;++i){{const row=height[i&1];{body}}}return sum;}}"
        );
        let canonical = canonicalize_graph(lower_source(&source).unwrap(), true).unwrap();
        let mut repository = specialize::Repository::default();
        for data in canonical.graph.insts.values() {
            if !matches!(data.op, Op::GetIndex(..)) {
                continue;
            }
            let text = &source[data.span.start as usize..data.span.end as usize];
            repository.read_feedback.insert(
                (
                    crate::jit::profile::Site {
                        source: 0,
                        start: data.span.start,
                        end: data.span.end,
                    },
                    crate::jit::profile::ReadKind::Receiver,
                ),
                if text == "height[i&1]" {
                    crate::jit::profile::ReadType::Array
                } else {
                    crate::jit::profile::ReadType::Typed(crate::heap::TypedKind::Float32)
                },
            );
        }
        let analysis = optimize_canonical(canonical, true, Some((&repository, 0))).unwrap();
        let rows: Vec<_> = analysis
            .graph
            .insts
            .values()
            .filter(|data| {
                matches!(data.op, Op::ArrayLoad(..))
                    && &source[data.span.start as usize..data.span.end as usize] == "height[i&1]"
            })
            .collect();
        assert!(!rows.is_empty());
        assert_eq!(
            analysis.graph.insts.values().any(|data| {
                matches!(data.op, Op::TypedIndex(object, ..) if analysis.graph.borrowed.contains(&object))
            }),
            borrowed
        );
        analysis.graph.verify().unwrap();
    }
}

// Varying read guards alone do not trigger region versioning, so generic reads still
// merge with the fast path and prevent borrowing. Enabling these regions also requires
// repairing loop values correctly on guard-failure continuations.
#[test]
#[should_panic(expected = "guard-only region should borrow the array row")]
fn guard_only_region_borrows_rows_without_entry_assumptions() {
    use crate::jit::profile::{ReadKind, ReadType, Site};

    let source =
        "function f(height){let sum=0;for(let i=0;i<8;++i)sum+=height[i][0]/1;return sum;}";
    let canonical = canonicalize_graph(lower_source(source).unwrap(), true).unwrap();
    let mut repository = specialize::Repository::default();
    for data in canonical.graph.insts.values() {
        if !matches!(data.op, Op::GetIndex(..)) {
            continue;
        }
        let text = &source[data.span.start as usize..data.span.end as usize];
        repository.read_feedback.insert(
            (
                Site {
                    source: 0,
                    start: data.span.start,
                    end: data.span.end,
                },
                ReadKind::Receiver,
            ),
            if text == "height[i]" {
                ReadType::Array
            } else {
                ReadType::Typed(crate::heap::TypedKind::Float32)
            },
        );
    }
    let analysis = optimize_canonical(canonical, true, Some((&repository, 0))).unwrap();
    analysis.graph.verify().unwrap();
    let graph = &analysis.graph;
    let cfg = Cfg::new(graph);
    let rows: Vec<_> = cfg
        .rpo
        .iter()
        .filter(|block| !graph.cold.contains(block))
        .flat_map(|&block| graph.blocks[block].insts.iter())
        .filter_map(|&inst| match graph.insts[inst].op {
            Op::TypedIndex(
                object,
                _,
                TypedRead {
                    element: Some(crate::heap::TypedKind::Float32),
                    ..
                },
            ) => Some(object),
            _ => None,
        })
        .collect();
    assert!(!rows.is_empty());
    assert!(
        rows.iter().all(|row| graph.borrowed.contains(row)),
        "guard-only region should borrow the array row"
    );
}

#[test]
fn typed_loop_reads_are_reused_before_inner_loop_versioning() {
    let graph = lower_source("function f(grid,width){let members=0;for(let y=0;y<2;++y){const indexY=y*width;for(let x=0;x<2;++x){if(grid[indexY+x])for(let i=0;i<16;++i)if(grid[indexY+x]&(1<<i))++members;}}return members;}").unwrap();
    let canonical = canonicalize_graph(graph, true).unwrap();
    let read = canonical
        .graph
        .insts
        .values()
        .filter(|data| matches!(data.op, Op::GetIndex(..)))
        .min_by_key(|data| data.span.start)
        .unwrap();
    let mut bodies = specialize::Repository::default();
    bodies.read_feedback.insert(
        (
            crate::jit::profile::Site {
                source: 0,
                start: read.span.start,
                end: read.span.end,
            },
            crate::jit::profile::ReadKind::Receiver,
        ),
        crate::jit::profile::ReadType::Typed(crate::heap::TypedKind::Uint16),
    );
    let analysis = optimize_canonical(canonical, true, Some((&bodies, 0))).unwrap();
    let operations = ops(&analysis.graph);
    assert!(operations.iter().any(|op| matches!(op, Op::TypedIndex(..))));
    assert!(!operations.iter().any(|op| matches!(op, Op::ProbeIndex(..))));
    analysis.graph.verify().unwrap();
}

#[test]
fn typeof_string_comparisons_avoid_string_materialization() {
    for expression in [
        "typeof value === 'string'",
        "'string' === typeof value",
        "typeof value !== 'string'",
        "'string' !== typeof value",
    ] {
        let analysis = analyze_source(&format!("function f(value) {{ return {expression}; }}"));
        let operations = ops(&analysis.graph);
        assert!(operations.iter().any(|op| matches!(op, Op::IsString(_))));
        assert!(!operations.iter().any(|op| matches!(
            op,
            Op::Constant(Constant::String(_)) | Op::Unary(UnaryOp::Typeof, _)
        )));
    }
}

#[test]
fn string_guard_refines_only_the_dominated_edge() {
    let analysis = analyze_source(
        "function f(value){if(typeof value==='string'){if(typeof value==='string')return 1;return 2;}return 3;}",
    );
    assert_eq!(
        ops(&analysis.graph)
            .iter()
            .filter(|op| matches!(op, Op::IsString(_)))
            .count(),
        1
    );
    assert!(
        !ops(&analysis.graph)
            .iter()
            .any(|op| matches!(op, Op::Constant(Constant::Number(number)) if *number == 2.0))
    );
}

#[test]
fn typeof_comparisons_avoid_string_materialization() {
    for name in ["number", "boolean", "undefined", "object", "function"] {
        for operator in ["===", "!=="] {
            for expression in [
                format!("typeof value {operator} '{name}'"),
                format!("'{name}' {operator} typeof value"),
            ] {
                let analysis =
                    analyze_source(&format!("function f(value) {{ return {expression}; }}"));
                let operations = ops(&analysis.graph);
                assert!(!operations.iter().any(|op| matches!(
                    op,
                    Op::Constant(Constant::String(_)) | Op::Unary(UnaryOp::Typeof, _)
                )));
            }
        }
    }
}

const CONTROL_FLOW: &str = "function f(a, b = g.h) {
        const r = { min: Infinity, list: [a, 'x', null] };
        let i = 0;
        while (i < a.length) {
            let v = a[i++];
            if (v === undefined) continue;
            if (v > 10) { const w = v; break; }
            r.min = Math.min(r.min, v || b ? v : -v);
            a[i] += 1;
        }
        for (let x = 0; x < 3; ++x) {
            if (x == 2) return r;
            else { let y = x; }
        }
        return b ?? r;
    }";

#[test]
fn lowers_control_flow_into_verified_ssa() {
    let graph = lower_source(CONTROL_FLOW).unwrap();
    assert!(
        ops(&graph)
            .iter()
            .any(|op| matches!(op, Op::Constant(Constant::Number(n)) if n.is_infinite()))
    );
    assert!(graph.blocks.values().any(|block| !block.params.is_empty()));
    analyze_source(CONTROL_FLOW);
}

#[test]
fn postfix_update_yields_numeric_old_value() {
    let graph = lower_source("function f(x) { return x++; }").unwrap();
    let block = &graph.blocks[graph.entry];
    let TerminatorKind::Return(value) = block.terminator.as_ref().unwrap().kind else {
        panic!()
    };
    assert!(matches!(graph.defining_op(value), Some(Op::ToNumeric(_))));
}

#[test]
fn keeps_shadowed_bindings_and_call_order() {
    let graph = lower_source("function f(x) { { let x = 1; obj.m(x); } return x; }").unwrap();
    assert_eq!(graph.bindings.len(), 2);
    let ops = ops(&graph);
    let get = ops
        .iter()
        .position(|op| matches!(op, Op::Get(_, name) if name == "m"))
        .unwrap();
    let read = ops
        .iter()
        .position(|op| matches!(op, Op::Read(binding) if binding.index() == 1))
        .unwrap();
    let call = ops
        .iter()
        .position(|op| {
            matches!(
                op,
                Op::Call {
                    receiver: Some(_),
                    ..
                }
            )
        })
        .unwrap();
    assert!(get < read && read < call);
}

#[test]
fn rejects_unsupported_syntax() {
    let Err(error) = lower_source("function f(a) { return delete a.x; }") else {
        panic!()
    };
    assert!(error.contains("unsupported"));
}

#[test]
fn promotes_bindings_and_infers_numbers() {
    let analysis = analyze_source(
        "function f(a) { let s = 0; for (let i = 0; i < a; ++i) s = s + i; return s; }",
    );
    let graph = &analysis.graph;
    assert!(!ops(graph).iter().any(|op| matches!(
        op,
        Op::Read(_) | Op::Write(..) | Op::Initialize(..) | Op::CheckInitialized(_)
    )));
    let returned = graph
        .blocks
        .values()
        .find_map(|block| match block.terminator.as_ref()?.kind {
            TerminatorKind::Return(value) => Some(value),
            _ => None,
        })
        .unwrap();
    assert!(analysis.facts.of(returned).within(Types::NUMBER));
}

#[test]
fn keeps_initialization_check_before_declaration() {
    let analysis = analyze_source("function f() { let x = y; const y = 2; return x; }");
    assert!(
        ops(&analysis.graph)
            .iter()
            .any(|op| matches!(op, Op::CheckInitialized(_)))
    );
}

#[test]
fn resets_promoted_bindings_on_scope_entry() {
    for source in [
        "function f() { for (let i = 0; i < 2; i++) { if (i) return x; let x = 1; } }",
        "function f() { for (let i = 0; i < 2; i++) { if (i) x; let x = 1; } }",
    ] {
        let analysis = analyze_source(source);
        let graph = &analysis.graph;
        let checks: Vec<_> = ops(graph)
            .into_iter()
            .filter_map(|op| match op {
                Op::CheckInitialized(value) => Some(*value),
                _ => None,
            })
            .collect();
        assert!(!checks.is_empty());
        for value in checks {
            assert!(matches!(
                graph.defining_op(value),
                Some(Op::Constant(Constant::Uninitialized))
            ));
        }
    }
}

#[test]
fn global_writes_prevent_versioning_and_load_reuse() {
    let analysis = analyze_source(
        "function f() { let before = shared; Math = shared; shared = Math.min(3, 4); return before + shared; }",
    );
    let ops = ops(&analysis.graph);
    assert!(ops.iter().any(|op| matches!(op, Op::GlobalWrite(..))));
    assert_eq!(
        ops.iter()
            .filter(|op| matches!(op, Op::GlobalRead(name) if name == "shared"))
            .count(),
        2
    );
    assert!(!ops.iter().any(|op| matches!(op, Op::CallNative(..))));
}

#[test]
fn small_constant_products_preserve_integer_types() {
    for (factor, expected) in [
        ("3", Types::INT64),
        ("4194304", Types::INT64),
        ("4194305", Types::NUMBER),
        ("0", Types::NUMBER),
        ("-1", Types::NUMBER),
        ("0.5", Types::NUMBER),
    ] {
        for expression in [format!("(x|0)*{factor}"), format!("{factor}*(x|0)")] {
            let graph = lower_source(&format!("function f(x){{return {expression};}}")).unwrap();
            let facts = Facts::compute(&graph);
            let result = graph
                .insts
                .values()
                .find_map(|data| {
                    matches!(data.op, Op::Binary(BinaryOperator::Multiplication, ..))
                        .then(|| data.result.unwrap())
                })
                .unwrap();
            assert_eq!(facts.of(result), expected, "{expression}");
        }
    }
}

#[test]
fn integer_coercions_are_reused_and_hoisted_out_of_bit_loops() {
    let analysis = analyze_source(
        "function f() { const grid=new Float64Array(4); let sum=0; for(let x=0;x<grid.length;++x) { const word=grid[x]; for(let i=0;i<16;++i) { sum+=word&(1<<i); sum+=word^i; } } return sum; }",
    );
    let graph = &analysis.graph;
    let cfg = Cfg::new(graph);
    let conversions: Vec<_> = cfg
        .rpo
        .iter()
        .flat_map(|&block| &graph.blocks[block].insts)
        .filter_map(|&inst| match graph.insts[inst].op {
            Op::ToInt32(value) if matches!(graph.defining_op(value), Some(Op::TypedIndex(..))) => {
                Some(inst)
            }
            _ => None,
        })
        .collect();
    assert_eq!(conversions.len(), 1);
    let conversion = &graph.insts[conversions[0]];
    let converted = conversion.result.unwrap();
    let users: Vec<_> = cfg
        .rpo
        .iter()
        .flat_map(|&block| &graph.blocks[block].insts)
        .filter(|&&inst| graph.insts[inst].op.operands().contains(&converted))
        .collect();
    assert_eq!(users.len(), 2);
    for &user in users {
        assert!(cfg.rpo.iter().any(|&header| {
            let Some(Terminator {
                kind: TerminatorKind::Branch(condition, _, _),
                ..
            }) = &graph.blocks[header].terminator
            else {
                return false;
            };
            let Some(&Op::Binary(BinaryOperator::LessThan, _, bound)) =
                graph.defining_op(*condition)
            else {
                return false;
            };
            matches!(
                graph.defining_op(bound),
                Some(Op::Constant(Constant::Number(16.0)))
            ) && conversion.block != header
                && cfg.dominates(conversion.block, header)
                && cfg.dominates(header, graph.insts[user].block)
        }));
    }
}

#[test]
fn capture_cell_writes_preserve_guarded_globals_and_properties() {
    let mut graph = lower_source(
        "function f(n){let count=0;for(let i=0;i<n;++i){++count;Math.floor(2);}return count;}",
    )
    .unwrap();
    let binding = graph
        .bindings
        .iter()
        .find_map(|(binding, data)| (data.name == "count").then_some(binding))
        .unwrap();
    graph.bindings[binding].captured = true;
    let canonical = canonicalize_graph(graph, true).unwrap();
    let mut graph = canonical.graph.clone();
    let regions = optimize::version(
        &mut graph,
        &specialize::ConditionalPlan::default(),
        None,
        &mut Vec::new(),
    )
    .unwrap();
    graph.verify().unwrap();
    assert!(!regions.is_empty());
    assert!(
        ops(&graph)
            .iter()
            .any(|op| matches!(op, Op::CallNative(..)))
    );
    assert!(
        ops(&graph)
            .iter()
            .any(|op| matches!(op, Op::Write(id, _) if *id == binding))
    );
    let analysis = optimize_canonical(canonical, true, None).unwrap();
    assert!(!analysis.graph.bindings[binding].captured);
    assert!(!ops(&analysis.graph).iter().any(|op| matches!(op,
            Op::Read(id) | Op::Write(id, _) | Op::Initialize(id, _) if *id == binding)));
}

#[test]
fn versions_on_native_calls_only_when_closed() {
    let source = "function f(m) {
            const r = { min: Infinity };
            for (let x = 0; x < m.length; ++x)
                for (let y = 0; y < m[x].length; ++y)
                    r.min = Math.min(r.min, m[x][y]);
            return r;
        }";
    let analysis = analyze_source(source);
    let versioned = ops(&analysis.graph);
    assert!(versioned.iter().any(|op| matches!(op, Op::CallNative(..))));
    assert!(
        super::continuation::bodies(&analysis.graph)
            .iter()
            .any(|body| ops(&body.graph)
                .iter()
                .any(|op| matches!(op, Op::Call { .. })))
    );
    let unknown = analyze_source(
        "function f(m) { let s = 0; for (let x = 0; x < m.length; ++x) s = Math.min(s, g(x)); return s; }",
    );
    assert!(
        !ops(&unknown.graph)
            .iter()
            .any(|op| matches!(op, Op::CallNative(..)))
    );
}

#[test]
fn verifier_rejects_missing_late_and_mismatched_definitions() {
    let graph = canonicalize_graph(
        lower_source("function f(x) { return x + 1; }").unwrap(),
        true,
    )
    .unwrap()
    .graph;
    graph.verify().unwrap();
    let (block, user, value, def) = graph
        .blocks
        .iter()
        .find_map(|(block, data)| {
            data.insts.iter().find_map(|&user| {
                graph.insts[user].op.operands_iter().find_map(|value| {
                    let ValueDef::Result(def) = graph.values[value] else {
                        return None;
                    };
                    (graph.insts[def].block == block).then_some((block, user, value, def))
                })
            })
        })
        .unwrap();

    let mut missing = graph.clone();
    missing.remove_inst(def);
    assert!(missing.verify().is_err());

    let mut late = graph.clone();
    late.remove_inst(def);
    let user_index = late.blocks[block]
        .insts
        .iter()
        .position(|&inst| inst == user)
        .unwrap();
    late.blocks[block].insts.insert(user_index + 1, def);
    assert!(late.verify().is_err());

    let mut mismatch = graph;
    assert_eq!(mismatch.insts[def].result, Some(value));
    mismatch.insts[def].result = None;
    assert!(mismatch.verify().is_err());
}

#[test]
fn borrowed_rows_flow_through_phi_parameters() {
    for (selection, body, expected) in [
        ("choose ? rows[0] : rows[1]", "", true),
        ("choose ? rows[0] : rows[1]", "change(rows);", false),
        ("choose ? rows[0] : rows[1]", "rows[0]=null;", false),
        ("choose ? rows[0] : {0: 3}", "", false),
    ] {
        let source =
            format!("function f(rows,choose){{const row={selection};{body}return row[0];}}");
        let mut graph = canonicalize_graph(lower_source(&source).unwrap(), true)
            .unwrap()
            .graph;
        let mut result = None;
        for (_, data) in graph.insts.iter_mut() {
            if let Op::GetIndex(object, key) = data.op {
                let text = &source[data.span.start as usize..data.span.end as usize];
                if text == "row[0]" {
                    result = Some(object);
                    data.op = Op::TypedIndex(
                        object,
                        key,
                        TypedRead {
                            element: Some(crate::heap::TypedKind::Float32),
                            bounded: false,
                        },
                    );
                } else {
                    data.op = Op::IndexedGet(object, key);
                }
            }
        }
        let row = result.unwrap();
        assert!(matches!(graph.values[row], ValueDef::Param(..)));
        let facts = Facts::compute(&graph);
        borrow::borrows(&mut graph, &facts, &mut Vec::new());
        assert_eq!(graph.borrowed.contains(&row), expected, "{source}");
    }
}

#[test]
fn integer_facts_survive_loop_backedges_without_length_bounds() {
    for step in ["++k", "k = k + 1", "--k", "k = k - 1"] {
        let source = format!("function f(limit){{let k=0;while(k!==limit){{{step};}}return k;}}");
        let graph = canonicalize_graph(lower_source(&source).unwrap(), true)
            .unwrap()
            .graph;
        let facts = Facts::compute(&graph);
        let mut steps = 0;
        for data in graph.insts.values() {
            if matches!(
                data.op,
                Op::Unary(UnaryOp::Increment | UnaryOp::Decrement, _)
                    | Op::Binary(BinaryOperator::Addition | BinaryOperator::Subtraction, ..)
            ) {
                assert!(facts.of(data.result.unwrap()).within(Types::INTEGER));
                steps += 1;
            }
        }
        assert!(steps > 0);
        assert!(
            graph
                .blocks
                .values()
                .flat_map(|block| &block.params)
                .any(|&value| facts.of(value).within(Types::INTEGER)
                    && facts.of(value).intersects(Types::INT64))
        );
    }
}

#[test]
fn unsupported_numeric_reads_are_not_observed_again() {
    use crate::jit::profile::{Observation, ReadKind, ReadType, Site};

    let mut graph = lower_source("function f(point){return point.x + 1;}").unwrap();
    observations::plan(&mut graph, None);
    let (value, site) = ops(&graph)
        .into_iter()
        .find_map(|op| match op {
            Op::Observe(value, Observation::Read(ReadKind::Number), site) => Some((*value, *site)),
            _ => None,
        })
        .unwrap();
    let mut repository = specialize::Repository::default();
    let site = Site { source: 7, ..site };
    repository
        .read_feedback
        .insert((site, ReadKind::Number), ReadType::Unknown);
    assert!(observations::plan(&mut graph, Some((&repository, 7))).is_empty());
    assert!(
        !ops(&graph)
            .iter()
            .any(|op| matches!(op, Op::Observe(_, Observation::Read(ReadKind::Number), _)))
    );
    assert_eq!(
        observations::numeric_hints(&graph).get(&value),
        Some(&ReadType::Unknown)
    );
    graph.verify().unwrap();
}

#[test]
fn checked_typed_reads_expose_shared_numeric_checks() {
    let mut graph = canonicalize_graph(
        lower_source("function f(a,b,k){k=+k;return a[k]+b[k];}").unwrap(),
        true,
    )
    .unwrap()
    .graph;
    let reads: Vec<_> = graph
        .insts
        .iter()
        .filter_map(|(inst, data)| matches!(data.op, Op::GetIndex(..)).then_some(inst))
        .collect();
    assert_eq!(reads.len(), 2);
    let Op::GetIndex(_, key) = graph.insts[reads[0]].op else {
        unreachable!()
    };
    for &inst in &reads {
        let Op::GetIndex(object, _) = graph.insts[inst].op else {
            unreachable!()
        };
        graph.insts[inst].op = Op::TypedIndex(
            object,
            key,
            TypedRead {
                element: Some(crate::heap::TypedKind::Float64),
                bounded: false,
            },
        );
    }
    let region = graph.regions.push(Region {
        blocks: graph.blocks.keys().collect(),
        fallback: None,
    });
    for &inst in &reads {
        graph.insts[inst].source_name = Some((17, 3));
    }
    let previous_insts = graph.insts.len();
    let mut function = FunctionState::new(graph, Vec::new(), true);
    assert!(typed_lowering::lower(&mut function).unwrap());
    let graph = function.graph;
    let cfg = Cfg::new(&graph);
    assert!(
        cfg.rpo
            .iter()
            .all(|block| graph.regions[region].blocks.contains(block))
    );
    assert!(
        graph
            .insts
            .iter()
            .skip(previous_insts)
            .all(|(_, data)| data.source_name == Some((17, 3)))
    );
    for &inst in &reads {
        let load = graph.insts[inst].block;
        assert_eq!(cfg.preds[load].len(), 1);
        let bounds = cfg.preds[load][0];
        let TerminatorKind::Branch(condition, yes, no) =
            &graph.blocks[bounds].terminator.as_ref().unwrap().kind
        else {
            panic!()
        };
        assert_eq!(yes.block, load);
        assert_ne!(no.block, load);
        assert!(matches!(
            graph.defining_op(*condition),
            Some(Op::IsInBounds(..))
        ));
        assert_eq!(cfg.preds[bounds].len(), 1);
        let check = cfg.preds[bounds][0];
        let TerminatorKind::Branch(condition, yes, no) =
            &graph.blocks[check].terminator.as_ref().unwrap().kind
        else {
            panic!()
        };
        assert_eq!(yes.block, bounds);
        assert_ne!(no.block, bounds);
        assert!(matches!(
            graph.defining_op(*condition),
            Some(Op::IsExactPosition(..))
        ));
    }
    let mut function = FunctionState::new(graph, Vec::new(), true);
    optimize::reuse_loads(&mut function);
    let graph = function.graph;
    graph.verify().unwrap();
    let ops: Vec<_> = Cfg::new(&graph)
        .rpo
        .into_iter()
        .flat_map(|block| {
            graph.blocks[block]
                .insts
                .iter()
                .map(|&inst| &graph.insts[inst].op)
        })
        .collect();
    assert_eq!(
        ops.iter()
            .filter(|op| matches!(op, Op::ArrayPosition(_)))
            .count(),
        1
    );
    assert_eq!(
        ops.iter()
            .filter(|op| matches!(op, Op::IsExactPosition(..)))
            .count(),
        1
    );
    assert_eq!(
        ops.iter()
            .filter(|op| matches!(op, Op::IsInBounds(..)))
            .count(),
        2
    );
    assert_eq!(
        ops.iter()
            .filter(|op| matches!(op, Op::TypedIndex(_, _, TypedRead { bounded: true, .. })))
            .count(),
        2
    );
    let facts = Facts::compute(&graph);
    for &inst in &reads {
        assert!(
            facts
                .of(graph.insts[inst].result.unwrap())
                .within(Types::NUMBER)
        );
    }
}

#[test]
fn integer_typed_read_skips_exactness_roundtrip() {
    let mut graph = canonicalize_graph(
        lower_source("function f(a,k){k=k|0;return a[k];}").unwrap(),
        true,
    )
    .unwrap()
    .graph;
    let inst = graph
        .insts
        .iter()
        .find_map(|(inst, data)| matches!(data.op, Op::GetIndex(..)).then_some(inst))
        .unwrap();
    let Op::GetIndex(object, key) = graph.insts[inst].op else {
        unreachable!()
    };
    graph.insts[inst].op = Op::TypedIndex(
        object,
        key,
        TypedRead {
            element: Some(crate::heap::TypedKind::Int32),
            bounded: false,
        },
    );
    let mut function = FunctionState::new(graph, Vec::new(), true);
    assert!(typed_lowering::lower(&mut function).unwrap());
    let graph = function.graph;
    assert!(matches!(
        graph.insts[inst].op,
        Op::TypedIndex(_, _, TypedRead { bounded: true, .. })
    ));
    assert!(!Cfg::new(&graph).rpo.iter().any(|&block| {
        graph.blocks[block]
            .insts
            .iter()
            .any(|&inst| matches!(graph.insts[inst].op, Op::IsExactPosition(..)))
    }));
}

#[test]
fn undefined_comparisons_refine_typed_read_results() {
    for (operator, comparison) in [("===", "x===undefined"), ("!==", "undefined!==x")] {
        let source = format!(
            "function f(a,k){{const x=a[k];return {comparison}?{}:{};}}",
            if operator == "===" { "0" } else { "x" },
            if operator == "===" { "x" } else { "0" }
        );
        let mut graph = canonicalize_graph(lower_source(&source).unwrap(), true)
            .unwrap()
            .graph;
        let inst = graph
            .insts
            .iter()
            .find_map(|(inst, data)| matches!(data.op, Op::GetIndex(..)).then_some(inst))
            .unwrap();
        let Op::GetIndex(object, key) = graph.insts[inst].op else {
            unreachable!()
        };
        graph.insts[inst].op = Op::TypedIndex(
            object,
            key,
            TypedRead {
                element: Some(crate::heap::TypedKind::Float64),
                bounded: false,
            },
        );
        let mut function = FunctionState::new(graph, Vec::new(), true);
        comparisons::expose(&mut function);
        let graph = function.graph;
        graph.verify().unwrap();
        assert!(
            ops(&graph)
                .iter()
                .any(|op| matches!(op, Op::IsUndefined(_)))
        );
        let result = graph
            .blocks
            .values()
            .find_map(|block| match block.terminator.as_ref()?.kind {
                TerminatorKind::Return(value) => Some(value),
                _ => None,
            })
            .unwrap();
        let types = Facts::compute(&graph).of(result);
        assert!(
            types.intersects(Types::NUMBER) && types.within(Types::NUMBER),
            "{source}: {types}"
        );
    }
    for source in [
        "function f(x){return x==undefined;}",
        "function f(x){return x!=undefined;}",
        "function f(x,undefined){return x===undefined;}",
    ] {
        let graph = canonicalize_graph(lower_source(source).unwrap(), true)
            .unwrap()
            .graph;
        let mut function = FunctionState::new(graph, Vec::new(), true);
        comparisons::expose(&mut function);
        let graph = function.graph;
        assert!(
            !ops(&graph)
                .iter()
                .any(|op| matches!(op, Op::IsUndefined(_))),
            "{source}"
        );
    }
}

#[test]
fn forwarded_undefined_facts_stop_at_joins_and_alternate_predecessors() {
    let mut graph = lower_source("function f(){return 0;}").unwrap();
    let entry = graph.entry;
    let span = graph.span;
    let value = graph.append_value(entry, span, Op::Argument(0));
    let condition = graph.append_value(entry, span, Op::IsUndefined(value));
    let yes = graph.blocks.push(BlockData::default());
    let no = graph.blocks.push(BlockData::default());
    let forward = graph.blocks.push(BlockData::default());
    let join = graph.blocks.push(BlockData::default());
    let passing = graph.blocks.push(BlockData::default());
    let narrowed = graph.add_param(passing);
    let parameter = graph.add_param(join);
    graph.terminate(
        entry,
        span,
        TerminatorKind::Branch(condition, Target::new(yes), Target::new(no)),
    );
    graph.terminate(
        yes,
        span,
        TerminatorKind::Jump(Target {
            block: join,
            args: vec![value],
        }),
    );
    graph.terminate(no, span, TerminatorKind::Jump(Target::new(forward)));
    graph.terminate(
        forward,
        span,
        TerminatorKind::Jump(Target {
            block: passing,
            args: vec![value],
        }),
    );
    graph.terminate(
        passing,
        span,
        TerminatorKind::Jump(Target {
            block: join,
            args: vec![narrowed],
        }),
    );
    graph.terminate(join, span, TerminatorKind::Return(parameter));
    graph.verify().unwrap();
    let facts = Facts::compute(&graph);
    assert_eq!(
        facts.of(narrowed),
        facts.of(value).without(Types::UNDEFINED)
    );
    assert_eq!(facts.of(parameter), facts.of(value));
    graph.terminate(yes, span, TerminatorKind::Jump(Target::new(no)));
    let facts = Facts::compute(&graph);
    assert_eq!(facts.of(narrowed), facts.of(value));
    assert_eq!(facts.of(parameter), facts.of(value));
    graph.terminate(
        join,
        span,
        TerminatorKind::Branch(
            condition,
            Target::new(no),
            Target {
                block: join,
                args: vec![parameter],
            },
        ),
    );
    graph.verify().unwrap();
    let facts = Facts::compute(&graph);
    assert_eq!(facts.of(parameter), facts.of(value));
}

#[test]
fn facts_refresh_after_instruction_and_control_flow_changes() {
    let graph = lower_source("function f(){return 1;}").unwrap();
    let mut function = FunctionState::new(graph, Vec::new(), true);
    function.ensure_facts();
    let (inst, value) = function
        .graph
        .insts
        .iter()
        .find_map(|(inst, data)| {
            matches!(data.op, Op::Constant(Constant::Number(1.0)))
                .then_some((inst, data.result.unwrap()))
        })
        .unwrap();
    let cfg_storage = function.facts.cfg.rpo.as_ptr();
    assert_eq!(function.facts.of(value), Types::INT32);
    function.graph.insts[inst].op = Op::Constant(Constant::String("one".into()));
    function.invalidate_values();
    function.ensure_facts();
    assert_eq!(function.facts.of(value), Types::STRING);
    assert_eq!(function.facts.cfg.rpo.as_ptr(), cfg_storage);
    assert!(function.facts.is_pure(inst));

    let graph = &mut function.graph;
    let exit = graph.blocks.push(BlockData::default());
    graph.terminate(exit, graph.span, TerminatorKind::Return(value));
    graph.terminate(
        graph.entry,
        graph.span,
        TerminatorKind::Jump(Target::new(exit)),
    );
    function.invalidate_cfg();
    function.ensure_facts();
    function.graph.verify().unwrap();
    assert!(function.facts.cfg.rpo.contains(&exit));
    assert!(function.facts.cfg.dominates(function.graph.entry, exit));
    assert_eq!(function.facts.of(value), Types::STRING);
}

#[test]
fn unchanged_hoisting_preserves_facts() {
    let graph = canonicalize_graph(
        lower_source("function f(a){while(a){a=a.next;}return a;}").unwrap(),
        true,
    )
    .unwrap()
    .graph;
    let mut function = FunctionState::new(graph, Vec::new(), true);
    let region = function.graph.regions.push(Region {
        blocks: function.graph.blocks.keys().collect(),
        fallback: None,
    });
    function.ensure_facts();
    assert!(!optimize::natural_loops(&function.graph, &function.facts.cfg).is_empty());
    let fresh = function.facts.fresh(&function.graph);
    optimize::hoist(&mut function, region);
    function.ensure_facts();
    assert!(std::rc::Rc::ptr_eq(
        &fresh,
        &function.facts.fresh(&function.graph)
    ));
    function.invalidate_effects();
    function.ensure_facts();
    assert!(!std::rc::Rc::ptr_eq(
        &fresh,
        &function.facts.fresh(&function.graph)
    ));
}

#[test]
fn trusted_numeric_inputs_do_not_acquire_speculative_entry_guards() {
    let mut function =
        canonicalize_graph(lower_source("function f(x){return x*2;}").unwrap(), true).unwrap();
    call_variants::assume(
        &mut function.graph,
        &vec![(
            crate::jit::variants::Input::Argument(0),
            crate::jit::variants::Fact::Number,
        )],
    );
    let function = plan_canonical(function, true, None).unwrap();
    let function = optimize_planned(
        function,
        Versioning::Speculate {
            fallback: None,
            continuations: true,
        },
        false,
    )
    .unwrap();
    function.graph.verify().unwrap();
    assert!(
        ops(&function.graph)
            .iter()
            .any(|op| matches!(op, Op::EntryAssume(_, guards::Fact::Type(_))))
    );
    assert!(
        !ops(&function.graph)
            .iter()
            .any(|op| matches!(op, Op::IsNumber(_)))
    );
}

#[test]
fn trusted_shapes_expire_after_writes_and_reentry() {
    for source in [
        "function f(object, mutate){let before=object.x; mutate(object);return object.x;}",
        "function f(object){let before=object.x; object.extra=1;return object.x;}",
    ] {
        let mut function = canonicalize_graph(lower_source(source).unwrap(), true).unwrap();
        call_variants::assume(
            &mut function.graph,
            &vec![(
                crate::jit::variants::Input::Argument(0),
                crate::jit::variants::Fact::Shape(10),
            )],
        );
        let reads: Vec<_> = Cfg::new(&function.graph)
            .rpo
            .into_iter()
            .flat_map(|block| function.graph.blocks[block].insts.clone())
            .filter_map(|inst| match function.graph.insts[inst].op {
                Op::Get(object, _) => Some((inst, object)),
                _ => None,
            })
            .collect();
        assert_eq!(reads.len(), 2);
        for (inst, object) in reads {
            let data = function.graph.insts[inst].clone();
            let position = function.graph.blocks[data.block]
                .insts
                .iter()
                .position(|&other| other == inst)
                .unwrap();
            function.graph.insert(
                data.block,
                position,
                data.span,
                Op::IsShape(object, 10),
                true,
            );
        }
        function.invalidate_cfg();
        optimize::reuse_loads(&mut function);
        function.graph.verify().unwrap();
        assert_eq!(
            ops(&function.graph)
                .iter()
                .filter(|op| matches!(op, Op::IsShape(_, 10)))
                .count(),
            1
        );
    }
}

#[test]
fn trusted_input_shapes_remove_redundant_callee_checks() {
    let mut function = canonicalize_graph(
        lower_source("function f(object){return object.x;}").unwrap(),
        true,
    )
    .unwrap();
    let mut repository = specialize::Repository::default();
    for block in Cfg::new(&function.graph).rpo {
        for &inst in &function.graph.blocks[block].insts {
            let data = &function.graph.insts[inst];
            if matches!(data.op, Op::Get(..)) {
                repository.shape_feedback.insert(
                    (
                        crate::jit::profile::Site {
                            source: 0,
                            start: data.span.start,
                            end: data.span.end,
                        },
                        crate::jit::profile::PropertyAccess::Get,
                    ),
                    crate::jit::profile::ShapeAccess::Read {
                        shape: 10,
                        index: 0,
                    },
                );
            }
        }
    }
    call_variants::assume(
        &mut function.graph,
        &vec![(
            crate::jit::variants::Input::Argument(0),
            crate::jit::variants::Fact::Shape(10),
        )],
    );
    let function = plan_canonical(function, true, Some((&repository, 0))).unwrap();
    let function = optimize_planned(
        function,
        Versioning::Speculate {
            fallback: None,
            continuations: true,
        },
        false,
    )
    .unwrap();
    function.graph.verify().unwrap();
    assert!(
        ops(&function.graph)
            .iter()
            .any(|op| matches!(op, Op::ShapedGet(..)))
    );
    assert!(
        !ops(&function.graph)
            .iter()
            .any(|op| matches!(op, Op::IsShape(..)))
    );
}

#[test]
fn caller_shapes_select_useful_polymorphic_callee_contracts() {
    use crate::jit::profile::{PropertyAccess, ShapeAccess, Site};
    use crate::jit::variants::{Fact as InputFact, Input};
    for (source, useful) in [
        ("function f(object){return object.x;}", true),
        ("function f(object){return 1;}", false),
        (
            "function f(object, callback){callback();return object.x;}",
            false,
        ),
    ] {
        let function = canonicalize_graph(lower_source(source).unwrap(), true).unwrap();
        let mut repository = specialize::Repository::default();
        for block in Cfg::new(&function.graph).rpo {
            for &inst in &function.graph.blocks[block].insts {
                let data = &function.graph.insts[inst];
                if matches!(data.op, Op::Get(..)) {
                    repository.shape_feedback.insert(
                        (
                            Site {
                                source: 0,
                                start: data.span.start,
                                end: data.span.end,
                            },
                            PropertyAccess::Get,
                        ),
                        ShapeAccess::PolymorphicRead {
                            entries: [(10, 0), (20, 1), (0, 0), (0, 0)],
                            len: 2,
                        },
                    );
                }
            }
        }
        assert!(
            call_variants::requirements(
                canonicalize_graph(lower_source(source).unwrap(), true).unwrap(),
                &repository,
                0
            )
            .unwrap()
            .is_empty()
        );
        for shape in [10, 20, 30] {
            let offered = vec![(Input::Argument(0), InputFact::Shape(shape))];
            let contract = call_variants::contextual_requirements(
                canonicalize_graph(lower_source(source).unwrap(), true).unwrap(),
                &repository,
                0,
                &offered,
            )
            .unwrap();
            assert_eq!(
                contract,
                if useful && shape != 30 {
                    offered
                } else {
                    Vec::new()
                },
                "{source}, shape {shape}"
            );
            if contract.is_empty() {
                continue;
            }
            let mut specialized = FunctionState::from_graph(function.graph.clone());
            call_variants::assume(&mut specialized.graph, &contract);
            let specialized = plan_canonical(specialized, true, Some((&repository, 0))).unwrap();
            let specialized = optimize_planned(
                specialized,
                Versioning::Speculate {
                    fallback: None,
                    continuations: true,
                },
                false,
            )
            .unwrap();
            assert!(ops(&specialized.graph).iter().any(
                |op| matches!(op, Op::ShapedGet(_, _, index) if *index == u32::from(shape == 20))
            ));
            assert!(
                !ops(&specialized.graph)
                    .iter()
                    .any(|op| matches!(op, Op::IsShape(..)))
            );
        }
    }
}

#[test]
fn call_variant_contracts_only_lift_entry_guards() {
    use crate::jit::variants::{Fact as InputFact, Input};
    for (source, expected) in [
        (
            "function f(x){return x*2;}",
            vec![(Input::Argument(0), InputFact::Number)],
        ),
        ("function f(x){return x;}", vec![]),
        ("function f(x, callback){callback();return x*2;}", vec![]),
    ] {
        let function = canonicalize_graph(lower_source(source).unwrap(), true).unwrap();
        assert_eq!(
            call_variants::requirements(function, &specialize::Repository::default(), 0).unwrap(),
            expected,
            "{source}"
        );
    }
}

#[test]
fn call_variant_contracts_do_not_lift_shape_guards_after_reentry() {
    use crate::jit::profile::{PropertyAccess, ShapeAccess, Site};
    use crate::jit::variants::{Fact as InputFact, Input};
    for (source, expected) in [
        (
            "function f(object){return object.x;}",
            vec![(Input::Argument(0), InputFact::Shape(10))],
        ),
        (
            "function f(object, callback){callback();return object.x;}",
            vec![],
        ),
        (
            "function f(flag, object){if(flag)return object.x;return 0;}",
            vec![],
        ),
    ] {
        let function = canonicalize_graph(lower_source(source).unwrap(), true).unwrap();
        let mut repository = specialize::Repository::default();
        for block in Cfg::new(&function.graph).rpo {
            for &inst in &function.graph.blocks[block].insts {
                let data = &function.graph.insts[inst];
                if matches!(data.op, Op::Get(..)) {
                    repository.shape_feedback.insert(
                        (
                            Site {
                                source: 0,
                                start: data.span.start,
                                end: data.span.end,
                            },
                            PropertyAccess::Get,
                        ),
                        ShapeAccess::Read {
                            shape: 10,
                            index: 0,
                        },
                    );
                }
            }
        }
        assert_eq!(
            call_variants::requirements(function, &repository, 0).unwrap(),
            expected,
            "{source}"
        );
    }
}

#[test]
fn call_variant_contracts_respect_disabled_numeric_speculation() {
    use crate::jit::profile::{ReadKind, ReadType, Site};
    let function =
        canonicalize_graph(lower_source("function f(x){return x*2;}").unwrap(), true).unwrap();
    let mut repository = specialize::Repository::default();
    for block in Cfg::new(&function.graph).rpo {
        for &inst in &function.graph.blocks[block].insts {
            let data = &function.graph.insts[inst];
            if matches!(data.op, Op::Argument(_)) {
                repository.read_feedback.insert(
                    (
                        Site {
                            source: 0,
                            start: data.span.start,
                            end: data.span.end,
                        },
                        ReadKind::Number,
                    ),
                    ReadType::Unknown,
                );
            }
        }
    }
    assert!(
        call_variants::requirements(function, &repository, 0)
            .unwrap()
            .is_empty()
    );
}

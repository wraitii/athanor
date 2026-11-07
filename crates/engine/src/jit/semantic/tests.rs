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

fn analyze_source(source: &str) -> Analysis {
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
        let instance = graph.append(block, span, Op::NewObject, true).1.unwrap();
        let object_return = matches!(returned, Op::NewObject);
        let unknown_return = matches!(returned, Op::Argument(_));
        let returned = graph.append(block, span, returned, true).1.unwrap();
        let result = graph
            .append(block, span, Op::ConstructorResult(returned, instance), true)
            .1
            .unwrap();
        graph.blocks[block].terminator = Some(Terminator {
            source_name: None,
            span,
            kind: TerminatorKind::Return(result),
        });
        assert_eq!(Facts::compute(&graph).of(result), Types::OBJECT);
        optimize::simplify(&mut graph);
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
        let value = graph.append(entry, span, Op::Argument(0), true).1.unwrap();
        let (op, mask) = match kind {
            0 => (Op::IsUndefined(value), Types::UNDEFINED),
            1 => (Op::IsNullish(value), Types::UNDEFINED.union(Types::NULL)),
            2 => (Op::IsString(value), Types::STRING),
            3 => (Op::IsNumber(value), Types::NUMBER),
            _ => (Op::IsUninitialized(value), Types::UNINITIALIZED),
        };
        let condition = graph.append(entry, span, op, true).1.unwrap();
        let yes = graph.blocks.push(BlockData::default());
        let no = graph.blocks.push(BlockData::default());
        let join = graph.blocks.push(BlockData::default());
        let positive = graph.add_param(yes);
        let negative = graph.add_param(no);
        let merged = graph.add_param(join);
        graph.blocks[entry].terminator = Some(Terminator {
            source_name: None,
            span,
            kind: TerminatorKind::Branch(
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
        });
        for (block, arg) in [(yes, positive), (no, negative)] {
            graph.blocks[block].terminator = Some(Terminator {
                source_name: None,
                span,
                kind: TerminatorKind::Jump(Target {
                    block: join,
                    args: vec![arg],
                }),
            });
        }
        graph.blocks[join].terminator = Some(Terminator {
            source_name: None,
            span,
            kind: TerminatorKind::Return(merged),
        });
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
    let mut graph = canonicalize(graph, true).unwrap().graph;
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
                    crate::jit::planner::Site {
                        source: 0,
                        start: data.span.start,
                        end: data.span.end,
                    },
                    crate::jit::planner::ReadKind::Receiver,
                ),
                crate::jit::planner::ReadType::Array,
            );
        }
    }
    let reads = reads::plan(&graph, &bodies, 0);
    reads::apply(&mut graph, &reads, &mut Vec::new()).unwrap();
    let blocks = graph.blocks.len();
    comparisons::expose(&mut graph);
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
    let mut graph = canonicalize(graph, true).unwrap().graph;
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
                    crate::jit::planner::Site {
                        source: 0,
                        start: data.span.start,
                        end: data.span.end,
                    },
                    crate::jit::planner::ReadKind::Receiver,
                ),
                crate::jit::planner::ReadType::Array,
            );
        }
    }
    let reads = reads::plan(&graph, &bodies, 0);
    reads::apply(&mut graph, &reads, &mut Vec::new()).unwrap();
    comparisons::expose(&mut graph);
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
    optimize::simplify(&mut graph);
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
    let mut graph = canonicalize(graph, true).unwrap().graph;
    comparisons::expose(&mut graph);
    assert!(ops(&graph).iter().any(|op| matches!(op, Op::IsNumber(_))));
    for data in graph.insts.values_mut() {
        if matches!(data.op, Op::Argument(0)) {
            data.op = Op::Constant(Constant::Boolean(true));
        }
    }
    optimize::simplify(&mut graph);
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
    let limit = graph
        .insts
        .values()
        .find_map(|data| {
            matches!(data.op, Op::Argument(0))
                .then_some(data.result)
                .flatten()
        })
        .unwrap();
    assert_eq!(
        ops(graph)
            .into_iter()
            .filter(|op| matches!(op, Op::IsNumber(value) if *value == limit))
            .count(),
        1
    );
}

#[test]
fn comparison_checks_do_not_cross_non_dominating_branches() {
    let graph = lower_source("function f(x,y,flag){if(flag)return x<y;return x>y;}").unwrap();
    let mut graph = canonicalize(graph, true).unwrap().graph;
    comparisons::expose(&mut graph);
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
    let mut canonical = canonicalize(graph, true).unwrap();
    for data in canonical.graph.insts.values_mut() {
        if let Op::GetIndex(object, key) = data.op {
            data.op = Op::IndexedGet(object, key);
        }
    }
    let mut viewed = canonical.graph.clone();
    optimize::views(&mut viewed, &mut Vec::new());
    assert!(
        viewed
            .views
            .keys()
            .any(|&inst| matches!(viewed.insts[inst].op, Op::IndexedGet(..)))
    );
    let analysis = optimize_canonical(canonical, true, None).unwrap();
    assert!(
        analysis
            .notes
            .iter()
            .any(|note| note.contains("get_indexed") && note.contains("reuses"))
    );
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
        let canonical = canonicalize(lower_source(&source).unwrap(), true).unwrap();
        let mut repository = specialize::Repository::default();
        for data in canonical.graph.insts.values() {
            if !matches!(data.op, Op::GetIndex(..)) {
                continue;
            }
            let text = &source[data.span.start as usize..data.span.end as usize];
            repository.read_feedback.insert(
                (
                    crate::jit::planner::Site {
                        source: 0,
                        start: data.span.start,
                        end: data.span.end,
                    },
                    crate::jit::planner::ReadKind::Receiver,
                ),
                if text == "height[i&1]" {
                    crate::jit::planner::ReadType::Array
                } else {
                    crate::jit::planner::ReadType::Typed(crate::heap::TypedKind::Float32)
                },
            );
        }
        let analysis = optimize_canonical(canonical, true, Some((&repository, 0))).unwrap();
        let rows: Vec<_> = analysis
            .graph
            .insts
            .values()
            .filter(|data| {
                matches!(data.op, Op::IndexedGet(..))
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

#[test]
fn typed_loop_reads_are_reused_before_inner_loop_versioning() {
    let graph = lower_source("function f(grid,width){let members=0;for(let y=0;y<2;++y){const indexY=y*width;for(let x=0;x<2;++x){if(grid[indexY+x])for(let i=0;i<16;++i)if(grid[indexY+x]&(1<<i))++members;}}return members;}").unwrap();
    let canonical = canonicalize(graph, true).unwrap();
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
            crate::jit::planner::Site {
                source: 0,
                start: read.span.start,
                end: read.span.end,
            },
            crate::jit::planner::ReadKind::Receiver,
        ),
        crate::jit::planner::ReadType::Typed(crate::heap::TypedKind::Uint16),
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
        3
    );
    assert!(!ops.iter().any(|op| matches!(op, Op::CallNative(..))));
}

#[test]
fn integer_coercions_are_reused_and_hoisted_out_of_bit_loops() {
    let analysis = analyze_source(
        "function f(grid) { let sum=0; for(let x=0;x<grid.length;++x) { const word=grid[x]; for(let i=0;i<16;++i) { sum+=word&(1<<i); sum+=word^i; } } return sum; }",
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
    let canonical = canonicalize(graph, true).unwrap();
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
    assert!(versioned.iter().any(|op| matches!(op, Op::Call { .. })));
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
    let graph = canonicalize(
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

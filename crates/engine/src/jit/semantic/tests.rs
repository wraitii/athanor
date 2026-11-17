use super::cursor::Insert;
use super::js::comparisons;
use super::runtime::{arrays as array_checks, numeric, properties, storage, typed, views};
use super::specialization::{
    call_variants, calls, observations, properties as property_accesses, reads, regions, shapes,
    writes,
};
use super::transforms::{conversions, cse, loops, reuse};
use super::variables::Variables;
use super::*;
use crate::FxHashSet;
use crate::jit::layout::ReadLayout;
use oxc_allocator::Allocator;
use oxc_ast::ast::Statement;
use oxc_parser::Parser;
use oxc_semantic::Scoping;
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
        build::lower(function, scoping, &FxHashSet::default())
    })
}

fn analyze_source(source: &str) -> FunctionState {
    with_function(source, |function, scoping| {
        analyze(function, scoping, &FxHashSet::default())
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
fn baseline_keeps_generic_property_accesses() {
    let graph =
        lower_source("function f(a, k, v) { a.x = v; return a.y + a[k] + shared; }").unwrap();
    let function = canonicalize_graph(graph, true).unwrap();
    let planned = inline_and_plan_specialization(function, true, None, 1).unwrap();
    let mut function = specialize_observed(
        planned,
        Versioning::Generic,
        false,
        None,
        None,
        true,
        &mut |_, _| {},
    )
    .unwrap();
    let mut checked = false;
    let mut checked_helpers = false;
    function
        .lower_runtime_observed(true, &mut |stage, function| {
            if stage == "helpers" {
                assert!(
                    !ops(&function.graph)
                        .iter()
                        .any(|op| matches!(op, Op::Get(..)))
                );
                let helper = ops(&function.graph)
                    .into_iter()
                    .find(|op| {
                        matches!(
                            op,
                            Op::HelperCall {
                                helper: crate::jit::helpers::Helper::PropertyGetCached,
                                ..
                            }
                        )
                    })
                    .unwrap();
                assert_eq!(
                    function.graph.contract(helper, &function.facts).result,
                    Ownership::Borrowed
                );
                for helper in [
                    crate::jit::helpers::Helper::PropertySetCached,
                    crate::jit::helpers::Helper::GlobalGetBorrowed,
                ] {
                    assert!(ops(&function.graph).iter().any(|op| matches!(
                        op, Op::HelperCall { helper: selected, .. } if *selected == helper
                    )));
                }
                assert!(
                    !ops(&function.graph).iter().any(|op| matches!(
                        op,
                        Op::Set(..) | Op::ShapedSet(..) | Op::GlobalRead(..)
                    ))
                );
                checked_helpers = true;
            }
            if stage != "property-accesses" {
                return;
            }
            let operations = ops(&function.graph);
            assert!(
                operations
                    .iter()
                    .any(|op| matches!(op, Op::Get(_, name) if name == "y"))
            );
            assert!(
                operations
                    .iter()
                    .any(|op| matches!(op, Op::Set(_, name, _) if name == "x"))
            );
            assert!(operations.iter().any(|op| matches!(op, Op::GetIndex(..))));
            assert!(
                !operations
                    .iter()
                    .any(|op| matches!(op, Op::PropertyCache(..)))
            );
            checked = true;
        })
        .unwrap();
    assert!(checked);
    assert!(checked_helpers);
}

#[test]
fn cold_property_accesses_keep_helpers() {
    for source in [
        "function f(a){return a.x;}",
        "function f(a){return a.length;}",
        "function f(a,v){a.x=v;return {x:v};}",
    ] {
        let graph = lower_source(source).unwrap();
        let mut function = canonicalize_graph(graph, true).unwrap();
        function.graph.cold.extend(function.graph.blocks.keys());
        let blocks = function.graph.blocks.len();
        assert!(!properties::lower(&mut function).unwrap());
        assert_eq!(function.graph.blocks.len(), blocks);
        assert!(!ops(&function.graph).iter().any(|op| matches!(
            op,
            Op::PropertyCache(..) | Op::CachedProperty(..) | Op::FillPropertyStore(..)
        )));
        runtime::helpers::lower(&mut function);
        assert!(
            !ops(&function.graph)
                .iter()
                .any(|op| matches!(op, Op::Get(..) | Op::Set(..) | Op::Define(..)))
        );
        function.graph.verify().unwrap();
    }
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
        assert_eq!(Facts::compute(&graph).of(result), ValueType::OBJECT);
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
            0 => (Op::IsUndefined(value), ValueType::UNDEFINED),
            1 => (
                Op::IsNullish(value),
                ValueType::UNDEFINED.union(ValueType::NULL),
            ),
            2 => (Op::IsString(value), ValueType::STRING),
            3 => (Op::IsNumber(value), ValueType::NUMBER),
            _ => (Op::IsUninitialized(value), ValueType::UNINITIALIZED),
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
            if kind == 4 {
                ValueType::default()
            } else {
                mask
            }
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
                .then_some(data.result())
                .flatten()
        })
        .unwrap();
    let mut bodies = calls::Repository::default();
    for data in graph.insts.values() {
        if matches!(data.op, Op::GetIndex(..)) {
            bodies.read_feedback.insert(
                (
                    crate::jit::profile::Site {
                        inline_context: 0,
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
                .then_some(data.result())
                .flatten()
        })
        .unwrap();
    let mut bodies = calls::Repository::default();
    for data in graph.insts.values() {
        if matches!(data.op, Op::GetIndex(..)) {
            bodies.read_feedback.insert(
                (
                    crate::jit::profile::Site {
                        inline_context: 0,
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
fn array_reads_are_reused_and_probed_without_implicit_views() {
    let graph = lower_source("function f(array){const zero=0;let sum=0;for(let i=0;i<2;++i){sum+=array[zero]+array[zero];}return sum;}").unwrap();
    let mut canonical = canonicalize_graph(graph, true).unwrap();
    for data in canonical.graph.insts.values_mut() {
        if let Op::GetIndex(object, key) = data.op {
            data.op = Op::ArrayGet(object, key);
        }
    }
    let viewed = canonical.graph.clone();
    let mut function = FunctionState::new(viewed, Vec::new(), true);
    views::views(&mut function);
    let viewed = function.graph;
    assert!(
        viewed
            .views
            .keys()
            .all(|&inst| !matches!(viewed.insts[inst].op, Op::ArrayGet(..)))
    );
    let analysis = optimize_canonical(canonical, true, None).unwrap();
    assert!(ops(&analysis.graph).iter().any(|op| matches!(
        op, Op::LowLevel(low_level::Op::Binary { op: low_level::Binary::FloatAdd, left, right }) if left == right
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
        let mut repository = calls::Repository::default();
        for data in canonical.graph.insts.values() {
            if !matches!(data.op, Op::GetIndex(..)) {
                continue;
            }
            let text = &source[data.span.start as usize..data.span.end as usize];
            repository.read_feedback.insert(
                (
                    crate::jit::profile::Site {
                        inline_context: 0,
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

#[test]
fn guard_only_region_borrows_rows_without_entry_assumptions() {
    use crate::jit::profile::{ReadKind, ReadType, Site};

    let source =
        "function f(height){let sum=0;for(let i=0;i<8;++i)sum+=height[i][0]/1;return sum;}";
    let canonical = canonicalize_graph(lower_source(source).unwrap(), true).unwrap();
    let mut repository = calls::Repository::default();
    for data in canonical.graph.insts.values() {
        if !matches!(data.op, Op::GetIndex(..)) {
            continue;
        }
        let text = &source[data.span.start as usize..data.span.end as usize];
        repository.read_feedback.insert(
            (
                Site {
                    inline_context: 0,
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
    let mut bodies = calls::Repository::default();
    bodies.read_feedback.insert(
        (
            crate::jit::profile::Site {
                inline_context: 0,
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
    assert!(analysis.facts.of(returned).within(ValueType::NUMBER));
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
        ("3", ValueType::INT64),
        ("4194304", ValueType::INT64),
        ("4194305", ValueType::NUMBER),
        ("0", ValueType::NUMBER),
        ("-1", ValueType::NUMBER),
        ("0.5", ValueType::NUMBER),
    ] {
        for expression in [format!("(x|0)*{factor}"), format!("{factor}*(x|0)")] {
            let graph = lower_source(&format!("function f(x){{return {expression};}}")).unwrap();
            let facts = Facts::compute(&graph);
            let result = graph
                .insts
                .values()
                .find_map(|data| {
                    matches!(data.op, Op::Binary(BinaryOperator::Multiplication, ..))
                        .then(|| data.result().unwrap())
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
    let converted = conversion.result().unwrap();
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
    let regions = regions::version(
        &mut graph,
        &calls::ConditionalPlan::default(),
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
                    let ValueDef::Result(def, _) = graph.values[value] else {
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
    assert_eq!(mismatch.insts[def].result(), Some(value));
    mismatch.insts[def].results.clear();
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
                    data.op = Op::ArrayGet(object, key);
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
                assert!(facts.of(data.result().unwrap()).within(ValueType::INTEGER));
                steps += 1;
            }
        }
        assert!(steps > 0);
        assert!(
            graph
                .blocks
                .values()
                .flat_map(|block| &block.params)
                .any(|&value| facts.of(value).within(ValueType::INTEGER)
                    && facts.of(value).intersects(ValueType::INT64))
        );
    }
}

#[test]
fn numeric_observations_skip_conditions_and_receivers() {
    use crate::jit::profile::{Observation, ReadKind};

    let source =
        "function f(a) { if (a.flag) {} let x = a.shared; if (x) {} return a.object.value + x; }";
    let mut graph = canonicalize_graph(lower_source(source).unwrap(), true)
        .unwrap()
        .graph;
    observations::plan(&mut graph, None);
    let observed: Vec<_> = ops(&graph)
        .into_iter()
        .filter_map(|op| match op {
            Op::Observe(_, Observation::Read(ReadKind::Number), site) => {
                Some(&source[site.start as usize..site.end as usize])
            }
            _ => None,
        })
        .collect();
    assert!(!observed.contains(&"a.flag"));
    assert!(!observed.contains(&"a.object"));
    assert!(observed.contains(&"a.object.value"));
    assert!(observed.contains(&"a.shared"));
    graph.verify().unwrap();
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
    let mut repository = calls::Repository::default();
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
    assert!(typed::lower(&mut function).unwrap());
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
    reuse::reuse_loads(&mut function);
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
                .of(graph.insts[inst].result().unwrap())
                .within(ValueType::NUMBER)
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
    assert!(typed::lower(&mut function).unwrap());
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
            types.intersects(ValueType::NUMBER) && types.within(ValueType::NUMBER),
            "{source}: {types}"
        );
        let mut lowered = FunctionState::from_graph(graph.clone());
        numeric::operations(&mut lowered);
        lowered.graph.verify().unwrap();
        assert_eq!(Facts::compute(&lowered.graph).of(result), types, "{source}");
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
        facts.of(value).without(ValueType::UNDEFINED)
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
                .then_some((inst, data.result().unwrap()))
        })
        .unwrap();
    let cfg_storage = function.facts.cfg.rpo.as_ptr();
    assert_eq!(function.facts.of(value), ValueType::INT32);
    function.graph.insts[inst].op = Op::Constant(Constant::String("one".into()));
    function.invalidate_values();
    function.ensure_facts();
    assert_eq!(function.facts.of(value), ValueType::STRING);
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
    assert_eq!(function.facts.of(value), ValueType::STRING);
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
    assert!(!loops::natural_loops(&function.graph, &function.facts.cfg).is_empty());
    let fresh = function.facts.memory_analysis(&function.graph);
    regions::hoist(&mut function, region);
    function.ensure_facts();
    assert!(std::rc::Rc::ptr_eq(
        &fresh,
        &function.facts.memory_analysis(&function.graph)
    ));
    function.invalidate_effects();
    function.ensure_facts();
    assert!(!std::rc::Rc::ptr_eq(
        &fresh,
        &function.facts.memory_analysis(&function.graph)
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
    let function = plan_specialization(function, true, None).unwrap();
    let function = lower_planned(
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
        reuse::reuse_loads(&mut function);
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
    let mut repository = calls::Repository::default();
    for block in Cfg::new(&function.graph).rpo {
        for &inst in &function.graph.blocks[block].insts {
            let data = &function.graph.insts[inst];
            if matches!(data.op, Op::Get(..)) {
                repository.shape_feedback.insert(
                    (
                        crate::jit::profile::Site {
                            inline_context: 0,
                            source: 0,
                            start: data.span.start,
                            end: data.span.end,
                        },
                        crate::jit::profile::PropertyAccess::Get,
                    ),
                    crate::jit::profile::ShapeAccess::Read(ReadLayout::own(10, 0)),
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
    let function = plan_specialization(function, true, Some((&repository, 0))).unwrap();
    let function = lower_planned(
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
        let mut repository = calls::Repository::default();
        for block in Cfg::new(&function.graph).rpo {
            for &inst in &function.graph.blocks[block].insts {
                let data = &function.graph.insts[inst];
                if matches!(data.op, Op::Get(..)) {
                    repository.shape_feedback.insert(
                        (
                            Site {
                                inline_context: 0,
                                source: 0,
                                start: data.span.start,
                                end: data.span.end,
                            },
                            PropertyAccess::Get,
                        ),
                        ShapeAccess::PolymorphicRead {
                            entries: [
                                ReadLayout::own(10, 0),
                                ReadLayout::own(20, 1),
                                ReadLayout::default(),
                                ReadLayout::default(),
                            ],
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
            let specialized =
                plan_specialization(specialized, true, Some((&repository, 0))).unwrap();
            let specialized = lower_planned(
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
            call_variants::requirements(function, &calls::Repository::default(), 0).unwrap(),
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
        let mut repository = calls::Repository::default();
        for block in Cfg::new(&function.graph).rpo {
            for &inst in &function.graph.blocks[block].insts {
                let data = &function.graph.insts[inst];
                if matches!(data.op, Op::Get(..)) {
                    repository.shape_feedback.insert(
                        (
                            Site {
                                inline_context: 0,
                                source: 0,
                                start: data.span.start,
                                end: data.span.end,
                            },
                            PropertyAccess::Get,
                        ),
                        ShapeAccess::Read(ReadLayout::own(10, 0)),
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
    let mut repository = calls::Repository::default();
    for block in Cfg::new(&function.graph).rpo {
        for &inst in &function.graph.blocks[block].insts {
            let data = &function.graph.insts[inst];
            if matches!(data.op, Op::Argument(_)) {
                repository.read_feedback.insert(
                    (
                        Site {
                            inline_context: 0,
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

#[test]
fn arithmetic_lowering_preserves_value_contracts_before_legalization() {
    use low_level::Binary as B;

    for (expression, expected) in [
        ("(a|0)+(b|0)", B::Add),
        ("(a|0)-(b|0)", B::Subtract),
        ("(a|0)*2", B::Multiply),
        ("(a|0)/(b|0)", B::FloatDivide),
        ("(+a)+(+b)", B::FloatAdd),
        ("(+a)-(+b)", B::FloatSubtract),
        ("(+a)*(+b)", B::FloatMultiply),
        ("undefined+(+a)", B::FloatAdd),
    ] {
        let graph = lower_source(&format!("function f(a,b){{return {expression};}}")).unwrap();
        let block = graph.entry;
        let TerminatorKind::Return(result) = graph.blocks[block].terminator.as_ref().unwrap().kind
        else {
            panic!()
        };
        let mut function = FunctionState::new(graph, Vec::new(), true);
        assert!(numeric::arithmetic(&mut function), "{expression}");
        let Some(Op::LowLevel(low_level::Op::Binary { op, .. })) =
            function.graph.defining_op(result)
        else {
            panic!("{expression}")
        };
        assert_eq!(
            std::mem::discriminant(op),
            std::mem::discriminant(&expected)
        );
        assert!(!numeric::arithmetic(&mut function));
        function.ensure_facts();
        let types = if expected.is_float() {
            ValueType::NUMBER
        } else {
            ValueType::INTEGER
        };
        assert!(function.facts.of(result).intersects(types));
        assert!(function.facts.of(result).within(types));
        function.graph.verify().unwrap();
        function.lower_runtime().unwrap();
    }

    let graph = lower_source("function f(a,b){return a+b;}").unwrap();
    let mut function = FunctionState::new(graph, Vec::new(), true);
    assert!(!numeric::arithmetic(&mut function));
    assert!(
        ops(&function.graph)
            .iter()
            .any(|op| matches!(op, Op::Binary(BinaryOperator::Addition, ..)))
    );
}

#[test]
fn lowered_comparisons_and_steps_preserve_loop_facts() {
    for (initial, condition, step) in [
        ("0", "i<16", "++i"),
        ("16", "i>0", "--i"),
        ("0", "i<=15", "i=i+1"),
        ("16", "i>=1", "i=i-1"),
    ] {
        let source =
            format!("function f(){{let i={initial};while({condition}){{{step};}}return i;}}");
        let mut function = canonicalize_graph(lower_source(&source).unwrap(), true).unwrap();
        function.ensure_values();
        let steps: Vec<_> = function
            .graph
            .insts
            .values()
            .filter_map(|data| {
                matches!(
                    data.op,
                    Op::Unary(UnaryOp::Increment | UnaryOp::Decrement, _)
                        | Op::Binary(BinaryOperator::Addition | BinaryOperator::Subtraction, ..)
                )
                .then(|| data.result().unwrap())
            })
            .collect();
        assert!(!steps.is_empty());
        for &step in &steps {
            assert_eq!(function.facts.of(step), ValueType::INT32, "{source}");
        }
        assert!(numeric::operations(&mut function));
        assert!(!numeric::operations(&mut function));
        function.ensure_values();
        for step in steps {
            assert_eq!(function.facts.of(step), ValueType::INT32, "{source}");
        }
        assert!(
            ops(&function.graph)
                .iter()
                .any(|op| matches!(op, Op::LowLevel(low_level::Op::IntCompare { .. })))
        );
        function.lower_runtime().unwrap();
    }
}

#[test]
fn numeric_lowering_reuses_guard_conversions_across_calls() {
    let mut graph = lower_source("function f(a, callback){return a;}").unwrap();
    let block = graph.entry;
    let span = graph.span;
    let TerminatorKind::Return(value) = graph.blocks[block].terminator.as_ref().unwrap().kind
    else {
        panic!()
    };
    let success = graph.blocks.push(BlockData::default());
    let fallback = graph.blocks.push(BlockData::default());
    let callback = graph.append_value(block, span, Op::Argument(1));
    let guard = graph.append_value(block, span, Op::IsInteger(value));
    graph.terminate(
        block,
        span,
        TerminatorKind::Branch(guard, Target::new(success), Target::new(fallback)),
    );
    graph.append_value(
        success,
        span,
        Op::Call {
            callee: callback,
            receiver: None,
            args: Vec::new(),
            target: None,
        },
    );
    let integer = graph.append_value(success, span, Op::AsInteger(value));
    let factor = graph.append_value(success, span, Op::Constant(Constant::Number(0.5)));
    let product = graph.append_value(
        success,
        span,
        Op::Binary(BinaryOperator::Multiplication, integer, factor),
    );
    graph.terminate(success, span, TerminatorKind::Return(product));
    graph.terminate(fallback, span, TerminatorKind::Return(value));
    let mut function = FunctionState::new(graph, Vec::new(), true);
    numeric::lower(&mut function);
    reuse::reuse_loads(&mut function);
    function.graph.verify().unwrap();
    function.ensure_values();
    assert_eq!(function.facts.of(integer), ValueType::INT64);
    let graph = &function.graph;
    let Some(Op::IntegerCheck(_, float, _)) = graph.defining_op(guard) else {
        panic!()
    };
    let Some(Op::LowLevel(low_level::Op::Binary {
        op: low_level::Binary::FloatMultiply,
        left,
        ..
    })) = graph.defining_op(product)
    else {
        panic!()
    };
    assert_eq!(left, float);
    assert_eq!(
        ops(graph)
            .iter()
            .filter(|op| matches!(op, Op::SaturatingInteger(_)))
            .count(),
        1
    );
    super::js::simplify(&mut function);
    numeric::relax_checks(&mut function);
    super::js::simplify(&mut function);
    assert!(matches!(
        function.graph.defining_op(guard),
        Some(Op::IsNumber(_))
    ));
    assert!(
        !ops(&function.graph)
            .iter()
            .any(|op| matches!(op, Op::SaturatingInteger(_)))
    );
}

#[test]
fn hoisting_moves_constant_arithmetic_and_lowered_conversions() {
    for operator in [
        BinaryOperator::Multiplication,
        BinaryOperator::Addition,
        BinaryOperator::Subtraction,
    ] {
        let mut graph = lower_source("function f(a){while(a){a=a.next;}return a;}").unwrap();
        let cfg = Cfg::new(&graph);
        let (header, body) = loops::natural_loops(&graph, &cfg).pop().unwrap();
        let preheader = loops::preheader(&graph, &cfg, header, &body).unwrap();
        let block = *body.iter().find(|&&block| block != header).unwrap();
        let span = graph.span;
        let input = graph.append_value(preheader, span, Op::Argument(1));
        let integer = graph.append_value(preheader, span, Op::AsInteger(input));
        let factor = graph.append_value(block, span, Op::Constant(Constant::Number(0.5)));
        let product = graph.append_value(block, span, Op::Binary(operator, integer, factor));
        let mut function = FunctionState::new(graph, Vec::new(), true);
        let region = function.graph.regions.push(Region {
            blocks: function.graph.blocks.keys().collect(),
            fallback: None,
        });
        numeric::lower(&mut function);
        reuse::reuse_loads(&mut function);
        regions::hoist(&mut function, region);
        function.graph.verify().unwrap();
        let graph = &function.graph;
        assert_eq!(graph.def_block(factor), preheader);
        assert_eq!(graph.def_block(product), preheader);
        let Some(Op::LowLevel(low_level::Op::Binary { left, right, .. })) =
            graph.defining_op(product)
        else {
            panic!()
        };
        assert_eq!(graph.def_block(*left), preheader);
        assert_eq!(graph.def_block(*right), preheader);
        assert!(
            graph.blocks[block]
                .insts
                .iter()
                .any(|&inst| matches!(graph.insts[inst].op, Op::Get(..)))
        );
    }
}

#[test]
fn cell_reads_hoist_across_elements_but_not_cell_writes_or_calls() {
    for mutation in 0..3 {
        let mut graph = lower_source("function f(cell,array,condition){return cell;}").unwrap();
        let entry = graph.entry;
        let span = graph.span;
        let cell = graph.append_value(entry, span, Op::Argument(0));
        let array = graph.append_value(entry, span, Op::Argument(1));
        let condition = graph.append_value(entry, span, Op::Argument(2));
        let index = graph.append_value(entry, span, Op::Constant(Constant::Number(0.0)));
        let header = graph.blocks.push(BlockData::default());
        let body = graph.blocks.push(BlockData::default());
        let exit = graph.blocks.push(BlockData::default());
        graph.terminate(entry, span, TerminatorKind::Jump(Target::new(header)));
        let read = graph.append_value(header, span, Op::CellGet(cell));
        graph.terminate(
            header,
            span,
            TerminatorKind::Branch(condition, Target::new(body), Target::new(exit)),
        );
        match mutation {
            0 => {
                graph.append(body, span, Op::ArrayStore(array, index, index), false);
            }
            1 => {
                graph.append(body, span, Op::CellSet(cell, index), false);
            }
            _ => {
                graph.append_value(
                    body,
                    span,
                    Op::Call {
                        callee: condition,
                        receiver: None,
                        args: vec![],
                        target: None,
                    },
                );
            }
        }
        graph.terminate(body, span, TerminatorKind::Jump(Target::new(header)));
        graph.terminate(exit, span, TerminatorKind::Return(read));
        let mut function = FunctionState::new(graph, Vec::new(), true);
        let region = function.graph.regions.push(Region {
            blocks: function.graph.blocks.keys().collect(),
            fallback: None,
        });
        regions::hoist(&mut function, region);
        function.graph.verify().unwrap();
        assert_eq!(
            function.graph.def_block(read),
            if mutation == 0 { entry } else { header }
        );
    }
}

#[test]
fn cell_read_reuse_stops_at_cell_writes_and_reentry() {
    for mutation in 0..3 {
        let mut graph = lower_source("function f(cell,array){return cell;}").unwrap();
        let entry = graph.entry;
        let span = graph.span;
        let cell = graph.append_value(entry, span, Op::Argument(0));
        let array = graph.append_value(entry, span, Op::Argument(1));
        let index = graph.append_value(entry, span, Op::Constant(Constant::Number(0.0)));
        let first = graph.append_value(entry, span, Op::CellGet(cell));
        match mutation {
            0 => {
                graph.append(entry, span, Op::ArrayStore(array, index, index), false);
            }
            1 => {
                graph.append(entry, span, Op::CellSet(cell, index), false);
            }
            _ => {
                graph.append_value(
                    entry,
                    span,
                    Op::Call {
                        callee: array,
                        receiver: None,
                        args: vec![],
                        target: None,
                    },
                );
            }
        }
        let second = graph.append_value(entry, span, Op::CellGet(cell));
        let result = graph.append_value(entry, span, Op::NewArray(vec![first, second]));
        graph.terminate(entry, span, TerminatorKind::Return(result));
        let mut function = FunctionState::new(graph, Vec::new(), true);
        reuse::reuse_loads(&mut function);
        function.graph.verify().unwrap();
        assert_eq!(
            ops(&function.graph)
                .iter()
                .filter(|op| matches!(op, Op::CellGet(_)))
                .count(),
            if mutation == 0 { 1 } else { 2 }
        );
    }
}

#[test]
fn array_kind_reuse_stops_at_calls_and_writes() {
    for call in [false, true] {
        let mut graph = lower_source("function f(a){return a;}").unwrap();
        let block = graph.entry;
        let span = graph.span;
        let object = graph.append_value(block, span, Op::Argument(0));
        graph.append_value(block, span, Op::IsArray(object, super::ArrayLayout::Packed));
        graph.append_value(block, span, Op::IsArray(object, super::ArrayLayout::Holey));
        if call {
            let callback = graph.append_value(block, span, Op::Argument(1));
            graph.append_value(
                block,
                span,
                Op::Call {
                    callee: callback,
                    receiver: None,
                    args: vec![object],
                    target: None,
                },
            );
        } else {
            let index = graph.append_value(block, span, Op::Constant(Constant::Number(100.0)));
            graph.append_value(block, span, Op::SetIndex(object, index, index));
        }
        let result =
            graph.append_value(block, span, Op::IsArray(object, super::ArrayLayout::Packed));
        graph.terminate(block, span, TerminatorKind::Return(result));
        let mut function = FunctionState::new(graph, Vec::new(), true);
        array_checks::lower_array_kinds(&mut function);
        reuse::reuse_loads(&mut function);
        function.graph.verify().unwrap();
        assert_eq!(
            ops(&function.graph)
                .iter()
                .filter(|op| matches!(op, Op::ArrayKind(_)))
                .count(),
            2
        );
    }
}

#[test]
fn array_metadata_uses_shared_loop_views() {
    let mut graph = lower_source("function f(a){while(a){a=a.next;}return a;}").unwrap();
    let cfg = Cfg::new(&graph);
    let (header, body) = loops::natural_loops(&graph, &cfg).pop().unwrap();
    let preheader = loops::preheader(&graph, &cfg, header, &body).unwrap();
    let block = *body.iter().find(|&&block| block != header).unwrap();
    let span = graph.span;
    let object = graph.append_value(preheader, span, Op::Argument(1));
    graph.append_value(block, span, Op::IsArray(object, super::ArrayLayout::Packed));
    graph.append_value(block, span, Op::IsArray(object, super::ArrayLayout::Holey));
    graph.append_value(block, span, Op::IndexedLength(object));
    let mut function = FunctionState::new(graph, Vec::new(), true);
    array_checks::lower_array_kinds(&mut function);
    reuse::reuse_loads(&mut function);
    views::views(&mut function);
    function.graph.verify().unwrap();
    let views: Vec<_> = function
        .graph
        .views
        .iter()
        .filter(|(inst, _)| {
            matches!(
                function.graph.insts[**inst].op,
                Op::ArrayKind(_) | Op::IndexedLength(_)
            )
        })
        .map(|(_, view)| *view)
        .collect();
    assert_eq!(views.len(), 2);
    assert_eq!(views[0], views[1]);
    assert_eq!(function.graph.def_block(views[0]), preheader);
}

#[test]
fn hoisting_distinguishes_pure_unknown_reads_and_ordered_operations() {
    for write in 0..4 {
        let mut graph = lower_source("function f(a){return a;}").unwrap();
        let entry = graph.entry;
        let span = graph.span;
        let input = graph.append_value(entry, span, Op::Argument(0));
        let other = graph.append_value(entry, span, Op::Argument(1));
        let number = graph.append_value(entry, span, Op::Constant(Constant::Number(2.0)));
        let header = graph.blocks.push(BlockData::default());
        let body = graph.blocks.push(BlockData::default());
        let exit = graph.blocks.push(BlockData::default());
        graph.terminate(entry, span, TerminatorKind::Jump(Target::new(header)));
        let ordered = graph.append_value(header, span, Op::ArrayPosition(number));
        let allocated = graph.append_value(header, span, Op::NewObject);
        graph.terminate(
            header,
            span,
            TerminatorKind::Branch(input, Target::new(body), Target::new(exit)),
        );
        let unknown = graph.append_value(
            body,
            span,
            Op::Binary(BinaryOperator::StrictEquality, input, other),
        );
        let pure = graph.append_value(
            body,
            span,
            Op::Binary(BinaryOperator::StrictEquality, number, number),
        );
        let unequal = graph.append_value(
            body,
            span,
            Op::Binary(BinaryOperator::StrictInequality, number, number),
        );
        match write {
            0 => {}
            1 => {
                graph.append_value(body, span, Op::SetIndex(input, number, other));
            }
            2 => {
                graph.append_value(body, span, Op::GlobalWrite("value".into(), other));
            }
            _ => {
                graph.append_value(
                    body,
                    span,
                    Op::Call {
                        callee: other,
                        receiver: None,
                        args: vec![input],
                        target: None,
                    },
                );
            }
        }
        graph.terminate(body, span, TerminatorKind::Jump(Target::new(header)));
        graph.terminate(exit, span, TerminatorKind::Return(input));
        let mut function = FunctionState::new(graph, Vec::new(), true);
        let region = function.graph.regions.push(Region {
            blocks: function.graph.blocks.keys().collect(),
            fallback: None,
        });
        regions::hoist(&mut function, region);
        function.graph.verify().unwrap();
        assert_eq!(
            function.graph.def_block(unknown),
            if write == 0 { entry } else { body }
        );
        assert_eq!(function.graph.def_block(pure), entry);
        assert_eq!(function.graph.def_block(unequal), entry);
        assert_eq!(function.graph.def_block(ordered), header);
        assert_eq!(function.graph.def_block(allocated), header);
    }
}

#[test]
fn continuation_exit_stubs_preserve_cold_paths() {
    for cold_source in [false, true] {
        let mut graph = lower_source("function f(a){return a;}").unwrap();
        let entry = graph.entry;
        let span = graph.span;
        let value = graph.append_value(entry, span, Op::Argument(0));
        let hot = graph.blocks.push(BlockData::default());
        let cold = graph.blocks.push(BlockData::default());
        graph.terminate(
            entry,
            span,
            TerminatorKind::Branch(value, Target::new(hot), Target::new(cold)),
        );
        graph.terminate(hot, span, TerminatorKind::Return(value));
        graph.terminate(cold, span, TerminatorKind::Return(value));
        graph.cold.insert(cold);
        if cold_source {
            graph.cold.insert(entry);
        }
        let shape = continuation::Shape {
            body: FxHashSet::from_iter([entry]),
            exits: [hot, cold]
                .into_iter()
                .map(|block| continuation::Exit {
                    block,
                    outputs: Vec::new(),
                    params: Vec::new(),
                })
                .collect(),
            returns: None,
        };
        let mut result = graph.clone();
        shape.store_exits(&graph, &mut result, entry);
        result.verify().unwrap();
        let targets = result.blocks[entry]
            .terminator
            .as_ref()
            .unwrap()
            .kind
            .targets()
            .collect::<Vec<_>>();
        assert_eq!(result.cold.contains(&targets[0].block), cold_source);
        assert!(result.cold.contains(&targets[1].block));
        for target in targets {
            assert!(
                result.blocks[target.block]
                    .insts
                    .iter()
                    .any(|&inst| matches!(result.insts[inst].op, Op::ContinuationStore(_)))
            );
        }
    }
}

#[test]
fn property_lowering_exposes_named_reads_and_leaves_computed_reads_generic() {
    let source = "function f(o,a,k){return o.x+a.length+a[k];}";
    let graph = canonicalize_graph(lower_source(source).unwrap(), true)
        .unwrap()
        .graph;
    let mut function = FunctionState::new(graph, Vec::new(), true);
    assert!(properties::lower(&mut function).unwrap());
    function.graph.verify().unwrap();
    let operations = ops(&function.graph);
    assert!(!operations.iter().any(|op| matches!(op, Op::Get(..))));
    assert!(
        operations
            .iter()
            .any(|op| matches!(op, Op::IsPropertyCached(..)))
    );
    assert!(
        operations
            .iter()
            .any(|op| matches!(op, Op::CachedProperty(..)))
    );
    assert!(
        operations
            .iter()
            .any(|op| matches!(op, Op::FillPropertyCache(_, _, name) if name == "x"))
    );
    assert!(
        operations
            .iter()
            .any(|op| matches!(op, Op::IndexedLength(..)))
    );
    assert!(operations.iter().any(|op| matches!(op, Op::GetIndex(..))));
    assert!(!operations.iter().any(|op| matches!(op, Op::ArrayLoad(..))));
    assert!(!properties::lower(&mut function).unwrap());
    function.lower_runtime().unwrap();
}

#[test]
fn array_storage_lowering_preserves_planned_ownership() {
    for layout in [ArrayLayout::Packed, ArrayLayout::Holey] {
        for mutate in [false, true] {
            let mutation = if mutate { "a[0]=null;" } else { "" };
            let source = format!(
                "function f(){{const a=[{{n:1}}];const value=a[0];const length=a.length;{mutation}return value.n+length;}}"
            );
            let mut graph = canonicalize_graph(lower_source(&source).unwrap(), true)
                .unwrap()
                .graph;
            let mut load = None;
            for (inst, data) in graph.insts.iter_mut() {
                match &data.op {
                    Op::GetIndex(object, key) => {
                        let result = data.result().unwrap();
                        load = Some((inst, result));
                        data.op = Op::ArrayLoad(*object, *key, layout);
                    }
                    Op::Get(object, name) if name == "length" => {
                        data.op = Op::IndexedLength(*object)
                    }
                    _ => {}
                }
            }
            let (inst, result) = load.unwrap();
            let mut function = FunctionState::new(graph, Vec::new(), true);
            function.lower_typed_storage().unwrap();
            function.ensure_facts();
            let borrows =
                borrow::borrows(&mut function.graph, &function.facts, &mut function.notes);
            assert!(matches!(function.graph.insts[inst].op, Op::ArrayLoad(..)));
            assert_eq!(function.graph.borrowed.contains(&result), !mutate);
            assert!(
                !ops(&function.graph)
                    .iter()
                    .any(|op| matches!(op, Op::IndexedLength(..)))
            );
            function.ensure_facts();
            repr::assign(&mut function.graph, &function.facts);
            boxing::insert(&mut function.graph, &mut function.facts);
            let plan = ownership::plan(&mut function.graph, &mut function.facts, borrows);
            ownership::verify(&function.graph, &function.facts, &plan).unwrap();
            function.graph.ownership = Some(plan);
            let retained = |graph: &Graph| {
                ops(graph)
                    .iter()
                    .any(|op| matches!(op, Op::Retain(value) if *value == result))
            };
            assert_eq!(retained(&function.graph), mutate);
            let refs: Vec<_> = function
                .graph
                .insts
                .iter()
                .filter_map(|(inst, data)| {
                    matches!(data.op, Op::Retain(_) | Op::Release(_) | Op::SlotStore(_))
                        .then_some(inst)
                })
                .collect();
            storage::lower_array_loads(&mut function);
            function.graph.verify().unwrap();
            assert_eq!(function.graph.repr(result), Repr::Boxed);
            assert_eq!(retained(&function.graph), mutate);
            assert_eq!(
                refs,
                function
                    .graph
                    .insts
                    .iter()
                    .filter_map(|(inst, data)| matches!(
                        data.op,
                        Op::Retain(_) | Op::Release(_) | Op::SlotStore(_)
                    )
                    .then_some(inst))
                    .collect::<Vec<_>>()
            );
            assert!(
                !ops(&function.graph)
                    .iter()
                    .any(|op| matches!(op, Op::ArrayLoad(..)))
            );
            let Op::AsBoxed(word) = function.graph.insts[inst].op else {
                panic!("{inst} is not reinterpreted as boxed");
            };
            assert!(matches!(
                (function.graph.defining_op(word), layout),
                (
                    Some(Op::LowLevel(low_level::Op::Load { .. })),
                    ArrayLayout::Packed
                ) | (
                    Some(Op::LowLevel(low_level::Op::Select { .. })),
                    ArrayLayout::Holey
                )
            ));
        }
    }
}

#[test]
fn typed_loads_lower_to_explicit_memory_operations() {
    use crate::heap::TypedKind;
    for (kind, format) in [
        (TypedKind::Uint8, MemoryFormat::U8),
        (TypedKind::Uint16, MemoryFormat::U16),
        (TypedKind::Uint32, MemoryFormat::U32),
        (TypedKind::Int8, MemoryFormat::I8),
        (TypedKind::Int16, MemoryFormat::I16),
        (TypedKind::Int32, MemoryFormat::I32),
        (TypedKind::Float32, MemoryFormat::F32),
        (TypedKind::Float64, MemoryFormat::F64),
    ] {
        let mut graph =
            canonicalize_graph(lower_source("function f(a,k){return a[k];}").unwrap(), true)
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
                element: Some(kind),
                bounded: true,
            },
        );
        let mut function = FunctionState::new(graph, Vec::new(), true);
        assert!(typed::lower_memory(&mut function));
        let graph = function.graph;
        graph.verify().unwrap();
        let Op::MemoryNumber(raw, actual) = graph.insts[inst].op else {
            panic!()
        };
        assert_eq!(actual, format);
        assert!(
            matches!(graph.defining_op(raw), Some(Op::Memory(MemoryOp::IndexedLoad { format: actual, .. })) if *actual == format)
        );
        assert!(
            graph.blocks[graph.insts[inst].block]
                .insts
                .iter()
                .any(|&i| matches!(graph.insts[i].op, Op::KeepAlive(value) if value == object))
        );
        let mut fallback = graph.clone();
        crate::jit::cranelift::split_indexed_loads(&mut fallback);
        fallback.verify().unwrap();
        assert!(
            matches!(fallback.defining_op(raw), Some(Op::Memory(MemoryOp::Load { format: actual, .. })) if *actual == format)
        );
        assert!(matches!(
            graph.defining_op(raw),
            Some(Op::Memory(MemoryOp::IndexedLoad { .. }))
        ));
        let mut invalid = graph.clone();
        let block = invalid.def_block(raw);
        invalid.blocks[block].terminator.as_mut().unwrap().kind = TerminatorKind::Return(raw);
        assert!(invalid.verify().is_err());
    }
}

#[test]
fn runtime_conversions_leave_backend_neutral_operations() {
    use cranelift_codegen::ir::{AbiParam, Signature, types};
    use cranelift_codegen::isa::CallConv;
    let mut signature = Signature::new(CallConv::AppleAarch64);
    signature.params.extend([
        AbiParam::new(types::I64),
        AbiParam::new(types::F64),
        AbiParam::new(types::I32),
    ]);
    signature.returns.extend([
        AbiParam::new(types::I32),
        AbiParam::new(types::I8),
        AbiParam::new(types::I64),
        AbiParam::new(types::I8),
    ]);
    let mut graph = Graph::new_low_level("conversions".into(), Span::new(0, 0), signature);
    let entry = graph.entry;
    let mut builder = low_level::Builder::new(&mut graph);
    builder.switch_to_block(entry);
    builder.append_block_params_for_function_params(entry);
    let ctx = builder.block_params(entry)[0];
    let number = builder.block_params(entry)[1];
    let integer = builder.block_params(entry)[2];
    let wrapped = builder.semantic(Op::Convert(number, Repr::I32), Repr::I32);
    let truth = builder.semantic(Op::Convert(number, Repr::Boolean), Repr::Boolean);
    let wide = builder.semantic(Op::Convert(integer, Repr::I64), Repr::I64);
    let masked = builder.ins().test_mask(integer, 0x80);
    builder.ins().return_(&[wrapped, truth, wide, masked]);
    let mut passes = Vec::new();
    lower_low_level(
        &mut graph,
        ctx,
        CallConv::AppleAarch64,
        types::I64,
        &mut |name, _| {
            passes.push(name.to_owned());
        },
    )
    .unwrap();
    assert_eq!(passes.last().unwrap(), "scalar-cleanup");
    assert!(!ops(&graph).iter().any(|op| matches!(op,
        Op::Convert(value, Repr::I32 | Repr::Boolean) if *value == number
    )));
    assert!(ops(&graph).iter().any(|op| matches!(
        op,
        Op::LowLevel(low_level::Op::Unary {
            op: low_level::Unary::SignedSaturatingFromFloat,
            ..
        })
    )));
    assert!(
        ops(&graph)
            .iter()
            .any(|op| matches!(op, Op::LowLevel(low_level::Op::FloatCompare { .. })))
    );
    let mut fallback = graph.clone();
    crate::jit::cranelift::prepare(&mut fallback, &mut |_, _| {}).unwrap();
    assert!(matches!(graph.defining_op(wide), Some(Op::Convert(..))));
    assert!(matches!(
        graph.defining_op(masked),
        Some(Op::LowLevel(low_level::Op::TestMask { .. }))
    ));
    assert!(!ops(&fallback).iter().any(|op| matches!(
        op,
        Op::Convert(..) | Op::LowLevel(low_level::Op::TestMask { .. })
    )));
}

#[test]
fn low_level_variables_forward_through_chains_and_loop() {
    use cranelift_codegen::ir::{AbiParam, Signature, types};
    use cranelift_codegen::isa::CallConv;
    let mut signature = Signature::new(CallConv::SystemV);
    signature
        .params
        .extend([AbiParam::new(types::I64), AbiParam::new(types::I8)]);
    signature
        .returns
        .extend([AbiParam::new(types::I64), AbiParam::new(types::I64)]);
    let mut graph = Graph::new_low_level("chain-loop".into(), Span::new(0, 0), signature);
    let entry = graph.entry;
    let span = graph.span;
    let mut builder = low_level::Builder::new(&mut graph);
    builder.switch_to_block(entry);
    builder.append_block_params_for_function_params(entry);
    let input = builder.block_params(entry)[0];
    let condition = builder.block_params(entry)[1];
    let first = builder.create_block();
    let second = builder.create_block();
    let header = builder.create_block();
    let body = builder.create_block();
    let backedge = builder.create_block();
    let exit = builder.create_block();
    let mut variables = Variables::default();
    let variable0 = variables.declare(builder.graph, input, super::Repr::Word);
    let variable1 = variables.declare(builder.graph, input, super::Repr::Word);
    builder.terminate(entry, span, TerminatorKind::Jump(Target::new(first)));
    builder.terminate(first, span, TerminatorKind::Jump(Target::new(second)));
    builder.terminate(second, span, TerminatorKind::Jump(Target::new(header)));
    let invariant = variables.read(builder.graph, header, span, variable0);
    let induction = variables.read(builder.graph, header, span, variable1);
    builder.terminate(
        header,
        span,
        TerminatorKind::Branch(condition, Target::new(body), Target::new(exit)),
    );
    builder.terminate(body, span, TerminatorKind::Jump(Target::new(backedge)));
    let prior = variables.read(builder.graph, backedge, span, variable1);
    let one = builder.value(
        backedge,
        span,
        low_level::Op::Integer {
            ty: types::I64,
            bits: 1,
        },
        types::I64,
    );
    let next = builder.value(
        backedge,
        span,
        low_level::Op::Binary {
            op: low_level::Binary::Add,
            left: prior,
            right: one,
        },
        types::I64,
    );
    variables.define(builder.graph, backedge, variable1, next);
    builder.terminate(backedge, span, TerminatorKind::Jump(Target::new(header)));
    builder.terminate(
        exit,
        span,
        TerminatorKind::ReturnValues(vec![invariant, induction]),
    );
    variables.finish(builder.graph).unwrap();
    graph.verify().unwrap();
    assert_eq!(graph.blocks[entry].params.len(), 2);
    assert_eq!(graph.blocks[header].params.len(), 1);
    for block in [first, second, body, backedge, exit] {
        assert!(graph.blocks[block].params.is_empty());
    }
    let result = &graph.blocks[exit].terminator.as_ref().unwrap().kind;
    assert!(
        matches!(result, TerminatorKind::ReturnValues(values) if values == &[input, graph.blocks[header].params[0]])
    );
}

#[test]
fn low_level_variables_resolve_diamond_and_loop() {
    let mut graph = lower_source("function f(a){return a;}").unwrap();
    let entry = graph.entry;
    let span = graph.span;
    let initial = match graph.blocks[entry].terminator.as_ref().unwrap().kind {
        TerminatorKind::Return(value) => value,
        _ => panic!(),
    };
    let left = graph.blocks.push(BlockData::default());
    let right = graph.blocks.push(BlockData::default());
    let join = graph.blocks.push(BlockData::default());
    let body = graph.blocks.push(BlockData::default());
    let exit = graph.blocks.push(BlockData::default());
    let changed = graph.append_value(left, span, Op::Constant(Constant::Number(7.0)));
    graph.terminate(
        entry,
        span,
        TerminatorKind::Branch(initial, Target::new(left), Target::new(right)),
    );
    graph.terminate(left, span, TerminatorKind::Jump(Target::new(join)));
    graph.terminate(right, span, TerminatorKind::Jump(Target::new(join)));
    graph.terminate(
        join,
        span,
        TerminatorKind::Branch(initial, Target::new(body), Target::new(exit)),
    );
    graph.terminate(body, span, TerminatorKind::Jump(Target::new(join)));
    let mut variables = Variables::default();
    let variable = variables.declare(&mut graph, initial, super::Repr::Boxed);
    variables.define(&mut graph, left, variable, changed);
    let joined = variables.read(&mut graph, join, span, variable);
    variables.define(&mut graph, body, variable, joined);
    let result = variables.read(&mut graph, exit, span, variable);
    graph.terminate(exit, span, TerminatorKind::Return(result));
    variables.finish(&mut graph).unwrap();
    let param = graph.blocks[join].params[0];
    assert_eq!(graph.blocks[join].params.len(), 1);
    assert!(
        graph.blocks[left]
            .terminator
            .as_ref()
            .unwrap()
            .kind
            .targets()
            .any(|target| target.args == [changed])
    );
    assert!(
        graph.blocks[right]
            .terminator
            .as_ref()
            .unwrap()
            .kind
            .targets()
            .any(|target| target.args == [initial])
    );
    assert!(
        graph.blocks[body]
            .terminator
            .as_ref()
            .unwrap()
            .kind
            .targets()
            .any(|target| target.args == [param])
    );
    assert!(
        matches!(graph.blocks[exit].terminator.as_ref().unwrap().kind, TerminatorKind::Return(value) if value == param)
    );
    assert_ne!(joined, param);
    graph.verify().unwrap();
}

#[test]
fn low_level_call_results_keep_types_and_operands() {
    use cranelift_codegen::ir::types;
    let mut graph = lower_source("function f(a){return a;}").unwrap();
    let block = graph.entry;
    let span = graph.span;
    let arg = match graph.blocks[block].terminator.as_ref().unwrap().kind {
        TerminatorKind::Return(value) => value,
        _ => panic!(),
    };
    let mut signature =
        cranelift_codegen::ir::Signature::new(cranelift_codegen::isa::CallConv::SystemV);
    signature
        .params
        .push(cranelift_codegen::ir::AbiParam::new(types::I64));
    signature.returns.extend([
        cranelift_codegen::ir::AbiParam::new(types::I64),
        cranelift_codegen::ir::AbiParam::new(types::I8),
    ]);
    let mut builder = low_level::Builder::new(&mut graph);
    let results = builder.call(
        block,
        span,
        low_level::CallTarget::Indirect(arg),
        signature,
        vec![arg],
    );
    Variables::default().finish(&mut graph).unwrap();
    let [result, status] = results[..] else {
        panic!()
    };
    let ValueDef::Result(call, 0) = graph.values[result] else {
        panic!()
    };
    assert_eq!(graph.values[status], ValueDef::Result(call, 1));
    assert_eq!(
        graph.insts[call].op.operands_iter().collect::<Vec<_>>(),
        [arg, arg]
    );
    assert_eq!(graph.reprs[result].map(Repr::machine), Some(types::I64));
    assert_eq!(graph.reprs[status].map(Repr::machine), Some(types::I8));
    graph.verify().unwrap();
    let facts = Facts::compute(&graph);
    assert_eq!(facts.of(result), ValueType::default());
    assert_eq!(facts.of(status), ValueType::default());
    let cfg = Cfg::new(&graph);
    let _ = boundary::liveness(&graph, &cfg.rpo);
    assert!(
        graph
            .listing(&facts)
            .lines()
            .iter()
            .any(|(_, _, line)| line.contains("low_level"))
    );
    boundary::compact(&mut graph);
    graph.verify().unwrap();
    let call = graph
        .insts
        .values()
        .find(|data| matches!(data.op, Op::LowLevel(low_level::Op::Call { .. })))
        .unwrap();
    assert_eq!(call.results.len(), 2);
    let call = graph
        .insts
        .keys()
        .find(|&inst| {
            matches!(
                graph.insts[inst].op,
                Op::LowLevel(low_level::Op::Call { .. })
            )
        })
        .unwrap();
    graph.insts[call].results.pop();
    assert!(graph.verify().is_err());
}

#[test]
fn low_level_ssa_same_target_edges_receive_one_argument() {
    let mut graph = lower_source("function f(a){return a;}").unwrap();
    let entry = graph.entry;
    let span = graph.span;
    let initial = graph.blocks[entry]
        .terminator
        .as_ref()
        .unwrap()
        .kind
        .operands_iter()
        .next()
        .unwrap();
    let join = graph.blocks.push(BlockData::default());
    let body = graph.blocks.push(BlockData::default());
    let exit = graph.blocks.push(BlockData::default());
    let changed = graph.append_value(body, span, Op::Constant(Constant::Number(2.0)));
    graph.terminate(
        entry,
        span,
        TerminatorKind::Branch(initial, Target::new(join), Target::new(join)),
    );
    graph.terminate(
        join,
        span,
        TerminatorKind::Branch(initial, Target::new(body), Target::new(exit)),
    );
    graph.terminate(body, span, TerminatorKind::Jump(Target::new(join)));
    let mut variables = Variables::default();
    let variable = variables.declare(&mut graph, initial, super::Repr::Boxed);
    let value = variables.read(&mut graph, join, span, variable);
    variables.define(&mut graph, body, variable, changed);
    graph.terminate(exit, span, TerminatorKind::Return(value));
    variables.finish(&mut graph).unwrap();
    assert!(
        graph.blocks[entry]
            .terminator
            .as_ref()
            .unwrap()
            .kind
            .targets()
            .all(|edge| edge.args == [initial])
    );
    graph.verify().unwrap();
}

#[test]
fn low_level_graph_emits_executable_cranelift() {
    use cranelift_codegen::ir::{AbiParam, types};
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::{Linkage, Module, default_libcall_names};
    let mut module = JITModule::new(JITBuilder::new(default_libcall_names()).unwrap());
    let mut signature = module.make_signature();
    signature.params.push(AbiParam::new(types::I64));
    signature.returns.push(AbiParam::new(types::I64));
    let mut graph = Graph::new_low_level("test".into(), Span::new(0, 0), signature.clone());
    let entry = graph.entry;
    let span = graph.span;
    let input = graph.add_param(entry);
    graph.reprs[input] = Some(Repr::from_machine(types::I64));
    let header = graph.blocks.push(BlockData::default());
    let body = graph.blocks.push(BlockData::default());
    let exit = graph.blocks.push(BlockData::default());
    let mut builder = low_level::Builder::new(&mut graph);
    builder.set_location(Some(42));
    let source = "lowered-loop.js";
    let source_name = (source.as_ptr() as usize, source.len());
    builder.set_source_name(Some(source_name));
    let zero = builder.value(
        entry,
        span,
        low_level::Op::Integer {
            ty: types::I64,
            bits: 0,
        },
        types::I64,
    );
    let one = builder.value(
        entry,
        span,
        low_level::Op::Integer {
            ty: types::I64,
            bits: 1,
        },
        types::I64,
    );
    let three = builder.value(
        entry,
        span,
        low_level::Op::Integer {
            ty: types::I64,
            bits: 3,
        },
        types::I64,
    );
    let eight = builder.value(
        entry,
        span,
        low_level::Op::Integer {
            ty: types::I64,
            bits: 8,
        },
        types::I64,
    );
    let mut variables = Variables::default();
    let index_variable = variables.declare(builder.graph, zero, super::Repr::Word);
    let sum_variable = variables.declare(builder.graph, zero, super::Repr::Word);
    builder.terminate(entry, span, TerminatorKind::Jump(Target::new(header)));
    let index = variables.read(builder.graph, header, span, index_variable);
    let sum = variables.read(builder.graph, header, span, sum_variable);
    let condition = builder.value(
        header,
        span,
        low_level::Op::IntCompare {
            condition: cranelift_codegen::ir::condcodes::IntCC::UnsignedLessThan,
            left: index,
            right: three,
        },
        types::I8,
    );
    builder.terminate(
        header,
        span,
        TerminatorKind::Branch(condition, Target::new(body), Target::new(exit)),
    );
    let offset = builder.value(
        body,
        span,
        low_level::Op::Binary {
            op: low_level::Binary::Multiply,
            left: index,
            right: eight,
        },
        types::I64,
    );
    let address = builder.value(
        body,
        span,
        low_level::Op::Binary {
            op: low_level::Binary::Add,
            left: input,
            right: offset,
        },
        types::I64,
    );
    let loaded = builder.value(
        body,
        span,
        low_level::Op::Load {
            ty: types::I64,
            address,
            offset: 0,
            flags: cranelift_codegen::ir::MemFlagsData::trusted(),
        },
        types::I64,
    );
    let next_sum = builder.value(
        body,
        span,
        low_level::Op::Binary {
            op: low_level::Binary::Add,
            left: sum,
            right: loaded,
        },
        types::I64,
    );
    let next_index = builder.value(
        body,
        span,
        low_level::Op::Binary {
            op: low_level::Binary::Add,
            left: index,
            right: one,
        },
        types::I64,
    );
    variables.define(builder.graph, body, index_variable, next_index);
    variables.define(builder.graph, body, sum_variable, next_sum);
    builder.terminate(body, span, TerminatorKind::Jump(Target::new(header)));
    builder.terminate(exit, span, TerminatorKind::ReturnValues(vec![sum]));
    variables.finish(builder.graph).unwrap();
    boundary::compact(&mut graph);
    graph.verify().unwrap();
    for block in Cfg::new(&graph).rpo {
        for &inst in &graph.blocks[block].insts {
            assert_eq!(graph.insts[inst].source_name, Some(source_name));
            assert_eq!(graph.insts[inst].emission_location, Some(42));
        }
        let terminator = graph.blocks[block].terminator.as_ref().unwrap();
        assert_eq!(terminator.source_name, Some(source_name));
        assert_eq!(terminator.emission_location, Some(42));
    }
    let id = module
        .declare_function("low_level_graph_test", Linkage::Local, &signature)
        .unwrap();
    let mut context = module.make_context();
    context.func.signature = signature;
    let mut builder_context = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut context.func, &mut builder_context);
    crate::jit::cranelift::emit(&mut builder, &mut module, &graph, None).unwrap();
    builder.seal_all_blocks();
    builder.finalize(module.target_config());
    module.define_function(id, &mut context).unwrap();
    module.finalize_definitions().unwrap();
    let entry: extern "C" fn(*const u64) -> u64 =
        unsafe { std::mem::transmute(module.get_finalized_function(id)) };
    assert_eq!(entry([10, 20, 12].as_ptr()), 42);
}

#[test]
fn low_level_variables_resolve_long_forwarding_chain() {
    use cranelift_codegen::ir::{AbiParam, Signature, types};
    use cranelift_codegen::isa::CallConv;
    let mut signature = Signature::new(CallConv::SystemV);
    signature.params.push(AbiParam::new(types::I64));
    signature.returns.push(AbiParam::new(types::I64));
    let span = Span::new(0, 0);
    let mut graph = Graph::new_low_level("long-chain".into(), span, signature);
    let entry = graph.entry;
    let mut builder = low_level::Builder::new(&mut graph);
    builder.append_block_params_for_function_params(entry);
    let input = builder.block_params(entry)[0];
    let mut variables = Variables::default();
    let variable = variables.declare(builder.graph, input, super::Repr::Word);
    let mut block = entry;
    for _ in 0..10_000 {
        let next = builder.create_block();
        builder.terminate(block, span, TerminatorKind::Jump(Target::new(next)));
        block = next;
    }
    let result = variables.read(builder.graph, block, span, variable);
    builder.terminate(block, span, TerminatorKind::ReturnValues(vec![result]));
    variables.finish(builder.graph).unwrap();
    graph.verify().unwrap();
    assert!(matches!(
        &graph.blocks[block].terminator.as_ref().unwrap().kind,
        TerminatorKind::ReturnValues(values) if values == &[input]
    ));
}

#[test]
fn semantic_constant_cse_preserves_bits_and_representations() {
    use cranelift_codegen::ir::{AbiParam, Signature, types};
    use cranelift_codegen::isa::CallConv;
    let constants = [
        Constant::Undefined,
        Constant::Null,
        Constant::Boolean(false),
        Constant::Boolean(true),
        Constant::Number(0.0),
        Constant::Number(-0.0),
        Constant::Number(7.0),
        Constant::Number(f64::NAN),
        Constant::Number(f64::from_bits(f64::NAN.to_bits() + 1)),
    ];
    let mut signature = Signature::new(CallConv::SystemV);
    signature.returns.extend([AbiParam::new(types::I64); 20]);
    let mut graph = Graph::new_low_level("constants".into(), Span::new(0, 0), signature);
    let entry = graph.entry;
    let mut builder = low_level::Builder::new(&mut graph);
    builder.switch_to_block(entry);
    let mut originals = Vec::new();
    let mut repeated = Vec::new();
    for constant in constants {
        originals.push(builder.semantic(Op::Constant(constant.clone()), Repr::Boxed));
        repeated.push(builder.semantic(Op::Constant(constant), Repr::Boxed));
    }
    let integer = builder.semantic(Op::Constant(Constant::Number(7.0)), Repr::I64);
    let string = builder.semantic(Op::Constant(Constant::String("7".into())), Repr::Boxed);
    let mut results = originals.clone();
    results.extend(repeated);
    results.extend([integer, string]);
    builder.ins().return_(&results);
    let facts = Facts::compute(&graph);
    assert!(cse::eliminate(&mut graph, &facts, None));
    graph.verify().unwrap();
    let TerminatorKind::ReturnValues(values) =
        &graph.blocks[entry].terminator.as_ref().unwrap().kind
    else {
        panic!()
    };
    let mut expected = originals.clone();
    expected.extend(originals);
    expected.extend([integer, string]);
    assert_eq!(*values, expected);
}

#[test]
fn scalar_dce_removes_unused_chains_and_keeps_effects() {
    use cranelift_codegen::ir::{AbiParam, MemFlagsData, Signature, types};
    use cranelift_codegen::isa::CallConv;
    let mut signature = Signature::new(CallConv::SystemV);
    signature.params.push(AbiParam::new(types::I64));
    signature.returns.push(AbiParam::new(types::I64));
    let mut graph = Graph::new_low_level("dead".into(), Span::new(0, 0), signature);
    let entry = graph.entry;
    let mut builder = low_level::Builder::new(&mut graph);
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    let address = builder.block_params(entry)[0];
    let one = builder.ins().iconst(types::I64, 1);
    let loaded = builder
        .ins()
        .load(types::I64, MemFlagsData::trusted(), address, 0);
    let mut unused = loaded;
    for _ in 0..100 {
        unused = builder.ins().iadd(unused, one);
    }
    builder
        .ins()
        .store(MemFlagsData::trusted(), one, address, 0);
    let mut called = Signature::new(CallConv::SystemV);
    called.returns.extend([AbiParam::new(types::I64); 2]);
    let called = builder.import_signature(called);
    builder.ins().call_indirect(called, address, &[]);
    builder.ins().return_(&[address]);
    reuse::lowered(&mut graph);
    graph.verify().unwrap();
    assert_eq!(ops(&graph).len(), 3);
    assert!(
        ops(&graph)
            .iter()
            .any(|op| matches!(op, Op::LowLevel(low_level::Op::Store { .. })))
    );
    assert!(
        ops(&graph)
            .iter()
            .any(|op| matches!(op, Op::LowLevel(low_level::Op::Call { .. })))
    );
}

#[test]
fn scalar_cse_respects_dominance_and_canonicalizes_operands() {
    use cranelift_codegen::ir::condcodes::IntCC;
    use cranelift_codegen::ir::{AbiParam, Signature, types};
    use cranelift_codegen::isa::CallConv;
    let mut signature = Signature::new(CallConv::SystemV);
    signature.params.push(AbiParam::new(types::I64));
    signature.returns.extend([AbiParam::new(types::I64); 2]);
    let mut graph = Graph::new_low_level("cse".into(), Span::new(0, 0), signature);
    let entry = graph.entry;
    let mut builder = low_level::Builder::new(&mut graph);
    builder.append_block_params_for_function_params(entry);
    let left = builder.create_block();
    let right = builder.create_block();
    let join = builder.create_block();
    builder.switch_to_block(entry);
    let input = builder.block_params(entry)[0];
    let one = builder.ins().iconst(types::I64, 1);
    let sum = builder.ins().iadd(input, one);
    let condition = builder.ins().icmp(IntCC::Equal, input, one);
    builder.ins().brif(condition, left, &[], right, &[]);
    let mut products = Vec::new();
    for block in [left, right, join] {
        builder.switch_to_block(block);
        let repeated_one = builder.ins().iconst(types::I64, 1);
        let repeated_sum = builder.ins().iadd(input, repeated_one);
        let product = builder
            .ins()
            .binary(low_level::Binary::Multiply, repeated_sum, input);
        products.push(product);
        if block == join {
            builder.ins().return_(&[repeated_sum, product]);
        } else {
            builder.ins().jump(join, &[]);
        }
    }
    let facts = Facts::compute(&graph);
    assert!(cse::eliminate(&mut graph, &facts, None));
    graph.verify().unwrap();
    assert!(matches!(
        &graph.blocks[join].terminator.as_ref().unwrap().kind,
        TerminatorKind::ReturnValues(values) if values == &[sum, products[2]]
    ));
    for (block, product) in [left, right, join].into_iter().zip(products) {
        let ValueDef::Result(inst, _) = graph.values[product] else {
            panic!()
        };
        assert!(graph.blocks[block].insts.contains(&inst));
        assert!(matches!(
            graph.defining_op(product),
            Some(Op::LowLevel(low_level::Op::Binary { left, .. })) if *left == sum
        ));
    }
}

#[test]
fn scalar_value_numbering_keeps_words_distinct_from_numbers() {
    use cranelift_codegen::ir::{AbiParam, Signature, types};
    use cranelift_codegen::isa::CallConv;
    let mut signature = Signature::new(CallConv::SystemV);
    signature.returns.extend([AbiParam::new(types::I64); 2]);
    let mut graph = Graph::new_low_level("words".into(), Span::new(0, 0), signature);
    let entry = graph.entry;
    let mut builder = low_level::Builder::new(&mut graph);
    builder.switch_to_block(entry);
    let word = builder.ins().iconst(types::I64, 7);
    let number = builder.ins().iconst(types::I64, 7);
    builder.graph.reprs[word] = Some(Repr::Word);
    builder.graph.reprs[number] = Some(Repr::I64);
    builder.ins().return_(&[word, number]);
    reuse::lowered(&mut graph);
    graph.verify().unwrap();
    assert!(
        matches!(&graph.blocks[entry].terminator.as_ref().unwrap().kind,
        TerminatorKind::ReturnValues(values) if values == &[word, number])
    );
}

#[test]
fn lowered_value_numbering_preserves_loads_across_effects() {
    use cranelift_codegen::ir::{AbiParam, MemFlagsData, Signature, types};
    use cranelift_codegen::isa::CallConv;
    for boundary in 0..3 {
        let mut signature = Signature::new(CallConv::SystemV);
        signature
            .params
            .extend([AbiParam::new(types::I64), AbiParam::new(types::I64)]);
        signature.returns.extend([AbiParam::new(types::I64); 6]);
        let mut graph = Graph::new_low_level("reuse".into(), Span::new(0, 0), signature.clone());
        let entry = graph.entry;
        let mut builder = low_level::Builder::new(&mut graph);
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        let address = builder.block_params(entry)[0];
        let input = builder.block_params(entry)[1];
        let first = builder
            .ins()
            .load(types::I64, MemFlagsData::trusted(), address, 0);
        let sum = builder.ins().iadd(address, input);
        let dependent = builder.ins().iadd(first, input);
        if boundary == 1 {
            builder
                .ins()
                .store(MemFlagsData::trusted(), input, address, 0);
        } else if boundary == 2 {
            let signature = builder.import_signature(signature);
            builder
                .ins()
                .call_indirect(signature, address, &[address, input]);
        }
        let second = builder
            .ins()
            .load(types::I64, MemFlagsData::trusted(), address, 0);
        let repeated = builder.ins().iadd(address, input);
        let repeated_dependent = builder.ins().iadd(second, input);
        builder
            .ins()
            .return_(&[first, second, sum, repeated, dependent, repeated_dependent]);
        reuse::lowered(&mut graph);
        graph.verify().unwrap();
        let TerminatorKind::ReturnValues(values) =
            &graph.blocks[entry].terminator.as_ref().unwrap().kind
        else {
            panic!()
        };
        assert_eq!(values[0] == values[1], boundary == 0);
        assert_eq!(values[2], values[3]);
        assert_eq!(values[4] == values[5], boundary == 0);
    }
}

#[test]
fn scalar_cleanup_hoists_conversions_without_speculating_loads() {
    use cranelift_codegen::ir::{AbiParam, MemFlagsData, Signature, types};
    use cranelift_codegen::isa::CallConv;
    let mut signature = Signature::new(CallConv::SystemV);
    signature
        .params
        .extend([AbiParam::new(types::I64), AbiParam::new(types::F64)]);
    signature.returns.push(AbiParam::new(types::I64));
    let mut graph = Graph::new_low_level("hoist".into(), Span::new(0, 0), signature);
    let entry = graph.entry;
    let mut builder = low_level::Builder::new(&mut graph);
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    let address = builder.block_params(entry)[0];
    let input = builder.block_params(entry)[1];
    let header = builder.create_block();
    let body = builder.create_block();
    let exit = builder.create_block();
    builder.ins().jump(header, &[]);
    builder.switch_to_block(header);
    let condition = builder.ins().icmp_imm_s(
        cranelift_codegen::ir::condcodes::IntCC::NotEqual,
        address,
        0,
    );
    builder.ins().brif(condition, body, &[], exit, &[]);
    builder.switch_to_block(body);
    let converted = builder.ins().fcvt_to_sint_sat(types::I64, input);
    let loaded = builder
        .ins()
        .load(types::I64, MemFlagsData::trusted(), address, 0);
    let stored = builder.ins().iadd(converted, loaded);
    builder
        .ins()
        .store(MemFlagsData::trusted(), stored, address, 0);
    builder.ins().jump(header, &[]);
    builder.switch_to_block(exit);
    builder.ins().return_(&[address]);
    reuse::lowered(&mut graph);
    graph.verify().unwrap();
    assert_eq!(graph.def_block(converted), entry);
    assert_eq!(graph.def_block(loaded), body);
}

#[test]
fn conversion_round_trips_require_lossless_representations() {
    for (from, through, boolean, folded) in [
        (Repr::I32, Repr::I64, Some(false), true),
        (Repr::I32, Repr::F64, Some(false), true),
        (Repr::I32, Repr::Boxed, Some(false), true),
        (Repr::Boolean, Repr::Boxed, Some(true), true),
        (Repr::Boolean, Repr::Boxed, Some(false), false),
        (Repr::Boolean, Repr::Boxed, None, false),
        (Repr::I64, Repr::F64, Some(false), false),
        (Repr::I64, Repr::Boxed, Some(false), false),
        (Repr::Boxed, Repr::I32, Some(false), false),
        (Repr::F64, Repr::I64, Some(false), false),
    ] {
        let mut graph = lower_source(if boolean == Some(true) {
            "function f(){return true;}"
        } else {
            "function f(){return 7;}"
        })
        .unwrap();
        let block = graph.entry;
        let TerminatorKind::Return(original) =
            graph.blocks[block].terminator.as_ref().unwrap().kind
        else {
            panic!()
        };
        graph.reprs[original] = Some(from);
        let (_, intermediate) =
            graph.append(block, graph.span, Op::Convert(original, through), true);
        let intermediate = intermediate.unwrap();
        graph.reprs[intermediate] = Some(through);
        let (_, result) = graph.append(block, graph.span, Op::Convert(intermediate, from), true);
        let result = result.unwrap();
        graph.reprs[result] = Some(from);
        graph.blocks[block].terminator.as_mut().unwrap().kind = TerminatorKind::Return(result);
        let mut facts = Facts::compute(&graph);
        if boolean.is_none() {
            facts.set(original, ValueType::default());
        }
        conversions::simplify(&mut graph, &facts);
        assert!(
            matches!(graph.blocks[block].terminator.as_ref().unwrap().kind,
            TerminatorKind::Return(value) if value == if folded { original } else { result }),
            "{from:?} through {through:?}"
        );
        graph.verify().unwrap();
    }
}

#[test]
fn conversion_hoisting_keeps_heap_values_inside_the_loop() {
    let mut graph =
        lower_source("function f(n){let total=0;while(n>0){total++;n--;}return total;}").unwrap();
    let entry = graph.entry;
    let argument = graph.blocks[entry]
        .insts
        .iter()
        .find_map(|&inst| {
            matches!(graph.insts[inst].op, Op::Argument(0))
                .then(|| graph.insts[inst].result().unwrap())
        })
        .unwrap();
    let constant = graph.blocks[entry]
        .insts
        .iter()
        .find_map(|&inst| {
            matches!(graph.insts[inst].op, Op::Constant(Constant::Number(_)))
                .then(|| graph.insts[inst].result().unwrap())
        })
        .unwrap();
    graph.reprs[constant] = Some(Repr::I32);
    graph.reprs[argument] = Some(Repr::Boxed);
    let facts = Facts::compute(&graph);
    let header = loops::natural_loops(&graph, &facts.cfg)[0].0;
    let (scalar, scalar_value) =
        graph.append(header, graph.span, Op::Convert(constant, Repr::F64), true);
    let (unknown, unknown_value) =
        graph.append(header, graph.span, Op::Convert(argument, Repr::I32), true);
    graph.reprs[scalar_value.unwrap()] = Some(Repr::F64);
    graph.reprs[unknown_value.unwrap()] = Some(Repr::I32);
    conversions::hoist(&mut graph, &facts);
    assert_eq!(graph.insts[scalar].block, entry);
    assert_eq!(graph.insts[unknown].block, header);
    graph.verify().unwrap();
}

#[test]
fn cache_requirements_leave_property_accesses_until_lowering() {
    let mut graph = canonicalize_graph(
        lower_source("function f(object){return object.method();}").unwrap(),
        true,
    )
    .unwrap()
    .graph;
    let (block, span, callee) = graph
        .insts
        .values()
        .find_map(|data| {
            matches!(data.op, Op::Get(_, ref name) if name == "method")
                .then(|| (data.block, data.span, data.result().unwrap()))
        })
        .unwrap();
    graph.append_value(
        block,
        span,
        Op::IsCode(callee, cranelift_module::FuncId::from_u32(0)),
    );
    let blocks = graph.blocks.len();
    property_accesses::assume(&mut graph, &shapes::Plan::default());
    assert_eq!(graph.property_accesses.len(), 1);
    assert_eq!(graph.blocks.len(), blocks);
    assert!(
        ops(&graph)
            .iter()
            .any(|op| matches!(op, Op::Get(_, name) if name == "method"))
    );
    assert!(
        ops(&graph)
            .iter()
            .any(|op| matches!(op, Op::Assume(_, guards::Fact::Property(_), _)))
    );
    assert!(!ops(&graph).iter().any(|op| matches!(
        op,
        Op::IsPropertyCached(..) | Op::CachedProperty(..) | Op::FillPropertyCache(..)
    )));
    graph.strip_assumes();
    regions::materialize_properties(&mut graph).unwrap();
    property_accesses::rewrite(&mut graph).unwrap();
    assert!(
        ops(&graph)
            .iter()
            .any(|op| matches!(op, Op::CachedProperty(..)))
    );
    assert!(
        ops(&graph)
            .iter()
            .any(|op| matches!(op, Op::FillPropertyCache(..)))
    );
    graph.verify().unwrap();
}

#[test]
fn cache_entry_guards_preserve_receiver_proofs() {
    for (name, builtins) in [("method", false), ("push", true)] {
        let source = format!("function f(object){{return object.{name};}}");
        let mut graph = canonicalize_graph(lower_source(&source).unwrap(), true)
            .unwrap()
            .graph;
        let (block, span, callee) = graph
            .insts
            .values()
            .find_map(|data| {
                matches!(data.op, Op::Get(..))
                    .then(|| (data.block, data.span, data.result().unwrap()))
            })
            .unwrap();
        graph.append_value(
            block,
            span,
            Op::IsCode(callee, cranelift_module::FuncId::from_u32(0)),
        );
        property_accesses::assume(&mut graph, &shapes::Plan::default());
        let regions =
            regions::version(&mut graph, &Default::default(), Some(1), &mut Vec::new()).unwrap();
        assert!(!regions.is_empty());
        assert_eq!(
            ops(&graph).iter().any(|op| matches!(op, Op::AsObject(..))),
            !builtins
        );
        let blocks = graph.blocks.len();
        regions::materialize_properties(&mut graph).unwrap();
        property_accesses::rewrite(&mut graph).unwrap();
        assert_eq!(graph.blocks.len(), blocks);
        boundary::compact(&mut graph);
        graph.verify().unwrap();
    }
}

#[test]
fn guarded_cache_loads_preserve_their_builtin_mode() {
    for (name, builtins) in [("method", false), ("push", true)] {
        let source = format!("function f(object){{return object.{name};}}");
        let mut graph = canonicalize_graph(lower_source(&source).unwrap(), true)
            .unwrap()
            .graph;
        let (inst, object) = graph
            .insts
            .iter()
            .find_map(|(inst, data)| {
                if let Op::Get(object, _) = data.op {
                    Some((inst, object))
                } else {
                    None
                }
            })
            .unwrap();
        let span = graph.insts[inst].span;
        property_accesses::assume(&mut graph, &shapes::Plan::default());
        graph.strip_assumes();
        let cache = graph.property_accesses[&inst].cache.unwrap();
        let original = graph.insts[inst].result().unwrap();
        let (before, continuation, read) = dispatch::split(&mut graph, inst);
        let fallback_result = graph.insts[inst].result().unwrap();
        assert_ne!(original, fallback_result);
        assert_eq!(graph.values[original], ValueDef::Param(continuation, 0));
        assert_eq!(graph.values[fallback_result], ValueDef::Result(inst, 0));
        assert_eq!(graph.blocks[continuation].params, vec![original]);
        assert!(matches!(
            &graph.blocks[read].terminator.as_ref().unwrap().kind,
            TerminatorKind::Jump(target) if target.args == [fallback_result]
        ));
        let failed = graph.blocks.push(BlockData::default());
        let undefined = graph.append_value(failed, span, Op::Constant(Constant::Undefined));
        graph.terminate(failed, span, TerminatorKind::Return(undefined));
        let check = graph.append_value(before, span, Op::IsPropertyCached(object, cache, builtins));
        graph.terminate(
            before,
            span,
            TerminatorKind::Branch(check, Target::new(read), Target::new(failed)),
        );
        let blocks = graph.blocks.len();
        regions::materialize_properties(&mut graph).unwrap();
        property_accesses::rewrite(&mut graph).unwrap();
        assert_eq!(graph.blocks.len(), blocks);
        assert!(matches!(graph.insts[inst].op, Op::CachedProperty(_, _, mode) if mode == builtins));
        graph.verify().unwrap();
    }
}

#[test]
fn write_lowering_keeps_generic_keys_and_materializes_selected_stores() {
    use crate::heap::TypedKind;
    let graph = canonicalize_graph(
        lower_source("function f(a,k,v){a[k]=v;return v;}").unwrap(),
        true,
    )
    .unwrap()
    .graph;
    let inst = Cfg::new(&graph)
        .rpo
        .iter()
        .flat_map(|&block| &graph.blocks[block].insts)
        .copied()
        .find(|&inst| matches!(graph.insts[inst].op, Op::SetIndex(..)))
        .unwrap();
    for element in [
        None,
        Some(TypedKind::Uint8),
        Some(TypedKind::Uint16),
        Some(TypedKind::Uint32),
        Some(TypedKind::Int8),
        Some(TypedKind::Int16),
        Some(TypedKind::Int32),
        Some(TypedKind::Float32),
        Some(TypedKind::Float64),
    ] {
        let mut graph = graph.clone();
        let mut repository = calls::Repository::default();
        repository.read_feedback.insert(
            (
                crate::jit::profile::Site {
                    inline_context: 0,
                    source: 0,
                    start: graph.insts[inst].span.start,
                    end: graph.insts[inst].span.end,
                },
                crate::jit::profile::ReadKind::Receiver,
            ),
            element.map_or(
                crate::jit::profile::ReadType::Array,
                crate::jit::profile::ReadType::Typed,
            ),
        );
        let planned = reads::plan_stores(&graph, &repository, 0);
        assert_eq!(planned.len(), 1);
        writes::lower(&mut graph, &planned).unwrap();
        assert!(ops(&graph).iter().any(|op| matches!(op, Op::SetIndex(..))));
        assert_eq!(
            ops(&graph)
                .iter()
                .any(|op| matches!(op, Op::ArrayStore(..))),
            element.is_none()
        );
        let mut function = FunctionState::new(graph, Vec::new(), true);
        typed::lower(&mut function).unwrap();
        array_checks::lower_array_kinds(&mut function);
        function.lower_runtime().unwrap();
        assert!(
            !ops(&function.graph)
                .iter()
                .any(|op| matches!(op, Op::ArrayStore(..)))
        );
        assert!(
            ops(&function.graph)
                .iter()
                .any(|op| matches!(op, Op::LowLevel(low_level::Op::Store { .. })))
        );
        if element.is_none() {
            let old_reads = ops(&function.graph).iter().filter(|op| matches!(op,
                Op::LowLevel(low_level::Op::Load { ty, offset: 0, .. }) if *ty == cranelift_codegen::ir::types::I64)).count();
            assert_eq!(old_reads, 1);
        }
    }
    let mut function = FunctionState::new(graph, Vec::new(), true);
    function.lower_runtime().unwrap();
    assert!(
        !ops(&function.graph)
            .iter()
            .any(|op| matches!(op, Op::ArrayStore(..) | Op::CanOverwrite(..)))
    );
}

#[test]
fn property_write_lowering_exposes_addition_and_overwrite() {
    let graph = canonicalize_graph(
        lower_source("function f(o,v){const p={x:v};o.x=p;return o.x;}").unwrap(),
        true,
    )
    .unwrap()
    .graph;
    let mut function = FunctionState::new(graph, Vec::new(), true);
    assert!(properties::lower(&mut function).unwrap());
    let operations = ops(&function.graph);
    assert!(!operations.iter().any(|op| matches!(op, Op::Set(..))));
    assert!(
        operations
            .iter()
            .any(|op| matches!(op, Op::PropertyAdd(..)))
    );
    assert!(
        operations
            .iter()
            .any(|op| matches!(op, Op::PropertyStore(..)))
    );
    assert!(
        operations
            .iter()
            .any(|op| matches!(op, Op::FillPropertyStore(..)))
    );
    assert!(!properties::lower(&mut function).unwrap());
    function.ensure_facts();
    let defines: Vec<_> = Cfg::new(&function.graph)
        .rpo
        .iter()
        .flat_map(|&block| &function.graph.blocks[block].insts)
        .filter(|&&inst| {
            matches!(
                function.graph.insts[inst].op,
                Op::PropertyStore(.., PropertyWrite::Define)
                    | Op::PropertyAdd(.., PropertyWrite::Define)
                    | Op::FillPropertyStore(.., PropertyWrite::Define)
            )
        })
        .copied()
        .collect();
    assert!(!defines.is_empty());
    assert!(
        defines
            .iter()
            .all(|&inst| !function.facts.memory(inst).writes_data())
    );
    function.lower_runtime().unwrap();
    assert!(
        !ops(&function.graph)
            .iter()
            .any(|op| matches!(op, Op::PropertyAdd(..) | Op::PropertyStore(..)))
    );
    let operations = ops(&function.graph);
    assert!(operations.iter().any(|op| matches!(
        op,
        Op::LowLevel(low_level::Op::Reference {
            op: low_level::Reference::Retain,
            ..
        })
    )));
    assert!(operations.iter().any(|op| matches!(
        op,
        Op::LowLevel(low_level::Op::Reference {
            op: low_level::Reference::Release { .. },
            ..
        })
    )));
}

#[test]
fn polymorphic_property_dispatch_is_planned_before_read_rewriting() {
    use crate::jit::profile::{PropertyAccess, ShapeAccess, Site};

    let mut graph = canonicalize_graph(
        lower_source("function f(object){return object.x;}").unwrap(),
        true,
    )
    .unwrap()
    .graph;
    let (read, site) = graph
        .insts
        .iter()
        .find_map(|(inst, data)| {
            matches!(data.op, Op::Get(..)).then_some((
                inst,
                Site {
                    inline_context: 0,
                    source: 0,
                    start: data.span.start,
                    end: data.span.end,
                },
            ))
        })
        .unwrap();
    let mut repository = calls::Repository::default();
    repository.shape_feedback.insert(
        (site, PropertyAccess::Get),
        ShapeAccess::PolymorphicRead {
            entries: [
                ReadLayout::own(10, 0),
                ReadLayout::own(20, 1),
                ReadLayout::default(),
                ReadLayout::default(),
            ],
            len: 2,
        },
    );
    let shapes = shapes::plan(&graph, &repository, 0);
    property_accesses::assume(&mut graph, &shapes);
    let cache = graph.property_accesses[&read].cache.unwrap();
    assert!(ops(&graph).iter().any(|op| matches!(op,
        Op::Assume(_, guards::Fact::Property(value), _) if *value == cache)));
    assert!(!ops(&graph).iter().any(|op| matches!(
        op,
        Op::Assume(_, guards::Fact::Shape(_), _) | Op::IsShape(..)
    )));
    regions::version(&mut graph, &Default::default(), None, &mut Vec::new()).unwrap();
    assert_eq!(
        ops(&graph)
            .iter()
            .filter(|op| matches!(op, Op::Get(..)))
            .count(),
        3
    );
    assert_eq!(
        ops(&graph)
            .iter()
            .filter(|op| matches!(op, Op::IsShape(..)))
            .count(),
        2
    );
    assert!(!ops(&graph).iter().any(|op| matches!(op, Op::ShapedGet(..))));
    graph.verify().unwrap();
    property_accesses::rewrite(&mut graph).unwrap();
    assert_eq!(
        ops(&graph)
            .iter()
            .filter(|op| matches!(op, Op::ShapedGet(..)))
            .count(),
        2
    );
    assert!(!ops(&graph).iter().any(|op| matches!(op, Op::Get(..))));
    assert!(
        ops(&graph)
            .iter()
            .any(|op| matches!(op, Op::FillPropertyCache(..)))
    );
    assert!(graph.property_accesses.is_empty());
    graph.verify().unwrap();
}

#[test]
fn cache_dependencies_follow_versioning_and_compaction() {
    let mut graph = canonicalize_graph(
        lower_source(
            "function f(object,n){let total=0;for(let i=0;i<n;++i)total+=object.x;return total;}",
        )
        .unwrap(),
        true,
    )
    .unwrap()
    .graph;
    property_accesses::assume(&mut graph, &shapes::Plan::default());
    let cache = graph
        .property_accesses
        .values()
        .find_map(|read| read.cache)
        .unwrap();
    assert!(matches!(
        graph.defining_op(cache),
        Some(Op::PropertyCache(..))
    ));
    regions::version(&mut graph, &Default::default(), None, &mut Vec::new()).unwrap();
    graph.verify().unwrap();
    boundary::compact(&mut graph);
    graph.verify().unwrap();
    property_accesses::rewrite(&mut graph).unwrap();
    for op in ops(&graph) {
        let cache = match op {
            Op::IsPropertyCached(_, cache, _)
            | Op::CachedProperty(_, cache, _)
            | Op::FillPropertyCache(_, cache, _)
            | Op::PrimePropertyCache(_, cache) => *cache,
            _ => continue,
        };
        assert!(matches!(
            graph.defining_op(cache),
            Some(Op::PropertyCache(..))
        ));
    }
    graph.verify().unwrap();
}

#[test]
fn cache_guards_hoist_only_when_the_receiver_is_available() {
    for (source, hoisted) in [
        (
            "function f(object,n){let total=0;for(let i=0;i<n;++i)total+=object.x;return total;}",
            true,
        ),
        (
            "function f(objects,n){let total=0;for(let i=0;i<n;++i)total+=objects[i].x;return total;}",
            false,
        ),
    ] {
        let mut graph = canonicalize_graph(lower_source(source).unwrap(), true)
            .unwrap()
            .graph;
        property_accesses::assume(&mut graph, &shapes::Plan::default());
        regions::version(&mut graph, &Default::default(), None, &mut Vec::new()).unwrap();
        property_accesses::rewrite(&mut graph).unwrap();
        let cfg = Cfg::new(&graph);
        let loops = loops::natural_loops(&graph, &cfg);
        let guards: Vec<_> = cfg
            .rpo
            .iter()
            .copied()
            .flat_map(|block| graph.blocks[block].insts.iter().copied())
            .filter(|&inst| matches!(graph.insts[inst].op, Op::IsPropertyCached(..)))
            .collect();
        assert!(!guards.is_empty());
        assert_eq!(
            guards.iter().any(|&inst| !loops
                .iter()
                .any(|(_, body)| body.contains(&graph.insts[inst].block))),
            hoisted,
            "{source}"
        );
        graph.verify().unwrap();
    }
}

#[test]
fn conflicting_property_shapes_remain_alternative_local_guards() {
    use crate::jit::profile::{PropertyAccess, ShapeAccess, Site};

    let mut graph = canonicalize_graph(
        lower_source("function f(object){return [object.x,object.y];}").unwrap(),
        true,
    )
    .unwrap()
    .graph;
    let mut repository = calls::Repository::default();
    for data in graph.insts.values() {
        if let Op::Get(_, name) = &data.op {
            let site = Site {
                inline_context: 0,
                source: 0,
                start: data.span.start,
                end: data.span.end,
            };
            let shape = if name == "x" { 10 } else { 20 };
            repository.shape_feedback.insert(
                (site, PropertyAccess::Get),
                ShapeAccess::Read(ReadLayout::own(shape, 0)),
            );
        }
    }
    let shapes = shapes::plan(&graph, &repository, 0);
    property_accesses::assume(&mut graph, &shapes);
    let mut notes = Vec::new();
    regions::version(&mut graph, &Default::default(), None, &mut notes).unwrap();
    assert!(graph.entry_guards.is_empty());
    assert!(
        notes
            .iter()
            .any(|note| note.contains("conflicting shape requirements"))
    );
    assert_eq!(
        ops(&graph)
            .iter()
            .filter(|op| matches!(op, Op::IsShape(..)))
            .count(),
        2
    );
    property_accesses::rewrite(&mut graph).unwrap();
    assert_eq!(
        ops(&graph)
            .iter()
            .filter(|op| matches!(op, Op::ShapedGet(..)))
            .count(),
        2
    );
    graph.verify().unwrap();
}

#[test]
fn successful_shape_probes_eliminate_only_valid_shape_checks() {
    for mutation in [None, Some(false), Some(true)] {
        let mut graph = canonicalize_graph(
            lower_source("function f(object){return object.x;}").unwrap(),
            true,
        )
        .unwrap()
        .graph;
        let inst = graph
            .insts
            .iter()
            .find_map(|(inst, data)| matches!(data.op, Op::Get(..)).then_some(inst))
            .unwrap();
        let Op::Get(object, _) = graph.insts[inst].op else {
            unreachable!()
        };
        graph.set_op(
            inst,
            Op::ProbeShapeProperty(object, "x".into(), ReadLayout::own(34, 0)),
        );
        let probe = graph.insts[inst].result().unwrap();
        let block = graph.insts[inst].block;
        let span = graph.span;
        let success = graph.blocks.push(BlockData::default());
        let failure = graph.blocks.push(BlockData::default());
        if mutation == Some(false) {
            graph.append(block, span, Op::Set(object, "y".into(), probe), false);
        }
        let failed = graph.append_value(block, span, Op::IsUninitialized(probe));
        graph.terminate(
            block,
            span,
            TerminatorKind::Branch(failed, Target::new(failure), Target::new(success)),
        );
        if mutation == Some(true) {
            graph.append(success, span, Op::Set(object, "y".into(), probe), false);
        }
        let (check, result) = graph.append(success, span, Op::IsShape(object, 34), true);
        graph.terminate(success, span, TerminatorKind::Return(result.unwrap()));
        let (miss_check, result) = graph.append(failure, span, Op::IsShape(object, 34), true);
        graph.terminate(failure, span, TerminatorKind::Return(result.unwrap()));
        let mut function = FunctionState::from_graph(graph);
        reuse::reuse_loads(&mut function);
        assert_eq!(
            matches!(
                function.graph.insts[check].op,
                Op::Constant(Constant::Boolean(true))
            ),
            mutation.is_none()
        );
        assert!(matches!(
            function.graph.insts[miss_check].op,
            Op::IsShape(..)
        ));
        function.graph.verify().unwrap();
    }
}

#[test]
fn property_reads_share_cache_search_and_holder_validation() {
    use crate::jit::helpers::{CACHE_HOLDER_SHAPE_OFFSET, CACHE_LEN_OFFSET};
    use cranelift_codegen::ir::types;

    for name in ["x", "push"] {
        let source = format!("function f(object){{return object.{name};}}");
        let mut function = canonicalize_graph(lower_source(&source).unwrap(), true).unwrap();
        properties::lower(&mut function).unwrap();
        properties::lower_accesses(&mut function).unwrap();
        let operations = ops(&function.graph);
        for offset in [CACHE_LEN_OFFSET, CACHE_HOLDER_SHAPE_OFFSET] {
            assert_eq!(
                count(&operations, |op| matches!(op,
                    Op::LowLevel(low_level::Op::Load { ty, offset: actual, .. })
                        if *ty == types::I32 && *actual == offset)),
                1,
                "{function}"
            );
        }
        assert_eq!(
            count(&operations, |op| matches!(op, Op::PropertyLoad(..))),
            1
        );
        assert_eq!(
            count(&operations, |op| matches!(
                op,
                Op::LowLevel(low_level::Op::IntCompare {
                    condition: cranelift_codegen::ir::condcodes::IntCC::UnsignedGreaterThan,
                    ..
                })
            )),
            1,
            "{function}"
        );
        function.graph.verify().unwrap();
    }
}

#[test]
fn property_access_lowering_preserves_boxed_loads_until_ownership_planning() {
    for shaped in [false, true] {
        let mut graph = canonicalize_graph(
            lower_source("function f(object){return object.x;}").unwrap(),
            true,
        )
        .unwrap()
        .graph;
        if shaped {
            for data in graph.insts.values_mut() {
                if let Op::Get(object, ref name) = data.op {
                    data.op = Op::ShapedGet(object, name.clone(), 0);
                }
            }
        } else {
            property_accesses::assume(&mut graph, &shapes::Plan::default());
            graph.strip_assumes();
            property_accesses::rewrite(&mut graph).unwrap();
        }
        let mut function = FunctionState::from_graph(graph);
        properties::lower_accesses(&mut function).unwrap();
        assert!(!ops(&function.graph).iter().any(|op| matches!(
            op,
            Op::ShapedGet(..)
                | Op::CachedProperty(..)
                | Op::IsPropertyCached(..)
                | Op::FillPropertyCache(..)
        )));
        function.ensure_facts();
        let loads: Vec<_> = Cfg::new(&function.graph)
            .rpo
            .into_iter()
            .flat_map(|block| function.graph.blocks[block].insts.clone())
            .filter(|&inst| matches!(function.graph.insts[inst].op, Op::PropertyLoad(..)))
            .collect();
        assert!(!loads.is_empty());
        for &inst in &loads {
            assert_eq!(
                function
                    .graph
                    .contract(&function.graph.insts[inst].op, &function.facts)
                    .result,
                Ownership::Borrowed
            );
        }
        if !shaped {
            assert!(ops(&function.graph).iter().any(|op| matches!(
                op,
                Op::HelperCall {
                    helper: crate::jit::helpers::Helper::PropertyGetCached,
                    ..
                }
            )));
        }
        function.lower_runtime().unwrap();
        assert!(
            !ops(&function.graph)
                .iter()
                .any(|op| matches!(op, Op::PropertyLoad(..)))
        );
        for inst in loads {
            let Op::AsBoxed(word) = function.graph.insts[inst].op else {
                panic!("{inst} is not reinterpreted as boxed");
            };
            assert!(matches!(
                function.graph.defining_op(word),
                Some(Op::LowLevel(low_level::Op::Load { .. }))
            ));
            assert_eq!(
                function
                    .graph
                    .repr(function.graph.insts[inst].result().unwrap()),
                Repr::Boxed
            );
        }
        function.graph.verify().unwrap();
    }
}

fn optimize_with_shapes(
    source: &str,
    feedback: &[(
        &str,
        crate::jit::profile::PropertyAccess,
        crate::jit::profile::ShapeAccess,
    )],
) -> FunctionState {
    use crate::jit::profile::{PropertyAccess, Site};

    let canonical = canonicalize_graph(lower_source(source).unwrap(), true).unwrap();
    let mut repository = calls::Repository::default();
    for data in canonical.graph.insts.values() {
        let (name, access) = match &data.op {
            Op::Get(_, name) => (name, PropertyAccess::Get),
            Op::Set(_, name, _) => (name, PropertyAccess::Set),
            _ => continue,
        };
        let site = Site {
            inline_context: 0,
            source: 0,
            start: data.span.start,
            end: data.span.end,
        };
        if let Some(&(.., shape)) = feedback
            .iter()
            .find(|&&(property, kind, _)| property == name && kind == access)
        {
            repository.shape_feedback.insert((site, access), shape);
        }
    }
    optimize_canonical(canonical, true, Some((&repository, 0))).unwrap()
}

#[test]
fn related_storage_reads_use_explicit_layout_guards_and_probes() {
    use crate::jit::layout::{ReadLayout, RelatedStorage};
    use crate::jit::profile::{PropertyAccess, ShapeAccess};
    let layout = ReadLayout {
        shape: 10,
        index: 1,
        storage: Some(RelatedStorage {
            identity: 128,
            shape: 20,
        }),
    };
    for source in [
        "function f(o){return o.x;}",
        "function f(o){return o.x+o.x;}",
    ] {
        let function = optimize_with_shapes(
            source,
            &[("x", PropertyAccess::Get, ShapeAccess::Read(layout))],
        );
        assert!(
            ops(&function.graph).iter().any(|op| matches!(op,
            Op::LayoutGet(_, _, known) | Op::ProbeShapeProperty(_, _, known) if *known == layout)),
            "{function}"
        );
        function.graph.verify().unwrap();
    }
}

#[test]
fn nested_read_probes_use_shape_feedback() {
    use crate::jit::profile::{PropertyAccess, ShapeAccess};

    let function = optimize_with_shapes(
        "function f(o){const r={};if(o.section){r.value=o.section.value;r.scaled=o.section.value*o.section.scale;}return r;}",
        &[
            (
                "section",
                PropertyAccess::Get,
                ShapeAccess::Read(ReadLayout::own(10, 0)),
            ),
            (
                "value",
                PropertyAccess::Get,
                ShapeAccess::Read(ReadLayout::own(20, 0)),
            ),
            (
                "scale",
                PropertyAccess::Get,
                ShapeAccess::Read(ReadLayout::own(20, 1)),
            ),
        ],
    );
    for (name, shape, index) in [("section", 10, 0), ("value", 20, 0), ("scale", 20, 1)] {
        assert!(
            ops(&function.graph).iter().any(|op| matches!(op,
            Op::ProbeShapeProperty(_, property, layout)
                if property == name && layout.shape == shape && layout.index == index)),
            "{function}"
        );
        assert!(
            !ops(&function.graph).iter().any(|op| matches!(op,
            Op::ProbeProperty(_, property) if property == name)),
            "{function}"
        );
    }
    function.graph.verify().unwrap();
}

fn loop_ops(graph: &Graph) -> Vec<&Op> {
    let cfg = Cfg::new(graph);
    let bodies: FxHashSet<Block> = loops::natural_loops(graph, &cfg)
        .into_iter()
        .flat_map(|(_, body)| body)
        .filter(|block| !graph.cold.contains(block))
        .collect();
    cfg.rpo
        .iter()
        .filter(|block| bodies.contains(block))
        .flat_map(|&block| &graph.blocks[block].insts)
        .map(|&inst| &graph.insts[inst].op)
        .collect()
}

fn count(ops: &[&Op], predicate: impl Fn(&Op) -> bool) -> usize {
    ops.iter().filter(|op| predicate(op)).count()
}

fn generic_property_access(op: &Op) -> bool {
    matches!(
        op,
        Op::Get(..)
            | Op::Set(..)
            | Op::IsPropertyCached(..)
            | Op::CachedProperty(..)
            | Op::FillPropertyCache(..)
            | Op::PropertyAdd(..)
    )
}

#[test]
fn shaped_overwrites_share_hoisted_guards_with_reads() {
    use crate::jit::profile::{PropertyAccess::*, ShapeAccess::*};

    let function = optimize_with_shapes(
        "function f(o,n){for(let i=0;i<n;++i)o.x=o.x+o.y;return o.x;}",
        &[
            ("x", Get, Read(ReadLayout::own(10, 0))),
            ("y", Get, Read(ReadLayout::own(10, 1))),
            (
                "x",
                Set,
                Overwrite {
                    shape: 10,
                    index: 0,
                },
            ),
        ],
    );
    let graph = &function.graph;
    let body = loop_ops(graph);
    assert!(
        !body.iter().any(|op| generic_property_access(op)),
        "{function}"
    );
    assert_eq!(
        count(&body, |op| matches!(op, Op::IsShape(..))),
        0,
        "{function}"
    );
    assert_eq!(
        count(&body, |op| matches!(op, Op::ShapedGet(..))),
        1,
        "{function}"
    );
    assert_eq!(
        count(&body, |op| matches!(op, Op::PropertyStore(..))),
        1,
        "{function}"
    );
    assert_eq!(
        count(&ops(graph), |op| matches!(op, Op::IsShape(_, 10))),
        1,
        "{function}"
    );
    graph.verify().unwrap();
}

#[test]
fn shaped_overwrites_keep_one_entry_guard_across_reads() {
    use crate::jit::profile::{PropertyAccess::*, ShapeAccess::*};

    let function = optimize_with_shapes(
        "function f(o){o.x=1;o.y=o.x+1;return o.y;}",
        &[
            ("x", Get, Read(ReadLayout::own(10, 0))),
            ("y", Get, Read(ReadLayout::own(10, 1))),
            (
                "x",
                Set,
                Overwrite {
                    shape: 10,
                    index: 0,
                },
            ),
            (
                "y",
                Set,
                Overwrite {
                    shape: 10,
                    index: 1,
                },
            ),
        ],
    );
    let graph = &function.graph;
    let Some(TerminatorKind::Branch(condition, guarded, _)) = graph.blocks[graph.entry]
        .terminator
        .as_ref()
        .map(|t| &t.kind)
    else {
        panic!("{function}")
    };
    assert!(matches!(
        graph.defining_op(*condition),
        Some(Op::IsShape(_, 10))
    ));
    let mut pending = vec![guarded.block];
    let mut seen = FxHashSet::default();
    let mut speculative = Vec::new();
    while let Some(block) = pending.pop() {
        if !seen.insert(block) || graph.cold.contains(&block) {
            continue;
        }
        speculative.extend(
            graph.blocks[block]
                .insts
                .iter()
                .map(|&inst| &graph.insts[inst].op),
        );
        if let Some(terminator) = &graph.blocks[block].terminator {
            pending.extend(terminator.kind.targets().map(|target| target.block));
        }
    }
    assert!(
        !speculative.iter().any(|op| generic_property_access(op)),
        "{function}"
    );
    assert_eq!(
        count(&speculative, |op| matches!(op, Op::IsShape(..))),
        0,
        "{function}"
    );
    assert_eq!(
        count(&speculative, |op| matches!(op, Op::ShapedGet(..))),
        2,
        "{function}"
    );
    assert_eq!(
        count(&speculative, |op| matches!(op, Op::PropertyStore(..))),
        2,
        "{function}"
    );
    graph.verify().unwrap();
}

#[test]
fn property_additions_keep_shape_guards_inside_loops() {
    use crate::jit::profile::{PropertyAccess::*, ShapeAccess::*};

    let function = optimize_with_shapes(
        "function f(o,n){for(let i=0;i<n;++i){o.x=o.x+1;o.z=i;}return o.x;}",
        &[
            ("x", Get, Read(ReadLayout::own(10, 0))),
            (
                "x",
                Set,
                Overwrite {
                    shape: 10,
                    index: 0,
                },
            ),
        ],
    );
    let body = loop_ops(&function.graph);
    assert_eq!(
        count(&body, |op| matches!(op, Op::IsShape(_, 10))),
        1,
        "{function}"
    );
    assert_eq!(
        count(&body, |op| matches!(op, Op::ShapedGet(..))),
        1,
        "{function}"
    );
    assert_eq!(
        count(&body, |op| matches!(op, Op::PropertyAdd(..))),
        1,
        "{function}"
    );
    assert!(
        !body.iter().any(|op| matches!(op, Op::ShapedSet(..))),
        "{function}"
    );
    function.graph.verify().unwrap();
}

#[test]
fn shaped_accesses_on_two_receivers_hoist_both_guards() {
    use crate::jit::profile::{PropertyAccess::*, ShapeAccess::*};

    let function = optimize_with_shapes(
        "function f(a,b,n){for(let i=0;i<n;++i){a.p=b.r;b.s=a.q+a.p;}return b.s;}",
        &[
            ("p", Get, Read(ReadLayout::own(10, 0))),
            ("q", Get, Read(ReadLayout::own(10, 1))),
            (
                "p",
                Set,
                Overwrite {
                    shape: 10,
                    index: 0,
                },
            ),
            ("r", Get, Read(ReadLayout::own(20, 0))),
            ("s", Get, Read(ReadLayout::own(20, 1))),
            (
                "s",
                Set,
                Overwrite {
                    shape: 20,
                    index: 1,
                },
            ),
        ],
    );
    let graph = &function.graph;
    let body = loop_ops(graph);
    assert!(
        !body.iter().any(|op| generic_property_access(op)),
        "{function}"
    );
    assert_eq!(
        count(&body, |op| matches!(op, Op::IsShape(..))),
        0,
        "{function}"
    );
    assert_eq!(
        count(&body, |op| matches!(op, Op::PropertyStore(..))),
        2,
        "{function}"
    );
    let all = ops(graph);
    assert_eq!(
        count(&all, |op| matches!(op, Op::IsShape(_, 10))),
        1,
        "{function}"
    );
    assert_eq!(
        count(&all, |op| matches!(op, Op::IsShape(_, 20))),
        1,
        "{function}"
    );
    graph.verify().unwrap();
}

#[test]
fn block_liveness_matches_instruction_liveness() {
    for source in [
        "function f(a,n){let value=a;for(let i=0;i<n;++i){if(i&1)value=a[i];else value=a[i+1];for(let j=0;j<i;++j){if(j===2)break;value=value[j];}}return value;}",
        "function f(a,n){let value=a;while(n--){if(n===4)return value;if(n&1)continue;value=value.next;}return value;}",
        "function f(a,n){let x=a;do{let y=x.next;if(n&1)x=y;else x=a;--n;}while(n>0);return x;}",
    ] {
        let function = canonicalize_graph(lower_source(source).unwrap(), true).unwrap();
        let graph = &function.graph;
        let cfg = Cfg::new(graph);
        assert_eq!(Cfg::reverse_postorder(graph), cfg.rpo);
        let expected = boundary::liveness(graph, &cfg.rpo);
        let mut uses = SecondaryMap::<Block, liveness::Set>::new();
        let mut defs = SecondaryMap::<Block, liveness::Set>::new();
        for &block in &cfg.rpo {
            let data = &graph.blocks[block];
            let mut live = liveness::Set::default();
            live.extend(data.terminator.as_ref().unwrap().kind.operands_iter());
            for &inst in data.insts.iter().rev() {
                let data = &graph.insts[inst];
                for &result in &data.results {
                    live.remove(&result);
                    defs[block].insert(result);
                }
                live.extend(data.op.operands_iter());
            }
            for &param in &data.params {
                live.remove(&param);
                defs[block].insert(param);
            }
            uses[block] = live;
        }
        let actual = liveness::solve(graph, &cfg, &uses, &defs);
        for &block in &cfg.rpo {
            assert_eq!(
                actual.input[block].iter().collect::<FxHashSet<_>>(),
                expected[block],
                "{block}: {source}"
            );
            let output: FxHashSet<_> = graph
                .successors(block)
                .flat_map(|next| expected[next].iter().copied())
                .collect();
            assert_eq!(
                actual.output[block].iter().collect::<FxHashSet<_>>(),
                output,
                "{block}: {source}"
            );
        }
    }
}

#[test]
fn load_availability_retracts_aliases_after_loop_writes() {
    use cranelift_codegen::ir::{AbiParam, MemFlagsData, Signature, types};
    use cranelift_codegen::isa::CallConv;
    for write in [false, true] {
        let mut signature = Signature::new(CallConv::SystemV);
        signature.params.extend([AbiParam::new(types::I64); 2]);
        signature.returns.extend([AbiParam::new(types::I64); 4]);
        let mut graph = Graph::new_low_level("loop-aliases".into(), Span::new(0, 0), signature);
        let entry = graph.entry;
        let mut builder = low_level::Builder::new(&mut graph);
        builder.append_block_params_for_function_params(entry);
        let header = builder.create_block();
        let body = builder.create_block();
        let exit = builder.create_block();
        builder.switch_to_block(entry);
        let address = builder.block_params(entry)[0];
        let input = builder.block_params(entry)[1];
        let first = builder
            .ins()
            .load(types::I64, MemFlagsData::trusted(), address, 0);
        let nested = builder
            .ins()
            .load(types::I64, MemFlagsData::trusted(), first, 0);
        let condition =
            builder
                .ins()
                .icmp_imm_s(cranelift_codegen::ir::condcodes::IntCC::NotEqual, input, 0);
        builder.ins().jump(header, &[]);
        builder.switch_to_block(header);
        let repeated = builder
            .ins()
            .load(types::I64, MemFlagsData::trusted(), address, 0);
        let repeated_nested = builder
            .ins()
            .load(types::I64, MemFlagsData::trusted(), repeated, 0);
        builder.ins().brif(condition, body, &[], exit, &[]);
        builder.switch_to_block(body);
        if write {
            builder
                .ins()
                .store(MemFlagsData::trusted(), input, address, 0);
        }
        builder.ins().jump(header, &[]);
        builder.switch_to_block(exit);
        let last = builder
            .ins()
            .load(types::I64, MemFlagsData::trusted(), address, 0);
        let last_nested = builder
            .ins()
            .load(types::I64, MemFlagsData::trusted(), last, 0);
        builder
            .ins()
            .return_(&[repeated, repeated_nested, last, last_nested]);
        graph.verify().unwrap();
        let facts = Facts::compute(&graph);
        let solved =
            available::solve(&graph, &facts, &facts.cfg, FxHashSet::default(), false).unwrap();
        assert_eq!(
            solved.canonical(repeated),
            if write { repeated } else { first }
        );
        assert_eq!(
            solved.canonical(repeated_nested),
            if write { repeated_nested } else { nested }
        );
        assert_eq!(solved.canonical(last), solved.canonical(repeated));
        assert_eq!(
            solved.canonical(last_nested),
            solved.canonical(repeated_nested)
        );
    }
}

#[test]
fn ownership_edges_require_a_reference_for_each_parameter() {
    let source = "function f(value, choose){let a,b;if(choose){a=value;b=value;}else{a={};b={};}return [a,b];}";
    let mut graph = canonicalize_graph(lower_source(source).unwrap(), true)
        .unwrap()
        .graph;
    let mut facts = Facts::compute(&graph);
    let borrows = borrow::borrows(&mut graph, &facts, &mut Vec::new());
    let plan = ownership::plan(&mut graph, &mut facts, borrows);
    ownership::verify(&graph, &facts, &plan).unwrap();
    for &block in &facts.cfg.rpo {
        let mut storing = false;
        for &inst in &graph.blocks[block].insts {
            match graph.insts[inst].op {
                Op::ParamStore(..) => storing = true,
                Op::Retain(_) | Op::SlotClear(_) => assert!(!storing),
                _ => {}
            }
        }
    }
    let (block, inst) = facts
        .cfg
        .rpo
        .iter()
        .find_map(|&block| {
            let retained: Vec<_> = graph.blocks[block]
                .insts
                .iter()
                .copied()
                .filter_map(|inst| match graph.insts[inst].op {
                    Op::Retain(value) => Some((inst, value)),
                    _ => None,
                })
                .collect();
            retained
                .iter()
                .find(|(_, value)| retained.iter().filter(|(_, other)| other == value).count() > 1)
                .map(|&(inst, _)| (block, inst))
        })
        .expect("an edge must retain the duplicated argument twice");
    graph.blocks[block].insts.retain(|&other| other != inst);
    assert!(
        ownership::verify(&graph, &facts, &plan)
            .unwrap_err()
            .contains("reference")
    );
}

#[test]
fn ownership_verifier_keeps_borrow_anchors_live() {
    let source = "function f(){const owner={child:{}};return owner.child;}";
    let mut graph = canonicalize_graph(lower_source(source).unwrap(), true)
        .unwrap()
        .graph;
    let (inst, owner, value) = graph
        .insts
        .iter()
        .find_map(|(inst, data)| {
            if let Op::Get(owner, ref name) = data.op
                && name == "child"
            {
                Some((inst, owner, data.result().unwrap()))
            } else {
                None
            }
        })
        .unwrap();
    let mut facts = Facts::compute(&graph);
    let borrows = borrow::borrows(&mut graph, &facts, &mut Vec::new());
    assert!(graph.borrowed.contains(&value));
    let plan = ownership::plan(&mut graph, &mut facts, borrows);
    assert_eq!(plan.holder(owner), ownership::Holder::Slot);
    ownership::verify(&graph, &facts, &plan).unwrap();
    cursor::Cursor::after(&mut graph, inst).add_effect(Op::SlotClear(owner));
    assert!(
        ownership::verify(&graph, &facts, &plan)
            .unwrap_err()
            .contains("clears live")
    );
}

#[test]
fn indexed_borrow_anchors_survive_helper_lowering() {
    let source = "function f(){const owner=[[3]];const row=owner[0];const scratch=[1];return [row,scratch];}";
    let graph = canonicalize_graph(lower_source(source).unwrap(), true)
        .unwrap()
        .graph;
    let (owner, row) = graph
        .insts
        .values()
        .find_map(|data| match data.op {
            Op::GetIndex(owner, _) => Some((owner, data.result().unwrap())),
            _ => None,
        })
        .unwrap();
    let scratch = graph
        .insts
        .values()
        .find(|data| &source[data.span.start as usize..data.span.end as usize] == "[1]")
        .unwrap()
        .result()
        .unwrap();
    let mut function = FunctionState::new(graph, Vec::new(), true);
    function.ensure_facts();
    let borrows = borrow::borrows(&mut function.graph, &function.facts, &mut function.notes);
    assert!(function.graph.borrowed.contains(&row));
    super::runtime::helpers::lower(&mut function);
    assert!(matches!(
        function.graph.defining_op(row),
        Some(Op::HelperCall { .. })
    ));
    let plan = ownership::plan(&mut function.graph, &mut function.facts, borrows);
    assert_eq!(plan.holder(owner), ownership::Holder::Slot);
    assert_eq!(plan.holder(scratch), ownership::Holder::Slot);
    assert_ne!(plan.slots.of(owner), plan.slots.of(scratch));
    ownership::verify(&function.graph, &function.facts, &plan).unwrap();
}

#[test]
fn property_borrows_follow_owning_slots_across_writes() {
    for (read, write, expected) in [
        ("o.x", "o.y=null;", true),
        ("o.x", "o.x=null;", false),
        ("o.x", "p.x=null;", false),
        ("o.x.child", "o.y=null;", true),
        ("o.x.child", "o.x=null;", false),
        ("o.x.child", "o.x.child=null;", false),
        ("o.x", "change(o);", false),
    ] {
        let source = format!("function f(o,p,choose){{let value={read};{write}return value.n;}}");
        let mut graph = canonicalize_graph(lower_source(&source).unwrap(), true)
            .unwrap()
            .graph;
        let value = graph
            .insts
            .values()
            .find_map(|data| {
                matches!(&data.op, Op::Get(_, name) if name == "n")
                    .then(|| data.op.operands_iter().next().unwrap())
            })
            .unwrap();
        let facts = Facts::compute(&graph);
        borrow::borrows(&mut graph, &facts, &mut Vec::new());
        assert_eq!(graph.borrowed.contains(&value), expected, "{source}");
    }
}

#[test]
fn lowered_property_accesses_preserve_borrowed_slots() {
    for setup in ["function f(){let o={x:{n:7},y:null};", "function f(o){"] {
        for (write, expected) in [("o.y=null;", true), ("o.x=null;", false)] {
            let source = format!("{setup}let value=o.x;{write}return value.n;}}");
            let graph = canonicalize_graph(lower_source(&source).unwrap(), true)
                .unwrap()
                .graph;
            let mut function = FunctionState::new(graph, Vec::new(), true);
            let mut borrowed = false;
            function
                .lower_runtime_observed(false, &mut |pass, state| {
                    if pass != "ownership" {
                        return;
                    }
                    borrowed = state.graph.insts.values().any(|data| {
                        matches!(&data.op, Op::PropertyLoad(receiver, _, js::memory::Field::Named(name))
                        if name == "n" && state.graph.borrowed.contains(receiver))
                    });
                })
                .unwrap();
            assert_eq!(borrowed, expected, "{source}");
        }
    }
}

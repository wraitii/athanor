use crate::jit::profile::{ReadKind, ReadType, Site};

use super::super::{
    BlockData, Cfg, Facts, Graph, Inst, Op, Target, Terminator, TerminatorKind, TypedRead,
    ValueDef, ValueType,
};

#[derive(Clone, PartialEq, Eq)]
pub(in crate::jit) struct Read {
    pub(in crate::jit::semantic) inst: Inst,
    pub(in crate::jit::semantic) typed: bool,
    pub(in crate::jit::semantic) element: Option<crate::heap::TypedKind>,
    pub(in crate::jit::semantic) guard_number: bool,
}

pub(in crate::jit::semantic) fn plan(
    graph: &Graph,
    bodies: &super::calls::Repository,
    source: usize,
) -> Vec<Read> {
    plan_accesses(graph, bodies, source, false)
}

pub(in crate::jit::semantic) fn plan_stores(
    graph: &Graph,
    bodies: &super::calls::Repository,
    source: usize,
) -> Vec<Read> {
    plan_accesses(graph, bodies, source, true)
}

fn plan_accesses(
    graph: &Graph,
    bodies: &super::calls::Repository,
    source: usize,
    stores: bool,
) -> Vec<Read> {
    let facts = Facts::compute(graph);
    let hints = super::observations::read_hints(graph);
    let mut reads = Vec::new();
    for block in Cfg::new(graph).rpo {
        for &inst in &graph.blocks[block].insts {
            let (object, key) = match graph.insts[inst].op {
                Op::SetIndex(object, key, _) if stores => (object, key),
                Op::GetIndex(object, key) if !stores => (object, key),
                Op::TypedIndex(object, key, TypedRead { element: None, .. }) if !stores => {
                    (object, key)
                }
                _ => continue,
            };
            let site = |inst: Inst| {
                let data = &graph.insts[inst];
                Site {
                    source: data.source_name.map_or(source, |name| name.0),
                    start: data.span.start,
                    end: data.span.end,
                }
            };
            let observed = bodies
                .read_feedback
                .get(&(site(inst), ReadKind::Receiver))
                .copied()
                .or_else(|| hints.get(&object).copied())
                .or_else(|| {
                    let ValueDef::Result(producer, _) = graph.values[object] else {
                        return None;
                    };
                    bodies
                        .read_feedback
                        .get(&(site(producer), ReadKind::Result))
                        .copied()
                })
                .or_else(|| super::regions::element_guess(graph, object).map(ReadType::Typed));
            let Some(observed) = observed else {
                continue;
            };
            let types = facts.of(key);
            if !types.intersects(ValueType::NUMBER) {
                continue;
            }
            reads.push(Read {
                inst,
                typed: matches!(observed, ReadType::Typed(_)),
                element: match observed {
                    ReadType::Typed(kind) => Some(kind),
                    ReadType::Array => None,
                    ReadType::Unknown | ReadType::Integer | ReadType::Number => continue,
                },
                guard_number: !types.within(ValueType::NUMBER),
            });
        }
    }
    reads
}

pub(in crate::jit::semantic) fn assume(graph: &mut Graph, reads: &[Read]) {
    use super::super::guards::{Fact, Kind, Origin};
    for read in reads {
        let data = graph.insts[read.inst].clone();
        let (object, key) = match data.op {
            Op::GetIndex(object, key)
            | Op::TypedIndex(object, key, _)
            | Op::SetIndex(object, key, _) => (object, key),
            _ => unreachable!(),
        };
        let at = graph.blocks[data.block]
            .insts
            .iter()
            .position(|&i| i == read.inst)
            .unwrap();
        let receiver = if read.typed {
            Fact::Typed(read.element)
        } else {
            Fact::Indexed
        };
        graph.insert(
            data.block,
            at,
            data.span,
            Op::Assume(
                object,
                receiver,
                Origin {
                    site: None,
                    kind: Kind::Receiver,
                    gain: 1,
                },
            ),
            false,
        );
        if read.guard_number {
            graph.insert(
                data.block,
                at,
                data.span,
                Op::Assume(
                    key,
                    Fact::Type(ValueType::NUMBER),
                    Origin {
                        site: None,
                        kind: Kind::Number,
                        gain: 1,
                    },
                ),
                false,
            );
        }
    }
}

pub(in crate::jit::semantic) fn lower(
    graph: &mut Graph,
    planned: &[Read],
    notes: &mut Vec<String>,
) -> super::super::Result<()> {
    let reads: Vec<_> = Cfg::new(graph)
        .rpo
        .into_iter()
        .filter(|block| !graph.cold.contains(block))
        .flat_map(|block| graph.blocks[block].insts.clone())
        .filter_map(|inst| {
            let data = &graph.insts[inst];
            if !matches!(
                data.op,
                Op::GetIndex(..) | Op::TypedIndex(_, _, TypedRead { element: None, .. })
            ) {
                return None;
            }
            planned
                .iter()
                .find(|read| {
                    let original = &graph.insts[read.inst];
                    original.span == data.span && original.source_name == data.source_name
                })
                .map(|read| Read {
                    inst,
                    ..read.clone()
                })
        })
        .collect();
    apply(graph, &reads, notes)?;
    let facts = Facts::compute(graph);
    let arrays: Vec<_> = facts
        .cfg
        .rpo
        .iter()
        .filter(|block| !graph.cold.contains(block))
        .flat_map(|&block| graph.blocks[block].insts.iter().copied())
        .filter(|&inst| {
            matches!(graph.insts[inst].op, Op::ArrayGet(_, key)
            if facts.of(key).within(ValueType::NUMBER))
        })
        .collect();
    for inst in arrays {
        lower_array(graph, inst, false);
        notes.push(format!(
            "read at {}: packed/holey array checks",
            graph.note_inst(inst)
        ));
    }
    graph.verify()
}

pub(in crate::jit::semantic) fn apply(
    graph: &mut Graph,
    reads: &[Read],
    notes: &mut Vec<String>,
) -> super::super::Result<()> {
    for read in reads {
        let site = graph.note_inst(read.inst);
        let data = graph.insts[read.inst].clone();
        let (object, key) = match data.op {
            Op::GetIndex(object, key)
            | Op::TypedIndex(object, key, TypedRead { element: None, .. }) => (object, key),
            _ => unreachable!(),
        };
        let cfg = Cfg::new(graph);
        let facts = Facts::compute(graph);
        let block = data.block;
        // A guarded indexed receiver owns the result even when the index needs a slow path.
        if !read.typed
            && super::super::guards::holds(
                graph,
                &cfg,
                block,
                object,
                super::super::guards::Fact::Indexed,
            )
            && facts.of(key).within(ValueType::NUMBER)
        {
            graph.set_op(read.inst, Op::ArrayGet(object, key));
        }
        if read.typed
            && super::super::guards::holds(
                graph,
                &cfg,
                block,
                object,
                super::super::guards::Fact::Typed(read.element),
            )
            && facts.of(key).within(ValueType::NUMBER)
        {
            graph.set_op(
                read.inst,
                Op::TypedIndex(
                    object,
                    key,
                    TypedRead {
                        element: read.element,
                        bounded: false,
                    },
                ),
            );
            continue;
        }
        if !read.typed {
            lower_array(graph, read.inst, read.guard_number);
            notes.push(format!("read at {site}: packed/holey array checks"));
            continue;
        }
        let first_block = graph.blocks.len();
        let (block, continuation, fallback) = super::super::dispatch::split(graph, read.inst);
        let fast = graph.blocks.push(BlockData::default());
        let check = graph.append_value(block, data.span, Op::IsTyped(object, read.element));
        graph.blocks[block].terminator = Some(Terminator {
            source_name: data.source_name,
            emission_location: None,
            span: data.span,
            kind: TerminatorKind::Branch(check, Target::new(fast), Target::new(fallback)),
        });
        let (fast, key) = if read.guard_number {
            let typed = graph.blocks.push(BlockData::default());
            let check = graph.append_value(fast, data.span, Op::IsNumber(key));
            graph.blocks[fast].terminator = Some(Terminator {
                source_name: data.source_name,
                emission_location: None,
                span: data.span,
                kind: TerminatorKind::Branch(check, Target::new(typed), Target::new(fallback)),
            });
            let key = graph.append_value(typed, data.span, Op::AsNumber(key));
            (typed, key)
        } else {
            (fast, key)
        };
        let (inst, value) = graph.append(
            fast,
            data.span,
            Op::TypedIndex(
                object,
                key,
                TypedRead {
                    element: read.element,
                    bounded: false,
                },
            ),
            true,
        );
        graph.insts[inst].source_name = data.source_name;
        graph.blocks[fast].terminator = Some(Terminator {
            source_name: data.source_name,
            emission_location: None,
            span: data.span,
            kind: TerminatorKind::Jump(Target {
                block: continuation,
                args: vec![value.unwrap()],
            }),
        });
        extend_regions(graph, data.block, first_block);
        notes.push(format!("read at {site}: typed receiver check"));
    }
    graph.verify()
}

fn lower_array(graph: &mut Graph, inst: Inst, guard_number: bool) {
    let data = graph.insts[inst].clone();
    let (object, key) = match data.op {
        Op::GetIndex(object, key) | Op::ArrayGet(object, key) => (object, key),
        _ => unreachable!(),
    };
    let terminate = |graph: &mut Graph, block, kind| {
        graph.blocks[block].terminator = Some(Terminator {
            source_name: data.source_name,
            emission_location: None,
            span: data.span,
            kind,
        });
    };
    let first_block = graph.blocks.len();
    let (block, continuation, fallback) = super::super::dispatch::split(graph, inst);
    let (fast, key) = if guard_number {
        let fast = graph.blocks.push(BlockData::default());
        let check = graph.append_value(block, data.span, Op::IsNumber(key));
        terminate(
            graph,
            block,
            TerminatorKind::Branch(check, Target::new(fast), Target::new(fallback)),
        );
        let key = graph.append_value(fast, data.span, Op::AsNumber(key));
        (fast, key)
    } else {
        (block, key)
    };
    let position = graph.append_value(fast, data.span, Op::ArrayPosition(key));
    let mut guard = fast;
    for layout in [
        super::super::ArrayLayout::Packed,
        super::super::ArrayLayout::Holey,
    ] {
        let load = graph.blocks.push(BlockData::default());
        let otherwise = if layout == super::super::ArrayLayout::Holey {
            fallback
        } else {
            graph.blocks.push(BlockData::default())
        };
        let valid = graph.append_value(
            guard,
            data.span,
            Op::IsArrayIndex(object, key, position, layout),
        );
        terminate(
            graph,
            guard,
            TerminatorKind::Branch(valid, Target::new(load), Target::new(otherwise)),
        );
        let (load_inst, value) = graph.append(
            load,
            data.span,
            Op::ArrayLoad(object, position, layout),
            true,
        );
        graph.insts[load_inst].source_name = data.source_name;
        terminate(
            graph,
            load,
            TerminatorKind::Jump(Target {
                block: continuation,
                args: vec![value.unwrap()],
            }),
        );
        guard = otherwise;
    }
    extend_regions(graph, data.block, first_block);
}

fn extend_regions(graph: &mut Graph, block: super::super::Block, first: usize) {
    let added: Vec<_> = graph.blocks.keys().skip(first).collect();
    for region in graph.regions.values_mut() {
        if region.blocks.contains(&block) {
            region.blocks.extend(added.iter().copied());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FxHashSet;
    use crate::heap::TypedKind;
    use oxc_allocator::Allocator;
    use oxc_ast::ast::Statement;
    use oxc_parser::Parser;
    use oxc_span::SourceType;

    #[test]
    fn nested_gets_select_indexed_and_typed_reads_independently() {
        let allocator = Allocator::default();
        let parsed = Parser::new(
            &allocator,
            "function read(height,x,y){return height[x][y];}",
            SourceType::script(),
        )
        .parse();
        assert!(!parsed.diagnostics.has_errors());
        let Statement::FunctionDeclaration(function) = &parsed.program.body[0] else {
            panic!();
        };
        let semantic = oxc_semantic::SemanticBuilder::new_compiler()
            .build(&parsed.program)
            .semantic;
        let graph =
            super::super::super::build::lower(function, semantic.scoping(), &FxHashSet::default())
                .unwrap();
        let mut graph = super::super::super::canonicalize_graph(graph, true)
            .unwrap()
            .graph;
        let mut bodies = super::super::calls::Repository::default();
        for data in graph.insts.values() {
            let Op::GetIndex(object, _) = data.op else {
                continue;
            };
            let observed = if matches!(graph.defining_op(object), Some(Op::GetIndex(..))) {
                ReadType::Typed(TypedKind::Float64)
            } else {
                ReadType::Array
            };
            bodies.read_feedback.insert(
                (
                    Site {
                        source: 0,
                        start: data.span.start,
                        end: data.span.end,
                    },
                    ReadKind::Receiver,
                ),
                observed,
            );
        }
        let decisions = plan(&graph, &bodies, 0);
        assert_eq!(decisions.len(), 2);
        assert!(!decisions[0].typed);
        assert!(decisions[1].typed);
        let blocks = graph.blocks.len();
        assume(&mut graph, &decisions);
        assert_eq!(graph.blocks.len(), blocks);
        assert!(
            !graph
                .insts
                .values()
                .any(|data| matches!(data.op, Op::ArrayLoad(..)))
        );
        assert!(
            graph
                .insts
                .values()
                .any(|data| matches!(data.op, Op::Assume(..)))
        );
        graph.strip_assumes();
        lower(&mut graph, &decisions, &mut Vec::new()).unwrap();
        assert_eq!(
            graph
                .insts
                .values()
                .filter(|data| matches!(data.op, Op::ArrayLoad(..)))
                .count(),
            2
        );
        assert_eq!(
            graph
                .insts
                .values()
                .filter(|data| matches!(
                    data.op,
                    Op::TypedIndex(
                        _,
                        _,
                        TypedRead {
                            element: Some(TypedKind::Float64),
                            ..
                        }
                    )
                ))
                .count(),
            1
        );
    }
}

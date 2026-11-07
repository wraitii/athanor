use crate::jit::planner::{ReadKind, ReadType, Site};

use super::{
    BlockData, Cfg, Facts, Graph, Inst, Op, Target, Terminator, TerminatorKind, TypedRead, Types,
    ValueDef,
};

#[derive(Clone, PartialEq, Eq)]
pub(in crate::jit) struct Read {
    pub(super) inst: Inst,
    pub(super) typed: bool,
    pub(super) element: Option<crate::heap::TypedKind>,
    pub(super) guard_number: bool,
}

pub(super) fn plan(
    graph: &Graph,
    bodies: &super::specialize::Repository,
    source: usize,
) -> Vec<Read> {
    let facts = Facts::compute(graph);
    let mut reads = Vec::new();
    for block in Cfg::new(graph).rpo {
        for &inst in &graph.blocks[block].insts {
            let (object, key) = match graph.insts[inst].op {
                Op::GetIndex(object, key)
                | Op::TypedIndex(object, key, TypedRead { element: None, .. }) => (object, key),
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
                .or_else(|| {
                    let ValueDef::Result(producer) = graph.values[object] else {
                        return None;
                    };
                    bodies
                        .read_feedback
                        .get(&(site(producer), ReadKind::Result))
                        .copied()
                });
            let Some(observed) = observed else {
                continue;
            };
            let types = facts.of(key);
            if !types.intersects(Types::NUMBER) {
                continue;
            }
            reads.push(Read {
                inst,
                typed: matches!(observed, ReadType::Typed(_)),
                element: match observed {
                    ReadType::Typed(kind) => Some(kind),
                    ReadType::Array => None,
                },
                guard_number: !types.within(Types::NUMBER),
            });
        }
    }
    reads
}

pub(super) fn apply(
    graph: &mut Graph,
    reads: &[Read],
    notes: &mut Vec<String>,
) -> super::Result<()> {
    for read in reads {
        let data = graph.insts[read.inst].clone();
        let (object, key) = match data.op {
            Op::GetIndex(object, key)
            | Op::TypedIndex(object, key, TypedRead { element: None, .. }) => (object, key),
            _ => unreachable!(),
        };
        let (block, continuation, fallback) = super::inlining::split_call(graph, read.inst);
        let fast = graph.blocks.push(BlockData::default());
        let guard = if read.typed {
            Op::IsTyped(object, read.element)
        } else {
            Op::IsIndexed(object)
        };
        let check = graph.append(block, data.span, guard, true).1.unwrap();
        graph.blocks[block].terminator = Some(Terminator {
            source_name: data.source_name,
            span: data.span,
            kind: TerminatorKind::Branch(check, Target::new(fast), Target::new(fallback)),
        });
        let (fast, key) = if read.guard_number {
            let typed = graph.blocks.push(BlockData::default());
            let check = graph
                .append(fast, data.span, Op::IsNumber(key), true)
                .1
                .unwrap();
            graph.blocks[fast].terminator = Some(Terminator {
                source_name: data.source_name,
                span: data.span,
                kind: TerminatorKind::Branch(check, Target::new(typed), Target::new(fallback)),
            });
            let key = graph
                .append(typed, data.span, Op::AsNumber(key), true)
                .1
                .unwrap();
            (typed, key)
        } else {
            (fast, key)
        };
        let (inst, value) = graph.append(
            fast,
            data.span,
            if read.typed {
                Op::TypedIndex(
                    object,
                    key,
                    TypedRead {
                        element: read.element,
                        bounded: false,
                    },
                )
            } else {
                Op::IndexedGet(object, key)
            },
            true,
        );
        graph.insts[inst].source_name = data.source_name;
        graph.blocks[fast].terminator = Some(Terminator {
            source_name: data.source_name,
            span: data.span,
            kind: TerminatorKind::Jump(Target {
                block: continuation,
                args: vec![value.unwrap()],
            }),
        });
        notes.push(format!(
            "read at {}..{}: {} receiver check",
            data.span.start,
            data.span.end,
            if read.typed { "typed" } else { "indexed" }
        ));
    }
    graph.verify()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heap::TypedKind;
    use oxc_allocator::Allocator;
    use oxc_ast::ast::Statement;
    use oxc_parser::Parser;
    use oxc_span::SourceType;
    use std::collections::HashSet;

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
            super::super::build::lower(function, semantic.scoping(), &HashSet::new()).unwrap();
        let mut graph = super::super::canonicalize(graph, true).unwrap().graph;
        let mut bodies = super::super::specialize::Repository::default();
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
        apply(&mut graph, &decisions, &mut Vec::new()).unwrap();
        assert_eq!(
            graph
                .insts
                .values()
                .filter(|data| matches!(data.op, Op::IndexedGet(..)))
                .count(),
            1
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

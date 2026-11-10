use crate::jit::planner::{ShapeAccess, Site};

use super::{BlockData, Cfg, Graph, Inst, Op, Target, Terminator, TerminatorKind};

#[derive(Clone, PartialEq, Eq)]
pub(in crate::jit) struct Shape {
    pub(super) inst: Inst,
    pub(super) shape: u32,
    pub(super) index: u32,
}

pub(super) fn plan(
    graph: &Graph,
    bodies: &super::specialize::Repository,
    source: usize,
) -> Vec<Shape> {
    let mut shapes = Vec::new();
    for block in Cfg::new(graph).rpo {
        if graph.cold.contains(&block) {
            continue;
        }
        for &inst in &graph.blocks[block].insts {
            let data = &graph.insts[inst];
            let site = Site {
                source: data.source_name.map_or(source, |name| name.0),
                start: data.span.start,
                end: data.span.end,
            };
            let access = match data.op {
                Op::Get(_, ref name) if name != "length" => {
                    bodies.shape_feedback.get(&site).and_then(|access| match *access {
                        ShapeAccess::Read { shape, index } => Some((shape, index)),
                        _ => None,
                    })
                }
                Op::Set(..) => bodies.shape_feedback.get(&site).and_then(|access| match *access {
                    ShapeAccess::Overwrite { shape, index } => Some((shape, index)),
                    _ => None,
                }),
                _ => None,
            };
            if let Some((shape, index)) = access {
                shapes.push(Shape { inst, shape, index });
            }
        }
    }
    shapes
}

pub(super) fn apply(
    graph: &mut Graph,
    shapes: &[Shape],
    notes: &mut Vec<String>,
) -> super::Result<()> {
    let mut applied = 0;
    for &Shape { inst, shape, index } in shapes {
        let data = graph.insts[inst].clone();
        let (object, name, value) = match data.op {
            Op::Get(object, ref name) => (object, name.clone(), None),
            Op::Set(object, ref name, value) => (object, name.clone(), Some(value)),
            _ => continue,
        };
        let (block, continuation, fallback) = super::inlining::split_call(graph, inst);
        let fast = graph.blocks.push(BlockData::default());
        let check = graph
            .append(block, data.span, Op::IsShape(object, shape), true)
            .1
            .unwrap();
        graph.blocks[block].terminator = Some(Terminator {
            source_name: data.source_name,
            span: data.span,
            kind: TerminatorKind::Branch(check, Target::new(fast), Target::new(fallback)),
        });
        let args = match value {
            Some(value) => {
                let (access, _) = graph.append(
                    fast,
                    data.span,
                    Op::ShapedSet(object, name, index, value),
                    false,
                );
                graph.insts[access].source_name = data.source_name;
                Vec::new()
            }
            None => {
                let (access, result) =
                    graph.append(fast, data.span, Op::ShapedGet(object, name, index), true);
                graph.insts[access].source_name = data.source_name;
                vec![result.unwrap()]
            }
        };
        graph.blocks[fast].terminator = Some(Terminator {
            source_name: data.source_name,
            span: data.span,
            kind: TerminatorKind::Jump(Target {
                block: continuation,
                args,
            }),
        });
        applied += 1;
    }
    if applied != 0 {
        notes.push(format!("shape guards: {applied} property accesses"));
    }
    graph.verify()
}

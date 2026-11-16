use crate::FxHashSet;

use super::super::guards::{Fact, Kind, Origin};
use super::super::{Cfg, Facts, Graph, Inst, Op, Value, available, dispatch};

#[derive(Clone, Default)]
pub(in crate::jit::semantic) struct Access {
    pub(in crate::jit::semantic) cache: Option<Value>,
    pub(in crate::jit::semantic) shapes: Vec<(u32, u32)>,
}

pub(in crate::jit::semantic) fn assume(graph: &mut Graph, shapes: &super::shapes::Plan) {
    assume_natives(graph);
    let builtins: FxHashSet<_> = graph
        .insts
        .values()
        .filter_map(|data| match data.op {
            Op::Assume(value, Fact::Native(_), _)
            | Op::IsNative(value, _)
            | Op::IsArrayIteration(value, ..)
            | Op::IsStringSplit(value) => Some(value),
            _ => None,
        })
        .collect();
    let accesses: Vec<_> = Cfg::new(graph)
        .rpo
        .into_iter()
        .flat_map(|block| graph.blocks[block].insts.clone())
        .filter(|&inst| match graph.insts[inst].op {
            Op::Get(_, ref name) => name != "length" && !shapes.megamorphic(graph, inst),
            Op::Set(..) => shapes.write(graph, inst).is_some(),
            _ => false,
        })
        .collect();
    for inst in accesses {
        let data = graph.insts[inst].clone();
        let (object, name, alternatives) = match data.op.clone() {
            Op::Get(object, name) => (
                object,
                name,
                shapes
                    .alternatives(graph, inst)
                    .map(<[_]>::to_vec)
                    .or_else(|| shapes.read(graph, inst).map(|access| vec![access]))
                    .unwrap_or_default(),
            ),
            Op::Set(object, name, _) => (
                object,
                name,
                shapes.write(graph, inst).into_iter().collect(),
            ),
            _ => unreachable!(),
        };
        let origin = Origin {
            site: None,
            kind: Kind::Receiver,
            gain: 1,
        };
        let at = graph.blocks[data.block]
            .insts
            .iter()
            .position(|&i| i == inst)
            .unwrap();
        let (allocation, cache) = graph.insert(
            data.block,
            at,
            data.span,
            Op::PropertyCache(inst, name),
            true,
        );
        graph.insts[allocation].source_name = data.source_name;
        let cache = cache.unwrap();
        if !data.result().is_some_and(|value| builtins.contains(&value)) {
            let (assume, _) = graph.insert(
                data.block,
                at + 1,
                data.span,
                Op::Assume(object, Fact::Property(cache), origin),
                false,
            );
            graph.insts[assume].source_name = data.source_name;
        }
        graph.property_accesses.insert(
            inst,
            Access {
                cache: Some(cache),
                shapes: alternatives,
            },
        );
    }
}

pub(in crate::jit::semantic) fn shape_writes(graph: &mut Graph) {
    let mut candidates: Vec<_> = Cfg::new(graph)
        .rpo
        .into_iter()
        .flat_map(|block| graph.blocks[block].insts.clone())
        .filter_map(|inst| {
            let Op::Set(object, ref name, value) = graph.insts[inst].op else {
                return None;
            };
            let &[(shape, index)] = graph.property_accesses.get(&inst)?.shapes.as_slice() else {
                return None;
            };
            let set = graph.insts[inst].op.clone();
            graph.insts[inst].op = Op::ShapedSet(object, name.clone(), index, value);
            Some((inst, object, shape, set))
        })
        .collect();
    while !candidates.is_empty() {
        let facts = Facts::compute(graph);
        let solved = available::solve(graph, &facts, &facts.cfg, Default::default(), false);
        let (proven, failed): (Vec<_>, Vec<_>) =
            candidates
                .into_iter()
                .partition(|&(inst, object, shape, _)| {
                    solved.as_ref().is_some_and(|solved| {
                        let state = solved.before(graph, &facts, &facts.cfg, inst);
                        solved.holds(graph, &state, object, Fact::Shape(shape))
                    })
                });
        candidates = proven;
        if failed.is_empty() {
            break;
        }
        for (inst, _, _, set) in failed {
            graph.set_op(inst, set);
        }
    }
    for (inst, object, shape, _) in candidates {
        graph.property_accesses.remove(&inst);
        if graph.insts[inst].block == graph.entry {
            super::super::guards::record_entry(graph, object, Fact::Shape(shape));
        }
    }
}

pub(in crate::jit::semantic) fn rewrite(graph: &mut Graph) -> super::super::Result<()> {
    shape_writes(graph);
    let facts = Facts::compute(graph);
    let solved = available::solve(graph, &facts, &facts.cfg, Default::default(), false);
    let reads: Vec<_> = facts
        .cfg
        .rpo
        .iter()
        .copied()
        .flat_map(|block| graph.blocks[block].insts.iter().copied())
        .filter_map(|inst| {
            let Op::Get(object, ref name) = graph.insts[inst].op else {
                return None;
            };
            let read = graph.property_accesses.get(&inst)?.clone();
            let state = solved
                .as_ref()
                .map(|solved| (solved, solved.before(graph, &facts, &facts.cfg, inst)));
            let holds = |fact| {
                state
                    .as_ref()
                    .is_some_and(|(solved, state)| solved.holds(graph, state, object, fact))
            };
            let shaped = read
                .shapes
                .iter()
                .find(|&&(shape, _)| holds(Fact::Shape(shape)))
                .copied();
            let cached = read.cache.is_some_and(|cache| {
                holds(Fact::Cached(cache, crate::builtins::builtin_method(name)))
            });
            Some((inst, read, shaped, cached))
        })
        .collect();
    for (inst, read, shaped, cached) in reads {
        graph.property_accesses.remove(&inst);
        let data = graph.insts[inst].clone();
        let Op::Get(object, name) = data.op.clone() else {
            continue;
        };
        if let Some((shape, index)) = shaped {
            if data.block == graph.entry {
                super::super::guards::record_entry(graph, object, Fact::Shape(shape));
            }
            graph.set_op(inst, Op::ShapedGet(object, name, index));
            continue;
        }
        let cache = read.cache.unwrap_or_else(|| {
            let at = graph.blocks[data.block]
                .insts
                .iter()
                .position(|&i| i == inst)
                .unwrap();
            let (allocation, cache) = graph.insert(
                data.block,
                at,
                data.span,
                Op::PropertyCache(inst, name.clone()),
                true,
            );
            graph.insts[allocation].source_name = data.source_name;
            cache.unwrap()
        });
        if cached {
            graph.insts[inst].op =
                Op::CachedProperty(object, cache, crate::builtins::builtin_method(&name));
            continue;
        }
        lower_cache(graph, inst, cache, object, &name);
    }
    graph.property_accesses.clear();
    graph.verify()
}

pub(in crate::jit::semantic) fn lower_cache(
    graph: &mut Graph,
    inst: Inst,
    cache: Value,
    object: Value,
    name: &str,
) {
    let data = graph.insts[inst].clone();
    let first = graph.blocks.len();
    let (block, done, fallback) = dispatch::split(graph, inst);
    graph.cold.remove(&fallback);
    graph.set_op(inst, Op::FillPropertyCache(object, cache, name.to_owned()));
    let check = dispatch::append(
        graph,
        &data,
        block,
        Op::IsPropertyCached(object, cache, crate::builtins::builtin_method(name)),
    );
    let fast = dispatch::branch(graph, inst, block, check, fallback);
    let value = dispatch::append(
        graph,
        &data,
        fast,
        Op::CachedProperty(object, cache, crate::builtins::builtin_method(name)),
    );
    graph.terminate(
        fast,
        data.span,
        super::super::TerminatorKind::Jump(super::super::Target {
            block: done,
            args: vec![value],
        }),
    );
    dispatch::extend_regions(graph, block, first);
}

pub(in crate::jit::semantic) fn assume_natives(graph: &mut Graph) {
    let blocks = Cfg::new(graph).rpo;
    let mut seen: FxHashSet<_> = blocks
        .iter()
        .flat_map(|&block| &graph.blocks[block].insts)
        .filter_map(|&inst| match graph.insts[inst].op {
            Op::Assume(value, Fact::Native(native), _) => {
                Some((graph.insts[inst].block, value, native))
            }
            _ => None,
        })
        .collect();
    for requirement in super::regions::requirements(graph, &blocks) {
        let data = graph.insts[requirement.call].clone();
        let Op::Call { callee, .. } = data.op else {
            continue;
        };
        if seen.insert((data.block, callee, requirement.native)) {
            let origin = Origin {
                site: None,
                kind: Kind::Native,
                gain: 1,
            };
            let (inst, _) = graph.append(
                data.block,
                data.span,
                Op::Assume(callee, Fact::Native(requirement.native), origin),
                false,
            );
            graph.insts[inst].source_name = data.source_name;
        }
    }
}

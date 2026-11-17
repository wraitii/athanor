use super::super::transforms::loops::{natural_loops, preheader};
use super::super::{Block, FunctionState, Op, Value, ValueType};
use crate::FxHashMap;

pub(in crate::jit::semantic) fn views(function: &mut FunctionState) {
    function.ensure_values();
    let FunctionState {
        graph,
        facts,
        notes,
        ..
    } = function;
    let cfg = &facts.cfg;
    let mut loops = natural_loops(graph, cfg);
    loops.reverse();
    let mut placed: FxHashMap<(Block, Value), Value> = FxHashMap::default();
    for &block in &cfg.rpo {
        for inst in graph.blocks[block].insts.clone() {
            let object = match graph.insts[inst].op {
                Op::TypedIndex(
                    object,
                    key,
                    super::super::TypedRead {
                        element: Some(_), ..
                    },
                )
                | Op::ArrayLoad(object, key, _)
                | Op::CanOverwrite(object, key)
                | Op::ArrayStore(object, key, _)
                | Op::PropertyStore(object, key, ..)
                    if facts.of(key).intersects(ValueType::NUMBER) =>
                {
                    object
                }
                Op::IsArray(object, _) | Op::ArrayKind(object) | Op::IndexedLength(object) => {
                    object
                }
                _ => continue,
            };
            if !facts.of(object).intersects(ValueType::OBJECT) {
                continue;
            }
            let Some(preheader) = loops.iter().find_map(|(header, body)| {
                (body.contains(&block) && !body.contains(&graph.def_block(object)))
                    .then(|| preheader(graph, cfg, *header, body))
                    .flatten()
            }) else {
                continue;
            };
            let view = match placed.get(&(preheader, object)) {
                Some(&view) => view,
                None => {
                    let span = graph.insts[inst].span;
                    let view = graph.append_value(preheader, span, Op::View(object));
                    notes.push(format!("{view} = view {object} in {preheader}"));
                    placed.insert((preheader, object), view);
                    view
                }
            };
            graph.views.insert(inst, view);
        }
    }
    if !placed.is_empty() {
        function.invalidate_values();
    }
}

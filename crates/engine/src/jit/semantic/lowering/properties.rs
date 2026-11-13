use super::super::{BlockData, Cfg, FunctionState, Op, Target, Terminator, TerminatorKind};

// TODO: split the legacy named-property dispatch once cache-entry selection is explicit.
pub(in crate::jit::semantic) fn lower(function: &mut FunctionState) -> super::super::Result<bool> {
    let shapes = &function.plan.shapes;
    let graph = &mut function.graph;
    if graph.has_handlers || graph.generator {
        return Ok(false);
    }
    let reads: Vec<_> = Cfg::new(graph)
        .rpo
        .into_iter()
        .flat_map(|block| graph.blocks[block].insts.clone())
        .filter(|&inst| {
            matches!(graph.insts[inst].op, Op::Get(..)) && !shapes.megamorphic(graph, inst)
        })
        .collect();
    let mut changed = !reads.is_empty();
    changed |= lower_stores(graph);
    let mut cached = Vec::new();
    for inst in reads {
        let data = graph.insts[inst].clone();
        let Op::Get(object, name) = &data.op else {
            unreachable!()
        };
        if name != "length" {
            cached.push(inst);
            continue;
        }
        let object = *object;
        let first = graph.blocks.len();
        let (block, continuation, fallback) = super::super::dispatch::split(graph, inst);
        graph.insts[inst].op = Op::FillPropertyCache(object, inst, name.clone());
        let fast = graph.blocks.push(BlockData::default());
        let check = graph.append_value(block, data.span, Op::IsIndexed(object));
        graph.blocks[block].terminator = Some(Terminator {
            source_name: data.source_name,
            emission_location: None,
            span: data.span,
            kind: TerminatorKind::Branch(check, Target::new(fast), Target::new(fallback)),
        });
        let (load, value) = graph.append(fast, data.span, Op::IndexedLength(object), true);
        graph.insts[load].source_name = data.source_name;
        graph.blocks[fast].terminator = Some(Terminator {
            source_name: data.source_name,
            emission_location: None,
            span: data.span,
            kind: TerminatorKind::Jump(Target {
                block: continuation,
                args: vec![value.unwrap()],
            }),
        });
        let added: Vec<_> = graph.blocks.keys().skip(first).collect();
        for region in graph.regions.values_mut() {
            if region.blocks.contains(&block) {
                region.blocks.extend(added.iter().copied());
            }
        }
    }
    super::super::caches::lower(graph, &cached)?;
    function.invalidate_cfg();
    Ok(changed)
}

fn lower_stores(graph: &mut super::super::Graph) -> bool {
    use super::super::{Constant, low_level};
    use crate::heap::{KIND_OFFSET, SHAPE_OFFSET, kind};
    use crate::jit::helpers::{
        CACHE_ADDITION_SOURCE_OFFSET, CACHE_OVERWRITE_INDEX_OFFSET, CACHE_OVERWRITE_SHAPE_OFFSET,
    };
    use cranelift_codegen::ir::{MemFlagsData, condcodes::IntCC, types};
    let stores: Vec<_> = Cfg::new(graph)
        .rpo
        .into_iter()
        .flat_map(|block| graph.blocks[block].insts.clone())
        .filter(|&inst| {
            matches!(
                graph.insts[inst].op,
                Op::Set(..) | Op::Define(..) | Op::ShapedSet(..)
            )
        })
        .collect();
    for &inst in &stores {
        let data = graph.insts[inst].clone();
        let write = if matches!(data.op, Op::Define(..)) {
            super::super::PropertyWrite::Define
        } else {
            super::super::PropertyWrite::Set
        };
        let (object, name, value, shaped) = match data.op {
            Op::Set(object, name, value) | Op::Define(object, name, value) => {
                (object, name, value, None)
            }
            Op::ShapedSet(object, name, index, value) => (object, name, value, Some(index)),
            _ => unreachable!(),
        };
        let first = graph.blocks.len();
        let (block, done, fallback) = super::super::dispatch::split(graph, inst);
        graph.insts[inst].op = Op::FillPropertyStore(object, inst, name.clone(), value, write);
        let view = graph.append_value(block, data.span, Op::View(object));
        let mut append = |block, op| {
            let (i, v) = graph.append(block, data.span, op, true);
            graph.insts[i].source_name = data.source_name;
            v.unwrap()
        };
        let (overwrite, index) = if let Some(index) = shaped {
            (
                block,
                append(block, Op::Constant(Constant::Number(index as f64))),
            )
        } else {
            let cache = append(block, Op::PropertyWriteCache(inst, name));
            let mut load = |address, offset| {
                append(
                    block,
                    Op::LowLevel(low_level::Op::Load {
                        ty: types::I32,
                        address,
                        offset,
                        flags: MemFlagsData::trusted(),
                    }),
                )
            };
            let actual_kind = load(view, KIND_OFFSET);
            let shape = load(view, SHAPE_OFFSET);
            let addition_shape = load(cache, CACHE_ADDITION_SOURCE_OFFSET);
            let overwrite_shape = load(cache, CACHE_OVERWRITE_SHAPE_OFFSET);
            let index = load(cache, CACHE_OVERWRITE_INDEX_OFFSET);
            let inline_kind = append(
                block,
                Op::LowLevel(low_level::Op::Integer {
                    ty: types::I32,
                    bits: (kind::OBJECT | kind::INLINE) as u64,
                }),
            );
            let inline = append(
                block,
                Op::LowLevel(low_level::Op::IntCompare {
                    condition: IntCC::Equal,
                    left: actual_kind,
                    right: inline_kind,
                }),
            );
            let matching = append(
                block,
                Op::LowLevel(low_level::Op::IntCompare {
                    condition: IntCC::Equal,
                    left: shape,
                    right: addition_shape,
                }),
            );
            let addition = append(block, Op::BooleanAnd(inline, matching));
            let object_bits = append(
                block,
                Op::LowLevel(low_level::Op::Integer {
                    ty: types::I32,
                    bits: kind::OBJECT as u64,
                }),
            );
            let bits = append(
                block,
                Op::LowLevel(low_level::Op::Binary {
                    op: low_level::Binary::And,
                    left: actual_kind,
                    right: object_bits,
                }),
            );
            let is_object = append(
                block,
                Op::LowLevel(low_level::Op::IntCompare {
                    condition: IntCC::Equal,
                    left: bits,
                    right: object_bits,
                }),
            );
            let matching = append(
                block,
                Op::LowLevel(low_level::Op::IntCompare {
                    condition: IntCC::Equal,
                    left: shape,
                    right: overwrite_shape,
                }),
            );
            let overwrite = append(block, Op::BooleanAnd(is_object, matching));
            let add = graph.blocks.push(BlockData::default());
            let check = graph.blocks.push(BlockData::default());
            graph.terminate(
                block,
                data.span,
                TerminatorKind::Branch(addition, Target::new(add), Target::new(check)),
            );
            let (store, _) = graph.append(
                add,
                data.span,
                Op::PropertyAdd(object, cache, value, write),
                false,
            );
            graph.views.insert(store, view);
            graph.terminate(add, data.span, TerminatorKind::Jump(Target::new(done)));
            let fast = super::super::writes::branch(graph, inst, check, overwrite, fallback);
            let index = graph.append_value(
                fast,
                data.span,
                Op::LowLevel(low_level::Op::Unary {
                    op: low_level::Unary::ZeroExtend,
                    ty: types::I64,
                    value: index,
                }),
            );
            (fast, index)
        };
        let (check, writable) =
            graph.append(overwrite, data.span, Op::CanOverwrite(object, index), true);
        graph.views.insert(check, view);
        let writable = writable.unwrap();
        let fast = super::super::writes::branch(graph, inst, overwrite, writable, fallback);
        let (store, _) = graph.append(
            fast,
            data.span,
            Op::PropertyStore(object, index, value, write),
            false,
        );
        graph.views.insert(store, view);
        graph.terminate(fast, data.span, TerminatorKind::Jump(Target::new(done)));
        super::super::writes::extend_regions(graph, block, first);
    }
    !stores.is_empty()
}

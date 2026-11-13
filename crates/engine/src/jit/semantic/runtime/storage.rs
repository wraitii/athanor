use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{MemFlagsData, types};

use super::super::selection::{Implementation, Lowering};
use super::super::{ArrayLayout, Cfg, FunctionState, Graph, Op, Repr, ValueType, low_level};

pub(in crate::jit::semantic) fn lower_metadata(function: &mut FunctionState) -> bool {
    let graph = &mut function.graph;
    let mut changed = false;
    for block in Cfg::new(graph).rpo {
        for inst in graph.blocks[block].insts.clone() {
            let data = graph.insts[inst].clone();
            let (object, existing_view, ty, offset) = match data.op {
                Op::ArrayKind(object) => (
                    object,
                    graph.views.remove(&inst),
                    types::I32,
                    crate::heap::KIND_OFFSET,
                ),
                Op::IndexedLength(object) => (
                    object,
                    graph.views.remove(&inst),
                    types::I64,
                    crate::heap::LENGTH_OFFSET,
                ),
                Op::TypedData(view) => {
                    let Some(Op::View(object)) = graph.defining_op(view) else {
                        unreachable!()
                    };
                    (*object, Some(view), types::I64, crate::heap::DATA_OFFSET)
                }
                _ => continue,
            };
            let mut at = graph.blocks[block]
                .insts
                .iter()
                .position(|&i| i == inst)
                .unwrap();
            let view = existing_view.unwrap_or_else(|| {
                let (view_inst, view) = graph.insert(block, at, data.span, Op::View(object), true);
                graph.insts[view_inst].source_name = data.source_name;
                at += 1;
                view.unwrap()
            });
            graph.insts[inst].op = Op::LowLevel(low_level::Op::Load {
                ty,
                address: view,
                offset,
                flags: MemFlagsData::trusted(),
            });
            let (keep, _) = graph.insert(block, at + 1, data.span, Op::KeepAlive(object), false);
            graph.insts[keep].source_name = data.source_name;
            changed = true;
        }
    }
    if changed {
        function.invalidate_values();
    }
    changed
}

pub(in crate::jit::semantic) fn lower_array_loads(function: &mut FunctionState) {
    // Ownership is fixed: preserve result identities and the scheduled reference operations.
    let graph = &mut function.graph;
    let mut plan = graph
        .ownership
        .take()
        .expect("array storage lowering requires ownership planning");
    for block in Cfg::new(graph).rpo {
        for inst in graph.blocks[block].insts.clone() {
            let data = graph.insts[inst].clone();
            let Op::ArrayLoad(object, position, layout) = data.op else {
                continue;
            };
            let mut at = graph.blocks[block]
                .insts
                .iter()
                .position(|&i| i == inst)
                .unwrap();
            let existing_view = graph.views.remove(&inst);
            let mut insert = |graph: &mut Graph, op, repr, value_type| {
                let (inserted, value) = graph.insert(block, at, data.span, op, true);
                let value = value.unwrap();
                graph.insts[inserted].source_name = data.source_name;
                graph.insts[inserted].emission_location = data.emission_location;
                graph.reprs[value] = Some(repr);
                function.facts.set(value, value_type);
                plan.implementations[inserted] = Implementation {
                    lowering: if matches!(graph.insts[inserted].op, Op::View(_)) {
                        Lowering::Generic
                    } else {
                        Lowering::Typed(Some(repr))
                    },
                    ..Default::default()
                };
                at += 1;
                value
            };
            let view = existing_view
                .unwrap_or_else(|| insert(graph, Op::View(object), Repr::Boxed, ValueType::VIEW));
            let base = insert(
                graph,
                Op::LowLevel(low_level::Op::Load {
                    ty: types::I64,
                    address: view,
                    offset: crate::heap::DATA_OFFSET,
                    flags: MemFlagsData::trusted(),
                }),
                Repr::I64,
                ValueType::INT64,
            );
            let shift = insert(
                graph,
                Op::LowLevel(low_level::Op::Integer {
                    ty: types::I64,
                    bits: 3,
                }),
                Repr::I64,
                ValueType::INT64,
            );
            let offset = insert(
                graph,
                Op::LowLevel(low_level::Op::Binary {
                    op: low_level::Binary::ShiftLeft,
                    left: position,
                    right: shift,
                }),
                Repr::I64,
                ValueType::INT64,
            );
            let address = insert(
                graph,
                Op::LowLevel(low_level::Op::Binary {
                    op: low_level::Binary::Add,
                    left: base,
                    right: offset,
                }),
                Repr::I64,
                ValueType::INT64,
            );
            let load = low_level::Op::Load {
                ty: types::I64,
                address,
                offset: 0,
                flags: MemFlagsData::trusted(),
            };
            graph.insts[inst].op = if layout == ArrayLayout::Packed {
                Op::LowLevel(load)
            } else {
                let raw = insert(graph, Op::LowLevel(load), Repr::I64, ValueType::INT64);
                let hole = insert(
                    graph,
                    Op::LowLevel(low_level::Op::Integer {
                        ty: types::I64,
                        bits: crate::heap::HOLE.bits(),
                    }),
                    Repr::I64,
                    ValueType::INT64,
                );
                let absent = insert(
                    graph,
                    Op::LowLevel(low_level::Op::IntCompare {
                        condition: IntCC::Equal,
                        left: raw,
                        right: hole,
                    }),
                    Repr::I8,
                    ValueType::BOOLEAN,
                );
                let undefined = insert(
                    graph,
                    Op::LowLevel(low_level::Op::Integer {
                        ty: types::I64,
                        bits: crate::value::Value::UNDEFINED.bits(),
                    }),
                    Repr::I64,
                    ValueType::INT64,
                );
                Op::LowLevel(low_level::Op::Select {
                    condition: absent,
                    yes: undefined,
                    no: raw,
                })
            };
            plan.implementations[inst].lowering = Lowering::Typed(Some(Repr::Boxed));
            let (keep, _) = graph.insert(block, at + 1, data.span, Op::KeepAlive(object), false);
            graph.insts[keep].source_name = data.source_name;
            plan.implementations[keep] = Implementation {
                lowering: Lowering::Typed(None),
                ..Default::default()
            };
        }
    }
    graph.ownership = Some(plan);
}

pub(in crate::jit::semantic) fn lower_stores(function: &mut FunctionState) {
    // Storage owns these references; preserve the function ownership plan while expanding them.
    use crate::heap::{INLINE_PROPERTY_LENGTH_OFFSET, LENGTH_OFFSET, SHAPE_OFFSET};
    use crate::jit::helpers::{CACHE_ADDITION_DESTINATION_OFFSET, CACHE_ADDITION_INDEX_OFFSET};
    let graph = &mut function.graph;
    let mut plan = graph.ownership.take().unwrap();
    let mut checked = crate::FxHashMap::default();
    for block in Cfg::new(graph).rpo {
        for inst in graph.blocks[block].insts.clone() {
            let data = graph.insts[inst].clone();
            let (object, key, value, addition) = match data.op {
                Op::ArrayStore(object, key, value) | Op::PropertyStore(object, key, value, _) => {
                    (object, key, value, false)
                }
                Op::PropertyAdd(object, key, value, _) => (object, key, value, true),
                Op::CanOverwrite(object, key) => (object, key, object, false),
                _ => continue,
            };
            let reuse = checked.remove(&inst);
            let store = checked_store(graph, inst, object, key);
            let existing_view = graph.views.remove(&inst);
            let checking = matches!(data.op, Op::CanOverwrite(..));
            let mut at = graph.blocks[block]
                .insts
                .iter()
                .position(|&i| i == inst)
                .unwrap();
            let mut insert = |op: Op, repr: Option<Repr>| {
                let result = storage_value(graph, &mut plan, &data, block, at, op, repr);
                at += 1;
                result
            };
            let (address, old, addition_data) = if let Some((address, old)) = reuse {
                (address, Some(old), None)
            } else {
                let view = existing_view
                    .unwrap_or_else(|| insert(Op::View(object), Some(Repr::Boxed)).unwrap());
                let base = insert(
                    Op::LowLevel(low_level::Op::Load {
                        ty: types::I64,
                        address: view,
                        offset: crate::heap::DATA_OFFSET,
                        flags: MemFlagsData::trusted(),
                    }),
                    Some(Repr::I64),
                )
                .unwrap();
                let index = if addition {
                    let raw = insert(
                        Op::LowLevel(low_level::Op::Load {
                            ty: types::I32,
                            address: key,
                            offset: CACHE_ADDITION_INDEX_OFFSET,
                            flags: MemFlagsData::trusted(),
                        }),
                        Some(Repr::I32),
                    )
                    .unwrap();
                    insert(
                        Op::LowLevel(low_level::Op::Unary {
                            op: low_level::Unary::ZeroExtend,
                            ty: types::I64,
                            value: raw,
                        }),
                        Some(Repr::I64),
                    )
                    .unwrap()
                } else {
                    key
                };
                let shift = insert(
                    Op::LowLevel(low_level::Op::Integer {
                        ty: types::I64,
                        bits: 3,
                    }),
                    Some(Repr::I64),
                )
                .unwrap();
                let offset = insert(
                    Op::LowLevel(low_level::Op::Binary {
                        op: low_level::Binary::ShiftLeft,
                        left: index,
                        right: shift,
                    }),
                    Some(Repr::I64),
                )
                .unwrap();
                let address = insert(
                    Op::LowLevel(low_level::Op::Binary {
                        op: low_level::Binary::Add,
                        left: base,
                        right: offset,
                    }),
                    Some(Repr::I64),
                )
                .unwrap();
                let old = (!addition).then(|| {
                    insert(
                        Op::LowLevel(low_level::Op::Load {
                            ty: types::I64,
                            address,
                            offset: 0,
                            flags: MemFlagsData::trusted(),
                        }),
                        Some(Repr::Boxed),
                    )
                    .unwrap()
                });
                (address, old, Some((view, base, index)))
            };
            if checking {
                let old = old.unwrap();
                let shift = insert(
                    Op::LowLevel(low_level::Op::Integer {
                        ty: types::I64,
                        bits: crate::value::TAG_SHIFT as u64,
                    }),
                    Some(Repr::I64),
                )
                .unwrap();
                let tag = insert(
                    Op::LowLevel(low_level::Op::Binary {
                        op: low_level::Binary::ShiftRightUnsigned,
                        left: old,
                        right: shift,
                    }),
                    Some(Repr::I64),
                )
                .unwrap();
                let first = insert(
                    Op::LowLevel(low_level::Op::Integer {
                        ty: types::I64,
                        bits: crate::value::TAG_OBJECT >> crate::value::TAG_SHIFT,
                    }),
                    Some(Repr::I64),
                )
                .unwrap();
                let tag = insert(
                    Op::LowLevel(low_level::Op::Binary {
                        op: low_level::Binary::Subtract,
                        left: tag,
                        right: first,
                    }),
                    Some(Repr::I64),
                )
                .unwrap();
                let two = insert(
                    Op::LowLevel(low_level::Op::Integer {
                        ty: types::I64,
                        bits: 2,
                    }),
                    Some(Repr::I64),
                )
                .unwrap();
                let heap = insert(
                    Op::LowLevel(low_level::Op::IntCompare {
                        condition: IntCC::UnsignedLessThan,
                        left: tag,
                        right: two,
                    }),
                    Some(Repr::I8),
                )
                .unwrap();
                let mask = insert(
                    Op::LowLevel(low_level::Op::Integer {
                        ty: types::I64,
                        bits: (1 << crate::value::TAG_SHIFT) - 1,
                    }),
                    Some(Repr::I64),
                )
                .unwrap();
                let raw_address = insert(
                    Op::LowLevel(low_level::Op::Binary {
                        op: low_level::Binary::And,
                        left: old,
                        right: mask,
                    }),
                    Some(Repr::I64),
                )
                .unwrap();
                let zero = insert(
                    Op::LowLevel(low_level::Op::Integer {
                        ty: types::I64,
                        bits: 0,
                    }),
                    Some(Repr::I64),
                )
                .unwrap();
                let old_address = insert(
                    Op::LowLevel(low_level::Op::Select {
                        condition: heap,
                        yes: raw_address,
                        no: zero,
                    }),
                    Some(Repr::I64),
                )
                .unwrap();
                let present = insert(
                    Op::LowLevel(low_level::Op::IntCompare {
                        condition: IntCC::NotEqual,
                        left: old_address,
                        right: zero,
                    }),
                    Some(Repr::I8),
                )
                .unwrap();
                if let Some(store) = store {
                    checked.insert(store, (address, old));
                }
                let first_block = graph.blocks.len();
                let (block, continuation, native) = super::super::dispatch::split(graph, inst);
                if !graph.cold.contains(&block) {
                    graph.cold.remove(&native);
                }
                let scalar = graph.blocks.push(super::super::BlockData::default());
                graph.terminate(
                    block,
                    data.span,
                    super::super::TerminatorKind::Branch(
                        present,
                        super::super::Target::new(native),
                        super::super::Target::new(scalar),
                    ),
                );
                let result = graph.blocks[continuation].params[0];
                graph.reprs[result] = Some(Repr::I8);
                function.facts.set(result, ValueType::BOOLEAN);
                let yes = storage_value(
                    graph,
                    &mut plan,
                    &data,
                    scalar,
                    0,
                    Op::LowLevel(low_level::Op::Integer {
                        ty: types::I8,
                        bits: 1,
                    }),
                    Some(Repr::I8),
                )
                .unwrap();
                graph.terminate(
                    scalar,
                    data.span,
                    super::super::TerminatorKind::Jump(super::super::Target {
                        block: continuation,
                        args: vec![yes],
                    }),
                );
                let mut at = 0;
                let mut insert = |op, repr| {
                    let result = storage_value(graph, &mut plan, &data, native, at, op, repr);
                    at += 1;
                    result.unwrap()
                };
                let kind = insert(
                    Op::LowLevel(low_level::Op::Load {
                        ty: types::I32,
                        address: old_address,
                        offset: crate::heap::KIND_OFFSET,
                        flags: MemFlagsData::trusted(),
                    }),
                    Some(Repr::I32),
                );
                let mask = insert(
                    Op::LowLevel(low_level::Op::Integer {
                        ty: types::I32,
                        bits: crate::heap::kind::NATIVE as u64,
                    }),
                    Some(Repr::I32),
                );
                let native_kind = insert(
                    Op::LowLevel(low_level::Op::Binary {
                        op: low_level::Binary::And,
                        left: kind,
                        right: mask,
                    }),
                    Some(Repr::I32),
                );
                let zero = insert(
                    Op::LowLevel(low_level::Op::Integer {
                        ty: types::I32,
                        bits: 0,
                    }),
                    Some(Repr::I32),
                );
                graph.insts[inst].op = Op::LowLevel(low_level::Op::IntCompare {
                    condition: IntCC::Equal,
                    left: native_kind,
                    right: zero,
                });
                plan.implementations[inst].lowering = Lowering::Typed(Some(Repr::I8));
                super::super::dispatch::extend_regions(graph, block, first_block);
                continue;
            }
            insert(
                Op::LowLevel(low_level::Op::Reference {
                    op: low_level::Reference::Retain,
                    value,
                }),
                None,
            );
            insert(
                Op::LowLevel(low_level::Op::Store {
                    value,
                    address,
                    offset: 0,
                    flags: MemFlagsData::trusted(),
                }),
                None,
            );
            if let Some(old) = old {
                insert(
                    Op::LowLevel(low_level::Op::Reference {
                        op: low_level::Reference::Release {
                            heap: false,
                            cold: false,
                        },
                        value: old,
                    }),
                    None,
                );
            } else {
                let (view, base, index) = addition_data.unwrap();
                let shape = insert(
                    Op::LowLevel(low_level::Op::Load {
                        ty: types::I32,
                        address: key,
                        offset: CACHE_ADDITION_DESTINATION_OFFSET,
                        flags: MemFlagsData::trusted(),
                    }),
                    Some(Repr::I32),
                )
                .unwrap();
                insert(
                    Op::LowLevel(low_level::Op::Store {
                        value: shape,
                        address: view,
                        offset: SHAPE_OFFSET,
                        flags: MemFlagsData::trusted(),
                    }),
                    None,
                );
                let one = insert(
                    Op::LowLevel(low_level::Op::Integer {
                        ty: types::I64,
                        bits: 1,
                    }),
                    Some(Repr::I64),
                )
                .unwrap();
                let length = insert(
                    Op::LowLevel(low_level::Op::Binary {
                        op: low_level::Binary::Add,
                        left: index,
                        right: one,
                    }),
                    Some(Repr::I64),
                )
                .unwrap();
                insert(
                    Op::LowLevel(low_level::Op::Store {
                        value: length,
                        address: view,
                        offset: LENGTH_OFFSET,
                        flags: MemFlagsData::trusted(),
                    }),
                    None,
                );
                let small = insert(
                    Op::LowLevel(low_level::Op::Unary {
                        op: low_level::Unary::Reduce,
                        ty: types::I8,
                        value: length,
                    }),
                    Some(Repr::I8),
                )
                .unwrap();
                insert(
                    Op::LowLevel(low_level::Op::Store {
                        value: small,
                        address: base,
                        offset: INLINE_PROPERTY_LENGTH_OFFSET,
                        flags: MemFlagsData::trusted(),
                    }),
                    None,
                );
            }
            graph.insts[inst].op = Op::KeepAlive(object);
            plan.implementations[inst] = Implementation {
                lowering: Lowering::Typed(None),
                ..Default::default()
            };
        }
    }
    graph.ownership = Some(plan);
    function.facts.cfg = Cfg::new(graph);
}

fn storage_value(
    graph: &mut Graph,
    plan: &mut super::super::ownership::Plan,
    source: &super::super::InstData,
    block: super::super::Block,
    at: usize,
    op: Op,
    repr: Option<Repr>,
) -> Option<super::super::Value> {
    let (inst, value) = graph.insert(block, at, source.span, op, repr.is_some());
    graph.insts[inst].source_name = source.source_name;
    graph.insts[inst].emission_location = source.emission_location;
    if let Some(value) = value {
        graph.reprs[value] = repr;
    }
    plan.implementations[inst] = Implementation {
        lowering: Lowering::Typed(repr),
        ..Default::default()
    };
    value
}

fn checked_store(
    graph: &Graph,
    inst: super::super::Inst,
    object: super::super::Value,
    key: super::super::Value,
) -> Option<super::super::Inst> {
    let data = &graph.insts[inst];
    if !matches!(data.op, Op::CanOverwrite(..)) {
        return None;
    }
    let super::super::TerminatorKind::Branch(condition, yes, _) =
        &graph.blocks[data.block].terminator.as_ref()?.kind
    else {
        return None;
    };
    if Some(*condition) != data.result() {
        return None;
    }
    for &inst in &graph.blocks[yes.block].insts {
        match graph.insts[inst].op {
            Op::ArrayStore(receiver, position, _) | Op::PropertyStore(receiver, position, _, _)
                if receiver == object && position == key =>
            {
                return Some(inst);
            }
            Op::Convert(..)
            | Op::Retain(_)
            | Op::Release(_)
            | Op::SlotStore(_)
            | Op::SlotClear(_)
            | Op::KeepAlive(_) => {}
            _ => return None,
        }
    }
    None
}

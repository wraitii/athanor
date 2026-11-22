use crate::jit::semantic::JsOp;
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{MemFlagsData, types};

use super::super::cursor::{Cursor, Insert};
use super::super::{ArrayLayout, FunctionState, Graph, Op, Repr, ValueType, low_level};
use super::layout;

pub(in crate::jit::semantic) fn lower_metadata(function: &mut FunctionState) -> bool {
    function.ensure_cfg();
    let graph = &mut function.graph;
    let mut changed = false;
    let mut insts = Vec::new();
    for block in function.facts.cfg.rpo.clone() {
        insts.clone_from(&graph.blocks[block].insts);
        for &inst in &insts {
            let data = &graph.insts[inst];
            let (object, existing_view, ty, offset) = match data.op {
                Op::Javascript(JsOp::ArrayKind(object)) => (
                    object,
                    graph.views.remove(&inst),
                    types::I32,
                    crate::heap::KIND_OFFSET,
                ),
                Op::Javascript(JsOp::IndexedLength(object)) => (
                    object,
                    graph.views.remove(&inst),
                    types::I64,
                    crate::heap::LENGTH_OFFSET,
                ),
                Op::Javascript(JsOp::TypedData(view)) => {
                    let Some(Op::Javascript(JsOp::View(object))) = graph.defining_op(view) else {
                        unreachable!()
                    };
                    (*object, Some(view), types::I64, crate::heap::DATA_OFFSET)
                }
                _ => continue,
            };
            let view = existing_view.unwrap_or_else(|| {
                Cursor::before(graph, inst)
                    .place(Op::Javascript(JsOp::View(object)), true)
                    .1
                    .unwrap()
            });
            graph.set_op(
                inst,
                Op::LowLevel(low_level::Op::Load {
                    ty,
                    address: view,
                    offset,
                    flags: MemFlagsData::trusted(),
                }),
            );
            Cursor::after(graph, inst).add_effect(Op::KeepAlive(object));
            changed = true;
        }
    }
    if changed {
        function.invalidate_values();
    }
    changed
}

pub(in crate::jit::semantic) fn lower_array_loads(function: &mut FunctionState) {
    function.ensure_cfg();
    // Ownership is fixed: preserve result identities and the scheduled reference operations.
    let graph = &mut function.graph;
    let mut insts = Vec::new();
    for block in function.facts.cfg.rpo.clone() {
        insts.clone_from(&graph.blocks[block].insts);
        for &inst in &insts {
            let data = &graph.insts[inst];
            let Op::Javascript(JsOp::ArrayLoad(object, position, layout)) = data.op else {
                continue;
            };
            let existing_view = graph.views.remove(&inst);
            let mut at = Cursor::before(graph, inst);
            let mut typed = |value, value_type| {
                function.facts.set(value, value_type);
                value
            };
            let view = existing_view.unwrap_or_else(|| {
                typed(
                    at.add(Op::Javascript(JsOp::View(object)), Repr::Boxed),
                    ValueType::VIEW,
                )
            });
            let base = typed(
                at.ins().load(
                    types::I64,
                    MemFlagsData::trusted(),
                    view,
                    crate::heap::DATA_OFFSET,
                ),
                ValueType::INT64,
            );
            let shift = typed(at.ins().iconst(types::I64, 3), ValueType::INT64);
            let offset = typed(at.ins().ishl(position, shift), ValueType::INT64);
            let address = typed(at.ins().iadd(base, offset), ValueType::INT64);
            let raw = typed(
                at.ins()
                    .load(types::I64, MemFlagsData::trusted(), address, 0),
                ValueType::INT64,
            );
            let word = if layout == ArrayLayout::Packed {
                raw
            } else {
                let hole = typed(
                    at.ins().iconst(types::I64, crate::heap::HOLE.bits() as i64),
                    ValueType::INT64,
                );
                let absent = typed(at.ins().icmp(IntCC::Equal, raw, hole), ValueType::BOOLEAN);
                let undefined = typed(
                    at.ins()
                        .iconst(types::I64, crate::value::Value::UNDEFINED.bits() as i64),
                    ValueType::INT64,
                );
                typed(at.ins().select(absent, undefined, raw), ValueType::INT64)
            };
            graph.set_op(inst, Op::AsBoxed(word));
        }
    }
}

pub(in crate::jit::semantic) fn lower_stores(function: &mut FunctionState) {
    function.ensure_cfg();
    // Storage owns these references; preserve the function ownership plan while expanding them.
    use crate::heap::{INLINE_PROPERTY_LENGTH_OFFSET, LENGTH_OFFSET, SHAPE_OFFSET};
    use crate::jit::helpers::{CACHE_ADDITION_DESTINATION_OFFSET, CACHE_ADDITION_INDEX_OFFSET};
    function.ensure_values();
    let inline_guards = super::values::inline_objects(&function.graph, &function.facts.cfg);
    let graph = &mut function.graph;
    let mut checked = crate::FxHashMap::default();
    let mut insts = Vec::new();
    for block in function.facts.cfg.rpo.clone() {
        insts.clone_from(&graph.blocks[block].insts);
        for &inst in &insts {
            let data = &graph.insts[inst];
            let (object, key, value, addition) = match data.op {
                Op::Javascript(
                    JsOp::ArrayStore(object, key, value)
                    | JsOp::PropertyStore(object, key, value, _, _),
                ) => (object, key, value, false),
                Op::Javascript(JsOp::PropertyAdd(object, key, value, _)) => {
                    (object, key, value, true)
                }
                Op::Javascript(JsOp::CanOverwrite(object, key, _)) => (object, key, object, false),
                _ => continue,
            };
            let span = data.span;
            let checking = matches!(data.op, Op::Javascript(JsOp::CanOverwrite(..)));
            let reuse = checked.remove(&inst);
            let store = checked_store(graph, inst, object, key);
            let indexed = matches!(
                data.op,
                Op::Javascript(
                    JsOp::ArrayStore(..)
                        | JsOp::PropertyStore(_, _, _, super::super::js::memory::Field::Cells, _)
                        | JsOp::CanOverwrite(_, _, super::super::js::memory::Field::Elements)
                )
            );
            let inline = !indexed
                && (addition
                    || super::values::is_proven(
                        &inline_guards,
                        &function.facts.cfg,
                        block,
                        object,
                    ));
            let constant_index = (!addition && inline)
                .then(|| super::super::ranges::integer_literal(graph, key))
                .flatten()
                .filter(|&index| index >= 0);
            let data_offset = if indexed {
                crate::heap::DATA_OFFSET
            } else {
                crate::heap::PROPERTY_DATA_OFFSET
            };
            let existing_view = graph.views.remove(&inst);
            let mut at = Cursor::before(graph, inst);
            let (address, old, addition_data) = if let Some((address, old)) = reuse {
                (address, Some(old), None)
            } else {
                let view = existing_view
                    .unwrap_or_else(|| at.add(Op::Javascript(JsOp::View(object)), Repr::Boxed));
                let (address, base, index) = if let Some(index_imm) = constant_index {
                    let offset = at.ins().iconst(
                        types::I64,
                        i64::from(crate::heap::INLINE_VALUES_OFFSET) + index_imm * 8,
                    );
                    let address = at.ins().iadd(view, offset);
                    (address, view, key)
                } else {
                    let base = if inline {
                        let offset = at
                            .ins()
                            .iconst(types::I64, i64::from(crate::heap::INLINE_VALUES_OFFSET));
                        at.ins().iadd(view, offset)
                    } else {
                        at.ins()
                            .load(types::I64, MemFlagsData::trusted(), view, data_offset)
                    };
                    let index = if addition {
                        let raw = at.ins().load(
                            types::I32,
                            MemFlagsData::trusted(),
                            key,
                            CACHE_ADDITION_INDEX_OFFSET,
                        );
                        at.ins().uextend(types::I64, raw)
                    } else {
                        key
                    };
                    let shift = at.ins().iconst(types::I64, 3);
                    let offset = at.ins().ishl(index, shift);
                    let address = at.ins().iadd(base, offset);
                    (address, base, index)
                };
                let old = (!addition).then(|| {
                    at.ins()
                        .load(types::I64, MemFlagsData::trusted(), address, 0)
                });
                (address, old, Some((view, base, index)))
            };
            if checking {
                let old = old.unwrap();
                let old_address = layout::heap_address(&mut at, old);
                let present = at.ins().icmp_imm_s(IntCC::NotEqual, old_address, 0);
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
                    span,
                    super::super::TerminatorKind::Branch(
                        present,
                        super::super::Target::new(native),
                        super::super::Target::new(scalar),
                    ),
                );
                let result = graph.blocks[continuation].params[0];
                graph.reprs[result] = Some(Repr::Boolean);
                function.facts.set(result, ValueType::BOOLEAN);
                let yes = Cursor::at_end(graph, scalar, inst).ins().boolean(true);
                graph.terminate(
                    scalar,
                    span,
                    super::super::TerminatorKind::Jump(super::super::Target {
                        block: continuation,
                        args: vec![yes],
                    }),
                );
                let mut at = Cursor::before(graph, inst);
                let kind = at.ins().load(
                    types::I32,
                    MemFlagsData::trusted(),
                    old_address,
                    crate::heap::KIND_OFFSET,
                );
                let mask = at
                    .ins()
                    .iconst(types::I32, crate::heap::kind::NATIVE as i64);
                let native_kind = at.ins().band(kind, mask);
                let zero = at.ins().iconst(types::I32, 0);
                graph.set_op(
                    inst,
                    Op::LowLevel(low_level::Op::IntCompare {
                        condition: IntCC::Equal,
                        left: native_kind,
                        right: zero,
                    }),
                );
                super::super::dispatch::extend_regions(graph, block, first_block);
                continue;
            }
            let types = function.facts.of(value);
            if types == ValueType::default()
                || !types.within(ValueType::PRIMITIVE.without(ValueType::STRING))
            {
                at.add_effect(Op::LowLevel(low_level::Op::Reference {
                    op: low_level::Reference::Retain,
                    value,
                }));
            }
            let word = at.as_word(value);
            at.ins().store(MemFlagsData::trusted(), word, address, 0);
            if let Some(old) = old {
                let old = at.as_boxed(old);
                at.add_effect(Op::LowLevel(low_level::Op::Reference {
                    op: low_level::Reference::Release {
                        heap: false,
                        cold: false,
                    },
                    value: old,
                }));
            } else {
                let (view, base, index) = addition_data.unwrap();
                let shape = at.ins().load(
                    types::I32,
                    MemFlagsData::trusted(),
                    key,
                    CACHE_ADDITION_DESTINATION_OFFSET,
                );
                at.ins()
                    .store(MemFlagsData::trusted(), shape, view, SHAPE_OFFSET);
                let one = at.ins().iconst(types::I64, 1);
                let length = at.ins().iadd(index, one);
                at.ins()
                    .store(MemFlagsData::trusted(), length, view, LENGTH_OFFSET);
                let small = at.ins().ireduce(types::I8, length);
                at.ins().store(
                    MemFlagsData::trusted(),
                    small,
                    base,
                    INLINE_PROPERTY_LENGTH_OFFSET,
                );
            }
            graph.detach_inst(inst);
        }
    }
    function.invalidate_cfg();
}

fn checked_store(
    graph: &Graph,
    inst: super::super::Inst,
    object: super::super::Value,
    key: super::super::Value,
) -> Option<super::super::Inst> {
    let data = &graph.insts[inst];
    if !matches!(data.op, Op::Javascript(JsOp::CanOverwrite(..))) {
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
            Op::Javascript(
                JsOp::ArrayStore(receiver, position, _)
                | JsOp::PropertyStore(receiver, position, ..),
            ) if receiver == object && position == key => {
                return Some(inst);
            }
            Op::Convert(..)
            | Op::Box(_)
            | Op::AsBoxed(_)
            | Op::AsWord(_)
            | Op::Unbox(..)
            | Op::Retain(_)
            | Op::Release(_)
            | Op::SlotStore(_)
            | Op::SlotClear(_) => {}
            _ => return None,
        }
    }
    None
}

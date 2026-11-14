use super::super::{
    Block, BlockData, Cfg, FunctionState, Graph, Inst, InstData, Op, Target, Terminator,
    TerminatorKind, Value, low_level,
};

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
        let position = graph.blocks[data.block]
            .insts
            .iter()
            .position(|&i| i == inst)
            .unwrap();
        let (allocation, cache) = graph.insert(
            data.block,
            position,
            data.span,
            Op::PropertyCache(inst, name.clone()),
            true,
        );
        graph.insts[allocation].source_name = data.source_name;
        let cache = cache.unwrap();
        let first = graph.blocks.len();
        let (block, continuation, fallback) = super::super::dispatch::split(graph, inst);
        graph.insts[inst].op = Op::FillPropertyCache(object, cache, name.clone());
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
    for inst in cached {
        let data = graph.insts[inst].clone();
        let Op::Get(object, name) = &data.op else {
            unreachable!()
        };
        let object = *object;
        let position = graph.blocks[data.block]
            .insts
            .iter()
            .position(|&i| i == inst)
            .unwrap();
        let (allocation, cache) = graph.insert(
            data.block,
            position,
            data.span,
            Op::PropertyCache(inst, name.clone()),
            true,
        );
        graph.insts[allocation].source_name = data.source_name;
        super::super::specialization::properties::lower_cache(
            graph,
            inst,
            cache.unwrap(),
            object,
            name,
        );
    }
    function.invalidate_cfg();
    Ok(changed)
}

fn lower_stores(graph: &mut Graph) -> bool {
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
        let position = graph.blocks[data.block]
            .insts
            .iter()
            .position(|&i| i == inst)
            .unwrap();
        let (allocation, cache) = graph.insert(
            data.block,
            position,
            data.span,
            Op::PropertyCache(inst, name.clone()),
            true,
        );
        graph.insts[allocation].source_name = data.source_name;
        let cache = cache.unwrap();
        let first = graph.blocks.len();
        let (block, done, fallback) = super::super::dispatch::split(graph, inst);
        graph.insts[inst].op = Op::FillPropertyStore(object, cache, name, value, write);
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
            let fast = super::super::dispatch::branch(graph, inst, check, overwrite, fallback);
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
        let fast = super::super::dispatch::branch(graph, inst, overwrite, writable, fallback);
        let (store, _) = graph.append(
            fast,
            data.span,
            Op::PropertyStore(object, index, value, write),
            false,
        );
        graph.views.insert(store, view);
        graph.terminate(fast, data.span, TerminatorKind::Jump(Target::new(done)));
        super::super::dispatch::extend_regions(graph, block, first);
    }
    !stores.is_empty()
}

pub(in crate::jit::semantic) fn lower_accesses(
    function: &mut FunctionState,
) -> super::super::Result<()> {
    use super::super::Repr;
    use crate::jit::helpers::Helper;

    function.ensure_facts();
    let graph = &mut function.graph;
    let accesses: Vec<_> = Cfg::new(graph)
        .rpo
        .into_iter()
        .flat_map(|block| graph.blocks[block].insts.clone())
        .filter(|&inst| {
            matches!(
                graph.insts[inst].op,
                Op::ShapedGet(..)
                    | Op::CachedProperty(..)
                    | Op::IsPropertyCached(..)
                    | Op::FillPropertyCache(..)
                    | Op::PrimePropertyCache(..)
                    | Op::FillPropertyStore(..)
            )
        })
        .collect();
    for inst in accesses {
        let data = graph.insts[inst].clone();
        match data.op {
            Op::ShapedGet(object, _, index) => lower_shaped(graph, inst, object, index),
            Op::CachedProperty(object, cache, builtins)
            | Op::IsPropertyCached(object, cache, builtins) => {
                let validate = matches!(data.op, Op::IsPropertyCached(..));
                let first = graph.blocks.len();
                let (block, continuation, finish) = super::super::dispatch::split(graph, inst);
                graph.remove_inst(inst);
                graph.cold.remove(&finish);
                graph.blocks[finish].terminator = None;
                let mut lowering = PropertyLowering {
                    graph,
                    source: data,
                };
                let (entry, view) =
                    lowering.entry(block, finish, object, cache, builtins, validate);
                let (last, value) = if validate {
                    let zero = lowering.integer(finish, cranelift_codegen::ir::types::I64, 0);
                    let hit = lowering.compare(
                        finish,
                        cranelift_codegen::ir::condcodes::IntCC::NotEqual,
                        entry,
                        zero,
                    );
                    lowering.keep_alive(finish, object);
                    (finish, hit)
                } else {
                    let (last, address) = lowering.slot(finish, view, entry, builtins);
                    let value = lowering.value(last, Op::PropertyLoad(object, address));
                    (last, value)
                };
                lowering.jump(last, continuation, vec![value]);
                super::super::dispatch::extend_regions(lowering.graph, block, first);
            }
            Op::FillPropertyCache(object, cache, _) | Op::PrimePropertyCache(object, cache) => {
                let fill = matches!(data.op, Op::FillPropertyCache(..));
                let contract = graph.contract(&data.op, &function.facts);
                graph.insts[inst].op = Op::HelperCall {
                    helper: if fill {
                        Helper::PropertyGetCached
                    } else {
                        Helper::PropertyCachePrime
                    },
                    args: vec![object, cache],
                    uses: vec![Repr::Boxed, Repr::I64],
                    produces: fill.then_some(Repr::Boxed),
                    contract,
                };
            }
            Op::FillPropertyStore(object, cache, _, value, _) => {
                let contract = graph.contract(&data.op, &function.facts);
                graph.insts[inst].op = Op::HelperCall {
                    helper: Helper::PropertySetCached,
                    args: vec![object, cache, value],
                    uses: vec![Repr::Boxed, Repr::I64, Repr::Boxed],
                    produces: None,
                    contract,
                };
            }
            _ => unreachable!(),
        }
    }
    function.invalidate_cfg();
    function.graph.verify()
}

fn lower_shaped(graph: &mut Graph, inst: Inst, object: Value, index: u32) {
    use cranelift_codegen::ir::{MemFlagsData, types};

    let data = graph.insts[inst].clone();
    let mut at = graph.blocks[data.block]
        .insts
        .iter()
        .position(|&i| i == inst)
        .unwrap();
    let mut insert = |op| {
        let (inst, value) = graph.insert(data.block, at, data.span, op, true);
        graph.insts[inst].source_name = data.source_name;
        graph.insts[inst].emission_location = data.emission_location;
        at += 1;
        value.unwrap()
    };
    let view = insert(Op::View(object));
    let values = insert(Op::LowLevel(low_level::Op::Load {
        ty: types::I64,
        address: view,
        offset: crate::heap::DATA_OFFSET,
        flags: MemFlagsData::trusted(),
    }));
    let offset = insert(Op::LowLevel(low_level::Op::Integer {
        ty: types::I64,
        bits: u64::from(index) * 8,
    }));
    let address = insert(Op::LowLevel(low_level::Op::Binary {
        op: low_level::Binary::Add,
        left: values,
        right: offset,
    }));
    graph.insts[inst].op = Op::PropertyLoad(object, address);
}

struct PropertyLowering<'a> {
    graph: &'a mut Graph,
    source: InstData,
}

impl PropertyLowering<'_> {
    fn value(&mut self, block: Block, op: Op) -> Value {
        let (inst, value) = self.graph.append(block, self.source.span, op, true);
        self.graph.insts[inst].source_name = self.source.source_name;
        self.graph.insts[inst].emission_location = self.source.emission_location;
        value.unwrap()
    }

    fn keep_alive(&mut self, block: Block, object: Value) {
        let (inst, _) = self
            .graph
            .append(block, self.source.span, Op::KeepAlive(object), false);
        self.graph.insts[inst].source_name = self.source.source_name;
        self.graph.insts[inst].emission_location = self.source.emission_location;
    }

    fn integer(&mut self, block: Block, ty: cranelift_codegen::ir::Type, bits: u64) -> Value {
        self.value(block, Op::LowLevel(low_level::Op::Integer { ty, bits }))
    }

    fn load(
        &mut self,
        block: Block,
        ty: cranelift_codegen::ir::Type,
        address: Value,
        offset: i32,
    ) -> Value {
        self.value(
            block,
            Op::LowLevel(low_level::Op::Load {
                ty,
                address,
                offset,
                flags: cranelift_codegen::ir::MemFlagsData::trusted(),
            }),
        )
    }

    fn binary(&mut self, block: Block, op: low_level::Binary, left: Value, right: Value) -> Value {
        self.value(
            block,
            Op::LowLevel(low_level::Op::Binary { op, left, right }),
        )
    }

    fn compare(
        &mut self,
        block: Block,
        condition: cranelift_codegen::ir::condcodes::IntCC,
        left: Value,
        right: Value,
    ) -> Value {
        self.value(
            block,
            Op::LowLevel(low_level::Op::IntCompare {
                condition,
                left,
                right,
            }),
        )
    }

    fn select(&mut self, block: Block, condition: Value, yes: Value, no: Value) -> Value {
        self.value(
            block,
            Op::LowLevel(low_level::Op::Select { condition, yes, no }),
        )
    }

    fn block(&mut self) -> Block {
        self.graph.blocks.push(BlockData::default())
    }

    fn branch(&mut self, block: Block, condition: Value, yes: Target, no: Target) {
        self.graph.terminate(
            block,
            self.source.span,
            TerminatorKind::Branch(condition, yes, no),
        );
    }

    fn jump(&mut self, block: Block, target: Block, args: Vec<Value>) {
        self.graph.terminate(
            block,
            self.source.span,
            TerminatorKind::Jump(Target {
                block: target,
                args,
            }),
        );
    }

    fn view(&mut self, block: Block, object: Value) -> Value {
        use super::super::low_level::Binary;
        use crate::value::{TAG_OBJECT, TAG_SHIFT};
        use cranelift_codegen::ir::{condcodes::IntCC, types};

        let bits = self.value(block, Op::AsWord(object));
        let shift = self.integer(block, types::I64, TAG_SHIFT as u64);
        let tag = self.binary(block, Binary::ShiftRightUnsigned, bits, shift);
        let first = self.integer(block, types::I64, TAG_OBJECT >> TAG_SHIFT);
        let offset = self.binary(block, Binary::Subtract, tag, first);
        let two = self.integer(block, types::I64, 2);
        let heap = self.compare(block, IntCC::UnsignedLessThan, offset, two);
        let mask = self.integer(block, types::I64, (1u64 << TAG_SHIFT) - 1);
        let address = self.binary(block, Binary::And, bits, mask);
        let zero = self.integer(block, types::I64, 0);
        let present = self.compare(block, IntCC::NotEqual, address, zero);
        let valid = self.value(block, Op::BooleanAnd(heap, present));
        let empty = self.integer(block, types::I64, crate::heap::NO_ELEMENTS.as_ptr() as u64);
        self.select(block, valid, address, empty)
    }

    fn entry(
        &mut self,
        mut block: Block,
        done: Block,
        object: Value,
        cache: Value,
        builtins: bool,
        validate: bool,
    ) -> (Value, Value) {
        use crate::heap::{KIND_OFFSET, SHAPE_OFFSET};
        use crate::jit::helpers::{
            CACHE_BUILTIN_KIND_OFFSET, CACHE_ENTRY_OFFSET, CACHE_ENTRY_SIZE, CACHE_HOLDER_OFFSET,
            CACHE_HOLDER_SHAPE_OFFSET, CACHE_LEN_OFFSET, CACHE_MEGAMORPHIC_OFFSET,
            CACHE_OVERFLOW_OFFSET, CACHE_SHAPE_OFFSET, PROPERTY_CACHE_LIMIT,
        };
        use cranelift_codegen::ir::{condcodes::IntCC, types};
        use low_level::Binary;

        let result = self.graph.add_param(done);
        let miss = self.block();
        if validate {
            let megamorphic = self.load(block, types::I32, cache, CACHE_MEGAMORPHIC_OFFSET);
            let lookup = self.block();
            self.branch(block, megamorphic, Target::new(miss), Target::new(lookup));
            block = lookup;
        }
        let view = self.view(block, object);
        let shape = self.load(block, types::I32, view, SHAPE_OFFSET);
        let kind = builtins.then(|| self.load(block, types::I32, view, KIND_OFFSET));
        let len = self.load(block, types::I32, cache, CACHE_LEN_OFFSET);
        let mut overflow = None;
        for index in 0..PROPERTY_CACHE_LIMIT {
            let entry = if index == 0 {
                let offset = self.integer(block, types::I64, CACHE_ENTRY_OFFSET as u64);
                self.binary(block, Binary::Add, cache, offset)
            } else {
                let overflow = *overflow.get_or_insert_with(|| {
                    self.load(block, types::I64, cache, CACHE_OVERFLOW_OFFSET)
                });
                let offset = self.integer(
                    block,
                    types::I64,
                    (index - 1) as u64 * CACHE_ENTRY_SIZE as u64,
                );
                self.binary(block, Binary::Add, overflow, offset)
            };
            let next = self.block();
            let mut shaped_kind = None;
            if let Some(kind) = kind {
                let expected = self.load(block, types::I32, entry, CACHE_BUILTIN_KIND_OFFSET);
                let zero = self.integer(block, types::I32, 0);
                let cached = self.compare(block, IntCC::NotEqual, expected, zero);
                shaped_kind = Some(self.compare(block, IntCC::Equal, expected, zero));
                let matches = self.compare(block, IntCC::Equal, kind, expected);
                let hit = self.value(block, Op::BooleanAnd(cached, matches));
                let shaped = self.block();
                self.branch(
                    block,
                    hit,
                    Target {
                        block: done,
                        args: vec![entry],
                    },
                    Target::new(shaped),
                );
                block = shaped;
            }
            let expected = self.load(block, types::I32, entry, CACHE_SHAPE_OFFSET);
            let mut matches = self.compare(block, IntCC::Equal, shape, expected);
            if let Some(shaped) = shaped_kind {
                matches = self.value(block, Op::BooleanAnd(matches, shaped));
            }
            if validate {
                let holder_check = self.block();
                self.branch(block, matches, Target::new(holder_check), Target::new(next));
                block = holder_check;
                let holder = self.load(block, types::I64, entry, CACHE_HOLDER_OFFSET);
                let zero = self.integer(block, types::I64, 0);
                let own = self.compare(block, IntCC::Equal, holder, zero);
                let holder = self.select(block, own, view, holder);
                let actual = self.load(block, types::I32, holder, SHAPE_OFFSET);
                let expected = self.load(block, types::I32, entry, CACHE_HOLDER_SHAPE_OFFSET);
                matches = self.compare(block, IntCC::Equal, actual, expected);
            }
            self.branch(
                block,
                matches,
                Target {
                    block: done,
                    args: vec![entry],
                },
                Target::new(next),
            );
            block = next;
            if index + 1 < PROPERTY_CACHE_LIMIT {
                let count = self.integer(block, types::I32, (index + 1) as u64);
                let more = self.compare(block, IntCC::UnsignedGreaterThan, len, count);
                let lookup = self.block();
                self.branch(block, more, Target::new(lookup), Target::new(miss));
                block = lookup;
            } else {
                self.jump(block, miss, Vec::new());
            }
        }
        let zero = self.integer(miss, types::I64, 0);
        self.jump(miss, done, vec![zero]);
        (result, view)
    }

    fn slot(
        &mut self,
        mut block: Block,
        view: Value,
        entry: Value,
        builtins: bool,
    ) -> (Block, Value) {
        use crate::heap::DATA_OFFSET;
        use crate::jit::helpers::{
            CACHE_BUILTIN_KIND_OFFSET, CACHE_BUILTIN_VALUE_OFFSET, CACHE_HOLDER_OFFSET,
            CACHE_INDEX_OFFSET,
        };
        use cranelift_codegen::ir::{condcodes::IntCC, types};
        use low_level::{Binary, Op as Low, Unary};

        let ready = builtins.then(|| self.block());
        let address = ready.map(|ready| self.graph.add_param(ready));
        if let Some(ready) = ready {
            let kind = self.load(block, types::I32, entry, CACHE_BUILTIN_KIND_OFFSET);
            let builtin = self.block();
            let shaped = self.block();
            self.branch(block, kind, Target::new(builtin), Target::new(shaped));
            let offset = self.integer(builtin, types::I64, CACHE_BUILTIN_VALUE_OFFSET as u64);
            let slot = self.binary(builtin, Binary::Add, entry, offset);
            self.jump(builtin, ready, vec![slot]);
            block = shaped;
        }
        let holder = self.load(block, types::I64, entry, CACHE_HOLDER_OFFSET);
        let zero = self.integer(block, types::I64, 0);
        let own = self.compare(block, IntCC::Equal, holder, zero);
        let holder = self.select(block, own, view, holder);
        let index = self.load(block, types::I32, entry, CACHE_INDEX_OFFSET);
        let index = self.value(
            block,
            Op::LowLevel(Low::Unary {
                op: Unary::ZeroExtend,
                ty: types::I64,
                value: index,
            }),
        );
        let shift = self.integer(block, types::I64, 3);
        let offset = self.binary(block, Binary::ShiftLeft, index, shift);
        let data = self.load(block, types::I64, holder, DATA_OFFSET);
        let slot = self.binary(block, Binary::Add, data, offset);
        if let Some(ready) = ready {
            self.jump(block, ready, vec![slot]);
            (ready, address.unwrap())
        } else {
            (block, slot)
        }
    }
}

pub(in crate::jit::semantic) fn lower_loads(function: &mut FunctionState) {
    use super::super::low_level;
    use cranelift_codegen::ir::{MemFlagsData, types};

    let graph = &mut function.graph;
    for block in Cfg::new(graph).rpo {
        for inst in graph.blocks[block].insts.clone() {
            let Op::PropertyLoad(_, address) = graph.insts[inst].op else {
                continue;
            };
            let load = low_level::Op::Load {
                ty: types::I64,
                address,
                offset: 0,
                flags: MemFlagsData::trusted(),
            };
            let word =
                super::helpers::insert_before(graph, inst, load, Some(super::super::Repr::I64))
                    .1
                    .unwrap();
            graph.insts[inst].op = Op::AsBoxed(word);
        }
    }
}

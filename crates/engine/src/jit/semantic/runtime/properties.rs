use super::super::cursor::{Cursor, Insert};
use super::super::{
    Block, BlockData, Cfg, FunctionState, Graph, Inst, Op, Repr, Target, Terminator,
    TerminatorKind, Value, ValueType,
};
use super::layout;

pub(in crate::jit::semantic) fn lower(function: &mut FunctionState) -> super::super::Result<bool> {
    function.ensure_cfg();
    let shapes = &function.plan.shapes;
    let graph = &mut function.graph;
    if graph.has_handlers || graph.generator {
        return Ok(false);
    }
    let reads: Vec<_> = function
        .facts
        .cfg
        .rpo
        .iter()
        .filter(|block| !graph.cold.contains(block))
        .flat_map(|&block| graph.blocks[block].insts.iter().copied())
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
        let cache =
            Cursor::before(graph, inst).add_unassigned(Op::PropertyCache(inst, name.clone()));
        let first = graph.blocks.len();
        let (block, continuation, fallback) = super::super::dispatch::split(graph, inst);
        graph.set_op(inst, Op::FillPropertyCache(object, cache, name.clone()));
        let fast = graph.blocks.push(BlockData::default());
        let check = Cursor::at_end(graph, block, inst).add_unassigned(Op::IsIndexed(object));
        graph.blocks[block].terminator = Some(Terminator {
            source_name: data.source_name,
            emission_location: None,
            span: data.span,
            kind: TerminatorKind::Branch(check, Target::new(fast), Target::new(fallback)),
        });
        let value = Cursor::at_end(graph, fast, inst).add_unassigned(Op::IndexedLength(object));
        graph.blocks[fast].terminator = Some(Terminator {
            source_name: data.source_name,
            emission_location: None,
            span: data.span,
            kind: TerminatorKind::Jump(Target {
                block: continuation,
                args: vec![value],
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
        let cache =
            Cursor::before(graph, inst).add_unassigned(Op::PropertyCache(inst, name.clone()));
        super::super::specialization::properties::lower_cache(graph, inst, cache, object, name);
    }
    if changed {
        function.invalidate_cfg();
    }
    Ok(changed)
}

fn lower_stores(graph: &mut Graph) -> bool {
    use super::super::Constant;
    use crate::heap::{KIND_OFFSET, SHAPE_OFFSET, kind};
    use crate::jit::helpers::{
        CACHE_ADDITION_SOURCE_OFFSET, CACHE_OVERWRITE_INDEX_OFFSET, CACHE_OVERWRITE_SHAPE_OFFSET,
    };
    use cranelift_codegen::ir::{MemFlagsData, condcodes::IntCC, types};
    let stores: Vec<_> = Cfg::reverse_postorder(graph)
        .into_iter()
        .filter(|block| !graph.cold.contains(block))
        .flat_map(|block| graph.blocks[block].insts.iter().copied())
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
        let cache =
            Cursor::before(graph, inst).add_unassigned(Op::PropertyCache(inst, name.clone()));
        let first = graph.blocks.len();
        let (block, done, fallback) = super::super::dispatch::split(graph, inst);
        graph.set_op(
            inst,
            Op::FillPropertyStore(object, cache, name, value, write),
        );
        let view = Cursor::at_end(graph, block, inst).add_unassigned(Op::View(object));
        let mut at = Cursor::at_end(graph, block, inst);
        let (overwrite, index) = if let Some(index) = shaped {
            let index = at.add_unassigned(Op::Constant(Constant::Number(index as f64)));
            (block, index)
        } else {
            let flags = MemFlagsData::trusted();
            let mut load = |address, offset| at.ins().load(types::I32, flags, address, offset);
            let actual_kind = load(view, KIND_OFFSET);
            let shape = load(view, SHAPE_OFFSET);
            let addition_shape = load(cache, CACHE_ADDITION_SOURCE_OFFSET);
            let overwrite_shape = load(cache, CACHE_OVERWRITE_SHAPE_OFFSET);
            let index = load(cache, CACHE_OVERWRITE_INDEX_OFFSET);
            let inline_kind = at
                .ins()
                .iconst(types::I32, (kind::OBJECT | kind::INLINE) as i64);
            let inline = at.ins().icmp(IntCC::Equal, actual_kind, inline_kind);
            let matching = at.ins().icmp(IntCC::Equal, shape, addition_shape);
            let addition = at.add_unassigned(Op::BooleanAnd(inline, matching));
            let object_bits = at.ins().iconst(types::I32, kind::OBJECT as i64);
            let bits = at.ins().band(actual_kind, object_bits);
            let is_object = at.ins().icmp(IntCC::Equal, bits, object_bits);
            let matching = at.ins().icmp(IntCC::Equal, shape, overwrite_shape);
            let overwrite = at.add_unassigned(Op::BooleanAnd(is_object, matching));
            let add = graph.blocks.push(BlockData::default());
            let check = graph.blocks.push(BlockData::default());
            graph.terminate(
                block,
                data.span,
                TerminatorKind::Branch(addition, Target::new(add), Target::new(check)),
            );
            let store = Cursor::at_end(graph, add, inst)
                .add_effect(Op::PropertyAdd(object, cache, value, write));
            graph.views.insert(store, view);
            graph.terminate(add, data.span, TerminatorKind::Jump(Target::new(done)));
            let fast = super::super::dispatch::branch(graph, inst, check, overwrite, fallback);
            let index = Cursor::at_end(graph, fast, inst)
                .ins()
                .uextend(types::I64, index);
            (fast, index)
        };
        let mut at = Cursor::at_end(graph, overwrite, inst);
        let (check, writable) = at.place(Op::CanOverwrite(object, index), true);
        graph.views.insert(check, view);
        let writable = writable.unwrap();
        let fast = super::super::dispatch::branch(graph, inst, overwrite, writable, fallback);
        let store = Cursor::at_end(graph, fast, inst)
            .add_effect(Op::PropertyStore(object, index, value, write));
        graph.views.insert(store, view);
        graph.terminate(fast, data.span, TerminatorKind::Jump(Target::new(done)));
        super::super::dispatch::extend_regions(graph, block, first);
    }
    !stores.is_empty()
}

pub(in crate::jit::semantic) fn lower_accesses(
    function: &mut FunctionState,
) -> super::super::Result<()> {
    use crate::jit::helpers::Helper;

    function.ensure_cfg();
    let graph = &function.graph;
    let accesses: Vec<_> = function
        .facts
        .cfg
        .rpo
        .iter()
        .flat_map(|&block| graph.blocks[block].insts.iter().copied())
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
    if accesses.is_empty() {
        return Ok(());
    }
    function.ensure_facts();
    let graph = &mut function.graph;
    let mut entries = crate::FxHashMap::default();
    for inst in accesses {
        let data = graph.insts[inst].clone();
        match data.op {
            Op::ShapedGet(object, _, index) => lower_shaped(graph, inst, object, index),
            Op::CachedProperty(object, cache, builtins)
            | Op::IsPropertyCached(object, cache, builtins) => {
                let validate = matches!(data.op, Op::IsPropertyCached(..));
                let cached = if validate && graph.blocks[data.block].insts.last() == Some(&inst) {
                    match graph.blocks[data.block].terminator.as_ref().map(|term| &term.kind) {
                        Some(TerminatorKind::Branch(condition, yes, no))
                            if Some(*condition) == data.result()
                                && yes.block != no.block
                                && function.facts.cfg.preds[yes.block].len() == 1 =>
                        {
                            graph.blocks[yes.block].insts.first().copied().filter(|&next| {
                                matches!(graph.insts[next].op, Op::CachedProperty(receiver, site, native)
                                    if receiver == object && site == cache && native == builtins)
                            })
                        }
                        _ => None,
                    }
                } else {
                    None
                };
                let first = graph.blocks.len();
                let (block, continuation, finish) = super::super::dispatch::split(graph, inst);
                graph.remove_inst(inst);
                graph.cold.remove(&finish);
                graph.blocks[finish].terminator = None;
                let mut lowering = PropertyLowering { graph, inst };
                let (entry, view) = if let Some((entry, view)) = entries.remove(&inst) {
                    lowering.jump(block, finish, Vec::new());
                    (entry, view)
                } else {
                    lowering.entry(block, finish, object, cache, builtins, validate)
                };
                if let Some(cached) = cached {
                    entries.insert(cached, (entry, view));
                }
                let (last, value) = if validate {
                    let zero = lowering
                        .at(finish)
                        .ins()
                        .iconst(cranelift_codegen::ir::types::I64, 0);
                    let hit = lowering.at(finish).ins().icmp(
                        cranelift_codegen::ir::condcodes::IntCC::NotEqual,
                        entry,
                        zero,
                    );
                    lowering.at(finish).add_effect(Op::KeepAlive(object));
                    let result = lowering.graph.blocks[continuation].params[0];
                    lowering.graph.reprs[result] = Some(Repr::Boolean);
                    (finish, hit)
                } else {
                    let (last, address) = lowering.slot(finish, view, entry, builtins);
                    let value = lowering
                        .at(last)
                        .add_unassigned(Op::PropertyLoad(object, address));
                    (last, value)
                };
                lowering.jump(last, continuation, vec![value]);
                super::super::dispatch::extend_regions(lowering.graph, block, first);
            }
            Op::FillPropertyCache(object, cache, _) | Op::PrimePropertyCache(object, cache) => {
                let fill = matches!(data.op, Op::FillPropertyCache(..));
                let contract = graph.contract(&data.op, &function.facts);
                graph.set_op(
                    inst,
                    Op::HelperCall {
                        helper: if fill {
                            Helper::PropertyGetCached
                        } else {
                            Helper::PropertyCachePrime
                        },
                        args: vec![object, cache],
                        uses: vec![Repr::Boxed, Repr::Word],
                        produces: fill.then_some(Repr::Boxed),
                        result_type: if fill {
                            ValueType::ALL
                        } else {
                            ValueType::default()
                        },
                        effects: contract.effects,
                    },
                );
            }
            Op::FillPropertyStore(object, cache, _, value, _) => {
                let contract = graph.contract(&data.op, &function.facts);
                graph.set_op(
                    inst,
                    Op::HelperCall {
                        helper: Helper::PropertySetCached,
                        args: vec![object, cache, value],
                        uses: vec![Repr::Boxed, Repr::Word, Repr::Boxed],
                        produces: None,
                        result_type: ValueType::default(),
                        effects: contract.effects,
                    },
                );
            }
            _ => unreachable!(),
        }
    }
    function.invalidate_cfg();
    function.graph.verify()
}

fn lower_shaped(graph: &mut Graph, inst: Inst, object: Value, index: u32) {
    use cranelift_codegen::ir::{MemFlagsData, types};

    let mut at = Cursor::before(graph, inst);
    let view = at.add_unassigned(Op::View(object));
    let values = at.ins().load(
        types::I64,
        MemFlagsData::trusted(),
        view,
        crate::heap::DATA_OFFSET,
    );
    let offset = at.ins().iconst(types::I64, i64::from(index) * 8);
    let address = at.ins().iadd(values, offset);
    graph.set_op(inst, Op::PropertyLoad(object, address));
}

struct PropertyLowering<'a> {
    graph: &'a mut Graph,
    inst: Inst,
}

impl PropertyLowering<'_> {
    fn at(&mut self, block: Block) -> Cursor<'_> {
        Cursor::at_end(self.graph, block, self.inst)
    }

    fn block(&mut self) -> Block {
        self.graph.blocks.push(BlockData::default())
    }

    fn branch(&mut self, block: Block, condition: Value, yes: Target, no: Target) {
        let span = self.graph.insts[self.inst].span;
        self.graph
            .terminate(block, span, TerminatorKind::Branch(condition, yes, no));
    }

    fn jump(&mut self, block: Block, target: Block, args: Vec<Value>) {
        let span = self.graph.insts[self.inst].span;
        self.graph.terminate(
            block,
            span,
            TerminatorKind::Jump(Target {
                block: target,
                args,
            }),
        );
    }

    fn view(&mut self, block: Block, object: Value) -> Value {
        let mut at = self.at(block);
        let bits = at.as_word(object);
        layout::heap_view(&mut at, bits)
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
        use cranelift_codegen::ir::{MemFlagsData, condcodes::IntCC, types};

        let flags = MemFlagsData::trusted();
        let result = self.graph.add_param(done);
        self.graph.reprs[result] = Some(Repr::Word);
        let miss = self.block();
        let matched = validate.then(|| {
            let block = self.block();
            let entry = self.graph.add_param(block);
            self.graph.reprs[entry] = Some(Repr::Word);
            (block, entry)
        });
        let view = self.view(block, object);
        if validate {
            let megamorphic =
                self.at(block)
                    .ins()
                    .load(types::I32, flags, cache, CACHE_MEGAMORPHIC_OFFSET);
            let lookup = self.block();
            self.branch(block, megamorphic, Target::new(miss), Target::new(lookup));
            block = lookup;
        }
        let shape = self
            .at(block)
            .ins()
            .load(types::I32, flags, view, SHAPE_OFFSET);
        let kind = builtins.then(|| {
            self.at(block)
                .ins()
                .load(types::I32, flags, view, KIND_OFFSET)
        });
        // Unused overflow entries carry DICTIONARY and participate in the search.
        let dictionary = self
            .at(block)
            .ins()
            .iconst(types::I32, crate::heap::DICTIONARY as i64);
        let cacheable = self
            .at(block)
            .ins()
            .icmp(IntCC::NotEqual, shape, dictionary);
        if !builtins {
            let lookup = self.block();
            self.branch(block, cacheable, Target::new(lookup), Target::new(miss));
            block = lookup;
        }
        let mut overflow = None;
        for index in 0..PROPERTY_CACHE_LIMIT {
            let entry = if index == 0 {
                let offset = self
                    .at(block)
                    .ins()
                    .iconst(types::I64, CACHE_ENTRY_OFFSET as i64);
                self.at(block).ins().iadd(cache, offset)
            } else {
                let overflow = *overflow.get_or_insert_with(|| {
                    self.at(block)
                        .ins()
                        .load(types::I64, flags, cache, CACHE_OVERFLOW_OFFSET)
                });
                let bits = (index - 1) as u64 * CACHE_ENTRY_SIZE as u64;
                let offset = self.at(block).ins().iconst(types::I64, bits as i64);
                self.at(block).ins().iadd(overflow, offset)
            };
            let next = self.block();
            let mut shaped_kind = None;
            if let Some(kind) = kind {
                let expected =
                    self.at(block)
                        .ins()
                        .load(types::I32, flags, entry, CACHE_BUILTIN_KIND_OFFSET);
                let zero = self.at(block).ins().iconst(types::I32, 0);
                let cached = self.at(block).ins().icmp(IntCC::NotEqual, expected, zero);
                let shaped = self.at(block).ins().icmp(IntCC::Equal, expected, zero);
                shaped_kind = Some(
                    self.at(block)
                        .add_unassigned(Op::BooleanAnd(shaped, cacheable)),
                );
                let matches = self.at(block).ins().icmp(IntCC::Equal, kind, expected);
                let hit = self
                    .at(block)
                    .add_unassigned(Op::BooleanAnd(cached, matches));
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
            let expected = self
                .at(block)
                .ins()
                .load(types::I32, flags, entry, CACHE_SHAPE_OFFSET);
            let mut matches = self.at(block).ins().icmp(IntCC::Equal, shape, expected);
            if let Some(shaped) = shaped_kind {
                matches = self
                    .at(block)
                    .add_unassigned(Op::BooleanAnd(matches, shaped));
            }
            self.branch(
                block,
                matches,
                Target {
                    block: matched.map_or(done, |(block, _)| block),
                    args: vec![entry],
                },
                Target::new(next),
            );
            block = next;
            if index == 0 {
                let len = self
                    .at(block)
                    .ins()
                    .load(types::I32, flags, cache, CACHE_LEN_OFFSET);
                let one = self.at(block).ins().iconst(types::I32, 1);
                let more = self
                    .at(block)
                    .ins()
                    .icmp(IntCC::UnsignedGreaterThan, len, one);
                let lookup = self.block();
                self.branch(block, more, Target::new(lookup), Target::new(miss));
                block = lookup;
            } else if index + 1 == PROPERTY_CACHE_LIMIT {
                self.jump(block, miss, Vec::new());
            }
        }
        if let Some((block, entry)) = matched {
            let holder = self
                .at(block)
                .ins()
                .load(types::I64, flags, entry, CACHE_HOLDER_OFFSET);
            let zero = self.at(block).ins().iconst(types::I64, 0);
            let own = self.at(block).ins().icmp(IntCC::Equal, holder, zero);
            let inherited = self.block();
            self.branch(
                block,
                own,
                Target {
                    block: done,
                    args: vec![entry],
                },
                Target::new(inherited),
            );
            let block = inherited;
            let actual = self
                .at(block)
                .ins()
                .load(types::I32, flags, holder, SHAPE_OFFSET);
            let expected =
                self.at(block)
                    .ins()
                    .load(types::I32, flags, entry, CACHE_HOLDER_SHAPE_OFFSET);
            let valid = self.at(block).ins().icmp(IntCC::Equal, actual, expected);
            self.branch(
                block,
                valid,
                Target {
                    block: done,
                    args: vec![entry],
                },
                Target::new(miss),
            );
        }
        let zero = self.at(miss).ins().iconst(types::I64, 0);
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
        use cranelift_codegen::ir::{MemFlagsData, condcodes::IntCC, types};

        let flags = MemFlagsData::trusted();
        let ready = builtins.then(|| self.block());
        let address = ready.map(|ready| {
            let address = self.graph.add_param(ready);
            self.graph.reprs[address] = Some(Repr::Word);
            address
        });
        if let Some(ready) = ready {
            let kind =
                self.at(block)
                    .ins()
                    .load(types::I32, flags, entry, CACHE_BUILTIN_KIND_OFFSET);
            let builtin = self.block();
            let shaped = self.block();
            self.branch(block, kind, Target::new(builtin), Target::new(shaped));
            let offset = self
                .at(builtin)
                .ins()
                .iconst(types::I64, CACHE_BUILTIN_VALUE_OFFSET as i64);
            let slot = self.at(builtin).ins().iadd(entry, offset);
            self.jump(builtin, ready, vec![slot]);
            block = shaped;
        }
        let holder = self
            .at(block)
            .ins()
            .load(types::I64, flags, entry, CACHE_HOLDER_OFFSET);
        let zero = self.at(block).ins().iconst(types::I64, 0);
        let own = self.at(block).ins().icmp(IntCC::Equal, holder, zero);
        let holder = self.at(block).ins().select(own, view, holder);
        let index = self
            .at(block)
            .ins()
            .load(types::I32, flags, entry, CACHE_INDEX_OFFSET);
        let index = self.at(block).ins().uextend(types::I64, index);
        let shift = self.at(block).ins().iconst(types::I64, 3);
        let offset = self.at(block).ins().ishl(index, shift);
        let data = self
            .at(block)
            .ins()
            .load(types::I64, flags, holder, DATA_OFFSET);
        let slot = self.at(block).ins().iadd(data, offset);
        if let Some(ready) = ready {
            self.jump(block, ready, vec![slot]);
            (ready, address.unwrap())
        } else {
            (block, slot)
        }
    }
}

pub(in crate::jit::semantic) fn lower_loads(function: &mut FunctionState) {
    use cranelift_codegen::ir::{MemFlagsData, types};

    function.ensure_cfg();
    let graph = &mut function.graph;
    for &block in &function.facts.cfg.rpo {
        for inst in graph.blocks[block].insts.clone() {
            let Op::PropertyLoad(_, address) = graph.insts[inst].op else {
                continue;
            };
            let word = Cursor::before(graph, inst).ins().load(
                types::I64,
                MemFlagsData::trusted(),
                address,
                0,
            );
            graph.set_op(inst, Op::AsBoxed(word));
        }
    }
}

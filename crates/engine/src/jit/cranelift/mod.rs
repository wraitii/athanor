mod legalize;
pub(super) mod stubs;

pub(in crate::jit) use legalize::prepare;
#[cfg(test)]
pub(in crate::jit) use legalize::split_indexed_loads;

use cranelift_codegen::entity::SecondaryMap;
use cranelift_codegen::ir::{
    BlockArg, InstBuilder, StackSlotData, StackSlotKind, Value as ClifValue, types,
};
use cranelift_frontend::FunctionBuilder;
use cranelift_jit::JITModule;
use cranelift_module::Module;

use crate::error::Error;
use crate::jit::semantic::Repr;
use crate::jit::semantic::{
    Cfg, Graph, MemoryFormat, MemoryOp, Op as GraphOp, TerminatorKind, Value,
    low_level::{Binary, CallTarget, Op, Unary},
};

pub(in crate::jit) fn emit(
    builder: &mut FunctionBuilder<'_>,
    module: &mut JITModule,
    graph: &Graph,
    mut annotations: Option<&mut crate::jit::clif::Annotations>,
) -> Result<(), Error> {
    let mut blocks = SecondaryMap::new();
    for block in graph.blocks.keys() {
        blocks[block] = Some(builder.create_block());
    }
    let mut values: SecondaryMap<Value, Option<ClifValue>> = SecondaryMap::new();
    for (block, data) in &graph.blocks {
        let target = blocks[block].unwrap();
        for &param in &data.params {
            let ty = graph.reprs[param]
                .map(Repr::machine)
                .ok_or_else(|| Error::Compile(format!("missing scalar type for {param}")))?;
            values[param] = Some(builder.append_block_param(target, ty));
        }
    }
    let slots: Vec<_> = graph
        .low_level_slots
        .iter()
        .map(|slot| {
            builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                slot.size,
                slot.align_shift,
            ))
        })
        .collect();
    for block in Cfg::new(graph).rpo {
        builder.switch_to_block(blocks[block].unwrap());
        if graph.cold.contains(&block) {
            builder.set_cold_block(blocks[block].unwrap());
        }
        for &inst in &graph.blocks[block].insts {
            let data = &graph.insts[inst];
            builder.set_srcloc(data.emission_location.map_or_else(
                cranelift_codegen::ir::SourceLoc::default,
                cranelift_codegen::ir::SourceLoc::new,
            ));
            let get = |value: Value| values[value].expect("low-level value must dominate its use");
            if let GraphOp::Memory(memory) = &data.op {
                let emitted = match memory {
                    MemoryOp::Address {
                        base,
                        index,
                        format,
                    } => {
                        let offset = builder.ins().imul_imm_s(get(*index), format.bytes() as i64);
                        builder.ins().iadd(get(*base), offset)
                    }
                    MemoryOp::Load { address, format } => {
                        let ty = match format {
                            MemoryFormat::U8 | MemoryFormat::I8 => types::I8,
                            MemoryFormat::U16 | MemoryFormat::I16 => types::I16,
                            MemoryFormat::U32 | MemoryFormat::I32 => types::I32,
                            MemoryFormat::F32 => types::F32,
                            MemoryFormat::F64 => types::F64,
                        };
                        builder.ins().load(
                            ty,
                            cranelift_codegen::ir::MemFlagsData::trusted(),
                            get(*address),
                            0,
                        )
                    }
                    MemoryOp::IndexedLoad { .. } => {
                        return Err(Error::Compile(
                            "indexed load reached Cranelift emission".into(),
                        ));
                    }
                };
                let [result] = data.results.as_slice() else {
                    return Err(Error::Compile(format!("wrong result count at {inst}")));
                };
                values[*result] = Some(emitted);
                continue;
            }
            let GraphOp::LowLevel(op) = &data.op else {
                return Err(Error::Compile(format!("non-low-level operation at {inst}")));
            };
            let result = match op {
                Op::Integer { ty, bits } => vec![builder.ins().iconst(*ty, *bits as i64)],
                Op::Float64(bits) => vec![builder.ins().f64const(f64::from_bits(*bits))],
                Op::Unary { op, ty, value } => {
                    let value = get(*value);
                    vec![match op {
                        Unary::BitNot => builder.ins().bnot(value),
                        Unary::FloatNeg => builder.ins().fneg(value),
                        Unary::FloatAbs => builder.ins().fabs(value),
                        Unary::Ceil => builder.ins().ceil(value),
                        Unary::Floor => builder.ins().floor(value),
                        Unary::Trunc => builder.ins().trunc(value),
                        Unary::Sqrt => builder.ins().sqrt(value),
                        Unary::Reduce => builder.ins().ireduce(*ty, value),
                        Unary::SignExtend => builder.ins().sextend(*ty, value),
                        Unary::ZeroExtend => builder.ins().uextend(*ty, value),
                        Unary::Bitcast => builder.ins().bitcast(
                            *ty,
                            cranelift_codegen::ir::MemFlagsData::new(),
                            value,
                        ),
                        Unary::FloatPromote => builder.ins().fpromote(*ty, value),
                        Unary::FloatDemote => builder.ins().fdemote(*ty, value),
                        Unary::FloatFromSigned => builder.ins().fcvt_from_sint(*ty, value),
                        Unary::FloatFromUnsigned => builder.ins().fcvt_from_uint(*ty, value),
                        Unary::SignedSaturatingFromFloat => {
                            builder.ins().fcvt_to_sint_sat(*ty, value)
                        }
                    }]
                }
                Op::Binary {
                    op, left, right, ..
                } => {
                    let (left, right) = (get(*left), get(*right));
                    vec![match op {
                        Binary::Add => builder.ins().iadd(left, right),
                        Binary::Subtract => builder.ins().isub(left, right),
                        Binary::Multiply => builder.ins().imul(left, right),
                        Binary::And => builder.ins().band(left, right),
                        Binary::Or => builder.ins().bor(left, right),
                        Binary::Xor => builder.ins().bxor(left, right),
                        Binary::ShiftLeft => builder.ins().ishl(left, right),
                        Binary::ShiftRightSigned => builder.ins().sshr(left, right),
                        Binary::ShiftRightUnsigned => builder.ins().ushr(left, right),
                        Binary::FloatAdd => builder.ins().fadd(left, right),
                        Binary::FloatSubtract => builder.ins().fsub(left, right),
                        Binary::FloatMultiply => builder.ins().fmul(left, right),
                        Binary::FloatDivide => builder.ins().fdiv(left, right),
                        Binary::FloatMin => builder.ins().fmin(left, right),
                        Binary::FloatMax => builder.ins().fmax(left, right),
                        Binary::BooleanAnd
                        | Binary::BooleanOr
                        | Binary::BooleanXor
                        | Binary::BooleanEqual => unreachable!("unexpanded boolean operation"),
                    }]
                }
                Op::IntCompare {
                    condition,
                    left,
                    right,
                } => vec![builder.ins().icmp(*condition, get(*left), get(*right))],
                Op::FloatCompare {
                    condition,
                    left,
                    right,
                } => vec![builder.ins().fcmp(*condition, get(*left), get(*right))],
                Op::Select { condition, yes, no } => {
                    vec![builder.ins().select(get(*condition), get(*yes), get(*no))]
                }
                Op::Load {
                    ty,
                    address,
                    offset,
                    flags,
                } => vec![builder.ins().load(*ty, *flags, get(*address), *offset)],
                Op::AtomicLoad { ty, address, flags } => {
                    vec![builder.ins().atomic_load(*ty, *flags, get(*address))]
                }
                Op::Store {
                    value,
                    address,
                    offset,
                    flags,
                } => {
                    builder
                        .ins()
                        .store(*flags, get(*value), get(*address), *offset);
                    Vec::new()
                }
                Op::StackAddress { slot, offset, ty } => {
                    vec![builder.ins().stack_addr(*ty, slots[slot.0], *offset)]
                }
                Op::StackPointer { ty } => vec![builder.ins().get_stack_pointer(*ty)],
                Op::Call {
                    target,
                    signature,
                    args,
                    helper,
                } => {
                    let args: Vec<_> = args.iter().map(|&arg| get(arg)).collect();
                    let call = match target {
                        CallTarget::Direct(id) => {
                            if &module.declarations().get_function_decl(*id).signature != signature
                            {
                                return Err(Error::Compile(
                                    "direct call signature mismatch".into(),
                                ));
                            }
                            let func = module.declare_func_in_func(*id, builder.func);
                            builder.ins().call(func, &args)
                        }
                        CallTarget::Indirect(value) => {
                            let signature = builder.import_signature(signature.clone());
                            builder.ins().call_indirect(signature, get(*value), &args)
                        }
                    };
                    if let (Some(annotations), Some(helper)) = (annotations.as_deref_mut(), helper)
                    {
                        annotations.helpers.insert(call, *helper);
                    }
                    builder.inst_results(call).to_vec()
                }
                Op::Cleanup { .. }
                | Op::Reference { .. }
                | Op::FrameStore { .. } => {
                    return Err(Error::Compile("unlowered runtime operation".into()));
                }
                Op::VariableRead => {
                    return Err(Error::Compile("unresolved low-level variable".into()));
                }
                Op::ExtractBits { .. } | Op::InRange { .. } | Op::TestMask { .. } => {
                    return Err(Error::Compile("unexpanded bit operation".into()));
                }
            };
            if result.len() != data.results.len() {
                return Err(Error::Compile(format!("wrong result count at {inst}")));
            }
            for (&value, emitted) in data.results.iter().zip(result) {
                values[value] = Some(emitted);
            }
        }
        builder.set_srcloc(
            graph.blocks[block]
                .terminator
                .as_ref()
                .and_then(|term| term.emission_location)
                .map_or_else(
                    cranelift_codegen::ir::SourceLoc::default,
                    cranelift_codegen::ir::SourceLoc::new,
                ),
        );
        let terminator = &graph.blocks[block]
            .terminator
            .as_ref()
            .ok_or_else(|| Error::Compile(format!("missing terminator at {block}")))?
            .kind;
        let get = |value: Value| values[value].expect("low-level value must dominate its use");
        let target = |target: &crate::jit::semantic::Target| -> (cranelift_codegen::ir::Block, Vec<BlockArg>) { (blocks[target.block].unwrap(), target.args.iter().map(|&value| get(value).into()).collect()) };
        match terminator {
            TerminatorKind::Jump(to) => {
                let (block, args) = target(to);
                builder.ins().jump(block, &args);
            }
            TerminatorKind::Branch(condition, yes, no) => {
                let (yes, yes_args) = target(yes);
                let (no, no_args) = target(no);
                builder
                    .ins()
                    .brif(get(*condition), yes, &yes_args, no, &no_args);
            }
            TerminatorKind::ReturnValues(results) => {
                let results: Vec<_> = results.iter().map(|&value| get(value)).collect();
                builder.ins().return_(&results);
            }
            TerminatorKind::TailCall {
                target: CallTarget::Direct(id),
                signature,
                args,
            } => {
                if &module.declarations().get_function_decl(*id).signature != signature {
                    return Err(Error::Compile("direct tail-call signature mismatch".into()));
                }
                let args: Vec<_> = args.iter().map(|&value| get(value)).collect();
                let function = module.declare_func_in_func(*id, builder.func);
                builder.ins().return_call(function, &args);
            }
            TerminatorKind::TailCall {
                target: CallTarget::Indirect(value),
                signature,
                args,
            } => {
                let args: Vec<_> = args.iter().map(|&value| get(value)).collect();
                let signature = builder.import_signature(signature.clone());
                builder
                    .ins()
                    .return_call_indirect(signature, get(*value), &args);
            }
            _ => {
                return Err(Error::Compile(format!(
                    "unsupported low-level terminator at {block}"
                )));
            }
        }
    }
    Ok(())
}

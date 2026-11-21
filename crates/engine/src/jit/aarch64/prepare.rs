use cranelift_codegen::entity::SecondaryMap;
use cranelift_codegen::ir::{ArgumentExtension, Signature, types};
use cranelift_codegen::isa::CallConv;

use super::Unsupported;
use super::abi::{self, Assignment, Location};
use super::encode::{self as e, Extend};
use super::inst;
use crate::jit::alloc::RegClass;
use crate::jit::alloc::graph::{Hint, Input};
use crate::jit::semantic::{Block, Cfg, Graph, Inst, Op, Repr, TerminatorKind, Value};

#[derive(Clone)]
pub(super) struct Call {
    pub args: Assignment,
    pub returns: Vec<Location>,
    pub extensions: Vec<(u8, Extend)>,
    pub callee_pops: bool,
}

impl Call {
    fn new(signature: &Signature) -> Result<Self, Unsupported> {
        let args = abi::assign(signature.call_conv, &signature.params)?;
        let returns = abi::assign_returns(signature.call_conv, &signature.returns)?;
        let mut extensions = Vec::new();
        if signature.call_conv == CallConv::AppleAarch64 {
            for (param, location) in signature.params.iter().zip(&args.locations) {
                let Location::Reg { reg, .. } = *location else {
                    continue;
                };
                let extend = match (param.extension, param.value_type) {
                    (ArgumentExtension::Uext, types::I8) => Extend::Uxtb,
                    (ArgumentExtension::Uext, types::I16) => Extend::Uxth,
                    (ArgumentExtension::Sext, types::I8) => Extend::Sxtb,
                    (ArgumentExtension::Sext, types::I16) => Extend::Sxth,
                    _ => continue,
                };
                extensions.push((reg, extend));
            }
        }
        Ok(Self {
            args,
            returns,
            extensions,
            callee_pops: signature.call_conv == CallConv::Tail,
        })
    }
}

pub(super) struct Prepared {
    pub cfg: Cfg,
    pub order: Vec<Block>,
    pub input: Input,
    pub constants: SecondaryMap<Value, Option<u64>>,
    pub entry: Vec<Location>,
    pub returns: Vec<Location>,
    pub calls: SecondaryMap<Inst, Option<Call>>,
    pub tails: SecondaryMap<Block, Option<Call>>,
    pub incoming: u32,
    pub outgoing: u32,
    pub tail_args: u32,
    pub pop: u32,
}

pub(super) fn run(graph: &Graph) -> Result<Prepared, Unsupported> {
    let signature = graph
        .low_level_signature
        .as_ref()
        .ok_or_else(|| Unsupported("missing signature".into()))?;
    let entry = abi::assign(signature.call_conv, &signature.params)?;
    let returns = abi::assign_returns(signature.call_conv, &signature.returns)?;
    let cfg = Cfg::new(graph);
    let (cold, mut order): (Vec<_>, Vec<_>) = cfg
        .rpo
        .iter()
        .copied()
        .partition(|&block| graph.cold.contains(&block) && block != graph.entry);
    order.extend(cold);
    let mut prepared = Prepared {
        cfg,
        order,
        input: Input::default(),
        constants: SecondaryMap::new(),
        incoming: entry.stack_size,
        outgoing: 0,
        tail_args: entry.stack_size,
        pop: 0,
        entry: entry.locations,
        returns,
        calls: SecondaryMap::new(),
        tails: SecondaryMap::new(),
    };
    for value in graph.values.keys() {
        prepared.input.classes[value] =
            graph.reprs[value].map(Repr::machine).and_then(RegClass::of);
    }
    hints(
        &mut prepared.input,
        &graph.blocks[graph.entry].params,
        &prepared.entry,
        graph.entry,
    );
    for &block in &prepared.order {
        for &id in &graph.blocks[block].insts {
            let data = &graph.insts[id];
            let Op::Aarch64(op) = &data.op else {
                return Err(Unsupported("unselected operation".into()));
            };
            match op {
                inst::Op::Vector(vector) => {
                    for &value in &data.results {
                        prepared.input.classes[value] = Some(RegClass::Vector);
                    }
                    if let inst::Vector::MultiplyAdd { accumulator, .. } = vector {
                        prepared
                            .input
                            .copies
                            .push((*accumulator, data.results[0], block));
                    }
                }
                inst::Op::Constant { bits, class } => {
                    let value = data.results[0];
                    prepared.constants[value] = Some(*bits);
                    prepared.input.remat[value] = remat_cost(*bits, *class);
                }
                inst::Op::Copy { value } => {
                    prepared.input.copies.push((*value, data.results[0], block))
                }
                inst::Op::Call {
                    signature, args, ..
                } => {
                    let call = Call::new(signature)?;
                    prepared.outgoing = prepared.outgoing.max(call.args.stack_size);
                    hints(&mut prepared.input, args, &call.args.locations, block);
                    hints(&mut prepared.input, &data.results, &call.returns, block);
                    prepared.input.calls[id] = true;
                    prepared.calls[id] = Some(call);
                }
                _ => {}
            }
        }
        match &graph.blocks[block]
            .terminator
            .as_ref()
            .ok_or_else(|| Unsupported("missing terminator".into()))?
            .kind
        {
            TerminatorKind::ReturnValues(values) => {
                hints(&mut prepared.input, values, &prepared.returns, block)
            }
            TerminatorKind::TailCall {
                signature: callee,
                args,
                ..
            } => {
                if signature.call_conv != CallConv::Tail {
                    return Err(Unsupported(format!(
                        "tail call from {}",
                        signature.call_conv
                    )));
                }
                let call = Call::new(callee)?;
                prepared.tail_args = prepared.tail_args.max(call.args.stack_size);
                hints(&mut prepared.input, args, &call.args.locations, block);
                prepared.tails[block] = Some(call);
            }
            TerminatorKind::Jump(_) | TerminatorKind::Aarch64Branch(..) => {}
            _ => return Err(Unsupported("unselected terminator".into())),
        }
    }
    prepared.pop = if signature.call_conv == CallConv::Tail {
        prepared.tail_args
    } else {
        0
    };
    Ok(prepared)
}

fn hints(input: &mut Input, values: &[Value], locations: &[Location], block: Block) {
    for (&value, location) in values.iter().zip(locations) {
        if let Location::Reg { class, reg } = *location {
            input.hints.push(Hint {
                value,
                class,
                reg,
                block,
            });
        }
    }
}

fn remat_cost(bits: u64, class: RegClass) -> u32 {
    if class == RegClass::Float && (bits == 0 || e::float_imm8(bits).is_some()) {
        return 1;
    }
    let mut count = 0;
    e::load_imm(0, bits, |_| count += 1);
    count + u32::from(class == RegClass::Float)
}

use std::collections::BTreeMap;

use cranelift_codegen::ir::{Inst, InstBuilder, InstructionData, Opcode, ValueDef, types};
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Module};

use crate::heap::MathFunction;

use super::effects::{self, GLOBAL_WRITE, HEAP_WRITE, Reentry};
use super::helpers::Helper;
use super::numeric::FunctionBody;

pub(super) const METHODS: &[(&str, MathFunction)] = &[
    ("ceil", MathFunction::Ceil),
    ("floor", MathFunction::Floor),
    ("min", MathFunction::Min),
    ("max", MathFunction::Max),
    ("sqrt", MathFunction::Sqrt),
];

pub(super) struct Candidate {
    calls: Vec<(Inst, Inst, Inst, usize)>,
    mask: u64,
    receiver_property: Option<(String, Vec<Inst>)>,
}

pub(super) fn analyze(body: &FunctionBody) -> Option<Candidate> {
    let function = &body.function;
    let annotations = &body.annotations;
    let mut calls = Vec::new();
    let mut mask = 0;
    let mut properties: BTreeMap<String, Vec<Inst>> = BTreeMap::new();
    let entry = function.layout.entry_block()?;
    let receiver = function.dfg.block_params(entry)[2];
    for block in function.layout.blocks() {
        for inst in function.layout.block_insts(block) {
            let opcode = function.dfg.insts[inst].opcode();
            if let Some(&helper) = annotations.helpers.get(&inst) {
                if helper == Helper::PropertyGet {
                    let args = function.dfg.inst_args(inst);
                    if function.dfg.resolve_aliases(args[1]) == receiver {
                        if let Some(name) = annotations.names.get(&inst) {
                            properties.entry(name.clone()).or_default().push(inst);
                        }
                    }
                }
                if helper == Helper::Call {
                    let name = effects::math_target(function, annotations, inst)?;
                    let index = METHODS.iter().position(|(method, _)| *method == name)?;
                    let args = function.dfg.inst_args(inst);
                    let property = effects::defining_helper(
                        function,
                        annotations,
                        args[1],
                        Helper::PropertyGet,
                    )?;
                    let global = effects::defining_helper(
                        function,
                        annotations,
                        args[2],
                        Helper::GlobalGet,
                    )?;
                    calls.push((inst, property, global, index));
                    mask |= 1 << index;
                } else {
                    let effect = effects::effects(helper);
                    if effect.reentry != Reentry::Closed
                        || effect.flags & (HEAP_WRITE | GLOBAL_WRITE) != 0
                    {
                        return None;
                    }
                }
            } else if opcode.is_call() {
                return None;
            } else if opcode.can_store() {
                match function.dfg.insts[inst] {
                    InstructionData::StackStore { .. } => {}
                    InstructionData::Store { args, .. } => {
                        let address = function.dfg.resolve_aliases(args[1]);
                        let ValueDef::Result(address_inst, _) = function.dfg.value_def(address)
                        else {
                            return None;
                        };
                        if function.dfg.insts[address_inst].opcode() != Opcode::StackAddr {
                            return None;
                        }
                    }
                    _ => return None,
                }
            }
        }
    }
    if calls.is_empty() {
        return None;
    }
    for &(call, property, global, _) in &calls {
        let property_value = function.dfg.inst_results(property)[0];
        let global_value = function.dfg.inst_results(global)[0];
        for block in function.layout.blocks() {
            for inst in function.layout.block_insts(block) {
                for value in function.dfg.inst_values(inst) {
                    let value = function.dfg.resolve_aliases(value);
                    if value == property_value && inst != call {
                        return None;
                    }
                    if value == global_value && inst != property && inst != call {
                        return None;
                    }
                }
            }
        }
    }
    let receiver_property = properties
        .into_iter()
        .filter(|(_, sites)| sites.len() > 1)
        .max_by(|left, right| {
            left.1
                .len()
                .cmp(&right.1.len())
                .then_with(|| right.0.cmp(&left.0))
        });
    Some(Candidate {
        calls,
        mask,
        receiver_property,
    })
}

pub(super) fn specialize(
    body: &mut FunctionBody,
    candidate: &Candidate,
    module: &mut JITModule,
    direct_id: FuncId,
) {
    let direct = module.declare_func_in_func(direct_id, &mut body.function);
    if let Some((_, sites)) = &candidate.receiver_property {
        body.function
            .signature
            .params
            .push(cranelift_codegen::ir::AbiParam::new(types::I64));
        let entry = body.function.layout.entry_block().unwrap();
        let cached = body.function.dfg.append_block_param(entry, types::I64);
        body.annotations.invocation_roots.insert(cached);
        for &site in sites {
            body.function.dfg.replace(site).iadd_imm(cached, 0);
            body.annotations.helpers.remove(&site);
            body.annotations.names.remove(&site);
        }
    }
    for &(call, property, global, index) in &candidate.calls {
        body.function.dfg.replace(global).iconst(types::I64, 0);
        body.function
            .dfg
            .replace(property)
            .iconst(types::I64, index as i64);
        body.annotations.helpers.remove(&global);
        body.annotations.helpers.remove(&property);
        body.annotations.names.remove(&global);
        body.annotations.names.remove(&property);
        if let InstructionData::Call { func_ref, .. } = &mut body.function.dfg.insts[call] {
            *func_ref = direct;
        }
        body.annotations.helpers.insert(call, Helper::MathDirect);
    }
}

pub(super) fn mask(candidate: &Candidate) -> u64 {
    candidate.mask
}

pub(super) fn count(candidate: &Candidate) -> usize {
    candidate.calls.len()
}

pub(super) fn target_names(candidate: &Candidate) -> String {
    METHODS
        .iter()
        .enumerate()
        .filter_map(|(index, (name, _))| (candidate.mask & (1 << index) != 0).then_some(*name))
        .collect::<Vec<_>>()
        .join(", ")
}

pub(super) fn receiver_property(candidate: &Candidate) -> Option<(&str, usize)> {
    candidate
        .receiver_property
        .as_ref()
        .map(|(name, sites)| (name.as_str(), sites.len()))
}

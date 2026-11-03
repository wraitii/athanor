use std::collections::VecDeque;

use cranelift_codegen::ir::{
    Block, Function, Inst, InstBuilder, InstructionData, Opcode, Value, ValueDef,
};
use rustc_hash::FxHashMap as HashMap;

use super::helpers::Helper;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    Number,
    Other,
}

#[derive(Clone, Default)]
pub(super) struct Annotations {
    pub constants: HashMap<Value, Kind>,
    pub helpers: HashMap<Inst, Helper>,
    pub guards: HashMap<Inst, Value>,
}

#[derive(Clone)]
pub(super) struct FunctionBody {
    pub function: Function,
    pub annotations: Annotations,
}

type Facts = HashMap<Value, Kind>;

pub(super) struct Analysis {
    // A missing block is unreachable; a missing value kind is unknown.
    entries: Vec<Option<Facts>>,
    guards: Vec<(Inst, Value, Option<Kind>)>,
}

fn value_kind(function: &Function, facts: &Facts, value: Value) -> Option<Kind> {
    facts.get(&function.dfg.resolve_aliases(value)).copied()
}

fn set_kind(function: &Function, facts: &mut Facts, value: Value, kind: Option<Kind>) {
    let value = function.dfg.resolve_aliases(value);
    if let Some(kind) = kind {
        facts.insert(value, kind);
    } else {
        facts.remove(&value);
    }
}

fn block_facts(function: &Function, block: Block, entry: &Facts) -> Facts {
    let mut facts = entry.clone();
    // Backedges must not reuse facts from this block's previous execution.
    for inst in function.layout.block_insts(block) {
        for &result in function.dfg.inst_results(inst) {
            facts.remove(&function.dfg.resolve_aliases(result));
        }
    }
    facts
}

fn helper_kind(function: &Function, facts: &Facts, inst: Inst, helper: Helper) -> Option<Kind> {
    let args = function.dfg.inst_args(inst);
    let number = |index| value_kind(function, facts, args[index]) == Some(Kind::Number);
    match helper {
        Helper::Add | Helper::Sub | Helper::Mul | Helper::Div | Helper::Rem | Helper::Pow
            if number(1) && number(2) =>
        {
            Some(Kind::Number)
        }
        Helper::UnaryPlus | Helper::UnaryNegation | Helper::BitwiseNot if number(1) => {
            Some(Kind::Number)
        }
        Helper::TempRetain | Helper::FrameExit => value_kind(function, facts, args[1]),
        Helper::LocalSet => value_kind(function, facts, args[2]),
        Helper::Truthy
        | Helper::LogicalNot
        | Helper::StrictEqual
        | Helper::StrictNotEqual
        | Helper::LooseEqual
        | Helper::LooseNotEqual
        | Helper::Less
        | Helper::LessEqual
        | Helper::Greater
        | Helper::GreaterEqual
        | Helper::ErrorPending
        | Helper::ExceptionPending => Some(Kind::Other),
        _ => None,
    }
}

fn transfer(function: &Function, annotations: &Annotations, facts: &mut Facts, inst: Inst) {
    let Some(&result) = function.dfg.inst_results(inst).first() else {
        return;
    };
    let kind = if let Some(kind) = annotations.constants.get(&result) {
        Some(*kind)
    } else if let Some(&helper) = annotations.helpers.get(&inst) {
        helper_kind(function, facts, inst, helper)
    } else if function.dfg.insts[inst].opcode() == Opcode::Select {
        let args = function.dfg.inst_args(inst);
        let left = value_kind(function, facts, args[1]);
        left.filter(|_| left == value_kind(function, facts, args[2]))
    } else {
        None
    };
    set_kind(function, facts, result, kind);
}

fn assume_condition(
    function: &Function,
    annotations: &Annotations,
    facts: &mut Facts,
    condition: Value,
    truth: bool,
) {
    let condition = function.dfg.resolve_aliases(condition);
    let ValueDef::Result(inst, _) = function.dfg.value_def(condition) else {
        return;
    };
    if let Some(&value) = annotations.guards.get(&inst) {
        set_kind(
            function,
            facts,
            value,
            Some(if truth { Kind::Number } else { Kind::Other }),
        );
    } else if truth && function.dfg.insts[inst].opcode() == Opcode::Band {
        for &arg in function.dfg.inst_args(inst) {
            assume_condition(function, annotations, facts, arg, true);
        }
    }
}

fn meet(target: &mut Option<Facts>, incoming: Facts) -> bool {
    let Some(target) = target else {
        *target = Some(incoming);
        return true;
    };
    let previous = target.len();
    target.retain(|value, kind| incoming.get(value) == Some(kind));
    target.len() != previous
}

impl Analysis {
    pub fn compute(function: &Function, annotations: &Annotations) -> Self {
        let mut entries = vec![None; function.dfg.num_blocks()];
        let Some(entry) = function.layout.entry_block() else {
            return Self {
                entries,
                guards: Vec::new(),
            };
        };
        entries[entry.as_u32() as usize] = Some(Facts::default());
        let mut queue = VecDeque::from([entry]);
        let mut queued = vec![false; entries.len()];
        queued[entry.as_u32() as usize] = true;
        while let Some(block) = queue.pop_front() {
            queued[block.as_u32() as usize] = false;
            let mut facts = block_facts(
                function,
                block,
                entries[block.as_u32() as usize].as_ref().unwrap(),
            );
            for inst in function.layout.block_insts(block) {
                transfer(function, annotations, &mut facts, inst);
                for (edge, destination) in function.dfg.insts[inst]
                    .branch_destination(&function.dfg.jump_tables, &function.dfg.exception_tables)
                    .iter()
                    .enumerate()
                {
                    let successor = destination.block(&function.dfg.value_lists);
                    let mut incoming = facts.clone();
                    if let InstructionData::Brif { arg, .. } = function.dfg.insts[inst] {
                        assume_condition(function, annotations, &mut incoming, arg, edge == 0);
                    }
                    let kinds: Vec<_> = destination
                        .args(&function.dfg.value_lists)
                        .map(|arg| {
                            arg.as_value()
                                .and_then(|value| value_kind(function, &incoming, value))
                        })
                        .collect();
                    for (&param, kind) in function.dfg.block_params(successor).iter().zip(kinds) {
                        set_kind(function, &mut incoming, param, kind);
                    }
                    let index = successor.as_u32() as usize;
                    if meet(&mut entries[index], incoming) && !queued[index] {
                        queued[index] = true;
                        queue.push_back(successor);
                    }
                }
            }
        }
        let mut guards = Vec::new();
        for block in function.layout.blocks() {
            let Some(entry) = &entries[block.as_u32() as usize] else {
                continue;
            };
            let mut facts = block_facts(function, block, entry);
            for inst in function.layout.block_insts(block) {
                if let Some(&value) = annotations.guards.get(&inst) {
                    guards.push((
                        inst,
                        function.dfg.resolve_aliases(value),
                        value_kind(function, &facts, value),
                    ));
                }
                transfer(function, annotations, &mut facts, inst);
            }
        }
        Self { entries, guards }
    }

    pub fn simplify(&self, function: &mut Function) -> usize {
        let mut count = 0;
        for &(inst, _, kind) in &self.guards {
            let Some(kind) = kind else { continue };
            let ty = function.dfg.value_type(function.dfg.first_result(inst));
            function
                .dfg
                .replace(inst)
                .iconst(ty, i64::from(kind == Kind::Number));
            count += 1;
        }
        count
    }

    pub fn dump(&self, function: &Function, name: &str, annotations: &Annotations) {
        eprintln!(
            "js ssa [{name}]: {} numeric guards, {} proven, {} helper calls",
            self.guards.len(),
            self.guards
                .iter()
                .filter(|(_, _, kind)| kind.is_some())
                .count(),
            annotations.helpers.len()
        );
        for block in function.layout.blocks() {
            let Some(facts) = &self.entries[block.as_u32() as usize] else {
                continue;
            };
            for &param in function.dfg.block_params(block) {
                eprintln!(
                    "  {block} {param}: {:?}",
                    value_kind(function, facts, param)
                );
            }
        }
        for &(inst, value, kind) in &self.guards {
            eprintln!("  {inst}: is_number({value}) = {kind:?}");
        }
    }
}

#[cfg(test)]
mod tests {
    use cranelift_codegen::ir::{AbiParam, InstBuilder, types};
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};

    use super::*;
    use crate::value::{TAG_FIRST, TAG_SHIFT, TAG_SPAN, Value as JsValue};

    fn constant(
        b: &mut FunctionBuilder<'_>,
        annotations: &mut Annotations,
        value: JsValue,
    ) -> Value {
        let result = b.ins().iconst(types::I64, value.bits() as i64);
        annotations.constants.insert(
            result,
            if value.is_number() {
                Kind::Number
            } else {
                Kind::Other
            },
        );
        result
    }

    fn guard(b: &mut FunctionBuilder<'_>, annotations: &mut Annotations, value: Value) -> Value {
        let tag = b.ins().ushr_imm(value, TAG_SHIFT as i64);
        let offset = b.ins().iadd_imm(tag, -(TAG_FIRST as i64));
        let condition = b.ins().icmp_imm(
            cranelift_codegen::ir::condcodes::IntCC::UnsignedGreaterThan,
            offset,
            TAG_SPAN as i64,
        );
        annotations
            .guards
            .insert(b.func.dfg.value_def(condition).unwrap_inst(), value);
        condition
    }

    fn build(emit: impl FnOnce(&mut FunctionBuilder<'_>, &mut Annotations)) -> FunctionBody {
        let mut function = Function::new();
        function.signature.params.extend([
            AbiParam::new(types::I64),
            AbiParam::new(types::I64),
            AbiParam::new(types::I8),
        ]);
        function.signature.returns.push(AbiParam::new(types::I8));
        let mut context = FunctionBuilderContext::new();
        let mut annotations = Annotations::default();
        let mut b = FunctionBuilder::new(&mut function, &mut context);
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        emit(&mut b, &mut annotations);
        b.seal_all_blocks();
        b.finalize();
        FunctionBody {
            function,
            annotations,
        }
    }

    fn kinds(body: &FunctionBody) -> Vec<Option<Kind>> {
        Analysis::compute(&body.function, &body.annotations)
            .guards
            .into_iter()
            .map(|(_, _, kind)| kind)
            .collect()
    }

    #[test]
    fn distinguishes_boxed_constants_from_raw_integers() {
        let body = build(|b, annotations| {
            let number = constant(b, annotations, JsValue::number(3.0));
            let other = constant(b, annotations, JsValue::UNINITIALIZED);
            let raw = b.ins().iconst(types::I64, 0);
            guard(b, annotations, number);
            guard(b, annotations, other);
            let condition = guard(b, annotations, raw);
            b.ins().return_(&[condition]);
        });
        assert_eq!(kinds(&body), [Some(Kind::Number), Some(Kind::Other), None]);
        let mut copy = body.clone();
        let analysis = Analysis::compute(&copy.function, &copy.annotations);
        assert_eq!(analysis.simplify(&mut copy.function), 2);
        cranelift_codegen::verify_function(
            &copy.function,
            &cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder()),
        )
        .unwrap();
        assert_eq!(kinds(&body), [Some(Kind::Number), Some(Kind::Other), None]);
    }

    #[test]
    fn refines_guard_edges_without_leaking_through_merges() {
        let body = build(|b, annotations| {
            let value = b.block_params(b.current_block().unwrap())[0];
            let yes = b.create_block();
            let no = b.create_block();
            let join = b.create_block();
            let condition = guard(b, annotations, value);
            b.ins().brif(condition, yes, &[], no, &[]);
            b.switch_to_block(yes);
            guard(b, annotations, value);
            b.ins().jump(join, &[]);
            b.switch_to_block(no);
            guard(b, annotations, value);
            b.ins().jump(join, &[]);
            b.switch_to_block(join);
            let condition = guard(b, annotations, value);
            b.ins().return_(&[condition]);
        });
        assert_eq!(
            kinds(&body),
            [None, Some(Kind::Number), Some(Kind::Other), None]
        );
    }

    #[test]
    fn resolves_trivial_block_parameter_aliases() {
        let body = build(|b, annotations| {
            let variable = b.declare_var(types::I64);
            let initial = constant(b, annotations, JsValue::number(7.0));
            b.def_var(variable, initial);
            let successor = b.create_block();
            b.ins().jump(successor, &[]);
            b.switch_to_block(successor);
            let value = b.use_var(variable);
            let condition = guard(b, annotations, value);
            b.ins().return_(&[condition]);
        });
        let value = *body.annotations.guards.values().next().unwrap();
        assert_ne!(value, body.function.dfg.resolve_aliases(value));
        assert_eq!(kinds(&body), [Some(Kind::Number)]);
    }

    #[test]
    fn propagates_conjunction_only_on_the_success_edge() {
        let body = build(|b, annotations| {
            let params = b.block_params(b.current_block().unwrap()).to_vec();
            let left = guard(b, annotations, params[0]);
            let right = guard(b, annotations, params[1]);
            let both = b.ins().band(left, right);
            let yes = b.create_block();
            let no = b.create_block();
            b.ins().brif(both, yes, &[], no, &[]);
            b.switch_to_block(yes);
            guard(b, annotations, params[0]);
            let condition = guard(b, annotations, params[1]);
            b.ins().return_(&[condition]);
            b.switch_to_block(no);
            let condition = guard(b, annotations, params[0]);
            b.ins().return_(&[condition]);
        });
        assert_eq!(
            kinds(&body),
            [None, None, Some(Kind::Number), Some(Kind::Number), None]
        );
    }

    #[test]
    fn loop_facts_include_backedges_and_frontend_aliases() {
        for mixed in [false, true] {
            let body = build(|b, annotations| {
                let flag = b.block_params(b.current_block().unwrap())[2];
                let variable = b.declare_var(types::I64);
                let initial = constant(b, annotations, JsValue::number(0.0));
                b.def_var(variable, initial);
                let header = b.create_block();
                let backedge = b.create_block();
                let exit = b.create_block();
                b.ins().jump(header, &[]);
                b.switch_to_block(header);
                let value = b.use_var(variable);
                let condition = guard(b, annotations, value);
                b.ins().brif(flag, backedge, &[], exit, &[]);
                b.switch_to_block(backedge);
                let next = constant(
                    b,
                    annotations,
                    if mixed {
                        JsValue::FALSE
                    } else {
                        JsValue::number(1.0)
                    },
                );
                b.def_var(variable, next);
                b.ins().jump(header, &[]);
                b.switch_to_block(exit);
                b.ins().return_(&[condition]);
            });
            assert_eq!(
                kinds(&body),
                [if mixed { None } else { Some(Kind::Number) }]
            );
        }
    }
}

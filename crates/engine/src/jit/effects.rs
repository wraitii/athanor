use std::collections::BTreeMap;

use cranelift_codegen::ir::{Function, Inst, Value, ValueDef};

use super::helpers::{HELPERS, Helper};
use super::numeric::Annotations;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Effects {
    pub(super) reentry: Reentry,
    pub(super) flags: u16,
    pub(super) assumption: Option<&'static str>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Reentry {
    Closed,
    Arbitrary,
    Unknown,
}

pub(super) const HEAP_READ: u16 = 1;
pub(super) const HEAP_WRITE: u16 = 2;
pub(super) const GLOBAL_READ: u16 = 4;
pub(super) const GLOBAL_WRITE: u16 = 8;
pub(super) const INTERNAL: u16 = 16;
pub(super) const ALLOCATION: u16 = 32;
pub(super) const ERROR: u16 = 64;
const UNKNOWN: u16 =
    HEAP_READ | HEAP_WRITE | GLOBAL_READ | GLOBAL_WRITE | INTERNAL | ALLOCATION | ERROR;

pub(super) fn effects(helper: Helper) -> Effects {
    use Helper::*;
    let (reentry, flags, assumption) = match helper {
        Call | CallSpread | Construct | ConstructSpread => (Reentry::Arbitrary, UNKNOWN, None),
        FrameEnter | FrameExit | LocalSet | LocalGet | TempClear | TempRetain | ErrorPending
        | ExceptionPending | ErrorOrigin | Catch | Throw => {
            (Reentry::Closed, INTERNAL | ERROR, None)
        }
        MathGuard => (Reentry::Closed, HEAP_READ | GLOBAL_READ, None),
        PropertyProbe => (
            Reentry::Closed,
            HEAP_READ | INTERNAL,
            Some("direct property probe does not invoke accessors"),
        ),
        PropertyRelease => (Reentry::Closed, INTERNAL, None),
        MathDirect => (
            Reentry::Closed,
            ALLOCATION | INTERNAL | ERROR,
            Some(
                "native identity checked by entry guard; current coercion does not invoke JS callbacks",
            ),
        ),
        Truthy => (Reentry::Closed, HEAP_READ, None),
        ArrayNew | Rest | ObjectNew | StringNew => {
            (Reentry::Closed, ALLOCATION | INTERNAL | ERROR, None)
        }
        GlobalGet | GlobalGetOptional => (Reentry::Closed, GLOBAL_READ | INTERNAL | ERROR, None),
        GlobalSet | GlobalInitialize => (Reentry::Closed, GLOBAL_WRITE | INTERNAL | ERROR, None),
        PropertyGet | IndexGet | IndexGetNumeric => (
            Reentry::Closed,
            HEAP_READ | GLOBAL_READ | ALLOCATION | INTERNAL | ERROR,
            Some("current property access has no getters or coercion callbacks"),
        ),
        IterateNew | IterateNext | EnumerateNew | EnumerateNext => (
            Reentry::Closed,
            HEAP_READ | INTERNAL | ALLOCATION | ERROR,
            Some(
                "current iteration/enumeration uses direct engine state, not JS iterator callbacks",
            ),
        ),
        BitwiseAnd | BitwiseOr | BitwiseXor | ShiftLeft | ShiftRight | ShiftRightUnsigned | Add
        | Sub | Mul | Div | Rem | Pow | LooseEqual | LooseNotEqual | StrictEqual
        | StrictNotEqual | Less | LessEqual | Greater | GreaterEqual | UnaryPlus
        | UnaryNegation | LogicalNot | BitwiseNot => (
            Reentry::Closed,
            HEAP_READ | ALLOCATION | INTERNAL | ERROR,
            Some("current coercion does not invoke JS callbacks"),
        ),
        _ => (Reentry::Unknown, UNKNOWN, None),
    };
    Effects {
        reentry,
        flags,
        assumption,
    }
}

fn describe(flags: u16) -> String {
    [
        (HEAP_READ, "heap read"),
        (HEAP_WRITE, "heap write"),
        (GLOBAL_READ, "global read"),
        (GLOBAL_WRITE, "global write"),
        (INTERNAL, "internal bookkeeping/ownership"),
        (ALLOCATION, "allocation"),
        (ERROR, "errors"),
    ]
    .into_iter()
    .filter_map(|(bit, name)| (flags & bit != 0).then_some(name))
    .collect::<Vec<_>>()
    .join(", ")
}

pub(super) fn defining_helper(
    function: &Function,
    annotations: &Annotations,
    value: Value,
    expected: Helper,
) -> Option<Inst> {
    let value = function.dfg.resolve_aliases(value);
    let ValueDef::Result(inst, _) = function.dfg.value_def(value) else {
        return None;
    };
    (annotations.helpers.get(&inst) == Some(&expected)).then_some(inst)
}

pub(super) fn math_target<'a>(
    function: &Function,
    annotations: &'a Annotations,
    call: Inst,
) -> Option<&'a str> {
    let args = function.dfg.inst_args(call);
    let property = defining_helper(function, annotations, *args.get(1)?, Helper::PropertyGet)?;
    let property_args = function.dfg.inst_args(property);
    let receiver = function.dfg.resolve_aliases(*property_args.get(1)?);
    if receiver != function.dfg.resolve_aliases(*args.get(2)?) {
        return None;
    }
    let global = defining_helper(function, annotations, receiver, Helper::GlobalGet)?;
    if annotations.names.get(&global)?.as_str() != "Math" {
        return None;
    }
    Some(annotations.names.get(&property)?.as_str())
}

fn site(function: &Function, inst: Inst, lines: &[u32]) -> String {
    let location = function.srcloc(inst);
    if location.is_default() {
        return inst.to_string();
    }
    let line = lines.partition_point(|&start| start <= location.bits());
    format!("{inst}@line {line}")
}

fn report(function: &Function, annotations: &Annotations, lines: &[u32]) -> String {
    let mut groups: BTreeMap<usize, (Helper, Vec<String>)> = BTreeMap::new();
    let mut unclassified = Vec::new();
    let mut math_candidates = Vec::new();
    for block in function.layout.blocks() {
        for inst in function.layout.block_insts(block) {
            if let Some(&helper) = annotations.helpers.get(&inst) {
                if helper == Helper::Call {
                    if let Some(name) = math_target(function, annotations, inst) {
                        math_candidates
                            .push(format!("{}: Math.{name}", site(function, inst, lines)));
                    }
                }
                groups
                    .entry(helper as usize)
                    .or_insert_with(|| (helper, Vec::new()))
                    .1
                    .push(site(function, inst, lines));
            } else if function.dfg.insts[inst].opcode().is_call() {
                unclassified.push(site(function, inst, lines));
            }
        }
    }
    let mut result = String::new();
    let mut blocked = !unclassified.is_empty();
    for (index, (helper, sites)) in groups {
        let effect = effects(helper);
        let status = match effect.reentry {
            Reentry::Closed => "discharged",
            Reentry::Arbitrary => {
                blocked = true;
                "unresolved: arbitrary call"
            }
            Reentry::Unknown => {
                blocked = true;
                "unresolved: unclassified helper"
            }
        };
        result.push_str(&format!(
            "  {}: {} site(s), {status}; {}",
            HELPERS[index].name,
            sites.len(),
            describe(effect.flags)
        ));
        if effect.reentry != Reentry::Closed {
            result.push_str(&format!(" [{}]", sites.join(", ")));
        }
        if effect.reentry == Reentry::Arbitrary {
            result.push_str("; obligation: establish this generic call target and its effects");
        }
        if let Some(assumption) = effect.assumption {
            result.push_str(&format!("; assumes {assumption}"));
        }
        result.push('\n');
    }
    if !math_candidates.is_empty() {
        result.push_str(&format!(
            "  candidate call targets (identity unproven): {}\n",
            math_candidates.join(", ")
        ));
    }
    if !unclassified.is_empty() {
        result.push_str(&format!(
            "  unannotated calls (including internal dispatch calls): unresolved [{}]\n",
            unclassified.join(", ")
        ));
    }
    format!(
        "  scope: helper effects and calls only; raw memory effects are not analyzed\n  admission: {}\n{result}",
        if blocked {
            "unresolved"
        } else {
            "closed against reentry in current runtime"
        }
    )
}

pub(super) fn dump(function: &Function, annotations: &Annotations, name: &str, lines: &[u32]) {
    println!(
        "[{name}] effects (before SSA optimization)\n{}",
        report(function, annotations, lines)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unannotated_indirect_call_is_unresolved() {
        use cranelift_codegen::ir::{InstBuilder, Signature, types};
        use cranelift_codegen::isa::CallConv;
        use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};

        let mut function = Function::new();
        let mut context = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut function, &mut context);
        let block = builder.create_block();
        builder.switch_to_block(block);
        let signature = builder
            .func
            .import_signature(Signature::new(CallConv::SystemV));
        let target = builder.ins().iconst(types::I64, 0);
        let call = builder.ins().call_indirect(signature, target, &[]);
        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize();

        let output = report(&function, &Annotations::default(), &[]);
        assert!(output.contains("admission: unresolved"));
        assert!(output.contains(&format!(
            "unannotated calls (including internal dispatch calls): unresolved [{call}]"
        )));
    }
}

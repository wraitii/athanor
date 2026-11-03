use cranelift_codegen::cursor::{Cursor, FuncCursor};
use std::collections::VecDeque;

use cranelift_codegen::ir::{
    Block, FuncRef, Function, Inst, InstBuilder, InstructionData, MemFlags, Opcode, Value,
    ValueDef, types,
};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use super::effects::{self, TemporaryEffect};
use super::helpers::Helper;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    Number,
    Int32,
    OtherNumber,
    NaN,
    Other,
}

impl Kind {
    pub(super) fn of_constant(value: crate::value::Value) -> Self {
        let Some(number) = value.as_number() else {
            return Self::Other;
        };
        if number.is_nan() {
            return Self::NaN;
        }
        if number == (number as i32) as f64 && !(number == 0.0 && number.is_sign_negative()) {
            return Self::Int32;
        }
        Self::OtherNumber
    }

    pub(super) fn is_number(self) -> bool {
        self != Self::Other
    }
}

fn join(left: Option<Kind>, right: Option<Kind>) -> Option<Kind> {
    let (Some(left), Some(right)) = (left, right) else {
        return None;
    };
    if left == right {
        return Some(left);
    }
    (left.is_number() && right.is_number()).then_some(Kind::Number)
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum IndexedReadContract {
    RuntimeValue,
}

#[derive(Clone, Copy)]
pub(super) struct IndexedRead {
    pub call: Inst,
    pub receiver: Value,
    pub key: Value,
    pub result: Value,
    pub success: Block,
    pub source: (u32, u32),
    pub contract: IndexedReadContract,
}

#[derive(Clone, Copy)]
pub(super) struct IndexedStorageGuard {
    pub call: Inst,
    pub result: Value,
    pub contract: crate::heap::IndexedStorageContract,
}

#[derive(Clone, Default)]
pub(super) struct Annotations {
    pub semantic_regions: Vec<String>,
    pub invocation_roots: HashSet<Value>,
    pub indexed_storage_guards: HashMap<Inst, IndexedStorageGuard>,
    pub indexed_reads: Vec<IndexedRead>,
    pub constants: HashMap<Value, Kind>,
    pub results: HashMap<Value, Kind>,
    pub continuations: HashMap<Block, Vec<(Value, Kind)>>,
    pub helpers: HashMap<Inst, Helper>,
    pub names: HashMap<Inst, String>,
    pub guards: HashMap<Inst, Value>,
    pub int32_guards: HashMap<Inst, Value>,
    pub retain_guards: HashMap<Inst, Value>,
}

impl Annotations {
    pub fn prune(&mut self, function: &Function) {
        let live = |inst: &Inst| function.layout.inst_block(*inst).is_some();
        self.indexed_reads
            .retain(|read| live(&read.call) && function.layout.is_block_inserted(read.success));
        self.indexed_storage_guards
            .retain(|source, guard| live(source) && live(&guard.call));
        self.helpers.retain(|inst, _| live(inst));
        self.names.retain(|inst, _| live(inst));
        self.guards.retain(|inst, _| live(inst));
        self.int32_guards.retain(|inst, _| live(inst));
        self.retain_guards.retain(|inst, _| live(inst));
        self.continuations
            .retain(|block, _| function.layout.is_block_inserted(*block));
    }
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
    retain_guards: Vec<Inst>,
    int32_guards: Vec<(Inst, bool)>,
    numeric_index_gets: Vec<Inst>,
    unowned_local_stores: Vec<Inst>,
}

fn unowned_local_stores(
    function: &Function,
    annotations: &Annotations,
    entries: &[Option<Facts>],
) -> Vec<Inst> {
    // Generators restore SSA locals from these slots after suspension.
    if annotations
        .helpers
        .values()
        .any(|helper| matches!(helper, Helper::GeneratorState | Helper::LocalGet))
    {
        return Vec::new();
    }
    let mut slots: HashMap<i64, (bool, Vec<Inst>)> = HashMap::default();
    for block in function.layout.blocks() {
        let mut facts = entries[block.as_u32() as usize]
            .as_ref()
            .map(|entry| block_facts(function, annotations, block, entry))
            .unwrap_or_default();
        for inst in function.layout.block_insts(block) {
            if annotations.helpers.get(&inst) == Some(&Helper::LocalSet) {
                let args = function.dfg.inst_args(inst);
                let slot = function.dfg.resolve_aliases(args[1]);
                let ValueDef::Result(slot_inst, _) = function.dfg.value_def(slot) else {
                    return Vec::new();
                };
                let InstructionData::UnaryImm {
                    opcode: Opcode::Iconst,
                    imm,
                } = function.dfg.insts[slot_inst]
                else {
                    return Vec::new();
                };
                let value = function.dfg.resolve_aliases(args[2]);
                let unowned = value_kind(function, &facts, value).is_some_and(Kind::is_number)
                    || is_unowned_constant(function, annotations, value);
                let entry = slots
                    .entry(imm.bits())
                    .or_insert_with(|| (true, Vec::new()));
                // A primitive store must still release any previous owning value.
                entry.0 &= unowned;
                entry.1.push(inst);
            }
            transfer(function, annotations, &mut facts, inst);
        }
    }
    slots
        .into_values()
        .filter(|(unowned, _)| *unowned)
        .flat_map(|(_, stores)| stores)
        .collect()
}

fn is_unowned_constant(function: &Function, annotations: &Annotations, value: Value) -> bool {
    if !annotations.constants.contains_key(&value) {
        return false;
    }
    let ValueDef::Result(inst, _) = function.dfg.value_def(value) else {
        return false;
    };
    let InstructionData::UnaryImm {
        opcode: Opcode::Iconst,
        imm,
    } = function.dfg.insts[inst]
    else {
        return false;
    };
    let value = crate::value::Value::from_bits(imm.bits() as u64);
    !value.is_object() && !value.is_string()
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

fn block_facts(
    function: &Function,
    annotations: &Annotations,
    block: Block,
    entry: &Facts,
) -> Facts {
    let mut facts = entry.clone();
    // Backedges must not reuse facts from this block's previous execution.
    for inst in function.layout.block_insts(block) {
        for &result in function.dfg.inst_results(inst) {
            facts.remove(&function.dfg.resolve_aliases(result));
        }
    }
    if let Some(values) = annotations.continuations.get(&block) {
        for &(value, kind) in values {
            set_kind(function, &mut facts, value, Some(kind));
        }
    }
    facts
}

fn helper_kind(function: &Function, facts: &Facts, inst: Inst, helper: Helper) -> Option<Kind> {
    let args = function.dfg.inst_args(inst);
    let number = |index| value_kind(function, facts, args[index]).is_some_and(Kind::is_number);
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
    let kind = if let Some(kind) = annotations
        .constants
        .get(&result)
        .or_else(|| annotations.results.get(&result))
    {
        Some(*kind)
    } else if let Some(&helper) = annotations.helpers.get(&inst) {
        helper_kind(function, facts, inst, helper)
    } else if function.dfg.insts[inst].opcode() == Opcode::Select {
        let args = function.dfg.inst_args(inst);
        let left = value_kind(function, facts, args[1]);
        join(left, value_kind(function, facts, args[2]))
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
            Some(if truth {
                value_kind(function, facts, value)
                    .filter(|kind| kind.is_number())
                    .unwrap_or(Kind::Number)
            } else {
                Kind::Other
            }),
        );
    } else if let Some(&value) = annotations.int32_guards.get(&inst) {
        if truth {
            set_kind(function, facts, value, Some(Kind::Int32));
        }
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
    let mut changed = false;
    target.retain(|value, kind| {
        let merged = join(Some(*kind), incoming.get(value).copied());
        changed |= merged != Some(*kind);
        if let Some(merged) = merged {
            *kind = merged;
            return true;
        }
        false
    });
    changed
}

fn constant_bits(function: &Function, value: Value, depth: usize) -> Option<u64> {
    if depth == 0 {
        return None;
    }
    let value = function.dfg.resolve_aliases(value);
    let ValueDef::Result(inst, _) = function.dfg.value_def(value) else {
        return None;
    };
    match function.dfg.insts[inst] {
        InstructionData::UnaryImm {
            opcode: Opcode::Iconst,
            imm,
        } => {
            let ty = function.dfg.value_type(value);
            if !ty.is_int() || ty.bits() > 64 {
                return None;
            }
            Some((imm.bits() as u64) & (u64::MAX >> (64 - ty.bits())))
        }
        InstructionData::Binary {
            opcode: Opcode::Band,
            args,
        } => {
            let left = constant_bits(function, args[0], depth - 1);
            let right = constant_bits(function, args[1], depth - 1);
            match (left, right) {
                (Some(0), _) | (_, Some(0)) => Some(0),
                (Some(left), Some(right)) => Some(left & right),
                _ => None,
            }
        }
        _ => None,
    }
}

pub(super) fn simplify_constant_branches(function: &mut Function) -> usize {
    let mut branches = Vec::new();
    for block in function.layout.blocks() {
        for inst in function.layout.block_insts(block) {
            let InstructionData::Brif { arg, blocks, .. } = function.dfg.insts[inst] else {
                continue;
            };
            let Some(condition) = constant_bits(function, arg, 16) else {
                continue;
            };
            let destination = blocks[usize::from(condition == 0)];
            let target = destination.block(&function.dfg.value_lists);
            let args = destination
                .args(&function.dfg.value_lists)
                .collect::<Vec<_>>();
            branches.push((inst, target, args));
        }
    }
    for (inst, target, args) in &branches {
        function.dfg.replace(*inst).jump(*target, args);
    }
    branches.len()
}

fn temporary_state(
    function: &Function,
    annotations: &Annotations,
    facts: &Facts,
    inst: Inst,
    dirty: bool,
) -> bool {
    if let Some(&helper) = annotations.helpers.get(&inst) {
        let args = function.dfg.inst_args(inst);
        let nonowning = |value| {
            value_kind(function, facts, value).is_some_and(Kind::is_number)
                || is_unowned_constant(function, annotations, value)
        };
        return match effects::temporary_effect(helper) {
            TemporaryEffect::Enter | TemporaryEffect::Clear => false,
            TemporaryEffect::None => dirty,
            TemporaryEffect::RetainArgument => dirty || !nonowning(args[1]),
            TemporaryEffect::NumericAdd => dirty || !(nonowning(args[1]) && nonowning(args[2])),
            TemporaryEffect::MayRegister => true,
        };
    }
    let opcode = function.dfg.insts[inst].opcode();
    if opcode.is_call() {
        return true;
    }
    if !opcode.can_store() {
        return dirty;
    }
    match function.dfg.insts[inst] {
        InstructionData::StackStore { .. } => dirty,
        InstructionData::Store { args, .. } => {
            let address = function.dfg.resolve_aliases(args[1]);
            let ValueDef::Result(address_inst, _) = function.dfg.value_def(address) else {
                return true;
            };
            dirty || function.dfg.insts[address_inst].opcode() != Opcode::StackAddr
        }
        _ => true,
    }
}

impl Analysis {
    pub fn compute(function: &Function, annotations: &Annotations) -> Self {
        let mut entries = vec![None; function.dfg.num_blocks()];
        let Some(entry) = function.layout.entry_block() else {
            return Self {
                entries,
                guards: Vec::new(),
                retain_guards: Vec::new(),
                int32_guards: Vec::new(),
                numeric_index_gets: Vec::new(),
                unowned_local_stores: Vec::new(),
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
                annotations,
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
        let mut retain_guards = Vec::new();
        let mut int32_guards = Vec::new();
        let mut numeric_index_gets = Vec::new();
        for block in function.layout.blocks() {
            let Some(entry) = &entries[block.as_u32() as usize] else {
                continue;
            };
            let mut facts = block_facts(function, annotations, block, entry);
            for inst in function.layout.block_insts(block) {
                if annotations.helpers.get(&inst) == Some(&Helper::IndexGet) {
                    let index = function.dfg.inst_args(inst)[2];
                    if value_kind(function, &facts, index).is_some_and(Kind::is_number) {
                        numeric_index_gets.push(inst);
                    }
                }
                if let Some(&value) = annotations.guards.get(&inst) {
                    guards.push((
                        inst,
                        function.dfg.resolve_aliases(value),
                        value_kind(function, &facts, value),
                    ));
                }
                if let Some(&value) = annotations.int32_guards.get(&inst) {
                    if let Some(kind) = value_kind(function, &facts, value) {
                        if kind != Kind::Number {
                            int32_guards.push((inst, kind == Kind::Int32));
                        }
                    }
                }
                if let Some(&value) = annotations.retain_guards.get(&inst) {
                    if value_kind(function, &facts, value).is_some_and(Kind::is_number) {
                        retain_guards.push(inst);
                    }
                }
                transfer(function, annotations, &mut facts, inst);
            }
        }
        let unowned_local_stores = unowned_local_stores(function, annotations, &entries);
        Self {
            entries,
            guards,
            retain_guards,
            int32_guards,
            numeric_index_gets,
            unowned_local_stores,
        }
    }

    pub fn remove_empty_temp_clears(
        &self,
        function: &mut Function,
        annotations: &mut Annotations,
    ) -> usize {
        let Some(entry) = function.layout.entry_block() else {
            return 0;
        };
        let mut entries = vec![None; function.dfg.num_blocks()];
        entries[entry.as_u32() as usize] = Some(true);
        let mut queue = VecDeque::from([entry]);
        let mut queued = vec![false; entries.len()];
        queued[entry.as_u32() as usize] = true;
        while let Some(block) = queue.pop_front() {
            queued[block.as_u32() as usize] = false;
            let Some(entry_facts) = &self.entries[block.as_u32() as usize] else {
                continue;
            };
            let mut facts = block_facts(function, annotations, block, entry_facts);
            let mut dirty = entries[block.as_u32() as usize].unwrap();
            for inst in function.layout.block_insts(block) {
                dirty = temporary_state(function, annotations, &facts, inst, dirty);
                transfer(function, annotations, &mut facts, inst);
                for (edge, destination) in function.dfg.insts[inst]
                    .branch_destination(&function.dfg.jump_tables, &function.dfg.exception_tables)
                    .iter()
                    .enumerate()
                {
                    if let InstructionData::Brif { arg, .. } = function.dfg.insts[inst] {
                        if constant_bits(function, arg, 16)
                            .is_some_and(|value| (value != 0) != (edge == 0))
                        {
                            continue;
                        }
                    }
                    let successor = destination.block(&function.dfg.value_lists);
                    let index = successor.as_u32() as usize;
                    let merged = entries[index].map_or(dirty, |previous| previous || dirty);
                    if entries[index] != Some(merged) {
                        entries[index] = Some(merged);
                        if !queued[index] {
                            queued[index] = true;
                            queue.push_back(successor);
                        }
                    }
                }
            }
        }
        let mut remove = Vec::new();
        for block in function.layout.blocks() {
            let (Some(mut dirty), Some(entry_facts)) = (
                entries[block.as_u32() as usize],
                &self.entries[block.as_u32() as usize],
            ) else {
                continue;
            };
            let mut facts = block_facts(function, annotations, block, entry_facts);
            for inst in function.layout.block_insts(block) {
                if annotations.helpers.get(&inst) == Some(&Helper::TempClear) && !dirty {
                    remove.push(inst);
                }
                dirty = temporary_state(function, annotations, &facts, inst, dirty);
                transfer(function, annotations, &mut facts, inst);
            }
        }
        for &inst in &remove {
            annotations.helpers.remove(&inst);
            function
                .dfg
                .replace(inst)
                .iconst(types::I64, crate::value::Value::UNDEFINED.bits() as i64);
        }
        remove.len()
    }

    pub fn specialize_index_gets(
        &self,
        function: &mut Function,
        annotations: &mut Annotations,
        helper: FuncRef,
    ) {
        for &inst in &self.numeric_index_gets {
            let args = function.dfg.inst_args(inst).to_vec();
            let mut cursor = FuncCursor::new(function).at_inst(inst);
            cursor.use_srcloc(inst);
            let index = cursor.ins().bitcast(types::F64, MemFlags::new(), args[2]);
            cursor
                .func
                .dfg
                .replace(inst)
                .call(helper, &[args[0], args[1], index]);
            annotations.helpers.insert(inst, Helper::IndexGetNumeric);
        }
    }

    pub fn simplify(&self, function: &mut Function) -> usize {
        let mut count = 0;
        for &(inst, _, kind) in &self.guards {
            let Some(kind) = kind else { continue };
            let ty = function.dfg.value_type(function.dfg.first_result(inst));
            function
                .dfg
                .replace(inst)
                .iconst(ty, i64::from(kind.is_number()));
            count += 1;
        }
        for &(inst, exact) in &self.int32_guards {
            let ty = function.dfg.value_type(function.dfg.first_result(inst));
            function.dfg.replace(inst).iconst(ty, i64::from(exact));
            count += 1;
        }
        for &inst in &self.retain_guards {
            let ty = function.dfg.value_type(function.dfg.first_result(inst));
            function.dfg.replace(inst).iconst(ty, 0);
            count += 1;
        }
        for &inst in &self.unowned_local_stores {
            let value = function.dfg.inst_args(inst)[2];
            let result = function.dfg.first_result(inst);
            function.dfg.clear_results(inst);
            function.dfg.change_to_alias(result, value);
            function.layout.remove_inst(inst);
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
        eprintln!(
            "  {} numeric local reads need no retain",
            self.retain_guards.len()
        );
        eprintln!("  {} int32 guards proven", self.int32_guards.len());
        eprintln!("  {} numeric index getters", self.numeric_index_gets.len());
        eprintln!(
            "  {} unowned local stores removable",
            self.unowned_local_stores.len()
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
        annotations
            .constants
            .insert(result, Kind::of_constant(value));
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
    fn constant_branches_preserve_selected_block_arguments() {
        use cranelift_codegen::ir::BlockArg;
        for (left, selected) in [(1, 17), (2, 29)] {
            let mut body = build(|b, _| {
                let target = b.create_block();
                b.append_block_param(target, types::I8);
                let left = b.ins().iconst(types::I8, left);
                let right = b.ins().iconst(types::I8, 1);
                let condition = b.ins().band(left, right);
                let yes = b.ins().iconst(types::I8, 17);
                let no = b.ins().iconst(types::I8, 29);
                b.ins().brif(
                    condition,
                    target,
                    &[BlockArg::Value(yes)],
                    target,
                    &[BlockArg::Value(no)],
                );
                b.switch_to_block(target);
                let result = b.block_params(target)[0];
                b.ins().return_(&[result]);
            });
            assert_eq!(simplify_constant_branches(&mut body.function), 1);
            let entry = body.function.layout.entry_block().unwrap();
            let jump = body.function.layout.last_inst(entry).unwrap();
            let destinations = body.function.dfg.insts[jump].branch_destination(
                &body.function.dfg.jump_tables,
                &body.function.dfg.exception_tables,
            );
            assert_eq!(destinations.len(), 1);
            let arg = destinations[0]
                .args(&body.function.dfg.value_lists)
                .next()
                .unwrap()
                .as_value()
                .unwrap();
            assert_eq!(constant_bits(&body.function, arg, 16), Some(selected));
            cranelift_codegen::verify_function(
                &body.function,
                &cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder()),
            )
            .unwrap();
        }
    }

    #[test]
    fn temporary_dataflow_respects_constant_bits_and_dirty_joins() {
        for (bits, expected) in [(Some(1), 1), (Some(2), 0), (None, 0)] {
            let mut body = build(|b, annotations| {
                let ctx = b.block_params(b.current_block().unwrap())[0];
                let unknown = b.block_params(b.current_block().unwrap())[2];
                let mut signature = cranelift_codegen::ir::Signature::new(
                    cranelift_codegen::isa::CallConv::SystemV,
                );
                signature.params = vec![AbiParam::new(types::I64); 2];
                signature.returns.push(AbiParam::new(types::I64));
                let signature = b.func.import_signature(signature);
                let helper = b.import_function(cranelift_codegen::ir::ExtFuncData {
                    name: cranelift_codegen::ir::ExternalName::testcase("temporary"),
                    signature,
                    colocated: false,
                    patchable: false,
                });
                let dummy = b.ins().iconst(types::I64, 0);
                let enter = b.ins().call(helper, &[ctx, dummy]);
                annotations.helpers.insert(enter, Helper::FrameEnter);
                let yes = b.create_block();
                let no = b.create_block();
                let merge = b.create_block();
                let one = b.ins().iconst(types::I8, 1);
                let left = bits
                    .map(|bits| b.ins().iconst(types::I8, bits))
                    .unwrap_or(unknown);
                let condition = b.ins().band(left, one);
                b.ins().brif(condition, yes, &[], no, &[]);
                b.switch_to_block(yes);
                b.ins().jump(merge, &[]);
                b.switch_to_block(no);
                let call = b.ins().call(helper, &[ctx, dummy]);
                annotations.helpers.insert(call, Helper::ObjectNew);
                b.ins().jump(merge, &[]);
                b.switch_to_block(merge);
                let clear = b.ins().call(helper, &[ctx, dummy]);
                annotations.helpers.insert(clear, Helper::TempClear);
                b.ins().return_(&[one]);
            });
            let analysis = Analysis::compute(&body.function, &body.annotations);
            assert_eq!(
                analysis.remove_empty_temp_clears(&mut body.function, &mut body.annotations),
                expected
            );
        }
    }

    #[test]
    fn temporary_dataflow_preserves_unknown_call_effects_for_numeric_results() {
        let mut body = build(|b, annotations| {
            let ctx = b.block_params(b.current_block().unwrap())[0];
            let mut signature =
                cranelift_codegen::ir::Signature::new(cranelift_codegen::isa::CallConv::SystemV);
            signature.params = vec![AbiParam::new(types::I64); 2];
            signature.returns.push(AbiParam::new(types::I64));
            let signature = b.func.import_signature(signature);
            let helper = b.import_function(cranelift_codegen::ir::ExtFuncData {
                name: cranelift_codegen::ir::ExternalName::testcase("temporary"),
                signature,
                colocated: false,
                patchable: false,
            });
            let dummy = b.ins().iconst(types::I64, 0);
            let enter = b.ins().call(helper, &[ctx, dummy]);
            annotations.helpers.insert(enter, Helper::FrameEnter);
            let clean = b.ins().call(helper, &[ctx, dummy]);
            annotations.helpers.insert(clean, Helper::TempClear);
            let call = b.ins().call(helper, &[ctx, dummy]);
            annotations.helpers.insert(call, Helper::Call);
            annotations
                .results
                .insert(b.inst_results(call)[0], Kind::Int32);
            let dirty = b.ins().call(helper, &[ctx, dummy]);
            annotations.helpers.insert(dirty, Helper::TempClear);
            let done = b.ins().iconst(types::I8, 1);
            b.ins().return_(&[done]);
        });
        let analysis = Analysis::compute(&body.function, &body.annotations);
        assert_eq!(
            analysis.remove_empty_temp_clears(&mut body.function, &mut body.annotations),
            1
        );
        assert_eq!(
            body.annotations
                .helpers
                .values()
                .filter(|&&helper| helper == Helper::TempClear)
                .count(),
            1
        );
    }

    #[test]
    fn temporary_dataflow_includes_dirty_loop_backedges() {
        let mut body = build(|b, annotations| {
            let ctx = b.block_params(b.current_block().unwrap())[0];
            let condition = b.block_params(b.current_block().unwrap())[2];
            let mut signature =
                cranelift_codegen::ir::Signature::new(cranelift_codegen::isa::CallConv::SystemV);
            signature.params = vec![AbiParam::new(types::I64); 2];
            signature.returns.push(AbiParam::new(types::I64));
            let signature = b.func.import_signature(signature);
            let helper = b.import_function(cranelift_codegen::ir::ExtFuncData {
                name: cranelift_codegen::ir::ExternalName::testcase("temporary"),
                signature,
                colocated: false,
                patchable: false,
            });
            let dummy = b.ins().iconst(types::I64, 0);
            let enter = b.ins().call(helper, &[ctx, dummy]);
            annotations.helpers.insert(enter, Helper::FrameEnter);
            let header = b.create_block();
            let body = b.create_block();
            let done = b.create_block();
            b.ins().jump(header, &[]);
            b.switch_to_block(header);
            let clear = b.ins().call(helper, &[ctx, dummy]);
            annotations.helpers.insert(clear, Helper::TempClear);
            b.ins().brif(condition, body, &[], done, &[]);
            b.switch_to_block(body);
            let call = b.ins().call(helper, &[ctx, dummy]);
            annotations.helpers.insert(call, Helper::ObjectNew);
            b.ins().jump(header, &[]);
            b.switch_to_block(done);
            b.ins().return_(&[condition]);
        });
        let analysis = Analysis::compute(&body.function, &body.annotations);
        assert_eq!(
            analysis.remove_empty_temp_clears(&mut body.function, &mut body.annotations),
            0
        );
    }

    #[test]
    fn removes_local_stores_only_when_the_entire_slot_is_unowned() {
        for mixed in [false, true] {
            let body = build(|b, annotations| {
                let ctx = b.block_params(b.current_block().unwrap())[0];
                let unknown = b.block_params(b.current_block().unwrap())[1];
                let mut signature = cranelift_codegen::ir::Signature::new(
                    cranelift_codegen::isa::CallConv::SystemV,
                );
                signature.params = vec![AbiParam::new(types::I64); 3];
                signature.returns.push(AbiParam::new(types::I64));
                let signature = b.func.import_signature(signature);
                let helper = b.import_function(cranelift_codegen::ir::ExtFuncData {
                    name: cranelift_codegen::ir::ExternalName::testcase("local_set"),
                    signature,
                    colocated: false,
                    patchable: false,
                });
                for slot in [0, 1] {
                    let slot_value = b.ins().iconst(types::I64, slot);
                    let initial = if mixed && slot == 0 {
                        unknown
                    } else {
                        constant(b, annotations, JsValue::UNINITIALIZED)
                    };
                    let number = constant(b, annotations, JsValue::number(7.0));
                    let undefined = constant(b, annotations, JsValue::UNDEFINED);
                    for value in [initial, number, undefined] {
                        let call = b.ins().call(helper, &[ctx, slot_value, value]);
                        annotations.helpers.insert(call, Helper::LocalSet);
                    }
                }
                let result = b.ins().iconst(types::I8, 1);
                b.ins().return_(&[result]);
            });
            let analysis = Analysis::compute(&body.function, &body.annotations);
            let expected = if mixed { 3 } else { 6 };
            assert_eq!(analysis.unowned_local_stores.len(), expected);
            let mut function = body.function;
            assert_eq!(analysis.simplify(&mut function), expected);
            cranelift_codegen::verify_function(
                &function,
                &cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder()),
            )
            .unwrap();
        }
    }

    #[test]
    fn classifies_numeric_constants_without_losing_negative_zero() {
        for number in [0.0, -1.0, i32::MIN as f64, i32::MAX as f64] {
            assert_eq!(Kind::of_constant(JsValue::number(number)), Kind::Int32);
        }
        for number in [
            -0.0,
            0.5,
            i32::MIN as f64 - 1.0,
            i32::MAX as f64 + 1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            assert_eq!(
                Kind::of_constant(JsValue::number(number)),
                Kind::OtherNumber
            );
        }
        assert_eq!(Kind::of_constant(JsValue::number(f64::NAN)), Kind::NaN);
        assert_eq!(Kind::of_constant(JsValue::UNINITIALIZED), Kind::Other);
    }

    #[test]
    fn bitwise_contracts_distinguish_signed_and_unsigned_results() {
        for helper in [
            Helper::BitwiseAnd,
            Helper::BitwiseOr,
            Helper::BitwiseXor,
            Helper::ShiftLeft,
            Helper::ShiftRight,
            Helper::BitwiseNot,
        ] {
            assert_eq!(helper.result_kind_on_success(), Some(Kind::Int32));
        }
        assert_eq!(
            Helper::ShiftRightUnsigned.result_kind_on_success(),
            Some(Kind::Number)
        );
        assert_eq!(Helper::Add.result_kind_on_success(), None);
    }

    #[test]
    fn joins_numeric_subtypes_and_reports_widening() {
        let value = Value::from_u32(0);
        for left in [
            Kind::Int32,
            Kind::OtherNumber,
            Kind::NaN,
            Kind::Number,
            Kind::Other,
        ] {
            for right in [
                Kind::Int32,
                Kind::OtherNumber,
                Kind::NaN,
                Kind::Number,
                Kind::Other,
            ] {
                let expected = if left == right {
                    Some(left)
                } else if left.is_number() && right.is_number() {
                    Some(Kind::Number)
                } else {
                    None
                };
                assert_eq!(join(Some(left), Some(right)), expected);
                let mut target = Some(HashMap::from_iter([(value, left)]));
                assert_eq!(
                    meet(&mut target, HashMap::from_iter([(value, right)])),
                    expected != Some(left)
                );
                assert_eq!(target.unwrap().get(&value).copied(), expected);
            }
        }
    }

    #[test]
    fn numeric_guard_preserves_subtype_on_success() {
        for number in [3.0, -0.0, f64::NAN] {
            let body = build(|b, annotations| {
                let value = constant(b, annotations, JsValue::number(number));
                let yes = b.create_block();
                let no = b.create_block();
                let condition = guard(b, annotations, value);
                b.ins().brif(condition, yes, &[], no, &[]);
                b.switch_to_block(yes);
                let condition = guard(b, annotations, value);
                b.ins().return_(&[condition]);
                b.switch_to_block(no);
                let condition = b.ins().iconst(types::I8, 0);
                b.ins().return_(&[condition]);
            });
            assert_eq!(
                kinds(&body),
                [Some(Kind::of_constant(JsValue::number(number))); 2]
            );
            let mut function = body.function.clone();
            assert_eq!(
                Analysis::compute(&function, &body.annotations).simplify(&mut function),
                2
            );
        }
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
        assert_eq!(kinds(&body), [Some(Kind::Int32), Some(Kind::Other), None]);
        let mut copy = body.clone();
        let analysis = Analysis::compute(&copy.function, &copy.annotations);
        assert_eq!(analysis.simplify(&mut copy.function), 2);
        cranelift_codegen::verify_function(
            &copy.function,
            &cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder()),
        )
        .unwrap();
        assert_eq!(kinds(&body), [Some(Kind::Int32), Some(Kind::Other), None]);
    }

    #[test]
    fn continuation_facts_do_not_escape_failure_or_merge() {
        let body = build(|b, annotations| {
            let params = b.block_params(b.current_block().unwrap()).to_vec();
            let success = b.create_block();
            let failure = b.create_block();
            let join = b.create_block();
            annotations
                .continuations
                .insert(success, vec![(params[0], Kind::Number)]);
            guard(b, annotations, params[0]);
            b.ins().brif(params[2], success, &[], failure, &[]);
            b.switch_to_block(success);
            guard(b, annotations, params[0]);
            b.ins().jump(join, &[]);
            b.switch_to_block(failure);
            guard(b, annotations, params[0]);
            b.ins().jump(join, &[]);
            b.switch_to_block(join);
            let condition = guard(b, annotations, params[0]);
            b.ins().return_(&[condition]);
        });
        assert_eq!(kinds(&body), [None, Some(Kind::Number), None, None]);
    }

    #[test]
    fn removes_retain_guards_only_for_proven_numbers() {
        let body = build(|b, annotations| {
            let unknown = b.block_params(b.current_block().unwrap())[0];
            let number = constant(b, annotations, JsValue::number(3.0));
            let uninitialized = constant(b, annotations, JsValue::UNINITIALIZED);
            for value in [number, uninitialized, unknown] {
                let tag = b.ins().ushr_imm(value, TAG_SHIFT as i64);
                let offset = b
                    .ins()
                    .iadd_imm(tag, -((crate::value::TAG_OBJECT >> TAG_SHIFT) as i64));
                let condition = b.ins().icmp_imm(
                    cranelift_codegen::ir::condcodes::IntCC::UnsignedLessThanOrEqual,
                    offset,
                    2,
                );
                annotations
                    .retain_guards
                    .insert(b.func.dfg.value_def(condition).unwrap_inst(), value);
            }
            let condition = guard(b, annotations, unknown);
            b.ins().return_(&[condition]);
        });
        let analysis = Analysis::compute(&body.function, &body.annotations);
        assert_eq!(analysis.retain_guards.len(), 1);
        let mut function = body.function.clone();
        assert_eq!(analysis.simplify(&mut function), 1);
        cranelift_codegen::verify_function(
            &function,
            &cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder()),
        )
        .unwrap();
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
        assert_eq!(kinds(&body), [Some(Kind::Int32)]);
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
        for (next_value, expected) in [
            (JsValue::number(1.0), Some(Kind::Int32)),
            (JsValue::number(-0.0), Some(Kind::Number)),
            (JsValue::number(f64::NAN), Some(Kind::Number)),
            (JsValue::FALSE, None),
        ] {
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
                let next = constant(b, annotations, next_value);
                b.def_var(variable, next);
                b.ins().jump(header, &[]);
                b.switch_to_block(exit);
                b.ins().return_(&[condition]);
            });
            assert_eq!(kinds(&body), [expected]);
        }
    }
}

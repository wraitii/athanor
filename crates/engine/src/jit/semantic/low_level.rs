use crate::FxHashMap;
use cranelift_codegen::entity::SecondaryMap;
use cranelift_codegen::ir::{
    MemFlagsData, Signature, Type,
    condcodes::{FloatCC, IntCC},
};
use cranelift_module::FuncId;
use oxc_span::Span;

use super::{Block, Cfg, Graph, Inst, Op as GraphOp, Value};
use crate::jit::helpers::Helper;

#[derive(Clone, Debug)]
pub(in crate::jit) enum CallTarget {
    Direct(FuncId),
    Indirect(Value),
}

#[derive(Clone, Copy, Debug)]
pub(in crate::jit) enum Unary {
    BitNot,
    FloatNeg,
    FloatAbs,
    Ceil,
    Floor,
    Trunc,
    Sqrt,
    Reduce,
    SignExtend,
    ZeroExtend,
    Bitcast,
    FloatPromote,
    FloatDemote,
    FloatFromSigned,
    FloatFromUnsigned,
    SignedSaturatingFromFloat,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::jit) enum Binary {
    Add,
    Subtract,
    Multiply,
    And,
    Or,
    Xor,
    ShiftLeft,
    ShiftRightSigned,
    ShiftRightUnsigned,
    FloatAdd,
    FloatSubtract,
    FloatMultiply,
    FloatDivide,
    FloatMin,
    FloatMax,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::jit) struct StackSlot(pub usize);

#[derive(Clone, Copy, Debug)]
pub(in crate::jit) struct StackSlotData {
    pub size: u32,
    pub align_shift: u8,
}

#[derive(Clone, Debug)]
pub(in crate::jit) enum Op {
    Integer {
        ty: Type,
        bits: u64,
    },
    Float64(u64),
    Unary {
        op: Unary,
        ty: Type,
        value: Value,
    },
    Binary {
        op: Binary,
        left: Value,
        right: Value,
    },
    IntCompare {
        condition: IntCC,
        left: Value,
        right: Value,
    },
    FloatCompare {
        condition: FloatCC,
        left: Value,
        right: Value,
    },
    Select {
        condition: Value,
        yes: Value,
        no: Value,
    },
    Load {
        ty: Type,
        address: Value,
        offset: i32,
        flags: MemFlagsData,
    },
    AtomicLoad {
        ty: Type,
        address: Value,
        flags: MemFlagsData,
    },
    Store {
        value: Value,
        address: Value,
        offset: i32,
        flags: MemFlagsData,
    },
    StackAddress {
        slot: StackSlot,
        offset: i32,
        ty: Type,
    },
    StackPointer {
        ty: Type,
    },
    Call {
        target: CallTarget,
        signature: Signature,
        args: Vec<Value>,
        helper: Option<Helper>,
    },
    VariableRead,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::jit) enum BlockArg {
    Value(Value),
}

impl From<Value> for BlockArg {
    fn from(value: Value) -> Self {
        Self::Value(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in crate::jit) struct Variable(usize);

impl Variable {
    pub(in crate::jit) fn as_u32(self) -> u32 {
        self.0 as u32
    }
}

#[derive(Clone, Copy, Debug)]
pub(in crate::jit) struct SignatureRef(usize);

#[derive(Clone, Copy, Debug)]
pub(in crate::jit) struct FunctionRef(usize);

pub(in crate::jit) struct Builder<'a> {
    pub(in crate::jit) graph: &'a mut Graph,
    definitions: FxHashMap<(Block, usize), Value>,
    reads: Vec<(Block, usize, Inst, Value)>,
    types: FxHashMap<usize, Type>,
    location: Option<u32>,
    source_name: Option<(usize, usize)>,
    current: Option<Block>,
    variable_types: Vec<Type>,
    signatures: Vec<Signature>,
    functions: Vec<(FuncId, Signature)>,
    span: Span,
}

impl<'a> Builder<'a> {
    pub(in crate::jit) fn new(graph: &'a mut Graph) -> Self {
        let span = graph.span;
        Self {
            graph,
            definitions: FxHashMap::default(),
            reads: Vec::new(),
            types: FxHashMap::default(),
            location: None,
            source_name: None,
            current: None,
            variable_types: Vec::new(),
            signatures: Vec::new(),
            functions: Vec::new(),
            span,
        }
    }

    pub(in crate::jit) fn create_block(&mut self) -> Block {
        self.graph.blocks.push(super::BlockData::default())
    }

    pub(in crate::jit) fn switch_to_block(&mut self, block: Block) {
        self.current = Some(block);
    }

    pub(in crate::jit) fn create_sized_stack_slot(
        &mut self,
        data: cranelift_codegen::ir::StackSlotData,
    ) -> StackSlot {
        self.stack_slot(data.size, data.align_shift)
    }

    pub(in crate::jit) fn num_blocks(&self) -> usize {
        self.graph.blocks.len()
    }

    pub(in crate::jit) fn defining_inst(&self, value: Value) -> Inst {
        match self.graph.values[value] {
            super::ValueDef::Result(inst, _) => inst,
            _ => panic!("not an instruction result"),
        }
    }

    pub(in crate::jit) fn patch_integer(&mut self, inst: Inst, bits: u64) {
        let GraphOp::LowLevel(Op::Integer { bits: value, .. }) = &mut self.graph.insts[inst].op
        else {
            panic!("not an integer constant")
        };
        *value = bits;
    }

    pub(in crate::jit) fn mark_helper(&mut self, inst: Inst, identity: Helper) {
        let GraphOp::LowLevel(Op::Call { helper, .. }) = &mut self.graph.insts[inst].op else {
            panic!("not a call")
        };
        *helper = Some(identity);
    }

    pub(in crate::jit) fn last_helper(&self) -> Option<Helper> {
        let block = self.current?;
        let inst = *self.graph.blocks[block].insts.last()?;
        match self.graph.insts[inst].op {
            GraphOp::LowLevel(Op::Call { helper, .. }) => helper,
            _ => None,
        }
    }

    pub(in crate::jit) fn indexed_load(
        &mut self,
        base: Value,
        index: Value,
        format: super::MemoryFormat,
        ty: Type,
    ) -> Value {
        let block = self.current.expect("no current block");
        let (inst, result) = self.graph.append(
            block,
            self.span,
            GraphOp::Memory(super::MemoryOp::IndexedLoad {
                base,
                index,
                format,
            }),
            true,
        );
        self.graph.insts[inst].emission_location = self.location;
        self.graph.insts[inst].source_name = self.source_name;
        let result = result.unwrap();
        self.graph.low_level_types[result] = Some(ty);
        result
    }

    pub(in crate::jit) fn set_srcloc(&mut self, location: cranelift_codegen::ir::SourceLoc) {
        self.location = (!location.is_default()).then(|| location.bits());
    }

    pub(in crate::jit) fn append_block_params_for_function_params(&mut self, block: Block) {
        let params: Vec<_> = self
            .graph
            .low_level_signature
            .as_ref()
            .expect("missing function signature")
            .params
            .iter()
            .map(|param| param.value_type)
            .collect();
        for ty in params {
            self.append_block_param(block, ty);
        }
    }

    pub(in crate::jit) fn current_block(&self) -> Option<Block> {
        self.current
    }

    pub(in crate::jit) fn import_signature(&mut self, signature: Signature) -> SignatureRef {
        let reference = SignatureRef(self.signatures.len());
        self.signatures.push(signature);
        reference
    }

    pub(in crate::jit) fn import_function(
        &mut self,
        function: FuncId,
        signature: Signature,
    ) -> FunctionRef {
        let reference = FunctionRef(self.functions.len());
        self.functions.push((function, signature));
        reference
    }

    pub(in crate::jit) fn inst_results(&self, inst: Inst) -> &[Value] {
        &self.graph.insts[inst].results
    }

    pub(in crate::jit) fn set_span(&mut self, span: Span) {
        self.span = span;
    }

    pub(in crate::jit) fn set_cold_block(&mut self, block: Block) {
        self.graph.cold.insert(block);
    }

    pub(in crate::jit) fn append_block_param(&mut self, block: Block, ty: Type) -> Value {
        let value = self.graph.add_param(block);
        self.graph.low_level_types[value] = Some(ty);
        value
    }

    pub(in crate::jit) fn block_params(&self, block: Block) -> &[Value] {
        &self.graph.blocks[block].params
    }

    pub(in crate::jit) fn declare_var(&mut self, ty: Type) -> Variable {
        let variable = Variable(self.variable_types.len());
        self.variable_types.push(ty);
        variable
    }

    pub(in crate::jit) fn declare_var_with_entry_value(
        &mut self,
        ty: Type,
        value: Value,
    ) -> Variable {
        assert_eq!(self.graph.low_level_types[value], Some(ty));
        let variable = self.declare_var(ty);
        self.definitions
            .insert((self.graph.entry, variable.0), value);
        variable
    }

    pub(in crate::jit) fn def_var(&mut self, variable: Variable, value: Value) {
        self.write(
            self.current.expect("no current block"),
            variable.0,
            value,
            self.variable_types[variable.0],
        );
    }

    pub(in crate::jit) fn use_var(&mut self, variable: Variable) -> Value {
        self.read(
            self.current.expect("no current block"),
            variable.0,
            self.variable_types[variable.0],
            self.span,
        )
    }

    pub(in crate::jit) fn ins(&mut self) -> Instructions<'_, 'a> {
        Instructions { builder: self }
    }

    pub(in crate::jit) fn set_source_name(&mut self, source_name: Option<(usize, usize)>) {
        self.source_name = source_name;
    }

    #[cfg(test)]
    pub(in crate::jit) fn set_location(&mut self, location: Option<u32>) {
        self.location = location;
    }

    pub(in crate::jit) fn terminate(
        &mut self,
        block: Block,
        span: Span,
        kind: super::TerminatorKind,
    ) {
        self.graph.terminate(block, span, kind);
        let terminator = self.graph.blocks[block].terminator.as_mut().unwrap();
        terminator.emission_location = self.location;
        terminator.source_name = self.source_name;
    }

    pub(in crate::jit) fn value(&mut self, block: Block, span: Span, op: Op, ty: Type) -> Value {
        let (inst, value) = self.graph.append(block, span, GraphOp::LowLevel(op), true);
        self.graph.insts[inst].emission_location = self.location;
        self.graph.insts[inst].source_name = self.source_name;
        let value = value.unwrap();
        self.graph.low_level_types[value] = Some(ty);
        value
    }

    pub(in crate::jit) fn effect(&mut self, block: Block, span: Span, op: Op) -> Inst {
        let inst = self
            .graph
            .append(block, span, GraphOp::LowLevel(op), false)
            .0;
        self.graph.insts[inst].emission_location = self.location;
        self.graph.insts[inst].source_name = self.source_name;
        inst
    }

    pub(in crate::jit) fn stack_slot(&mut self, size: u32, align_shift: u8) -> StackSlot {
        let slot = StackSlot(self.graph.low_level_slots.len());
        self.graph
            .low_level_slots
            .push(StackSlotData { size, align_shift });
        slot
    }

    #[cfg(test)]
    pub(in crate::jit) fn call(
        &mut self,
        block: Block,
        span: Span,
        target: CallTarget,
        signature: Signature,
        args: Vec<Value>,
    ) -> Vec<Value> {
        let inst = self.call_inst(block, span, target, signature, args);
        self.graph.insts[inst].results.clone()
    }

    fn call_inst(
        &mut self,
        block: Block,
        span: Span,
        target: CallTarget,
        signature: Signature,
        args: Vec<Value>,
    ) -> Inst {
        let types: Vec<_> = signature
            .returns
            .iter()
            .map(|param| param.value_type)
            .collect();
        let op = GraphOp::LowLevel(Op::Call {
            target,
            signature,
            args,
            helper: None,
        });
        let (inst, results) = self.graph.append_results(block, span, op, types.len());
        self.graph.insts[inst].emission_location = self.location;
        self.graph.insts[inst].source_name = self.source_name;
        for (&value, ty) in results.iter().zip(types) {
            self.graph.low_level_types[value] = Some(ty);
        }
        inst
    }

    pub(in crate::jit) fn write(&mut self, block: Block, variable: usize, value: Value, ty: Type) {
        if let Some(previous) = self.types.insert(variable, ty) {
            assert_eq!(previous, ty);
        }
        if let Some(existing) = self.graph.low_level_types[value] {
            assert_eq!(existing, ty);
        }
        self.graph.low_level_types[value] = Some(ty);
        self.definitions.insert((block, variable), value);
    }

    pub(in crate::jit) fn read(
        &mut self,
        block: Block,
        variable: usize,
        ty: Type,
        span: Span,
    ) -> Value {
        if let Some(previous) = self.types.insert(variable, ty) {
            assert_eq!(previous, ty);
        }
        if let Some(&value) = self.definitions.get(&(block, variable)) {
            return value;
        }
        let (inst, result) =
            self.graph
                .append(block, span, GraphOp::LowLevel(Op::VariableRead), true);
        let value = result.unwrap();
        self.graph.low_level_types[value] = Some(ty);
        self.reads.push((block, variable, inst, value));
        self.definitions.insert((block, variable), value);
        value
    }

    pub(in crate::jit) fn finish(mut self) -> Result<(), String> {
        let cfg = Cfg::new(self.graph);
        let mut reachable = SecondaryMap::<Block, bool>::new();
        for &block in &cfg.rpo {
            reachable[block] = true;
        }
        let mut entries = FxHashMap::default();
        let mut aliases = SecondaryMap::new();
        let mut work = Vec::new();
        let mut changed = false;
        for (block, variable, inst, value) in std::mem::take(&mut self.reads) {
            if !reachable[block] {
                self.graph.remove_inst(inst);
                continue;
            }
            let definition = self.entry_value(&cfg, &mut entries, &mut work, block, variable)?;
            aliases[value] = Some(definition);
            self.graph.remove_inst(inst);
            changed = true;
        }
        super::transforms::simplify_parameters(self.graph, &cfg.rpo, &mut aliases, changed);
        Ok(())
    }

    fn entry_value(
        &mut self,
        cfg: &Cfg,
        entries: &mut FxHashMap<(Block, usize), Value>,
        work: &mut Vec<(Block, Option<usize>)>,
        block: Block,
        variable: usize,
    ) -> Result<Value, String> {
        if let Some(&value) = entries.get(&(block, variable)) {
            return Ok(value);
        }
        work.push((block, None));
        while let Some(&(current, next)) = work.last() {
            let next = match next {
                Some(next) => next,
                None => {
                    if current == self.graph.entry || cfg.preds[current].is_empty() {
                        return Err(format!(
                            "variable {variable} has no definition at entry to {current}"
                        ));
                    }
                    if cfg.preds[current].len() > 1 {
                        // Backedges must find this phi before its inputs are resolved.
                        let param = self.graph.add_param(current);
                        entries.insert((current, variable), param);
                        self.graph.low_level_types[param] = Some(self.types[&variable]);
                    }
                    work.last_mut().unwrap().1 = Some(0);
                    0
                }
            };
            let pred = cfg.preds[current][next];
            let value = self
                .definitions
                .get(&(pred, variable))
                .or_else(|| entries.get(&(pred, variable)))
                .copied();
            let Some(value) = value else {
                work.push((pred, None));
                continue;
            };
            if cfg.preds[current].len() == 1 {
                entries.insert((current, variable), value);
            } else {
                self.graph.push_edge_arg(pred, current, value);
            }
            if next + 1 == cfg.preds[current].len() {
                work.pop();
            } else {
                work.last_mut().unwrap().1 = Some(next + 1);
            }
        }
        Ok(entries[&(block, variable)])
    }
}

#[derive(Default)]
pub(in crate::jit) struct Switch {
    entries: Vec<(u128, Block)>,
}

impl Switch {
    pub(in crate::jit) fn new() -> Self {
        Self::default()
    }
    pub(in crate::jit) fn set_entry(&mut self, index: u128, block: Block) {
        self.entries.push((index, block));
    }
    pub(in crate::jit) fn emit(self, builder: &mut Builder<'_>, value: Value, default: Block) {
        let block = builder.current.expect("no current block");
        let cases = self
            .entries
            .into_iter()
            .map(|(index, block)| (index, super::Target::new(block)))
            .collect();
        builder.terminate(
            block,
            builder.span,
            super::TerminatorKind::Switch(value, cases, super::Target::new(default)),
        );
    }
}

pub(in crate::jit) struct Instructions<'b, 'g> {
    builder: &'b mut Builder<'g>,
}

macro_rules! binary_methods {
    ($($name:ident => $op:ident),* $(,)?) => { $(
        pub(in crate::jit) fn $name(&mut self, left: Value, right: Value) -> Value {
            self.binary(Binary::$op, left, right)
        }
    )* };
}

macro_rules! immediate_methods {
    ($($name:ident => $op:ident),* $(,)?) => { $(
        pub(in crate::jit) fn $name(&mut self, value: Value, immediate: i64) -> Value {
            let ty = self.builder.graph.low_level_types[value].expect("untyped low-level operand");
            let immediate = self.iconst(ty, immediate);
            self.binary(Binary::$op, value, immediate)
        }
    )* };
}

macro_rules! unary_methods {
    ($($name:ident => $op:ident),* $(,)?) => { $(
        pub(in crate::jit) fn $name(&mut self, value: Value) -> Value {
            let ty = self.builder.graph.low_level_types[value].expect("untyped low-level operand");
            self.value(Op::Unary { op: Unary::$op, ty, value }, ty)
        }
    )* };
}

macro_rules! conversion_methods {
    ($($name:ident => $op:ident),* $(,)?) => { $(
        pub(in crate::jit) fn $name(&mut self, ty: Type, value: Value) -> Value {
            self.value(Op::Unary { op: Unary::$op, ty, value }, ty)
        }
    )* };
}

impl Instructions<'_, '_> {
    fn value(&mut self, op: Op, ty: Type) -> Value {
        let block = self.builder.current.expect("no current block");
        self.builder.value(block, self.builder.span, op, ty)
    }

    fn binary(&mut self, op: Binary, left: Value, right: Value) -> Value {
        let ty = self.builder.graph.low_level_types[left].expect("untyped low-level operand");
        assert_eq!(self.builder.graph.low_level_types[right], Some(ty));
        self.value(Op::Binary { op, left, right }, ty)
    }

    binary_methods!(isub => Subtract, band => And, bor => Or, bxor => Xor,
        ishl => ShiftLeft, sshr => ShiftRightSigned, ushr => ShiftRightUnsigned,
        fadd => FloatAdd, fsub => FloatSubtract, fmul => FloatMultiply,
        fdiv => FloatDivide, fmin => FloatMin, fmax => FloatMax);
    immediate_methods!(iadd_imm_s => Add, iadd_imm_u => Add, imul_imm_s => Multiply,
        band_imm_s => And, band_imm_u => And, ishl_imm_s => ShiftLeft,
        ushr_imm_s => ShiftRightUnsigned);

    unary_methods!(bnot => BitNot, fneg => FloatNeg, fabs => FloatAbs,
        ceil => Ceil, floor => Floor, trunc => Trunc, sqrt => Sqrt);
    conversion_methods!(ireduce => Reduce, sextend => SignExtend, uextend => ZeroExtend,
        fpromote => FloatPromote, fdemote => FloatDemote, fcvt_from_sint => FloatFromSigned,
        fcvt_from_uint => FloatFromUnsigned, fcvt_to_sint_sat => SignedSaturatingFromFloat);

    pub(in crate::jit) fn f64const(&mut self, value: f64) -> Value {
        self.value(
            Op::Float64(value.to_bits()),
            cranelift_codegen::ir::types::F64,
        )
    }

    pub(in crate::jit) fn bitcast(&mut self, ty: Type, flags: MemFlagsData, value: Value) -> Value {
        assert_eq!(flags, MemFlagsData::new());
        self.value(
            Op::Unary {
                op: Unary::Bitcast,
                ty,
                value,
            },
            ty,
        )
    }

    pub(in crate::jit) fn icmp(&mut self, condition: IntCC, left: Value, right: Value) -> Value {
        self.value(
            Op::IntCompare {
                condition,
                left,
                right,
            },
            cranelift_codegen::ir::types::I8,
        )
    }

    pub(in crate::jit) fn icmp_imm_s(
        &mut self,
        condition: IntCC,
        left: Value,
        immediate: i64,
    ) -> Value {
        let ty = self.builder.graph.low_level_types[left].expect("untyped low-level operand");
        let right = self.iconst(ty, immediate);
        self.icmp(condition, left, right)
    }

    pub(in crate::jit) fn icmp_imm_u(
        &mut self,
        condition: IntCC,
        left: Value,
        immediate: i64,
    ) -> Value {
        self.icmp_imm_s(condition, left, immediate)
    }

    pub(in crate::jit) fn fcmp(&mut self, condition: FloatCC, left: Value, right: Value) -> Value {
        self.value(
            Op::FloatCompare {
                condition,
                left,
                right,
            },
            cranelift_codegen::ir::types::I8,
        )
    }

    pub(in crate::jit) fn select(&mut self, condition: Value, yes: Value, no: Value) -> Value {
        let ty = self.builder.graph.low_level_types[yes].expect("untyped low-level operand");
        assert_eq!(self.builder.graph.low_level_types[no], Some(ty));
        self.value(Op::Select { condition, yes, no }, ty)
    }

    pub(in crate::jit) fn atomic_load(
        &mut self,
        ty: Type,
        flags: MemFlagsData,
        address: Value,
    ) -> Value {
        self.value(Op::AtomicLoad { ty, flags, address }, ty)
    }

    pub(in crate::jit) fn stack_addr(&mut self, ty: Type, slot: StackSlot, offset: i32) -> Value {
        self.value(Op::StackAddress { ty, slot, offset }, ty)
    }

    pub(in crate::jit) fn get_stack_pointer(&mut self, ty: Type) -> Value {
        self.value(Op::StackPointer { ty }, ty)
    }

    pub(in crate::jit) fn iconst(&mut self, ty: Type, bits: i64) -> Value {
        self.value(
            Op::Integer {
                ty,
                bits: bits as u64,
            },
            ty,
        )
    }

    pub(in crate::jit) fn iadd(&mut self, left: Value, right: Value) -> Value {
        self.binary(Binary::Add, left, right)
    }

    pub(in crate::jit) fn imul(&mut self, left: Value, right: Value) -> Value {
        self.binary(Binary::Multiply, left, right)
    }

    pub(in crate::jit) fn load(
        &mut self,
        ty: Type,
        flags: MemFlagsData,
        address: Value,
        offset: i32,
    ) -> Value {
        self.value(
            Op::Load {
                ty,
                address,
                offset,
                flags,
            },
            ty,
        )
    }

    pub(in crate::jit) fn store(
        &mut self,
        flags: MemFlagsData,
        value: Value,
        address: Value,
        offset: i32,
    ) -> Inst {
        let block = self.builder.current.expect("no current block");
        self.builder.effect(
            block,
            self.builder.span,
            Op::Store {
                value,
                address,
                offset,
                flags,
            },
        )
    }

    pub(in crate::jit) fn call(&mut self, function: FunctionRef, args: &[Value]) -> Inst {
        let (id, signature) = self.builder.functions[function.0].clone();
        let block = self.builder.current.expect("no current block");
        self.builder.call_inst(
            block,
            self.builder.span,
            CallTarget::Direct(id),
            signature,
            args.to_vec(),
        )
    }

    pub(in crate::jit) fn call_indirect(
        &mut self,
        signature: SignatureRef,
        address: Value,
        args: &[Value],
    ) -> Inst {
        let signature = self.builder.signatures[signature.0].clone();
        let block = self.builder.current.expect("no current block");
        self.builder.call_inst(
            block,
            self.builder.span,
            CallTarget::Indirect(address),
            signature,
            args.to_vec(),
        )
    }

    pub(in crate::jit) fn return_call_indirect(
        &mut self,
        signature: SignatureRef,
        address: Value,
        args: &[Value],
    ) {
        let signature = self.builder.signatures[signature.0].clone();
        let block = self.builder.current.expect("no current block");
        self.builder.terminate(
            block,
            self.builder.span,
            super::TerminatorKind::TailCall {
                target: CallTarget::Indirect(address),
                signature,
                args: args.to_vec(),
            },
        );
    }

    pub(in crate::jit) fn jump(&mut self, destination: Block, args: &[BlockArg]) {
        let block = self.builder.current.expect("no current block");
        self.builder.terminate(
            block,
            self.builder.span,
            super::TerminatorKind::Jump(super::Target {
                block: destination,
                args: args
                    .iter()
                    .map(|arg| match arg {
                        BlockArg::Value(value) => *value,
                    })
                    .collect(),
            }),
        );
    }

    pub(in crate::jit) fn brif(
        &mut self,
        condition: Value,
        yes: Block,
        yes_args: &[BlockArg],
        no: Block,
        no_args: &[BlockArg],
    ) {
        let block = self.builder.current.expect("no current block");
        self.builder.terminate(
            block,
            self.builder.span,
            super::TerminatorKind::Branch(
                condition,
                super::Target {
                    block: yes,
                    args: yes_args
                        .iter()
                        .map(|arg| match arg {
                            BlockArg::Value(value) => *value,
                        })
                        .collect(),
                },
                super::Target {
                    block: no,
                    args: no_args
                        .iter()
                        .map(|arg| match arg {
                            BlockArg::Value(value) => *value,
                        })
                        .collect(),
                },
            ),
        );
    }

    pub(in crate::jit) fn return_(&mut self, values: &[Value]) {
        let block = self.builder.current.expect("no current block");
        self.builder.terminate(
            block,
            self.builder.span,
            super::TerminatorKind::ReturnValues(values.to_vec()),
        );
    }
}

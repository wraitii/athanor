use cranelift_codegen::ir::{
    MemFlagsData, Signature, Type,
    condcodes::{FloatCC, IntCC},
};
use cranelift_module::FuncId;
use oxc_span::Span;

use super::cursor::Insert;
use super::{Block, Constant, Graph, Inst, Op as GraphOp, Repr, Value};
use crate::jit::helpers::Helper;

#[derive(Clone, Debug)]
pub(in crate::jit) enum CallTarget {
    Direct(FuncId),
    Indirect(Value),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
    BooleanAnd,
    BooleanOr,
    BooleanXor,
    BooleanEqual,
}

impl Binary {
    pub(in crate::jit) fn is_integer_arithmetic(self) -> bool {
        matches!(self, Self::Add | Self::Subtract | Self::Multiply)
    }

    pub(in crate::jit) fn is_float(self) -> bool {
        matches!(
            self,
            Self::FloatAdd
                | Self::FloatSubtract
                | Self::FloatMultiply
                | Self::FloatDivide
                | Self::FloatMin
                | Self::FloatMax
        )
    }

    pub(in crate::jit) fn is_boolean(self) -> bool {
        matches!(
            self,
            Self::BooleanAnd | Self::BooleanOr | Self::BooleanXor | Self::BooleanEqual
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub(in crate::jit) struct StackSlot(pub usize);

#[derive(Clone, Copy, Debug)]
pub(in crate::jit) struct StackSlotData {
    pub size: u32,
    pub align_shift: u8,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::jit) enum Reference {
    Retain,
    Release { heap: bool, cold: bool },
}

#[derive(Clone, Debug)]
pub(in crate::jit) enum Scratch {
    Begin,
    Own,
    Forward {
        target: Block,
    },
    Bind {
        owner: u32,
        required: bool,
        append: bool,
    },
    MoveOwner {
        owner: u32,
    },
    ReleaseOwner {
        owner: u32,
    },
    MoveValue,
    ReleaseToken {
        token: Inst,
    },
    End,
    Unwind {
        exclude: Option<Inst>,
    },
}

#[derive(Clone, Debug)]
pub(in crate::jit) enum Op {
    Cleanup {
        values: Vec<Value>,
        unwind: bool,
    },
    Scratch {
        op: Scratch,
        values: Vec<Value>,
    },
    FrameStore {
        ctx: Value,
        value: Value,
        slot: u32,
    },
    Reference {
        op: Reference,
        value: Value,
    },
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
    ExtractBits {
        value: Value,
        shift: u8,
        width: u8,
    },
    InRange {
        value: Value,
        low: u64,
        high: u64,
        inside: bool,
    },
    TestMask {
        value: Value,
        mask: u64,
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

#[derive(Clone, Copy, Debug)]
pub(in crate::jit) struct SignatureRef(usize);

#[derive(Clone, Copy, Debug)]
pub(in crate::jit) struct FunctionRef(usize);

pub(in crate::jit) struct Builder<'a> {
    pub(in crate::jit) graph: &'a mut Graph,
    location: Option<u32>,
    source_name: Option<(usize, usize)>,
    current: Option<Block>,
    cursor: Option<usize>,
    signatures: Vec<Signature>,
    functions: Vec<(FuncId, Signature)>,
    span: Span,
}

impl<'a> Builder<'a> {
    pub(in crate::jit) fn new(graph: &'a mut Graph) -> Self {
        let span = graph.span;
        Self {
            graph,
            location: None,
            source_name: None,
            current: None,
            cursor: None,
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
        self.cursor = None;
    }

    pub(in crate::jit) fn insert_at(&mut self, block: Block, position: usize) {
        self.current = Some(block);
        self.cursor = Some(position);
    }

    pub(in crate::jit) fn cursor(&self) -> Option<usize> {
        self.cursor
    }

    fn place_in(
        &mut self,
        block: Block,
        span: Span,
        op: GraphOp,
        has_result: bool,
    ) -> (Inst, Option<Value>) {
        let Some(position) = self.cursor.filter(|_| self.current == Some(block)) else {
            return self.graph.append(block, span, op, has_result);
        };
        self.cursor = Some(position + 1);
        self.graph.insert(block, position, span, op, has_result)
    }

    pub(in crate::jit) fn semantic(&mut self, op: GraphOp, repr: Repr) -> Value {
        self.add(op, repr)
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
        // Lifetime requests do not change which helper a status check belongs to.
        let inst = *self.graph.blocks[block].insts.iter().rev().find(|&&inst| {
            !matches!(
                self.graph.insts[inst].op,
                GraphOp::LowLevel(Op::Scratch { .. }) | GraphOp::AsBoxed(_) | GraphOp::AsWord(_)
            )
        })?;
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
        self.graph.reprs[result] = Some(Repr::numeric(ty));
        result
    }

    pub(in crate::jit) fn adopt(&mut self, inst: Inst) {
        let block = self.current.expect("no current block");
        self.graph.blocks[block].insts.push(inst);
        let data = &mut self.graph.insts[inst];
        data.block = block;
        data.emission_location = self.location;
        data.source_name = self.source_name;
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
        self.append_block_param_repr(block, Repr::from_machine(ty))
    }

    pub(in crate::jit) fn append_block_param_repr(&mut self, block: Block, repr: Repr) -> Value {
        let value = self.graph.add_param(block);
        self.graph.reprs[value] = Some(repr);
        value
    }

    #[track_caller]
    pub(in crate::jit) fn as_boxed(&mut self, word: Value) -> Value {
        debug_assert_eq!(self.graph.repr(word), Repr::Word);
        Insert::as_boxed(self, word)
    }

    #[track_caller]
    pub(in crate::jit) fn as_word(&mut self, value: Value) -> Value {
        debug_assert_eq!(self.graph.repr(value), Repr::Boxed);
        Insert::as_word(self, value)
    }

    pub(in crate::jit) fn block_params(&self, block: Block) -> &[Value] {
        &self.graph.blocks[block].params
    }

    pub(in crate::jit) fn scratch(&mut self, op: Scratch, values: &[Value]) -> Inst {
        self.scratch_owned(op, values.to_vec())
    }

    pub(in crate::jit) fn scratch_owned(&mut self, op: Scratch, values: Vec<Value>) -> Inst {
        self.effect(
            self.current.expect("no current block"),
            self.span,
            Op::Scratch { op, values },
        )
    }

    pub(in crate::jit) fn store_frame(&mut self, ctx: Value, value: Value, slot: u32) -> Value {
        self.semantic(
            GraphOp::LowLevel(Op::FrameStore { ctx, value, slot }),
            Repr::Boxed,
        )
    }

    pub(in crate::jit) fn reference(&mut self, op: Reference, value: Value) {
        self.effect(
            self.current.expect("no current block"),
            self.span,
            Op::Reference { op, value },
        );
    }

    pub(in crate::jit) fn ins(&mut self) -> Instructions<'_, Self> {
        Instructions { inserter: self }
    }

    pub(in crate::jit) fn set_source_name(&mut self, source_name: Option<(usize, usize)>) {
        self.source_name = source_name;
    }

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

    #[cfg(test)]
    pub(in crate::jit) fn value(&mut self, block: Block, span: Span, op: Op, ty: Type) -> Value {
        let (inst, value) = self.place_in(block, span, GraphOp::LowLevel(op), true);
        self.graph.insts[inst].emission_location = self.location;
        self.graph.insts[inst].source_name = self.source_name;
        let value = value.unwrap();
        self.graph.reprs[value] = Some(Repr::from_machine(ty));
        value
    }

    pub(in crate::jit) fn effect(&mut self, block: Block, span: Span, op: Op) -> Inst {
        let inst = self.place_in(block, span, GraphOp::LowLevel(op), false).0;
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
            self.graph.reprs[value] = Some(Repr::from_machine(ty));
        }
        inst
    }
}

impl Insert for Builder<'_> {
    fn graph(&self) -> &Graph {
        self.graph
    }

    fn graph_mut(&mut self) -> &mut Graph {
        self.graph
    }

    fn place(&mut self, op: GraphOp, has_result: bool) -> (Inst, Option<Value>) {
        let block = self.current.expect("no current block");
        let (inst, value) = self.place_in(block, self.span, op, has_result);
        self.graph.insts[inst].emission_location = self.location;
        self.graph.insts[inst].source_name = self.source_name;
        (inst, value)
    }

    fn terminate_current(&mut self, kind: super::TerminatorKind) {
        let block = self.current.expect("no current block");
        self.terminate(block, self.span, kind);
    }
}

pub(in crate::jit) struct Instructions<'b, I: Insert + ?Sized> {
    pub(super) inserter: &'b mut I,
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
            let ty = self.inserter.graph().reprs[value].map(Repr::machine).expect("untyped low-level operand");
            let immediate = self.iconst(ty, immediate);
            self.binary(Binary::$op, value, immediate)
        }
    )* };
}

macro_rules! unary_methods {
    ($($name:ident => $op:ident),* $(,)?) => { $(
        pub(in crate::jit) fn $name(&mut self, value: Value) -> Value {
            let repr = self.inserter.graph().reprs[value].expect("untyped low-level operand");
            let ty = repr.machine();
            self.inserter.add(GraphOp::LowLevel(Op::Unary { op: Unary::$op, ty, value }), repr)
        }
    )* };
}

macro_rules! conversion_methods {
    ($($name:ident => $op:ident),* $(,)?) => { $(
        pub(in crate::jit) fn $name(&mut self, ty: Type, value: Value) -> Value {
            self.inserter.add(GraphOp::LowLevel(Op::Unary { op: Unary::$op, ty, value }), Repr::numeric(ty))
        }
    )* };
}

impl<I: Insert + ?Sized> Instructions<'_, I> {
    fn value(&mut self, op: Op, ty: Type) -> Value {
        self.inserter
            .add(GraphOp::LowLevel(op), Repr::from_machine(ty))
    }

    fn predicate(&mut self, op: Op) -> Value {
        self.inserter.add(GraphOp::LowLevel(op), Repr::Boolean)
    }

    fn derived(&mut self, op: Op, first: Value, second: Value) -> Value {
        let graph = self.inserter.graph();
        let typed = |value: Value| graph.reprs[value].filter(|&repr| repr != Repr::Boxed);
        let (first_repr, second_repr) = (typed(first), typed(second));
        if let (Some(first), Some(second)) = (first_repr, second_repr) {
            assert_eq!(
                first.machine(),
                second.machine(),
                "mismatched low-level operands"
            );
        }
        let constant = matches!(
            graph.defining_op(first),
            Some(GraphOp::LowLevel(Op::Integer { .. } | Op::Float64(_)))
        );
        let repr = if constant {
            second_repr.or(first_repr)
        } else {
            first_repr.or(second_repr)
        };
        match repr {
            Some(repr) => self.inserter.add(GraphOp::LowLevel(op), repr),
            None => self.inserter.place(GraphOp::LowLevel(op), true).1.unwrap(),
        }
    }

    pub(in crate::jit) fn binary(&mut self, op: Binary, left: Value, right: Value) -> Value {
        let binary = Op::Binary { op, left, right };
        if op.is_boolean() {
            return self.inserter.add(GraphOp::LowLevel(binary), Repr::Boolean);
        }
        let graph = self.inserter.graph();
        assert!(
            [left, right]
                .iter()
                .all(|&value| graph.reprs[value] != Some(Repr::Boolean)),
            "boolean operand to {op:?}"
        );
        self.derived(binary, left, right)
    }

    binary_methods!(isub => Subtract, band => And, ishl => ShiftLeft,
        fsub => FloatSubtract, fmul => FloatMultiply, boolean_and => BooleanAnd,
        boolean_or => BooleanOr, boolean_equal => BooleanEqual);
    immediate_methods!(iadd_imm_s => Add, iadd_imm_u => Add, imul_imm_s => Multiply,
        band_imm_s => And, band_imm_u => And, ishl_imm_s => ShiftLeft,
        ushr_imm_s => ShiftRightUnsigned);

    unary_methods!(trunc => Trunc);
    conversion_methods!(ireduce => Reduce, sextend => SignExtend, uextend => ZeroExtend,
        fpromote => FloatPromote, fcvt_from_sint => FloatFromSigned,
        fcvt_from_uint => FloatFromUnsigned, fcvt_to_sint_sat => SignedSaturatingFromFloat);

    pub(in crate::jit) fn boolean(&mut self, value: bool) -> Value {
        self.inserter
            .add(GraphOp::Constant(Constant::Boolean(value)), Repr::Boolean)
    }

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
        self.predicate(Op::IntCompare {
            condition,
            left,
            right,
        })
    }

    pub(in crate::jit) fn icmp_imm_s(
        &mut self,
        condition: IntCC,
        left: Value,
        immediate: i64,
    ) -> Value {
        let ty = self.inserter.graph().reprs[left]
            .map(Repr::machine)
            .expect("untyped low-level operand");
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
        self.predicate(Op::FloatCompare {
            condition,
            left,
            right,
        })
    }

    pub(in crate::jit) fn select(&mut self, condition: Value, yes: Value, no: Value) -> Value {
        self.derived(Op::Select { condition, yes, no }, yes, no)
    }

    pub(in crate::jit) fn extract_bits(&mut self, value: Value, shift: u8, width: u8) -> Value {
        let repr = self.inserter.graph().reprs[value].expect("untyped low-level operand");
        assert!(u32::from(shift) + u32::from(width) <= repr.machine().bits() && width > 0);
        self.inserter.add(
            GraphOp::LowLevel(Op::ExtractBits {
                value,
                shift,
                width,
            }),
            repr,
        )
    }

    pub(in crate::jit) fn in_range(&mut self, value: Value, low: u64, high: u64) -> Value {
        self.range(value, low, high, true)
    }

    pub(in crate::jit) fn outside_range(&mut self, value: Value, low: u64, high: u64) -> Value {
        self.range(value, low, high, false)
    }

    fn range(&mut self, value: Value, low: u64, high: u64, inside: bool) -> Value {
        assert!(low <= high);
        let op = Op::InRange {
            value,
            low,
            high,
            inside,
        };
        self.predicate(op)
    }

    pub(in crate::jit) fn test_mask(&mut self, value: Value, mask: u64) -> Value {
        self.predicate(Op::TestMask { value, mask })
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
        self.inserter.add_effect(GraphOp::LowLevel(Op::Store {
            value,
            address,
            offset,
            flags,
        }))
    }

    pub(in crate::jit) fn jump(&mut self, destination: Block, args: &[BlockArg]) {
        self.inserter
            .terminate_current(super::TerminatorKind::Jump(super::Target {
                block: destination,
                args: args
                    .iter()
                    .map(|arg| match arg {
                        BlockArg::Value(value) => *value,
                    })
                    .collect(),
            }));
    }

    pub(in crate::jit) fn brif(
        &mut self,
        condition: Value,
        yes: Block,
        yes_args: &[BlockArg],
        no: Block,
        no_args: &[BlockArg],
    ) {
        self.inserter
            .terminate_current(super::TerminatorKind::Branch(
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
            ));
    }

    pub(in crate::jit) fn return_(&mut self, values: &[Value]) {
        self.inserter
            .terminate_current(super::TerminatorKind::ReturnValues(values.to_vec()));
    }
}

impl Instructions<'_, Builder<'_>> {
    pub(in crate::jit) fn call(&mut self, function: FunctionRef, args: &[Value]) -> Inst {
        let (id, signature) = self.inserter.functions[function.0].clone();
        let block = self.inserter.current.expect("no current block");
        self.inserter.call_inst(
            block,
            self.inserter.span,
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
        let signature = self.inserter.signatures[signature.0].clone();
        let block = self.inserter.current.expect("no current block");
        self.inserter.call_inst(
            block,
            self.inserter.span,
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
        let signature = self.inserter.signatures[signature.0].clone();
        let block = self.inserter.current.expect("no current block");
        self.inserter.terminate(
            block,
            self.inserter.span,
            super::TerminatorKind::TailCall {
                target: CallTarget::Indirect(address),
                signature,
                args: args.to_vec(),
            },
        );
    }
}

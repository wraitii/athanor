use cranelift_codegen::ir::{
    BlockArg, InstBuilder, MemFlags, Value as ClifValue,
    condcodes::{FloatCC, IntCC},
    types,
};
use cranelift_frontend::Variable;
use oxc_ast::ast::{
    BindingPattern, Expression, ForStatementInit, IfStatement, SimpleAssignmentTarget, Statement,
    VariableDeclarationKind,
};
use oxc_syntax::{
    operator::{BinaryOperator, UpdateOperator},
    symbol::SymbolId,
};

use super::{FunctionCompiler, ir_helpers};
use crate::jit::helpers::Helper;
use crate::jit::numeric::{IndexedStorageGuard, Kind};
use crate::value::Value;

#[derive(Clone, PartialEq, Eq)]
enum Key {
    Local(SymbolId),
    Add(SymbolId, SymbolId),
}

#[derive(Clone, PartialEq, Eq)]
enum Receiver {
    ThisProperty(String),
    Local(SymbolId),
}

#[derive(Clone, PartialEq, Eq)]
struct IndexedValue {
    receiver: Receiver,
    key: Key,
}

struct InclusiveInduction {
    index: SymbolId,
    start: SymbolId,
    end: SymbolId,
}

struct IndexedBitTest {
    read: IndexedValue,
    shift_index: SymbolId,
}

struct CounterIncrement {
    target: SymbolId,
}

struct BitLoop {
    induction: InclusiveInduction,
    bit_test: IndexedBitTest,
    increment: CounterIncrement,
}

pub(super) struct PreparedBitLoop {
    start: Variable,
    end: Variable,
    counter: Variable,
    description: String,
}

impl FunctionCompiler<'_, '_> {
    fn identifier(&self, expression: &Expression<'_>) -> Option<SymbolId> {
        let Expression::Identifier(identifier) = expression else {
            return None;
        };
        if self
            .arguments_references
            .contains_key(&identifier.span.start)
        {
            return None;
        }
        identifier
            .reference_id
            .get()
            .and_then(|id| self.scoping.get_reference(id).symbol_id())
    }

    fn increment(&self, expression: &Expression<'_>) -> Option<SymbolId> {
        let Expression::UpdateExpression(update) = expression else {
            return None;
        };
        if update.operator != UpdateOperator::Increment {
            return None;
        }
        let SimpleAssignmentTarget::AssignmentTargetIdentifier(identifier) = &update.argument
        else {
            return None;
        };
        if self
            .arguments_references
            .contains_key(&identifier.span.start)
        {
            return None;
        }
        identifier
            .reference_id
            .get()
            .and_then(|id| self.scoping.get_reference(id).symbol_id())
    }

    fn indexed_value(&self, expression: &Expression<'_>) -> Option<IndexedValue> {
        let Expression::ComputedMemberExpression(member) = expression else {
            return None;
        };
        if member.optional {
            return None;
        }
        let receiver = if let Some(symbol) = self.identifier(&member.object) {
            Receiver::Local(symbol)
        } else if let Expression::StaticMemberExpression(object) = &member.object {
            if object.optional || !matches!(object.object, Expression::ThisExpression(_)) {
                return None;
            }
            Receiver::ThisProperty(object.property.name.to_string())
        } else {
            return None;
        };
        let key = if let Some(symbol) = self.identifier(&member.expression) {
            Key::Local(symbol)
        } else if let Expression::BinaryExpression(binary) = &member.expression {
            if binary.operator != BinaryOperator::Addition {
                return None;
            }
            Key::Add(
                self.identifier(&binary.left)?,
                self.identifier(&binary.right)?,
            )
        } else {
            return None;
        };
        Some(IndexedValue { receiver, key })
    }

    fn recognize_bit_loop(&self, outer: &IfStatement<'_>) -> Option<BitLoop> {
        if !self.semantic_loops || self.generator.is_some() || outer.alternate.is_some() {
            return None;
        }
        let read = self.indexed_value(&outer.test)?;
        let Statement::ForStatement(loop_) = &outer.consequent else {
            return None;
        };
        let Some(ForStatementInit::VariableDeclaration(decl)) = &loop_.init else {
            return None;
        };
        if decl.kind != VariableDeclarationKind::Let || decl.declarations.len() != 1 {
            return None;
        }
        let declaration = &decl.declarations[0];
        let BindingPattern::BindingIdentifier(binding) = &declaration.id else {
            return None;
        };
        let index = binding.symbol_id.get()?;
        let start = self.identifier(declaration.init.as_ref()?)?;
        let Expression::BinaryExpression(test) = loop_.test.as_ref()? else {
            return None;
        };
        if test.operator != BinaryOperator::LessEqualThan || self.identifier(&test.left)? != index {
            return None;
        }
        let end = self.identifier(&test.right)?;
        if self.increment(loop_.update.as_ref()?)? != index {
            return None;
        }
        let Statement::IfStatement(body) = &loop_.body else {
            return None;
        };
        if body.alternate.is_some() {
            return None;
        }
        let Expression::BinaryExpression(bit_test) = &body.test else {
            return None;
        };
        if bit_test.operator != BinaryOperator::BitwiseAnd
            || self.indexed_value(&bit_test.left)? != read
        {
            return None;
        }
        let Expression::BinaryExpression(shift) = bit_test.right.without_parentheses() else {
            return None;
        };
        let shift_index = self.identifier(&shift.right)?;
        if shift.operator != BinaryOperator::ShiftLeft
            || !matches!(&shift.left, Expression::NumericLiteral(n) if n.value == 1.0)
            || shift_index != index
        {
            return None;
        }
        let Statement::ExpressionStatement(update) = &body.consequent else {
            return None;
        };
        let counter = self.increment(&update.expression)?;
        Some(BitLoop {
            induction: InclusiveInduction { index, start, end },
            bit_test: IndexedBitTest { read, shift_index },
            increment: CounterIncrement { target: counter },
        })
    }

    fn lower_bit_loop(&self, region: BitLoop) -> Option<PreparedBitLoop> {
        let InclusiveInduction { index, start, end } = region.induction;
        let counter = region.increment.target;
        let read = &region.bit_test.read;
        if region.bit_test.shift_index != index
            || [start, end, counter, index]
                .into_iter()
                .any(|symbol| self.captured.contains(&symbol))
            || start == counter
            || end == counter
            || counter == index
        {
            return None;
        }
        if let Receiver::Local(symbol) = &read.receiver {
            if *symbol == counter || *symbol == index || self.captured.contains(symbol) {
                return None;
            }
            self.local(*symbol)?;
        }
        match read.key {
            Key::Local(symbol)
                if symbol == counter || symbol == index || self.captured.contains(&symbol) =>
            {
                return None;
            }
            Key::Add(left, right)
                if [left, right].contains(&counter)
                    || [left, right].contains(&index)
                    || [left, right]
                        .into_iter()
                        .any(|symbol| self.captured.contains(&symbol)) =>
            {
                return None;
            }
            _ => {}
        }
        let name = |symbol| self.scoping.symbol_name(symbol);
        let receiver = match &read.receiver {
            Receiver::Local(symbol) => name(*symbol).to_owned(),
            Receiver::ThisProperty(property) => format!("this.{property}"),
        };
        let key = match read.key {
            Key::Local(symbol) => {
                self.local(symbol)?;
                name(symbol).to_owned()
            }
            Key::Add(left, right) => {
                self.local(left)?;
                self.local(right)?;
                format!("{} + {}", name(left), name(right))
            }
        };
        Some(PreparedBitLoop {
            start: self.local(start)?,
            end: self.local(end)?,
            counter: self.local(counter)?,
            description: format!(
                "word = index {receiver}[{key}]\nif truthy(word):\n  induction: {} = {}..={} (inclusive, step 1)\n  test: index {receiver}[{key}] & (1 << {})\n  on true: increment {}\n  fast-path guards: direct numeric index storage; word, {}, {} are int32; 0 <= {} <= {} <= 31; {} is number\n  fast path: reuse word; increment counter as f64 on each hit\n  fallback: original loop",
                name(index),
                name(start),
                name(end),
                name(index),
                name(counter),
                name(start),
                name(end),
                name(start),
                name(end),
                name(counter)
            ),
        })
    }

    pub(super) fn prepare_bit_loop(&self, outer: &IfStatement<'_>) -> Option<PreparedBitLoop> {
        self.lower_bit_loop(self.recognize_bit_loop(outer)?)
    }

    pub(super) fn record_bit_loop(&mut self, region: &PreparedBitLoop) {
        self.annotations
            .semantic_regions
            .push(region.description.clone());
    }

    pub(super) fn probe_bit_loop(&mut self, word: ClifValue) -> Option<ClifValue> {
        let read = *self.annotations.indexed_reads.last()?;
        if read.result != word {
            return None;
        }
        let result = self.call_helper(Helper::IndexProbe, &[read.receiver, read.key]);
        let call = self.b.func.dfg.value_def(result).unwrap_inst();
        self.annotations.indexed_storage_guards.insert(
            read.call,
            IndexedStorageGuard {
                call,
                result,
                contract: crate::heap::IndexedStorageContract::DirectNumericIndexStorage,
            },
        );
        Some(result)
    }

    pub(super) fn emit_bit_loop(
        &mut self,
        region: &PreparedBitLoop,
        word: ClifValue,
        probe: ClifValue,
        generic: cranelift_codegen::ir::Block,
        done: cranelift_codegen::ir::Block,
    ) {
        let start = self.b.use_var(region.start);
        let end = self.b.use_var(region.end);
        let counter = self.b.use_var(region.counter);
        let (word, word_ok) = ir_helpers::unbox_int32(self.b, word);
        let (start, start_ok) = ir_helpers::unbox_int32(self.b, start);
        let (end, end_ok) = ir_helpers::unbox_int32(self.b, end);
        let nonnegative = self
            .b
            .ins()
            .icmp_imm(IntCC::SignedGreaterThanOrEqual, start, 0);
        let bounded = self.b.ins().icmp_imm(IntCC::SignedLessThanOrEqual, end, 31);
        let ordered = self.b.ins().icmp(IntCC::SignedLessThanOrEqual, start, end);
        let number = self.is_number(counter);
        let ready = self
            .b
            .ins()
            .icmp_imm(IntCC::Equal, probe, Value::TRUE.bits() as i64);
        let mut valid = ready;
        for condition in [
            word_ok,
            start_ok,
            end_ok,
            nonnegative,
            bounded,
            ordered,
            number,
        ] {
            valid = self.b.ins().band(valid, condition);
        }
        let header = self.b.create_block();
        self.b.append_block_param(header, types::I32);
        self.b.append_block_param(header, types::F64);
        let finish = self.b.create_block();
        self.b.append_block_param(finish, types::F64);
        let counter = self.b.ins().bitcast(types::F64, MemFlags::new(), counter);
        self.brif(
            valid,
            header,
            &[BlockArg::Value(start), BlockArg::Value(counter)],
            generic,
            &[],
        );
        self.switch(header);
        let index = self.b.block_params(header)[0];
        let counter = self.b.block_params(header)[1];
        let one = self.b.ins().iconst(types::I32, 1);
        let mask = self.b.ins().ishl(one, index);
        let selected = self.b.ins().band(word, mask);
        let hit = self.b.ins().icmp_imm(IntCC::NotEqual, selected, 0);
        let one = self.b.ins().f64const(1.0);
        let added = self.b.ins().fadd(counter, one);
        let next_counter = self.b.ins().select(hit, added, counter);
        let next = self.b.ins().iadd_imm(index, 1);
        let more = self.b.ins().icmp(IntCC::SignedLessThanOrEqual, next, end);
        self.brif(
            more,
            header,
            &[BlockArg::Value(next), BlockArg::Value(next_counter)],
            finish,
            &[BlockArg::Value(next_counter)],
        );
        self.switch(finish);
        let result = self.b.block_params(finish)[0];
        let nan = self.b.ins().fcmp(FloatCC::Unordered, result, result);
        let bits = self.b.ins().bitcast(types::I64, MemFlags::new(), result);
        let canonical = self.constant(Value::number(f64::NAN));
        let result = self.b.ins().select(nan, canonical, bits);
        self.annotations.results.insert(result, Kind::Number);
        self.store_local(region.counter, result);
        self.jump(done, &[]);
    }
}

use crate::{FxHashMap, FxHashSet};
use std::fmt;

use cranelift_codegen::entity::SecondaryMap;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use oxc_syntax::operator::BinaryOperator;

use super::super::effects::{Control, Memory};
use super::access::Access;
use super::available;
use super::fresh::Fresh;
use super::guards;
use super::low_level::{Binary as LowBinary, Op as LowOp};
use super::{Block, Cfg, Constant, Graph, Inst, MemoryFormat, Op, TerminatorKind, UnaryOp, Value};
use crate::heap::TypedKind;
use cranelift_module::FuncId;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(in crate::jit) struct ValueType(u16);

impl ValueType {
    pub(in crate::jit) const INT32: Self = Self(1);
    pub(in crate::jit) const DOUBLE: Self = Self(2);
    pub(in crate::jit) const INT64: Self = Self(2048);
    pub(in crate::jit) const INTEGER: Self = Self(1 | 2048);
    pub(in crate::jit) const BOOLEAN: Self = Self(4);
    pub(in crate::jit) const UNDEFINED: Self = Self(8);
    pub(in crate::jit) const NULL: Self = Self(16);
    pub(in crate::jit) const STRING: Self = Self(32);
    pub(in crate::jit) const OBJECT: Self = Self(64);
    pub(in crate::jit) const OTHER: Self = Self(128);
    pub(in crate::jit) const UNINITIALIZED: Self = Self(256);
    pub(in crate::jit) const PACK: Self = Self(1024);
    pub(in crate::jit) const VIEW: Self = Self(512);
    pub(in crate::jit) const NUMBER: Self = Self(1 | 2 | 2048);
    pub(in crate::jit) const PRIMITIVE: Self = Self(1 | 2 | 4 | 8 | 16 | 32 | 2048);
    pub(in crate::jit) const VALUE: Self = Self((511 & !256) | 2048);
    pub(in crate::jit) const ALL: Self = Self(511 | 2048);

    pub(in crate::jit) const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub(in crate::jit) fn within(self, other: Self) -> bool {
        self.0 & !other.0 == 0
    }

    pub(in crate::jit) const fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    pub(in crate::jit) fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    pub(in crate::jit) fn is_informative(self) -> bool {
        !Self::VALUE.within(self)
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut names = Vec::new();
        if Self::NUMBER.within(*self) {
            names.push("number");
        } else {
            if self.intersects(Self::INT32) {
                names.push("int32");
            }
            if self.intersects(Self::INT64) {
                names.push("int64");
            }
            if self.intersects(Self::DOUBLE) {
                names.push("double");
            }
        }
        for (types, name) in [
            (Self::BOOLEAN, "bool"),
            (Self::UNDEFINED, "undefined"),
            (Self::NULL, "null"),
            (Self::STRING, "string"),
            (Self::OBJECT, "object"),
            (Self::OTHER, "other"),
            (Self::UNINITIALIZED, "uninit"),
            (Self::VIEW, "view"),
            (Self::PACK, "pack"),
        ] {
            if self.intersects(types) {
                names.push(name);
            }
        }
        if names.is_empty() {
            names.push("none");
        }
        f.write_str(&names.join("|"))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::jit) enum Identity {
    #[default]
    Unset,
    Code(FuncId),
    Many,
}

impl Identity {
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unset, other) | (other, Self::Unset) => other,
            (left, right) if left == right => left,
            _ => Self::Many,
        }
    }
}

struct Refinement {
    block: Block,
    mask: Option<ValueType>,
    positive: bool,
    identity: Option<FuncId>,
}

pub(in crate::jit) struct Facts {
    pub(in crate::jit) cfg: Cfg,
    refinements: FxHashMap<Value, Vec<Refinement>>,
    probes: Vec<(String, Value, Block, FuncId)>,
    rejected: FxHashSet<Inst>,
    identities: SecondaryMap<Value, Identity>,
    assumed: FxHashMap<Value, FuncId>,
    types: SecondaryMap<Value, ValueType>,
    indexed: FxHashSet<Value>,
    known: bool,
    fresh: std::cell::OnceCell<std::rc::Rc<Fresh>>,
    memory: SecondaryMap<Inst, Memory>,
    control: SecondaryMap<Inst, Control>,
}

impl Default for Facts {
    fn default() -> Self {
        Self {
            cfg: Cfg::default(),
            refinements: FxHashMap::default(),
            probes: Vec::new(),
            rejected: FxHashSet::default(),
            identities: SecondaryMap::new(),
            assumed: FxHashMap::default(),
            types: SecondaryMap::new(),
            indexed: FxHashSet::default(),
            known: false,
            fresh: Default::default(),
            memory: SecondaryMap::with_default(Memory::UNKNOWN),
            control: SecondaryMap::with_default(Control::UNKNOWN),
        }
    }
}

impl Facts {
    pub(super) fn indexed(&self, value: Value) -> bool {
        self.indexed.contains(&value)
    }

    pub(in crate::jit) fn of(&self, value: Value) -> ValueType {
        if self.known {
            self.types[value]
        } else {
            ValueType::ALL
        }
    }

    pub(super) fn set(&mut self, value: Value, types: ValueType) {
        self.types[value] = types;
    }

    pub(super) fn fresh(&self, graph: &Graph) -> std::rc::Rc<Fresh> {
        self.fresh
            .get_or_init(|| std::rc::Rc::new(Fresh::compute(graph, &self.cfg)))
            .clone()
    }

    pub(super) fn assumes(&self, value: Value) -> bool {
        self.assumed.contains_key(&value)
    }

    pub(in crate::jit) fn identity_at(&self, value: Value, block: Block) -> Identity {
        if let Some(&code) = self.assumed.get(&value) {
            return Identity::Code(code);
        }
        let mut identity = self.identities[value];
        for refinement in self.refinements.get(&value).into_iter().flatten() {
            if let Some(code) = refinement.identity
                && self.cfg.dominates(refinement.block, block)
            {
                identity = identity.join(Identity::Code(code));
            }
        }
        identity
    }

    pub(super) fn guarded_global(
        &self,
        name: &str,
        code: FuncId,
        inst: Inst,
        block: Block,
    ) -> bool {
        !self.rejected.contains(&inst)
            && self.probes.iter().any(|(probed, _, entry, guarded)| {
                probed == name && *guarded == code && self.cfg.dominates(*entry, block)
            })
    }

    pub(super) fn kills(&self, graph: &Graph, inst: Inst, write: &Access, read: &Access) -> bool {
        write.overlaps(read)
            && self.fresh(graph).may_alias(inst, write, read)
            && write.field.conflicts(&read.field)
    }

    fn unavailable_probes(&self, graph: &Graph) -> Vec<Inst> {
        let mut pending = Vec::new();
        for &block in &self.cfg.rpo {
            for &inst in &graph.blocks[block].insts {
                let Op::Call { callee, .. } = graph.insts[inst].op else {
                    continue;
                };
                if self.rejected.contains(&inst) || self.assumes(callee) {
                    continue;
                }
                if graph
                    .call_summary(inst, callee, self)
                    .is_some_and(|summary| !summary.requires.is_empty())
                {
                    pending.push(inst);
                }
            }
        }
        if pending.is_empty() {
            return pending;
        }
        let Some(solved) = available::solve(graph, self, &self.cfg, FxHashSet::default(), true)
        else {
            return pending;
        };
        pending
            .into_iter()
            .filter(|&inst| {
                let Op::Call { callee, .. } = graph.insts[inst].op else {
                    return false;
                };
                let Some(summary) = graph.call_summary(inst, callee, self) else {
                    return false;
                };
                let block = graph.insts[inst].block;
                let state = solved.before(graph, self, &self.cfg, inst);
                !summary.requires.iter().all(|(name, code)| {
                    self.probes.iter().any(|(probed, probe, entry, guarded)| {
                        probed == name
                            && guarded == code
                            && self.cfg.dominates(*entry, block)
                            && state
                                .get(&available::LoadKey::Probe(name.clone()))
                                .is_some_and(|available| {
                                    available.value == solved.canonical(*probe)
                                })
                    })
                })
            })
            .collect()
    }

    pub(super) fn is_pure(&self, inst: Inst) -> bool {
        self.memory[inst].is_empty() && self.control[inst].is_empty()
    }

    pub(super) fn removable(&self, inst: Inst) -> bool {
        self.memory[inst].writes.is_empty() && self.control[inst].is_empty()
    }

    pub(super) fn memory(&self, inst: Inst) -> Memory {
        self.memory[inst]
    }

    pub(in crate::jit) fn compute(graph: &Graph) -> Self {
        let mut facts = Self::values(graph);
        facts.compute_effects(graph);
        facts
    }

    // Effect queries remain conservative until compute_effects runs.
    pub(super) fn values(graph: &Graph) -> Self {
        Self::values_with_cfg(graph, Cfg::new(graph))
    }

    pub(super) fn values_with_cfg(graph: &Graph, cfg: Cfg) -> Self {
        Self::assuming_values(graph, cfg, &[])
    }

    pub(super) fn assuming(graph: &Graph, candidates: &[(Value, guards::Fact)]) -> Self {
        let mut facts = Self::assuming_values(graph, Cfg::new(graph), candidates);
        facts.compute_effects(graph);
        facts
    }

    pub(super) fn assuming_values(
        graph: &Graph,
        cfg: Cfg,
        candidates: &[(Value, guards::Fact)],
    ) -> Self {
        let entry_facts: Vec<_> = cfg
            .rpo
            .iter()
            .flat_map(|&block| &graph.blocks[block].insts)
            .filter_map(|&inst| match graph.insts[inst].op {
                Op::EntryAssume(value, fact) => Some((value, fact)),
                _ => None,
            })
            .chain(candidates.iter().copied())
            .collect();
        let candidates = &entry_facts;
        let assumptions: FxHashMap<Value, ValueType> = candidates
            .iter()
            .filter_map(|&(value, fact)| Some((value, fact.mask()?)))
            .collect();
        let identities: FxHashMap<Value, FuncId> = candidates
            .iter()
            .filter_map(|&(value, fact)| Some((value, fact.code()?)))
            .collect();
        let assumptions = &assumptions;
        let identities = &identities;
        let steps = steps(graph, &cfg);
        let mut refinements: FxHashMap<Value, Vec<Refinement>> = FxHashMap::default();
        for &block in &cfg.rpo {
            let Some(terminator) = &graph.blocks[block].terminator else {
                continue;
            };
            let Some(edge) = guards::edge_of(graph, &terminator.kind) else {
                continue;
            };
            for target in terminator.kind.targets() {
                // Dominance carries this edge's restriction only when the successor has no
                // other entry path. The function entry also has an implicit incoming path.
                if target.block != graph.entry && cfg.preds[target.block].len() == 1 {
                    let positive = target.block == edge.yes;
                    let identity = edge.fact.code().filter(|_| positive);
                    let mask = edge.fact.mask();
                    if mask.is_none() && identity.is_none() {
                        continue;
                    }
                    refinements.entry(edge.value).or_default().push(Refinement {
                        block: target.block,
                        mask,
                        positive,
                        identity,
                    });
                }
            }
        }
        let mut edge_facts: FxHashMap<(Block, Value), Vec<(ValueType, bool)>> =
            FxHashMap::default();
        for &block in &cfg.rpo {
            let Some(terminator) = &graph.blocks[block].terminator else {
                continue;
            };
            for arg in terminator
                .kind
                .targets()
                .into_iter()
                .flat_map(|target| &target.args)
            {
                for refinement in refinements.get(arg).into_iter().flatten() {
                    let Some(mask) = refinement.mask else {
                        continue;
                    };
                    if cfg.dominates(refinement.block, block) {
                        edge_facts
                            .entry((block, *arg))
                            .or_default()
                            .push((mask, refinement.positive));
                    }
                }
            }
        }
        let (indexed, reads) = super::string_split::result_facts(graph, &cfg);
        let mut probes = Vec::new();
        for (&value, entries) in &refinements {
            let Some(Op::ProbeGlobal(name)) = graph.defining_op(value) else {
                continue;
            };
            for refinement in entries {
                if let Some(code) = refinement.identity {
                    probes.push((name.clone(), value, refinement.block, code));
                }
            }
        }
        let rpo = cfg.rpo.clone();
        let marked: FxHashSet<Value> = graph
            .insts
            .values()
            .filter_map(|data| match data.op {
                Op::MarkClass(value) => Some(value),
                _ => None,
            })
            .collect();
        let mut facts = Self {
            cfg,
            refinements,
            probes,
            rejected: FxHashSet::default(),
            identities: SecondaryMap::new(),
            assumed: identities.clone(),
            types: SecondaryMap::new(),
            indexed,
            known: true,
            fresh: Default::default(),
            memory: SecondaryMap::with_default(Memory::UNKNOWN),
            control: SecondaryMap::with_default(Control::UNKNOWN),
        };
        for (&value, &types) in assumptions {
            facts.types[value] = types;
        }
        let mut changed = true;
        while changed {
            changed = false;
            for &block in &rpo {
                for &inst in &graph.blocks[block].insts {
                    let data = &graph.insts[inst];
                    for &result in &data.results {
                        if let Op::Closure(_, target, _, false, _, _, _) = data.op
                            && !marked.contains(&result)
                        {
                            changed |= facts.join_identity(result, Identity::Code(target));
                        }
                        if assumptions.contains_key(&result) {
                            continue;
                        }
                        let types = if let Some(&types) = reads.get(&inst) {
                            types
                        } else {
                            match steps.get(&inst) {
                                Some(bounds) if facts.bounded(&data.op, bounds) => ValueType::INT32,
                                _ => result_types(graph, &data.op, &facts),
                            }
                        };
                        changed |= facts.widen(result, types);
                    }
                }
                let Some(terminator) = &graph.blocks[block].terminator else {
                    continue;
                };
                let refinement = guards::edge_of(graph, &terminator.kind);
                for target in terminator.kind.targets() {
                    for (&param, &arg) in graph.blocks[target.block].params.iter().zip(&target.args)
                    {
                        let mut identity = facts.identity_at(arg, block);
                        if let Some(edge) = &refinement
                            && edge.value == arg
                            && target.block == edge.yes
                            && let Some(code) = edge.fact.code()
                        {
                            identity = identity.join(Identity::Code(code));
                        }
                        changed |= facts.join_identity(param, identity);
                        if assumptions.contains_key(&param) {
                            continue;
                        }
                        let mut types = facts.of(arg);
                        if let Some(edge) = &refinement
                            && edge.value == arg
                            && let Some(mask) = edge.fact.mask()
                        {
                            types = if target.block == edge.yes {
                                ValueType(types.0 & mask.0)
                            } else {
                                types.without(mask)
                            };
                        }
                        for &(mask, positive) in edge_facts.get(&(block, arg)).into_iter().flatten()
                        {
                            types = if positive {
                                ValueType(types.0 & mask.0)
                            } else {
                                types.without(mask)
                            };
                        }
                        changed |= facts.widen(param, types);
                    }
                }
            }
        }
        facts
    }

    pub(super) fn compute_effects(&mut self, graph: &Graph) {
        self.rejected.clear();
        self.fresh.take();
        self.memory.clear();
        self.control.clear();
        loop {
            for &block in &self.cfg.rpo {
                for &inst in &graph.blocks[block].insts {
                    let effects = graph.inst_contract(inst, self).effects;
                    self.memory[inst] = effects.memory;
                    self.control[inst] = effects.control;
                }
            }
            let unavailable = self.unavailable_probes(graph);
            if unavailable.is_empty() {
                break;
            }
            self.rejected.extend(unavailable);
        }
    }

    fn bounded(&self, op: &Op, bounds: &[Value]) -> bool {
        step(op).is_some_and(|(operand, _)| self.of(operand).within(ValueType::INT32))
            && bounds
                .iter()
                .any(|&bound| self.of(bound).within(ValueType::INT32))
    }

    fn join_identity(&mut self, value: Value, identity: Identity) -> bool {
        let joined = self.identities[value].join(identity);
        if joined == self.identities[value] {
            return false;
        }
        self.identities[value] = joined;
        true
    }

    fn widen(&mut self, value: Value, types: ValueType) -> bool {
        let widened = self.types[value].union(types);
        if widened == self.types[value] {
            return false;
        }
        self.types[value] = widened;
        true
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Relation {
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

impl Relation {
    fn comparison(op: &Op) -> Option<(Self, Value, Value)> {
        let (relation, left, right) = match *op {
            Op::Binary(operator, left, right)
            | Op::GuardedCompare {
                operator,
                left,
                right,
                ..
            } => (Self::of(operator)?, left, right),
            Op::LowLevel(LowOp::IntCompare {
                condition,
                left,
                right,
            }) => {
                let relation = match condition {
                    IntCC::SignedLessThan => Self::Less,
                    IntCC::SignedLessThanOrEqual => Self::LessEqual,
                    IntCC::SignedGreaterThan => Self::Greater,
                    IntCC::SignedGreaterThanOrEqual => Self::GreaterEqual,
                    _ => return None,
                };
                (relation, left, right)
            }
            Op::LowLevel(LowOp::FloatCompare {
                condition,
                left,
                right,
            }) => {
                let relation = match condition {
                    FloatCC::LessThan => Self::Less,
                    FloatCC::LessThanOrEqual => Self::LessEqual,
                    FloatCC::GreaterThan => Self::Greater,
                    FloatCC::GreaterThanOrEqual => Self::GreaterEqual,
                    _ => return None,
                };
                (relation, left, right)
            }
            _ => return None,
        };
        Some((relation, left, right))
    }

    fn of(operator: BinaryOperator) -> Option<Self> {
        Some(match operator {
            BinaryOperator::LessThan => Self::Less,
            BinaryOperator::LessEqualThan => Self::LessEqual,
            BinaryOperator::GreaterThan => Self::Greater,
            BinaryOperator::GreaterEqualThan => Self::GreaterEqual,
            _ => return None,
        })
    }

    fn negate(self) -> Self {
        match self {
            Self::Less => Self::GreaterEqual,
            Self::LessEqual => Self::Greater,
            Self::Greater => Self::LessEqual,
            Self::GreaterEqual => Self::Less,
        }
    }

    fn swap(self) -> Self {
        match self {
            Self::Less => Self::Greater,
            Self::LessEqual => Self::GreaterEqual,
            Self::Greater => Self::Less,
            Self::GreaterEqual => Self::LessEqual,
        }
    }
}

fn one(graph: &Graph, value: Value) -> bool {
    matches!(graph.defining_op(value), Some(Op::Constant(Constant::Number(n))) if *n == 1.0)
}

fn step(op: &Op) -> Option<(Value, bool)> {
    match *op {
        Op::Unary(UnaryOp::Increment, operand) => Some((operand, true)),
        Op::Unary(UnaryOp::Decrement, operand) => Some((operand, false)),
        Op::Binary(BinaryOperator::Addition, left, _) => Some((left, true)),
        Op::Binary(BinaryOperator::Subtraction, left, _) => Some((left, false)),
        Op::LowLevel(LowOp::Binary {
            op: LowBinary::Add,
            left,
            ..
        }) => Some((left, true)),
        Op::LowLevel(LowOp::Binary {
            op: LowBinary::Subtract,
            left,
            ..
        }) => Some((left, false)),
        _ => None,
    }
}

fn steps(graph: &Graph, cfg: &Cfg) -> FxHashMap<Inst, Vec<Value>> {
    let mut incoming: FxHashMap<Value, Vec<Value>> = FxHashMap::default();
    let mut relations: Vec<(Block, Value, Relation, Value)> = Vec::new();
    for &block in &cfg.rpo {
        let Some(terminator) = &graph.blocks[block].terminator else {
            continue;
        };
        for target in terminator.kind.targets() {
            for (&param, &arg) in graph.blocks[target.block].params.iter().zip(&target.args) {
                incoming.entry(param).or_default().push(arg);
            }
        }
        let TerminatorKind::Branch(condition, yes, no) = &terminator.kind else {
            continue;
        };
        let Some((relation, left, right)) =
            graph.defining_op(*condition).and_then(Relation::comparison)
        else {
            continue;
        };
        let source = |mut value| {
            while let Some(Op::NumericFloat(inner)) = graph.defining_op(value) {
                value = *inner;
            }
            value
        };
        let (left, right) = (source(left), source(right));
        if yes.block == no.block {
            continue;
        }
        for (target, relation) in [(yes.block, relation), (no.block, relation.negate())] {
            if cfg.preds[target].len() == 1 {
                relations.push((target, left, relation, right));
                relations.push((target, right, relation.swap(), left));
            }
        }
    }
    let mut steps = FxHashMap::default();
    for &block in &cfg.rpo {
        for &inst in &graph.blocks[block].insts {
            let op = &graph.insts[inst].op;
            let Some((operand, up)) = step(op) else {
                continue;
            };
            if let Op::Binary(_, left, right) | Op::LowLevel(LowOp::Binary { left, right, .. }) =
                *op
                && (left != operand || !one(graph, right))
            {
                continue;
            }
            let bounds: Vec<Value> = relations
                .iter()
                .filter(|(target, value, ..)| *value == operand && cfg.dominates(*target, block))
                .filter(|(.., relation, bound)| {
                    let (low, high) = range(graph, &incoming, *bound, 0);
                    match (up, relation) {
                        (true, Relation::Less) | (false, Relation::Greater) => true,
                        (true, Relation::LessEqual) => high < i32::MAX as i64,
                        (false, Relation::GreaterEqual) => low > i32::MIN as i64,
                        _ => false,
                    }
                })
                .map(|(.., bound)| *bound)
                .collect();
            if !bounds.is_empty() {
                steps.insert(inst, bounds);
            }
        }
    }
    steps
}

fn range(
    graph: &Graph,
    incoming: &FxHashMap<Value, Vec<Value>>,
    value: Value,
    depth: usize,
) -> (i64, i64) {
    const FULL: (i64, i64) = (i32::MIN as i64, i32::MAX as i64);
    let constant = |value: Value| match graph.defining_op(value) {
        Some(Op::Constant(Constant::Number(n)))
            if n.fract() == 0.0 && *n >= i32::MIN as f64 && *n <= i32::MAX as f64 =>
        {
            Some(*n as i64)
        }
        _ => None,
    };
    if depth > 8 {
        return FULL;
    }
    if let Some(n) = constant(value) {
        return (n, n);
    }
    match graph.defining_op(value) {
        Some(&Op::Binary(BinaryOperator::BitwiseAnd, left, right)) => {
            match constant(left).or(constant(right)) {
                Some(mask) if mask >= 0 => (0, mask),
                _ => FULL,
            }
        }
        Some(&Op::Binary(BinaryOperator::ShiftRight, _, count)) => match constant(count) {
            Some(count) => (FULL.0 >> (count & 31), FULL.1 >> (count & 31)),
            None => FULL,
        },
        Some(_) => FULL,
        None => incoming.get(&value).map_or(FULL, |args| {
            args.iter()
                .map(|&arg| range(graph, incoming, arg, depth + 1))
                .reduce(|(low, high), (l, h)| (low.min(l), high.max(h)))
                .unwrap_or(FULL)
        }),
    }
}

fn number(value: f64) -> ValueType {
    let integral = value.fract() == 0.0 && value >= i32::MIN as f64 && value <= i32::MAX as f64;
    if integral && !(value == 0.0 && value.is_sign_negative()) {
        ValueType::INT32
    } else {
        ValueType::DOUBLE
    }
}

fn integer_product(graph: &Graph, facts: &Facts, left: Value, right: Value) -> bool {
    // Positive factors avoid negative zero; 2^31 * 2^22 stays exactly representable.
    let scaled = |value, factor| {
        facts.of(value).within(ValueType::INT32)
            && matches!(graph.defining_op(factor), Some(Op::Constant(Constant::Number(n)))
                if *n > 0.0 && n.fract() == 0.0 && *n <= 4_194_304.0)
    };
    scaled(left, right) || scaled(right, left)
}

fn result_types(graph: &Graph, op: &Op, facts: &Facts) -> ValueType {
    match op {
        Op::OsrStart(..) => ValueType::INT32,
        Op::Continuation(body, _) => body.result,
        Op::ContinuationOutput(..) | Op::BindingLoad(_) => ValueType::ALL,
        Op::BindingRestore(..)
        | Op::ContinuationStore(..)
        | Op::Observe(..)
        | Op::ReadHint(..)
        | Op::Assume(..) => ValueType::default(),
        Op::Constant(constant) => match constant {
            Constant::Undefined => ValueType::UNDEFINED,
            Constant::Null => ValueType::NULL,
            Constant::Boolean(_) => ValueType::BOOLEAN,
            Constant::Number(value) => number(*value),
            Constant::String(_) => ValueType::STRING,
            Constant::Uninitialized => ValueType::UNINITIALIZED,
        },
        Op::NewObject
        | Op::NewInstance(_)
        | Op::ConstructorResult(..)
        | Op::NewArray(_)
        | Op::NewHoleyArray(_)
        | Op::IteratorNew(_)
        | Op::EnumerateNew(_) => ValueType::OBJECT,
        Op::MapElement(..) => ValueType::ALL,
        Op::IsArrayIteration(..) => ValueType::BOOLEAN,
        Op::IteratorNext(_) | Op::EnumerateNext(_) => ValueType::ALL,
        Op::EntryAssume(..)
        | Op::IsUninitialized(_)
        | Op::IsPropertyCached(..)
        | Op::IsShape(..) => ValueType::BOOLEAN,
        Op::ProbeProperty(..)
        | Op::ProbeShapeProperty(..)
        | Op::ProbeIndex(..)
        | Op::ConstructorPrototype(_)
        | Op::CachedProperty(..)
        | Op::ShapedGet(..)
        | Op::FillPropertyCache(..) => ValueType::ALL,
        Op::View(_) => ValueType::VIEW,
        Op::Convert(value, repr) => repr.coerce(facts.of(*value)),
        Op::LowLevel(LowOp::Binary { op, .. }) if op.is_float() => ValueType::NUMBER,
        Op::LowLevel(LowOp::Binary { op, .. }) if op.is_integer_arithmetic() => ValueType::INT64,
        Op::LowLevel(LowOp::IntCompare { .. } | LowOp::FloatCompare { .. }) => ValueType::BOOLEAN,
        Op::LowLevel(LowOp::Unary {
            op: super::low_level::Unary::FloatNeg,
            ..
        }) => ValueType::NUMBER,
        Op::LowLevel(LowOp::Unary {
            op: super::low_level::Unary::BitNot,
            ..
        }) => ValueType::INT32,
        Op::LowLevel(LowOp::Load { ty, .. }) if *ty == cranelift_codegen::ir::types::I32 => {
            ValueType::INT32
        }
        Op::LowLevel(LowOp::Load { ty, .. }) if *ty == cranelift_codegen::ir::types::I64 => {
            ValueType::INT64
        }
        Op::LowLevel(_) | Op::TypedData(_) | Op::Memory(_) => ValueType::default(),
        Op::MemoryNumber(_, format) => match format {
            MemoryFormat::U32 | MemoryFormat::F32 | MemoryFormat::F64 => ValueType::NUMBER,
            _ => ValueType::INT32,
        },
        Op::KeepAlive(_) => ValueType::default(),
        Op::TypedIndex(.., read) if read.bounded => match read.element {
            Some(TypedKind::Float32 | TypedKind::Float64 | TypedKind::Uint32) | None => {
                ValueType::NUMBER
            }
            Some(_) => ValueType::INT32,
        },
        Op::TypedIndex(..) => ValueType::NUMBER.union(ValueType::UNDEFINED),
        Op::IsTyped(..)
        | Op::IsIndexed(_)
        | Op::IsArrayIndex(..)
        | Op::IsArray(..)
        | Op::IsExactPosition(..)
        | Op::IsInBounds(..)
        | Op::BooleanAnd(..) => ValueType::BOOLEAN,
        Op::PropertyWriteCache(..) => ValueType::INT64,
        Op::CanOverwrite(..) => ValueType::BOOLEAN,
        Op::IndexedLength(_) => ValueType::INT64,
        Op::ArrayKind(_) => ValueType::INT32,
        Op::IsNumber(_)
        | Op::IsInteger(_)
        | Op::IntegerCheck(..)
        | Op::IsString(_)
        | Op::IsStringSplit(_)
        | Op::IsTypeof(..) => ValueType::BOOLEAN,
        Op::NewPack | Op::Spread(_) | Op::StringSplitParts(..) => ValueType::PACK,
        Op::StringSplit(..) => ValueType::OBJECT,
        Op::ArrayPosition(_) => ValueType::INT64,
        Op::ToInt32(_) => ValueType::INT32,
        Op::AsInteger(_) => ValueType::INT64,
        Op::NumericFloat(_) | Op::SaturatingInteger(_) => ValueType::NUMBER,
        Op::AsObject(_) => ValueType::OBJECT,
        Op::AsNumber(value) | Op::ToNumeric(value) | Op::Unary(UnaryOp::Plus, value)
            if facts.of(*value).within(ValueType::NUMBER) =>
        {
            facts.of(*value)
        }
        Op::AsNumber(_) | Op::ToNumeric(_) => ValueType::NUMBER,
        Op::Unary(UnaryOp::Increment | UnaryOp::Decrement, value)
            if facts.of(*value).within(ValueType::INTEGER) =>
        {
            ValueType::INT64
        }
        Op::Unary(operator, _) => match operator {
            UnaryOp::Not => ValueType::BOOLEAN,
            UnaryOp::Typeof => ValueType::STRING,
            UnaryOp::Plus => ValueType::NUMBER,
            UnaryOp::BitwiseNot => ValueType::INT32,
            UnaryOp::Negate | UnaryOp::Increment | UnaryOp::Decrement => ValueType::NUMBER,
        },
        Op::Binary(BinaryOperator::Addition | BinaryOperator::Subtraction, left, right)
            if facts.of(*left).within(ValueType::INTEGER)
                && facts.of(*right).within(ValueType::INTEGER) =>
        {
            ValueType::INT64
        }
        Op::Binary(BinaryOperator::Multiplication, left, right)
            if integer_product(graph, facts, *left, *right) =>
        {
            ValueType::INT64
        }
        Op::Binary(operator, left, right) => match operator {
            BinaryOperator::Addition => {
                let (left, right) = (facts.of(*left), facts.of(*right));
                let nonstring = ValueType::NUMBER
                    .union(ValueType::BOOLEAN)
                    .union(ValueType::UNDEFINED)
                    .union(ValueType::NULL);
                if left.within(nonstring) && right.within(nonstring) {
                    ValueType::NUMBER
                } else if left.within(ValueType::STRING) || right.within(ValueType::STRING) {
                    ValueType::STRING
                } else {
                    ValueType::NUMBER.union(ValueType::STRING)
                }
            }
            BinaryOperator::Subtraction
            | BinaryOperator::Multiplication
            | BinaryOperator::Division
            | BinaryOperator::Remainder
            | BinaryOperator::Exponential => ValueType::NUMBER,
            BinaryOperator::ShiftLeft
            | BinaryOperator::ShiftRight
            | BinaryOperator::BitwiseOR
            | BinaryOperator::BitwiseXOR
            | BinaryOperator::BitwiseAnd => ValueType::INT32,
            BinaryOperator::ShiftRightZeroFill => ValueType::NUMBER,
            _ => ValueType::BOOLEAN,
        },
        Op::GuardedCompare { .. } => ValueType::BOOLEAN,
        Op::CallNative(..) => ValueType::NUMBER,
        Op::IsUndefined(_)
        | Op::IsNullish(_)
        | Op::IsNative(..)
        | Op::IsCode(..)
        | Op::GuardValid(_) => ValueType::BOOLEAN,
        _ => ValueType::VALUE,
    }
}

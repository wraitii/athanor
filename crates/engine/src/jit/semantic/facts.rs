use std::collections::{HashMap, HashSet};
use std::fmt;

use cranelift_codegen::entity::SecondaryMap;
use oxc_syntax::operator::BinaryOperator;

use super::super::effects::{Control, Memory};
use super::access::Access;
use super::available;
use super::fresh::Fresh;
use super::guards;
use super::{Block, Cfg, Constant, Graph, Inst, Op, TerminatorKind, UnaryOp, Value};
use crate::heap::TypedKind;
use cranelift_module::FuncId;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(in crate::jit) struct Types(u16);

impl Types {
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

impl fmt::Display for Types {
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
    mask: Option<Types>,
    positive: bool,
    identity: Option<FuncId>,
}

pub(in crate::jit) struct Facts {
    pub(in crate::jit) cfg: Cfg,
    refinements: HashMap<Value, Vec<Refinement>>,
    probes: Vec<(String, Value, Block, FuncId)>,
    rejected: HashSet<Inst>,
    identities: SecondaryMap<Value, Identity>,
    assumed: HashMap<Value, FuncId>,
    types: SecondaryMap<Value, Types>,
    indexed: HashSet<Value>,
    known: bool,
    fresh: std::cell::OnceCell<std::rc::Rc<Fresh>>,
    memory: SecondaryMap<Inst, Memory>,
    control: SecondaryMap<Inst, Control>,
}

impl Default for Facts {
    fn default() -> Self {
        Self {
            cfg: Cfg::default(),
            refinements: HashMap::new(),
            probes: Vec::new(),
            rejected: HashSet::new(),
            identities: SecondaryMap::new(),
            assumed: HashMap::new(),
            types: SecondaryMap::new(),
            indexed: HashSet::new(),
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

    pub(in crate::jit) fn of(&self, value: Value) -> Types {
        if self.known {
            self.types[value]
        } else {
            Types::ALL
        }
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
        let Some(solved) = available::solve(graph, self, &self.cfg, HashSet::new(), true) else {
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
        let assumptions: HashMap<Value, Types> = candidates
            .iter()
            .filter_map(|&(value, fact)| Some((value, fact.mask()?)))
            .collect();
        let identities: HashMap<Value, FuncId> = candidates
            .iter()
            .filter_map(|&(value, fact)| Some((value, fact.code()?)))
            .collect();
        let assumptions = &assumptions;
        let identities = &identities;
        let steps = steps(graph, &cfg);
        let mut refinements: HashMap<Value, Vec<Refinement>> = HashMap::new();
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
        let mut edge_facts: HashMap<(Block, Value), Vec<(Types, bool)>> = HashMap::new();
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
        let marked: HashSet<Value> = graph
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
            rejected: HashSet::new(),
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
                    if let Some(result) = data.result {
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
                                Some(bounds) if facts.bounded(&data.op, bounds) => Types::INT32,
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
                                Types(types.0 & mask.0)
                            } else {
                                types.without(mask)
                            };
                        }
                        for &(mask, positive) in edge_facts.get(&(block, arg)).into_iter().flatten()
                        {
                            types = if positive {
                                Types(types.0 & mask.0)
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
        step(op).is_some_and(|(operand, _)| self.of(operand).within(Types::INT32))
            && bounds
                .iter()
                .any(|&bound| self.of(bound).within(Types::INT32))
    }

    fn join_identity(&mut self, value: Value, identity: Identity) -> bool {
        let joined = self.identities[value].join(identity);
        if joined == self.identities[value] {
            return false;
        }
        self.identities[value] = joined;
        true
    }

    fn widen(&mut self, value: Value, types: Types) -> bool {
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
        _ => None,
    }
}

fn steps(graph: &Graph, cfg: &Cfg) -> HashMap<Inst, Vec<Value>> {
    let mut incoming: HashMap<Value, Vec<Value>> = HashMap::new();
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
        let Some(
            &(Op::Binary(operator, left, right)
            | Op::GuardedCompare {
                operator,
                left,
                right,
                ..
            }),
        ) = graph.defining_op(*condition)
        else {
            continue;
        };
        let Some(relation) = Relation::of(operator) else {
            continue;
        };
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
    let mut steps = HashMap::new();
    for &block in &cfg.rpo {
        for &inst in &graph.blocks[block].insts {
            let op = &graph.insts[inst].op;
            let Some((operand, up)) = step(op) else {
                continue;
            };
            if let Op::Binary(_, left, right) = *op
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
    incoming: &HashMap<Value, Vec<Value>>,
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

fn number(value: f64) -> Types {
    let integral = value.fract() == 0.0 && value >= i32::MIN as f64 && value <= i32::MAX as f64;
    if integral && !(value == 0.0 && value.is_sign_negative()) {
        Types::INT32
    } else {
        Types::DOUBLE
    }
}

fn integer_product(graph: &Graph, facts: &Facts, left: Value, right: Value) -> bool {
    // Positive factors avoid negative zero; 2^31 * 2^22 stays exactly representable.
    let scaled = |value, factor| {
        facts.of(value).within(Types::INT32)
            && matches!(graph.defining_op(factor), Some(Op::Constant(Constant::Number(n)))
                if *n > 0.0 && n.fract() == 0.0 && *n <= 4_194_304.0)
    };
    scaled(left, right) || scaled(right, left)
}

fn result_types(graph: &Graph, op: &Op, facts: &Facts) -> Types {
    match op {
        Op::OsrStart(..) => Types::INT32,
        Op::Continuation(body, _) => body.result,
        Op::ContinuationOutput(..) => Types::ALL,
        Op::ContinuationStore(..) | Op::Observe(..) | Op::ReadHint(..) | Op::Assume(..) => {
            Types::default()
        }
        Op::Constant(constant) => match constant {
            Constant::Undefined => Types::UNDEFINED,
            Constant::Null => Types::NULL,
            Constant::Boolean(_) => Types::BOOLEAN,
            Constant::Number(value) => number(*value),
            Constant::String(_) => Types::STRING,
            Constant::Uninitialized => Types::UNINITIALIZED,
        },
        Op::NewObject
        | Op::NewInstance(_)
        | Op::ConstructorResult(..)
        | Op::NewArray(_)
        | Op::NewHoleyArray(_)
        | Op::IteratorNew(_)
        | Op::EnumerateNew(_) => Types::OBJECT,
        Op::MapElement(..) => Types::ALL,
        Op::IsArrayIteration(..) => Types::BOOLEAN,
        Op::IteratorNext(_) | Op::EnumerateNext(_) => Types::ALL,
        Op::EntryAssume(..)
        | Op::IsUninitialized(_)
        | Op::IsPropertyCached(..)
        | Op::IsShape(..) => Types::BOOLEAN,
        Op::ProbeProperty(..)
        | Op::ProbeShapeProperty(..)
        | Op::ProbeIndex(..)
        | Op::ConstructorPrototype(_)
        | Op::CachedProperty(..)
        | Op::ShapedGet(..)
        | Op::FillPropertyCache(..) => Types::ALL,
        Op::View(_) => Types::VIEW,
        Op::TypedIndex(.., read) if read.bounded => match read.element {
            Some(TypedKind::Float32 | TypedKind::Float64 | TypedKind::Uint32) | None => {
                Types::NUMBER
            }
            Some(_) => Types::INT32,
        },
        Op::TypedIndex(..) => Types::NUMBER.union(Types::UNDEFINED),
        Op::IsTyped(..)
        | Op::IsIndexed(_)
        | Op::IsArrayIndex(..)
        | Op::IsArray(..)
        | Op::IsExactPosition(..)
        | Op::IsInBounds(..)
        | Op::BooleanAnd(..) => Types::BOOLEAN,
        Op::IndexedLength(_) => Types::INT64,
        Op::ArrayKind(_) => Types::INT32,
        Op::IsNumber(_)
        | Op::IsInteger(_)
        | Op::IntegerCheck(..)
        | Op::IsString(_)
        | Op::IsStringSplit(_)
        | Op::IsTypeof(..) => Types::BOOLEAN,
        Op::NewPack | Op::Spread(_) | Op::StringSplitParts(..) => Types::PACK,
        Op::StringSplit(..) => Types::OBJECT,
        Op::ArrayPosition(_) => Types::INT64,
        Op::ToInt32(_) => Types::INT32,
        Op::AsInteger(_) => Types::INT64,
        Op::NumericFloat(_) | Op::SaturatingInteger(_) => Types::NUMBER,
        Op::AsObject(_) => Types::OBJECT,
        Op::AsNumber(value) | Op::ToNumeric(value) | Op::Unary(UnaryOp::Plus, value)
            if facts.of(*value).within(Types::NUMBER) =>
        {
            facts.of(*value)
        }
        Op::AsNumber(_) | Op::ToNumeric(_) => Types::NUMBER,
        Op::Unary(UnaryOp::Increment | UnaryOp::Decrement, value)
            if facts.of(*value).within(Types::INTEGER) =>
        {
            Types::INT64
        }
        Op::Unary(operator, _) => match operator {
            UnaryOp::Not => Types::BOOLEAN,
            UnaryOp::Typeof => Types::STRING,
            UnaryOp::Plus => Types::NUMBER,
            UnaryOp::BitwiseNot => Types::INT32,
            UnaryOp::Negate | UnaryOp::Increment | UnaryOp::Decrement => Types::NUMBER,
        },
        Op::Binary(BinaryOperator::Addition | BinaryOperator::Subtraction, left, right)
            if facts.of(*left).within(Types::INTEGER)
                && facts.of(*right).within(Types::INTEGER) =>
        {
            Types::INT64
        }
        Op::Binary(BinaryOperator::Multiplication, left, right)
            if integer_product(graph, facts, *left, *right) =>
        {
            Types::INT64
        }
        Op::Binary(operator, left, right) => match operator {
            BinaryOperator::Addition => {
                let (left, right) = (facts.of(*left), facts.of(*right));
                let nonstring = Types::NUMBER
                    .union(Types::BOOLEAN)
                    .union(Types::UNDEFINED)
                    .union(Types::NULL);
                if left.within(nonstring) && right.within(nonstring) {
                    Types::NUMBER
                } else if left.within(Types::STRING) || right.within(Types::STRING) {
                    Types::STRING
                } else {
                    Types::NUMBER.union(Types::STRING)
                }
            }
            BinaryOperator::Subtraction
            | BinaryOperator::Multiplication
            | BinaryOperator::Division
            | BinaryOperator::Remainder
            | BinaryOperator::Exponential => Types::NUMBER,
            BinaryOperator::ShiftLeft
            | BinaryOperator::ShiftRight
            | BinaryOperator::BitwiseOR
            | BinaryOperator::BitwiseXOR
            | BinaryOperator::BitwiseAnd => Types::INT32,
            BinaryOperator::ShiftRightZeroFill => Types::NUMBER,
            _ => Types::BOOLEAN,
        },
        Op::GuardedCompare { .. } => Types::BOOLEAN,
        Op::CallNative(..) => Types::NUMBER,
        Op::IsUndefined(_)
        | Op::IsNullish(_)
        | Op::IsNative(..)
        | Op::IsCode(..)
        | Op::GuardValid(_) => Types::BOOLEAN,
        _ => Types::VALUE,
    }
}

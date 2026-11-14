use super::variables::{Variable, Variables};
use super::{Block, Graph, Inst, InstData, Op, Value, low_level};
use crate::FxHashMap;
use low_level::Scratch;

struct Resolver<'g> {
    graph: &'g mut Graph,
    variables: Variables,
    at: InstData,
    empty: Value,
    active: Vec<Variable>,
    free: Vec<Variable>,
    tokens: FxHashMap<Inst, Variable>,
    values: FxHashMap<Value, Vec<Variable>>,
    owners: FxHashMap<u32, Vec<Variable>>,
}

impl Resolver<'_> {
    fn define(&mut self, variable: Variable, value: Value) {
        self.variables
            .define(self.graph, self.at.block, variable, value);
    }

    fn read(&mut self, variable: Variable) -> Value {
        self.variables
            .read(self.graph, self.at.block, self.at.span, variable)
    }

    fn cleanup(&mut self, values: Vec<Value>, unwind: bool) {
        if values.is_empty() {
            return;
        }
        let op = Op::LowLevel(low_level::Op::Cleanup { values, unwind });
        let (inst, _) = self.graph.append(self.at.block, self.at.span, op, false);
        self.graph.insts[inst].source_name = self.at.source_name;
        self.graph.insts[inst].emission_location = self.at.emission_location;
    }

    fn own(&mut self, token: Inst, value: Value) {
        let variable = match self.free.pop() {
            Some(variable) => variable,
            None => {
                let repr = self.graph.repr(self.empty);
                self.variables.declare(self.graph, self.empty, repr)
            }
        };
        self.define(variable, value);
        self.active.push(variable);
        self.tokens.insert(token, variable);
        self.values.insert(value, vec![variable]);
    }

    fn retire(&mut self, variables: &[Variable]) {
        debug_assert!(
            variables
                .iter()
                .all(|variable| self.active.contains(variable))
        );
        for references in self.values.values_mut().chain(self.owners.values_mut()) {
            references.retain(|variable| !variables.contains(variable));
        }
        self.active.retain(|variable| !variables.contains(variable));
        self.tokens
            .retain(|_, variable| !variables.contains(variable));
        self.free.extend_from_slice(variables);
    }

    fn release(&mut self, variables: &[Variable]) {
        let values = variables
            .iter()
            .map(|&variable| self.read(variable))
            .collect();
        self.cleanup(values, false);
        for &variable in variables {
            self.define(variable, self.empty);
        }
        self.retire(variables);
    }

    fn move_to_slot(&mut self, variables: &[Variable]) {
        for &variable in variables {
            self.define(variable, self.empty);
        }
        self.retire(variables);
    }

    // Incoming tokens are mutually exclusive; untaken paths keep empty holders.
    fn forward(&mut self, target: Block, args: &[Value]) {
        for (&param, value) in self.graph.blocks[target].params.iter().zip(args) {
            let Some(tokens) = self.values.get(value).cloned() else {
                continue;
            };
            let incoming = self.values.entry(param).or_default();
            for token in tokens {
                if !incoming.contains(&token) {
                    incoming.push(token);
                }
            }
        }
    }

    fn bind(
        &mut self,
        owner: u32,
        value: Value,
        required: bool,
        append: bool,
    ) -> Result<(), String> {
        let Some(tokens) = self.values.get(&value) else {
            if required {
                return Err(format!("v{owner}: owned result has no reference token"));
            }
            return Ok(());
        };
        if append {
            self.owners.entry(owner).or_default().extend(tokens);
        } else {
            self.owners.insert(owner, tokens.clone());
        }
        Ok(())
    }

    fn unwind(&mut self, exclude: Option<Inst>) {
        let excluded = exclude.and_then(|token| self.tokens.get(&token)).copied();
        let live = self.active.clone();
        let values = live
            .into_iter()
            .filter(|&token| Some(token) != excluded)
            .map(|token| self.read(token))
            .collect();
        self.cleanup(values, true);
    }

    fn apply(&mut self, inst: Inst, op: Scratch, values: &[Value]) -> Result<(), String> {
        match op {
            Scratch::Begin => self.values.clear(),
            Scratch::Own => self.own(inst, values[0]),
            Scratch::Forward { target } => self.forward(target, values),
            Scratch::Bind {
                owner,
                required,
                append,
            } => self.bind(owner, values[0], required, append)?,
            Scratch::MoveOwner { owner } => {
                if let Some(tokens) = self.owners.remove(&owner) {
                    self.move_to_slot(&tokens);
                }
            }
            Scratch::ReleaseOwner { owner } => {
                if let Some(tokens) = self.owners.remove(&owner) {
                    self.release(&tokens);
                }
            }
            Scratch::MoveValue => {
                let tokens = self
                    .values
                    .get(&values[0])
                    .cloned()
                    .ok_or_else(|| format!("{inst}: moved result has no reference token"))?;
                self.move_to_slot(&tokens);
            }
            Scratch::ReleaseToken { token } => {
                let variable = self
                    .tokens
                    .get(&token)
                    .copied()
                    .ok_or_else(|| format!("{inst}: reference token {token} is not live"))?;
                self.release(&[variable]);
            }
            Scratch::End => {
                let active = self.active.clone();
                self.release(&active);
                self.owners.clear();
                self.values.clear();
            }
            Scratch::Unwind { exclude } => self.unwind(exclude),
        }
        Ok(())
    }
}

// Resolve emission scopes into SSA cleanup operands before allowing graph rewrites.
pub(in crate::jit) fn resolve(graph: &mut Graph, empty: Value) -> Result<(), String> {
    let events: Vec<_> = graph
        .insts
        .iter()
        .filter_map(|(inst, data)| {
            matches!(
                data.op,
                Op::LowLevel(low_level::Op::Scratch { .. }) | Op::AsBoxed(_)
            )
            .then_some(inst)
        })
        .collect();
    let Some(&first) = events.first() else {
        return Variables::default().finish(graph);
    };
    let at = graph.insts[first].clone();
    let mut planner = Resolver {
        graph,
        variables: Variables::default(),
        at,
        empty,
        active: Vec::new(),
        free: Vec::new(),
        tokens: FxHashMap::default(),
        values: FxHashMap::default(),
        owners: FxHashMap::default(),
    };
    let mut replacements = FxHashMap::default();
    for inst in events {
        if let Op::AsBoxed(word) = planner.graph.insts[inst].op {
            if let Some(tokens) = planner.values.get(&word).cloned() {
                let result = planner.graph.insts[inst].result().unwrap();
                planner.values.insert(result, tokens);
            }
            continue;
        }
        let data = &mut planner.graph.insts[inst];
        let Op::LowLevel(low_level::Op::Scratch { op, values }) =
            std::mem::replace(&mut data.op, Op::LowLevel(low_level::Op::VariableRead))
        else {
            unreachable!()
        };
        planner.at = data.clone();
        let block = planner.at.block;
        let original_len = planner.graph.blocks[block].insts.len();
        planner.apply(inst, op, &values)?;
        let added = planner.graph.blocks[block].insts.split_off(original_len);
        replacements.insert(inst, added);
    }
    for block in planner.graph.blocks.keys().collect::<Vec<_>>() {
        if !planner.graph.blocks[block]
            .insts
            .iter()
            .any(|inst| replacements.contains_key(inst))
        {
            continue;
        }
        let insts = std::mem::take(&mut planner.graph.blocks[block].insts);
        let mut rebuilt = Vec::with_capacity(insts.len());
        for inst in insts {
            if let Some(added) = replacements.remove(&inst) {
                rebuilt.extend(added);
            } else {
                rebuilt.push(inst);
            }
        }
        planner.graph.blocks[block].insts = rebuilt;
    }
    debug_assert!(replacements.is_empty());
    debug_assert!(planner.active.is_empty());
    let Resolver {
        graph, variables, ..
    } = planner;
    variables.finish(graph)
}

use cranelift_codegen::ir::types;

use super::{Block, Inst, Op, Value, low_level};
use crate::FxHashMap;
use low_level::{Builder, Scratch, Variable};

struct Resolver<'a, 'g> {
    b: &'a mut Builder<'g>,
    empty: Value,
    active: Vec<Variable>,
    free: Vec<Variable>,
    tokens: FxHashMap<Inst, Variable>,
    values: FxHashMap<Value, Vec<Variable>>,
    owners: FxHashMap<u32, Vec<Variable>>,
}

impl Resolver<'_, '_> {
    fn own(&mut self, token: Inst, value: Value) {
        let variable = self
            .free
            .pop()
            .unwrap_or_else(|| self.b.declare_var_with_entry_value(types::I64, self.empty));
        self.b.def_var(variable, value);
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
            .map(|&variable| self.b.use_var(variable))
            .collect();
        self.b.cleanup(values, false);
        for &variable in variables {
            self.b.def_var(variable, self.empty);
        }
        self.retire(variables);
    }

    fn move_to_slot(&mut self, variables: &[Variable]) {
        for &variable in variables {
            self.b.def_var(variable, self.empty);
        }
        self.retire(variables);
    }

    // Incoming tokens are mutually exclusive; untaken paths keep empty holders.
    fn forward(&mut self, target: Block, args: &[Value]) {
        for (&param, value) in self.b.block_params(target).iter().zip(args) {
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
            .map(|token| self.b.use_var(token))
            .collect();
        self.b.cleanup(values, true);
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
pub(in crate::jit) fn resolve(b: &mut Builder<'_>, empty: Value) -> Result<(), String> {
    let events: Vec<_> = b
        .graph
        .insts
        .iter()
        .filter_map(|(inst, data)| {
            matches!(data.op, Op::LowLevel(low_level::Op::Scratch { .. })).then_some(inst)
        })
        .collect();
    let mut planner = Resolver {
        b,
        empty,
        active: Vec::new(),
        free: Vec::new(),
        tokens: FxHashMap::default(),
        values: FxHashMap::default(),
        owners: FxHashMap::default(),
    };
    let mut replacements = FxHashMap::default();
    for inst in events {
        let data = &mut planner.b.graph.insts[inst];
        let Op::LowLevel(low_level::Op::Scratch { op, values }) =
            std::mem::replace(&mut data.op, Op::LowLevel(low_level::Op::VariableRead))
        else {
            unreachable!()
        };
        let (block, span, source_name, location) = (
            data.block,
            data.span,
            data.source_name,
            data.emission_location,
        );
        let original_len = planner.b.graph.blocks[block].insts.len();
        planner.b.switch_to_block(block);
        planner.b.set_span(span);
        planner.b.set_source_name(source_name);
        planner.b.set_location(location);
        planner.apply(inst, op, &values)?;
        let added = planner.b.graph.blocks[block].insts.split_off(original_len);
        replacements.insert(inst, added);
    }
    for block in planner.b.graph.blocks.keys().collect::<Vec<_>>() {
        if !planner.b.graph.blocks[block]
            .insts
            .iter()
            .any(|inst| replacements.contains_key(inst))
        {
            continue;
        }
        let insts = std::mem::take(&mut planner.b.graph.blocks[block].insts);
        let mut rebuilt = Vec::with_capacity(insts.len());
        for inst in insts {
            if let Some(added) = replacements.remove(&inst) {
                rebuilt.extend(added);
            } else {
                rebuilt.push(inst);
            }
        }
        planner.b.graph.blocks[block].insts = rebuilt;
    }
    debug_assert!(replacements.is_empty());
    debug_assert!(planner.active.is_empty());
    Ok(())
}

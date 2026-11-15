use crate::FxHashMap;
use crate::jit::semantic::variables::{Variable, Variables};
use crate::jit::semantic::{Block, Graph, Op, Value, low_level};
use low_level::{BlockArg, Builder};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Token(usize);

pub(super) struct References {
    variables: Variables,
    empty: Value,
    next_token: usize,
    active: Vec<Variable>,
    free: Vec<Variable>,
    tokens: FxHashMap<Token, Variable>,
    values: FxHashMap<Value, Vec<Variable>>,
    owners: FxHashMap<u32, Vec<Variable>>,
}

impl References {
    pub(super) fn new(empty: Value) -> Self {
        Self {
            variables: Variables::default(),
            empty,
            next_token: 0,
            active: Vec::new(),
            free: Vec::new(),
            tokens: FxHashMap::default(),
            values: FxHashMap::default(),
            owners: FxHashMap::default(),
        }
    }

    pub(super) fn begin(&mut self) {
        self.values.clear();
    }

    fn tokens_for(&self, graph: &Graph, mut value: Value) -> Option<&Vec<Variable>> {
        loop {
            if let Some(tokens) = self.values.get(&value) {
                return Some(tokens);
            }
            let Some(&Op::AsBoxed(word)) = graph.defining_op(value) else {
                return None;
            };
            value = word;
        }
    }

    fn define(&mut self, b: &mut Builder<'_>, variable: Variable, value: Value) {
        self.variables
            .define(b.graph, b.current_block().unwrap(), variable, value);
    }

    fn read(&mut self, b: &mut Builder<'_>, variable: Variable) -> Value {
        self.variables.read(
            b.graph,
            b.current_block().unwrap(),
            b.current_span(),
            variable,
        )
    }

    fn cleanup(&mut self, b: &mut Builder<'_>, values: Vec<Value>, unwind: bool) {
        if values.is_empty() {
            return;
        }
        b.cleanup(values, unwind);
    }

    pub(super) fn own(&mut self, b: &mut Builder<'_>, value: Value) -> Token {
        let token = Token(self.next_token);
        self.next_token += 1;
        let variable = match self.free.pop() {
            Some(variable) => variable,
            None => {
                let repr = b.graph.repr(self.empty);
                self.variables.declare(b.graph, self.empty, repr)
            }
        };
        self.define(b, variable, value);
        self.active.push(variable);
        self.tokens.insert(token, variable);
        self.values.insert(value, vec![variable]);
        token
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

    fn release(&mut self, b: &mut Builder<'_>, variables: &[Variable]) {
        let values = variables
            .iter()
            .map(|&variable| self.read(b, variable))
            .collect();
        self.cleanup(b, values, false);
        for &variable in variables {
            self.define(b, variable, self.empty);
        }
        self.retire(variables);
    }

    fn move_to_slot(&mut self, b: &mut Builder<'_>, variables: &[Variable]) {
        for &variable in variables {
            self.define(b, variable, self.empty);
        }
        self.retire(variables);
    }

    pub(super) fn forward(&mut self, b: &Builder<'_>, target: Block, args: &[BlockArg]) {
        for (&param, &BlockArg::Value(value)) in b.graph.blocks[target].params.iter().zip(args) {
            let Some(tokens) = self.tokens_for(b.graph, value).cloned() else {
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

    pub(super) fn bind(
        &mut self,
        b: &Builder<'_>,
        owner: u32,
        value: Value,
        required: bool,
        append: bool,
    ) -> Result<(), String> {
        let Some(tokens) = self.tokens_for(b.graph, value).cloned() else {
            if required {
                return Err(format!("v{owner}: owned result has no reference token"));
            }
            return Ok(());
        };
        if append {
            self.owners.entry(owner).or_default().extend(tokens);
        } else {
            self.owners.insert(owner, tokens);
        }
        Ok(())
    }

    pub(super) fn unwind(&mut self, b: &mut Builder<'_>, exclude: Option<Token>) {
        let excluded = exclude.and_then(|token| self.tokens.get(&token)).copied();
        let live = self.active.clone();
        let values = live
            .into_iter()
            .filter(|&token| Some(token) != excluded)
            .map(|token| self.read(b, token))
            .collect();
        self.cleanup(b, values, true);
    }

    pub(super) fn move_owner(&mut self, b: &mut Builder<'_>, owner: u32) {
        if let Some(tokens) = self.owners.remove(&owner) {
            self.move_to_slot(b, &tokens);
        }
    }

    pub(super) fn release_owner(&mut self, b: &mut Builder<'_>, owner: u32) {
        if let Some(tokens) = self.owners.remove(&owner) {
            self.release(b, &tokens);
        }
    }

    pub(super) fn move_value(&mut self, b: &mut Builder<'_>, value: Value) -> Result<(), String> {
        let tokens = self
            .tokens_for(b.graph, value)
            .cloned()
            .ok_or_else(|| format!("{value}: moved result has no reference token"))?;
        self.move_to_slot(b, &tokens);
        Ok(())
    }

    pub(super) fn release_token(
        &mut self,
        b: &mut Builder<'_>,
        token: Token,
    ) -> Result<(), String> {
        let variable = self
            .tokens
            .get(&token)
            .copied()
            .ok_or_else(|| format!("reference token {token:?} is not live"))?;
        self.release(b, &[variable]);
        Ok(())
    }

    pub(super) fn end(&mut self, b: &mut Builder<'_>) {
        let active = self.active.clone();
        self.release(b, &active);
        self.owners.clear();
        self.values.clear();
    }

    pub(super) fn finish(self, graph: &mut Graph) -> Result<(), String> {
        debug_assert!(self.active.is_empty());
        self.variables.finish(graph)
    }
}

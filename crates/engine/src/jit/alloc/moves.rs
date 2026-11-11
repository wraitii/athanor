use super::RegClass;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Loc {
    Reg(RegClass, u8),
    Slot(u32),
    Imm(u64),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Move {
    pub class: RegClass,
    pub from: Loc,
    pub to: Loc,
}

pub(crate) fn sequence(mut pending: Vec<Move>, scratch: impl Fn(RegClass) -> Loc) -> Vec<Move> {
    pending.retain(|m| m.from != m.to);
    let mut ordered = Vec::with_capacity(pending.len());
    while !pending.is_empty() {
        let ready = (0..pending.len())
            .find(|&index| pending.iter().all(|other| other.from != pending[index].to));
        if let Some(index) = ready {
            ordered.push(pending.remove(index));
            continue;
        }
        let blocked = pending[0].to;
        let class = pending
            .iter()
            .find(|m| m.from == blocked)
            .expect("a cycle reads every destination")
            .class;
        let temporary = scratch(class);
        ordered.push(Move {
            class,
            from: blocked,
            to: temporary,
        });
        for m in &mut pending {
            if m.from == blocked {
                m.from = temporary;
            }
        }
    }
    ordered
}

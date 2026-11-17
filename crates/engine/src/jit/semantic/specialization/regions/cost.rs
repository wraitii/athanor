use super::super::super::guards::{Fact, Origin};

pub(in crate::jit::semantic) struct Candidate<'a> {
    pub(in crate::jit::semantic) size: usize,
    pub(in crate::jit::semantic) checks: usize,
    pub(in crate::jit::semantic) reentry: Option<&'a str>,
    pub(in crate::jit::semantic) survivors: &'a [(Fact, Origin)],
}

impl Candidate<'_> {
    pub(in crate::jit::semantic) fn gain(&self) -> u32 {
        self.survivors
            .iter()
            .map(|(_, origin)| u32::from(origin.gain))
            .sum()
    }

    pub(in crate::jit::semantic) fn summary(&self) -> String {
        format!(
            "{} checks, gain {}, size {}, {}",
            self.checks,
            self.gain(),
            self.size,
            if self.reentry.is_some() {
                "reentrant"
            } else {
                "closed"
            }
        )
    }
}

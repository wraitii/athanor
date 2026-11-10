use super::super::guards::{Fact, Origin};

pub(super) struct Candidate<'a> {
    pub(super) size: usize,
    pub(super) checks: usize,
    pub(super) reentry: Option<&'a str>,
    pub(super) survivors: &'a [(Fact, Origin)],
}

pub(super) enum Verdict {
    Open,
    Skip(String),
}

impl Candidate<'_> {
    pub(super) fn gain(&self) -> u32 {
        self.survivors
            .iter()
            .map(|(_, origin)| u32::from(origin.gain))
            .sum()
    }

    pub(super) fn summary(&self) -> String {
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

pub(super) fn worth(candidate: &Candidate<'_>) -> Verdict {
    match candidate.reentry {
        Some(reason) => Verdict::Skip(reason.to_owned()),
        None => Verdict::Open,
    }
}

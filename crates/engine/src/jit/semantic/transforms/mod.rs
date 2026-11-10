mod promote;
mod simplify;
mod trim;

pub(super) use promote::{promote, remove_trivial_params};
pub(super) use simplify::{Fold, Folder, simplify};
pub(super) use trim::trim;

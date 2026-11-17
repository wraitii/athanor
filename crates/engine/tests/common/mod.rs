use std::path::Path;

pub fn map_library(name: &str) -> String {
    let parent = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let paths = ["binaries", "0ad/binaries"].map(|root| {
        parent
            .join(root)
            .join("data/mods/public/maps/random/rmgen")
            .join(name)
    });
    let path = paths
        .iter()
        .find(|path| path.is_file())
        .unwrap_or_else(|| panic!("map fixture {name} not found in {paths:?}"));
    std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("cannot read map fixture {}: {error}", path.display()))
}

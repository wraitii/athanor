pub trait Host {
    fn read_file(&self, path: &str) -> Result<String, String>;
    fn list_files(&self, path: &str, filter: &str, recursive: bool) -> Result<String, String>;
    fn print(&self, _message: &str) {}
}

use std::fs;
use std::path::{Component, Path, PathBuf};

pub struct FileHost;

fn matches_filter(filename: &str, filter: &str) -> bool {
    if filter.is_empty() || filter == "*" {
        return true;
    }
    if let Some(suffix) = filter.strip_prefix('*') {
        return filename.ends_with(suffix);
    }
    filename == filter
}

fn normalized(path: &Path) -> Result<String, String> {
    let mut result = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir
                if matches!(result.components().next_back(), Some(Component::Normal(_))) =>
            {
                result.pop();
            }
            Component::ParentDir if result.has_root() && result.parent().is_none() => {}
            component => result.push(component.as_os_str()),
        }
    }
    result
        .to_str()
        .map(|text| {
            if cfg!(windows) {
                text.replace('\\', "/")
            } else {
                text.to_owned()
            }
        })
        .ok_or_else(|| "path is not valid UTF-8".into())
}

fn list(path: &Path, filter: &str, recursive: bool, files: &mut Vec<String>) -> Result<(), String> {
    let entries = fs::read_dir(path)
        .map_err(|error| format!("Unable to list {}: {error}", path.display()))?;
    for entry in entries {
        let entry = entry.map_err(|error| error.to_string())?;
        let file_type = entry.file_type().map_err(|error| error.to_string())?;
        if recursive && file_type.is_dir() {
            list(&entry.path(), filter, true, files)?;
        } else if fs::metadata(entry.path())
            .map_err(|error| error.to_string())?
            .is_file()
        {
            let name = entry.file_name();
            let name = name.to_str().ok_or("filename is not valid UTF-8")?;
            if matches_filter(name, filter) {
                files.push(normalized(&entry.path())?);
            }
        }
    }
    Ok(())
}

impl js_engine::Host for FileHost {
    fn read_file(&self, path: &str) -> Result<String, String> {
        fs::read_to_string(path).map_err(|error| format!("Unable to read {path}: {error}"))
    }

    fn print(&self, message: &str) {
        use std::io::Write;
        let mut stdout = std::io::stdout().lock();
        let _ = stdout.write_all(message.as_bytes());
        let _ = stdout.flush();
    }

    fn list_files(&self, path: &str, filter: &str, recursive: bool) -> Result<String, String> {
        let mut files = Vec::new();
        list(Path::new(path), filter, recursive, &mut files)?;
        files.sort();
        if files.is_empty() {
            return Ok(String::new());
        }
        Ok(files.join("\n") + "\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use js_engine::Host;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT: AtomicUsize = AtomicUsize::new(0);

    struct Directory(PathBuf);

    impl Directory {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "0ad-host-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for Directory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn filesystem_host_reads_utf8_and_lists_sorted_matching_paths() {
        let directory = Directory::new();
        fs::write(directory.0.join("b.js"), "é😀").unwrap();
        fs::write(directory.0.join("a.js"), "a").unwrap();
        fs::write(directory.0.join("skip.txt"), "skip").unwrap();
        fs::create_dir(directory.0.join("sub")).unwrap();
        fs::write(directory.0.join("sub/c.js"), "c").unwrap();
        let path = directory.0.to_str().unwrap();
        assert_eq!(
            FileHost
                .read_file(directory.0.join("b.js").to_str().unwrap())
                .unwrap(),
            "é😀"
        );
        let expected = [directory.0.join("a.js"), directory.0.join("b.js")]
            .map(|path| normalized(&path).unwrap());
        assert_eq!(
            FileHost.list_files(path, "*.js", false).unwrap(),
            expected.join("\n") + "\n"
        );
        let recursive = FileHost.list_files(path, "*.js", true).unwrap();
        assert_eq!(recursive.lines().count(), 3);
        assert!(recursive.contains("sub/c.js"));
        assert_eq!(
            FileHost.list_files(path, "a.js", false).unwrap(),
            expected[0].clone() + "\n"
        );
        assert!(
            FileHost
                .read_file(directory.0.join("absent").to_str().unwrap())
                .is_err()
        );
        assert!(
            FileHost
                .list_files(directory.0.join("absent").to_str().unwrap(), "*", false)
                .is_err()
        );
    }
}

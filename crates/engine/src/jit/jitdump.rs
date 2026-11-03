use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::PathBuf;

const JITDUMP_MAGIC: u32 = 0x4a69_5444;
const JITDUMP_VERSION: u32 = 1;
const HEADER_SIZE: u32 = 40;
const CODE_LOAD: u32 = 0;
const CODE_DEBUG_INFO: u32 = 2;
const DEBUG_INFO_HEADER_SIZE: usize = 32;
const DEBUG_ENTRY_SIZE: usize = 16;
const CODE_LOAD_HEADER_SIZE: usize = 56;
const ELF_MACHINE_AARCH64: u32 = 183;
const ELF_MACHINE_X86_64: u32 = 62;

#[cfg(target_os = "macos")]
mod sys {
    unsafe extern "C" {
        fn clock_gettime_nsec_np(clock_id: u32) -> u64;
        fn pthread_threadid_np(thread: usize, id: *mut u64) -> i32;
    }

    pub fn clock_ns() -> u64 {
        const CLOCK_UPTIME_RAW: u32 = 8;
        unsafe { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) }
    }

    pub fn thread_id() -> u32 {
        let mut id = 0u64;
        unsafe { pthread_threadid_np(0, &mut id) };
        id as u32
    }
}

#[cfg(target_os = "linux")]
mod sys {
    #[repr(C)]
    struct Timespec {
        sec: i64,
        nsec: i64,
    }

    unsafe extern "C" {
        fn clock_gettime(clock_id: i32, out: *mut Timespec) -> i32;
        fn gettid() -> i32;
    }

    pub fn clock_ns() -> u64 {
        const CLOCK_MONOTONIC: i32 = 1;
        let mut ts = Timespec { sec: 0, nsec: 0 };
        unsafe { clock_gettime(CLOCK_MONOTONIC, &mut ts) };
        ts.sec as u64 * 1_000_000_000 + ts.nsec as u64
    }

    pub fn thread_id() -> u32 {
        unsafe { gettid() as u32 }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
mod sys {
    pub fn clock_ns() -> u64 {
        0
    }

    pub fn thread_id() -> u32 {
        std::process::id()
    }
}

pub(super) struct Writer {
    pid: u32,
    file: File,
    path: PathBuf,
    next_index: u64,
    next_listing: u64,
}

fn write_listing(
    source: &str,
    instructions: &[(u32, String)],
    out: &mut impl Write,
) -> io::Result<(Vec<u32>, BTreeMap<u32, u32>)> {
    let mut mapping = Vec::new();
    let mut ir_mapping = BTreeMap::new();
    let mut next_line = 1;
    for (index, line) in source.split('\n').enumerate() {
        mapping.push(next_line);
        writeln!(out, "{:>5} | {line}", index + 1)?;
        next_line += 1;
    }
    for (id, instruction) in instructions {
        writeln!(out, "{instruction}")?;
        if *id != 0 {
            ir_mapping.insert(*id, next_line);
        }
        next_line += 1;
    }
    Ok((mapping, ir_mapping))
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn absolute_filename(filename: &str) -> String {
    let path = std::path::Path::new(filename);
    if path.is_absolute() {
        return filename.to_string();
    }
    std::path::absolute(path)
        .ok()
        .and_then(|path| path.to_str().map(str::to_owned))
        .unwrap_or_else(|| filename.to_string())
}

impl Writer {
    pub(super) fn new() -> Option<Self> {
        match Self::create() {
            Ok(writer) => Some(writer),
            Err(error) => {
                eprintln!("js jit: jitdump disabled: {error}");
                None
            }
        }
    }

    fn create() -> io::Result<Self> {
        let arch = if cfg!(target_arch = "aarch64") {
            ELF_MACHINE_AARCH64
        } else if cfg!(target_arch = "x86_64") {
            ELF_MACHINE_X86_64
        } else {
            return Err(io::Error::other("unsupported host architecture"));
        };
        let pid = std::process::id();
        let timestamp = sys::clock_ns();
        let path = std::env::temp_dir().join(format!("jit-{pid}-{timestamp}-js.dump"));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        let mut header = Vec::with_capacity(HEADER_SIZE as usize);
        for value in [JITDUMP_MAGIC, JITDUMP_VERSION, HEADER_SIZE, arch, 0, pid] {
            push_u32(&mut header, value);
        }
        push_u64(&mut header, timestamp);
        push_u64(&mut header, 0);
        file.write_all(&header)?;
        eprintln!("js jit: jitdump {}", path.display());
        Ok(Self {
            pid,
            file,
            path,
            next_index: 0,
            next_listing: 0,
        })
    }

    pub(super) fn listing(
        &mut self,
        source: &str,
        complete_ir: &[(u32, String)],
    ) -> io::Result<(PathBuf, Vec<u32>, BTreeMap<u32, u32>)> {
        let path = self
            .path
            .with_extension(format!("{}.jsir", self.next_listing));
        self.next_listing += 1;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        let (mapping, ir_mapping) = write_listing(source, complete_ir, &mut file)?;
        Ok((path, mapping, ir_mapping))
    }

    pub(super) fn record(
        &mut self,
        name: &str,
        filename: &str,
        locations: &[(u32, u32)],
        code: &[u8],
    ) {
        if let Err(error) = self.try_record(name, filename, locations, code) {
            eprintln!("js jit: jitdump write failed: {error}");
        }
    }

    fn try_record(
        &mut self,
        name: &str,
        filename: &str,
        locations: &[(u32, u32)],
        code: &[u8],
    ) -> io::Result<()> {
        let filename = absolute_filename(filename);
        let filename = filename.as_str();
        let timestamp = sys::clock_ns();
        let address = code.as_ptr() as u64;

        let debug_size =
            DEBUG_INFO_HEADER_SIZE + locations.len() * (DEBUG_ENTRY_SIZE + filename.len() + 1);
        let mut debug = Vec::with_capacity(debug_size);
        push_u32(&mut debug, CODE_DEBUG_INFO);
        push_u32(&mut debug, debug_size as u32);
        push_u64(&mut debug, timestamp);
        push_u64(&mut debug, address);
        push_u64(&mut debug, locations.len() as u64);
        for &(offset, line) in locations {
            push_u64(&mut debug, address + u64::from(offset));
            push_u32(&mut debug, line);
            push_u32(&mut debug, 0);
            debug.extend_from_slice(filename.as_bytes());
            debug.push(0);
        }
        self.file.write_all(&debug)?;

        let total = CODE_LOAD_HEADER_SIZE + name.len() + 1 + code.len();
        let mut record = Vec::with_capacity(total);
        push_u32(&mut record, CODE_LOAD);
        push_u32(&mut record, total as u32);
        push_u64(&mut record, timestamp);
        push_u32(&mut record, self.pid);
        push_u32(&mut record, sys::thread_id());
        push_u64(&mut record, address);
        push_u64(&mut record, address);
        push_u64(&mut record, code.len() as u64);
        push_u64(&mut record, self.next_index);
        record.extend_from_slice(name.as_bytes());
        record.push(0);
        record.extend_from_slice(code);
        self.file.write_all(&record)?;
        self.next_index += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_filenames_are_absolute() {
        let relative = absolute_filename("relative.js");
        assert!(std::path::Path::new(&relative).is_absolute());
        assert_eq!(absolute_filename(&relative), relative);
    }

    #[test]
    fn listing_preserves_source_lines_and_maps_annotations() {
        let instructions = vec![(0, "block0(v0: i64):".into()), (1, "    return v0".into())];
        let mut listing = Vec::new();
        let (mapping, ir_mapping) =
            write_listing("let a;\nlet b;\n", &instructions, &mut listing).unwrap();
        assert_eq!(mapping, vec![1, 2, 3]);
        assert_eq!(ir_mapping.get(&1), Some(&5));
        assert_eq!(
            String::from_utf8(listing).unwrap(),
            "    1 | let a;\n    2 | let b;\n    3 | \nblock0(v0: i64):\n    return v0\n"
        );
    }
}

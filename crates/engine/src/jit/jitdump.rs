use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::PathBuf;

use crate::FxHashMap;

use super::semantic::ListingLine;

#[derive(Clone)]
pub(super) struct Listing {
    pub lines: Vec<ListingLine>,
    pub locations: FxHashMap<u32, u32>,
}

impl Listing {
    pub fn new(lines: Vec<ListingLine>) -> Self {
        Self {
            lines,
            locations: FxHashMap::default(),
        }
    }

    pub fn remap(
        &mut self,
        lowered: Vec<ListingLine>,
        source: usize,
        mut line_at: impl FnMut(usize, u32) -> Option<u32>,
    ) {
        let mut spans = FxHashMap::default();
        let mut ids = FxHashMap::default();
        let mut lines = FxHashMap::default();
        for (id, location, _) in &self.lines {
            let Some(location) = location else {
                continue;
            };
            let source = location.source.unwrap_or(source);
            let key = (source, location.span.start, location.span.end);
            spans.entry(key).or_insert(*id);
            ids.insert(*id, key);
            if let Some(line) = line_at(source, location.span.start) {
                lines.entry((source, line)).or_insert(*id);
            }
        }
        for (id, location, _) in lowered {
            let Some(location) = location else {
                continue;
            };
            let source = location.source.unwrap_or(source);
            let key = (source, location.span.start, location.span.end);
            let target = if ids.get(&id) == Some(&key) {
                Some(id)
            } else {
                spans.get(&key).copied().or_else(|| {
                    let line = line_at(source, location.span.start)?;
                    lines.get(&(source, line)).copied()
                })
            };
            if let Some(target) = target {
                self.locations.insert(id, target);
            }
        }
    }
}

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
    instructions: &[(u32, String)],
    out: &mut impl Write,
) -> io::Result<BTreeMap<u32, u32>> {
    let mut ir_mapping = BTreeMap::new();
    for (index, (id, instruction)) in instructions.iter().enumerate() {
        writeln!(out, "{instruction}")?;
        if *id != 0 {
            ir_mapping.insert(*id, index as u32 + 1);
        }
    }
    Ok(ir_mapping)
}

fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .take(80)
        .collect()
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
        name: &str,
        ir: &[(u32, String)],
    ) -> io::Result<(PathBuf, BTreeMap<u32, u32>)> {
        let path =
            self.path
                .with_extension(format!("{}.{}.jsir", self.next_listing, sanitize(name)));
        self.next_listing += 1;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        let mut out = io::BufWriter::new(file);
        let mapping = write_listing(ir, &mut out)?;
        out.flush()?;
        Ok((path, mapping))
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
    fn listing_maps_annotations_to_lines() {
        let instructions = vec![
            (0, "block0(v0: i64):".into()),
            (0, "  ;     1 | let a;".into()),
            (1, "    return v0".into()),
        ];
        let mut listing = Vec::new();
        let mapping = write_listing(&instructions, &mut listing).unwrap();
        assert_eq!(mapping.get(&1), Some(&3));
    }

    #[test]
    fn listing_remaps_lowered_locations_to_specialized_ir() {
        use super::super::semantic::SourceLocation;
        use oxc_span::Span;

        let line = |id, source, start, end| {
            (
                id,
                Some(SourceLocation {
                    source,
                    span: Span::new(start, end),
                }),
                String::new(),
            )
        };
        let mut listing = Listing::new(vec![
            line(1, None, 10, 30),
            line(2, None, 15, 20),
            line(3, None, 15, 20),
            line(4, Some(1), 15, 20),
            line(5, None, 40, 50),
        ]);
        listing.remap(
            vec![
                line(3, None, 15, 20),
                line(5, None, 15, 20),
                line(6, None, 16, 19),
                line(7, Some(1), 15, 20),
                line(8, Some(2), 15, 20),
                line(9, Some(0), 40, 50),
                line(10, None, 30, 35),
            ],
            0,
            |_, start| Some(start / 10),
        );
        assert_eq!(listing.locations.get(&3), Some(&3));
        assert_eq!(listing.locations.get(&5), Some(&2));
        assert_eq!(listing.locations.get(&6), Some(&1));
        assert_eq!(listing.locations.get(&7), Some(&4));
        assert_eq!(listing.locations.get(&8), None);
        assert_eq!(listing.locations.get(&9), Some(&5));
        assert_eq!(listing.locations.get(&10), None);
    }
}

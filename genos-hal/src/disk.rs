use alloc::vec::Vec;
use core::ops::{Deref, DerefMut};

/// Disk operation errors.
pub enum DiskError {
    NoFileSystem,
    ProtocolError,
    InvalidPath,
    ReadError,
    WriteError,
    NotFound,
    OutOfMemory,
}

impl DiskError {
    pub fn as_str(&self) -> &str {
        match self {
            DiskError::NoFileSystem => "no filesystem found",
            DiskError::ProtocolError => "protocol error",
            DiskError::InvalidPath => "invalid path",
            DiskError::ReadError => "read error",
            DiskError::WriteError => "write error",
            DiskError::NotFound => "file not found",
            DiskError::OutOfMemory => "out of memory",
        }
    }
}

/// Buffer for large files that should avoid the default UEFI pool allocator.
pub struct LargeFileBuffer {
    #[cfg(not(feature = "hosted"))]
    ptr: core::ptr::NonNull<u8>,
    #[cfg(not(feature = "hosted"))]
    len: usize,
    #[cfg(not(feature = "hosted"))]
    page_count: usize,
    #[cfg(feature = "hosted")]
    data: Vec<u8>,
}

impl LargeFileBuffer {
    #[cfg(not(feature = "hosted"))]
    fn new_paged(len: usize) -> Result<Self, DiskError> {
        use uefi::boot::{self, AllocateType, MemoryType};

        if len == 0 {
            return Ok(Self {
                ptr: core::ptr::NonNull::dangling(),
                len: 0,
                page_count: 0,
            });
        }

        let page_count = len.div_ceil(4096);
        let ptr = boot::allocate_pages(AllocateType::AnyPages, MemoryType::LOADER_DATA, page_count)
            .map_err(|_| DiskError::OutOfMemory)?;

        Ok(Self {
            ptr,
            len,
            page_count,
        })
    }

    #[cfg(feature = "hosted")]
    fn from_vec(data: Vec<u8>) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        #[cfg(not(feature = "hosted"))]
        {
            self.len
        }
        #[cfg(feature = "hosted")]
        {
            self.data.len()
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        #[cfg(not(feature = "hosted"))]
        {
            unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
        #[cfg(feature = "hosted")]
        {
            &self.data
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        #[cfg(not(feature = "hosted"))]
        {
            unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
        #[cfg(feature = "hosted")]
        {
            &mut self.data
        }
    }
}

impl Deref for LargeFileBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for LargeFileBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl AsRef<[u8]> for LargeFileBuffer {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

#[cfg(not(feature = "hosted"))]
impl Drop for LargeFileBuffer {
    fn drop(&mut self) {
        if self.page_count == 0 {
            return;
        }

        unsafe {
            let _ = uefi::boot::free_pages(self.ptr, self.page_count);
        }
    }
}

// ── UEFI implementation ────────────────────────────────────────────
#[cfg(not(feature = "hosted"))]
mod uefi_impl {
    use super::*;
    use uefi::boot;
    use uefi::fs::FileSystem;
    use uefi::proto::media::file::{File, FileAttribute, FileInfo, FileMode, RegularFile};
    use uefi::proto::media::fs::SimpleFileSystem;
    use uefi::CString16;

    fn open_regular_file(path: &str) -> Result<RegularFile, DiskError> {
        let handle = boot::get_handle_for_protocol::<SimpleFileSystem>()
            .map_err(|_| DiskError::NoFileSystem)?;
        let mut sfs = boot::open_protocol_exclusive::<SimpleFileSystem>(handle)
            .map_err(|_| DiskError::ProtocolError)?;
        let mut root = sfs.open_volume().map_err(|_| DiskError::ProtocolError)?;
        let path16 = CString16::try_from(path).map_err(|_| DiskError::InvalidPath)?;
        let file = root
            .open(path16.as_ref(), FileMode::Read, FileAttribute::empty())
            .map_err(|_| DiskError::ReadError)?;

        file.into_regular_file().ok_or(DiskError::ReadError)
    }

    pub fn read_file(path: &str) -> Result<Vec<u8>, DiskError> {
        let handle = boot::get_handle_for_protocol::<SimpleFileSystem>()
            .map_err(|_| DiskError::NoFileSystem)?;
        let sfs = boot::open_protocol_exclusive::<SimpleFileSystem>(handle)
            .map_err(|_| DiskError::ProtocolError)?;
        let mut fs = FileSystem::new(sfs);
        let path16 = CString16::try_from(path).map_err(|_| DiskError::InvalidPath)?;
        fs.read(path16.as_ref()).map_err(|_| DiskError::ReadError)
    }

    pub fn write_file(path: &str, data: &[u8]) -> Result<(), DiskError> {
        let handle = boot::get_handle_for_protocol::<SimpleFileSystem>()
            .map_err(|_| DiskError::NoFileSystem)?;
        let sfs = boot::open_protocol_exclusive::<SimpleFileSystem>(handle)
            .map_err(|_| DiskError::ProtocolError)?;
        let mut fs = FileSystem::new(sfs);
        let path16 = CString16::try_from(path).map_err(|_| DiskError::InvalidPath)?;
        fs.write(path16.as_ref(), data).map_err(|_| DiskError::WriteError)
    }

    pub fn read_file_paged(path: &str) -> Result<LargeFileBuffer, DiskError> {
        read_file_paged_with_progress(path, |_, _| {})
    }

    /// Read a large file with a progress callback called every chunk.
    /// callback(bytes_read_so_far, total_bytes)
    pub fn read_file_paged_with_progress(
        path: &str,
        progress: impl Fn(usize, usize),
    ) -> Result<LargeFileBuffer, DiskError> {
        let mut file = open_regular_file(path)?;
        let info = file
            .get_boxed_info::<FileInfo>()
            .map_err(|_| DiskError::ReadError)?;
        let len = info.file_size() as usize;
        let mut data = LargeFileBuffer::new_paged(len)?;

        // Read in 4 MB chunks with progress reporting.
        // UEFI file read under QEMU TCG is very slow for multi-GB files.
        const CHUNK: usize = 4 * 1024 * 1024;
        let mut offset = 0usize;
        while offset < len {
            let remaining = len - offset;
            let to_read = if remaining < CHUNK { remaining } else { CHUNK };
            let read_bytes = file
                .read(&mut data.as_mut_slice()[offset..offset + to_read])
                .map_err(|_| DiskError::ReadError)?;
            if read_bytes == 0 {
                return Err(DiskError::ReadError);
            }
            offset += read_bytes;
            progress(offset, len);
        }

        Ok(data)
    }
}

// ── Hosted (std) implementation ────────────────────────────────────
#[cfg(feature = "hosted")]
mod hosted_impl {
    use super::*;
    use std::path::PathBuf;

    /// Convert a genos backslash path to a host filesystem path.
    /// Root is `GENOS_TEST_ROOT` env var or `/tmp/genos_test`.
    fn to_host_path(path: &str) -> PathBuf {
        let root = std::env::var("GENOS_TEST_ROOT")
            .unwrap_or_else(|_| String::from("/tmp/genos_test"));
        let normalized = path.replace('\\', "/");
        let trimmed = normalized.trim_start_matches('/');
        PathBuf::from(root).join(trimmed)
    }

    pub fn read_file(path: &str) -> Result<Vec<u8>, DiskError> {
        let host_path = to_host_path(path);
        std::fs::read(&host_path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                DiskError::NotFound
            } else {
                DiskError::ReadError
            }
        })
    }

    pub fn write_file(path: &str, data: &[u8]) -> Result<(), DiskError> {
        let host_path = to_host_path(path);
        if let Some(parent) = host_path.parent() {
            std::fs::create_dir_all(parent).map_err(|_| DiskError::WriteError)?;
        }
        std::fs::write(&host_path, data).map_err(|_| DiskError::WriteError)
    }

    pub fn read_file_paged(path: &str) -> Result<LargeFileBuffer, DiskError> {
        read_file(path).map(LargeFileBuffer::from_vec)
    }

    pub fn read_file_paged_with_progress(
        path: &str,
        _progress: impl Fn(usize, usize),
    ) -> Result<LargeFileBuffer, DiskError> {
        read_file(path).map(LargeFileBuffer::from_vec)
    }
}

// ── Public API (delegates to active implementation) ────────────────

/// Read a file from the EFI system partition.
/// Path should use backslashes, e.g. "\\models\\stories15m.bin"
pub fn read_file(path: &str) -> Result<Vec<u8>, DiskError> {
    #[cfg(not(feature = "hosted"))]
    { uefi_impl::read_file(path) }
    #[cfg(feature = "hosted")]
    { hosted_impl::read_file(path) }
}

/// Read a large file from the EFI system partition into a page-backed buffer.
/// Intended for multi-gigabyte model files that should avoid pool allocation.
pub fn read_file_paged(path: &str) -> Result<LargeFileBuffer, DiskError> {
    #[cfg(not(feature = "hosted"))]
    { uefi_impl::read_file_paged(path) }
    #[cfg(feature = "hosted")]
    { hosted_impl::read_file_paged(path) }
}

/// Read a large file with a progress callback.
/// callback(bytes_read_so_far, total_bytes) is called every chunk.
pub fn read_file_paged_with_progress(
    path: &str,
    progress: impl Fn(usize, usize),
) -> Result<LargeFileBuffer, DiskError> {
    #[cfg(not(feature = "hosted"))]
    { uefi_impl::read_file_paged_with_progress(path, progress) }
    #[cfg(feature = "hosted")]
    { hosted_impl::read_file_paged_with_progress(path, progress) }
}

/// Write data to a file on the EFI system partition.
pub fn write_file(path: &str, data: &[u8]) -> Result<(), DiskError> {
    #[cfg(not(feature = "hosted"))]
    { uefi_impl::write_file(path, data) }
    #[cfg(feature = "hosted")]
    { hosted_impl::write_file(path, data) }
}

use alloc::vec::Vec;

/// Disk operation errors.
pub enum DiskError {
    NoFileSystem,
    ProtocolError,
    InvalidPath,
    ReadError,
    WriteError,
    NotFound,
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
        }
    }
}

// ── UEFI implementation ────────────────────────────────────────────
#[cfg(not(feature = "hosted"))]
mod uefi_impl {
    use super::*;
    use uefi::boot;
    use uefi::fs::FileSystem;
    use uefi::proto::media::fs::SimpleFileSystem;
    use uefi::CString16;

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

/// Write data to a file on the EFI system partition.
pub fn write_file(path: &str, data: &[u8]) -> Result<(), DiskError> {
    #[cfg(not(feature = "hosted"))]
    { uefi_impl::write_file(path, data) }
    #[cfg(feature = "hosted")]
    { hosted_impl::write_file(path, data) }
}

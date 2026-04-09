use alloc::vec::Vec;
use uefi::boot;
use uefi::fs::FileSystem;
use uefi::proto::media::fs::SimpleFileSystem;
use uefi::CString16;

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

/// Read a file from the EFI system partition.
/// Path should use backslashes, e.g. "\\models\\stories15m.bin"
pub fn read_file(path: &str) -> Result<Vec<u8>, DiskError> {
    let handle = boot::get_handle_for_protocol::<SimpleFileSystem>()
        .map_err(|_| DiskError::NoFileSystem)?;

    let sfs = boot::open_protocol_exclusive::<SimpleFileSystem>(handle)
        .map_err(|_| DiskError::ProtocolError)?;

    let mut fs = FileSystem::new(sfs);

    let path16 = CString16::try_from(path).map_err(|_| DiskError::InvalidPath)?;

    fs.read(path16.as_ref()).map_err(|_| DiskError::ReadError)
}

/// Write data to a file on the EFI system partition.
pub fn write_file(path: &str, data: &[u8]) -> Result<(), DiskError> {
    let handle = boot::get_handle_for_protocol::<SimpleFileSystem>()
        .map_err(|_| DiskError::NoFileSystem)?;

    let sfs = boot::open_protocol_exclusive::<SimpleFileSystem>(handle)
        .map_err(|_| DiskError::ProtocolError)?;

    let mut fs = FileSystem::new(sfs);

    let path16 = CString16::try_from(path).map_err(|_| DiskError::InvalidPath)?;

    fs.write(path16.as_ref(), data)
        .map_err(|_| DiskError::WriteError)
}

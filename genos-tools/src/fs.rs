use alloc::string::String;
use alloc::vec::Vec;
use genos_hal::disk;

/// Filesystem tool — read, write, and list files on the EFI partition.
pub struct FsTool;

impl FsTool {
    /// Read a file and return its contents as a UTF-8 string.
    pub fn read(path: &str) -> Result<String, &'static str> {
        let data = disk::read_file(path).map_err(|_| "failed to read file")?;
        String::from_utf8(data).map_err(|_| "file is not valid UTF-8")
    }

    /// Read raw bytes from a file.
    pub fn read_bytes(path: &str) -> Result<Vec<u8>, &'static str> {
        disk::read_file(path).map_err(|_| "failed to read file")
    }

    /// Write string content to a file.
    pub fn write(path: &str, content: &str) -> Result<(), &'static str> {
        disk::write_file(path, content.as_bytes()).map_err(|_| "failed to write file")
    }

    /// Write raw bytes to a file.
    pub fn write_bytes(path: &str, data: &[u8]) -> Result<(), &'static str> {
        disk::write_file(path, data).map_err(|_| "failed to write file")
    }
}

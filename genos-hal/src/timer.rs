use alloc::format;
use alloc::string::String;
use core::time::Duration;
use uefi::boot;

/// Wall-clock timestamp from UEFI GetTime().
#[derive(Clone, Copy)]
pub struct WallTime {
    pub year: u16,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
}

impl WallTime {
    /// Format as ISO 8601: "2026-04-09T14:22:00Z"
    pub fn iso8601(&self) -> String {
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )
    }

    /// Format for session IDs: "20260409_142200"
    pub fn session_id_suffix(&self) -> String {
        format!(
            "{:04}{:02}{:02}_{:02}{:02}{:02}",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )
    }
}

/// Get the current wall-clock time from UEFI runtime services.
pub fn get_wall_time() -> Option<WallTime> {
    let time = uefi::runtime::get_time().ok()?;
    Some(WallTime {
        year: time.year(),
        month: time.month(),
        day: time.day(),
        hour: time.hour(),
        minute: time.minute(),
        second: time.second(),
    })
}

/// Sleep for the given number of milliseconds.
pub fn sleep_ms(ms: u64) {
    boot::stall(Duration::from_millis(ms));
}

/// Monotonic millisecond counter — cooperative, advanced after each turn.
static mut BOOT_STALL_COUNTER: u64 = 0;

/// Get the current cooperative clock in milliseconds.
pub fn now_ms() -> u64 {
    unsafe { BOOT_STALL_COUNTER }
}

/// Advance the cooperative clock by `elapsed_ms` milliseconds.
pub fn advance_ms(elapsed_ms: u64) {
    unsafe {
        BOOT_STALL_COUNTER += elapsed_ms;
    }
}

/// RAM info from UEFI memory map.
pub struct MemInfo {
    pub free_kb: u64,
    pub total_kb: u64,
}

/// Query UEFI memory map for free/total conventional RAM.
pub fn get_memory_info() -> MemInfo {
    use uefi::boot;
    use uefi::mem::memory_map::{MemoryMap, MemoryType};

    let mut free_kb: u64 = 0;
    let mut total_kb: u64 = 0;

    match boot::memory_map(MemoryType::LOADER_DATA) {
        Ok(map) => {
            for desc in map.entries() {
                let kb = desc.page_count * 4;
                total_kb += kb;
                match desc.ty {
                    MemoryType::CONVENTIONAL
                    | MemoryType::BOOT_SERVICES_CODE
                    | MemoryType::BOOT_SERVICES_DATA => {
                        free_kb += kb;
                    }
                    _ => {}
                }
            }
        }
        Err(_) => {}
    }

    MemInfo { free_kb, total_kb }
}

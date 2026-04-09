use uefi::system;
use uefi::CString16;

/// Print a string to the UEFI console output.
pub fn print(s: &str) {
    if let Ok(s16) = CString16::try_from(s) {
        system::with_stdout(|stdout| {
            let _ = stdout.output_string(&s16);
        });
    }
}

/// Print a string followed by a newline.
pub fn println(s: &str) {
    print(s);
    print("\r\n");
}

/// Clear the screen.
pub fn clear() {
    system::with_stdout(|stdout| {
        let _ = stdout.clear();
    });
}

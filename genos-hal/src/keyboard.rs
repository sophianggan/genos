use alloc::string::String;
use core::time::Duration;
use uefi::boot;
use uefi::proto::console::text::{Key, ScanCode};

/// Read a single key press (blocking).
pub fn read_key() -> Option<Key> {
    // Poll stdin until a key is available
    loop {
        let result = uefi::system::with_stdin(|stdin| stdin.read_key());
        match result {
            Ok(Some(key)) => return Some(key),
            Ok(None) => {
                // No key ready, stall briefly to avoid busy spin
                boot::stall(Duration::from_millis(10));
            }
            Err(_) => return None,
        }
    }
}

/// Read a line of text from the keyboard, echoing characters to screen.
/// Returns the line on Enter. Handles backspace.
pub fn read_line() -> String {
    let mut buf = String::new();

    loop {
        if let Some(key) = read_key() {
            match key {
                Key::Printable(c) => {
                    let ch: char = c.into();
                    if ch == '\r' || ch == '\n' {
                        // Enter pressed
                        crate::screen::print("\r\n");
                        return buf;
                    } else if ch == '\x08' {
                        // Backspace
                        if !buf.is_empty() {
                            buf.pop();
                            crate::screen::print("\x08 \x08");
                        }
                    } else {
                        buf.push(ch);
                        // Echo the character
                        let mut tmp = [0u8; 4];
                        let s = ch.encode_utf8(&mut tmp);
                        crate::screen::print(s);
                    }
                }
                Key::Special(scan) => {
                    match scan {
                        ScanCode::ESCAPE => {
                            // Clear current line
                            while !buf.is_empty() {
                                buf.pop();
                                crate::screen::print("\x08 \x08");
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

use alloc::string::String;
use alloc::vec::Vec;
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

/// Input history ring buffer for up-arrow recall.
pub struct InputHistory {
    entries: Vec<String>,
    max_entries: usize,
    cursor: usize,
}

impl InputHistory {
    pub fn new(max_entries: usize) -> Self {
        InputHistory {
            entries: Vec::new(),
            max_entries,
            cursor: 0,
        }
    }

    /// Record a new input line.
    pub fn push(&mut self, line: &str) {
        if line.is_empty() {
            return;
        }
        // Don't duplicate the last entry
        if let Some(last) = self.entries.last() {
            if last == line {
                self.cursor = self.entries.len();
                return;
            }
        }
        if self.entries.len() >= self.max_entries {
            self.entries.remove(0);
        }
        self.entries.push(String::from(line));
        self.cursor = self.entries.len();
    }

    /// Move cursor up (older). Returns the entry or None.
    fn prev(&mut self) -> Option<&str> {
        if self.entries.is_empty() || self.cursor == 0 {
            return None;
        }
        self.cursor -= 1;
        Some(&self.entries[self.cursor])
    }

    /// Move cursor down (newer). Returns the entry or None (at end = empty).
    fn next(&mut self) -> Option<&str> {
        if self.cursor >= self.entries.len() {
            return None;
        }
        self.cursor += 1;
        if self.cursor >= self.entries.len() {
            None // Past end = clear line
        } else {
            Some(&self.entries[self.cursor])
        }
    }

    /// Reset cursor to end (after new input).
    fn reset_cursor(&mut self) {
        self.cursor = self.entries.len();
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

/// Read a line with input history support (up/down arrows).
pub fn read_line_with_history(history: &mut InputHistory) -> String {
    let mut buf = String::new();

    loop {
        if let Some(key) = read_key() {
            match key {
                Key::Printable(c) => {
                    let ch: char = c.into();
                    if ch == '\r' || ch == '\n' {
                        crate::screen::print("\r\n");
                        history.push(&buf);
                        history.reset_cursor();
                        return buf;
                    } else if ch == '\x08' {
                        if !buf.is_empty() {
                            buf.pop();
                            crate::screen::print("\x08 \x08");
                        }
                    } else {
                        buf.push(ch);
                        let mut tmp = [0u8; 4];
                        let s = ch.encode_utf8(&mut tmp);
                        crate::screen::print(s);
                    }
                }
                Key::Special(scan) => {
                    match scan {
                        ScanCode::UP => {
                            if let Some(prev) = history.prev() {
                                let prev_owned = String::from(prev);
                                // Clear current line
                                while !buf.is_empty() {
                                    buf.pop();
                                    crate::screen::print("\x08 \x08");
                                }
                                // Display recalled line
                                crate::screen::print(&prev_owned);
                                buf = prev_owned;
                            }
                        }
                        ScanCode::DOWN => {
                            // Clear current line
                            while !buf.is_empty() {
                                buf.pop();
                                crate::screen::print("\x08 \x08");
                            }
                            if let Some(next) = history.next() {
                                let next_owned = String::from(next);
                                crate::screen::print(&next_owned);
                                buf = next_owned;
                            }
                            // If None, line stays empty (past end of history)
                        }
                        ScanCode::ESCAPE => {
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

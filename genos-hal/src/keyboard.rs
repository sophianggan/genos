// ── UEFI implementation ────────────────────────────────────────────
#[cfg(not(feature = "hosted"))]
mod inner {
    use core::time::Duration;
    use uefi::boot;
    use uefi::proto::console::text::{Key, ScanCode};
    use alloc::string::String;

    pub fn read_key() -> Option<Key> {
        loop {
            let result = uefi::system::with_stdin(|stdin| stdin.read_key());
            match result {
                Ok(Some(key)) => return Some(key),
                Ok(None) => { boot::stall(Duration::from_millis(10)); }
                Err(_) => return None,
            }
        }
    }

    pub fn read_line() -> String {
        let mut buf = String::new();
        loop {
            if let Some(key) = read_key() {
                match key {
                    Key::Printable(c) => {
                        let ch: char = c.into();
                        if ch == '\r' || ch == '\n' { crate::screen::print("\r\n"); return buf; }
                        else if ch == '\x08' { if !buf.is_empty() { buf.pop(); crate::screen::print("\x08 \x08"); } }
                        else { buf.push(ch); let mut tmp = [0u8; 4]; let s = ch.encode_utf8(&mut tmp); crate::screen::print(s); }
                    }
                    Key::Special(scan) => {
                        if scan == ScanCode::ESCAPE { while !buf.is_empty() { buf.pop(); crate::screen::print("\x08 \x08"); } }
                    }
                }
            }
        }
    }

    pub fn read_line_with_history(history: &mut super::InputHistory) -> String {
        let mut buf = String::new();
        loop {
            if let Some(key) = read_key() {
                match key {
                    Key::Printable(c) => {
                        let ch: char = c.into();
                        if ch == '\r' || ch == '\n' {
                            crate::screen::print("\r\n");
                            history.push(&buf); history.reset_cursor();
                            return buf;
                        } else if ch == '\x08' {
                            if !buf.is_empty() { buf.pop(); crate::screen::print("\x08 \x08"); }
                        } else {
                            buf.push(ch); let mut tmp = [0u8; 4]; let s = ch.encode_utf8(&mut tmp); crate::screen::print(s);
                        }
                    }
                    Key::Special(scan) => match scan {
                        ScanCode::UP => {
                            if let Some(prev) = history.prev() {
                                let prev_owned = String::from(prev);
                                while !buf.is_empty() { buf.pop(); crate::screen::print("\x08 \x08"); }
                                crate::screen::print(&prev_owned); buf = prev_owned;
                            }
                        }
                        ScanCode::DOWN => {
                            while !buf.is_empty() { buf.pop(); crate::screen::print("\x08 \x08"); }
                            if let Some(next) = history.next() {
                                let next_owned = String::from(next); crate::screen::print(&next_owned); buf = next_owned;
                            }
                        }
                        ScanCode::ESCAPE => { while !buf.is_empty() { buf.pop(); crate::screen::print("\x08 \x08"); } }
                        _ => {}
                    },
                }
            }
        }
    }
}

// ── Hosted (std) implementation ────────────────────────────────────
#[cfg(feature = "hosted")]
mod inner {
    use alloc::string::String;

    pub fn read_line() -> String {
        let mut buf = String::new();
        std::io::stdin().read_line(&mut buf).ok();
        if buf.ends_with('\n') { buf.pop(); }
        if buf.ends_with('\r') { buf.pop(); }
        buf
    }

    pub fn read_line_with_history(history: &mut super::InputHistory) -> String {
        let line = read_line();
        history.push(&line);
        history.reset_cursor();
        line
    }
}

// ── InputHistory (platform-independent) ────────────────────────────
use alloc::string::String;
use alloc::vec::Vec;

pub struct InputHistory {
    entries: Vec<String>,
    max_entries: usize,
    cursor: usize,
}

impl InputHistory {
    pub fn new(max_entries: usize) -> Self {
        InputHistory { entries: Vec::new(), max_entries, cursor: 0 }
    }

    pub fn push(&mut self, line: &str) {
        if line.is_empty() { return; }
        if let Some(last) = self.entries.last() { if last == line { self.cursor = self.entries.len(); return; } }
        if self.entries.len() >= self.max_entries { self.entries.remove(0); }
        self.entries.push(String::from(line));
        self.cursor = self.entries.len();
    }

    #[allow(dead_code)]
    fn prev(&mut self) -> Option<&str> {
        if self.entries.is_empty() || self.cursor == 0 { return None; }
        self.cursor -= 1; Some(&self.entries[self.cursor])
    }

    #[allow(dead_code)]
    fn next(&mut self) -> Option<&str> {
        if self.cursor >= self.entries.len() { return None; }
        self.cursor += 1;
        if self.cursor >= self.entries.len() { None } else { Some(&self.entries[self.cursor]) }
    }

    fn reset_cursor(&mut self) { self.cursor = self.entries.len(); }
}

// ── Public API ─────────────────────────────────────────────────────
pub fn read_line() -> String { inner::read_line() }
pub fn read_line_with_history(history: &mut InputHistory) -> String { inner::read_line_with_history(history) }

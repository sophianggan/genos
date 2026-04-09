use alloc::string::String;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use uefi::proto::console::text::Color;
use uefi::system;
use uefi::CString16;

/// Ring buffer for scrollback (last N lines).
const SCROLLBACK_SIZE: usize = 100;

struct Scrollback {
    lines: Vec<String>,
    head: usize,
    count: usize,
}

impl Scrollback {
    fn new() -> Self {
        let mut lines = Vec::with_capacity(SCROLLBACK_SIZE);
        for _ in 0..SCROLLBACK_SIZE {
            lines.push(String::new());
        }
        Scrollback { lines, head: 0, count: 0 }
    }

    fn push(&mut self, line: &str) {
        let idx = self.head % SCROLLBACK_SIZE;
        self.lines[idx].clear();
        self.lines[idx].push_str(line);
        self.head += 1;
        if self.count < SCROLLBACK_SIZE {
            self.count += 1;
        }
    }
}

// Global scrollback buffer - only safe in single-threaded UEFI
// Use UnsafeCell to avoid static_mut_refs warnings
struct ScrollbackCell(UnsafeCell<Option<Scrollback>>);
unsafe impl Sync for ScrollbackCell {}
static SCROLLBACK: ScrollbackCell = ScrollbackCell(UnsafeCell::new(None));

struct LineCell(UnsafeCell<Option<String>>);
unsafe impl Sync for LineCell {}
static CURRENT_LINE: LineCell = LineCell(UnsafeCell::new(None));

fn with_scrollback<F, R>(f: F) -> R
where
    F: FnOnce(&mut Scrollback) -> R,
{
    unsafe {
        let sb = &mut *SCROLLBACK.0.get();
        if sb.is_none() {
            *sb = Some(Scrollback::new());
        }
        f(sb.as_mut().unwrap())
    }
}

fn with_current_line<F, R>(f: F) -> R
where
    F: FnOnce(&mut String) -> R,
{
    unsafe {
        let line = &mut *CURRENT_LINE.0.get();
        if line.is_none() {
            *line = Some(String::new());
        }
        f(line.as_mut().unwrap())
    }
}

/// Print a string to the UEFI console output.
pub fn print(s: &str) {
    if let Ok(s16) = CString16::try_from(s) {
        system::with_stdout(|stdout| {
            let _ = stdout.output_string(&s16);
        });
    }
    // Track current line for scrollback
    with_current_line(|line| {
        for ch in s.chars() {
            if ch == '\n' {
                with_scrollback(|sb| sb.push(line));
                line.clear();
            } else if ch != '\r' {
                line.push(ch);
            }
        }
    });
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

/// Set text color attributes.
pub fn set_color(foreground: Color, background: Color) {
    system::with_stdout(|stdout| {
        let _ = stdout.set_color(foreground, background);
    });
}

/// Reset to default colors (light gray on black).
pub fn reset_color() {
    set_color(Color::LightGray, Color::Black);
}

/// Print with a specific color, then reset.
pub fn print_colored(s: &str, fg: Color) {
    set_color(fg, Color::Black);
    print(s);
    reset_color();
}

/// Print tool call rendering: [TOOL tool_name | key=val | N tokens]
pub fn print_tool_call(tool_name: &str, summary: &str) {
    set_color(Color::Cyan, Color::Black);
    print("[TOOL ");
    print(tool_name);
    print(" | ");
    print(summary);
    print("]");
    reset_color();
    print("\r\n");
}

/// Print tool result rendering.
pub fn print_tool_result(tool_name: &str, ok: bool, summary: &str) {
    if ok {
        set_color(Color::Green, Color::Black);
        print("[OK ");
    } else {
        set_color(Color::Red, Color::Black);
        print("[ERR ");
    }
    print(tool_name);
    print("] ");
    reset_color();
    println(summary);
}

/// Print a status bar on a single line (no newline).
pub fn print_status(status: &str) {
    set_color(Color::White, Color::Blue);
    print(status);
    reset_color();
}

/// TUI status bar — renders a persistent footer at the bottom of the screen.
/// Format: [model] | tok/s | RAM used/totalMB | turns: N | uptime: HH:MM:SS
pub fn render_status_bar(
    model: &str,
    tokens_per_sec: f64,
    ram_used_mb: usize,
    ram_total_mb: usize,
    turns: usize,
    uptime_secs: u64,
) {
    // Get screen dimensions
    let (cols, rows) = get_screen_size();
    if rows == 0 || cols == 0 {
        return;
    }

    // Save cursor position — move to bottom row
    // UEFI SimpleTextOutput: set cursor position
    system::with_stdout(|stdout| {
        let _ = stdout.set_cursor_position(0, rows - 1);
    });

    // Format uptime
    let hours = uptime_secs / 3600;
    let mins = (uptime_secs % 3600) / 60;
    let secs = uptime_secs % 60;

    // Build status line
    let tok_s_int = tokens_per_sec as u64;
    let status = alloc::format!(
        " [{}] | {} tok/s | RAM {}/{}MB | turns: {} | uptime: {:02}:{:02}:{:02} ",
        model, tok_s_int, ram_used_mb, ram_total_mb, turns, hours, mins, secs
    );

    // Pad to fill width
    let pad_len = if cols > status.len() { cols - status.len() } else { 0 };

    set_color(Color::White, Color::Blue);
    print(&status);
    for _ in 0..pad_len {
        print(" ");
    }
    reset_color();

    // Restore cursor to previous position (move up one from bottom)
    // We can't truly save/restore, so just move to bottom - 2
    system::with_stdout(|stdout| {
        if rows > 2 {
            let _ = stdout.set_cursor_position(0, rows - 2);
        }
    });
}

/// Get screen dimensions (columns, rows).
pub fn get_screen_size() -> (usize, usize) {
    system::with_stdout(|stdout| {
        let mode = stdout.current_mode();
        match mode {
            Ok(Some(m)) => (m.columns(), m.rows()),
            _ => (80, 25), // Default fallback
        }
    })
}

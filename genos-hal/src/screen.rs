// ── UEFI implementation ────────────────────────────────────────────
#[cfg(not(feature = "hosted"))]
mod inner {
    use alloc::string::String;
    use alloc::vec::Vec;
    use core::cell::UnsafeCell;
    use uefi::proto::console::text::Color;
    use uefi::system;
    use uefi::CString16;

    const SCROLLBACK_SIZE: usize = 100;

    struct Scrollback {
        lines: Vec<String>,
        head: usize,
        count: usize,
    }
    impl Scrollback {
        fn new() -> Self {
            let mut lines = Vec::with_capacity(SCROLLBACK_SIZE);
            for _ in 0..SCROLLBACK_SIZE { lines.push(String::new()); }
            Scrollback { lines, head: 0, count: 0 }
        }
        fn push(&mut self, line: &str) {
            let idx = self.head % SCROLLBACK_SIZE;
            self.lines[idx].clear();
            self.lines[idx].push_str(line);
            self.head += 1;
            if self.count < SCROLLBACK_SIZE { self.count += 1; }
        }
    }

    struct ScrollbackCell(UnsafeCell<Option<Scrollback>>);
    unsafe impl Sync for ScrollbackCell {}
    static SCROLLBACK: ScrollbackCell = ScrollbackCell(UnsafeCell::new(None));

    struct LineCell(UnsafeCell<Option<String>>);
    unsafe impl Sync for LineCell {}
    static CURRENT_LINE: LineCell = LineCell(UnsafeCell::new(None));

    fn with_scrollback<F, R>(f: F) -> R where F: FnOnce(&mut Scrollback) -> R {
        unsafe {
            let sb = &mut *SCROLLBACK.0.get();
            if sb.is_none() { *sb = Some(Scrollback::new()); }
            f(sb.as_mut().unwrap())
        }
    }
    fn with_current_line<F, R>(f: F) -> R where F: FnOnce(&mut String) -> R {
        unsafe {
            let line = &mut *CURRENT_LINE.0.get();
            if line.is_none() { *line = Some(String::new()); }
            f(line.as_mut().unwrap())
        }
    }

    pub fn print(s: &str) {
        if let Ok(s16) = CString16::try_from(s) {
            system::with_stdout(|stdout| { let _ = stdout.output_string(&s16); });
        }
        with_current_line(|line| {
            for ch in s.chars() {
                if ch == '\n' { with_scrollback(|sb| sb.push(line)); line.clear(); }
                else if ch != '\r' { line.push(ch); }
            }
        });
    }
    pub fn println(s: &str) { print(s); print("\r\n"); }
    pub fn clear() { system::with_stdout(|stdout| { let _ = stdout.clear(); }); }
    pub fn set_color(fg: Color, bg: Color) { system::with_stdout(|stdout| { let _ = stdout.set_color(fg, bg); }); }
    pub fn reset_color() { set_color(Color::LightGray, Color::Black); }
    #[allow(dead_code)]
    pub fn print_colored(s: &str, fg: Color) { set_color(fg, Color::Black); print(s); reset_color(); }

    pub fn print_tool_call(tool_name: &str, summary: &str) {
        set_color(Color::Cyan, Color::Black);
        print("[TOOL "); print(tool_name); print(" | "); print(summary); print("]");
        reset_color(); print("\r\n");
    }
    pub fn print_tool_result(tool_name: &str, ok: bool, summary: &str) {
        if ok { set_color(Color::Green, Color::Black); print("[OK "); }
        else  { set_color(Color::Red,   Color::Black); print("[ERR "); }
        print(tool_name); print("] "); reset_color(); println(summary);
    }
    pub fn print_status(status: &str) { set_color(Color::White, Color::Blue); print(status); reset_color(); }

    pub fn render_status_bar(
        model: &str, tokens_per_sec: f64, ram_used_mb: usize, ram_total_mb: usize,
        turns: usize, uptime_secs: u64,
    ) {
        let (cols, rows) = get_screen_size();
        if rows == 0 || cols == 0 { return; }
        system::with_stdout(|stdout| { let _ = stdout.set_cursor_position(0, rows - 1); });
        let hours = uptime_secs / 3600;
        let mins = (uptime_secs % 3600) / 60;
        let secs = uptime_secs % 60;
        let tok_s_int = tokens_per_sec as u64;
        let status = alloc::format!(
            " [{}] | {} tok/s | RAM {}/{}MB | turns: {} | uptime: {:02}:{:02}:{:02} ",
            model, tok_s_int, ram_used_mb, ram_total_mb, turns, hours, mins, secs
        );
        let pad_len = if cols > status.len() { cols - status.len() } else { 0 };
        set_color(Color::White, Color::Blue);
        print(&status);
        for _ in 0..pad_len { print(" "); }
        reset_color();
        system::with_stdout(|stdout| { if rows > 2 { let _ = stdout.set_cursor_position(0, rows - 2); } });
    }

    pub fn get_screen_size() -> (usize, usize) {
        system::with_stdout(|stdout| {
            match stdout.current_mode() {
                Ok(Some(m)) => (m.columns(), m.rows()),
                _ => (80, 25),
            }
        })
    }
}

// ── Hosted (std) implementation ────────────────────────────────────
#[cfg(feature = "hosted")]
mod inner {
    pub fn print(s: &str) { std::print!("{}", s); }
    pub fn println(s: &str) { std::println!("{}", s); }
    pub fn clear() {}
    pub fn print_tool_call(tool_name: &str, summary: &str) {
        std::println!("[TOOL {} | {}]", tool_name, summary);
    }
    pub fn print_tool_result(tool_name: &str, ok: bool, summary: &str) {
        if ok { std::println!("[OK {}] {}", tool_name, summary); }
        else  { std::println!("[ERR {}] {}", tool_name, summary); }
    }
    pub fn print_status(status: &str) { std::println!("[STATUS] {}", status); }
    pub fn render_status_bar(
        model: &str, _tokens_per_sec: f64, ram_used_mb: usize, ram_total_mb: usize,
        turns: usize, uptime_secs: u64,
    ) {
        std::println!(
            "[STATUS] [{}] | RAM {}/{}MB | turns: {} | uptime: {}s",
            model, ram_used_mb, ram_total_mb, turns, uptime_secs
        );
    }
    pub fn get_screen_size() -> (usize, usize) { (80, 25) }
}

// ── Public API ─────────────────────────────────────────────────────
pub fn print(s: &str) { inner::print(s); }
pub fn println(s: &str) { inner::println(s); }
pub fn clear() { inner::clear(); }
pub fn print_tool_call(tool_name: &str, summary: &str) { inner::print_tool_call(tool_name, summary); }
pub fn print_tool_result(tool_name: &str, ok: bool, summary: &str) { inner::print_tool_result(tool_name, ok, summary); }
pub fn print_status(status: &str) { inner::print_status(status); }
pub fn render_status_bar(
    model: &str, tokens_per_sec: f64, ram_used_mb: usize, ram_total_mb: usize,
    turns: usize, uptime_secs: u64,
) {
    inner::render_status_bar(model, tokens_per_sec, ram_used_mb, ram_total_mb, turns, uptime_secs);
}
pub fn get_screen_size() -> (usize, usize) { inner::get_screen_size() }

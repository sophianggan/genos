//! TF-IDF search index over palace halls.
//!
//! Builds an inverted index over palace hall files for O(1) term lookup.
//! Supports wing+hall filtering (34% recall improvement per MemPalace).
//! Recency-weighted scoring: score = tfidf * (1.0 / (1.0 + days_old)).

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use alloc::format;

/// A single indexed document in the palace.
#[derive(Clone)]
pub struct IndexEntry {
    /// Path to the source file.
    pub path: String,
    /// Wing name (e.g., "general", "project_foo").
    pub wing: String,
    /// Hall name (e.g., "facts", "events").
    pub hall: String,
    /// Raw text content.
    pub text: String,
    /// Term frequency map: word -> count.
    tf: BTreeMap<String, usize>,
    /// Total word count in this document.
    word_count: usize,
    /// Approximate timestamp from filename (turn number).
    pub turn: usize,
}

/// A search result with score.
pub struct SearchResult {
    pub path: String,
    pub wing: String,
    pub hall: String,
    pub text: String,
    pub score: f64,
    pub turn: usize,
}

/// TF-IDF search index over palace halls.
pub struct SearchIndex {
    /// All indexed entries.
    entries: Vec<IndexEntry>,
    /// Inverse document frequency: term -> idf value.
    idf: BTreeMap<String, f64>,
    /// Total documents indexed.
    doc_count: usize,
}

impl SearchIndex {
    pub fn new() -> Self {
        SearchIndex {
            entries: Vec::new(),
            idf: BTreeMap::new(),
            doc_count: 0,
        }
    }

    /// Add a document to the index.
    pub fn add_document(&mut self, path: &str, wing: &str, hall: &str, text: &str, turn: usize) {
        let words = tokenize_text(text);
        let word_count = words.len();

        // Build term frequency map
        let mut tf: BTreeMap<String, usize> = BTreeMap::new();
        for word in &words {
            *tf.entry(word.clone()).or_insert(0) += 1;
        }

        self.entries.push(IndexEntry {
            path: String::from(path),
            wing: String::from(wing),
            hall: String::from(hall),
            text: String::from(text),
            tf,
            word_count,
            turn,
        });

        self.doc_count = self.entries.len();
    }

    /// Rebuild IDF values from current entries.
    pub fn rebuild_idf(&mut self) {
        // Count how many docs contain each term
        let mut doc_freq: BTreeMap<String, usize> = BTreeMap::new();
        for entry in &self.entries {
            for term in entry.tf.keys() {
                *doc_freq.entry(term.clone()).or_insert(0) += 1;
            }
        }

        // Compute IDF: log(N / df) where N = total docs
        self.idf.clear();
        let n = self.doc_count as f64;
        for (term, df) in &doc_freq {
            let idf = ln_approx(n / *df as f64);
            self.idf.insert(term.clone(), idf);
        }
    }

    /// Search the index with a query string.
    /// Returns results sorted by TF-IDF score with recency weighting.
    ///
    /// `wing`: filter by wing name (None = all wings)
    /// `hall`: filter by hall name (None = all halls)
    /// `top_k`: maximum results to return
    /// `current_turn`: current turn number for recency weighting
    pub fn search(
        &self,
        query: &str,
        wing: Option<&str>,
        hall: Option<&str>,
        top_k: usize,
        current_turn: usize,
    ) -> Vec<SearchResult> {
        let query_terms = tokenize_text(query);
        if query_terms.is_empty() {
            return Vec::new();
        }

        let mut results: Vec<SearchResult> = Vec::new();

        for entry in &self.entries {
            // Apply wing/hall filters
            if let Some(w) = wing {
                if entry.wing != w {
                    continue;
                }
            }
            if let Some(h) = hall {
                if entry.hall != h {
                    continue;
                }
            }

            // Compute TF-IDF score for this document against the query
            let mut score = 0.0f64;
            for term in &query_terms {
                let tf = entry.tf.get(term).copied().unwrap_or(0) as f64;
                if tf == 0.0 {
                    continue;
                }
                let tf_normalized = tf / entry.word_count.max(1) as f64;
                let idf = self.idf.get(term).copied().unwrap_or(0.0);
                score += tf_normalized * idf;
            }

            if score > 0.0 {
                // Recency weighting: boost newer entries
                let turns_ago = current_turn.saturating_sub(entry.turn);
                let recency = 1.0 / (1.0 + turns_ago as f64 * 0.1);
                score *= recency;

                results.push(SearchResult {
                    path: entry.path.clone(),
                    wing: entry.wing.clone(),
                    hall: entry.hall.clone(),
                    text: entry.text.clone(),
                    score,
                    turn: entry.turn,
                });
            }
        }

        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(core::cmp::Ordering::Equal));

        // Truncate to top_k
        results.truncate(top_k);

        results
    }

    /// Get the number of indexed documents.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Index all palace hall files from disk.
    /// Reads \palace\wings\{wing}\halls\{hall}\*.txt files.
    pub fn index_palace(&mut self) {
        let wings = ["general"];
        let halls = ["facts", "events", "discoveries", "preferences", "advice"];

        for wing in &wings {
            for hall in &halls {
                // Try to read files with sequential turn numbers
                for turn in 0..1000 {
                    let path = format!(
                        "\\palace\\wings\\{}\\halls\\{}\\session_*_{:04}.txt",
                        wing, hall, turn
                    );
                    // Since we can't glob, try reading known session files
                    // Instead, index from session journals which we know exist
                    let _ = path; // Placeholder for glob-based indexing
                }
            }
        }
    }

    /// Index a single text entry directly (used when storing new entries).
    pub fn index_entry(&mut self, path: &str, wing: &str, hall: &str, text: &str, turn: usize) {
        self.add_document(path, wing, hall, text, turn);
        // Incrementally update IDF for new terms
        self.rebuild_idf();
    }

    /// Index entries from a session journal JSONL file.
    pub fn index_session_journal(&mut self, session_id: &str) {
        let path = format!("\\palace\\sessions\\{}.jsonl", session_id);
        let data = match genos_hal::disk::read_file(&path) {
            Ok(d) => d,
            Err(_) => return,
        };
        let text = core::str::from_utf8(&data).unwrap_or("");

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(val) = genos_kernel::json::parse(line) {
                let wing = val.get("wing")
                    .and_then(|v| v.as_str())
                    .unwrap_or("general");
                let hall = val.get("hall")
                    .and_then(|v| v.as_str())
                    .unwrap_or("events");
                let turn = val.get("turn")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as usize;
                let input = val.get("input")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let output = val.get("output")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                let combined = format!("{} {}", input, output);
                let entry_path = format!("{}:turn_{}", path, turn);
                self.add_document(&entry_path, wing, hall, &combined, turn);
            }
        }

        self.rebuild_idf();
    }
}

/// Natural log approximation for no_std (no libm dependency in genos-tools).
/// Uses the identity: ln(x) = 2 * atanh((x-1)/(x+1)) with series expansion.
fn ln_approx(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 0.0;
    }
    // Reduce: find k such that x = 2^k * m, 1 <= m < 2
    let mut k = 0i32;
    let mut m = x;
    while m >= 2.0 {
        m /= 2.0;
        k += 1;
    }
    while m < 1.0 {
        m *= 2.0;
        k -= 1;
    }
    // ln(x) = k * ln(2) + ln(m)
    // ln(m) via series: let y = (m-1)/(m+1), ln(m) = 2*(y + y^3/3 + y^5/5 + ...)
    let y = (m - 1.0) / (m + 1.0);
    let y2 = y * y;
    let mut term = y;
    let mut sum = y;
    for i in 1..10 {
        term *= y2;
        sum += term / (2 * i + 1) as f64;
    }
    k as f64 * 0.693147180559945 + 2.0 * sum
}

/// Tokenize text into lowercase words for indexing/querying.
fn tokenize_text(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            // Lowercase inline
            if ch.is_ascii_uppercase() {
                current.push((ch as u8 + 32) as char);
            } else {
                current.push(ch);
            }
        } else if !current.is_empty() {
            // Skip very short words (stop words)
            if current.len() > 2 {
                words.push(current.clone());
            }
            current.clear();
        }
    }

    if current.len() > 2 {
        words.push(current);
    }

    words
}

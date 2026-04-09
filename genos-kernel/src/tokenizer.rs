use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use alloc::format;

/// BPE tokenizer supporting both llama2.c binary format and Hugging Face
/// vocab.json + merges.txt format (needed for Gemma 4's 262k vocab in Phase D).
///
/// llama2.c format (Phase A/B):
///   max_token_length: i32
///   For each token (vocab_size total):
///     score: f32, len: i32, bytes: [u8; len]
///
/// Hugging Face format (Phase C+):
///   vocab.json: JSON object mapping token_string -> token_id
///   merges.txt: one merge per line "token_a token_b"
pub struct Tokenizer {
    vocab: Vec<String>,
    scores: Vec<f32>,
    /// Reverse lookup: token string -> token id (for large vocabs).
    vocab_index: BTreeMap<String, usize>,
    #[allow(dead_code)]
    max_token_length: usize,
    #[allow(dead_code)]
    vocab_size: usize,
    /// BPE merge rules in priority order (from merges.txt).
    /// Each entry is (token_a, token_b) -> merged rank (lower = higher priority).
    #[allow(dead_code)]
    merges: Vec<(String, String)>,
    /// If true, merges take priority over scores for BPE.
    use_merges: bool,
}

impl Tokenizer {
    /// Load tokenizer from the llama2.c tokenizer.bin format.
    pub fn load(data: &[u8], vocab_size: usize) -> Self {
        let mut offset = 0;

        // Need at least 4 bytes for the header
        if data.len() < 4 {
            return Tokenizer {
                vocab: Vec::new(),
                scores: Vec::new(),
                vocab_index: BTreeMap::new(),
                max_token_length: 0,
                vocab_size,
                merges: Vec::new(),
                use_merges: false,
            };
        }

        let max_token_length = read_i32(data, &mut offset) as usize;

        let mut vocab = Vec::with_capacity(vocab_size);
        let mut scores = Vec::with_capacity(vocab_size);
        let mut vocab_index = BTreeMap::new();

        for i in 0..vocab_size {
            // Each entry: f32 score + i32 len + len bytes
            if offset + 8 > data.len() { break; }
            let score = read_f32(data, &mut offset);
            let len = read_i32(data, &mut offset) as usize;
            if offset + len > data.len() { break; }
            let bytes = &data[offset..offset + len];
            offset += len;

            let s = String::from_utf8_lossy(bytes).into_owned();
            vocab_index.insert(s.clone(), i);
            vocab.push(s);
            scores.push(score);
        }

        Tokenizer {
            vocab,
            scores,
            vocab_index,
            max_token_length,
            vocab_size,
            merges: Vec::new(),
            use_merges: false,
        }
    }

    /// Load tokenizer from Hugging Face vocab.json + merges.txt format.
    ///
    /// vocab.json: `{"token": id, ...}` — maps token strings to IDs.
    /// merges.txt: one merge per line `"token_a token_b"`, first line may be `#version:`.
    pub fn load_hf(vocab_json: &[u8], merges_txt: &[u8]) -> Self {
        let vocab_str = core::str::from_utf8(vocab_json).unwrap_or("{}");
        let merges_str = core::str::from_utf8(merges_txt).unwrap_or("");

        // Parse vocab.json using our JSON parser
        let vocab_val = match genos_kernel_json_parse(vocab_str) {
            Some(v) => v,
            None => {
                return Self {
                    vocab: Vec::new(),
                    scores: Vec::new(),
                    vocab_index: BTreeMap::new(),
                    max_token_length: 0,
                    vocab_size: 0,
                    merges: Vec::new(),
                    use_merges: false,
                };
            }
        };

        // Build vocab from JSON object {token: id}
        let mut max_id: usize = 0;
        let mut entries: Vec<(String, usize)> = Vec::new();

        if let Some(obj) = vocab_val.as_object() {
            for (token, id_val) in obj {
                if let Some(id) = id_val.as_f64() {
                    let id = id as usize;
                    entries.push((token.clone(), id));
                    if id > max_id {
                        max_id = id;
                    }
                }
            }
        }

        let vocab_size = max_id + 1;
        let mut vocab = Vec::with_capacity(vocab_size);
        let mut scores = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            vocab.push(String::new());
            scores.push(0.0f32);
        }

        let mut vocab_index = BTreeMap::new();
        for (token, id) in &entries {
            if *id < vocab_size {
                vocab[*id] = token.clone();
                vocab_index.insert(token.clone(), *id);
            }
        }

        // Parse merges.txt
        let mut merges = Vec::new();
        for line in merges_str.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(space_pos) = line.find(' ') {
                let a = String::from(&line[..space_pos]);
                let b = String::from(&line[space_pos + 1..]);
                merges.push((a, b));
            }
        }

        // Assign scores: merges listed first have higher priority = higher score.
        // Also assign scores so the BPE merge loop can use them.
        // The merged token gets a score based on its merge rank.
        for (rank, (a, b)) in merges.iter().enumerate() {
            let merged = format!("{}{}", a, b);
            if let Some(&id) = vocab_index.get(&merged) {
                // Higher score for earlier merges (higher priority).
                // Use negative rank so that lower rank = higher score.
                scores[id] = -(rank as f32);
            }
        }

        let max_token_length = vocab.iter().map(|s| s.len()).max().unwrap_or(0);

        Tokenizer {
            vocab,
            scores,
            vocab_index,
            max_token_length,
            vocab_size,
            merges,
            use_merges: true,
        }
    }

    /// Encode a string into a sequence of token IDs using BPE.
    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<u32> {
        // Use usize internally for vocab indexing; convert to u32 at the boundary.
        let mut tokens: Vec<usize> = Vec::new();

        if bos {
            tokens.push(1); // BOS token
        }

        // If text is not empty, try to add the dummy prefix space token
        if !text.is_empty() {
            if let Some(id) = self.lookup(" ") {
                tokens.push(id);
            }
        }

        // Encode each UTF-8 byte / character as an initial token
        for ch in text.chars() {
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);

            if let Some(id) = self.lookup(s) {
                tokens.push(id);
            } else {
                // Fallback: encode each byte individually using byte fallback tokens <0xNN>
                for byte in s.as_bytes() {
                    let byte_token = alloc::format!("<0x{:02X}>", byte);
                    if let Some(id) = self.lookup(&byte_token) {
                        tokens.push(id);
                    }
                }
            }
        }

        // BPE merge loop: greedily merge the pair with the highest score
        loop {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_id = 0usize;
            let mut best_idx = usize::MAX;

            for i in 0..tokens.len().saturating_sub(1) {
                // Concatenate the string forms of tokens[i] and tokens[i+1]
                let merged = alloc::format!("{}{}", self.vocab[tokens[i]], self.vocab[tokens[i + 1]]);

                if let Some(id) = self.lookup(&merged) {
                    if self.scores[id] > best_score {
                        best_score = self.scores[id];
                        best_id = id;
                        best_idx = i;
                    }
                }
            }

            if best_idx == usize::MAX {
                break; // No more merges possible
            }

            // Replace the pair with the merged token
            tokens[best_idx] = best_id;
            tokens.remove(best_idx + 1);
        }

        if eos {
            tokens.push(2); // EOS token
        }

        tokens.into_iter().map(|t| t as u32).collect()
    }

    /// Decode a single token to its string representation.
    /// Handles byte fallback tokens like <0xNN>.
    /// Strips leading space if previous token was BOS (token 1).
    pub fn decode(&self, prev_token: u32, token: u32) -> &str {
        let piece = &self.vocab[token as usize];

        // Handle byte fallback tokens
        if piece.starts_with("<0x") && piece.ends_with('>') && piece.len() == 6 {
            // This is a raw byte token — for now just return it as-is
            // A proper implementation would decode the hex byte
            return piece;
        }

        // Strip leading space after BOS
        if prev_token == 1 && piece.starts_with(' ') {
            return &piece[1..];
        }

        piece
    }

    /// Look up a string in the vocabulary. Returns its token ID if found.
    fn lookup(&self, s: &str) -> Option<usize> {
        // BTreeMap lookup for O(log n) lookups — scales to Gemma 4's 262k vocab.
        self.vocab_index.get(s).copied()
    }

    /// Get the vocabulary size.
    pub fn vocab_len(&self) -> usize {
        self.vocab_size
    }

    /// Check if this tokenizer uses Hugging Face merges format.
    pub fn is_hf(&self) -> bool {
        self.use_merges
    }
}

fn read_i32(data: &[u8], offset: &mut usize) -> i32 {
    let val = i32::from_le_bytes([
        data[*offset],
        data[*offset + 1],
        data[*offset + 2],
        data[*offset + 3],
    ]);
    *offset += 4;
    val
}

fn read_f32(data: &[u8], offset: &mut usize) -> f32 {
    let val = f32::from_le_bytes([
        data[*offset],
        data[*offset + 1],
        data[*offset + 2],
        data[*offset + 3],
    ]);
    *offset += 4;
    val
}

/// Minimal JSON object parser for vocab.json.
/// Uses the crate's json module to parse vocabs.
fn genos_kernel_json_parse(input: &str) -> Option<crate::json::JsonValue> {
    crate::json::parse(input).ok()
}

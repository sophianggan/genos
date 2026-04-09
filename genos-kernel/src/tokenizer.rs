use alloc::string::String;
use alloc::vec::Vec;

/// BPE tokenizer compatible with the llama2.c tokenizer.bin format.
///
/// File format:
///   max_token_length: i32
///   For each token (vocab_size total):
///     score: f32
///     len: i32
///     bytes: [u8; len]
pub struct Tokenizer {
    vocab: Vec<String>,
    scores: Vec<f32>,
    #[allow(dead_code)]
    max_token_length: usize,
    #[allow(dead_code)]
    vocab_size: usize,
}

impl Tokenizer {
    /// Load tokenizer from the llama2.c tokenizer.bin format.
    pub fn load(data: &[u8], vocab_size: usize) -> Self {
        let mut offset = 0;

        // Need at least 4 bytes for the header
        if data.len() < 4 {
            return Tokenizer { vocab: Vec::new(), scores: Vec::new(), max_token_length: 0, vocab_size };
        }

        let max_token_length = read_i32(data, &mut offset) as usize;

        let mut vocab = Vec::with_capacity(vocab_size);
        let mut scores = Vec::with_capacity(vocab_size);

        for _ in 0..vocab_size {
            // Each entry: f32 score + i32 len + len bytes
            if offset + 8 > data.len() { break; }
            let score = read_f32(data, &mut offset);
            let len = read_i32(data, &mut offset) as usize;
            if offset + len > data.len() { break; }
            let bytes = &data[offset..offset + len];
            offset += len;

            let s = String::from_utf8_lossy(bytes).into_owned();
            vocab.push(s);
            scores.push(score);
        }

        Tokenizer {
            vocab,
            scores,
            max_token_length,
            vocab_size,
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
        // Linear scan — for Stories15M (32k vocab) this is fast enough.
        // For larger models, use a sorted index.
        self.vocab.iter().position(|v| v == s)
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

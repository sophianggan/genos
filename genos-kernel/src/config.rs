/// Model configuration parsed from the llama2.c binary format header.
/// The header is 7 consecutive i32 values (28 bytes).
///
/// Phase A/B/C: Stories15M (llama2.c format, 15M params, 32k vocab)
/// Phase D+:    Gemma 4 E2B (GGUF format, 2.3B effective params / 5.1B total,
///              262k vocab, 35 layers, hybrid sliding-window + global attention,
///              128k context, Per-Layer Embeddings, ~3.2 GB at int4 quantization)
#[derive(Debug, Clone, Copy)]
pub struct ModelConfig {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    /// True if classifier weights are shared with token embeddings.
    pub shared_weights: bool,
}

impl ModelConfig {
    /// Parse config from the first 28 bytes of a model file (llama2.c format).
    /// Returns None if the buffer is too short.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 28 {
            return None;
        }

        let read_i32 = |offset: usize| -> i32 {
            i32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ])
        };

        let dim = read_i32(0) as usize;
        let hidden_dim = read_i32(4) as usize;
        let n_layers = read_i32(8) as usize;
        let n_heads = read_i32(12) as usize;
        let n_kv_heads = read_i32(16) as usize;
        let raw_vocab_size = read_i32(20);
        let seq_len = read_i32(24) as usize;

        // Negative vocab_size indicates shared classifier weights
        let shared_weights = raw_vocab_size > 0;
        let vocab_size = raw_vocab_size.unsigned_abs() as usize;

        if dim == 0 || n_layers == 0 || n_heads == 0 || vocab_size == 0 || seq_len == 0 {
            return None;
        }

        Some(ModelConfig {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            seq_len,
            shared_weights,
        })
    }

    /// Head size (dim / n_heads).
    pub fn head_size(&self) -> usize {
        self.dim / self.n_heads
    }

    /// KV dimension (head_size * n_kv_heads).
    pub fn kv_dim(&self) -> usize {
        self.head_size() * self.n_kv_heads
    }

    /// Number of key-value head groups.
    pub fn kv_mul(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stories15m_config() {
        // Stories15M config: dim=288, hidden_dim=768, n_layers=6, n_heads=6,
        // n_kv_heads=6, vocab_size=32000 (shared), seq_len=256
        let mut data = [0u8; 28];
        let vals: [i32; 7] = [288, 768, 6, 6, 6, 32000, 256];
        for (i, &v) in vals.iter().enumerate() {
            let bytes = v.to_le_bytes();
            data[i * 4..i * 4 + 4].copy_from_slice(&bytes);
        }

        let config = ModelConfig::from_bytes(&data).unwrap();
        assert_eq!(config.dim, 288);
        assert_eq!(config.hidden_dim, 768);
        assert_eq!(config.n_layers, 6);
        assert_eq!(config.n_heads, 6);
        assert_eq!(config.n_kv_heads, 6);
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.seq_len, 256);
        assert!(config.shared_weights);
        assert_eq!(config.head_size(), 48);
        assert_eq!(config.kv_dim(), 288);
        assert_eq!(config.kv_mul(), 1);
    }
}

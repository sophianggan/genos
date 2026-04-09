use crate::config::ModelConfig;
use alloc::vec::Vec;

/// Pre-computed offsets into the flat weight array for each weight tensor.
/// All weights are stored as a single contiguous f32 array in llama2.c format.
pub struct WeightOffsets {
    pub token_embedding: usize,
    pub rms_att_weight: usize,
    pub wq: usize,
    pub wk: usize,
    pub wv: usize,
    pub wo: usize,
    pub rms_ffn_weight: usize,
    pub w1: usize,
    pub w2: usize,
    pub w3: usize,
    pub rms_final_weight: usize,
    pub freq_cis_real: usize,
    pub freq_cis_imag: usize,
    pub wcls: usize,
}

impl WeightOffsets {
    /// Compute offsets for the llama2.c weight layout.
    pub fn new(config: &ModelConfig) -> Self {
        let head_size = config.head_size();
        let n_layers = config.n_layers;
        let dim = config.dim;
        let kv_dim = config.kv_dim();
        let hidden_dim = config.hidden_dim;
        let vocab_size = config.vocab_size;
        let seq_len = config.seq_len;

        let mut offset = 0usize;

        let token_embedding = offset;
        offset += vocab_size * dim;

        let rms_att_weight = offset;
        offset += n_layers * dim;

        let wq = offset;
        offset += n_layers * dim * (config.n_heads * head_size);

        let wk = offset;
        offset += n_layers * dim * kv_dim;

        let wv = offset;
        offset += n_layers * dim * kv_dim;

        let wo = offset;
        offset += n_layers * (config.n_heads * head_size) * dim;

        let rms_ffn_weight = offset;
        offset += n_layers * dim;

        let w1 = offset;
        offset += n_layers * dim * hidden_dim;

        let w2 = offset;
        offset += n_layers * hidden_dim * dim;

        let w3 = offset;
        offset += n_layers * dim * hidden_dim;

        let rms_final_weight = offset;
        offset += dim;

        let freq_cis_real = offset;
        offset += seq_len * head_size / 2;

        let freq_cis_imag = offset;
        offset += seq_len * head_size / 2;

        let wcls = if config.shared_weights {
            token_embedding
        } else {
            offset
        };

        WeightOffsets {
            token_embedding,
            rms_att_weight,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_weight,
            w1,
            w2,
            w3,
            rms_final_weight,
            freq_cis_real,
            freq_cis_imag,
            wcls,
        }
    }
}

/// Provides access to model weights stored as a flat f32 array.
pub struct Weights {
    data: Vec<f32>,
    pub offsets: WeightOffsets,
}

impl Weights {
    /// Create weights from raw bytes (everything after the 28-byte header).
    /// Interprets bytes as little-endian f32 values.
    pub fn from_bytes(raw: &[u8], config: &ModelConfig) -> Self {
        let data: Vec<f32> = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        let offsets = WeightOffsets::new(config);

        Weights { data, offsets }
    }

    /// Get a slice of the weight data.
    #[inline]
    pub fn slice(&self, offset: usize, len: usize) -> &[f32] {
        &self.data[offset..offset + len]
    }

    /// Token embedding for a given token index.
    #[inline]
    pub fn token_embedding(&self, token: usize, dim: usize) -> &[f32] {
        self.slice(self.offsets.token_embedding + token * dim, dim)
    }

    /// RMSNorm attention weights for a given layer.
    #[inline]
    pub fn rms_att_weight(&self, layer: usize, dim: usize) -> &[f32] {
        self.slice(self.offsets.rms_att_weight + layer * dim, dim)
    }

    /// Query weight matrix for a given layer (dim x dim).
    #[inline]
    pub fn wq(&self, layer: usize, dim: usize) -> &[f32] {
        self.slice(self.offsets.wq + layer * dim * dim, dim * dim)
    }

    /// Key weight matrix for a given layer (dim x kv_dim).
    #[inline]
    pub fn wk(&self, layer: usize, dim: usize, kv_dim: usize) -> &[f32] {
        self.slice(self.offsets.wk + layer * dim * kv_dim, dim * kv_dim)
    }

    /// Value weight matrix for a given layer (dim x kv_dim).
    #[inline]
    pub fn wv(&self, layer: usize, dim: usize, kv_dim: usize) -> &[f32] {
        self.slice(self.offsets.wv + layer * dim * kv_dim, dim * kv_dim)
    }

    /// Output projection weight matrix for a given layer (dim x dim).
    #[inline]
    pub fn wo(&self, layer: usize, dim: usize) -> &[f32] {
        self.slice(self.offsets.wo + layer * dim * dim, dim * dim)
    }

    /// RMSNorm FFN weights for a given layer.
    #[inline]
    pub fn rms_ffn_weight(&self, layer: usize, dim: usize) -> &[f32] {
        self.slice(self.offsets.rms_ffn_weight + layer * dim, dim)
    }

    /// FFN gate weight matrix (w1) for a given layer (dim x hidden_dim).
    #[inline]
    pub fn w1(&self, layer: usize, dim: usize, hidden_dim: usize) -> &[f32] {
        self.slice(self.offsets.w1 + layer * dim * hidden_dim, dim * hidden_dim)
    }

    /// FFN down projection (w2) for a given layer (hidden_dim x dim).
    #[inline]
    pub fn w2(&self, layer: usize, dim: usize, hidden_dim: usize) -> &[f32] {
        self.slice(self.offsets.w2 + layer * hidden_dim * dim, hidden_dim * dim)
    }

    /// FFN up projection (w3) for a given layer (dim x hidden_dim).
    #[inline]
    pub fn w3(&self, layer: usize, dim: usize, hidden_dim: usize) -> &[f32] {
        self.slice(self.offsets.w3 + layer * dim * hidden_dim, dim * hidden_dim)
    }

    /// Final RMSNorm weight.
    #[inline]
    pub fn rms_final_weight(&self, dim: usize) -> &[f32] {
        self.slice(self.offsets.rms_final_weight, dim)
    }

    /// Classifier weights (vocab_size x dim). May alias token_embedding.
    #[inline]
    pub fn wcls(&self, vocab_size: usize, dim: usize) -> &[f32] {
        self.slice(self.offsets.wcls, vocab_size * dim)
    }
}

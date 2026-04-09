use alloc::vec;
use alloc::vec::Vec;
use libm::{cosf, expf, sinf, sqrtf};

use crate::config::ModelConfig;
use crate::weights::Weights;

/// Runtime state buffers for transformer inference.
struct RunState {
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    q: Vec<f32>,
    att: Vec<f32>,
    logits: Vec<f32>,
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
}

impl RunState {
    fn new(config: &ModelConfig) -> Self {
        let kv_dim = config.kv_dim();
        RunState {
            x: vec![0.0; config.dim],
            xb: vec![0.0; config.dim],
            xb2: vec![0.0; config.dim],
            hb: vec![0.0; config.hidden_dim],
            hb2: vec![0.0; config.hidden_dim],
            q: vec![0.0; config.dim],
            att: vec![0.0; config.n_heads * config.seq_len],
            logits: vec![0.0; config.vocab_size],
            key_cache: vec![0.0; config.n_layers * config.seq_len * kv_dim],
            value_cache: vec![0.0; config.n_layers * config.seq_len * kv_dim],
        }
    }

    fn reset(&mut self) {
        self.key_cache.fill(0.0);
        self.value_cache.fill(0.0);
    }
}

/// The transformer model: config + weights + runtime state.
pub struct Transformer {
    pub config: ModelConfig,
    weights: Weights,
    state: RunState,
}

impl Transformer {
    /// Create a new transformer from a parsed config and raw weight bytes.
    /// `weight_bytes` should be everything after the 28-byte config header.
    pub fn new(config: ModelConfig, weight_bytes: &[u8]) -> Self {
        let weights = Weights::from_bytes(weight_bytes, &config);
        let state = RunState::new(&config);
        Transformer {
            config,
            weights,
            state,
        }
    }

    /// Reset the KV cache (call between independent sequences).
    pub fn reset(&mut self) {
        self.state.reset();
    }

    /// Run one forward pass for a single token at the given position.
    /// Returns a reference to the logits (vocab_size floats).
    pub fn forward(&mut self, token: usize, pos: usize) -> &[f32] {
        let cfg = self.config;
        let dim = cfg.dim;
        let kv_dim = cfg.kv_dim();
        let kv_mul = cfg.kv_mul();
        let head_size = cfg.head_size();
        let hidden_dim = cfg.hidden_dim;

        // Copy token embedding into x
        self.state.x.copy_from_slice(self.weights.token_embedding(token, dim));

        for l in 0..cfg.n_layers {
            // --- Attention ---

            // RMSNorm on x -> xb
            rmsnorm(&mut self.state.xb, &self.state.x, self.weights.rms_att_weight(l, dim));

            // QKV projections
            // q = Wq * xb
            matmul(&mut self.state.q, &self.state.xb, self.weights.wq(l, dim), dim, dim);

            // We need temporary k and v buffers; use slices of the KV cache directly
            let loff = l * cfg.seq_len * kv_dim;
            let kv_start = loff + pos * kv_dim;

            // k = Wk * xb -> directly into key_cache
            {
                let wk = self.weights.wk(l, dim, kv_dim);
                for i in 0..kv_dim {
                    let mut val = 0.0f32;
                    let row = &wk[i * dim..(i + 1) * dim];
                    for j in 0..dim {
                        val += row[j] * self.state.xb[j];
                    }
                    self.state.key_cache[kv_start + i] = val;
                }
            }

            // v = Wv * xb -> directly into value_cache
            {
                let wv = self.weights.wv(l, dim, kv_dim);
                for i in 0..kv_dim {
                    let mut val = 0.0f32;
                    let row = &wv[i * dim..(i + 1) * dim];
                    for j in 0..dim {
                        val += row[j] * self.state.xb[j];
                    }
                    self.state.value_cache[kv_start + i] = val;
                }
            }

            // Apply RoPE to q and k
            for i in (0..dim).step_by(2) {
                let head_dim = i % head_size;
                let freq = 1.0 / powf_int(10000.0, head_dim as f32 / head_size as f32);
                let val = pos as f32 * freq;
                let fcr = cosf(val);
                let fci = sinf(val);

                // Rotate q
                let v0 = self.state.q[i];
                let v1 = self.state.q[i + 1];
                self.state.q[i] = v0 * fcr - v1 * fci;
                self.state.q[i + 1] = v0 * fci + v1 * fcr;

                // Rotate k (only for kv_dim elements)
                if i < kv_dim {
                    let v0 = self.state.key_cache[kv_start + i];
                    let v1 = self.state.key_cache[kv_start + i + 1];
                    self.state.key_cache[kv_start + i] = v0 * fcr - v1 * fci;
                    self.state.key_cache[kv_start + i + 1] = v0 * fci + v1 * fcr;
                }
            }

            // Multi-head attention
            for h in 0..cfg.n_heads {
                let q_off = h * head_size;
                let att_off = h * cfg.seq_len;

                // Compute attention scores for all cached positions
                for t in 0..=pos {
                    let k_off = loff + t * kv_dim + (h / kv_mul) * head_size;
                    let mut score = 0.0f32;
                    for i in 0..head_size {
                        score += self.state.q[q_off + i] * self.state.key_cache[k_off + i];
                    }
                    score /= sqrtf(head_size as f32);
                    self.state.att[att_off + t] = score;
                }

                // Softmax over attention scores [0..=pos]
                softmax(&mut self.state.att[att_off..att_off + pos + 1]);

                // Weighted sum of values
                let xb_off = h * head_size;
                for i in 0..head_size {
                    self.state.xb[xb_off + i] = 0.0;
                }
                for t in 0..=pos {
                    let v_off = loff + t * kv_dim + (h / kv_mul) * head_size;
                    let a = self.state.att[att_off + t];
                    for i in 0..head_size {
                        self.state.xb[xb_off + i] += a * self.state.value_cache[v_off + i];
                    }
                }
            }

            // Output projection: xb2 = Wo * xb
            matmul(&mut self.state.xb2, &self.state.xb, self.weights.wo(l, dim), dim, dim);

            // Residual connection
            for i in 0..dim {
                self.state.x[i] += self.state.xb2[i];
            }

            // --- FFN ---

            // RMSNorm
            rmsnorm(&mut self.state.xb, &self.state.x, self.weights.rms_ffn_weight(l, dim));

            // w1 and w3 projections
            matmul(
                &mut self.state.hb,
                &self.state.xb,
                self.weights.w1(l, dim, hidden_dim),
                dim,
                hidden_dim,
            );
            matmul(
                &mut self.state.hb2,
                &self.state.xb,
                self.weights.w3(l, dim, hidden_dim),
                dim,
                hidden_dim,
            );

            // SiLU activation on hb, then element-wise multiply with hb2
            for i in 0..hidden_dim {
                let x = self.state.hb[i];
                // SiLU(x) = x * sigmoid(x)
                self.state.hb[i] = x * (1.0 / (1.0 + expf(-x)));
                self.state.hb[i] *= self.state.hb2[i];
            }

            // w2 down projection
            matmul(
                &mut self.state.xb,
                &self.state.hb,
                self.weights.w2(l, dim, hidden_dim),
                hidden_dim,
                dim,
            );

            // Residual connection
            for i in 0..dim {
                self.state.x[i] += self.state.xb[i];
            }
        }

        // Final RMSNorm (in-place on x, using xb as temp)
        {
            let w = self.weights.rms_final_weight(dim);
            let mut ss = 0.0f32;
            for i in 0..dim {
                ss += self.state.x[i] * self.state.x[i];
            }
            ss = 1.0 / sqrtf(ss / dim as f32 + 1e-5);
            for i in 0..dim {
                self.state.x[i] = w[i] * (ss * self.state.x[i]);
            }
        }

        // Classifier: logits = Wcls * x
        matmul(
            &mut self.state.logits,
            &self.state.x,
            self.weights.wcls(self.config.vocab_size, dim),
            dim,
            self.config.vocab_size,
        );

        &self.state.logits
    }
}

// --- Math utilities ---

/// RMSNorm: out[i] = weight[i] * (x[i] / rms(x))
fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32]) {
    let n = x.len();
    let mut ss = 0.0f32;
    for i in 0..n {
        ss += x[i] * x[i];
    }
    ss = 1.0 / sqrtf(ss / n as f32 + 1e-5);
    for i in 0..n {
        out[i] = weight[i] * (ss * x[i]);
    }
}

/// Matrix-vector multiply: out = W * x
/// W is (d, n) stored row-major: W[i*n + j]
/// x is (n,), out is (d,)
fn matmul(out: &mut [f32], x: &[f32], w: &[f32], n: usize, d: usize) {
    for i in 0..d {
        let mut val = 0.0f32;
        let row_start = i * n;
        for j in 0..n {
            val += w[row_start + j] * x[j];
        }
        out[i] = val;
    }
}

/// Softmax in-place over a mutable slice.
fn softmax(x: &mut [f32]) {
    let n = x.len();
    if n == 0 {
        return;
    }

    // Find max for numerical stability
    let mut max_val = x[0];
    for i in 1..n {
        if x[i] > max_val {
            max_val = x[i];
        }
    }

    // exp and sum
    let mut sum = 0.0f32;
    for i in 0..n {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    // Normalize
    let inv_sum = 1.0 / sum;
    for i in 0..n {
        x[i] *= inv_sum;
    }
}

/// Compute a^b using exp and ln, since we need fractional powers.
fn powf_int(a: f32, b: f32) -> f32 {
    expf(b * libm::logf(a))
}

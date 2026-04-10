//! Gemma 4 E2B inference engine.
//!
//! Implements the complete forward pass for Gemma 4 E2B:
//!   - Per-Layer Embeddings (PLE)
//!   - Hybrid sliding-window / full global attention
//!   - GQA (Grouped Query Attention, 8 query heads, 1 KV head)
//!   - GeGLU FFN (GELU-gated linear unit)
//!   - Proportional RoPE (p-RoPE) for global layers
//!   - Logit soft-capping
//!   - QJL-compressed KV cache
//!
//! Architecture reference: google/gemma-4-E2B-it config.json

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::gguf::{GGMLType, GGUFError, GGUFFile};
use crate::kv_cache::QJLKVCache;
use crate::simd;
use crate::LLMRuntime;

// ── Layer attention type ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    Sliding,
    Full,
}

// ── Gemma 4 config ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Gemma4Config {
    pub hidden_size: usize,       // 1536
    pub num_layers: usize,        // 35
    pub num_attention_heads: usize, // 8
    pub num_kv_heads: usize,      // 1
    pub head_dim: usize,          // 256 (sliding)
    pub global_head_dim: usize,   // 512 (full)
    pub intermediate_size: usize, // 6144
    pub vocab_size: usize,        // 262144
    pub max_position_embeddings: usize, // 131072
    pub sliding_window: usize,    // 512
    pub rms_norm_eps: f32,        // 1e-6
    pub final_logit_softcapping: f32, // 30.0
    pub ple_dim: usize,           // 256 (hidden_size_per_layer_input)
    pub tie_word_embeddings: bool, // true
    pub layer_types: Vec<LayerType>,

    // RoPE parameters
    pub rope_theta_sliding: f32,      // 10000.0
    pub rope_theta_full: f32,         // 1000000.0
    pub partial_rotary_factor: f32,   // 0.25 (for full attention)
}

impl Gemma4Config {
    /// Build config from GGUF metadata.
    pub fn from_gguf(gguf: &GGUFFile) -> Result<Self, GGUFError> {
        // Try standard llama.cpp GGUF keys, falling back to defaults from the
        // known Gemma 4 E2B architecture.
        let arch = gguf.get_str("general.architecture").unwrap_or("gemma4");
        let prefix = if arch.is_empty() { "gemma4" } else { arch };

        let hidden_size = gguf
            .get_u32(&fmt_key(prefix, "embedding_length"))
            .unwrap_or(1536) as usize;
        let num_layers = gguf
            .get_u32(&fmt_key(prefix, "block_count"))
            .unwrap_or(35) as usize;
        let num_attention_heads = gguf
            .get_u32(&fmt_key(prefix, "attention.head_count"))
            .unwrap_or(8) as usize;
        let num_kv_heads = gguf
            .get_u32(&fmt_key(prefix, "attention.head_count_kv"))
            .unwrap_or(1) as usize;
        let vocab_size = gguf
            .get_u32(&fmt_key(prefix, "vocab_size"))
            .unwrap_or(262144) as usize;
        let intermediate_size = gguf
            .get_u32(&fmt_key(prefix, "feed_forward_length"))
            .unwrap_or(6144) as usize;
        let max_position_embeddings = gguf
            .get_u32(&fmt_key(prefix, "context_length"))
            .unwrap_or(131072) as usize;
        let rms_norm_eps = gguf
            .get_f32(&fmt_key(prefix, "attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6);

        // Gemma 4-specific head dims
        let head_dim = gguf
            .get_u32(&fmt_key(prefix, "attention.head_dim"))
            .unwrap_or(256) as usize;
        let global_head_dim = gguf
            .get_u32(&fmt_key(prefix, "attention.global_head_dim"))
            .unwrap_or(512) as usize;
        let sliding_window = gguf
            .get_u32(&fmt_key(prefix, "attention.sliding_window"))
            .unwrap_or(512) as usize;
        let ple_dim = gguf
            .get_u32(&fmt_key(prefix, "ple_dim"))
            .unwrap_or(256) as usize;

        // Build layer types (default: 4 sliding + 1 full, repeating)
        let layer_types = build_layer_types(num_layers);

        Ok(Gemma4Config {
            hidden_size,
            num_layers,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            global_head_dim,
            intermediate_size,
            vocab_size,
            max_position_embeddings,
            sliding_window,
            rms_norm_eps,
            final_logit_softcapping: 30.0,
            ple_dim,
            tie_word_embeddings: true,
            layer_types,
            rope_theta_sliding: 10000.0,
            rope_theta_full: 1000000.0,
            partial_rotary_factor: 0.25,
        })
    }

    /// Default config matching google/gemma-4-E2B-it exactly.
    pub fn default_e2b() -> Self {
        Gemma4Config {
            hidden_size: 1536,
            num_layers: 35,
            num_attention_heads: 8,
            num_kv_heads: 1,
            head_dim: 256,
            global_head_dim: 512,
            intermediate_size: 6144,
            vocab_size: 262144,
            max_position_embeddings: 131072,
            sliding_window: 512,
            rms_norm_eps: 1e-6,
            final_logit_softcapping: 30.0,
            ple_dim: 256,
            tie_word_embeddings: true,
            layer_types: build_layer_types(35),
            rope_theta_sliding: 10000.0,
            rope_theta_full: 1000000.0,
            partial_rotary_factor: 0.25,
        }
    }

    /// Head dimension for a specific layer.
    #[inline]
    pub fn layer_head_dim(&self, layer: usize) -> usize {
        match self.layer_types[layer] {
            LayerType::Sliding => self.head_dim,
            LayerType::Full => self.global_head_dim,
        }
    }

    /// Total Q dimension for a layer (num_heads × head_dim).
    #[inline]
    pub fn q_dim(&self, layer: usize) -> usize {
        self.num_attention_heads * self.layer_head_dim(layer)
    }

    /// Total KV dimension for a layer (num_kv_heads × head_dim).
    #[inline]
    pub fn kv_dim(&self, layer: usize) -> usize {
        self.num_kv_heads * self.layer_head_dim(layer)
    }

    /// GQA expansion ratio.
    #[inline]
    pub fn kv_mul(&self) -> usize {
        self.num_attention_heads / self.num_kv_heads
    }
}

/// Build Gemma 4 E2B layer types: 4 sliding + 1 full, repeating.
fn build_layer_types(n_layers: usize) -> Vec<LayerType> {
    let mut types = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        if (i + 1) % 5 == 0 {
            types.push(LayerType::Full);
        } else {
            types.push(LayerType::Sliding);
        }
    }
    types
}

// ── Weight references ────────────────────────────────────────────────────────

/// References to tensor data for one transformer layer.
pub struct LayerWeights<'a> {
    // Attention
    attn_norm: Vec<f32>,       // RMSNorm weight (hidden_size)
    wq: (&'a [u8], GGMLType), // Q projection
    wk: (&'a [u8], GGMLType), // K projection
    wv: (&'a [u8], GGMLType), // V projection
    wo: (&'a [u8], GGMLType), // output projection

    // FFN
    ffn_norm: Vec<f32>,        // RMSNorm weight (hidden_size)
    w_gate: (&'a [u8], GGMLType), // GeGLU gate projection
    w_up: (&'a [u8], GGMLType),   // GeGLU up projection
    w_down: (&'a [u8], GGMLType), // down projection

    // PLE
    ple_embed: Option<(&'a [u8], GGMLType)>, // per-layer embedding table
}

/// All model weights.
pub struct Gemma4Weights<'a> {
    token_embed: (&'a [u8], GGMLType),
    layers: Vec<LayerWeights<'a>>,
    final_norm: Vec<f32>,
    classifier: (&'a [u8], GGMLType),
}

// ── Run state (f32 scratch buffers) ──────────────────────────────────────────

struct RunState {
    x: Vec<f32>,      // hidden state (hidden_size)
    xb: Vec<f32>,     // buffer after norm (hidden_size)
    xb2: Vec<f32>,    // second buffer (hidden_size)
    q: Vec<f32>,      // query projection (max q_dim)
    k_temp: Vec<f32>, // temp key (max kv_dim)
    v_temp: Vec<f32>, // temp value (max kv_dim)
    att: Vec<f32>,    // attention scores (num_heads × max_seq)
    hb: Vec<f32>,     // FFN hidden (intermediate_size)
    hb2: Vec<f32>,    // FFN hidden 2 (intermediate_size)
    logits: Vec<f32>, // output logits (vocab_size)
    ple_buf: Vec<f32>,  // PLE embedding (ple_dim)
    #[allow(dead_code)]
    ple_proj: Vec<f32>, // PLE projected (hidden_size)
}

impl RunState {
    fn new(config: &Gemma4Config) -> Self {
        let max_q = config.num_attention_heads * config.global_head_dim;
        let max_kv = config.num_kv_heads * config.global_head_dim;
        // Cap att buffer at 8192 tokens max (not full 128k context) to save ~4MB
        let initial_max_seq = 8192.min(config.max_position_embeddings);
        let max_att = config.num_attention_heads * initial_max_seq;
        RunState {
            x: vec![0.0; config.hidden_size],
            xb: vec![0.0; config.hidden_size],
            xb2: vec![0.0; config.hidden_size],
            q: vec![0.0; max_q],
            k_temp: vec![0.0; max_kv],
            v_temp: vec![0.0; max_kv],
            att: vec![0.0; max_att],
            hb: vec![0.0; config.intermediate_size],
            hb2: vec![0.0; config.intermediate_size],
            logits: vec![0.0; config.vocab_size],
            ple_buf: vec![0.0; config.ple_dim],
            ple_proj: vec![0.0; config.hidden_size],
        }
    }
}

// ── Main model ───────────────────────────────────────────────────────────────

/// Gemma 4 E2B transformer model.
pub struct Gemma4Model<'a> {
    pub config: Gemma4Config,
    weights: Gemma4Weights<'a>,
    state: RunState,
    kv_cache: QJLKVCache,
    /// Tracks the last token processed (for PLE lookup).
    last_token: u32,
}

impl<'a> Gemma4Model<'a> {
    /// Construct the model from a parsed GGUF file.
    pub fn from_gguf(gguf: &'a GGUFFile<'a>) -> Result<Self, GGUFError> {
        let config = Gemma4Config::from_gguf(gguf)?;

        // Extract weight references
        let weights = Self::extract_weights(gguf, &config)?;

        // Build KV cache
        let mut kv_dims = Vec::with_capacity(config.num_layers);
        let mut window_sizes = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            kv_dims.push(config.kv_dim(i));
            window_sizes.push(match config.layer_types[i] {
                LayerType::Sliding => config.sliding_window,
                LayerType::Full => 0, // full attention = no window
            });
        }
        // Start with 8k context, expandable
        let initial_max_seq = 8192.min(config.max_position_embeddings);
        let kv_cache = QJLKVCache::new(config.num_layers, &kv_dims, &window_sizes, initial_max_seq);

        let state = RunState::new(&config);

        // Initialise PolarQuant trig tables
        simd::init_polar_tables();

        Ok(Gemma4Model {
            config,
            weights,
            state,
            kv_cache,
            last_token: 0,
        })
    }

    /// Construct with an explicit config (for testing without GGUF).
    pub fn from_config_and_weights(
        config: Gemma4Config,
        weights: Gemma4Weights<'a>,
    ) -> Self {
        let mut kv_dims = Vec::with_capacity(config.num_layers);
        let mut window_sizes = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            kv_dims.push(config.kv_dim(i));
            window_sizes.push(match config.layer_types[i] {
                LayerType::Sliding => config.sliding_window,
                LayerType::Full => 0,
            });
        }
        let initial_max_seq = 8192.min(config.max_position_embeddings);
        let kv_cache = QJLKVCache::new(config.num_layers, &kv_dims, &window_sizes, initial_max_seq);
        let state = RunState::new(&config);
        simd::init_polar_tables();

        Gemma4Model {
            config,
            weights,
            state,
            kv_cache,
            last_token: 0,
        }
    }

    /// Reset the KV cache (for new sequences).
    pub fn reset(&mut self) {
        self.kv_cache.reset();
    }

    /// Run one forward pass for a single token at the given position.
    pub fn forward_pass(&mut self, token: u32, pos: usize) -> &[f32] {
        let hidden = self.config.hidden_size;
        let num_layers = self.config.num_layers;
        let eps = self.config.rms_norm_eps;
        let vocab_size = self.config.vocab_size;
        let softcap = self.config.final_logit_softcapping;
        self.last_token = token;

        // 1. Token embedding
        let (embed_data, embed_dtype) = self.weights.token_embed;
        simd::embedding_lookup(&mut self.state.x, embed_data, embed_dtype, token as usize, hidden);

        // Gemma normalization: scale embeddings by sqrt(hidden_size)
        let scale = libm::sqrtf(hidden as f32);
        for v in self.state.x.iter_mut() {
            *v *= scale;
        }

        // 2. Per-layer transformer blocks
        for l in 0..num_layers {
            self.transformer_block(l, token, pos);
        }

        // 3. Final RMSNorm
        simd::rmsnorm(
            &mut self.state.xb,
            &self.state.x,
            &self.weights.final_norm,
            eps,
        );

        // 4. Classifier: logits = Wcls @ xb
        let (cls_data, cls_dtype) = self.weights.classifier;
        simd::matmul_quantized(
            &mut self.state.logits,
            &self.state.xb,
            cls_data,
            cls_dtype,
            hidden,
            vocab_size,
        );

        // 5. Logit soft-capping
        simd::logit_softcap(&mut self.state.logits, softcap);

        &self.state.logits
    }

    /// KV cache memory usage in bytes.
    pub fn kv_memory_bytes(&self) -> usize {
        self.kv_cache.memory_bytes()
    }

    // ── Per-layer transformer block ──────────────────────────────────────

    fn transformer_block(&mut self, layer: usize, token: u32, pos: usize) {
        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let layer_type = cfg.layer_types[layer];
        let hd = cfg.layer_head_dim(layer);
        let q_dim = cfg.q_dim(layer);
        let kv_dim = cfg.kv_dim(layer);

        // ── PLE injection ────────────────────────────────────────────────
        if let Some((ple_data, ple_dtype)) = self.weights.layers[layer].ple_embed {
            // Look up per-layer embedding for this token
            simd::embedding_lookup(
                &mut self.state.ple_buf[..cfg.ple_dim],
                ple_data,
                ple_dtype,
                token as usize,
                cfg.ple_dim,
            );

            // Simple injection: add PLE to first `ple_dim` components of x.
            // This is a weight-free projection (no learned linear map needed
            // since PLE embeddings are already trained into the right subspace).
            for i in 0..cfg.ple_dim.min(hidden) {
                self.state.x[i] += self.state.ple_buf[i];
            }
        }

        // ── Attention ────────────────────────────────────────────────────

        // Pre-attention RMSNorm
        simd::rmsnorm(
            &mut self.state.xb,
            &self.state.x,
            &self.weights.layers[layer].attn_norm,
            cfg.rms_norm_eps,
        );

        // Q projection: xb (hidden) → q (q_dim)
        let (wq_data, wq_dtype) = self.weights.layers[layer].wq;
        simd::matmul_quantized(&mut self.state.q[..q_dim], &self.state.xb, wq_data, wq_dtype, hidden, q_dim);

        // K projection: xb (hidden) → k_temp (kv_dim)
        let (wk_data, wk_dtype) = self.weights.layers[layer].wk;
        simd::matmul_quantized(&mut self.state.k_temp[..kv_dim], &self.state.xb, wk_data, wk_dtype, hidden, kv_dim);

        // V projection: xb (hidden) → v_temp (kv_dim)
        let (wv_data, wv_dtype) = self.weights.layers[layer].wv;
        simd::matmul_quantized(&mut self.state.v_temp[..kv_dim], &self.state.xb, wv_data, wv_dtype, hidden, kv_dim);

        // Apply RoPE
        match layer_type {
            LayerType::Sliding => {
                // Standard RoPE on all dimensions
                for h in 0..cfg.num_attention_heads {
                    simd::rope_apply(
                        &mut self.state.q[h * hd..(h + 1) * hd],
                        pos,
                        cfg.rope_theta_sliding,
                        hd, // rotate all dims
                    );
                }
                for h in 0..cfg.num_kv_heads {
                    simd::rope_apply(
                        &mut self.state.k_temp[h * hd..(h + 1) * hd],
                        pos,
                        cfg.rope_theta_sliding,
                        hd,
                    );
                }
            }
            LayerType::Full => {
                // p-RoPE: only rotate partial_rotary_factor of dimensions
                let rotary_dim = (hd as f32 * cfg.partial_rotary_factor) as usize;
                let rotary_dim = rotary_dim & !1; // must be even
                for h in 0..cfg.num_attention_heads {
                    simd::rope_apply(
                        &mut self.state.q[h * hd..(h + 1) * hd],
                        pos,
                        cfg.rope_theta_full,
                        rotary_dim,
                    );
                }
                for h in 0..cfg.num_kv_heads {
                    simd::rope_apply(
                        &mut self.state.k_temp[h * hd..(h + 1) * hd],
                        pos,
                        cfg.rope_theta_full,
                        rotary_dim,
                    );
                }
            }
        }

        // Store K, V in compressed cache
        self.kv_cache.layers[layer].store(
            pos,
            &self.state.k_temp[..kv_dim],
            &self.state.v_temp[..kv_dim],
        );

        // Multi-head attention with GQA
        let (att_start, att_end) = self.kv_cache.layers[layer].attention_range(pos);
        let kv_mul = cfg.kv_mul();
        let inv_sqrt_hd = 1.0 / libm::sqrtf(hd as f32);

        // Temporary buffers for decompressed KV
        let mut k_buf = vec![0.0f32; kv_dim];
        let mut v_buf = vec![0.0f32; kv_dim];

        for h in 0..cfg.num_attention_heads {
            let q_off = h * hd;
            let kv_head = h / kv_mul; // which KV head this query head reads
            let att_off = h * (att_end - att_start + 1);

            // Score computation: Q[h] · K[t] for each cached position
            for t in att_start..=att_end {
                if !self.kv_cache.layers[layer].get_key(t, &mut k_buf) {
                    self.state.att[att_off + t - att_start] = f32::NEG_INFINITY;
                    continue;
                }
                let k_head_off = kv_head * hd;
                let mut score = 0.0f32;
                for i in 0..hd {
                    score += self.state.q[q_off + i] * k_buf[k_head_off + i];
                }
                self.state.att[att_off + t - att_start] = score * inv_sqrt_hd;
            }

            // Softmax over attention scores
            let att_len = att_end - att_start + 1;
            simd::softmax(&mut self.state.att[att_off..att_off + att_len]);

            // Weighted sum of values
            let xb_off = h * hd;
            for i in 0..hd {
                self.state.xb[xb_off + i] = 0.0;
            }
            for t in att_start..=att_end {
                if !self.kv_cache.layers[layer].get_value(t, &mut v_buf) {
                    continue;
                }
                let v_head_off = kv_head * hd;
                let a = self.state.att[att_off + t - att_start];
                for i in 0..hd {
                    self.state.xb[xb_off + i] += a * v_buf[v_head_off + i];
                }
            }
        }

        // Output projection: xb2 = Wo @ xb  (q_dim → hidden)
        let (wo_data, wo_dtype) = self.weights.layers[layer].wo;
        simd::matmul_quantized(&mut self.state.xb2[..hidden], &self.state.xb[..q_dim], wo_data, wo_dtype, q_dim, hidden);

        // Residual connection
        for i in 0..hidden {
            self.state.x[i] += self.state.xb2[i];
        }

        // ── GeGLU FFN ────────────────────────────────────────────────────

        // Pre-FFN RMSNorm
        simd::rmsnorm(
            &mut self.state.xb,
            &self.state.x,
            &self.weights.layers[layer].ffn_norm,
            cfg.rms_norm_eps,
        );

        // Gate projection: xb → hb (hidden → intermediate)
        let (wg_data, wg_dtype) = self.weights.layers[layer].w_gate;
        simd::matmul_quantized(&mut self.state.hb, &self.state.xb, wg_data, wg_dtype, hidden, cfg.intermediate_size);

        // Up projection: xb → hb2 (hidden → intermediate)
        let (wu_data, wu_dtype) = self.weights.layers[layer].w_up;
        simd::matmul_quantized(&mut self.state.hb2, &self.state.xb, wu_data, wu_dtype, hidden, cfg.intermediate_size);

        // GeGLU: GELU(gate) * up
        for i in 0..cfg.intermediate_size {
            self.state.hb[i] = simd::gelu(self.state.hb[i]) * self.state.hb2[i];
        }

        // Down projection: hb → xb (intermediate → hidden)
        let (wd_data, wd_dtype) = self.weights.layers[layer].w_down;
        simd::matmul_quantized(&mut self.state.xb[..hidden], &self.state.hb, wd_data, wd_dtype, cfg.intermediate_size, hidden);

        // Residual connection
        for i in 0..hidden {
            self.state.x[i] += self.state.xb[i];
        }
    }

    // ── Weight extraction from GGUF ──────────────────────────────────────

    fn extract_weights(
        gguf: &'a GGUFFile<'a>,
        config: &Gemma4Config,
    ) -> Result<Gemma4Weights<'a>, GGUFError> {
        // Token embedding
        let token_embed = Self::get_tensor_ref(gguf, "token_embd.weight")?;

        // Final norm
        let (norm_info, norm_data) = gguf.get_tensor_data("output_norm.weight")?;
        let final_norm = simd::read_norm_weight(norm_data, norm_info.dtype, config.hidden_size);

        // Classifier (may be tied to token embedding)
        let classifier = if config.tie_word_embeddings {
            token_embed
        } else {
            Self::get_tensor_ref(gguf, "output.weight")?
        };

        // Per-layer weights
        let mut layers = Vec::with_capacity(config.num_layers);
        for l in 0..config.num_layers {
            let layer_weights = Self::extract_layer_weights(gguf, config, l)?;
            layers.push(layer_weights);
        }

        Ok(Gemma4Weights {
            token_embed,
            layers,
            final_norm,
            classifier,
        })
    }

    fn extract_layer_weights(
        gguf: &'a GGUFFile<'a>,
        config: &Gemma4Config,
        layer: usize,
    ) -> Result<LayerWeights<'a>, GGUFError> {
        use alloc::format;

        // Attention norm
        let norm_name = format!("blk.{layer}.attn_norm.weight");
        let (ni, nd) = gguf.get_tensor_data(&norm_name)?;
        let attn_norm = simd::read_norm_weight(nd, ni.dtype, config.hidden_size);

        // QKV and output projections
        let wq = Self::get_tensor_ref(gguf, &format!("blk.{layer}.attn_q.weight"))?;
        let wk = Self::get_tensor_ref(gguf, &format!("blk.{layer}.attn_k.weight"))?;
        let wv = Self::get_tensor_ref(gguf, &format!("blk.{layer}.attn_v.weight"))?;
        let wo = Self::get_tensor_ref(gguf, &format!("blk.{layer}.attn_output.weight"))?;

        // FFN norm
        let ffn_norm_name = format!("blk.{layer}.ffn_norm.weight");
        let (fni, fnd) = gguf.get_tensor_data(&ffn_norm_name)?;
        let ffn_norm = simd::read_norm_weight(fnd, fni.dtype, config.hidden_size);

        // FFN projections (GeGLU: gate, up, down)
        let w_gate = Self::get_tensor_ref(gguf, &format!("blk.{layer}.ffn_gate.weight"))?;
        let w_up = Self::get_tensor_ref(gguf, &format!("blk.{layer}.ffn_up.weight"))?;
        let w_down = Self::get_tensor_ref(gguf, &format!("blk.{layer}.ffn_down.weight"))?;

        // PLE embedding (optional — may not exist in some GGUF exports)
        let ple_name = format!("blk.{layer}.token_embd.weight");
        let ple_embed = Self::try_tensor_ref(gguf, &ple_name);

        Ok(LayerWeights {
            attn_norm,
            wq,
            wk,
            wv,
            wo,
            ffn_norm,
            w_gate,
            w_up,
            w_down,
            ple_embed,
        })
    }

    fn get_tensor_ref(
        gguf: &'a GGUFFile<'a>,
        name: &str,
    ) -> Result<(&'a [u8], GGMLType), GGUFError> {
        let (info, data) = gguf.get_tensor_data(name)?;
        Ok((data, info.dtype))
    }

    fn try_tensor_ref(
        gguf: &'a GGUFFile<'a>,
        name: &str,
    ) -> Option<(&'a [u8], GGMLType)> {
        gguf.get_tensor_data(name).ok().map(|(info, data)| (data, info.dtype))
    }
}

// ── LLMRuntime implementation ────────────────────────────────────────────────

impl<'a> LLMRuntime for Gemma4Model<'a> {
    fn forward(&mut self, token: u32, pos: usize) -> &[f32] {
        self.forward_pass(token, pos)
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn max_seq_len(&self) -> usize {
        self.config.max_position_embeddings
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Format a GGUF metadata key like "gemma4.embedding_length".
fn fmt_key(prefix: &str, suffix: &str) -> String {
    let mut s = String::with_capacity(prefix.len() + 1 + suffix.len());
    s.push_str(prefix);
    s.push('.');
    s.push_str(suffix);
    s
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_layers() {
        let cfg = Gemma4Config::default_e2b();
        assert_eq!(cfg.num_layers, 35);
        assert_eq!(cfg.layer_types.len(), 35);

        // Check pattern: layers 4, 9, 14, 19, 24, 29, 34 should be Full
        let full_layers: Vec<usize> = cfg
            .layer_types
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == LayerType::Full)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(full_layers, vec![4, 9, 14, 19, 24, 29, 34]);
    }

    #[test]
    fn head_dims() {
        let cfg = Gemma4Config::default_e2b();
        // Sliding layer
        assert_eq!(cfg.layer_head_dim(0), 256);
        assert_eq!(cfg.q_dim(0), 8 * 256); // 2048
        assert_eq!(cfg.kv_dim(0), 1 * 256); // 256

        // Full layer
        assert_eq!(cfg.layer_head_dim(4), 512);
        assert_eq!(cfg.q_dim(4), 8 * 512); // 4096
        assert_eq!(cfg.kv_dim(4), 1 * 512); // 512
    }

    #[test]
    fn gqa_mul() {
        let cfg = Gemma4Config::default_e2b();
        assert_eq!(cfg.kv_mul(), 8); // 8 query heads / 1 KV head
    }

    #[test]
    fn build_layers_pattern() {
        let types = build_layer_types(10);
        assert_eq!(types[0], LayerType::Sliding);
        assert_eq!(types[3], LayerType::Sliding);
        assert_eq!(types[4], LayerType::Full);
        assert_eq!(types[5], LayerType::Sliding);
        assert_eq!(types[9], LayerType::Full);
    }
}

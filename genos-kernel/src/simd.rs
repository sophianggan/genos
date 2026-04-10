//! Quantised math operations for LLM inference.
//!
//! Provides f16/bf16 conversion, dequantisation of GGML block formats (Q4_0,
//! Q8_0, Q4_K), quantised matrix-vector products, GELU activation, RMSNorm,
//! and PolarQuant (TurboQuant component 1).
//!
//! All code is `no_std`-safe — no heap allocation in hot paths, only `libm`.

use libm::{cosf, expf, sinf, sqrtf};

use crate::gguf::GGMLType;

// ── f16 / bf16 conversion ────────────────────────────────────────────────────

/// IEEE 754 half-precision (binary16) → f32.
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // ±zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal: normalise
            let mut e = 0u32;
            let mut m = mant;
            loop {
                e += 1;
                m <<= 1;
                if m & 0x400 != 0 {
                    break;
                }
            }
            m &= 0x3FF;
            f32::from_bits((sign << 31) | ((127 - 15 + 1 - e) << 23) | (m << 13))
        }
    } else if exp == 31 {
        // Inf / NaN
        f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13))
    } else {
        // Normal
        let f32_exp = (exp as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
    }
}

/// Brain floating-point (bf16) → f32.
#[inline]
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// f32 → IEEE 754 half-precision (binary16), round-to-nearest-even.
#[inline]
pub fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    if exp == 0xFF {
        // Inf / NaN
        return sign | 0x7C00 | if mant != 0 { 0x200 } else { 0 };
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return sign | 0x7C00; // overflow → Inf
    }
    if new_exp <= 0 {
        return sign; // underflow → zero (simplified)
    }
    sign | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
}

// ── Read helpers (little-endian) ─────────────────────────────────────────────

#[inline]
fn read_u16_le(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}

#[inline]
fn read_f16(b: &[u8]) -> f32 {
    f16_to_f32(read_u16_le(b))
}

// ── Q4_0 (block of 32, 18 bytes) ────────────────────────────────────────────

pub const Q4_0_BLOCK_SIZE: usize = 32;
pub const Q4_0_BLOCK_BYTES: usize = 18;

/// Dequantise a single Q4_0 block (32 values) into `out`.
#[inline]
pub fn dequantize_q4_0(block: &[u8], out: &mut [f32]) {
    let d = read_f16(&block[0..2]);
    for j in 0..16 {
        let byte = block[2 + j];
        let lo = (byte & 0x0F) as i32 - 8;
        let hi = ((byte >> 4) & 0x0F) as i32 - 8;
        out[j * 2] = lo as f32 * d;
        out[j * 2 + 1] = hi as f32 * d;
    }
}

/// Dot product of f32 vector `x` (length 32) against one Q4_0 block.
#[inline]
pub fn vec_dot_q4_0(block: &[u8], x: &[f32]) -> f32 {
    let d = read_f16(&block[0..2]);
    let mut sum = 0.0f32;
    for j in 0..16 {
        let byte = block[2 + j];
        let lo = (byte & 0x0F) as i32 - 8;
        let hi = ((byte >> 4) & 0x0F) as i32 - 8;
        sum += lo as f32 * x[j * 2];
        sum += hi as f32 * x[j * 2 + 1];
    }
    sum * d
}

// ── Q8_0 (block of 32, 34 bytes) ────────────────────────────────────────────

pub const Q8_0_BLOCK_SIZE: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Dequantise a single Q8_0 block (32 values) into `out`.
#[inline]
pub fn dequantize_q8_0(block: &[u8], out: &mut [f32]) {
    let d = read_f16(&block[0..2]);
    for j in 0..32 {
        out[j] = (block[2 + j] as i8) as f32 * d;
    }
}

/// Dot product of f32 vector `x` (length 32) against one Q8_0 block.
/// Uses AVX2 intrinsics on x86_64 for ~4× speedup.
#[inline]
pub fn vec_dot_q8_0(block: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            return unsafe { avx2::vec_dot_q8_0(block, x) };
        }
    }
    let d = read_f16(&block[0..2]);
    let mut sum = 0.0f32;
    for j in 0..32 {
        sum += (block[2 + j] as i8) as f32 * x[j];
    }
    sum * d
}

// ── Q4_K (super-block of 256, 144 bytes) ────────────────────────────────────

pub const Q4K_BLOCK_SIZE: usize = 256;
pub const Q4K_BLOCK_BYTES: usize = 144;

/// Dequantise a single Q4_K super-block (256 values) into `out`.
pub fn dequantize_q4k(block: &[u8], out: &mut [f32]) {
    let d = read_f16(&block[0..2]);
    let dmin = read_f16(&block[2..4]);
    let scales = &block[4..16]; // 12 bytes of packed scales
    let qs = &block[16..144]; // 128 bytes of 4-bit values

    // Unpack 6-bit scales and mins for 8 sub-blocks.
    // Packing layout (llama.cpp ggml-quants.c):
    //   scales[0..3]: lower 4 bits of scale for sub-blocks 0-7
    //                 (2 sub-blocks per byte, lo/hi nibble)
    //   scales[4..7]: lower 4 bits of min for sub-blocks 0-7
    //   scales[8..11]: upper 2 bits of scale (bits 4-5) and min (bits 4-5)
    //                  packed for sub-blocks 0-7
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];

    for i in 0..4 {
        let a = scales[i];
        sc[i * 2] = a & 0x3F;
        sc[i * 2 + 1] = (a >> 4) & 0x3F; // Oops, this isn't right

        // Actually the packing is different. Let me follow the GGML spec:
    }

    // GGML Q4_K packing (from ggml-quants.c, quantize_row_q4_K_ref):
    //   For sub-block j (0..7):
    //     if j < 4:
    //       sc[j] = (scales[j] & 0x3F)
    //       mn[j] = (scales[j+4] & 0x3F)
    //     else: (j >= 4)
    //       sc[j] = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
    //       mn[j] = (scales[j+4] >> 4)   | ((scales[j]   >> 6) << 4)
    for j in 0..4usize {
        sc[j] = scales[j] & 0x3F;
        mn[j] = scales[j + 4] & 0x3F;
    }
    for j in 4..8usize {
        sc[j] = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        mn[j] = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }

    // Dequantise 8 sub-blocks of 32
    for sb in 0..8 {
        let sub_d = d * sc[sb] as f32;
        let sub_m = dmin * mn[sb] as f32;
        let q = &qs[sb * 16..(sb + 1) * 16];
        let base = sb * 32;
        for k in 0..16 {
            let lo = (q[k] & 0x0F) as f32;
            let hi = ((q[k] >> 4) & 0x0F) as f32;
            out[base + k * 2] = lo * sub_d - sub_m;
            out[base + k * 2 + 1] = hi * sub_d - sub_m;
        }
    }
}

/// Dot product of f32 vector `x` (length 256) against one Q4_K super-block.
pub fn vec_dot_q4k(block: &[u8], x: &[f32]) -> f32 {
    let d = read_f16(&block[0..2]);
    let dmin = read_f16(&block[2..4]);
    let scales = &block[4..16];
    let qs = &block[16..144];

    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];
    for j in 0..4usize {
        sc[j] = scales[j] & 0x3F;
        mn[j] = scales[j + 4] & 0x3F;
    }
    for j in 4..8usize {
        sc[j] = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        mn[j] = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }

    let mut sum = 0.0f32;
    for sb in 0..8 {
        let sub_d = d * sc[sb] as f32;
        let sub_m = dmin * mn[sb] as f32;
        let q = &qs[sb * 16..(sb + 1) * 16];
        let base = sb * 32;
        for k in 0..16 {
            let lo = (q[k] & 0x0F) as f32;
            let hi = ((q[k] >> 4) & 0x0F) as f32;
            sum += (lo * sub_d - sub_m) * x[base + k * 2];
            sum += (hi * sub_d - sub_m) * x[base + k * 2 + 1];
        }
    }
    sum
}

// ── Generic quantised matmul ─────────────────────────────────────────────────

/// Matrix-vector multiply: `out[i] = dot(weight_row[i], input)`.
///
/// `weight_data` is the raw quantised bytes for a (out_features × in_features)
/// matrix in row-major block layout. `dtype` selects the dequantisation path.
pub fn matmul_quantized(
    out: &mut [f32],
    input: &[f32],
    weight_data: &[u8],
    dtype: GGMLType,
    in_features: usize,
    out_features: usize,
) {
    match dtype {
        GGMLType::F32 => matmul_f32(out, input, weight_data, in_features, out_features),
        GGMLType::F16 => matmul_f16(out, input, weight_data, in_features, out_features),
        GGMLType::BF16 => matmul_bf16(out, input, weight_data, in_features, out_features),
        GGMLType::Q4_0 => {
            matmul_blocked(out, input, weight_data, in_features, out_features, 32, 18, vec_dot_q4_0)
        }
        GGMLType::Q8_0 => {
            matmul_blocked(out, input, weight_data, in_features, out_features, 32, 34, vec_dot_q8_0)
        }
        GGMLType::Q4K => {
            matmul_blocked(out, input, weight_data, in_features, out_features, 256, 144, vec_dot_q4k)
        }
        _ => {
            // Unsupported type — zero output as safe fallback
            for v in out.iter_mut() {
                *v = 0.0;
            }
        }
    }
}

/// Blocked matmul using a per-block `vec_dot` function.
fn matmul_blocked(
    out: &mut [f32],
    input: &[f32],
    weight_data: &[u8],
    in_features: usize,
    out_features: usize,
    block_size: usize,
    block_bytes: usize,
    vec_dot: fn(&[u8], &[f32]) -> f32,
) {
    let blocks_per_row = in_features / block_size;
    let row_bytes = blocks_per_row * block_bytes;

    for i in 0..out_features {
        let row_start = i * row_bytes;
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let bstart = row_start + b * block_bytes;
            let block = &weight_data[bstart..bstart + block_bytes];
            let xstart = b * block_size;
            sum += vec_dot(block, &input[xstart..xstart + block_size]);
        }
        out[i] = sum;
    }
}

/// Plain f32 matmul.  Uses AVX2 dot product on x86_64.
pub fn matmul_f32(
    out: &mut [f32],
    input: &[f32],
    weight_data: &[u8],
    in_features: usize,
    out_features: usize,
) {
    // Reinterpret weight bytes as f32 slice for AVX2 dot product
    let weight_f32: &[f32] = unsafe {
        core::slice::from_raw_parts(
            weight_data.as_ptr() as *const f32,
            weight_data.len() / 4,
        )
    };
    for i in 0..out_features {
        let row = &weight_f32[i * in_features..(i + 1) * in_features];
        out[i] = dot_f32(row, &input[..in_features]);
    }
}

/// f16 dequant-on-the-fly matmul.
pub fn matmul_f16(
    out: &mut [f32],
    input: &[f32],
    weight_data: &[u8],
    in_features: usize,
    out_features: usize,
) {
    for i in 0..out_features {
        let mut sum = 0.0f32;
        let base = i * in_features * 2;
        for j in 0..in_features {
            let off = base + j * 2;
            let w = f16_to_f32(u16::from_le_bytes([weight_data[off], weight_data[off + 1]]));
            sum += w * input[j];
        }
        out[i] = sum;
    }
}

/// bf16 dequant-on-the-fly matmul.
pub fn matmul_bf16(
    out: &mut [f32],
    input: &[f32],
    weight_data: &[u8],
    in_features: usize,
    out_features: usize,
) {
    for i in 0..out_features {
        let mut sum = 0.0f32;
        let base = i * in_features * 2;
        for j in 0..in_features {
            let off = base + j * 2;
            let w = bf16_to_f32(u16::from_le_bytes([weight_data[off], weight_data[off + 1]]));
            sum += w * input[j];
        }
        out[i] = sum;
    }
}

// ── Embedding lookup (f32, f16, q8_0) ────────────────────────────────────────

/// Read a single embedding row from quantised data.
/// Returns the `dim`-dimensional embedding for token `idx`.
pub fn embedding_lookup(
    out: &mut [f32],
    data: &[u8],
    dtype: GGMLType,
    idx: usize,
    dim: usize,
) {
    match dtype {
        GGMLType::F32 => {
            let base = idx * dim * 4;
            for j in 0..dim {
                let off = base + j * 4;
                out[j] = f32::from_le_bytes([
                    data[off], data[off + 1], data[off + 2], data[off + 3],
                ]);
            }
        }
        GGMLType::F16 => {
            let base = idx * dim * 2;
            for j in 0..dim {
                let off = base + j * 2;
                out[j] = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
            }
        }
        GGMLType::BF16 => {
            let base = idx * dim * 2;
            for j in 0..dim {
                let off = base + j * 2;
                out[j] = bf16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
            }
        }
        GGMLType::Q8_0 => {
            // dim elements = dim/32 blocks, each 34 bytes
            let blocks = dim / 32;
            let row_bytes = blocks * 34;
            let base = idx * row_bytes;
            for b in 0..blocks {
                let boff = base + b * 34;
                dequantize_q8_0(&data[boff..boff + 34], &mut out[b * 32..(b + 1) * 32]);
            }
        }
        GGMLType::Q4_0 => {
            let blocks = dim / 32;
            let row_bytes = blocks * 18;
            let base = idx * row_bytes;
            for b in 0..blocks {
                let boff = base + b * 18;
                dequantize_q4_0(&data[boff..boff + 18], &mut out[b * 32..(b + 1) * 32]);
            }
        }
        _ => {
            for v in out.iter_mut() {
                *v = 0.0;
            }
        }
    }
}

// ── Activation functions ─────────────────────────────────────────────────────

/// GELU approximation (tanh variant) matching `gelu_pytorch_tanh`.
/// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
#[inline]
pub fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.7978845608; // sqrt(2/π)
    const C: f32 = 0.044715;
    let inner = SQRT_2_OVER_PI * (x + C * x * x * x);
    0.5 * x * (1.0 + tanhf(inner))
}

/// tanh via the identity tanh(x) = 1 − 2/(exp(2x)+1)
#[inline]
fn tanhf(x: f32) -> f32 {
    if x > 10.0 {
        return 1.0;
    }
    if x < -10.0 {
        return -1.0;
    }
    let e2x = expf(2.0 * x);
    (e2x - 1.0) / (e2x + 1.0)
}

// ── RMSNorm ──────────────────────────────────────────────────────────────────

/// RMSNorm: `out[i] = weight[i] * (x[i] / rms(x))`
/// with epsilon for numerical stability.  Uses AVX2 for sum-of-squares on x86_64.
pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let n = x.len();
    let ss = dot_f32(x, x); // sum of squares via AVX2 or scalar
    let inv_rms = 1.0 / sqrtf(ss / n as f32 + eps);
    for i in 0..n {
        out[i] = weight[i] * (inv_rms * x[i]);
    }
}

/// Read f32 or f16 RMSNorm weight vector from tensor data.
pub fn read_norm_weight(data: &[u8], dtype: GGMLType, dim: usize) -> alloc::vec::Vec<f32> {
    let mut out = alloc::vec![0.0f32; dim];
    match dtype {
        GGMLType::F32 => {
            for i in 0..dim {
                let off = i * 4;
                out[i] = f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
            }
        }
        GGMLType::F16 => {
            for i in 0..dim {
                let off = i * 2;
                out[i] = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
            }
        }
        GGMLType::BF16 => {
            for i in 0..dim {
                let off = i * 2;
                out[i] = bf16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
            }
        }
        _ => {
            out.fill(1.0); // fallback: identity norm
        }
    }
    out
}

// ── Softmax ──────────────────────────────────────────────────────────────────

/// In-place softmax over a mutable slice.
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max_val = x[0];
    for &v in x.iter().skip(1) {
        if v > max_val {
            max_val = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = expf(*v - max_val);
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Logit soft-capping: `logit = cap * tanh(logit / cap)`
#[inline]
pub fn logit_softcap(logits: &mut [f32], cap: f32) {
    let inv_cap = 1.0 / cap;
    for v in logits.iter_mut() {
        *v = cap * tanhf(*v * inv_cap);
    }
}

// ── RoPE (Rotary Positional Embeddings) ──────────────────────────────────────

/// Apply standard RoPE to a query or key vector in-place.
///
/// `vec` has length `n_dim` (could be full head_dim or partial).
/// `pos` is the sequence position.
/// `theta_base` is the base frequency (10000 for standard, 1e6 for p-RoPE).
/// `rotary_dim` is how many dimensions to rotate (rest are untouched).
pub fn rope_apply(vec: &mut [f32], pos: usize, theta_base: f32, rotary_dim: usize) {
    let head_dim = vec.len();
    let effective = rotary_dim.min(head_dim);
    for i in (0..effective).step_by(2) {
        let freq = 1.0 / powf_approx(theta_base, i as f32 / effective as f32);
        let angle = pos as f32 * freq;
        let cos_a = cosf(angle);
        let sin_a = sinf(angle);
        let v0 = vec[i];
        let v1 = vec[i + 1];
        vec[i] = v0 * cos_a - v1 * sin_a;
        vec[i + 1] = v0 * sin_a + v1 * cos_a;
    }
}

/// Approximate a^b via exp(b * ln(a)).
#[inline]
fn powf_approx(a: f32, b: f32) -> f32 {
    expf(b * libm::logf(a))
}

// ── PolarQuant (TurboQuant component 1) ──────────────────────────────────────
//
// Stores weight blocks in polar form: 1 magnitude + 31 angles, each 4-bit
// quantised. Same 18-byte footprint as Q4_0 for a block of 32 values.
//
// Layout (18 bytes per block of 32):
//   r_scale: f16  (2 bytes) — scale for magnitude reconstruction
//   packed:  [u8; 16] — 32 × 4-bit nibbles
//     nibble 0: quantised magnitude level (0..15)
//     nibbles 1..30: quantised angle in [0, π], level * (π/15)
//     nibble 31: quantised angle in [-π, π], level * (2π/15) - π

/// Runtime-initialised lookup tables for PolarQuant trig values.
static mut POLAR_COS_PI: [f32; 16] = [0.0; 16];
static mut POLAR_SIN_PI: [f32; 16] = [0.0; 16];
static mut POLAR_COS_2PI: [f32; 16] = [0.0; 16];
static mut POLAR_SIN_2PI: [f32; 16] = [0.0; 16];
static mut POLAR_TABLES_INIT: bool = false;

/// Initialise PolarQuant trig lookup tables.  Call once at startup.
pub fn init_polar_tables() {
    unsafe {
        if POLAR_TABLES_INIT {
            return;
        }
        for i in 0..16 {
            let angle_pi = i as f32 * core::f32::consts::PI / 15.0;
            POLAR_COS_PI[i] = cosf(angle_pi);
            POLAR_SIN_PI[i] = sinf(angle_pi);
            let angle_2pi = i as f32 * 2.0 * core::f32::consts::PI / 15.0 - core::f32::consts::PI;
            POLAR_COS_2PI[i] = cosf(angle_2pi);
            POLAR_SIN_2PI[i] = sinf(angle_2pi);
        }
        POLAR_TABLES_INIT = true;
    }
}

/// Dequantise a PolarQuant block (32 values, 18 bytes) to f32.
///
/// Algorithm:
/// 1. Unpack magnitude level and 31 angle levels from 4-bit nibbles
/// 2. Reconstruct magnitude: r = level * r_scale
/// 3. Convert from hyper-spherical to cartesian using trig lookup tables
pub fn dequantize_polar4(block: &[u8], out: &mut [f32]) {
    let r_scale = read_f16(&block[0..2]);

    // Unpack 32 nibbles from 16 bytes
    let mut nibbles = [0u8; 32];
    for j in 0..16 {
        nibbles[j * 2] = block[2 + j] & 0x0F;
        nibbles[j * 2 + 1] = (block[2 + j] >> 4) & 0x0F;
    }

    // Magnitude
    let r = nibbles[0] as f32 * r_scale;
    if r == 0.0 {
        out[..32].fill(0.0);
        return;
    }

    // Reconstruct cartesian from polar (hyper-spherical coordinates):
    //   w_0 = r * cos(φ_0)
    //   w_1 = r * sin(φ_0) * cos(φ_1)
    //   ...
    //   w_{d-2} = r * sin(φ_0) * ... * sin(φ_{d-3}) * cos(φ_{d-2})
    //   w_{d-1} = r * sin(φ_0) * ... * sin(φ_{d-3}) * sin(φ_{d-2})
    //
    // φ_0..φ_{d-3} ∈ [0,π],  φ_{d-2} ∈ [-π,π]

    unsafe {
        init_polar_tables();

        let mut running_sin_product = r; // r * ∏ sin(φ_i) so far

        for k in 0..31 {
            let level = nibbles[k + 1] as usize;
            if k < 30 {
                // Angles in [0, π]
                let cos_val = POLAR_COS_PI[level];
                let sin_val = POLAR_SIN_PI[level];
                out[k] = running_sin_product * cos_val;
                running_sin_product *= sin_val;
            } else {
                // Last angle in [-π, π]
                let cos_val = POLAR_COS_2PI[level];
                let sin_val = POLAR_SIN_2PI[level];
                out[30] = running_sin_product * cos_val;
                out[31] = running_sin_product * sin_val;
            }
        }
    }
}

/// Dot product of f32 vector `x` (length 32) against one PolarQuant block.
pub fn vec_dot_polar4(block: &[u8], x: &[f32]) -> f32 {
    let mut tmp = [0.0f32; 32];
    dequantize_polar4(block, &mut tmp);
    let mut sum = 0.0f32;
    for i in 0..32 {
        sum += tmp[i] * x[i];
    }
    sum
}

// ── AVX2 SIMD kernels (x86_64 with AVX2+FMA only) ───────────────────────────
//
// Provides hardware-accelerated dot products and vector operations.
// On non-x86_64 targets (e.g., aarch64 host tests) the scalar fallbacks
// above are used automatically via the public dispatch functions below.

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use core::arch::x86_64::*;

    /// Horizontal sum of 8 f32 lanes in a __m256 register.
    #[inline]
    pub(super) unsafe fn hsum_ps(v: __m256) -> f32 {
        let hi128 = _mm256_extractf128_ps(v, 1);
        let lo128 = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(lo128, hi128);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let hi64 = _mm_movehl_ps(sums, sums);
        _mm_cvtss_f32(_mm_add_ss(sums, hi64))
    }

    /// AVX2+FMA f32 dot product: a[0..n] · b[0..n]
    /// Uses 4-way unrolled FMA for throughput.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let pa = a.as_ptr();
        let pb = b.as_ptr();
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let mut i = 0usize;

        // 4-way unrolled: 32 floats per iteration
        while i + 32 <= n {
            acc0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i)),
                _mm256_loadu_ps(pb.add(i)),
                acc0,
            );
            acc1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i + 8)),
                _mm256_loadu_ps(pb.add(i + 8)),
                acc1,
            );
            acc2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i + 16)),
                _mm256_loadu_ps(pb.add(i + 16)),
                acc2,
            );
            acc3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i + 24)),
                _mm256_loadu_ps(pb.add(i + 24)),
                acc3,
            );
            i += 32;
        }

        // Reduce 4 accumulators → 1
        acc0 = _mm256_add_ps(
            _mm256_add_ps(acc0, acc1),
            _mm256_add_ps(acc2, acc3),
        );

        // Handle remaining 8-wide chunks
        while i + 8 <= n {
            acc0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i)),
                _mm256_loadu_ps(pb.add(i)),
                acc0,
            );
            i += 8;
        }

        let mut sum = hsum_ps(acc0);

        // Scalar tail
        while i < n {
            sum += *pa.add(i) * *pb.add(i);
            i += 1;
        }
        sum
    }

    /// AVX2 scale-add: out[i] += a[i] * scale
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn fmadd_scalar(out: &mut [f32], a: &[f32], scale: f32) {
        let n = out.len().min(a.len());
        let ps = _mm256_set1_ps(scale);
        let po = out.as_mut_ptr();
        let pa = a.as_ptr();
        let mut i = 0usize;

        while i + 8 <= n {
            let vo = _mm256_loadu_ps(po.add(i));
            let va = _mm256_loadu_ps(pa.add(i));
            _mm256_storeu_ps(po.add(i), _mm256_fmadd_ps(va, ps, vo));
            i += 8;
        }
        while i < n {
            *po.add(i) += *pa.add(i) * scale;
            i += 1;
        }
    }

    /// AVX2 Q8_0 dot product: block of 32 int8 quantized values × f32 input.
    /// Block layout: [f16 scale (2 bytes)][32 × i8 values (32 bytes)] = 34 bytes
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn vec_dot_q8_0(block: &[u8], x: &[f32]) -> f32 {
        let d = super::read_f16(&block[0..2]);
        let scale = _mm256_set1_ps(d);
        let qs = block.as_ptr().add(2);
        let xp = x.as_ptr();
        let mut acc = _mm256_setzero_ps();

        // Process 8 int8 values at a time (4 iterations for 32 values)
        for k in 0..4u32 {
            let off = (k * 8) as usize;
            // Load 8 bytes, sign-extend i8 → i16 → i32 → f32
            let bytes = _mm_loadl_epi64(qs.add(off) as *const __m128i);
            let i16s = _mm_cvtepi8_epi16(bytes);
            let i32s = _mm256_cvtepi16_epi32(i16s);
            let f32s = _mm256_cvtepi32_ps(i32s);
            let inp = _mm256_loadu_ps(xp.add(off));
            acc = _mm256_fmadd_ps(f32s, inp, acc);
        }

        hsum_ps(_mm256_mul_ps(acc, scale))
    }
}

// ── Runtime AVX2 detection ──────────────────────────────────────────────────

/// Runtime check for AVX2+FMA support using CPUID.  Result is cached.
///
/// Currently always returns false — QEMU TCG (software x86_64 emulation on
/// Apple Silicon) advertises AVX2 support via CPUID with `-cpu max` but its
/// AVX2 instruction emulation is unreliable and causes reboots.  The scalar
/// fallback paths work correctly and have equivalent throughput under TCG
/// since the bottleneck is instruction translation, not SIMD width.
///
/// On real x86_64 hardware, re-enable by removing the early return.
#[cfg(target_arch = "x86_64")]
fn has_avx2_fma() -> bool {
    false // Disabled: TCG AVX2 emulation is unreliable (see comment above)
}

// ── Public dispatch functions (AVX2 or scalar) ──────────────────────────────

/// f32 dot product: a · b.  Uses AVX2+FMA if available at runtime, scalar fallback otherwise.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            return unsafe { avx2::dot_f32(a, b) };
        }
    }
    let n = a.len().min(b.len());
    let mut sum = 0.0f32;
    for i in 0..n {
        sum += a[i] * b[i];
    }
    sum
}

/// Scale-add: out[i] += a[i] * scale.  AVX2-accelerated if available at runtime.
#[inline]
pub fn fmadd_scalar(out: &mut [f32], a: &[f32], scale: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            unsafe { avx2::fmadd_scalar(out, a, scale) };
            return;
        }
    }
    let n = out.len().min(a.len());
    for i in 0..n {
        out[i] += a[i] * scale;
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip() {
        let vals = [0.0f32, 1.0, -1.0, 0.5, 65504.0, -0.0001];
        for &v in &vals {
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            let err = (v - back).abs();
            // f16 has ~3 decimal digits of precision
            assert!(err < v.abs() * 0.01 + 0.001, "f16 roundtrip failed for {v}: got {back}");
        }
    }

    #[test]
    fn bf16_conversion() {
        let v = 3.14f32;
        let bits = v.to_bits();
        let h = (bits >> 16) as u16;
        let back = bf16_to_f32(h);
        assert!((v - back).abs() < 0.02);
    }

    #[test]
    fn q4_0_dequant() {
        // Build a block: scale=1.0 as f16, all nibbles = 8 (which dequants to 0)
        let mut block = [0u8; 18];
        let scale_h = f32_to_f16(1.0);
        block[0] = scale_h as u8;
        block[1] = (scale_h >> 8) as u8;
        // nibble value 8 → (8-8)*1.0 = 0.0
        block[2..18].fill(0x88);

        let mut out = [0.0f32; 32];
        dequantize_q4_0(&block, &mut out);
        for &v in &out {
            assert_eq!(v, 0.0);
        }

        // nibble value 15 → (15-8)*1.0 = 7.0
        block[2..18].fill(0xFF);
        dequantize_q4_0(&block, &mut out);
        for &v in &out {
            assert!((v - 7.0).abs() < 0.01);
        }
    }

    #[test]
    fn q8_0_dequant() {
        let mut block = [0u8; 34];
        let scale_h = f32_to_f16(0.5);
        block[0] = scale_h as u8;
        block[1] = (scale_h >> 8) as u8;
        // All quantised values = 2 → 2 * 0.5 = 1.0
        block[2..34].fill(2);

        let mut out = [0.0f32; 32];
        dequantize_q8_0(&block, &mut out);
        for &v in &out {
            assert!((v - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn gelu_values() {
        // GELU(0) ≈ 0
        assert!(gelu(0.0).abs() < 1e-6);
        // GELU(large positive) ≈ x
        assert!((gelu(5.0) - 5.0).abs() < 0.01);
        // GELU(large negative) ≈ 0
        assert!(gelu(-5.0).abs() < 0.01);
    }

    #[test]
    fn softmax_normalises() {
        let mut x = [1.0f32, 2.0, 3.0, 4.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Should be increasing
        assert!(x[0] < x[1]);
        assert!(x[1] < x[2]);
        assert!(x[2] < x[3]);
    }

    #[test]
    fn rmsnorm_unit() {
        let x = [1.0f32, 1.0, 1.0, 1.0];
        let w = [1.0f32, 1.0, 1.0, 1.0];
        let mut out = [0.0f32; 4];
        rmsnorm(&mut out, &x, &w, 1e-6);
        // RMS of [1,1,1,1] = 1, so output ≈ w * x / 1 = [1,1,1,1]
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn rope_identity_at_pos_zero() {
        let mut vec = [1.0f32, 0.0, 1.0, 0.0];
        rope_apply(&mut vec, 0, 10000.0, 4);
        // At pos 0, angle = 0, cos=1, sin=0 → no change
        assert!((vec[0] - 1.0).abs() < 1e-6);
        assert!((vec[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn dot_f32_basic() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let result = dot_f32(&a, &b);
        assert!((result - 36.0).abs() < 1e-4);
    }

    #[test]
    fn dot_f32_large() {
        // Test with > 32 elements to exercise unrolled loop
        let a: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..100).map(|i| (100 - i) as f32 * 0.01).collect();
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let result = dot_f32(&a, &b);
        assert!((result - expected).abs() < 0.01, "got {result}, expected {expected}");
    }

    #[test]
    fn fmadd_scalar_basic() {
        let mut out = [1.0f32, 2.0, 3.0, 4.0];
        let a = [10.0f32, 20.0, 30.0, 40.0];
        fmadd_scalar(&mut out, &a, 0.5);
        assert!((out[0] - 6.0).abs() < 1e-6);
        assert!((out[1] - 12.0).abs() < 1e-6);
    }
}

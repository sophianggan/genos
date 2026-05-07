//! QJL-compressed KV cache (TurboQuant component 2).
//!
//! Uses Quantised Johnson–Lindenstrauss inspired compression:
//!   - Keys:   3-bit per dimension (8 levels, group-wise scale)
//!   - Values: 2-bit per dimension (4 levels, group-wise scale)
//!
//! Compression ratio vs f32:
//!   Keys:  32/3 ≈ 10.7×   Values: 32/2 = 16×
//!
//! All allocations use `alloc` and the cache is `no_std`-safe.

use alloc::vec::Vec;

use crate::simd;

// ── Constants ────────────────────────────────────────────────────────────────

/// Number of elements per quantisation group (one scale factor per group).
const GROUP_SIZE: usize = 32;

// ── 3-bit key compression ────────────────────────────────────────────────────
//
// 3-bit signed: values in {-4, -3, -2, -1, 0, 1, 2, 3}
// Packed: 32 values (= 1 group) → 32 × 3 = 96 bits = 12 bytes
// Plus 2 bytes for f16 scale → 14 bytes per group of 32.
// Effective: 14 / 32 = 0.4375 bytes per element (vs 4 for f32 = 9.1× compression).

/// Bytes per group for 3-bit key storage: 2 (f16 scale) + 12 (packed 3-bit).
const KEY_GROUP_BYTES: usize = 14;

/// Compress `dim` f32 values to 3-bit quantised with group-wise scaling.
/// Returns packed byte vector.
fn compress_3bit(input: &[f32]) -> Vec<u8> {
    let dim = input.len();
    let n_groups = (dim + GROUP_SIZE - 1) / GROUP_SIZE;
    let mut packed = Vec::with_capacity(n_groups * KEY_GROUP_BYTES);

    for g in 0..n_groups {
        let start = g * GROUP_SIZE;
        let end = (start + GROUP_SIZE).min(dim);
        let count = end - start;

        // Find max absolute value for this group
        let mut amax = 0.0f32;
        for i in start..end {
            let a = if input[i] < 0.0 { -input[i] } else { input[i] };
            if a > amax {
                amax = a;
            }
        }

        // Scale: map [-amax, amax] to [-4, 3] (3-bit signed)
        let scale = if amax > 0.0 { amax / 4.0 } else { 0.0 };
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        // Write f16 scale
        let scale_h = simd::f32_to_f16(scale);
        packed.push(scale_h as u8);
        packed.push((scale_h >> 8) as u8);

        // Quantise to 3-bit signed: clamp to [-4, 3], round to nearest
        let mut q3 = [0i8; GROUP_SIZE];
        for i in 0..count {
            let v = input[start + i] * inv_scale;
            let rounded = if v < 0.0 {
                (v - 0.5) as i8
            } else {
                (v + 0.5) as i8
            };
            q3[i] = rounded.max(-4).min(3);
        }

        // Pack 32 × 3-bit values into 12 bytes (96 bits).
        // Encoding: offset by +4 to get unsigned 0..7, then pack sequentially.
        // Bit layout: values packed LSB-first into bytes.
        let mut bits = [0u8; 12];
        let mut bit_pos = 0usize;
        for i in 0..GROUP_SIZE {
            let unsigned = (q3[i] + 4) as u8; // 0..7
            let byte_idx = bit_pos / 8;
            let bit_offset = bit_pos % 8;
            if byte_idx < 12 {
                bits[byte_idx] |= (unsigned & 0x07) << bit_offset;
                // If bits spill into next byte
                if bit_offset > 5 && byte_idx + 1 < 12 {
                    bits[byte_idx + 1] |= unsigned >> (8 - bit_offset);
                }
            }
            bit_pos += 3;
        }
        packed.extend_from_slice(&bits);
    }
    packed
}

/// Decompress 3-bit quantised data back to f32.
fn decompress_3bit(packed: &[u8], dim: usize, out: &mut [f32]) {
    let n_groups = (dim + GROUP_SIZE - 1) / GROUP_SIZE;
    for g in 0..n_groups {
        let base = g * KEY_GROUP_BYTES;
        let scale = simd::f16_to_f32(u16::from_le_bytes([packed[base], packed[base + 1]]));
        let bits = &packed[base + 2..base + 14];

        let start = g * GROUP_SIZE;
        let end = (start + GROUP_SIZE).min(dim);

        let mut bit_pos = 0usize;
        for i in start..end {
            let byte_idx = bit_pos / 8;
            let bit_offset = bit_pos % 8;
            let mut val = (bits[byte_idx] >> bit_offset) & 0x07;
            if bit_offset > 5 && byte_idx + 1 < bits.len() {
                val |= (bits[byte_idx + 1] << (8 - bit_offset)) & 0x07;
            }
            let signed = val as i8 - 4; // back to [-4, 3]
            out[i] = signed as f32 * scale;
            bit_pos += 3;
        }
    }
}

// ── 2-bit value compression ──────────────────────────────────────────────────
//
// 2-bit signed: values in {-2, -1, 0, 1}
// Packed: 32 values → 64 bits = 8 bytes
// Plus 2 bytes for f16 scale → 10 bytes per group of 32.
// Effective: 10 / 32 = 0.3125 bytes per element (vs 4 for f32 = 12.8× compression).

const VAL_GROUP_BYTES: usize = 10;

/// Compress `dim` f32 values to 2-bit quantised with group-wise scaling.
fn compress_2bit(input: &[f32]) -> Vec<u8> {
    let dim = input.len();
    let n_groups = (dim + GROUP_SIZE - 1) / GROUP_SIZE;
    let mut packed = Vec::with_capacity(n_groups * VAL_GROUP_BYTES);

    for g in 0..n_groups {
        let start = g * GROUP_SIZE;
        let end = (start + GROUP_SIZE).min(dim);
        let count = end - start;

        let mut amax = 0.0f32;
        for i in start..end {
            let a = if input[i] < 0.0 { -input[i] } else { input[i] };
            if a > amax {
                amax = a;
            }
        }

        let scale = if amax > 0.0 { amax / 2.0 } else { 0.0 };
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        let scale_h = simd::f32_to_f16(scale);
        packed.push(scale_h as u8);
        packed.push((scale_h >> 8) as u8);

        // Quantise to 2-bit signed [-2, 1]
        let mut q2 = [0i8; GROUP_SIZE];
        for i in 0..count {
            let v = input[start + i] * inv_scale;
            let rounded = if v < 0.0 {
                (v - 0.5) as i8
            } else {
                (v + 0.5) as i8
            };
            q2[i] = rounded.max(-2).min(1);
        }

        // Pack 32 × 2-bit values into 8 bytes.
        // Offset by +2 to get unsigned 0..3.
        let mut bytes = [0u8; 8];
        for i in 0..GROUP_SIZE {
            let unsigned = (q2[i] + 2) as u8; // 0..3
            let byte_idx = i / 4;
            let shift = (i % 4) * 2;
            bytes[byte_idx] |= (unsigned & 0x03) << shift;
        }
        packed.extend_from_slice(&bytes);
    }
    packed
}

/// Decompress 2-bit quantised data back to f32.
fn decompress_2bit(packed: &[u8], dim: usize, out: &mut [f32]) {
    let n_groups = (dim + GROUP_SIZE - 1) / GROUP_SIZE;
    for g in 0..n_groups {
        let base = g * VAL_GROUP_BYTES;
        let scale = simd::f16_to_f32(u16::from_le_bytes([packed[base], packed[base + 1]]));
        let bytes = &packed[base + 2..base + 10];

        let start = g * GROUP_SIZE;
        let end = (start + GROUP_SIZE).min(dim);

        for i in start..end {
            let local = i - start;
            let byte_idx = local / 4;
            let shift = (local % 4) * 2;
            let unsigned = (bytes[byte_idx] >> shift) & 0x03;
            let signed = unsigned as i8 - 2; // back to [-2, 1]
            out[i] = signed as f32 * scale;
        }
    }
}

// ── Per-layer KV cache ───────────────────────────────────────────────────────

/// Compressed KV entry for one sequence position.
struct KVEntry {
    key_packed: Vec<u8>,
    value_packed: Vec<u8>,
}

/// KV cache for a single attention layer.
pub struct LayerKVCache {
    /// Max sequence positions.
    max_seq: usize,
    /// KV head dimension.
    kv_dim: usize,
    /// Stored entries (indexed by position). `None` = not yet filled.
    entries: Vec<Option<KVEntry>>,
    /// Sliding window size (0 = full attention).
    window_size: usize,
}

impl LayerKVCache {
    pub fn new(max_seq: usize, kv_dim: usize, window_size: usize) -> Self {
        let mut entries = Vec::with_capacity(max_seq);
        for _ in 0..max_seq {
            entries.push(None);
        }
        LayerKVCache {
            max_seq,
            kv_dim,
            entries,
            window_size,
        }
    }

    /// Store a KV pair at the given position (compresses in place).
    pub fn store(&mut self, pos: usize, key: &[f32], value: &[f32]) {
        if pos >= self.max_seq {
            return;
        }
        self.entries[pos] = Some(KVEntry {
            key_packed: compress_3bit(key),
            value_packed: compress_2bit(value),
        });
    }

    /// Decompress key at `pos` into `out`. Returns false if position is empty.
    pub fn get_key(&self, pos: usize, out: &mut [f32]) -> bool {
        if pos >= self.max_seq {
            return false;
        }
        match &self.entries[pos] {
            Some(entry) => {
                decompress_3bit(&entry.key_packed, self.kv_dim, out);
                true
            }
            None => false,
        }
    }

    /// Decompress value at `pos` into `out`. Returns false if position is empty.
    pub fn get_value(&self, pos: usize, out: &mut [f32]) -> bool {
        if pos >= self.max_seq {
            return false;
        }
        match &self.entries[pos] {
            Some(entry) => {
                decompress_2bit(&entry.value_packed, self.kv_dim, out);
                true
            }
            None => false,
        }
    }

    /// Determine valid range of positions for attention given current position
    /// and this layer's window configuration.
    pub fn attention_range(&self, current_pos: usize) -> (usize, usize) {
        if self.window_size == 0 {
            // Full attention
            (0, current_pos)
        } else {
            // Sliding window
            let start = if current_pos >= self.window_size {
                current_pos - self.window_size + 1
            } else {
                0
            };
            (start, current_pos)
        }
    }

    /// Clear all stored entries (for sequence reset).
    pub fn reset(&mut self) {
        for entry in self.entries.iter_mut() {
            *entry = None;
        }
    }
}

// ── Full KV cache (all layers) ───────────────────────────────────────────────

/// QJL-compressed KV cache for all transformer layers.
pub struct QJLKVCache {
    pub layers: Vec<LayerKVCache>,
}

impl QJLKVCache {
    /// Create a KV cache for `n_layers` layers.
    ///
    /// `kv_dims[i]` is the KV dimension for layer `i`.
    /// `window_sizes[i]` is the sliding window for layer `i` (0 = full attention).
    /// `max_seq` is the maximum sequence length.
    pub fn new(
        n_layers: usize,
        kv_dims: &[usize],
        window_sizes: &[usize],
        max_seq: usize,
    ) -> Self {
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            layers.push(LayerKVCache::new(max_seq, kv_dims[i], window_sizes[i]));
        }
        QJLKVCache { layers }
    }

    /// Reset all layers.
    pub fn reset(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.reset();
        }
    }

    /// Estimate memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let mut total = 0;
        for layer in &self.layers {
            for entry in &layer.entries {
                if let Some(e) = entry {
                    total += e.key_packed.len() + e.value_packed.len();
                }
            }
        }
        total
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_3bit() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let packed = compress_3bit(&input);
        let mut out = vec![0.0f32; 64];
        decompress_3bit(&packed, 64, &mut out);
        // 3-bit has limited precision — check within tolerance
        for i in 0..64 {
            let err = (input[i] - out[i]).abs();
            // Scale-dependent tolerance: ~scale/4 quantisation step
            assert!(err < 1.0, "3-bit roundtrip error at {i}: {err}");
        }
    }

    #[test]
    fn roundtrip_2bit() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
        let packed = compress_2bit(&input);
        let mut out = vec![0.0f32; 64];
        decompress_2bit(&packed, 64, &mut out);
        for i in 0..64 {
            let err = (input[i] - out[i]).abs();
            assert!(err < 1.0, "2-bit roundtrip error at {i}: {err}");
        }
    }

    #[test]
    fn layer_cache_store_retrieve() {
        let mut cache = LayerKVCache::new(128, 256, 0);
        let key: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
        let value: Vec<f32> = (0..256).map(|i| -(i as f32) * 0.01).collect();

        cache.store(0, &key, &value);

        let mut k_out = vec![0.0f32; 256];
        let mut v_out = vec![0.0f32; 256];
        assert!(cache.get_key(0, &mut k_out));
        assert!(cache.get_value(0, &mut v_out));

        // Verify approximate reconstruction (lossy compression)
        let k_err: f32 = key.iter().zip(k_out.iter()).map(|(a, b)| (a - b).abs()).sum::<f32>() / 256.0;
        assert!(k_err < 0.5, "Average key error too high: {k_err}");
    }

    #[test]
    fn sliding_window_range() {
        let cache = LayerKVCache::new(1024, 64, 512);
        assert_eq!(cache.attention_range(10), (0, 10));
        assert_eq!(cache.attention_range(600), (89, 600));

        let full = LayerKVCache::new(1024, 64, 0);
        assert_eq!(full.attention_range(600), (0, 600));
    }

    #[test]
    fn empty_position_returns_false() {
        let cache = LayerKVCache::new(64, 32, 0);
        let mut out = vec![0.0f32; 32];
        assert!(!cache.get_key(0, &mut out));
    }
}

use alloc::vec::Vec;
use libm::expf;

/// Simple PRNG (xorshift64).
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng {
            state: if seed == 0 { 0xdeadbeef } else { seed },
        }
    }

    fn next_u32(&mut self) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state & 0xFFFFFFFF) as u32
    }

    /// Random f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16777216.0
    }
}

/// Token sampler with temperature, top-k, top-p (nucleus), min-p,
/// and repetition penalty.
pub struct Sampler {
    #[allow(dead_code)]
    vocab_size: usize,
    temperature: f32,
    topp: f32,
    topk: usize,
    min_p: f32,
    rep_penalty: f32,
    rng: Rng,
}

impl Sampler {
    pub fn new(vocab_size: usize, temperature: f32, topp: f32, seed: u64) -> Self {
        Sampler {
            vocab_size,
            temperature,
            topp,
            topk: 0,
            min_p: 0.0,
            rep_penalty: 1.0,
            rng: Rng::new(seed),
        }
    }

    /// Extended constructor with all sampling parameters.
    pub fn new_full(
        vocab_size: usize,
        temperature: f32,
        topp: f32,
        topk: usize,
        min_p: f32,
        rep_penalty: f32,
        seed: u64,
    ) -> Self {
        Sampler {
            vocab_size,
            temperature,
            topp,
            topk,
            min_p,
            rep_penalty,
            rng: Rng::new(seed),
        }
    }

    /// Apply repetition penalty to logits based on recent token history.
    /// Tokens that appear in `recent_tokens` have their logits divided (if
    /// positive) or multiplied (if negative) by `penalty`.
    pub fn apply_repetition_penalty(&self, logits: &mut [f32], recent_tokens: &[u32]) {
        if self.rep_penalty <= 1.0 {
            return;
        }
        for &tok in recent_tokens {
            let idx = tok as usize;
            if idx < logits.len() {
                if logits[idx] > 0.0 {
                    logits[idx] /= self.rep_penalty;
                } else {
                    logits[idx] *= self.rep_penalty;
                }
            }
        }
    }

    /// Sample a token from the logits distribution.
    /// Modifies logits in-place (temperature scaling + softmax).
    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        if self.temperature == 0.0 {
            return argmax(logits) as u32;
        }

        // Apply temperature
        let inv_temp = 1.0 / self.temperature;
        for v in logits.iter_mut() {
            *v *= inv_temp;
        }

        // Softmax
        softmax(logits);

        // Top-k: zero out everything below the k-th highest probability
        if self.topk > 0 && self.topk < logits.len() {
            apply_top_k(logits, self.topk);
        }

        // Min-p: zero out tokens below min_p * max_prob
        if self.min_p > 0.0 && self.min_p < 1.0 {
            apply_min_p(logits, self.min_p);
        }

        // Top-p (nucleus) or multinomial sampling
        if self.topp > 0.0 && self.topp < 1.0 {
            self.sample_topp(logits) as u32
        } else {
            self.sample_multinomial(logits) as u32
        }
    }

    fn sample_multinomial(&mut self, probs: &[f32]) -> usize {
        let coin = self.rng.next_f32();
        let mut cumulative = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if cumulative > coin {
                return i;
            }
        }
        probs.len() - 1
    }

    fn sample_topp(&mut self, probs: &[f32]) -> usize {
        // Collect (index, prob) pairs and sort by probability descending
        let mut indices: Vec<(usize, f32)> = probs
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.0)
            .map(|(i, &p)| (i, p))
            .collect();

        // Sort descending by probability
        indices.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

        // Find cutoff where cumulative probability exceeds topp
        let mut cumulative = 0.0f32;
        let mut cutoff_idx = indices.len();
        for (i, &(_, p)) in indices.iter().enumerate() {
            cumulative += p;
            if cumulative > self.topp {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Renormalize the kept probabilities
        let kept = &indices[..cutoff_idx];
        let sum: f32 = kept.iter().map(|&(_, p)| p).sum();
        let inv_sum = 1.0 / sum;

        // Sample from the kept distribution
        let coin = self.rng.next_f32();
        let mut cum = 0.0f32;
        for &(idx, p) in kept {
            cum += p * inv_sum;
            if cum > coin {
                return idx;
            }
        }

        kept.last().map(|&(idx, _)| idx).unwrap_or(0)
    }
}

fn argmax(x: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = x[0];
    for (i, &v) in x.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

fn softmax(x: &mut [f32]) {
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
    let inv_sum = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

/// Top-k: keep only the `k` highest-probability tokens, zero out the rest.
fn apply_top_k(probs: &mut [f32], k: usize) {
    // Find the k-th largest probability using partial selection.
    // For large vocabs (262k), a full sort is expensive. Instead we find the
    // k-th threshold by scanning: collect top-k values, then threshold.
    let n = probs.len();
    if k >= n {
        return;
    }

    // Collect (prob, index) for top-k values
    let mut top: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
    for (i, &p) in probs.iter().enumerate() {
        if top.len() < k {
            top.push((p, i));
            if top.len() == k {
                // Sort ascending so top[0] is the smallest of the top-k
                top.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
            }
        } else if p > top[0].0 {
            top[0] = (p, i);
            // Re-sort to maintain heap property (insertion sort for small k)
            top.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
        }
    }

    let threshold = if top.is_empty() { 0.0 } else { top[0].0 };

    // Zero out everything below threshold and renormalise
    let mut sum = 0.0f32;
    for p in probs.iter_mut() {
        if *p < threshold {
            *p = 0.0;
        } else {
            sum += *p;
        }
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for p in probs.iter_mut() {
            *p *= inv;
        }
    }
}

/// Min-p: zero out tokens with probability < `min_p * max_probability`.
fn apply_min_p(probs: &mut [f32], min_p: f32) {
    let max_prob = probs.iter().copied().fold(0.0f32, f32::max);
    let threshold = min_p * max_prob;

    let mut sum = 0.0f32;
    for p in probs.iter_mut() {
        if *p < threshold {
            *p = 0.0;
        } else {
            sum += *p;
        }
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for p in probs.iter_mut() {
            *p *= inv;
        }
    }
}

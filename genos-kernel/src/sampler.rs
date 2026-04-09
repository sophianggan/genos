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

/// Token sampler with temperature and top-p (nucleus) sampling.
pub struct Sampler {
    #[allow(dead_code)]
    vocab_size: usize,
    temperature: f32,
    topp: f32,
    rng: Rng,
}

impl Sampler {
    pub fn new(vocab_size: usize, temperature: f32, topp: f32, seed: u64) -> Self {
        Sampler {
            vocab_size,
            temperature,
            topp,
            rng: Rng::new(seed),
        }
    }

    /// Sample a token from the logits distribution.
    /// Modifies logits in-place (temperature scaling + softmax).
    pub fn sample(&mut self, logits: &mut [f32]) -> usize {
        if self.temperature == 0.0 {
            return argmax(logits);
        }

        // Apply temperature
        let inv_temp = 1.0 / self.temperature;
        for v in logits.iter_mut() {
            *v *= inv_temp;
        }

        // Softmax
        softmax(logits);

        // Top-p sampling
        if self.topp > 0.0 && self.topp < 1.0 {
            self.sample_topp(logits)
        } else {
            self.sample_multinomial(logits)
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

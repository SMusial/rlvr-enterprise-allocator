//! Poisson arrival sampler — work order spikes (Ch01)
use rand::Rng;
use crate::rng::seeded_rng;

pub fn sample_arrivals(lambda: f64, n_steps: usize, seed: u64) -> Vec<usize> {
    let mut rng = seeded_rng(seed);
    (0..n_steps).map(|_| {
        // Knuth algorithm for Poisson
        let l = (-lambda).exp();
        let mut k = 0usize;
        let mut p = 1.0f64;
        loop { p *= rng.gen::<f64>(); if p <= l { break; } k += 1; }
        k
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mean_close_to_lambda() {
        let arrivals = sample_arrivals(3.0, 10_000, 42);
        let mean = arrivals.iter().sum::<usize>() as f64 / 10_000.0;
        assert!((mean - 3.0).abs() < 0.1, "mean={mean}");
    }
}

//! Seeded RNG — determinism guarantee for all chapters
use rand::{SeedableRng, rngs::StdRng};

pub fn seeded_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    #[test]
    fn same_seed_same_sequence() {
        let mut r1 = seeded_rng(42);
        let mut r2 = seeded_rng(42);
        for _ in 0..100 {
            assert_eq!(r1.gen::<u64>(), r2.gen::<u64>());
        }
    }
    #[test]
    fn different_seeds_differ() {
        let mut r1 = seeded_rng(1);
        let mut r2 = seeded_rng(2);
        assert_ne!(r1.gen::<u64>(), r2.gen::<u64>());
    }
}

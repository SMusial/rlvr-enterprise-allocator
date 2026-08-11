//! 2D Gaussian spatial sampler — technician GPS positions (Ch01)
use rand_distr::{Normal, Distribution};
use crate::rng::seeded_rng;

pub fn sample_positions(center_lat: f64, center_lon: f64,
                         std_deg: f64, n: usize, seed: u64) -> Vec<(f64,f64)> {
    let mut rng = seeded_rng(seed);
    let dist_lat = Normal::new(center_lat, std_deg).unwrap();
    let dist_lon = Normal::new(center_lon, std_deg).unwrap();
    (0..n).map(|_| (dist_lat.sample(&mut rng), dist_lon.sample(&mut rng))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn positions_near_center() {
        let pos = sample_positions(52.23, 21.01, 0.1, 100, 42);
        for (lat, lon) in &pos {
            assert!((lat - 52.23).abs() < 0.5);
            assert!((lon - 21.01).abs() < 0.5);
        }
    }
}

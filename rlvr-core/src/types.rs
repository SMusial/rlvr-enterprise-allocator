use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MdpParams {
    pub gamma:   f64,
    pub alpha:   f64,
    pub epsilon: f64,
    pub seed:    u64,
}

impl Default for MdpParams {
    fn default() -> Self {
        Self { gamma: 0.95, alpha: 0.1, epsilon: 1.0, seed: 42 }
    }
}

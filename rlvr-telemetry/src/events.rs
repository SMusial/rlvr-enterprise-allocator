use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TelemetryEvent {
    Reward   { step: usize, value: f64 },
    Return   { episode: usize, gt: f64 },
    Epsilon  { episode: usize, value: f64 },
    Loss     { step: usize, value: f64 },
    Latency  { step: usize, ms: f64 },
}

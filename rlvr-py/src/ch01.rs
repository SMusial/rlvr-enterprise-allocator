//! PyO3 bindings for Chapter 01 — ASP Task Dispatch
//!
//! Python call:
//!   rlvr_py.run_ch01_episode(seed, n_tech, n_task, epsilon, gamma) -> JSON str

use pyo3::prelude::*;
use rlvr_core::ch01_asp_dispatch::{AspConfig, run_episode};

/// Run one Ch01 episode. Returns JSON array of flat step objects.
///
/// Args:
///   seed    — deterministic seed (same seed = same episode)
///   n_tech  — number of technicians (2-8)
///   n_task  — number of work orders (3-20)
///   epsilon — exploration rate 0.0-1.0
///   gamma   — discount factor 0.0-1.0
///
/// Returns: JSON string — array of FlatStep objects
#[pyfunction]
#[pyo3(signature = (seed=42, n_tech=5, n_task=10, epsilon=0.8, gamma=0.95))]
pub fn run_ch01_episode(
    seed:    u64,
    n_tech:  usize,
    n_task:  usize,
    epsilon: f64,
    gamma:   f64,
) -> PyResult<String> {
    let config = AspConfig {
        n_technicians: n_tech,
        n_work_orders: n_task,
        skill_levels:  5,
        seed,
        epsilon,
        gamma,
    };
    let record = run_episode(config);
    serde_json::to_string(&record.steps)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

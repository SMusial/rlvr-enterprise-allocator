//! Safety invariant harness — one function per chapter invariant
//! These are the TDD red→green targets for every chapter.

/// Ch01 — Q-table must start at zeros (untrained)
pub fn ch01_q_table_zero(q: &ndarray::Array2<f64>) -> bool {
    q.iter().all(|&v| v == 0.0)
}

/// Ch02 — Transition probabilities must sum to 1 ± 1e-6
pub fn ch02_prob_conservation(row: &[f64]) -> bool {
    (row.iter().sum::<f64>() - 1.0).abs() < 1e-6
}

/// Ch03 — Every ASP must have selection probability ≥ 0.05
pub fn ch03_min_selection_prob(probs: &[f64]) -> bool {
    probs.iter().all(|&p| p >= 0.05)
}

/// Ch04 — Bellman contraction: ‖V_{k+1}−V_k‖ < γ·‖V_k−V_{k-1}‖
pub fn ch04_bellman_contraction(v_prev: f64, v_curr: f64, v_next: f64, gamma: f64) -> bool {
    let delta_next = (v_next - v_curr).abs();
    let delta_curr = (v_curr - v_prev).abs();
    if delta_curr < 1e-10 { return true; }
    delta_next < gamma * delta_curr + 1e-8
}

/// Ch06 — Q-values bounded by R_max/(1-γ)
pub fn ch06_q_bound(q_max: f64, r_max: f64, gamma: f64) -> bool {
    q_max <= r_max / (1.0 - gamma) + 1e-6
}

/// Ch08 — Eligibility trace bounded by 1/(1-γλ)
pub fn ch08_trace_bound(e_max: f64, gamma: f64, lambda: f64) -> bool {
    e_max <= 1.0 / (1.0 - gamma * lambda) + 1e-6
}

/// Ch09 — Policy must lie on probability simplex
pub fn ch09_simplex(policy: &[f64]) -> bool {
    let sum_ok = (policy.iter().sum::<f64>() - 1.0).abs() < 1e-6;
    let non_neg = policy.iter().all(|&p| p >= 0.0);
    sum_ok && non_neg
}

/// Ch12 — Nash regret ≤ 1e-6
pub fn ch12_nash_regret(regret: f64) -> bool { regret <= 1e-6 }

/// Ch15 — Inference latency ≤ 2 ms
pub fn ch15_latency_ms(tau_ms: f64) -> bool { tau_ms <= 2.0 }

/// Ch18 — Mixer weights must be non-negative
pub fn ch18_mixer_nonneg(w: &[f64]) -> bool { w.iter().all(|&v| v >= 0.0) }

/// Ch19 — Privacy loss ε ≤ budget
pub fn ch19_privacy_budget(epsilon: f64, budget: f64) -> bool { epsilon <= budget }

/// Ch20 — Zero unsafe pointer dereferences across FFI
pub fn ch20_ffi_safe(unsafe_count: usize) -> bool { unsafe_count == 0 }

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test] fn t_ch01() { assert!(ch01_q_table_zero(&Array2::zeros((5,5)))); }
    #[test] fn t_ch02() { assert!(ch02_prob_conservation(&[0.3,0.3,0.4])); }
    #[test] fn t_ch03() { assert!(ch03_min_selection_prob(&[0.1,0.5,0.4])); }
    #[test] fn t_ch03_fail() { assert!(!ch03_min_selection_prob(&[0.01,0.98,0.01])); }
    #[test] fn t_ch04() { assert!(ch04_bellman_contraction(2.0,1.5,1.2,0.95)); }
    #[test] fn t_ch06() { assert!(ch06_q_bound(18.0, 10.0, 0.5)); }
    #[test] fn t_ch08() { assert!(ch08_trace_bound(5.0, 0.9, 0.8)); }
    #[test] fn t_ch09() { assert!(ch09_simplex(&[0.2,0.5,0.3])); }
    #[test] fn t_ch09_fail() { assert!(!ch09_simplex(&[0.2,0.5,0.4])); }
    #[test] fn t_ch12() { assert!(ch12_nash_regret(0.0)); }
    #[test] fn t_ch15() { assert!(ch15_latency_ms(1.5)); }
    #[test] fn t_ch18() { assert!(ch18_mixer_nonneg(&[0.0,1.0,2.0])); }
    #[test] fn t_ch19() { assert!(ch19_privacy_budget(0.5, 1.0)); }
    #[test] fn t_ch20() { assert!(ch20_ffi_safe(0)); }
}

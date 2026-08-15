//! Ch08 — Eligibility Traces & TD(λ)
//! SARSA(λ) · Q(λ) Watkins · λ=0 baseline · λ=0.99 baseline
//! Backward-view implementation — online, per-step updates.
//! Sparse HashMap traces, auto-pruned at 1e-8.

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

use crate::ch02_bellman::{
    build_asp_transitions, build_asp_rewards, verify_transition_matrix,
    N_STATES, N_ACTIONS,
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct EligibilityConfig {
    pub seed:          u64,
    pub gamma:         f64,
    pub alpha:         f64,
    pub epsilon:       f64,
    pub epsilon_decay: f64,
    pub n_episodes:    usize,
    pub lambda:        f64,   // trace decay λ ∈ [0, 1]
    pub replacing:     bool,  // true = replacing traces, false = accumulating
}

#[derive(Debug, Clone)]
pub struct EligibilityResult {
    pub algorithm:         String,
    pub values:            Vec<f64>,
    pub policy:            Vec<usize>,
    pub q_table:           Vec<Vec<f64>>,
    pub returns_curve:     Vec<f64>,
    pub td_error_curve:    Vec<f64>,
    pub convergence_curve: Vec<f64>,
    pub n_episodes:        usize,
    pub total_steps:       usize,
    pub trace_stats:       Vec<f64>, // max active traces per episode
}

#[derive(Debug, Clone)]
pub struct Ch08Result {
    pub sarsa_lambda: EligibilityResult,
    pub q_lambda:     EligibilityResult,
    pub sarsa_td0:    EligibilityResult, // λ=0  baseline ≡ TD(0)
    pub sarsa_mc:     EligibilityResult, // λ=0.99 baseline ≈ MC
}

// ---------------------------------------------------------------------------
// Shared helpers  (mirror ch07_nstep.rs style exactly)
// ---------------------------------------------------------------------------

fn sample_next_state(
    s: usize, a: usize,
    transitions: &ndarray::Array3<f64>,
    rng: &mut StdRng,
) -> usize {
    let p: f64 = rng.gen();
    let mut cumsum = 0.0_f64;
    for sp in 0..N_STATES {
        cumsum += transitions[[s, a, sp]];
        if p <= cumsum { return sp; }
    }
    N_STATES - 1
}

fn epsilon_greedy(q: &[Vec<f64>], s: usize, epsilon: f64, rng: &mut StdRng) -> usize {
    if rng.gen::<f64>() < epsilon {
        rng.gen_range(0..N_ACTIONS)
    } else {
        q[s].iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0)
    }
}

fn greedy_action(q: &[Vec<f64>], s: usize) -> usize {
    q[s].iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i).unwrap_or(0)
}

fn v_from_q(q: &[Vec<f64>]) -> Vec<f64> {
    q.iter().map(|row| row.iter().cloned().fold(f64::NEG_INFINITY, f64::max)).collect()
}

fn greedy_policy(q: &[Vec<f64>]) -> Vec<usize> {
    q.iter().map(|row| {
        row.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0)
    }).collect()
}

// ---------------------------------------------------------------------------
// SARSA(λ) — on-policy TD(λ) with eligibility traces
//
// Backward view:
//   δ_t = R_{t+1} + γ Q(S_{t+1},A_{t+1}) − Q(S_t,A_t)
//   e_t(s,a) = γλ e_{t-1}(s,a) + 𝟙[s=S_t, a=A_t]   (replacing: set to 1)
//   Q(s,a)  ← Q(s,a) + α δ_t e_t(s,a)   for all (s,a)
// ---------------------------------------------------------------------------
pub fn sarsa_lambda(
    config: &EligibilityConfig,
    transitions: &ndarray::Array3<f64>,
    rewards: &Array2<f64>,
) -> EligibilityResult {
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut q: Vec<Vec<f64>> = vec![vec![0.0_f64; N_ACTIONS]; N_STATES];
    let mut et: HashMap<(usize, usize), f64> = HashMap::new();

    let mut returns_curve     = Vec::new();
    let mut td_error_curve    = Vec::new();
    let mut convergence_curve = Vec::new();
    let mut trace_stats       = Vec::new();
    let mut v_prev            = v_from_q(&q);
    let mut total_steps       = 0usize;
    let threshold             = 1e-8_f64;

    for ep in 0..config.n_episodes {
        let epsilon_t = (config.epsilon
            / (1.0 + config.epsilon_decay * ep as f64))
            .max(0.01);

        et.clear(); // reset traces at episode start

        let mut s = rng.gen_range(0..N_STATES);
        let mut a = epsilon_greedy(&q, s, epsilon_t, &mut rng);

        let mut ep_return  = 0.0_f64;
        let mut ep_td_err  = 0.0_f64;
        let mut steps      = 0usize;
        let mut gamma_pow  = 1.0_f64;
        let mut max_traces = 0usize;

        loop {
            let r  = rewards[[s, a]];
            let sp = sample_next_state(s, a, transitions, &mut rng);
            let ap = epsilon_greedy(&q, sp, epsilon_t, &mut rng);

            let done   = sp == 0 || sp == 7 || steps >= 50;
            let q_next = if done { 0.0 } else { q[sp][ap] };

            // TD error δ
            let delta = r + config.gamma * q_next - q[s][a];

            // Update trace for (s, a)
            if config.replacing {
                et.insert((s, a), 1.0);                    // replacing
            } else {
                *et.entry((s, a)).or_insert(0.0) += 1.0;  // accumulating
            }

            // Update ALL active (s,a) pairs proportional to their trace
            for (&(qs, qa), e_val) in et.iter_mut() {
                q[qs][qa] += config.alpha * delta * *e_val;
                *e_val    *= config.gamma * config.lambda;
            }
            // Prune near-zero traces
            et.retain(|_, v| *v > threshold);

            max_traces  = max_traces.max(et.len());
            ep_return  += gamma_pow * r;
            ep_td_err  += delta.abs();
            gamma_pow  *= config.gamma;
            total_steps += 1;
            steps       += 1;

            if done { break; }
            s = sp;
            a = ap;
        }

        returns_curve.push(ep_return);
        td_error_curve.push(ep_td_err / steps.max(1) as f64);
        trace_stats.push(max_traces as f64);

        let v     = v_from_q(&q);
        let delta = v.iter().zip(v_prev.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
        convergence_curve.push(delta);
        v_prev = v;
    }

    let label = if config.replacing { "replacing" } else { "accumulating" };
    EligibilityResult {
        algorithm: format!("sarsa_lambda_{:.2}_{}", config.lambda, label),
        values:    v_from_q(&q),
        policy:    greedy_policy(&q),
        q_table:   q,
        returns_curve,
        td_error_curve,
        convergence_curve,
        n_episodes: config.n_episodes,
        total_steps,
        trace_stats,
    }
}

// ---------------------------------------------------------------------------
// Q(λ) — off-policy TD(λ) with Watkins' trace cutting
//
// Same as SARSA(λ) but:
//   - TD target uses greedy next action: max_a Q(S',a)
//   - Traces CUT (cleared) when a non-greedy action is taken
//     → prevents off-policy contamination (Watkins, 1989)
// ---------------------------------------------------------------------------
pub fn q_lambda(
    config: &EligibilityConfig,
    transitions: &ndarray::Array3<f64>,
    rewards: &Array2<f64>,
) -> EligibilityResult {
    let mut rng = StdRng::seed_from_u64(config.seed + 10);
    let mut q: Vec<Vec<f64>> = vec![vec![0.0_f64; N_ACTIONS]; N_STATES];
    let mut et: HashMap<(usize, usize), f64> = HashMap::new();

    let mut returns_curve     = Vec::new();
    let mut td_error_curve    = Vec::new();
    let mut convergence_curve = Vec::new();
    let mut trace_stats       = Vec::new();
    let mut v_prev            = v_from_q(&q);
    let mut total_steps       = 0usize;
    let threshold             = 1e-8_f64;

    for ep in 0..config.n_episodes {
        let epsilon_t = (config.epsilon
            / (1.0 + config.epsilon_decay * ep as f64))
            .max(0.01);

        et.clear();

        let mut s = rng.gen_range(0..N_STATES);
        let mut ep_return  = 0.0_f64;
        let mut ep_td_err  = 0.0_f64;
        let mut steps      = 0usize;
        let mut gamma_pow  = 1.0_f64;
        let mut max_traces = 0usize;

        loop {
            let a      = epsilon_greedy(&q, s, epsilon_t, &mut rng);
            let r      = rewards[[s, a]];
            let sp     = sample_next_state(s, a, transitions, &mut rng);
            let done   = sp == 0 || sp == 7 || steps >= 50;

            // Off-policy: greedy next action for TD target
            let a_star = greedy_action(&q, sp);
            let q_next = if done { 0.0 } else { q[sp][a_star] };
            let delta  = r + config.gamma * q_next - q[s][a];

            // Update trace for (s, a) — replacing
            et.insert((s, a), 1.0);

            // Update ALL active (s,a) pairs
            for (&(qs, qa), e_val) in et.iter_mut() {
                q[qs][qa] += config.alpha * delta * *e_val;
                *e_val    *= config.gamma * config.lambda;
            }
            et.retain(|_, v| *v > threshold);

            // Watkins' cut: reset all traces if action was non-greedy
            if a != a_star {
                et.clear();
            }

            max_traces  = max_traces.max(et.len());
            ep_return  += gamma_pow * r;
            ep_td_err  += delta.abs();
            gamma_pow  *= config.gamma;
            total_steps += 1;
            steps       += 1;

            if done { break; }
            s = sp;
        }

        returns_curve.push(ep_return);
        td_error_curve.push(ep_td_err / steps.max(1) as f64);
        trace_stats.push(max_traces as f64);

        let v     = v_from_q(&q);
        let delta = v.iter().zip(v_prev.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
        convergence_curve.push(delta);
        v_prev = v;
    }

    EligibilityResult {
        algorithm: format!("q_lambda_{:.2}_watkins", config.lambda),
        values:    v_from_q(&q),
        policy:    greedy_policy(&q),
        q_table:   q,
        returns_curve,
        td_error_curve,
        convergence_curve,
        n_episodes: config.n_episodes,
        total_steps,
        trace_stats,
    }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------
pub fn run_ch08(config: EligibilityConfig) -> Ch08Result {
    let mut rng     = StdRng::seed_from_u64(config.seed);
    let transitions = build_asp_transitions(&mut rng);
    let rewards     = build_asp_rewards();
    verify_transition_matrix(&transitions)
        .expect("Transition matrix probability conservation violated");

    let sarsa_lambda_r = sarsa_lambda(&config, &transitions, &rewards);
    let q_lambda_r     = q_lambda(&config, &transitions, &rewards);

    // λ=0 baseline ≡ TD(0) / SARSA
    let td0_config = EligibilityConfig { lambda: 0.0, ..config.clone() };
    let sarsa_td0  = sarsa_lambda(&td0_config, &transitions, &rewards);

    // λ=0.99 baseline ≈ Monte Carlo
    let mc_config  = EligibilityConfig { lambda: 0.99, ..config.clone() };
    let sarsa_mc   = sarsa_lambda(&mc_config, &transitions, &rewards);

    Ch08Result { sarsa_lambda: sarsa_lambda_r, q_lambda: q_lambda_r, sarsa_td0, sarsa_mc }
}

// ---------------------------------------------------------------------------
// Tests  (inline, matching ch07 naming convention: test_*)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> EligibilityConfig {
        EligibilityConfig {
            seed: 42, gamma: 0.95, alpha: 0.1,
            epsilon: 0.3, epsilon_decay: 0.01,
            n_episodes: 500, lambda: 0.7, replacing: true,
        }
    }

    // ── Smoke ────────────────────────────────────────────────────────────

    #[test]
    fn test_sarsa_lambda_runs() {
        let r = run_ch08(default_config());
        assert_eq!(r.sarsa_lambda.values.len(), N_STATES);
        assert!(r.sarsa_lambda.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_q_lambda_runs() {
        let r = run_ch08(default_config());
        assert_eq!(r.q_lambda.values.len(), N_STATES);
        assert!(r.q_lambda.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_baselines_run() {
        let r = run_ch08(default_config());
        assert_eq!(r.sarsa_td0.values.len(), N_STATES);
        assert_eq!(r.sarsa_mc.values.len(),  N_STATES);
    }

    // ── Shape / validity ─────────────────────────────────────────────────

    #[test]
    fn test_q_table_shape() {
        let r = run_ch08(default_config());
        assert_eq!(r.sarsa_lambda.q_table.len(), N_STATES);
        for row in &r.sarsa_lambda.q_table { assert_eq!(row.len(), N_ACTIONS); }
        assert_eq!(r.q_lambda.q_table.len(), N_STATES);
        for row in &r.q_lambda.q_table     { assert_eq!(row.len(), N_ACTIONS); }
    }

    #[test]
    fn test_policy_valid() {
        let r = run_ch08(default_config());
        for &a in &r.sarsa_lambda.policy { assert!(a < N_ACTIONS); }
        for &a in &r.q_lambda.policy     { assert!(a < N_ACTIONS); }
        for &a in &r.sarsa_td0.policy    { assert!(a < N_ACTIONS); }
        for &a in &r.sarsa_mc.policy     { assert!(a < N_ACTIONS); }
    }

    #[test]
    fn test_curves_length() {
        let config = default_config();
        let r = run_ch08(config.clone());
        assert_eq!(r.sarsa_lambda.returns_curve.len(),  config.n_episodes);
        assert_eq!(r.q_lambda.returns_curve.len(),      config.n_episodes);
        assert_eq!(r.sarsa_lambda.td_error_curve.len(), config.n_episodes);
        assert_eq!(r.sarsa_lambda.trace_stats.len(),    config.n_episodes);
    }

    #[test]
    fn test_all_values_finite() {
        let r = run_ch08(default_config());
        for v in &r.sarsa_lambda.returns_curve { assert!(v.is_finite()); }
        for v in &r.q_lambda.returns_curve     { assert!(v.is_finite()); }
        for v in &r.sarsa_td0.returns_curve    { assert!(v.is_finite()); }
        for v in &r.sarsa_mc.returns_curve     { assert!(v.is_finite()); }
    }

    // ── Algorithm correctness ─────────────────────────────────────────────

    #[test]
    fn test_lambda_zero_no_traces() {
        // λ=0 → traces decay to 0 immediately → max active traces ≤ 1
        let config = EligibilityConfig { lambda: 0.0, ..default_config() };
        let mut rng = StdRng::seed_from_u64(config.seed);
        let transitions = build_asp_transitions(&mut rng);
        let rewards     = build_asp_rewards();
        let r = sarsa_lambda(&config, &transitions, &rewards);
        for &ts in &r.trace_stats {
            assert!(ts <= 1.0,
                "With λ=0, max active traces should be ≤ 1, got {}", ts);
        }
    }

    #[test]
    fn test_trace_stats_positive_with_lambda() {
        // λ=0.7 → active traces > 0 during learning
        let r = run_ch08(default_config());
        let avg: f64 = r.sarsa_lambda.trace_stats.iter().sum::<f64>()
            / r.sarsa_lambda.trace_stats.len() as f64;
        assert!(avg > 0.0,
            "Expected active traces > 0 with λ=0.7, got {:.4}", avg);
    }

    #[test]
    fn test_replacing_vs_accumulating_differ() {
        let config_r = EligibilityConfig { replacing: true,  ..default_config() };
        let config_a = EligibilityConfig { replacing: false, ..default_config() };
        let mut rng = StdRng::seed_from_u64(42);
        let transitions = build_asp_transitions(&mut rng);
        let rewards     = build_asp_rewards();
        let r_rep = sarsa_lambda(&config_r, &transitions, &rewards);
        let r_acc = sarsa_lambda(&config_a, &transitions, &rewards);
        let any_diff = r_rep.q_table.iter().zip(r_acc.q_table.iter())
            .any(|(rr, ra)| rr.iter().zip(ra.iter()).any(|(a, b)| (a - b).abs() > 1e-9));
        assert!(any_diff,
            "Replacing and accumulating traces should produce different Q-tables");
    }

    #[test]
    fn test_sarsa_lambda_converges() {
        let config = EligibilityConfig { n_episodes: 1000, ..default_config() };
        let r = run_ch08(config);
        let early: f64 = r.sarsa_lambda.returns_curve[..100].iter().sum::<f64>() / 100.0;
        let late:  f64 = r.sarsa_lambda.returns_curve[900..].iter().sum::<f64>() / 100.0;
        assert!(late > early,
            "SARSA(λ) should improve: early={:.3} late={:.3}", early, late);
    }

    #[test]
    fn test_q_lambda_watkins_converges() {
        let config = EligibilityConfig { n_episodes: 1000, ..default_config() };
        let r = run_ch08(config);
        let late: f64 = r.q_lambda.returns_curve[900..].iter().sum::<f64>() / 100.0;
        assert!(late.is_finite() && late > -100.0,
            "Q(λ) should converge, got late avg={:.3}", late);
    }

    #[test]
    fn test_deterministic() {
        let r1 = run_ch08(default_config());
        let r2 = run_ch08(default_config());
        for (v1, v2) in r1.sarsa_lambda.values.iter().zip(r2.sarsa_lambda.values.iter()) {
            assert_eq!(v1.to_bits(), v2.to_bits(), "Results must be deterministic");
        }
    }

    #[test]
    fn test_worst_state_lower_value() {
        // S7 (SLA breach) should have lower value than S0 (all available)
        let r = run_ch08(default_config());
        assert!(r.sarsa_lambda.values[7] < r.sarsa_lambda.values[0],
            "S7 should have lower value than S0: {} vs {}",
            r.sarsa_lambda.values[7], r.sarsa_lambda.values[0]);
    }
}

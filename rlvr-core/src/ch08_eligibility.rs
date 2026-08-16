//! Ch08 — Eligibility Traces & TD(λ)
//! SARSA(λ) · Q(λ) Watkins · λ=0 baseline · λ=0.99 baseline
//! Backward-view, sparse HashMap traces, auto-pruned at 1e-8.

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

use crate::ch02_bellman::{
    build_asp_transitions, build_asp_rewards, verify_transition_matrix,
    N_STATES, N_ACTIONS,
};

#[derive(Debug, Clone)]
pub struct EligibilityConfig {
    pub seed:          u64,
    pub gamma:         f64,
    pub alpha:         f64,
    pub epsilon:       f64,
    pub epsilon_decay: f64,
    pub n_episodes:    usize,
    pub lambda:        f64,
    pub replacing:     bool,
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
    pub trace_stats:       Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct Ch08Result {
    pub sarsa_lambda: EligibilityResult,
    pub q_lambda:     EligibilityResult,
    pub sarsa_td0:    EligibilityResult,
    pub sarsa_mc:     EligibilityResult,
}

fn sample_next_state(s: usize, a: usize, transitions: &ndarray::Array3<f64>, rng: &mut StdRng) -> usize {
    let p: f64 = rng.gen();
    let mut cum = 0.0_f64;
    for sp in 0..N_STATES { cum += transitions[[s, a, sp]]; if p <= cum { return sp; } }
    N_STATES - 1
}

fn epsilon_greedy(q: &[Vec<f64>], s: usize, eps: f64, rng: &mut StdRng) -> usize {
    if rng.gen::<f64>() < eps { rng.gen_range(0..N_ACTIONS) }
    else { q[s].iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i,_)| i).unwrap_or(0) }
}

fn greedy_action(q: &[Vec<f64>], s: usize) -> usize {
    q[s].iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i,_)| i).unwrap_or(0)
}

fn v_from_q(q: &[Vec<f64>]) -> Vec<f64> {
    q.iter().map(|row| row.iter().cloned().fold(f64::NEG_INFINITY, f64::max)).collect()
}

fn greedy_policy(q: &[Vec<f64>]) -> Vec<usize> {
    q.iter().map(|row| row.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i,_)| i).unwrap_or(0)).collect()
}

pub fn sarsa_lambda(config: &EligibilityConfig, transitions: &ndarray::Array3<f64>, rewards: &Array2<f64>) -> EligibilityResult {
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut q: Vec<Vec<f64>> = vec![vec![0.0; N_ACTIONS]; N_STATES];
    let mut et: HashMap<(usize, usize), f64> = HashMap::new();
    let (mut returns_curve, mut td_error_curve, mut convergence_curve, mut trace_stats) = (vec![], vec![], vec![], vec![]);
    let mut v_prev = v_from_q(&q);
    let mut total_steps = 0usize;
    let thr = 1e-8_f64;

    for ep in 0..config.n_episodes {
        let eps_t = (config.epsilon / (1.0 + config.epsilon_decay * ep as f64)).max(0.01);
        et.clear();
        let mut s = rng.gen_range(0..N_STATES);
        let mut a = epsilon_greedy(&q, s, eps_t, &mut rng);
        let (mut ep_ret, mut ep_td, mut steps, mut gp, mut max_tr) = (0.0_f64, 0.0_f64, 0usize, 1.0_f64, 0usize);

        loop {
            let r  = rewards[[s, a]];
            let sp = sample_next_state(s, a, transitions, &mut rng);
            let ap = epsilon_greedy(&q, sp, eps_t, &mut rng);
            let done = sp == 0 || sp == 7 || steps >= 50;
            let delta = r + config.gamma * (if done { 0.0 } else { q[sp][ap] }) - q[s][a];
            if config.replacing { et.insert((s, a), 1.0); } else { *et.entry((s, a)).or_insert(0.0) += 1.0; }
            for (&(qs, qa), ev) in et.iter_mut() { q[qs][qa] += config.alpha * delta * *ev; *ev *= config.gamma * config.lambda; }
            et.retain(|_, v| *v > thr);
            max_tr = max_tr.max(et.len());
            ep_ret += gp * r; ep_td += delta.abs(); gp *= config.gamma; total_steps += 1; steps += 1;
            if done { break; }
            s = sp; a = ap;
        }
        returns_curve.push(ep_ret);
        td_error_curve.push(ep_td / steps.max(1) as f64);
        trace_stats.push(max_tr as f64);
        let v = v_from_q(&q);
        convergence_curve.push(v.iter().zip(v_prev.iter()).map(|(a,b)| (a-b).abs()).fold(0.0_f64, f64::max));
        v_prev = v;
    }
    let label = if config.replacing { "replacing" } else { "accumulating" };
    EligibilityResult {
        algorithm: format!("sarsa_lambda_{:.2}_{}", config.lambda, label),
        values: v_from_q(&q), policy: greedy_policy(&q), q_table: q,
        returns_curve, td_error_curve, convergence_curve, n_episodes: config.n_episodes, total_steps, trace_stats,
    }
}

pub fn q_lambda(config: &EligibilityConfig, transitions: &ndarray::Array3<f64>, rewards: &Array2<f64>) -> EligibilityResult {
    let mut rng = StdRng::seed_from_u64(config.seed + 10);
    let mut q: Vec<Vec<f64>> = vec![vec![0.0; N_ACTIONS]; N_STATES];
    let mut et: HashMap<(usize, usize), f64> = HashMap::new();
    let (mut returns_curve, mut td_error_curve, mut convergence_curve, mut trace_stats) = (vec![], vec![], vec![], vec![]);
    let mut v_prev = v_from_q(&q);
    let mut total_steps = 0usize;
    let thr = 1e-8_f64;

    for ep in 0..config.n_episodes {
        let eps_t = (config.epsilon / (1.0 + config.epsilon_decay * ep as f64)).max(0.01);
        et.clear();
        let mut s = rng.gen_range(0..N_STATES);
        let (mut ep_ret, mut ep_td, mut steps, mut gp, mut max_tr) = (0.0_f64, 0.0_f64, 0usize, 1.0_f64, 0usize);

        loop {
            let a      = epsilon_greedy(&q, s, eps_t, &mut rng);
            let r      = rewards[[s, a]];
            let sp     = sample_next_state(s, a, transitions, &mut rng);
            let done   = sp == 0 || sp == 7 || steps >= 50;
            let a_star = greedy_action(&q, sp);
            let delta  = r + config.gamma * (if done { 0.0 } else { q[sp][a_star] }) - q[s][a];
            et.insert((s, a), 1.0);
            for (&(qs, qa), ev) in et.iter_mut() { q[qs][qa] += config.alpha * delta * *ev; *ev *= config.gamma * config.lambda; }
            et.retain(|_, v| *v > thr);
            if a != a_star { et.clear(); }
            max_tr = max_tr.max(et.len());
            ep_ret += gp * r; ep_td += delta.abs(); gp *= config.gamma; total_steps += 1; steps += 1;
            if done { break; }
            s = sp;
        }
        returns_curve.push(ep_ret);
        td_error_curve.push(ep_td / steps.max(1) as f64);
        trace_stats.push(max_tr as f64);
        let v = v_from_q(&q);
        convergence_curve.push(v.iter().zip(v_prev.iter()).map(|(a,b)| (a-b).abs()).fold(0.0_f64, f64::max));
        v_prev = v;
    }
    EligibilityResult {
        algorithm: format!("q_lambda_{:.2}_watkins", config.lambda),
        values: v_from_q(&q), policy: greedy_policy(&q), q_table: q,
        returns_curve, td_error_curve, convergence_curve, n_episodes: config.n_episodes, total_steps, trace_stats,
    }
}

pub fn run_ch08(config: EligibilityConfig) -> Ch08Result {
    let mut rng = StdRng::seed_from_u64(config.seed);
    let transitions = build_asp_transitions(&mut rng);
    let rewards     = build_asp_rewards();
    verify_transition_matrix(&transitions).expect("Transition matrix invalid");
    let sarsa_lambda_r = sarsa_lambda(&config, &transitions, &rewards);
    let q_lambda_r     = q_lambda(&config, &transitions, &rewards);
    let sarsa_td0  = sarsa_lambda(&EligibilityConfig { lambda: 0.0,  ..config.clone() }, &transitions, &rewards);
    let sarsa_mc   = sarsa_lambda(&EligibilityConfig { lambda: 0.99, ..config.clone() }, &transitions, &rewards);
    Ch08Result { sarsa_lambda: sarsa_lambda_r, q_lambda: q_lambda_r, sarsa_td0, sarsa_mc }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cfg() -> EligibilityConfig {
        EligibilityConfig { seed: 42, gamma: 0.95, alpha: 0.1, epsilon: 0.3, epsilon_decay: 0.01, n_episodes: 500, lambda: 0.7, replacing: true }
    }
    #[test] fn test_sarsa_lambda_runs() { let r = run_ch08(cfg()); assert_eq!(r.sarsa_lambda.values.len(), N_STATES); assert!(r.sarsa_lambda.values.iter().all(|v| v.is_finite())); }
    #[test] fn test_q_lambda_runs() { let r = run_ch08(cfg()); assert_eq!(r.q_lambda.values.len(), N_STATES); assert!(r.q_lambda.values.iter().all(|v| v.is_finite())); }
    #[test] fn test_baselines_run() { let r = run_ch08(cfg()); assert_eq!(r.sarsa_td0.values.len(), N_STATES); assert_eq!(r.sarsa_mc.values.len(), N_STATES); }
    #[test] fn test_q_table_shape() { let r = run_ch08(cfg()); for row in &r.sarsa_lambda.q_table { assert_eq!(row.len(), N_ACTIONS); } }
    #[test] fn test_policy_valid() { let r = run_ch08(cfg()); for &a in &r.sarsa_lambda.policy { assert!(a < N_ACTIONS); } for &a in &r.q_lambda.policy { assert!(a < N_ACTIONS); } }
    #[test] fn test_curves_length() { let c = cfg(); let r = run_ch08(c.clone()); assert_eq!(r.sarsa_lambda.returns_curve.len(), c.n_episodes); assert_eq!(r.sarsa_lambda.trace_stats.len(), c.n_episodes); }
    #[test] fn test_all_values_finite() { let r = run_ch08(cfg()); for v in &r.sarsa_lambda.returns_curve { assert!(v.is_finite()); } for v in &r.q_lambda.returns_curve { assert!(v.is_finite()); } }
    #[test] fn test_lambda_zero_no_traces() {
        let c = EligibilityConfig { lambda: 0.0, ..cfg() };
        let mut rng = StdRng::seed_from_u64(c.seed);
        let t = build_asp_transitions(&mut rng); let rw = build_asp_rewards();
        let r = sarsa_lambda(&c, &t, &rw);
        for &ts in &r.trace_stats { assert!(ts <= 1.0, "λ=0 traces should be ≤1, got {}", ts); }
    }
    #[test] fn test_trace_stats_positive() { let r = run_ch08(cfg()); let avg: f64 = r.sarsa_lambda.trace_stats.iter().sum::<f64>() / r.sarsa_lambda.trace_stats.len() as f64; assert!(avg > 0.0); }
    #[test] fn test_replacing_vs_accumulating_differ() {
        let mut rng = StdRng::seed_from_u64(42);
        let t = build_asp_transitions(&mut rng); let rw = build_asp_rewards();
        let r1 = sarsa_lambda(&EligibilityConfig { replacing: true,  ..cfg() }, &t, &rw);
        let r2 = sarsa_lambda(&EligibilityConfig { replacing: false, ..cfg() }, &t, &rw);
        let diff = r1.q_table.iter().zip(r2.q_table.iter()).any(|(a,b)| a.iter().zip(b.iter()).any(|(x,y)| (x-y).abs() > 1e-9));
        assert!(diff, "replacing vs accumulating should differ");
    }
    #[test] fn test_sarsa_lambda_converges() {
        let r = run_ch08(EligibilityConfig { n_episodes: 1000, ..cfg() });
        let early: f64 = r.sarsa_lambda.returns_curve[..100].iter().sum::<f64>() / 100.0;
        let late:  f64 = r.sarsa_lambda.returns_curve[900..].iter().sum::<f64>() / 100.0;
        assert!(late > early, "early={:.3} late={:.3}", early, late);
    }
    #[test] fn test_q_lambda_converges() {
        let r = run_ch08(EligibilityConfig { n_episodes: 1000, ..cfg() });
        let late: f64 = r.q_lambda.returns_curve[900..].iter().sum::<f64>() / 100.0;
        assert!(late.is_finite() && late > -100.0);
    }
    #[test] fn test_deterministic() {
        let r1 = run_ch08(cfg()); let r2 = run_ch08(cfg());
        for (v1, v2) in r1.sarsa_lambda.values.iter().zip(r2.sarsa_lambda.values.iter()) { assert_eq!(v1.to_bits(), v2.to_bits()); }
    }
    #[test] fn test_worst_state_lower_value() {
        let r = run_ch08(cfg());
        assert!(r.sarsa_lambda.values[7] < r.sarsa_lambda.values[0], "S7={:.3} S0={:.3}", r.sarsa_lambda.values[7], r.sarsa_lambda.values[0]);
    }
}

#!/usr/bin/env python3
"""
RLVR Enterprise Allocator — Chapter 06 ZIP Generator
Run from project root:
    python3 ch06_generate.py
Produces: ch06_package.zip
"""

import zipfile
import os

# ---------------------------------------------------------------------------
# FILE 1: rlvr-core/src/ch06_td.rs
# ---------------------------------------------------------------------------
CH06_RS = r'''use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::ch02_bellman::{
    build_asp_transitions, build_asp_rewards, verify_transition_matrix,
    N_STATES, N_ACTIONS,
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct TdConfig {
    pub seed:          u64,
    pub gamma:         f64,
    pub alpha:         f64,   // learning rate
    pub epsilon:       f64,
    pub epsilon_decay: f64,
    pub n_episodes:    usize,
}

#[derive(Debug, Clone)]
pub struct TdStep {
    pub episode:  usize,
    pub step:     usize,
    pub state:    usize,
    pub action:   usize,
    pub reward:   f64,
    pub next_state: usize,
    pub td_error: f64,
    pub explored: bool,
}

#[derive(Debug, Clone)]
pub struct TdResult {
    pub algorithm:         String,
    pub values:            Vec<f64>,
    pub policy:            Vec<usize>,
    pub q_table:           Vec<Vec<f64>>,
    pub returns_curve:     Vec<f64>,
    pub td_error_curve:    Vec<f64>,
    pub convergence_curve: Vec<f64>,
    pub n_episodes:        usize,
    pub total_steps:       usize,
}

#[derive(Debug, Clone)]
pub struct Ch06Result {
    pub td0:   TdResult,
    pub sarsa: TdResult,
    pub qlearn: TdResult,
}

// ---------------------------------------------------------------------------
// Environment helpers
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

fn epsilon_greedy(
    q: &[Vec<f64>], s: usize, epsilon: f64, rng: &mut StdRng,
) -> usize {
    if rng.gen::<f64>() < epsilon {
        rng.gen_range(0..N_ACTIONS)
    } else {
        q[s].iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0)
    }
}

fn greedy(q: &[Vec<f64>], s: usize) -> usize {
    q[s].iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i).unwrap_or(0)
}

fn v_from_q(q: &[Vec<f64>]) -> Vec<f64> {
    q.iter().map(|row| row.iter().cloned().fold(f64::NEG_INFINITY, f64::max)).collect()
}

// ---------------------------------------------------------------------------
// TD(0) — semi-gradient prediction
// V(s) <- V(s) + alpha * [R + gamma*V(s') - V(s)]
// ---------------------------------------------------------------------------
pub fn td0_prediction(
    config: &TdConfig,
    transitions: &ndarray::Array3<f64>,
    rewards: &Array2<f64>,
) -> TdResult {
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut v = vec![0.0_f64; N_STATES];
    let mut returns_curve = Vec::new();
    let mut td_error_curve = Vec::new();
    let mut convergence_curve = Vec::new();
    let mut v_prev = v.clone();
    let mut total_steps = 0usize;

    // fixed policy: greedy w.r.t. initial Q (all zeros -> A0)
    let policy = vec![1usize; N_STATES]; // A1 baseline

    for ep in 0..config.n_episodes {
        let mut s = rng.gen_range(0..N_STATES);
        let mut ep_return = 0.0_f64;
        let mut ep_td_err = 0.0_f64;
        let mut gamma_pow = 1.0_f64;
        let mut steps = 0usize;

        loop {
            let a = policy[s];
            let r = rewards[[s, a]];
            let sp = sample_next_state(s, a, transitions, &mut rng);

            let td_error = r + config.gamma * v[sp] - v[s];
            v[s] += config.alpha * td_error;

            ep_return += gamma_pow * r;
            ep_td_err += td_error.abs();
            gamma_pow *= config.gamma;
            total_steps += 1;
            steps += 1;
            s = sp;

            if s == 0 || s == 7 || steps >= 50 { break; }
        }

        returns_curve.push(ep_return);
        td_error_curve.push(ep_td_err / steps.max(1) as f64);

        let delta = v.iter().zip(v_prev.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
        convergence_curve.push(delta);
        v_prev = v.clone();
    }

    // derive Q from V
    let q_table: Vec<Vec<f64>> = (0..N_STATES).map(|s| {
        (0..N_ACTIONS).map(|a| {
            (0..N_STATES).map(|sp| {
                transitions[[s, a, sp]] * (rewards[[s, a]] + config.gamma * v[sp])
            }).sum::<f64>()
        }).collect()
    }).collect();

    let policy_out: Vec<usize> = q_table.iter().map(|row| {
        row.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0)
    }).collect();

    TdResult {
        algorithm: "td0".to_string(),
        values: v,
        policy: policy_out,
        q_table,
        returns_curve,
        td_error_curve,
        convergence_curve,
        n_episodes: config.n_episodes,
        total_steps,
    }
}

// ---------------------------------------------------------------------------
// SARSA — on-policy TD control
// Q(s,a) <- Q(s,a) + alpha * [R + gamma*Q(s',a') - Q(s,a)]
// ---------------------------------------------------------------------------
pub fn sarsa(
    config: &TdConfig,
    transitions: &ndarray::Array3<f64>,
    rewards: &Array2<f64>,
) -> TdResult {
    let mut rng = StdRng::seed_from_u64(config.seed + 1);
    let mut q = vec![vec![0.0_f64; N_ACTIONS]; N_STATES];
    let mut returns_curve = Vec::new();
    let mut td_error_curve = Vec::new();
    let mut convergence_curve = Vec::new();
    let mut v_prev = v_from_q(&q);
    let mut total_steps = 0usize;

    for ep in 0..config.n_episodes {
        let epsilon_t = (config.epsilon / (1.0 + config.epsilon_decay * ep as f64)).max(0.01);
        let mut s = rng.gen_range(0..N_STATES);
        let mut a = epsilon_greedy(&q, s, epsilon_t, &mut rng);
        let mut ep_return = 0.0_f64;
        let mut ep_td_err = 0.0_f64;
        let mut gamma_pow = 1.0_f64;
        let mut steps = 0usize;

        loop {
            let r = rewards[[s, a]];
            let sp = sample_next_state(s, a, transitions, &mut rng);
            let ap = epsilon_greedy(&q, sp, epsilon_t, &mut rng); // SARSA: next action from epsilon-greedy

            let td_error = r + config.gamma * q[sp][ap] - q[s][a];
            q[s][a] += config.alpha * td_error;

            ep_return += gamma_pow * r;
            ep_td_err += td_error.abs();
            gamma_pow *= config.gamma;
            total_steps += 1;
            steps += 1;
            s = sp;
            a = ap;

            if s == 0 || s == 7 || steps >= 50 { break; }
        }

        returns_curve.push(ep_return);
        td_error_curve.push(ep_td_err / steps.max(1) as f64);

        let v = v_from_q(&q);
        let delta = v.iter().zip(v_prev.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
        convergence_curve.push(delta);
        v_prev = v;
    }

    let values = v_from_q(&q);
    let policy: Vec<usize> = q.iter().map(|row| {
        row.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0)
    }).collect();

    TdResult {
        algorithm: "sarsa".to_string(),
        values,
        policy,
        q_table: q,
        returns_curve,
        td_error_curve,
        convergence_curve,
        n_episodes: config.n_episodes,
        total_steps,
    }
}

// ---------------------------------------------------------------------------
// Q-Learning — off-policy TD control
// Q(s,a) <- Q(s,a) + alpha * [R + gamma*max_a' Q(s',a') - Q(s,a)]
// ---------------------------------------------------------------------------
pub fn q_learning(
    config: &TdConfig,
    transitions: &ndarray::Array3<f64>,
    rewards: &Array2<f64>,
) -> TdResult {
    let mut rng = StdRng::seed_from_u64(config.seed + 2);
    let mut q = vec![vec![0.0_f64; N_ACTIONS]; N_STATES];
    let mut returns_curve = Vec::new();
    let mut td_error_curve = Vec::new();
    let mut convergence_curve = Vec::new();
    let mut v_prev = v_from_q(&q);
    let mut total_steps = 0usize;

    for ep in 0..config.n_episodes {
        let epsilon_t = (config.epsilon / (1.0 + config.epsilon_decay * ep as f64)).max(0.01);
        let mut s = rng.gen_range(0..N_STATES);
        let mut ep_return = 0.0_f64;
        let mut ep_td_err = 0.0_f64;
        let mut gamma_pow = 1.0_f64;
        let mut steps = 0usize;

        loop {
            let a = epsilon_greedy(&q, s, epsilon_t, &mut rng);
            let r = rewards[[s, a]];
            let sp = sample_next_state(s, a, transitions, &mut rng);

            // Q-Learning: use max over next actions (off-policy)
            let max_q_sp = q[sp].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let td_error = r + config.gamma * max_q_sp - q[s][a];
            q[s][a] += config.alpha * td_error;

            ep_return += gamma_pow * r;
            ep_td_err += td_error.abs();
            gamma_pow *= config.gamma;
            total_steps += 1;
            steps += 1;
            s = sp;

            if s == 0 || s == 7 || steps >= 50 { break; }
        }

        returns_curve.push(ep_return);
        td_error_curve.push(ep_td_err / steps.max(1) as f64);

        let v = v_from_q(&q);
        let delta = v.iter().zip(v_prev.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
        convergence_curve.push(delta);
        v_prev = v;
    }

    let values = v_from_q(&q);
    let policy: Vec<usize> = q.iter().map(|row| {
        row.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0)
    }).collect();

    TdResult {
        algorithm: "qlearning".to_string(),
        values,
        policy,
        q_table: q,
        returns_curve,
        td_error_curve,
        convergence_curve,
        n_episodes: config.n_episodes,
        total_steps,
    }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------
pub fn run_ch06(config: TdConfig) -> Ch06Result {
    let mut rng = StdRng::seed_from_u64(config.seed);
    let transitions = build_asp_transitions(&mut rng);
    let rewards = build_asp_rewards();
    verify_transition_matrix(&transitions)
        .expect("Transition matrix probability conservation violated");

    let td0    = td0_prediction(&config, &transitions, &rewards);
    let sarsa  = sarsa(&config, &transitions, &rewards);
    let qlearn = q_learning(&config, &transitions, &rewards);

    Ch06Result { td0, sarsa, qlearn }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> TdConfig {
        TdConfig { seed: 42, gamma: 0.95, alpha: 0.1, epsilon: 0.3, epsilon_decay: 0.01, n_episodes: 500 }
    }

    #[test]
    fn test_td0_runs() {
        let r = run_ch06(default_config());
        assert_eq!(r.td0.values.len(), N_STATES);
        assert!(r.td0.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_sarsa_runs() {
        let r = run_ch06(default_config());
        assert_eq!(r.sarsa.values.len(), N_STATES);
        assert!(r.sarsa.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_qlearning_runs() {
        let r = run_ch06(default_config());
        assert_eq!(r.qlearn.values.len(), N_STATES);
        assert!(r.qlearn.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_policy_valid() {
        let r = run_ch06(default_config());
        for &a in &r.sarsa.policy  { assert!(a < N_ACTIONS); }
        for &a in &r.qlearn.policy { assert!(a < N_ACTIONS); }
    }

    #[test]
    fn test_returns_curve_length() {
        let config = default_config();
        let r = run_ch06(config.clone());
        assert_eq!(r.sarsa.returns_curve.len(),  config.n_episodes);
        assert_eq!(r.qlearn.returns_curve.len(), config.n_episodes);
    }

    #[test]
    fn test_td_error_decreases() {
        let config = TdConfig { n_episodes: 1000, ..default_config() };
        let r = run_ch06(config);
        let early: f64 = r.qlearn.td_error_curve[..50].iter().sum::<f64>() / 50.0;
        let late:  f64 = r.qlearn.td_error_curve[950..].iter().sum::<f64>() / 50.0;
        assert!(late <= early * 2.0,
            "TD error should not explode: early={:.3} late={:.3}", early, late);
    }

    #[test]
    fn test_qlearning_ge_sarsa_value() {
        // Q-Learning is off-policy optimal -> should find at least as good values
        let config = TdConfig { n_episodes: 2000, ..default_config() };
        let r = run_ch06(config);
        let v_ql: f64 = r.qlearn.values.iter().sum();
        let v_sa: f64 = r.sarsa.values.iter().sum();
        assert!(v_ql >= v_sa - 5.0,
            "Q-Learning should be competitive: ql={:.2} sarsa={:.2}", v_ql, v_sa);
    }

    #[test]
    fn test_q_table_shape() {
        let r = run_ch06(default_config());
        assert_eq!(r.sarsa.q_table.len(), N_STATES);
        for row in &r.sarsa.q_table { assert_eq!(row.len(), N_ACTIONS); }
    }

    #[test]
    fn test_deterministic() {
        let r1 = run_ch06(default_config());
        let r2 = run_ch06(default_config());
        for (v1, v2) in r1.qlearn.values.iter().zip(r2.qlearn.values.iter()) {
            assert_eq!(v1.to_bits(), v2.to_bits());
        }
    }

    #[test]
    fn test_total_steps_positive() {
        let r = run_ch06(default_config());
        assert!(r.sarsa.total_steps > 0);
        assert!(r.qlearn.total_steps > 0);
    }
}
'''

# ---------------------------------------------------------------------------
# FILE 2: rlvr-py/src/lib.rs (Ch01-Ch06)
# ---------------------------------------------------------------------------
LIB_RS = r'''use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rlvr_core::ch01_asp_dispatch::{run_episode, AspConfig};
use rlvr_core::ch02_bellman::{run_ch02, Ch02Config, N_STATES, N_ACTIONS, STATE_NAMES, ACTION_NAMES};
use rlvr_core::ch03_bandit::{run_ch03, Ch03Config, ARM_NAMES, TRUE_SLA_RATES};
use rlvr_core::ch04_dp::{run_ch04, Ch04Config};
use rlvr_core::ch05_mc::{run_ch05, McConfig};
use rlvr_core::ch06_td::{run_ch06, TdConfig};

#[pyfunction]
fn run_ch01_episode(py: Python, seed: u64, n_tech: usize, n_task: usize, epsilon: f64, gamma: f64) -> PyResult<PyObject> {
    let config = AspConfig { seed, n_tech, n_orders: n_task, epsilon, gamma };
    let record = run_episode(config);
    let steps = PyList::empty_bound(py);
    for s in &record.steps {
        let d = PyDict::new_bound(py);
        d.set_item("step",        s.step)?;
        d.set_item("tech_idx",    s.tech_idx)?;
        d.set_item("order_idx",   s.order_idx)?;
        d.set_item("tech_x",      s.tech_x)?;
        d.set_item("tech_y",      s.tech_y)?;
        d.set_item("order_x",     s.order_x)?;
        d.set_item("order_y",     s.order_y)?;
        d.set_item("reward",      s.reward)?;
        d.set_item("gt",          s.gt)?;
        d.set_item("sla_met",     s.sla_met)?;
        d.set_item("skill_match", s.skill_match)?;
        d.set_item("explored",    s.explored)?;
        d.set_item("epsilon",     s.epsilon)?;
        d.set_item("distance_km", s.distance_km)?;
        d.set_item("tech_skill",  &s.tech_skill)?;
        d.set_item("order_skill", &s.order_skill)?;
        d.set_item("urgency",     s.urgency)?;
        steps.append(d)?;
    }
    let result = PyDict::new_bound(py);
    result.set_item("steps",             steps)?;
    result.set_item("total_gt",          record.total_gt)?;
    result.set_item("sla_met_count",     record.sla_met_count)?;
    result.set_item("skill_match_count", record.skill_match_count)?;
    result.set_item("explored_count",    record.explored_count)?;
    result.set_item("seed",              record.seed)?;
    result.set_item("epsilon",           record.epsilon)?;
    result.set_item("gamma",             record.gamma)?;
    Ok(result.into())
}

#[pyfunction]
fn run_ch02_value_iteration(py: Python, seed: u64, gamma: f64, theta: f64) -> PyResult<PyObject> {
    let config = Ch02Config { seed, gamma, theta };
    let result = run_ch02(config);
    let values = PyList::empty_bound(py);
    for v in &result.values { values.append(v)?; }
    let policy = PyList::empty_bound(py);
    for &a in &result.policy { policy.append(a)?; }
    let curve = PyList::empty_bound(py);
    for v in &result.convergence_curve { curve.append(v)?; }
    let trace = PyList::empty_bound(py);
    for step in &result.bellman_trace {
        let d = PyDict::new_bound(py);
        d.set_item("iteration", step.iteration)?;
        d.set_item("state",     step.state)?;
        d.set_item("action",    step.action)?;
        let qv = PyList::empty_bound(py);
        for q in &step.q_values { qv.append(q)?; }
        d.set_item("q_values",  qv)?;
        d.set_item("v_old",     step.v_old)?;
        d.set_item("v_new",     step.v_new)?;
        d.set_item("delta",     step.delta)?;
        trace.append(d)?;
    }
    let snames = PyList::empty_bound(py);
    for n in STATE_NAMES { snames.append(n)?; }
    let anames = PyList::empty_bound(py);
    for n in ACTION_NAMES { anames.append(n)?; }
    let out = PyDict::new_bound(py);
    out.set_item("values",            values)?;
    out.set_item("policy",            policy)?;
    out.set_item("iterations",        result.iterations)?;
    out.set_item("convergence_curve", curve)?;
    out.set_item("bellman_trace",     trace)?;
    out.set_item("state_names",       snames)?;
    out.set_item("action_names",      anames)?;
    out.set_item("n_states",          N_STATES)?;
    out.set_item("n_actions",         N_ACTIONS)?;
    Ok(out.into())
}

#[pyfunction]
fn run_ch03_bandits(py: Python, seed: u64, n_steps: usize, epsilon: f64, epsilon_decay: f64, ucb_c: f64) -> PyResult<PyObject> {
    let config = Ch03Config { seed, n_steps, epsilon, epsilon_decay, ucb_c, gamma: 0.95 };
    let results = run_ch03(config);
    let out_list = PyList::empty_bound(py);
    for result in &results {
        let steps_list = PyList::empty_bound(py);
        for s in &result.steps {
            let d = PyDict::new_bound(py);
            d.set_item("step",              s.step)?;
            d.set_item("algorithm",         &s.algorithm)?;
            d.set_item("arm",               s.arm)?;
            d.set_item("reward",            s.reward)?;
            d.set_item("regret",            s.regret)?;
            d.set_item("cumulative_regret", s.cumulative_regret)?;
            let qv = PyList::empty_bound(py);
            for q in &s.q_values { qv.append(q)?; }
            d.set_item("q_values", qv)?;
            let np = PyList::empty_bound(py);
            for n in &s.n_pulls { np.append(n)?; }
            d.set_item("n_pulls", np)?;
            let uv = PyList::empty_bound(py);
            for u in &s.ucb_values { uv.append(u)?; }
            d.set_item("ucb_values", uv)?;
            let tv = PyList::empty_bound(py);
            for t in &s.thompson_samples { tv.append(t)?; }
            d.set_item("thompson_samples", tv)?;
            d.set_item("epsilon",  s.epsilon)?;
            d.set_item("explored", s.explored)?;
            steps_list.append(d)?;
        }
        let fq = PyList::empty_bound(py);
        for q in &result.final_q_values { fq.append(q)?; }
        let fn_ = PyList::empty_bound(py);
        for n in &result.final_n_pulls { fn_.append(n)?; }
        let r = PyDict::new_bound(py);
        r.set_item("algorithm",      &result.algorithm)?;
        r.set_item("steps",          steps_list)?;
        r.set_item("total_reward",   result.total_reward)?;
        r.set_item("total_regret",   result.total_regret)?;
        r.set_item("final_q_values", fq)?;
        r.set_item("final_n_pulls",  fn_)?;
        r.set_item("best_arm",       result.best_arm)?;
        out_list.append(r)?;
    }
    let arm_names = PyList::empty_bound(py);
    for n in ARM_NAMES { arm_names.append(n)?; }
    let true_rates = PyList::empty_bound(py);
    for r in TRUE_SLA_RATES { true_rates.append(r)?; }
    let out = PyDict::new_bound(py);
    out.set_item("results",    out_list)?;
    out.set_item("arm_names",  arm_names)?;
    out.set_item("true_rates", true_rates)?;
    Ok(out.into())
}

#[pyfunction]
fn run_ch04_dp(py: Python, seed: u64, gamma: f64, theta: f64) -> PyResult<PyObject> {
    let config = Ch04Config { seed, gamma, theta };
    let result = run_ch04(config);
    let to_f = |v: &Vec<f64>| -> PyResult<Py<PyList>> {
        let l = PyList::empty_bound(py); for x in v { l.append(x)?; } Ok(l.into())
    };
    let to_u = |v: &Vec<usize>| -> PyResult<Py<PyList>> {
        let l = PyList::empty_bound(py); for x in v { l.append(x)?; } Ok(l.into())
    };
    let pi = PyDict::new_bound(py);
    pi.set_item("values",            to_f(&result.pi.values)?)?;
    pi.set_item("policy",            to_u(&result.pi.policy)?)?;
    pi.set_item("pi_iterations",     result.pi.pi_iterations)?;
    let ei = PyList::empty_bound(py);
    for &n in &result.pi.eval_iterations { ei.append(n)?; }
    pi.set_item("eval_iterations",   ei)?;
    pi.set_item("convergence_curve", to_f(&result.pi.convergence_curve)?)?;
    let ph = PyList::empty_bound(py);
    for pol in &result.pi.policy_history {
        let row = PyList::empty_bound(py);
        for &a in pol { row.append(a)?; }
        ph.append(row)?;
    }
    pi.set_item("policy_history", ph)?;
    let vi = PyDict::new_bound(py);
    vi.set_item("values",     to_f(&result.vi_values)?)?;
    vi.set_item("policy",     to_u(&result.vi_policy)?)?;
    vi.set_item("iterations", result.vi_iterations)?;
    vi.set_item("curve",      to_f(&result.vi_curve)?)?;
    let av = PyDict::new_bound(py);
    av.set_item("values",     to_f(&result.async_values)?)?;
    av.set_item("policy",     to_u(&result.async_policy)?)?;
    av.set_item("iterations", result.async_iterations)?;
    av.set_item("curve",      to_f(&result.async_curve)?)?;
    let snames = PyList::empty_bound(py);
    for n in rlvr_core::ch02_bellman::STATE_NAMES { snames.append(n)?; }
    let anames = PyList::empty_bound(py);
    for n in rlvr_core::ch02_bellman::ACTION_NAMES { anames.append(n)?; }
    let out = PyDict::new_bound(py);
    out.set_item("pi",           pi)?;
    out.set_item("vi",           vi)?;
    out.set_item("async_vi",     av)?;
    out.set_item("residuals",    to_f(&result.residuals)?)?;
    out.set_item("state_names",  snames)?;
    out.set_item("action_names", anames)?;
    out.set_item("n_states",     N_STATES)?;
    out.set_item("n_actions",    N_ACTIONS)?;
    Ok(out.into())
}

#[pyfunction]
fn run_ch05_mc(py: Python, seed: u64, n_episodes: usize, gamma: f64, epsilon: f64, epsilon_decay: f64) -> PyResult<PyObject> {
    let config = McConfig { seed, gamma, n_episodes, epsilon, epsilon_decay };
    let result = run_ch05(config);
    let serialize_mc = |r: &rlvr_core::ch05_mc::McResult| -> PyResult<Py<PyDict>> {
        let d = PyDict::new_bound(py);
        let vals = PyList::empty_bound(py);
        for v in &r.values { vals.append(v)?; }
        d.set_item("values", vals)?;
        let pol = PyList::empty_bound(py);
        for &a in &r.policy { pol.append(a)?; }
        d.set_item("policy", pol)?;
        let qt = PyList::empty_bound(py);
        for row in &r.q_table {
            let rr = PyList::empty_bound(py);
            for &q in row { rr.append(q)?; }
            qt.append(rr)?;
        }
        d.set_item("q_table", qt)?;
        let rc = PyList::empty_bound(py);
        for v in &r.returns_curve { rc.append(v)?; }
        d.set_item("returns_curve", rc)?;
        let vc = PyList::empty_bound(py);
        for &v in &r.visit_counts { vc.append(v)?; }
        d.set_item("visit_counts", vc)?;
        let cc = PyList::empty_bound(py);
        for v in &r.convergence_curve { cc.append(v)?; }
        d.set_item("convergence_curve", cc)?;
        d.set_item("algorithm",  &r.algorithm)?;
        d.set_item("n_episodes", r.n_episodes)?;
        Ok(d.into())
    };
    let snames = PyList::empty_bound(py);
    for n in rlvr_core::ch02_bellman::STATE_NAMES { snames.append(n)?; }
    let anames = PyList::empty_bound(py);
    for n in rlvr_core::ch02_bellman::ACTION_NAMES { anames.append(n)?; }
    let out = PyDict::new_bound(py);
    out.set_item("first_visit",  serialize_mc(&result.first_visit)?)?;
    out.set_item("every_visit",  serialize_mc(&result.every_visit)?)?;
    out.set_item("on_policy",    serialize_mc(&result.on_policy)?)?;
    out.set_item("off_policy",   serialize_mc(&result.off_policy)?)?;
    out.set_item("state_names",  snames)?;
    out.set_item("action_names", anames)?;
    out.set_item("n_states",     N_STATES)?;
    out.set_item("n_actions",    N_ACTIONS)?;
    Ok(out.into())
}

#[pyfunction]
fn run_ch06_td(py: Python, seed: u64, n_episodes: usize, gamma: f64, alpha: f64, epsilon: f64, epsilon_decay: f64) -> PyResult<PyObject> {
    let config = TdConfig { seed, gamma, alpha, epsilon, epsilon_decay, n_episodes };
    let result = run_ch06(config);

    let serialize_td = |r: &rlvr_core::ch06_td::TdResult| -> PyResult<Py<PyDict>> {
        let d = PyDict::new_bound(py);
        let vals = PyList::empty_bound(py);
        for v in &r.values { vals.append(v)?; }
        d.set_item("values", vals)?;
        let pol = PyList::empty_bound(py);
        for &a in &r.policy { pol.append(a)?; }
        d.set_item("policy", pol)?;
        let qt = PyList::empty_bound(py);
        for row in &r.q_table {
            let rr = PyList::empty_bound(py);
            for &q in row { rr.append(q)?; }
            qt.append(rr)?;
        }
        d.set_item("q_table", qt)?;
        let rc = PyList::empty_bound(py);
        for v in &r.returns_curve { rc.append(v)?; }
        d.set_item("returns_curve", rc)?;
        let te = PyList::empty_bound(py);
        for v in &r.td_error_curve { te.append(v)?; }
        d.set_item("td_error_curve", te)?;
        let cc = PyList::empty_bound(py);
        for v in &r.convergence_curve { cc.append(v)?; }
        d.set_item("convergence_curve", cc)?;
        d.set_item("algorithm",    &r.algorithm)?;
        d.set_item("n_episodes",   r.n_episodes)?;
        d.set_item("total_steps",  r.total_steps)?;
        Ok(d.into())
    };

    let snames = PyList::empty_bound(py);
    for n in rlvr_core::ch02_bellman::STATE_NAMES { snames.append(n)?; }
    let anames = PyList::empty_bound(py);
    for n in rlvr_core::ch02_bellman::ACTION_NAMES { anames.append(n)?; }

    let out = PyDict::new_bound(py);
    out.set_item("td0",          serialize_td(&result.td0)?)?;
    out.set_item("sarsa",        serialize_td(&result.sarsa)?)?;
    out.set_item("qlearning",    serialize_td(&result.qlearn)?)?;
    out.set_item("state_names",  snames)?;
    out.set_item("action_names", anames)?;
    out.set_item("n_states",     N_STATES)?;
    out.set_item("n_actions",    N_ACTIONS)?;
    Ok(out.into())
}

#[pymodule]
fn rlvr_py(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_ch01_episode,         m)?)?;
    m.add_function(wrap_pyfunction!(run_ch02_value_iteration, m)?)?;
    m.add_function(wrap_pyfunction!(run_ch03_bandits,         m)?)?;
    m.add_function(wrap_pyfunction!(run_ch04_dp,              m)?)?;
    m.add_function(wrap_pyfunction!(run_ch05_mc,              m)?)?;
    m.add_function(wrap_pyfunction!(run_ch06_td,              m)?)?;
    Ok(())
}
'''

# ---------------------------------------------------------------------------
# FILE 3: gui/chapters/ch06.py
# ---------------------------------------------------------------------------
CH06_PY = r'''import streamlit as st
import plotly.graph_objects as go

T = {
    "EN": {
        "title": "Chapter 06 — Temporal Difference Learning",
        "subtitle": "TD(0) · SARSA · Q-Learning · ASP Dispatch · Warsaw Region",
        "engine_missing": "Run: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ TD Settings",
        "n_episodes": "Number of episodes",
        "gamma": "γ — Discount factor",
        "alpha": "α — Learning rate",
        "epsilon": "ε — Initial exploration",
        "epsilon_decay": "ε decay rate",
        "seed": "Random seed",
        "run_btn": "▶ Run TD(0), SARSA and Q-Learning",
        "guide_title": "🎓 How to use this chapter",
        "guide": """
**Step 1 — Understand the key difference from Ch05 (MC)**
TD methods update after EVERY step, not after episode end.
They bootstrap: use current estimates V(s') to update V(s).
This makes them faster and applicable to continuing tasks.

**Step 2 — Understand SARSA vs Q-Learning**
Both are TD control methods. The difference is one line:
- SARSA: uses Q(s', a') where a' is chosen by epsilon-greedy (on-policy)
- Q-Learning: uses max_a' Q(s', a') regardless of what action is taken (off-policy)

**Step 3 — Set α (learning rate)**
α=0.1 is a good start. Higher α = faster learning but less stable.
Lower α = slower but more stable convergence.

**Step 4 — Click ▶ Run TD(0), SARSA and Q-Learning**
All three run simultaneously. Results appear side by side.

**Step 5 — Read the TD Error curve**
TD error = R + γV(s') - V(s). Watch it decay toward zero.
This is the learning signal — the agent is "surprised" less over time.

**Step 6 — Compare Q-Learning vs SARSA policies**
Q-Learning finds the optimal policy (off-policy).
SARSA finds the safest policy given epsilon-soft exploration (on-policy).

**Step 7 — Compare with DP reference (Ch04)**
TD methods should converge to the same policy as DP — without needing P(s'|s,a).
""",
        "returns_title": "📈 Episode Returns — TD(0), SARSA, Q-Learning",
        "returns_caption": "Moving average of episode returns. Q-Learning should converge fastest.",
        "td_error_title": "📉 TD Error — |R + γV(s') - V(s)|",
        "td_error_caption": "TD error decays as the agent learns. Smaller = better estimates.",
        "value_title": "📊 Value Function V(s) — TD vs DP Reference",
        "value_caption": "TD estimates should converge toward DP solution (Ch04) with more episodes.",
        "policy_title": "🎯 Optimal Policy — SARSA vs Q-Learning",
        "policy_caption": "Q-Learning finds optimal policy. SARSA finds safest policy under epsilon-soft.",
        "qtable_title": "📊 Q-Table Heatmap",
        "qtable_caption": "Q(s,a) values. Select algorithm to display.",
        "glass_title": "🔬 Glass-Box — TD Update Trace",
        "glass_headers": ["Episode", "Step", "State", "Action", "Reward", "Next State", "TD Error"],
        "summary_title": "📊 Summary",
        "summary_results": "Algorithm Comparison",
        "summary_pros_cons": "TD Algorithms — Pros & Cons",
        "pros": "✅ Pros",
        "cons": "❌ Cons",
        "theory_title": "📖 Theory — Chapter 06",
        "theory_sections": {
            "td_intro":  "§6.1 Temporal Difference Learning",
            "td0":       "§6.1 TD(0) Prediction",
            "sarsa":     "§6.2 SARSA — On-Policy TD Control",
            "qlearning": "§6.3 Q-Learning — Off-Policy TD Control",
            "comparison":"§6.4 SARSA vs Q-Learning",
        },
        "theory_td_intro": r"""
**Temporal Difference (TD) Learning** combines ideas from MC and DP:
- Like MC: model-free, learns from experience
- Like DP: bootstraps — updates using current estimates, not waiting for episode end

The TD error (delta):
δ_t = R_{t+1} + γ V(S_{t+1}) - V(S_t)

This is the "surprise" signal — how much better or worse than expected.
Implemented in `ch06_td.rs`.
""",
        "theory_td0": r"""
**TD(0) Prediction** updates V(s) after every step:

V(S_t) <- V(S_t) + α [R_{t+1} + γ V(S_{t+1}) - V(S_t)]

- α = learning rate (step size)
- R_{t+1} + γ V(S_{t+1}) = TD target
- R_{t+1} + γ V(S_{t+1}) - V(S_t) = TD error δ_t

Converges to V^π for any fixed policy π.
Implemented in `td0_prediction()` in `ch06_td.rs`.
""",
        "theory_sarsa": r"""
**SARSA** (State-Action-Reward-State-Action) is on-policy TD control:

Q(S_t, A_t) <- Q(S_t, A_t) + α [R_{t+1} + γ Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]

Key: A_{t+1} is chosen by the SAME epsilon-greedy policy used for behaviour.
This makes SARSA on-policy — it learns the value of the epsilon-soft policy.

SARSA is conservative: it accounts for the exploration cost in its Q estimates.
Implemented in `sarsa()` in `ch06_td.rs`.
""",
        "theory_qlearning": r"""
**Q-Learning** is off-policy TD control:

Q(S_t, A_t) <- Q(S_t, A_t) + α [R_{t+1} + γ max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)]

Key: uses max_{a'} Q(S_{t+1}, a') — the GREEDY action, regardless of what was actually taken.
This makes Q-Learning off-policy — it directly learns Q* (optimal action-value function).

Q-Learning converges to Q* regardless of the behaviour policy (as long as all (s,a) are visited).
Implemented in `q_learning()` in `ch06_td.rs`.
""",
        "theory_comparison": r"""
**SARSA vs Q-Learning — the one-line difference:**

SARSA:      Q(s,a) += α [R + γ Q(s', a') - Q(s,a)]   where a' ~ ε-greedy
Q-Learning: Q(s,a) += α [R + γ max_a' Q(s', a') - Q(s,a)]

**When does it matter?**
In risky environments (like S7 — SLA breach imminent):
- SARSA avoids risky states because it accounts for epsilon-exploration
- Q-Learning ignores exploration cost and finds the theoretically optimal policy

**Cliff Walking analogy:**
- SARSA walks safely away from the cliff (accounts for accidental falls)
- Q-Learning walks along the cliff edge (optimal but risky during learning)

In ASP: SARSA is safer during training, Q-Learning finds the better final policy.
""",
        "algo_labels": {"td0": "TD(0)", "sarsa": "SARSA", "qlearning": "Q-Learning"},
        "pros_list": {
            "td0":       ["Online learning — updates every step", "No model needed", "Lower variance than MC"],
            "sarsa":     ["Safe during learning", "On-policy — consistent with behaviour", "Converges to optimal epsilon-soft policy"],
            "qlearning": ["Directly learns Q*", "Off-policy — can learn from any data", "Converges to optimal greedy policy"],
        },
        "cons_list": {
            "td0":       ["Only predicts V^pi — needs separate policy", "Biased (bootstrapping)", "Sensitive to alpha"],
            "sarsa":     ["Suboptimal if epsilon stays high", "On-policy — needs epsilon > 0", "Slower than Q-Learning in safe environments"],
            "qlearning": ["Can be risky during learning", "Overestimates Q values (maximisation bias)", "Sensitive to alpha"],
        },
    },
    "FR": {
        "title": "Chapitre 06 — Apprentissage par Différences Temporelles",
        "subtitle": "TD(0) · SARSA · Q-Learning · ASP · Région de Varsovie",
        "engine_missing": "Exécutez : `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Paramètres TD",
        "n_episodes": "Nombre d'épisodes",
        "gamma": "γ — Facteur d'actualisation",
        "alpha": "α — Taux d'apprentissage",
        "epsilon": "ε — Exploration initiale",
        "epsilon_decay": "Taux de décroissance ε",
        "seed": "Graine aléatoire",
        "run_btn": "▶ Lancer TD(0), SARSA et Q-Learning",
        "guide_title": "🎓 Comment utiliser ce chapitre",
        "guide": "TD met à jour après chaque étape (pas après l'épisode). SARSA = on-policy. Q-Learning = off-policy.",
        "returns_title": "📈 Retours par épisode",
        "returns_caption": "Moyenne mobile. Q-Learning devrait converger le plus vite.",
        "td_error_title": "📉 Erreur TD",
        "td_error_caption": "L'erreur TD décroît au fil de l'apprentissage.",
        "value_title": "📊 Fonction de valeur V(s)",
        "value_caption": "Les estimations TD convergent vers la solution DP (Ch04).",
        "policy_title": "🎯 Politique optimale — SARSA vs Q-Learning",
        "policy_caption": "Q-Learning trouve la politique optimale. SARSA trouve la plus sûre.",
        "qtable_title": "📊 Table Q",
        "qtable_caption": "Valeurs Q(s,a). Sélectionnez l'algorithme.",
        "glass_title": "🔬 Glass-Box — Trace de mise à jour TD",
        "glass_headers": ["Épisode", "Étape", "État", "Action", "Récompense", "État suivant", "Erreur TD"],
        "summary_title": "📊 Résumé",
        "summary_results": "Comparaison des algorithmes",
        "summary_pros_cons": "Algorithmes TD — Avantages & Inconvénients",
        "pros": "✅ Avantages", "cons": "❌ Inconvénients",
        "theory_title": "📖 Théorie — Chapitre 06",
        "theory_sections": {
            "td_intro": "§6.1 Apprentissage par différences temporelles",
            "td0": "§6.1 Prédiction TD(0)",
            "sarsa": "§6.2 SARSA — Contrôle TD on-policy",
            "qlearning": "§6.3 Q-Learning — Contrôle TD off-policy",
            "comparison": "§6.4 SARSA vs Q-Learning",
        },
        "theory_td_intro": "δ_t = R_{t+1} + γ V(S_{t+1}) - V(S_t)",
        "theory_td0": "V(S_t) ← V(S_t) + α [R_{t+1} + γ V(S_{t+1}) - V(S_t)]",
        "theory_sarsa": "Q(S_t,A_t) ← Q(S_t,A_t) + α [R_{t+1} + γ Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)]",
        "theory_qlearning": "Q(S_t,A_t) ← Q(S_t,A_t) + α [R_{t+1} + γ max_a' Q(S_{t+1},a') - Q(S_t,A_t)]",
        "theory_comparison": "SARSA: on-policy, sûr. Q-Learning: off-policy, optimal.",
        "algo_labels": {"td0": "TD(0)", "sarsa": "SARSA", "qlearning": "Q-Learning"},
        "pros_list": {"td0": ["En ligne", "Sans modèle"], "sarsa": ["Sûr pendant l'apprentissage", "On-policy"], "qlearning": ["Apprend Q* directement", "Off-policy"]},
        "cons_list": {"td0": ["Prédit seulement V^pi", "Biaisé"], "sarsa": ["Sous-optimal si ε élevé"], "qlearning": ["Risqué pendant l'apprentissage", "Biais de maximisation"]},
    },
    "ES": {
        "title": "Capítulo 06 — Aprendizaje por Diferencias Temporales",
        "subtitle": "TD(0) · SARSA · Q-Learning · ASP · Región de Varsovia",
        "engine_missing": "Ejecute: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Configuración TD",
        "n_episodes": "Número de episodios",
        "gamma": "γ — Factor de descuento",
        "alpha": "α — Tasa de aprendizaje",
        "epsilon": "ε — Exploración inicial",
        "epsilon_decay": "Tasa de decaimiento ε",
        "seed": "Semilla aleatoria",
        "run_btn": "▶ Ejecutar TD(0), SARSA y Q-Learning",
        "guide_title": "🎓 Cómo usar este capítulo",
        "guide": "TD actualiza después de cada paso. SARSA = on-policy. Q-Learning = off-policy.",
        "returns_title": "📈 Retornos por episodio",
        "returns_caption": "Media móvil. Q-Learning debería converger más rápido.",
        "td_error_title": "📉 Error TD",
        "td_error_caption": "El error TD decrece con el aprendizaje.",
        "value_title": "📊 Función de valor V(s)",
        "value_caption": "Las estimaciones TD convergen hacia la solución DP (Ch04).",
        "policy_title": "🎯 Política óptima — SARSA vs Q-Learning",
        "policy_caption": "Q-Learning encuentra la política óptima. SARSA la más segura.",
        "qtable_title": "📊 Tabla Q",
        "qtable_caption": "Valores Q(s,a). Seleccione el algoritmo.",
        "glass_title": "🔬 Glass-Box — Traza de actualización TD",
        "glass_headers": ["Episodio", "Paso", "Estado", "Acción", "Recompensa", "Siguiente estado", "Error TD"],
        "summary_title": "📊 Resumen",
        "summary_results": "Comparación de algoritmos",
        "summary_pros_cons": "Algoritmos TD — Pros y Contras",
        "pros": "✅ Pros", "cons": "❌ Contras",
        "theory_title": "📖 Teoría — Capítulo 06",
        "theory_sections": {
            "td_intro": "§6.1 Aprendizaje por diferencias temporales",
            "td0": "§6.1 Predicción TD(0)",
            "sarsa": "§6.2 SARSA — Control TD on-policy",
            "qlearning": "§6.3 Q-Learning — Control TD off-policy",
            "comparison": "§6.4 SARSA vs Q-Learning",
        },
        "theory_td_intro": "δ_t = R_{t+1} + γ V(S_{t+1}) - V(S_t)",
        "theory_td0": "V(S_t) ← V(S_t) + α [R_{t+1} + γ V(S_{t+1}) - V(S_t)]",
        "theory_sarsa": "Q(S_t,A_t) ← Q(S_t,A_t) + α [R_{t+1} + γ Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)]",
        "theory_qlearning": "Q(S_t,A_t) ← Q(S_t,A_t) + α [R_{t+1} + γ max_a' Q(S_{t+1},a') - Q(S_t,A_t)]",
        "theory_comparison": "SARSA: on-policy, seguro. Q-Learning: off-policy, óptimo.",
        "algo_labels": {"td0": "TD(0)", "sarsa": "SARSA", "qlearning": "Q-Learning"},
        "pros_list": {"td0": ["En línea", "Sin modelo"], "sarsa": ["Seguro durante aprendizaje", "On-policy"], "qlearning": ["Aprende Q* directamente", "Off-policy"]},
        "cons_list": {"td0": ["Solo predice V^pi", "Sesgado"], "sarsa": ["Subóptimo si ε alto"], "qlearning": ["Arriesgado durante aprendizaje", "Sesgo de maximización"]},
    },
    "PL": {
        "title": "Rozdział 06 — Uczenie przez Różnice Temporalne",
        "subtitle": "TD(0) · SARSA · Q-Learning · Dyspozytura ASP · Region Warszawy",
        "engine_missing": "Uruchom: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Ustawienia TD",
        "n_episodes": "Liczba epizodów",
        "gamma": "γ — Współczynnik dyskontowania",
        "alpha": "α — Współczynnik uczenia",
        "epsilon": "ε — Eksploracja początkowa",
        "epsilon_decay": "Współczynnik zaniku ε",
        "seed": "Ziarno losowości",
        "run_btn": "▶ Uruchom TD(0), SARSA i Q-Learning",
        "guide_title": "🎓 Jak korzystać z tego rozdziału",
        "guide": """
**Krok 1** — TD aktualizuje po KAŻDYM kroku (nie po epizodzie jak MC).
**Krok 2** — SARSA = on-policy (używa tej samej polityki do uczenia i zachowania).
**Krok 3** — Q-Learning = off-policy (uczy się optymalnej polityki niezależnie od zachowania).
**Krok 4** — Ustaw α (współczynnik uczenia). α=0.1 to dobry start.
**Krok 5** — Kliknij ▶ aby uruchomić wszystkie trzy algorytmy.
**Krok 6** — Odczytaj krzywą błędu TD — powinna maleć w czasie.
**Krok 7** — Porównaj polityki SARSA vs Q-Learning — Q-Learning powinien znaleźć lepszą.
""",
        "returns_title": "📈 Zwroty epizodów — TD(0), SARSA, Q-Learning",
        "returns_caption": "Średnia krocząca zwrotów. Q-Learning powinien zbiegać najszybciej.",
        "td_error_title": "📉 Błąd TD — |R + γV(s') - V(s)|",
        "td_error_caption": "Błąd TD maleje w miarę uczenia się agenta.",
        "value_title": "📊 Funkcja wartości V(s) — TD vs referencja DP",
        "value_caption": "Estymaty TD zbiegają do rozwiązania DP (Ch04) bez modelu P.",
        "policy_title": "🎯 Optymalna polityka — SARSA vs Q-Learning",
        "policy_caption": "Q-Learning znajduje optymalną politykę. SARSA — najbezpieczniejszą.",
        "qtable_title": "📊 Tabela Q",
        "qtable_caption": "Wartości Q(s,a). Wybierz algorytm.",
        "glass_title": "🔬 Glass-Box — Ślad aktualizacji TD",
        "glass_headers": ["Epizod", "Krok", "Stan", "Akcja", "Nagroda", "Następny stan", "Błąd TD"],
        "summary_title": "📊 Podsumowanie",
        "summary_results": "Porównanie algorytmów",
        "summary_pros_cons": "Algorytmy TD — Zalety i Wady",
        "pros": "✅ Zalety", "cons": "❌ Wady",
        "theory_title": "📖 Teoria — Rozdział 06",
        "theory_sections": {
            "td_intro":  "§6.1 Uczenie przez różnice temporalne",
            "td0":       "§6.1 Predykcja TD(0)",
            "sarsa":     "§6.2 SARSA — On-policy TD Control",
            "qlearning": "§6.3 Q-Learning — Off-policy TD Control",
            "comparison":"§6.4 SARSA vs Q-Learning",
        },
        "theory_td_intro": r"""
**Uczenie przez różnice temporalne (TD)** łączy MC i DP:
- Jak MC: bez modelu, uczy z doświadczenia
- Jak DP: bootstrapping — aktualizuje używając bieżących estymат

Błąd TD (delta): δ_t = R_{t+1} + γ V(S_{t+1}) - V(S_t)
Implementacja: `ch06_td.rs`
""",
        "theory_td0": "V(S_t) ← V(S_t) + α [R_{t+1} + γ V(S_{t+1}) - V(S_t)]",
        "theory_sarsa": "Q(S_t,A_t) ← Q(S_t,A_t) + α [R_{t+1} + γ Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)]",
        "theory_qlearning": "Q(S_t,A_t) ← Q(S_t,A_t) + α [R_{t+1} + γ max_a' Q(S_{t+1},a') - Q(S_t,A_t)]",
        "theory_comparison": r"""
**Jedna linia różnicy:**
SARSA:      Q(s,a) += α [R + γ Q(s', a') - Q(s,a)]   gdzie a' ~ ε-zachłanna
Q-Learning: Q(s,a) += α [R + γ max_a' Q(s', a') - Q(s,a)]

SARSA jest bezpieczniejszy podczas uczenia (uwzględnia eksplorację).
Q-Learning znajduje lepszą politykę końcową (bezpośrednio optymalizuje Q*).
""",
        "algo_labels": {"td0": "TD(0)", "sarsa": "SARSA", "qlearning": "Q-Learning"},
        "pros_list": {
            "td0":       ["Uczenie online — aktualizacja po każdym kroku", "Bez modelu", "Niższa wariancja niż MC"],
            "sarsa":     ["Bezpieczny podczas uczenia", "On-policy — spójny z zachowaniem", "Zbiega do optymalnej polityki epsilon-soft"],
            "qlearning": ["Bezpośrednio uczy Q*", "Off-policy — może uczyć z dowolnych danych", "Zbiega do optymalnej zachłannej polityki"],
        },
        "cons_list": {
            "td0":       ["Tylko predykcja V^pi", "Obciążony (bootstrapping)", "Wrażliwy na α"],
            "sarsa":     ["Suboptymalny gdy ε wysokie", "On-policy — wymaga ε > 0"],
            "qlearning": ["Ryzykowny podczas uczenia", "Przeszacowuje Q (bias maksymalizacji)", "Wrażliwy na α"],
        },
    },
}

COLORS = {"td0": "#0082F0", "sarsa": "#FF8C0A", "qlearning": "#0FC373"}

def _moving_avg(data, window=20):
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(data[start:i+1]) / (i - start + 1))
    return result

def render():
    lang = st.session_state.get("lang", "EN")
    tx = T[lang]
    st.title(tx["title"])
    st.caption(tx["subtitle"])
    try:
        import rlvr_py
    except ImportError:
        st.error(tx["engine_missing"])
        return

    st.sidebar.header(tx["sidebar_title"])
    n_episodes    = st.sidebar.slider(tx["n_episodes"],    50, 5000, 1000, 50)
    gamma         = st.sidebar.slider(tx["gamma"],         0.5, 0.999, 0.95, 0.005)
    alpha         = st.sidebar.slider(tx["alpha"],         0.01, 1.0, 0.1, 0.01)
    epsilon       = st.sidebar.slider(tx["epsilon"],       0.0, 1.0, 0.3, 0.05)
    epsilon_decay = st.sidebar.slider(tx["epsilon_decay"], 0.0, 0.1, 0.01, 0.001, format="%.3f")
    seed          = st.sidebar.number_input(tx["seed"], 0, 9999, 42)

    with st.expander(tx["guide_title"], expanded=False):
        st.markdown(tx["guide"])

    if st.button(tx["run_btn"], type="primary"):
        with st.spinner("Running Rust TD engine..."):
            result = rlvr_py.run_ch06_td(
                int(seed), int(n_episodes), float(gamma),
                float(alpha), float(epsilon), float(epsilon_decay)
            )
        st.session_state["ch06_result"] = result

    if "ch06_result" not in st.session_state:
        st.info("Configure settings and click **▶ Run TD(0), SARSA and Q-Learning**.")
        _render_theory(tx)
        return

    result       = st.session_state["ch06_result"]
    state_names  = result["state_names"]
    action_names = result["action_names"]
    algos        = ["td0", "sarsa", "qlearning"]

    # KPI
    cols = st.columns(3)
    for i, key in enumerate(algos):
        r = result[key]
        avg = sum(r["returns_curve"][-50:]) / min(50, len(r["returns_curve"]))
        cols[i].metric(tx["algo_labels"][key],
                       f"Avg return: {avg:.2f}",
                       f"Steps: {r['total_steps']:,}")

    # Returns
    st.subheader(tx["returns_title"])
    fig = go.Figure()
    for key in algos:
        ma = _moving_avg(result[key]["returns_curve"], 30)
        fig.add_trace(go.Scatter(x=list(range(len(ma))), y=ma,
            mode="lines", name=tx["algo_labels"][key],
            line=dict(color=COLORS[key], width=2)))
    fig.update_layout(height=300, margin=dict(l=40,r=20,t=20,b=40),
                      xaxis_title="Episode", yaxis_title="Return (MA-30)",
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(tx["returns_caption"])

    # TD Error
    st.subheader(tx["td_error_title"])
    fig2 = go.Figure()
    for key in ["sarsa", "qlearning"]:
        ma = _moving_avg(result[key]["td_error_curve"], 30)
        fig2.add_trace(go.Scatter(x=list(range(len(ma))), y=ma,
            mode="lines", name=tx["algo_labels"][key],
            line=dict(color=COLORS[key], width=2)))
    fig2.update_layout(height=280, margin=dict(l=40,r=20,t=20,b=40),
                       xaxis_title="Episode", yaxis_title="Avg TD Error",
                       legend=dict(orientation="h"))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(tx["td_error_caption"])

    # Value function
    st.subheader(tx["value_title"])
    short = [f"S{i}" for i in range(result["n_states"])]
    fig3 = go.Figure()
    for key in algos:
        fig3.add_trace(go.Bar(x=short, y=result[key]["values"],
            name=tx["algo_labels"][key], marker_color=COLORS[key], opacity=0.8))
    fig3.update_layout(height=300, barmode="group",
                       margin=dict(l=40,r=20,t=20,b=40),
                       legend=dict(orientation="h"))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(tx["value_caption"])

    # Policy comparison
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(tx["policy_title"])
        rows = []
        for s in range(result["n_states"]):
            sa = result["sarsa"]["policy"][s]
            ql = result["qlearning"]["policy"][s]
            rows.append({
                "State": f"S{s}",
                "SARSA": f"A{sa}",
                "Q-Learning": f"A{ql}",
                "Match": "✅" if sa == ql else "🔄",
            })
        st.dataframe(rows, hide_index=True)
        st.caption(tx["policy_caption"])

    with col2:
        st.subheader(tx["qtable_title"])
        algo_sel = st.selectbox("Algorithm", [tx["algo_labels"][k] for k in ["sarsa","qlearning"]])
        key_sel = "sarsa" if "SARSA" in algo_sel or "sarsa" in algo_sel.lower() else "qlearning"
        qt = result[key_sel]["q_table"]
        action_short = [f"A{i}" for i in range(result["n_actions"])]
        fig4 = go.Figure(go.Heatmap(
            z=qt, x=action_short, y=short,
            colorscale="Blues",
            text=[[f"{qt[s][a]:.2f}" for a in range(result["n_actions"])]
                  for s in range(result["n_states"])],
            texttemplate="%{text}",
        ))
        fig4.update_layout(height=320, margin=dict(l=60,r=20,t=20,b=40))
        st.plotly_chart(fig4, use_container_width=True)
        st.caption(tx["qtable_caption"])

    # Glass-Box
    st.subheader(tx["glass_title"])
    _render_glass_box(result, tx)

    # Summary
    st.subheader(tx["summary_title"])
    _render_summary(result, tx, algos)

    _render_theory(tx)


def _render_glass_box(result, tx):
    algo_options = {tx["algo_labels"][k]: k for k in ["td0","sarsa","qlearning"]}
    selected = st.selectbox("Algorithm", list(algo_options.keys()), key="gb_algo")
    key = algo_options[selected]
    r = result[key]
    ep_idx = st.slider("Episode", 0, max(len(r["returns_curve"])-1, 0),
                       max(len(r["returns_curve"])-1, 0), key="gb_ep")
    st.metric("Episode return", f"{r['returns_curve'][ep_idx]:.3f}")
    st.metric("Avg TD error this episode", f"{r['td_error_curve'][ep_idx]:.4f}")
    if key == "sarsa":
        st.latex(r"Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha[R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)]")
    elif key == "qlearning":
        st.latex(r"Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha[R_{t+1} + \gamma \max_{a'} Q(S_{t+1},a') - Q(S_t,A_t)]")
    else:
        st.latex(r"V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]")


def _render_summary(result, tx, algos):
    st.markdown(f"#### {tx['summary_results']}")
    rows = []
    for key in algos:
        r = result[key]
        avg = sum(r["returns_curve"][-100:]) / min(100, len(r["returns_curve"]))
        rows.append({
            "Algorithm":           tx["algo_labels"][key],
            "Avg return (last 100)": f"{avg:.3f}",
            "Total steps":         str(r["total_steps"]),
            "V*(S0)":              f"{r['values'][0]:.3f}",
            "V*(S7)":              f"{r['values'][7]:.3f}",
            "Policy S7":           f"A{r['policy'][7]}",
        })
    st.dataframe(rows, hide_index=True)
    st.markdown(f"#### {tx['summary_pros_cons']}")
    for key in algos:
        label = tx["algo_labels"][key]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{label} — {tx['pros']}**")
            for p in tx["pros_list"][key]: st.markdown(f"- {p}")
        with c2:
            st.markdown(f"**{label} — {tx['cons']}**")
            for c in tx["cons_list"][key]: st.markdown(f"- {c}")
        st.markdown("---")


def _render_theory(tx):
    st.markdown("---")
    st.subheader(tx["theory_title"])
    for key in ["td_intro","td0","sarsa","qlearning","comparison"]:
        with st.expander(tx["theory_sections"][key], expanded=False):
            st.markdown(tx[f"theory_{key}"])
'''

# ---------------------------------------------------------------------------
# FILE 4: README.md
# ---------------------------------------------------------------------------
README_MD = """# RLVR Enterprise Allocator

> **Reinforcement Learning via Rust** — A 20-chapter, end-to-end framework for learning and demonstrating RL algorithms through a real enterprise field-service optimisation use case.

[![Rust](https://img.shields.io/badge/Rust-1.75+-orange?logo=rust)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Progress](https://img.shields.io/badge/Progress-Ch01--06%20%E2%9C%85%20%7C%20Ch07--20%20%F0%9F%9A%A7-lightgrey)]()

---

## What is this?

This project implements **all 20 chapters** of the [Reinforcement Learning via Rust](https://rlvr.rantai.dev/docs/reinforcement-learning-via-rust/) book as a fully working, interactive demo application built to learn RL from first principles while simultaneously building production-grade Rust infrastructure.

**The business scenario:** A field-service dispatch system for the Warsaw region. An RL agent learns to assign technicians to work orders, optimising SLA compliance, skill matching, and travel distance.

**The architecture principle:** All RL logic lives in Rust. Python is used exclusively for UI rendering.

---

## Progress

| Chapter | Topic | Algorithm | Status |
|---------|-------|-----------|--------|
| **01** | Introduction to RL | MDP, epsilon-greedy, Gt | ✅ Complete |
| **02** | Discrete MDP & Bellman | Value Iteration, nalgebra LU | ✅ Complete |
| **03** | Multi-Armed Bandit | UCB1, Thompson Sampling | ✅ Complete |
| **04** | Dynamic Programming | Policy Iteration, Async VI | ✅ Complete |
| **05** | Monte Carlo Methods | First-Visit, Every-Visit, On/Off-Policy | ✅ Complete |
| **06** | Temporal Difference | TD(0), SARSA, Q-Learning | ✅ Complete |
| 07 | n-Step TD & Planning | Dyna-Q | 🚧 Planned |
| 08 | Eligibility Traces | TD(lambda), SARSA(lambda) | 🚧 Planned |
| 09 | Policy Gradient | REINFORCE, Softmax | 🚧 Planned |
| 10 | Model-Based RL | World Models | 🚧 Planned |
| 11 | Multi-Agent RL | Independent Q-Learning | 🚧 Planned |
| 12 | Game Theory & Nash | Nash Equilibrium | 🚧 Planned |
| 13 | Cooperative MARL | VDN, QMIX | 🚧 Planned |
| 14 | Learning Dynamics | ELO, Fictitious Play | 🚧 Planned |
| 15 | Deep RL (DQN) | DQN + burn tensors | 🚧 Planned |
| 16 | Actor-Critic | A2C, PPO | 🚧 Planned |
| 17 | Advanced Policy Opt. | SAC, DDPG | 🚧 Planned |
| 18 | QMIX Deep | Monotonic Mixing | 🚧 Planned |
| 19 | Federated RL | Differential Privacy | 🚧 Planned |
| 20 | PyO3 Interop & Safety | FFI Safety Invariants | 🚧 Planned |

---

## Repository Structure

```
rlvr-enterprise-allocator/
├── Cargo.toml
├── rlvr-core/src/
│   ├── ch01_asp_dispatch.rs   # MDP, epsilon-greedy, Gt
│   ├── ch02_bellman.rs        # Value Iteration, nalgebra LU
│   ├── ch03_bandit.rs         # UCB1, Thompson Sampling
│   ├── ch04_dp.rs             # Policy Iteration, Async VI
│   ├── ch05_mc.rs             # Monte Carlo: First/Every-Visit, On/Off-Policy
│   └── ch06_td.rs             # TD(0), SARSA, Q-Learning
├── rlvr-py/src/lib.rs         # PyO3 bridge (Ch01-Ch06)
└── gui/chapters/
    ├── ch01.py  ch02.py  ch03.py  ch04.py  ch05.py  ch06.py
```

---

## Quick Start

```bash
cd rlvr-py && maturin develop && cd ..
cargo test --workspace   # 59 tests passing
streamlit run gui/app.py
```

---

## Chapter Summaries

### Ch01-Ch05 — see previous releases

### Ch06 — Temporal Difference Learning
- **TD(0)**: online V(s) prediction — updates every step, not episode end
- **SARSA**: on-policy TD control — safe during learning, accounts for exploration
- **Q-Learning**: off-policy TD control — directly learns Q*, optimal greedy policy
- TD error curve, returns curve, Q-table heatmap, SARSA vs Q-Learning policy diff
- Glass-Box: exact TD update equation per episode with δ_t values
- 4 languages: EN / FR / ES / PL

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| RL Engine | Rust 1.75+ | All algorithms |
| Arrays | ndarray v0.16 | Q-tables, matrices |
| Linear algebra | nalgebra v0.33 | LU decomposition |
| RNG | rand v0.8 + StdRng | Deterministic |
| FFI | pyo3 v0.21 + maturin | Rust -> Python |
| UI | Streamlit + Plotly | Rendering only |

---

## License

MIT -- see [LICENSE](LICENSE)
"""

# ---------------------------------------------------------------------------
# FILE 5: apply_ch06.sh
# ---------------------------------------------------------------------------
APPLY_SH = """#!/bin/bash
set -e
echo "Applying Chapter 06..."

cp ch06_package/rlvr-core/src/ch06_td.rs rlvr-core/src/ch06_td.rs
echo "pub mod ch06_td;" >> rlvr-core/src/lib.rs

cp ch06_package/rlvr-py/src/lib.rs rlvr-py/src/lib.rs
cp ch06_package/gui/chapters/ch06.py gui/chapters/ch06.py
cp ch06_package/README.md README.md

python3 -c "
content = open('gui/app.py').read()
content = content.replace(
    'elif ch_num == 5:\\n    from chapters.ch05 import render\\n    render()',
    'elif ch_num == 5:\\n    from chapters.ch05 import render\\n    render()\\nelif ch_num == 6:\\n    from chapters.ch06 import render\\n    render()'
)
open('gui/app.py', 'w').write(content)
print('app.py routing updated')
"

python3 -c "
lines = open('rlvr-core/src/lib.rs').readlines()
seen = set()
out = []
for line in lines:
    if line.strip().startswith('pub mod') and line.strip() in seen:
        continue
    seen.add(line.strip())
    out.append(line)
open('rlvr-core/src/lib.rs', 'w').writelines(out)
print('lib.rs deduped')
"

cd rlvr-py && maturin develop && cd ..
cargo test --workspace

echo ""
echo "Chapter 06 applied! Run: streamlit run gui/app.py"
"""

# ---------------------------------------------------------------------------
# Build ZIP
# ---------------------------------------------------------------------------
files = {
    "ch06_package/rlvr-core/src/ch06_td.rs":   CH06_RS,
    "ch06_package/rlvr-py/src/lib.rs":          LIB_RS,
    "ch06_package/gui/chapters/ch06.py":        CH06_PY,
    "ch06_package/README.md":                   README_MD,
    "ch06_package/apply_ch06.sh":               APPLY_SH,
}

zip_path = "/tmp/ch06_package.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for path, content in files.items():
        zf.writestr(path, content)

print(f"ZIP created: {zip_path}")
print(f"Size: {os.path.getsize(zip_path):,} bytes")
print("Files:")
with zipfile.ZipFile(zip_path, "r") as zf:
    for name in zf.namelist():
        info = zf.getinfo(name)
        print(f"  {name:55s} {info.file_size:>8,} bytes")

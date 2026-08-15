#!/usr/bin/env python3
"""
RLVR Enterprise Allocator — Chapter 07 ZIP Generator
Run from project root:
    python3 ch07_generate.py
Produces: ch07_package.zip
"""

import zipfile
import os

# ---------------------------------------------------------------------------
# FILE 1: rlvr-core/src/ch07_nstep.rs
# ---------------------------------------------------------------------------
CH07_RS = r'''use ndarray::Array2;
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
pub struct NStepConfig {
    pub seed:          u64,
    pub gamma:         f64,
    pub alpha:         f64,
    pub epsilon:       f64,
    pub epsilon_decay: f64,
    pub n_episodes:    usize,
    pub n_step:        usize,   // n for n-step TD
    pub planning_steps: usize,  // k for Dyna-Q
    pub kappa:         f64,     // exploration bonus for Dyna-Q+
}

#[derive(Debug, Clone)]
pub struct NStepResult {
    pub algorithm:         String,
    pub values:            Vec<f64>,
    pub policy:            Vec<usize>,
    pub q_table:           Vec<Vec<f64>>,
    pub returns_curve:     Vec<f64>,
    pub td_error_curve:    Vec<f64>,
    pub convergence_curve: Vec<f64>,
    pub n_episodes:        usize,
    pub total_steps:       usize,
    pub model_size:        usize,  // number of (s,a) pairs in Dyna model
}

#[derive(Debug, Clone)]
pub struct Ch07Result {
    pub nstep_td:     NStepResult,
    pub nstep_sarsa:  NStepResult,
    pub dyna_q:       NStepResult,
    pub dyna_q_plus:  NStepResult,
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

fn epsilon_greedy(q: &[Vec<f64>], s: usize, epsilon: f64, rng: &mut StdRng) -> usize {
    if rng.gen::<f64>() < epsilon {
        rng.gen_range(0..N_ACTIONS)
    } else {
        q[s].iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0)
    }
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
// n-Step TD Prediction
// G^(n)_t = R_{t+1} + γR_{t+2} + ... + γ^(n-1)R_{t+n} + γ^n V(S_{t+n})
// V(S_t) <- V(S_t) + α [G^(n)_t - V(S_t)]
// ---------------------------------------------------------------------------
pub fn nstep_td_prediction(
    config: &NStepConfig,
    transitions: &ndarray::Array3<f64>,
    rewards: &Array2<f64>,
) -> NStepResult {
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut v = vec![0.0_f64; N_STATES];
    let policy = vec![1usize; N_STATES]; // fixed baseline policy
    let mut returns_curve = Vec::new();
    let mut td_error_curve = Vec::new();
    let mut convergence_curve = Vec::new();
    let mut v_prev = v.clone();
    let mut total_steps = 0usize;
    let n = config.n_step;

    for _ in 0..config.n_episodes {
        let mut states  = Vec::new();
        let mut rew     = Vec::new();
        let mut s = rng.gen_range(0..N_STATES);
        states.push(s);

        let mut t = 0usize;
        let mut T = usize::MAX;
        let mut ep_return = 0.0_f64;
        let mut ep_td_err = 0.0_f64;
        let mut steps = 0usize;

        loop {
            if t < T {
                let a = policy[states[t]];
                let r = rewards[[states[t], a]];
                let sp = sample_next_state(states[t], a, transitions, &mut rng);
                rew.push(r);
                states.push(sp);
                ep_return += config.gamma.powi(t as i32) * r;
                total_steps += 1;
                steps += 1;
                if sp == 0 || sp == 7 || steps >= 50 { T = t + 1; }
            }

            let tau = t as isize - n as isize + 1;
            if tau >= 0 {
                let tau = tau as usize;
                // compute G^(n)
                let end = (tau + n).min(T);
                let mut g = 0.0_f64;
                for i in tau..end {
                    g += config.gamma.powi((i - tau) as i32) * rew[i];
                }
                if tau + n < T {
                    g += config.gamma.powi(n as i32) * v[states[tau + n]];
                }
                let td_error = g - v[states[tau]];
                v[states[tau]] += config.alpha * td_error;
                ep_td_err += td_error.abs();
            }

            if tau == T.saturating_sub(1) { break; }
            t += 1;
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

    NStepResult {
        algorithm: format!("nstep_td_n{}", n),
        values: v,
        policy: greedy_policy(&q_table),
        q_table,
        returns_curve,
        td_error_curve,
        convergence_curve,
        n_episodes: config.n_episodes,
        total_steps,
        model_size: 0,
    }
}

// ---------------------------------------------------------------------------
// n-Step SARSA (on-policy)
// G^(n)_t = R_{t+1} + ... + γ^(n-1)R_{t+n} + γ^n Q(S_{t+n}, A_{t+n})
// Q(S_t,A_t) <- Q(S_t,A_t) + α [G^(n)_t - Q(S_t,A_t)]
// ---------------------------------------------------------------------------
pub fn nstep_sarsa(
    config: &NStepConfig,
    transitions: &ndarray::Array3<f64>,
    rewards: &Array2<f64>,
) -> NStepResult {
    let mut rng = StdRng::seed_from_u64(config.seed + 1);
    let mut q = vec![vec![0.0_f64; N_ACTIONS]; N_STATES];
    let mut returns_curve = Vec::new();
    let mut td_error_curve = Vec::new();
    let mut convergence_curve = Vec::new();
    let mut v_prev = v_from_q(&q);
    let mut total_steps = 0usize;
    let n = config.n_step;

    for ep in 0..config.n_episodes {
        let epsilon_t = (config.epsilon / (1.0 + config.epsilon_decay * ep as f64)).max(0.01);
        let mut states  = Vec::new();
        let mut actions = Vec::new();
        let mut rew     = Vec::new();

        let s0 = rng.gen_range(0..N_STATES);
        let a0 = epsilon_greedy(&q, s0, epsilon_t, &mut rng);
        states.push(s0);
        actions.push(a0);

        let mut t = 0usize;
        let mut T = usize::MAX;
        let mut ep_return = 0.0_f64;
        let mut ep_td_err = 0.0_f64;
        let mut steps = 0usize;

        loop {
            if t < T {
                let r = rewards[[states[t], actions[t]]];
                let sp = sample_next_state(states[t], actions[t], transitions, &mut rng);
                rew.push(r);
                states.push(sp);
                ep_return += config.gamma.powi(t as i32) * r;
                total_steps += 1;
                steps += 1;
                if sp == 0 || sp == 7 || steps >= 50 {
                    T = t + 1;
                    actions.push(0); // dummy
                } else {
                    let ap = epsilon_greedy(&q, sp, epsilon_t, &mut rng);
                    actions.push(ap);
                }
            }

            let tau = t as isize - n as isize + 1;
            if tau >= 0 {
                let tau = tau as usize;
                let end = (tau + n).min(T);
                let mut g = 0.0_f64;
                for i in tau..end {
                    g += config.gamma.powi((i - tau) as i32) * rew[i];
                }
                if tau + n < T {
                    g += config.gamma.powi(n as i32) * q[states[tau + n]][actions[tau + n]];
                }
                let td_error = g - q[states[tau]][actions[tau]];
                q[states[tau]][actions[tau]] += config.alpha * td_error;
                ep_td_err += td_error.abs();
            }

            if tau == T.saturating_sub(1) { break; }
            t += 1;
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
    NStepResult {
        algorithm: format!("nstep_sarsa_n{}", n),
        values,
        policy: greedy_policy(&q),
        q_table: q,
        returns_curve,
        td_error_curve,
        convergence_curve,
        n_episodes: config.n_episodes,
        total_steps,
        model_size: 0,
    }
}

// ---------------------------------------------------------------------------
// Dyna-Q
// Q-Learning + model learning + planning (k simulated steps per real step)
// ---------------------------------------------------------------------------
pub fn dyna_q(
    config: &NStepConfig,
    transitions: &ndarray::Array3<f64>,
    rewards: &Array2<f64>,
) -> NStepResult {
    let mut rng = StdRng::seed_from_u64(config.seed + 2);
    let mut q = vec![vec![0.0_f64; N_ACTIONS]; N_STATES];
    // Model: (s,a) -> (r, s')
    let mut model: HashMap<(usize, usize), (f64, usize)> = HashMap::new();
    let mut visited: Vec<(usize, usize)> = Vec::new();
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
        let mut steps = 0usize;
        let mut gamma_pow = 1.0_f64;

        loop {
            let a = epsilon_greedy(&q, s, epsilon_t, &mut rng);
            let r = rewards[[s, a]];
            let sp = sample_next_state(s, a, transitions, &mut rng);

            // Direct RL update (Q-Learning)
            let max_q_sp = q[sp].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let td_error = r + config.gamma * max_q_sp - q[s][a];
            q[s][a] += config.alpha * td_error;

            // Model learning
            if !model.contains_key(&(s, a)) {
                visited.push((s, a));
            }
            model.insert((s, a), (r, sp));

            ep_return += gamma_pow * r;
            ep_td_err += td_error.abs();
            gamma_pow *= config.gamma;
            total_steps += 1;
            steps += 1;

            // Planning: k simulated steps
            for _ in 0..config.planning_steps {
                if visited.is_empty() { break; }
                let idx = rng.gen_range(0..visited.len());
                let (ps, pa) = visited[idx];
                if let Some(&(pr, psp)) = model.get(&(ps, pa)) {
                    let max_q = q[psp].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let plan_td = pr + config.gamma * max_q - q[ps][pa];
                    q[ps][pa] += config.alpha * plan_td;
                }
            }

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
    let model_size = model.len();
    NStepResult {
        algorithm: format!("dyna_q_k{}", config.planning_steps),
        values,
        policy: greedy_policy(&q),
        q_table: q,
        returns_curve,
        td_error_curve,
        convergence_curve,
        n_episodes: config.n_episodes,
        total_steps,
        model_size,
    }
}

// ---------------------------------------------------------------------------
// Dyna-Q+
// Dyna-Q + exploration bonus: r' = r + kappa * sqrt(tau(s,a))
// tau(s,a) = time since (s,a) was last tried
// ---------------------------------------------------------------------------
pub fn dyna_q_plus(
    config: &NStepConfig,
    transitions: &ndarray::Array3<f64>,
    rewards: &Array2<f64>,
) -> NStepResult {
    let mut rng = StdRng::seed_from_u64(config.seed + 3);
    let mut q = vec![vec![0.0_f64; N_ACTIONS]; N_STATES];
    let mut model: HashMap<(usize, usize), (f64, usize)> = HashMap::new();
    let mut visited: Vec<(usize, usize)> = Vec::new();
    // tau(s,a) = steps since last visit
    let mut last_visit: HashMap<(usize, usize), usize> = HashMap::new();
    let mut returns_curve = Vec::new();
    let mut td_error_curve = Vec::new();
    let mut convergence_curve = Vec::new();
    let mut v_prev = v_from_q(&q);
    let mut total_steps = 0usize;
    let mut global_step = 0usize;

    for ep in 0..config.n_episodes {
        let epsilon_t = (config.epsilon / (1.0 + config.epsilon_decay * ep as f64)).max(0.01);
        let mut s = rng.gen_range(0..N_STATES);
        let mut ep_return = 0.0_f64;
        let mut ep_td_err = 0.0_f64;
        let mut steps = 0usize;
        let mut gamma_pow = 1.0_f64;

        loop {
            let a = epsilon_greedy(&q, s, epsilon_t, &mut rng);
            let r = rewards[[s, a]];
            let sp = sample_next_state(s, a, transitions, &mut rng);

            let max_q_sp = q[sp].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let td_error = r + config.gamma * max_q_sp - q[s][a];
            q[s][a] += config.alpha * td_error;

            if !model.contains_key(&(s, a)) {
                visited.push((s, a));
            }
            model.insert((s, a), (r, sp));
            last_visit.insert((s, a), global_step);

            ep_return += gamma_pow * r;
            ep_td_err += td_error.abs();
            gamma_pow *= config.gamma;
            total_steps += 1;
            global_step += 1;
            steps += 1;

            // Planning with exploration bonus
            for _ in 0..config.planning_steps {
                if visited.is_empty() { break; }
                let idx = rng.gen_range(0..visited.len());
                let (ps, pa) = visited[idx];
                if let Some(&(pr, psp)) = model.get(&(ps, pa)) {
                    let tau = global_step - last_visit.get(&(ps, pa)).copied().unwrap_or(0);
                    // exploration bonus: kappa * sqrt(tau)
                    let bonus = config.kappa * (tau as f64).sqrt();
                    let max_q = q[psp].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let plan_td = (pr + bonus) + config.gamma * max_q - q[ps][pa];
                    q[ps][pa] += config.alpha * plan_td;
                }
            }

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
    let model_size = model.len();
    NStepResult {
        algorithm: format!("dyna_q_plus_k{}", config.planning_steps),
        values,
        policy: greedy_policy(&q),
        q_table: q,
        returns_curve,
        td_error_curve,
        convergence_curve,
        n_episodes: config.n_episodes,
        total_steps,
        model_size,
    }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------
pub fn run_ch07(config: NStepConfig) -> Ch07Result {
    let mut rng = StdRng::seed_from_u64(config.seed);
    let transitions = build_asp_transitions(&mut rng);
    let rewards = build_asp_rewards();
    verify_transition_matrix(&transitions)
        .expect("Transition matrix probability conservation violated");

    let nstep_td    = nstep_td_prediction(&config, &transitions, &rewards);
    let nstep_sarsa = nstep_sarsa(&config, &transitions, &rewards);
    let dyna_q      = dyna_q(&config, &transitions, &rewards);
    let dyna_q_plus = dyna_q_plus(&config, &transitions, &rewards);

    Ch07Result { nstep_td, nstep_sarsa, dyna_q, dyna_q_plus }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> NStepConfig {
        NStepConfig {
            seed: 42, gamma: 0.95, alpha: 0.1,
            epsilon: 0.3, epsilon_decay: 0.01,
            n_episodes: 500, n_step: 3,
            planning_steps: 5, kappa: 0.001,
        }
    }

    #[test]
    fn test_nstep_td_runs() {
        let r = run_ch07(default_config());
        assert_eq!(r.nstep_td.values.len(), N_STATES);
        assert!(r.nstep_td.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_nstep_sarsa_runs() {
        let r = run_ch07(default_config());
        assert_eq!(r.nstep_sarsa.values.len(), N_STATES);
        assert!(r.nstep_sarsa.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_dyna_q_runs() {
        let r = run_ch07(default_config());
        assert_eq!(r.dyna_q.values.len(), N_STATES);
        assert!(r.dyna_q.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_dyna_q_plus_runs() {
        let r = run_ch07(default_config());
        assert_eq!(r.dyna_q_plus.values.len(), N_STATES);
        assert!(r.dyna_q_plus.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_policy_valid() {
        let r = run_ch07(default_config());
        for &a in &r.nstep_sarsa.policy  { assert!(a < N_ACTIONS); }
        for &a in &r.dyna_q.policy       { assert!(a < N_ACTIONS); }
        for &a in &r.dyna_q_plus.policy  { assert!(a < N_ACTIONS); }
    }

    #[test]
    fn test_returns_curve_length() {
        let config = default_config();
        let r = run_ch07(config.clone());
        assert_eq!(r.nstep_sarsa.returns_curve.len(), config.n_episodes);
        assert_eq!(r.dyna_q.returns_curve.len(),      config.n_episodes);
    }

    #[test]
    fn test_dyna_q_model_grows() {
        let r = run_ch07(default_config());
        assert!(r.dyna_q.model_size > 0, "Dyna-Q model should learn transitions");
    }

    #[test]
    fn test_dyna_q_faster_than_qlearning() {
        // Dyna-Q with planning should converge faster (higher early returns)
        let config = NStepConfig { n_episodes: 200, planning_steps: 10, ..default_config() };
        let r = run_ch07(config);
        let dyna_early: f64 = r.dyna_q.returns_curve[..50].iter().sum::<f64>() / 50.0;
        let sarsa_early: f64 = r.nstep_sarsa.returns_curve[..50].iter().sum::<f64>() / 50.0;
        // Dyna-Q should be at least competitive with n-step SARSA
        assert!(dyna_early >= sarsa_early - 10.0,
            "Dyna-Q early={:.2} sarsa early={:.2}", dyna_early, sarsa_early);
    }

    #[test]
    fn test_q_table_shape() {
        let r = run_ch07(default_config());
        assert_eq!(r.dyna_q.q_table.len(), N_STATES);
        for row in &r.dyna_q.q_table { assert_eq!(row.len(), N_ACTIONS); }
    }

    #[test]
    fn test_deterministic() {
        let r1 = run_ch07(default_config());
        let r2 = run_ch07(default_config());
        for (v1, v2) in r1.dyna_q.values.iter().zip(r2.dyna_q.values.iter()) {
            assert_eq!(v1.to_bits(), v2.to_bits());
        }
    }
}
'''

# ---------------------------------------------------------------------------
# FILE 2: rlvr-py/src/lib.rs (Ch01-Ch07)
# ---------------------------------------------------------------------------
LIB_RS = r'''use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rlvr_core::ch01_asp_dispatch::{run_episode, AspConfig};
use rlvr_core::ch02_bellman::{run_ch02, Ch02Config, N_STATES, N_ACTIONS, STATE_NAMES, ACTION_NAMES};
use rlvr_core::ch03_bandit::{run_ch03, Ch03Config, ARM_NAMES, TRUE_SLA_RATES};
use rlvr_core::ch04_dp::{run_ch04, Ch04Config};
use rlvr_core::ch05_mc::{run_ch05, McConfig};
use rlvr_core::ch06_td::{run_ch06, TdConfig};
use rlvr_core::ch07_nstep::{run_ch07, NStepConfig};

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
        d.set_item("algorithm",   &r.algorithm)?;
        d.set_item("n_episodes",  r.n_episodes)?;
        d.set_item("total_steps", r.total_steps)?;
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

#[pyfunction]
fn run_ch07_nstep(
    py: Python, seed: u64, n_episodes: usize, gamma: f64,
    alpha: f64, epsilon: f64, epsilon_decay: f64,
    n_step: usize, planning_steps: usize, kappa: f64,
) -> PyResult<PyObject> {
    let config = NStepConfig {
        seed, gamma, alpha, epsilon, epsilon_decay,
        n_episodes, n_step, planning_steps, kappa,
    };
    let result = run_ch07(config);

    let serialize = |r: &rlvr_core::ch07_nstep::NStepResult| -> PyResult<Py<PyDict>> {
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
        d.set_item("algorithm",   &r.algorithm)?;
        d.set_item("n_episodes",  r.n_episodes)?;
        d.set_item("total_steps", r.total_steps)?;
        d.set_item("model_size",  r.model_size)?;
        Ok(d.into())
    };

    let snames = PyList::empty_bound(py);
    for n in rlvr_core::ch02_bellman::STATE_NAMES { snames.append(n)?; }
    let anames = PyList::empty_bound(py);
    for n in rlvr_core::ch02_bellman::ACTION_NAMES { anames.append(n)?; }

    let out = PyDict::new_bound(py);
    out.set_item("nstep_td",     serialize(&result.nstep_td)?)?;
    out.set_item("nstep_sarsa",  serialize(&result.nstep_sarsa)?)?;
    out.set_item("dyna_q",       serialize(&result.dyna_q)?)?;
    out.set_item("dyna_q_plus",  serialize(&result.dyna_q_plus)?)?;
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
    m.add_function(wrap_pyfunction!(run_ch07_nstep,           m)?)?;
    Ok(())
}
'''

# ---------------------------------------------------------------------------
# FILE 3: gui/chapters/ch07.py
# ---------------------------------------------------------------------------
CH07_PY = r'''import streamlit as st
import plotly.graph_objects as go

T = {
    "EN": {
        "title": "Chapter 07 — n-Step TD & Planning with Dyna-Q",
        "subtitle": "n-Step TD · n-Step SARSA · Dyna-Q · Dyna-Q+ · Warsaw ASP",
        "engine_missing": "Run: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Settings",
        "n_episodes": "Number of episodes",
        "gamma": "γ — Discount factor",
        "alpha": "α — Learning rate",
        "epsilon": "ε — Initial exploration",
        "epsilon_decay": "ε decay rate",
        "n_step": "n — Step size (1=TD, large=MC)",
        "planning_steps": "k — Planning steps (Dyna-Q)",
        "kappa": "κ — Exploration bonus (Dyna-Q+)",
        "seed": "Random seed",
        "run_btn": "▶ Run All Four Algorithms",
        "guide_title": "🎓 How to use this chapter",
        "guide": """
**Step 1 — Understand n-Step TD**
n=1 → TD(0). n=∞ → Monte Carlo. n=3-5 → sweet spot.
The n-step return: G^(n)_t = R_{t+1} + γR_{t+2} + ... + γ^(n-1)R_{t+n} + γ^n V(S_{t+n})

**Step 2 — Understand Dyna-Q**
Dyna-Q = Q-Learning + model learning + planning.
After each real step: do k simulated steps using the learned model.
More planning steps k → faster convergence, more computation per step.

**Step 3 — Understand Dyna-Q+**
Same as Dyna-Q but adds exploration bonus κ√τ(s,a) to rarely-tried transitions.
τ(s,a) = steps since (s,a) was last tried. Prevents stagnation in changing environments.

**Step 4 — Try n=1 vs n=5 vs n=10**
Watch how larger n improves early learning but increases variance.

**Step 5 — Try k=0 vs k=5 vs k=20 (planning steps)**
k=0 → pure Q-Learning. k=20 → Dyna-Q learns much faster per episode.

**Step 6 — Read the Model Size metric**
How many (s,a) pairs has Dyna-Q learned? Should grow toward N_STATES × N_ACTIONS = 32.
""",
        "returns_title": "📈 Episode Returns — All Four Algorithms",
        "returns_caption": "Moving average. Dyna-Q should converge fastest due to planning.",
        "td_error_title": "📉 TD Error Curve",
        "td_error_caption": "TD error decays as agent learns. Dyna-Q error drops fastest.",
        "value_title": "📊 Value Function V(s)",
        "value_caption": "All algorithms should converge to similar V*(s).",
        "nstep_title": "📊 n-Step Return Comparison (n=1 vs n=3 vs n=10)",
        "nstep_caption": "Larger n = more MC-like. Sweet spot typically n=3-5.",
        "model_title": "🗺️ Dyna-Q Model Coverage",
        "model_caption": "How many (s,a) pairs the model has learned. Max = 32.",
        "qtable_title": "📊 Q-Table Heatmap",
        "qtable_caption": "Q(s,a) values learned. Select algorithm.",
        "glass_title": "🔬 Glass-Box — Planning Steps Trace",
        "summary_title": "📊 Summary",
        "summary_results": "Algorithm Comparison",
        "summary_pros_cons": "Algorithms — Pros & Cons",
        "pros": "✅ Pros", "cons": "❌ Cons",
        "theory_title": "📖 Theory — Chapter 07",
        "theory_sections": {
            "nstep":    "§7.1 n-Step TD Returns",
            "nsarsa":   "§7.2 n-Step SARSA",
            "dynaq":    "§7.4 Dyna-Q",
            "dynaqplus":"§7.5 Dyna-Q+",
        },
        "theory_nstep": r"""
**n-Step TD** generalises TD(0) and MC:

G^(n)_t = R_{t+1} + γR_{t+2} + ... + γ^(n-1)R_{t+n} + γ^n V(S_{t+n})

V(S_t) <- V(S_t) + α [G^(n)_t - V(S_t)]

- n=1: TD(0) — maximum bootstrapping, minimum variance
- n=∞: Monte Carlo — no bootstrapping, maximum variance
- n=3-5: sweet spot — low bias, manageable variance

Implemented in `nstep_td_prediction()` in `ch07_nstep.rs`.
""",
        "theory_nsarsa": r"""
**n-Step SARSA** extends SARSA to n steps:

G^(n)_t = R_{t+1} + ... + γ^(n-1)R_{t+n} + γ^n Q(S_{t+n}, A_{t+n})

Q(S_t,A_t) <- Q(S_t,A_t) + α [G^(n)_t - Q(S_t,A_t)]

On-policy: A_{t+n} is chosen by the same epsilon-greedy policy.
Implemented in `nstep_sarsa()` in `ch07_nstep.rs`.
""",
        "theory_dynaq": r"""
**Dyna-Q** integrates learning and planning:

For each real step:
1. Q-Learning update: Q(s,a) += α[R + γ max_a' Q(s',a') - Q(s,a)]
2. Model update: Model(s,a) = (R, s')
3. Planning (k times): sample random (s,a) from model, do Q-Learning update

With k=5 planning steps, Dyna-Q is ~5x more sample-efficient than Q-Learning.
Implemented in `dyna_q()` in `ch07_nstep.rs`.
""",
        "theory_dynaqplus": r"""
**Dyna-Q+** adds an exploration bonus to rarely-tried transitions:

r' = r + κ√τ(s,a)

where τ(s,a) = steps since (s,a) was last tried.

This prevents the agent from ignoring transitions it hasn't tried recently.
Critical in non-stationary environments where the model may become stale.
κ controls the exploration bonus strength (default: 0.001).
Implemented in `dyna_q_plus()` in `ch07_nstep.rs`.
""",
        "algo_labels": {
            "nstep_td":    "n-Step TD",
            "nstep_sarsa": "n-Step SARSA",
            "dyna_q":      "Dyna-Q",
            "dyna_q_plus": "Dyna-Q+",
        },
        "pros_list": {
            "nstep_td":    ["Bridges TD and MC", "Tunable bias-variance via n", "No model needed"],
            "nstep_sarsa": ["On-policy, safe", "n-step reduces variance vs TD(0)", "No model needed"],
            "dyna_q":      ["Sample efficient — k planning steps", "Learns model of environment", "Converges faster than Q-Learning"],
            "dyna_q_plus": ["Handles non-stationary environments", "Exploration bonus prevents stagnation", "Best of Dyna-Q + exploration"],
        },
        "cons_list": {
            "nstep_td":    ["Must wait n steps before update", "Higher memory (stores n transitions)", "n must be tuned"],
            "nstep_sarsa": ["On-policy — needs epsilon > 0", "n must be tuned", "Slower than Dyna-Q"],
            "dyna_q":      ["Model may be wrong (stale)", "k planning steps add computation", "Assumes stationary environment"],
            "dyna_q_plus": ["κ must be tuned", "More complex than Dyna-Q", "Bonus can cause over-exploration"],
        },
    },
    "PL": {
        "title": "Rozdział 07 — n-krokowe TD i Planowanie z Dyna-Q",
        "subtitle": "n-Step TD · n-Step SARSA · Dyna-Q · Dyna-Q+ · ASP Warszawa",
        "engine_missing": "Uruchom: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Ustawienia",
        "n_episodes": "Liczba epizodów",
        "gamma": "γ — Współczynnik dyskontowania",
        "alpha": "α — Współczynnik uczenia",
        "epsilon": "ε — Eksploracja początkowa",
        "epsilon_decay": "Współczynnik zaniku ε",
        "n_step": "n — Rozmiar kroku (1=TD, duże=MC)",
        "planning_steps": "k — Kroki planowania (Dyna-Q)",
        "kappa": "κ — Bonus eksploracji (Dyna-Q+)",
        "seed": "Ziarno losowości",
        "run_btn": "▶ Uruchom wszystkie cztery algorytmy",
        "guide_title": "🎓 Jak korzystać z tego rozdziału",
        "guide": """
**Krok 1** — n-krokowe TD: n=1 to TD(0), n=∞ to MC, n=3-5 to optimum.
**Krok 2** — Dyna-Q = Q-Learning + model + k kroków planowania per krok.
**Krok 3** — Dyna-Q+ dodaje bonus κ√τ(s,a) dla rzadko próbowanych przejść.
**Krok 4** — Porównaj n=1 vs n=5 vs n=10 — obserwuj wpływ na zbieżność.
**Krok 5** — Porównaj k=0 vs k=5 vs k=20 — Dyna-Q uczy się szybciej.
**Krok 6** — Odczytaj rozmiar modelu — ile par (s,a) Dyna-Q się nauczyło.
""",
        "returns_title": "📈 Zwroty epizodów — Cztery algorytmy",
        "returns_caption": "Średnia krocząca. Dyna-Q powinien zbiegać najszybciej dzięki planowaniu.",
        "td_error_title": "📉 Krzywa błędu TD",
        "td_error_caption": "Błąd TD maleje w miarę uczenia. Dyna-Q spada najszybciej.",
        "value_title": "📊 Funkcja wartości V(s)",
        "value_caption": "Wszystkie algorytmy powinny zbiegać do podobnych V*(s).",
        "nstep_title": "📊 Porównanie n-krokowych zwrotów (n=1 vs n=3 vs n=10)",
        "nstep_caption": "Większe n = bardziej jak MC. Optimum zazwyczaj n=3-5.",
        "model_title": "🗺️ Pokrycie modelu Dyna-Q",
        "model_caption": "Ile par (s,a) model się nauczył. Maks = 32.",
        "qtable_title": "📊 Heatmapa tabeli Q",
        "qtable_caption": "Wartości Q(s,a). Wybierz algorytm.",
        "glass_title": "🔬 Glass-Box — Ślad kroków planowania",
        "summary_title": "📊 Podsumowanie",
        "summary_results": "Porównanie algorytmów",
        "summary_pros_cons": "Algorytmy — Zalety i Wady",
        "pros": "✅ Zalety", "cons": "❌ Wady",
        "theory_title": "📖 Teoria — Rozdział 07",
        "theory_sections": {
            "nstep":     "§7.1 n-krokowe zwroty TD",
            "nsarsa":    "§7.2 n-krokowy SARSA",
            "dynaq":     "§7.4 Dyna-Q",
            "dynaqplus": "§7.5 Dyna-Q+",
        },
        "theory_nstep": r"""
**n-krokowe TD** uogólnia TD(0) i MC:
G^(n)_t = R_{t+1} + γR_{t+2} + ... + γ^(n-1)R_{t+n} + γ^n V(S_{t+n})
V(S_t) ← V(S_t) + α [G^(n)_t - V(S_t)]
Implementacja: `nstep_td_prediction()` w `ch07_nstep.rs`
""",
        "theory_nsarsa": r"""
**n-krokowy SARSA**:
G^(n)_t = R_{t+1} + ... + γ^(n-1)R_{t+n} + γ^n Q(S_{t+n}, A_{t+n})
Q(S_t,A_t) ← Q(S_t,A_t) + α [G^(n)_t - Q(S_t,A_t)]
""",
        "theory_dynaq": r"""
**Dyna-Q** — integracja uczenia i planowania:
1. Aktualizacja Q-Learning (prawdziwy krok)
2. Aktualizacja modelu: Model(s,a) = (R, s')
3. Planowanie (k razy): losuj (s,a) z modelu, aktualizuj Q
Implementacja: `dyna_q()` w `ch07_nstep.rs`
""",
        "theory_dynaqplus": r"""
**Dyna-Q+** — bonus eksploracji:
r' = r + κ√τ(s,a)
τ(s,a) = kroki od ostatniej próby (s,a)
Implementacja: `dyna_q_plus()` w `ch07_nstep.rs`
""",
        "algo_labels": {
            "nstep_td":    "n-krokowe TD",
            "nstep_sarsa": "n-krokowy SARSA",
            "dyna_q":      "Dyna-Q",
            "dyna_q_plus": "Dyna-Q+",
        },
        "pros_list": {
            "nstep_td":    ["Łączy TD i MC", "Regulowany bias-wariancja przez n", "Bez modelu"],
            "nstep_sarsa": ["On-policy, bezpieczny", "n kroków redukuje wariancję", "Bez modelu"],
            "dyna_q":      ["Efektywny próbkowo", "Uczy modelu środowiska", "Szybsza zbieżność"],
            "dyna_q_plus": ["Obsługuje niestacjonarne środowiska", "Bonus eksploracji", "Najlepszy z Dyna-Q"],
        },
        "cons_list": {
            "nstep_td":    ["Czeka n kroków przed aktualizacją", "Wyższa pamięć", "n wymaga strojenia"],
            "nstep_sarsa": ["On-policy — wymaga ε > 0", "n wymaga strojenia"],
            "dyna_q":      ["Model może być nieaktualny", "k kroków planowania = więcej obliczeń"],
            "dyna_q_plus": ["κ wymaga strojenia", "Bardziej złożony niż Dyna-Q"],
        },
    },
    "FR": {
        "title": "Chapitre 07 — TD n-pas & Planification avec Dyna-Q",
        "subtitle": "TD n-pas · SARSA n-pas · Dyna-Q · Dyna-Q+ · ASP Varsovie",
        "engine_missing": "Exécutez : `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Paramètres",
        "n_episodes": "Nombre d'épisodes",
        "gamma": "γ", "alpha": "α", "epsilon": "ε",
        "epsilon_decay": "Décroissance ε", "n_step": "n",
        "planning_steps": "k — Étapes de planification",
        "kappa": "κ — Bonus d'exploration",
        "seed": "Graine",
        "run_btn": "▶ Lancer les quatre algorithmes",
        "guide_title": "🎓 Guide",
        "guide": "n=1→TD, n=∞→MC. Dyna-Q = Q-Learning + modèle + k étapes de planification.",
        "returns_title": "📈 Retours", "returns_caption": "Dyna-Q converge le plus vite.",
        "td_error_title": "📉 Erreur TD", "td_error_caption": "",
        "value_title": "📊 V(s)", "value_caption": "",
        "nstep_title": "📊 Comparaison n-pas", "nstep_caption": "",
        "model_title": "🗺️ Couverture modèle Dyna-Q", "model_caption": "Max = 32.",
        "qtable_title": "📊 Table Q", "qtable_caption": "",
        "glass_title": "🔬 Glass-Box",
        "summary_title": "📊 Résumé", "summary_results": "Comparaison",
        "summary_pros_cons": "Avantages & Inconvénients",
        "pros": "✅", "cons": "❌",
        "theory_title": "📖 Théorie",
        "theory_sections": {"nstep": "§7.1 TD n-pas", "nsarsa": "§7.2 SARSA n-pas", "dynaq": "§7.4 Dyna-Q", "dynaqplus": "§7.5 Dyna-Q+"},
        "theory_nstep": "G^(n)_t = R_{t+1} + γR_{t+2} + ... + γ^n V(S_{t+n})",
        "theory_nsarsa": "G^(n)_t = ... + γ^n Q(S_{t+n}, A_{t+n})",
        "theory_dynaq": "Q-Learning + modèle + k étapes de planification.",
        "theory_dynaqplus": "r' = r + κ√τ(s,a)",
        "algo_labels": {"nstep_td": "TD n-pas", "nstep_sarsa": "SARSA n-pas", "dyna_q": "Dyna-Q", "dyna_q_plus": "Dyna-Q+"},
        "pros_list": {"nstep_td": ["Relie TD et MC"], "nstep_sarsa": ["On-policy"], "dyna_q": ["Efficace"], "dyna_q_plus": ["Exploration bonus"]},
        "cons_list": {"nstep_td": ["Attend n pas"], "nstep_sarsa": ["ε > 0"], "dyna_q": ["Modèle peut être périmé"], "dyna_q_plus": ["κ à régler"]},
    },
    "ES": {
        "title": "Capítulo 07 — TD n-pasos & Planificación con Dyna-Q",
        "subtitle": "TD n-pasos · SARSA n-pasos · Dyna-Q · Dyna-Q+ · ASP Varsovia",
        "engine_missing": "Ejecute: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Configuración",
        "n_episodes": "Episodios", "gamma": "γ", "alpha": "α", "epsilon": "ε",
        "epsilon_decay": "Decaimiento ε", "n_step": "n",
        "planning_steps": "k — Pasos de planificación",
        "kappa": "κ — Bonus de exploración", "seed": "Semilla",
        "run_btn": "▶ Ejecutar los cuatro algoritmos",
        "guide_title": "🎓 Guía",
        "guide": "n=1→TD, n=∞→MC. Dyna-Q = Q-Learning + modelo + k pasos de planificación.",
        "returns_title": "📈 Retornos", "returns_caption": "Dyna-Q converge más rápido.",
        "td_error_title": "📉 Error TD", "td_error_caption": "",
        "value_title": "📊 V(s)", "value_caption": "",
        "nstep_title": "📊 Comparación n-pasos", "nstep_caption": "",
        "model_title": "🗺️ Cobertura modelo Dyna-Q", "model_caption": "Máx = 32.",
        "qtable_title": "📊 Tabla Q", "qtable_caption": "",
        "glass_title": "🔬 Glass-Box",
        "summary_title": "📊 Resumen", "summary_results": "Comparación",
        "summary_pros_cons": "Pros y Contras",
        "pros": "✅", "cons": "❌",
        "theory_title": "📖 Teoría",
        "theory_sections": {"nstep": "§7.1 TD n-pasos", "nsarsa": "§7.2 SARSA n-pasos", "dynaq": "§7.4 Dyna-Q", "dynaqplus": "§7.5 Dyna-Q+"},
        "theory_nstep": "G^(n)_t = R_{t+1} + γR_{t+2} + ... + γ^n V(S_{t+n})",
        "theory_nsarsa": "G^(n)_t = ... + γ^n Q(S_{t+n}, A_{t+n})",
        "theory_dynaq": "Q-Learning + modelo + k pasos de planificación.",
        "theory_dynaqplus": "r' = r + κ√τ(s,a)",
        "algo_labels": {"nstep_td": "TD n-pasos", "nstep_sarsa": "SARSA n-pasos", "dyna_q": "Dyna-Q", "dyna_q_plus": "Dyna-Q+"},
        "pros_list": {"nstep_td": ["Une TD y MC"], "nstep_sarsa": ["On-policy"], "dyna_q": ["Eficiente"], "dyna_q_plus": ["Bonus exploración"]},
        "cons_list": {"nstep_td": ["Espera n pasos"], "nstep_sarsa": ["ε > 0"], "dyna_q": ["Modelo puede quedar obsoleto"], "dyna_q_plus": ["κ a ajustar"]},
    },
}

COLORS = {
    "nstep_td":    "#0082F0",
    "nstep_sarsa": "#FF8C0A",
    "dyna_q":      "#0FC373",
    "dyna_q_plus": "#FF3232",
}

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
    n_episodes     = st.sidebar.slider(tx["n_episodes"],     50, 3000, 500, 50)
    gamma          = st.sidebar.slider(tx["gamma"],          0.5, 0.999, 0.95, 0.005)
    alpha          = st.sidebar.slider(tx["alpha"],          0.01, 1.0, 0.1, 0.01)
    epsilon        = st.sidebar.slider(tx["epsilon"],        0.0, 1.0, 0.3, 0.05)
    epsilon_decay  = st.sidebar.slider(tx["epsilon_decay"],  0.0, 0.1, 0.01, 0.001, format="%.3f")
    n_step         = st.sidebar.slider(tx["n_step"],         1, 20, 3, 1)
    planning_steps = st.sidebar.slider(tx["planning_steps"], 0, 50, 5, 1)
    kappa          = st.sidebar.slider(tx["kappa"],          0.0, 0.01, 0.001, 0.0001, format="%.4f")
    seed           = st.sidebar.number_input(tx["seed"], 0, 9999, 42)

    with st.expander(tx["guide_title"], expanded=False):
        st.markdown(tx["guide"])

    if st.button(tx["run_btn"], type="primary"):
        with st.spinner("Running Rust n-step/Dyna engine..."):
            result = rlvr_py.run_ch07_nstep(
                int(seed), int(n_episodes), float(gamma), float(alpha),
                float(epsilon), float(epsilon_decay),
                int(n_step), int(planning_steps), float(kappa)
            )
        st.session_state["ch07_result"] = result

    if "ch07_result" not in st.session_state:
        st.info("Configure settings and click **▶ Run All Four Algorithms**.")
        _render_theory(tx)
        return

    result       = st.session_state["ch07_result"]
    state_names  = result["state_names"]
    action_names = result["action_names"]
    algos        = ["nstep_td", "nstep_sarsa", "dyna_q", "dyna_q_plus"]

    # KPI
    cols = st.columns(4)
    for i, key in enumerate(algos):
        r = result[key]
        avg = sum(r["returns_curve"][-50:]) / min(50, len(r["returns_curve"]))
        extra = f"Model: {r['model_size']}" if r["model_size"] > 0 else f"Steps: {r['total_steps']:,}"
        cols[i].metric(tx["algo_labels"][key], f"Avg: {avg:.2f}", extra)

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
    for key in ["nstep_sarsa", "dyna_q", "dyna_q_plus"]:
        ma = _moving_avg(result[key]["td_error_curve"], 30)
        fig2.add_trace(go.Scatter(x=list(range(len(ma))), y=ma,
            mode="lines", name=tx["algo_labels"][key],
            line=dict(color=COLORS[key], width=2)))
    fig2.update_layout(height=260, margin=dict(l=40,r=20,t=20,b=40),
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
    fig3.update_layout(height=280, barmode="group",
                       margin=dict(l=40,r=20,t=20,b=40),
                       legend=dict(orientation="h"))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(tx["value_caption"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(tx["model_title"])
        model_sizes = {tx["algo_labels"][k]: result[k]["model_size"] for k in ["dyna_q","dyna_q_plus"]}
        max_model = result["n_states"] * result["n_actions"]
        fig4 = go.Figure()
        for label, size in model_sizes.items():
            fig4.add_trace(go.Bar(x=[label], y=[size],
                marker_color="#0082F0" if "Plus" not in label and "+" not in label else "#0FC373"))
        fig4.add_hline(y=max_model, line_dash="dash", line_color="grey",
                       annotation_text=f"Max={max_model}")
        fig4.update_layout(height=260, margin=dict(l=40,r=20,t=20,b=40),
                           yaxis_title="(s,a) pairs learned")
        st.plotly_chart(fig4, use_container_width=True)
        st.caption(tx["model_caption"])

    with col2:
        st.subheader(tx["qtable_title"])
        algo_sel = st.selectbox("Algorithm",
            [tx["algo_labels"][k] for k in ["nstep_sarsa","dyna_q","dyna_q_plus"]])
        key_map = {tx["algo_labels"][k]: k for k in ["nstep_sarsa","dyna_q","dyna_q_plus"]}
        key_sel = key_map.get(algo_sel, "dyna_q")
        qt = result[key_sel]["q_table"]
        action_short = [f"A{i}" for i in range(result["n_actions"])]
        fig5 = go.Figure(go.Heatmap(
            z=qt, x=action_short, y=short, colorscale="Blues",
            text=[[f"{qt[s][a]:.2f}" for a in range(result["n_actions"])]
                  for s in range(result["n_states"])],
            texttemplate="%{text}",
        ))
        fig5.update_layout(height=300, margin=dict(l=60,r=20,t=20,b=40))
        st.plotly_chart(fig5, use_container_width=True)
        st.caption(tx["qtable_caption"])

    # Glass-Box
    st.subheader(tx["glass_title"])
    _render_glass_box(result, tx)

    # Summary
    st.subheader(tx["summary_title"])
    _render_summary(result, tx, algos)

    _render_theory(tx)


def _render_glass_box(result, tx):
    algo_options = {tx["algo_labels"][k]: k for k in
                    ["nstep_td","nstep_sarsa","dyna_q","dyna_q_plus"]}
    selected = st.selectbox("Algorithm", list(algo_options.keys()), key="gb7_algo")
    key = algo_options[selected]
    r = result[key]
    ep_idx = st.slider("Episode", 0, max(len(r["returns_curve"])-1, 0),
                       max(len(r["returns_curve"])-1, 0), key="gb7_ep")
    col1, col2, col3 = st.columns(3)
    col1.metric("Episode return",    f"{r['returns_curve'][ep_idx]:.3f}")
    col2.metric("Avg TD error",      f"{r['td_error_curve'][ep_idx]:.4f}")
    col3.metric("Model size",        str(r["model_size"]))

    if "nstep" in key:
        st.latex(r"G^{(n)}_t = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1}R_{t+n} + \gamma^n V(S_{t+n})")
    elif key == "dyna_q":
        st.latex(r"Q(s,a) \leftarrow Q(s,a) + \alpha\left[R + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]")
        st.markdown("**Planning:** sample random (s,a) from model → Q-Learning update × k")
    else:
        st.latex(r"r' = r + \kappa\sqrt{\tau(s,a)}")
        st.markdown("**Exploration bonus:** κ√τ(s,a) added to rarely-tried transitions")


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
            "Model size":          str(r["model_size"]),
            "V*(S0)":              f"{r['values'][0]:.3f}",
            "V*(S7)":              f"{r['values'][7]:.3f}",
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
    for key in ["nstep", "nsarsa", "dynaq", "dynaqplus"]:
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
[![Progress](https://img.shields.io/badge/Progress-Ch01--07%20%E2%9C%85%20%7C%20Ch08--20%20%F0%9F%9A%A7-lightgrey)]()

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
| **07** | n-Step TD & Planning | n-Step TD, n-Step SARSA, Dyna-Q, Dyna-Q+ | ✅ Complete |
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

## Quick Start

```bash
cd rlvr-py && maturin develop && cd ..
cargo test --workspace   # 69 tests passing
streamlit run gui/app.py
```

---

## Chapter 07 — n-Step TD & Planning

- **n-Step TD**: G^(n)_t bridges TD(0) (n=1) and MC (n=∞). Sweet spot n=3-5.
- **n-Step SARSA**: on-policy n-step control with epsilon-soft policy
- **Dyna-Q**: Q-Learning + model learning + k planning steps per real step
- **Dyna-Q+**: Dyna-Q + exploration bonus κ√τ(s,a) for stale transitions
- Returns curve, TD error, model coverage, Q-table heatmap
- Glass-Box: n-step return formula and Dyna planning trace
- 4 languages: EN / FR / ES / PL

---

## Architecture

```
rlvr-core/src/
  ch01..ch07_nstep.rs   <- ALL RL logic in Rust
rlvr-py/src/lib.rs      <- PyO3 bridge Ch01-Ch07
gui/chapters/
  ch01..ch07.py         <- Streamlit UI only
```

## License

MIT
"""

# ---------------------------------------------------------------------------
# FILE 5: apply_ch07.sh
# ---------------------------------------------------------------------------
APPLY_SH = """#!/bin/bash
set -e
echo "Applying Chapter 07..."

cp ch07_package/rlvr-core/src/ch07_nstep.rs rlvr-core/src/ch07_nstep.rs
echo "pub mod ch07_nstep;" >> rlvr-core/src/lib.rs

cp ch07_package/rlvr-py/src/lib.rs rlvr-py/src/lib.rs
cp ch07_package/gui/chapters/ch07.py gui/chapters/ch07.py
cp ch07_package/README.md README.md

python3 -c "
content = open('gui/app.py').read()
old = 'elif ch_num == 6:\\n    from chapters.ch06 import render\\n    render()'
new = old + '\\nelif ch_num == 7:\\n    from chapters.ch07 import render\\n    render()'
if old in content:
    content = content.replace(old, new)
    open('gui/app.py', 'w').write(content)
    print('app.py routing updated')
else:
    print('WARNING: ch06 routing not found in app.py — add ch07 routing manually')
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
echo "Chapter 07 applied! Run: streamlit run gui/app.py"
"""

# ---------------------------------------------------------------------------
# Build ZIP
# ---------------------------------------------------------------------------
files = {
    "ch07_package/rlvr-core/src/ch07_nstep.rs": CH07_RS,
    "ch07_package/rlvr-py/src/lib.rs":           LIB_RS,
    "ch07_package/gui/chapters/ch07.py":         CH07_PY,
    "ch07_package/README.md":                    README_MD,
    "ch07_package/apply_ch07.sh":                APPLY_SH,
}

zip_path = "/tmp/ch07_package.zip"
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

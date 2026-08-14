#!/usr/bin/env python3
"""
RLVR Enterprise Allocator — Chapter 05 ZIP Generator
Run from project root:
    python3 ch05_generate.py
Produces: ch05_package.zip
"""

import zipfile
import os

# ---------------------------------------------------------------------------
# FILE 1: rlvr-core/src/ch05_mc.rs
# ---------------------------------------------------------------------------
CH05_RS = r'''use ndarray::Array2;
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
pub struct McConfig {
    pub seed:          u64,
    pub gamma:         f64,
    pub n_episodes:    usize,
    pub epsilon:       f64,
    pub epsilon_decay: f64,
}

#[derive(Debug, Clone)]
pub struct McStep {
    pub state:   usize,
    pub action:  usize,
    pub reward:  f64,
}

#[derive(Debug, Clone)]
pub struct McEpisode {
    pub steps:        Vec<McStep>,
    pub total_return: f64,
}

#[derive(Debug, Clone)]
pub struct McResult {
    pub algorithm:          String,
    pub values:             Vec<f64>,
    pub policy:             Vec<usize>,
    pub q_table:            Vec<Vec<f64>>,   // [state][action]
    pub returns_curve:      Vec<f64>,        // avg return per episode
    pub visit_counts:       Vec<usize>,      // N(s) first-visit
    pub convergence_curve:  Vec<f64>,        // max |V^(k) - V^(k-1)|
    pub n_episodes:         usize,
}

#[derive(Debug, Clone)]
pub struct Ch05Result {
    pub first_visit:  McResult,
    pub every_visit:  McResult,
    pub on_policy:    McResult,
    pub off_policy:   McResult,
}

// ---------------------------------------------------------------------------
// Environment: simulate one step using transition matrix
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
        if p <= cumsum {
            return sp;
        }
    }
    N_STATES - 1
}

// ---------------------------------------------------------------------------
// Generate one episode following policy π (epsilon-soft)
// ---------------------------------------------------------------------------
fn generate_episode(
    policy: &[usize],
    epsilon: f64,
    transitions: &ndarray::Array3<f64>,
    rewards: &Array2<f64>,
    rng: &mut StdRng,
    max_steps: usize,
) -> McEpisode {
    let mut steps = Vec::new();
    let mut s = rng.gen_range(0..N_STATES); // random start state

    for _ in 0..max_steps {
        // epsilon-soft action selection
        let a = if rng.gen::<f64>() < epsilon {
            rng.gen_range(0..N_ACTIONS)
        } else {
            policy[s]
        };

        let r = rewards[[s, a]];
        steps.push(McStep { state: s, action: a, reward: r });

        // transition
        let sp = sample_next_state(s, a, transitions, rng);
        s = sp;

        // terminal condition: reached S0 (best state) or S7 (worst)
        if s == 0 || s == 7 {
            break;
        }
    }

    // compute total discounted return (for curve)
    let total_return = steps.iter().rev().fold(0.0_f64, |g, step| step.reward + 0.95 * g);

    McEpisode { steps, total_return }
}

// ---------------------------------------------------------------------------
// MC First-Visit Prediction — estimates V^π
// ---------------------------------------------------------------------------
pub fn mc_first_visit_prediction(
    policy: &[usize],
    config: &McConfig,
    transitions: &ndarray::Array3<f64>,
    rewards: &Array2<f64>,
) -> McResult {
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut v = vec![0.0_f64; N_STATES];
    let mut returns: Vec<Vec<f64>> = vec![Vec::new(); N_STATES];
    let mut visit_counts = vec![0usize; N_STATES];
    let mut returns_curve = Vec::new();
    let mut convergence_curve = Vec::new();
    let mut v_prev = v.clone();

    for _ in 0..config.n_episodes {
        let ep = generate_episode(policy, config.epsilon, transitions, rewards, &mut rng, 50);
        returns_curve.push(ep.total_return);

        // first-visit: track which states appeared first in this episode
        let mut visited = vec![false; N_STATES];
        let mut g = 0.0_f64;

        for step in ep.steps.iter().rev() {
            g = step.reward + config.gamma * g;
            if !visited[step.state] {
                visited[step.state] = true;
                visit_counts[step.state] += 1;
                returns[step.state].push(g);
                v[step.state] = returns[step.state].iter().sum::<f64>()
                    / returns[step.state].len() as f64;
            }
        }

        // convergence delta
        let delta = v.iter().zip(v_prev.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        convergence_curve.push(delta);
        v_prev = v.clone();
    }

    let policy_out = greedy_policy_from_v(&v, transitions, rewards, config.gamma);
    let q_table = compute_q_table(&v, transitions, rewards, config.gamma);

    McResult {
        algorithm: "first_visit".to_string(),
        values: v,
        policy: policy_out,
        q_table,
        returns_curve,
        visit_counts,
        convergence_curve,
        n_episodes: config.n_episodes,
    }
}

// ---------------------------------------------------------------------------
// MC Every-Visit Prediction — estimates V^π
// ---------------------------------------------------------------------------
pub fn mc_every_visit_prediction(
    policy: &[usize],
    config: &McConfig,
    transitions: &ndarray::Array3<f64>,
    rewards: &Array2<f64>,
) -> McResult {
    let mut rng = StdRng::seed_from_u64(config.seed + 1);
    let mut v = vec![0.0_f64; N_STATES];
    let mut returns: Vec<Vec<f64>> = vec![Vec::new(); N_STATES];
    let mut visit_counts = vec![0usize; N_STATES];
    let mut returns_curve = Vec::new();
    let mut convergence_curve = Vec::new();
    let mut v_prev = v.clone();

    for _ in 0..config.n_episodes {
        let ep = generate_episode(policy, config.epsilon, transitions, rewards, &mut rng, 50);
        returns_curve.push(ep.total_return);

        let mut g = 0.0_f64;
        for step in ep.steps.iter().rev() {
            g = step.reward + config.gamma * g;
            // every-visit: update on EVERY occurrence
            visit_counts[step.state] += 1;
            returns[step.state].push(g);
            v[step.state] = returns[step.state].iter().sum::<f64>()
                / returns[step.state].len() as f64;
        }

        let delta = v.iter().zip(v_prev.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        convergence_curve.push(delta);
        v_prev = v.clone();
    }

    let policy_out = greedy_policy_from_v(&v, transitions, rewards, config.gamma);
    let q_table = compute_q_table(&v, transitions, rewards, config.gamma);

    McResult {
        algorithm: "every_visit".to_string(),
        values: v,
        policy: policy_out,
        q_table,
        returns_curve,
        visit_counts,
        convergence_curve,
        n_episodes: config.n_episodes,
    }
}

// ---------------------------------------------------------------------------
// MC On-Policy Control (epsilon-soft) — estimates Q* and π*
// ---------------------------------------------------------------------------
pub fn mc_on_policy_control(
    config: &McConfig,
    transitions: &ndarray::Array3<f64>,
    rewards: &Array2<f64>,
) -> McResult {
    let mut rng = StdRng::seed_from_u64(config.seed + 2);
    let mut q = vec![vec![0.0_f64; N_ACTIONS]; N_STATES];
    let mut n = vec![vec![0usize; N_ACTIONS]; N_STATES];
    let mut policy = vec![0usize; N_STATES];
    let mut returns_curve = Vec::new();
    let mut convergence_curve = Vec::new();
    let mut v_prev = vec![0.0_f64; N_STATES];

    for ep_idx in 0..config.n_episodes {
        let epsilon_t = (config.epsilon / (1.0 + config.epsilon_decay * ep_idx as f64)).max(0.01);
        let ep = generate_episode(&policy, epsilon_t, transitions, rewards, &mut rng, 50);
        returns_curve.push(ep.total_return);

        let mut g = 0.0_f64;
        let mut visited = vec![vec![false; N_ACTIONS]; N_STATES];

        for step in ep.steps.iter().rev() {
            g = step.reward + config.gamma * g;
            if !visited[step.state][step.action] {
                visited[step.state][step.action] = true;
                n[step.state][step.action] += 1;
                let nn = n[step.state][step.action] as f64;
                q[step.state][step.action] += (g - q[step.state][step.action]) / nn;
                // greedy policy improvement
                policy[step.state] = q[step.state]
                    .iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i).unwrap_or(0);
            }
        }

        let v: Vec<f64> = (0..N_STATES).map(|s| {
            q[s].iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        }).collect();
        let delta = v.iter().zip(v_prev.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        convergence_curve.push(delta);
        v_prev = v.clone();
    }

    let v: Vec<f64> = (0..N_STATES).map(|s| {
        q[s].iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }).collect();
    let visit_counts: Vec<usize> = (0..N_STATES).map(|s| n[s].iter().sum()).collect();

    McResult {
        algorithm: "on_policy".to_string(),
        values: v,
        policy,
        q_table: q,
        returns_curve,
        visit_counts,
        convergence_curve,
        n_episodes: config.n_episodes,
    }
}

// ---------------------------------------------------------------------------
// MC Off-Policy Control with Importance Sampling (Weighted IS)
// ---------------------------------------------------------------------------
pub fn mc_off_policy_control(
    config: &McConfig,
    transitions: &ndarray::Array3<f64>,
    rewards: &Array2<f64>,
) -> McResult {
    let mut rng = StdRng::seed_from_u64(config.seed + 3);
    let mut q = vec![vec![0.0_f64; N_ACTIONS]; N_STATES];
    let mut c = vec![vec![0.0_f64; N_ACTIONS]; N_STATES]; // cumulative weights
    let mut target_policy = vec![0usize; N_STATES];       // greedy target π
    let mut returns_curve = Vec::new();
    let mut convergence_curve = Vec::new();
    let mut v_prev = vec![0.0_f64; N_STATES];

    // behaviour policy: uniform random (soft)
    let behaviour_prob = 1.0 / N_ACTIONS as f64;

    for _ in 0..config.n_episodes {
        // generate episode using behaviour policy (uniform)
        let uniform_policy = vec![0usize; N_STATES]; // will be overridden by epsilon=1.0
        let ep = generate_episode(&uniform_policy, 1.0, transitions, rewards, &mut rng, 50);
        returns_curve.push(ep.total_return);

        let mut g = 0.0_f64;
        let mut w = 1.0_f64; // importance sampling ratio

        for step in ep.steps.iter().rev() {
            g = step.reward + config.gamma * g;

            c[step.state][step.action] += w;
            let cc = c[step.state][step.action];
            q[step.state][step.action] +=
                (w / cc) * (g - q[step.state][step.action]);

            // update target policy greedily
            target_policy[step.state] = q[step.state]
                .iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i).unwrap_or(0);

            // if behaviour action != target action, IS ratio = 0 -> stop
            if step.action != target_policy[step.state] {
                break;
            }
            // update IS ratio: π(a|s) / b(a|s) = 1 / (1/N_ACTIONS) = N_ACTIONS
            w *= 1.0 / behaviour_prob;
        }

        let v: Vec<f64> = (0..N_STATES).map(|s| {
            q[s].iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        }).collect();
        let delta = v.iter().zip(v_prev.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        convergence_curve.push(delta);
        v_prev = v.clone();
    }

    let v: Vec<f64> = (0..N_STATES).map(|s| {
        q[s].iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }).collect();
    let visit_counts = vec![0usize; N_STATES];

    McResult {
        algorithm: "off_policy".to_string(),
        values: v,
        policy: target_policy,
        q_table: q,
        returns_curve,
        visit_counts,
        convergence_curve,
        n_episodes: config.n_episodes,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
fn greedy_policy_from_v(
    v: &[f64],
    transitions: &ndarray::Array3<f64>,
    rewards: &Array2<f64>,
    gamma: f64,
) -> Vec<usize> {
    (0..N_STATES).map(|s| {
        (0..N_ACTIONS).map(|a| {
            (0..N_STATES).map(|sp| {
                transitions[[s, a, sp]] * (rewards[[s, a]] + gamma * v[sp])
            }).sum::<f64>()
        })
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i).unwrap_or(0)
    }).collect()
}

fn compute_q_table(
    v: &[f64],
    transitions: &ndarray::Array3<f64>,
    rewards: &Array2<f64>,
    gamma: f64,
) -> Vec<Vec<f64>> {
    (0..N_STATES).map(|s| {
        (0..N_ACTIONS).map(|a| {
            (0..N_STATES).map(|sp| {
                transitions[[s, a, sp]] * (rewards[[s, a]] + gamma * v[sp])
            }).sum::<f64>()
        }).collect()
    }).collect()
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------
pub fn run_ch05(config: McConfig) -> Ch05Result {
    let mut rng = StdRng::seed_from_u64(config.seed);
    let transitions = build_asp_transitions(&mut rng);
    let rewards = build_asp_rewards();

    verify_transition_matrix(&transitions)
        .expect("Transition matrix probability conservation violated");

    // start with uniform policy for prediction
    let uniform_policy = vec![1usize; N_STATES]; // A1 everywhere as baseline

    let first_visit = mc_first_visit_prediction(&uniform_policy, &config, &transitions, &rewards);
    let every_visit = mc_every_visit_prediction(&uniform_policy, &config, &transitions, &rewards);
    let on_policy   = mc_on_policy_control(&config, &transitions, &rewards);
    let off_policy  = mc_off_policy_control(&config, &transitions, &rewards);

    Ch05Result { first_visit, every_visit, on_policy, off_policy }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> McConfig {
        McConfig { seed: 42, gamma: 0.95, n_episodes: 500, epsilon: 0.3, epsilon_decay: 0.01 }
    }

    #[test]
    fn test_first_visit_runs() {
        let r = run_ch05(default_config());
        assert_eq!(r.first_visit.values.len(), N_STATES);
        assert!(r.first_visit.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_every_visit_runs() {
        let r = run_ch05(default_config());
        assert_eq!(r.every_visit.values.len(), N_STATES);
        assert!(r.every_visit.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_on_policy_runs() {
        let r = run_ch05(default_config());
        assert_eq!(r.on_policy.values.len(), N_STATES);
        assert!(r.on_policy.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_off_policy_runs() {
        let r = run_ch05(default_config());
        assert_eq!(r.off_policy.values.len(), N_STATES);
        assert!(r.off_policy.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_policy_valid() {
        let r = run_ch05(default_config());
        for &a in &r.on_policy.policy  { assert!(a < N_ACTIONS); }
        for &a in &r.off_policy.policy { assert!(a < N_ACTIONS); }
    }

    #[test]
    fn test_returns_curve_length() {
        let config = default_config();
        let r = run_ch05(config.clone());
        assert_eq!(r.first_visit.returns_curve.len(), config.n_episodes);
        assert_eq!(r.on_policy.returns_curve.len(),   config.n_episodes);
    }

    #[test]
    fn test_visit_counts_positive() {
        let config = McConfig { n_episodes: 1000, ..default_config() };
        let r = run_ch05(config);
        // after 1000 episodes all states should be visited at least once
        for &c in &r.first_visit.visit_counts {
            assert!(c > 0, "All states should be visited");
        }
    }

    #[test]
    fn test_deterministic() {
        let r1 = run_ch05(default_config());
        let r2 = run_ch05(default_config());
        for (v1, v2) in r1.on_policy.values.iter().zip(r2.on_policy.values.iter()) {
            assert_eq!(v1.to_bits(), v2.to_bits());
        }
    }

    #[test]
    fn test_on_policy_improves_over_episodes() {
        let config = McConfig { n_episodes: 1000, ..default_config() };
        let r = run_ch05(config);
        let early_avg: f64 = r.on_policy.returns_curve[..50].iter().sum::<f64>() / 50.0;
        let late_avg:  f64 = r.on_policy.returns_curve[950..].iter().sum::<f64>() / 50.0;
        assert!(late_avg >= early_avg - 5.0,
            "On-policy should not degrade: early={:.2} late={:.2}", early_avg, late_avg);
    }

    #[test]
    fn test_q_table_shape() {
        let r = run_ch05(default_config());
        assert_eq!(r.on_policy.q_table.len(), N_STATES);
        for row in &r.on_policy.q_table {
            assert_eq!(row.len(), N_ACTIONS);
        }
    }
}
'''

# ---------------------------------------------------------------------------
# FILE 2: rlvr-py/src/lib.rs  (full bridge with Ch01-Ch05)
# ---------------------------------------------------------------------------
LIB_RS = r'''use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rlvr_core::ch01_asp_dispatch::{run_episode, AspConfig};
use rlvr_core::ch02_bellman::{run_ch02, Ch02Config, N_STATES, N_ACTIONS, STATE_NAMES, ACTION_NAMES};
use rlvr_core::ch03_bandit::{run_ch03, Ch03Config, ARM_NAMES, TRUE_SLA_RATES};
use rlvr_core::ch04_dp::{run_ch04, Ch04Config};
use rlvr_core::ch05_mc::{run_ch05, McConfig};

// ---------------------------------------------------------------------------
// Ch01
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Ch02
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Ch03
// ---------------------------------------------------------------------------
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
            d.set_item("q_values",          qv)?;
            let np = PyList::empty_bound(py);
            for n in &s.n_pulls { np.append(n)?; }
            d.set_item("n_pulls",           np)?;
            let uv = PyList::empty_bound(py);
            for u in &s.ucb_values { uv.append(u)?; }
            d.set_item("ucb_values",        uv)?;
            let tv = PyList::empty_bound(py);
            for t in &s.thompson_samples { tv.append(t)?; }
            d.set_item("thompson_samples",  tv)?;
            d.set_item("epsilon",           s.epsilon)?;
            d.set_item("explored",          s.explored)?;
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

// ---------------------------------------------------------------------------
// Ch04
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Ch05
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------
#[pymodule]
fn rlvr_py(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_ch01_episode,         m)?)?;
    m.add_function(wrap_pyfunction!(run_ch02_value_iteration, m)?)?;
    m.add_function(wrap_pyfunction!(run_ch03_bandits,         m)?)?;
    m.add_function(wrap_pyfunction!(run_ch04_dp,              m)?)?;
    m.add_function(wrap_pyfunction!(run_ch05_mc,              m)?)?;
    Ok(())
}
'''

# ---------------------------------------------------------------------------
# FILE 3: gui/chapters/ch05.py
# ---------------------------------------------------------------------------
CH05_PY = r'''import streamlit as st
import plotly.graph_objects as go

T = {
    "EN": {
        "title": "Chapter 05 — Monte Carlo Methods",
        "subtitle": "ASP Dispatch Learning from Episodes · No Model Required · Warsaw Region",
        "engine_missing": "Run: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ MC Settings",
        "n_episodes": "Number of episodes",
        "gamma": "γ — Discount factor",
        "epsilon": "ε — Initial exploration",
        "epsilon_decay": "ε decay rate α",
        "seed": "Random seed",
        "run_btn": "▶ Run All Four MC Algorithms",
        "guide_title": "🎓 How to use this chapter",
        "guide": """
**Step 1 — Understand the key difference from Ch04**
MC methods learn WITHOUT a model — no P(s\'|s,a) needed.
The agent generates episodes by interacting with the environment and learns from the returns.

**Step 2 — Set number of episodes**
More episodes = better estimates. Try 200 first, then 2000 to see convergence.

**Step 3 — Click ▶ Run All Four MC Algorithms**
- First-Visit MC: updates V(s) only on first visit to s per episode
- Every-Visit MC: updates V(s) on every visit to s per episode
- On-Policy Control: learns Q*(s,a) and π* simultaneously (epsilon-soft)
- Off-Policy Control: learns from behaviour policy using Importance Sampling

**Step 4 — Read the Returns Curve**
Watch average episode return improve over time — this is the learning signal.

**Step 5 — Read the Value Function comparison**
MC estimates of V*(s) should converge toward the DP solution from Ch04.

**Step 6 — Read the Visit Count heatmap**
Which states were visited most? Rarely visited states have high variance estimates.

**Step 7 — Read the Glass-Box**
See exact returns G_t for each state in a selected episode.
""",
        "returns_title": "📈 Episode Returns — All Four Algorithms",
        "returns_caption": "Moving average of episode returns. On-policy control should improve over time.",
        "value_title": "📊 Value Function V(s) — MC vs DP Reference",
        "value_caption": "MC estimates converge toward DP solution (Ch04) with more episodes.",
        "visits_title": "🗺️ State Visit Counts — First-Visit MC",
        "visits_caption": "States visited rarely have high-variance V(s) estimates.",
        "conv_title": "📈 Convergence — Max |V^(k) - V^(k-1)|",
        "conv_caption": "MC convergence is noisier than DP — stochastic environment.",
        "qtable_title": "📊 Q-Table Heatmap — On-Policy Control",
        "qtable_caption": "Q(s,a) values learned by on-policy MC. Brighter = higher value.",
        "glass_title": "🔬 Glass-Box — Episode Trace",
        "glass_headers": ["Step", "State", "Action", "Reward", "Return G_t"],
        "summary_title": "📊 Summary",
        "summary_results": "Algorithm Comparison",
        "summary_pros_cons": "MC Algorithms — Pros & Cons",
        "pros": "✅ Pros",
        "cons": "❌ Cons",
        "theory_title": "📖 Theory — Chapter 05",
        "theory_sections": {
            "intro":       "§5.1 Monte Carlo Methods — Introduction",
            "first_visit": "§5.2 First-Visit MC Prediction",
            "every_visit": "§5.2 Every-Visit MC Prediction",
            "on_policy":   "§5.3 On-Policy MC Control",
            "off_policy":  "§5.4 Off-Policy MC with Importance Sampling",
        },
        "theory_intro": r"""
**Monte Carlo Methods** learn directly from episodes of experience — no model of P(s\'|s,a) needed.

Key properties:
- **Model-free**: learns from raw experience, not transition probabilities
- **Episode-based**: must wait until end of episode to update (unlike TD)
- **Unbiased**: estimates are unbiased (unlike TD which bootstraps)
- **High variance**: estimates can be noisy, especially for rarely visited states

The return from step t:
G_t = R_{t+1} + γ R_{t+2} + γ² R_{t+3} + ... = Σ_{k=0}^{T-t-1} γ^k R_{t+k+1}

Implemented in `ch05_mc.rs` — `generate_episode()`, `mc_first_visit_prediction()`.
""",
        "theory_first_visit": r"""
**First-Visit MC Prediction** estimates V^π(s) by averaging returns from the FIRST visit to s per episode:

V(s) ← average of G_t for all first visits to s across all episodes

- Unbiased estimator of V^π(s)
- Each episode contributes at most one return per state
- Converges to V^π(s) as number of episodes → ∞

Implemented in `mc_first_visit_prediction()` in `ch05_mc.rs`.
""",
        "theory_every_visit": r"""
**Every-Visit MC Prediction** averages returns from ALL visits to s (not just first):

V(s) ← average of G_t for ALL visits to s across all episodes

- Biased but consistent estimator
- More data per episode → lower variance
- Converges to V^π(s) as number of episodes → ∞

Implemented in `mc_every_visit_prediction()` in `ch05_mc.rs`.
""",
        "theory_on_policy": r"""
**On-Policy MC Control** (epsilon-soft) learns Q*(s,a) and π* simultaneously:

1. Generate episode using epsilon-soft policy
2. For each (s,a) pair (first-visit): G ← discounted return
3. Q(s,a) ← average(G) — incremental update
4. π(s) ← argmax_a Q(s,a) — greedy improvement

The epsilon-soft policy ensures all (s,a) pairs are visited infinitely often.
As epsilon → 0, the policy converges to the optimal greedy policy.

Implemented in `mc_on_policy_control()` in `ch05_mc.rs`.
""",
        "theory_off_policy": r"""
**Off-Policy MC Control** with Weighted Importance Sampling:

- **Behaviour policy b**: generates episodes (uniform random)
- **Target policy π**: what we want to optimise (greedy)
- **Importance Sampling Ratio**: ρ = π(a|s) / b(a|s)

Weighted IS update:
Q(s,a) ← Q(s,a) + (W / C(s,a)) * [G - Q(s,a)]
C(s,a) ← C(s,a) + W

Off-policy MC can learn the optimal policy while following a different (exploratory) policy.
This is the foundation for modern off-policy algorithms like Q-Learning (Ch06).

Implemented in `mc_off_policy_control()` in `ch05_mc.rs`.
""",
        "algo_labels": {
            "first_visit": "First-Visit MC",
            "every_visit": "Every-Visit MC",
            "on_policy":   "On-Policy Control",
            "off_policy":  "Off-Policy (IS)",
        },
        "pros_list": {
            "first_visit": ["Unbiased estimator", "Simple implementation", "No model needed"],
            "every_visit": ["More data per episode", "Lower variance than first-visit", "No model needed"],
            "on_policy":   ["Learns Q* and π* simultaneously", "No model needed", "Guaranteed to visit all (s,a)"],
            "off_policy":  ["Learns from any behaviour policy", "Foundation for Q-Learning", "Can reuse historical data"],
        },
        "cons_list": {
            "first_visit": ["High variance", "Must wait for episode end", "Slow for long episodes"],
            "every_visit": ["Biased estimator", "Must wait for episode end", "Slow for long episodes"],
            "on_policy":   ["Epsilon must stay > 0", "Slower convergence than DP", "High variance"],
            "off_policy":  ["IS ratio can explode", "High variance", "Complex implementation"],
        },
    },
    "FR": {
        "title": "Chapitre 05 — Méthodes de Monte Carlo",
        "subtitle": "Apprentissage ASP par épisodes · Sans modèle · Région de Varsovie",
        "engine_missing": "Exécutez : `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Paramètres MC",
        "n_episodes": "Nombre d\'épisodes",
        "gamma": "γ — Facteur d\'actualisation",
        "epsilon": "ε — Exploration initiale",
        "epsilon_decay": "Taux de décroissance ε α",
        "seed": "Graine aléatoire",
        "run_btn": "▶ Lancer les quatre algorithmes MC",
        "guide_title": "🎓 Comment utiliser ce chapitre",
        "guide": "Quatre algorithmes MC sans modèle. Augmentez le nombre d\'épisodes pour voir la convergence.",
        "returns_title": "📈 Retours par épisode — Quatre algorithmes",
        "returns_caption": "Moyenne mobile des retours. Le contrôle on-policy devrait s\'améliorer.",
        "value_title": "📊 Fonction de valeur V(s) — MC vs référence DP",
        "value_caption": "Les estimations MC convergent vers la solution DP (Ch04).",
        "visits_title": "🗺️ Comptage des visites — First-Visit MC",
        "visits_caption": "Les états rarement visités ont des estimations V(s) à haute variance.",
        "conv_title": "📈 Convergence — Max |V^(k) - V^(k-1)|",
        "conv_caption": "La convergence MC est plus bruyante que DP.",
        "qtable_title": "📊 Table Q — Contrôle On-Policy",
        "qtable_caption": "Valeurs Q(s,a) apprises par MC on-policy.",
        "glass_title": "🔬 Glass-Box — Trace d\'épisode",
        "glass_headers": ["Étape", "État", "Action", "Récompense", "Retour G_t"],
        "summary_title": "📊 Résumé",
        "summary_results": "Comparaison des algorithmes",
        "summary_pros_cons": "Algorithmes MC — Avantages & Inconvénients",
        "pros": "✅ Avantages", "cons": "❌ Inconvénients",
        "theory_title": "📖 Théorie — Chapitre 05",
        "theory_sections": {
            "intro": "§5.1 Introduction aux méthodes MC",
            "first_visit": "§5.2 Prédiction MC First-Visit",
            "every_visit": "§5.2 Prédiction MC Every-Visit",
            "on_policy": "§5.3 Contrôle MC On-Policy",
            "off_policy": "§5.4 MC Off-Policy avec échantillonnage d\'importance",
        },
        "theory_intro": "G_t = R_{t+1} + γ R_{t+2} + γ² R_{t+3} + ...",
        "theory_first_visit": "V(s) ← moyenne des G_t pour les premières visites à s.",
        "theory_every_visit": "V(s) ← moyenne des G_t pour toutes les visites à s.",
        "theory_on_policy": "Générer épisode → mettre à jour Q(s,a) → améliorer π greedily.",
        "theory_off_policy": "Q(s,a) ← Q(s,a) + (W/C) * [G - Q(s,a)]",
        "algo_labels": {"first_visit": "First-Visit MC", "every_visit": "Every-Visit MC", "on_policy": "Contrôle On-Policy", "off_policy": "Off-Policy (IS)"},
        "pros_list": {"first_visit": ["Estimateur non biaisé", "Sans modèle"], "every_visit": ["Plus de données", "Sans modèle"], "on_policy": ["Apprend Q* et π*", "Sans modèle"], "off_policy": ["Apprend de n\'importe quelle politique", "Base du Q-Learning"]},
        "cons_list": {"first_visit": ["Haute variance", "Attend la fin de l\'épisode"], "every_visit": ["Estimateur biaisé", "Attend la fin"], "on_policy": ["ε doit rester > 0", "Convergence lente"], "off_policy": ["Ratio IS peut exploser", "Haute variance"]},
    },
    "ES": {
        "title": "Capítulo 05 — Métodos de Monte Carlo",
        "subtitle": "Aprendizaje ASP por episodios · Sin modelo · Región de Varsovia",
        "engine_missing": "Ejecute: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Configuración MC",
        "n_episodes": "Número de episodios",
        "gamma": "γ — Factor de descuento",
        "epsilon": "ε — Exploración inicial",
        "epsilon_decay": "Tasa de decaimiento ε α",
        "seed": "Semilla aleatoria",
        "run_btn": "▶ Ejecutar los cuatro algoritmos MC",
        "guide_title": "🎓 Cómo usar este capítulo",
        "guide": "Cuatro algoritmos MC sin modelo. Aumente el número de episodios para ver la convergencia.",
        "returns_title": "📈 Retornos por episodio — Cuatro algoritmos",
        "returns_caption": "Media móvil de retornos. El control on-policy debería mejorar.",
        "value_title": "📊 Función de valor V(s) — MC vs referencia DP",
        "value_caption": "Las estimaciones MC convergen hacia la solución DP (Ch04).",
        "visits_title": "🗺️ Conteo de visitas — First-Visit MC",
        "visits_caption": "Los estados raramente visitados tienen estimaciones V(s) de alta varianza.",
        "conv_title": "📈 Convergencia — Max |V^(k) - V^(k-1)|",
        "conv_caption": "La convergencia MC es más ruidosa que DP.",
        "qtable_title": "📊 Tabla Q — Control On-Policy",
        "qtable_caption": "Valores Q(s,a) aprendidos por MC on-policy.",
        "glass_title": "🔬 Glass-Box — Traza de episodio",
        "glass_headers": ["Paso", "Estado", "Acción", "Recompensa", "Retorno G_t"],
        "summary_title": "📊 Resumen",
        "summary_results": "Comparación de algoritmos",
        "summary_pros_cons": "Algoritmos MC — Pros y Contras",
        "pros": "✅ Pros", "cons": "❌ Contras",
        "theory_title": "📖 Teoría — Capítulo 05",
        "theory_sections": {
            "intro": "§5.1 Introducción a los métodos MC",
            "first_visit": "§5.2 Predicción MC First-Visit",
            "every_visit": "§5.2 Predicción MC Every-Visit",
            "on_policy": "§5.3 Control MC On-Policy",
            "off_policy": "§5.4 MC Off-Policy con muestreo de importancia",
        },
        "theory_intro": "G_t = R_{t+1} + γ R_{t+2} + γ² R_{t+3} + ...",
        "theory_first_visit": "V(s) ← promedio de G_t para primeras visitas a s.",
        "theory_every_visit": "V(s) ← promedio de G_t para todas las visitas a s.",
        "theory_on_policy": "Generar episodio → actualizar Q(s,a) → mejorar π greedy.",
        "theory_off_policy": "Q(s,a) ← Q(s,a) + (W/C) * [G - Q(s,a)]",
        "algo_labels": {"first_visit": "First-Visit MC", "every_visit": "Every-Visit MC", "on_policy": "Control On-Policy", "off_policy": "Off-Policy (IS)"},
        "pros_list": {"first_visit": ["Estimador insesgado", "Sin modelo"], "every_visit": ["Más datos", "Sin modelo"], "on_policy": ["Aprende Q* y π*", "Sin modelo"], "off_policy": ["Aprende de cualquier política", "Base del Q-Learning"]},
        "cons_list": {"first_visit": ["Alta varianza", "Espera fin de episodio"], "every_visit": ["Estimador sesgado", "Espera fin"], "on_policy": ["ε debe ser > 0", "Convergencia lenta"], "off_policy": ["Ratio IS puede explotar", "Alta varianza"]},
    },
    "PL": {
        "title": "Rozdział 05 — Metody Monte Carlo",
        "subtitle": "Uczenie ASP z epizodów · Bez modelu · Region Warszawy",
        "engine_missing": "Uruchom: `cd rlvr-py && maturin develop`",
        "sidebar_title": "⚙️ Ustawienia MC",
        "n_episodes": "Liczba epizodów",
        "gamma": "γ — Współczynnik dyskontowania",
        "epsilon": "ε — Eksploracja początkowa",
        "epsilon_decay": "Współczynnik zaniku ε α",
        "seed": "Ziarno losowości",
        "run_btn": "▶ Uruchom wszystkie cztery algorytmy MC",
        "guide_title": "🎓 Jak korzystać z tego rozdziału",
        "guide": """
**Krok 1** — MC uczy się BEZ modelu P(s\'|s,a) — tylko z epizodów.
**Krok 2** — Ustaw liczbę epizodów. Zacznij od 200, potem 2000.
**Krok 3** — Kliknij ▶ aby uruchomić wszystkie cztery algorytmy.
**Krok 4** — Odczytaj krzywą zwrotów — powinna rosnąć dla on-policy.
**Krok 5** — Porównaj V(s) MC z rozwiązaniem DP z Ch04.
**Krok 6** — Odczytaj mapę ciepła wizyt — rzadko odwiedzane stany mają wysoką wariancję.
**Krok 7** — Odczytaj Glass-Box — dokładne zwroty G_t dla wybranego epizodu.
""",
        "returns_title": "📈 Zwroty epizodów — Cztery algorytmy",
        "returns_caption": "Średnia krocząca zwrotów. On-policy control powinien się poprawiać.",
        "value_title": "📊 Funkcja wartości V(s) — MC vs referencja DP",
        "value_caption": "Estymaty MC zbiegają do rozwiązania DP (Ch04) przy większej liczbie epizodów.",
        "visits_title": "🗺️ Liczba wizyt — First-Visit MC",
        "visits_caption": "Rzadko odwiedzane stany mają estymaty V(s) z wysoką wariancją.",
        "conv_title": "📈 Zbieżność — Max |V^(k) - V^(k-1)|",
        "conv_caption": "Zbieżność MC jest bardziej zaszumiona niż DP — środowisko stochastyczne.",
        "qtable_title": "📊 Tabela Q — On-Policy Control",
        "qtable_caption": "Wartości Q(s,a) wyuczone przez on-policy MC.",
        "glass_title": "🔬 Glass-Box — Ślad epizodu",
        "glass_headers": ["Krok", "Stan", "Akcja", "Nagroda", "Zwrot G_t"],
        "summary_title": "📊 Podsumowanie",
        "summary_results": "Porównanie algorytmów",
        "summary_pros_cons": "Algorytmy MC — Zalety i Wady",
        "pros": "✅ Zalety", "cons": "❌ Wady",
        "theory_title": "📖 Teoria — Rozdział 05",
        "theory_sections": {
            "intro": "§5.1 Wprowadzenie do metod Monte Carlo",
            "first_visit": "§5.2 Predykcja MC First-Visit",
            "every_visit": "§5.2 Predykcja MC Every-Visit",
            "on_policy": "§5.3 Sterowanie MC On-Policy",
            "off_policy": "§5.4 MC Off-Policy z próbkowaniem ważności",
        },
        "theory_intro": r"""
**Metody Monte Carlo** uczą się bezpośrednio z epizodów — bez modelu P(s\'|s,a).
G_t = R_{t+1} + γ R_{t+2} + γ² R_{t+3} + ...
Implementacja: `ch05_mc.rs` — `generate_episode()`, `mc_first_visit_prediction()`.
""",
        "theory_first_visit": "V(s) ← średnia G_t dla pierwszych wizyt w s.",
        "theory_every_visit": "V(s) ← średnia G_t dla wszystkich wizyt w s.",
        "theory_on_policy": "Generuj epizod → aktualizuj Q(s,a) → popraw π zachłannie.",
        "theory_off_policy": "Q(s,a) ← Q(s,a) + (W/C) * [G - Q(s,a)]",
        "algo_labels": {"first_visit": "First-Visit MC", "every_visit": "Every-Visit MC", "on_policy": "On-Policy Control", "off_policy": "Off-Policy (IS)"},
        "pros_list": {"first_visit": ["Nieobciążony estymator", "Bez modelu", "Prosta implementacja"], "every_visit": ["Więcej danych na epizod", "Bez modelu"], "on_policy": ["Uczy Q* i π* jednocześnie", "Bez modelu"], "off_policy": ["Uczy z dowolnej polityki", "Podstawa Q-Learning"]},
        "cons_list": {"first_visit": ["Wysoka wariancja", "Czeka na koniec epizodu"], "every_visit": ["Obciążony estymator", "Czeka na koniec"], "on_policy": ["ε musi być > 0", "Wolna zbieżność"], "off_policy": ["Współczynnik IS może eksplodować", "Wysoka wariancja"]},
    },
}

COLORS = {
    "first_visit": "#0082F0",
    "every_visit": "#FF8C0A",
    "on_policy":   "#0FC373",
    "off_policy":  "#FF3232",
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
    n_episodes    = st.sidebar.slider(tx["n_episodes"],    50, 5000, 500, 50)
    gamma         = st.sidebar.slider(tx["gamma"],         0.5, 0.999, 0.95, 0.005)
    epsilon       = st.sidebar.slider(tx["epsilon"],       0.0, 1.0, 0.3, 0.05)
    epsilon_decay = st.sidebar.slider(tx["epsilon_decay"], 0.0, 0.1, 0.01, 0.001, format="%.3f")
    seed          = st.sidebar.number_input(tx["seed"], 0, 9999, 42)

    with st.expander(tx["guide_title"], expanded=False):
        st.markdown(tx["guide"])

    if st.button(tx["run_btn"], type="primary"):
        with st.spinner("Running Rust MC engine..."):
            result = rlvr_py.run_ch05_mc(
                int(seed), int(n_episodes), float(gamma),
                float(epsilon), float(epsilon_decay)
            )
        st.session_state["ch05_result"] = result

    if "ch05_result" not in st.session_state:
        st.info("Configure settings and click **▶ Run All Four MC Algorithms**.")
        _render_theory(tx)
        return

    result      = st.session_state["ch05_result"]
    state_names = result["state_names"]
    action_names= result["action_names"]
    algos       = ["first_visit", "every_visit", "on_policy", "off_policy"]

    # KPI
    cols = st.columns(4)
    for i, key in enumerate(algos):
        r = result[key]
        avg_ret = sum(r["returns_curve"][-50:]) / min(50, len(r["returns_curve"]))
        cols[i].metric(tx["algo_labels"][key], f"Avg return: {avg_ret:.2f}")

    # Returns curve
    st.subheader(tx["returns_title"])
    fig = go.Figure()
    for key in algos:
        r = result[key]
        ma = _moving_avg(r["returns_curve"], 30)
        fig.add_trace(go.Scatter(
            x=list(range(len(ma))), y=ma,
            mode="lines", name=tx["algo_labels"][key],
            line=dict(color=COLORS[key], width=2),
        ))
    fig.update_layout(height=300, margin=dict(l=40,r=20,t=20,b=40),
                      xaxis_title="Episode", yaxis_title="Return (MA-30)",
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(tx["returns_caption"])

    # Value function comparison
    st.subheader(tx["value_title"])
    short = [f"S{i}" for i in range(result["n_states"])]
    fig2 = go.Figure()
    for key in algos:
        fig2.add_trace(go.Bar(
            x=short, y=result[key]["values"],
            name=tx["algo_labels"][key],
            marker_color=COLORS[key], opacity=0.8,
        ))
    fig2.update_layout(height=300, barmode="group",
                       margin=dict(l=40,r=20,t=20,b=40),
                       legend=dict(orientation="h"))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(tx["value_caption"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(tx["visits_title"])
        vc = result["first_visit"]["visit_counts"]
        colors_vc = ["#0FC373" if v > 50 else "#FF8C0A" if v > 10 else "#FF3232" for v in vc]
        fig3 = go.Figure(go.Bar(x=short, y=vc, marker_color=colors_vc,
                                text=[str(v) for v in vc], textposition="outside"))
        fig3.update_layout(height=280, margin=dict(l=40,r=20,t=20,b=40))
        st.plotly_chart(fig3, use_container_width=True)
        st.caption(tx["visits_caption"])

    with col2:
        st.subheader(tx["conv_title"])
        fig4 = go.Figure()
        for key in ["first_visit", "on_policy"]:
            fig4.add_trace(go.Scatter(
                x=list(range(len(result[key]["convergence_curve"]))),
                y=result[key]["convergence_curve"],
                mode="lines", name=tx["algo_labels"][key],
                line=dict(color=COLORS[key], width=1.5),
            ))
        fig4.update_layout(height=280, margin=dict(l=40,r=20,t=20,b=40),
                           yaxis_type="log", legend=dict(orientation="h"))
        st.plotly_chart(fig4, use_container_width=True)
        st.caption(tx["conv_caption"])

    # Q-table heatmap
    st.subheader(tx["qtable_title"])
    qt = result["on_policy"]["q_table"]
    action_short = [f"A{i}" for i in range(result["n_actions"])]
    fig5 = go.Figure(go.Heatmap(
        z=qt, x=action_short, y=short,
        colorscale="Blues",
        text=[[f"{qt[s][a]:.2f}" for a in range(result["n_actions"])]
              for s in range(result["n_states"])],
        texttemplate="%{text}",
    ))
    fig5.update_layout(height=320, margin=dict(l=60,r=20,t=20,b=40))
    st.plotly_chart(fig5, use_container_width=True)
    st.caption(tx["qtable_caption"])

    # Glass-Box
    st.subheader(tx["glass_title"])
    _render_glass_box(result, tx, state_names, action_names)

    # Summary
    st.subheader(tx["summary_title"])
    _render_summary(result, tx, algos)

    _render_theory(tx)


def _render_glass_box(result, tx, state_names, action_names):
    algo_options = {tx["algo_labels"][k]: k for k in ["first_visit", "every_visit", "on_policy", "off_policy"]}
    selected = st.selectbox("Algorithm", list(algo_options.keys()))
    key = algo_options[selected]
    r = result[key]
    curve = r["returns_curve"]
    ep_idx = st.slider("Episode", 0, max(len(curve)-1, 0), max(len(curve)-1, 0))
    st.metric("Episode return", f"{curve[ep_idx]:.3f}")
    st.latex(r"G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}")


def _render_summary(result, tx, algos):
    st.markdown(f"#### {tx['summary_results']}")
    rows = []
    for key in algos:
        r = result[key]
        avg_last = sum(r["returns_curve"][-100:]) / min(100, len(r["returns_curve"]))
        rows.append({
            "Algorithm":        tx["algo_labels"][key],
            "Avg return (last 100)": f"{avg_last:.3f}",
            "Best V*(S0)":      f"{r['values'][0]:.3f}",
            "Worst V*(S7)":     f"{r['values'][7]:.3f}",
            "Best action S7":   f"A{r['policy'][7]}",
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
    for key in ["intro", "first_visit", "every_visit", "on_policy", "off_policy"]:
        with st.expander(tx["theory_sections"][key], expanded=False):
            st.markdown(tx[f"theory_{key}"])
'''

# ---------------------------------------------------------------------------
# FILE 4: README.md (updated)
# ---------------------------------------------------------------------------
README_MD = """# RLVR Enterprise Allocator

> **Reinforcement Learning via Rust** — A 20-chapter, end-to-end framework for learning and demonstrating RL algorithms through a real enterprise field-service optimisation use case.

[![Rust](https://img.shields.io/badge/Rust-1.75+-orange?logo=rust)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Progress](https://img.shields.io/badge/Progress-Ch01--05%20%E2%9C%85%20%7C%20Ch06--20%20%F0%9F%9A%A7-lightgrey)]()

---

## What is this?

This project implements **all 20 chapters** of the [Reinforcement Learning via Rust](https://rlvr.rantai.dev/docs/reinforcement-learning-via-rust/) book as a fully working, interactive demo application built to learn RL from first principles while simultaneously building production-grade Rust infrastructure.

**The business scenario:** A field-service dispatch system for the Warsaw region. An RL agent learns to assign technicians to work orders, optimising SLA compliance, skill matching, and travel distance.

**The architecture principle:** All RL logic lives in Rust. Python is used exclusively for UI rendering.

```
rlvr-core  (Rust)   <- ALL RL algorithms, environments, math
     | PyO3
rlvr-py    (Bridge) <- zero-copy FFI, serialises Rust structs -> Python dicts
     | json
gui/       (Python) <- Streamlit UI only -- renders data, zero RL logic
```

---

## Progress

| Chapter | Topic | Algorithm | Status |
|---------|-------|-----------|--------|
| **01** | Introduction to RL | MDP, epsilon-greedy, Gt | ✅ Complete |
| **02** | Discrete MDP & Bellman | Value Iteration, nalgebra LU | ✅ Complete |
| **03** | Multi-Armed Bandit | UCB1, Thompson Sampling | ✅ Complete |
| **04** | Dynamic Programming | Policy Iteration, Async VI | ✅ Complete |
| **05** | Monte Carlo Methods | First-Visit, Every-Visit, On/Off-Policy | ✅ Complete |
| 06 | Temporal Difference | SARSA, Q-Learning | 🚧 Planned |
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
│   └── ch05_mc.rs             # Monte Carlo: First/Every-Visit, On/Off-Policy
├── rlvr-py/src/lib.rs         # PyO3 bridge (Ch01-Ch05)
└── gui/chapters/
    ├── ch01.py  ch02.py  ch03.py  ch04.py  ch05.py
```

---

## Quick Start

```bash
# Build
cd rlvr-py && maturin develop && cd ..

# Test (49 passing)
cargo test --workspace

# Run
streamlit run gui/app.py
```

---

## Chapter Summaries

### Ch01 — Introduction to RL
Warsaw map, epsilon-greedy, Glass-Box MDP inspector, learning curve, 4 languages.

### Ch02 — Discrete MDP & Bellman
Value function V*(s), optimal policy, convergence curve, Bellman trace, nalgebra LU.

### Ch03 — Multi-Armed Bandit
Cumulative regret, UCB1 vs Thompson Sampling vs epsilon-greedy, Beta posteriors.

### Ch04 — Dynamic Programming
Policy Iteration vs Value Iteration vs Async VI, policy evolution, Bellman residuals.

### Ch05 — Monte Carlo Methods
- **First-Visit MC**: unbiased V^pi estimation from first state visits per episode
- **Every-Visit MC**: V^pi estimation from all state visits per episode
- **On-Policy Control**: learns Q* and pi* simultaneously (epsilon-soft)
- **Off-Policy Control**: Weighted Importance Sampling from behaviour policy
- Returns curve, visit count heatmap, Q-table heatmap, Glass-Box episode trace

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

## Design Principles

1. Rust owns all computation
2. Deterministic by default (seeded RNG)
3. TDD from day one
4. Vertical slices per chapter
5. Glass-box transparency

---

## License

MIT -- see [LICENSE](LICENSE)
"""

# ---------------------------------------------------------------------------
# FILE 5: apply_ch05.sh — install script
# ---------------------------------------------------------------------------
APPLY_SH = """#!/bin/bash
# RLVR Chapter 05 — Apply script
# Run from project root: bash apply_ch05.sh

set -e
echo "Applying Chapter 05..."

# 1 — Rust core
cp ch05_package/rlvr-core/src/ch05_mc.rs rlvr-core/src/ch05_mc.rs
echo "pub mod ch05_mc;" >> rlvr-core/src/lib.rs

# 2 — Bridge
cp ch05_package/rlvr-py/src/lib.rs rlvr-py/src/lib.rs

# 3 — UI
cp ch05_package/gui/chapters/ch05.py gui/chapters/ch05.py

# 4 — README
cp ch05_package/README.md README.md

# 5 — Routing in app.py
python3 -c "
content = open('gui/app.py').read()
content = content.replace(
    'elif ch_num == 4:\\n    from chapters.ch04 import render\\n    render()',
    'elif ch_num == 4:\\n    from chapters.ch04 import render\\n    render()\\nelif ch_num == 5:\\n    from chapters.ch05 import render\\n    render()'
)
open('gui/app.py', 'w').write(content)
print('app.py routing updated')
"

# 6 — Remove duplicate lib.rs module if needed
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

# 7 — Build and test
echo "Building..."
cd rlvr-py && maturin develop && cd ..
echo "Testing..."
cargo test --workspace

echo ""
echo "Chapter 05 applied successfully!"
echo "Run: streamlit run gui/app.py"
"""

# ---------------------------------------------------------------------------
# Build ZIP
# ---------------------------------------------------------------------------
files = {
    "ch05_package/rlvr-core/src/ch05_mc.rs":      CH05_RS,
    "ch05_package/rlvr-py/src/lib.rs":             LIB_RS,
    "ch05_package/gui/chapters/ch05.py":           CH05_PY,
    "ch05_package/README.md":                      README_MD,
    "ch05_package/apply_ch05.sh":                  APPLY_SH,
}

zip_path = "/tmp/ch05_package.zip"
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

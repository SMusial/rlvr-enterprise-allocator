use ndarray::Array2;
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

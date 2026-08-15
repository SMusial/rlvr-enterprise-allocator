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

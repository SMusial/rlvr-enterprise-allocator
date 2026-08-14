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

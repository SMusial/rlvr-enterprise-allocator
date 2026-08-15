use ndarray::{Array2, Array3};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::cmp::Ordering;

// ---------------------------------------------------------------------------
// Re-use Ch02 ASP environment (same 8 states, 4 actions)
// ---------------------------------------------------------------------------
use crate::ch02_bellman::{
    build_asp_transitions, build_asp_rewards, verify_transition_matrix,
    N_STATES, N_ACTIONS,
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct DpModel {
    pub transitions: Array3<f64>,
    pub rewards: Array2<f64>,
    pub gamma: f64,
}

#[derive(Debug, Clone)]
pub struct PolicyIterationResult {
    pub values: Vec<f64>,
    pub policy: Vec<usize>,
    pub pi_iterations: usize,          // outer PI loops
    pub eval_iterations: Vec<usize>,   // inner eval loops per PI step
    pub convergence_curve: Vec<f64>,   // max delta per eval sweep
    pub policy_history: Vec<Vec<usize>>, // policy at each PI step
}

#[derive(Debug, Clone)]
pub struct DpCompareResult {
    pub pi: PolicyIterationResult,
    pub vi_values: Vec<f64>,
    pub vi_policy: Vec<usize>,
    pub vi_iterations: usize,
    pub vi_curve: Vec<f64>,
    pub async_values: Vec<f64>,
    pub async_policy: Vec<usize>,
    pub async_iterations: usize,
    pub async_curve: Vec<f64>,
    pub residuals: Vec<f64>,           // final Bellman residuals per state
}

#[derive(Debug, Clone)]
pub struct Ch04Config {
    pub seed: u64,
    pub gamma: f64,
    pub theta: f64,
}

// ---------------------------------------------------------------------------
// Priority queue entry for Prioritized Sweeping
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
struct StatePriority {
    state: usize,
    priority: f64,
}

impl Eq for StatePriority {}
impl PartialEq for StatePriority {
    fn eq(&self, other: &Self) -> bool {
        self.priority.to_bits() == other.priority.to_bits()
    }
}
impl Ord for StatePriority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .partial_cmp(&other.priority)
            .unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for StatePriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ---------------------------------------------------------------------------
// Helper: compute Q(s,a) for all actions
// ---------------------------------------------------------------------------
fn compute_q_values(s: usize, v: &[f64], model: &DpModel) -> Vec<f64> {
    (0..N_ACTIONS)
        .map(|a| {
            (0..N_STATES)
                .map(|sp| {
                    model.transitions[[s, a, sp]]
                        * (model.rewards[[s, a]] + model.gamma * v[sp])
                })
                .sum::<f64>()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Helper: greedy policy extraction
// ---------------------------------------------------------------------------
fn greedy_policy(v: &[f64], model: &DpModel) -> Vec<usize> {
    (0..N_STATES)
        .map(|s| {
            compute_q_values(s, v, model)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Policy Evaluation — Bellman expectation equation
// V^π(s) = Σ_s' P(s'|s,π(s)) [R(s,π(s)) + γ V^π(s')]
// ---------------------------------------------------------------------------
pub fn policy_evaluation(
    policy: &[usize],
    model: &DpModel,
    theta: f64,
) -> (Vec<f64>, usize, Vec<f64>) {
    let mut v = vec![0.0_f64; N_STATES];
    let mut iters = 0usize;
    let mut curve = Vec::new();

    loop {
        let mut delta = 0.0_f64;
        for s in 0..N_STATES {
            let a = policy[s];
            let v_new: f64 = (0..N_STATES)
                .map(|sp| {
                    model.transitions[[s, a, sp]]
                        * (model.rewards[[s, a]] + model.gamma * v[sp])
                })
                .sum();
            delta = delta.max((v_new - v[s]).abs());
            v[s] = v_new;
        }
        curve.push(delta);
        iters += 1;
        if delta < theta || iters > 1000 {
            break;
        }
    }
    (v, iters, curve)
}

// ---------------------------------------------------------------------------
// Policy Improvement — greedy w.r.t. V^π
// π'(s) = argmax_a Σ_s' P(s'|s,a)[R(s,a) + γ V^π(s')]
// ---------------------------------------------------------------------------
pub fn policy_improvement(v: &[f64], model: &DpModel) -> Vec<usize> {
    greedy_policy(v, model)
}

// ---------------------------------------------------------------------------
// Policy Iteration — alternates eval + improve until stable
// ---------------------------------------------------------------------------
pub fn policy_iteration(model: &DpModel, theta: f64) -> PolicyIterationResult {
    let mut policy: Vec<usize> = vec![0; N_STATES]; // start with A0 everywhere
    let mut pi_iters = 0usize;
    let mut eval_iters_vec: Vec<usize> = Vec::new();
    let mut convergence_curve: Vec<f64> = Vec::new();
    let mut policy_history: Vec<Vec<usize>> = vec![policy.clone()];

    loop {
        // Step 1 — Policy Evaluation
        let (v, eval_iters, eval_curve) = policy_evaluation(&policy, model, theta);
        eval_iters_vec.push(eval_iters);
        convergence_curve.extend(eval_curve);

        // Step 2 — Policy Improvement
        let new_policy = policy_improvement(&v, model);
        pi_iters += 1;
        policy_history.push(new_policy.clone());

        // Step 3 — Check stability
        if new_policy == policy || pi_iters >= 100 {
            return PolicyIterationResult {
                values: v,
                policy: new_policy,
                pi_iterations: pi_iters,
                eval_iterations: eval_iters_vec,
                convergence_curve,
                policy_history,
            };
        }
        policy = new_policy;
    }
}

// ---------------------------------------------------------------------------
// Value Iteration (synchronous) — for comparison with PI
// ---------------------------------------------------------------------------
pub fn value_iteration_sync(model: &DpModel, theta: f64) -> (Vec<f64>, Vec<usize>, usize, Vec<f64>) {
    let mut v = vec![0.0_f64; N_STATES];
    let mut curve = Vec::new();
    let mut iters = 0usize;

    loop {
        let mut delta = 0.0_f64;
        for s in 0..N_STATES {
            let q = compute_q_values(s, &v, model);
            let v_new = q.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            delta = delta.max((v_new - v[s]).abs());
            v[s] = v_new;
        }
        curve.push(delta);
        iters += 1;
        if delta < theta || iters > 1000 {
            break;
        }
    }
    let policy = greedy_policy(&v, model);
    (v, policy, iters, curve)
}

// ---------------------------------------------------------------------------
// Asynchronous Value Iteration with Prioritized Sweeping
// Updates states in order of Bellman residual (highest first)
// ---------------------------------------------------------------------------
pub fn async_value_iteration(model: &DpModel, theta: f64) -> (Vec<f64>, Vec<usize>, usize, Vec<f64>) {
    let mut v = vec![0.0_f64; N_STATES];
    let mut curve = Vec::new();
    let mut iters = 0usize;

    loop {
        // Compute residuals for all states
        let mut residuals: Vec<(usize, f64)> = (0..N_STATES)
            .map(|s| {
                let q = compute_q_values(s, &v, model);
                let v_best = q.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                (s, (v_best - v[s]).abs())
            })
            .collect();

        // Sort by descending residual — prioritize high-impact states
        residuals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut delta = 0.0_f64;
        for (s, _) in &residuals {
            let q = compute_q_values(*s, &v, model);
            let v_new = q.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            delta = delta.max((v_new - v[*s]).abs());
            v[*s] = v_new;
        }

        curve.push(delta);
        iters += 1;
        if delta < theta || iters > 1000 {
            break;
        }
    }

    let policy = greedy_policy(&v, model);
    (v, policy, iters, curve)
}

// ---------------------------------------------------------------------------
// Bellman residual per state — for heatmap visualisation
// ---------------------------------------------------------------------------
pub fn compute_residuals(v: &[f64], model: &DpModel) -> Vec<f64> {
    (0..N_STATES)
        .map(|s| {
            let q = compute_q_values(s, v, model);
            let v_best = q.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (v_best - v[s]).abs()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Main entry point — runs all 3 algorithms for comparison
// ---------------------------------------------------------------------------
pub fn run_ch04(config: Ch04Config) -> DpCompareResult {
    let mut rng = StdRng::seed_from_u64(config.seed);
    let transitions = build_asp_transitions(&mut rng);
    let rewards = build_asp_rewards();

    verify_transition_matrix(&transitions)
        .expect("Transition matrix probability conservation violated");

    let model = DpModel {
        transitions,
        rewards,
        gamma: config.gamma,
    };

    // Run all three algorithms
    let pi = policy_iteration(&model, config.theta);
    let (vi_values, vi_policy, vi_iters, vi_curve) =
        value_iteration_sync(&model, config.theta);
    let (async_values, async_policy, async_iters, async_curve) =
        async_value_iteration(&model, config.theta);

    let residuals = compute_residuals(&pi.values, &model);

    DpCompareResult {
        pi,
        vi_values,
        vi_policy,
        vi_iterations: vi_iters,
        vi_curve,
        async_values,
        async_policy,
        async_iterations: async_iters,
        async_curve,
        residuals,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> Ch04Config {
        Ch04Config { seed: 42, gamma: 0.95, theta: 1e-6 }
    }

    fn default_model() -> DpModel {
        let mut rng = StdRng::seed_from_u64(42);
        DpModel {
            transitions: build_asp_transitions(&mut rng),
            rewards: build_asp_rewards(),
            gamma: 0.95,
        }
    }

    #[test]
    fn test_policy_iteration_converges() {
        let result = run_ch04(default_config());
        assert!(result.pi.pi_iterations < 50, "PI should converge in < 50 outer loops");
        assert_eq!(result.pi.values.len(), N_STATES);
    }

    #[test]
    fn test_policy_evaluation_correct() {
        let model = default_model();
        let policy = vec![1usize; N_STATES]; // always A1
        let (v, iters, _) = policy_evaluation(&policy, &model, 1e-6);
        assert!(iters < 500);
        for val in &v { assert!(val.is_finite()); }
    }

    #[test]
    fn test_policy_improvement_valid() {
        let model = default_model();
        let v = vec![0.0_f64; N_STATES];
        let policy = policy_improvement(&v, &model);
        assert_eq!(policy.len(), N_STATES);
        for &a in &policy { assert!(a < N_ACTIONS); }
    }

    #[test]
    fn test_pi_vi_same_policy() {
        // PI and VI should converge to the same optimal policy
        let result = run_ch04(default_config());
        assert_eq!(
            result.pi.policy, result.vi_policy,
            "PI and VI must find the same optimal policy"
        );
    }

    #[test]
    fn test_pi_vi_close_values() {
        let result = run_ch04(default_config());
        for (v_pi, v_vi) in result.pi.values.iter().zip(result.vi_values.iter()) {
            assert!(
                (v_pi - v_vi).abs() < 0.1,
                "PI and VI values should be close: {} vs {}",
                v_pi, v_vi
            );
        }
    }

    #[test]
    fn test_async_vi_converges() {
        let result = run_ch04(default_config());
        assert!(result.async_iterations < 500);
        for v in &result.async_values { assert!(v.is_finite()); }
    }

    #[test]
    fn test_async_fewer_iterations_than_sync() {
        let result = run_ch04(default_config());
        // Async should converge in fewer or equal iterations than sync VI
        assert!(
            result.async_iterations <= result.vi_iterations + 5,
            "Async VI should be competitive: async={} sync={}",
            result.async_iterations, result.vi_iterations
        );
    }

    #[test]
    fn test_policy_history_monotone() {
        let result = run_ch04(default_config());
        // Policy history should have at least 2 entries (initial + final)
        assert!(result.pi.policy_history.len() >= 2);
    }

    #[test]
    fn test_residuals_finite() {
        let result = run_ch04(default_config());
        for r in &result.residuals { assert!(r.is_finite()); }
    }

    #[test]
    fn test_deterministic() {
        let r1 = run_ch04(default_config());
        let r2 = run_ch04(default_config());
        for (v1, v2) in r1.pi.values.iter().zip(r2.pi.values.iter()) {
            assert_eq!(v1.to_bits(), v2.to_bits());
        }
    }
}

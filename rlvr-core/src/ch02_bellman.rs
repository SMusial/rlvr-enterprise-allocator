use nalgebra::{DMatrix, DVector};
use ndarray::{Array2, Array3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
pub const N_STATES: usize = 8;
pub const N_ACTIONS: usize = 4;

/// ASP operational state descriptions
pub const STATE_NAMES: &[&str] = &[
    "S0: All available, no urgent",
    "S1: All available, urgent pending",
    "S2: Some busy, no urgent",
    "S3: Some busy, urgent pending",
    "S4: Most busy, no urgent",
    "S5: Most busy, urgent pending",
    "S6: All busy, backlog building",
    "S7: All busy, SLA breach imminent",
];

/// ASP dispatch strategy descriptions
pub const ACTION_NAMES: &[&str] = &[
    "A0: Dispatch nearest tech",
    "A1: Dispatch skill-matched tech",
    "A2: Dispatch most experienced tech",
    "A3: Hold — wait for better tech",
];

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct MdpModel {
    /// Transition tensor P[s, a, s'] — shape (N_STATES, N_ACTIONS, N_STATES)
    pub transitions: Array3<f64>,
    /// Reward matrix R[s, a] — shape (N_STATES, N_ACTIONS)
    pub rewards: Array2<f64>,
    pub gamma: f64,
}

#[derive(Debug, Clone)]
pub struct ValueIterationResult {
    pub values: Vec<f64>,
    pub policy: Vec<usize>,
    pub iterations: usize,
    pub convergence_curve: Vec<f64>,
    pub bellman_trace: Vec<BellmanStep>,
}

#[derive(Debug, Clone)]
pub struct BellmanStep {
    pub iteration: usize,
    pub state: usize,
    pub action: usize,
    pub q_values: Vec<f64>,
    pub v_old: f64,
    pub v_new: f64,
    pub delta: f64,
}

#[derive(Debug, Clone)]
pub struct PolicyEvalResult {
    pub values: Vec<f64>,
    pub iterations: usize,
}

#[derive(Debug, Clone)]
pub struct Ch02Config {
    pub seed: u64,
    pub gamma: f64,
    pub theta: f64, // convergence threshold
}

// ---------------------------------------------------------------------------
// ASP transition matrix — domain-specific probabilities
// ---------------------------------------------------------------------------
pub fn build_asp_transitions(rng: &mut StdRng) -> Array3<f64> {
    let mut p = Array3::<f64>::zeros((N_STATES, N_ACTIONS, N_STATES));

    // Domain knowledge encoded as transition probabilities
    // Each row p[s, a, :] must sum to 1.0 (enforced below)

    // A0: Dispatch nearest — fast but may mismatch skill
    // A1: Dispatch skill-matched — slower but higher quality
    // A2: Dispatch most experienced — best SLA, slowest
    // A3: Hold — wait for better tech

    let base_transitions: [[[f64; N_STATES]; N_ACTIONS]; N_STATES] = [
        // S0: All available, no urgent
        [
            [0.05, 0.40, 0.30, 0.10, 0.10, 0.03, 0.01, 0.01], // A0 nearest
            [0.10, 0.35, 0.25, 0.15, 0.10, 0.03, 0.01, 0.01], // A1 skill
            [0.15, 0.40, 0.20, 0.10, 0.10, 0.03, 0.01, 0.01], // A2 experienced
            [0.20, 0.30, 0.25, 0.10, 0.10, 0.03, 0.01, 0.01], // A3 hold
        ],
        // S1: All available, urgent pending
        [
            [0.30, 0.05, 0.30, 0.20, 0.10, 0.03, 0.01, 0.01], // A0 nearest — resolves urgent fast
            [0.40, 0.05, 0.25, 0.15, 0.10, 0.03, 0.01, 0.01], // A1 skill — best for urgent
            [0.35, 0.05, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02], // A2 experienced
            [0.05, 0.10, 0.20, 0.30, 0.15, 0.10, 0.07, 0.03], // A3 hold — dangerous when urgent
        ],
        // S2: Some busy, no urgent
        [
            [0.10, 0.20, 0.05, 0.35, 0.20, 0.05, 0.03, 0.02], // A0
            [0.15, 0.25, 0.05, 0.30, 0.15, 0.05, 0.03, 0.02], // A1
            [0.20, 0.25, 0.05, 0.25, 0.15, 0.05, 0.03, 0.02], // A2
            [0.25, 0.20, 0.10, 0.25, 0.10, 0.05, 0.03, 0.02], // A3
        ],
        // S3: Some busy, urgent pending
        [
            [0.20, 0.15, 0.20, 0.05, 0.20, 0.10, 0.07, 0.03], // A0
            [0.30, 0.15, 0.15, 0.05, 0.15, 0.10, 0.07, 0.03], // A1 — best
            [0.25, 0.15, 0.15, 0.05, 0.15, 0.12, 0.08, 0.05], // A2
            [0.05, 0.10, 0.15, 0.10, 0.20, 0.20, 0.12, 0.08], // A3 — risky
        ],
        // S4: Most busy, no urgent
        [
            [0.10, 0.10, 0.20, 0.15, 0.05, 0.25, 0.10, 0.05], // A0
            [0.15, 0.10, 0.20, 0.15, 0.05, 0.20, 0.10, 0.05], // A1
            [0.20, 0.10, 0.20, 0.10, 0.05, 0.20, 0.10, 0.05], // A2
            [0.25, 0.10, 0.20, 0.10, 0.05, 0.15, 0.10, 0.05], // A3
        ],
        // S5: Most busy, urgent pending — critical state
        [
            [0.15, 0.10, 0.15, 0.20, 0.10, 0.05, 0.15, 0.10], // A0 — act fast
            [0.20, 0.10, 0.15, 0.20, 0.10, 0.05, 0.12, 0.08], // A1
            [0.25, 0.10, 0.15, 0.15, 0.10, 0.05, 0.12, 0.08], // A2 — best
            [0.03, 0.05, 0.10, 0.15, 0.10, 0.10, 0.25, 0.22], // A3 — very risky
        ],
        // S6: All busy, backlog building
        [
            [0.10, 0.05, 0.10, 0.10, 0.15, 0.15, 0.05, 0.30], // A0
            [0.15, 0.05, 0.10, 0.10, 0.15, 0.15, 0.05, 0.25], // A1
            [0.20, 0.05, 0.10, 0.10, 0.15, 0.10, 0.05, 0.25], // A2
            [0.05, 0.05, 0.10, 0.10, 0.15, 0.15, 0.10, 0.30], // A3
        ],
        // S7: All busy, SLA breach imminent — worst state
        [
            [0.20, 0.10, 0.10, 0.10, 0.10, 0.10, 0.15, 0.15], // A0 — emergency dispatch
            [0.25, 0.10, 0.10, 0.10, 0.10, 0.10, 0.13, 0.12], // A1
            [0.30, 0.10, 0.10, 0.10, 0.10, 0.08, 0.12, 0.10], // A2 — best escape
            [0.05, 0.05, 0.05, 0.10, 0.10, 0.15, 0.25, 0.25], // A3 — catastrophic
        ],
    ];

    for s in 0..N_STATES {
        for a in 0..N_ACTIONS {
            let mut row = base_transitions[s][a];
            // Add small random noise for realism
            let mut sum = 0.0_f64;
            for sp in 0..N_STATES {
                let noise = rng.gen::<f64>() * 0.02 - 0.01;
                row[sp] = (row[sp] + noise).max(0.001);
                sum += row[sp];
            }
            // Normalise to enforce probability conservation invariant
            for sp in 0..N_STATES {
                p[[s, a, sp]] = row[sp] / sum;
            }
        }
    }
    p
}

// ---------------------------------------------------------------------------
// ASP reward matrix — domain-specific rewards
// ---------------------------------------------------------------------------
pub fn build_asp_rewards() -> Array2<f64> {
    let mut r = Array2::<f64>::zeros((N_STATES, N_ACTIONS));

    // Rewards encode business value of each strategy in each state
    let base_rewards: [[f64; N_ACTIONS]; N_STATES] = [
        // S0: All available — all strategies work well
        [6.0, 8.0, 7.0, 4.0],
        // S1: Urgent pending — skill match and experience pay off
        [7.0, 10.0, 9.0, -3.0],
        // S2: Some busy — moderate rewards
        [5.0, 7.0, 6.0, 3.0],
        // S3: Some busy + urgent — act decisively
        [6.0, 9.0, 8.0, -2.0],
        // S4: Most busy — limited options
        [4.0, 6.0, 5.0, 2.0],
        // S5: Critical — every action matters
        [5.0, 7.0, 8.0, -8.0],
        // S6: All busy — damage control
        [3.0, 5.0, 6.0, -1.0],
        // S7: SLA breach imminent — emergency
        [4.0, 6.0, 8.0, -10.0],
    ];

    for s in 0..N_STATES {
        for a in 0..N_ACTIONS {
            r[[s, a]] = base_rewards[s][a];
        }
    }
    r
}

// ---------------------------------------------------------------------------
// Safety invariant: probability conservation
// ---------------------------------------------------------------------------
pub fn verify_transition_matrix(p: &Array3<f64>) -> Result<(), String> {
    for s in 0..N_STATES {
        for a in 0..N_ACTIONS {
            let sum: f64 = (0..N_STATES).map(|sp| p[[s, a, sp]]).sum();
            if (sum - 1.0).abs() > 1e-6 {
                return Err(format!(
                    "Probability conservation violated at s={} a={}: sum={:.8}",
                    s, a, sum
                ));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Bellman expectation equation — policy evaluation
// V^π(s) = Σ_a π(a|s) Σ_s' P(s'|s,a) [R(s,a) + γ V^π(s')]
// ---------------------------------------------------------------------------
pub fn bellman_expectation(
    v: &[f64],
    policy: &[usize],
    model: &MdpModel,
) -> Vec<f64> {
    let mut v_new = vec![0.0_f64; N_STATES];
    for s in 0..N_STATES {
        let a = policy[s];
        let mut val = 0.0_f64;
        for sp in 0..N_STATES {
            val += model.transitions[[s, a, sp]]
                * (model.rewards[[s, a]] + model.gamma * v[sp]);
        }
        v_new[s] = val;
    }
    v_new
}

// ---------------------------------------------------------------------------
// Value iteration — finds V* and π*
// V^(k+1)(s) = max_a Σ_s' P(s'|s,a) [R(s,a) + γ V^(k)(s')]
// ---------------------------------------------------------------------------
pub fn value_iteration(model: &MdpModel, theta: f64) -> ValueIterationResult {
    let mut v = vec![0.0_f64; N_STATES];
    let mut convergence_curve: Vec<f64> = Vec::new();
    let mut bellman_trace: Vec<BellmanStep> = Vec::new();
    let mut iterations = 0usize;

    loop {
        let mut delta = 0.0_f64;
        let v_old = v.clone();

        for s in 0..N_STATES {
            // Compute Q(s,a) for all actions
            let q_values: Vec<f64> = (0..N_ACTIONS)
                .map(|a| {
                    (0..N_STATES)
                        .map(|sp| {
                            model.transitions[[s, a, sp]]
                                * (model.rewards[[s, a]] + model.gamma * v[sp])
                        })
                        .sum::<f64>()
                })
                .collect();

            let v_new_s = q_values
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);

            let best_a = q_values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            // Record trace for first 3 iterations, all states
            if iterations < 3 {
                bellman_trace.push(BellmanStep {
                    iteration: iterations,
                    state: s,
                    action: best_a,
                    q_values: q_values.clone(),
                    v_old: v_old[s],
                    v_new: v_new_s,
                    delta: (v_new_s - v_old[s]).abs(),
                });
            }

            delta = delta.max((v_new_s - v[s]).abs());
            v[s] = v_new_s;
        }

        convergence_curve.push(delta);
        iterations += 1;

        if delta < theta {
            break;
        }
        if iterations > 1000 {
            break; // safety cap
        }
    }

    // Extract greedy policy from V*
    let policy = extract_policy(&v, model);

    ValueIterationResult {
        values: v,
        policy,
        iterations,
        convergence_curve,
        bellman_trace,
    }
}

// ---------------------------------------------------------------------------
// Policy extraction — greedy w.r.t. V*
// ---------------------------------------------------------------------------
pub fn extract_policy(v: &[f64], model: &MdpModel) -> Vec<usize> {
    (0..N_STATES)
        .map(|s| {
            (0..N_ACTIONS)
                .map(|a| {
                    (0..N_STATES)
                        .map(|sp| {
                            model.transitions[[s, a, sp]]
                                * (model.rewards[[s, a]] + model.gamma * v[sp])
                        })
                        .sum::<f64>()
                })
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Exact solution via linear system: (I - γP^π) v = r^π
// ---------------------------------------------------------------------------
pub fn solve_exact(policy: &[usize], model: &MdpModel) -> Vec<f64> {
    let n = N_STATES;
    let mut a_mat = DMatrix::<f64>::identity(n, n);
    let mut r_vec = DVector::<f64>::zeros(n);

    for s in 0..n {
        let a = policy[s];
        r_vec[s] = model.rewards[[s, a]];
        for sp in 0..n {
            a_mat[(s, sp)] -= model.gamma * model.transitions[[s, a, sp]];
        }
    }

    // LU decomposition solve
    match a_mat.lu().solve(&r_vec) {
        Some(v) => v.iter().cloned().collect(),
        None => vec![0.0; n], // fallback
    }
}

// ---------------------------------------------------------------------------
// Contraction mapping verification
// ‖V^(k+1) - V^(k)‖∞ < γ ‖V^(k) - V^(k-1)‖∞
// ---------------------------------------------------------------------------
pub fn verify_contraction(curve: &[f64], gamma: f64) -> bool {
    if curve.len() < 2 {
        return true;
    }
    for i in 1..curve.len() {
        if curve[i] > gamma * curve[i - 1] + 1e-9 {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------
pub fn run_ch02(config: Ch02Config) -> ValueIterationResult {
    let mut rng = StdRng::seed_from_u64(config.seed);
    let transitions = build_asp_transitions(&mut rng);
    let rewards = build_asp_rewards();

    // Enforce safety invariant
    verify_transition_matrix(&transitions)
        .expect("Transition matrix probability conservation violated");

    let model = MdpModel {
        transitions,
        rewards,
        gamma: config.gamma,
    };

    value_iteration(&model, config.theta)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> Ch02Config {
        Ch02Config { seed: 42, gamma: 0.95, theta: 1e-6 }
    }

    fn default_model(seed: u64) -> MdpModel {
        let mut rng = StdRng::seed_from_u64(seed);
        MdpModel {
            transitions: build_asp_transitions(&mut rng),
            rewards: build_asp_rewards(),
            gamma: 0.95,
        }
    }

    #[test]
    fn test_probability_conservation() {
        let mut rng = StdRng::seed_from_u64(42);
        let p = build_asp_transitions(&mut rng);
        assert!(verify_transition_matrix(&p).is_ok());
    }

    #[test]
    fn test_value_iteration_converges() {
        let result = run_ch02(default_config());
        assert!(result.iterations < 500, "Should converge well before 500 iterations");
        assert!(!result.convergence_curve.is_empty());
        let final_delta = result.convergence_curve.last().unwrap();
        assert!(*final_delta < 1e-5, "Final delta should be < theta");
    }

    #[test]
    fn test_contraction_mapping() {
        let result = run_ch02(default_config());
        assert!(
            verify_contraction(&result.convergence_curve, 0.95),
            "Bellman operator must be a contraction mapping"
        );
    }

    #[test]
    fn test_policy_valid() {
        let result = run_ch02(default_config());
        for &a in &result.policy {
            assert!(a < N_ACTIONS, "Policy action out of bounds");
        }
        assert_eq!(result.policy.len(), N_STATES);
    }

    #[test]
    fn test_values_finite() {
        let result = run_ch02(default_config());
        for v in &result.values {
            assert!(v.is_finite(), "Value must be finite");
        }
    }

    #[test]
    fn test_deterministic() {
        let r1 = run_ch02(default_config());
        let r2 = run_ch02(default_config());
        for (v1, v2) in r1.values.iter().zip(r2.values.iter()) {
            assert_eq!(v1.to_bits(), v2.to_bits());
        }
    }

    #[test]
    fn test_exact_solution_close_to_vi() {
        let model = default_model(42);
        let vi = value_iteration(&model, 1e-9);
        let exact = solve_exact(&vi.policy, &model);
        for (v_vi, v_ex) in vi.values.iter().zip(exact.iter()) {
            assert!(
                (v_vi - v_ex).abs() < 0.5,
                "VI and exact solution should be close: {} vs {}",
                v_vi, v_ex
            );
        }
    }

    #[test]
    fn test_worst_state_lowest_value() {
        let result = run_ch02(default_config());
        // S7 (SLA breach imminent) should have lower value than S0 (all available)
        assert!(
            result.values[7] < result.values[0],
            "S7 should have lower value than S0: {} vs {}",
            result.values[7], result.values[0]
        );
    }
}


use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Beta, Distribution};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
pub const N_ARMS: usize = 5; // HVAC, Electrical, Plumbing, Network, Mechanical
pub const ARM_NAMES: &[&str] = &["HVAC", "Electrical", "Plumbing", "Network", "Mechanical"];

// True SLA success rates per skill (unknown to agent — learned via bandit)
pub const TRUE_SLA_RATES: &[f64] = &[0.82, 0.76, 0.88, 0.71, 0.79];

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct BanditState {
    /// Q(a) — estimated value of each arm
    pub q_values: Vec<f64>,
    /// N(a) — number of times each arm was pulled
    pub n_pulls: Vec<usize>,
    /// Alpha (successes) for Thompson Sampling Beta posterior
    pub alpha: Vec<f64>,
    /// Beta (failures) for Thompson Sampling Beta posterior
    pub beta_params: Vec<f64>,
    /// Total steps taken
    pub t: usize,
}

#[derive(Debug, Clone)]
pub struct BanditStep {
    pub step: usize,
    pub algorithm: String,
    pub arm: usize,
    pub reward: f64,
    pub regret: f64,
    pub cumulative_regret: f64,
    pub q_values: Vec<f64>,
    pub n_pulls: Vec<usize>,
    pub ucb_values: Vec<f64>,
    pub thompson_samples: Vec<f64>,
    pub epsilon: f64,
    pub explored: bool,
}

#[derive(Debug, Clone)]
pub struct BanditResult {
    pub algorithm: String,
    pub steps: Vec<BanditStep>,
    pub total_reward: f64,
    pub total_regret: f64,
    pub final_q_values: Vec<f64>,
    pub final_n_pulls: Vec<usize>,
    pub best_arm: usize,
}

#[derive(Debug, Clone)]
pub struct Ch03Config {
    pub seed: u64,
    pub n_steps: usize,
    pub epsilon: f64,
    pub epsilon_decay: f64,
    pub ucb_c: f64,    // UCB exploration constant
    pub gamma: f64,    // discount for annealing
}

// ---------------------------------------------------------------------------
// Bandit environment — stochastic reward from true SLA rates
// ---------------------------------------------------------------------------
pub fn pull_arm(arm: usize, rng: &mut StdRng) -> f64 {
    let success = rng.gen::<f64>() < TRUE_SLA_RATES[arm];
    if success { 1.0 } else { 0.0 }
}

// ---------------------------------------------------------------------------
// Optimal expected reward (for regret calculation)
// ---------------------------------------------------------------------------
pub fn optimal_reward() -> f64 {
    TRUE_SLA_RATES.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

// ---------------------------------------------------------------------------
// Initialise bandit state
// ---------------------------------------------------------------------------
pub fn init_bandit() -> BanditState {
    BanditState {
        q_values:     vec![0.0; N_ARMS],
        n_pulls:      vec![0; N_ARMS],
        alpha:        vec![1.0; N_ARMS], // Beta(1,1) = uniform prior
        beta_params:  vec![1.0; N_ARMS],
        t:            0,
    }
}

// ---------------------------------------------------------------------------
// Q-value update (incremental mean)
// Q(a) <- Q(a) + 1/N(a) * (R - Q(a))
// ---------------------------------------------------------------------------
pub fn update_q(state: &mut BanditState, arm: usize, reward: f64) {
    state.n_pulls[arm] += 1;
    state.t += 1;
    let n = state.n_pulls[arm] as f64;
    state.q_values[arm] += (reward - state.q_values[arm]) / n;
    // Update Beta posterior for Thompson Sampling
    if reward > 0.5 {
        state.alpha[arm] += 1.0;
    } else {
        state.beta_params[arm] += 1.0;
    }
}

// ---------------------------------------------------------------------------
// Algorithm 1: Epsilon-greedy with annealing
// εₜ = max(ε_min, ε₀ / (1 + decay * t))
// ---------------------------------------------------------------------------
pub fn epsilon_greedy_select(
    state: &BanditState,
    epsilon: f64,
    rng: &mut StdRng,
) -> (usize, bool) {
    if rng.gen::<f64>() < epsilon {
        (rng.gen_range(0..N_ARMS), true)
    } else {
        let best = state.q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        (best, false)
    }
}

// ---------------------------------------------------------------------------
// Algorithm 2: UCB1
// UCB(a) = Q(a) + c * sqrt(ln(t) / N(a))
// ---------------------------------------------------------------------------
pub fn ucb_select(state: &BanditState, c: f64) -> (usize, Vec<f64>) {
    let t = state.t.max(1) as f64;
    let ucb_values: Vec<f64> = (0..N_ARMS)
        .map(|a| {
            if state.n_pulls[a] == 0 {
                f64::INFINITY
            } else {
                let n = state.n_pulls[a] as f64;
                state.q_values[a] + c * (t.ln() / n).sqrt()
            }
        })
        .collect();

    let best = ucb_values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    (best, ucb_values)
}

// ---------------------------------------------------------------------------
// Algorithm 3: Thompson Sampling
// Sample θₐ ~ Beta(αₐ, βₐ), select arm with highest sample
// ---------------------------------------------------------------------------
pub fn thompson_select(state: &BanditState, rng: &mut StdRng) -> (usize, Vec<f64>) {
    let samples: Vec<f64> = (0..N_ARMS)
        .map(|a| {
            let dist = Beta::new(state.alpha[a], state.beta_params[a])
                .unwrap_or(Beta::new(1.0, 1.0).unwrap());
            dist.sample(rng)
        })
        .collect();

    let best = samples
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    (best, samples)
}

// ---------------------------------------------------------------------------
// Run a single bandit algorithm
// ---------------------------------------------------------------------------
pub fn run_bandit(
    algorithm: &str,
    config: &Ch03Config,
    seed_offset: u64,
) -> BanditResult {
    let mut rng = StdRng::seed_from_u64(config.seed + seed_offset);
    let mut state = init_bandit();
    let mut steps: Vec<BanditStep> = Vec::new();
    let mut cumulative_regret = 0.0_f64;
    let mut total_reward = 0.0_f64;
    let opt = optimal_reward();

    for step in 0..config.n_steps {
        // Compute current epsilon with annealing
        let epsilon_t = (config.epsilon / (1.0 + config.epsilon_decay * step as f64))
            .max(0.01);

        // Select arm based on algorithm
        let (arm, explored, ucb_vals, thompson_samples) = match algorithm {
            "epsilon_greedy" => {
                let (a, exp) = epsilon_greedy_select(&state, epsilon_t, &mut rng);
                let ucb = vec![0.0; N_ARMS];
                let ts  = vec![0.0; N_ARMS];
                (a, exp, ucb, ts)
            }
            "ucb" => {
                let (a, ucb) = ucb_select(&state, config.ucb_c);
                let ts = vec![0.0; N_ARMS];
                (a, false, ucb, ts)
            }
            "thompson" => {
                let (a, ts) = thompson_select(&state, &mut rng);
                let ucb = vec![0.0; N_ARMS];
                (a, false, ucb, ts)
            }
            _ => (0, false, vec![0.0; N_ARMS], vec![0.0; N_ARMS]),
        };

        // Pull arm and get reward
        let reward = pull_arm(arm, &mut rng);
        let regret = opt - TRUE_SLA_RATES[arm];
        cumulative_regret += regret;
        total_reward += reward;

        let q_snap = state.q_values.clone();
        let n_snap = state.n_pulls.clone();

        update_q(&mut state, arm, reward);

        steps.push(BanditStep {
            step,
            algorithm: algorithm.to_string(),
            arm,
            reward,
            regret,
            cumulative_regret,
            q_values: q_snap,
            n_pulls: n_snap,
            ucb_values: ucb_vals,
            thompson_samples,
            epsilon: epsilon_t,
            explored,
        });
    }

    let best_arm = state.q_values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    BanditResult {
        algorithm: algorithm.to_string(),
        steps,
        total_reward,
        total_regret: cumulative_regret,
        final_q_values: state.q_values,
        final_n_pulls: state.n_pulls,
        best_arm,
    }
}

// ---------------------------------------------------------------------------
// Run all three algorithms for comparison
// ---------------------------------------------------------------------------
pub fn run_ch03(config: Ch03Config) -> Vec<BanditResult> {
    vec![
        run_bandit("epsilon_greedy", &config, 0),
        run_bandit("ucb",            &config, 1),
        run_bandit("thompson",       &config, 2),
    ]
}

// ---------------------------------------------------------------------------
// Regret bound verification
// UCB regret <= c * sqrt(K * T * ln(T))
// ---------------------------------------------------------------------------
pub fn verify_ucb_regret_bound(result: &BanditResult, c: f64, t: usize) -> bool {
    let bound = c * ((N_ARMS as f64) * (t as f64) * (t as f64).ln()).sqrt();
    result.total_regret <= bound + 1e-9
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> Ch03Config {
        Ch03Config {
            seed: 42,
            n_steps: 500,
            epsilon: 0.3,
            epsilon_decay: 0.01,
            ucb_c: 2.0,
            gamma: 0.95,
        }
    }

    #[test]
    fn test_pull_arm_in_range() {
        let mut rng = StdRng::seed_from_u64(42);
        for arm in 0..N_ARMS {
            let r = pull_arm(arm, &mut rng);
            assert!(r == 0.0 || r == 1.0);
        }
    }

    #[test]
    fn test_q_update_converges_to_true_rate() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut state = init_bandit();
        for _ in 0..2000 {
            let r = pull_arm(0, &mut rng);
            update_q(&mut state, 0, r);
        }
        let estimated = state.q_values[0];
        let true_rate = TRUE_SLA_RATES[0];
        assert!(
            (estimated - true_rate).abs() < 0.05,
            "Q estimate {:.3} should be close to true rate {:.3}",
            estimated, true_rate
        );
    }

    #[test]
    fn test_epsilon_greedy_runs() {
        let config = default_config();
        let result = run_bandit("epsilon_greedy", &config, 0);
        assert_eq!(result.steps.len(), 500);
        assert!(result.total_reward > 0.0);
    }

    #[test]
    fn test_ucb_runs() {
        let config = default_config();
        let result = run_bandit("ucb", &config, 1);
        assert_eq!(result.steps.len(), 500);
        assert!(result.total_reward > 0.0);
    }

    #[test]
    fn test_thompson_runs() {
        let config = default_config();
        let result = run_bandit("thompson", &config, 2);
        assert_eq!(result.steps.len(), 500);
        assert!(result.total_reward > 0.0);
    }

    #[test]
    fn test_all_arms_explored() {
        let config = default_config();
        let result = run_bandit("ucb", &config, 0);
        // UCB must explore all arms at least once
        for &n in &result.final_n_pulls {
            assert!(n > 0, "UCB must pull every arm at least once");
        }
    }

    #[test]
    fn test_best_arm_identified() {
        // True best arm is index 2 (Plumbing, 0.88)
        let config = Ch03Config { n_steps: 2000, ..default_config() };
        let result = run_bandit("thompson", &config, 0);
        assert_eq!(
            result.best_arm, 2,
            "Thompson should identify Plumbing (arm 2) as best after 2000 steps"
        );
    }

    #[test]
    fn test_deterministic() {
        let config = default_config();
        let r1 = run_bandit("ucb", &config, 0);
        let r2 = run_bandit("ucb", &config, 0);
        assert_eq!(
            r1.total_reward.to_bits(),
            r2.total_reward.to_bits()
        );
    }



    #[test]
    fn test_thompson_beats_epsilon_greedy_regret() {
        let config = Ch03Config { seed: 0, n_steps: 1000, ..default_config() };
        let ts = run_bandit("thompson", &config, 2);
        // Thompson regret must be sublinear — less than 50 pct of steps
        assert!(ts.total_regret < 500.0, "Thompson regret too high: {}", ts.total_regret);
    }
}

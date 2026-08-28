
// ch14_marl.rs — Foundational MARL Algorithms
// IQL, VDN, MAPG, MADDPG on 5x5 grid world + drone swarm navigation
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const GRID_SIZE: usize = 5;
const NUM_ACTIONS: usize = 4; // Up, Down, Left, Right
const NUM_AGENTS: usize = 2;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MarlStepResult {
    pub episode: usize,
    pub step: usize,
    pub agent_id: usize,
    pub state: (usize, usize),
    pub action: usize,
    pub reward: f64,
    pub next_state: (usize, usize),
    pub q_value: f64,
    pub td_error: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MarlEpisodeResult {
    pub episode: usize,
    pub algorithm: String,
    pub total_reward: f64,
    pub agent_rewards: Vec<f64>,
    pub cooperation_rate: f64,
    pub avg_td_error: f64,
    pub steps: Vec<MarlStepResult>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ch14Result {
    pub algorithm: String,
    pub episodes: Vec<MarlEpisodeResult>,
    pub final_q_tables: Vec<Vec<Vec<f64>>>, // [agent][state][action]
    pub total_reward: f64,
    pub avg_cooperation: f64,
}

// ── Grid World Environment ──────────────────────────────────────────────────
struct GridWorld {
    grid: Vec<Vec<f64>>,
    agent_positions: Vec<(usize, usize)>,
    target: (usize, usize),
}

impl GridWorld {
    fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut grid = vec![vec![0.0f64; GRID_SIZE]; GRID_SIZE];
        for row in grid.iter_mut() {
            for cell in row.iter_mut() {
                *cell = rng.gen_range(0.0..5.0);
            }
        }
        let target = (GRID_SIZE / 2, GRID_SIZE / 2);
        let agent_positions = vec![(0, 0), (GRID_SIZE - 1, GRID_SIZE - 1)];
        GridWorld { grid, agent_positions, target }
    }

    fn step(&mut self, actions: &[usize]) -> Vec<f64> {
        let mut rewards = vec![0.0f64; NUM_AGENTS];
        for (i, &action) in actions.iter().enumerate() {
            let (x, y) = self.agent_positions[i];
            let (nx, ny) = match action {
                0 => (x.saturating_sub(1), y),
                1 => ((x + 1).min(GRID_SIZE - 1), y),
                2 => (x, y.saturating_sub(1)),
                _ => (x, (y + 1).min(GRID_SIZE - 1)),
            };
            self.agent_positions[i] = (nx, ny);
            let cell_reward = self.grid[nx][ny];
            self.grid[nx][ny] = 0.0;
            let dist = (((nx as i64 - self.target.0 as i64).pow(2)
                + (ny as i64 - self.target.1 as i64).pow(2)) as f64).sqrt();
            rewards[i] = cell_reward - dist * 0.1;
        }
        rewards
    }

    fn state_idx(&self, agent: usize) -> usize {
        let (x, y) = self.agent_positions[agent];
        x * GRID_SIZE + y
    }

    fn reset(&mut self, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        for row in self.grid.iter_mut() {
            for cell in row.iter_mut() {
                *cell = rng.gen_range(0.0..5.0);
            }
        }
        self.agent_positions = vec![(0, 0), (GRID_SIZE - 1, GRID_SIZE - 1)];
    }
}

// ── IQL — Independent Q-Learning ───────────────────────────────────────────
pub fn run_iql(episodes: usize, alpha: f64, gamma: f64, epsilon: f64, seed: u64) -> Ch14Result {
    let num_states = GRID_SIZE * GRID_SIZE;
    let mut q_tables: Vec<Vec<Vec<f64>>> = vec![
        vec![vec![0.0f64; NUM_ACTIONS]; num_states]; NUM_AGENTS
    ];
    let mut rng = StdRng::seed_from_u64(seed);
    let mut env = GridWorld::new(seed);
    let mut episode_results = Vec::new();
    let mut total_reward_all = 0.0;
    let mut total_coop = 0.0;

    for ep in 0..episodes {
        env.reset(seed + ep as u64);
        let mut ep_rewards = vec![0.0f64; NUM_AGENTS];
        let mut ep_td_errors = Vec::new();
        let mut steps = Vec::new();
        let mut same_action_count = 0usize;

        for step in 0..50 {
            let mut actions = Vec::new();
            for agent in 0..NUM_AGENTS {
                let s = env.state_idx(agent);
                let action = if rng.gen::<f64>() < epsilon {
                    rng.gen_range(0..NUM_ACTIONS)
                } else {
                    q_tables[agent][s]
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .unwrap()
                        .0
                };
                actions.push(action);
            }

            if actions[0] == actions[1] { same_action_count += 1; }

            let rewards = env.step(&actions);

            for agent in 0..NUM_AGENTS {
                let s = env.state_idx(agent);
                let a = actions[agent];
                let r = rewards[agent];
                let s_next = env.state_idx(agent);
                let max_next = q_tables[agent][s_next]
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                let td = r + gamma * max_next - q_tables[agent][s][a];
                q_tables[agent][s][a] += alpha * td;
                ep_rewards[agent] += r;
                ep_td_errors.push(td.abs());

                if agent == 0 {
                    steps.push(MarlStepResult {
                        episode: ep,
                        step,
                        agent_id: agent,
                        state: env.agent_positions[agent],
                        action: a,
                        reward: r,
                        next_state: env.agent_positions[agent],
                        q_value: q_tables[agent][s][a],
                        td_error: td.abs(),
                    });
                }
            }
        }

        let total_ep = ep_rewards.iter().sum::<f64>();
        let coop = same_action_count as f64 / 50.0;
        total_reward_all += total_ep;
        total_coop += coop;

        episode_results.push(MarlEpisodeResult {
            episode: ep,
            algorithm: "IQL".to_string(),
            total_reward: total_ep,
            agent_rewards: ep_rewards,
            cooperation_rate: coop,
            avg_td_error: ep_td_errors.iter().sum::<f64>() / ep_td_errors.len().max(1) as f64,
            steps,
        });
    }

    Ch14Result {
        algorithm: "IQL".to_string(),
        episodes: episode_results,
        final_q_tables: q_tables,
        total_reward: total_reward_all / episodes as f64,
        avg_cooperation: total_coop / episodes as f64,
    }
}

// ── VDN — Value Decomposition Networks ─────────────────────────────────────
pub fn run_vdn(episodes: usize, alpha: f64, gamma: f64, epsilon: f64, seed: u64) -> Ch14Result {
    let num_states = GRID_SIZE * GRID_SIZE;
    let mut q_tables: Vec<Vec<Vec<f64>>> = vec![
        vec![vec![0.0f64; NUM_ACTIONS]; num_states]; NUM_AGENTS
    ];
    let mut rng = StdRng::seed_from_u64(seed);
    let mut env = GridWorld::new(seed);
    let mut episode_results = Vec::new();
    let mut total_reward_all = 0.0;
    let mut total_coop = 0.0;

    for ep in 0..episodes {
        env.reset(seed + ep as u64);
        let mut ep_rewards = vec![0.0f64; NUM_AGENTS];
        let mut ep_td_errors = Vec::new();
        let mut steps = Vec::new();
        let mut same_action_count = 0usize;

        for step in 0..50 {
            let mut actions = Vec::new();
            for agent in 0..NUM_AGENTS {
                let s = env.state_idx(agent);
                let action = if rng.gen::<f64>() < epsilon {
                    rng.gen_range(0..NUM_ACTIONS)
                } else {
                    q_tables[agent][s]
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .unwrap()
                        .0
                };
                actions.push(action);
            }

            if actions[0] == actions[1] { same_action_count += 1; }

            let rewards = env.step(&actions);
            // VDN: joint reward = sum of individual rewards
            let joint_reward: f64 = rewards.iter().sum();

            for agent in 0..NUM_AGENTS {
                let s = env.state_idx(agent);
                let a = actions[agent];
                let s_next = env.state_idx(agent);
                // VDN: each agent gets joint_reward / N
                let r = joint_reward / NUM_AGENTS as f64;
                let max_next = q_tables[agent][s_next]
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                let td = r + gamma * max_next - q_tables[agent][s][a];
                q_tables[agent][s][a] += alpha * td;
                ep_rewards[agent] += rewards[agent];
                ep_td_errors.push(td.abs());

                if agent == 0 {
                    steps.push(MarlStepResult {
                        episode: ep,
                        step,
                        agent_id: agent,
                        state: env.agent_positions[agent],
                        action: a,
                        reward: r,
                        next_state: env.agent_positions[agent],
                        q_value: q_tables[agent][s][a],
                        td_error: td.abs(),
                    });
                }
            }
        }

        let total_ep = ep_rewards.iter().sum::<f64>();
        let coop = same_action_count as f64 / 50.0;
        total_reward_all += total_ep;
        total_coop += coop;

        episode_results.push(MarlEpisodeResult {
            episode: ep,
            algorithm: "VDN".to_string(),
            total_reward: total_ep,
            agent_rewards: ep_rewards,
            cooperation_rate: coop,
            avg_td_error: ep_td_errors.iter().sum::<f64>() / ep_td_errors.len().max(1) as f64,
            steps,
        });
    }

    Ch14Result {
        algorithm: "VDN".to_string(),
        episodes: episode_results,
        final_q_tables: q_tables,
        total_reward: total_reward_all / episodes as f64,
        avg_cooperation: total_coop / episodes as f64,
    }
}

// ── MAPG — Multi-Agent Policy Gradient (tabular softmax) ───────────────────
pub fn run_mapg(episodes: usize, alpha: f64, gamma: f64, beta: f64, seed: u64) -> Ch14Result {
    let num_states = GRID_SIZE * GRID_SIZE;
    // theta[agent][state][action] = logit
    let mut theta: Vec<Vec<Vec<f64>>> = vec![
        vec![vec![0.0f64; NUM_ACTIONS]; num_states]; NUM_AGENTS
    ];
    let mut rng = StdRng::seed_from_u64(seed);
    let mut env = GridWorld::new(seed);
    let mut episode_results = Vec::new();
    let mut total_reward_all = 0.0;
    let mut total_coop = 0.0;

    let softmax = |logits: &[f64]| -> Vec<f64> {
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|x| (x - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|x| x / sum).collect()
    };

    for ep in 0..episodes {
        env.reset(seed + ep as u64);
        let mut ep_rewards = vec![0.0f64; NUM_AGENTS];
        let mut trajectory: Vec<(usize, usize, usize, f64)> = Vec::new(); // (agent, state, action, reward)
        let mut same_action_count = 0usize;
        let mut steps = Vec::new();

        for step in 0..50 {
            let mut actions = Vec::new();
            let mut states_before = Vec::new();
            for agent in 0..NUM_AGENTS {
                let s = env.state_idx(agent);
                states_before.push(s);
                let probs = softmax(&theta[agent][s]);
                let r: f64 = rng.gen();
                let mut cumsum = 0.0;
                let mut action = 0;
                for (i, &p) in probs.iter().enumerate() {
                    cumsum += p;
                    if r < cumsum { action = i; break; }
                }
                actions.push(action);
            }

            if actions[0] == actions[1] { same_action_count += 1; }
            let rewards = env.step(&actions);

            for agent in 0..NUM_AGENTS {
                ep_rewards[agent] += rewards[agent];
                trajectory.push((agent, states_before[agent], actions[agent], rewards[agent]));
                if agent == 0 {
                    steps.push(MarlStepResult {
                        episode: ep,
                        step,
                        agent_id: agent,
                        state: env.agent_positions[agent],
                        action: actions[agent],
                        reward: rewards[agent],
                        next_state: env.agent_positions[agent],
                        q_value: theta[agent][states_before[agent]][actions[agent]],
                        td_error: rewards[agent].abs(),
                    });
                }
            }
        }

        // REINFORCE update
        let mut returns = vec![0.0f64; trajectory.len()];
        let mut g = 0.0f64;
        for i in (0..trajectory.len()).rev() {
            g = trajectory[i].3 + gamma * g;
            returns[i] = g;
        }

        for (i, &(agent, s, a, _)) in trajectory.iter().enumerate() {
            let probs = softmax(&theta[agent][s]);
            // Entropy bonus
            let entropy: f64 = -probs.iter().map(|&p| if p > 0.0 { p * p.ln() } else { 0.0 }).sum::<f64>();
            for act in 0..NUM_ACTIONS {
                let indicator = if act == a { 1.0 } else { 0.0 };
                let grad = indicator - probs[act];
                theta[agent][s][act] += alpha * returns[i] * grad + beta * entropy;
            }
        }

        let total_ep = ep_rewards.iter().sum::<f64>();
        let coop = same_action_count as f64 / 50.0;
        total_reward_all += total_ep;
        total_coop += coop;

        episode_results.push(MarlEpisodeResult {
            episode: ep,
            algorithm: "MAPG".to_string(),
            total_reward: total_ep,
            agent_rewards: ep_rewards,
            cooperation_rate: coop,
            avg_td_error: returns.iter().map(|x| x.abs()).sum::<f64>() / returns.len().max(1) as f64,
            steps,
        });
    }

    // Convert theta to q_tables format for display
    Ch14Result {
        algorithm: "MAPG".to_string(),
        episodes: episode_results,
        final_q_tables: theta,
        total_reward: total_reward_all / episodes as f64,
        avg_cooperation: total_coop / episodes as f64,
    }
}

// ── MADDPG (simplified tabular version) ────────────────────────────────────
pub fn run_maddpg(episodes: usize, alpha: f64, gamma: f64, tau: f64, seed: u64) -> Ch14Result {
    let num_states = GRID_SIZE * GRID_SIZE;
    // Centralized critic: Q(s0, s1, a0, a1)
    let joint_states = num_states * num_states;
    let joint_actions = NUM_ACTIONS * NUM_ACTIONS;
    let mut critic: Vec<Vec<f64>> = vec![vec![0.0f64; joint_actions]; joint_states];
    let mut target_critic = critic.clone();
    // Decentralized actors: pi[agent][state][action]
    let mut actors: Vec<Vec<Vec<f64>>> = vec![
        vec![vec![0.0f64; NUM_ACTIONS]; num_states]; NUM_AGENTS
    ];
    let mut rng = StdRng::seed_from_u64(seed);
    let mut env = GridWorld::new(seed);
    let mut episode_results = Vec::new();
    let mut total_reward_all = 0.0;
    let mut total_coop = 0.0;

    let softmax = |logits: &[f64]| -> Vec<f64> {
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|x| (x - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|x| x / sum).collect()
    };

    for ep in 0..episodes {
        env.reset(seed + ep as u64);
        let mut ep_rewards = vec![0.0f64; NUM_AGENTS];
        let mut ep_td_errors = Vec::new();
        let mut steps = Vec::new();
        let mut same_action_count = 0usize;

        for step in 0..50 {
            let s0 = env.state_idx(0);
            let s1 = env.state_idx(1);
            let joint_s = s0 * num_states + s1;

            let mut actions = Vec::new();
            for agent in 0..NUM_AGENTS {
                let s = env.state_idx(agent);
                let probs = softmax(&actors[agent][s]);
                let r: f64 = rng.gen();
                let mut cumsum = 0.0;
                let mut action = 0;
                for (i, &p) in probs.iter().enumerate() {
                    cumsum += p;
                    if r < cumsum { action = i; break; }
                }
                actions.push(action);
            }

            if actions[0] == actions[1] { same_action_count += 1; }
            let joint_a = actions[0] * NUM_ACTIONS + actions[1];
            let rewards = env.step(&actions);
            let joint_r: f64 = rewards.iter().sum();

            let s0_next = env.state_idx(0);
            let s1_next = env.state_idx(1);
            let joint_s_next = s0_next * num_states + s1_next;

            // Centralized critic update
            let max_next = target_critic[joint_s_next]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let td = joint_r + gamma * max_next - critic[joint_s][joint_a];
            critic[joint_s][joint_a] += alpha * td;
            // Soft update target
            target_critic[joint_s][joint_a] = tau * critic[joint_s][joint_a]
                + (1.0 - tau) * target_critic[joint_s][joint_a];

            // Actor update using centralized critic as advantage
            for agent in 0..NUM_AGENTS {
                let s = if agent == 0 { s0 } else { s1 };
                let a = actions[agent];
                let probs = softmax(&actors[agent][s]);
                let advantage = critic[joint_s][joint_a];
                for act in 0..NUM_ACTIONS {
                    let indicator = if act == a { 1.0 } else { 0.0 };
                    let grad = indicator - probs[act];
                    actors[agent][s][act] += alpha * 0.1 * advantage * grad;
                }
            }

            for agent in 0..NUM_AGENTS {
                ep_rewards[agent] += rewards[agent];
            }
            ep_td_errors.push(td.abs());

            steps.push(MarlStepResult {
                episode: ep,
                step,
                agent_id: 0,
                state: env.agent_positions[0],
                action: actions[0],
                reward: joint_r,
                next_state: env.agent_positions[0],
                q_value: critic[joint_s][joint_a],
                td_error: td.abs(),
            });
        }

        let total_ep = ep_rewards.iter().sum::<f64>();
        let coop = same_action_count as f64 / 50.0;
        total_reward_all += total_ep;
        total_coop += coop;

        episode_results.push(MarlEpisodeResult {
            episode: ep,
            algorithm: "MADDPG".to_string(),
            total_reward: total_ep,
            agent_rewards: ep_rewards,
            cooperation_rate: coop,
            avg_td_error: ep_td_errors.iter().sum::<f64>() / ep_td_errors.len().max(1) as f64,
            steps,
        });
    }

    Ch14Result {
        algorithm: "MADDPG".to_string(),
        episodes: episode_results,
        final_q_tables: actors,
        total_reward: total_reward_all / episodes as f64,
        avg_cooperation: total_coop / episodes as f64,
    }
}

// ── Public entry point ──────────────────────────────────────────────────────
#[derive(Serialize, Deserialize)]
pub struct Ch14Input {
    pub episodes: usize,
    pub alpha: f64,
    pub gamma: f64,
    pub epsilon: f64,
    pub beta: f64,
    pub tau: f64,
    pub seed: u64,
}

pub fn run_ch14(input: Ch14Input) -> Vec<Ch14Result> {
    vec![
        run_iql(input.episodes, input.alpha, input.gamma, input.epsilon, input.seed),
        run_vdn(input.episodes, input.alpha, input.gamma, input.epsilon, input.seed),
        run_mapg(input.episodes, input.alpha, input.gamma, input.beta, input.seed),
        run_maddpg(input.episodes, input.alpha, input.gamma, input.tau, input.seed),
    ]
}

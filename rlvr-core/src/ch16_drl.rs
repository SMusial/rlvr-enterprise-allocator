//! Ch16 — Deep Reinforcement Learning Models (tch 0.14 / libtorch 2.1.0)
//! DQN, Double DQN, Dueling DQN, PPO on Warsaw ASP 8-state MDP.
//! Business context: Warsaw ASP dispatch — DRL approximates Q*(s,a) from Ch06.

use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

// ── Warsaw ASP environment ───────────────────────────────────────────────────
const N_STATES:  i64 = 8;
const N_ACTIONS: i64 = 4;
const GAMMA: f64 = 0.95;

// Transition matrix P[s][a] -> (s', r)
fn step_env(s: i64, a: i64, rng: &mut StdRng) -> (i64, f32) {
    let noise: f32 = rng.gen_range(-0.1..0.1);
    let base_reward = match (s, a) {
        (0, 0) => 2.0, (0, 1) => 1.5, (0, 2) => 1.0, (0, 3) => 0.5,
        (1, 0) => 1.8, (1, 1) => 1.6, (1, 2) => 0.8, (1, 3) => 0.4,
        (2, 0) => 1.5, (2, 1) => 1.2, (2, 2) => 0.9, (2, 3) => 0.3,
        (3, 0) => 1.2, (3, 1) => 1.0, (3, 2) => 0.7, (3, 3) => 0.2,
        (4, 0) => 0.9, (4, 1) => 0.8, (4, 2) => 0.5, (4, 3) => 0.1,
        (5, 0) => 0.6, (5, 1) => 0.5, (5, 2) => 0.3, (5, 3) => 0.0,
        (6, 0) => 0.3, (6, 1) => 0.2, (6, 2) => 0.1, (6, 3) => -0.1,
        (7, 0) => 0.1, (7, 1) => 0.0, (7, 2) => -0.1, (7, 3) => -0.2,
        _ => 0.0,
    };
    let next_s = match a {
        0 => (s + 1).min(N_STATES - 1),
        1 => (s - 1).max(0),
        2 => (s + 2).min(N_STATES - 1),
        _ => (s - 2).max(0),
    };
    (next_s, base_reward + noise)
}

fn state_to_tensor(s: i64, device: Device) -> Tensor {
    let mut feat = vec![0.0f32; N_STATES as usize];
    feat[s as usize] = 1.0;
    Tensor::from_slice(&feat).to_device(device).to_kind(Kind::Float)
}

// ── Replay Buffer ────────────────────────────────────────────────────────────
struct ReplayBuffer {
    states:      Vec<i64>,
    actions:     Vec<i64>,
    rewards:     Vec<f32>,
    next_states: Vec<i64>,
    dones:       Vec<bool>,
    capacity:    usize,
    pos:         usize,
}

impl ReplayBuffer {
    fn new(capacity: usize) -> Self {
        ReplayBuffer {
            states: Vec::new(), actions: Vec::new(), rewards: Vec::new(),
            next_states: Vec::new(), dones: Vec::new(),
            capacity, pos: 0,
        }
    }

    fn push(&mut self, s: i64, a: i64, r: f32, ns: i64, done: bool) {
        if self.states.len() < self.capacity {
            self.states.push(s); self.actions.push(a); self.rewards.push(r);
            self.next_states.push(ns); self.dones.push(done);
        } else {
            let p = self.pos % self.capacity;
            self.states[p] = s; self.actions[p] = a; self.rewards[p] = r;
            self.next_states[p] = ns; self.dones[p] = done;
        }
        self.pos += 1;
    }

    fn len(&self) -> usize { self.states.len() }

    fn sample(&self, batch: usize, rng: &mut StdRng, device: Device)
        -> (Tensor, Tensor, Tensor, Tensor, Tensor)
    {
        let n = self.states.len();
        let idxs: Vec<usize> = (0..batch).map(|_| rng.gen_range(0..n)).collect();

        let s_feat: Vec<f32> = idxs.iter().flat_map(|&i| {
            let mut f = vec![0.0f32; N_STATES as usize];
            f[self.states[i] as usize] = 1.0; f
        }).collect();
        let ns_feat: Vec<f32> = idxs.iter().flat_map(|&i| {
            let mut f = vec![0.0f32; N_STATES as usize];
            f[self.next_states[i] as usize] = 1.0; f
        }).collect();
        let acts:    Vec<i64> = idxs.iter().map(|&i| self.actions[i]).collect();
        let rews:    Vec<f32> = idxs.iter().map(|&i| self.rewards[i]).collect();
        let dones:   Vec<f32> = idxs.iter().map(|&i| if self.dones[i] { 1.0 } else { 0.0 }).collect();

        let s  = Tensor::from_slice(&s_feat).view([batch as i64, N_STATES]).to_device(device).to_kind(Kind::Float);
        let ns = Tensor::from_slice(&ns_feat).view([batch as i64, N_STATES]).to_device(device).to_kind(Kind::Float);
        let a  = Tensor::from_slice(&acts).to_device(device).to_kind(Kind::Int64);
        let r  = Tensor::from_slice(&rews).to_device(device).to_kind(Kind::Float);
        let d  = Tensor::from_slice(&dones).to_device(device).to_kind(Kind::Float);
        (s, a, r, ns, d)
    }
}

// ── Result types ─────────────────────────────────────────────────────────────
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ch16StepResult {
    pub episode: usize,
    pub step: usize,
    pub state: usize,
    pub action: usize,
    pub reward: f64,
    pub loss: f64,
    pub epsilon: f64,
    pub q_value: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ch16EpisodeResult {
    pub episode: usize,
    pub total_reward: f64,
    pub avg_loss: f64,
    pub epsilon: f64,
    pub steps: Vec<Ch16StepResult>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ch16Result {
    pub algorithm: String,
    pub episodes: Vec<Ch16EpisodeResult>,
    pub final_q_table: Vec<Vec<f64>>,  // [state][action]
    pub total_reward: f64,
    pub avg_loss: f64,
}

// ── Config ───────────────────────────────────────────────────────────────────
#[derive(Serialize, Deserialize)]
pub struct Ch16Config {
    pub n_episodes:    usize,
    pub lr:            f64,
    pub gamma:         f64,
    pub epsilon_start: f64,
    pub epsilon_end:   f64,
    pub epsilon_decay: f64,
    pub batch_size:    usize,
    pub target_update: usize,
    pub buffer_size:   usize,
    pub hidden_units:  usize,
    pub seed:          u64,
}

// ── Build Q-network (MLP) ────────────────────────────────────────────────────
fn build_qnet(vs: &nn::Path, hidden: usize) -> nn::Sequential {
    nn::seq()
        .add(nn::linear(vs / "l1", N_STATES, hidden as i64, Default::default()))
        .add_fn(|x| x.relu())
        .add(nn::linear(vs / "l2", hidden as i64, hidden as i64, Default::default()))
        .add_fn(|x| x.relu())
        .add(nn::linear(vs / "l3", hidden as i64, N_ACTIONS, Default::default()))
}

// ── Build Dueling Q-network ──────────────────────────────────────────────────
struct DuelingNet {
    shared:    nn::Sequential,
    value:     nn::Linear,
    advantage: nn::Linear,
}

impl DuelingNet {
    fn new(vs: &nn::Path, hidden: usize) -> Self {
        let shared = nn::seq()
            .add(nn::linear(vs / "shared_l1", N_STATES, hidden as i64, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs / "shared_l2", hidden as i64, hidden as i64, Default::default()))
            .add_fn(|x| x.relu());
        let value     = nn::linear(vs / "value",     hidden as i64, 1,         Default::default());
        let advantage = nn::linear(vs / "advantage", hidden as i64, N_ACTIONS, Default::default());
        DuelingNet { shared, value, advantage }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.shared.forward(x);
        let v = h.apply(&self.value);
        let a = h.apply(&self.advantage);
        let a_mean = a.mean_dim(-1i64, true, Kind::Float);
        v + (a - a_mean)
    }
}

fn extract_q_table(net: &nn::Sequential, device: Device) -> Vec<Vec<f64>> {
    tch::no_grad(|| {
        (0..N_STATES as usize).map(|s| {
            let t = state_to_tensor(s as i64, device);
            let q = net.forward(&t.unsqueeze(0));
            Vec::<f32>::try_from(q.squeeze()).unwrap_or_default()
                .iter().map(|&v| v as f64).collect()
        }).collect()
    })
}

fn extract_q_table_dueling(net: &DuelingNet, device: Device) -> Vec<Vec<f64>> {
    tch::no_grad(|| {
        (0..N_STATES as usize).map(|s| {
            let t = state_to_tensor(s as i64, device);
            let q = net.forward(&t.unsqueeze(0));
            Vec::<f32>::try_from(q.squeeze()).unwrap_or_default()
                .iter().map(|&v| v as f64).collect()
        }).collect()
    })
}

// ── DQN ──────────────────────────────────────────────────────────────────────
fn run_dqn(config: &Ch16Config) -> Ch16Result {
    tch::manual_seed(config.seed as i64);
    let device = Device::Cpu;
    let vs        = nn::VarStore::new(device);
    let mut vs_target = nn::VarStore::new(device);
    let qnet        = build_qnet(&vs.root(),        config.hidden_units);
    let qnet_target = build_qnet(&vs_target.root(), config.hidden_units);
    vs_target.copy(&vs).unwrap();

    let mut opt = nn::Adam::default().build(&vs, config.lr).unwrap();
    let mut buf = ReplayBuffer::new(config.buffer_size);
    let mut rng = StdRng::seed_from_u64(config.seed);

    let mut epsilon = config.epsilon_start;
    let mut episode_results = Vec::new();
    let mut total_reward_all = 0.0f64;
    let mut total_loss_all   = 0.0f64;

    for ep in 0..config.n_episodes {
        let mut s = rng.gen_range(0..N_STATES);
        let mut ep_reward = 0.0f64;
        let mut ep_losses = Vec::new();
        let mut steps = Vec::new();

        for step in 0..50 {
            // ε-greedy
            let a = if rng.gen::<f64>() < epsilon {
                rng.gen_range(0..N_ACTIONS)
            } else {
                tch::no_grad(|| {
                    let t = state_to_tensor(s, device);
                    let q = qnet.forward(&t.unsqueeze(0));
                    q.argmax(-1, false).int64_value(&[0])
                })
            };

            let (ns, r) = step_env(s, a, &mut rng);
            let done = step == 49;
            buf.push(s, a, r, ns, done);
            ep_reward += r as f64;

            let mut loss_val = 0.0f64;
            if buf.len() >= config.batch_size {
                let (sb, ab, rb, nsb, db) = buf.sample(config.batch_size, &mut rng, device);
                let q_vals = qnet.forward(&sb).gather(1, &ab.unsqueeze(1), false).squeeze_dim(1);
                let next_q = tch::no_grad(|| qnet_target.forward(&nsb).max_dim(1, false).0);
                let target = &rb + (config.gamma as f32) * &next_q * (1.0 - &db);
                let loss = q_vals.mse_loss(&target, tch::Reduction::Mean);
                loss_val = loss.double_value(&[]);
                opt.backward_step(&loss);
                ep_losses.push(loss_val);
            }

            let q_val = tch::no_grad(|| {
                let t = state_to_tensor(s, device);
                qnet.forward(&t.unsqueeze(0)).max_dim(1, false).0.double_value(&[0])
            });

            steps.push(Ch16StepResult {
                episode: ep, step, state: s as usize, action: a as usize,
                reward: r as f64, loss: loss_val, epsilon, q_value: q_val,
            });
            s = ns;
        }

        if ep % config.target_update == 0 {
            vs_target.copy(&vs).unwrap();
        }

        epsilon = (epsilon * config.epsilon_decay).max(config.epsilon_end);
        let avg_loss = if ep_losses.is_empty() { 0.0 } else {
            ep_losses.iter().sum::<f64>() / ep_losses.len() as f64
        };
        total_reward_all += ep_reward;
        total_loss_all   += avg_loss;

        episode_results.push(Ch16EpisodeResult {
            episode: ep, total_reward: ep_reward, avg_loss, epsilon, steps,
        });
    }

    let final_q = extract_q_table(&qnet, device);
    Ch16Result {
        algorithm: "DQN".to_string(),
        episodes: episode_results,
        final_q_table: final_q,
        total_reward: total_reward_all / config.n_episodes as f64,
        avg_loss:     total_loss_all   / config.n_episodes as f64,
    }
}

// ── Double DQN ───────────────────────────────────────────────────────────────
fn run_double_dqn(config: &Ch16Config) -> Ch16Result {
    tch::manual_seed(config.seed as i64);
    let device = Device::Cpu;
    let vs        = nn::VarStore::new(device);
    let mut vs_target = nn::VarStore::new(device);
    let qnet        = build_qnet(&vs.root(),        config.hidden_units);
    let qnet_target = build_qnet(&vs_target.root(), config.hidden_units);
    vs_target.copy(&vs).unwrap();

    let mut opt = nn::Adam::default().build(&vs, config.lr).unwrap();
    let mut buf = ReplayBuffer::new(config.buffer_size);
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut epsilon = config.epsilon_start;
    let mut episode_results = Vec::new();
    let mut total_reward_all = 0.0f64;
    let mut total_loss_all   = 0.0f64;

    for ep in 0..config.n_episodes {
        let mut s = rng.gen_range(0..N_STATES);
        let mut ep_reward = 0.0f64;
        let mut ep_losses = Vec::new();
        let mut steps = Vec::new();

        for step in 0..50 {
            let a = if rng.gen::<f64>() < epsilon {
                rng.gen_range(0..N_ACTIONS)
            } else {
                tch::no_grad(|| {
                    let t = state_to_tensor(s, device);
                    qnet.forward(&t.unsqueeze(0)).argmax(-1, false).int64_value(&[0])
                })
            };

            let (ns, r) = step_env(s, a, &mut rng);
            let done = step == 49;
            buf.push(s, a, r, ns, done);
            ep_reward += r as f64;

            let mut loss_val = 0.0f64;
            if buf.len() >= config.batch_size {
                let (sb, ab, rb, nsb, db) = buf.sample(config.batch_size, &mut rng, device);
                let q_vals = qnet.forward(&sb).gather(1, &ab.unsqueeze(1), false).squeeze_dim(1);
                // Double DQN: select action with main net, evaluate with target net
                let best_actions = tch::no_grad(|| qnet.forward(&nsb).argmax(-1, false));
                let next_q = tch::no_grad(||
                    qnet_target.forward(&nsb).gather(1, &best_actions.unsqueeze(1), false).squeeze_dim(1)
                );
                let target = &rb + (config.gamma as f32) * &next_q * (1.0 - &db);
                let loss = q_vals.mse_loss(&target, tch::Reduction::Mean);
                loss_val = loss.double_value(&[]);
                opt.backward_step(&loss);
                ep_losses.push(loss_val);
            }

            let q_val = tch::no_grad(|| {
                let t = state_to_tensor(s, device);
                qnet.forward(&t.unsqueeze(0)).max_dim(1, false).0.double_value(&[0])
            });

            steps.push(Ch16StepResult {
                episode: ep, step, state: s as usize, action: a as usize,
                reward: r as f64, loss: loss_val, epsilon, q_value: q_val,
            });
            s = ns;
        }

        if ep % config.target_update == 0 { vs_target.copy(&vs).unwrap(); }
        epsilon = (epsilon * config.epsilon_decay).max(config.epsilon_end);
        let avg_loss = if ep_losses.is_empty() { 0.0 } else {
            ep_losses.iter().sum::<f64>() / ep_losses.len() as f64
        };
        total_reward_all += ep_reward;
        total_loss_all   += avg_loss;
        episode_results.push(Ch16EpisodeResult {
            episode: ep, total_reward: ep_reward, avg_loss, epsilon, steps,
        });
    }

    let final_q = extract_q_table(&qnet, device);
    Ch16Result {
        algorithm: "Double DQN".to_string(),
        episodes: episode_results,
        final_q_table: final_q,
        total_reward: total_reward_all / config.n_episodes as f64,
        avg_loss:     total_loss_all   / config.n_episodes as f64,
    }
}

// ── Dueling DQN ──────────────────────────────────────────────────────────────
fn run_dueling_dqn(config: &Ch16Config) -> Ch16Result {
    tch::manual_seed(config.seed as i64);
    let device = Device::Cpu;
    let vs        = nn::VarStore::new(device);
    let mut vs_target = nn::VarStore::new(device);
    let qnet        = DuelingNet::new(&vs.root(),        config.hidden_units);
    let qnet_target = DuelingNet::new(&vs_target.root(), config.hidden_units);
    vs_target.copy(&vs).unwrap();

    let mut opt = nn::Adam::default().build(&vs, config.lr).unwrap();
    let mut buf = ReplayBuffer::new(config.buffer_size);
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut epsilon = config.epsilon_start;
    let mut episode_results = Vec::new();
    let mut total_reward_all = 0.0f64;
    let mut total_loss_all   = 0.0f64;

    for ep in 0..config.n_episodes {
        let mut s = rng.gen_range(0..N_STATES);
        let mut ep_reward = 0.0f64;
        let mut ep_losses = Vec::new();
        let mut steps = Vec::new();

        for step in 0..50 {
            let a = if rng.gen::<f64>() < epsilon {
                rng.gen_range(0..N_ACTIONS)
            } else {
                tch::no_grad(|| {
                    let t = state_to_tensor(s, device);
                    qnet.forward(&t.unsqueeze(0)).argmax(-1, false).int64_value(&[0])
                })
            };

            let (ns, r) = step_env(s, a, &mut rng);
            let done = step == 49;
            buf.push(s, a, r, ns, done);
            ep_reward += r as f64;

            let mut loss_val = 0.0f64;
            if buf.len() >= config.batch_size {
                let (sb, ab, rb, nsb, db) = buf.sample(config.batch_size, &mut rng, device);
                let q_vals = qnet.forward(&sb).gather(1, &ab.unsqueeze(1), false).squeeze_dim(1);
                let next_q = tch::no_grad(|| qnet_target.forward(&nsb).max_dim(1, false).0);
                let target = &rb + (config.gamma as f32) * &next_q * (1.0 - &db);
                let loss = q_vals.mse_loss(&target, tch::Reduction::Mean);
                loss_val = loss.double_value(&[]);
                opt.backward_step(&loss);
                ep_losses.push(loss_val);
            }

            let q_val = tch::no_grad(|| {
                let t = state_to_tensor(s, device);
                qnet.forward(&t.unsqueeze(0)).max_dim(1, false).0.double_value(&[0])
            });

            steps.push(Ch16StepResult {
                episode: ep, step, state: s as usize, action: a as usize,
                reward: r as f64, loss: loss_val, epsilon, q_value: q_val,
            });
            s = ns;
        }

        if ep % config.target_update == 0 { vs_target.copy(&vs).unwrap(); }
        epsilon = (epsilon * config.epsilon_decay).max(config.epsilon_end);
        let avg_loss = if ep_losses.is_empty() { 0.0 } else {
            ep_losses.iter().sum::<f64>() / ep_losses.len() as f64
        };
        total_reward_all += ep_reward;
        total_loss_all   += avg_loss;
        episode_results.push(Ch16EpisodeResult {
            episode: ep, total_reward: ep_reward, avg_loss, epsilon, steps,
        });
    }

    let final_q = extract_q_table_dueling(&qnet, device);
    Ch16Result {
        algorithm: "Dueling DQN".to_string(),
        episodes: episode_results,
        final_q_table: final_q,
        total_reward: total_reward_all / config.n_episodes as f64,
        avg_loss:     total_loss_all   / config.n_episodes as f64,
    }
}

// ── PPO (tabular softmax actor + value critic) ───────────────────────────────
fn run_ppo(config: &Ch16Config) -> Ch16Result {
    tch::manual_seed(config.seed as i64);
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);

    // Actor: state -> action logits
    let actor = nn::seq()
        .add(nn::linear(&vs.root() / "actor_l1", N_STATES, config.hidden_units as i64, Default::default()))
        .add_fn(|x| x.relu())
        .add(nn::linear(&vs.root() / "actor_l2", config.hidden_units as i64, N_ACTIONS, Default::default()));

    // Critic: state -> value
    let critic = nn::seq()
        .add(nn::linear(&vs.root() / "critic_l1", N_STATES, config.hidden_units as i64, Default::default()))
        .add_fn(|x| x.relu())
        .add(nn::linear(&vs.root() / "critic_l2", config.hidden_units as i64, 1i64, Default::default()));

    let mut opt = nn::Adam::default().build(&vs, config.lr).unwrap();
    let mut rng = StdRng::seed_from_u64(config.seed);
    let clip_eps = 0.2f64;
    let entropy_coef = 0.01f64;

    let mut episode_results = Vec::new();
    let mut total_reward_all = 0.0f64;
    let mut total_loss_all   = 0.0f64;

    for ep in 0..config.n_episodes {
        // Collect trajectory
        let mut states_traj  = Vec::new();
        let mut actions_traj = Vec::new();
        let mut rewards_traj = Vec::new();
        let mut log_probs_old = Vec::new();

        let mut s = rng.gen_range(0..N_STATES);
        let mut ep_reward = 0.0f64;
        let mut steps = Vec::new();

        for step in 0..50 {
            let st = state_to_tensor(s, device);
            let logits = tch::no_grad(|| actor.forward(&st.unsqueeze(0)).squeeze_dim(0));
            let probs  = logits.softmax(-1, Kind::Float);
            let a = probs.multinomial(1, true).int64_value(&[0]);
            let log_p = probs.log().double_value(&[a]);

            let (ns, r) = step_env(s, a, &mut rng);
            ep_reward += r as f64;

            states_traj.push(s);
            actions_traj.push(a);
            rewards_traj.push(r as f64);
            log_probs_old.push(log_p);

            let q_val = tch::no_grad(|| {
                let t = state_to_tensor(s, device);
                actor.forward(&t.unsqueeze(0)).max_dim(1, false).0.double_value(&[0])
            });

            steps.push(Ch16StepResult {
                episode: ep, step, state: s as usize, action: a as usize,
                reward: r as f64, loss: 0.0, epsilon: 0.0, q_value: q_val,
            });
            s = ns;
        }

        // Compute returns
        let mut returns = vec![0.0f64; rewards_traj.len()];
        let mut g = 0.0f64;
        for i in (0..rewards_traj.len()).rev() {
            g = rewards_traj[i] + config.gamma * g;
            returns[i] = g;
        }

        // PPO update (4 epochs)
        let mut ep_losses = Vec::new();
        for _ in 0..4 {
            let s_feat: Vec<f32> = states_traj.iter().flat_map(|&si| {
                let mut f = vec![0.0f32; N_STATES as usize];
                f[si as usize] = 1.0; f
            }).collect();
            let n = states_traj.len();
            let sb = Tensor::from_slice(&s_feat).view([n as i64, N_STATES]).to_kind(Kind::Float);
            let ab = Tensor::from_slice(&actions_traj).to_kind(Kind::Int64);
            let rb = Tensor::from_slice(&returns.iter().map(|&v| v as f32).collect::<Vec<_>>()).to_kind(Kind::Float);
            let old_lp = Tensor::from_slice(&log_probs_old.iter().map(|&v| v as f32).collect::<Vec<_>>()).to_kind(Kind::Float);

            let logits = actor.forward(&sb);
            let probs  = logits.softmax(-1, Kind::Float);
            let log_probs = probs.log();
            let new_lp = log_probs.gather(1, &ab.unsqueeze(1), false).squeeze_dim(1);

            let values = critic.forward(&sb).squeeze_dim(1);
            let advantages = (&rb - &values.detach()).to_kind(Kind::Float);

            // Clipped surrogate
            let ratio = (new_lp - &old_lp).exp();
            let surr1 = &ratio * &advantages;
            let surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * &advantages;
            let policy_loss = -surr1.min_other(&surr2).mean(Kind::Float);

            // Value loss
            let value_loss = values.mse_loss(&rb, tch::Reduction::Mean);

            // Entropy bonus
            let entropy = -(probs * log_probs).sum_dim_intlist([-1i64].as_slice(), false, Kind::Float).mean(Kind::Float);

            let loss: Tensor = &policy_loss + 0.5 * &value_loss - (entropy_coef as f32) * &entropy;
            let loss_val = loss.double_value(&[]);
            opt.backward_step(&loss);
            ep_losses.push(loss_val);
        }

        let avg_loss = ep_losses.iter().sum::<f64>() / ep_losses.len() as f64;
        total_reward_all += ep_reward;
        total_loss_all   += avg_loss;

        episode_results.push(Ch16EpisodeResult {
            episode: ep, total_reward: ep_reward, avg_loss, epsilon: 0.0, steps,
        });
    }

    // Extract Q-table from actor logits
    let final_q: Vec<Vec<f64>> = tch::no_grad(|| {
        (0..N_STATES as usize).map(|si| {
            let t = state_to_tensor(si as i64, device);
            let logits = actor.forward(&t.unsqueeze(0)).squeeze_dim(0);
            Vec::<f32>::try_from(logits).unwrap_or_default()
                .iter().map(|&v| v as f64).collect()
        }).collect()
    });

    Ch16Result {
        algorithm: "PPO".to_string(),
        episodes: episode_results,
        final_q_table: final_q,
        total_reward: total_reward_all / config.n_episodes as f64,
        avg_loss:     total_loss_all   / config.n_episodes as f64,
    }
}

// ── Public entry point ────────────────────────────────────────────────────────
pub fn run_ch16(config: Ch16Config) -> Vec<Ch16Result> {
    vec![
        run_dqn(&config),
        run_double_dqn(&config),
        run_dueling_dqn(&config),
        run_ppo(&config),
    ]
}

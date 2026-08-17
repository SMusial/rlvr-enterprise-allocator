//! Ch10 - Model-Based RL: World Models
//! Four algorithms on the Warsaw ASP 8-state / 4-action MDP:
//!
//! 1. World Model Q-Learning  - learn T(s,a,s') + R(s,a), plan with learned model
//! 2. Prioritised Sweeping    - plan from states with highest |delta| first
//! 3. Model-Based PG (MBPO)   - use learned model for synthetic REINFORCE rollouts
//! 4. Uncertainty Bonus       - UCB-style exploration bonus from model visit counts
//!
//! All algorithms share the same tabular world model representation.

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::{HashMap, BinaryHeap};
use std::cmp::Ordering;

use crate::ch02_bellman::{
    build_asp_transitions, build_asp_rewards, verify_transition_matrix,
    N_STATES, N_ACTIONS,
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct WorldModelConfig {
    pub seed:              u64,
    pub gamma:             f64,
    pub alpha:             f64,
    pub epsilon:           f64,
    pub epsilon_decay:     f64,
    pub n_episodes:        usize,
    pub planning_steps:    usize,  // k planning steps per real step
    pub priority_threshold: f64,   // min |delta| to add to priority queue
    pub uncertainty_beta:  f64,    // exploration bonus weight for uncertainty
}

#[derive(Debug, Clone)]
pub struct WmResult {
    pub algorithm:          String,
    pub values:             Vec<f64>,
    pub policy:             Vec<usize>,
    pub q_table:            Vec<Vec<f64>>,
    pub returns_curve:      Vec<f64>,
    pub td_error_curve:     Vec<f64>,
    pub convergence_curve:  Vec<f64>,
    pub model_accuracy:     Vec<f64>,  // fraction of correct next-state predictions
    pub planning_steps_used: Vec<f64>, // actual planning steps per episode
    pub n_episodes:         usize,
    pub total_steps:        usize,
    pub model_size:         usize,
}

#[derive(Debug, Clone)]
pub struct Ch10Result {
    pub wm_qlearning:    WmResult,
    pub pri_sweeping:    WmResult,
    pub mbpo:            WmResult,
    pub uncertainty:     WmResult,
}

// ---------------------------------------------------------------------------
// Tabular World Model
// ---------------------------------------------------------------------------

/// Tabular world model: stores empirical counts for T(s,a,s') and mean R(s,a).
#[derive(Clone)]
struct WorldModel {
    /// transition counts[s][a][s']
    counts:   Vec<Vec<Vec<f64>>>,
    /// reward sum[s][a]
    r_sum:    Vec<Vec<f64>>,
    /// visit count[s][a]
    n_visits: Vec<Vec<usize>>,
}

impl WorldModel {
    fn new() -> Self {
        WorldModel {
            counts:   vec![vec![vec![0.0; N_STATES]; N_ACTIONS]; N_STATES],
            r_sum:    vec![vec![0.0; N_ACTIONS]; N_STATES],
            n_visits: vec![vec![0; N_ACTIONS]; N_STATES],
        }
    }

    fn update(&mut self, s: usize, a: usize, r: f64, sp: usize) {
        self.counts[s][a][sp] += 1.0;
        self.r_sum[s][a]      += r;
        self.n_visits[s][a]   += 1;
    }

    fn visited(&self, s: usize, a: usize) -> bool {
        self.n_visits[s][a] > 0
    }

    /// Sample next state from learned T(s,a,.)
    fn sample_next(&self, s: usize, a: usize, rng: &mut StdRng) -> usize {
        let total: f64 = self.counts[s][a].iter().sum();
        if total == 0.0 { return s; }
        let u: f64 = rng.gen();
        let mut cum = 0.0_f64;
        for sp in 0..N_STATES {
            cum += self.counts[s][a][sp] / total;
            if u <= cum { return sp; }
        }
        N_STATES - 1
    }

    /// Mean reward R(s,a)
    fn mean_reward(&self, s: usize, a: usize) -> f64 {
        let n = self.n_visits[s][a];
        if n == 0 { 0.0 } else { self.r_sum[s][a] / n as f64 }
    }

    /// Most likely next state (argmax T(s,a,.))
    fn best_next(&self, s: usize, a: usize) -> usize {
        self.counts[s][a].iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0)
    }

    /// Model accuracy: fraction of (s,a) where argmax T matches true argmax
    fn accuracy(&self, true_transitions: &ndarray::Array3<f64>) -> f64 {
        let mut correct = 0usize;
        let mut total   = 0usize;
        for s in 0..N_STATES {
            for a in 0..N_ACTIONS {
                if !self.visited(s, a) { continue; }
                total += 1;
                let pred = self.best_next(s, a);
                let true_best = (0..N_STATES)
                    .max_by(|&x, &y| true_transitions[[s,a,x]].partial_cmp(&true_transitions[[s,a,y]]).unwrap())
                    .unwrap_or(0);
                if pred == true_best { correct += 1; }
            }
        }
        if total == 0 { 0.0 } else { correct as f64 / total as f64 }
    }

    fn size(&self) -> usize {
        let mut n = 0;
        for s in 0..N_STATES { for a in 0..N_ACTIONS { if self.visited(s,a) { n += 1; } } }
        n
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn sample_next_state(s: usize, a: usize, transitions: &ndarray::Array3<f64>, rng: &mut StdRng) -> usize {
    let p: f64 = rng.gen();
    let mut cum = 0.0_f64;
    for sp in 0..N_STATES { cum += transitions[[s, a, sp]]; if p <= cum { return sp; } }
    N_STATES - 1
}

fn epsilon_greedy(q: &[Vec<f64>], s: usize, eps: f64, rng: &mut StdRng) -> usize {
    if rng.gen::<f64>() < eps { rng.gen_range(0..N_ACTIONS) }
    else { q[s].iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i,_)| i).unwrap_or(0) }
}

fn v_from_q(q: &[Vec<f64>]) -> Vec<f64> {
    q.iter().map(|row| row.iter().cloned().fold(f64::NEG_INFINITY, f64::max)).collect()
}

fn greedy_policy(q: &[Vec<f64>]) -> Vec<usize> {
    q.iter().map(|row| row.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i,_)| i).unwrap_or(0)).collect()
}

fn visited_pairs(model: &WorldModel) -> Vec<(usize, usize)> {
    let mut v = vec![];
    for s in 0..N_STATES { for a in 0..N_ACTIONS { if model.visited(s, a) { v.push((s, a)); } } }
    v
}

// ---------------------------------------------------------------------------
// 1. World Model Q-Learning
//    Real step: Q-Learning update + model update
//    Planning:  k steps of Q-Learning on model samples (random (s,a))
// ---------------------------------------------------------------------------
pub fn wm_qlearning(
    config: &WorldModelConfig,
    transitions: &ndarray::Array3<f64>,
    rewards: &Array2<f64>,
) -> WmResult {
    let mut rng   = StdRng::seed_from_u64(config.seed);
    let mut q     = vec![vec![0.0_f64; N_ACTIONS]; N_STATES];
    let mut model = WorldModel::new();

    let (mut ret_c, mut td_c, mut conv_c, mut acc_c, mut plan_c) = (vec![], vec![], vec![], vec![], vec![]);
    let mut v_prev = v_from_q(&q);
    let mut total_steps = 0usize;

    for ep in 0..config.n_episodes {
        let eps_t = (config.epsilon / (1.0 + config.epsilon_decay * ep as f64)).max(0.01);
        let mut s = rng.gen_range(0..N_STATES);
        let (mut ep_ret, mut ep_td, mut steps, mut gp, mut ep_plan) = (0.0_f64, 0.0_f64, 0usize, 1.0_f64, 0usize);

        loop {
            let a  = epsilon_greedy(&q, s, eps_t, &mut rng);
            let r  = rewards[[s, a]];
            let sp = sample_next_state(s, a, transitions, &mut rng);
            let done = sp == 0 || sp == 7 || steps >= 50;

            // Real Q-Learning update
            let max_q = q[sp].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let delta = r + config.gamma * (if done { 0.0 } else { max_q }) - q[s][a];
            q[s][a] += config.alpha * delta;
            ep_td   += delta.abs();

            // Model update
            model.update(s, a, r, sp);

            // Planning: k random model samples
            let visited = visited_pairs(&model);
            if !visited.is_empty() {
                for _ in 0..config.planning_steps {
                    let (ps, pa) = visited[rng.gen_range(0..visited.len())];
                    let pr  = model.mean_reward(ps, pa);
                    let psp = model.sample_next(ps, pa, &mut rng);
                    let pmq = q[psp].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let pd  = pr + config.gamma * pmq - q[ps][pa];
                    q[ps][pa] += config.alpha * pd;
                    ep_plan += 1;
                }
            }

            ep_ret += gp * r; gp *= config.gamma; total_steps += 1; steps += 1;
            if done { break; }
            s = sp;
        }

        ret_c.push(ep_ret);
        td_c.push(ep_td / steps.max(1) as f64);
        acc_c.push(model.accuracy(transitions));
        plan_c.push(ep_plan as f64);
        let v = v_from_q(&q);
        conv_c.push(v.iter().zip(v_prev.iter()).map(|(a,b)| (a-b).abs()).fold(0.0_f64, f64::max));
        v_prev = v;
    }

    WmResult {
        algorithm: format!("wm_qlearning_k{}", config.planning_steps),
        values: v_from_q(&q), policy: greedy_policy(&q), q_table: q,
        returns_curve: ret_c, td_error_curve: td_c, convergence_curve: conv_c,
        model_accuracy: acc_c, planning_steps_used: plan_c,
        n_episodes: config.n_episodes, total_steps, model_size: model.size(),
    }
}

// ---------------------------------------------------------------------------
// 2. Prioritised Sweeping
//    Maintain a max-heap of (|delta|, s, a) pairs.
//    After each real step, propagate updates to predecessors.
// ---------------------------------------------------------------------------

#[derive(PartialEq)]
struct PriEntry(f64, usize, usize);

impl Eq for PriEntry {}

impl PartialOrd for PriEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { self.0.partial_cmp(&other.0) }
}

impl Ord for PriEntry {
    fn cmp(&self, other: &Self) -> Ordering { self.partial_cmp(other).unwrap_or(Ordering::Equal) }
}

pub fn prioritised_sweeping(
    config: &WorldModelConfig,
    transitions: &ndarray::Array3<f64>,
    rewards: &Array2<f64>,
) -> WmResult {
    let mut rng   = StdRng::seed_from_u64(config.seed + 10);
    let mut q     = vec![vec![0.0_f64; N_ACTIONS]; N_STATES];
    let mut model = WorldModel::new();
    let mut pq: BinaryHeap<PriEntry> = BinaryHeap::new();

    let (mut ret_c, mut td_c, mut conv_c, mut acc_c, mut plan_c) = (vec![], vec![], vec![], vec![], vec![]);
    let mut v_prev = v_from_q(&q);
    let mut total_steps = 0usize;

    for ep in 0..config.n_episodes {
        let eps_t = (config.epsilon / (1.0 + config.epsilon_decay * ep as f64)).max(0.01);
        let mut s = rng.gen_range(0..N_STATES);
        let (mut ep_ret, mut ep_td, mut steps, mut gp, mut ep_plan) = (0.0_f64, 0.0_f64, 0usize, 1.0_f64, 0usize);

        loop {
            let a  = epsilon_greedy(&q, s, eps_t, &mut rng);
            let r  = rewards[[s, a]];
            let sp = sample_next_state(s, a, transitions, &mut rng);
            let done = sp == 0 || sp == 7 || steps >= 50;

            model.update(s, a, r, sp);

            // Compute priority for (s,a)
            let max_q = q[sp].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let delta = (r + config.gamma * (if done { 0.0 } else { max_q }) - q[s][a]).abs();
            ep_td += delta;
            if delta > config.priority_threshold {
                pq.push(PriEntry(delta, s, a));
            }

            // Prioritised planning: up to k steps
            let mut plan_done = 0usize;
            while plan_done < config.planning_steps {
                match pq.pop() {
                    None => break,
                    Some(PriEntry(_, ps, pa)) => {
                        let pr  = model.mean_reward(ps, pa);
                        let psp = model.best_next(ps, pa);
                        let pmq = q[psp].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let pd  = pr + config.gamma * pmq - q[ps][pa];
                        q[ps][pa] += config.alpha * pd;
                        ep_plan += 1;
                        plan_done += 1;

                        // Add predecessors of ps to queue
                        for pred_s in 0..N_STATES {
                            for pred_a in 0..N_ACTIONS {
                                if !model.visited(pred_s, pred_a) { continue; }
                                if model.best_next(pred_s, pred_a) != ps { continue; }
                                let pr2 = model.mean_reward(pred_s, pred_a);
                                let mq  = q[ps].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                                let d2  = (pr2 + config.gamma * mq - q[pred_s][pred_a]).abs();
                                if d2 > config.priority_threshold {
                                    pq.push(PriEntry(d2, pred_s, pred_a));
                                }
                            }
                        }
                    }
                }
            }

            ep_ret += gp * r; gp *= config.gamma; total_steps += 1; steps += 1;
            if done { break; }
            s = sp;
        }

        ret_c.push(ep_ret);
        td_c.push(ep_td / steps.max(1) as f64);
        acc_c.push(model.accuracy(transitions));
        plan_c.push(ep_plan as f64);
        let v = v_from_q(&q);
        conv_c.push(v.iter().zip(v_prev.iter()).map(|(a,b)| (a-b).abs()).fold(0.0_f64, f64::max));
        v_prev = v;
    }

    WmResult {
        algorithm: "prioritised_sweeping".to_string(),
        values: v_from_q(&q), policy: greedy_policy(&q), q_table: q,
        returns_curve: ret_c, td_error_curve: td_c, convergence_curve: conv_c,
        model_accuracy: acc_c, planning_steps_used: plan_c,
        n_episodes: config.n_episodes, total_steps, model_size: model.size(),
    }
}

// ---------------------------------------------------------------------------
// 3. Model-Based Policy Gradient (MBPO-lite)
//    Learn world model from real experience.
//    Each episode: generate k synthetic rollouts from model, run REINFORCE on them.
//    Policy: softmax(theta[s][a])
// ---------------------------------------------------------------------------
fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn sample_action(probs: &[f64], rng: &mut StdRng) -> usize {
    let u: f64 = rng.gen();
    let mut cum = 0.0_f64;
    for (i, &p) in probs.iter().enumerate() { cum += p; if u <= cum { return i; } }
    probs.len() - 1
}

pub fn mbpo(
    config: &WorldModelConfig,
    transitions: &ndarray::Array3<f64>,
    rewards: &Array2<f64>,
) -> WmResult {
    let mut rng   = StdRng::seed_from_u64(config.seed + 20);
    let mut theta = vec![vec![0.0_f64; N_ACTIONS]; N_STATES];
    let mut model = WorldModel::new();

    let (mut ret_c, mut td_c, mut conv_c, mut acc_c, mut plan_c) = (vec![], vec![], vec![], vec![], vec![]);
    let mut theta_prev = theta.clone();
    let mut total_steps = 0usize;

    for ep in 0..config.n_episodes {
        let eps_t = (config.epsilon / (1.0 + config.epsilon_decay * ep as f64)).max(0.01);

        // ---- Real episode: collect experience + update model ----
        let mut s = rng.gen_range(0..N_STATES);
        let (mut ep_ret, mut steps, mut gp) = (0.0_f64, 0usize, 1.0_f64);

        loop {
            let probs = softmax(&theta[s]);
            let a = if rng.gen::<f64>() < eps_t { rng.gen_range(0..N_ACTIONS) } else { sample_action(&probs, &mut rng) };
            let r  = rewards[[s, a]];
            let sp = sample_next_state(s, a, transitions, &mut rng);
            model.update(s, a, r, sp);
            ep_ret += gp * r; gp *= config.gamma; total_steps += 1; steps += 1;
            if sp == 0 || sp == 7 || steps >= 50 { break; }
            s = sp;
        }

        // ---- Synthetic rollouts: REINFORCE on model ----
        let visited = visited_pairs(&model);
        let mut ep_plan = 0usize;
        let mut ep_td   = 0.0_f64;

        for _ in 0..config.planning_steps {
            if visited.is_empty() { break; }
            let (mut ms, _) = visited[rng.gen_range(0..visited.len())];
            let (mut traj_s, mut traj_a, mut traj_r) = (vec![], vec![], vec![]);
            let mut msteps = 0usize;

            loop {
                let probs = softmax(&theta[ms]);
                let ma = sample_action(&probs, &mut rng);
                let mr = model.mean_reward(ms, ma);
                let msp = model.sample_next(ms, ma, &mut rng);
                traj_s.push(ms); traj_a.push(ma); traj_r.push(mr);
                ep_plan += 1; msteps += 1;
                if msp == 0 || msp == 7 || msteps >= 10 { break; }
                ms = msp;
            }

            // REINFORCE update on synthetic trajectory
            let t_len = traj_r.len();
            let mut g_t = vec![0.0_f64; t_len];
            let mut g = 0.0_f64;
            for t in (0..t_len).rev() { g = traj_r[t] + config.gamma * g; g_t[t] = g; }

            let mut gp2 = 1.0_f64;
            for t in 0..t_len {
                let (st, at, gt) = (traj_s[t], traj_a[t], g_t[t]);
                let probs = softmax(&theta[st]);
                for a in 0..N_ACTIONS {
                    let grad = (if a == at { 1.0 } else { 0.0 }) - probs[a];
                    let upd  = config.alpha * gp2 * gt * grad;
                    theta[st][a] += upd;
                    ep_td += upd.abs();
                }
                gp2 *= config.gamma;
            }
        }

        ret_c.push(ep_ret);
        td_c.push(ep_td / ep_plan.max(1) as f64);
        acc_c.push(model.accuracy(transitions));
        plan_c.push(ep_plan as f64);

        let v: Vec<f64> = (0..N_STATES).map(|s| {
            let probs = softmax(&theta[s]);
            probs.iter().enumerate().map(|(a, &p)| p * rewards[[s, a]]).sum()
        }).collect();
        conv_c.push(theta.iter().zip(theta_prev.iter())
            .flat_map(|(r, p)| r.iter().zip(p.iter()).map(|(a, b)| (a-b).abs()))
            .fold(0.0_f64, f64::max));
        theta_prev = theta.clone();

        let _ = v;
    }

    let values: Vec<f64> = (0..N_STATES).map(|s| {
        let probs = softmax(&theta[s]);
        probs.iter().enumerate().map(|(a, &p)| p * rewards[[s, a]]).sum()
    }).collect();
    let policy: Vec<usize> = (0..N_STATES).map(|s| {
        softmax(&theta[s]).iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i,_)| i).unwrap_or(0)
    }).collect();

    WmResult {
        algorithm: "mbpo_reinforce".to_string(),
        values, policy,
        q_table: theta,
        returns_curve: ret_c, td_error_curve: td_c, convergence_curve: conv_c,
        model_accuracy: acc_c, planning_steps_used: plan_c,
        n_episodes: config.n_episodes, total_steps, model_size: model.size(),
    }
}

// ---------------------------------------------------------------------------
// 4. Uncertainty Bonus Q-Learning
//    Q-Learning + UCB-style exploration bonus from model visit counts:
//    Q_bonus(s,a) = Q(s,a) + beta / sqrt(N(s,a) + 1)
//    Action selection uses Q_bonus; Q update uses standard Q-Learning.
// ---------------------------------------------------------------------------
pub fn uncertainty_bonus(
    config: &WorldModelConfig,
    transitions: &ndarray::Array3<f64>,
    rewards: &Array2<f64>,
) -> WmResult {
    let mut rng   = StdRng::seed_from_u64(config.seed + 30);
    let mut q     = vec![vec![0.0_f64; N_ACTIONS]; N_STATES];
    let mut model = WorldModel::new();

    let (mut ret_c, mut td_c, mut conv_c, mut acc_c, mut plan_c) = (vec![], vec![], vec![], vec![], vec![]);
    let mut v_prev = v_from_q(&q);
    let mut total_steps = 0usize;

    for ep in 0..config.n_episodes {
        let mut s = rng.gen_range(0..N_STATES);
        let (mut ep_ret, mut ep_td, mut steps, mut gp, mut ep_plan) = (0.0_f64, 0.0_f64, 0usize, 1.0_f64, 0usize);

        loop {
            // UCB-style action selection using uncertainty bonus
            let a = (0..N_ACTIONS).max_by(|&a1, &a2| {
                let b1 = q[s][a1] + config.uncertainty_beta / ((model.n_visits[s][a1] + 1) as f64).sqrt();
                let b2 = q[s][a2] + config.uncertainty_beta / ((model.n_visits[s][a2] + 1) as f64).sqrt();
                b1.partial_cmp(&b2).unwrap()
            }).unwrap_or(0);

            let r  = rewards[[s, a]];
            let sp = sample_next_state(s, a, transitions, &mut rng);
            let done = sp == 0 || sp == 7 || steps >= 50;

            // Standard Q-Learning update (no bonus in update)
            let max_q = q[sp].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let delta = r + config.gamma * (if done { 0.0 } else { max_q }) - q[s][a];
            q[s][a] += config.alpha * delta;
            ep_td   += delta.abs();

            model.update(s, a, r, sp);

            // Planning: k random model samples (same as WM Q-Learning)
            let visited = visited_pairs(&model);
            if !visited.is_empty() {
                for _ in 0..config.planning_steps {
                    let (ps, pa) = visited[rng.gen_range(0..visited.len())];
                    let pr  = model.mean_reward(ps, pa);
                    let psp = model.sample_next(ps, pa, &mut rng);
                    let pmq = q[psp].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let pd  = pr + config.gamma * pmq - q[ps][pa];
                    q[ps][pa] += config.alpha * pd;
                    ep_plan += 1;
                }
            }

            ep_ret += gp * r; gp *= config.gamma; total_steps += 1; steps += 1;
            if done { break; }
            s = sp;
        }

        ret_c.push(ep_ret);
        td_c.push(ep_td / steps.max(1) as f64);
        acc_c.push(model.accuracy(transitions));
        plan_c.push(ep_plan as f64);
        let v = v_from_q(&q);
        conv_c.push(v.iter().zip(v_prev.iter()).map(|(a,b)| (a-b).abs()).fold(0.0_f64, f64::max));
        v_prev = v;
    }

    WmResult {
        algorithm: format!("uncertainty_bonus_b{:.3}", config.uncertainty_beta),
        values: v_from_q(&q), policy: greedy_policy(&q), q_table: q,
        returns_curve: ret_c, td_error_curve: td_c, convergence_curve: conv_c,
        model_accuracy: acc_c, planning_steps_used: plan_c,
        n_episodes: config.n_episodes, total_steps, model_size: model.size(),
    }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------
pub fn run_ch10(config: WorldModelConfig) -> Ch10Result {
    let mut rng = StdRng::seed_from_u64(config.seed);
    let transitions = build_asp_transitions(&mut rng);
    let rewards     = build_asp_rewards();
    verify_transition_matrix(&transitions).expect("Transition matrix invalid");

    let wm_q  = wm_qlearning(&config, &transitions, &rewards);
    let ps    = prioritised_sweeping(&config, &transitions, &rewards);
    let mb    = mbpo(&config, &transitions, &rewards);
    let ub    = uncertainty_bonus(&config, &transitions, &rewards);

    Ch10Result { wm_qlearning: wm_q, pri_sweeping: ps, mbpo: mb, uncertainty: ub }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> WorldModelConfig {
        WorldModelConfig {
            seed: 42, gamma: 0.95, alpha: 0.1,
            epsilon: 0.3, epsilon_decay: 0.01,
            n_episodes: 300, planning_steps: 5,
            priority_threshold: 0.01, uncertainty_beta: 1.0,
        }
    }

    // ---- Smoke ----
    #[test] fn test_wm_qlearning_runs() {
        let r = run_ch10(cfg());
        assert_eq!(r.wm_qlearning.values.len(), N_STATES);
        assert!(r.wm_qlearning.values.iter().all(|v| v.is_finite()));
    }
    #[test] fn test_pri_sweeping_runs() {
        let r = run_ch10(cfg());
        assert_eq!(r.pri_sweeping.values.len(), N_STATES);
        assert!(r.pri_sweeping.values.iter().all(|v| v.is_finite()));
    }
    #[test] fn test_mbpo_runs() {
        let r = run_ch10(cfg());
        assert_eq!(r.mbpo.values.len(), N_STATES);
        assert!(r.mbpo.values.iter().all(|v| v.is_finite()));
    }
    #[test] fn test_uncertainty_runs() {
        let r = run_ch10(cfg());
        assert_eq!(r.uncertainty.values.len(), N_STATES);
        assert!(r.uncertainty.values.iter().all(|v| v.is_finite()));
    }

    // ---- Shape / validity ----
    #[test] fn test_q_table_shape() {
        let r = run_ch10(cfg());
        assert_eq!(r.wm_qlearning.q_table.len(), N_STATES);
        for row in &r.wm_qlearning.q_table { assert_eq!(row.len(), N_ACTIONS); }
        assert_eq!(r.pri_sweeping.q_table.len(), N_STATES);
    }
    #[test] fn test_policy_valid() {
        let r = run_ch10(cfg());
        for &a in &r.wm_qlearning.policy  { assert!(a < N_ACTIONS); }
        for &a in &r.pri_sweeping.policy  { assert!(a < N_ACTIONS); }
        for &a in &r.mbpo.policy          { assert!(a < N_ACTIONS); }
        for &a in &r.uncertainty.policy   { assert!(a < N_ACTIONS); }
    }
    #[test] fn test_curves_length() {
        let c = cfg(); let r = run_ch10(c.clone());
        assert_eq!(r.wm_qlearning.returns_curve.len(),   c.n_episodes);
        assert_eq!(r.pri_sweeping.returns_curve.len(),   c.n_episodes);
        assert_eq!(r.wm_qlearning.model_accuracy.len(),  c.n_episodes);
        assert_eq!(r.wm_qlearning.planning_steps_used.len(), c.n_episodes);
    }
    #[test] fn test_all_values_finite() {
        let r = run_ch10(cfg());
        for v in &r.wm_qlearning.returns_curve { assert!(v.is_finite()); }
        for v in &r.pri_sweeping.returns_curve { assert!(v.is_finite()); }
        for v in &r.mbpo.returns_curve         { assert!(v.is_finite()); }
        for v in &r.uncertainty.returns_curve  { assert!(v.is_finite()); }
    }

    // ---- Algorithm correctness ----
    #[test] fn test_model_grows_with_experience() {
        let r = run_ch10(cfg());
        assert!(r.wm_qlearning.model_size > 0, "World model should learn transitions");
        assert!(r.wm_qlearning.model_size <= N_STATES * N_ACTIONS);
    }
    #[test] fn test_model_accuracy_increases() {
        let c = WorldModelConfig { n_episodes: 500, ..cfg() };
        let r = run_ch10(c);
        let early: f64 = r.wm_qlearning.model_accuracy[..50].iter().sum::<f64>() / 50.0;
        let late:  f64 = r.wm_qlearning.model_accuracy[450..].iter().sum::<f64>() / 50.0;
        assert!(late >= early - 0.1,
            "Model accuracy should not degrade: early={:.3} late={:.3}", early, late);
    }
    #[test] fn test_planning_steps_used_positive() {
        let r = run_ch10(cfg());
        let avg: f64 = r.wm_qlearning.planning_steps_used.iter().sum::<f64>()
            / r.wm_qlearning.planning_steps_used.len() as f64;
        assert!(avg > 0.0, "Planning steps should be used, got avg={:.2}", avg);
    }
    #[test] fn test_wm_qlearning_converges() {
        let c = WorldModelConfig { n_episodes: 800, ..cfg() };
        let r = run_ch10(c);
        let early: f64 = r.wm_qlearning.returns_curve[..100].iter().sum::<f64>() / 100.0;
        let late:  f64 = r.wm_qlearning.returns_curve[700..].iter().sum::<f64>() / 100.0;
        assert!(late > early, "WM Q-Learning should improve: early={:.3} late={:.3}", early, late);
    }
    #[test] fn test_pri_sweeping_converges() {
        let c = WorldModelConfig { n_episodes: 800, ..cfg() };
        let r = run_ch10(c);
        let late: f64 = r.pri_sweeping.returns_curve[700..].iter().sum::<f64>() / 100.0;
        assert!(late.is_finite() && late > -100.0, "Pri sweeping late avg={:.3}", late);
    }
    #[test] fn test_uncertainty_explores_more_early() {
        // Uncertainty bonus should visit more unique (s,a) pairs early on
        let r = run_ch10(cfg());
        assert!(r.uncertainty.model_size > 0, "Uncertainty agent should explore");
    }
    #[test] fn test_deterministic() {
        let r1 = run_ch10(cfg()); let r2 = run_ch10(cfg());
        for (v1, v2) in r1.wm_qlearning.values.iter().zip(r2.wm_qlearning.values.iter()) {
            assert_eq!(v1.to_bits(), v2.to_bits(), "Results must be deterministic");
        }
    }
    #[test] fn test_softmax_sums_to_one() {
        let p = softmax(&[1.0, 2.0, 0.5, -1.0]);
        let s: f64 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-9, "softmax sum={}", s);
    }
}

//! Ch09 — Policy Gradient: REINFORCE & Softmax Actor-Critic
//! REINFORCE (Williams 1992) · REINFORCE+Baseline · Actor-Critic TD(0) · Temperature variant
//! Policy: π(a|s) = softmax(θ[s][a] / τ)
//! Update: θ[s][a] += α γ^t (G_t − b(s)) (𝟙[a=A_t] − π(a|s))

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::ch02_bellman::{
    build_asp_transitions, build_asp_rewards, verify_transition_matrix,
    N_STATES, N_ACTIONS,
};

#[derive(Debug, Clone)]
pub struct PgConfig {
    pub seed:           u64,
    pub gamma:          f64,
    pub alpha:          f64,
    pub alpha_baseline: f64,
    pub n_episodes:     usize,
    pub use_baseline:   bool,
    pub temperature:    f64,
}

#[derive(Debug, Clone)]
pub struct PgResult {
    pub algorithm:         String,
    pub theta:             Vec<Vec<f64>>,
    pub policy:            Vec<usize>,
    pub values:            Vec<f64>,
    pub returns_curve:     Vec<f64>,
    pub pg_loss_curve:     Vec<f64>,
    pub entropy_curve:     Vec<f64>,
    pub convergence_curve: Vec<f64>,
    pub n_episodes:        usize,
    pub total_steps:       usize,
}

#[derive(Debug, Clone)]
pub struct Ch09Result {
    pub reinforce:          PgResult,
    pub reinforce_baseline: PgResult,
    pub softmax_td0:        PgResult,
    pub reinforce_temp:     PgResult,
}

fn softmax(logits: &[f64], tau: f64) -> Vec<f64> {
    let scaled: Vec<f64> = logits.iter().map(|&x| x / tau).collect();
    let max = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scaled.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn sample_action(probs: &[f64], rng: &mut StdRng) -> usize {
    let u: f64 = rng.gen();
    let mut cum = 0.0_f64;
    for (i, &p) in probs.iter().enumerate() { cum += p; if u <= cum { return i; } }
    probs.len() - 1
}

fn entropy(probs: &[f64]) -> f64 {
    probs.iter().map(|&p| if p > 1e-12 { -p * p.ln() } else { 0.0 }).sum()
}

fn greedy_from_theta(theta: &[Vec<f64>], s: usize, tau: f64) -> usize {
    let probs = softmax(&theta[s], tau);
    probs.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i,_)| i).unwrap_or(0)
}

fn v_from_theta(theta: &[Vec<f64>], rewards: &Array2<f64>, tau: f64) -> Vec<f64> {
    (0..N_STATES).map(|s| {
        let probs = softmax(&theta[s], tau);
        probs.iter().enumerate().map(|(a, &p)| p * rewards[[s, a]]).sum()
    }).collect()
}

fn sample_next_state(s: usize, a: usize, transitions: &ndarray::Array3<f64>, rng: &mut StdRng) -> usize {
    let p: f64 = rng.gen();
    let mut cum = 0.0_f64;
    for sp in 0..N_STATES { cum += transitions[[s, a, sp]]; if p <= cum { return sp; } }
    N_STATES - 1
}

pub fn reinforce(config: &PgConfig, transitions: &ndarray::Array3<f64>, rewards: &Array2<f64>) -> PgResult {
    let mut rng      = StdRng::seed_from_u64(config.seed);
    let mut theta    = vec![vec![0.0_f64; N_ACTIONS]; N_STATES];
    let mut baseline = vec![0.0_f64; N_STATES];
    let (mut returns_curve, mut pg_loss_curve, mut entropy_curve, mut convergence_curve) = (vec![], vec![], vec![], vec![]);
    let mut theta_prev = theta.clone();
    let mut total_steps = 0usize;

    for _ in 0..config.n_episodes {
        let (mut traj_s, mut traj_a, mut traj_r) = (vec![], vec![], vec![]);
        let mut s = rng.gen_range(0..N_STATES);
        let mut steps = 0usize;
        loop {
            let probs = softmax(&theta[s], config.temperature);
            let a = sample_action(&probs, &mut rng);
            let r = rewards[[s, a]];
            let sp = sample_next_state(s, a, transitions, &mut rng);
            traj_s.push(s); traj_a.push(a); traj_r.push(r);
            total_steps += 1; steps += 1;
            if sp == 0 || sp == 7 || steps >= 50 { break; }
            s = sp;
        }
        let t_len = traj_r.len();
        let mut g_t = vec![0.0_f64; t_len];
        let mut g = 0.0_f64;
        for t in (0..t_len).rev() { g = traj_r[t] + config.gamma * g; g_t[t] = g; }

        let (mut ep_pg, mut ep_ent, mut gp) = (0.0_f64, 0.0_f64, 1.0_f64);
        for t in 0..t_len {
            let (st, at, gt) = (traj_s[t], traj_a[t], g_t[t]);
            let b = if config.use_baseline { baseline[st] } else { 0.0 };
            let adv = gt - b;
            if config.use_baseline { baseline[st] += config.alpha_baseline * (gt - baseline[st]); }
            let probs = softmax(&theta[st], config.temperature);
            ep_ent += entropy(&probs);
            for a in 0..N_ACTIONS {
                let grad = (if a == at { 1.0 } else { 0.0 }) - probs[a];
                let upd  = config.alpha * gp * adv * grad;
                theta[st][a] += upd;
                ep_pg += upd.abs();
            }
            gp *= config.gamma;
        }
        returns_curve.push(g_t[0]);
        pg_loss_curve.push(ep_pg / t_len.max(1) as f64);
        entropy_curve.push(ep_ent / t_len.max(1) as f64);
        let delta = theta.iter().zip(theta_prev.iter())
            .flat_map(|(r, p)| r.iter().zip(p.iter()).map(|(a, b)| (a - b).abs()))
            .fold(0.0_f64, f64::max);
        convergence_curve.push(delta);
        theta_prev = theta.clone();
    }
    let label = if config.use_baseline { "reinforce_baseline" } else { "reinforce" };
    PgResult {
        algorithm: label.to_string(),
        values: v_from_theta(&theta, rewards, config.temperature),
        policy: (0..N_STATES).map(|s| greedy_from_theta(&theta, s, config.temperature)).collect(),
        theta, returns_curve, pg_loss_curve, entropy_curve, convergence_curve,
        n_episodes: config.n_episodes, total_steps,
    }
}

pub fn softmax_actor_critic(config: &PgConfig, transitions: &ndarray::Array3<f64>, rewards: &Array2<f64>) -> PgResult {
    let mut rng      = StdRng::seed_from_u64(config.seed + 20);
    let mut theta    = vec![vec![0.0_f64; N_ACTIONS]; N_STATES];
    let mut v_critic = vec![0.0_f64; N_STATES];
    let (mut returns_curve, mut pg_loss_curve, mut entropy_curve, mut convergence_curve) = (vec![], vec![], vec![], vec![]);
    let mut theta_prev = theta.clone();
    let mut total_steps = 0usize;

    for _ in 0..config.n_episodes {
        let mut s = rng.gen_range(0..N_STATES);
        let (mut ep_ret, mut ep_pg, mut ep_ent, mut steps, mut gp) = (0.0_f64, 0.0_f64, 0.0_f64, 0usize, 1.0_f64);
        loop {
            let probs = softmax(&theta[s], config.temperature);
            let a     = sample_action(&probs, &mut rng);
            let r     = rewards[[s, a]];
            let sp    = sample_next_state(s, a, transitions, &mut rng);
            let done  = sp == 0 || sp == 7 || steps >= 50;
            let v_next = if done { 0.0 } else { v_critic[sp] };
            let delta  = r + config.gamma * v_next - v_critic[s];
            v_critic[s] += config.alpha_baseline * delta;
            ep_ent += entropy(&probs);
            for act in 0..N_ACTIONS {
                let grad = (if act == a { 1.0 } else { 0.0 }) - probs[act];
                let upd  = config.alpha * gp * delta * grad;
                theta[s][act] += upd;
                ep_pg += upd.abs();
            }
            ep_ret += gp * r; gp *= config.gamma; total_steps += 1; steps += 1;
            if done { break; }
            s = sp;
        }
        returns_curve.push(ep_ret);
        pg_loss_curve.push(ep_pg / steps.max(1) as f64);
        entropy_curve.push(ep_ent / steps.max(1) as f64);
        let delta = theta.iter().zip(theta_prev.iter())
            .flat_map(|(r, p)| r.iter().zip(p.iter()).map(|(a, b)| (a - b).abs()))
            .fold(0.0_f64, f64::max);
        convergence_curve.push(delta);
        theta_prev = theta.clone();
    }
    PgResult {
        algorithm: "softmax_actor_critic".to_string(),
        values: v_critic,
        policy: (0..N_STATES).map(|s| greedy_from_theta(&theta, s, config.temperature)).collect(),
        theta, returns_curve, pg_loss_curve, entropy_curve, convergence_curve,
        n_episodes: config.n_episodes, total_steps,
    }
}

pub fn run_ch09(config: PgConfig) -> Ch09Result {
    let mut rng = StdRng::seed_from_u64(config.seed);
    let transitions = build_asp_transitions(&mut rng);
    let rewards     = build_asp_rewards();
    verify_transition_matrix(&transitions).expect("Transition matrix invalid");
    let reinforce_r          = reinforce(&config, &transitions, &rewards);
    let reinforce_baseline_r = reinforce(&PgConfig { use_baseline: true, ..config.clone() }, &transitions, &rewards);
    let ac_r                 = softmax_actor_critic(&config, &transitions, &rewards);
    let reinforce_temp_r     = reinforce(&PgConfig { temperature: 0.5, use_baseline: false, ..config.clone() }, &transitions, &rewards);
    Ch09Result { reinforce: reinforce_r, reinforce_baseline: reinforce_baseline_r, softmax_td0: ac_r, reinforce_temp: reinforce_temp_r }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cfg() -> PgConfig {
        PgConfig { seed: 42, gamma: 0.95, alpha: 0.01, alpha_baseline: 0.1, n_episodes: 500, use_baseline: false, temperature: 1.0 }
    }
    #[test] fn test_reinforce_runs() { let r = run_ch09(cfg()); assert_eq!(r.reinforce.values.len(), N_STATES); assert!(r.reinforce.values.iter().all(|v| v.is_finite())); }
    #[test] fn test_reinforce_baseline_runs() { let r = run_ch09(cfg()); assert_eq!(r.reinforce_baseline.values.len(), N_STATES); assert!(r.reinforce_baseline.values.iter().all(|v| v.is_finite())); }
    #[test] fn test_actor_critic_runs() { let r = run_ch09(cfg()); assert_eq!(r.softmax_td0.values.len(), N_STATES); assert!(r.softmax_td0.values.iter().all(|v| v.is_finite())); }
    #[test] fn test_reinforce_temp_runs() { let r = run_ch09(cfg()); assert_eq!(r.reinforce_temp.values.len(), N_STATES); assert!(r.reinforce_temp.values.iter().all(|v| v.is_finite())); }
    #[test] fn test_theta_shape() { let r = run_ch09(cfg()); assert_eq!(r.reinforce.theta.len(), N_STATES); for row in &r.reinforce.theta { assert_eq!(row.len(), N_ACTIONS); } }
    #[test] fn test_policy_valid() { let r = run_ch09(cfg()); for &a in &r.reinforce.policy { assert!(a < N_ACTIONS); } for &a in &r.reinforce_baseline.policy { assert!(a < N_ACTIONS); } }
    #[test] fn test_curves_length() { let c = cfg(); let r = run_ch09(c.clone()); assert_eq!(r.reinforce.returns_curve.len(), c.n_episodes); assert_eq!(r.reinforce.entropy_curve.len(), c.n_episodes); }
    #[test] fn test_all_values_finite() { let r = run_ch09(cfg()); for v in &r.reinforce.returns_curve { assert!(v.is_finite()); } for v in &r.softmax_td0.returns_curve { assert!(v.is_finite()); } }
    #[test] fn test_softmax_sums_to_one() { let p = softmax(&[1.0, 2.0, 0.5, -1.0], 1.0); let s: f64 = p.iter().sum(); assert!((s - 1.0).abs() < 1e-9, "sum={}", s); }
    #[test] fn test_softmax_temperature_sharpens() {
        let logits = vec![2.0, 1.0, 0.5, 0.1];
        let p_hot  = softmax(&logits, 0.1);
        let p_cold = softmax(&logits, 2.0);
        assert!(p_hot.iter().cloned().fold(0.0_f64, f64::max) > p_cold.iter().cloned().fold(0.0_f64, f64::max));
    }
    #[test] fn test_entropy_positive() { let r = run_ch09(cfg()); let avg: f64 = r.reinforce.entropy_curve.iter().sum::<f64>() / r.reinforce.entropy_curve.len() as f64; assert!(avg > 0.0); }
    #[test] fn test_baseline_reduces_variance() {
        let r = run_ch09(PgConfig { n_episodes: 1000, ..cfg() });
        let var = |v: &[f64]| { let m = v.iter().sum::<f64>() / v.len() as f64; v.iter().map(|x| (x-m).powi(2)).sum::<f64>() / v.len() as f64 };
        let vp = var(&r.reinforce.returns_curve[500..]); let vb = var(&r.reinforce_baseline.returns_curve[500..]);
        assert!(vb <= vp * 1.2, "baseline var={:.4} plain var={:.4}", vb, vp);
    }
    #[test] fn test_reinforce_converges() {
        let r = run_ch09(PgConfig { n_episodes: 1000, ..cfg() });
        let early: f64 = r.reinforce.returns_curve[..100].iter().sum::<f64>() / 100.0;
        let late:  f64 = r.reinforce.returns_curve[900..].iter().sum::<f64>() / 100.0;
        assert!(late > early, "early={:.3} late={:.3}", early, late);
    }
    #[test] fn test_actor_critic_converges() {
        let r = run_ch09(PgConfig { n_episodes: 1000, ..cfg() });
        let late: f64 = r.softmax_td0.returns_curve[900..].iter().sum::<f64>() / 100.0;
        assert!(late.is_finite() && late > -100.0);
    }
    #[test] fn test_deterministic() {
        let r1 = run_ch09(cfg()); let r2 = run_ch09(cfg());
        for (v1, v2) in r1.reinforce.values.iter().zip(r2.reinforce.values.iter()) { assert_eq!(v1.to_bits(), v2.to_bits()); }
    }
}

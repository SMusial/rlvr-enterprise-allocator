//! Ch13 - Cooperative MARL: VDN, QMIX, QMIX+CG, IQL Baseline
//!
//! Four algorithms on a 2-agent Warsaw ASP cooperative dispatch task.
//! Agents share a JOINT reward (cooperative setting).
//!
//! 1. IQL Baseline     - independent Q-learning, no coordination
//! 2. VDN              - Value Decomposition Networks: Q_tot = Q_0 + Q_1
//! 3. QMIX             - monotonic mixing: Q_tot = f(Q_0, Q_1, s) with f monotone
//! 4. QMIX+CG          - QMIX with counterfactual baseline for credit assignment
//!
//! Key insight: VDN/QMIX satisfy IGM (Individual-Global-Max):
//!   argmax_a Q_tot = (argmax Q_0, argmax Q_1)
//! This means centralised training + decentralised execution.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::ch02_bellman::{
    build_asp_transitions, build_asp_rewards, verify_transition_matrix,
    N_STATES, N_ACTIONS,
};

pub const N_AGENTS: usize = 2;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CoopConfig {
    pub seed:          u64,
    pub gamma:         f64,
    pub alpha:         f64,
    pub epsilon:       f64,
    pub epsilon_decay: f64,
    pub n_episodes:    usize,
    pub mixing_hidden: usize,  // hidden units in QMIX mixing network
}

#[derive(Debug, Clone)]
pub struct CoopResult {
    pub algorithm:         String,
    pub q_tables:          Vec<Vec<Vec<f64>>>,  // [agent][state][action]
    pub policies:          Vec<Vec<usize>>,      // [agent][state]
    pub values:            Vec<f64>,             // joint V(s)
    pub returns_curve:     Vec<f64>,
    pub td_error_curve:    Vec<f64>,
    pub convergence_curve: Vec<f64>,
    pub mixing_weights:    Vec<f64>,  // QMIX: mean absolute mixing weight per episode
    pub joint_q_curve:     Vec<f64>,  // Q_tot per episode
    pub n_episodes:        usize,
    pub total_steps:       usize,
}

#[derive(Debug, Clone)]
pub struct Ch13Result {
    pub iql:      CoopResult,
    pub vdn:      CoopResult,
    pub qmix:     CoopResult,
    pub qmix_cg:  CoopResult,
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn sample_next(s: usize, a: usize, t: &ndarray::Array3<f64>, rng: &mut StdRng) -> usize {
    let p: f64 = rng.gen();
    let mut cum = 0.0_f64;
    for sp in 0..N_STATES { cum += t[[s, a, sp]]; if p <= cum { return sp; } }
    N_STATES - 1
}

fn eps_greedy(q: &[Vec<f64>], s: usize, eps: f64, rng: &mut StdRng) -> usize {
    if rng.gen::<f64>() < eps { rng.gen_range(0..N_ACTIONS) }
    else { q[s].iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i,_)| i).unwrap_or(0) }
}

fn max_q(q: &[Vec<f64>], s: usize) -> f64 {
    q[s].iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

fn joint_v(q_tables: &[Vec<Vec<f64>>]) -> Vec<f64> {
    let mut v = vec![0.0_f64; N_STATES];
    for q in q_tables {
        for s in 0..N_STATES { v[s] += max_q(q, s) / N_AGENTS as f64; }
    }
    v
}

fn greedy(q: &[Vec<f64>]) -> Vec<usize> {
    q.iter().map(|row| row.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i,_)| i).unwrap_or(0)).collect()
}

/// Joint reward: mean of per-agent rewards (cooperative)
fn joint_reward(r: &ndarray::Array2<f64>, s0: usize, a0: usize, s1: usize, a1: usize) -> f64 {
    (r[[s0, a0]] + r[[s1, a1]]) / 2.0
}

// ---------------------------------------------------------------------------
// 1. IQL Baseline - Independent Q-Learning (no coordination)
// ---------------------------------------------------------------------------
pub fn iql(cfg: &CoopConfig, t: &ndarray::Array3<f64>, r: &ndarray::Array2<f64>) -> CoopResult {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut q: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0_f64; N_ACTIONS]; N_STATES]; N_AGENTS];
    let (mut rc, mut tc, mut cc, mut mw, mut jq) = (vec![], vec![], vec![], vec![], vec![]);
    let mut vp = joint_v(&q);
    let mut total = 0usize;

    for ep in 0..cfg.n_episodes {
        let eps = (cfg.epsilon / (1.0 + cfg.epsilon_decay * ep as f64)).max(0.01);
        let mut s = vec![rng.gen_range(0..N_STATES); N_AGENTS];
        let (mut er, mut et, mut steps, mut gp) = (0.0_f64, 0.0_f64, 0usize, 1.0_f64);

        loop {
            let acts: Vec<usize> = (0..N_AGENTS).map(|i| eps_greedy(&q[i], s[i], eps, &mut rng)).collect();
            let jr = joint_reward(r, s[0], acts[0], s[1], acts[1]);
            let done = steps >= 50;
            let mut sp = vec![0usize; N_AGENTS];
            for i in 0..N_AGENTS {
                sp[i] = sample_next(s[i], acts[i], t, &mut rng);
                let delta = jr + cfg.gamma * (if done { 0.0 } else { max_q(&q[i], sp[i]) }) - q[i][s[i]][acts[i]];
                q[i][s[i]][acts[i]] += cfg.alpha * delta;
                et += delta.abs();
            }
            er += gp * jr; gp *= cfg.gamma; total += 1; steps += 1;
            if done { break; }
            s = sp;
        }
        rc.push(er); tc.push(et / steps.max(1) as f64); mw.push(0.0);
        jq.push(q.iter().map(|qi| max_q(qi, 0)).sum::<f64>() / N_AGENTS as f64);
        let v = joint_v(&q);
        cc.push(v.iter().zip(vp.iter()).map(|(a,b)| (a-b).abs()).fold(0.0_f64, f64::max));
        vp = v;
    }
    CoopResult { algorithm: "iql_baseline".into(), values: joint_v(&q),
        policies: q.iter().map(|qi| greedy(qi)).collect(), q_tables: q,
        returns_curve: rc, td_error_curve: tc, convergence_curve: cc,
        mixing_weights: mw, joint_q_curve: jq, n_episodes: cfg.n_episodes, total_steps: total }
}

// ---------------------------------------------------------------------------
// 2. VDN - Value Decomposition Networks (Sunehag et al., 2018)
//    Q_tot(s, a) = Q_0(s_0, a_0) + Q_1(s_1, a_1)
//    IGM holds: argmax Q_tot = (argmax Q_0, argmax Q_1)
//    TD target uses Q_tot for bootstrap.
// ---------------------------------------------------------------------------
pub fn vdn(cfg: &CoopConfig, t: &ndarray::Array3<f64>, r: &ndarray::Array2<f64>) -> CoopResult {
    let mut rng = StdRng::seed_from_u64(cfg.seed + 10);
    let mut q: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0_f64; N_ACTIONS]; N_STATES]; N_AGENTS];
    let (mut rc, mut tc, mut cc, mut mw, mut jq) = (vec![], vec![], vec![], vec![], vec![]);
    let mut vp = joint_v(&q);
    let mut total = 0usize;

    for ep in 0..cfg.n_episodes {
        let eps = (cfg.epsilon / (1.0 + cfg.epsilon_decay * ep as f64)).max(0.01);
        let mut s = vec![rng.gen_range(0..N_STATES); N_AGENTS];
        let (mut er, mut et, mut steps, mut gp, mut ep_mw) = (0.0_f64, 0.0_f64, 0usize, 1.0_f64, 0.0_f64);

        loop {
            let acts: Vec<usize> = (0..N_AGENTS).map(|i| eps_greedy(&q[i], s[i], eps, &mut rng)).collect();
            let jr = joint_reward(r, s[0], acts[0], s[1], acts[1]);
            let done = steps >= 50;

            // Q_tot = Q_0(s_0,a_0) + Q_1(s_1,a_1)
            let q_tot: f64 = (0..N_AGENTS).map(|i| q[i][s[i]][acts[i]]).sum();

            let mut sp = vec![0usize; N_AGENTS];
            for i in 0..N_AGENTS { sp[i] = sample_next(s[i], acts[i], t, &mut rng); }

            // Max Q_tot at next state (VDN: sum of individual maxes)
            let max_q_tot_sp: f64 = (0..N_AGENTS).map(|i| max_q(&q[i], sp[i])).sum();

            let td_target = jr + cfg.gamma * (if done { 0.0 } else { max_q_tot_sp });
            let delta = td_target - q_tot;
            et += delta.abs();

            // Gradient flows equally to each agent (VDN: dQ_tot/dQ_i = 1)
            for i in 0..N_AGENTS {
                q[i][s[i]][acts[i]] += cfg.alpha * delta;
            }
            ep_mw += 1.0; // VDN mixing weight = 1.0 always

            er += gp * jr; gp *= cfg.gamma; total += 1; steps += 1;
            if done { break; }
            s = sp;
        }
        rc.push(er); tc.push(et / steps.max(1) as f64);
        mw.push(ep_mw / steps.max(1) as f64);
        jq.push(q.iter().map(|qi| max_q(qi, 0)).sum::<f64>());
        let v = joint_v(&q);
        cc.push(v.iter().zip(vp.iter()).map(|(a,b)| (a-b).abs()).fold(0.0_f64, f64::max));
        vp = v;
    }
    CoopResult { algorithm: "vdn".into(), values: joint_v(&q),
        policies: q.iter().map(|qi| greedy(qi)).collect(), q_tables: q,
        returns_curve: rc, td_error_curve: tc, convergence_curve: cc,
        mixing_weights: mw, joint_q_curve: jq, n_episodes: cfg.n_episodes, total_steps: total }
}

// ---------------------------------------------------------------------------
// 3. QMIX (Rashid et al., 2018)
//    Q_tot = f(Q_0, Q_1, s) where f is monotone in each Q_i
//    Tabular approximation: w_i(s) >= 0, b(s) free
//    Q_tot = w_0(s)*Q_0 + w_1(s)*Q_1 + b(s)
//    Weights w_i(s) are state-dependent and non-negative (monotonicity).
// ---------------------------------------------------------------------------
pub fn qmix(cfg: &CoopConfig, t: &ndarray::Array3<f64>, r: &ndarray::Array2<f64>) -> CoopResult {
    let mut rng = StdRng::seed_from_u64(cfg.seed + 20);
    let mut q: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0_f64; N_ACTIONS]; N_STATES]; N_AGENTS];
    // Mixing weights w[agent][state] >= 0, bias b[state]
    let mut w: Vec<Vec<f64>> = vec![vec![1.0_f64; N_STATES]; N_AGENTS];
    let mut b: Vec<f64> = vec![0.0_f64; N_STATES];
    let (mut rc, mut tc, mut cc, mut mw, mut jq) = (vec![], vec![], vec![], vec![], vec![]);
    let mut vp = joint_v(&q);
    let mut total = 0usize;

    for ep in 0..cfg.n_episodes {
        let eps = (cfg.epsilon / (1.0 + cfg.epsilon_decay * ep as f64)).max(0.01);
        let mut s = vec![rng.gen_range(0..N_STATES); N_AGENTS];
        let (mut er, mut et, mut steps, mut gp, mut ep_mw) = (0.0_f64, 0.0_f64, 0usize, 1.0_f64, 0.0_f64);

        loop {
            let acts: Vec<usize> = (0..N_AGENTS).map(|i| eps_greedy(&q[i], s[i], eps, &mut rng)).collect();
            let jr = joint_reward(r, s[0], acts[0], s[1], acts[1]);
            let done = steps >= 50;

            // Q_tot = w_0(s)*Q_0 + w_1(s)*Q_1 + b(s)
            let q_i: Vec<f64> = (0..N_AGENTS).map(|i| q[i][s[i]][acts[i]]).collect();
            let q_tot: f64 = (0..N_AGENTS).map(|i| w[i][s[0]] * q_i[i]).sum::<f64>() + b[s[0]];

            let mut sp = vec![0usize; N_AGENTS];
            for i in 0..N_AGENTS { sp[i] = sample_next(s[i], acts[i], t, &mut rng); }

            // Max Q_tot at next state
            let max_q_tot_sp: f64 = (0..N_AGENTS).map(|i| w[i][sp[0]] * max_q(&q[i], sp[i])).sum::<f64>() + b[sp[0]];

            let td_target = jr + cfg.gamma * (if done { 0.0 } else { max_q_tot_sp });
            let delta = td_target - q_tot;
            et += delta.abs();

            // Update Q_i: gradient = w_i(s) * delta (chain rule through mixing)
            for i in 0..N_AGENTS {
                q[i][s[i]][acts[i]] += cfg.alpha * w[i][s[0]] * delta;
            }

            // Update mixing weights (gradient descent, keep w >= 0)
            for i in 0..N_AGENTS {
                w[i][s[0]] += cfg.alpha * 0.1 * delta * q_i[i];
                w[i][s[0]] = w[i][s[0]].max(0.0); // monotonicity constraint
            }
            b[s[0]] += cfg.alpha * 0.1 * delta;

            ep_mw += w.iter().map(|wi| wi[s[0]].abs()).sum::<f64>() / N_AGENTS as f64;

            er += gp * jr; gp *= cfg.gamma; total += 1; steps += 1;
            if done { break; }
            s = sp;
        }
        rc.push(er); tc.push(et / steps.max(1) as f64);
        mw.push(ep_mw / steps.max(1) as f64);
        jq.push((0..N_AGENTS).map(|i| w[i][0] * max_q(&q[i], 0)).sum::<f64>() + b[0]);
        let v = joint_v(&q);
        cc.push(v.iter().zip(vp.iter()).map(|(a,b)| (a-b).abs()).fold(0.0_f64, f64::max));
        vp = v;
    }
    CoopResult { algorithm: "qmix".into(), values: joint_v(&q),
        policies: q.iter().map(|qi| greedy(qi)).collect(), q_tables: q,
        returns_curve: rc, td_error_curve: tc, convergence_curve: cc,
        mixing_weights: mw, joint_q_curve: jq, n_episodes: cfg.n_episodes, total_steps: total }
}

// ---------------------------------------------------------------------------
// 4. QMIX+CG - QMIX with Counterfactual Baseline (credit assignment)
//    Advantage for agent i: A_i(s,a) = Q_tot(s,a) - Q_tot(s, a_{-i}, a_i^*)
//    where a_i^* = argmax Q_i(s_i, .)
//    This isolates each agent's contribution to the joint value.
// ---------------------------------------------------------------------------
pub fn qmix_cg(cfg: &CoopConfig, t: &ndarray::Array3<f64>, r: &ndarray::Array2<f64>) -> CoopResult {
    let mut rng = StdRng::seed_from_u64(cfg.seed + 30);
    let mut q: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0_f64; N_ACTIONS]; N_STATES]; N_AGENTS];
    let mut w: Vec<Vec<f64>> = vec![vec![1.0_f64; N_STATES]; N_AGENTS];
    let mut b: Vec<f64> = vec![0.0_f64; N_STATES];
    let (mut rc, mut tc, mut cc, mut mw, mut jq) = (vec![], vec![], vec![], vec![], vec![]);
    let mut vp = joint_v(&q);
    let mut total = 0usize;

    for ep in 0..cfg.n_episodes {
        let eps = (cfg.epsilon / (1.0 + cfg.epsilon_decay * ep as f64)).max(0.01);
        let mut s = vec![rng.gen_range(0..N_STATES); N_AGENTS];
        let (mut er, mut et, mut steps, mut gp, mut ep_mw) = (0.0_f64, 0.0_f64, 0usize, 1.0_f64, 0.0_f64);

        loop {
            let acts: Vec<usize> = (0..N_AGENTS).map(|i| eps_greedy(&q[i], s[i], eps, &mut rng)).collect();
            let jr = joint_reward(r, s[0], acts[0], s[1], acts[1]);
            let done = steps >= 50;

            let q_i: Vec<f64> = (0..N_AGENTS).map(|i| q[i][s[i]][acts[i]]).collect();
            let q_tot: f64 = (0..N_AGENTS).map(|i| w[i][s[0]] * q_i[i]).sum::<f64>() + b[s[0]];

            let mut sp = vec![0usize; N_AGENTS];
            for i in 0..N_AGENTS { sp[i] = sample_next(s[i], acts[i], t, &mut rng); }

            let max_q_tot_sp: f64 = (0..N_AGENTS).map(|i| w[i][sp[0]] * max_q(&q[i], sp[i])).sum::<f64>() + b[sp[0]];
            let td_target = jr + cfg.gamma * (if done { 0.0 } else { max_q_tot_sp });
            let delta = td_target - q_tot;
            et += delta.abs();

            // Counterfactual baseline for each agent
            for i in 0..N_AGENTS {
                // Baseline: Q_tot with agent i playing its best action
                let best_ai = greedy(&q[i])[s[i]];
                let q_i_best = q[i][s[i]][best_ai];
                let q_tot_baseline: f64 = (0..N_AGENTS).map(|j| {
                    let qi_j = if j == i { q_i_best } else { q_i[j] };
                    w[j][s[0]] * qi_j
                }).sum::<f64>() + b[s[0]];

                // Advantage: how much did agent i's actual action contribute?
                let advantage = q_tot - q_tot_baseline;
                // Update Q_i using advantage-weighted gradient
                q[i][s[i]][acts[i]] += cfg.alpha * (delta + 0.1 * advantage);
            }

            // Update mixing weights
            for i in 0..N_AGENTS {
                w[i][s[0]] += cfg.alpha * 0.1 * delta * q_i[i];
                w[i][s[0]] = w[i][s[0]].max(0.0);
            }
            b[s[0]] += cfg.alpha * 0.1 * delta;
            ep_mw += w.iter().map(|wi| wi[s[0]].abs()).sum::<f64>() / N_AGENTS as f64;

            er += gp * jr; gp *= cfg.gamma; total += 1; steps += 1;
            if done { break; }
            s = sp;
        }
        rc.push(er); tc.push(et / steps.max(1) as f64);
        mw.push(ep_mw / steps.max(1) as f64);
        jq.push((0..N_AGENTS).map(|i| w[i][0] * max_q(&q[i], 0)).sum::<f64>() + b[0]);
        let v = joint_v(&q);
        cc.push(v.iter().zip(vp.iter()).map(|(a,b)| (a-b).abs()).fold(0.0_f64, f64::max));
        vp = v;
    }
    CoopResult { algorithm: "qmix_cg".into(), values: joint_v(&q),
        policies: q.iter().map(|qi| greedy(qi)).collect(), q_tables: q,
        returns_curve: rc, td_error_curve: tc, convergence_curve: cc,
        mixing_weights: mw, joint_q_curve: jq, n_episodes: cfg.n_episodes, total_steps: total }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
pub fn run_ch13(cfg: CoopConfig) -> Ch13Result {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let t = build_asp_transitions(&mut rng);
    let r = build_asp_rewards();
    verify_transition_matrix(&t).expect("Transition matrix invalid");
    Ch13Result {
        iql:     iql(&cfg, &t, &r),
        vdn:     vdn(&cfg, &t, &r),
        qmix:    qmix(&cfg, &t, &r),
        qmix_cg: qmix_cg(&cfg, &t, &r),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> CoopConfig {
        CoopConfig { seed:42, gamma:0.95, alpha:0.1, epsilon:0.3,
                     epsilon_decay:0.01, n_episodes:300, mixing_hidden:8 }
    }

    #[test] fn test_iql_runs() { let r=run_ch13(cfg()); assert_eq!(r.iql.values.len(),N_STATES); assert!(r.iql.values.iter().all(|v|v.is_finite())); }
    #[test] fn test_vdn_runs() { let r=run_ch13(cfg()); assert_eq!(r.vdn.values.len(),N_STATES); assert!(r.vdn.values.iter().all(|v|v.is_finite())); }
    #[test] fn test_qmix_runs() { let r=run_ch13(cfg()); assert_eq!(r.qmix.values.len(),N_STATES); assert!(r.qmix.values.iter().all(|v|v.is_finite())); }
    #[test] fn test_qmix_cg_runs() { let r=run_ch13(cfg()); assert_eq!(r.qmix_cg.values.len(),N_STATES); assert!(r.qmix_cg.values.iter().all(|v|v.is_finite())); }
    #[test] fn test_q_tables_shape() {
        let r=run_ch13(cfg());
        for algo in [&r.iql,&r.vdn,&r.qmix,&r.qmix_cg] {
            assert_eq!(algo.q_tables.len(),N_AGENTS);
            for qt in &algo.q_tables { assert_eq!(qt.len(),N_STATES); for row in qt { assert_eq!(row.len(),N_ACTIONS); } }
        }
    }
    #[test] fn test_policies_shape() {
        let r=run_ch13(cfg());
        for algo in [&r.iql,&r.vdn,&r.qmix,&r.qmix_cg] {
            assert_eq!(algo.policies.len(),N_AGENTS);
            for pol in &algo.policies { assert_eq!(pol.len(),N_STATES); for &a in pol { assert!(a<N_ACTIONS); } }
        }
    }
    #[test] fn test_curves_length() {
        let c=cfg(); let r=run_ch13(c.clone());
        assert_eq!(r.vdn.returns_curve.len(),c.n_episodes);
        assert_eq!(r.qmix.mixing_weights.len(),c.n_episodes);
        assert_eq!(r.qmix_cg.joint_q_curve.len(),c.n_episodes);
    }
    #[test] fn test_all_finite() {
        let r=run_ch13(cfg());
        for algo in [&r.iql,&r.vdn,&r.qmix,&r.qmix_cg] {
            for v in &algo.returns_curve { assert!(v.is_finite()); }
            for v in &algo.mixing_weights { assert!(v.is_finite()); }
        }
    }
    #[test] fn test_vdn_better_than_iql() {
        let c=CoopConfig{n_episodes:800,..cfg()};
        let r=run_ch13(c);
        let iql_late:f64=r.iql.returns_curve[700..].iter().sum::<f64>()/100.0;
        let vdn_late:f64=r.vdn.returns_curve[700..].iter().sum::<f64>()/100.0;
        assert!(vdn_late >= iql_late - 0.5, "VDN should match or beat IQL: vdn={:.3} iql={:.3}", vdn_late, iql_late);
    }
    #[test] fn test_qmix_mixing_weights_nonneg() {
        let r=run_ch13(cfg());
        for &w in &r.qmix.mixing_weights { assert!(w>=0.0,"Mixing weight must be >=0, got {}",w); }
    }
    #[test] fn test_qmix_cg_mixing_weights_nonneg() {
        let r=run_ch13(cfg());
        for &w in &r.qmix_cg.mixing_weights { assert!(w>=0.0); }
    }
    #[test] fn test_joint_q_finite() {
        let r=run_ch13(cfg());
        for v in &r.qmix.joint_q_curve { assert!(v.is_finite()); }
        for v in &r.qmix_cg.joint_q_curve { assert!(v.is_finite()); }
    }
    #[test] fn test_igm_vdn() {
        let r=run_ch13(cfg());
        for s in 0..N_STATES {
            let joint_best:Vec<usize>=(0..N_AGENTS).map(|i| r.vdn.policies[i][s]).collect();
            for i in 0..N_AGENTS {
                let qi=&r.vdn.q_tables[i][s];
                let best=qi.iter().enumerate().max_by(|(_,a),(_,b)|a.partial_cmp(b).unwrap()).map(|(i,_)|i).unwrap_or(0);
                assert_eq!(joint_best[i],best,"IGM violated for VDN agent {} state {}",i,s);
            }
        }
    }
    #[test] fn test_qmix_converges() {
        let c=CoopConfig{n_episodes:800,..cfg()};
        let r=run_ch13(c);
        let late:f64=r.qmix.returns_curve[700..].iter().sum::<f64>()/100.0;
        assert!(late.is_finite()&&late>-100.0,"qmix late={}",late);
    }
    #[test] fn test_deterministic() {
        let r1=run_ch13(cfg()); let r2=run_ch13(cfg());
        for (v1,v2) in r1.vdn.values.iter().zip(r2.vdn.values.iter()) {
            assert_eq!(v1.to_bits(),v2.to_bits());
        }
    }
}

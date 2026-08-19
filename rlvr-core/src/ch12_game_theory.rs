//! Ch12 - Game Theory and Nash Equilibrium
//! Nash Q-Learning, Correlated Q-Learning, Minimax Q-Learning, Fictitious Play

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use crate::ch02_bellman::{
    build_asp_transitions, build_asp_rewards, verify_transition_matrix,
    N_STATES, N_ACTIONS,
};

pub const N_PLAYERS: usize = 2;

#[derive(Debug, Clone)]
pub struct GameConfig {
    pub seed:          u64,
    pub gamma:         f64,
    pub alpha:         f64,
    pub epsilon:       f64,
    pub epsilon_decay: f64,
    pub n_episodes:    usize,
    pub zero_sum:      bool,
}

#[derive(Debug, Clone)]
pub struct GameResult {
    pub algorithm:         String,
    pub q_tables:          Vec<Vec<Vec<Vec<f64>>>>,
    pub strategies:        Vec<Vec<Vec<f64>>>,
    pub values:            Vec<f64>,
    pub returns_curve:     Vec<f64>,
    pub exploitability:    Vec<f64>,
    pub convergence_curve: Vec<f64>,
    pub nash_gap_curve:    Vec<f64>,
    pub n_episodes:        usize,
    pub total_steps:       usize,
}

#[derive(Debug, Clone)]
pub struct Ch12Result {
    pub nash_q:       GameResult,
    pub correlated_q: GameResult,
    pub minimax_q:    GameResult,
    pub fictitious:   GameResult,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn snext(s: usize, a: usize, t: &ndarray::Array3<f64>, rng: &mut StdRng) -> usize {
    let p: f64 = rng.gen();
    let mut c = 0.0_f64;
    for sp in 0..N_STATES { c += t[[s, a, sp]]; if p <= c { return sp; } }
    N_STATES - 1
}

fn uniform() -> Vec<f64> { vec![1.0 / N_ACTIONS as f64; N_ACTIONS] }

fn softmax_t(v: &[f64], temp: f64) -> Vec<f64> {
    let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = v.iter().map(|&x| ((x - mx) / temp).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|&x| x / s).collect()
}

fn sample_s(s: &[f64], rng: &mut StdRng) -> usize {
    let u: f64 = rng.gen();
    let mut c = 0.0_f64;
    for (i, &p) in s.iter().enumerate() { c += p; if u <= c { return i; } }
    s.len() - 1
}

fn pay(r: &ndarray::Array2<f64>, s: usize, a0: usize, a1: usize, pl: usize, zs: bool) -> f64 {
    let r0 = r[[s, a0]];
    let r1 = r[[s, a1]];
    if zs { if pl == 0 { r0 - r1 } else { r1 - r0 } }
    else  { if pl == 0 { r0 } else { r1 } }
}

/// Joint value: mean V(s) across players under current mixed strategies.
/// q: [player][state][a0][a1], st: [player][state][action]
fn jv(q: &[Vec<Vec<Vec<f64>>>], st: &[Vec<Vec<f64>>]) -> Vec<f64> {
    let mut v = vec![0.0_f64; N_STATES];
    for p in 0..N_PLAYERS {
        for s in 0..N_STATES {
            let vs: f64 = st[p][s].iter().enumerate()
                .map(|(a, &pr)| pr * q[p][s][a].iter().sum::<f64>() / N_ACTIONS as f64)
                .sum();
            v[s] += vs / N_PLAYERS as f64;
        }
    }
    v
}

/// Nash gap for one player in one state.
/// q_s: [a0][a1] matrix for (player, state), st_s: mixed strategy [action]
/// FIX: q_s is &[Vec<f64>] i.e. a slice of a1-rows, one per a0.
fn ngap(q_s: &[Vec<f64>], st_s: &[f64]) -> f64 {
    // Marginal Q for each action a0: average over a1
    let marginal: Vec<f64> = q_s.iter()
        .map(|row| row.iter().sum::<f64>() / N_ACTIONS as f64)
        .collect();
    // Expected value under current strategy
    let v: f64 = st_s.iter().zip(marginal.iter()).map(|(&p, &m)| p * m).sum();
    // Best response value
    let br: f64 = marginal.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (br - v).max(0.0)
}

// ---------------------------------------------------------------------------
// 1. Nash Q-Learning
// ---------------------------------------------------------------------------
pub fn nash_q(cfg: &GameConfig, t: &ndarray::Array3<f64>, r: &ndarray::Array2<f64>) -> GameResult {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut q: Vec<Vec<Vec<Vec<f64>>>> =
        vec![vec![vec![vec![0.0_f64; N_ACTIONS]; N_ACTIONS]; N_STATES]; N_PLAYERS];
    let mut st: Vec<Vec<Vec<f64>>> = vec![vec![uniform(); N_STATES]; N_PLAYERS];
    let (mut rc, mut ec, mut cc, mut nc) = (vec![], vec![], vec![], vec![]);
    let mut vp = jv(&q, &st);
    let mut total = 0usize;

    for ep in 0..cfg.n_episodes {
        let eps = (cfg.epsilon / (1.0 + cfg.epsilon_decay * ep as f64)).max(0.01);
        let mut states = vec![rng.gen_range(0..N_STATES); N_PLAYERS];
        let (mut er, mut steps, mut gp) = (0.0_f64, 0usize, 1.0_f64);
        loop {
            let acts: Vec<usize> = (0..N_PLAYERS).map(|i| {
                if rng.gen::<f64>() < eps { rng.gen_range(0..N_ACTIONS) }
                else { sample_s(&st[i][states[i]], &mut rng) }
            }).collect();
            let (a0, a1) = (acts[0], acts[1]);
            let done = steps >= 50;
            let mut sr = 0.0_f64;
            for pl in 0..N_PLAYERS {
                let s = states[pl];
                let rew = pay(r, s, a0, a1, pl, cfg.zero_sum);
                let sp  = snext(s, acts[pl], t, &mut rng);
                let nv: f64 = st[pl][sp].iter().enumerate()
                    .map(|(a, &p)| p * q[pl][sp][a].iter().sum::<f64>() / N_ACTIONS as f64)
                    .sum();
                let tgt = rew + cfg.gamma * (if done { 0.0 } else { nv });
                q[pl][s][a0][a1] += cfg.alpha * (tgt - q[pl][s][a0][a1]);
                let mg: Vec<f64> = (0..N_ACTIONS)
                    .map(|a| q[pl][s][a].iter().sum::<f64>() / N_ACTIONS as f64)
                    .collect();
                st[pl][s] = softmax_t(&mg, 1.0_f64.max(eps * 5.0));
                sr += rew;
                states[pl] = sp;
            }
            er += gp * sr / N_PLAYERS as f64;
            gp *= cfg.gamma;
            total += 1;
            steps += 1;
            if done { break; }
        }
        rc.push(er);
        let ng: f64 = (0..N_STATES)
            .map(|s| ngap(&q[0][s], &st[0][s]))
            .sum::<f64>() / N_STATES as f64;
        nc.push(ng); ec.push(ng);
        let v = jv(&q, &st);
        cc.push(v.iter().zip(vp.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max));
        vp = v;
    }
    GameResult {
        algorithm: "nash_q".into(), values: jv(&q, &st),
        q_tables: q, strategies: st,
        returns_curve: rc, exploitability: ec,
        convergence_curve: cc, nash_gap_curve: nc,
        n_episodes: cfg.n_episodes, total_steps: total,
    }
}

// ---------------------------------------------------------------------------
// 2. Correlated Q-Learning (regret matching)
// ---------------------------------------------------------------------------
pub fn correlated_q(cfg: &GameConfig, t: &ndarray::Array3<f64>, r: &ndarray::Array2<f64>) -> GameResult {
    let mut rng = StdRng::seed_from_u64(cfg.seed + 10);
    let mut q: Vec<Vec<Vec<Vec<f64>>>> =
        vec![vec![vec![vec![0.0_f64; N_ACTIONS]; N_ACTIONS]; N_STATES]; N_PLAYERS];
    let mut corr: Vec<Vec<Vec<f64>>> =
        vec![vec![vec![1.0 / (N_ACTIONS * N_ACTIONS) as f64; N_ACTIONS]; N_ACTIONS]; N_STATES];
    let mut regret: Vec<Vec<Vec<Vec<f64>>>> =
        vec![vec![vec![vec![0.0_f64; N_ACTIONS]; N_ACTIONS]; N_PLAYERS]; N_STATES];
    let mut st: Vec<Vec<Vec<f64>>> = vec![vec![uniform(); N_STATES]; N_PLAYERS];
    let (mut rc, mut ec, mut cc, mut nc) = (vec![], vec![], vec![], vec![]);
    let mut vp = jv(&q, &st);
    let mut total = 0usize;

    for ep in 0..cfg.n_episodes {
        let eps = (cfg.epsilon / (1.0 + cfg.epsilon_decay * ep as f64)).max(0.01);
        let mut states = vec![rng.gen_range(0..N_STATES); N_PLAYERS];
        let (mut er, mut steps, mut gp) = (0.0_f64, 0usize, 1.0_f64);
        loop {
            let s    = states[0];
            let done = steps >= 50;
            let (a0, a1) = if rng.gen::<f64>() < eps {
                (rng.gen_range(0..N_ACTIONS), rng.gen_range(0..N_ACTIONS))
            } else {
                let u: f64 = rng.gen();
                let mut cum = 0.0_f64;
                let mut sam = (0usize, 0usize);
                'outer: for i in 0..N_ACTIONS {
                    for j in 0..N_ACTIONS {
                        cum += corr[s][i][j];
                        if u <= cum { sam = (i, j); break 'outer; }
                    }
                }
                sam
            };
            let mut sr = 0.0_f64;
            for pl in 0..N_PLAYERS {
                let rew = pay(r, s, a0, a1, pl, cfg.zero_sum);
                let sp  = snext(s, if pl == 0 { a0 } else { a1 }, t, &mut rng);
                let nv: f64 = st[pl][sp].iter().enumerate()
                    .map(|(a, &p)| p * q[pl][sp][a].iter().sum::<f64>() / N_ACTIONS as f64)
                    .sum();
                let tgt = rew + cfg.gamma * (if done { 0.0 } else { nv });
                q[pl][s][a0][a1] += cfg.alpha * (tgt - q[pl][s][a0][a1]);
                let rec = if pl == 0 { a0 } else { a1 };
                for dev in 0..N_ACTIONS {
                    if dev == rec { continue; }
                    let r_dev = pay(r, s,
                        if pl == 0 { dev } else { a0 },
                        if pl == 1 { dev } else { a1 },
                        pl, cfg.zero_sum);
                    regret[s][pl][rec][dev] += (r_dev - rew).max(0.0);
                }
                sr += rew;
                states[pl] = sp;
            }
            // Update correlated distribution via regret matching
            for i in 0..N_ACTIONS {
                for j in 0..N_ACTIONS {
                    let r0: f64 = regret[s][0][i].iter().sum();
                    let r1: f64 = regret[s][1][j].iter().sum();
                    corr[s][i][j] = (1.0 + r0 + r1).max(0.0);
                }
            }
            let tot: f64 = corr[s].iter().flat_map(|x| x.iter()).sum();
            if tot > 0.0 {
                for i in 0..N_ACTIONS {
                    for j in 0..N_ACTIONS { corr[s][i][j] /= tot; }
                }
            }
            // Extract marginal strategies from corr
            for pl in 0..N_PLAYERS {
                let mg: Vec<f64> = (0..N_ACTIONS).map(|a| {
                    if pl == 0 { corr[s][a].iter().sum() }
                    else       { (0..N_ACTIONS).map(|i| corr[s][i][a]).sum() }
                }).collect();
                let ms: f64 = mg.iter().sum();
                st[pl][s] = if ms > 0.0 { mg.iter().map(|&x| x / ms).collect() } else { uniform() };
            }
            er += gp * sr / N_PLAYERS as f64;
            gp *= cfg.gamma;
            total += 1;
            steps += 1;
            if done { break; }
        }
        rc.push(er);
        let ng: f64 = (0..N_STATES)
            .map(|s| ngap(&q[0][s], &st[0][s]))
            .sum::<f64>() / N_STATES as f64;
        nc.push(ng); ec.push(ng);
        let v = jv(&q, &st);
        cc.push(v.iter().zip(vp.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max));
        vp = v;
    }
    GameResult {
        algorithm: "correlated_q".into(), values: jv(&q, &st),
        q_tables: q, strategies: st,
        returns_curve: rc, exploitability: ec,
        convergence_curve: cc, nash_gap_curve: nc,
        n_episodes: cfg.n_episodes, total_steps: total,
    }
}

// ---------------------------------------------------------------------------
// 3. Minimax Q-Learning (zero-sum)
// ---------------------------------------------------------------------------
pub fn minimax_q(cfg: &GameConfig, t: &ndarray::Array3<f64>, r: &ndarray::Array2<f64>) -> GameResult {
    let mut rng = StdRng::seed_from_u64(cfg.seed + 20);
    // q0[state][a0][a1]
    let mut q0: Vec<Vec<Vec<f64>>> =
        vec![vec![vec![0.0_f64; N_ACTIONS]; N_ACTIONS]; N_STATES];
    let mut strat0: Vec<Vec<f64>> = vec![uniform(); N_STATES];
    let (mut rc, mut ec, mut cc, mut nc) = (vec![], vec![], vec![], vec![]);
    let mut vp = vec![0.0_f64; N_STATES];
    let mut total = 0usize;

    for ep in 0..cfg.n_episodes {
        let eps = (cfg.epsilon / (1.0 + cfg.epsilon_decay * ep as f64)).max(0.01);
        let mut s = rng.gen_range(0..N_STATES);
        let (mut er, mut steps, mut gp) = (0.0_f64, 0usize, 1.0_f64);
        loop {
            let done = steps >= 50;
            let a0 = if rng.gen::<f64>() < eps { rng.gen_range(0..N_ACTIONS) }
                     else { sample_s(&strat0[s], &mut rng) };
            // Adversary minimises
            let a1 = if rng.gen::<f64>() < eps { rng.gen_range(0..N_ACTIONS) }
                     else {
                         (0..N_ACTIONS)
                             .min_by(|&x, &y| q0[s][a0][x].partial_cmp(&q0[s][a0][y]).unwrap())
                             .unwrap_or(0)
                     };
            let rew = pay(r, s, a0, a1, 0, true);
            let sp  = snext(s, a0, t, &mut rng);
            // Minimax value of next state
            let mv: f64 = strat0[sp].iter().enumerate()
                .map(|(na0, &p)| {
                    let worst: f64 = (0..N_ACTIONS)
                        .map(|na1| q0[sp][na0][na1])
                        .fold(f64::INFINITY, f64::min);
                    p * worst
                })
                .sum();
            let tgt = rew + cfg.gamma * (if done { 0.0 } else { mv });
            q0[s][a0][a1] += cfg.alpha * (tgt - q0[s][a0][a1]);
            // Update minimax strategy
            let wc: Vec<f64> = (0..N_ACTIONS)
                .map(|na0| (0..N_ACTIONS).map(|na1| q0[s][na0][na1]).fold(f64::INFINITY, f64::min))
                .collect();
            strat0[s] = softmax_t(&wc, 1.0_f64.max(eps * 5.0));
            er += gp * rew;
            gp *= cfg.gamma;
            total += 1;
            steps += 1;
            if done { break; }
            s = sp;
        }
        rc.push(er);
        let ng: f64 = (0..N_STATES).map(|st| {
            let wc: Vec<f64> = (0..N_ACTIONS)
                .map(|a| (0..N_ACTIONS).map(|b| q0[st][a][b]).fold(f64::INFINITY, f64::min))
                .collect();
            let v: f64 = strat0[st].iter().zip(wc.iter()).map(|(p, w)| p * w).sum();
            let br: f64 = wc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (br - v).max(0.0)
        }).sum::<f64>() / N_STATES as f64;
        nc.push(ng); ec.push(ng);
        let v: Vec<f64> = (0..N_STATES).map(|st| {
            strat0[st].iter().enumerate()
                .map(|(a, &p)| {
                    let worst: f64 = (0..N_ACTIONS).map(|b| q0[st][a][b]).fold(f64::INFINITY, f64::min);
                    p * worst
                })
                .sum()
        }).collect();
        cc.push(v.iter().zip(vp.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max));
        vp = v.clone();
    }

    // Build q_tables: [player][state][a0][a1]
    // FIX: explicit type annotation on neg_q0 to avoid inference issues
    let neg_q0: Vec<Vec<Vec<f64>>> = q0.iter()
        .map(|state| state.iter()
            .map(|row| row.iter().map(|&x| -x).collect::<Vec<f64>>())
            .collect::<Vec<Vec<f64>>>())
        .collect::<Vec<Vec<Vec<f64>>>>();
    let q_tables: Vec<Vec<Vec<Vec<f64>>>> = vec![q0.clone(), neg_q0];
    let strats: Vec<Vec<Vec<f64>>> = vec![strat0.clone(), vec![uniform(); N_STATES]];
    let vals: Vec<f64> = (0..N_STATES).map(|st| {
        strat0[st].iter().enumerate()
            .map(|(a, &p)| {
                let worst: f64 = (0..N_ACTIONS).map(|b| q0[st][a][b]).fold(f64::INFINITY, f64::min);
                p * worst
            })
            .sum()
    }).collect();

    GameResult {
        algorithm: "minimax_q".into(), values: vals,
        q_tables, strategies: strats,
        returns_curve: rc, exploitability: ec,
        convergence_curve: cc, nash_gap_curve: nc,
        n_episodes: cfg.n_episodes, total_steps: total,
    }
}

// ---------------------------------------------------------------------------
// 4. Fictitious Play
// ---------------------------------------------------------------------------
pub fn fictitious_play(cfg: &GameConfig, t: &ndarray::Array3<f64>, r: &ndarray::Array2<f64>) -> GameResult {
    let mut rng = StdRng::seed_from_u64(cfg.seed + 30);
    let mut q: Vec<Vec<Vec<Vec<f64>>>> =
        vec![vec![vec![vec![0.0_f64; N_ACTIONS]; N_ACTIONS]; N_STATES]; N_PLAYERS];
    let mut counts: Vec<Vec<Vec<f64>>> = vec![vec![vec![1.0_f64; N_ACTIONS]; N_STATES]; N_PLAYERS];
    let mut st: Vec<Vec<Vec<f64>>> = vec![vec![uniform(); N_STATES]; N_PLAYERS];
    let (mut rc, mut ec, mut cc, mut nc) = (vec![], vec![], vec![], vec![]);
    let mut vp = jv(&q, &st);
    let mut total = 0usize;

    for ep in 0..cfg.n_episodes {
        let eps = (cfg.epsilon / (1.0 + cfg.epsilon_decay * ep as f64)).max(0.01);
        let mut states = vec![rng.gen_range(0..N_STATES); N_PLAYERS];
        let (mut er, mut steps, mut gp) = (0.0_f64, 0usize, 1.0_f64);
        loop {
            let done = steps >= 50;
            let acts: Vec<usize> = (0..N_PLAYERS).map(|pl| {
                if rng.gen::<f64>() < eps { return rng.gen_range(0..N_ACTIONS); }
                let s   = states[pl];
                let opp = 1 - pl;
                let ot: f64 = counts[opp][s].iter().sum();
                let op: Vec<f64> = counts[opp][s].iter().map(|&c| c / ot).collect();
                (0..N_ACTIONS).max_by(|&a1, &a2| {
                    let ev = |my: usize| -> f64 {
                        op.iter().enumerate().map(|(oa, &p)| {
                            let (ma, opa) = if pl == 0 { (my, oa) } else { (oa, my) };
                            p * pay(r, s, ma, opa, pl, cfg.zero_sum)
                        }).sum()
                    };
                    ev(a1).partial_cmp(&ev(a2)).unwrap()
                }).unwrap_or(0)
            }).collect();
            let (a0, a1) = (acts[0], acts[1]);
            let mut sr = 0.0_f64;
            for pl in 0..N_PLAYERS {
                let s   = states[pl];
                let rew = pay(r, s, a0, a1, pl, cfg.zero_sum);
                let sp  = snext(s, acts[pl], t, &mut rng);
                let nv: f64 = st[pl][sp].iter().enumerate()
                    .map(|(a, &p)| p * q[pl][sp][a].iter().sum::<f64>() / N_ACTIONS as f64)
                    .sum();
                let tgt = rew + cfg.gamma * (if done { 0.0 } else { nv });
                q[pl][s][a0][a1] += cfg.alpha * (tgt - q[pl][s][a0][a1]);
                counts[pl][s][acts[pl]] += 1.0;
                let tot: f64 = counts[pl][s].iter().sum();
                st[pl][s] = counts[pl][s].iter().map(|&c| c / tot).collect();
                sr += rew;
                states[pl] = sp;
            }
            er += gp * sr / N_PLAYERS as f64;
            gp *= cfg.gamma;
            total += 1;
            steps += 1;
            if done { break; }
        }
        rc.push(er);
        let ng: f64 = (0..N_STATES)
            .map(|s| ngap(&q[0][s], &st[0][s]))
            .sum::<f64>() / N_STATES as f64;
        nc.push(ng); ec.push(ng);
        let v = jv(&q, &st);
        cc.push(v.iter().zip(vp.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max));
        vp = v;
    }
    GameResult {
        algorithm: "fictitious_play".into(), values: jv(&q, &st),
        q_tables: q, strategies: st,
        returns_curve: rc, exploitability: ec,
        convergence_curve: cc, nash_gap_curve: nc,
        n_episodes: cfg.n_episodes, total_steps: total,
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
pub fn run_ch12(cfg: GameConfig) -> Ch12Result {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let t = build_asp_transitions(&mut rng);
    let r = build_asp_rewards();
    verify_transition_matrix(&t).expect("Transition matrix invalid");
    Ch12Result {
        nash_q:       nash_q(&cfg, &t, &r),
        correlated_q: correlated_q(&cfg, &t, &r),
        minimax_q:    minimax_q(&cfg, &t, &r),
        fictitious:   fictitious_play(&cfg, &t, &r),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    fn cfg() -> GameConfig {
        GameConfig { seed:42, gamma:0.95, alpha:0.1, epsilon:0.3,
            epsilon_decay:0.01, n_episodes:300, zero_sum:false }
    }
    fn cfgz() -> GameConfig { GameConfig { zero_sum:true, ..cfg() } }

    #[test] fn test_nash_q_runs() {
        let r=run_ch12(cfg()); assert_eq!(r.nash_q.values.len(),N_STATES);
        assert!(r.nash_q.values.iter().all(|v|v.is_finite()));
    }
    #[test] fn test_correlated_q_runs() {
        let r=run_ch12(cfg()); assert_eq!(r.correlated_q.values.len(),N_STATES);
        assert!(r.correlated_q.values.iter().all(|v|v.is_finite()));
    }
    #[test] fn test_minimax_q_runs() {
        let r=run_ch12(cfgz()); assert_eq!(r.minimax_q.values.len(),N_STATES);
        assert!(r.minimax_q.values.iter().all(|v|v.is_finite()));
    }
    #[test] fn test_fictitious_runs() {
        let r=run_ch12(cfg()); assert_eq!(r.fictitious.values.len(),N_STATES);
        assert!(r.fictitious.values.iter().all(|v|v.is_finite()));
    }
    #[test] fn test_q_tables_shape() {
        let r=run_ch12(cfg());
        assert_eq!(r.nash_q.q_tables.len(),N_PLAYERS);
        for qt in &r.nash_q.q_tables {
            assert_eq!(qt.len(),N_STATES);
            for s in qt { assert_eq!(s.len(),N_ACTIONS); for row in s { assert_eq!(row.len(),N_ACTIONS); } }
        }
    }
    #[test] fn test_strategies_shape() {
        let r=run_ch12(cfg());
        for ps in &r.nash_q.strategies {
            assert_eq!(ps.len(),N_STATES);
            for s in ps { assert_eq!(s.len(),N_ACTIONS); let sum:f64=s.iter().sum(); assert!((sum-1.0).abs()<1e-6,"sum={}",sum); }
        }
    }
    #[test] fn test_curves_length() {
        let c=cfg(); let r=run_ch12(c.clone());
        assert_eq!(r.nash_q.returns_curve.len(),c.n_episodes);
        assert_eq!(r.nash_q.nash_gap_curve.len(),c.n_episodes);
        assert_eq!(r.fictitious.returns_curve.len(),c.n_episodes);
    }
    #[test] fn test_all_finite() {
        let r=run_ch12(cfg());
        for v in &r.nash_q.returns_curve { assert!(v.is_finite()); }
        for v in &r.fictitious.returns_curve { assert!(v.is_finite()); }
    }
    #[test] fn test_strategies_sum_to_one() {
        let r=run_ch12(cfg());
        for algo in [&r.nash_q,&r.correlated_q,&r.fictitious] {
            for ps in &algo.strategies { for s in ps { let sum:f64=s.iter().sum(); assert!((sum-1.0).abs()<1e-6,"sum={}",sum); } }
        }
    }
    #[test] fn test_nash_gap_nonneg() {
        let r=run_ch12(cfg());
        for &g in &r.nash_q.nash_gap_curve { assert!(g>=0.0,"nash gap={}",g); }
    }
    #[test] fn test_exploitability_nonneg() {
        let r=run_ch12(cfg());
        for &e in &r.nash_q.exploitability { assert!(e>=0.0); }
        for &e in &r.fictitious.exploitability { assert!(e>=0.0); }
    }
    #[test] fn test_minimax_zero_sum() {
        let r=run_ch12(cfgz());
        assert!(r.minimax_q.values.iter().all(|v|v.is_finite()));
    }
    #[test] fn test_fictitious_converges() {
        let c=GameConfig{n_episodes:800,..cfg()}; let r=run_ch12(c);
        let late:f64=r.fictitious.returns_curve[700..].iter().sum::<f64>()/100.0;
        assert!(late.is_finite()&&late>-100.0,"late={}",late);
    }
    #[test] fn test_nash_q_converges() {
        let c=GameConfig{n_episodes:800,..cfg()}; let r=run_ch12(c);
        let late:f64=r.nash_q.returns_curve[700..].iter().sum::<f64>()/100.0;
        assert!(late.is_finite()&&late>-100.0);
    }
    #[test] fn test_deterministic() {
        let r1=run_ch12(cfg()); let r2=run_ch12(cfg());
        for (v1,v2) in r1.nash_q.values.iter().zip(r2.nash_q.values.iter()) {
            assert_eq!(v1.to_bits(),v2.to_bits(),"not deterministic");
        }
    }
}

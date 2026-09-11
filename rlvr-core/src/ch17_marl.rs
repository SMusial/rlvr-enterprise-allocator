//! Ch17 — Multi-Agent Reinforcement Learning (MARL)
//! Warsaw ASP: Multiple MARL variants
//! V1: IQL — individual rewards, no communication
//! V2: IQL — shared reward, no communication
//! V3: IQL — individual rewards, partial observability (distance-aware)
//! V4: CTDE (VDN) — shared reward, full observability, centralised training
//! V5: CTDE — mixed reward, partial observability, centralised training

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

const SKILLS: &[&str] = &["HVAC", "Electrical", "Network"];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ch17Step {
    pub episode:              usize,
    pub step:                 usize,
    pub tech_idx:             usize,
    pub order_idx:            usize,
    pub tech_x:               f64,
    pub tech_y:               f64,
    pub order_x:              f64,
    pub order_y:              f64,
    pub distance_km:          f64,
    pub reward:               f64,
    pub individual_reward:    f64,  // V5: individual component
    pub team_reward:          f64,  // V5: team component
    pub gt:                   f64,
    pub sla_met:              bool,
    pub skill_match:          bool,
    pub explored:             bool,
    pub epsilon:              f64,
    pub tech_skill:           String,
    pub order_skill:          String,
    pub q_before:             f64,
    pub q_after:              f64,
    pub td_error:             f64,
    pub nearest_colleague_km: f64,
    pub collision_avoided:    bool,
    pub joint_q:              f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ch17AgentStats {
    pub tech_idx:      usize,
    pub tech_skill:    String,
    pub total_gt:      f64,
    pub sla_rate:      f64,
    pub skill_rate:    f64,
    pub avg_distance:  f64,
    pub orders_served: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ch17EpisodeResult {
    pub episode:            usize,
    pub steps:              Vec<Ch17Step>,
    pub total_gt:           f64,
    pub agent_stats:        Vec<Ch17AgentStats>,
    pub team_sla_rate:      f64,
    pub collisions_avoided: usize,
    pub mean_joint_q:       f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ch17Result {
    pub episodes:       Vec<Ch17EpisodeResult>,
    pub curve:          Vec<f64>,
    pub agent_curves:   Vec<Vec<f64>>,
    pub final_q_tables: Vec<Vec<f64>>,
}

fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let r = 6371.0;
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    2.0 * r * a.sqrt().atan2((1.0 - a).sqrt())
}

fn init_environment(n_orders: usize, n_tech: usize) -> (
    Vec<f64>, Vec<f64>, Vec<String>, Vec<f64>,
    Vec<f64>, Vec<f64>, Vec<String>
) {
    let mut env_rng = StdRng::seed_from_u64(42);
    let orders_x: Vec<f64> = (0..n_orders).map(|_| 20.90 + env_rng.gen::<f64>() * 0.20).collect();
    let orders_y: Vec<f64> = (0..n_orders).map(|_| 52.18 + env_rng.gen::<f64>() * 0.14).collect();
    let order_skills: Vec<String> = (0..n_orders)
        .map(|_| SKILLS[env_rng.gen_range(0..SKILLS.len())].to_string()).collect();
    let order_sla: Vec<f64> = (0..n_orders).map(|_| 0.3 + env_rng.gen::<f64>() * 0.5).collect();
    let init_tech_x: Vec<f64> = (0..n_tech).map(|_| 20.90 + env_rng.gen::<f64>() * 0.20).collect();
    let init_tech_y: Vec<f64> = (0..n_tech).map(|_| 52.18 + env_rng.gen::<f64>() * 0.14).collect();
    let tech_skills: Vec<String> = (0..n_tech)
        .map(|_| SKILLS[env_rng.gen_range(0..SKILLS.len())].to_string()).collect();
    (orders_x, orders_y, order_skills, order_sla, init_tech_x, init_tech_y, tech_skills)
}

fn compute_reward(
    tech_idx: usize, order_idx: usize,
    tech_x: &[f64], tech_y: &[f64],
    orders_x: &[f64], orders_y: &[f64],
    tech_skills: &[String], order_skills: &[String],
    order_sla: &[f64],
) -> (f64, f64, bool, bool) {
    let tx = tech_x[tech_idx]; let ty = tech_y[tech_idx];
    let ox = orders_x[order_idx]; let oy = orders_y[order_idx];
    let distance_km = haversine_km(ty, tx, oy, ox);
    let skill_match = tech_skills[tech_idx] == order_skills[order_idx];
    let sla_met = skill_match && distance_km < order_sla[order_idx] * 20.0;
    let reward = if sla_met { 2.0 - distance_km * 0.05 }
                 else if skill_match { 0.5 - distance_km * 0.05 }
                 else { -1.0 - distance_km * 0.02 };
    (reward, distance_km, sla_met, skill_match)
}

fn iql_update(
    q_tables: &mut Vec<Vec<f64>>,
    tech_idx: usize, order_idx: usize, reward: f64,
    order_indices: &[usize], step: usize, n_orders: usize,
    alpha: f64, gamma: f64,
) -> (f64, f64, f64) {
    let q_before = q_tables[tech_idx][order_idx];
    let max_q_next = if step + 1 < n_orders {
        order_indices[step + 1..].iter()
            .map(|&o| q_tables[tech_idx][o])
            .fold(f64::NEG_INFINITY, f64::max)
    } else { 0.0 };
    let td_error = reward + gamma * max_q_next - q_before;
    let q_after = q_before + alpha * td_error;
    q_tables[tech_idx][order_idx] = q_after;
    (q_before, q_after, td_error)
}

fn build_agent_stats(
    steps: &[Ch17Step], n_tech: usize, tech_skills: &[String],
    gamma: f64, agent_curves: &mut Vec<Vec<f64>>,
) -> Vec<Ch17AgentStats> {
    let mut stats = Vec::new();
    for t in 0..n_tech {
        let ts: Vec<&Ch17Step> = steps.iter().filter(|s| s.tech_idx == t).collect();
        let nt = ts.len();
        let agent_gt = if nt == 0 { 0.0 } else {
            let mut g = 0.0f64;
            for s in ts.iter().rev() { g = s.reward + gamma * g; }
            g
        };
        let sla_rate   = if nt == 0 { 0.0 } else { ts.iter().filter(|s| s.sla_met).count() as f64 / nt as f64 };
        let skill_rate = if nt == 0 { 0.0 } else { ts.iter().filter(|s| s.skill_match).count() as f64 / nt as f64 };
        let avg_dist   = if nt == 0 { 0.0 } else { ts.iter().map(|s| s.distance_km).sum::<f64>() / nt as f64 };
        agent_curves[t].push(agent_gt);
        stats.push(Ch17AgentStats {
            tech_idx: t, tech_skill: tech_skills[t].clone(),
            total_gt: agent_gt, sla_rate, skill_rate,
            avg_distance: avg_dist, orders_served: nt,
        });
    }
    stats
}

// ---------------------------------------------------------------------------
// V1 — IQL + Individual Reward + No Communication
// ---------------------------------------------------------------------------
pub fn run_ch17(
    seed: u64, n_tech: usize, n_orders: usize, n_ep: usize,
    alpha: f64, gamma: f64, epsilon_start: f64, epsilon_end: f64,
) -> Ch17Result {
    let (orders_x, orders_y, order_skills, order_sla, init_tech_x, init_tech_y, tech_skills) =
        init_environment(n_orders, n_tech);
    let mut q_tables: Vec<Vec<f64>> = vec![vec![0.0f64; n_orders]; n_tech];
    let mut curve: Vec<f64> = Vec::new();
    let mut agent_curves: Vec<Vec<f64>> = vec![Vec::new(); n_tech];
    let mut episodes: Vec<Ch17EpisodeResult> = Vec::new();

    for ep in 0..n_ep {
        let mut rng = StdRng::seed_from_u64(seed + ep as u64);
        let epsilon = epsilon_start - (epsilon_start - epsilon_end) * (ep as f64 / n_ep.max(1) as f64);
        let mut order_indices: Vec<usize> = (0..n_orders).collect();
        for i in (1..n_orders).rev() { let j = rng.gen_range(0..=i); order_indices.swap(i, j); }
        let mut tech_x = init_tech_x.clone();
        let mut tech_y = init_tech_y.clone();
        let mut steps: Vec<Ch17Step> = Vec::new();

        for step in 0..n_orders {
            let order_idx = order_indices[step];
            let explored = rng.gen::<f64>() < epsilon;
            let matching: Vec<usize> = (0..n_tech).filter(|&t| tech_skills[t] == order_skills[order_idx]).collect();
            let tech_idx = if explored {
                if !matching.is_empty() && rng.gen::<f64>() < 0.80 { matching[rng.gen_range(0..matching.len())] }
                else { rng.gen_range(0..n_tech) }
            } else {
                (0..n_tech).max_by(|&a, &b| q_tables[a][order_idx].partial_cmp(&q_tables[b][order_idx]).unwrap()).unwrap_or(0)
            };
            let (reward, distance_km, sla_met, skill_match) = compute_reward(
                tech_idx, order_idx, &tech_x, &tech_y, &orders_x, &orders_y, &tech_skills, &order_skills, &order_sla);
            let tx = tech_x[tech_idx]; let ty = tech_y[tech_idx];
            let (q_before, q_after, td_error) = iql_update(&mut q_tables, tech_idx, order_idx, reward, &order_indices, step, n_orders, alpha, gamma);
            tech_x[tech_idx] = orders_x[order_idx]; tech_y[tech_idx] = orders_y[order_idx];
            steps.push(Ch17Step {
                episode: ep, step, tech_idx, order_idx, tech_x: tx, tech_y: ty,
                order_x: orders_x[order_idx], order_y: orders_y[order_idx],
                distance_km, reward, individual_reward: reward, team_reward: reward,
                gt: 0.0, sla_met, skill_match, explored, epsilon,
                tech_skill: tech_skills[tech_idx].clone(), order_skill: order_skills[order_idx].clone(),
                q_before, q_after, td_error, nearest_colleague_km: 0.0, collision_avoided: false, joint_q: 0.0,
            });
        }
        let n = steps.len(); let mut gt = 0.0f64;
        for i in (0..n).rev() { gt = steps[i].reward + gamma * gt; steps[i].gt = gt; }
        let total_gt = steps.first().map(|s| s.gt).unwrap_or(0.0);
        let agent_stats = build_agent_stats(&steps, n_tech, &tech_skills, gamma, &mut agent_curves);
        let team_sla = steps.iter().filter(|s| s.sla_met).count() as f64 / steps.len() as f64;
        curve.push(total_gt);
        episodes.push(Ch17EpisodeResult { episode: ep, steps, total_gt, agent_stats, team_sla_rate: team_sla, collisions_avoided: 0, mean_joint_q: 0.0 });
    }
    Ch17Result { episodes, curve, agent_curves, final_q_tables: q_tables }
}

// ---------------------------------------------------------------------------
// V2 — IQL + Shared Reward + No Communication
// ---------------------------------------------------------------------------
pub fn run_ch17_v2(
    seed: u64, n_tech: usize, n_orders: usize, n_ep: usize,
    alpha: f64, gamma: f64, epsilon_start: f64, epsilon_end: f64,
) -> Ch17Result {
    let (orders_x, orders_y, order_skills, order_sla, init_tech_x, init_tech_y, tech_skills) =
        init_environment(n_orders, n_tech);
    let mut q_tables: Vec<Vec<f64>> = vec![vec![0.0f64; n_orders]; n_tech];
    let mut curve: Vec<f64> = Vec::new();
    let mut agent_curves: Vec<Vec<f64>> = vec![Vec::new(); n_tech];
    let mut episodes: Vec<Ch17EpisodeResult> = Vec::new();

    for ep in 0..n_ep {
        let mut rng = StdRng::seed_from_u64(seed + ep as u64);
        let epsilon = epsilon_start - (epsilon_start - epsilon_end) * (ep as f64 / n_ep.max(1) as f64);
        let mut order_indices: Vec<usize> = (0..n_orders).collect();
        for i in (1..n_orders).rev() { let j = rng.gen_range(0..=i); order_indices.swap(i, j); }
        let mut tech_x = init_tech_x.clone();
        let mut tech_y = init_tech_y.clone();

        struct StepRaw { tech_idx: usize, order_idx: usize, tx: f64, ty: f64, individual_reward: f64, distance_km: f64, sla_met: bool, skill_match: bool, explored: bool, epsilon: f64, tech_skill: String, order_skill: String }
        let mut raw_steps: Vec<StepRaw> = Vec::new();

        for step in 0..n_orders {
            let order_idx = order_indices[step];
            let explored = rng.gen::<f64>() < epsilon;
            let matching: Vec<usize> = (0..n_tech).filter(|&t| tech_skills[t] == order_skills[order_idx]).collect();
            let tech_idx = if explored {
                if !matching.is_empty() && rng.gen::<f64>() < 0.80 { matching[rng.gen_range(0..matching.len())] }
                else { rng.gen_range(0..n_tech) }
            } else {
                (0..n_tech).max_by(|&a, &b| q_tables[a][order_idx].partial_cmp(&q_tables[b][order_idx]).unwrap()).unwrap_or(0)
            };
            let (individual_reward, distance_km, sla_met, skill_match) = compute_reward(
                tech_idx, order_idx, &tech_x, &tech_y, &orders_x, &orders_y, &tech_skills, &order_skills, &order_sla);
            let tx = tech_x[tech_idx]; let ty = tech_y[tech_idx];
            tech_x[tech_idx] = orders_x[order_idx]; tech_y[tech_idx] = orders_y[order_idx];
            raw_steps.push(StepRaw { tech_idx, order_idx, tx, ty, individual_reward, distance_km, sla_met, skill_match, explored, epsilon, tech_skill: tech_skills[tech_idx].clone(), order_skill: order_skills[order_idx].clone() });
        }

        let shared_reward: f64 = raw_steps.iter().map(|s| s.individual_reward).sum::<f64>() / raw_steps.len() as f64;
        let mut steps: Vec<Ch17Step> = Vec::new();
        for (step, raw) in raw_steps.iter().enumerate() {
            let (q_before, q_after, td_error) = iql_update(&mut q_tables, raw.tech_idx, raw.order_idx, shared_reward, &order_indices, step, n_orders, alpha, gamma);
            steps.push(Ch17Step {
                episode: ep, step, tech_idx: raw.tech_idx, order_idx: raw.order_idx,
                tech_x: raw.tx, tech_y: raw.ty, order_x: orders_x[raw.order_idx], order_y: orders_y[raw.order_idx],
                distance_km: raw.distance_km, reward: shared_reward,
                individual_reward: raw.individual_reward, team_reward: shared_reward,
                gt: 0.0, sla_met: raw.sla_met, skill_match: raw.skill_match,
                explored: raw.explored, epsilon: raw.epsilon,
                tech_skill: raw.tech_skill.clone(), order_skill: raw.order_skill.clone(),
                q_before, q_after, td_error, nearest_colleague_km: 0.0, collision_avoided: false, joint_q: 0.0,
            });
        }
        let n = steps.len(); let mut gt = 0.0f64;
        for i in (0..n).rev() { gt = steps[i].reward + gamma * gt; steps[i].gt = gt; }
        let total_gt = steps.first().map(|s| s.gt).unwrap_or(0.0);
        let agent_stats = build_agent_stats(&steps, n_tech, &tech_skills, gamma, &mut agent_curves);
        let team_sla = steps.iter().filter(|s| s.sla_met).count() as f64 / steps.len() as f64;
        curve.push(total_gt);
        episodes.push(Ch17EpisodeResult { episode: ep, steps, total_gt, agent_stats, team_sla_rate: team_sla, collisions_avoided: 0, mean_joint_q: 0.0 });
    }
    Ch17Result { episodes, curve, agent_curves, final_q_tables: q_tables }
}

// ---------------------------------------------------------------------------
// V3 — IQL + Individual Reward + Partial Observability (distance-aware)
// ---------------------------------------------------------------------------
pub fn run_ch17_v3(
    seed: u64, n_tech: usize, n_orders: usize, n_ep: usize,
    alpha: f64, gamma: f64, epsilon_start: f64, epsilon_end: f64,
    lambda: f64,
) -> Ch17Result {
    let (orders_x, orders_y, order_skills, order_sla, init_tech_x, init_tech_y, tech_skills) =
        init_environment(n_orders, n_tech);
    let mut q_tables: Vec<Vec<f64>> = vec![vec![0.0f64; n_orders]; n_tech];
    let mut curve: Vec<f64> = Vec::new();
    let mut agent_curves: Vec<Vec<f64>> = vec![Vec::new(); n_tech];
    let mut episodes: Vec<Ch17EpisodeResult> = Vec::new();

    for ep in 0..n_ep {
        let mut rng = StdRng::seed_from_u64(seed + ep as u64);
        let epsilon = epsilon_start - (epsilon_start - epsilon_end) * (ep as f64 / n_ep.max(1) as f64);
        let mut order_indices: Vec<usize> = (0..n_orders).collect();
        for i in (1..n_orders).rev() { let j = rng.gen_range(0..=i); order_indices.swap(i, j); }
        let mut tech_x = init_tech_x.clone();
        let mut tech_y = init_tech_y.clone();
        let mut steps: Vec<Ch17Step> = Vec::new();
        let mut collisions_avoided = 0usize;

        for step in 0..n_orders {
            let order_idx = order_indices[step];
            let explored = rng.gen::<f64>() < epsilon;
            let matching: Vec<usize> = (0..n_tech).filter(|&t| tech_skills[t] == order_skills[order_idx]).collect();
            let tech_idx = if explored {
                if !matching.is_empty() && rng.gen::<f64>() < 0.80 { matching[rng.gen_range(0..matching.len())] }
                else { rng.gen_range(0..n_tech) }
            } else {
                (0..n_tech).max_by(|&t_a, &t_b| {
                    let score = |t: usize| {
                        let q = q_tables[t][order_idx];
                        let min_coll = (0..n_tech).filter(|&x| x != t)
                            .map(|x| haversine_km(tech_y[x], tech_x[x], orders_y[order_idx], orders_x[order_idx]))
                            .fold(f64::INFINITY, f64::min);
                        q + lambda * min_coll
                    };
                    score(t_a).partial_cmp(&score(t_b)).unwrap()
                }).unwrap_or(0)
            };
            let (reward, distance_km, sla_met, skill_match) = compute_reward(
                tech_idx, order_idx, &tech_x, &tech_y, &orders_x, &orders_y, &tech_skills, &order_skills, &order_sla);
            let tx = tech_x[tech_idx]; let ty = tech_y[tech_idx];
            let nearest_colleague_km = (0..n_tech).filter(|&t| t != tech_idx)
                .map(|t| haversine_km(tech_y[t], tech_x[t], ty, tx))
                .fold(f64::INFINITY, f64::min);
            let closest_coll_to_order = (0..n_tech).filter(|&t| t != tech_idx)
                .map(|t| haversine_km(tech_y[t], tech_x[t], orders_y[order_idx], orders_x[order_idx]))
                .fold(f64::INFINITY, f64::min);
            let collision_avoided = !explored && closest_coll_to_order < distance_km;
            if collision_avoided { collisions_avoided += 1; }
            let (q_before, q_after, td_error) = iql_update(&mut q_tables, tech_idx, order_idx, reward, &order_indices, step, n_orders, alpha, gamma);
            tech_x[tech_idx] = orders_x[order_idx]; tech_y[tech_idx] = orders_y[order_idx];
            steps.push(Ch17Step {
                episode: ep, step, tech_idx, order_idx, tech_x: tx, tech_y: ty,
                order_x: orders_x[order_idx], order_y: orders_y[order_idx],
                distance_km, reward, individual_reward: reward, team_reward: reward,
                gt: 0.0, sla_met, skill_match, explored, epsilon,
                tech_skill: tech_skills[tech_idx].clone(), order_skill: order_skills[order_idx].clone(),
                q_before, q_after, td_error, nearest_colleague_km, collision_avoided, joint_q: 0.0,
            });
        }
        let n = steps.len(); let mut gt = 0.0f64;
        for i in (0..n).rev() { gt = steps[i].reward + gamma * gt; steps[i].gt = gt; }
        let total_gt = steps.first().map(|s| s.gt).unwrap_or(0.0);
        let agent_stats = build_agent_stats(&steps, n_tech, &tech_skills, gamma, &mut agent_curves);
        let team_sla = steps.iter().filter(|s| s.sla_met).count() as f64 / steps.len() as f64;
        curve.push(total_gt);
        episodes.push(Ch17EpisodeResult { episode: ep, steps, total_gt, agent_stats, team_sla_rate: team_sla, collisions_avoided, mean_joint_q: 0.0 });
    }
    Ch17Result { episodes, curve, agent_curves, final_q_tables: q_tables }
}

// ---------------------------------------------------------------------------
// V4 — CTDE (VDN) + Shared Reward + Full Observability
// ---------------------------------------------------------------------------
pub fn run_ch17_v4(
    seed: u64, n_tech: usize, n_orders: usize, n_ep: usize,
    alpha: f64, gamma: f64, epsilon_start: f64, epsilon_end: f64,
) -> Ch17Result {
    let (orders_x, orders_y, order_skills, order_sla, init_tech_x, init_tech_y, tech_skills) =
        init_environment(n_orders, n_tech);
    let mut q_tables: Vec<Vec<f64>> = vec![vec![0.0f64; n_orders]; n_tech];
    let mut curve: Vec<f64> = Vec::new();
    let mut agent_curves: Vec<Vec<f64>> = vec![Vec::new(); n_tech];
    let mut episodes: Vec<Ch17EpisodeResult> = Vec::new();

    for ep in 0..n_ep {
        let mut rng = StdRng::seed_from_u64(seed + ep as u64);
        let epsilon = epsilon_start - (epsilon_start - epsilon_end) * (ep as f64 / n_ep.max(1) as f64);
        let mut order_indices: Vec<usize> = (0..n_orders).collect();
        for i in (1..n_orders).rev() { let j = rng.gen_range(0..=i); order_indices.swap(i, j); }
        let mut tech_x = init_tech_x.clone();
        let mut tech_y = init_tech_y.clone();
        let mut steps: Vec<Ch17Step> = Vec::new();
        let mut joint_q_sum = 0.0f64;

        for step in 0..n_orders {
            let order_idx = order_indices[step];
            let explored = rng.gen::<f64>() < epsilon;
            let tech_idx = if explored {
                let matching: Vec<usize> = (0..n_tech).filter(|&t| tech_skills[t] == order_skills[order_idx]).collect();
                if !matching.is_empty() && rng.gen::<f64>() < 0.80 { matching[rng.gen_range(0..matching.len())] }
                else { rng.gen_range(0..n_tech) }
            } else {
                (0..n_tech).max_by(|&t_a, &t_b| {
                    let dist_a = haversine_km(tech_y[t_a], tech_x[t_a], orders_y[order_idx], orders_x[order_idx]);
                    let dist_b = haversine_km(tech_y[t_b], tech_x[t_b], orders_y[order_idx], orders_x[order_idx]);
                    let max_dist = 30.0_f64;
                    let score_a = q_tables[t_a][order_idx] + 0.5 * (max_dist - dist_a.min(max_dist)) / max_dist;
                    let score_b = q_tables[t_b][order_idx] + 0.5 * (max_dist - dist_b.min(max_dist)) / max_dist;
                    score_a.partial_cmp(&score_b).unwrap()
                }).unwrap_or(0)
            };
            let (reward, distance_km, sla_met, skill_match) = compute_reward(
                tech_idx, order_idx, &tech_x, &tech_y, &orders_x, &orders_y, &tech_skills, &order_skills, &order_sla);
            let tx = tech_x[tech_idx]; let ty = tech_y[tech_idx];
            let joint_q: f64 = (0..n_tech).map(|t| q_tables[t][order_idx]).sum();
            joint_q_sum += joint_q;
            let (q_before, q_after, td_error) = iql_update(&mut q_tables, tech_idx, order_idx, reward, &order_indices, step, n_orders, alpha, gamma);
            for t in 0..n_tech {
                if t != tech_idx {
                    let q_t = q_tables[t][order_idx];
                    let max_q_next_t = if step + 1 < n_orders {
                        order_indices[step + 1..].iter().map(|&o| q_tables[t][o]).fold(f64::NEG_INFINITY, f64::max)
                    } else { 0.0 };
                    let td_t = reward + gamma * max_q_next_t - q_t;
                    q_tables[t][order_idx] = q_t + (alpha * 0.3) * td_t;
                }
            }
            tech_x[tech_idx] = orders_x[order_idx]; tech_y[tech_idx] = orders_y[order_idx];
            let nearest_colleague_km = (0..n_tech).filter(|&t| t != tech_idx)
                .map(|t| haversine_km(tech_y[t], tech_x[t], ty, tx))
                .fold(f64::INFINITY, f64::min);
            steps.push(Ch17Step {
                episode: ep, step, tech_idx, order_idx, tech_x: tx, tech_y: ty,
                order_x: orders_x[order_idx], order_y: orders_y[order_idx],
                distance_km, reward, individual_reward: reward, team_reward: reward,
                gt: 0.0, sla_met, skill_match, explored, epsilon,
                tech_skill: tech_skills[tech_idx].clone(), order_skill: order_skills[order_idx].clone(),
                q_before, q_after, td_error, nearest_colleague_km, collision_avoided: false, joint_q,
            });
        }
        let n = steps.len(); let mut gt = 0.0f64;
        for i in (0..n).rev() { gt = steps[i].reward + gamma * gt; steps[i].gt = gt; }
        let total_gt = steps.first().map(|s| s.gt).unwrap_or(0.0);
        let mean_joint_q = if n > 0 { joint_q_sum / n as f64 } else { 0.0 };
        let agent_stats = build_agent_stats(&steps, n_tech, &tech_skills, gamma, &mut agent_curves);
        let team_sla = steps.iter().filter(|s| s.sla_met).count() as f64 / steps.len() as f64;
        curve.push(total_gt);
        episodes.push(Ch17EpisodeResult { episode: ep, steps, total_gt, agent_stats, team_sla_rate: team_sla, collisions_avoided: 0, mean_joint_q });
    }
    Ch17Result { episodes, curve, agent_curves, final_q_tables: q_tables }
}

// ---------------------------------------------------------------------------
// V5 — CTDE + Mixed Reward + Partial Observability
// ---------------------------------------------------------------------------
// Mixed reward: R_mixed = alpha_ind * R^i + (1 - alpha_ind) * R_team
// alpha_ind = 0.0 → pure shared (like V4)
// alpha_ind = 1.0 → pure individual (like V1)
// alpha_ind = 0.5 → balanced (default)
// Partial observability: distance-aware selection (like V3)
// CTDE: soft update for non-assigned agents (like V4)
pub fn run_ch17_v5(
    seed: u64, n_tech: usize, n_orders: usize, n_ep: usize,
    alpha: f64, gamma: f64, epsilon_start: f64, epsilon_end: f64,
    alpha_ind: f64,  // individual reward weight [0.0, 1.0]
    lambda: f64,     // collision avoidance weight
) -> Ch17Result {
    let (orders_x, orders_y, order_skills, order_sla, init_tech_x, init_tech_y, tech_skills) =
        init_environment(n_orders, n_tech);
    let mut q_tables: Vec<Vec<f64>> = vec![vec![0.0f64; n_orders]; n_tech];
    let mut curve: Vec<f64> = Vec::new();
    let mut agent_curves: Vec<Vec<f64>> = vec![Vec::new(); n_tech];
    let mut episodes: Vec<Ch17EpisodeResult> = Vec::new();

    for ep in 0..n_ep {
        let mut rng = StdRng::seed_from_u64(seed + ep as u64);
        let epsilon = epsilon_start - (epsilon_start - epsilon_end) * (ep as f64 / n_ep.max(1) as f64);
        let mut order_indices: Vec<usize> = (0..n_orders).collect();
        for i in (1..n_orders).rev() { let j = rng.gen_range(0..=i); order_indices.swap(i, j); }
        let mut tech_x = init_tech_x.clone();
        let mut tech_y = init_tech_y.clone();
        let mut collisions_avoided = 0usize;
        let mut joint_q_sum = 0.0f64;

        // Collect all individual rewards first for team mean
        struct StepRaw {
            tech_idx: usize, order_idx: usize,
            tx: f64, ty: f64,
            individual_reward: f64, distance_km: f64,
            sla_met: bool, skill_match: bool,
            explored: bool, epsilon: f64,
            tech_skill: String, order_skill: String,
            nearest_colleague_km: f64, collision_avoided: bool,
            joint_q: f64,
        }
        let mut raw_steps: Vec<StepRaw> = Vec::new();

        for step in 0..n_orders {
            let order_idx = order_indices[step];
            let explored = rng.gen::<f64>() < epsilon;

            // V5: CTDE + Partial Observability selection
            let tech_idx = if explored {
                let matching: Vec<usize> = (0..n_tech)
                    .filter(|&t| tech_skills[t] == order_skills[order_idx]).collect();
                if !matching.is_empty() && rng.gen::<f64>() < 0.80 {
                    matching[rng.gen_range(0..matching.len())]
                } else {
                    rng.gen_range(0..n_tech)
                }
            } else {
                // CTDE + distance-aware: Q[t][o] + lambda * min_colleague_dist + distance_bonus
                (0..n_tech).max_by(|&t_a, &t_b| {
                    let score = |t: usize| {
                        let q = q_tables[t][order_idx];
                        // Partial observability: colleague distance bonus
                        let min_coll = (0..n_tech).filter(|&x| x != t)
                            .map(|x| haversine_km(tech_y[x], tech_x[x], orders_y[order_idx], orders_x[order_idx]))
                            .fold(f64::INFINITY, f64::min);
                        // Full observability: own distance bonus (CTDE)
                        let own_dist = haversine_km(tech_y[t], tech_x[t], orders_y[order_idx], orders_x[order_idx]);
                        let max_dist = 30.0_f64;
                        let dist_bonus = 0.3 * (max_dist - own_dist.min(max_dist)) / max_dist;
                        q + lambda * min_coll + dist_bonus
                    };
                    score(t_a).partial_cmp(&score(t_b)).unwrap()
                }).unwrap_or(0)
            };

            let (individual_reward, distance_km, sla_met, skill_match) = compute_reward(
                tech_idx, order_idx, &tech_x, &tech_y,
                &orders_x, &orders_y, &tech_skills, &order_skills, &order_sla);

            let tx = tech_x[tech_idx]; let ty = tech_y[tech_idx];

            let nearest_colleague_km = (0..n_tech).filter(|&t| t != tech_idx)
                .map(|t| haversine_km(tech_y[t], tech_x[t], ty, tx))
                .fold(f64::INFINITY, f64::min);

            let closest_coll_to_order = (0..n_tech).filter(|&t| t != tech_idx)
                .map(|t| haversine_km(tech_y[t], tech_x[t], orders_y[order_idx], orders_x[order_idx]))
                .fold(f64::INFINITY, f64::min);
            let collision_avoided = !explored && closest_coll_to_order < distance_km;
            if collision_avoided { collisions_avoided += 1; }

            let joint_q: f64 = (0..n_tech).map(|t| q_tables[t][order_idx]).sum();
            joint_q_sum += joint_q;

            tech_x[tech_idx] = orders_x[order_idx];
            tech_y[tech_idx] = orders_y[order_idx];

            raw_steps.push(StepRaw {
                tech_idx, order_idx, tx, ty,
                individual_reward, distance_km,
                sla_met, skill_match, explored, epsilon,
                tech_skill: tech_skills[tech_idx].clone(),
                order_skill: order_skills[order_idx].clone(),
                nearest_colleague_km, collision_avoided, joint_q,
            });
        }

        // V5 KEY: compute team mean reward
        let team_mean: f64 = raw_steps.iter().map(|s| s.individual_reward).sum::<f64>()
            / raw_steps.len() as f64;

        let mut steps: Vec<Ch17Step> = Vec::new();
        for (step, raw) in raw_steps.iter().enumerate() {
            // V5 KEY: mixed reward = alpha_ind * R^i + (1 - alpha_ind) * R_team
            let mixed_reward = alpha_ind * raw.individual_reward + (1.0 - alpha_ind) * team_mean;

            // Update assigned agent with mixed reward
            let (q_before, q_after, td_error) = iql_update(
                &mut q_tables, raw.tech_idx, raw.order_idx, mixed_reward,
                &order_indices, step, n_orders, alpha, gamma);

            // CTDE: soft update for non-assigned agents with team mean reward
            for t in 0..n_tech {
                if t != raw.tech_idx {
                    let q_t = q_tables[t][raw.order_idx];
                    let max_q_next_t = if step + 1 < n_orders {
                        order_indices[step + 1..].iter()
                            .map(|&o| q_tables[t][o])
                            .fold(f64::NEG_INFINITY, f64::max)
                    } else { 0.0 };
                    // Non-assigned agents get team mean signal (softer update)
                    let td_t = team_mean + gamma * max_q_next_t - q_t;
                    q_tables[t][raw.order_idx] = q_t + (alpha * 0.3) * td_t;
                }
            }

            steps.push(Ch17Step {
                episode: ep, step,
                tech_idx: raw.tech_idx, order_idx: raw.order_idx,
                tech_x: raw.tx, tech_y: raw.ty,
                order_x: orders_x[raw.order_idx], order_y: orders_y[raw.order_idx],
                distance_km: raw.distance_km,
                reward: mixed_reward,
                individual_reward: raw.individual_reward,
                team_reward: team_mean,
                gt: 0.0,
                sla_met: raw.sla_met, skill_match: raw.skill_match,
                explored: raw.explored, epsilon: raw.epsilon,
                tech_skill: raw.tech_skill.clone(),
                order_skill: raw.order_skill.clone(),
                q_before, q_after, td_error,
                nearest_colleague_km: raw.nearest_colleague_km,
                collision_avoided: raw.collision_avoided,
                joint_q: raw.joint_q,
            });
        }

        let n = steps.len(); let mut gt = 0.0f64;
        for i in (0..n).rev() { gt = steps[i].reward + gamma * gt; steps[i].gt = gt; }
        let total_gt = steps.first().map(|s| s.gt).unwrap_or(0.0);
        let mean_joint_q = if n > 0 { joint_q_sum / n as f64 } else { 0.0 };
        let agent_stats = build_agent_stats(&steps, n_tech, &tech_skills, gamma, &mut agent_curves);
        let team_sla = steps.iter().filter(|s| s.sla_met).count() as f64 / steps.len() as f64;
        curve.push(total_gt);
        episodes.push(Ch17EpisodeResult {
            episode: ep, steps, total_gt, agent_stats,
            team_sla_rate: team_sla, collisions_avoided, mean_joint_q,
        });
    }
    Ch17Result { episodes, curve, agent_curves, final_q_tables: q_tables }
}

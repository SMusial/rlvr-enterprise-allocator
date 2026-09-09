//! Ch17 — Multi-Agent Reinforcement Learning (MARL)
//! Warsaw ASP: Independent Q-Learning (IQL)
//! Each technician is an independent agent with its own Q-table.
//! Individual rewards, no communication between agents.

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

const SKILLS: &[&str] = &["HVAC", "Electrical", "Network"];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ch17Step {
    pub episode:     usize,
    pub step:        usize,
    pub tech_idx:    usize,
    pub order_idx:   usize,
    pub tech_x:      f64,
    pub tech_y:      f64,
    pub order_x:     f64,
    pub order_y:     f64,
    pub distance_km: f64,
    pub reward:      f64,
    pub gt:          f64,
    pub sla_met:     bool,
    pub skill_match: bool,
    pub explored:    bool,
    pub epsilon:     f64,
    pub tech_skill:  String,
    pub order_skill: String,
    pub q_before:    f64,
    pub q_after:     f64,
    pub td_error:    f64,
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
    pub episode:       usize,
    pub steps:         Vec<Ch17Step>,
    pub total_gt:      f64,
    pub agent_stats:   Vec<Ch17AgentStats>,
    pub team_sla_rate: f64,
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

pub fn run_ch17(
    seed:          u64,
    n_tech:        usize,
    n_orders:      usize,
    n_ep:          usize,
    alpha:         f64,
    gamma:         f64,
    epsilon_start: f64,
    epsilon_end:   f64,
) -> Ch17Result {
    // Fixed environment (seed=42)
    let mut env_rng = StdRng::seed_from_u64(42);

    let orders_x: Vec<f64> = (0..n_orders).map(|_| 20.90 + env_rng.gen::<f64>() * 0.20).collect();
    let orders_y: Vec<f64> = (0..n_orders).map(|_| 52.18 + env_rng.gen::<f64>() * 0.14).collect();
    let order_skills: Vec<String> = (0..n_orders)
        .map(|_| SKILLS[env_rng.gen_range(0..SKILLS.len())].to_string())
        .collect();
    let order_sla: Vec<f64> = (0..n_orders).map(|_| 0.3 + env_rng.gen::<f64>() * 0.5).collect();

    let init_tech_x: Vec<f64> = (0..n_tech).map(|_| 20.90 + env_rng.gen::<f64>() * 0.20).collect();
    let init_tech_y: Vec<f64> = (0..n_tech).map(|_| 52.18 + env_rng.gen::<f64>() * 0.14).collect();
    let tech_skills: Vec<String> = (0..n_tech)
        .map(|_| SKILLS[env_rng.gen_range(0..SKILLS.len())].to_string())
        .collect();

    // Independent Q-tables: Q[tech][order] — each agent learns independently
    let mut q_tables: Vec<Vec<f64>> = vec![vec![0.0f64; n_orders]; n_tech];

    let mut curve: Vec<f64> = Vec::new();
    let mut agent_curves: Vec<Vec<f64>> = vec![Vec::new(); n_tech];
    let mut episodes: Vec<Ch17EpisodeResult> = Vec::new();

    for ep in 0..n_ep {
        let mut rng = StdRng::seed_from_u64(seed + ep as u64);

        // Linear epsilon decay
        let epsilon = epsilon_start
            - (epsilon_start - epsilon_end) * (ep as f64 / n_ep.max(1) as f64);

        // Shuffle order indices — each order dispatched exactly once
        let mut order_indices: Vec<usize> = (0..n_orders).collect();
        for i in (1..n_orders).rev() {
            let j = rng.gen_range(0..=i);
            order_indices.swap(i, j);
        }

        let mut tech_x = init_tech_x.clone();
        let mut tech_y = init_tech_y.clone();
        let mut steps: Vec<Ch17Step> = Vec::new();

        for step in 0..n_orders {
            let order_idx = order_indices[step];

            // IQL: each agent bids Q[tech][order_idx], highest bid wins
            // With epsilon-greedy: explore = random (with skill bias), exploit = highest Q
            let explored = rng.gen::<f64>() < epsilon;

            let matching_techs: Vec<usize> = (0..n_tech)
                .filter(|&t| tech_skills[t] == order_skills[order_idx])
                .collect();

            let tech_idx = if explored {
                if !matching_techs.is_empty() && rng.gen::<f64>() < 0.80 {
                    matching_techs[rng.gen_range(0..matching_techs.len())]
                } else {
                    rng.gen_range(0..n_tech)
                }
            } else {
                // Exploit: pick tech with highest Q for this order
                (0..n_tech)
                    .max_by(|&a, &b| {
                        q_tables[a][order_idx]
                            .partial_cmp(&q_tables[b][order_idx])
                            .unwrap()
                    })
                    .unwrap_or(0)
            };

            let tx = tech_x[tech_idx];
            let ty = tech_y[tech_idx];
            let ox = orders_x[order_idx];
            let oy = orders_y[order_idx];

            let distance_km = haversine_km(ty, tx, oy, ox);
            let skill_match = tech_skills[tech_idx] == order_skills[order_idx];
            let sla_met = skill_match && distance_km < order_sla[order_idx] * 20.0;

            let reward = if sla_met {
                2.0 - distance_km * 0.05
            } else if skill_match {
                0.5 - distance_km * 0.05
            } else {
                -1.0 - distance_km * 0.02
            };

            // IQL update: only assigned tech updates its own Q-table
            let q_before = q_tables[tech_idx][order_idx];
            let max_q_next = if step + 1 < n_orders {
                order_indices[step + 1..].iter()
                    .map(|&o| q_tables[tech_idx][o])
                    .fold(f64::NEG_INFINITY, f64::max)
            } else {
                0.0
            };
            let td_error = reward + gamma * max_q_next - q_before;
            let q_after = q_before + alpha * td_error;
            q_tables[tech_idx][order_idx] = q_after;

            // Technician moves to order location
            tech_x[tech_idx] = ox;
            tech_y[tech_idx] = oy;

            steps.push(Ch17Step {
                episode: ep,
                step,
                tech_idx,
                order_idx,
                tech_x: tx,
                tech_y: ty,
                order_x: ox,
                order_y: oy,
                distance_km,
                reward,
                gt: 0.0,
                sla_met,
                skill_match,
                explored,
                epsilon,
                tech_skill: tech_skills[tech_idx].clone(),
                order_skill: order_skills[order_idx].clone(),
                q_before,
                q_after,
                td_error,
            });
        }

        // Compute discounted returns backward
        let n = steps.len();
        let mut gt = 0.0f64;
        for i in (0..n).rev() {
            gt = steps[i].reward + gamma * gt;
            steps[i].gt = gt;
        }
        let total_gt = steps.first().map(|s| s.gt).unwrap_or(0.0);

        // Per-agent stats
        let mut agent_stats: Vec<Ch17AgentStats> = Vec::new();
        for t in 0..n_tech {
            let ts: Vec<&Ch17Step> = steps.iter().filter(|s| s.tech_idx == t).collect();
            let nt = ts.len();
            let agent_gt = if nt == 0 {
                0.0
            } else {
                let mut g = 0.0f64;
                for s in ts.iter().rev() { g = s.reward + gamma * g; }
                g
            };
            let sla_rate   = if nt == 0 { 0.0 } else { ts.iter().filter(|s| s.sla_met).count() as f64 / nt as f64 };
            let skill_rate = if nt == 0 { 0.0 } else { ts.iter().filter(|s| s.skill_match).count() as f64 / nt as f64 };
            let avg_dist   = if nt == 0 { 0.0 } else { ts.iter().map(|s| s.distance_km).sum::<f64>() / nt as f64 };
            agent_curves[t].push(agent_gt);
            agent_stats.push(Ch17AgentStats {
                tech_idx: t,
                tech_skill: tech_skills[t].clone(),
                total_gt: agent_gt,
                sla_rate,
                skill_rate,
                avg_distance: avg_dist,
                orders_served: nt,
            });
        }

        let team_sla = steps.iter().filter(|s| s.sla_met).count() as f64 / steps.len() as f64;
        curve.push(total_gt);

        episodes.push(Ch17EpisodeResult {
            episode: ep,
            steps,
            total_gt,
            agent_stats,
            team_sla_rate: team_sla,
        });
    }

    // Flatten Q-tables for serialization: [tech * n_orders + order]
    let final_q_tables: Vec<f64> = q_tables.iter().flat_map(|qt| qt.iter().cloned()).collect();

    Ch17Result {
        episodes,
        curve,
        agent_curves,
        final_q_tables,
    }
}

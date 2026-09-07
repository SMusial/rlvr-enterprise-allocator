//! Ch01 — MDP Baseline & ε-Greedy Dispatch
//! Warsaw ASP: technicians move to work order location after each dispatch.
//! Each work order is dispatched exactly once per episode.

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ch01Step {
    pub step:        usize,
    pub tech_idx:    usize,
    pub order_idx:   usize,
    pub tech_x:      f64,
    pub tech_y:      f64,
    pub order_x:     f64,
    pub order_y:     f64,
    pub distance:    f64,
    pub reward:      f64,
    pub gt:          f64,
    pub sla_met:     bool,
    pub skill_match: bool,
    pub explored:    bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ch01Result {
    pub steps:    Vec<Ch01Step>,
    pub total_gt: f64,
}

pub fn run_ch01_episode(
    seed:     u64,
    n_tech:   usize,
    n_orders: usize,
    epsilon:  f64,
    gamma:    f64,
) -> Ch01Result {
    // Policy RNG — different per episode
    let mut rng = StdRng::seed_from_u64(seed);

    // Environment RNG — fixed seed 42 so positions are always the same
    let mut env_rng = StdRng::seed_from_u64(42);

    // Fixed work order positions
    let orders_x: Vec<f64> = (0..n_orders)
        .map(|_| 20.90 + env_rng.gen::<f64>() * 0.20)
        .collect();
    let orders_y: Vec<f64> = (0..n_orders)
        .map(|_| 52.18 + env_rng.gen::<f64>() * 0.14)
        .collect();

    // Fixed initial technician positions
    let init_tech_x: Vec<f64> = (0..n_tech)
        .map(|_| 20.90 + env_rng.gen::<f64>() * 0.20)
        .collect();
    let init_tech_y: Vec<f64> = (0..n_tech)
        .map(|_| 52.18 + env_rng.gen::<f64>() * 0.14)
        .collect();

    // Fixed skills and SLA
    let tech_skills:  Vec<u8>  = (0..n_tech).map(|_| env_rng.gen_range(0u8..4u8)).collect();
    let order_skills: Vec<u8>  = (0..n_orders).map(|_| env_rng.gen_range(0u8..4u8)).collect();
    let order_sla:    Vec<f64> = (0..n_orders).map(|_| 0.3 + env_rng.gen::<f64>() * 0.5).collect();

    // Mutable technician positions — updated after each dispatch
    let mut tech_x = init_tech_x.clone();
    let mut tech_y = init_tech_y.clone();

    // Q-table = all zeros (baseline — no learning)
    let q_table = vec![vec![0.0f64; n_orders]; n_tech];

    // Shuffle order indices — each work order dispatched exactly once
    let mut order_indices: Vec<usize> = (0..n_orders).collect();
    for i in (1..n_orders).rev() {
        let j = rng.gen_range(0..=i);
        order_indices.swap(i, j);
    }

    let mut steps = Vec::new();

    for step in 0..n_orders {
        let order_idx = order_indices[step];

        // ε-greedy action selection
        let explored = rng.gen::<f64>() < epsilon;
        let tech_idx = if explored {
            rng.gen_range(0..n_tech)
        } else {
            (0..n_tech)
                .max_by(|&a, &b| {
                    q_table[a][order_idx]
                        .partial_cmp(&q_table[b][order_idx])
                        .unwrap()
                })
                .unwrap_or(0)
        };

        // Technician position BEFORE moving
        let tx = tech_x[tech_idx];
        let ty = tech_y[tech_idx];
        let ox = orders_x[order_idx];
        let oy = orders_y[order_idx];

        // Distance in km
        let dx = (tx - ox) * 111.0 * (52.2f64.to_radians().cos());
        let dy = (ty - oy) * 111.0;
        let distance = (dx * dx + dy * dy).sqrt();

        let skill_match = tech_skills[tech_idx] == order_skills[order_idx];
        let sla_met = skill_match && distance < order_sla[order_idx] * 20.0;

        let reward = if sla_met {
            2.0 - distance * 0.05
        } else if skill_match {
            0.5 - distance * 0.05
        } else {
            -1.0 - distance * 0.02
        };

        // Technician moves to work order location after dispatch
        tech_x[tech_idx] = ox;
        tech_y[tech_idx] = oy;

        steps.push(Ch01Step {
            step,
            tech_idx,
            order_idx,
            tech_x: tx,
            tech_y: ty,
            order_x: ox,
            order_y: oy,
            distance,
            reward,
            gt: 0.0,
            sla_met,
            skill_match,
            explored,
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
    Ch01Result { steps, total_gt }
}

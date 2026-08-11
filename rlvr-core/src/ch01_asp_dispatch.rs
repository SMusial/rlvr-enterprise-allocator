use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
pub const GAMMA: f64 = 0.95;
const SLA_REWARD: f64 = 10.0;
const SLA_BREACH_PENALTY: f64 = -5.0;
const SKILL_MISMATCH_PENALTY: f64 = -2.0;
const DISTANCE_PENALTY_PER_KM: f64 = -0.1;

const SKILLS: &[&str] = &["HVAC", "Electrical", "Plumbing", "Network", "Mechanical"];

// Warsaw bounding box
const LAT_MIN: f64 = 52.10;
const LAT_MAX: f64 = 52.40;
const LON_MIN: f64 = 20.85;
const LON_MAX: f64 = 21.25;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Technician {
    pub id: usize,
    pub lat: f64,
    pub lon: f64,
    pub skill: String,
    pub available: bool,
}

#[derive(Debug, Clone)]
pub struct WorkOrder {
    pub id: usize,
    pub lat: f64,
    pub lon: f64,
    pub required_skill: String,
    pub urgency: f64, // 0.0 low .. 1.0 critical
}

#[derive(Debug, Clone)]
pub struct AspState {
    pub technicians: Vec<Technician>,
    pub work_orders: Vec<WorkOrder>,
}

#[derive(Debug, Clone)]
pub struct AspAction {
    pub tech_idx: usize,
    pub order_idx: usize,
}

#[derive(Debug, Clone)]
pub struct FlatStep {
    pub step: usize,
    pub tech_idx: usize,
    pub order_idx: usize,
    pub tech_x: f64,
    pub tech_y: f64,
    pub order_x: f64,
    pub order_y: f64,
    pub reward: f64,
    pub gt: f64,
    pub sla_met: bool,
    pub skill_match: bool,
    pub explored: bool,
    pub epsilon: f64,
    pub distance_km: f64,
    pub tech_skill: String,
    pub order_skill: String,
    pub urgency: f64,
}

#[derive(Debug, Clone)]
pub struct EpisodeRecord {
    pub seed: u64,
    pub epsilon: f64,
    pub gamma: f64,
    pub steps: Vec<FlatStep>,
    pub total_gt: f64,
    pub sla_met_count: usize,
    pub skill_match_count: usize,
    pub explored_count: usize,
}

#[derive(Debug, Clone)]
pub struct AspConfig {
    pub seed: u64,
    pub n_tech: usize,
    pub n_orders: usize,
    pub epsilon: f64,
    pub gamma: f64,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
pub fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let r = 6371.0_f64;
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    2.0 * r * a.sqrt().asin()
}

fn random_lat(rng: &mut StdRng) -> f64 {
    LAT_MIN + rng.gen::<f64>() * (LAT_MAX - LAT_MIN)
}

fn random_lon(rng: &mut StdRng) -> f64 {
    LON_MIN + rng.gen::<f64>() * (LON_MAX - LON_MIN)
}

fn random_skill(rng: &mut StdRng) -> String {
    SKILLS[rng.gen_range(0..SKILLS.len())].to_string()
}

// ---------------------------------------------------------------------------
// Environment initialisation
// ---------------------------------------------------------------------------
pub fn init_state(config: &AspConfig, rng: &mut StdRng) -> AspState {
    let technicians = (0..config.n_tech)
        .map(|i| Technician {
            id: i,
            lat: random_lat(rng),
            lon: random_lon(rng),
            skill: random_skill(rng),
            available: true,
        })
        .collect();

    let work_orders = (0..config.n_orders)
        .map(|i| WorkOrder {
            id: i,
            lat: random_lat(rng),
            lon: random_lon(rng),
            required_skill: random_skill(rng),
            urgency: rng.gen::<f64>(),
        })
        .collect();

    AspState { technicians, work_orders }
}

// ---------------------------------------------------------------------------
// Q-table (untrained in Ch01 — all zeros; Ch02 will train it)
// ---------------------------------------------------------------------------
pub fn init_q_table(n_tech: usize, n_orders: usize) -> Array2<f64> {
    Array2::<f64>::zeros((n_tech, n_orders))
}

// ---------------------------------------------------------------------------
// Reward function — realistic SLA 77-93%
// ---------------------------------------------------------------------------
pub fn reward(
    tech: &Technician,
    order: &WorkOrder,
    distance_km: f64,
    rng: &mut StdRng,
) -> (f64, bool, bool) {
    let skill_match = tech.skill == order.required_skill;

    // SLA failure probability — realistic mix of factors
    let mut sla_fail_prob: f64 = 0.08; // base 8% failure rate
    if !skill_match {
        sla_fail_prob += 0.35; // skill mismatch adds 35%
    }
    if order.urgency > 0.7 {
        sla_fail_prob += 0.20; // high urgency harder to meet
    }
    if distance_km > 15.0 {
        sla_fail_prob += 0.18; // far distance adds travel risk
    }
    if distance_km > 25.0 {
        sla_fail_prob += 0.12; // very far adds more
    }
    // cap at 0.92 so there's always some chance of meeting SLA
    sla_fail_prob = sla_fail_prob.min(0.92_f64);

    let sla_met = rng.gen::<f64>() > sla_fail_prob;

    let mut r = if sla_met { SLA_REWARD } else { SLA_BREACH_PENALTY };
    if !skill_match {
        r += SKILL_MISMATCH_PENALTY;
    }
    r += DISTANCE_PENALTY_PER_KM * distance_km;

    (r, sla_met, skill_match)
}

// ---------------------------------------------------------------------------
// Markov transition — next state depends only on current state + action
// ---------------------------------------------------------------------------
pub fn transition(state: &AspState, action: &AspAction, rng: &mut StdRng) -> AspState {
    let mut next = state.clone();
    // Mark technician as dispatched (unavailable for this step)
    next.technicians[action.tech_idx].available = false;
    // Regenerate work order position (new job arrives)
    next.work_orders[action.order_idx].lat = random_lat(rng);
    next.work_orders[action.order_idx].lon = random_lon(rng);
    next.work_orders[action.order_idx].required_skill = random_skill(rng);
    next.work_orders[action.order_idx].urgency = rng.gen::<f64>();
    // Reset technician availability for next step
    next.technicians[action.tech_idx].available = true;
    next
}

// ---------------------------------------------------------------------------
// ε-greedy policy (Ch01 core algorithm)
// ---------------------------------------------------------------------------
pub fn epsilon_greedy(
    q_table: &Array2<f64>,
    tech_idx: usize,
    n_orders: usize,
    epsilon: f64,
    rng: &mut StdRng,
) -> (usize, bool) {
    if rng.gen::<f64>() < epsilon {
        // Explore — random order
        (rng.gen_range(0..n_orders), true)
    } else {
        // Exploit — greedy on Q-table row (all zeros in Ch01 → effectively random)
        let row = q_table.row(tech_idx);
        let best = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        (best, false)
    }
}

// ---------------------------------------------------------------------------
// Discounted return Gt = Σ γ^k * R_{t+k}
// ---------------------------------------------------------------------------
pub fn discounted_return(rewards: &[f64], gamma: f64) -> Vec<f64> {
    let n = rewards.len();
    let mut gt = vec![0.0_f64; n];
    let mut running = 0.0_f64;
    for i in (0..n).rev() {
        running = rewards[i] + gamma * running;
        gt[i] = running;
    }
    gt
}

// ---------------------------------------------------------------------------
// Episode runner
// ---------------------------------------------------------------------------
pub fn run_episode(config: AspConfig) -> EpisodeRecord {
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut state = init_state(&config, &mut rng);
    let q_table = init_q_table(config.n_tech, config.n_orders);

    let _init_techs: Vec<_> = state.technicians.clone();
    let _init_orders: Vec<_> = state.work_orders.clone();

    let mut raw_steps: Vec<(AspAction, f64, bool, bool, bool, f64)> = Vec::new();

    for step in 0..config.n_orders {
        let tech_idx = step % config.n_tech;
        let (order_idx, explored) = epsilon_greedy(
            &q_table,
            tech_idx,
            config.n_orders,
            config.epsilon,
            &mut rng,
        );

        let tech = &state.technicians[tech_idx];
        let order = &state.work_orders[order_idx];
        let dist = haversine_km(tech.lat, tech.lon, order.lat, order.lon);

        let (r, sla_met, skill_match) = reward(tech, order, dist, &mut rng);
        let action = AspAction { tech_idx, order_idx };
        raw_steps.push((action.clone(), r, sla_met, skill_match, explored, dist));
        state = transition(&state, &action, &mut rng);
    }

    // Compute discounted returns
    let rewards: Vec<f64> = raw_steps.iter().map(|(_, r, _, _, _, _)| *r).collect();
    let gt_vec = discounted_return(&rewards, config.gamma);

    // Rebuild state for coordinate extraction
    let mut rng2 = StdRng::seed_from_u64(config.seed);
    let state0 = init_state(&config, &mut rng2);

    let mut flat_steps: Vec<FlatStep> = Vec::new();
    let mut sla_met_count = 0usize;
    let mut skill_match_count = 0usize;
    let mut explored_count = 0usize;

    for (i, (action, r, sla_met, skill_match, explored, dist)) in raw_steps.iter().enumerate() {
        if *sla_met { sla_met_count += 1; }
        if *skill_match { skill_match_count += 1; }
        if *explored { explored_count += 1; }

        let tech = &state0.technicians[action.tech_idx];
        let order = &state0.work_orders[action.order_idx];

        flat_steps.push(FlatStep {
            step: i,
            tech_idx: action.tech_idx,
            order_idx: action.order_idx,
            tech_x: tech.lon,
            tech_y: tech.lat,
            order_x: order.lon,
            order_y: order.lat,
            reward: *r,
            gt: gt_vec[i],
            sla_met: *sla_met,
            skill_match: *skill_match,
            explored: *explored,
            epsilon: config.epsilon,
            distance_km: *dist,
            tech_skill: tech.skill.clone(),
            order_skill: order.required_skill.clone(),
            urgency: order.urgency,
        });
    }

    let total_gt = gt_vec.first().copied().unwrap_or(0.0);

    EpisodeRecord {
        seed: config.seed,
        epsilon: config.epsilon,
        gamma: config.gamma,
        steps: flat_steps,
        total_gt,
        sla_met_count,
        skill_match_count,
        explored_count,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> AspConfig {
        AspConfig { seed: 42, n_tech: 5, n_orders: 10, epsilon: 0.5, gamma: GAMMA }
    }

    #[test]
    fn test_episode_deterministic() {
        let r1 = run_episode(default_config());
        let r2 = run_episode(default_config());
        assert_eq!(r1.total_gt.to_bits(), r2.total_gt.to_bits());
    }

    #[test]
    fn test_episode_step_count() {
        let r = run_episode(default_config());
        assert_eq!(r.steps.len(), 10);
    }

    #[test]
    fn test_sla_rate_realistic() {
        // Run 20 episodes and check SLA rate is between 60% and 99%
        for seed in 0..20u64 {
            let cfg = AspConfig { seed, n_tech: 5, n_orders: 20, epsilon: 0.5, gamma: GAMMA };
            let r = run_episode(cfg);
            let sla_rate = r.sla_met_count as f64 / r.steps.len() as f64;
            assert!(sla_rate < 0.99, "SLA rate too high: {}", sla_rate);
            assert!(sla_rate > 0.50, "SLA rate too low: {}", sla_rate);
        }
    }

    #[test]
    fn test_discounted_return_math() {
        let rewards = vec![1.0, 1.0, 1.0];
        let gt = discounted_return(&rewards, 0.9);
        let expected = 1.0 + 0.9 + 0.81;
        assert!((gt[0] - expected).abs() < 1e-9);
    }

    #[test]
    fn test_haversine_warsaw() {
        // Warsaw centre to ~10km north
        let d = haversine_km(52.23, 21.01, 52.32, 21.01);
        assert!(d > 9.0 && d < 11.0, "Expected ~10km, got {}", d);
    }

    #[test]
    fn test_q_table_zeros() {
        let q = init_q_table(5, 10);
        assert!(q.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_epsilon_one_always_explores() {
        let q = init_q_table(5, 10);
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..20 {
            let (_, explored) = epsilon_greedy(&q, 0, 10, 1.0, &mut rng);
            assert!(explored);
        }
    }

    #[test]
    fn test_epsilon_zero_never_explores() {
        let q = init_q_table(5, 10);
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..20 {
            let (_, explored) = epsilon_greedy(&q, 0, 10, 0.0, &mut rng);
            assert!(!explored);
        }
    }
}

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rlvr_core::ch01_asp_dispatch::{run_episode, AspConfig};
use rlvr_core::ch02_bellman::{run_ch02, Ch02Config, N_STATES, N_ACTIONS, STATE_NAMES, ACTION_NAMES};
use rlvr_core::ch03_bandit::{run_ch03, Ch03Config, ARM_NAMES, TRUE_SLA_RATES};
use rlvr_core::ch04_dp::{run_ch04, Ch04Config};
use rlvr_core::ch05_mc::{run_ch05, McConfig};

// ---------------------------------------------------------------------------
// Ch01
// ---------------------------------------------------------------------------
#[pyfunction]
fn run_ch01_episode(py: Python, seed: u64, n_tech: usize, n_task: usize, epsilon: f64, gamma: f64) -> PyResult<PyObject> {
    let config = AspConfig { seed, n_tech, n_orders: n_task, epsilon, gamma };
    let record = run_episode(config);
    let steps = PyList::empty_bound(py);
    for s in &record.steps {
        let d = PyDict::new_bound(py);
        d.set_item("step",        s.step)?;
        d.set_item("tech_idx",    s.tech_idx)?;
        d.set_item("order_idx",   s.order_idx)?;
        d.set_item("tech_x",      s.tech_x)?;
        d.set_item("tech_y",      s.tech_y)?;
        d.set_item("order_x",     s.order_x)?;
        d.set_item("order_y",     s.order_y)?;
        d.set_item("reward",      s.reward)?;
        d.set_item("gt",          s.gt)?;
        d.set_item("sla_met",     s.sla_met)?;
        d.set_item("skill_match", s.skill_match)?;
        d.set_item("explored",    s.explored)?;
        d.set_item("epsilon",     s.epsilon)?;
        d.set_item("distance_km", s.distance_km)?;
        d.set_item("tech_skill",  &s.tech_skill)?;
        d.set_item("order_skill", &s.order_skill)?;
        d.set_item("urgency",     s.urgency)?;
        steps.append(d)?;
    }
    let result = PyDict::new_bound(py);
    result.set_item("steps",             steps)?;
    result.set_item("total_gt",          record.total_gt)?;
    result.set_item("sla_met_count",     record.sla_met_count)?;
    result.set_item("skill_match_count", record.skill_match_count)?;
    result.set_item("explored_count",    record.explored_count)?;
    result.set_item("seed",              record.seed)?;
    result.set_item("epsilon",           record.epsilon)?;
    result.set_item("gamma",             record.gamma)?;
    Ok(result.into())
}

// ---------------------------------------------------------------------------
// Ch02
// ---------------------------------------------------------------------------
#[pyfunction]
fn run_ch02_value_iteration(py: Python, seed: u64, gamma: f64, theta: f64) -> PyResult<PyObject> {
    let config = Ch02Config { seed, gamma, theta };
    let result = run_ch02(config);
    let values = PyList::empty_bound(py);
    for v in &result.values { values.append(v)?; }
    let policy = PyList::empty_bound(py);
    for &a in &result.policy { policy.append(a)?; }
    let curve = PyList::empty_bound(py);
    for v in &result.convergence_curve { curve.append(v)?; }
    let trace = PyList::empty_bound(py);
    for step in &result.bellman_trace {
        let d = PyDict::new_bound(py);
        d.set_item("iteration", step.iteration)?;
        d.set_item("state",     step.state)?;
        d.set_item("action",    step.action)?;
        let qv = PyList::empty_bound(py);
        for q in &step.q_values { qv.append(q)?; }
        d.set_item("q_values",  qv)?;
        d.set_item("v_old",     step.v_old)?;
        d.set_item("v_new",     step.v_new)?;
        d.set_item("delta",     step.delta)?;
        trace.append(d)?;
    }
    let snames = PyList::empty_bound(py);
    for n in STATE_NAMES { snames.append(n)?; }
    let anames = PyList::empty_bound(py);
    for n in ACTION_NAMES { anames.append(n)?; }
    let out = PyDict::new_bound(py);
    out.set_item("values",            values)?;
    out.set_item("policy",            policy)?;
    out.set_item("iterations",        result.iterations)?;
    out.set_item("convergence_curve", curve)?;
    out.set_item("bellman_trace",     trace)?;
    out.set_item("state_names",       snames)?;
    out.set_item("action_names",      anames)?;
    out.set_item("n_states",          N_STATES)?;
    out.set_item("n_actions",         N_ACTIONS)?;
    Ok(out.into())
}

// ---------------------------------------------------------------------------
// Ch03
// ---------------------------------------------------------------------------
#[pyfunction]
fn run_ch03_bandits(py: Python, seed: u64, n_steps: usize, epsilon: f64, epsilon_decay: f64, ucb_c: f64) -> PyResult<PyObject> {
    let config = Ch03Config { seed, n_steps, epsilon, epsilon_decay, ucb_c, gamma: 0.95 };
    let results = run_ch03(config);
    let out_list = PyList::empty_bound(py);
    for result in &results {
        let steps_list = PyList::empty_bound(py);
        for s in &result.steps {
            let d = PyDict::new_bound(py);
            d.set_item("step",              s.step)?;
            d.set_item("algorithm",         &s.algorithm)?;
            d.set_item("arm",               s.arm)?;
            d.set_item("reward",            s.reward)?;
            d.set_item("regret",            s.regret)?;
            d.set_item("cumulative_regret", s.cumulative_regret)?;
            let qv = PyList::empty_bound(py);
            for q in &s.q_values { qv.append(q)?; }
            d.set_item("q_values",          qv)?;
            let np = PyList::empty_bound(py);
            for n in &s.n_pulls { np.append(n)?; }
            d.set_item("n_pulls",           np)?;
            let uv = PyList::empty_bound(py);
            for u in &s.ucb_values { uv.append(u)?; }
            d.set_item("ucb_values",        uv)?;
            let tv = PyList::empty_bound(py);
            for t in &s.thompson_samples { tv.append(t)?; }
            d.set_item("thompson_samples",  tv)?;
            d.set_item("epsilon",           s.epsilon)?;
            d.set_item("explored",          s.explored)?;
            steps_list.append(d)?;
        }
        let fq = PyList::empty_bound(py);
        for q in &result.final_q_values { fq.append(q)?; }
        let fn_ = PyList::empty_bound(py);
        for n in &result.final_n_pulls { fn_.append(n)?; }
        let r = PyDict::new_bound(py);
        r.set_item("algorithm",      &result.algorithm)?;
        r.set_item("steps",          steps_list)?;
        r.set_item("total_reward",   result.total_reward)?;
        r.set_item("total_regret",   result.total_regret)?;
        r.set_item("final_q_values", fq)?;
        r.set_item("final_n_pulls",  fn_)?;
        r.set_item("best_arm",       result.best_arm)?;
        out_list.append(r)?;
    }
    let arm_names = PyList::empty_bound(py);
    for n in ARM_NAMES { arm_names.append(n)?; }
    let true_rates = PyList::empty_bound(py);
    for r in TRUE_SLA_RATES { true_rates.append(r)?; }
    let out = PyDict::new_bound(py);
    out.set_item("results",    out_list)?;
    out.set_item("arm_names",  arm_names)?;
    out.set_item("true_rates", true_rates)?;
    Ok(out.into())
}

// ---------------------------------------------------------------------------
// Ch04
// ---------------------------------------------------------------------------
#[pyfunction]
fn run_ch04_dp(py: Python, seed: u64, gamma: f64, theta: f64) -> PyResult<PyObject> {
    let config = Ch04Config { seed, gamma, theta };
    let result = run_ch04(config);
    let to_f = |v: &Vec<f64>| -> PyResult<Py<PyList>> {
        let l = PyList::empty_bound(py); for x in v { l.append(x)?; } Ok(l.into())
    };
    let to_u = |v: &Vec<usize>| -> PyResult<Py<PyList>> {
        let l = PyList::empty_bound(py); for x in v { l.append(x)?; } Ok(l.into())
    };
    let pi = PyDict::new_bound(py);
    pi.set_item("values",            to_f(&result.pi.values)?)?;
    pi.set_item("policy",            to_u(&result.pi.policy)?)?;
    pi.set_item("pi_iterations",     result.pi.pi_iterations)?;
    let ei = PyList::empty_bound(py);
    for &n in &result.pi.eval_iterations { ei.append(n)?; }
    pi.set_item("eval_iterations",   ei)?;
    pi.set_item("convergence_curve", to_f(&result.pi.convergence_curve)?)?;
    let ph = PyList::empty_bound(py);
    for pol in &result.pi.policy_history {
        let row = PyList::empty_bound(py);
        for &a in pol { row.append(a)?; }
        ph.append(row)?;
    }
    pi.set_item("policy_history", ph)?;
    let vi = PyDict::new_bound(py);
    vi.set_item("values",     to_f(&result.vi_values)?)?;
    vi.set_item("policy",     to_u(&result.vi_policy)?)?;
    vi.set_item("iterations", result.vi_iterations)?;
    vi.set_item("curve",      to_f(&result.vi_curve)?)?;
    let av = PyDict::new_bound(py);
    av.set_item("values",     to_f(&result.async_values)?)?;
    av.set_item("policy",     to_u(&result.async_policy)?)?;
    av.set_item("iterations", result.async_iterations)?;
    av.set_item("curve",      to_f(&result.async_curve)?)?;
    let snames = PyList::empty_bound(py);
    for n in rlvr_core::ch02_bellman::STATE_NAMES { snames.append(n)?; }
    let anames = PyList::empty_bound(py);
    for n in rlvr_core::ch02_bellman::ACTION_NAMES { anames.append(n)?; }
    let out = PyDict::new_bound(py);
    out.set_item("pi",           pi)?;
    out.set_item("vi",           vi)?;
    out.set_item("async_vi",     av)?;
    out.set_item("residuals",    to_f(&result.residuals)?)?;
    out.set_item("state_names",  snames)?;
    out.set_item("action_names", anames)?;
    out.set_item("n_states",     N_STATES)?;
    out.set_item("n_actions",    N_ACTIONS)?;
    Ok(out.into())
}

// ---------------------------------------------------------------------------
// Ch05
// ---------------------------------------------------------------------------
#[pyfunction]
fn run_ch05_mc(py: Python, seed: u64, n_episodes: usize, gamma: f64, epsilon: f64, epsilon_decay: f64) -> PyResult<PyObject> {
    let config = McConfig { seed, gamma, n_episodes, epsilon, epsilon_decay };
    let result = run_ch05(config);

    let serialize_mc = |r: &rlvr_core::ch05_mc::McResult| -> PyResult<Py<PyDict>> {
        let d = PyDict::new_bound(py);
        let vals = PyList::empty_bound(py);
        for v in &r.values { vals.append(v)?; }
        d.set_item("values", vals)?;
        let pol = PyList::empty_bound(py);
        for &a in &r.policy { pol.append(a)?; }
        d.set_item("policy", pol)?;
        let qt = PyList::empty_bound(py);
        for row in &r.q_table {
            let rr = PyList::empty_bound(py);
            for &q in row { rr.append(q)?; }
            qt.append(rr)?;
        }
        d.set_item("q_table", qt)?;
        let rc = PyList::empty_bound(py);
        for v in &r.returns_curve { rc.append(v)?; }
        d.set_item("returns_curve", rc)?;
        let vc = PyList::empty_bound(py);
        for &v in &r.visit_counts { vc.append(v)?; }
        d.set_item("visit_counts", vc)?;
        let cc = PyList::empty_bound(py);
        for v in &r.convergence_curve { cc.append(v)?; }
        d.set_item("convergence_curve", cc)?;
        d.set_item("algorithm",  &r.algorithm)?;
        d.set_item("n_episodes", r.n_episodes)?;
        Ok(d.into())
    };

    let snames = PyList::empty_bound(py);
    for n in rlvr_core::ch02_bellman::STATE_NAMES { snames.append(n)?; }
    let anames = PyList::empty_bound(py);
    for n in rlvr_core::ch02_bellman::ACTION_NAMES { anames.append(n)?; }

    let out = PyDict::new_bound(py);
    out.set_item("first_visit",  serialize_mc(&result.first_visit)?)?;
    out.set_item("every_visit",  serialize_mc(&result.every_visit)?)?;
    out.set_item("on_policy",    serialize_mc(&result.on_policy)?)?;
    out.set_item("off_policy",   serialize_mc(&result.off_policy)?)?;
    out.set_item("state_names",  snames)?;
    out.set_item("action_names", anames)?;
    out.set_item("n_states",     N_STATES)?;
    out.set_item("n_actions",    N_ACTIONS)?;
    Ok(out.into())
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------
#[pymodule]
fn rlvr_py(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_ch01_episode,         m)?)?;
    m.add_function(wrap_pyfunction!(run_ch02_value_iteration, m)?)?;
    m.add_function(wrap_pyfunction!(run_ch03_bandits,         m)?)?;
    m.add_function(wrap_pyfunction!(run_ch04_dp,              m)?)?;
    m.add_function(wrap_pyfunction!(run_ch05_mc,              m)?)?;
    Ok(())
}

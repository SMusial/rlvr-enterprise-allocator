use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rlvr_core::ch01_asp_dispatch::{run_episode, AspConfig};
use rlvr_core::ch02_bellman::{
    run_ch02, Ch02Config, N_STATES, N_ACTIONS, STATE_NAMES, ACTION_NAMES,
};

// ---------------------------------------------------------------------------
// Ch01
// ---------------------------------------------------------------------------
#[pyfunction]
fn run_ch01_episode(
    py: Python,
    seed: u64,
    n_tech: usize,
    n_task: usize,
    epsilon: f64,
    gamma: f64,
) -> PyResult<PyObject> {
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
fn run_ch02_value_iteration(
    py: Python,
    seed: u64,
    gamma: f64,
    theta: f64,
) -> PyResult<PyObject> {
    let config = Ch02Config { seed, gamma, theta };
    let result = run_ch02(config);

    // values
    let values = PyList::empty_bound(py);
    for v in &result.values {
        values.append(v)?;
    }

    // policy
    let policy = PyList::empty_bound(py);
    for &a in &result.policy {
        policy.append(a)?;
    }

    // convergence curve
    let curve = PyList::empty_bound(py);
    for v in &result.convergence_curve {
        curve.append(v)?;
    }

    // bellman trace
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

    // state and action names
    let snames = PyList::empty_bound(py);
    for n in STATE_NAMES { snames.append(n)?; }
    let anames = PyList::empty_bound(py);
    for n in ACTION_NAMES { anames.append(n)?; }

    let out = PyDict::new_bound(py);
    out.set_item("values",           values)?;
    out.set_item("policy",           policy)?;
    out.set_item("iterations",       result.iterations)?;
    out.set_item("convergence_curve", curve)?;
    out.set_item("bellman_trace",    trace)?;
    out.set_item("state_names",      snames)?;
    out.set_item("action_names",     anames)?;
    out.set_item("n_states",         N_STATES)?;
    out.set_item("n_actions",        N_ACTIONS)?;

    Ok(out.into())
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------
#[pymodule]
fn rlvr_py(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_ch01_episode, m)?)?;
    m.add_function(wrap_pyfunction!(run_ch02_value_iteration, m)?)?;
    Ok(())
}

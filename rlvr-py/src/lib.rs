use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rlvr_core::ch01_asp_dispatch::{run_episode, AspConfig};

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
        d.set_item("step",         s.step)?;
        d.set_item("tech_idx",     s.tech_idx)?;
        d.set_item("order_idx",    s.order_idx)?;
        d.set_item("tech_x",       s.tech_x)?;
        d.set_item("tech_y",       s.tech_y)?;
        d.set_item("order_x",      s.order_x)?;
        d.set_item("order_y",      s.order_y)?;
        d.set_item("reward",       s.reward)?;
        d.set_item("gt",           s.gt)?;
        d.set_item("sla_met",      s.sla_met)?;
        d.set_item("skill_match",  s.skill_match)?;
        d.set_item("explored",     s.explored)?;
        d.set_item("epsilon",      s.epsilon)?;
        d.set_item("distance_km",  s.distance_km)?;
        d.set_item("tech_skill",   &s.tech_skill)?;
        d.set_item("order_skill",  &s.order_skill)?;
        d.set_item("urgency",      s.urgency)?;
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

#[pymodule]
fn rlvr_py(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_ch01_episode, m)?)?;
    Ok(())
}

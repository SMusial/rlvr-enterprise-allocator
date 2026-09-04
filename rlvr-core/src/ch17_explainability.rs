//! Ch17 — Model Explainability and Interpretability (tch 0.14 / libtorch 2.1.0)
//! Gradient Feature Importance, Saliency Maps, LIME, SHAP on Warsaw ASP FNN.
//! Business context: Explain WHY the FNN from Ch15 makes each dispatch decision.

use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

// ── Warsaw ASP data (same as Ch15) ───────────────────────────────────────────
const STATE_FEATURES: [[f32; 4]; 8] = [
    [0.95, 0.9, 0.1, 1.0],
    [0.88, 0.8, 0.2, 0.9],
    [0.80, 0.7, 0.3, 0.8],
    [0.72, 0.6, 0.4, 0.7],
    [0.65, 0.5, 0.5, 0.6],
    [0.55, 0.4, 0.6, 0.5],
    [0.45, 0.3, 0.7, 0.4],
    [0.30, 0.2, 0.9, 0.2],
];
const TRUE_V_STAR: [f32; 8] = [18.5, 15.2, 12.8, 10.1, 7.4, 4.9, 2.3, 0.5];
const FEATURE_NAMES: [&str; 4] = ["SLA Rate", "Urgency", "Distance", "Skill Match"];

// ── Result types ─────────────────────────────────────────────────────────────
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ch17StateExplanation {
    pub state_idx: usize,
    pub features: Vec<f32>,
    pub prediction: f64,
    pub true_value: f64,
    // Gradient Feature Importance: |dL/dx_j| per feature
    pub gradient_fi: Vec<f64>,
    // Saliency: |df/dx_j| per feature
    pub saliency: Vec<f64>,
    // LIME: local linear coefficients per feature
    pub lime_coefs: Vec<f64>,
    pub lime_intercept: f64,
    pub lime_r2: f64,
    // SHAP: Shapley values per feature
    pub shap_values: Vec<f64>,
    pub shap_base: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ch17Result {
    pub method: String,
    pub feature_names: Vec<String>,
    pub global_importance: Vec<f64>,  // averaged over all states
    pub state_explanations: Vec<Ch17StateExplanation>,
    pub model_r2: f64,
}

#[derive(Serialize, Deserialize)]
pub struct Ch17Config {
    pub n_train_epochs: usize,
    pub lr: f64,
    pub hidden_units: usize,
    pub n_lime_samples: usize,
    pub n_shap_samples: usize,
    pub seed: u64,
}

// ── Build and train FNN (same as Ch15) ───────────────────────────────────────
fn build_and_train(config: &Ch17Config) -> (nn::VarStore, Vec<nn::Linear>) {
    tch::manual_seed(config.seed as i64);
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let root = vs.root();

    let linears = vec![
        nn::linear(&root / "l0", 4, config.hidden_units as i64, Default::default()),
        nn::linear(&root / "l1", config.hidden_units as i64, config.hidden_units as i64, Default::default()),
        nn::linear(&root / "l2", config.hidden_units as i64, 1, Default::default()),
    ];

    let v_max = TRUE_V_STAR.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let v_min = TRUE_V_STAR.iter().cloned().fold(f32::INFINITY, f32::min);
    let y_norm: Vec<f32> = TRUE_V_STAR.iter().map(|&v| (v - v_min) / (v_max - v_min)).collect();

    let x_flat: Vec<f32> = STATE_FEATURES.iter().flat_map(|r| r.iter().cloned()).collect();
    let x = Tensor::from_slice(&x_flat).view([8, 4]).to_kind(Kind::Float);
    let y = Tensor::from_slice(&y_norm).view([8, 1]).to_kind(Kind::Float);

    let mut opt = nn::Adam::default().build(&vs, config.lr).unwrap();

    for _ in 0..config.n_train_epochs {
        let mut h = x.shallow_clone();
        for (i, lin) in linears.iter().enumerate() {
            h = lin.forward(&h);
            if i < linears.len() - 1 { h = h.relu(); }
        }
        let loss = h.mse_loss(&y, tch::Reduction::Mean);
        opt.backward_step(&loss);
    }

    (vs, linears)
}

fn forward_pass(linears: &[nn::Linear], x: &Tensor) -> Tensor {
    let mut h = x.shallow_clone();
    for (i, lin) in linears.iter().enumerate() {
        h = lin.forward(&h);
        if i < linears.len() - 1 { h = h.relu(); }
    }
    h
}

fn predict_scalar(linears: &[nn::Linear], x: &Tensor) -> f64 {
    tch::no_grad(|| forward_pass(linears, x)).double_value(&[0, 0])
}

// ── Gradient Feature Importance ───────────────────────────────────────────────
fn gradient_feature_importance(
    linears: &[nn::Linear],
    state_idx: usize,
) -> Vec<f64> {
    let feat: Vec<f32> = STATE_FEATURES[state_idx].to_vec();
    let x = Tensor::from_slice(&feat)
        .view([1, 4])
        .to_kind(Kind::Float)
        .set_requires_grad(true);

    let pred = forward_pass(linears, &x);
    pred.backward();

    let grad = x.grad();
    Vec::<f32>::try_from(grad.view(-1)).unwrap_or_default()
        .iter().map(|&v| v.abs() as f64).collect()
}

// ── Saliency Map ─────────────────────────────────────────────────────────────
fn saliency_map(
    linears: &[nn::Linear],
    state_idx: usize,
) -> Vec<f64> {
    let feat: Vec<f32> = STATE_FEATURES[state_idx].to_vec();
    let x = Tensor::from_slice(&feat)
        .view([1, 4])
        .to_kind(Kind::Float)
        .set_requires_grad(true);

    let pred = forward_pass(linears, &x);
    pred.backward();

    let grad = x.grad();
    Vec::<f32>::try_from(grad.abs().view(-1)).unwrap_or_default()
        .iter().map(|&v| v as f64).collect()
}

// ── LIME ─────────────────────────────────────────────────────────────────────
fn lime_explain(
    linears: &[nn::Linear],
    state_idx: usize,
    n_samples: usize,
    rng: &mut StdRng,
) -> (Vec<f64>, f64, f64) {
    let x0: Vec<f32> = STATE_FEATURES[state_idx].to_vec();
    let n_feat = x0.len();

    // Generate perturbed samples around x0
    let mut xs: Vec<Vec<f32>> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    let mut ws: Vec<f64> = Vec::new();

    for _ in 0..n_samples {
        let perturbed: Vec<f32> = x0.iter().map(|&xi| {
            xi + rng.gen_range(-0.2f32..0.2f32)
        }).collect();

        let t = Tensor::from_slice(&perturbed).view([1, 4]).to_kind(Kind::Float);
        let pred = predict_scalar(linears, &t);

        // Kernel weight: exp(-d^2 / sigma^2)
        let d2: f32 = perturbed.iter().zip(x0.iter())
            .map(|(&a, &b)| (a - b).powi(2)).sum();
        let w = (-d2 / 0.25f32).exp() as f64;

        xs.push(perturbed);
        ys.push(pred);
        ws.push(w);
    }

    // Weighted linear regression: min Σ w_i (y_i - (β₀ + Σ βⱼ xⱼ))²
    // Using normal equations with weights
    let n = n_samples;
    let p = n_feat + 1; // +1 for intercept

    // Build weighted design matrix X_aug [1, x1, x2, x3, x4]
    let mut xtx = vec![vec![0.0f64; p]; p];
    let mut xty = vec![0.0f64; p];

    for i in 0..n {
        let w = ws[i];
        let mut row = vec![1.0f64];
        row.extend(xs[i].iter().map(|&v| v as f64));

        for j in 0..p {
            xty[j] += w * row[j] * ys[i];
            for k in 0..p {
                xtx[j][k] += w * row[j] * row[k];
            }
        }
    }

    // Solve via Gaussian elimination
    let coefs = solve_linear_system(&xtx, &xty);
    let intercept = coefs[0];
    let beta: Vec<f64> = coefs[1..].to_vec();

    // Compute R²
    let y_mean = ys.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = ys.iter().map(|&y| (y - y_mean).powi(2)).sum();
    let ss_res: f64 = (0..n).map(|i| {
        let y_hat = intercept + beta.iter().zip(xs[i].iter())
            .map(|(&b, &x)| b * x as f64).sum::<f64>();
        (ys[i] - y_hat).powi(2)
    }).sum();
    let r2 = if ss_tot > 1e-10 { 1.0 - ss_res / ss_tot } else { 0.0 };

    (beta, intercept, r2)
}

fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut mat: Vec<Vec<f64>> = a.iter().map(|row| row.clone()).collect();
    let mut rhs = b.to_vec();

    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        for row in col+1..n {
            if mat[row][col].abs() > mat[max_row][col].abs() { max_row = row; }
        }
        mat.swap(col, max_row);
        rhs.swap(col, max_row);

        let pivot = mat[col][col];
        if pivot.abs() < 1e-12 { continue; }

        for row in col+1..n {
            let factor = mat[row][col] / pivot;
            for k in col..n { mat[row][k] -= factor * mat[col][k]; }
            rhs[row] -= factor * rhs[col];
        }
    }

    // Back substitution
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        x[i] = rhs[i];
        for j in i+1..n { x[i] -= mat[i][j] * x[j]; }
        if mat[i][i].abs() > 1e-12 { x[i] /= mat[i][i]; }
    }
    x
}

// ── SHAP (KernelSHAP approximation) ──────────────────────────────────────────
fn shap_values(
    linears: &[nn::Linear],
    state_idx: usize,
    n_samples: usize,
    rng: &mut StdRng,
) -> (Vec<f64>, f64) {
    let x0: Vec<f32> = STATE_FEATURES[state_idx].to_vec();
    let n_feat = x0.len();

    // Baseline: mean of all states
    let baseline: Vec<f32> = (0..n_feat).map(|j| {
        STATE_FEATURES.iter().map(|s| s[j]).sum::<f32>() / 8.0
    }).collect();

    let base_t = Tensor::from_slice(&baseline).view([1, 4]).to_kind(Kind::Float);
    let base_pred = predict_scalar(linears, &base_t);

    let full_t = Tensor::from_slice(&x0).view([1, 4]).to_kind(Kind::Float);
    let full_pred = predict_scalar(linears, &full_t);

    // KernelSHAP: sample coalitions
    let mut phi = vec![0.0f64; n_feat];
    let mut counts = vec![0usize; n_feat];

    for _ in 0..n_samples {
        // Random coalition S (subset of features)
        let coalition: Vec<bool> = (0..n_feat).map(|_| rng.gen::<bool>()).collect();
        let s_size = coalition.iter().filter(|&&b| b).count();
        if s_size == 0 || s_size == n_feat { continue; }

        // f(S) — coalition with x0 features, baseline for others
        let x_s: Vec<f32> = (0..n_feat).map(|j| {
            if coalition[j] { x0[j] } else { baseline[j] }
        }).collect();
        let t_s = Tensor::from_slice(&x_s).view([1, 4]).to_kind(Kind::Float);
        let f_s = predict_scalar(linears, &t_s);

        // Shapley kernel weight
        let w = shapley_kernel(n_feat, s_size);

        // For each feature i in coalition, attribute marginal contribution
        for i in 0..n_feat {
            if coalition[i] {
                // f(S) - f(S \ {i})
                let mut x_s_minus_i = x_s.clone();
                x_s_minus_i[i] = baseline[i];
                let t_minus = Tensor::from_slice(&x_s_minus_i).view([1, 4]).to_kind(Kind::Float);
                let f_s_minus_i = predict_scalar(linears, &t_minus);
                phi[i] += w * (f_s - f_s_minus_i);
                counts[i] += 1;
            }
        }
    }

    // Normalize
    let phi_norm: Vec<f64> = phi.iter().zip(counts.iter())
        .map(|(&p, &c)| if c > 0 { p / c as f64 } else { 0.0 })
        .collect();

    // Scale so that sum(phi) = full_pred - base_pred
    let phi_sum: f64 = phi_norm.iter().sum();
    let target_sum = full_pred - base_pred;
    let scale = if phi_sum.abs() > 1e-10 { target_sum / phi_sum } else { 1.0 };
    let phi_scaled: Vec<f64> = phi_norm.iter().map(|&p| p * scale).collect();

    (phi_scaled, base_pred)
}

fn shapley_kernel(n: usize, s: usize) -> f64 {
    // w(S) = (n-1)! / (|S|! * (n-|S|-1)! * C(n,|S|))
    // Simplified: use 1/(s*(n-s)) as approximation
    let n = n as f64;
    let s = s as f64;
    (n - 1.0) / (s * (n - s))
}

// ── Compute R² of trained model ───────────────────────────────────────────────
fn compute_r2(linears: &[nn::Linear]) -> f64 {
    let v_max = TRUE_V_STAR.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let v_min = TRUE_V_STAR.iter().cloned().fold(f32::INFINITY, f32::min);

    let preds: Vec<f64> = (0..8).map(|i| {
        let feat: Vec<f32> = STATE_FEATURES[i].to_vec();
        let t = Tensor::from_slice(&feat).view([1, 4]).to_kind(Kind::Float);
        let p = predict_scalar(linears, &t);
        p * (v_max - v_min) as f64 + v_min as f64
    }).collect();

    let true_vals: Vec<f64> = TRUE_V_STAR.iter().map(|&v| v as f64).collect();
    let y_mean = true_vals.iter().sum::<f64>() / 8.0;
    let ss_tot: f64 = true_vals.iter().map(|&y| (y - y_mean).powi(2)).sum();
    let ss_res: f64 = preds.iter().zip(true_vals.iter()).map(|(&p, &y)| (p - y).powi(2)).sum();
    if ss_tot > 1e-10 { 1.0 - ss_res / ss_tot } else { 0.0 }
}

// ── Public entry point ────────────────────────────────────────────────────────
pub fn run_ch17(config: Ch17Config) -> Ch17Result {
    let (_vs, linears) = build_and_train(&config);
    let mut rng = StdRng::seed_from_u64(config.seed);

    let v_max = TRUE_V_STAR.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let v_min = TRUE_V_STAR.iter().cloned().fold(f32::INFINITY, f32::min);

    let model_r2 = compute_r2(&linears);

    let mut state_explanations = Vec::new();
    let mut global_grad_fi = vec![0.0f64; 4];
    let mut global_saliency = vec![0.0f64; 4];
    let mut global_shap = vec![0.0f64; 4];

    for s in 0..8 {
        let feat: Vec<f32> = STATE_FEATURES[s].to_vec();
        let t = Tensor::from_slice(&feat).view([1, 4]).to_kind(Kind::Float);
        let pred_norm = predict_scalar(&linears, &t);
        let pred = pred_norm * (v_max - v_min) as f64 + v_min as f64;

        let grad_fi  = gradient_feature_importance(&linears, s);
        let saliency = saliency_map(&linears, s);
        let (lime_coefs, lime_intercept, lime_r2) =
            lime_explain(&linears, s, config.n_lime_samples, &mut rng);
        let (shap_vals, shap_base) =
            shap_values(&linears, s, config.n_shap_samples, &mut rng);

        for j in 0..4 {
            global_grad_fi[j] += grad_fi[j] / 8.0;
            global_saliency[j] += saliency[j] / 8.0;
            global_shap[j] += shap_vals[j].abs() / 8.0;
        }

        state_explanations.push(Ch17StateExplanation {
            state_idx: s,
            features: feat,
            prediction: pred,
            true_value: TRUE_V_STAR[s] as f64,
            gradient_fi: grad_fi,
            saliency,
            lime_coefs,
            lime_intercept,
            lime_r2,
            shap_values: shap_vals,
            shap_base,
        });
    }

    // Global importance = average of SHAP absolute values
    let global_importance = global_shap;

    Ch17Result {
        method: "Gradient FI + Saliency + LIME + SHAP".to_string(),
        feature_names: FEATURE_NAMES.iter().map(|s| s.to_string()).collect(),
        global_importance,
        state_explanations,
        model_r2,
    }
}

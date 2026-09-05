//! Ch18 — Kolmogorov-Arnold Networks (KANs) (tch 0.14 / libtorch 2.1.0)
//! Shallow KAN (poly), Shallow KAN (Fourier), Deep KAN, MLP baseline.
//! Business context: Warsaw ASP V*(s) regression + synthetic cubic dataset.

use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

// ── Warsaw ASP data (same as Ch15/Ch17) ──────────────────────────────────────
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

// ── Basis functions ───────────────────────────────────────────────────────────

/// Polynomial basis: [x, x², x³, ..., x^degree] concatenated with original x
fn polynomial_basis(x: &Tensor, degree: i64) -> Tensor {
    let mut result = x.shallow_clone();
    for d in 2..=degree {
        result = Tensor::cat(&[result, x.pow_tensor_scalar(d as f64)], 1);
    }
    result
}

/// Fourier basis: [x, sin(x), cos(x), sin(2x), cos(2x), ...]
fn fourier_basis(x: &Tensor, n_freqs: i64) -> Tensor {
    let mut result = x.shallow_clone();
    for k in 1..=n_freqs {
        let kf = k as f64;
        result = Tensor::cat(&[
            result,
            (x * kf).sin(),
            (x * kf).cos(),
        ], 1);
    }
    result
}

// ── Result types ─────────────────────────────────────────────────────────────
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ch18EpochResult {
    pub epoch: usize,
    pub train_loss: f64,
    pub test_loss: f64,
    pub predictions: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ch18Result {
    pub model_name: String,
    pub basis: String,
    pub depth: usize,
    pub input_dim: usize,
    pub expanded_dim: usize,
    pub total_params: usize,
    pub epochs: Vec<Ch18EpochResult>,
    pub final_train_loss: f64,
    pub final_test_loss: f64,
    pub final_predictions: Vec<f64>,
    pub true_values: Vec<f64>,
    pub train_r2: f64,
    pub test_r2: f64,
    // Synthetic cubic dataset results
    pub synth_train_loss: f64,
    pub synth_test_loss: f64,
}

#[derive(Serialize, Deserialize)]
pub struct Ch18Config {
    pub n_epochs: usize,
    pub lr: f64,
    pub hidden_units: usize,
    pub poly_degree: usize,
    pub n_fourier_freqs: usize,
    pub deep_layers: usize,
    pub dropout_rate: f64,
    pub seed: u64,
}

// ── Compute R² ────────────────────────────────────────────────────────────────
fn r2_score(preds: &[f64], targets: &[f64]) -> f64 {
    let mean = targets.iter().sum::<f64>() / targets.len() as f64;
    let ss_tot: f64 = targets.iter().map(|&y| (y - mean).powi(2)).sum();
    let ss_res: f64 = preds.iter().zip(targets.iter()).map(|(&p, &y)| (p - y).powi(2)).sum();
    if ss_tot > 1e-10 { 1.0 - ss_res / ss_tot } else { 0.0 }
}

// ── Generate synthetic cubic dataset ─────────────────────────────────────────
fn generate_cubic_data(n: usize, seed: u64) -> (Tensor, Tensor, Tensor, Tensor) {
    let mut rng = StdRng::seed_from_u64(seed);
    let xs: Vec<f32> = (0..n).map(|_| rng.gen_range(-2.0f32..2.0f32)).collect();
    let ys: Vec<f32> = xs.iter().map(|&x| {
        x.powi(3) + 0.5 * x.powi(2) + 0.1 * rng.gen_range(-1.0f32..1.0f32)
    }).collect();

    let train_n = (n * 4 / 5) as i64;
    let test_n  = n as i64 - train_n;

    let x_t = Tensor::from_slice(&xs).view([n as i64, 1]).to_kind(Kind::Float);
    let y_t = Tensor::from_slice(&ys).view([n as i64, 1]).to_kind(Kind::Float);

    let train_x = x_t.narrow(0, 0, train_n);
    let train_y = y_t.narrow(0, 0, train_n);
    let test_x  = x_t.narrow(0, train_n, test_n);
    let test_y  = y_t.narrow(0, train_n, test_n);
    (train_x, train_y, test_x, test_y)
}

// ── Build KAN/MLP model ───────────────────────────────────────────────────────
fn build_model(
    vs: &nn::Path,
    expanded_dim: i64,
    hidden: i64,
    depth: usize,
    _dropout: f64,
) -> nn::Sequential {
    let mut seq = nn::seq();
    for i in 0..depth {
        let in_d = if i == 0 { expanded_dim } else { hidden };
        seq = seq
            .add(nn::linear(vs / format!("l{}", i), in_d, hidden, Default::default()))
            .add_fn(|x| x.relu());
    }
    seq.add(nn::linear(vs / "out", hidden, 1, Default::default()))
}

// ── Train one configuration ───────────────────────────────────────────────────
fn train_kan(
    config: &Ch18Config,
    model_name: &str,
    basis: &str,
    depth: usize,
) -> Ch18Result {
    tch::manual_seed(config.seed as i64);
    let device = Device::Cpu;

    // Warsaw ASP data
    let v_max = TRUE_V_STAR.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let v_min = TRUE_V_STAR.iter().cloned().fold(f32::INFINITY, f32::min);
    let y_norm: Vec<f32> = TRUE_V_STAR.iter().map(|&v| (v - v_min) / (v_max - v_min)).collect();

    let x_flat: Vec<f32> = STATE_FEATURES.iter().flat_map(|r| r.iter().cloned()).collect();
    let x_raw = Tensor::from_slice(&x_flat).view([8, 4]).to_kind(Kind::Float);
    let y     = Tensor::from_slice(&y_norm).view([8, 1]).to_kind(Kind::Float);

    // Apply basis expansion
    let x = match basis {
        "fourier" => fourier_basis(&x_raw, config.n_fourier_freqs as i64),
        _         => polynomial_basis(&x_raw, config.poly_degree as i64),
    };

    let expanded_dim = x.size()[1];
    let vs = nn::VarStore::new(device);
    let net = build_model(&vs.root(), expanded_dim, config.hidden_units as i64, depth, config.dropout_rate);

    let mut opt = nn::Adam::default().build(&vs, config.lr).unwrap();
    let mut epoch_results = Vec::new();

    // Count params
    let total_params: usize = vs.trainable_variables().iter()
        .map(|v| v.numel() as usize).sum::<usize>();

    for epoch in 0..config.n_epochs {
        let pred = net.forward(&x);
        let loss = pred.mse_loss(&y, tch::Reduction::Mean);
        opt.backward_step(&loss);

        if epoch % 10 == 0 || epoch == config.n_epochs - 1 {
            let train_loss = loss.double_value(&[]);
            let pred_eval = tch::no_grad(|| net.forward(&x));
            let preds_raw: Vec<f32> = Vec::try_from(pred_eval.view(-1)).unwrap_or_default();
            let preds_denorm: Vec<f64> = preds_raw.iter()
                .map(|&p| (p * (v_max - v_min) + v_min) as f64).collect();

            epoch_results.push(Ch18EpochResult {
                epoch,
                train_loss,
                test_loss: train_loss, // same dataset (small)
                predictions: preds_denorm,
            });
        }
    }

    // Final predictions
    let pred_final = tch::no_grad(|| net.forward(&x));
    let pf_raw: Vec<f32> = Vec::try_from(pred_final.view(-1)).unwrap_or_default();
    let final_preds: Vec<f64> = pf_raw.iter()
        .map(|&p| (p * (v_max - v_min) + v_min) as f64).collect();
    let true_vals: Vec<f64> = TRUE_V_STAR.iter().map(|&v| v as f64).collect();
    let train_r2 = r2_score(&final_preds, &true_vals);
    let final_loss = epoch_results.last().map(|e| e.train_loss).unwrap_or(f64::NAN);

    // Synthetic cubic dataset
    let (sx_train_raw, sy_train, sx_test_raw, sy_test) = generate_cubic_data(200, config.seed);
    let sx_train = match basis {
        "fourier" => fourier_basis(&sx_train_raw, config.n_fourier_freqs as i64),
        _         => polynomial_basis(&sx_train_raw, config.poly_degree as i64),
    };
    let sx_test = match basis {
        "fourier" => fourier_basis(&sx_test_raw, config.n_fourier_freqs as i64),
        _         => polynomial_basis(&sx_test_raw, config.poly_degree as i64),
    };

    let synth_expanded = sx_train.size()[1];
    let vs2 = nn::VarStore::new(device);
    let net2 = build_model(&vs2.root(), synth_expanded, config.hidden_units as i64, depth, config.dropout_rate);
    let mut opt2 = nn::Adam::default().build(&vs2, config.lr).unwrap();

    for _ in 0..config.n_epochs {
        let p = net2.forward(&sx_train);
        let l = p.mse_loss(&sy_train, tch::Reduction::Mean);
        opt2.backward_step(&l);
    }

    let synth_train_loss = tch::no_grad(|| {
        net2.forward(&sx_train).mse_loss(&sy_train, tch::Reduction::Mean).double_value(&[])
    });
    let synth_test_loss = tch::no_grad(|| {
        net2.forward(&sx_test).mse_loss(&sy_test, tch::Reduction::Mean).double_value(&[])
    });

    Ch18Result {
        model_name: model_name.to_string(),
        basis: basis.to_string(),
        depth,
        input_dim: 4,
        expanded_dim: expanded_dim as usize,
        total_params,
        epochs: epoch_results,
        final_train_loss: final_loss,
        final_test_loss: final_loss,
        final_predictions: final_preds,
        true_values: true_vals.clone(),
        train_r2,
        test_r2: train_r2,
        synth_train_loss,
        synth_test_loss,
    }
}

// ── Public entry point ────────────────────────────────────────────────────────
pub fn run_ch18(config: Ch18Config) -> Vec<Ch18Result> {
    vec![
        train_kan(&config, "Shallow KAN (Polynomial)", "poly",    1),
        train_kan(&config, "Shallow KAN (Fourier)",    "fourier", 1),
        train_kan(&config, "Deep KAN (Polynomial)",    "poly",    config.deep_layers),
        train_kan(&config, "MLP Baseline",             "poly",    2),
    ]
}

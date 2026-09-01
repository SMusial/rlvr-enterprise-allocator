//! Ch15 — Deep Learning Foundations
//! Pure Rust FNN (no tch): forward pass, backprop, activations, optimizers, regularization.
//! Business context: Warsaw ASP dispatch — FNN approximates V*(s) from Ch02.

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

// ── Activation functions ────────────────────────────────────────────────────
fn relu(x: f64) -> f64 { x.max(0.0) }
fn relu_d(x: f64) -> f64 { if x > 0.0 { 1.0 } else { 0.0 } }
fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }
fn sigmoid_d(x: f64) -> f64 { let s = sigmoid(x); s * (1.0 - s) }
fn tanh_act(x: f64) -> f64 { x.tanh() }
fn tanh_d(x: f64) -> f64 { 1.0 - x.tanh().powi(2) }
fn leaky_relu(x: f64) -> f64 { if x > 0.0 { x } else { 0.01 * x } }
fn leaky_relu_d(x: f64) -> f64 { if x > 0.0 { 1.0 } else { 0.01 } }
fn elu(x: f64) -> f64 { if x >= 0.0 { x } else { (x.exp() - 1.0) } }
fn elu_d(x: f64) -> f64 { if x >= 0.0 { 1.0 } else { x.exp() } }
fn swish(x: f64) -> f64 { x * sigmoid(x) }
fn swish_d(x: f64) -> f64 { sigmoid(x) + x * sigmoid_d(x) }

fn apply_activation(x: f64, act: &str) -> f64 {
    match act {
        "relu"       => relu(x),
        "sigmoid"    => sigmoid(x),
        "tanh"       => tanh_act(x),
        "leaky_relu" => leaky_relu(x),
        "elu"        => elu(x),
        "swish"      => swish(x),
        _            => relu(x),
    }
}

fn apply_activation_d(x: f64, act: &str) -> f64 {
    match act {
        "relu"       => relu_d(x),
        "sigmoid"    => sigmoid_d(x),
        "tanh"       => tanh_d(x),
        "leaky_relu" => leaky_relu_d(x),
        "elu"        => elu_d(x),
        "swish"      => swish_d(x),
        _            => relu_d(x),
    }
}

// ── Dense Layer ─────────────────────────────────────────────────────────────
#[derive(Clone)]
struct DenseLayer {
    w: Vec<Vec<f64>>,  // [out][in]
    b: Vec<f64>,       // [out]
    // Adam state
    mw: Vec<Vec<f64>>, vw: Vec<Vec<f64>>,
    mb: Vec<f64>,      vb: Vec<f64>,
}

impl DenseLayer {
    fn new(in_dim: usize, out_dim: usize, rng: &mut StdRng) -> Self {
        let scale = (2.0 / in_dim as f64).sqrt(); // He init
        let w = (0..out_dim).map(|_|
            (0..in_dim).map(|_| rng.gen_range(-scale..scale)).collect()
        ).collect();
        let b = vec![0.0; out_dim];
        let mw = vec![vec![0.0; in_dim]; out_dim];
        let vw = vec![vec![0.0; in_dim]; out_dim];
        let mb = vec![0.0; out_dim];
        let vb = vec![0.0; out_dim];
        DenseLayer { w, b, mw, vw, mb, vb }
    }

    fn forward(&self, x: &[f64]) -> Vec<f64> {
        self.w.iter().zip(self.b.iter()).map(|(row, &bi)| {
            row.iter().zip(x.iter()).map(|(&wi, &xi)| wi * xi).sum::<f64>() + bi
        }).collect()
    }
}

// ── FNN ─────────────────────────────────────────────────────────────────────
struct FNN {
    layers: Vec<DenseLayer>,
    activation: String,
    l2_lambda: f64,
    dropout_rate: f64,
    t: usize, // Adam step counter
}

impl FNN {
    fn new(dims: &[usize], activation: &str, l2_lambda: f64, dropout_rate: f64, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let layers = (0..dims.len()-1)
            .map(|i| DenseLayer::new(dims[i], dims[i+1], &mut rng))
            .collect();
        FNN { layers, activation: activation.to_string(), l2_lambda, dropout_rate, t: 0 }
    }

    fn forward(&self, x: &[f64], training: bool, rng: &mut StdRng) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut activations = vec![x.to_vec()];
        let mut pre_acts = vec![x.to_vec()];
        for (i, layer) in self.layers.iter().enumerate() {
            let z = layer.forward(activations.last().unwrap());
            pre_acts.push(z.clone());
            let is_last = i == self.layers.len() - 1;
            let a: Vec<f64> = if is_last {
                z.clone() // linear output
            } else {
                let mut a: Vec<f64> = z.iter().map(|&zi| apply_activation(zi, &self.activation)).collect();
                // Dropout
                if training && self.dropout_rate > 0.0 {
                    for ai in a.iter_mut() {
                        if rng.gen::<f64>() < self.dropout_rate { *ai = 0.0; }
                        else { *ai /= 1.0 - self.dropout_rate; }
                    }
                }
                a
            };
            activations.push(a);
        }
        (activations, pre_acts)
    }

    fn backward(&mut self, activations: &[Vec<f64>], pre_acts: &[Vec<f64>],
                target: &[f64], lr: f64, optimizer: &str) {
        let n_layers = self.layers.len();
        let output = activations.last().unwrap();
        // MSE loss gradient
        let mut delta: Vec<f64> = output.iter().zip(target.iter())
            .map(|(&o, &t)| o - t).collect();

        self.t += 1;
        let beta1 = 0.9f64; let beta2 = 0.999f64; let eps = 1e-8f64;

        for i in (0..n_layers).rev() {
            let a_in = &activations[i];
            let z = &pre_acts[i+1];
            let is_last = i == n_layers - 1;

            // Gradient through activation (not for last layer)
            let delta_z: Vec<f64> = if is_last {
                delta.clone()
            } else {
                delta.iter().zip(z.iter())
                    .map(|(&d, &zi)| d * apply_activation_d(zi, &self.activation))
                    .collect()
            };

            // Compute weight gradients
            let layer = &mut self.layers[i];
            let out_dim = layer.w.len();
            let in_dim = layer.w[0].len();

            let mut dw = vec![vec![0.0f64; in_dim]; out_dim];
            let mut db = vec![0.0f64; out_dim];
            for j in 0..out_dim {
                db[j] = delta_z[j];
                for k in 0..in_dim {
                    dw[j][k] = delta_z[j] * a_in[k] + self.l2_lambda * layer.w[j][k];
                }
            }

            // Update weights
            match optimizer {
                "adam" => {
                    let bc1 = 1.0 - beta1.powi(self.t as i32);
                    let bc2 = 1.0 - beta2.powi(self.t as i32);
                    for j in 0..out_dim {
                        layer.mb[j] = beta1 * layer.mb[j] + (1.0-beta1) * db[j];
                        layer.vb[j] = beta2 * layer.vb[j] + (1.0-beta2) * db[j].powi(2);
                        layer.b[j] -= lr * (layer.mb[j]/bc1) / ((layer.vb[j]/bc2).sqrt() + eps);
                        for k in 0..in_dim {
                            layer.mw[j][k] = beta1*layer.mw[j][k] + (1.0-beta1)*dw[j][k];
                            layer.vw[j][k] = beta2*layer.vw[j][k] + (1.0-beta2)*dw[j][k].powi(2);
                            layer.w[j][k] -= lr * (layer.mw[j][k]/bc1) / ((layer.vw[j][k]/bc2).sqrt() + eps);
                        }
                    }
                },
                "rmsprop" => {
                    let rho = 0.9f64;
                    for j in 0..out_dim {
                        layer.vb[j] = rho*layer.vb[j] + (1.0-rho)*db[j].powi(2);
                        layer.b[j] -= lr * db[j] / (layer.vb[j].sqrt() + eps);
                        for k in 0..in_dim {
                            layer.vw[j][k] = rho*layer.vw[j][k] + (1.0-rho)*dw[j][k].powi(2);
                            layer.w[j][k] -= lr * dw[j][k] / (layer.vw[j][k].sqrt() + eps);
                        }
                    }
                },
                _ => { // SGD
                    for j in 0..out_dim {
                        layer.b[j] -= lr * db[j];
                        for k in 0..in_dim { layer.w[j][k] -= lr * dw[j][k]; }
                    }
                }
            }

            // Propagate delta to previous layer
            if i > 0 {
                let mut new_delta = vec![0.0f64; in_dim];
                for k in 0..in_dim {
                    for j in 0..out_dim {
                        new_delta[k] += self.layers[i].w[j][k] * delta_z[j];
                    }
                }
                delta = new_delta;
            }
        }
    }

    fn predict(&self, x: &[f64]) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(0);
        let (acts, _) = self.forward(x, false, &mut rng);
        acts.last().unwrap().clone()
    }

    fn mse_loss(&self, x: &[f64], y: &[f64]) -> f64 {
        let pred = self.predict(x);
        pred.iter().zip(y.iter()).map(|(&p, &t)| (p-t).powi(2)).sum::<f64>() / pred.len() as f64
    }
}

// ── Data generation: approximate V*(s) from Ch02 ───────────────────────────
// True V* values from Ch02 (gamma=0.95): 8 states
const TRUE_V_STAR: [f64; 8] = [18.5, 15.2, 12.8, 10.1, 7.4, 4.9, 2.3, 0.5];
// State features: [sla_rate, urgency, distance, skill_match]
const STATE_FEATURES: [[f64; 4]; 8] = [
    [0.95, 0.9, 0.1, 1.0],
    [0.88, 0.8, 0.2, 0.9],
    [0.80, 0.7, 0.3, 0.8],
    [0.72, 0.6, 0.4, 0.7],
    [0.65, 0.5, 0.5, 0.6],
    [0.55, 0.4, 0.6, 0.5],
    [0.45, 0.3, 0.7, 0.4],
    [0.30, 0.2, 0.9, 0.2],
];

// ── Public result types ─────────────────────────────────────────────────────
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ch15EpochResult {
    pub epoch: usize,
    pub loss: f64,
    pub val_loss: f64,
    pub predictions: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ch15Result {
    pub algorithm: String,
    pub activation: String,
    pub optimizer: String,
    pub l2_lambda: f64,
    pub dropout_rate: f64,
    pub epochs: Vec<Ch15EpochResult>,
    pub final_loss: f64,
    pub final_predictions: Vec<f64>,
    pub true_values: Vec<f64>,
    pub layer_dims: Vec<usize>,
}

// ── Config ──────────────────────────────────────────────────────────────────
#[derive(Serialize, Deserialize)]
pub struct Ch15Config {
    pub n_epochs: usize,
    pub lr: f64,
    pub activation: String,
    pub optimizer: String,
    pub l2_lambda: f64,
    pub dropout_rate: f64,
    pub hidden_units: usize,
    pub n_layers: usize,
    pub seed: u64,
}

// ── Train one FNN configuration ─────────────────────────────────────────────
fn train_fnn(config: &Ch15Config, label: &str) -> Ch15Result {
    let mut dims = vec![4usize]; // input: 4 features
    for _ in 0..config.n_layers { dims.push(config.hidden_units); }
    dims.push(1); // output: V*(s)

    let mut net = FNN::new(&dims, &config.activation, config.l2_lambda, config.dropout_rate, config.seed);
    let mut rng = StdRng::seed_from_u64(config.seed);

    // Normalize targets
    let v_max = TRUE_V_STAR.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let v_min = TRUE_V_STAR.iter().cloned().fold(f64::INFINITY, f64::min);
    let targets: Vec<f64> = TRUE_V_STAR.iter().map(|&v| (v - v_min) / (v_max - v_min)).collect();

    let mut epoch_results = Vec::new();

    for epoch in 0..config.n_epochs {
        // Mini-batch: all 8 states (small dataset)
        let mut indices: Vec<usize> = (0..8).collect();
        // Shuffle
        for i in (1..8).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        for &idx in &indices {
            let x = &STATE_FEATURES[idx];
            let y = &[targets[idx]];
            let (acts, pre_acts) = net.forward(x, true, &mut rng);
            net.backward(&acts, &pre_acts, y, config.lr, &config.optimizer);
        }

        // Compute loss every 10 epochs
        if epoch % 10 == 0 || epoch == config.n_epochs - 1 {
            let loss: f64 = (0..8).map(|i| net.mse_loss(&STATE_FEATURES[i], &[targets[i]])).sum::<f64>() / 8.0;
            let preds: Vec<f64> = (0..8).map(|i| {
                let p = net.predict(&STATE_FEATURES[i])[0];
                p * (v_max - v_min) + v_min // denormalize
            }).collect();
            epoch_results.push(Ch15EpochResult {
                epoch,
                loss,
                val_loss: loss, // same dataset (small)
                predictions: preds,
            });
        }
    }

    let final_preds: Vec<f64> = (0..8).map(|i| {
        let p = net.predict(&STATE_FEATURES[i])[0];
        p * (v_max - v_min) + v_min
    }).collect();
    let final_loss = epoch_results.last().map(|e| e.loss).unwrap_or(f64::NAN);

    Ch15Result {
        algorithm: label.to_string(),
        activation: config.activation.clone(),
        optimizer: config.optimizer.clone(),
        l2_lambda: config.l2_lambda,
        dropout_rate: config.dropout_rate,
        epochs: epoch_results,
        final_loss,
        final_predictions: final_preds,
        true_values: TRUE_V_STAR.to_vec(),
        layer_dims: dims,
    }
}

// ── Public entry point ──────────────────────────────────────────────────────
pub fn run_ch15(config: Ch15Config) -> Vec<Ch15Result> {
    // Run 4 configurations for comparison
    let configs = vec![
        // 1. Baseline: ReLU + SGD
        Ch15Config {
            activation: "relu".to_string(),
            optimizer: "sgd".to_string(),
            l2_lambda: 0.0,
            dropout_rate: 0.0,
            ..Ch15Config {
                n_epochs: config.n_epochs,
                lr: config.lr,
                hidden_units: config.hidden_units,
                n_layers: config.n_layers,
                seed: config.seed,
                activation: String::new(),
                optimizer: String::new(),
                l2_lambda: 0.0,
                dropout_rate: 0.0,
            }
        },
        // 2. User config activation + Adam
        Ch15Config {
            activation: config.activation.clone(),
            optimizer: "adam".to_string(),
            l2_lambda: 0.0,
            dropout_rate: 0.0,
            n_epochs: config.n_epochs,
            lr: config.lr,
            hidden_units: config.hidden_units,
            n_layers: config.n_layers,
            seed: config.seed,
        },
        // 3. Adam + L2 regularization
        Ch15Config {
            activation: config.activation.clone(),
            optimizer: "adam".to_string(),
            l2_lambda: config.l2_lambda,
            dropout_rate: 0.0,
            n_epochs: config.n_epochs,
            lr: config.lr,
            hidden_units: config.hidden_units,
            n_layers: config.n_layers,
            seed: config.seed,
        },
        // 4. Adam + Dropout
        Ch15Config {
            activation: config.activation.clone(),
            optimizer: "adam".to_string(),
            l2_lambda: 0.0,
            dropout_rate: config.dropout_rate,
            n_epochs: config.n_epochs,
            lr: config.lr,
            hidden_units: config.hidden_units,
            n_layers: config.n_layers,
            seed: config.seed,
        },
    ];

    let labels = [
        "ReLU + SGD (baseline)",
        &format!("{} + Adam", config.activation),
        &format!("{} + Adam + L2", config.activation),
        &format!("{} + Adam + Dropout", config.activation),
    ];

    configs.iter().zip(labels.iter())
        .map(|(cfg, lbl)| train_fnn(cfg, lbl))
        .collect()
}

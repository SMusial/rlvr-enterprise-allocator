//! Ch15 — Deep Learning Foundations (tch 0.14 / libtorch 2.1.0)
//! FNN with real PyTorch tensors: activations, optimizers, regularization.
//! Business context: Warsaw ASP — FNN approximates V*(s) from Ch02.

use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};
use serde::{Deserialize, Serialize};

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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ch15EpochResult {
    pub epoch: usize,
    pub loss: f64,
    pub val_loss: f64,
    pub predictions: Vec<f64>,
    pub grad_norm: f64,
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
    pub total_params: usize,
}

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

fn apply_act(x: &Tensor, act: &str) -> Tensor {
    match act {
        "relu"       => x.relu(),
        "leaky_relu" => x.leaky_relu(),
        "elu"        => x.elu(),
        "swish"      => x * x.sigmoid(),
        "tanh"       => x.tanh(),
        "sigmoid"    => x.sigmoid(),
        _            => x.relu(),
    }
}

fn count_params(dims: &[usize]) -> usize {
    (0..dims.len()-1).map(|i| dims[i]*dims[i+1] + dims[i+1]).sum()
}

fn tensor_to_f64(t: &Tensor) -> f64 {
    t.detach().double_value(&[])
}

fn tensor_to_vec_f32(t: &Tensor) -> Vec<f32> {
    Vec::<f32>::try_from(t.detach().contiguous().view(-1)).unwrap_or_default()
}

fn train_config(config: &Ch15Config, label: &str, opt_name: &str) -> Ch15Result {
    tch::manual_seed(config.seed as i64);
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let root = vs.root();

    let mut dims = vec![4usize];
    for _ in 0..config.n_layers { dims.push(config.hidden_units); }
    dims.push(1);

    let linears: Vec<nn::Linear> = (0..dims.len()-1)
        .map(|i| nn::linear(
            &root / format!("l{}", i),
            dims[i] as i64,
            dims[i+1] as i64,
            Default::default(),
        ))
        .collect();

    let act   = config.activation.clone();
    let dr    = config.dropout_rate;
    let n_lin = linears.len();

    let x_flat: Vec<f32> = STATE_FEATURES.iter().flat_map(|r| r.iter().cloned()).collect();
    let x = Tensor::from_slice(&x_flat).view([8, 4]).to_kind(Kind::Float);

    let v_max = TRUE_V_STAR.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let v_min = TRUE_V_STAR.iter().cloned().fold(f32::INFINITY,     f32::min);
    let y_norm: Vec<f32> = TRUE_V_STAR.iter().map(|&v| (v-v_min)/(v_max-v_min)).collect();
    let y = Tensor::from_slice(&y_norm).view([8, 1]).to_kind(Kind::Float);

    let mut opt = match opt_name {
        "adam"    => nn::Adam::default().build(&vs, config.lr).unwrap(),
        "rmsprop" => nn::RmsProp::default().build(&vs, config.lr).unwrap(),
        _         => nn::Sgd::default().build(&vs, config.lr).unwrap(),
    };

    let mut epoch_results = Vec::new();
    let total_params = count_params(&dims);

    for epoch in 0..config.n_epochs {
        let mut h = x.shallow_clone();
        for (i, lin) in linears.iter().enumerate() {
            h = lin.forward(&h);
            if i < n_lin - 1 {
                h = apply_act(&h, &act);
                if dr > 0.0 { h = h.dropout(dr, true); }
            }
        }

        let mut loss = h.mse_loss(&y, tch::Reduction::Mean);
        if config.l2_lambda > 0.0 {
            let l2 = vs.trainable_variables().iter()
                .map(|v| v.pow_tensor_scalar(2).sum(Kind::Float))
                .fold(Tensor::zeros(&[1], (Kind::Float, device)), |a, b| a + b);
            loss = &loss + config.l2_lambda * l2;
        }
        opt.backward_step(&loss);

        if epoch % 10 == 0 || epoch == config.n_epochs - 1 {
            let loss_val = tensor_to_f64(&loss);

            let pred_eval = tch::no_grad(|| {
                let mut h2 = x.shallow_clone();
                for (i, lin) in linears.iter().enumerate() {
                    h2 = lin.forward(&h2);
                    if i < n_lin - 1 { h2 = apply_act(&h2, &act); }
                }
                h2
            });
            let preds_raw = tensor_to_vec_f32(&pred_eval);
            let preds_denorm: Vec<f64> = preds_raw.iter()
                .map(|&p| (p * (v_max - v_min) + v_min) as f64)
                .collect();

            let grad_norm: f64 = vs.trainable_variables().iter()
                .map(|v| {
                    let g = v.grad();
                    if g.defined() { tensor_to_f64(&g.norm()) } else { 0.0 }
                })
                .sum();

            epoch_results.push(Ch15EpochResult {
                epoch, loss: loss_val, val_loss: loss_val,
                predictions: preds_denorm, grad_norm,
            });
        }
    }

    let pred_final = tch::no_grad(|| {
        let mut h = x.shallow_clone();
        for (i, lin) in linears.iter().enumerate() {
            h = lin.forward(&h);
            if i < n_lin - 1 { h = apply_act(&h, &act); }
        }
        h
    });
    let pf_raw = tensor_to_vec_f32(&pred_final);
    let final_preds: Vec<f64> = pf_raw.iter()
        .map(|&p| (p * (v_max - v_min) + v_min) as f64)
        .collect();
    let final_loss = epoch_results.last().map(|e| e.loss).unwrap_or(f64::NAN);

    Ch15Result {
        algorithm: label.to_string(),
        activation: config.activation.clone(),
        optimizer: opt_name.to_string(),
        l2_lambda: config.l2_lambda,
        dropout_rate: config.dropout_rate,
        epochs: epoch_results,
        final_loss, final_predictions: final_preds,
        true_values: TRUE_V_STAR.iter().map(|&v| v as f64).collect(),
        layer_dims: dims, total_params,
    }
}

pub fn run_ch15(config: Ch15Config) -> Vec<Ch15Result> {
    let act = config.activation.clone();
    vec![
        train_config(&Ch15Config {
            activation: "relu".to_string(), optimizer: "sgd".to_string(),
            l2_lambda: 0.0, dropout_rate: 0.0,
            n_epochs: config.n_epochs, lr: config.lr,
            hidden_units: config.hidden_units, n_layers: config.n_layers, seed: config.seed,
        }, "ReLU + SGD (baseline)", "sgd"),
        train_config(&Ch15Config {
            activation: act.clone(), optimizer: "adam".to_string(),
            l2_lambda: 0.0, dropout_rate: 0.0,
            n_epochs: config.n_epochs, lr: config.lr,
            hidden_units: config.hidden_units, n_layers: config.n_layers, seed: config.seed,
        }, &format!("{} + Adam", act), "adam"),
        train_config(&Ch15Config {
            activation: act.clone(), optimizer: "adam".to_string(),
            l2_lambda: config.l2_lambda, dropout_rate: 0.0,
            n_epochs: config.n_epochs, lr: config.lr,
            hidden_units: config.hidden_units, n_layers: config.n_layers, seed: config.seed,
        }, &format!("{} + Adam + L2", act), "adam"),
        train_config(&Ch15Config {
            activation: act.clone(), optimizer: "adam".to_string(),
            l2_lambda: 0.0, dropout_rate: config.dropout_rate,
            n_epochs: config.n_epochs, lr: config.lr,
            hidden_units: config.hidden_units, n_layers: config.n_layers, seed: config.seed,
        }, &format!("{} + Adam + Dropout", act), "adam"),
    ]
}

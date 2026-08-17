# RLVR Enterprise Allocator

> **Reinforcement Learning via Rust** — A 20-chapter, end-to-end framework for learning and demonstrating RL algorithms through a real enterprise field-service optimisation use case.

[![Rust](https://img.shields.io/badge/Rust-1.75+-orange?logo=rust)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Progress](https://img.shields.io/badge/Progress-Ch01--07%20%E2%9C%85%20%7C%20Ch08--20%20%F0%9F%9A%A7-lightgrey)]()

---

## Progress

| Chapter | Topic | Algorithm | Status |
|---------|-------|-----------|--------|
| **01** | Introduction to RL | MDP, epsilon-greedy, Gt | ✅ Complete |
| **02** | Discrete MDP & Bellman | Value Iteration, nalgebra LU | ✅ Complete |
| **03** | Multi-Armed Bandit | UCB1, Thompson Sampling | ✅ Complete |
| **04** | Dynamic Programming | Policy Iteration, Async VI | ✅ Complete |
| **05** | Monte Carlo Methods | First-Visit, Every-Visit, On/Off-Policy | ✅ Complete |
| **06** | Temporal Difference | TD(0), SARSA, Q-Learning | ✅ Complete |
| **07** | n-Step TD & Planning | n-Step TD, n-Step SARSA, Dyna-Q, Dyna-Q+ | ✅ Complete |
| 08 | Eligibility Traces | TD(λ), SARSA(λ), Q(λ) Watkins | ✅ Complete |
| 09 | Policy Gradient | REINFORCE, Softmax, Actor-Critic | ✅ Complete |
| 10 | Model-Based RL | World Model, Prioritised Sweeping, MBPO, Uncertainty | Complete |
| 11 | Multi-Agent RL | Independent Q-Learning | 🚧 Planned |
| 12 | Game Theory & Nash | Nash Equilibrium | 🚧 Planned |
| 13 | Cooperative MARL | VDN, QMIX | 🚧 Planned |
| 14 | Learning Dynamics | ELO, Fictitious Play | 🚧 Planned |
| 15 | Deep RL (DQN) | DQN + burn tensors | 🚧 Planned |
| 16 | Actor-Critic | A2C, PPO | 🚧 Planned |
| 17 | Advanced Policy Opt. | SAC, DDPG | 🚧 Planned |
| 18 | QMIX Deep | Monotonic Mixing | 🚧 Planned |
| 19 | Federated RL | Differential Privacy | 🚧 Planned |
| 20 | PyO3 Interop & Safety | FFI Safety Invariants | 🚧 Planned |

---

## Quick Start

```bash
cd rlvr-py && maturin develop && cd ..
cargo test --workspace   # 112 tests passing
streamlit run gui/app.py
```

---

## Chapter 07 — n-Step TD & Planning

- **n-Step TD**: G^(n)_t bridges TD(0) (n=1) and MC (n=∞). Sweet spot n=3-5.
- **n-Step SARSA**: on-policy n-step control with epsilon-soft policy
- **Dyna-Q**: Q-Learning + model learning + k planning steps per real step
- **Dyna-Q+**: Dyna-Q + exploration bonus κ√τ(s,a) for stale transitions
- Returns curve, TD error, model coverage, Q-table heatmap
- Glass-Box: n-step return formula and Dyna planning trace
- 4 languages: EN / FR / ES / PL

---


## Chapter 08 — Eligibility Traces & TD(λ)

- **SARSA(λ)**: on-policy backward-view TD(λ) with eligibility traces
- **Q(λ) Watkins**: off-policy TD(λ) with trace cutting on non-greedy actions
- **λ=0 baseline**: equivalent to TD(0) / SARSA (Ch06)
- **λ=0.99 baseline**: approximates Monte Carlo (Ch05)
- Sparse HashMap traces, auto-pruned at 1e-8, Replacing & Accumulating variants
- Returns curve, TD error, active traces, Q-table heatmap, Glass-Box
- 4 languages: EN / FR / ES / PL
- 14 inline tests: smoke, shape, convergence, determinism, Watkins cut

## Chapter 09 — Policy Gradient: REINFORCE & Softmax

- **REINFORCE**: Monte Carlo policy gradient (Williams, 1992)
- **REINFORCE + Baseline**: variance-reduced with state-value baseline
- **Softmax Actor-Critic**: TD(0) critic + softmax actor, online per-step updates
- **REINFORCE τ=0.5**: sharper policy via lower softmax temperature
- Policy: π(a|s) = softmax(θ[s][a] / τ), no Q-table
- Returns, PG magnitude, policy entropy, θ heatmap, Glass-Box
- 4 languages: EN / FR / ES / PL
- 15 inline tests: smoke, shape, softmax correctness, baseline variance reduction, convergence

## Architecture

```
rlvr-core/src/
  ch01..ch10_world_model.rs        <- ALL RL logic in Rust
  rlvr-py/src/lib.rs              <- PyO3 bridge Ch01-Ch09
gui/chapters/
  ch01..ch09.py                   <- Streamlit UI only
```

## License

MIT

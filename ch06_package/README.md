# RLVR Enterprise Allocator

> **Reinforcement Learning via Rust** — A 20-chapter, end-to-end framework for learning and demonstrating RL algorithms through a real enterprise field-service optimisation use case.

[![Rust](https://img.shields.io/badge/Rust-1.75+-orange?logo=rust)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Progress](https://img.shields.io/badge/Progress-Ch01--06%20%E2%9C%85%20%7C%20Ch07--20%20%F0%9F%9A%A7-lightgrey)]()

---

## What is this?

This project implements **all 20 chapters** of the [Reinforcement Learning via Rust](https://rlvr.rantai.dev/docs/reinforcement-learning-via-rust/) book as a fully working, interactive demo application built to learn RL from first principles while simultaneously building production-grade Rust infrastructure.

**The business scenario:** A field-service dispatch system for the Warsaw region. An RL agent learns to assign technicians to work orders, optimising SLA compliance, skill matching, and travel distance.

**The architecture principle:** All RL logic lives in Rust. Python is used exclusively for UI rendering.

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
| 07 | n-Step TD & Planning | Dyna-Q | 🚧 Planned |
| 08 | Eligibility Traces | TD(lambda), SARSA(lambda) | 🚧 Planned |
| 09 | Policy Gradient | REINFORCE, Softmax | 🚧 Planned |
| 10 | Model-Based RL | World Models | 🚧 Planned |
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

## Repository Structure

```
rlvr-enterprise-allocator/
├── Cargo.toml
├── rlvr-core/src/
│   ├── ch01_asp_dispatch.rs   # MDP, epsilon-greedy, Gt
│   ├── ch02_bellman.rs        # Value Iteration, nalgebra LU
│   ├── ch03_bandit.rs         # UCB1, Thompson Sampling
│   ├── ch04_dp.rs             # Policy Iteration, Async VI
│   ├── ch05_mc.rs             # Monte Carlo: First/Every-Visit, On/Off-Policy
│   └── ch06_td.rs             # TD(0), SARSA, Q-Learning
├── rlvr-py/src/lib.rs         # PyO3 bridge (Ch01-Ch06)
└── gui/chapters/
    ├── ch01.py  ch02.py  ch03.py  ch04.py  ch05.py  ch06.py
```

---

## Quick Start

```bash
cd rlvr-py && maturin develop && cd ..
cargo test --workspace   # 59 tests passing
streamlit run gui/app.py
```

---

## Chapter Summaries

### Ch01-Ch05 — see previous releases

### Ch06 — Temporal Difference Learning
- **TD(0)**: online V(s) prediction — updates every step, not episode end
- **SARSA**: on-policy TD control — safe during learning, accounts for exploration
- **Q-Learning**: off-policy TD control — directly learns Q*, optimal greedy policy
- TD error curve, returns curve, Q-table heatmap, SARSA vs Q-Learning policy diff
- Glass-Box: exact TD update equation per episode with δ_t values
- 4 languages: EN / FR / ES / PL

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| RL Engine | Rust 1.75+ | All algorithms |
| Arrays | ndarray v0.16 | Q-tables, matrices |
| Linear algebra | nalgebra v0.33 | LU decomposition |
| RNG | rand v0.8 + StdRng | Deterministic |
| FFI | pyo3 v0.21 + maturin | Rust -> Python |
| UI | Streamlit + Plotly | Rendering only |

---

## License

MIT -- see [LICENSE](LICENSE)

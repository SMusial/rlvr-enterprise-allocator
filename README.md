# RLVR Enterprise Allocator

> **Reinforcement Learning via Rust** — A 20-chapter, end-to-end framework for learning and demonstrating RL algorithms through a real enterprise field-service optimisation use case.

[![Rust](https://img.shields.io/badge/Rust-1.75+-orange?logo=rust)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Chapter](https://img.shields.io/badge/Progress-Ch01%20%E2%9C%85%20%7C%20Ch02--20%20%F0%9F%9A%A7-lightgrey)]()

---

## What is this?

This project implements **all 20 chapters** of the [Reinforcement Learning via Rust](https://rlvr.rantai.dev/docs/reinforcement-learning-via-rust/) book as a fully working, interactive demo application — built to learn RL from first principles while simultaneously building production-grade Rust infrastructure.

**The business scenario:** A field-service dispatch system for the Warsaw region. An RL agent learns to assign technicians to work orders, optimising SLA compliance, skill matching, and travel distance — chapter by chapter, algorithm by algorithm.

**The architecture principle:** All RL logic lives in Rust. Python is used exclusively for UI rendering.

```
rlvr-core  (Rust)   ← ALL RL algorithms, environments, math
     ↓ PyO3
rlvr-py    (Bridge) ← zero-copy FFI, serialises Rust structs → Python dicts
     ↓ json
gui/       (Python) ← Streamlit UI only — renders data, zero RL logic
```

---

## Progress

| Chapter | Topic | Algorithm | Status |
|---------|-------|-----------|--------|
| **01** | Introduction to RL | ε-greedy, MDP, Gₜ | ✅ Complete |
| 02 | Discrete MDP & Bellman | Q-Learning | 🚧 Planned |
| 03 | Multi-Armed Bandit | UCB, Thompson Sampling | 🚧 Planned |
| 04 | Dynamic Programming | Value/Policy Iteration | 🚧 Planned |
| 05 | Monte Carlo Methods | MC Control, Off-policy | 🚧 Planned |
| 06 | Temporal Difference | SARSA, Q-Learning | 🚧 Planned |
| 07 | n-Step TD & Planning | Dyna-Q | 🚧 Planned |
| 08 | Eligibility Traces | TD(λ), SARSA(λ) | 🚧 Planned |
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
├── Cargo.toml                          # Workspace — 4 Rust crates
│
├── rlvr-core/                          # ALL RL logic (Rust)
│   └── src/
│       ├── lib.rs
│       ├── types.rs                    # Shared MDP types
│       ├── rng.rs                      # Seedable StdRng
│       ├── ch01_asp_dispatch.rs        # Ch01: MDP, ε-greedy, Gₜ
│       └── samplers/                   # Synthetic data generators
│
├── rlvr-verify/                        # Safety invariant harness (Rust)
│   └── src/invariants.rs              # TDD tests for all 20 chapters
│
├── rlvr-telemetry/                     # Lock-free ring buffer (Rust)
│   └── src/
│       ├── events.rs
│       └── ring_buffer.rs
│
├── rlvr-py/                            # PyO3 bridge (Rust → Python)
│   ├── pyproject.toml
│   └── src/lib.rs
│
└── gui/                                # Streamlit UI (Python — no RL logic)
    ├── app.py                          # Router + language selector
    ├── requirements.txt
    └── chapters/
        └── ch01.py                     # Ch01 UI: map, glass-box, theory
```

---

## Quick Start

### Prerequisites

```bash
# 1 — Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 2 — Python 3.10+ virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3 — Python dependencies
pip install -r gui/requirements.txt

# 4 — maturin (Rust → Python bridge builder)
pip install maturin
```

### Build & Run

```bash
# Build Rust engine + PyO3 bridge
cd rlvr-py && maturin develop && cd ..

# Run tests
cargo test --workspace

# Launch UI
streamlit run gui/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### After every Rust code change

```bash
cd rlvr-py && maturin develop && cd ..
cargo test --workspace
```

---

## Chapter 01 — What you'll see

The UI demonstrates the full **MDP framework** applied to field-service dispatch:

- **Warsaw map** — technicians (blue) and work orders (amber/red) on real OpenStreetMap tiles
- **ε-greedy policy** — move the ε slider to control exploration vs exploitation
- **Glass-Box inspector** — every step shows the full MDP tuple `(Sₜ, Aₜ, Rₜ, Gₜ, ε)`
- **Learning curve** — Gₜ over N episodes
- **Episode summary** — quantified business results + pros/cons of the algorithm
- **Theory panel** — 5 collapsible sections with math, code references, and book citations
- **4 languages** — EN / FR / ES / PL, full translation including theory

**Key concepts demonstrated:**
- MDP tuple `(S, A, P, R, γ)` — implemented in `ch01_asp_dispatch.rs`
- Markov property — `transition()` depends only on current state
- ε-greedy policy — `epsilon_greedy()` with exploration/exploitation toggle
- Discounted return `Gₜ = Σ γᵏ Rₜ₊ₖ` — computed backward from episode end
- `ndarray::Array2<f64>` Q-table — all zeros in Ch01, trained from Ch02

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| RL Engine | Rust 1.75+ | All algorithms, environments, math |
| Arrays (tabular) | `ndarray` v0.16 | Q-tables, state matrices (Ch01–Ch14) |
| Linear algebra | `nalgebra` v0.33 | Matrix ops, decompositions |
| Deep RL tensors | `burn` v0.21 | Neural networks, autodiff (Ch15–Ch20) |
| RNG | `rand` v0.8 + `StdRng` | Deterministic, seedable |
| FFI Bridge | `pyo3` v0.21 + `maturin` | Zero-copy Rust → Python |
| UI | Streamlit + Plotly | Rendering only — no RL logic |
| Maps | Plotly Scattermapbox | OpenStreetMap, no token needed |

---

## Design Principles

1. **Rust owns all computation** — Python never implements RL logic
2. **Deterministic by default** — every run is seeded and reproducible
3. **TDD from day one** — safety invariants written before algorithms
4. **Vertical slices** — each chapter is complete: Rust → bridge → UI → tests
5. **Glass-box transparency** — every algorithm decision is inspectable in the UI

---

## Book Reference

This project follows the [Reinforcement Learning via Rust](https://rlvr.rantai.dev/docs/reinforcement-learning-via-rust/) curriculum:

- **Part I** (Ch01–Ch04): Foundations — MDP, Bandits, Dynamic Programming
- **Part II** (Ch05–Ch10): Algorithms — MC, TD, Policy Gradient, Model-Based
- **Part III** (Ch11–Ch14): Multi-Agent RL — MARL, Game Theory
- **Part IV** (Ch15–Ch20): Deep RL — DQN, Actor-Critic, Federated, PyO3 Safety

---

## License

MIT — see [LICENSE](LICENSE)

# RLVR Enterprise Allocator

**Reinforcement Learning via Rust** — A 20-chapter, end-to-end framework for learning and demonstrating RL algorithms through a real enterprise field-service optimisation use case.

> **Warsaw ASP (After-Sales Point):** 5 technicians · up to 20 work orders per shift · 3 skills (HVAC, Electrical, Network) · SLA constraints · real Warsaw map coordinates · technicians move after each dispatch

---

## 🚀 Quick Start

```bash
# Build Rust engine
cd rlvr-py && maturin develop --release && cd ..

# Run Streamlit app
streamlit run gui/app.py --server.port 8001
```

---

## 📚 Chapter Progress

| Ch | Topic | Algorithm | Status |
|----|-------|-----------|--------|
| 01 | MDP Baseline & Random Policy | ε-greedy (ε=1), Gₜ, Monte Carlo return, Q=0 | ✅ Complete |
| 02 | Bellman Equation & Value Iteration | Value Iteration, nalgebra LU solver | ✅ Complete |
| 03 | Multi-Armed Bandit | UCB1, Thompson Sampling, ε-greedy | ✅ Complete |
| 04 | Dynamic Programming | Policy Iteration, Async Value Iteration | ✅ Complete |
| 05 | Monte Carlo Methods | First-Visit MC, Every-Visit MC, On/Off-Policy | ✅ Complete |
| 06 | Temporal Difference | TD(0), SARSA, Q-Learning | ✅ Complete |
| 07 | n-Step TD & Planning | n-Step TD, n-Step SARSA, Dyna-Q, Dyna-Q+ | ✅ Complete |
| 08 | Eligibility Traces | TD(λ), SARSA(λ), Replacing & Accumulating traces | ✅ Complete |
| 09 | Policy Gradient | REINFORCE, Softmax policy, Baseline | ✅ Complete |
| 10 | Model-Based RL | World Models, Dyna architecture | ✅ Complete |
| 11 | Multi-Agent RL (intro) | Independent Q-Learning (legacy) | ✅ Complete |
| 12 | Game Theory & Nash | Nash Equilibrium, Zero-sum games | ✅ Complete |
| 13 | Cooperative MARL | VDN, QMIX, Centralised training | ✅ Complete |
| 14 | Learning Dynamics | ELO rating, Fictitious Play | ✅ Complete |
| 15 | Deep RL — DQN | DQN, Experience Replay, Target Network (burn) | ✅ Complete |
| 16 | Actor-Critic | A2C, PPO, Advantage estimation | ✅ Complete |
| **17** | **MARL: Independent Q-Learning** | **IQL — 5 independent agents, individual rewards, no communication** | ✅ **Complete** |
| 18 | QMIX Deep MARL | Monotonic mixing network, centralised critic | 🔄 Planned |
| 19 | Scalable Deep RL | Distributed training, Federated RL | 🔄 Planned |
| 20 | PyO3 Interop & Safety | FFI safety invariants, Rust↔Python bridge | 🔄 Planned |

---

## 🤖 Chapter 17 — MARL: Independent Q-Learning (IQL)

Multi-Agent Reinforcement Learning applied to Warsaw ASP dispatch optimisation.

### Design

| Property | Value |
|----------|-------|
| Agents | 5 technicians — each is an independent RL agent |
| Q-table | One per agent: `Q[tech][order]` |
| Reward | Individual — each agent learns only from its own dispatches |
| Communication | None — agents do not observe each other |
| Policy | ε-greedy with linear decay: ε_start → ε_end |
| Update | IQL: `Q(t,o) ← Q(t,o) + α[R + γ·max Q(t,o') − Q(t,o)]` |
| Environment | Fixed (seed=42) — same positions every episode |
| Skill bias | 80% probability of selecting a skill-matching technician |

### IQL Update Rule

```
δ = R + γ · max_{o'} Q(tech, o') − Q(tech, order)
Q(tech, order) ← Q(tech, order) + α · δ
```

### Interactive Lab Features

- 🎬 **Episode selector** — all charts update for selected episode
- 🔍 **Step slider** — map, reward chart, TD error, glass-box all update
- 🗺️ **Warsaw Dispatch Map** — each agent has its own color
- 📊 **Reward per Step Chart** — per-step rewards with selected step highlighted
- 📉 **TD Error per Step** — convergence indicator (→ 0 as agent learns)
- 🔬 **Glass-Box MDP Trace** — Q before/after, TD error, skill match per step
- 📋 **Episode Summary** — team SLA rate, skill match %, avg distance
- 🤖 **Agent Stats Table** — per-agent SLA rate, skill match, orders served
- 📊 **Per-Agent Learning Curves (MA-5)** — each agent's independent learning progress
- 📈 **Team Learning Curve (MA-5)** — overall team performance
- 🧮 **Q-Table Heatmap** — final Q-values: agents × work orders

---

## 🏗️ Architecture

```
rlvr-enterprise-allocator/
├── rlvr-core/src/
│   ├── ch01_asp_dispatch.rs    ← Ch01 MDP baseline (random policy)
│   ├── ch02_*.rs               ← Value Iteration
│   ├── ch03_*.rs               ← Multi-Armed Bandit
│   ├── ch04_*.rs               ← Dynamic Programming
│   ├── ch05_*.rs               ← Monte Carlo
│   ├── ch06_*.rs               ← Temporal Difference
│   ├── ch07_*.rs               ← n-Step TD & Dyna
│   ├── ch08_*.rs               ← Eligibility Traces
│   ├── ch09_*.rs               ← Policy Gradient
│   ├── ch10_*.rs               ← World Models
│   ├── ch11_*.rs               ← Multi-Agent (intro)
│   ├── ch12_*.rs               ← Game Theory
│   ├── ch13_*.rs               ← Cooperative MARL
│   ├── ch14_*.rs               ← Learning Dynamics
│   ├── ch15_*.rs               ← DQN
│   ├── ch16_*.rs               ← Actor-Critic
│   └── ch17_marl.rs            ← Ch17 MARL IQL ← NEW
│
├── rlvr-py/src/lib.rs          ← PyO3 bridge (all chapters)
│
├── gui/
│   ├── app.py                  ← Streamlit router
│   └── chapters/
│       ├── ch01.py             ← Ch01 Interactive Lab
│       ├── ch02.py .. ch16.py  ← Ch02–Ch16 Interactive Labs
│       └── ch17.py             ← Ch17 Interactive Lab ← NEW
│
└── docs/
    ├── handson_ch01_en.html    ← Ch01 Hands-On Guide
    └── handson_ch*.html        ← Per-chapter guides
```

---

## 🧪 Warsaw ASP Use Case

The same business problem runs through all 20 chapters — enabling direct comparison of algorithms:

```
Environment (fixed, seed=42):
  - 5 technicians: T0–T4, skills: HVAC / Electrical / Network
  - 10 work orders: W0–W9, each requiring a specific skill
  - Warsaw coordinates: lon ∈ [20.90, 21.10], lat ∈ [52.18, 52.32]
  - Technicians move after each dispatch (realistic field service)

Reward function:
  R = +2.0 − 0.05·d   if SLA met (skill match + within distance threshold)
  R = +0.5 − 0.05·d   if skill match, SLA breached
  R = −1.0 − 0.02·d   if skill mismatch

Baseline (Ch01):  G₀ ≈ random, flat learning curve
Target (Ch02+):   G₀ > baseline, upward learning curve
```

---

## 📦 Dependencies

| Crate | Purpose |
|-------|---------|
| `rand` | Random number generation (StdRng, seeded) |
| `serde` / `serde_json` | Serialisation to Python |
| `pyo3` / `maturin` | Rust↔Python bridge |
| `nalgebra` | Linear algebra (Ch02 Value Iteration) |
| `burn` | Deep learning tensors (Ch15 DQN) |
| `streamlit` | Interactive UI |
| `plotly` / `altair` | Charts and visualisations |

---

## 📄 License

MIT — see [LICENSE](LICENSE)

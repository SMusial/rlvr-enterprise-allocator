# RLVR Enterprise Allocator

**Reinforcement Learning via Rust** — A 20-chapter, end-to-end framework for learning
and demonstrating RL algorithms through a real enterprise field-service optimisation use case.

---

## Progress

| Chapter | Topic | Algorithm | Status |
|---|---|---|---|
| 01 | Introduction to RL | MDP, epsilon-greedy, Gt | ✅ Complete |
| 02 | Discrete MDP & Bellman | Value Iteration, nalgebra LU | ✅ Complete |
| 03 | Multi-Armed Bandit | UCB1, Thompson Sampling | ✅ Complete |
| 04 | Dynamic Programming | Policy Iteration, Async VI | ✅ Complete |
| 05 | Monte Carlo Methods | First-Visit, Every-Visit, On/Off-Policy | ✅ Complete |
| 06 | Temporal Difference | TD(0), SARSA, Q-Learning | ✅ Complete |
| 07 | n-Step TD & Planning | n-Step TD, n-Step SARSA, Dyna-Q, Dyna-Q+ | ✅ Complete |
| 08 | Eligibility Traces | TD(λ), SARSA(λ), Q(λ) Watkins | ✅ Complete |
| 09 | Policy Gradient | REINFORCE, Softmax, Actor-Critic | ✅ Complete |
| 10 | Model-Based RL | World Model, Prioritised Sweeping, MBPO, Uncertainty | ✅ Complete |
| 11 | Multi-Agent RL | IQL, JAL, Lenient Q, Mean Field Q | Complete |
| 12 | Game Theory & Nash | Nash Equilibrium | Planned |
| 13 | Cooperative MARL | VDN, QMIX | Planned |
| 14 | Learning Dynamics | ELO, Fictitious Play | Planned |
| 15 | Deep RL (DQN) | DQN + burn tensors | Planned |
| 16 | Actor-Critic | A2C, PPO | Planned |
| 17 | Advanced Policy Opt. | SAC, DDPG | Planned |
| 18 | QMIX Deep | Monotonic Mixing | Planned |
| 19 | Federated RL | Differential Privacy | Planned |
| 20 | PyO3 Interop & Safety | FFI Safety Invariants | Planned |

---

## Quick Start

```bash
cd rlvr-py && maturin develop && cd ..
cargo test --workspace   # 127 tests passing
streamlit run gui/app.py
```

---

## Chapter 07 — n-Step TD & Planning

- **n-Step TD**: G^(n)_t bridges TD(0) (n=1) and MC (n=inf). Sweet spot n=3-5
- **n-Step SARSA**: on-policy n-step control with epsilon-soft policy
- **Dyna-Q**: Q-Learning + model learning + k planning steps per real step
- **Dyna-Q+**: Dyna-Q + exploration bonus for stale (s,a) transitions
- Returns curve, TD error, model coverage, Q-table heatmap, Glass-Box
- 4 languages: EN / FR / ES / PL | 14 inline tests

## Chapter 08 — Eligibility Traces & TD(lambda)

- **SARSA(lambda)**: on-policy backward-view TD(lambda) with eligibility traces
- **Q(lambda) Watkins**: off-policy TD(lambda) — traces cut on non-greedy actions
- **lambda=0 baseline**: equivalent to TD(0) / SARSA (Ch06)
- **lambda=0.99 baseline**: approximates Monte Carlo (Ch05)
- Sparse HashMap traces, auto-pruned at 1e-8 | Replacing & Accumulating variants
- Returns, TD error, active traces, Q-table heatmap, Glass-Box
- 4 languages: EN / FR / ES / PL | 14 inline tests

## Chapter 09 — Policy Gradient: REINFORCE & Softmax

- **REINFORCE**: Monte Carlo policy gradient (Williams, 1992)
- **REINFORCE + Baseline**: variance-reduced with state-value baseline
- **Softmax Actor-Critic**: TD(0) critic + softmax actor, online per-step updates
- **REINFORCE tau=0.5**: sharper policy via lower softmax temperature
- Policy: pi(a|s) = softmax(theta[s][a] / tau) — no Q-table
- Returns, PG magnitude, policy entropy, theta heatmap, Glass-Box
- 4 languages: EN / FR / ES / PL | 15 inline tests

## Chapter 10 — Model-Based RL: World Models

- **WM Q-Learning**: Dyna-Q with explicit tabular WorldModel — T(s,a,s') + R(s,a)
- **Prioritised Sweeping**: BinaryHeap of (|delta|, s, a) — plan from highest priority, propagate to predecessors
- **MBPO (lite)**: synthetic REINFORCE rollouts on learned model — bridges Ch09 + Ch10
- **Uncertainty Bonus**: Q_bonus(s,a) = Q(s,a) + beta/sqrt(N(s,a)+1) — UCB-style exploration
- Model accuracy curve, planning steps used, model size tracking
- Returns, TD error, model accuracy, planning steps, Q-table heatmap, Glass-Box
- 4 languages: EN / FR / ES / PL | 15 inline tests

---

## Architecture

```
rlvr-core/src/
  ch01..ch11_multiagent.rs         <- ALL RL logic in Rust
  rlvr-py/src/lib.rs              <- PyO3 bridge Ch01-Ch11
gui/chapters/
  ch01..ch11.py                   <- Streamlit UI only
```

---

## License

MIT

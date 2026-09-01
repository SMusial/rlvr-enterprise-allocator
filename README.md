# 🦀 RLVR Enterprise Allocator

> **Reinforcement Learning via Rust** — A complete, interactive RL curriculum built on a high-performance Rust engine with a Streamlit GUI.  
> Business context: Warsaw ASP (After-Sales Point) dispatch optimisation — 5 technicians, up to 20 orders per shift.

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| RL Engine | Rust (`rlvr-core`) — compiled to native Python extension via `maturin` |
| Python Bindings | `rlvr-py` (PyO3) |
| GUI | Streamlit |
| Math rendering | KaTeX (in-browser, HTML guides) |
| Charts | Altair, Plotly |

---

## 🚀 Quick Start

```bash
git clone https://github.com/SMusial/rlvr-enterprise-allocator.git
cd rlvr-enterprise-allocator

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd rlvr-py && maturin develop --release && cd ..

streamlit run gui/app.py --server.port 8001
```

---

## ✅ Chapter Progress

| # | Chapter | Topic | Algorithms | Status |
|---|---|---|---|---|
| 01 | MDP Baseline | ε-Greedy Dispatch | ε-greedy, Q=0 | ✅ |
| 02 | Bellman / Value Iteration | Optimal dispatch policy | Value Iteration, Bellman equation | ✅ |
| 03 | Multi-Armed Bandit | Skill-slot optimisation | ε-Greedy, UCB1, Thompson Sampling | ✅ |
| 04 | Dynamic Programming | Policy & Value Iteration | PI, VI, Async VI | ✅ |
| 05 | Monte Carlo | Model-free learning from episodes | First-Visit MC, Every-Visit MC, On-Policy, Off-Policy IS | ✅ |
| 06 | Temporal Difference | Online learning | TD(0), SARSA, Q-Learning | ✅ |
| 07 | n-Step TD & Planning | Sample efficiency | n-Step TD, n-Step SARSA, Dyna-Q, Dyna-Q+ | ✅ |
| 08 | Eligibility Traces | Backward credit assignment | SARSA(λ), Q(λ) Watkins | ✅ |
| 09 | Policy Gradient | Direct policy optimisation | REINFORCE, REINFORCE+Baseline, Actor-Critic TD(0) | ✅ |
| 10 | World Models | Model-based RL | WM Q-Learning, Prioritised Sweeping, MBPO, Uncertainty Bonus | ✅ |
| 11 | Multi-Agent RL | 2-agent dispatch | IQL, JAL, Lenient Q, Mean Field Q | ✅ |
| 12 | Game Theory | Nash equilibrium | Nash Q, Correlated Q, Minimax Q, Fictitious Play | ✅ |
| 13 | Cooperative MARL | Value decomposition | IQL baseline, VDN, QMIX, QMIX+CG | ✅ |
| 14 | Foundational MARL | Full MARL comparison | IQL, VDN, MAPG, MADDPG | ✅ |
| 15 | Deep Learning Foundations | FNN, activations, optimizers | ReLU/Swish/ELU, Adam/SGD/RMSProp, L2, Dropout | ✅ |
| 16–20 | Deep RL | DQN, A2C, PPO, SAC, TD3… | — | 🔜 |

---

## 📚 Chapter Summaries

### Chapter 01 — MDP Baseline & ε-Greedy Dispatch
**Key concept:** Markov Decision Process (MDP) — the formal foundation of all RL.  
**Business problem:** Warsaw ASP dispatch with 5 technicians and up to 20 orders. Q-table is all zeros — this is the random baseline.  
**Algorithms:** ε-greedy policy (Q=0 → always random).  
**Key output:** Baseline $G_0$ (discounted return) to compare all future chapters against.  
**Rust function:** `run_ch01_episode()`

---

### Chapter 02 — Bellman Equation & Value Iteration
**Key concept:** Bellman optimality equation — the recursive definition of optimal value.  
**Business problem:** Find the optimal dispatch policy $\pi^*$ for the 8-state Warsaw ASP MDP.  
**Algorithms:** Value Iteration (VI) — iterates $V^{(k+1)}(s) = \max_a \sum_{s'} P(s'|s,a)[R + \gamma V^{(k)}(s')]$ until convergence.  
**Key output:** $V^*(s)$ for all 8 states, optimal policy $\pi^*$, convergence curve, Glass-Box Bellman trace.  
**Rust function:** `run_ch02_value_iteration()`

---

### Chapter 03 — Multi-Armed Bandit & Exploration Strategies
**Key concept:** Stateless RL — the simplest exploration-exploitation trade-off.  
**Business problem:** Warsaw ASP skill-slot optimisation — 5 arms (HVAC, Electrical, Plumbing, Network, Mechanical) with unknown true SLA rates.  
**Algorithms:** ε-Greedy with annealing, UCB1 ($O(\sqrt{KT \ln T})$ regret), Thompson Sampling (Beta posterior).  
**Key output:** Cumulative regret curves, arm pull distribution, Q-value convergence toward true SLA rates.  
**Rust function:** `run_ch03_bandits()`

---

### Chapter 04 — Dynamic Programming: PI vs VI vs Async VI
**Key concept:** Exact model-based planning — requires full knowledge of $P(s'|s,a)$.  
**Business problem:** Compare three DP algorithms on the same Warsaw ASP MDP — all must find the same $\pi^*$.  
**Algorithms:** Policy Iteration (PI), Value Iteration (VI), Asynchronous VI (prioritised by Bellman residual).  
**Key output:** Convergence comparison, Policy Evolution table (PI outer iterations), Bellman residual heatmap.  
**Rust function:** `run_ch04_dp()`

---

### Chapter 05 — Monte Carlo Methods
**Key concept:** Model-free learning from complete episodes — no $P(s'|s,a)$ needed.  
**Business problem:** Learn $V^\pi(s)$ and $Q^*(s,a)$ from observed dispatch episodes.  
**Algorithms:** First-Visit MC (unbiased), Every-Visit MC (consistent), On-Policy MC Control, Off-Policy MC with Importance Sampling.  
**Key output:** V(s) estimates converging toward DP reference (Ch04), visit count heatmap, episode returns curve.  
**Rust function:** `run_ch05_mc()`

---

### Chapter 06 — Temporal Difference Learning
**Key concept:** Online learning — updates after every step, not episode end. Combines MC (model-free) and DP (bootstrapping).  
**Business problem:** Real-time dispatch learning — update Q-values after each dispatch decision.  
**Algorithms:** TD(0) prediction, SARSA (on-policy, safe), Q-Learning (off-policy, optimal).  
**Key output:** TD error curve decaying toward zero, SARSA vs Q-Learning policy comparison, Q-table heatmap.  
**Rust function:** `run_ch06_td()`

---

### Chapter 07 — n-Step TD & Dyna-Q Planning
**Key concept:** Bridging TD(0) and MC with n-step returns; model-based planning for sample efficiency.  
**Business problem:** Reduce real dispatch interactions needed to learn a good policy.  
**Algorithms:** n-Step TD (n=1=TD0, n=∞=MC, sweet spot n=3–5), n-Step SARSA, Dyna-Q (k planning steps), Dyna-Q+ (exploration bonus $\kappa\sqrt{\tau}$).  
**Key output:** ~5× sample efficiency gain with Dyna-Q k=5, model coverage chart growing toward 32 (|S|×|A|).  
**Rust function:** `run_ch07_nstep()`

---

### Chapter 08 — Eligibility Traces & TD(λ)
**Key concept:** Backward credit assignment — update all recently visited (s,a) pairs simultaneously.  
**Business problem:** Faster reward propagation through the Warsaw ASP state space.  
**Algorithms:** SARSA(λ) (on-policy), Q(λ) Watkins (off-policy, traces cut on non-greedy), SARSA λ=0 (TD0 baseline), SARSA λ=0.99 (≈MC).  
**Key output:** Active traces chart, λ=0.7 sweet spot, Watkins cut effect on trace count.  
**Rust function:** `run_ch08_eligibility()`

---

### Chapter 09 — Policy Gradient: REINFORCE & Softmax
**Key concept:** Direct policy optimisation — no Q-table, parameterise $\pi(a|s,\theta)$ directly.  
**Business problem:** Foundation for deep RL (A2C, PPO) — scales to continuous state spaces.  
**Algorithms:** REINFORCE (unbiased, high variance), REINFORCE+Baseline (lower variance), Actor-Critic TD(0) (online, biased), REINFORCE τ=0.5 (sharp policy).  
**Key output:** Policy entropy chart (healthy decay), θ[s][a] heatmap, PG magnitude curve.  
**Rust function:** `run_ch09_policy_gradient()`

---

### Chapter 10 — Model-Based RL: World Models
**Key concept:** Learn the transition model $\hat{T}(s,a,s')$ from experience, then plan with it.  
**Business problem:** Bridge between model-free (Ch06) and model-based (Ch04) — no analytical model available.  
**Algorithms:** WM Q-Learning (Dyna-Q with explicit model), Prioritised Sweeping (plan from highest Bellman residual), MBPO (synthetic rollouts for REINFORCE), Uncertainty Bonus ($Q + \beta/\sqrt{N+1}$).  
**Key output:** Model accuracy chart growing toward 1.0, planning steps chart, Prioritised Sweeping convergence advantage.  
**Rust function:** `run_ch10_world_model()`

---

### Chapter 11 — Multi-Agent RL
**Key concept:** Two agents share the Warsaw ASP MDP — non-stationarity, coordination without communication.  
**Business problem:** Two Warsaw ASP dispatchers acting independently but with interacting rewards.  
**Algorithms:** IQL (independent Q-Learning, non-stationary baseline), JAL (models partner policy $\hat{\pi}_j$), Lenient Q (ignores negative TD errors with probability μ), Mean Field Q (scales to N agents via mean action).  
**Key output:** Cooperation rate chart, joint V(s) comparison, Q-table heatmaps per agent.  
**Rust function:** `run_ch11_multiagent()`

---

### Chapter 12 — Game Theory & Nash Equilibrium
**Key concept:** Formalise multi-agent interaction — find stable equilibria where no agent can improve unilaterally.  
**Business problem:** Strategic dispatch coordination — find the Nash equilibrium operating point.  
**Algorithms:** Nash Q-Learning (converges to Nash equilibrium), Correlated Q (joint distribution via regret matching), Minimax Q (zero-sum adversarial), Fictitious Play (best response to empirical average).  
**Key output:** Nash Gap (exploitability) chart decaying toward 0, Mixed Strategy Profile per state.  
**Rust function:** `run_ch12_game_theory()`

---

### Chapter 13 — Cooperative MARL: VDN & QMIX
**Key concept:** Value decomposition — agents share joint reward, centralised training with decentralised execution (CTDE).  
**Business problem:** Two dispatchers maximise joint SLA performance — cooperative, not competitive.  
**Algorithms:** IQL baseline, VDN ($Q_{tot} = Q_0 + Q_1$, additive, IGM), QMIX (monotone mixing $w_i(s) \geq 0$, state-dependent weights), QMIX+CG (counterfactual baseline for credit assignment).  
**Key output:** Mixing weights chart, joint $Q_{tot}$ curve, IGM verification across all states.  
**Rust function:** `run_ch13_coop_marl()`

---

### Chapter 14 — Foundational MARL Algorithms
**Key concept:** Capstone of the MARL curriculum — unified comparison of value-based, policy-based, and actor-critic MARL methods.  
**Business problem:** 5×5 grid world with 2 agents collecting resources while minimising distance to target.  
**Algorithms:** IQL (independent Q-Learning baseline), VDN (additive decomposition), MAPG (Multi-Agent Policy Gradient with entropy bonus $\beta H(\pi)$), MADDPG (centralised critic + decentralised actors + soft target update $\tau$).  
**Key output:** Episode returns comparison (MA-20), cooperation rate, TD error curves, Q-table/logit heatmaps per agent.  
**Rust function:** `run_ch14()`

---

### Chapter 15 — Deep Learning Foundations
**Key concept:** Universal Approximation Theorem (UAT) — FNNs can approximate any continuous function. Bridge from tabular RL to Deep RL.  
**Business problem:** Approximate $V^*(s)$ from Ch02 using a FNN trained on 4 Warsaw ASP state features (SLA rate, urgency, distance, skill match) — without the exact Bellman model.  
**Algorithms:** FNN with backpropagation, 6 activation functions (ReLU, LeakyReLU, ELU, Swish, Tanh, Sigmoid), 3 optimizers (SGD, Adam, RMSProp), L2 regularization, Dropout. He initialisation.  
**4 configurations compared:** ReLU+SGD (baseline), User activation+Adam, Adam+L2, Adam+Dropout.  
**Key output:** Loss curves (log scale), V*(s) predictions vs Ch02 reference, activation function comparison chart, network architecture summary.  
**Rust function:** `run_ch15()`

---

## 🗂️ Repository Structure

```
rlvr-enterprise-allocator/
├── rlvr-core/src/          # Rust RL algorithms (ch01–ch15)
├── rlvr-py/src/lib.rs      # PyO3 Python bindings
├── gui/
│   ├── app.py              # Streamlit entry point
│   └── chapters/           # ch01.py – ch15.py (UI per chapter)
├── docs/                   # Hands-On Guide HTML files (EN/PL)
│   ├── handson_ch01_en.html
│   ├── handson_ch02_pl.html
│   └── handson_ch03_en.html … handson_ch15_en.html
└── Cargo.toml              # Rust workspace
```

---

## 🛠️ Development

```bash
# Rebuild Rust engine after changes
cd rlvr-py && maturin develop --release && cd ..

# Run tests
cargo test -p rlvr-core

# Restart GUI
pkill -f streamlit
streamlit run gui/app.py --server.port 8001
```

---

## 📖 Hands-On Guides

Each chapter includes a self-contained HTML guide with:
- 10 interactive tabs (Introduction, Theory, Environment, UI, Interpretation, Exercises, Tasks, Quiz, Summary)
- KaTeX math rendering (golden formulas)
- 5 practical tasks with hidden answers
- 10-question quiz (90% pass threshold)

Available languages: **EN** (Ch01–Ch15), **PL** (Ch02)

---

## 📄 License

MIT License — see [opensource.org/licenses/MIT](https://opensource.org/licenses/MIT)

Copyright (c) 2026 Sylwester Musial

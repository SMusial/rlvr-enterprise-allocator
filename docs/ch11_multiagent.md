# Chapter 11 - Multi-Agent RL

> **RLVR Enterprise Allocator** - Warsaw ASP - `rlvr-core::ch11_multiagent`

## Scenario

2 dispatchers (Agent 0, Agent 1) share the Warsaw ASP 8-state / 4-action MDP.
Each agent acts independently but their rewards interact through the shared environment.

## Algorithms

| Algorithm | Key idea | Communication |
|---|---|---|
| IQL | Independent Q-Learning - ignore other agents | None |
| JAL | Joint Action Learning - model partner's policy | Observe partner actions |
| Lenient Q | Ignore negative TD errors with prob mu | None |
| Mean Field Q | Approximate joint action by mean action | Mean action only |

## New output fields vs Ch07-Ch10

| Field | Type | Description |
|---|---|---|
| `q_tables` | list[agent][state][action] | Per-agent Q-tables |
| `policies` | list[agent][state] | Per-agent greedy policies |
| `cooperation_curve` | Vec<f64> | Fraction of steps both agents chose same action |
| `n_agents` | usize | Number of agents (2) |

## Key Formulas

```
# IQL
Q_i(s,a) += alpha * [r + gamma * max Q_i(s') - Q_i(s,a)]

# JAL
pi_j(a|s) = N_j(s,a) / sum_a N_j(s,a)   (empirical frequency)
Q_i(s,a) += alpha * [r + gamma * E_{pi_j}[max Q_i(s')] - Q_i(s,a)]

# Lenient Q
delta = r + gamma * max Q_i(s') - Q_i(s,a)
apply if delta >= 0, or with prob (1-mu) if delta < 0

# Mean Field Q
mean_a_j(s) = running mean of partner's actions in state s
Q_i(s,a) += alpha * [r + beta*mean_a_j/N_A + gamma*max Q_i(s') - Q_i(s,a)]
```

## Tests (15 inline)

smoke x4, shape x2 (q_tables, policies), curves, finite, cooperation in range,
n_agents correct, IQL converges, lenient mu=0 equals IQL, JAL converges,
MF converges, deterministic.

## Hyperparameters

| Param | Default | Notes |
|---|---|---|
| leniency_mu | 0.5 | 0=IQL, 1=never penalise. Start at 0.5 |
| mf_beta | 0.5 | Mean field influence weight. 0=IQL, 2=strong MF |
| alpha | 0.1 | Standard TD rate |
| gamma | 0.95 | Same as Ch01-Ch10 |

## Connection to other chapters

- Ch06 Q-Learning: IQL is Q-Learning applied independently per agent
- Ch12 Game Theory: JAL is the tabular precursor to Nash Q-Learning
- Ch13 Cooperative MARL: VDN/QMIX extend IQL with joint value decomposition
- Ch14 Learning Dynamics: cooperation_curve shows emergent coordination

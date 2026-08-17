# Chapter 10 - Model-Based RL: World Models

> **RLVR Enterprise Allocator** - Warsaw ASP - `rlvr-core::ch10_world_model`

## Algorithms

| Algorithm | Planning | Key idea |
|---|---|---|
| WM Q-Learning | Random k samples | Dyna-Q with explicit WorldModel object |
| Prioritised Sweeping | Priority queue | Plan from highest delta first, propagate to predecessors |
| MBPO (lite) | Synthetic rollouts | REINFORCE on model-generated trajectories |
| Uncertainty Bonus | Random k samples | UCB-style Q_bonus = Q + beta/sqrt(N+1) |

## World Model

Tabular model storing:
- `counts[s][a][s']` - empirical transition counts
- `r_sum[s][a]` - cumulative reward
- `n_visits[s][a]` - visit count

Mean reward: `R(s,a) = r_sum[s][a] / n_visits[s][a]`
Transition: sample from `counts[s][a][.] / sum`

## Key Formulas

```
# WM Q-Learning (planning step)
Q(s,a) <- Q(s,a) + alpha * [R_model(s,a) + gamma * max Q(s') - Q(s,a)]

# Prioritised Sweeping
Priority(s,a) = |R(s,a) + gamma * max Q(s') - Q(s,a)|
Plan from max priority. After update, add predecessors to queue.

# Uncertainty Bonus
Q_bonus(s,a) = Q(s,a) + beta / sqrt(N(s,a) + 1)
Action = argmax Q_bonus(s,.)
Q update = standard Q-Learning (no bonus)
```

## New output fields vs Ch07-Ch09

| Field | Type | Description |
|---|---|---|
| `model_accuracy` | Vec<f64> | Fraction of (s,a) where learned T matches true T |
| `planning_steps_used` | Vec<f64> | Actual planning steps per episode |
| `model_size` | usize | Number of visited (s,a) pairs in model |

## Tests (15 inline)

smoke x4, shape x2, curves, finite, model grows, accuracy increases,
planning steps positive, WM converges, PS converges, uncertainty explores,
deterministic, softmax sums to 1.

## Hyperparameters

| Param | Default | Notes |
|---|---|---|
| k (planning_steps) | 5 | Higher = faster convergence, more compute |
| priority_threshold | 0.01 | Lower = more updates, higher = fewer |
| uncertainty_beta | 1.0 | Higher = more exploration |
| alpha | 0.1 | Standard TD rate |
| gamma | 0.95 | Same as Ch01-Ch09 |

## Connection to other chapters

- Ch07 Dyna-Q: WM Q-Learning is Dyna-Q with an explicit model object
- Ch09 REINFORCE: MBPO uses REINFORCE on synthetic model rollouts
- Ch11+ Multi-Agent: each agent can maintain its own world model

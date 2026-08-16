# Chapter 09 — Policy Gradient: REINFORCE & Softmax

> **RLVR Enterprise Allocator** · Warsaw ASP · `rlvr-core::ch09_policy_gradient`

## Algorithms

| Algorithm | Type | Update | Notes |
|---|---|---|---|
| REINFORCE | On-policy MC | Episode-end | Unbiased, high variance |
| REINFORCE+Baseline | On-policy MC | Episode-end | Variance-reduced |
| Actor-Critic (TD0) | On-policy TD | Per-step | Lower variance, biased |
| REINFORCE τ=0.5 | On-policy MC | Episode-end | Sharper policy |

## Policy Representation

```
π(a|s,θ) = exp(θ[s][a] / τ) / Σ_b exp(θ[s][b] / τ)
```

## Key Update Rules

```
# REINFORCE
G_t = Σ_{k=t}^T γ^{k-t} R_{k+1}
θ[s][a] += α γ^t (G_t − b(s)) (𝟙[a=A_t] − π(a|s))

# Actor-Critic
δ_t = R_{t+1} + γ V(S_{t+1}) − V(S_t)
V(S_t) += α_v δ_t
θ[s][a] += α γ^t δ_t (𝟙[a=A_t] − π(a|s))
```

## Tests (15 inline)

smoke × 4, shape × 2, curves, finite, softmax sums to 1, temperature sharpens,
entropy positive, baseline reduces variance, REINFORCE converges,
Actor-Critic converges, deterministic.

## Hyperparameters

| Param | Default | Notes |
|---|---|---|
| α (policy) | 0.01 | Lower than TD — PG is noisier |
| α_v (baseline) | 0.1 | Standard TD(0) rate |
| τ (temperature) | 1.0 | Start at 1.0, reduce to 0.5 for sharper policy |
| n_episodes | 500 | PG needs more episodes — use 1000+ for convergence |

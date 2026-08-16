# Chapter 08 — Eligibility Traces & TD(λ)

> **RLVR Enterprise Allocator** · Warsaw ASP · `rlvr-core::ch08_eligibility`

## Algorithms

| Algorithm | Type | λ | Notes |
|---|---|---|---|
| SARSA(λ) | On-policy | user (default 0.7) | Replacing traces |
| Q(λ) Watkins | Off-policy | user (default 0.7) | Traces cut on non-greedy |
| SARSA λ=0 | On-policy | 0.0 | Baseline ≡ TD(0) |
| SARSA λ=0.99 | On-policy | 0.99 | Baseline ≈ MC |

## Key Formula

```
e_t(s,a) = γλ · e_{t-1}(s,a) + 𝟙[s=S_t, a=A_t]   (replacing: set to 1)
δ_t = R + γ Q(S',A') − Q(S,A)
Q(s,a) ← Q(s,a) + α δ_t e_t(s,a)   for ALL (s,a)
```

## Tests (14 inline)

smoke × 3, shape × 2, curves, finite, λ=0 no traces, trace stats positive,
replacing vs accumulating differ, converges, Q(λ) converges, deterministic, worst state lower value.

## Hyperparameters

| Param | Default | Notes |
|---|---|---|
| λ | 0.7 | Start here; reduce to 0.5 if stochastic env |
| α | 0.1 | Standard TD rate |
| γ | 0.95 | Same as Ch01–Ch07 |
| Traces | Replacing | Stable for tabular |

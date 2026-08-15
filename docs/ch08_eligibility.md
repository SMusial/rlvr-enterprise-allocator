# Chapter 08 — Eligibility Traces & TD(λ)

> **RLVR Enterprise Allocator** · Warsaw ASP Field-Service Optimisation
> Rust crate: `rlvr-core` · Module: `ch08_eligibility` · GUI: `gui/chapters/ch08.py`

---

## Overview

Chapter 08 introduces **eligibility traces** — a memory mechanism that bridges
TD(0) (single-step bootstrapping) and Monte Carlo (full-episode returns) through
a single parameter **λ ∈ [0, 1]**.

Four algorithms are implemented and compared:

| Algorithm | Type | λ | Notes |
|---|---|---|---|
| **SARSA(λ)** | On-policy | user-set (default 0.7) | Backward-view, replacing traces |
| **Q(λ) Watkins** | Off-policy | user-set (default 0.7) | Traces cut on non-greedy action |
| **SARSA λ=0** | On-policy | 0.0 | Baseline ≡ TD(0) / SARSA (Ch06) |
| **SARSA λ=0.99** | On-policy | 0.99 | Baseline ≈ Monte Carlo (Ch05) |

---

## The Problem — Credit Assignment in Warsaw ASP

A dispatcher assigns Technician #2 to Job #5. The technician drives 20 minutes.
The SLA reward (+12.3) arrives only on arrival.

- **TD(0)**: updates only the last step — reward propagates back over *many episodes*
- **TD(λ=0.7)**: propagates reward back through the entire trace *within one episode*

The eligibility trace for the dispatch decision decays as `(γλ)^k`:

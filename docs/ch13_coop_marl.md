# Chapter 13 - Cooperative MARL: VDN and QMIX

> RLVR Enterprise Allocator - Warsaw ASP - rlvr-core::ch13_coop_marl

## Scenario

2 dispatchers cooperate to maximise a JOINT reward.
Both agents share the same reward signal.
Goal: centralised training, decentralised execution (CTDE).

## Algorithms

| Algorithm | Q_tot formula | Key property |
|---|---|---|
| IQL Baseline | none | independent, no coordination |
| VDN | Q_0 + Q_1 | additive decomposition, IGM holds |
| QMIX | w_0(s)*Q_0 + w_1(s)*Q_1 + b(s) | monotone mixing, IGM holds |
| QMIX+CG | QMIX + counterfactual advantage | better credit assignment |

## IGM Property

```
argmax_a Q_tot(s,a) = (argmax Q_0(s_0,.), argmax Q_1(s_1,.))
Enables decentralised execution: each agent greedily picks its own best action.
```

## New output fields vs Ch12

| Field | Type | Description |
|---|---|---|
| mixing_weights | Vec<f64> | mean absolute mixing weight per episode |
| joint_q_curve | Vec<f64> | Q_tot value per episode |

## Tests (15 inline)

smoke x4, shape x2, curves, finite, VDN >= IQL,
QMIX mixing weights non-negative, QMIX+CG mixing weights non-negative,
joint Q finite, IGM for VDN, QMIX converges, deterministic.

## Connection to other chapters

- Ch11 IQL: IQL baseline is identical to Ch11 IQL with joint reward
- Ch12 Nash Q: QMIX assumes cooperative game (team reward), not Nash
- Ch14 Learning Dynamics: QMIX mixing weight evolution shows coordination
- Ch18 QMIX Deep: deep neural network mixing network replaces tabular w(s)
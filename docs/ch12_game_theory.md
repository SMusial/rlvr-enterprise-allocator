# Chapter 12 - Game Theory and Nash Equilibrium

> RLVR Enterprise Allocator - Warsaw ASP - rlvr-core::ch12_game_theory

## Algorithms

| Algorithm | Game type | Key idea |
|---|---|---|
| Nash Q-Learning | General-sum | Converges to Nash equilibrium |
| Correlated Q-Learning | General-sum | Coordinates via correlated equilibrium |
| Minimax Q-Learning | Zero-sum | Optimal vs worst-case opponent |
| Fictitious Play | General/zero-sum | Best response to empirical average |

## Tests (15 inline)

smoke x4, shape x2, curves, finite, strategies sum to 1,
nash gap non-negative, exploitability non-negative,
minimax zero-sum, fictitious converges, nash Q converges, deterministic.

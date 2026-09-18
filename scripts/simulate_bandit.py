#!/usr/bin/env python
"""Synthetic regret simulation: Thompson sampling vs. epsilon-greedy vs.
random, on a made-up "true" preference vector. This isn't a test (there's no
assertion) -- it's the portfolio-artifact sanity check the spec asks for in
section 9: show the bandit actually converges instead of just trusting the
math. Real per-item rewards from actual usage live in data/state.json and are
far too sparse to plot a regret curve from directly; this uses a synthetic
ground truth instead, which is the standard way to validate a bandit
implementation before trusting it on real (sparse, noisy) feedback.

    PYTHONPATH=src python scripts/simulate_bandit.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from recsys.bandit import ThompsonBandit

N_ARMS = 12
N_FEATURES = 5
N_ROUNDS = 400
SEED = 0


def run_thompson(true_theta, arm_features, rng) -> np.ndarray:
    bandit = ThompsonBandit([f"f{i}" for i in range(N_FEATURES)],
                             prior_mean=np.zeros(N_FEATURES), lambda_reg=1.0, sigma=1.0)
    regrets = np.zeros(N_ROUNDS)
    best_value = max(true_theta @ phi for phi in arm_features)
    for t in range(N_ROUNDS):
        scores = bandit.rank(arm_features, rng)
        choice = int(np.argmax(scores))
        phi = arm_features[choice]
        reward = float(true_theta @ phi + rng.normal(0, 0.3))
        bandit.update(phi, reward)
        regrets[t] = best_value - float(true_theta @ phi)
    return regrets


def run_epsilon_greedy(true_theta, arm_features, rng, eps0=1.0) -> np.ndarray:
    theta_hat = np.zeros(N_FEATURES)
    A = np.eye(N_FEATURES)
    b = np.zeros(N_FEATURES)
    regrets = np.zeros(N_ROUNDS)
    best_value = max(true_theta @ phi for phi in arm_features)
    for t in range(N_ROUNDS):
        eps = min(1.0, eps0 / (t + 1))
        if rng.random() < eps:
            choice = rng.integers(0, len(arm_features))
        else:
            choice = int(np.argmax([theta_hat @ phi for phi in arm_features]))
        phi = arm_features[choice]
        reward = float(true_theta @ phi + rng.normal(0, 0.3))
        A += np.outer(phi, phi)
        b += reward * phi
        theta_hat = np.linalg.solve(A, b)
        regrets[t] = best_value - float(true_theta @ phi)
    return regrets


def run_random(true_theta, arm_features, rng) -> np.ndarray:
    best_value = max(true_theta @ phi for phi in arm_features)
    regrets = np.zeros(N_ROUNDS)
    for t in range(N_ROUNDS):
        choice = rng.integers(0, len(arm_features))
        regrets[t] = best_value - float(true_theta @ arm_features[choice])
    return regrets


def bucketed_avg(regrets: np.ndarray, bucket: int = 50) -> list[float]:
    return [float(regrets[i:i + bucket].mean()) for i in range(0, len(regrets), bucket)]


def main():
    rng = np.random.default_rng(SEED)
    true_theta = rng.normal(0, 1, size=N_FEATURES)
    arm_features = [rng.normal(0, 1, size=N_FEATURES) for _ in range(N_ARMS)]

    results = {
        "Thompson sampling": run_thompson(true_theta, arm_features, np.random.default_rng(SEED)),
        "epsilon-greedy":    run_epsilon_greedy(true_theta, arm_features, np.random.default_rng(SEED)),
        "random baseline":   run_random(true_theta, arm_features, np.random.default_rng(SEED)),
    }

    print(f"Synthetic bandit comparison: {N_ARMS} arms, {N_FEATURES} features, {N_ROUNDS} rounds\n")
    print(f"{'rounds':>12} | " + " | ".join(f"{name:>18}" for name in results))
    buckets = {name: bucketed_avg(r) for name, r in results.items()}
    n_buckets = len(next(iter(buckets.values())))
    for i in range(n_buckets):
        label = f"{i*50:>5}-{i*50+49:<5}"
        row = " | ".join(f"{buckets[name][i]:>18.3f}" for name in results)
        print(f"{label:>12} | {row}")

    print("\nCumulative regret (lower is better):")
    for name, r in results.items():
        print(f"  {name:20s} {r.sum():8.1f}")


if __name__ == "__main__":
    main()

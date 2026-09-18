"""Step 2 -- the contextual bandit. Hand-written Bayesian linear regression +
Thompson sampling, per the spec: no black-box bandit library.

One shared model, not one per arm. "Arms" here are individual activities in a
option bank that has hundreds of entries and keeps growing, each seen only a
handful of times by a single user -- there's never enough exposure to fit a
separate regression per item. So instead of per-arm theta_a, there's a single
theta over the shared feature vector phi(option, context) from
content_scoring.featurize(). This is exactly the cold-start argument the spec
makes for contextual (feature-based) bandits over per-arm bandits: a new item
gets a reasonable score immediately because it shares features with items
already seen, and the prior mean below IS the Step-1 rules-based weight
vector -- Thompson sampling only has to learn a CORRECTION to my own
hand-set profile, not start from nothing.

Math (linear-Gaussian conjugate Bayesian linear regression):
    posterior:      theta ~ N(theta_hat, sigma^2 A^-1)
    theta_hat     = A^-1 b
    A             = lambda*I + sum_t phi_t phi_t^T
    b             = lambda*I*prior_mean + sum_t r_t * phi_t   (so theta_hat == prior_mean with no data)
    act:            sample theta_tilde ~ posterior, then argmax_a  phi(a)^T theta_tilde
Exploration falls out of posterior uncertainty automatically: a feature
combination seen rarely has a wide posterior, so its sampled theta_tilde
swings more, occasionally pushing it to the top even without hard-coded
epsilon-exploration.

Non-stationarity (taste drift, spec section on the topic): `update()` takes an
optional discount `gamma < 1` that shrinks A and b toward the prior before
folding in the new observation, so old evidence decays instead of locking in
forever.
"""
from __future__ import annotations

import numpy as np

from .state import BanditPosterior


class ThompsonBandit:
    def __init__(self, feature_names: list[str], prior_mean: np.ndarray,
                 lambda_reg: float = 2.0, sigma: float = 1.0):
        self.feature_names = list(feature_names)
        self.prior_mean = np.asarray(prior_mean, dtype=float)
        self.lambda_reg = lambda_reg
        self.sigma = sigma
        d = len(feature_names)
        self.A = lambda_reg * np.eye(d)
        self.b = lambda_reg * np.eye(d) @ self.prior_mean

    @property
    def theta_hat(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.b)

    def sample_theta(self, rng: np.random.Generator) -> np.ndarray:
        cov = (self.sigma ** 2) * np.linalg.inv(self.A)
        return rng.multivariate_normal(self.theta_hat, cov)

    def rank(self, phis: list[np.ndarray], rng: np.random.Generator) -> list[float]:
        """One posterior sample, scored against every candidate -- this IS the
        explore/exploit step. Returns scores in the same order as `phis`."""
        theta_tilde = self.sample_theta(rng)
        return [float(phi @ theta_tilde) for phi in phis]

    def update(self, phi: np.ndarray, reward: float, gamma: float = 1.0) -> None:
        if gamma < 1.0:
            d = len(phi)
            prior_A = self.lambda_reg * np.eye(d)
            prior_b = prior_A @ self.prior_mean
            self.A = gamma * self.A + (1 - gamma) * prior_A
            self.b = gamma * self.b + (1 - gamma) * prior_b
        self.A += np.outer(phi, phi)
        self.b += reward * phi

    def to_posterior(self, n_updates: int) -> BanditPosterior:
        return BanditPosterior(
            A=self.A.tolist(), b=self.b.tolist(),
            feature_names=self.feature_names, n_updates=n_updates,
        )

    @staticmethod
    def from_posterior(posterior: BanditPosterior, prior_mean: np.ndarray,
                        lambda_reg: float = 2.0, sigma: float = 1.0) -> "ThompsonBandit":
        bandit = ThompsonBandit(posterior.feature_names, prior_mean, lambda_reg, sigma)
        bandit.A = np.array(posterior.A, dtype=float)
        bandit.b = np.array(posterior.b, dtype=float)
        return bandit


def reward_from_feedback(action: str, rating: float | None = None) -> float:
    """Turn my feedback into a scalar reward. Explicit feedback (a rating)
    wins when given, since it's the cleaner (if sparser) signal the spec
    prefers; otherwise fall back to the coarser implicit action label."""
    if rating is not None:
        return max(-1.0, min(1.0, (rating - 3.0) / 2.0))  # 1..5 -> -1..1
    return {
        "accepted_repeat": 1.0,
        "accepted_no_repeat": -0.3,
        "skipped": -0.5,
    }.get(action, 0.0)

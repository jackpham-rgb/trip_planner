"""Wires retrieval -> scoring -> selection -> feedback into the two entry
points the spec's mental model calls for: "what should I do next" (the
bandit picks one, with exploration) and "plan my day/trip" (scores feed the
orienteering optimizer as prizes). See recommender-planner-spec.txt section 3.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .bandit import ThompsonBandit, reward_from_feedback
from .content_scoring import (
    DEFAULT_WEIGHTS, FEATURE_NAMES, ScoredOption, diversify, featurize,
    filter_candidates, rank_by_rules,
)
from .optimizer import Itinerary, greedy_two_opt, milp_orienteering
from .schema import Context, Option
from .state import UserState, load_state, save_state


def _get_bandit(state: UserState) -> ThompsonBandit:
    if state.bandit is not None:
        return ThompsonBandit.from_posterior(state.bandit, DEFAULT_WEIGHTS)
    return ThompsonBandit(FEATURE_NAMES, DEFAULT_WEIGHTS)


def score_candidates(
    options: list[Option], context: Context, state: UserState,
    use_bandit: bool = True, rng: np.random.Generator | None = None,
) -> list[ScoredOption]:
    """Retrieval + scoring. Falls back to pure content rules until the bandit
    has seen at least one piece of feedback (cold start), then lets the
    bandit's learned/sampled weights take over. Applies the done-and-would-
    repeat recovery multiplier either way."""
    candidates = filter_candidates(options, context, state.excluded_ids())
    if not candidates:
        return []

    if use_bandit and state.bandit is not None and state.bandit.n_updates > 0:
        rng = rng or np.random.default_rng()
        bandit = _get_bandit(state)
        phis = [featurize(o, context) for o in candidates]
        scores = bandit.rank(phis, rng)
        scored = [ScoredOption(option=o, score=s, features=phi)
                  for o, s, phi in zip(candidates, scores, phis)]
    else:
        scored = rank_by_rules(candidates, context)

    for s in scored:
        s.score *= state.recovery_multiplier(s.option.id)
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored


def recommend_next(options, context, state, top_k: int = 5, rng=None) -> list[ScoredOption]:
    """A shown menu benefits from diversity even though the optimizer (plan_trip)
    should not: five near-identical high-value hikes is a bad suggestion list
    but could be a perfectly good day plan."""
    scored = score_candidates(options, context, state, use_bandit=True, rng=rng)
    return diversify(scored, top_k)


def plan_trip(options, context, state, budget_hours: float,
              candidate_pool: int = 15, use_milp: bool = True) -> Itinerary:
    scored = score_candidates(options, context, state, use_bandit=True)[:candidate_pool]
    if not scored:
        return Itinerary(order=[], total_value=0.0, total_hours=0.0)
    if use_milp:
        try:
            milp = milp_orienteering(scored, budget_hours)
            if milp.order:
                return milp
        except Exception:
            pass  # fall through to the always-available heuristic
    return greedy_two_opt(scored, budget_hours)


def apply_feedback(state: UserState, option: Option, context: Context,
                    action: str, rating: float | None = None) -> None:
    """action: 'accepted_repeat' | 'accepted_no_repeat' | 'skipped'."""
    reward = reward_from_feedback(action, rating)
    bandit = _get_bandit(state)
    phi = featurize(option, context)
    bandit.update(phi, reward, gamma=0.98)  # mild decay for non-stationarity
    n_updates = (state.bandit.n_updates if state.bandit else 0) + 1
    state.bandit = bandit.to_posterior(n_updates)
    if action in ("accepted_repeat", "accepted_no_repeat"):
        state.record_outcome(option.id, would_repeat=(action == "accepted_repeat"), rating=rating)


DEFAULT_OPTION_BANK = Path("data/option_bank.json")
DEFAULT_STATE = Path("data/state.json")


def load_options(path: Path = DEFAULT_OPTION_BANK) -> list[Option]:
    from .data_loader import load_option_bank
    return load_option_bank(path)


def load_user_state(path: Path = DEFAULT_STATE) -> UserState:
    return load_state(path)


def save_user_state(state: UserState, path: Path = DEFAULT_STATE) -> None:
    save_state(state, path)

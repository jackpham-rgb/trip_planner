"""Step 1 — Rules + content scoring (cold start, works with zero feedback data).

Two things live here:
  1. `filter_candidates`: cheap retrieval — trip length, budget, and
     already-decided (done & would-not-repeat) items drop out before scoring.
  2. `featurize`: the SHARED feature function phi(option, context). Step 1
     scores candidates with a hand-set weight vector dotted against phi.
     Step 2 (bandit.py) reuses this exact phi and *learns* the weight vector
     from feedback instead of hand-setting it, with the hand-set weights below
     serving as the bandit's prior mean. That's the "hybrid: content prior +
     bandit exploration" cold-start strategy the spec calls for — Step 1 isn't
     thrown away once the bandit exists, it becomes the bandit's starting
     belief.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .schema import (
    COSTS, PRIORITIES, VIRAL_RATINGS, OUTDOOR_LEAN, ENERGY_DEMAND,
    Context, Option,
)

# Types that lean toward an adventurous/high-stimulation evening vs. a relaxed one.
# A second judgment-call vocabulary, same spirit as OUTDOOR_LEAN/ENERGY_DEMAND.
MOOD_ADVENTURE_LEAN = {
    "Historical": -0.1, "Nature": 0.3, "Food": 0.0, "Architecture": -0.2,
    "Shopping": -0.3, "Nightlife": 0.6, "Cultural": -0.1, "Technology": 0.1,
    "Automotive": 0.4, "Anime/pop culture": 0.2, "Photography": 0.1,
    "Adventure": 1.0, "Relaxation": -0.9, "Festival/event": 0.7,
    "Social-media-famous": 0.2, "Transport experience": 0.3,
    "Academic/research": -0.4,
}

FEATURE_NAMES = [
    "bias", "indoor_outdoor_match", "energy_match", "mood_match",
    "budget_fit", "priority_level", "viral_level", "short_and_easy",
]


def _ordinal(value: Optional[str], vocab: list[str]) -> Optional[int]:
    if value is None:
        return None
    for i, v in enumerate(vocab):
        if v.lower() == str(value).strip().lower():
            return i
    return None


def _to_pm1(index: Optional[int], size: int) -> float:
    """Map an ordinal index in [0, size-1] to [-1, 1]; missing -> 0 (neutral)."""
    if index is None or size <= 1:
        return 0.0
    return 2.0 * index / (size - 1) - 1.0


# Which region sheets are even reachable for a given trip type. Dwell time
# ("time needed" on an individual attraction, e.g. "1.5h" for a shrine visit)
# is NOT a proxy for how far away it is -- a short Tokyo shrine visit is not
# a valid "hangout nearby" suggestion just because the visit itself is quick.
# So trip_type scoping filters by SHEET (i.e. by region) first, and only uses
# est_hours as a secondary sanity check within that region.
TRIP_TYPE_SHEETS = {
    "hangout_nearby": {"Cali"},
    "day_trip": {"Cali", "Weekend Trips"},
    "long_trip": {
        "USA", "USA MEX CAN", "South America", "Europe", "East Asia",
        "South East Asia", "Middle East", "Africa", "Rest of World",
    },
}


def filter_candidates(
    options: list[Option],
    context: Context,
    excluded_ids: set[str],
) -> list[Option]:
    """Cheap retrieval: drop what's out of scope before anyone scores anything."""
    budget_idx = max(0, min(context.budget_level, len(COSTS) - 1))
    allowed_sheets = TRIP_TYPE_SHEETS.get(context.trip_type)  # None for "custom" -> no restriction
    out = []
    for o in options:
        if o.id in excluded_ids:
            continue
        cost_idx = _ordinal(o.cost, COSTS)
        if cost_idx is not None and cost_idx > budget_idx:
            continue
        if allowed_sheets is not None and o.source_sheet not in allowed_sheets:
            continue
        if context.trip_type in ("hangout_nearby", "day_trip"):
            if o.est_hours is not None and o.est_hours > 30:
                continue
        out.append(o)
    return out


def featurize(option: Option, context: Context) -> np.ndarray:
    """phi(option, context): the feature vector every scorer (rules or bandit) uses."""
    outdoor_lean = OUTDOOR_LEAN.get(option.type, 0.0)
    energy_demand = ENERGY_DEMAND.get(option.type, 0.0)
    mood_lean = MOOD_ADVENTURE_LEAN.get(option.type, 0.0)

    context_energy_pm1 = context.energy * 2.0 - 1.0
    indoor_outdoor_match = 1.0 - abs(outdoor_lean - context.indoor_outdoor) / 2.0
    energy_match = 1.0 - abs(energy_demand - context_energy_pm1) / 2.0
    mood_match = 1.0 - abs(mood_lean - context.mood_adventurous) / 2.0

    cost_idx = _ordinal(option.cost, COSTS)
    budget_idx = max(0, min(context.budget_level, len(COSTS) - 1))
    if cost_idx is None:
        budget_fit = 0.5
    else:
        headroom = budget_idx - cost_idx
        budget_fit = 1.0 if headroom >= 0 else max(0.0, 1.0 + headroom / len(COSTS))

    priority_level = _to_pm1(_ordinal(option.priority, PRIORITIES), len(PRIORITIES))
    viral_level = _to_pm1(_ordinal(option.viral_rating, VIRAL_RATINGS), len(VIRAL_RATINGS))

    if context.trip_type in ("hangout_nearby", "day_trip") and option.est_hours:
        short_and_easy = 1.0 if option.est_hours <= 6 else -0.5
    else:
        short_and_easy = 0.0

    return np.array([
        1.0,  # bias
        indoor_outdoor_match,
        energy_match,
        mood_match,
        budget_fit,
        priority_level,
        viral_level,
        short_and_easy,
    ])


# Hand-set weights for the cold-start ranker (Step 1) and the bandit's prior
# mean (Step 2). Tuned by inspection, not fit to data -- there is no data yet.
DEFAULT_WEIGHTS = np.array([
    0.0,   # bias
    1.2,   # indoor_outdoor_match
    1.0,   # energy_match
    1.0,   # mood_match
    1.5,   # budget_fit
    0.8,   # priority_level (workbook already curated MUST/HIGH as worth doing)
    0.6,   # viral_level (mild preference for "actually worth it")
    0.5,   # short_and_easy
])


@dataclass
class ScoredOption:
    option: Option
    score: float
    features: np.ndarray


def rank_by_rules(
    options: list[Option],
    context: Context,
    weights: np.ndarray = DEFAULT_WEIGHTS,
) -> list[ScoredOption]:
    """Step 1: score every candidate by feature similarity to a hand-set profile."""
    scored = []
    for o in options:
        phi = featurize(o, context)
        scored.append(ScoredOption(option=o, score=float(weights @ phi), features=phi))
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored


def diversify(scored: list[ScoredOption], top_k: int, penalty: float = 1.0) -> list[ScoredOption]:
    """Step 4 polish -- a cheap Maximal-Marginal-Relevance pass so a top-K
    suggestion list doesn't tunnel into one category (spec section 5,
    limitation #4: "a greedy bandit narrows suggestions ... you only ever see
    the same 3 cafes"). Similarity is just "same Type", which is enough to
    break up a list that would otherwise be five nature hikes in a row; a
    proper embedding-based similarity is a straightforward upgrade later."""
    if top_k >= len(scored):
        return sorted(scored, key=lambda s: s.score, reverse=True)
    pool = list(scored)
    chosen: list[ScoredOption] = []
    while pool and len(chosen) < top_k:
        def penalized(s: ScoredOption) -> float:
            same_type_count = sum(1 for c in chosen if c.option.type == s.option.type)
            return s.score - penalty * same_type_count
        best = max(pool, key=penalized)
        chosen.append(best)
        pool.remove(best)
    return chosen

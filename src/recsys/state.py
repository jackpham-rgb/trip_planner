"""User state: what Jack has done, whether he'd repeat it, and the bandit's
learned posterior. Kept in its own file, completely separate from the content
(`data/option_bank.json`), so rebuilding the option bank from a refreshed
workbook can never silently wipe out history -- the single design decision
current_setup.md flagged as most important from the earlier Android planning,
carried over here.

Also implements the interest decay/recovery curve for "done, would repeat"
items: interest drops right after doing something, then recovers the longer
it's been. This is the one idea from project_purpose_and_aim.md's calculus
framing that's cheap enough to earn its place -- everything else proposed
there (PageRank over the option bank, full Markov/MDP trip states) is left
out of this build; see docs/design.md for why.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np

DONE_NO_REPEAT = "done_no_repeat"
DONE_REPEAT = "done_repeat"

# Recovery curve: interest(t) = 1 - (1 - FLOOR) * exp(-t / TAU)
# t = days since last done. Right after doing it, interest ~= FLOOR (still in
# the pool, just heavily discounted). By ~3*TAU days it's back near 1.0.
RECOVERY_FLOOR = 0.25
RECOVERY_TAU_DAYS = 10.0


@dataclass
class ItemState:
    status: str  # DONE_NO_REPEAT | DONE_REPEAT
    times_done: int = 0
    last_done: Optional[str] = None  # ISO date
    rating: Optional[float] = None


@dataclass
class BanditPosterior:
    """Sufficient statistics for the shared-feature Bayesian linear regression."""
    A: list  # d x d, starts at lambda*I
    b: list  # d
    feature_names: list
    n_updates: int = 0


@dataclass
class UserState:
    items: dict = field(default_factory=dict)  # option_id -> ItemState
    bandit: Optional[BanditPosterior] = None

    def excluded_ids(self) -> set[str]:
        return {oid for oid, s in self.items.items() if s.status == DONE_NO_REPEAT}

    def recovery_multiplier(self, option_id: str, today: Optional[date] = None) -> float:
        st = self.items.get(option_id)
        if st is None or st.status != DONE_REPEAT or not st.last_done:
            return 1.0
        today = today or date.today()
        last = datetime.strptime(st.last_done, "%Y-%m-%d").date()
        t = max(0, (today - last).days)
        return RECOVERY_FLOOR + (1.0 - RECOVERY_FLOOR) * (1.0 - math.exp(-t / RECOVERY_TAU_DAYS))

    def record_outcome(self, option_id: str, would_repeat: bool, when: Optional[date] = None,
                        rating: Optional[float] = None) -> None:
        when = when or date.today()
        st = self.items.get(option_id)
        times_done = (st.times_done if st else 0) + 1
        self.items[option_id] = ItemState(
            status=DONE_REPEAT if would_repeat else DONE_NO_REPEAT,
            times_done=times_done,
            last_done=when.isoformat(),
            rating=rating,
        )


def load_state(path: Path) -> UserState:
    if not Path(path).exists():
        return UserState()
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = {oid: ItemState(**v) for oid, v in raw.get("items", {}).items()}
    bandit = None
    if raw.get("bandit"):
        bandit = BanditPosterior(**raw["bandit"])
    return UserState(items=items, bandit=bandit)


def save_state(state: UserState, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    raw = {
        "items": {oid: vars(s) for oid, s in state.items.items()},
        "bandit": vars(state.bandit) if state.bandit else None,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)

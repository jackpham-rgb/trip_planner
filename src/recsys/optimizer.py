"""Step 3 -- the orienteering optimizer.

Given scored candidates (value v_i = the bandit's/rules' score), a dwell time
per stop, a travel-time estimate between any two stops, and a time budget T,
choose a subset AND an order that maximizes total value without blowing the
budget. This is the prize-collecting / orienteering problem from the spec,
NP-hard in general, so there are two solvers:

  `greedy_two_opt`: value/time-ratio greedy construction + 2-opt local search
    on the resulting route. Instant, always available, good enough for a
    personal day plan. No external solver dependency.

  `milp_orienteering`: the exact MILP from the spec (flow conservation +
    MTZ subtour elimination + a time-budget knapsack constraint), solved with
    OR-Tools CP-SAT. Optimal for the (small, <=25-ish candidate) instances a
    single day plan actually has.

SIMPLIFICATION (documented, not hidden): there's no real address for "home"
and no live travel-time API wired up yet (that's a Step-4 polish item -- see
docs/design.md), so `estimate_travel_hours` below is a heuristic based on the
workbook's own city/section text fields, including a flat "getting there"
leg from an abstract depot. Swapping in a real distance-matrix API later only
means replacing that one function; both solvers already take a general
travel-time function.
"""
from __future__ import annotations

from dataclasses import dataclass

from .content_scoring import ScoredOption
from .schema import Option

DEPOT_ID = "__HOME__"


def estimate_travel_hours(a: Option, b: Option) -> float:
    """Heuristic travel time between two options (or from the depot, if one
    side is None), based on how much of the location text they share."""
    if a is None or b is None:
        return 0.5  # depot leg: "getting going"
    if a.city_region and b.city_region and a.city_region == b.city_region:
        return 0.3
    if a.section and b.section and a.section == b.section:
        return 1.0
    if a.country and b.country and a.country == b.country:
        return 2.0
    return 4.0


@dataclass
class Itinerary:
    order: list[Option]  # visiting order, depot excluded
    total_value: float
    total_hours: float

    def summary(self) -> str:
        lines = [f"Itinerary: {self.total_hours:.1f}h planned, value={self.total_value:.2f}"]
        for i, o in enumerate(self.order, 1):
            lines.append(f"  {i}. {o.label()} (~{o.est_hours or 0:.1f}h)")
        return "\n".join(lines)


def _route_hours(order: list[Option]) -> float:
    hours = estimate_travel_hours(None, order[0]) if order else 0.0
    for i in range(len(order) - 1):
        hours += (order[i].est_hours or 1.0) + estimate_travel_hours(order[i], order[i + 1])
    if order:
        hours += order[-1].est_hours or 1.0
    return hours


def greedy_two_opt(scored: list[ScoredOption], budget_hours: float) -> Itinerary:
    pool = sorted(scored, key=lambda s: s.score, reverse=True)
    route: list[Option] = []

    # Greedy construction: repeatedly insert the candidate (at whichever
    # position is cheapest) with the best value / added-time ratio that still
    # fits the remaining budget.
    remaining = list(pool)
    while remaining:
        best = None  # (ratio, insert_idx, candidate, added_hours)
        for cand in remaining:
            o = cand.option
            dwell = o.est_hours or 1.0
            for idx in range(len(route) + 1):
                prev = route[idx - 1] if idx > 0 else None
                nxt = route[idx] if idx < len(route) else None
                added = dwell
                added += estimate_travel_hours(prev, o)
                if nxt is not None:
                    added += estimate_travel_hours(o, nxt) - estimate_travel_hours(prev, nxt)
                elif prev is None:
                    added += estimate_travel_hours(o, None)  # return-ish leg for a single stop
                if _route_hours(route[:idx] + [o] + route[idx:]) > budget_hours:
                    continue
                ratio = cand.score / max(added, 0.1)
                if best is None or ratio > best[0]:
                    best = (ratio, idx, cand, added)
        if best is None:
            break
        _, idx, cand, _ = best
        route.insert(idx, cand.option)
        remaining.remove(cand)

    route = _two_opt(route, budget_hours)
    value = sum(s.score for s in pool if s.option in route)
    return Itinerary(order=route, total_value=value, total_hours=_route_hours(route))


def _two_opt(route: list[Option], budget_hours: float) -> list[Option]:
    """Standard 2-opt: repeatedly reverse a segment if it shortens total
    travel+dwell time, keeping the same set of stops (just a better order)."""
    if len(route) < 3:
        return route
    improved = True
    best = route
    best_hours = _route_hours(best)
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                h = _route_hours(candidate)
                if h < best_hours - 1e-9 and h <= budget_hours + 1e-9:
                    best, best_hours = candidate, h
                    improved = True
    return best


def milp_orienteering(scored: list[ScoredOption], budget_hours: float,
                       time_limit_seconds: float = 10.0) -> Itinerary:
    """Exact MILP (OR-Tools CP-SAT): flow conservation + MTZ subtour
    elimination + a time-budget constraint, maximizing total value."""
    from ortools.sat.python import cp_model

    options = [s.option for s in scored]
    values = [s.score for s in scored]
    n = len(options)
    if n == 0:
        return Itinerary(order=[], total_value=0.0, total_hours=0.0)

    SCALE = 20  # scale hours -> integer "ticks" (1 tick = 3 minutes) for CP-SAT
    nodes = [None] + options  # index 0 = depot
    travel = [[int(round(estimate_travel_hours(a, b) * SCALE)) for b in nodes] for a in nodes]
    dwell = [0] + [int(round((o.est_hours or 1.0) * SCALE)) for o in options]
    budget_ticks = int(round(budget_hours * SCALE))

    model = cp_model.CpModel()
    N = n + 1  # includes depot at index 0
    y = [model.NewBoolVar(f"y{i}") for i in range(N)]
    model.Add(y[0] == 1)
    x = {}
    for i in range(N):
        for j in range(N):
            if i != j:
                x[i, j] = model.NewBoolVar(f"x{i}_{j}")

    for i in range(N):
        model.Add(sum(x[i, j] for j in range(N) if j != i) == y[i])
        model.Add(sum(x[j, i] for j in range(N) if j != i) == y[i])

    u = [model.NewIntVar(0, N - 1, f"u{i}") for i in range(N)]
    model.Add(u[0] == 0)
    for i in range(1, N):
        for j in range(1, N):
            if i != j:
                model.Add(u[i] - u[j] + (N - 1) * x[i, j] <= N - 2)

    model.Add(
        sum(travel[i][j] * x[i, j] for i in range(N) for j in range(N) if i != j)
        + sum(dwell[i] * y[i] for i in range(1, N))
        <= budget_ticks
    )
    model.Maximize(sum(int(round(values[i - 1] * 1000)) * y[i] for i in range(1, N)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return Itinerary(order=[], total_value=0.0, total_hours=0.0)

    visited = [i for i in range(1, N) if solver.Value(y[i]) == 1]
    visited.sort(key=lambda i: solver.Value(u[i]))
    order = [nodes[i] for i in visited]
    return Itinerary(order=order, total_value=sum(values[i - 1] for i in visited),
                      total_hours=_route_hours(order))

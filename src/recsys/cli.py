"""Interactive CLI: the "open it like a weather app" flow from
project_purpose_and_aim.md -- a short multiple-choice intake, a ranked list
back, and a way to record what happened so the bandit learns.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .content_scoring import COSTS
from .optimizer import Itinerary
from .pipeline import (
    DEFAULT_OPTION_BANK, DEFAULT_STATE, apply_feedback, load_options,
    load_user_state, plan_trip, recommend_next, save_user_state,
)
from .schema import Context


def _choice(prompt: str, options: list[tuple[str, object]]) -> object:
    print(prompt)
    for i, (label, _) in enumerate(options, 1):
        print(f"  {i}. {label}")
    while True:
        raw = input("> ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][1]
        print("Pick a number from the list.")


def intake() -> Context:
    trip_type = _choice("What kind of trip is this?", [
        ("Long trip (a real vacation)", "long_trip"),
        ("Day trip", "day_trip"),
        ("Just hanging out nearby", "hangout_nearby"),
        ("Something else / custom duration", "custom"),
    ])
    energy = _choice("Energy level right now?", [
        ("Wiped out", 0.1), ("Kind of tired", 0.3), ("Normal", 0.5),
        ("Good energy", 0.7), ("Wired / raring to go", 0.9),
    ])
    budget_label = _choice("Budget for this?", [(c, i) for i, c in enumerate(COSTS)])
    _choice_party = _choice("Who's coming?", [
        ("Solo", "solo"), ("Partner", "partner"), ("Friends", "friends"), ("Family", "family"),
    ])
    indoor_outdoor = _choice("Indoor or outdoor?", [
        ("Indoor", -1.0), ("No preference", 0.0), ("Outdoor", 1.0),
    ])
    mood = _choice("Adventurous or relaxed?", [
        ("Relaxed", -1.0), ("No preference", 0.0), ("Adventurous", 1.0),
    ])
    duration_hint = None
    if trip_type == "custom":
        duration_hint = input("Roughly how long (free text, e.g. '5 hours', '4 days')? ").strip()
    return Context(
        trip_type=trip_type, duration_hint=duration_hint, energy=energy,
        budget_level=budget_label, party=_choice_party,
        indoor_outdoor=indoor_outdoor, mood_adventurous=mood,
    )


def _feedback_loop(option, context, state):
    action = _choice(f"Feedback on '{option.label()}'?", [
        ("Did it, would do again", "accepted_repeat"),
        ("Did it, wouldn't repeat", "accepted_no_repeat"),
        ("Skipping this one", "skipped"),
        ("No feedback right now", None),
    ])
    if action:
        apply_feedback(state, option, context, action)
        print("  (noted)")


def run_suggest(args):
    options = load_options(Path(args.bank))
    state = load_user_state(Path(args.state))
    context = intake()
    results = recommend_next(options, context, state, top_k=args.top_k)
    if not results:
        print("Nothing matches right now -- try loosening the budget or trip type.")
        return
    print(f"\nTop {len(results)} suggestions:")
    for i, s in enumerate(results, 1):
        o = s.option
        print(f"\n{i}. {o.label()}  [{o.type}, {o.cost}, ~{o.est_hours or '?'}h]  score={s.score:.2f}")
        if o.why_worth_it:
            print(f"   {o.why_worth_it}")
    pick = input("\nWhich one did you do (number, or blank to skip)? ").strip()
    if pick.isdigit() and 1 <= int(pick) <= len(results):
        _feedback_loop(results[int(pick) - 1].option, context, state)
    save_user_state(state, Path(args.state))


def run_plan(args):
    options = load_options(Path(args.bank))
    state = load_user_state(Path(args.state))
    context = intake()
    budget_hours = args.hours or float(input("Time budget in hours for this plan? ").strip())
    itinerary: Itinerary = plan_trip(options, context, state, budget_hours=budget_hours,
                                      use_milp=not args.no_milp)
    print()
    print(itinerary.summary())
    save_user_state(state, Path(args.state))


def main():
    parser = argparse.ArgumentParser(description="Personal recommender + trip planner")
    parser.add_argument("--bank", default=str(DEFAULT_OPTION_BANK))
    parser.add_argument("--state", default=str(DEFAULT_STATE))
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_suggest = sub.add_parser("suggest", help="What should I do next?")
    p_suggest.add_argument("--top-k", type=int, default=5)
    p_suggest.set_defaults(func=run_suggest)

    p_plan = sub.add_parser("plan", help="Plan a time-boxed outing/trip")
    p_plan.add_argument("--hours", type=float, default=None)
    p_plan.add_argument("--no-milp", action="store_true", help="skip OR-Tools, use greedy+2-opt only")
    p_plan.set_defaults(func=run_plan)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

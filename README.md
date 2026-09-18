# Recommender + Trip Planner (personal)

A personal tool that suggests the **next best activity** and **plans a trip /
itinerary**, built on top of Jack's own `Jack-Master-Travel-Database.xlsx` --
and, as a bonus, an applied-algorithms showcase (hand-written Bayesian
contextual bandit + a real MILP orienteering optimizer).

**Independent, standalone project.** See [`CLAUDE.md`](CLAUDE.md) (bootstrap),
[`recommender-planner-spec.txt`](recommender-planner-spec.txt) (the design
spec), [`project_purpose_and_aim.md`](project_purpose_and_aim.md) (the
original ask), and [`docs/design.md`](docs/design.md) (how those three were
reconciled, and what shipped vs. what was deliberately left out).

## What's built (v1)

1. **Rules + content scoring** (`src/recsys/content_scoring.py`) -- cold
   start ranker, works with zero feedback data.
2. **Thompson-sampling contextual bandit** (`src/recsys/bandit.py`) -- hand-
   written Bayesian linear regression, shared across the whole option bank
   (not one model per arm -- see `docs/design.md` for why). Its prior mean
   *is* the Step 1 rules-based weight vector, so it starts from Jack's own
   hand-set profile and learns a correction, not from nothing.
3. **Orienteering optimizer** (`src/recsys/optimizer.py`) -- greedy + 2-opt
   (instant, always available) and an exact MILP (OR-Tools CP-SAT: flow
   conservation + MTZ subtour elimination + a time-budget constraint).
4. **Polish** -- a diversity (MMR) pass on the suggestion list, non-
   stationarity decay on the bandit's posterior, and a decay/recovery curve
   for "done, would repeat" items. Real weather/geo APIs and a GUI are the
   next polish items, not yet built (see `docs/design.md`).

Content (the option bank) and state (what Jack's done, ratings, the bandit's
learned posterior) are two separate local JSON files, so refreshing the
option bank from an updated workbook never overwrites history.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate  # Windows; use `source .venv/bin/activate` elsewhere
pip install -r requirements.txt
```

## Build the option bank from the workbook

```bash
PYTHONPATH=src python -m recsys.data_loader --xlsx "Jack-Master-Travel-Database.xlsx" --out data/option_bank.json
```

Re-run this any time the workbook changes. It never touches `data/state.json`.

## Use it

```bash
PYTHONPATH=src python -m recsys.cli suggest      # "what should I do next?"
PYTHONPATH=src python -m recsys.cli plan --hours 8   # plan a time-boxed outing
```

Both commands ask a short set of multiple-choice questions, then either
return a ranked shortlist (`suggest`) or a time-feasible ordered itinerary
(`plan`). Recording feedback after `suggest` (did it / would repeat / skip)
updates the bandit for next time.

## Tests

```bash
python -m pytest tests/ -v
```

## Privacy

`data/*.json` and the workbook `.xlsx` files are gitignored -- this is a
single-user tool and that data stays on-device, per `CLAUDE.md`.

## Git

This folder is a fresh local git repo (no remote configured). To push it
somewhere:

```bash
git remote add origin <your-repo-url>
git push -u origin main
```

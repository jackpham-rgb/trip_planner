# Recommender + Trip Planner

A personal tool that suggests what to do next and plans a trip, built on top
of my own travel database. Two algorithms under the hood: a hand-written
Thompson-sampling contextual bandit that learns my taste, and an OR-Tools
MILP optimizer that turns scored options into a time-feasible itinerary.

Details on how it's built and why: [docs/design.md](docs/design.md).

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate  # Windows; `source .venv/bin/activate` elsewhere
pip install -r requirements.txt
```

## Build the option bank from my workbook

```bash
python scripts/build_option_bank.py --xlsx "Jack-Master-Travel-Database.xlsx"
```

Re-run this any time the workbook changes. It never touches `data/state.json`.

## Use it

```bash
PYTHONPATH=src python -m recsys.cli suggest        # what should I do next?
PYTHONPATH=src python -m recsys.cli plan --hours 8 # plan a time-boxed outing
```

Both ask a few multiple-choice questions, then return a ranked shortlist or
an ordered itinerary. Recording feedback after `suggest` (did it / would
repeat / skip) teaches the bandit for next time.

## Tests

```bash
python -m pytest tests/ -v
```

## Privacy

`data/*.json` and the `.xlsx` files are gitignored — this is a single-user
tool and my data stays on-device.

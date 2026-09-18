# CLAUDE.md — Agent bootstrap (standalone project)

This is an **independent personal project**: a recommender + trip planner
("suggest the next best activity / plan a trip"). It is **self-contained and
unrelated to any other work** — do not look for or reference other repos, EECS
coursework docs, or portfolio planning. Everything you need is in this folder.

## Step 1 — Read
`recommender-planner-spec.txt` — the full spec: the two sub-problems (contextual
bandit + orienteering optimizer), the pipeline, formulas, limitations, build
order, and tech stack.

## Step 2 — Build (staged; ship something at each step)
Follow the spec's build order:
1. **Rules + content scoring** — filter candidates + score by feature similarity
   to a profile. Works with zero data (cold start).
2. **Contextual bandit (Thompson Sampling)** — the core; **write it yourself**
   (~40 lines of Bayesian linear regression) — do NOT import a black-box bandit;
   implementing it is the point. Add context features (time/weather/geo).
3. **Orienteering optimizer** — greedy + 2-opt first, then OR-Tools MILP for real
   itineraries with a time budget (and opening-hours time windows).
4. **Polish** — a diversity term, non-stationarity decay, weather/geo APIs, a
   small UI (CLI or Streamlit).
5. **(Only if needed)** RL/MDP for long-horizon multi-day sequencing.

## Conventions
- **Stack:** Python; numpy/scipy; OR-Tools (routing) or PuLP/cvxpy (LP/MILP);
  pandas; optional scikit-learn for the reward model; Streamlit/CLI UI.
- **Repo layout:** `src/` package, `tests/` (pytest), `docs/`, `scripts/`, a
  clear README. Keep logic importable; notebooks for exploration only.
- **Single user / privacy:** it's personal — keep data on-device/local; no
  collaborative filtering (there are no other users, so content + contextual
  bandit is the correct design).
- Small, reviewable commits; clear messages; never force-push.

## Interaction
Proceed with sensible defaults; ask only if a choice is truly irreversible.
When done, summarize what you built, what's stubbed, and the git commands to push.

> TL;DR: read `recommender-planner-spec.txt`, then build in order — rules/content
> scoring → your own Thompson-sampling contextual bandit → an OR-Tools
> orienteering planner → polish. Self-contained; ignore everything outside this folder.

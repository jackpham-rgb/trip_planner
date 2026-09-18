# Design notes

This document records the decisions made while building v1, especially where
the three planning documents (`CLAUDE.md`, `recommender-planner-spec.txt`,
`project_purpose_and_aim.md`) pointed in different directions. Read this
alongside those three for the "why," not just the "what."

## Where the source docs disagreed, and what shipped

**Platform.** `current_setup.md` describes a prior Kotlin + Jetpack Compose +
Room (two-database) Android plan. The spec and `CLAUDE.md` pin the stack to
Python. This build is Python, per the spec -- it's the concrete, buildable
instruction, and a personal CLI tool is a day's work where a native Android
app is a much bigger undertaking. The one idea worth keeping from the Android
plan is architectural, not platform-specific: **content and state are two
separate files** (`data/option_bank.json`, rebuilt from the workbook any
time, vs. `data/state.json`, holding done/repeat history and the bandit's
posterior, never touched by a content rebuild). See `state.py`'s module
docstring -- this was the one design decision current_setup.md flagged as
most important, and it carries over cleanly regardless of platform. If this
ever becomes a phone app, `state.json`'s shape is what a Room "state" table
would hold.

**Scope.** `project_purpose_and_aim.md` proposes a much wider academic
mapping than the spec: personalized PageRank over the option bank, and a full
Markov/MDP model of a trip as a state sequence. Neither shipped:

- *PageRank* would need a real "chosen together" or "near each other" graph.
  There isn't one -- the workbook doesn't record co-occurrence, and building
  a synthetic graph just to run PageRank on it would be ranking-theater, not
  a genuine improvement over content similarity + a learned bandit. Skipped.
- *Full MDP/RL* is explicitly a "usually not" in the spec's own section 4:
  it's justified only when an action's value depends on *future* state
  (multi-day sequencing effects), and is data-hungry for a single user with
  sparse feedback. A contextual bandit + re-planning (already how `plan_trip`
  works -- rescore, re-optimize, no persistent long-horizon state) gets most
  of the benefit for a fraction of the complexity and data. Skipped for v1;
  revisit only if multi-day sequencing effects turn out to matter in
  practice, per the spec's own escalation criterion.
- One idea from that document *did* ship, cheaply: the "interest fades then
  recovers" decay/recovery curve for "done, would repeat" items
  (`state.py::recovery_multiplier`). It's a few lines, and it's also exactly
  what the spec's own non-stationarity section asks for, so it earned its
  place on both documents' terms at once.

**Framing.** Both the spec and `project_purpose_and_aim.md` frame this as a
portfolio artifact demonstrating coursework (EECS126/127/189), which sits
oddly with this folder's own name ("non career related, just life and
hobby") and `CLAUDE.md`'s instruction to ignore portfolio planning. That
framing is treated here as write-up content only -- it explains *why* the
bandit is hand-written and *why* the optimizer is a real MILP instead of a
heuristic-only shortcut, but it didn't add scope beyond what the spec's own
build order calls for.

## What the pieces actually are

- **`schema.py`** -- the controlled vocabularies (`Type`, `Priority`, `Cost`,
  `Viral rating`, ...) are lifted directly from the workbook's own "Lists"
  sheet (its dropdown source), not invented. `OUTDOOR_LEAN`,
  `ENERGY_DEMAND`, `MOOD_ADVENTURE_LEAN` (in `content_scoring.py`) are
  judgment calls -- a rough per-Type lean, not workbook data -- and are the
  most editable/arguable numbers in the codebase.

- **`data_loader.py`** -- the workbook has no stable row IDs and repeats its
  27-column itinerary header once per country/version block, interleaved
  with free-text section titles and small "Field / Detail" overview tables
  that aren't itinerary rows. The parser scans for the header signature
  (`Version`, `Day` in the first two cells) and reads data rows below it
  until the block runs out, tagging each row with the nearest preceding
  short title as `section`. IDs are derived from
  `sheet + section + destination + counter`, so the same real row gets the
  same ID across rebuilds -- this is what lets `state.json` survive a
  content refresh, and is a lighter-weight version of the
  `JP-S2W-D06-03`-style stable ID current_setup.md flagged as a blocker for
  the Android plan.

- **`content_scoring.py`** -- Step 1 (cold start) and Step 2 (the bandit)
  share one feature function, `featurize(option, context)`. Step 1 dots it
  against a hand-set weight vector (`DEFAULT_WEIGHTS`); Step 2 dots it
  against a *learned* weight vector whose prior mean is that same
  `DEFAULT_WEIGHTS` vector. So the bandit doesn't start from nothing -- it
  starts at Jack's own hand-set profile and learns a correction from there.
  This is deliberately the "hybrid: content prior + bandit exploration"
  cold-start strategy the spec calls for, not a coincidence.

- **`bandit.py`** -- one shared linear-Gaussian Bayesian regression, not one
  per arm. The option bank has hundreds of entries and keeps growing per
  `project_purpose_and_aim.md`'s "should grow on its own over time" goal;
  a brand-new item would never get enough repeat exposure to fit its own
  per-arm regression. A shared feature-based model is exactly why the spec
  argues contextual bandits beat per-arm bandits at cold start -- a new
  item's score comes from its features overlapping with items already
  rated, not from its own trial history. Non-stationarity is handled with a
  discount `gamma` on the sufficient statistics (`A`, `b`) at each update,
  pulling old evidence back toward the prior instead of letting it lock in
  forever.

- **`optimizer.py`** -- both a greedy + 2-opt heuristic (always available,
  instant) and the exact MILP from the spec (flow conservation + MTZ subtour
  elimination + a time-budget constraint), solved with OR-Tools CP-SAT.
  `estimate_travel_hours` is a placeholder based on shared city/section text
  fields, not real travel times -- there's no live geo/traffic API wired up
  yet, and most workbook rows don't have lat/lon. Swapping in a real
  distance-matrix API later only means replacing that one function; nothing
  else in the optimizer depends on how the number was produced.

- **`content_scoring.py::diversify`** -- a plain MMR pass (penalize repeating
  a `Type` already chosen) used only for the shown suggestion list
  (`recommend_next`), not for `plan_trip`'s candidate pool -- an itinerary
  optimizer maximizing value over a time budget is a different, legitimate
  objective from "don't show five near-identical hikes in a row." Real
  early data (`data/option_bank.json` scored against a nature/outdoor
  context) showed the top 5 unfiltered suggestions were all `Nature` before
  this was added -- this isn't a hypothetical concern, it happened on the
  first real run.

## Explicitly out of scope for this build

- Weather/geo API integration (spec's own Step-4 polish item; the workbook
  has `Seasonal?` / `Weather Dep.?` flags but no live data source yet).
- A GUI (Streamlit or otherwise) -- the CLI (`recsys.cli`) satisfies the
  spec's "small UI (CLI or Streamlit)" requirement for v1; a Streamlit layer
  on top of `pipeline.py` would not require touching the core logic.
- Lat/lon-based real travel times -- most workbook rows don't have
  coordinates yet; `optimizer.py` is written so this is a one-function swap
  later.
- PageRank, full Markov/MDP -- see "Where the source docs disagreed" above.

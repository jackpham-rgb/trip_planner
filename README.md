# Trip Planner

A personal recommender + trip planner: I open a link, answer a few
multiple-choice questions, and get back the top 3 things I should do next --
or a whole time-boxed itinerary -- pulled from my own real travel database
and ranked by an algorithm that learns my taste over time. No login, no
server to start, no app to install (though it can install itself as one).
Just a link.

**Live demo:** open `index.html` on GitHub Pages once it's deployed (see
[Setup](#setup) below), or run it locally in thirty seconds:

```bash
python -m http.server 8000    # any static file server works
```

then open `http://localhost:8000/`.

- [`test.html`](test.html) -- the algorithm's own test suite, runnable in a browser tab
- [`simulation.html`](simulation.html) -- a live chart of *why* a bandit beats guessing

---

## 1. What this is

Three multiple-choice-and-slider questions, then a ranked shortlist of real
places pulled from a 279-entry option bank I built from my own travel
spreadsheet (see [Data](#5-the-data)). Mark what I actually did and whether
I'd repeat it, and the ranking quietly gets better -- without ever asking me
to rate hundreds of things up front, and without me ever creating an account.

It also plans a whole outing: give it a time budget and it returns an
ordered stop list that fits, not just a single suggestion.

## 2. Quickstart

Cloning the repo and opening `index.html` through a local static server (see
above) works immediately, with no setup: it uses the browser's own
`localStorage` to remember what I've done, and everything else (the
279 destinations, the ranking math) runs entirely client-side.

To get real cross-device sync (open it on my phone, then my laptop, see the
same history) I need my own free Firebase project -- a few minutes, walked
through in [Setup](#setup).

## 3. How it's built

```
                    ┌─────────────────────┐
   my phone/laptop  │   index.html/app.js  │   a plain static site --
   (any browser) ──▶│   (the UI)            │   no backend, no build step
                    └──────────┬───────────┘
                               │
              ┌────────────────┼─────────────────┐
              ▼                ▼                  ▼
     data/option_bank.json  recommender.js   Firebase Realtime DB
     (279 real places,      (all the math:    or, with no Firebase
      static file, never    scoring, the      project configured,
      changes at runtime)   bandit, trip      the browser's own
                             planning)         localStorage
```

- **GitHub Pages** serves the site: push HTML/CSS/JS, get a public URL,
  no server to run or pay for. The tradeoff is that it can only serve files
  as-is -- it cannot run a backend, which is why this version has no Python
  server (an earlier version did; see the [design journal](#8-design-journal)).
- **`recommender.js`** holds every bit of the actual algorithm and nothing
  else -- no DOM access, no network calls -- so it's the one file both
  `index.html` (the real app) and `test.html`/`simulation.html` (verification
  and demo) all load and trust equally.
- **Firebase Realtime Database** is the cloud database, used only if I've
  set up my own project (see [Setup](#setup)); otherwise everything falls
  back to `localStorage` automatically, so the app is never broken, just
  less synced.
- **Content and state are two different things, on purpose.** The option
  bank (the 279 places) is a static file that ships with the site and only
  changes when I rebuild it from my workbook. My history -- what I've done,
  rated, and want to repeat, plus the bandit's learned weights -- is the
  *only* thing that goes in Firebase/localStorage. Rebuilding the option
  bank from an updated workbook can never silently wipe out that history,
  because they're never in the same place.

## 4. The algorithm

### Step 1 -- cold start (rules)

Every candidate gets turned into an 8-number feature vector -- how well its
category matches my indoor/outdoor and energy preferences, whether it fits
my budget, its priority and "actually worth it" rating from the workbook,
and whether it's quick enough for a short outing. On day one, with zero
history, those 8 numbers get dotted against a hand-set weight vector to
produce a score. This is `rankByRules` in `recommender.js`.

### Step 2 -- learning from feedback (the bandit)

This is the part I actually wrote from scratch rather than importing:
**Thompson sampling** over a **Bayesian linear regression**, which is the
standard way to solve a *contextual bandit* -- the formal name for "I only
find out how much I liked the thing I actually picked, and I have to balance
trying new things against sticking with what's worked."

The math (linear-Gaussian conjugate Bayesian linear regression):

```
posterior:   theta ~ N(theta_hat, sigma^2 * A^-1)
theta_hat  = A^-1 b
A          = lambda*I + sum_t phi_t phi_t^T
b          = lambda*I*prior_mean + sum_t r_t * phi_t
```

`phi_t` is that same 8-number feature vector from Step 1. `theta` is a
belief about how much each feature matters -- a posterior *distribution*,
not just a single best guess, which is the whole point. To decide, the
algorithm doesn't take the average of that belief; it draws one random
sample from it (`theta_tilde`) and ranks candidates by `phi . theta_tilde`.
A feature combination I've rarely encountered has a *wide* posterior, so its
sampled value swings more -- which means it occasionally jumps to the top
even with no hard-coded "explore 10% of the time" rule. Exploration falls
out of uncertainty, for free.

Crucially, there's **one shared model, not one bandit per place**. With 279
entries (and growing) and just one person providing feedback, no single
place would ever get enough repeat visits to learn about *it specifically*.
Instead, everything shares one model over the *features*, so a brand-new
canyon hike I've never seen gets a reasonable score immediately, because it
shares "Nature, outdoor, $$, MUST-priority" with hikes I've already rated.
This is also why `theta`'s prior mean is set to Step 1's hand-picked weights:
the bandit doesn't start from nothing, it starts from my own judgment and
*learns a correction* to it.

**Taste drifts.** A discount factor (`gamma = 0.98` on every update) shrinks
old evidence back toward the prior over time, so a phase I was in six months
ago doesn't permanently outvote what I actually want now.

### Step 3 -- planning a whole trip (orienteering)

Given a time budget and a list of scored candidates, choosing which subset
to visit *and in what order* to maximize total value is the **orienteering
problem** -- provably NP-hard. `greedyTwoOpt` in `recommender.js` builds a
route by repeatedly inserting whichever candidate gives the best value added
per hour it costs, then runs **2-opt** (repeatedly reverse a segment of the
route if that shortens it) until nothing improves. Fast, always available,
good enough for a single day's plan.

*(An earlier Python version of this project also had an exact
solver -- a real mixed-integer program via Google OR-Tools, guaranteed
optimal, not just "good" -- see the [design journal](#8-design-journal) for
why that's not in this version.)*

### Step 4 -- polish

- **Diversity.** A pure top-3-by-score list tends to be three near-identical
  hikes. `diversify` penalizes repeating a category already picked, so the
  shortlist actually spans different kinds of things.
- **Interest decay and recovery.** Something I did and loved doesn't
  disappear from the pool forever, but it also shouldn't be suggested again
  tomorrow. `recoveryMultiplier` models interest as `1 - (1-floor) * e^(-t/tau)`
  -- low right after, climbing back toward normal over about a month.
- **Location and weather, as defaults, not filters.** The option bank
  doesn't have a precise latitude/longitude for most entries (see
  [Limitations](#6-known-limitations)), and running real geodesic distance
  math against 279 places for a single-user tool is effort spent on a
  precision the data can't back up. So: the Geolocation API gets my
  coordinates once, a free reverse-geocoding call turns that into a city/zip
  label (cheap, coarse, enough to sanity-check "does 'hangout nearby' make
  sense from here"), and a free weather call pre-fills the indoor/outdoor
  question based on current temperature and precipitation. Both are just
  starting points for the same dropdowns I could set myself -- never a
  silent override.

## 5. The data

`data/option_bank.json` -- 279 real destinations, parsed by
`tools/build_option_bank.py` from my own hand-built travel spreadsheet (19
tabs, built up over months; not included in this repo -- see
[Privacy](#7-privacy)). The parser is tolerant of that workbook's organic
layout: the same 27-column itinerary header repeats once per
country/trip-length block, interleaved with free-text section titles and
small "Field / Detail" summary tables that aren't itinerary rows at all. It
assigns each row a stable ID (`sheet-section-destination-counter`) so my
history stays attached to the right place even after the workbook changes
and the bank gets rebuilt.

Each entry's fields (type, cost, priority, "viral rating," estimated hours,
etc.) use the exact controlled vocabulary from the workbook's own dropdown
list, mirrored in both `tools/schema.py` (Python, for the build step) and
the top of `recommender.js` (JavaScript, for runtime) -- keep both in sync
if the workbook's categories ever change.

### Firebase tree layout

Everything that changes at runtime lives under one path:

```
trip_planner/
  state/
    items/
      <option-id>/
        status: "done_repeat" | "done_no_repeat"
        timesDone: number
        lastDone: "YYYY-MM-DD"
        rating: number | null
        notes: string
    bandit/
      A: number[8][8]        # the posterior's sufficient statistics
      b: number[8]
      featureNames: string[8]
      nUpdates: number
```

## 6. Known limitations

Written down on purpose, the same way the earlier Python version's
`docs/history/design.md` did -- an honest list is worth more than a
confident-sounding one:

- **Location is zip/city-level, not per-place distance.** Most of the 279
  entries don't have coordinates, so nothing here computes "how far is this
  specific hike from me" -- only "does this general trip type make sense
  from around here."
- **The security model keeps out casual access, not a determined reader.**
  Firebase's web config isn't a secret the way a server API key is (Firebase
  says so too) -- it's visible to anyone who reads the page's source. Access
  control is Anonymous Auth + a rule requiring "signed in" (see
  [Setup](#setup)), which blocks scraping bots, not someone who deliberately
  copies the config out and calls the database directly. Fine for a personal
  hackathon project; not a template for anything holding real secrets.
- **One shared data path, not one per device/user.** This is deliberate (see
  the [design journal](#8-design-journal)), but it does mean there's no
  actual access boundary between "my phone" and "anyone who has my deployed
  link and reads the page source."
- **Weather/location are defaults, not hard filters.** They pre-fill a
  dropdown; they never silently remove an option from consideration.
- **No exact optimizer.** Trip planning uses a fast heuristic (greedy +
  2-opt), not a solver that guarantees the mathematically best possible
  itinerary. For a single day's plan the difference is marginal.
- **This is sized for one person.** No accounts, no multi-tenancy, and the
  bandit is a single shared model by design -- see Step 2 above for why
  that's the *right* call here, not a shortcut.

## 7. Privacy

The raw workbook (`Jack-Master-Travel-Database.xlsx`) never leaves my
machine -- it's gitignored, and only `tools/build_option_bank.py` ever reads
it, offline, to produce `data/option_bank.json`. That parsed option bank
*is* committed and public: GitHub Pages is static, so the browser has to be
able to fetch it directly -- there's no backend left to keep it private
server-side. My actual history (what I've done, rated, and written notes
about) lives only in Firebase/localStorage, never in git.

Making the GitHub repo private hides the *source code* from casual
browsing. It does **not** hide the deployed site -- that stays reachable by
anyone with the link, which is required for "no login, just open it
anywhere" to work at all.

## Setup

### Run it locally, no account needed

```bash
git clone <this-repo>
cd trip-planner
python -m http.server 8000
```

Open `http://localhost:8000/`. History saves to that browser's
`localStorage` only.

### Your own Firebase project (for real cross-device sync)

1. Go to [console.firebase.google.com](https://console.firebase.google.com),
   create a free project.
2. **Build -> Realtime Database -> Create Database** (start in locked mode).
3. **Build -> Authentication -> Sign-in method -> Anonymous -> Enable.**
4. Paste the contents of [`firebase-rules.json`](firebase-rules.json) into
   **Realtime Database -> Rules**, and publish.
5. **Project settings -> General -> Your apps -> Web app (`</>`)** to
   register a web app; copy the `firebaseConfig` object it gives you.
6. Paste those values into [`firebase-config.js`](firebase-config.js),
   replacing the placeholders.

The app detects real values automatically (`window.FIREBASE_IS_CONFIGURED`)
and switches from `localStorage` to Firebase with no other changes.

### Deploy to GitHub Pages

**Settings -> Pages -> Source: `main` branch, `/ (root)`.** That's the whole
deploy step -- no build, no Action, no server.

If I want the *source* private while the *site* stays reachable by link (see
[Privacy](#7-privacy) for why those are different things): **Settings ->
Danger Zone -> Change visibility -> Private.**

### Rebuild the option bank after editing the workbook

```bash
pip install -r tools/requirements.txt
python tools/build_option_bank.py --xlsx "Jack-Master-Travel-Database.xlsx"
```

Never touches Firebase or localStorage -- content and state stay separate,
per Section 3.

### Tests

```bash
pip install -r tools/requirements.txt
python -m pytest tests/ -v      # the one Python piece left: the workbook parser
```

Open [`test.html`](test.html) in any browser for the JavaScript side (the
bandit, content scoring, trip planner) -- no npm, no build step, just a page
that runs assertions and prints pass/fail.

## 8. Design journal

The final design above reads like it was obvious from the start. It wasn't
-- here's what actually happened, including the parts that turned out to be
wrong.

**It started as a Python CLI.** Rules ranker, then the Thompson-sampling
bandit, then an orienteering optimizer -- built and tested in that order,
because that's what a first coursework-driven spec called for. It worked. It
also only ran on one laptop, and needed a terminal open.

**Wanting it on my phone forced the real rethink.** A backend can't live on
GitHub Pages. The choice was either give up free static hosting, or move the
math into the browser. I moved the math -- which meant retiring the Python
version's exact MILP trip solver (Google OR-Tools; genuinely optimal, not
just good) since OR-Tools has no practical way to run in a browser. The
greedy + 2-opt heuristic was already built as that version's "instant,
always-available" fallback, so nothing about the demo's honesty changed --
it was already the thing that ran when an exact solve wasn't worth the wait.

**The first cloud data model was wrong, and fixing it made things simpler.**
My first instinct, copied from how most Firebase tutorials show it, was to
store data per signed-in user: `/users/{id}/...`. Then I actually thought
through what "open the same link on my phone and my laptop" requires:
Firebase Anonymous Auth hands out a *different* invisible ID per browser and
device. Partitioning by that ID would mean my phone and laptop each get
their *own empty* history -- exactly backwards from what mattered most.
Since this is genuinely single-user, the fix wasn't a cleverer per-user
scheme -- it was to delete the per-user idea entirely and use one shared
path. Simpler *and* correct, which doesn't happen often.

**Fine-grained GPS distance-ranking got cut once I looked at the data.**
The plan on paper was hardware-precise location driving hardware-precise
"nearest to me" ranking. Two problems showed up fast: most of the 279
entries don't have coordinates, and computing geodesic distance against all
of them on every request is real computation for a signal that a zip code
already captures at the scale a personal tool like this actually needs.
Zip/city-level location, cheap and automatic, does the honest version of the
same job.

**PageRank over the option bank and a full Markov/MDP model of a trip** were
both on the table early -- they show up in the earlier planning docs
(archived in `docs/history/`) as genuinely interesting ideas. Neither
survived contact with what the data could support: PageRank wants a real
"chosen together" or "near each other" graph, which doesn't exist here, and
building a synthetic one just to run PageRank on it would be ranking-theater
rather than a real improvement over content similarity plus a learned
bandit. A full MDP is justified when *today's* choice changes what's optimal
*tomorrow* -- multi-day sequencing effects -- and is data-hungry in a way
that doesn't fit one person's sparse feedback. A contextual bandit that
re-plans each time gets most of the real benefit for a fraction of the
complexity and data.

**The one idea that did survive from that early brainstorming:** the
interest decay/recovery curve for "done, would repeat" places (Step 4
above). It's a few lines of calculus-shaped code, and it turned out to
directly satisfy the bandit's own need for non-stationarity handling --
which is the rare case where the "sounds nice academically" idea and the
"actually needed for the math to behave" idea were the same idea.

**Try it live:** [`simulation.html`](simulation.html) runs the actual
`ThompsonBandit` class from `recommender.js` against a synthetic problem and
charts the regret curve in real time -- the "why a bandit" argument, shown
instead of just asserted.

## Repository map

```
index.html, app.js, styles.css   -- the app: UI, storage glue, geolocation/weather
recommender.js                    -- all the algorithm math, dependency-free
firebase-config.js                -- paste your own Firebase project config here
firebase-rules.json                -- security rules to paste into Firebase console
manifest.json, icon.svg,
  service-worker.js                -- PWA installability + offline app shell
test.html                          -- in-browser test suite for recommender.js
simulation.html                    -- live regret-curve demo
data/option_bank.json              -- the content: 279 real destinations (static, public)
tools/
  build_option_bank.py             -- offline: workbook -> option_bank.json
  schema.py                        -- the Option record + controlled vocabulary
  requirements.txt
tests/
  test_data_loader.py              -- covers the one Python piece still shipped
docs/history/                       -- archived planning docs from earlier iterations
```

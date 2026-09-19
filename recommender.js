/**
 * recommender.js -- the whole recommendation engine, ported line-for-line
 * from the Python prototype (see docs/history/ for the original .py files
 * and design.md for the reasoning). No dependencies, no build step: this is
 * a plain script that defines a global `Recommender` object, so it works
 * equally well opened straight from disk (file://) or served from GitHub
 * Pages.
 *
 * Sections, in the order a reader would want them:
 *   1. Vocabularies         -- the workbook's own controlled vocabulary
 *   2. Linear algebra        -- tiny helpers (solve/invert/Cholesky) the
 *                               bandit needs and JS has no built-in for
 *   3. Content scoring        -- Step 1: cold-start rules ranker
 *   4. Thompson-sampling bandit -- Step 2: the learned ranker
 *   5. Diversity pass         -- Step 4 polish: no five-hikes-in-a-row
 *   6. Trip optimizer          -- Step 3: greedy + 2-opt (the MILP solver
 *                               from the Python version needs OR-Tools,
 *                               which has no browser story -- see README)
 *   7. Recovery curve         -- "done, would repeat" interest decay/recovery
 */
(function (global) {
  "use strict";

  // ======================================================================
  // 1. Vocabularies -- copied from tools/schema.py, which is itself copied
  //    from the workbook's own "Lists" sheet. Keep these two files in sync
  //    if the workbook's dropdowns ever change.
  // ======================================================================

  const COSTS = ["Free", "$", "$$", "$$$", "$$$$"]; // ordinal, low -> high
  const PRIORITIES = ["OPTIONAL", "HIGH", "MUST"]; // ordinal, low -> high
  const VIRAL_RATINGS = [
    "Skip unless personally interested",
    "Not a social-media place",
    "Mostly photo opportunity",
    "Worth it if nearby",
    "Actually worth it",
  ]; // ordinal, low -> high "genuinely worth it"

  // Judgment-call vocabularies (not workbook data) -- a rough per-Type lean,
  // the same three used by the Python content_scoring.py.
  const OUTDOOR_LEAN = {
    Historical: 0.3, Nature: 1.0, Food: -0.2, Architecture: 0.2,
    Shopping: -0.6, Nightlife: -0.3, Cultural: 0.0, Technology: -0.5,
    Automotive: 0.0, "Anime/pop culture": -0.4, Photography: 0.4,
    Adventure: 0.9, Relaxation: 0.1, "Festival/event": 0.3,
    "Social-media-famous": 0.2, "Transport experience": 0.1,
    "Academic/research": -0.5,
  };
  const ENERGY_DEMAND = {
    Historical: 0.4, Nature: 0.6, Food: 0.2, Architecture: 0.3,
    Shopping: 0.4, Nightlife: 0.6, Cultural: 0.3, Technology: 0.2,
    Automotive: 0.3, "Anime/pop culture": 0.3, Photography: 0.4,
    Adventure: 0.9, Relaxation: -0.7, "Festival/event": 0.6,
    "Social-media-famous": 0.3, "Transport experience": 0.2,
    "Academic/research": 0.1,
  };
  const MOOD_ADVENTURE_LEAN = {
    Historical: -0.1, Nature: 0.3, Food: 0.0, Architecture: -0.2,
    Shopping: -0.3, Nightlife: 0.6, Cultural: -0.1, Technology: 0.1,
    Automotive: 0.4, "Anime/pop culture": 0.2, Photography: 0.1,
    Adventure: 1.0, Relaxation: -0.9, "Festival/event": 0.7,
    "Social-media-famous": 0.2, "Transport experience": 0.3,
    "Academic/research": -0.4,
  };

  // Which region sheets are even reachable for a given trip type. Dwell time
  // ("time needed" on one attraction) is NOT a proxy for how far away it is
  // -- a short Tokyo shrine visit is not a valid "hangout nearby" suggestion
  // just because the visit itself is quick. So trip-type scoping filters by
  // SHEET (i.e. by region) first.
  const TRIP_TYPE_SHEETS = {
    hangout_nearby: new Set(["Cali"]),
    day_trip: new Set(["Cali", "Weekend Trips"]),
    long_trip: new Set([
      "USA", "USA MEX CAN", "South America", "Europe", "East Asia",
      "South East Asia", "Middle East", "Africa", "Rest of World",
    ]),
  };

  const FEATURE_NAMES = [
    "bias", "indoor_outdoor_match", "energy_match", "mood_match",
    "budget_fit", "priority_level", "viral_level", "short_and_easy",
  ];

  // Hand-set weights for the cold-start ranker (Step 1) and the bandit's
  // prior mean (Step 2). Tuned by inspection, not fit to data.
  const DEFAULT_WEIGHTS = [0.0, 1.2, 1.0, 1.0, 1.5, 0.8, 0.6, 0.5];

  // ======================================================================
  // 2. Linear algebra -- JS has no numpy. These are small, dense-matrix-
  //    only helpers (the bandit's matrices are 8x8: one entry per feature).
  //    Not written for speed at scale -- written to be obviously correct
  //    at the one scale this app ever runs at.
  // ======================================================================

  function dot(x, y) { return x.reduce((s, v, i) => s + v * y[i], 0); }
  function vecAdd(x, y) { return x.map((v, i) => v + y[i]); }
  function vecScale(x, s) { return x.map((v) => v * s); }
  function matVec(A, x) { return A.map((row) => dot(row, x)); }
  function matAdd(A, B) { return A.map((row, i) => row.map((v, j) => v + B[i][j])); }
  function matScale(A, s) { return A.map((row) => row.map((v) => v * s)); }
  function outer(x, y) { return x.map((xi) => y.map((yj) => xi * yj)); }
  function identity(n) {
    return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  }

  /** Solve A x = b via Gaussian elimination with partial pivoting. */
  function solveLinear(A, b) {
    const n = A.length;
    const M = A.map((row, i) => [...row, b[i]]);
    for (let col = 0; col < n; col++) {
      let piv = col;
      for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
      [M[col], M[piv]] = [M[piv], M[col]];
      const pivVal = M[col][col];
      for (let j = col; j <= n; j++) M[col][j] /= pivVal;
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const factor = M[r][col];
        for (let j = col; j <= n; j++) M[r][j] -= factor * M[col][j];
      }
    }
    return M.map((row) => row[n]);
  }

  /** Matrix inverse via Gauss-Jordan on the augmented [A | I] matrix. */
  function invert(A) {
    const n = A.length;
    const I = identity(n);
    const M = A.map((row, i) => [...row, ...I[i]]);
    for (let col = 0; col < n; col++) {
      let piv = col;
      for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
      [M[col], M[piv]] = [M[piv], M[col]];
      const pivVal = M[col][col];
      for (let j = 0; j < 2 * n; j++) M[col][j] /= pivVal;
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const factor = M[r][col];
        for (let j = 0; j < 2 * n; j++) M[r][j] -= factor * M[col][j];
      }
    }
    return M.map((row) => row.slice(n));
  }

  /** Cholesky decomposition: cov = L L^T, L lower-triangular. */
  function cholesky(A) {
    const n = A.length;
    const L = identity(n).map((row) => row.map(() => 0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = 0;
        for (let k = 0; k < j; k++) sum += L[i][k] * L[j][k];
        if (i === j) {
          L[i][j] = Math.sqrt(Math.max(A[i][i] - sum, 1e-12));
        } else {
          L[i][j] = (A[i][j] - sum) / L[j][j];
        }
      }
    }
    return L;
  }

  /** A tiny seedable PRNG (mulberry32) so the simulation demo is reproducible.
   * The live app uses Math.random by default -- see makeRng(). */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function makeRng(seed) { return seed === undefined ? Math.random : mulberry32(seed); }

  function gaussianRandom(rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function sampleMultivariateNormal(mean, cov, rng) {
    const L = cholesky(cov);
    const z = mean.map(() => gaussianRandom(rng));
    return vecAdd(mean, matVec(L, z));
  }

  // ======================================================================
  // 3. Content scoring -- Step 1 (cold start, works with zero feedback).
  // ======================================================================

  function ordinal(value, vocab) {
    if (value === null || value === undefined) return null;
    const needle = String(value).trim().toLowerCase();
    const idx = vocab.findIndex((v) => v.toLowerCase() === needle);
    return idx === -1 ? null : idx;
  }

  /** Map an ordinal index in [0, size-1] to [-1, 1]; missing -> 0 (neutral). */
  function toPm1(index, size) {
    if (index === null || size <= 1) return 0.0;
    return (2.0 * index) / (size - 1) - 1.0;
  }

  /** Cheap retrieval: drop what's out of scope before anyone scores anything. */
  function filterCandidates(options, context, excludedIds) {
    const budgetIdx = Math.max(0, Math.min(context.budgetLevel, COSTS.length - 1));
    const allowedSheets = TRIP_TYPE_SHEETS[context.tripType]; // undefined for "custom" -> no restriction
    return options.filter((o) => {
      if (excludedIds.has(o.id)) return false;
      const costIdx = ordinal(o.cost, COSTS);
      if (costIdx !== null && costIdx > budgetIdx) return false;
      if (allowedSheets && !allowedSheets.has(o.source_sheet)) return false;
      if (context.tripType === "hangout_nearby" || context.tripType === "day_trip") {
        if (o.est_hours !== null && o.est_hours !== undefined && o.est_hours > 30) return false;
      }
      return true;
    });
  }

  /** phi(option, context): the feature vector every scorer (rules or bandit) uses. */
  function featurize(option, context) {
    const outdoorLean = OUTDOOR_LEAN[option.type] ?? 0.0;
    const energyDemand = ENERGY_DEMAND[option.type] ?? 0.0;
    const moodLean = MOOD_ADVENTURE_LEAN[option.type] ?? 0.0;

    const contextEnergyPm1 = context.energy * 2.0 - 1.0;
    const indoorOutdoorMatch = 1.0 - Math.abs(outdoorLean - context.indoorOutdoor) / 2.0;
    const energyMatch = 1.0 - Math.abs(energyDemand - contextEnergyPm1) / 2.0;
    const moodMatch = 1.0 - Math.abs(moodLean - context.moodAdventurous) / 2.0;

    const costIdx = ordinal(option.cost, COSTS);
    const budgetIdx = Math.max(0, Math.min(context.budgetLevel, COSTS.length - 1));
    let budgetFit;
    if (costIdx === null) {
      budgetFit = 0.5;
    } else {
      const headroom = budgetIdx - costIdx;
      budgetFit = headroom >= 0 ? 1.0 : Math.max(0.0, 1.0 + headroom / COSTS.length);
    }

    const priorityLevel = toPm1(ordinal(option.priority, PRIORITIES), PRIORITIES.length);
    const viralLevel = toPm1(ordinal(option.viral_rating, VIRAL_RATINGS), VIRAL_RATINGS.length);

    let shortAndEasy = 0.0;
    if ((context.tripType === "hangout_nearby" || context.tripType === "day_trip") && option.est_hours) {
      shortAndEasy = option.est_hours <= 6 ? 1.0 : -0.5;
    }

    return [1.0, indoorOutdoorMatch, energyMatch, moodMatch, budgetFit, priorityLevel, viralLevel, shortAndEasy];
  }

  /** Step 1: score every candidate by feature similarity to a hand-set profile. */
  function rankByRules(options, context, weights = DEFAULT_WEIGHTS) {
    const scored = options.map((o) => {
      const phi = featurize(o, context);
      return { option: o, score: dot(weights, phi), features: phi };
    });
    scored.sort((a, b) => b.score - a.score);
    return scored;
  }

  // ======================================================================
  // 4. Thompson-sampling bandit -- Step 2, the learned ranker.
  //
  // One SHARED model, not one per arm. "Arms" are individual activities in
  // an option bank with hundreds of entries that keeps growing, each seen
  // only a handful of times by one person -- there's never enough exposure
  // to fit a separate regression per item. A single theta over the shared
  // feature vector phi(option, context) means a brand-new item gets a
  // reasonable score immediately, because it shares features with items
  // already rated.
  //
  // Math (linear-Gaussian conjugate Bayesian linear regression):
  //   posterior:  theta ~ N(theta_hat, sigma^2 A^-1)
  //   theta_hat = A^-1 b
  //   A = lambda*I + sum_t phi_t phi_t^T
  //   b = lambda*I*prior_mean + sum_t r_t * phi_t   (so theta_hat == prior_mean with no data)
  //   act: sample theta_tilde ~ posterior, then argmax_a phi(a)^T theta_tilde
  // Exploration falls out of posterior uncertainty automatically -- a
  // feature combination seen rarely has a wide posterior, so its sampled
  // theta_tilde swings more, occasionally surfacing it without any
  // hard-coded epsilon-exploration.
  // ======================================================================

  class ThompsonBandit {
    constructor(featureNames, priorMean, lambdaReg = 2.0, sigma = 1.0) {
      this.featureNames = featureNames.slice();
      this.priorMean = priorMean.slice();
      this.lambdaReg = lambdaReg;
      this.sigma = sigma;
      const d = featureNames.length;
      this.A = matScale(identity(d), lambdaReg);
      this.b = matVec(this.A, this.priorMean);
    }

    get thetaHat() { return solveLinear(this.A, this.b); }

    sampleTheta(rng) {
      const cov = matScale(invert(this.A), this.sigma * this.sigma);
      return sampleMultivariateNormal(this.thetaHat, cov, rng);
    }

    /** One posterior sample, scored against every candidate -- this IS the
     * explore/exploit step. Returns scores in the same order as `phis`. */
    rank(phis, rng) {
      const theta = this.sampleTheta(rng);
      return phis.map((phi) => dot(phi, theta));
    }

    /** Non-stationarity (taste drift): `gamma < 1` shrinks A and b toward
     * the prior before folding in the new observation, so old evidence
     * decays instead of locking in forever. */
    update(phi, reward, gamma = 1.0) {
      if (gamma < 1.0) {
        const d = phi.length;
        const priorA = matScale(identity(d), this.lambdaReg);
        const priorB = matVec(priorA, this.priorMean);
        this.A = matAdd(matScale(this.A, gamma), matScale(priorA, 1 - gamma));
        this.b = vecAdd(vecScale(this.b, gamma), vecScale(priorB, 1 - gamma));
      }
      this.A = matAdd(this.A, outer(phi, phi));
      this.b = vecAdd(this.b, vecScale(phi, reward));
    }

    toPosterior(nUpdates) {
      return { A: this.A, b: this.b, featureNames: this.featureNames, nUpdates };
    }

    static fromPosterior(posterior, priorMean, lambdaReg = 2.0, sigma = 1.0) {
      const bandit = new ThompsonBandit(posterior.featureNames, priorMean, lambdaReg, sigma);
      bandit.A = posterior.A;
      bandit.b = posterior.b;
      return bandit;
    }
  }

  /** Turn feedback into a scalar reward. An explicit rating wins when given
   * (cleaner, if sparser, than an implicit action); otherwise fall back to
   * the coarser action label. */
  function rewardFromFeedback(action, rating) {
    if (rating !== null && rating !== undefined) {
      return Math.max(-1.0, Math.min(1.0, (rating - 3.0) / 2.0)); // 1..5 -> -1..1
    }
    const table = { accepted_repeat: 1.0, accepted_no_repeat: -0.3, skipped: -0.5 };
    return table[action] ?? 0.0;
  }

  // ======================================================================
  // 5. Diversity pass -- Step 4 polish. A plain top-K ranking by score alone
  //    tends to tunnel into one category (five nature hikes in a row); this
  //    is a cheap Maximal-Marginal-Relevance pass that penalizes repeating
  //    a Type already chosen. Used only for the shown suggestion list, not
  //    for the trip optimizer's candidate pool -- an itinerary maximizing
  //    value over a time budget is a different, legitimate objective.
  // ======================================================================

  function diversify(scored, topK, penalty = 1.0) {
    if (topK >= scored.length) return [...scored].sort((a, b) => b.score - a.score);
    const pool = [...scored];
    const chosen = [];
    while (pool.length && chosen.length < topK) {
      let bestIdx = 0, bestVal = -Infinity;
      pool.forEach((s, i) => {
        const sameTypeCount = chosen.filter((c) => c.option.type === s.option.type).length;
        const val = s.score - penalty * sameTypeCount;
        if (val > bestVal) { bestVal = val; bestIdx = i; }
      });
      chosen.push(pool[bestIdx]);
      pool.splice(bestIdx, 1);
    }
    return chosen;
  }

  // ======================================================================
  // 6. Trip optimizer -- Step 3, the orienteering problem: choose a SUBSET
  //    and an ORDER of scored candidates that maximizes total value without
  //    exceeding a time budget. NP-hard in general.
  //
  //    The Python version also had an exact MILP solver (OR-Tools CP-SAT:
  //    flow conservation + MTZ subtour elimination + a time-budget
  //    constraint) -- dropped here because OR-Tools has no practical
  //    browser/JS equivalent. This greedy + 2-opt heuristic was already the
  //    "always available, instant" fallback in that version, so nothing
  //    about the demo's honesty changes: it's flagged the same way here.
  //
  //    SIMPLIFICATION (documented, not hidden): there's no real address for
  //    "home" and no live travel-time API, so `estimateTravelHours` is a
  //    heuristic based on shared city/section text, including a flat
  //    "getting there" leg from an abstract depot.
  // ======================================================================

  function estimateTravelHours(a, b) {
    if (!a || !b) return 0.5; // depot leg: "getting going"
    if (a.city_region && b.city_region && a.city_region === b.city_region) return 0.3;
    if (a.section && b.section && a.section === b.section) return 1.0;
    if (a.country && b.country && a.country === b.country) return 2.0;
    return 4.0;
  }

  function routeHours(order) {
    if (!order.length) return 0.0;
    let hours = estimateTravelHours(null, order[0]);
    for (let i = 0; i < order.length - 1; i++) {
      hours += (order[i].est_hours || 1.0) + estimateTravelHours(order[i], order[i + 1]);
    }
    hours += order[order.length - 1].est_hours || 1.0;
    return hours;
  }

  function twoOpt(route, budgetHours) {
    if (route.length < 3) return route;
    let best = route;
    let bestHours = routeHours(best);
    let improved = true;
    while (improved) {
      improved = false;
      for (let i = 0; i < best.length - 1; i++) {
        for (let j = i + 1; j < best.length; j++) {
          const candidate = [...best.slice(0, i), ...best.slice(i, j + 1).reverse(), ...best.slice(j + 1)];
          const h = routeHours(candidate);
          if (h < bestHours - 1e-9 && h <= budgetHours + 1e-9) {
            best = candidate; bestHours = h; improved = true;
          }
        }
      }
    }
    return best;
  }

  /** Greedy construction: repeatedly insert the candidate, at whichever
   * position is cheapest, with the best value/added-time ratio that still
   * fits the remaining budget -- then 2-opt to shorten the resulting route. */
  function greedyTwoOpt(scored, budgetHours) {
    let remaining = [...scored].sort((a, b) => b.score - a.score);
    let route = [];
    while (remaining.length) {
      let best = null; // { ratio, idx, cand }
      for (const cand of remaining) {
        const before = routeHours(route);
        for (let idx = 0; idx <= route.length; idx++) {
          const trial = [...route.slice(0, idx), cand.option, ...route.slice(idx)];
          const after = routeHours(trial);
          if (after > budgetHours) continue;
          const added = after - before;
          const ratio = cand.score / Math.max(added, 0.1);
          if (!best || ratio > best.ratio) best = { ratio, idx, cand };
        }
      }
      if (!best) break;
      route.splice(best.idx, 0, best.cand.option);
      remaining = remaining.filter((c) => c !== best.cand);
    }
    route = twoOpt(route, budgetHours);
    const inRoute = new Set(route);
    const value = scored.filter((s) => inRoute.has(s.option)).reduce((sum, s) => sum + s.score, 0);
    return { order: route, totalValue: value, totalHours: routeHours(route) };
  }

  // ======================================================================
  // 7. Recovery curve -- "done, would repeat" items stay in the pool but
  //    get discounted right after being done, then recover the longer it's
  //    been. interest(t) = 1 - (1 - FLOOR) * exp(-t / TAU), t = days since
  //    last done.
  // ======================================================================

  const RECOVERY_FLOOR = 0.25;
  const RECOVERY_TAU_DAYS = 10.0;

  function recoveryMultiplier(itemState, today) {
    if (!itemState || itemState.status !== "done_repeat" || !itemState.lastDone) return 1.0;
    today = today || new Date();
    const last = new Date(itemState.lastDone);
    const t = Math.max(0, Math.floor((today - last) / (1000 * 60 * 60 * 24)));
    return RECOVERY_FLOOR + (1.0 - RECOVERY_FLOOR) * (1.0 - Math.exp(-t / RECOVERY_TAU_DAYS));
  }

  // ======================================================================
  // Public API
  // ======================================================================

  global.Recommender = {
    // vocab
    COSTS, PRIORITIES, VIRAL_RATINGS, OUTDOOR_LEAN, ENERGY_DEMAND,
    MOOD_ADVENTURE_LEAN, TRIP_TYPE_SHEETS, FEATURE_NAMES, DEFAULT_WEIGHTS,
    // content scoring
    filterCandidates, featurize, rankByRules, diversify,
    // bandit
    ThompsonBandit, rewardFromFeedback,
    // trip optimizer
    estimateTravelHours, routeHours, greedyTwoOpt,
    // state
    recoveryMultiplier, RECOVERY_FLOOR, RECOVERY_TAU_DAYS,
    // linear algebra + rng (exposed for test.html / simulation.html)
    linalg: { dot, vecAdd, vecScale, matVec, matAdd, matScale, outer, identity, solveLinear, invert, cholesky },
    makeRng, gaussianRandom, sampleMultivariateNormal,
  };
})(typeof window !== "undefined" ? window : globalThis);

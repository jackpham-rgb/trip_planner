/**
 * app.js -- the orchestration layer. Everything that touches the DOM,
 * the network (option bank, Firebase, geolocation, weather), or browser
 * storage lives here; the actual recommendation math is all in
 * recommender.js and never touches any of that. See README.md for the
 * full architecture writeup.
 *
 * The intake questions and the post-activity review are both rendered as a
 * short sequence of one-question-at-a-time steps rather than one long form,
 * and which step comes next depends on the answer just given (e.g. "long
 * trip" asks which region; "hangout nearby" asks how far instead). Both
 * flows share the same tiny step-renderer (`renderChoiceStep`/`renderTextStep`)
 * further down.
 */
(function () {
  "use strict";

  const R = window.Recommender;
  const STATE_PATH = "trip_planner/state"; // one shared path -- see README/firebase-rules.json for why
  const LOCAL_KEY = "trip_planner_state";

  // ====================================================================
  // Storage: Firebase Realtime Database if firebase-config.js has real
  // values, localStorage otherwise. Same shape either way: { items, bandit }.
  // ====================================================================

  const Storage = (function () {
    let readyPromise = null;

    function useFirebase() {
      return !!(window.FIREBASE_IS_CONFIGURED && window.firebase);
    }

    function ensureFirebase() {
      if (!readyPromise) {
        const app = firebase.initializeApp(window.FIREBASE_CONFIG);
        const auth = firebase.auth(app);
        const db = firebase.database(app);
        readyPromise = auth.signInAnonymously().then(() => db);
      }
      return readyPromise;
    }

    async function load() {
      const empty = { items: {}, bandit: null };
      if (useFirebase()) {
        try {
          const db = await ensureFirebase();
          const snap = await db.ref(STATE_PATH).once("value");
          return snap.val() || empty;
        } catch (e) {
          console.warn("Firebase read failed, falling back to empty state:", e);
          return empty;
        }
      }
      try {
        const raw = localStorage.getItem(LOCAL_KEY);
        return raw ? JSON.parse(raw) : empty;
      } catch (e) {
        return empty;
      }
    }

    async function save(state) {
      if (useFirebase()) {
        try {
          const db = await ensureFirebase();
          await db.ref(STATE_PATH).set(state);
          return;
        } catch (e) {
          console.warn("Firebase write failed, falling back to localStorage:", e);
        }
      }
      try { localStorage.setItem(LOCAL_KEY, JSON.stringify(state)); } catch (e) { /* ignore */ }
    }

    return { load, save, useFirebase };
  })();

  // ====================================================================
  // Recommendation pipeline glue (thin wrappers over recommender.js)
  // ====================================================================

  function excludedIds(state) {
    const out = new Set();
    for (const id in state.items || {}) {
      if (state.items[id].status === "done_no_repeat") out.add(id);
    }
    return out;
  }

  function getBandit(state) {
    if (state.bandit) return R.ThompsonBandit.fromPosterior(state.bandit, R.DEFAULT_WEIGHTS);
    return new R.ThompsonBandit(R.FEATURE_NAMES, R.DEFAULT_WEIGHTS);
  }

  function scoreCandidates(options, context, state) {
    const candidates = R.filterCandidates(options, context, excludedIds(state));
    if (!candidates.length) return [];
    let scored;
    if (state.bandit && state.bandit.nUpdates > 0) {
      const bandit = getBandit(state);
      const phis = candidates.map((o) => R.featurize(o, context));
      const scores = bandit.rank(phis, Math.random);
      scored = candidates.map((o, i) => ({ option: o, score: scores[i], features: phis[i] }));
    } else {
      scored = R.rankByRules(candidates, context);
    }
    scored.forEach((s) => { s.score *= R.recoveryMultiplier((state.items || {})[s.option.id]); });
    scored.sort((a, b) => b.score - a.score);
    return scored;
  }

  function recommendNext(options, context, state, topK) {
    return R.diversify(scoreCandidates(options, context, state), topK || 5);
  }

  function planTrip(options, context, state, budgetHours) {
    const pool = scoreCandidates(options, context, state).slice(0, 15);
    if (!pool.length) return { order: [], totalValue: 0, totalHours: 0 };
    return R.greedyTwoOpt(pool, budgetHours);
  }

  function applyFeedback(state, option, context, action, rating, notes, extra) {
    const reward = R.rewardFromFeedback(action, rating);
    const bandit = getBandit(state);
    const phi = R.featurize(option, context);
    bandit.update(phi, reward, 0.98); // mild decay for non-stationarity, same as the Python build
    const nUpdates = (state.bandit ? state.bandit.nUpdates : 0) + 1;
    state.bandit = bandit.toPosterior(nUpdates);

    if (action === "accepted_repeat" || action === "accepted_no_repeat") {
      state.items = state.items || {};
      const prev = state.items[option.id];
      state.items[option.id] = {
        status: action === "accepted_repeat" ? "done_repeat" : "done_no_repeat",
        timesDone: (prev ? prev.timesDone : 0) + 1,
        lastDone: new Date().toISOString().slice(0, 10),
        rating: rating === undefined ? null : rating,
        notes: notes || "",
        ...(extra || {}),
      };
    }
  }

  // ====================================================================
  // Geolocation -> zip/city (not raw distance math) + weather defaults
  // ====================================================================

  let weatherIndoorOutdoorDefault = null; // set once geolocation+weather resolve, used as a suggested (not forced) answer

  async function reverseGeocode(lat, lon) {
    const res = await fetch(`https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${lat}&longitude=${lon}&localityLanguage=en`);
    if (!res.ok) throw new Error("reverse geocode failed");
    return res.json();
  }

  async function fetchWeather(lat, lon) {
    const res = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,precipitation&temperature_unit=fahrenheit`);
    if (!res.ok) throw new Error("weather fetch failed");
    const data = await res.json();
    return { temperature: data.current.temperature_2m, precipitation: data.current.precipitation };
  }

  function weatherToIndoorOutdoor(weather) {
    if (!weather) return null;
    if (weather.precipitation > 0.1 || weather.temperature < 45 || weather.temperature > 95) return "-1";
    if (weather.temperature >= 60 && weather.temperature <= 80) return "1";
    return null;
  }

  function initContextSignals() {
    const note = document.getElementById("context-note");
    if (!("geolocation" in navigator)) return;
    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        const { latitude, longitude } = pos.coords;
        try {
          const [geo, weather] = await Promise.allSettled([
            reverseGeocode(latitude, longitude), fetchWeather(latitude, longitude),
          ]);
          const parts = [];
          if (geo.status === "fulfilled" && geo.value) {
            const g = geo.value;
            const place = g.city || g.locality || g.principalSubdivision || "";
            parts.push(`\u{1F4CD} ${place}${g.postcode ? " " + g.postcode : ""}`.trim());
          }
          if (weather.status === "fulfilled" && weather.value) {
            const w = weather.value;
            parts.push(`${Math.round(w.temperature)}°F${w.precipitation > 0 ? ", precip" : ""}`);
            weatherIndoorOutdoorDefault = weatherToIndoorOutdoor(w);
          }
          if (note && parts.length) note.textContent = parts.join(" · ");
        } catch (e) { /* these are just defaults -- silently skip on failure */ }
      },
      () => { /* permission denied or unavailable -- the form still works without it */ },
      { timeout: 6000 }
    );
  }

  // ====================================================================
  // Yelp-style price legend and region groupings -- display/UI concerns
  // only, so they live here rather than in recommender.js.
  // ====================================================================

  const COST_HINTS = { Free: "$0", $: "under $20", $$: "$20–75", $$$: "$75–200", $$$$: "$200+" };

  const REGION_GROUPS = {
    americas: new Set(["USA", "USA MEX CAN", "South America"]),
    europe: new Set(["Europe"]),
    asia: new Set(["East Asia", "South East Asia"]),
    mea: new Set(["Middle East", "Africa"]),
    // "anywhere" is intentionally absent -- it means "no filter", which
    // also reaches "Rest of World", the one sheet no specific group covers.
  };

  // ====================================================================
  // A tiny step renderer shared by the intake wizard and the per-card
  // review log. `container` is any element; `onPick`/`onSubmit` receive the
  // chosen value and are expected to render whatever comes next themselves.
  // ====================================================================

  function renderChoiceStep(container, question, options, onPick, opts) {
    opts = opts || {};
    const suggested = opts.suggestedValue;
    container.innerHTML = `
      ${opts.progress ? `<div class="wizard-progress">${opts.progress}</div>` : ""}
      <h2 class="wizard-question">${question}</h2>
      <div class="choice-list">
        ${options.map(([val, label]) =>
          `<button type="button" class="choice-btn${val === suggested ? " suggested" : ""}" data-value="${val}">${label}</button>`
        ).join("")}
      </div>
      ${opts.onBack ? `<button type="button" class="secondary back-btn">Back</button>` : ""}
    `;
    container.querySelectorAll(".choice-btn").forEach((btn) => {
      btn.addEventListener("click", () => onPick(btn.dataset.value));
    });
    if (opts.onBack) container.querySelector(".back-btn").addEventListener("click", opts.onBack);
  }

  function renderMultiChoiceStep(container, question, options, onSubmit, opts) {
    opts = opts || {};
    container.innerHTML = `
      ${opts.progress ? `<div class="wizard-progress">${opts.progress}</div>` : ""}
      <h2 class="wizard-question">${question}</h2>
      <div class="choice-list multi">
        ${options.map(([val, label]) => `<button type="button" class="choice-btn chip-btn" data-value="${val}">${label}</button>`).join("")}
      </div>
      <div class="actions"><button type="button" class="continue-btn">Continue</button></div>
    `;
    const chosen = new Set();
    container.querySelectorAll(".choice-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        btn.classList.toggle("selected");
        if (chosen.has(btn.dataset.value)) chosen.delete(btn.dataset.value);
        else chosen.add(btn.dataset.value);
      });
    });
    container.querySelector(".continue-btn").addEventListener("click", () => onSubmit([...chosen]));
  }

  function renderTextStep(container, question, placeholder, onSubmit, opts) {
    opts = opts || {};
    container.innerHTML = `
      ${opts.progress ? `<div class="wizard-progress">${opts.progress}</div>` : ""}
      <h2 class="wizard-question">${question}</h2>
      <input type="text" class="step-text-input" placeholder="${placeholder || ""}">
      <div class="actions"><button type="button" class="continue-btn">Continue</button></div>
    `;
    const submit = () => onSubmit(container.querySelector(".step-text-input").value.trim());
    container.querySelector(".continue-btn").addEventListener("click", submit);
    container.querySelector(".step-text-input").addEventListener("keydown", (e) => { if (e.key === "Enter") submit(); });
  }

  // ====================================================================
  // The intake wizard -- one question at a time, later questions chosen
  // by earlier answers.
  // ====================================================================

  function intakeSteps() {
    return [
      {
        id: "trip_type", question: "What kind of trip is this?",
        options: [
          ["hangout_nearby", "Just hanging out nearby"],
          ["day_trip", "Day trip"],
          ["long_trip", "Long trip"],
          ["custom", "Custom duration"],
        ],
      },
      {
        id: "distance", question: "How far are you willing to go?",
        showIf: (a) => a.trip_type === "hangout_nearby" || a.trip_type === "day_trip",
        options: [
          ["3", "Just close by — there and back in a few hours"],
          ["8", "A bit of a trek — can fill most of a day"],
          ["16", "Give me the whole day, maybe overnight"],
        ],
      },
      {
        id: "region", question: "Which part of the world appeals right now?",
        showIf: (a) => a.trip_type === "long_trip",
        options: [
          ["americas", "The Americas"], ["europe", "Europe"], ["asia", "Asia"],
          ["mea", "Middle East & Africa"], ["anywhere", "Surprise me — anywhere"],
        ],
      },
      {
        id: "duration_hint", question: "Roughly how long?", type: "text",
        showIf: (a) => a.trip_type === "custom", placeholder: "e.g. '5 hours', '4 days'",
      },
      {
        id: "energy", question: "Energy level right now?",
        options: [["0.1", "Wiped out"], ["0.3", "Kind of tired"], ["0.5", "Normal"], ["0.7", "Good energy"], ["0.9", "Wired / raring to go"]],
      },
      {
        id: "budget", question: "Budget for this?",
        options: () => R.COSTS.map((label, i) => [String(i), `${label} — ${COST_HINTS[label]}`]),
      },
      {
        id: "party", question: "Who's coming?",
        options: [["solo", "Solo"], ["partner", "Partner"], ["friends", "Friends"], ["family", "Family"]],
      },
      {
        id: "indoor_outdoor", question: "Indoor or outdoor?",
        options: [["-1", "Indoor"], ["0", "No preference"], ["1", "Outdoor"]],
        suggestedValue: () => weatherIndoorOutdoorDefault,
      },
      {
        id: "mood", question: "Adventurous or relaxed?",
        options: [["-1", "Relaxed"], ["0", "No preference"], ["1", "Adventurous"]],
      },
    ];
  }

  let wizardAnswers = {};
  let wizardIndex = 0;

  function visibleIntakeSteps() {
    return intakeSteps().filter((s) => !s.showIf || s.showIf(wizardAnswers));
  }

  function renderWizard() {
    const steps = visibleIntakeSteps();
    const wizardEl = document.getElementById("wizard");
    if (wizardIndex >= steps.length) { renderWizardDone(wizardEl); return; }
    const step = steps[wizardIndex];
    const progress = `Step ${wizardIndex + 1} of ${steps.length}`;
    const goBack = wizardIndex > 0 ? () => { wizardIndex--; renderWizard(); } : null;

    if (step.type === "text") {
      renderTextStep(wizardEl, step.question, step.placeholder, (value) => {
        wizardAnswers[step.id] = value;
        wizardIndex++;
        renderWizard();
      }, { progress, onBack: goBack });
    } else {
      const options = typeof step.options === "function" ? step.options() : step.options;
      const suggested = typeof step.suggestedValue === "function" ? step.suggestedValue() : step.suggestedValue;
      renderChoiceStep(wizardEl, step.question, options, (value) => {
        wizardAnswers[step.id] = value;
        wizardIndex++;
        renderWizard();
      }, { progress, onBack: goBack, suggestedValue: suggested });
    }
  }

  function buildContextFromAnswers(a) {
    const ctx = {
      tripType: a.trip_type,
      energy: parseFloat(a.energy),
      budgetLevel: parseInt(a.budget, 10),
      party: a.party,
      indoorOutdoor: parseFloat(a.indoor_outdoor),
      moodAdventurous: parseFloat(a.mood),
    };
    if (a.distance) ctx.maxDwellHours = parseFloat(a.distance);
    if (a.region && a.region !== "anywhere" && REGION_GROUPS[a.region]) ctx.preferredSheets = REGION_GROUPS[a.region];
    return ctx;
  }

  function restartWizard() {
    wizardAnswers = {};
    wizardIndex = 0;
    document.getElementById("results").innerHTML = "";
    renderWizard();
  }

  function renderWizardDone(wizardEl) {
    const context = buildContextFromAnswers(wizardAnswers);
    wizardEl.innerHTML = `
      <h2 class="wizard-question">Ready.</h2>
      <div class="actions">
        <button type="button" id="btn-suggest">Suggest something</button>
        <button type="button" id="btn-plan">Plan a trip</button>
      </div>
      <button type="button" class="secondary" id="wizard-restart">Start over</button>
    `;
    document.getElementById("btn-suggest").addEventListener("click", () => {
      renderEmpty("Thinking…");
      const results = recommendNext(OPTIONS, context, STATE, 3); // top 3, per the brief
      renderSuggestions(results, context);
    });
    document.getElementById("btn-plan").addEventListener("click", () => renderHoursStep(wizardEl, context));
    document.getElementById("wizard-restart").addEventListener("click", restartWizard);
  }

  function renderHoursStep(wizardEl, context) {
    const presets = [["3", "3 hours"], ["5", "5 hours"], ["8", "8 hours"], ["12", "12 hours"]];
    renderChoiceStep(wizardEl, "How many hours do you have?", presets, (value) => {
      renderEmpty("Planning…");
      const itinerary = planTrip(OPTIONS, context, STATE, parseFloat(value));
      renderItinerary(itinerary);
    }, { onBack: () => renderWizardDone(wizardEl) });
  }

  // ====================================================================
  // Results rendering
  // ====================================================================

  let OPTIONS = [];
  let STATE = { items: {}, bandit: null };
  const resultsEl = document.getElementById("results");
  const cardTemplate = document.getElementById("suggestion-card");

  function renderEmpty(message) {
    resultsEl.innerHTML = `<p class="empty">${message}</p>`;
  }

  function optionLabel(option) {
    return option.label || option.specific_attraction || option.main_destination || option.id;
  }

  // ---- the review/log flow: also one question at a time, branching on the
  // rating just given, exactly like the intake wizard above. ----
  function startLogFlow(container, option, context) {
    renderChoiceStep(container, "How was it?",
      [["1", "1"], ["2", "2"], ["3", "3"], ["4", "4"], ["5", "5"]],
      (ratingStr) => {
        const rating = parseFloat(ratingStr);
        if (rating <= 2) {
          renderMultiChoiceStep(container, "What went wrong? (pick any)", [
            ["too_expensive", "Too expensive"], ["too_far", "Too far / too long"],
            ["not_as_expected", "Not what I expected"], ["closed", "Closed or unavailable"], ["other", "Other"],
          ], (reasons) => askNotes(container, option, context, rating, "accepted_no_repeat", { wentWrong: reasons }));
        } else if (rating >= 4) {
          renderChoiceStep(container, "When would you want to do this again?", [
            ["soon", "Soon — this week"], ["sometime", "Sometime — this month"],
            ["someday", "Someday"], ["not_eager", "Probably not eager to repeat"],
          ], (timing) => {
            const action = timing === "not_eager" ? "accepted_no_repeat" : "accepted_repeat";
            const extra = timing === "not_eager" ? {} : { recoveryTauDays: R.RECOVERY_TAU_PRESETS[timing] };
            askNotes(container, option, context, rating, action, extra);
          });
        } else {
          renderChoiceStep(container, "Would you do it again?", [["yes", "Yes"], ["no", "No"]], (answer) => {
            askNotes(container, option, context, rating, answer === "yes" ? "accepted_repeat" : "accepted_no_repeat", {});
          });
        }
      }
    );
  }

  function askNotes(container, option, context, rating, action, extra) {
    container.innerHTML = `
      <h2 class="wizard-question">Anything to remember?</h2>
      <label>Notes <span style="font-weight:400">(for you to read later -- not used in scoring)</span>
        <textarea class="notes-input" placeholder="What you'd want to remember next time."></textarea>
      </label>
      <div class="actions"><button type="button" class="save-btn">Save</button></div>
    `;
    container.querySelector(".save-btn").addEventListener("click", async () => {
      const notes = container.querySelector(".notes-input").value.trim();
      applyFeedback(STATE, option, context, action, rating, notes, extra);
      await Storage.save(STATE);
      container.innerHTML = `<p class="empty">Saved -- thanks, this updates future suggestions.</p>`;
    });
  }

  function renderSuggestions(results, context) {
    resultsEl.innerHTML = "";
    if (results.length === 0) {
      renderEmpty("Nothing matches right now -- try loosening the budget or trip type.");
      return;
    }
    for (const s of results) {
      const option = s.option;
      const node = cardTemplate.content.cloneNode(true);
      node.querySelector(".label").textContent = optionLabel(option);
      node.querySelector(".meta").textContent =
        `${option.type || "?"} · ${option.cost || "?"} (${COST_HINTS[option.cost] || "?"}) · ~${option.est_hours ?? "?"}h · score ${s.score?.toFixed(2) ?? ""}`;
      node.querySelector(".why").textContent = option.why_worth_it || "";

      const feedbackDiv = node.querySelector(".feedback");
      const openBtn = node.querySelector("[data-open-log]");
      const logFlow = node.querySelector(".log-flow");
      openBtn.addEventListener("click", () => {
        feedbackDiv.hidden = true;
        logFlow.hidden = false;
        startLogFlow(logFlow, option, context);
      });
      resultsEl.appendChild(node);
    }
  }

  function renderItinerary(itinerary) {
    resultsEl.innerHTML = "";
    if (itinerary.order.length === 0) {
      renderEmpty("Couldn't fit anything in that time budget -- try a larger one.");
      return;
    }
    const summary = document.createElement("p");
    summary.className = "itinerary-summary";
    summary.textContent = `${itinerary.totalHours.toFixed(1)}h planned, value ${itinerary.totalValue.toFixed(2)}`;
    resultsEl.appendChild(summary);

    itinerary.order.forEach((option, i) => {
      const card = document.createElement("article");
      card.className = "card";
      card.innerHTML = `<h3>${i + 1}. ${optionLabel(option)}</h3>
        <p class="meta">${option.type || "?"} · ${option.cost || "?"} (${COST_HINTS[option.cost] || "?"}) · ~${option.est_hours ?? "?"}h</p>
        <p class="why">${option.why_worth_it || ""}</p>`;
      resultsEl.appendChild(card);
    });
  }

  // ====================================================================
  // Boot
  // ====================================================================

  async function boot() {
    initContextSignals();
    renderWizard();
    try {
      const res = await fetch("data/option_bank.json");
      OPTIONS = await res.json();
    } catch (e) {
      renderEmpty("Couldn't load the option bank (data/option_bank.json) -- run tools/build_option_bank.py first.");
      return;
    }
    STATE = await Storage.load();
  }

  boot();

  if ("serviceWorker" in navigator) {
    window.addEventListener("load", () => {
      navigator.serviceWorker.register("service-worker.js").catch(() => { /* offline shell is a nice-to-have, not required */ });
    });
  }
})();

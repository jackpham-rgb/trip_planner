/**
 * app.js -- the orchestration layer. Everything that touches the DOM,
 * the network (option bank, Firebase, geolocation, weather), or browser
 * storage lives here; the actual recommendation math is all in
 * recommender.js and never touches any of that. See README.md for the
 * full architecture writeup.
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

  function applyFeedback(state, option, context, action, rating, notes) {
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
      };
    }
  }

  // ====================================================================
  // Geolocation -> zip/city (not raw distance math) + weather defaults
  // ====================================================================

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

  function applyWeatherDefault(weather) {
    const select = document.getElementById("indoor_outdoor");
    if (!weather || !select) return;
    if (weather.precipitation > 0.1 || weather.temperature < 45 || weather.temperature > 95) {
      select.value = "-1";
    } else if (weather.temperature >= 60 && weather.temperature <= 80) {
      select.value = "1";
    }
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
            applyWeatherDefault(w);
          }
          if (note && parts.length) note.textContent = parts.join(" · ") + " (auto-filled defaults below, override anytime)";
        } catch (e) { /* these are just defaults -- silently skip on failure */ }
      },
      () => { /* permission denied or unavailable -- the form still works without it */ },
      { timeout: 6000 }
    );
  }

  // ====================================================================
  // DOM wiring
  // ====================================================================

  let OPTIONS = [];
  let STATE = { items: {}, bandit: null };

  const form = document.getElementById("intake-form");
  const resultsEl = document.getElementById("results");
  const tripTypeEl = document.getElementById("trip_type");
  const hoursField = document.getElementById("hours-field");
  const cardTemplate = document.getElementById("suggestion-card");

  function fillBudgetOptions() {
    const budgetSelect = document.getElementById("budget_level");
    budgetSelect.innerHTML = R.COSTS
      .map((label, i) => `<option value="${i}" ${i === 2 ? "selected" : ""}>${label}</option>`)
      .join("");
  }

  function readContext() {
    const data = new FormData(form);
    return {
      tripType: data.get("trip_type"),
      energy: parseFloat(data.get("energy")),
      budgetLevel: parseInt(data.get("budget_level"), 10),
      party: data.get("party"),
      indoorOutdoor: parseFloat(data.get("indoor_outdoor")),
      moodAdventurous: parseFloat(data.get("mood_adventurous")),
    };
  }

  function renderEmpty(message) {
    resultsEl.innerHTML = `<p class="empty">${message}</p>`;
  }

  function buildPips(container, name) {
    let html = "";
    for (let i = 1; i <= 5; i++) {
      html += `<input type="radio" id="${name}-${i}" name="${name}" value="${i}">`
        + `<label for="${name}-${i}">${i}</label>`;
    }
    container.innerHTML = html;
  }

  function wireLogForm(cardNode, option, context, onSaved) {
    const openBtn = cardNode.querySelector("[data-open-log]");
    const logForm = cardNode.querySelector(".log-form");
    const pipsContainer = cardNode.querySelector(".rating-pips");
    const uniqueName = "rating-" + option.id.replace(/[^A-Za-z0-9]/g, "");
    buildPips(pipsContainer, uniqueName);

    openBtn.addEventListener("click", () => {
      logForm.hidden = false;
      openBtn.parentElement.hidden = true;
    });
    cardNode.querySelector("[data-cancel-log]").addEventListener("click", () => {
      logForm.hidden = true;
      openBtn.parentElement.hidden = false;
    });
    cardNode.querySelector("[data-save-log]").addEventListener("click", async () => {
      const ratingEl = logForm.querySelector(`input[name="${uniqueName}"]:checked`);
      const repeatEl = logForm.querySelector('input[name="repeat"]:checked');
      const notes = logForm.querySelector('textarea[name="notes"]').value.trim();
      const rating = ratingEl ? parseFloat(ratingEl.value) : undefined;
      const action = repeatEl
        ? (repeatEl.value === "yes" ? "accepted_repeat" : "accepted_no_repeat")
        : "skipped";
      applyFeedback(STATE, option, context, action, rating, notes);
      await Storage.save(STATE);
      logForm.innerHTML = "<p class=\"empty\">Saved -- thanks, this updates future suggestions.</p>";
      if (onSaved) onSaved();
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
      node.querySelector(".label").textContent = option.label || option.specific_attraction || option.main_destination || option.id;
      node.querySelector(".meta").textContent =
        `${option.type || "?"} · ${option.cost || "?"} · ~${option.est_hours ?? "?"}h · score ${s.score?.toFixed(2) ?? ""}`;
      node.querySelector(".why").textContent = option.why_worth_it || "";
      wireLogForm(node, option, context);
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
      const label = option.label || option.specific_attraction || option.main_destination || option.id;
      card.innerHTML = `<h3>${i + 1}. ${label}</h3>
        <p class="meta">${option.type || "?"} · ~${option.est_hours ?? "?"}h</p>
        <p class="why">${option.why_worth_it || ""}</p>`;
      resultsEl.appendChild(card);
    });
  }

  tripTypeEl.addEventListener("change", () => {
    hoursField.hidden = tripTypeEl.value !== "custom";
  });

  document.getElementById("btn-suggest").addEventListener("click", () => {
    renderEmpty("Thinking…");
    const context = readContext();
    const results = recommendNext(OPTIONS, context, STATE, 3); // top 3, per the brief
    renderSuggestions(results, context);
  });

  document.getElementById("btn-plan").addEventListener("click", () => {
    renderEmpty("Planning…");
    const context = readContext();
    const hours = parseFloat(form.elements["budget_hours"]?.value) || 8;
    const itinerary = planTrip(OPTIONS, context, STATE, hours);
    renderItinerary(itinerary);
  });

  async function boot() {
    fillBudgetOptions();
    initContextSignals();
    try {
      const res = await fetch("data/option_bank.json");
      OPTIONS = await res.json();
    } catch (e) {
      renderEmpty("Couldn't load the option bank (data/option_bank.json) -- run tools/build_option_bank.py first.");
      return;
    }
    STATE = await Storage.load();
    renderEmpty("Pick your preferences above, then hit Suggest or Plan.");
  }

  boot();

  if ("serviceWorker" in navigator) {
    window.addEventListener("load", () => {
      navigator.serviceWorker.register("service-worker.js").catch(() => { /* offline shell is a nice-to-have, not required */ });
    });
  }
})();

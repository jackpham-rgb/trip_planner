const API = "/api";

const form = document.getElementById("intake-form");
const resultsEl = document.getElementById("results");
const tripTypeEl = document.getElementById("trip_type");
const hoursField = document.getElementById("hours-field");
const cardTemplate = document.getElementById("suggestion-card");

async function loadMeta() {
  const res = await fetch(`${API}/meta`);
  const meta = await res.json();
  const budgetSelect = document.getElementById("budget_level");
  budgetSelect.innerHTML = meta.costs
    .map((label, i) => `<option value="${i}" ${i === 2 ? "selected" : ""}>${label}</option>`)
    .join("");
}

function readContext() {
  const data = new FormData(form);
  return {
    trip_type: data.get("trip_type"),
    energy: parseFloat(data.get("energy")),
    budget_level: parseInt(data.get("budget_level"), 10),
    party: data.get("party"),
    indoor_outdoor: parseFloat(data.get("indoor_outdoor")),
    mood_adventurous: parseFloat(data.get("mood_adventurous")),
  };
}

function renderEmpty(message) {
  resultsEl.innerHTML = `<p class="empty">${message}</p>`;
}

function renderSuggestions(results, context) {
  resultsEl.innerHTML = "";
  if (results.length === 0) {
    renderEmpty("Nothing matches right now — try loosening the budget or trip type.");
    return;
  }
  for (const option of results) {
    const node = cardTemplate.content.cloneNode(true);
    node.querySelector(".label").textContent = option.label;
    node.querySelector(".meta").textContent =
      `${option.type || "?"} · ${option.cost || "?"} · ~${option.est_hours ?? "?"}h · score ${option.score?.toFixed(2) ?? ""}`;
    node.querySelector(".why").textContent = option.why_worth_it || "";
    const buttons = node.querySelectorAll("[data-action]");
    buttons.forEach((btn) => {
      btn.addEventListener("click", async () => {
        await fetch(`${API}/feedback`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ option_id: option.id, context, action: btn.dataset.action }),
        });
        buttons.forEach((b) => b.classList.add("done"));
        btn.textContent = "Noted";
        btn.classList.remove("done");
      });
    });
    resultsEl.appendChild(node);
  }
}

function renderItinerary(itinerary) {
  resultsEl.innerHTML = "";
  if (itinerary.order.length === 0) {
    renderEmpty("Couldn't fit anything in that time budget — try a larger one.");
    return;
  }
  const summary = document.createElement("p");
  summary.className = "itinerary-summary";
  summary.textContent = `${itinerary.total_hours.toFixed(1)}h planned, value ${itinerary.total_value.toFixed(2)}`;
  resultsEl.appendChild(summary);

  itinerary.order.forEach((option, i) => {
    const card = document.createElement("article");
    card.className = "card";
    card.innerHTML = `<h3>${i + 1}. ${option.label}</h3>
      <p class="meta">${option.type || "?"} · ~${option.est_hours ?? "?"}h</p>
      <p class="why">${option.why_worth_it || ""}</p>`;
    resultsEl.appendChild(card);
  });
}

tripTypeEl.addEventListener("change", () => {
  hoursField.hidden = tripTypeEl.value !== "custom";
});

document.getElementById("btn-suggest").addEventListener("click", async () => {
  renderEmpty("Thinking…");
  const context = readContext();
  const res = await fetch(`${API}/suggest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ context, top_k: 5 }),
  });
  const data = await res.json();
  renderSuggestions(data.results, context);
});

document.getElementById("btn-plan").addEventListener("click", async () => {
  renderEmpty("Planning…");
  const context = readContext();
  const hours = parseFloat(form.elements["budget_hours"]?.value) || 8;
  const res = await fetch(`${API}/plan`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ context, budget_hours: hours, use_milp: true }),
  });
  const data = await res.json();
  renderItinerary(data);
});

loadMeta();

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("service-worker.js").catch(() => {});
  });
}

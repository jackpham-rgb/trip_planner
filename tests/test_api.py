import json

from fastapi.testclient import TestClient

import recsys.api as api
from recsys.data_loader import save_option_bank
from recsys.schema import Option


def make_option(id, **overrides):
    base = dict(id=id, source_sheet="Cali", section="Test", main_destination=id,
                type="Nature", cost="$$", priority="HIGH",
                viral_rating="Actually worth it", est_hours=2.0)
    base.update(overrides)
    return Option(**base)


def make_client(tmp_path, monkeypatch, options=None):
    options = options if options is not None else [make_option("a"), make_option("b", type="Food")]
    bank_path = tmp_path / "option_bank.json"
    state_path = tmp_path / "state.json"
    save_option_bank(options, bank_path)
    monkeypatch.setattr(api, "DEFAULT_OPTION_BANK", bank_path)
    monkeypatch.setattr(api, "DEFAULT_STATE", state_path)
    return TestClient(api.app), state_path


def test_health_reports_loaded_option_count(tmp_path, monkeypatch):
    client, _ = make_client(tmp_path, monkeypatch)
    with client:
        res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "num_options": 2}


def test_meta_exposes_cost_vocab(tmp_path, monkeypatch):
    client, _ = make_client(tmp_path, monkeypatch)
    with client:
        res = client.get("/api/meta")
    assert res.status_code == 200
    assert "Free" in res.json()["costs"]


def test_suggest_returns_scored_results(tmp_path, monkeypatch):
    client, _ = make_client(tmp_path, monkeypatch)
    body = {"context": {"trip_type": "hangout_nearby", "budget_level": 2}, "top_k": 5}
    with client:
        res = client.post("/api/suggest", json=body)
    assert res.status_code == 200
    results = res.json()["results"]
    assert len(results) == 2
    assert all("score" in r and r["score"] is not None for r in results)


def test_plan_returns_itinerary_within_budget(tmp_path, monkeypatch):
    client, _ = make_client(tmp_path, monkeypatch)
    body = {"context": {"trip_type": "hangout_nearby"}, "budget_hours": 3.0}
    with client:
        res = client.post("/api/plan", json=body)
    assert res.status_code == 200
    data = res.json()
    assert data["total_hours"] <= 3.0 + 1e-6


def test_feedback_persists_state_to_disk(tmp_path, monkeypatch):
    client, state_path = make_client(tmp_path, monkeypatch)
    body = {
        "option_id": "a",
        "context": {"trip_type": "hangout_nearby"},
        "action": "accepted_repeat",
    }
    with client:
        res = client.post("/api/feedback", json=body)
    assert res.status_code == 200
    assert res.json()["bandit_updates"] == 1

    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["items"]["a"]["status"] == "done_repeat"
    assert saved["bandit"]["n_updates"] == 1


def test_feedback_unknown_option_id_is_404(tmp_path, monkeypatch):
    client, _ = make_client(tmp_path, monkeypatch)
    body = {"option_id": "does-not-exist", "context": {}, "action": "skipped"}
    with client:
        res = client.post("/api/feedback", json=body)
    assert res.status_code == 404

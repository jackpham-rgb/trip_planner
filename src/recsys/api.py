"""FastAPI backend for the web app.

Thin HTTP layer over `pipeline.py` -- every endpoint here just validates a
request, calls a function that already exists and is already tested, and
serializes the result. No scoring/optimization logic lives in this file.

Single-user tool, so state is kept simple on purpose: the option bank loads
once at startup (it only changes when I rebuild it from the workbook), and
`data/state.json` is the on-disk source of truth for everything that
changes -- reloaded and resaved around each request that mutates it, rather
than trusting an in-memory copy to stay in sync with the file. That's more
I/O than strictly necessary for one user, but it means the CLI and the web
app can be used interchangeably without stepping on each other's state.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .content_scoring import COSTS
from .optimizer import Itinerary
from .pipeline import (
    DEFAULT_OPTION_BANK, DEFAULT_STATE, apply_feedback, load_options,
    load_user_state, plan_trip, recommend_next, save_user_state,
)
from .schema import Context, Option

WEB_DIR = Path(__file__).parent.parent.parent / "web"

_options: list[Option] = []
_options_by_id: dict[str, Option] = {}


def _load_option_bank() -> None:
    global _options, _options_by_id
    if DEFAULT_OPTION_BANK.exists():
        _options = load_options(DEFAULT_OPTION_BANK)
        _options_by_id = {o.id: o for o in _options}
    else:
        _options, _options_by_id = [], {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_option_bank()
    yield


app = FastAPI(title="Recommender + Trip Planner", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class ContextIn(BaseModel):
    trip_type: str = "hangout_nearby"
    duration_hint: Optional[str] = None
    energy: float = 0.5
    budget_level: int = 2
    party: str = "solo"
    indoor_outdoor: float = 0.0
    mood_adventurous: float = 0.0

    def to_context(self) -> Context:
        return Context(**self.model_dump())


class SuggestRequest(BaseModel):
    context: ContextIn
    top_k: int = 5


class PlanRequest(BaseModel):
    context: ContextIn
    budget_hours: float = Field(gt=0)
    use_milp: bool = True


class FeedbackRequest(BaseModel):
    option_id: str
    context: ContextIn
    action: str  # accepted_repeat | accepted_no_repeat | skipped
    rating: Optional[float] = None


def _option_out(option: Option, score: float | None = None) -> dict:
    return {
        "id": option.id, "label": option.label(), "type": option.type,
        "cost": option.cost, "priority": option.priority,
        "est_hours": option.est_hours, "why_worth_it": option.why_worth_it,
        "section": option.section, "source_sheet": option.source_sheet,
        "score": score,
    }


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "num_options": len(_options)}


@app.get("/api/meta")
def meta() -> dict:
    """Vocab the frontend needs to render its multiple-choice questions."""
    return {"costs": COSTS, "trip_types": ["long_trip", "day_trip", "hangout_nearby", "custom"]}


@app.post("/api/suggest")
def suggest(req: SuggestRequest) -> dict:
    if not _options:
        raise HTTPException(status_code=503, detail="Option bank not loaded -- run scripts/build_option_bank.py")
    state = load_user_state(DEFAULT_STATE)
    ctx = req.context.to_context()
    results = recommend_next(_options, ctx, state, top_k=req.top_k)
    return {"results": [_option_out(s.option, s.score) for s in results]}


@app.post("/api/plan")
def plan(req: PlanRequest) -> dict:
    if not _options:
        raise HTTPException(status_code=503, detail="Option bank not loaded -- run scripts/build_option_bank.py")
    state = load_user_state(DEFAULT_STATE)
    ctx = req.context.to_context()
    itinerary: Itinerary = plan_trip(_options, ctx, state, budget_hours=req.budget_hours,
                                      use_milp=req.use_milp)
    return {
        "order": [_option_out(o) for o in itinerary.order],
        "total_value": itinerary.total_value,
        "total_hours": itinerary.total_hours,
    }


@app.post("/api/feedback")
def feedback(req: FeedbackRequest) -> dict:
    option = _options_by_id.get(req.option_id)
    if option is None:
        raise HTTPException(status_code=404, detail=f"Unknown option id: {req.option_id}")
    state = load_user_state(DEFAULT_STATE)
    ctx = req.context.to_context()
    apply_feedback(state, option, ctx, req.action, rating=req.rating)
    save_user_state(state, DEFAULT_STATE)
    return {"ok": True, "bandit_updates": state.bandit.n_updates if state.bandit else 0}


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

"""Shared vocabularies and dataclasses for the option bank and context.

The vocabularies below are lifted directly from the "Lists" sheet of
Jack-Master-Travel-Database.xlsx, so encodings line up with how the workbook
already categorizes things instead of inventing a parallel taxonomy.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


TYPES = [
    "Historical", "Nature", "Food", "Architecture", "Shopping", "Nightlife",
    "Cultural", "Technology", "Automotive", "Anime/pop culture", "Photography",
    "Adventure", "Relaxation", "Festival/event", "Social-media-famous",
    "Transport experience", "Academic/research",
]

PRIORITIES = ["OPTIONAL", "HIGH", "MUST"]  # ordinal, low -> high
COSTS = ["Free", "$", "$$", "$$$", "$$$$"]  # ordinal, low -> high
VIRAL_RATINGS = [
    "Skip unless personally interested",
    "Not a social-media place",
    "Mostly photo opportunity",
    "Worth it if nearby",
    "Actually worth it",
]  # ordinal, low -> high "genuinely worth it"
SEASONS = ["Spring", "Summer", "Autumn", "Winter", "Shoulder", "Year-round", "Avoid monsoon"]
TIME_OF_DAY = [
    "Sunrise", "Early morning", "Morning", "Midday", "Afternoon", "Golden hour",
    "Sunset", "Evening", "Night", "Late night", "Any", "Overnight",
]

# Rough indoor/outdoor lean per Type, used as a content feature when the
# workbook doesn't say so explicitly. This is a judgment call, not workbook data.
OUTDOOR_LEAN = {
    "Historical": 0.3, "Nature": 1.0, "Food": -0.2, "Architecture": 0.2,
    "Shopping": -0.6, "Nightlife": -0.3, "Cultural": 0.0, "Technology": -0.5,
    "Automotive": 0.0, "Anime/pop culture": -0.4, "Photography": 0.4,
    "Adventure": 0.9, "Relaxation": 0.1, "Festival/event": 0.3,
    "Social-media-famous": 0.2, "Transport experience": 0.1,
    "Academic/research": -0.5,
}

# Rough energy demand per Type (low = chill, high = active), another judgment call.
ENERGY_DEMAND = {
    "Historical": 0.4, "Nature": 0.6, "Food": 0.2, "Architecture": 0.3,
    "Shopping": 0.4, "Nightlife": 0.6, "Cultural": 0.3, "Technology": 0.2,
    "Automotive": 0.3, "Anime/pop culture": 0.3, "Photography": 0.4,
    "Adventure": 0.9, "Relaxation": -0.7, "Festival/event": 0.6,
    "Social-media-famous": 0.3, "Transport experience": 0.2,
    "Academic/research": 0.1,
}

TRIP_TYPES = ["long_trip", "day_trip", "hangout_nearby", "custom"]


@dataclass
class Option:
    """One candidate activity/destination pulled from the workbook (content only)."""

    id: str
    source_sheet: str
    section: str  # e.g. sub-region/country header the row was found under
    version: Optional[str] = None
    day: Optional[str] = None
    country: Optional[str] = None
    city_region: Optional[str] = None
    overnight: Optional[str] = None
    route: Optional[str] = None
    transport: Optional[str] = None
    transit: Optional[str] = None
    main_destination: Optional[str] = None
    specific_attraction: Optional[str] = None
    address: Optional[str] = None
    website: Optional[str] = None
    why_worth_it: Optional[str] = None
    time_needed: Optional[str] = None
    priority: Optional[str] = None
    type: Optional[str] = None
    time_of_day: Optional[str] = None
    cost: Optional[str] = None
    reserve: Optional[str] = None
    guide: Optional[str] = None
    seasonal: Optional[str] = None
    weather_dependent: Optional[str] = None
    viral_rating: Optional[str] = None
    food_drink_pick: Optional[str] = None
    info_class: Optional[str] = None
    best_time: Optional[str] = None
    est_hours: Optional[float] = None  # parsed numeric estimate from time_needed
    lat: Optional[float] = None
    lon: Optional[float] = None

    def label(self) -> str:
        return self.specific_attraction or self.main_destination or self.id

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Option":
        return Option(**{k: v for k, v in d.items() if k in Option.__dataclass_fields__})


@dataclass
class Context:
    """A quick-intake snapshot of what's on my mind right now."""

    trip_type: str = "hangout_nearby"  # TRIP_TYPES
    duration_hint: Optional[str] = None  # free text if trip_type == "custom"
    energy: float = 0.5        # 0 (wiped out) .. 1 (wired)
    budget_level: int = 2      # index into COSTS the user is willing to spend up to
    party: str = "solo"        # solo / partner / friends / family
    indoor_outdoor: float = 0.0  # -1 indoor .. +1 outdoor, 0 = no preference
    mood_adventurous: float = 0.0  # -1 relaxed .. +1 adventurous
    season: Optional[str] = None
    time_of_day: Optional[str] = None
    extra_tags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

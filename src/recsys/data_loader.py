"""Parse Jack-Master-Travel-Database.xlsx into a flat option bank (content only).

The workbook was built by hand over months and its layout reflects that: each
region sheet repeats the same 27-column itinerary header once per
country/version block (see current_setup.md), interleaved with free-text
section titles and, in some sheets, small "Field / Detail" overview tables
that aren't itinerary rows at all. There are no stable row IDs.

This loader is intentionally tolerant of that shape: it scans every row for
the itinerary header signature, reads the data rows that follow each one, and
tags each row with the nearest preceding short title as its `section`. It
assigns a deterministic ID (sheet + section + destination + counter) so the
same real-world row maps to the same ID across rebuilds, which is what lets
`state.json` (done/repeat history) survive a content refresh.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Optional

import openpyxl

from .schema import Option

REGION_SHEETS = [
    "Cali", "Weekend Trips", "USA", "USA MEX CAN", "South America", "Europe",
    "East Asia", "South East Asia", "Middle East", "Africa", "Rest of World",
]

HEADER_SIGNATURE = ("Version", "Day")

FIELD_ORDER = [
    "version", "day", "date", "country", "city_region", "overnight", "route",
    "transport", "transit", "main_destination", "specific_attraction",
    "address", "website", "why_worth_it", "time_needed", "priority", "type",
    "time_of_day", "cost", "reserve", "guide", "seasonal", "weather_dependent",
    "viral_rating", "food_drink_pick", "info_class", "best_time",
]

_HOURS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(h|hour)", re.IGNORECASE)
_DAYS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(d|day)", re.IGNORECASE)


def _parse_hours(time_needed: Optional[str]) -> Optional[float]:
    """Best-effort numeric hours estimate from a free-text field like
    '2 to 3 days' or '~2h' or '3 days (tight; a 4th day helps)'."""
    if not time_needed:
        return None
    text = str(time_needed)
    days = _DAYS_RE.findall(text)
    if days:
        return max(float(v) for v, _ in days) * 8.0  # treat a "day" as ~8 waking hours
    hours = _HOURS_RE.findall(text)
    if hours:
        return max(float(v) for v, _ in hours)
    return None


def _is_section_title(row: tuple) -> Optional[str]:
    """A row with exactly one short populated cell is treated as a section title."""
    nonnull = [(i, v) for i, v in enumerate(row) if v not in (None, "")]
    if len(nonnull) == 1 and isinstance(nonnull[0][1], str):
        text = nonnull[0][1].strip()
        if 0 < len(text) <= 80:
            return text.lstrip("→").strip()
    return None


def _is_header_row(row: tuple) -> bool:
    return (
        len(row) > 1
        and isinstance(row[0], str) and row[0].strip() == HEADER_SIGNATURE[0]
        and isinstance(row[1], str) and row[1].strip() == HEADER_SIGNATURE[1]
    )


def _is_data_row(row: tuple) -> bool:
    nonnull = sum(1 for v in row[:27] if v not in (None, ""))
    return nonnull >= 6


def _slugify(text: str, maxlen: int = 24) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "-", str(text)).strip("-").upper()
    return text[:maxlen] if text else "X"


def parse_sheet(ws, sheet_name: str) -> list[Option]:
    rows = list(ws.iter_rows(values_only=True))
    options: list[Option] = []
    section = sheet_name
    counters: dict[str, int] = {}
    i = 0
    while i < len(rows):
        row = rows[i]
        title = _is_section_title(row)
        if title:
            section = title
            i += 1
            continue
        if _is_header_row(row):
            i += 1
            while i < len(rows) and _is_data_row(rows[i]):
                data_row = rows[i]
                values = dict(zip(FIELD_ORDER, data_row))
                dest = values.get("main_destination") or values.get("specific_attraction") or "ITEM"
                key = _slugify(f"{sheet_name}-{section}")
                counters[key] = counters.get(key, 0) + 1
                opt_id = f"{_slugify(sheet_name, 6)}-{_slugify(section, 12)}-{_slugify(dest, 10)}-{counters[key]:03d}"
                options.append(Option(
                    id=opt_id,
                    source_sheet=sheet_name,
                    section=section,
                    version=values.get("version"),
                    day=str(values.get("day")) if values.get("day") is not None else None,
                    country=values.get("country"),
                    city_region=values.get("city_region"),
                    overnight=values.get("overnight"),
                    route=values.get("route"),
                    transport=values.get("transport"),
                    transit=values.get("transit"),
                    main_destination=values.get("main_destination"),
                    specific_attraction=values.get("specific_attraction"),
                    address=values.get("address"),
                    website=values.get("website"),
                    why_worth_it=values.get("why_worth_it"),
                    time_needed=values.get("time_needed"),
                    priority=values.get("priority"),
                    type=values.get("type"),
                    time_of_day=values.get("time_of_day"),
                    cost=values.get("cost"),
                    reserve=values.get("reserve"),
                    guide=values.get("guide"),
                    seasonal=values.get("seasonal"),
                    weather_dependent=values.get("weather_dependent"),
                    viral_rating=values.get("viral_rating"),
                    food_drink_pick=values.get("food_drink_pick"),
                    info_class=values.get("info_class"),
                    best_time=values.get("best_time"),
                    est_hours=_parse_hours(values.get("time_needed")),
                ))
                i += 1
            continue
        i += 1
    return options


def build_option_bank(xlsx_path: Path, sheets: Iterable[str] = REGION_SHEETS) -> list[Option]:
    wb = openpyxl.load_workbook(xlsx_path, data_only=True, read_only=True)
    all_options: list[Option] = []
    for name in sheets:
        if name not in wb.sheetnames:
            continue
        all_options.extend(parse_sheet(wb[name], name))
    return all_options


def save_option_bank(options: list[Option], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([o.to_dict() for o in options], f, indent=2, ensure_ascii=False)


def load_option_bank(path: Path) -> list[Option]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Option.from_dict(d) for d in raw]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild data/option_bank.json from the master workbook.")
    parser.add_argument("--xlsx", default="Jack-Master-Travel-Database.xlsx")
    parser.add_argument("--out", default="data/option_bank.json")
    args = parser.parse_args()

    options = build_option_bank(Path(args.xlsx))
    save_option_bank(options, Path(args.out))
    print(f"Parsed {len(options)} options from {len(REGION_SHEETS)} sheets -> {args.out}")

    by_sheet: dict[str, int] = {}
    for o in options:
        by_sheet[o.source_sheet] = by_sheet.get(o.source_sheet, 0) + 1
    for sheet, count in by_sheet.items():
        print(f"  {sheet}: {count}")


if __name__ == "__main__":
    main()

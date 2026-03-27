from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import pandas as pd

try:
    import holidays
except ImportError:  # pragma: no cover - fallback when dependencies are not installed yet.
    holidays = None


STATE_CODES = {
    "NSW1": "NSW",
    "QLD1": "QLD",
    "SA1": "SA",
    "TAS1": "TAS",
    "VIC1": "VIC",
}


@lru_cache(maxsize=16)
def _country_holidays(year: int):
    if holidays is None:
        return {}
    return holidays.country_holidays("AU", years=[year])


@lru_cache(maxsize=32)
def _state_holidays(year: int, state: str):
    if holidays is None:
        return {}
    return holidays.country_holidays("AU", subdiv=STATE_CODES[state], years=[year])


def holiday_features(day: date, region: str) -> dict[str, int]:
    national = int(day in _country_holidays(day.year))
    regional = int(day in _state_holidays(day.year, region))
    prev_day = day - timedelta(days=1)
    next_day = day + timedelta(days=1)
    prev_or_next = int(
        prev_day in _country_holidays(prev_day.year)
        or next_day in _country_holidays(next_day.year)
        or prev_day in _state_holidays(prev_day.year, region)
        or next_day in _state_holidays(next_day.year, region)
    )
    return {
        "is_weekend": int(day.weekday() >= 5),
        "is_national_holiday": national,
        "is_state_holiday": regional,
        "adjacent_holiday": prev_or_next,
    }


def build_calendar_frame(timestamps: pd.DatetimeIndex, region: str) -> pd.DataFrame:
    frame = pd.DataFrame({"ds": timestamps})
    frame["hour"] = frame["ds"].dt.hour
    frame["minute"] = frame["ds"].dt.minute
    frame["slot_index"] = frame["hour"] * 2 + (frame["minute"] // 30)
    frame["day_of_week"] = frame["ds"].dt.dayofweek
    frame["month"] = frame["ds"].dt.month
    features = frame["ds"].dt.date.apply(lambda day: holiday_features(day, region))
    holiday_df = pd.DataFrame(list(features))
    return pd.concat([frame, holiday_df], axis=1)

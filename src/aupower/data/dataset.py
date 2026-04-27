from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aupower.calendar_utils import build_calendar_frame
from aupower.config import ForecastConfig


@dataclass
class SampleBundle:
    region: np.ndarray
    forecast_date: np.ndarray
    load_history: np.ndarray
    calendar_future: np.ndarray
    weather_future: np.ndarray
    event_context: np.ndarray
    target: np.ndarray


def _daily_forecast_dates(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")


def _event_feature_columns(event_frame: pd.DataFrame | None) -> list[str]:
    if event_frame is None or event_frame.empty:
        return []
    return [column for column in event_frame.columns if column not in {"date", "region"}]


def _region_one_hot(region: str, regions: list[str]) -> np.ndarray:
    vector = np.zeros(len(regions), dtype=np.float32)
    vector[regions.index(region)] = 1.0
    return vector


def build_samples(
    load_wide: pd.DataFrame,
    config: ForecastConfig,
    start: str,
    end: str,
    lookback_days: int,
    weather_frame: pd.DataFrame | None = None,
    event_frame: pd.DataFrame | None = None,
) -> SampleBundle:
    forecast_dates = _daily_forecast_dates(start, end)
    weather_indexed = None
    event_indexed = None
    weather_columns = []
    event_columns = _event_feature_columns(event_frame)
    if weather_frame is not None and not weather_frame.empty:
        weather_columns = [column for column in weather_frame.columns if column.startswith("forecast_")]
        weather_indexed = weather_frame.set_index(["date", "region"]).sort_index()
    if event_frame is not None and not event_frame.empty:
        event_indexed = event_frame.set_index(["date", "region"]).sort_index()

    history_steps = lookback_days * 48
    region_values = []
    date_values = []
    load_blocks = []
    calendar_blocks = []
    weather_blocks = []
    event_blocks = []
    target_blocks = []

    for region in config.data.regions:
        for forecast_date in forecast_dates:
            history_end = forecast_date - pd.Timedelta(minutes=30)
            history_index = pd.date_range(end=history_end, periods=history_steps, freq="30min")
            future_index = pd.date_range(start=forecast_date, periods=config.experts.horizon, freq="30min")
            if any(ts not in load_wide.index for ts in history_index) or any(ts not in load_wide.index for ts in future_index):
                continue
            history = load_wide.loc[history_index, config.data.regions].to_numpy(dtype=np.float32).reshape(-1)
            target = load_wide.loc[future_index, region].to_numpy(dtype=np.float32)
            calendar = build_calendar_frame(future_index, region).drop(columns=["ds"]).to_numpy(dtype=np.float32).reshape(-1)
            calendar = np.concatenate([calendar, _region_one_hot(region, config.data.regions)], axis=0)
            if weather_indexed is not None and (forecast_date, region) in weather_indexed.index:
                weather_values = weather_indexed.loc[(forecast_date, region), weather_columns]
                weather_vector = np.asarray(weather_values, dtype=np.float32).reshape(-1)
            else:
                weather_vector = np.zeros(len(weather_columns), dtype=np.float32)
            event_date = forecast_date - pd.Timedelta(days=1)
            if event_indexed is not None and (event_date, region) in event_indexed.index:
                event_values = event_indexed.loc[(event_date, region), event_columns]
                event_vector = np.asarray(event_values, dtype=np.float32).reshape(-1)
            else:
                event_vector = np.zeros(len(event_columns), dtype=np.float32)

            region_values.append(region)
            date_values.append(forecast_date.strftime("%Y-%m-%d"))
            load_blocks.append(history)
            calendar_blocks.append(calendar)
            weather_blocks.append(weather_vector)
            event_blocks.append(event_vector)
            target_blocks.append(target)

    return SampleBundle(
        region=np.asarray(region_values),
        forecast_date=np.asarray(date_values),
        load_history=np.asarray(load_blocks, dtype=np.float32),
        calendar_future=np.asarray(calendar_blocks, dtype=np.float32),
        weather_future=np.asarray(weather_blocks, dtype=np.float32) if weather_blocks else np.zeros((0, 0), dtype=np.float32),
        event_context=np.asarray(event_blocks, dtype=np.float32) if event_blocks else np.zeros((0, 0), dtype=np.float32),
        target=np.asarray(target_blocks, dtype=np.float32),
    )

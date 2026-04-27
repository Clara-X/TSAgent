from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from aupower.calendar_utils import holiday_features
from aupower.config import ForecastConfig

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None


WEATHER_SUFFIX = "_missing"


@dataclass
class WeatherArtifacts:
    clean_frame: pd.DataFrame
    proxy_forecasts: pd.DataFrame
    feature_columns: list[str]
    target_columns: list[str]


def clean_weather_frame(config: ForecastConfig) -> pd.DataFrame:
    frame = pd.read_csv(config.data.weather_path)
    frame["date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).copy()
    frame["region"] = frame["State"].map(config.data.state_to_region)
    frame = frame[frame["region"].isin(config.data.regions)].copy()
    target_columns = config.data.stable_weather_columns + ["Total Precipitation"]
    for column in target_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[f"{column}{WEATHER_SUFFIX}"] = frame[column].isna().astype(int)
    frame = frame.sort_values(["region", "date"]).reset_index(drop=True)

    stable = config.data.stable_weather_columns
    frame[stable] = frame.groupby("region")[stable].ffill()
    month_key = frame["date"].dt.month
    for column in stable:
        monthly_median = frame.groupby(["region", month_key])[column].transform("median")
        frame[column] = frame[column].fillna(monthly_median)
        frame[column] = frame.groupby("region")[column].transform(lambda series: series.fillna(series.median()))
    frame["precip_missing"] = frame["Total Precipitation"].isna().astype(int)
    keep_columns = ["date", "region", *stable, "precip_missing", *[f"{column}{WEATHER_SUFFIX}" for column in stable]]
    return frame[keep_columns].sort_values(["region", "date"]).reset_index(drop=True)


def _build_backend(config: ForecastConfig):
    if config.weather.backend in {"lightgbm", "auto"} and LGBMRegressor is not None:
        model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=42,
        )
        return MultiOutputRegressor(model)
    return MultiOutputRegressor(HistGradientBoostingRegressor(random_state=42, max_depth=8))


def _weather_feature_frame(frame: pd.DataFrame, config: ForecastConfig) -> tuple[pd.DataFrame, list[str], list[str]]:
    stable = config.data.stable_weather_columns
    base = frame.copy()
    base["month"] = base["date"].dt.month
    base["day_of_week"] = base["date"].dt.dayofweek
    base["day_of_year"] = base["date"].dt.dayofyear
    region_dummies = pd.get_dummies(base["region"], prefix="region", dtype=int)
    base = pd.concat([base, region_dummies], axis=1)
    for lag in config.weather.lags:
        lagged = (
            base.groupby("region")[stable]
            .shift(lag)
            .add_prefix(f"lag_{lag}_")
        )
        base = pd.concat([base, lagged], axis=1)

    holiday_rows = [holiday_features(day.date(), region) for day, region in zip(base["date"], base["region"])]
    base = pd.concat([base, pd.DataFrame(holiday_rows)], axis=1)
    feature_columns = [
        col
        for col in base.columns
        if col.startswith("lag_")
        or col.startswith("region_")
        or col in {"month", "day_of_week", "day_of_year", "is_weekend", "is_national_holiday", "is_state_holiday", "adjacent_holiday"}
    ]
    target_columns = stable
    dataset = base.dropna(subset=feature_columns + target_columns).reset_index(drop=True)
    return dataset, feature_columns, target_columns


def train_weather_proxy_models(frame: pd.DataFrame, config: ForecastConfig, output_path: str | Path) -> WeatherArtifacts:
    dataset, feature_columns, target_columns = _weather_feature_frame(frame, config)
    cutoff = pd.Timestamp(config.splits.pretrain_end)
    train = dataset[dataset["date"] <= cutoff].copy()
    model = _build_backend(config)
    model.fit(train[feature_columns], train[target_columns])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump({"model": model, "features": feature_columns, "targets": target_columns}, handle)

    future = dataset[dataset["date"] >= pd.Timestamp(config.splits.multimodal_start)].copy()
    predicted = model.predict(future[feature_columns])
    for idx, column in enumerate(target_columns):
        future[f"forecast_{column}"] = predicted[:, idx]

    keep_cols = ["date", "region", *[f"forecast_{column}" for column in target_columns]]
    proxy = future[keep_cols].drop_duplicates(["date", "region"]).reset_index(drop=True)
    return WeatherArtifacts(clean_frame=frame, proxy_forecasts=proxy, feature_columns=feature_columns, target_columns=target_columns)

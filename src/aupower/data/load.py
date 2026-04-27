from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from aupower.config import ForecastConfig


@dataclass
class LoadArtifacts:
    long_frame: pd.DataFrame
    wide_frame: pd.DataFrame


def _assign_split(day: pd.Timestamp, config: ForecastConfig) -> str:
    if day <= pd.Timestamp(config.splits.pretrain_end):
        return "pretrain"
    if day <= pd.Timestamp(config.splits.train_end):
        return "train"
    if day <= pd.Timestamp(config.splits.val_end):
        return "val"
    if day <= pd.Timestamp(config.splits.test_end):
        return "test"
    return "holdout"


def prepare_load_artifacts(config: ForecastConfig) -> LoadArtifacts:
    usecols = ["REGION", "SETTLEMENTDATE", "TOTALDEMAND"]
    frame = pd.read_csv(config.data.load_path, usecols=usecols)
    frame = frame[frame["REGION"].isin(config.data.regions)].copy()
    frame["SETTLEMENTDATE"] = pd.to_datetime(frame["SETTLEMENTDATE"], errors="coerce")
    frame = frame.dropna(subset=["SETTLEMENTDATE"])
    frame["TOTALDEMAND"] = pd.to_numeric(frame["TOTALDEMAND"], errors="coerce")
    frame = frame.dropna(subset=["TOTALDEMAND"])
    frame["ds"] = frame["SETTLEMENTDATE"].dt.floor("30min")

    grouped = (
        frame.groupby(["REGION", "ds"], as_index=False)["TOTALDEMAND"]
        .mean()
        .sort_values(["REGION", "ds"])
        .rename(columns={"REGION": "region", "TOTALDEMAND": "y"})
    )
    wide = grouped.pivot(index="ds", columns="region", values="y").sort_index()
    full_index = pd.date_range(wide.index.min(), wide.index.max(), freq="30min")
    wide = wide.reindex(full_index).sort_index()
    wide = wide.ffill().bfill()
    wide.index.name = "ds"
    wide = wide[config.data.regions]

    long_frame = (
        wide.reset_index()
        .melt(id_vars="ds", value_vars=config.data.regions, var_name="region", value_name="y")
        .sort_values(["region", "ds"])
        .reset_index(drop=True)
    )
    long_frame["date"] = long_frame["ds"].dt.normalize()
    long_frame["split"] = long_frame["date"].apply(lambda day: _assign_split(day, config))
    return LoadArtifacts(long_frame=long_frame, wide_frame=wide)

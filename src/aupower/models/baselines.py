from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from aupower.data.dataset import SampleBundle

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None


def seasonal_naive_matrix(load_wide, regions: list[str], forecast_dates: list[str], horizon: int = 48, season_days: int = 7):
    predictions = {}
    for region in regions:
        region_preds = []
        for forecast_date in forecast_dates:
            start = np.datetime64(forecast_date)
            history_start = start - np.timedelta64(season_days, "D")
            index = load_wide.loc[str(history_start) : str(history_start + np.timedelta64(23, "h") + np.timedelta64(30, "m")), region]
            region_preds.append(index.to_numpy(dtype=np.float32)[:horizon])
        predictions[region] = np.asarray(region_preds, dtype=np.float32)
    return predictions


@dataclass
class LagBoostingBaseline:
    model: MultiOutputRegressor | None = None

    def _build_backend(self) -> MultiOutputRegressor:
        if LGBMRegressor is not None:
            regressor = LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.8,
                random_state=42,
            )
            return MultiOutputRegressor(regressor)
        return MultiOutputRegressor(HistGradientBoostingRegressor(max_depth=8, random_state=42))

    def fit(self, samples: SampleBundle) -> "LagBoostingBaseline":
        features = np.concatenate([samples.load_history, samples.calendar_future], axis=1)
        self.model = self._build_backend()
        self.model.fit(features, samples.target)
        return self

    def predict(self, samples: SampleBundle) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LagBoostingBaseline must be fitted before calling predict()")
        features = np.concatenate([samples.load_history, samples.calendar_future], axis=1)
        return np.asarray(self.model.predict(features), dtype=np.float32)

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "LagBoostingBaseline":
        with Path(path).open("rb") as handle:
            return pickle.load(handle)

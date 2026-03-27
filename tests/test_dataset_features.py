from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from aupower.config import ForecastConfig
from aupower.data.dataset import build_samples


class DatasetFeatureTests(unittest.TestCase):
    def test_calendar_features_include_region_one_hot(self) -> None:
        config = ForecastConfig()
        idx = pd.date_range("2021-12-31 00:00:00", "2022-01-01 23:30:00", freq="30min")
        load_wide = pd.DataFrame(index=idx)
        for i, region in enumerate(config.data.regions, start=1):
            load_wide[region] = float(i) * np.ones(len(idx), dtype=np.float32)

        samples = build_samples(
            load_wide=load_wide,
            config=config,
            start="2022-01-01",
            end="2022-01-01",
            lookback_days=1,
        )
        self.assertEqual(samples.calendar_future.shape[0], len(config.data.regions))
        self.assertEqual(samples.calendar_future.shape[1], 48 * 9 + len(config.data.regions))

        for row_idx, region in enumerate(config.data.regions):
            one_hot = samples.calendar_future[row_idx, -len(config.data.regions) :]
            expected = np.zeros(len(config.data.regions), dtype=np.float32)
            expected[config.data.regions.index(region)] = 1.0
            self.assertTrue(np.array_equal(one_hot, expected))


if __name__ == "__main__":
    unittest.main()

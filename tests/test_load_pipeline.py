from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from aupower.config import ForecastConfig
from aupower.data.load import prepare_load_artifacts


class LoadPipelineTests(unittest.TestCase):
    def test_resamples_five_minute_rows_to_thirty_minute(self) -> None:
        rows = [
            "REGION,SETTLEMENTDATE,TOTALDEMAND",
            "NSW1,2021-10-01 00:00:00,10",
            "NSW1,2021-10-01 00:05:00,20",
            "NSW1,2021-10-01 00:10:00,30",
            "NSW1,2021-10-01 00:15:00,40",
            "NSW1,2021-10-01 00:20:00,50",
            "NSW1,2021-10-01 00:25:00,60",
            "QLD1,2021-10-01 00:00:00,5",
            "QLD1,2021-10-01 00:30:00,7",
            "SA1,2021-10-01 00:00:00,1",
            "SA1,2021-10-01 00:30:00,2",
            "TAS1,2021-10-01 00:00:00,3",
            "TAS1,2021-10-01 00:30:00,4",
            "VIC1,2021-10-01 00:00:00,8",
            "VIC1,2021-10-01 00:30:00,9",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "load.csv"
            csv_path.write_text("\n".join(rows), encoding="utf-8")
            config = ForecastConfig()
            config.data.load_path = str(csv_path)
            artifacts = prepare_load_artifacts(config)
        wide = artifacts.wide_frame
        self.assertAlmostEqual(float(wide.loc[pd.Timestamp("2021-10-01 00:00:00"), "NSW1"]), 35.0)


if __name__ == "__main__":
    unittest.main()

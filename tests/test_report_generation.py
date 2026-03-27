from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from aupower.pipeline import generate_report


class ReportGenerationTests(unittest.TestCase):
    def test_generate_report_writes_markdown_and_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifacts = root / "artifacts"
            reports = artifacts / "reports"
            models = artifacts / "models"
            reports.mkdir(parents=True, exist_ok=True)
            models.mkdir(parents=True, exist_ok=True)

            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "project:",
                        f"  processed_dir: {artifacts / 'processed'}",
                        f"  model_dir: {models}",
                        f"  report_dir: {reports}",
                        f"  log_dir: {artifacts / 'logs'}",
                    ]
                ),
                encoding="utf-8",
            )

            summary = {
                "lag_baseline": {"wmape": 0.10, "mae": 10.0, "rmse": 12.0, "smape": 0.11},
                "base_only": {"wmape": 0.09, "mae": 9.0, "rmse": 11.0, "smape": 0.10},
                "base_weather": {"wmape": 0.08, "mae": 8.0, "rmse": 10.0, "smape": 0.09},
                "fixed_multimodal": {"wmape": 0.07, "mae": 7.0, "rmse": 9.0, "smape": 0.08},
                "agent_routed": {"wmape": 0.068, "mae": 6.8, "rmse": 8.8, "smape": 0.078},
                "agent_routed_peak_wmape": 0.06,
                "fixed_multimodal_peak_wmape": 0.065,
                "agent_routed_event_day_wmape": 0.05,
                "fixed_multimodal_event_day_wmape": 0.055,
                "by_region": {
                    "NSW1": {"wmape": 0.05, "mae": 5.0, "rmse": 6.0, "smape": 0.05},
                    "QLD1": {"wmape": 0.06, "mae": 6.0, "rmse": 7.0, "smape": 0.06},
                    "SA1": {"wmape": 0.07, "mae": 7.0, "rmse": 8.0, "smape": 0.07},
                    "TAS1": {"wmape": 0.08, "mae": 8.0, "rmse": 9.0, "smape": 0.08},
                    "VIC1": {"wmape": 0.09, "mae": 9.0, "rmse": 10.0, "smape": 0.09},
                },
            }
            (reports / "backtest_summary.json").write_text(json.dumps(summary), encoding="utf-8")

            pd.DataFrame(
                [
                    {
                        "region": "NSW1",
                        "forecast_date": "2022-01-01",
                        "risk_level": "low",
                        "base_weight": 0.2,
                        "weather_weight": 0.7,
                        "event_weight": 0.1,
                        "issue_date": "2021-12-31",
                        "event_day": 0,
                        "actual_sum": 100.0,
                        "fixed_abs_error_sum": 7.0,
                        "agent_abs_error_sum": 6.5,
                        "fixed_rmse_day": 1.0,
                        "agent_rmse_day": 0.9,
                    },
                    {
                        "region": "QLD1",
                        "forecast_date": "2022-01-01",
                        "risk_level": "medium",
                        "base_weight": 0.3,
                        "weather_weight": 0.5,
                        "event_weight": 0.2,
                        "issue_date": "2021-12-31",
                        "event_day": 1,
                        "actual_sum": 120.0,
                        "fixed_abs_error_sum": 8.0,
                        "agent_abs_error_sum": 7.0,
                        "fixed_rmse_day": 1.1,
                        "agent_rmse_day": 0.95,
                    },
                ]
            ).to_csv(reports / "backtest_predictions.csv", index=False)

            prior_payload = {
                "region_weights": {
                    "NSW1": {"base": 0.2, "weather": 0.7, "event": 0.1},
                    "QLD1": {"base": 0.3, "weather": 0.5, "event": 0.2},
                    "SA1": {"base": 0.4, "weather": 0.4, "event": 0.2},
                    "TAS1": {"base": 0.4, "weather": 0.4, "event": 0.2},
                    "VIC1": {"base": 0.4, "weather": 0.4, "event": 0.2},
                },
                "global_weights": {"base": 0.3, "weather": 0.5, "event": 0.2},
                "prior_blend": 0.7,
                "metrics": {"global_val_wmape": 0.07, "region_val_wmape": {"NSW1": 0.05}},
            }
            (models / "region_weight_priors.json").write_text(json.dumps(prior_payload), encoding="utf-8")
            (models / "learned_router_metrics.json").write_text(
                json.dumps({"val_wmape_rule_router": 0.07, "val_wmape_best_blend": 0.069, "blend_alpha": 0.0}),
                encoding="utf-8",
            )

            paths = generate_report(str(config_path))
            self.assertTrue(paths.report_md.exists())
            self.assertTrue(paths.ablation_csv.exists())
            self.assertTrue(paths.routing_summary_csv.exists())
            self.assertTrue(paths.monthly_chart_png.exists())
            self.assertTrue(paths.holiday_chart_png.exists())
            self.assertTrue(paths.peak_chart_png.exists())
            report_text = paths.report_md.read_text(encoding="utf-8")
            self.assertIn("Executive Summary", report_text)
            self.assertIn("Acceptance result", report_text)


if __name__ == "__main__":
    unittest.main()

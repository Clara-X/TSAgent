from __future__ import annotations

import unittest

import numpy as np

from aupower.agent.ollama_router import PriorCalibratedRouter, context_to_feature_vector, fit_learned_router_model, oracle_weight_target
from aupower.config import ForecastConfig


class LearnedRouterTests(unittest.TestCase):
    def test_oracle_weight_prefers_best_expert(self) -> None:
        actual = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float32)
        base = actual.copy()
        weather = actual + 5.0
        event = actual - 4.0
        weights = oracle_weight_target(base, weather, event, actual, step=0.05)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        combined = weights[0] * base + weights[1] * weather + weights[2] * event
        self.assertLess(float(np.mean(np.abs(combined - actual))), 1e-4)

    def test_context_feature_vector_shape(self) -> None:
        regions = ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]
        context = {
            "forecast_temp_max": 305.0,
            "forecast_temp_min": 290.0,
            "forecast_wind": 8.0,
            "weather_extreme_score": 0.4,
            "event_severity_7d": 1.0,
            "outage_score_7d": 0.2,
            "market_stress_7d": 0.3,
            "base_residual_14d": 0.1,
            "weather_residual_14d": 0.2,
            "event_residual_14d": 0.15,
        }
        vec = context_to_feature_vector("SA1", "2022-05-14", context, regions)
        self.assertEqual(vec.shape[0], 19)
        self.assertEqual(float(vec[-3]), 1.0)

    def test_fit_learned_router_model_returns_blend(self) -> None:
        rng = np.random.default_rng(7)
        n_samples = 64
        horizon = 8
        features = rng.normal(size=(n_samples, 19)).astype(np.float32)
        base_preds = rng.normal(loc=100.0, scale=5.0, size=(n_samples, horizon)).astype(np.float32)
        weather_preds = base_preds + rng.normal(loc=0.3, scale=2.0, size=(n_samples, horizon)).astype(np.float32)
        event_preds = base_preds + rng.normal(loc=-0.2, scale=2.0, size=(n_samples, horizon)).astype(np.float32)
        actual = 0.5 * base_preds + 0.3 * weather_preds + 0.2 * event_preds
        oracle_weights = np.tile(np.array([0.5, 0.3, 0.2], dtype=np.float32), (n_samples, 1))
        rule_weights = np.tile(np.array([0.4, 0.4, 0.2], dtype=np.float32), (n_samples, 1))

        payload = fit_learned_router_model(
            features=features,
            oracle_weights=oracle_weights,
            rule_weights=rule_weights,
            base_preds=base_preds,
            weather_preds=weather_preds,
            event_preds=event_preds,
            actual=actual,
        )
        self.assertIn("model", payload)
        self.assertIn("blend_alpha", payload)
        self.assertGreaterEqual(float(payload["blend_alpha"]), 0.0)
        self.assertLessEqual(float(payload["blend_alpha"]), 1.0)

    def test_prior_router_uses_region_prior_blend(self) -> None:
        config = ForecastConfig()
        router = PriorCalibratedRouter(
            config=config,
            region_weights={"NSW1": {"base": 0.2, "weather": 0.7, "event": 0.1}},
            global_weights={"base": 0.4, "weather": 0.4, "event": 0.2},
            prior_blend=0.7,
        )
        context = {
            "forecast_temp_max": 305.0,
            "forecast_temp_min": 290.0,
            "forecast_wind": 10.0,
            "weather_extreme_score": 0.0,
            "event_severity_7d": 0.0,
            "outage_score_7d": 0.0,
            "market_stress_7d": 0.0,
            "base_residual_14d": 0.3,
            "weather_residual_14d": 0.1,
            "event_residual_14d": 0.5,
        }
        decision = router.route("NSW1", "2022-05-14", context)
        self.assertIn("region_prior_blend", decision.reason_codes)
        self.assertGreater(decision.expert_weights.weather, 0.6)


if __name__ == "__main__":
    unittest.main()

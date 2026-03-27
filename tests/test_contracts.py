from __future__ import annotations

import unittest

from aupower.contracts import AgentDecision, ExpertWeights, ForecastOutput


class ContractTests(unittest.TestCase):
    def test_expert_weights_must_sum_to_one(self) -> None:
        with self.assertRaises(ValueError):
            ExpertWeights(base=0.5, weather=0.5, event=0.5)

    def test_forecast_output_requires_48_steps(self) -> None:
        with self.assertRaises(ValueError):
            ForecastOutput(
                region="NSW1",
                forecast_date="2022-01-01",
                pred_30m=[1.0, 2.0],
                expert_preds={"base": [1.0] * 48, "weather": [1.0] * 48, "event": [1.0] * 48},
                agent_decision=AgentDecision(
                    region="NSW1",
                    issue_date="2021-12-31",
                    expert_weights=ExpertWeights(base=0.4, weather=0.4, event=0.2),
                    risk_level="low",
                    use_event_context=True,
                    reason_codes=["default"],
                ),
                risk_level="low",
            )


if __name__ == "__main__":
    unittest.main()

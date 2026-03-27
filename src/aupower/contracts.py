from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


Region = Literal["NSW1", "QLD1", "SA1", "TAS1", "VIC1", "NATIONAL"]
RiskLevel = Literal["low", "medium", "high"]


class ExpertWeights(BaseModel):
    base: float = Field(ge=0.0)
    weather: float = Field(ge=0.0)
    event: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_sum(self) -> "ExpertWeights":
        total = self.base + self.weather + self.event
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"expert weights must sum to 1.0, got {total:.6f}")
        return self


class AgentDecision(BaseModel):
    region: Region
    issue_date: str
    expert_weights: ExpertWeights
    risk_level: RiskLevel
    use_event_context: bool
    reason_codes: list[str] = Field(default_factory=list)


class NewsEventRecord(BaseModel):
    article_id: str
    publication_time: str
    regions: list[Region]
    is_energy_relevant: bool
    event_type: Literal[
        "outage",
        "extreme_weather",
        "market_stress",
        "policy",
        "generation",
        "network",
        "pricing",
        "other",
    ]
    impact_direction: Literal["up", "down", "neutral", "unknown"]
    impact_horizon: Literal["intraday", "1d", "3d", "7d", "longer", "unknown"]
    severity: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    summary_2lines: str

    @field_validator("regions")
    @classmethod
    def non_empty_regions(cls, regions: list[Region]) -> list[Region]:
        if not regions:
            raise ValueError("regions must not be empty")
        return regions


class ForecastRequest(BaseModel):
    region: Region
    issue_ts: str
    history_load_30m: list[list[float]]
    calendar_future: list[dict[str, float | int | bool]]
    weather_forecast_daily: dict[str, float | int | bool]
    event_features_daily: dict[str, float | int | bool]


class ForecastOutput(BaseModel):
    region: Region
    forecast_date: str
    pred_30m: list[float]
    expert_preds: dict[str, list[float]]
    agent_decision: AgentDecision
    risk_level: RiskLevel

    @field_validator("pred_30m")
    @classmethod
    def validate_horizon(cls, preds: list[float]) -> list[float]:
        if len(preds) != 48:
            raise ValueError("pred_30m must have 48 points for a day-ahead forecast")
        return preds

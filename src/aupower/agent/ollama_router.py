from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import requests
from sklearn.linear_model import Ridge

from aupower.config import ForecastConfig
from aupower.contracts import AgentDecision, ExpertWeights
from aupower.metrics import wmape


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("no JSON object found in Ollama response")


@dataclass
class OllamaClient:
    base_url: str
    timeout_seconds: int = 60

    def generate_json(self, model: str, prompt: str, temperature: int = 0) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": temperature},
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return _extract_json(payload.get("response", "{}"))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


_CONTEXT_NUMERIC_FEATURES = [
    "forecast_temp_max",
    "forecast_temp_min",
    "forecast_wind",
    "weather_extreme_score",
    "event_severity_7d",
    "outage_score_7d",
    "market_stress_7d",
    "base_residual_14d",
    "weather_residual_14d",
    "event_residual_14d",
]


def _normalize_weight_vector(values: np.ndarray | list[float], default: list[float] | None = None) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    vector = np.clip(vector, 0.0, None)
    if float(vector.sum()) <= 1e-8:
        if default is None:
            default = [0.4, 0.4, 0.2]
        vector = np.asarray(default, dtype=np.float32)
    vector = vector / float(vector.sum())
    return vector.astype(np.float32)


def context_to_feature_vector(region: str, issue_date: str, context: dict[str, float | int | str | bool], regions: list[str]) -> np.ndarray:
    issue_day = date.fromisoformat(issue_date)
    numeric = [float(context.get(key, 0.0)) for key in _CONTEXT_NUMERIC_FEATURES]
    calendar = [float(issue_day.weekday()), float(issue_day.month), float(issue_day.day), float(issue_day.timetuple().tm_yday)]
    region_ohe = np.zeros(len(regions), dtype=np.float32)
    if region in regions:
        region_ohe[regions.index(region)] = 1.0
    return np.concatenate([np.asarray(numeric + calendar, dtype=np.float32), region_ohe], axis=0).astype(np.float32)


def _weight_grid(step: float = 0.05) -> np.ndarray:
    units = int(round(1.0 / step))
    rows: list[list[float]] = []
    for base in range(units + 1):
        for weather in range(units + 1 - base):
            event = units - base - weather
            rows.append([base * step, weather * step, event * step])
    return np.asarray(rows, dtype=np.float32)


def oracle_weight_target(
    base_pred: np.ndarray,
    weather_pred: np.ndarray,
    event_pred: np.ndarray,
    actual: np.ndarray,
    step: float = 0.05,
) -> np.ndarray:
    grid = _weight_grid(step)
    experts = np.stack([base_pred, weather_pred, event_pred], axis=0).astype(np.float32)
    combined = grid @ experts
    losses = np.mean(np.abs(combined - actual[None, :]), axis=1)
    return grid[int(np.argmin(losses))]


def fit_learned_router_model(
    features: np.ndarray,
    oracle_weights: np.ndarray,
    rule_weights: np.ndarray,
    base_preds: np.ndarray,
    weather_preds: np.ndarray,
    event_preds: np.ndarray,
    actual: np.ndarray,
) -> dict[str, Any]:
    rule_weights = np.asarray([_normalize_weight_vector(row) for row in rule_weights], dtype=np.float32)
    n_samples = len(features)
    split_idx = int(round(n_samples * 0.75))
    if split_idx <= 0 or split_idx >= n_samples:
        split_idx = n_samples // 2
    train_idx = np.arange(split_idx)
    tune_idx = np.arange(split_idx, n_samples)

    train_model = Ridge(alpha=8.0, fit_intercept=True)
    train_model.fit(features[train_idx], oracle_weights[train_idx])
    tune_raw = train_model.predict(features[tune_idx])
    tune_learned = np.asarray([_normalize_weight_vector(row) for row in tune_raw], dtype=np.float32)

    rule_tune = rule_weights[tune_idx]
    rule_tune_pred = (
        rule_tune[:, 0:1] * base_preds[tune_idx]
        + rule_tune[:, 1:2] * weather_preds[tune_idx]
        + rule_tune[:, 2:3] * event_preds[tune_idx]
    )
    rule_tune_wmape = wmape(actual[tune_idx], rule_tune_pred)

    best_alpha = 0.0
    best_tune_wmape = rule_tune_wmape
    for alpha in np.linspace(0.0, 0.6, num=7):
        mixed = np.asarray(
            [_normalize_weight_vector((1.0 - alpha) * rule_tune[i] + alpha * tune_learned[i]) for i in range(len(tune_idx))],
            dtype=np.float32,
        )
        pred = mixed[:, 0:1] * base_preds[tune_idx] + mixed[:, 1:2] * weather_preds[tune_idx] + mixed[:, 2:3] * event_preds[tune_idx]
        score = wmape(actual[tune_idx], pred)
        if score < best_tune_wmape:
            best_tune_wmape = score
            best_alpha = float(alpha)

    # Require minimum gain on the tuning split, otherwise keep pure rule routing.
    min_relative_gain = 0.002
    if best_tune_wmape > rule_tune_wmape * (1.0 - min_relative_gain):
        best_alpha = 0.0

    model = Ridge(alpha=8.0, fit_intercept=True)
    model.fit(features, oracle_weights)
    learned_raw = model.predict(features)
    learned_weights = np.asarray([_normalize_weight_vector(row) for row in learned_raw], dtype=np.float32)

    mixed_full = np.asarray(
        [_normalize_weight_vector((1.0 - best_alpha) * rule_weights[i] + best_alpha * learned_weights[i]) for i in range(n_samples)],
        dtype=np.float32,
    )
    rule_pred = rule_weights[:, 0:1] * base_preds + rule_weights[:, 1:2] * weather_preds + rule_weights[:, 2:3] * event_preds
    learned_pred = learned_weights[:, 0:1] * base_preds + learned_weights[:, 1:2] * weather_preds + learned_weights[:, 2:3] * event_preds
    mixed_pred = mixed_full[:, 0:1] * base_preds + mixed_full[:, 1:2] * weather_preds + mixed_full[:, 2:3] * event_preds
    metrics = {
        "val_wmape_rule_router": wmape(actual, rule_pred),
        "val_wmape_learned_router": wmape(actual, learned_pred),
        "val_wmape_best_blend": wmape(actual, mixed_pred),
        "tune_wmape_rule_router": rule_tune_wmape,
        "tune_wmape_best_blend": best_tune_wmape,
        "blend_alpha": best_alpha,
        "n_samples": int(len(features)),
    }
    return {"model": model, "blend_alpha": best_alpha, "metrics": metrics}


class RuleBasedRouter:
    def __init__(self, config: ForecastConfig):
        self.defaults = config.agent.default_weights

    def route(self, region: str, issue_date: str, context: dict[str, float | int | str | bool]) -> AgentDecision:
        base = self.defaults["base"]
        weather = self.defaults["weather"]
        event = self.defaults["event"]
        reasons = []

        base_residual = float(context.get("base_residual_14d", 1.0))
        weather_residual = float(context.get("weather_residual_14d", 1.0))
        event_residual = float(context.get("event_residual_14d", 1.0))

        # Build a safe prior from recent residuals so the router tracks which expert
        # has been stronger on the latest validation window for this region.
        inv_base = 1.0 / max(base_residual, 1e-6)
        inv_weather = 1.0 / max(weather_residual, 1e-6)
        inv_event = 1.0 / max(event_residual, 1e-6)
        inv_total = inv_base + inv_weather + inv_event
        residual_prior_base = inv_base / inv_total
        residual_prior_weather = inv_weather / inv_total
        residual_prior_event = inv_event / inv_total
        prior_alpha = 0.65
        base = (1.0 - prior_alpha) * base + prior_alpha * residual_prior_base
        weather = (1.0 - prior_alpha) * weather + prior_alpha * residual_prior_weather
        event = (1.0 - prior_alpha) * event + prior_alpha * residual_prior_event
        reasons.append("residual_adaptive_prior")

        if float(context.get("event_severity_7d", 0.0)) > 2.5 or float(context.get("outage_score_7d", 0.0)) > 1.5:
            event += 0.12
            base -= 0.05
            weather -= 0.07
            reasons.append("event_spike")
        if float(context.get("weather_extreme_score", 0.0)) > 0.7:
            weather += 0.12
            base -= 0.08
            event -= 0.04
            reasons.append("extreme_weather")
        if event_residual <= 0.95 * min(base_residual, weather_residual):
            event += 0.08
            base -= 0.04
            weather -= 0.04
            reasons.append("event_recently_best")
        elif weather_residual <= 0.95 * min(base_residual, event_residual):
            weather += 0.08
            base -= 0.04
            event -= 0.04
            reasons.append("weather_recently_best")
        elif base_residual <= 0.95 * min(weather_residual, event_residual):
            base += 0.08
            weather -= 0.04
            event -= 0.04
            reasons.append("base_recently_better")
        if event_residual > 1.1 * min(base_residual, weather_residual):
            event -= 0.10
            base += 0.05
            weather += 0.05
            reasons.append("event_expert_weak")

        base = _clamp(base, 0.05, 0.85)
        weather = _clamp(weather, 0.05, 0.85)
        event = _clamp(event, 0.0, 0.6)
        total = base + weather + event
        weights = ExpertWeights(base=base / total, weather=weather / total, event=event / total)
        risk = "low"
        if float(context.get("event_severity_7d", 0.0)) > 2.5 or float(context.get("weather_extreme_score", 0.0)) > 0.7:
            risk = "high"
        elif float(context.get("event_severity_7d", 0.0)) > 1.0:
            risk = "medium"
        return AgentDecision(
            region=region,
            issue_date=issue_date,
            expert_weights=weights,
            risk_level=risk,
            use_event_context=weights.event > 0.05,
            reason_codes=reasons,
        )


class PriorCalibratedRouter:
    def __init__(
        self,
        config: ForecastConfig,
        region_weights: dict[str, dict[str, float]],
        global_weights: dict[str, float] | None = None,
        prior_blend: float = 0.7,
    ):
        self.config = config
        self.region_weights = region_weights
        self.global_weights = global_weights or dict(config.agent.default_weights)
        self.prior_blend = float(prior_blend)
        self.rule_router = RuleBasedRouter(config)

    @staticmethod
    def save(
        path: str | Path,
        region_weights: dict[str, dict[str, float]],
        global_weights: dict[str, float],
        prior_blend: float,
        metrics: dict[str, Any],
    ) -> None:
        payload = {
            "region_weights": region_weights,
            "global_weights": global_weights,
            "prior_blend": float(prior_blend),
            "metrics": metrics,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, config: ForecastConfig) -> "PriorCalibratedRouter":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            config=config,
            region_weights=payload.get("region_weights", {}),
            global_weights=payload.get("global_weights"),
            prior_blend=float(payload.get("prior_blend", 0.7)),
        )

    def _prior_for_region(self, region: str) -> dict[str, float]:
        return self.region_weights.get(region, self.global_weights)

    def route(self, region: str, issue_date: str, context: dict[str, float | int | str | bool]) -> AgentDecision:
        prior = self._prior_for_region(region)
        self.rule_router.defaults = {
            "base": float(prior.get("base", self.global_weights["base"])),
            "weather": float(prior.get("weather", self.global_weights["weather"])),
            "event": float(prior.get("event", self.global_weights["event"])),
        }
        adjusted = self.rule_router.route(region, issue_date, context)
        blended = _normalize_weight_vector(
            [
                self.prior_blend * float(prior.get("base", 0.0)) + (1.0 - self.prior_blend) * adjusted.expert_weights.base,
                self.prior_blend * float(prior.get("weather", 0.0)) + (1.0 - self.prior_blend) * adjusted.expert_weights.weather,
                self.prior_blend * float(prior.get("event", 0.0)) + (1.0 - self.prior_blend) * adjusted.expert_weights.event,
            ],
            default=[0.4, 0.4, 0.2],
        )
        return AgentDecision(
            region=region,
            issue_date=issue_date,
            expert_weights=ExpertWeights(
                base=float(blended[0]),
                weather=float(blended[1]),
                event=float(blended[2]),
            ),
            risk_level=adjusted.risk_level,
            use_event_context=float(blended[2]) > 0.05,
            reason_codes=adjusted.reason_codes + ["region_prior_blend"],
        )


class LearnedRouter:
    def __init__(self, config: ForecastConfig, model: Any, blend_alpha: float, regions: list[str]):
        self.config = config
        self.model = model
        self.blend_alpha = float(blend_alpha)
        self.regions = regions
        self.fallback_router = RuleBasedRouter(config)

    @staticmethod
    def save(path: str | Path, model: Any, blend_alpha: float, regions: list[str], metrics: dict[str, Any]) -> None:
        payload = {
            "model": model,
            "blend_alpha": float(blend_alpha),
            "regions": regions,
            "metrics": metrics,
        }
        with Path(path).open("wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: str | Path, config: ForecastConfig) -> "LearnedRouter":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        return cls(
            config=config,
            model=payload["model"],
            blend_alpha=float(payload.get("blend_alpha", 0.5)),
            regions=list(payload.get("regions", config.data.regions)),
        )

    def route(self, region: str, issue_date: str, context: dict[str, float | int | str | bool]) -> AgentDecision:
        fallback = self.fallback_router.route(region, issue_date, context)
        try:
            feature = np.asarray(context_to_feature_vector(region, issue_date, context, self.regions), dtype=np.float32).reshape(1, -1)
            learned_raw = self.model.predict(feature)[0]
            learned_weights = _normalize_weight_vector(learned_raw, default=[0.4, 0.4, 0.2])
            fallback_weights = np.asarray(
                [
                    fallback.expert_weights.base,
                    fallback.expert_weights.weather,
                    fallback.expert_weights.event,
                ],
                dtype=np.float32,
            )
            mixed_weights = _normalize_weight_vector(
                (1.0 - self.blend_alpha) * fallback_weights + self.blend_alpha * learned_weights,
                default=[0.4, 0.4, 0.2],
            )
            weights = ExpertWeights(
                base=float(mixed_weights[0]),
                weather=float(mixed_weights[1]),
                event=float(mixed_weights[2]),
            )
            return AgentDecision(
                region=region,
                issue_date=issue_date,
                expert_weights=weights,
                risk_level=fallback.risk_level,
                use_event_context=weights.event > 0.05,
                reason_codes=fallback.reason_codes + ["learned_router_blend"],
            )
        except Exception:
            return fallback


def build_agent_context(
    region: str,
    issue_date: str,
    weather_forecast: dict[str, float],
    event_features: dict[str, float],
    residuals: dict[str, float],
) -> dict[str, float | str]:
    return {
        "region": region,
        "issue_date": issue_date,
        "forecast_temp_max": float(weather_forecast.get("forecast_Max Temp (K)", 0.0)),
        "forecast_temp_min": float(weather_forecast.get("forecast_Min Temp (K)", 0.0)),
        "forecast_wind": float(weather_forecast.get("forecast_Max Wind Speed (m/s)", 0.0)),
        "weather_extreme_score": float(weather_forecast.get("extreme_weather_score", 0.0)),
        "event_severity_7d": float(event_features.get("severity_sum_7d", 0.0)),
        "outage_score_7d": float(event_features.get("outage_score_7d", 0.0)),
        "market_stress_7d": float(event_features.get("market_stress_score_7d", 0.0)),
        "base_residual_14d": float(residuals.get("base_residual_14d", 1.0)),
        "weather_residual_14d": float(residuals.get("weather_residual_14d", 1.0)),
        "event_residual_14d": float(residuals.get("event_residual_14d", 1.0)),
    }


class OllamaRouter:
    def __init__(self, config: ForecastConfig, fallback_router: Any | None = None):
        self.config = config
        self.client = OllamaClient(config.news.ollama_url, timeout_seconds=config.agent.timeout_seconds)
        self.rule_router = fallback_router or RuleBasedRouter(config)

    def route(self, region: str, issue_date: str, context: dict[str, float | int | str | bool]) -> AgentDecision:
        rule_decision = self.rule_router.route(region, issue_date, context)
        prompt = f"""
You are routing expert forecasters for Australian day-ahead power demand.
Only choose weights inside the same decision space as the default rule.
Return strict JSON with keys:
region, issue_date, expert_weights={{base,weather,event}}, risk_level, use_event_context, reason_codes.
Expert weights must be non-negative and sum to 1.

Context:
{json.dumps(context, ensure_ascii=False, indent=2)}

Default safe decision:
{rule_decision.model_dump_json(indent=2)}
"""
        try:
            payload = self.client.generate_json(
                model=self.config.news.routing_model,
                prompt=prompt,
                temperature=self.config.agent.temperature,
            )
            return AgentDecision(**payload)
        except Exception:
            return rule_decision

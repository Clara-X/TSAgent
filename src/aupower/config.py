from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectPaths:
    data_dir: str = "data"
    artifact_dir: str = "artifacts"
    processed_dir: str = "artifacts/processed"
    model_dir: str = "artifacts/models"
    report_dir: str = "artifacts/reports"
    log_dir: str = "artifacts/logs"


@dataclass
class SplitConfig:
    pretrain_start: str = "2012-01-01"
    pretrain_end: str = "2018-12-31"
    multimodal_start: str = "2019-01-01"
    train_end: str = "2021-06-30"
    val_end: str = "2021-12-31"
    test_end: str = "2022-05-15"


@dataclass
class DataConfig:
    load_path: str = "data/AULF-2019-2023.csv"
    weather_path: str = "data/combined_weather_data.csv"
    news_path: str = "data/news_processed_data_2019-2022.json"
    regions: list[str] = field(default_factory=lambda: ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"])
    state_to_region: dict[str, str] = field(
        default_factory=lambda: {"NSW": "NSW1", "QLD": "QLD1", "SA": "SA1", "TAS": "TAS1", "VIC": "VIC1"}
    )
    stable_weather_columns: list[str] = field(
        default_factory=lambda: [
            "Min Temp (K)",
            "Max Temp (K)",
            "Max Wind Speed (m/s)",
            "Afternoon Humidity",
            "Afternoon Pressure",
            "Morning Temp",
            "Afternoon Temp",
        ]
    )


@dataclass
class WeatherConfig:
    lags: list[int] = field(default_factory=lambda: [1, 2, 3, 7, 14, 30])
    backend: str = "auto"
    minimum_history_days: int = 30


@dataclass
class NewsConfig:
    keyword_terms: list[str] = field(
        default_factory=lambda: [
            "electricity",
            "blackout",
            "outage",
            "aemo",
            "renewable",
            "solar farm",
            "wind farm",
            "coal plant",
            "transmission",
            "interconnector",
            "wholesale price",
            "heatwave",
            "power bill",
            "energy market",
            "demand response",
        ]
    )
    extraction_model: str = "qwen3.5:9b"
    routing_model: str = "qwen2.5:7b"
    ollama_url: str = "http://127.0.0.1:11434"
    max_articles_per_day_region: int = 20
    max_body_chars: int = 5000


@dataclass
class ExpertConfig:
    horizon: int = 48
    base_lookback_days: int = 14
    weather_lookback_days: int = 28
    event_lookback_days: int = 56
    batch_size: int = 64
    max_epochs: int = 30
    patience: int = 5
    lr: float = 1e-3
    hidden_dim: int = 384
    dropout: float = 0.1
    backend: str = "auto"


@dataclass
class AgentConfig:
    temperature: int = 0
    default_weights: dict[str, float] = field(default_factory=lambda: {"base": 0.4, "weather": 0.4, "event": 0.2})
    timeout_seconds: int = 60


@dataclass
class ForecastConfig:
    project: ProjectPaths = field(default_factory=ProjectPaths)
    splits: SplitConfig = field(default_factory=SplitConfig)
    data: DataConfig = field(default_factory=DataConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    experts: ExpertConfig = field(default_factory=ExpertConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _update_dataclass(instance: Any, updates: dict[str, Any]) -> Any:
    for key, value in updates.items():
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path | None = None) -> ForecastConfig:
    config = ForecastConfig()
    if path is None:
        return config
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return _update_dataclass(config, payload)

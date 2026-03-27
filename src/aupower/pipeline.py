from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from aupower.agent.ollama_router import (
    LearnedRouter,
    OllamaRouter,
    PriorCalibratedRouter,
    RuleBasedRouter,
    build_agent_context,
    context_to_feature_vector,
    fit_learned_router_model,
    oracle_weight_target,
)
from aupower.calendar_utils import holiday_features
from aupower.config import ForecastConfig, load_config
from aupower.contracts import ForecastOutput
from aupower.data.dataset import SampleBundle, build_samples
from aupower.data.load import prepare_load_artifacts
from aupower.data.news import aggregate_event_features, read_event_records, write_event_records
from aupower.data.weather import clean_weather_frame, train_weather_proxy_models
from aupower.metrics import summarise_metrics, wmape
from aupower.models.baselines import LagBoostingBaseline
from aupower.models.experts import WindowExpert
from aupower.utils import ensure_dir, get_logger, save_json

LOGGER = get_logger("aupower.pipeline")


@dataclass
class ArtifactPaths:
    processed_dir: Path
    model_dir: Path
    report_dir: Path
    log_dir: Path
    load_wide_csv: Path
    load_long_csv: Path
    weather_clean_csv: Path
    weather_proxy_csv: Path
    weather_model_pkl: Path
    news_events_jsonl: Path
    daily_events_csv: Path
    lag_baseline_pkl: Path
    base_expert_pt: Path
    weather_expert_pt: Path
    event_expert_pt: Path
    region_priors_json: Path
    learned_router_pkl: Path
    learned_router_metrics_json: Path
    backtest_json: Path
    predictions_csv: Path
    ablation_csv: Path
    routing_summary_csv: Path
    monthly_breakdown_csv: Path
    holiday_breakdown_csv: Path
    peak_breakdown_csv: Path
    report_md: Path
    figures_dir: Path
    monthly_chart_png: Path
    holiday_chart_png: Path
    peak_chart_png: Path
    residuals_json: Path


def resolve_artifacts(config: ForecastConfig) -> ArtifactPaths:
    processed_dir = ensure_dir(config.project.processed_dir)
    model_dir = ensure_dir(config.project.model_dir)
    report_dir = ensure_dir(config.project.report_dir)
    log_dir = ensure_dir(config.project.log_dir)
    figures_dir = ensure_dir(report_dir / "figures")
    return ArtifactPaths(
        processed_dir=processed_dir,
        model_dir=model_dir,
        report_dir=report_dir,
        log_dir=log_dir,
        load_wide_csv=processed_dir / "load_30m_wide.csv",
        load_long_csv=processed_dir / "load_30m_long.csv",
        weather_clean_csv=processed_dir / "weather_clean.csv",
        weather_proxy_csv=processed_dir / "weather_proxy_forecasts.csv",
        weather_model_pkl=model_dir / "weather_proxy_model.pkl",
        news_events_jsonl=processed_dir / "news_event_records.jsonl",
        daily_events_csv=processed_dir / "daily_event_features.csv",
        lag_baseline_pkl=model_dir / "lag_baseline.pkl",
        base_expert_pt=model_dir / "base_expert.pt",
        weather_expert_pt=model_dir / "weather_expert.pt",
        event_expert_pt=model_dir / "event_expert.pt",
        region_priors_json=model_dir / "region_weight_priors.json",
        learned_router_pkl=model_dir / "learned_router.pkl",
        learned_router_metrics_json=model_dir / "learned_router_metrics.json",
        backtest_json=report_dir / "backtest_summary.json",
        predictions_csv=report_dir / "backtest_predictions.csv",
        ablation_csv=report_dir / "ablation_table.csv",
        routing_summary_csv=report_dir / "routing_summary.csv",
        monthly_breakdown_csv=report_dir / "monthly_breakdown.csv",
        holiday_breakdown_csv=report_dir / "holiday_breakdown.csv",
        peak_breakdown_csv=report_dir / "peak_breakdown.csv",
        report_md=report_dir / "experiment_report.md",
        figures_dir=figures_dir,
        monthly_chart_png=figures_dir / "monthly_performance.png",
        holiday_chart_png=figures_dir / "holiday_performance.png",
        peak_chart_png=figures_dir / "peak_day_performance.png",
        residuals_json=model_dir / "validation_residuals.json",
    )


def prepare_data(config_path: str | None = None) -> ArtifactPaths:
    config = load_config(config_path)
    paths = resolve_artifacts(config)
    load_artifacts = prepare_load_artifacts(config)
    load_artifacts.wide_frame.reset_index().to_csv(paths.load_wide_csv, index=False)
    load_artifacts.long_frame.to_csv(paths.load_long_csv, index=False)
    LOGGER.info("prepared load data: %s", paths.load_wide_csv)

    weather_clean = clean_weather_frame(config)
    weather_artifacts = train_weather_proxy_models(weather_clean, config, paths.weather_model_pkl)
    weather_artifacts.clean_frame.to_csv(paths.weather_clean_csv, index=False)
    weather_artifacts.proxy_forecasts.to_csv(paths.weather_proxy_csv, index=False)
    LOGGER.info("prepared weather data: %s", paths.weather_proxy_csv)
    return paths


def extract_events(config_path: str | None = None, limit: int | None = None, use_ollama: bool = True) -> ArtifactPaths:
    config = load_config(config_path)
    paths = resolve_artifacts(config)
    count = write_event_records(config, paths.news_events_jsonl, limit=limit, use_ollama=use_ollama)
    records = read_event_records(paths.news_events_jsonl)
    aggregate_event_features(records).to_csv(paths.daily_events_csv, index=False)
    LOGGER.info("extracted %s candidate energy events", count)
    return paths


def _load_processed_frames(paths: ArtifactPaths):
    load_wide = pd.read_csv(paths.load_wide_csv, parse_dates=["ds"]).set_index("ds").sort_index()
    weather_proxy = pd.read_csv(paths.weather_proxy_csv, parse_dates=["date"])
    event_frame = pd.read_csv(paths.daily_events_csv, parse_dates=["date"]) if paths.daily_events_csv.exists() else pd.DataFrame()
    return load_wide, weather_proxy, event_frame


def _split_sample_periods(config: ForecastConfig) -> dict[str, tuple[str, str]]:
    return {
        "pretrain": (config.splits.pretrain_start, config.splits.pretrain_end),
        "train": (config.splits.multimodal_start, config.splits.train_end),
        "val": ("2021-07-01", config.splits.val_end),
        "test": ("2022-01-01", config.splits.test_end),
    }


def train_models(config_path: str | None = None) -> ArtifactPaths:
    config = load_config(config_path)
    paths = resolve_artifacts(config)
    load_wide, weather_proxy, event_frame = _load_processed_frames(paths)
    periods = _split_sample_periods(config)

    base_pretrain_train = build_samples(
        load_wide=load_wide,
        config=config,
        start=periods["pretrain"][0],
        end="2018-09-30",
        lookback_days=config.experts.base_lookback_days,
    )
    base_pretrain_valid = build_samples(
        load_wide=load_wide,
        config=config,
        start="2018-10-01",
        end=periods["pretrain"][1],
        lookback_days=config.experts.base_lookback_days,
    )
    base_expert = WindowExpert("BaseExpert", config).fit(base_pretrain_train, base_pretrain_valid)
    base_expert.save(paths.base_expert_pt)

    lag_baseline = LagBoostingBaseline().fit(base_pretrain_train)
    lag_baseline.save(paths.lag_baseline_pkl)

    base_train = build_samples(load_wide, config, periods["train"][0], periods["train"][1], config.experts.base_lookback_days)
    base_val = build_samples(load_wide, config, periods["val"][0], periods["val"][1], config.experts.base_lookback_days)
    base_expert = WindowExpert("BaseExpert", config).fit(base_train, base_val, init_from=base_expert)
    base_expert.save(paths.base_expert_pt)

    weather_train = build_samples(
        load_wide,
        config,
        periods["train"][0],
        periods["train"][1],
        config.experts.weather_lookback_days,
        weather_frame=weather_proxy,
    )
    weather_val = build_samples(
        load_wide,
        config,
        periods["val"][0],
        periods["val"][1],
        config.experts.weather_lookback_days,
        weather_frame=weather_proxy,
    )
    weather_expert = WindowExpert("WeatherExpert", config).fit(weather_train, weather_val, init_from=base_expert)
    weather_expert.save(paths.weather_expert_pt)

    event_train = build_samples(
        load_wide,
        config,
        periods["train"][0],
        periods["train"][1],
        config.experts.event_lookback_days,
        weather_frame=weather_proxy,
        event_frame=event_frame,
    )
    event_val = build_samples(
        load_wide,
        config,
        periods["val"][0],
        periods["val"][1],
        config.experts.event_lookback_days,
        weather_frame=weather_proxy,
        event_frame=event_frame,
    )
    event_expert = WindowExpert("EventExpert", config).fit(event_train, event_val, init_from=base_expert)
    event_expert.save(paths.event_expert_pt)

    residuals = _validation_residuals(base_expert, weather_expert, event_expert, config, load_wide, weather_proxy, event_frame)
    save_json(paths.residuals_json, residuals)
    _train_region_weight_priors(
        base_expert=base_expert,
        weather_expert=weather_expert,
        event_expert=event_expert,
        config=config,
        load_wide=load_wide,
        weather_proxy=weather_proxy,
        event_frame=event_frame,
        output_path=paths.region_priors_json,
    )
    learned_metrics = _train_learned_router(
        base_expert=base_expert,
        weather_expert=weather_expert,
        event_expert=event_expert,
        config=config,
        load_wide=load_wide,
        weather_proxy=weather_proxy,
        event_frame=event_frame,
        residuals=residuals,
        output_path=paths.learned_router_pkl,
    )
    save_json(paths.learned_router_metrics_json, learned_metrics)
    return paths


def _validation_residuals(base_expert, weather_expert, event_expert, config, load_wide, weather_proxy, event_frame) -> dict[str, dict[str, float]]:
    val_base = build_samples(load_wide, config, "2021-07-01", config.splits.val_end, config.experts.base_lookback_days)
    val_weather = build_samples(load_wide, config, "2021-07-01", config.splits.val_end, config.experts.weather_lookback_days, weather_frame=weather_proxy)
    val_event = build_samples(
        load_wide,
        config,
        "2021-07-01",
        config.splits.val_end,
        config.experts.event_lookback_days,
        weather_frame=weather_proxy,
        event_frame=event_frame,
    )
    preds = {
        "base": base_expert.predict(val_base),
        "weather": weather_expert.predict(val_weather),
        "event": event_expert.predict(val_event),
    }
    actual = val_base.target
    payload: dict[str, dict[str, float]] = {}
    for region in config.data.regions:
        region_mask = val_base.region == region
        payload[region] = {
            "base_residual_14d": wmape(actual[region_mask], preds["base"][region_mask]),
            "weather_residual_14d": wmape(actual[region_mask], preds["weather"][region_mask]),
            "event_residual_14d": wmape(actual[region_mask], preds["event"][region_mask]),
        }
    return payload


def _train_learned_router(
    base_expert: WindowExpert,
    weather_expert: WindowExpert,
    event_expert: WindowExpert,
    config: ForecastConfig,
    load_wide: pd.DataFrame,
    weather_proxy: pd.DataFrame,
    event_frame: pd.DataFrame,
    residuals: dict[str, dict[str, float]],
    output_path: Path,
) -> dict[str, float | int]:
    val_base = build_samples(load_wide, config, "2021-07-01", config.splits.val_end, config.experts.base_lookback_days)
    val_weather = build_samples(
        load_wide,
        config,
        "2021-07-01",
        config.splits.val_end,
        config.experts.weather_lookback_days,
        weather_frame=weather_proxy,
    )
    val_event = build_samples(
        load_wide,
        config,
        "2021-07-01",
        config.splits.val_end,
        config.experts.event_lookback_days,
        weather_frame=weather_proxy,
        event_frame=event_frame,
    )
    base_preds = base_expert.predict(val_base)
    weather_preds = weather_expert.predict(val_weather)
    event_preds = event_expert.predict(val_event)
    actual = val_base.target

    weather_lookup = _sample_to_weather_lookup(weather_proxy)
    event_lookup = _sample_to_event_lookup(event_frame)
    rule_router = RuleBasedRouter(config)

    feature_rows = []
    oracle_weights = []
    rule_weights = []
    for idx, (region, forecast_date) in enumerate(zip(val_base.region, val_base.forecast_date)):
        forecast_date_str = str(forecast_date)
        issue_date = (pd.Timestamp(forecast_date_str) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        context = build_agent_context(
            region=region,
            issue_date=issue_date,
            weather_forecast=weather_lookup.get((forecast_date_str, region), {}),
            event_features=event_lookup.get((issue_date, region), {}),
            residuals=residuals.get(region, {}),
        )
        feature_rows.append(context_to_feature_vector(region, issue_date, context, config.data.regions))
        rule_decision = rule_router.route(region, issue_date, context)
        rule_weights.append(
            [
                rule_decision.expert_weights.base,
                rule_decision.expert_weights.weather,
                rule_decision.expert_weights.event,
            ]
        )
        oracle_weights.append(
            oracle_weight_target(
                base_pred=base_preds[idx],
                weather_pred=weather_preds[idx],
                event_pred=event_preds[idx],
                actual=actual[idx],
                step=0.05,
            )
        )

    sort_idx = np.argsort(np.asarray(val_base.forecast_date, dtype=str))
    fit_payload = fit_learned_router_model(
        features=np.asarray(feature_rows, dtype=np.float32)[sort_idx],
        oracle_weights=np.asarray(oracle_weights, dtype=np.float32)[sort_idx],
        rule_weights=np.asarray(rule_weights, dtype=np.float32)[sort_idx],
        base_preds=base_preds[sort_idx],
        weather_preds=weather_preds[sort_idx],
        event_preds=event_preds[sort_idx],
        actual=actual[sort_idx],
    )
    LearnedRouter.save(
        path=output_path,
        model=fit_payload["model"],
        blend_alpha=fit_payload["blend_alpha"],
        regions=config.data.regions,
        metrics=fit_payload["metrics"],
    )
    metrics = fit_payload["metrics"]
    LOGGER.info(
        "trained learned router: alpha=%.2f val_rule=%.6f val_blend=%.6f",
        metrics["blend_alpha"],
        metrics["val_wmape_rule_router"],
        metrics["val_wmape_best_blend"],
    )
    return metrics


def _search_best_weights(
    actual: np.ndarray,
    base_preds: np.ndarray,
    weather_preds: np.ndarray,
    event_preds: np.ndarray,
    step: float = 0.01,
) -> tuple[tuple[float, float, float], float]:
    best_weights = (0.4, 0.4, 0.2)
    best_score = float("inf")
    for wb in np.arange(0.0, 1.0 + 1e-8, step):
        for ww in np.arange(0.0, 1.0 - wb + 1e-8, step):
            we = float(1.0 - wb - ww)
            if we < -1e-8:
                continue
            preds = wb * base_preds + ww * weather_preds + we * event_preds
            score = wmape(actual, preds)
            if score < best_score:
                best_score = score
                best_weights = (float(wb), float(ww), float(we))
    return best_weights, best_score


def _train_region_weight_priors(
    base_expert: WindowExpert,
    weather_expert: WindowExpert,
    event_expert: WindowExpert,
    config: ForecastConfig,
    load_wide: pd.DataFrame,
    weather_proxy: pd.DataFrame,
    event_frame: pd.DataFrame,
    output_path: Path,
) -> None:
    val_base = build_samples(load_wide, config, "2021-07-01", config.splits.val_end, config.experts.base_lookback_days)
    val_weather = build_samples(
        load_wide,
        config,
        "2021-07-01",
        config.splits.val_end,
        config.experts.weather_lookback_days,
        weather_frame=weather_proxy,
    )
    val_event = build_samples(
        load_wide,
        config,
        "2021-07-01",
        config.splits.val_end,
        config.experts.event_lookback_days,
        weather_frame=weather_proxy,
        event_frame=event_frame,
    )
    actual = val_base.target
    base_preds = base_expert.predict(val_base)
    weather_preds = weather_expert.predict(val_weather)
    event_preds = event_expert.predict(val_event)

    global_weights, global_score = _search_best_weights(actual, base_preds, weather_preds, event_preds, step=0.01)
    region_weights: dict[str, dict[str, float]] = {}
    region_scores: dict[str, float] = {}
    for region in config.data.regions:
        mask = val_base.region == region
        weights, score = _search_best_weights(
            actual[mask],
            base_preds[mask],
            weather_preds[mask],
            event_preds[mask],
            step=0.01,
        )
        region_weights[region] = {
            "base": weights[0],
            "weather": weights[1],
            "event": weights[2],
        }
        region_scores[region] = score

    PriorCalibratedRouter.save(
        path=output_path,
        region_weights=region_weights,
        global_weights={"base": global_weights[0], "weather": global_weights[1], "event": global_weights[2]},
        prior_blend=0.7,
        metrics={
            "global_val_wmape": global_score,
            "region_val_wmape": region_scores,
        },
    )


def _sample_to_event_lookup(event_frame: pd.DataFrame | None) -> dict[tuple[str, str], dict[str, float]]:
    if event_frame is None or event_frame.empty:
        return {}
    rows = {}
    for _, row in event_frame.iterrows():
        date_key = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
        rows[(date_key, row["region"])] = {
            key: float(row[key]) for key in row.index if key not in {"date", "region"}
        }
    return rows


def _sample_to_weather_lookup(weather_frame: pd.DataFrame | None) -> dict[tuple[str, str], dict[str, float]]:
    if weather_frame is None or weather_frame.empty:
        return {}
    rows = {}
    for _, row in weather_frame.iterrows():
        date_key = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
        record = {key: float(row[key]) for key in row.index if key not in {"date", "region"}}
        record["extreme_weather_score"] = float(max(record.get("forecast_Max Temp (K)", 0.0) - 305.0, 0.0) / 10.0)
        rows[(date_key, row["region"])] = record
    return rows


def backtest(config_path: str | None = None, use_ollama: bool = False) -> ArtifactPaths:
    config = load_config(config_path)
    paths = resolve_artifacts(config)
    load_wide, weather_proxy, event_frame = _load_processed_frames(paths)

    base_expert = WindowExpert.load(paths.base_expert_pt, config)
    weather_expert = WindowExpert.load(paths.weather_expert_pt, config)
    event_expert = WindowExpert.load(paths.event_expert_pt, config)
    lag_baseline = LagBoostingBaseline.load(paths.lag_baseline_pkl)
    residuals = json.loads(paths.residuals_json.read_text(encoding="utf-8")) if paths.residuals_json.exists() else {}

    base_samples = build_samples(load_wide, config, "2022-01-01", config.splits.test_end, config.experts.base_lookback_days)
    weather_samples = build_samples(load_wide, config, "2022-01-01", config.splits.test_end, config.experts.weather_lookback_days, weather_frame=weather_proxy)
    event_samples = build_samples(
        load_wide,
        config,
        "2022-01-01",
        config.splits.test_end,
        config.experts.event_lookback_days,
        weather_frame=weather_proxy,
        event_frame=event_frame,
    )

    base_preds = base_expert.predict(base_samples)
    weather_preds = weather_expert.predict(weather_samples)
    event_preds = event_expert.predict(event_samples)
    lag_preds = lag_baseline.predict(base_samples)
    actual = base_samples.target

    weather_lookup = _sample_to_weather_lookup(weather_proxy)
    event_lookup = _sample_to_event_lookup(event_frame)
    if paths.region_priors_json.exists():
        fallback_router = PriorCalibratedRouter.load(paths.region_priors_json, config)
    elif paths.learned_router_pkl.exists():
        fallback_router = LearnedRouter.load(paths.learned_router_pkl, config)
    else:
        fallback_router = RuleBasedRouter(config)
    router = OllamaRouter(config, fallback_router=fallback_router) if use_ollama else fallback_router

    fixed_preds = 0.4 * base_preds + 0.4 * weather_preds + 0.2 * event_preds
    bw_preds = 0.5 * base_preds + 0.5 * weather_preds

    routed_preds = []
    routed_rows = []
    for idx, (region, forecast_date) in enumerate(zip(base_samples.region, base_samples.forecast_date)):
        forecast_date_str = str(forecast_date)
        weather_context = weather_lookup.get((forecast_date_str, region), {})
        issue_date = (pd.Timestamp(forecast_date_str) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        event_context = event_lookup.get((issue_date, region), {})
        context = build_agent_context(
            region=region,
            issue_date=issue_date,
            weather_forecast=weather_context,
            event_features=event_context,
            residuals=residuals.get(region, {}),
        )
        decision = router.route(region, context["issue_date"], context)
        combined = (
            decision.expert_weights.base * base_preds[idx]
            + decision.expert_weights.weather * weather_preds[idx]
            + decision.expert_weights.event * event_preds[idx]
        )
        routed_preds.append(combined)
        routed_rows.append(
            {
                "region": region,
                "forecast_date": forecast_date_str,
                "risk_level": decision.risk_level,
                "base_weight": decision.expert_weights.base,
                "weather_weight": decision.expert_weights.weather,
                "event_weight": decision.expert_weights.event,
                "issue_date": issue_date,
                "event_day": int(float(event_context.get("severity_sum_7d", 0.0)) > 0.0),
                "actual_sum": float(np.sum(actual[idx])),
                "fixed_abs_error_sum": float(np.sum(np.abs(actual[idx] - fixed_preds[idx]))),
                "agent_abs_error_sum": float(np.sum(np.abs(actual[idx] - combined))),
                "fixed_rmse_day": float(np.sqrt(np.mean(np.square(actual[idx] - fixed_preds[idx])))),
                "agent_rmse_day": float(np.sqrt(np.mean(np.square(actual[idx] - combined)))),
            }
        )
    routed_preds = np.asarray(routed_preds, dtype=np.float32)

    reports = {
        "lag_baseline": summarise_metrics(actual, lag_preds),
        "base_only": summarise_metrics(actual, base_preds),
        "base_weather": summarise_metrics(actual, bw_preds),
        "fixed_multimodal": summarise_metrics(actual, fixed_preds),
        "agent_routed": summarise_metrics(actual, routed_preds),
    }

    peak_threshold = np.quantile(actual.reshape(-1), 0.9)
    peak_mask = actual >= peak_threshold
    reports["agent_routed_peak_wmape"] = wmape(actual[peak_mask], routed_preds[peak_mask])
    reports["fixed_multimodal_peak_wmape"] = wmape(actual[peak_mask], fixed_preds[peak_mask])
    reports["by_region"] = {}
    for region in config.data.regions:
        mask = base_samples.region == region
        reports["by_region"][region] = summarise_metrics(actual[mask], routed_preds[mask])

    event_day_mask = np.asarray([row["event_day"] for row in routed_rows], dtype=bool)
    if event_day_mask.any():
        reports["agent_routed_event_day_wmape"] = wmape(actual[event_day_mask], routed_preds[event_day_mask])
        reports["fixed_multimodal_event_day_wmape"] = wmape(actual[event_day_mask], fixed_preds[event_day_mask])

    routed_frame = pd.DataFrame(routed_rows)
    routed_frame.to_csv(paths.predictions_csv, index=False)
    save_json(paths.backtest_json, reports)
    return paths


def predict(config_path: str | None, region: str, forecast_date: str, use_ollama: bool = True) -> ForecastOutput:
    config = load_config(config_path)
    paths = resolve_artifacts(config)
    load_wide, weather_proxy, event_frame = _load_processed_frames(paths)
    base_expert = WindowExpert.load(paths.base_expert_pt, config)
    weather_expert = WindowExpert.load(paths.weather_expert_pt, config)
    event_expert = WindowExpert.load(paths.event_expert_pt, config)
    residuals = json.loads(paths.residuals_json.read_text(encoding="utf-8")) if paths.residuals_json.exists() else {}

    base_samples = build_samples(load_wide, config, forecast_date, forecast_date, config.experts.base_lookback_days)
    weather_samples = build_samples(load_wide, config, forecast_date, forecast_date, config.experts.weather_lookback_days, weather_frame=weather_proxy)
    event_samples = build_samples(
        load_wide,
        config,
        forecast_date,
        forecast_date,
        config.experts.event_lookback_days,
        weather_frame=weather_proxy,
        event_frame=event_frame,
    )
    sample_index = int(np.where(base_samples.region == region)[0][0])
    base_pred = base_expert.predict(base_samples)[sample_index]
    weather_pred = weather_expert.predict(weather_samples)[sample_index]
    event_pred = event_expert.predict(event_samples)[sample_index]

    weather_lookup = _sample_to_weather_lookup(weather_proxy)
    event_lookup = _sample_to_event_lookup(event_frame)
    forecast_date_str = str(forecast_date)
    issue_date = (pd.Timestamp(forecast_date_str) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    context = build_agent_context(
        region=region,
        issue_date=issue_date,
        weather_forecast=weather_lookup.get((forecast_date_str, region), {}),
        event_features=event_lookup.get((issue_date, region), {}),
        residuals=residuals.get(region, {}),
    )
    if paths.region_priors_json.exists():
        fallback_router = PriorCalibratedRouter.load(paths.region_priors_json, config)
    elif paths.learned_router_pkl.exists():
        fallback_router = LearnedRouter.load(paths.learned_router_pkl, config)
    else:
        fallback_router = RuleBasedRouter(config)
    router = OllamaRouter(config, fallback_router=fallback_router) if use_ollama else fallback_router
    decision = router.route(region, issue_date, context)
    pred = (
        decision.expert_weights.base * base_pred
        + decision.expert_weights.weather * weather_pred
        + decision.expert_weights.event * event_pred
    )
    return ForecastOutput(
        region=region,  # type: ignore[arg-type]
        forecast_date=forecast_date,
        pred_30m=pred.tolist(),
        expert_preds={"base": base_pred.tolist(), "weather": weather_pred.tolist(), "event": event_pred.tolist()},
        agent_decision=decision,
        risk_level=decision.risk_level,
    )


def _safe_pct_improvement(reference: float, challenger: float) -> float:
    if abs(reference) <= 1e-12:
        return 0.0
    return (reference - challenger) / reference * 100.0


def _markdown_table(frame: pd.DataFrame, float_digits: int = 4) -> str:
    if frame.empty:
        return "_No data._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda x: f"{x:.{float_digits}f}")
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def _attach_calendar_categories(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["forecast_date"] = pd.to_datetime(enriched["forecast_date"])
    enriched["month"] = enriched["forecast_date"].dt.strftime("%Y-%m")

    holiday_payload = []
    for row in enriched.itertuples(index=False):
        features = holiday_features(pd.Timestamp(row.forecast_date).date(), row.region)
        if features["is_national_holiday"] or features["is_state_holiday"]:
            category = "Holiday"
        elif features["adjacent_holiday"]:
            category = "Adjacent Holiday"
        elif features["is_weekend"]:
            category = "Weekend"
        else:
            category = "Regular"
        holiday_payload.append(
            {
                "holiday_category": category,
                "is_weekend": features["is_weekend"],
                "is_national_holiday": features["is_national_holiday"],
                "is_state_holiday": features["is_state_holiday"],
                "adjacent_holiday": features["adjacent_holiday"],
            }
        )
    return pd.concat([enriched.reset_index(drop=True), pd.DataFrame(holiday_payload)], axis=1)


def _wmape_from_sums(error_sum: pd.Series, actual_sum: pd.Series) -> pd.Series:
    denominator = actual_sum.replace(0.0, np.nan)
    return (error_sum / denominator).fillna(0.0)


def _build_monthly_breakdown(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby("month", as_index=False)[["actual_sum", "fixed_abs_error_sum", "agent_abs_error_sum"]]
        .sum()
        .sort_values("month")
        .reset_index(drop=True)
    )
    grouped["fixed_wmape"] = _wmape_from_sums(grouped["fixed_abs_error_sum"], grouped["actual_sum"])
    grouped["agent_wmape"] = _wmape_from_sums(grouped["agent_abs_error_sum"], grouped["actual_sum"])
    grouped["improve_pct"] = [
        _safe_pct_improvement(float(fixed), float(agent))
        for fixed, agent in zip(grouped["fixed_wmape"], grouped["agent_wmape"])
    ]
    return grouped


def _build_holiday_breakdown(frame: pd.DataFrame) -> pd.DataFrame:
    category_order = ["Regular", "Weekend", "Adjacent Holiday", "Holiday"]
    grouped = (
        frame.groupby("holiday_category", as_index=False)[["actual_sum", "fixed_abs_error_sum", "agent_abs_error_sum"]]
        .sum()
    )
    grouped["holiday_category"] = pd.Categorical(grouped["holiday_category"], categories=category_order, ordered=True)
    grouped = grouped.sort_values("holiday_category").reset_index(drop=True)
    grouped["fixed_wmape"] = _wmape_from_sums(grouped["fixed_abs_error_sum"], grouped["actual_sum"])
    grouped["agent_wmape"] = _wmape_from_sums(grouped["agent_abs_error_sum"], grouped["actual_sum"])
    grouped["improve_pct"] = [
        _safe_pct_improvement(float(fixed), float(agent))
        for fixed, agent in zip(grouped["fixed_wmape"], grouped["agent_wmape"])
    ]
    grouped["holiday_category"] = grouped["holiday_category"].astype(str)
    return grouped


def _build_peak_breakdown(frame: pd.DataFrame, regions: list[str]) -> pd.DataFrame:
    rows = []
    for region in regions:
        region_frame = frame[frame["region"] == region].copy()
        if region_frame.empty:
            continue
        threshold = float(region_frame["actual_sum"].quantile(0.9))
        peak_frame = region_frame[region_frame["actual_sum"] >= threshold].copy()
        if peak_frame.empty:
            continue
        actual_sum = float(peak_frame["actual_sum"].sum())
        fixed_error = float(peak_frame["fixed_abs_error_sum"].sum())
        agent_error = float(peak_frame["agent_abs_error_sum"].sum())
        fixed_wmape = fixed_error / actual_sum if actual_sum else 0.0
        agent_wmape = agent_error / actual_sum if actual_sum else 0.0
        rows.append(
            {
                "region": region,
                "n_peak_days": int(len(peak_frame)),
                "fixed_wmape": fixed_wmape,
                "agent_wmape": agent_wmape,
                "improve_pct": _safe_pct_improvement(fixed_wmape, agent_wmape),
            }
        )
    return pd.DataFrame(rows)


def _save_monthly_chart(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(frame))
    ax.plot(x, frame["fixed_wmape"], marker="o", linewidth=2.2, label="Fixed Multimodal")
    ax.plot(x, frame["agent_wmape"], marker="o", linewidth=2.2, label="Agent Routed")
    ax.set_xticks(x)
    ax.set_xticklabels(frame["month"], rotation=45, ha="right")
    ax.set_ylabel("WMAPE")
    ax.set_title("Monthly WMAPE on Test Window")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_holiday_chart(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(frame))
    width = 0.34
    ax.bar(x - width / 2, frame["fixed_wmape"], width=width, label="Fixed Multimodal")
    ax.bar(x + width / 2, frame["agent_wmape"], width=width, label="Agent Routed")
    ax.set_xticks(x)
    ax.set_xticklabels(frame["holiday_category"], rotation=20, ha="right")
    ax.set_ylabel("WMAPE")
    ax.set_title("Holiday Category WMAPE")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_peak_chart(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(frame))
    width = 0.34
    ax.bar(x - width / 2, frame["fixed_wmape"], width=width, label="Fixed Multimodal")
    ax.bar(x + width / 2, frame["agent_wmape"], width=width, label="Agent Routed")
    ax.set_xticks(x)
    ax.set_xticklabels(frame["region"])
    ax.set_ylabel("WMAPE")
    ax.set_title("Top 10% Peak-Day WMAPE by Region")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_report(config_path: str | None = None) -> ArtifactPaths:
    config = load_config(config_path)
    paths = resolve_artifacts(config)
    summary = json.loads(paths.backtest_json.read_text(encoding="utf-8"))
    predictions = pd.read_csv(paths.predictions_csv)
    predictions = _attach_calendar_categories(predictions)
    prior_payload = (
        json.loads(paths.region_priors_json.read_text(encoding="utf-8")) if paths.region_priors_json.exists() else {}
    )
    learned_metrics = (
        json.loads(paths.learned_router_metrics_json.read_text(encoding="utf-8"))
        if paths.learned_router_metrics_json.exists()
        else {}
    )

    ablation_rows = []
    fixed_wmape = float(summary["fixed_multimodal"]["wmape"])
    ordered_models = ["lag_baseline", "base_only", "base_weather", "fixed_multimodal", "agent_routed"]
    labels = {
        "lag_baseline": "LagBoostingBaseline",
        "base_only": "BaseExpert only",
        "base_weather": "Base + Weather",
        "fixed_multimodal": "Base + Weather + Event (fixed weights)",
        "agent_routed": "Agent routed",
    }
    for model_key in ordered_models:
        metrics = summary[model_key]
        ablation_rows.append(
            {
                "model": labels[model_key],
                "wmape": float(metrics["wmape"]),
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "smape": float(metrics["smape"]),
                "improve_vs_fixed_pct": _safe_pct_improvement(fixed_wmape, float(metrics["wmape"])),
            }
        )
    ablation_frame = pd.DataFrame(ablation_rows)
    ablation_frame.to_csv(paths.ablation_csv, index=False)

    routing_summary = (
        predictions.groupby("region", as_index=False)[["base_weight", "weather_weight", "event_weight"]]
        .mean()
        .sort_values("region")
        .reset_index(drop=True)
    )
    routing_summary.columns = ["region", "avg_base_weight", "avg_weather_weight", "avg_event_weight"]
    routing_summary.to_csv(paths.routing_summary_csv, index=False)

    monthly_breakdown = _build_monthly_breakdown(predictions)
    holiday_breakdown = _build_holiday_breakdown(predictions)
    peak_breakdown = _build_peak_breakdown(predictions, config.data.regions)
    monthly_breakdown.to_csv(paths.monthly_breakdown_csv, index=False)
    holiday_breakdown.to_csv(paths.holiday_breakdown_csv, index=False)
    peak_breakdown.to_csv(paths.peak_breakdown_csv, index=False)

    _save_monthly_chart(monthly_breakdown, paths.monthly_chart_png)
    _save_holiday_chart(holiday_breakdown, paths.holiday_chart_png)
    _save_peak_chart(peak_breakdown, paths.peak_chart_png)

    region_frame = pd.DataFrame(
        [
            {
                "region": region,
                "wmape": float(metrics["wmape"]),
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "smape": float(metrics["smape"]),
            }
            for region, metrics in summary["by_region"].items()
        ]
    ).sort_values("region")
    prior_rows = []
    for region in config.data.regions:
        prior = prior_payload.get("region_weights", {}).get(region, {})
        prior_rows.append(
            {
                "region": region,
                "prior_base": float(prior.get("base", np.nan)),
                "prior_weather": float(prior.get("weather", np.nan)),
                "prior_event": float(prior.get("event", np.nan)),
            }
        )
    prior_frame = pd.DataFrame(prior_rows).sort_values("region")

    risk_counts = predictions["risk_level"].value_counts().rename_axis("risk_level").reset_index(name="count")
    total_predictions = max(int(risk_counts["count"].sum()), 1)
    risk_counts["share_pct"] = risk_counts["count"] / total_predictions * 100.0

    overall_improve = _safe_pct_improvement(
        float(summary["fixed_multimodal"]["wmape"]),
        float(summary["agent_routed"]["wmape"]),
    )
    peak_improve = _safe_pct_improvement(
        float(summary["fixed_multimodal_peak_wmape"]),
        float(summary["agent_routed_peak_wmape"]),
    )
    event_improve = _safe_pct_improvement(
        float(summary["fixed_multimodal_event_day_wmape"]),
        float(summary["agent_routed_event_day_wmape"]),
    )
    acceptance_pass = overall_improve >= 2.0 or (
        max(peak_improve, event_improve) >= 5.0 and overall_improve >= -0.5
    )

    lines = [
        "# Australia Five-Region Day-Ahead Power Forecasting Report",
        "",
        "## Executive Summary",
        "",
        f"- Task: five-region joint modeling for `NSW1/QLD1/SA1/TAS1/VIC1`, day-ahead `24h` load forecasting at `30 min` resolution (`48` points/day/region).",
        f"- Test window: `2022-01-01` to `{config.splits.test_end}`.",
        "- Final routing policy: region-calibrated expert priors blended with rule-based context routing; Ollama remains optional and sits on top of the same safe decision space.",
        f"- Acceptance result: `{'PASS' if acceptance_pass else 'FAIL'}`.",
        f"- Overall improvement vs strongest non-Agent baseline: `{overall_improve:.2f}%` WMAPE.",
        "",
        "## Data And Protocol",
        "",
        f"- Pretrain: `{config.splits.pretrain_start}` to `{config.splits.pretrain_end}`.",
        f"- Multimodal train: `{config.splits.multimodal_start}` to `{config.splits.train_end}`.",
        f"- Validation: `2021-07-01` to `{config.splits.val_end}`.",
        f"- Test: `2022-01-01` to `{config.splits.test_end}`.",
        "- Forecast issue time follows the project protocol: use information available up to `D-1 23:30`, predict `D 00:00` to `D 23:30`.",
        "",
        "## Core Results",
        "",
        f"- `fixed_multimodal WMAPE`: `{summary['fixed_multimodal']['wmape']:.6f}`.",
        f"- `agent_routed WMAPE`: `{summary['agent_routed']['wmape']:.6f}`.",
        f"- Peak-window WMAPE improvement: `{peak_improve:.2f}%`.",
        f"- Event-day WMAPE improvement: `{event_improve:.2f}%`.",
        "",
        _markdown_table(
            ablation_frame[
                ["model", "wmape", "mae", "rmse", "smape", "improve_vs_fixed_pct"]
            ],
            float_digits=4,
        ),
        "",
        "## Visual Analysis",
        "",
        "### Monthly Performance",
        "",
        "![Monthly WMAPE](figures/monthly_performance.png)",
        "",
        _markdown_table(monthly_breakdown[["month", "fixed_wmape", "agent_wmape", "improve_pct"]], float_digits=4),
        "",
        "### Holiday And Calendar Categories",
        "",
        "![Holiday WMAPE](figures/holiday_performance.png)",
        "",
        _markdown_table(
            holiday_breakdown[["holiday_category", "fixed_wmape", "agent_wmape", "improve_pct"]],
            float_digits=4,
        ),
        "",
        "### Peak-Day Performance",
        "",
        "![Peak-Day WMAPE](figures/peak_day_performance.png)",
        "",
        _markdown_table(peak_breakdown, float_digits=4),
        "",
        "## Region Breakdown",
        "",
        _markdown_table(region_frame, float_digits=4),
        "",
        "## Routing Diagnostics",
        "",
        "### Learned Region Priors",
        "",
        _markdown_table(prior_frame, float_digits=4),
        "",
        "### Realized Average Routing Weights On Test",
        "",
        _markdown_table(routing_summary, float_digits=4),
        "",
        "### Risk Distribution",
        "",
        _markdown_table(risk_counts, float_digits=2),
        "",
        "## Acceptance Check",
        "",
        f"- Overall WMAPE improvement: `{overall_improve:.2f}%`.",
        f"- Peak subset improvement: `{peak_improve:.2f}%`.",
        f"- Event-day subset improvement: `{event_improve:.2f}%`.",
        f"- Decision: `{'PASS' if acceptance_pass else 'FAIL'}` against the project threshold (`>=2%` overall or `>=5%` on peak/event subsets with no meaningful overall regression).",
        "",
        "## Artifacts",
        "",
        f"- Summary JSON: `{paths.backtest_json}`",
        f"- Ablation CSV: `{paths.ablation_csv}`",
        f"- Routing summary CSV: `{paths.routing_summary_csv}`",
        f"- Monthly breakdown CSV: `{paths.monthly_breakdown_csv}`",
        f"- Holiday breakdown CSV: `{paths.holiday_breakdown_csv}`",
        f"- Peak breakdown CSV: `{paths.peak_breakdown_csv}`",
        f"- Region priors JSON: `{paths.region_priors_json}`",
    ]
    if learned_metrics:
        lines.extend(
            [
                "",
                "## Learned Router Diagnostics",
                "",
                f"- Learned-router validation rule WMAPE: `{float(learned_metrics.get('val_wmape_rule_router', 0.0)):.6f}`.",
                f"- Learned-router validation blended WMAPE: `{float(learned_metrics.get('val_wmape_best_blend', 0.0)):.6f}`.",
                f"- Learned-router blend alpha after safeguard: `{float(learned_metrics.get('blend_alpha', 0.0)):.2f}`.",
            ]
        )

    paths.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("generated experiment report: %s", paths.report_md)
    return paths

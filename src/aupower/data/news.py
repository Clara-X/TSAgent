from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from aupower.agent.ollama_router import OllamaClient
from aupower.config import ForecastConfig
from aupower.contracts import NewsEventRecord

JSON_DECODER = json.JSONDecoder()

REGION_HINTS = {
    "NSW1": ["nsw", "new south wales", "sydney", "newcastle", "canberra", "act "],
    "QLD1": ["qld", "queensland", "brisbane", "townsville", "gold coast", "cairns"],
    "SA1": ["sa ", "south australia", "adelaide", "port augusta"],
    "TAS1": ["tas", "tasmania", "hobart", "launceston"],
    "VIC1": ["vic", "victoria", "melbourne", "geelong"],
}

OUT_OF_SCOPE_HINTS = [
    "western australia",
    " wa ",
    "perth",
    "darwin",
    "northern territory",
    " nt ",
]

NATIONAL_HINTS = [
    "national electricity market",
    "australian energy market operator",
    "aemo",
    "east coast grid",
    "electricity market",
    "power grid",
]

STRONG_POSITIVE_TERMS = [
    "electricity",
    "blackout",
    "power outage",
    "electricity outage",
    "grid",
    "aemo",
    "energy market",
    "national electricity market",
    "renewable energy",
    "electricity bill",
    "power bill",
    "coal plant",
    "gas plant",
    "solar farm",
    "wind farm",
    "transmission line",
    "transmission network",
    "interconnector",
    "substation",
    "load shedding",
    "generator outage",
]

CONTEXT_TERMS = [
    "heatwave",
    "storm",
    "cyclone",
    "flood",
    "bushfire",
    "hot weather",
    "record heat",
    "demand",
    "prices",
    "renewable",
    "battery",
]

NEGATIVE_PATTERNS = [
    "power couple",
    "powerball",
    "most powerful",
    "passport",
    "solar system",
    "whirlwind relationship",
    "in demand suburb",
    "property market",
    "real estate",
    "sporting prodigy",
    "fuel price",
    "petrol price",
    "air raid",
    "weapon",
    "rail cars",
    "netflix",
    "beach box",
    "stone cottage",
    "jockey",
    "bank outage",
    "mobile app",
    "internet outage",
    "website outage",
    "service outage",
    "customer outage",
]

EVENT_KEYWORDS = {
    "outage": ["outage", "blackout", "power cut", "grid failure"],
    "extreme_weather": ["heatwave", "storm", "cyclone", "flood", "bushfire", "extreme weather"],
    "market_stress": ["wholesale price", "aemo", "demand response", "reserve shortfall"],
    "generation": ["coal plant", "generator", "solar farm", "wind farm", "battery", "renewable"],
    "network": ["transmission", "interconnector", "substation", "network"],
    "pricing": ["power bill", "electricity bill", "energy price"],
    "policy": ["energy policy", "emissions target", "renewable target"],
}


def iter_json_array(path: str | Path, chunk_size: int = 1 << 20) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        buffer = ""
        in_array = False
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            buffer += chunk
            while True:
                stripped = buffer.lstrip()
                offset = len(buffer) - len(stripped)
                buffer = stripped
                if not in_array:
                    if not buffer:
                        break
                    if buffer[0] == "[":
                        in_array = True
                        buffer = buffer[1:]
                        continue
                    raise ValueError("expected JSON array")
                if not buffer:
                    break
                if buffer[0] in ",\n\r\t ":
                    buffer = buffer[1:]
                    continue
                if buffer[0] == "]":
                    return
                try:
                    obj, index = JSON_DECODER.raw_decode(buffer)
                except json.JSONDecodeError:
                    if offset == 0 and len(chunk) < chunk_size:
                        raise
                    break
                yield obj
                buffer = buffer[index:]


def _article_text(article: dict, max_body_chars: int) -> str:
    full_article = article.get("full_article", "") or ""
    return " ".join(
        [
            article.get("title", "") or "",
            article.get("summary", "") or "",
            full_article[:max_body_chars],
        ]
    ).strip()


def article_matches_energy(article: dict, config: ForecastConfig) -> bool:
    title = (article.get("title") or "").lower()
    summary = (article.get("summary") or "").lower()
    body = (article.get("full_article") or "")[: config.news.max_body_chars].lower()
    headline = f"{title} {summary}".strip()
    text = f"{headline} {body}".strip()

    def has_anchor(value: str) -> bool:
        return any(anchor in value for anchor in ["electricity", "power", "grid", "energy", "transmission", "interconnector", "substation", "blackout"])

    if any(pattern in headline or pattern in text for pattern in NEGATIVE_PATTERNS):
        return False

    if any(term in headline for term in STRONG_POSITIVE_TERMS) and has_anchor(headline):
        return True
    if any(term in text for term in STRONG_POSITIVE_TERMS) and has_anchor(text):
        return True

    has_context = any(term in text for term in CONTEXT_TERMS)
    has_keyword = any(term.lower() in text for term in config.news.keyword_terms)
    has_energy_anchor = has_anchor(text)
    if "outage" in text and not has_energy_anchor:
        return False
    return bool(has_keyword and has_context and has_energy_anchor)


def candidate_energy_score(article: dict, config: ForecastConfig) -> float:
    title = (article.get("title") or "").lower()
    summary = (article.get("summary") or "").lower()
    body = (article.get("full_article") or "")[: config.news.max_body_chars].lower()
    score = 0.0
    for term in STRONG_POSITIVE_TERMS:
        if term in title:
            score += 3.0
        elif term in summary:
            score += 2.0
        elif term in body:
            score += 1.0
    for term in CONTEXT_TERMS:
        if term in title:
            score += 1.0
        elif term in summary:
            score += 0.75
        elif term in body:
            score += 0.25
    for pattern in NEGATIVE_PATTERNS:
        if pattern in title or pattern in summary:
            score -= 4.0
        elif pattern in body:
            score -= 1.5
    return score


def detect_regions(article: dict) -> list[str]:
    headline = f"{article.get('title','')} {article.get('summary','')}".lower()
    text = _article_text(article, 4000).lower()
    if any(hint in headline for hint in OUT_OF_SCOPE_HINTS):
        if not any(any(hint in headline for hint in hints) for hints in REGION_HINTS.values()):
            return []
    regions = []
    for region, hints in REGION_HINTS.items():
        if any(hint in headline for hint in hints):
            regions.append(region)
    if not regions:
        for region, hints in REGION_HINTS.items():
            if any(hint in text for hint in hints):
                regions.append(region)
    if any(hint in text for hint in OUT_OF_SCOPE_HINTS) and not regions:
        return []
    if not regions:
        if any(term in text for term in NATIONAL_HINTS):
            return ["NATIONAL"]
        return []
    return sorted(set(regions))


def infer_event_type(article: dict) -> str:
    text = _article_text(article, 4000).lower()
    for event_type, keywords in EVENT_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return event_type
    return "other"


def infer_direction(event_type: str, article: dict) -> str:
    text = _article_text(article, 2000).lower()
    if event_type in {"outage", "extreme_weather", "market_stress", "pricing"}:
        return "up"
    if any(term in text for term in ["lower demand", "milder weather", "prices fall", "generation recovers"]):
        return "down"
    return "neutral"


def infer_horizon(text: str) -> str:
    lowered = text.lower()
    if any(term in lowered for term in ["today", "this evening", "tonight", "intraday"]):
        return "intraday"
    if any(term in lowered for term in ["tomorrow", "next day"]):
        return "1d"
    if any(term in lowered for term in ["this week", "next few days", "three days"]):
        return "3d"
    if any(term in lowered for term in ["next week", "seven days"]):
        return "7d"
    return "unknown"


def summarise_article(article: dict) -> str:
    summary = article.get("summary") or ""
    if summary:
        return summary[:320]
    full_article = article.get("full_article", "") or ""
    sentences = re.split(r"(?<=[.!?])\s+", full_article.strip())
    return " ".join(sentences[:2])[:320]


def heuristic_event_record(article: dict, config: ForecastConfig | None = None) -> NewsEventRecord:
    config = config or ForecastConfig()
    article_id = hashlib.sha1(
        f"{article.get('link','')}-{article.get('publication_time','')}-{article.get('title','')}".encode("utf-8")
    ).hexdigest()
    event_type = infer_event_type(article)
    summary = summarise_article(article)
    regions = detect_regions(article)
    score = candidate_energy_score(article, config)
    severity = 0.85 if event_type in {"outage", "extreme_weather", "market_stress"} else 0.55
    confidence = min(0.9, max(0.2, 0.45 + 0.05 * score))
    is_relevant = bool(regions) and event_type != "other" and score >= 2.5
    return NewsEventRecord(
        article_id=article_id,
        publication_time=article.get("publication_time", ""),
        regions=regions or ["NATIONAL"],
        is_energy_relevant=is_relevant,
        event_type=event_type,
        impact_direction=infer_direction(event_type, article),
        impact_horizon=infer_horizon(_article_text(article, 1000)),
        severity=severity,
        confidence=confidence,
        summary_2lines=summary,
    )


def ollama_event_record(article: dict, config: ForecastConfig) -> NewsEventRecord:
    client = OllamaClient(config.news.ollama_url, timeout_seconds=config.agent.timeout_seconds)
    prompt = f"""
You are extracting structured electricity-market events from Australian news.
Return strict JSON with keys:
article_id, publication_time, regions, is_energy_relevant, event_type, impact_direction, impact_horizon, severity, confidence, summary_2lines.

Allowed event_type values:
outage, extreme_weather, market_stress, policy, generation, network, pricing, other

Allowed impact_direction values:
up, down, neutral, unknown

Allowed impact_horizon values:
intraday, 1d, 3d, 7d, longer, unknown

Allowed regions values:
NSW1, QLD1, SA1, TAS1, VIC1, NATIONAL

Article:
title: {article.get("title", "")}
summary: {article.get("summary", "")}
publication_time: {article.get("publication_time", "")}
full_article: {_article_text(article, config.news.max_body_chars)}
"""
    article_id = hashlib.sha1(
        f"{article.get('link','')}-{article.get('publication_time','')}-{article.get('title','')}".encode("utf-8")
    ).hexdigest()
    payload = client.generate_json(config.news.extraction_model, prompt, temperature=config.agent.temperature)
    payload.setdefault("article_id", article_id)
    payload.setdefault("publication_time", article.get("publication_time", ""))
    if not payload.get("regions"):
        payload["regions"] = detect_regions(article)
    score = candidate_energy_score(article, config)
    if not payload.get("is_energy_relevant"):
        payload["is_energy_relevant"] = bool(payload.get("regions")) and score >= 2.5 and payload.get("event_type") != "other"
    payload["confidence"] = float(max(float(payload.get("confidence", 0.5)), min(0.95, 0.45 + 0.04 * score)))
    return NewsEventRecord(**payload)


def extract_candidate_articles(config: ForecastConfig, limit: int | None = None) -> Iterator[dict]:
    count = 0
    for article in iter_json_array(config.data.news_path):
        if article_matches_energy(article, config):
            yield article
            count += 1
            if limit is not None and count >= limit:
                return


def write_event_records(config: ForecastConfig, output_path: str | Path, limit: int | None = None, use_ollama: bool = True) -> int:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for article in extract_candidate_articles(config, limit=limit):
            if use_ollama:
                try:
                    record = ollama_event_record(article, config)
                except Exception:
                    record = heuristic_event_record(article, config)
            else:
                record = heuristic_event_record(article, config)
            if not record.is_energy_relevant:
                continue
            if not record.regions:
                continue
            if record.event_type == "other":
                continue
            if record.confidence < 0.45:
                continue
            handle.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_event_records(path: str | Path) -> list[NewsEventRecord]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(NewsEventRecord(**json.loads(line)))
    return records


def aggregate_event_features(records: Iterable[NewsEventRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        if not record.is_energy_relevant:
            continue
        if record.event_type == "other":
            continue
        if record.confidence < 0.45:
            continue
        publication = pd.to_datetime(record.publication_time).normalize()
        base_row = {
            "date": publication,
            "event_count": 1.0,
            "severity_sum": record.severity,
            "demand_up_score": record.severity if record.impact_direction == "up" else 0.0,
            "demand_down_score": record.severity if record.impact_direction == "down" else 0.0,
            "outage_score": record.severity if record.event_type == "outage" else 0.0,
            "extreme_weather_score": record.severity if record.event_type == "extreme_weather" else 0.0,
            "market_stress_score": record.severity if record.event_type == "market_stress" else 0.0,
            "llm_confidence_mean": record.confidence,
        }
        for region in record.regions:
            if region == "NATIONAL":
                for local_region in ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]:
                    weighted = dict(base_row)
                    weighted["region"] = local_region
                    for key in base_row:
                        if key not in {"date"}:
                            weighted[key] = base_row[key] * 0.5
                    rows.append(weighted)
            else:
                weighted = dict(base_row)
                weighted["region"] = region
                rows.append(weighted)
    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "region",
                "event_count_1d",
                "severity_sum_1d",
                "demand_up_score_1d",
                "demand_down_score_1d",
                "outage_score_1d",
                "extreme_weather_score_1d",
                "market_stress_score_1d",
                "llm_confidence_mean_1d",
            ]
        )

    frame = pd.DataFrame(rows)
    daily = frame.groupby(["date", "region"], as_index=False).agg(
        {
            "event_count": "sum",
            "severity_sum": "sum",
            "demand_up_score": "sum",
            "demand_down_score": "sum",
            "outage_score": "sum",
            "extreme_weather_score": "sum",
            "market_stress_score": "sum",
            "llm_confidence_mean": "mean",
        }
    )
    daily = daily.sort_values(["region", "date"]).reset_index(drop=True)
    feature_columns = [
        "event_count",
        "severity_sum",
        "demand_up_score",
        "demand_down_score",
        "outage_score",
        "extreme_weather_score",
        "market_stress_score",
        "llm_confidence_mean",
    ]
    expanded_rows = []
    for region, group in daily.groupby("region"):
        region_frame = group.set_index("date").sort_index()
        full_index = pd.date_range(region_frame.index.min(), region_frame.index.max(), freq="D")
        region_frame = region_frame.reindex(full_index).fillna(0.0)
        region_frame["region"] = region
        for window in (1, 3, 7):
            rolled = region_frame[feature_columns].rolling(window, min_periods=1).sum()
            conf = region_frame[["llm_confidence_mean"]].rolling(window, min_periods=1).mean()
            for column in feature_columns:
                source = conf if column == "llm_confidence_mean" else rolled
                region_frame[f"{column}_{window}d"] = source[column]
        region_frame = region_frame.reset_index().rename(columns={"index": "date"})
        keep = ["date", "region"] + [f"{column}_{window}d" for window in (1, 3, 7) for column in feature_columns]
        expanded_rows.append(region_frame[keep])
    return pd.concat(expanded_rows, ignore_index=True).sort_values(["region", "date"]).reset_index(drop=True)

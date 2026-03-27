from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aupower.config import ForecastConfig
from aupower.data.news import article_matches_energy, detect_regions, heuristic_event_record, iter_json_array


class NewsParserTests(unittest.TestCase):
    def test_streaming_json_array_parser(self) -> None:
        payload = [
            {"title": "Electricity outage hits Sydney", "summary": "Power outage", "publication_time": "2022-01-01T08:00:00"},
            {"title": "Sports update", "summary": "Football", "publication_time": "2022-01-01T09:00:00"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "news.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            rows = list(iter_json_array(path, chunk_size=16))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["title"], payload[0]["title"])

    def test_keyword_filter_and_heuristic_record(self) -> None:
        config = ForecastConfig()
        article = {
            "title": "Major electricity blackout across Sydney",
            "summary": "Power outage affects thousands.",
            "publication_time": "2022-01-02T06:00:00",
            "full_article": "A major electricity blackout and grid outage affected Sydney after a storm.",
            "link": "https://example.com/story",
        }
        self.assertTrue(article_matches_energy(article, config))
        record = heuristic_event_record(article)
        self.assertTrue(record.is_energy_relevant)
        self.assertIn("NSW1", record.regions)
        self.assertEqual(record.event_type, "outage")

    def test_negative_power_phrases_are_filtered(self) -> None:
        config = ForecastConfig()
        article = {
            "title": "Power couple expands property portfolio",
            "summary": "The real estate market is heating up again.",
            "publication_time": "2022-01-02T06:00:00",
            "full_article": "A celebrity power couple bought another home in Melbourne's property market.",
            "link": "https://example.com/property",
        }
        self.assertFalse(article_matches_energy(article, config))

    def test_out_of_scope_region_is_not_broadcast_to_national(self) -> None:
        article = {
            "title": "Major power outage hits Perth after storm",
            "summary": "Western Australia homes lose electricity.",
            "publication_time": "2022-01-02T06:00:00",
            "full_article": "A storm in Perth, Western Australia, caused a power outage across suburbs.",
        }
        self.assertEqual(detect_regions(article), [])

    def test_generic_service_outage_is_filtered(self) -> None:
        config = ForecastConfig()
        article = {
            "title": "Bank customers hit by app outage",
            "summary": "Mobile banking service outage frustrates users.",
            "publication_time": "2022-01-02T06:00:00",
            "full_article": "Customers could not access their app after a website outage.",
        }
        self.assertFalse(article_matches_energy(article, config))


if __name__ == "__main__":
    unittest.main()

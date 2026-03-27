from __future__ import annotations

import argparse
import json

from aupower.pipeline import backtest, extract_events, generate_report, prepare_data, predict, train_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Australia power forecasting pipeline")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config_option(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--config", default=None, help="Optional command-level config override")

    prepare_parser = subparsers.add_parser("prepare-data", help="Prepare load and weather artifacts")
    add_config_option(prepare_parser)

    extract_parser = subparsers.add_parser("extract-events", help="Extract news events into structured records")
    add_config_option(extract_parser)
    extract_parser.add_argument("--limit", type=int, default=None, help="Optional cap for faster iteration")
    extract_parser.add_argument("--no-ollama", action="store_true", help="Use heuristic event extraction instead of Ollama")

    train_parser = subparsers.add_parser("train", help="Train baselines and expert models")
    add_config_option(train_parser)

    backtest_parser = subparsers.add_parser("backtest", help="Run the fixed and agent-routed evaluations")
    add_config_option(backtest_parser)
    backtest_parser.add_argument("--use-ollama", action="store_true", help="Use the local Ollama router instead of rules only")

    report_parser = subparsers.add_parser("report", help="Generate the formal experiment report and ablation tables")
    add_config_option(report_parser)

    predict_parser = subparsers.add_parser("predict", help="Generate a single forecast")
    add_config_option(predict_parser)
    predict_parser.add_argument("--region", required=True)
    predict_parser.add_argument("--forecast-date", required=True, help="YYYY-MM-DD")
    predict_parser.add_argument("--no-ollama", action="store_true", help="Use rule router instead of Ollama")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config_path = args.config if getattr(args, "config", None) else "configs/default.yaml"
    if args.command == "prepare-data":
        prepare_data(config_path)
    elif args.command == "extract-events":
        extract_events(config_path, limit=args.limit, use_ollama=not args.no_ollama)
    elif args.command == "train":
        train_models(config_path)
    elif args.command == "backtest":
        backtest(config_path, use_ollama=args.use_ollama)
    elif args.command == "report":
        generate_report(config_path)
    elif args.command == "predict":
        output = predict(config_path, region=args.region, forecast_date=args.forecast_date, use_ollama=not args.no_ollama)
        print(output.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

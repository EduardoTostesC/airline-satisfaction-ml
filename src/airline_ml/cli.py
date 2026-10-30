"""Interface única para execução local ou chamada pelo notebook."""

import argparse
import json
import sys

from .config import load_config
from .data import read_csv
from .pipeline import audit, train
from .smoke import run_smoke
from .submission import validate_submission


def main():
    parser = argparse.ArgumentParser(description="Airline Satisfaction ML")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "train"):
        command = commands.add_parser(name)
        command.add_argument("--config", default="configs/catboost.json")
        command.add_argument("--data-dir")
        if name == "train":
            command.add_argument("--pilot-rows", type=int)
            command.add_argument("--folds", type=int, nargs="+")
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--output-dir", default="artifacts/smoke")
    smoke.add_argument("--model", choices=["catboost", "lightgbm", "xgboost"], default="catboost")
    validate = commands.add_parser("validate-submission")
    validate.add_argument("path")
    validate.add_argument("--data-dir", default="data/raw")
    args = parser.parse_args()
    try:
        if args.command == "smoke":
            result = run_smoke(args.output_dir, args.model)
        elif args.command == "validate-submission":
            from pathlib import Path

            folder = Path(args.data_dir)
            sample = read_csv(folder / "sample_submission.csv", "id")
            test = read_csv(folder / "test.csv", "id")
            validate_submission(read_csv(args.path, "id"), sample, test["id"])
            result = {"valid": True, "rows": len(sample), "submission_state": "validated_local"}
        else:
            config = load_config(args.config)
            if args.data_dir:
                config["data_dir"] = args.data_dir
            result = (
                audit(config)
                if args.command == "audit"
                else train(config, args.pilot_rows, args.folds)
            )
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (ValueError, FileNotFoundError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .benchmark import BenchmarkError, run_prediction_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score a complete local-model prediction file against the bundled benchmark."
    )
    parser.add_argument("predictions", help="JSON containing source and predictions")
    parser.add_argument("--output", help="optional JSON report destination (stdout by default)")
    args = parser.parse_args(argv)
    try:
        report = run_prediction_file(args.predictions)
    except (OSError, BenchmarkError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        try:
            Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        except OSError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
    else:
        print(rendered)
    return 0

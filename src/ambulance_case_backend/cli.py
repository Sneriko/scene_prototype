from __future__ import annotations

import argparse
from pathlib import Path

from .config import AppConfig
from .pipeline import AmbulanceCasePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ambulance case backend CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full pipeline for a case")
    run_parser.add_argument("--case-id", type=int, required=True, help="Case number to process")
    run_parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        pipeline = AmbulanceCasePipeline(AppConfig())
        result = pipeline.run_case(case_id=args.case_id, output_dir=args.output_dir)
        print(f"Wrote draft for case {result.case_id}.")


if __name__ == "__main__":
    main()

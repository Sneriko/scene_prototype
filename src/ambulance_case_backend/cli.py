from __future__ import annotations

import argparse
from pathlib import Path

from .config import AppConfig, KB_WHISPER_MODEL_MAP
from .pipeline import AmbulanceCasePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ambulance case backend CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full pipeline for a case")
    run_parser.add_argument("--case-id", type=int, required=True, help="Case number to process")
    run_parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory")
    run_parser.add_argument(
        "--transcription-backend",
        choices=["openai", "local_kb_whisper"],
        default="openai",
        help="Transcription backend to use.",
    )
    run_parser.add_argument(
        "--kb-whisper-size",
        choices=sorted(KB_WHISPER_MODEL_MAP),
        default="large",
        help="KB Whisper model size when using local_kb_whisper backend.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        pipeline = AmbulanceCasePipeline(
            AppConfig(
                transcription_backend=args.transcription_backend,
                kb_whisper_size=args.kb_whisper_size,
            )
        )
        result = pipeline.run_case(case_id=args.case_id, output_dir=args.output_dir)
        print(f"Wrote draft for case {result.case_id}.")


if __name__ == "__main__":
    main()

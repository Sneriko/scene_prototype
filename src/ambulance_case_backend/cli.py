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
        choices=["openai", "local_kb_whisper", "local_edge"],
        default="openai",
        help="Transcription/backend mode to use. local_edge keeps transcription and generation local.",
    )
    run_parser.add_argument(
        "--kb-whisper-size",
        choices=sorted(KB_WHISPER_MODEL_MAP),
        default="large",
        help="KB Whisper model size when using local_kb_whisper backend.",
    )

    serve_parser = subparsers.add_parser("serve-edge", help="Serve the ambulance edge API and Windows/PWA frontend")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    serve_parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    serve_parser.add_argument(
        "--transcription-backend",
        choices=["local_edge", "local_kb_whisper", "openai"],
        default="local_edge",
        help="Backend mode for the edge API. Use local_edge for production privacy.",
    )
    serve_parser.add_argument(
        "--kb-whisper-size",
        choices=sorted(KB_WHISPER_MODEL_MAP),
        default="large",
        help="KB Whisper model size for local ASR.",
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

    if args.command == "serve-edge":
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError("serve-edge requires the edge extra: pip install -e .[edge]") from exc

        from .edge_api import create_app

        app = create_app(
            AppConfig(
                transcription_backend=args.transcription_backend,
                kb_whisper_size=args.kb_whisper_size,
            )
        )
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

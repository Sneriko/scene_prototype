from ambulance_case_backend.cli import build_parser
from ambulance_case_backend.config import AppConfig


def test_kb_whisper_size_maps_to_model_id() -> None:
    config = AppConfig(kb_whisper_size="small")
    assert config.kb_whisper_model_id == "KBLab/kb-whisper-small"


def test_cli_accepts_local_backend_and_model_size() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--case-id",
            "2",
            "--transcription-backend",
            "local_kb_whisper",
            "--kb-whisper-size",
            "medium",
        ]
    )

    assert args.transcription_backend == "local_kb_whisper"
    assert args.kb_whisper_size == "medium"

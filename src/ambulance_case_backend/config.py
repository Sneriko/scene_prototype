from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass(slots=True)
class AppConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path = field(init=False)
    journals_dir: Path = field(init=False)
    audio_dir: Path = field(init=False)
    treatment_pdf: Path = field(init=False)
    output_dir: Path = field(init=False)
    transcription_model: str = "gpt-4o-transcribe"
    reasoning_model: str = "gpt-4.1"
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        self.journals_dir = self.data_dir / "journaler"
        self.audio_dir = self.data_dir / "ljudfiler"
        self.output_dir = self.project_root / "outputs"
        self.treatment_pdf = next(self.data_dir.glob("*.pdf"))

    def ensure_output_dir(self, output_dir: Path | None = None) -> Path:
        path = output_dir or self.output_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


KB_WHISPER_MODEL_MAP = {
    "tiny": "KBLab/kb-whisper-tiny",
    "base": "KBLab/kb-whisper-base",
    "small": "KBLab/kb-whisper-small",
    "medium": "KBLab/kb-whisper-medium",
    "large": "KBLab/kb-whisper-large",
}


@dataclass(slots=True)
class AppConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path = field(init=False)
    journals_dir: Path = field(init=False)
    audio_dir: Path = field(init=False)
    treatment_pdf: Path = field(init=False)
    output_dir: Path = field(init=False)
    transcription_model: str = "gpt-4o-transcribe"
    transcription_backend: str = field(default_factory=lambda: os.getenv("TRANSCRIPTION_BACKEND", "openai"))
    local_llm_base_url: str = field(default_factory=lambda: os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8001/v1"))
    local_llm_model: str = field(default_factory=lambda: os.getenv("LOCAL_LLM_MODEL", "qwen2.5-7b-instruct"))
    local_llm_api_key: str = field(default_factory=lambda: os.getenv("LOCAL_LLM_API_KEY", "local"))
    kb_whisper_size: str = field(default_factory=lambda: os.getenv("KB_WHISPER_SIZE", "large"))
    kb_whisper_model_id: str = field(init=False)
    reasoning_model: str = "gpt-4.1"
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    huggingface_token: str | None = field(default_factory=lambda: os.getenv("HUGGINGFACE_TOKEN"))

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        self.journals_dir = self.data_dir / "journaler"
        self.audio_dir = self.data_dir / "ljudfiler"
        self.output_dir = self.project_root / "outputs"
        self.treatment_pdf = next(self.data_dir.glob("*.pdf"))

        normalized_size = self.kb_whisper_size.lower().strip()
        if normalized_size not in KB_WHISPER_MODEL_MAP:
            valid_sizes = ", ".join(sorted(KB_WHISPER_MODEL_MAP))
            raise ValueError(f"Invalid KB_WHISPER_SIZE '{self.kb_whisper_size}'. Choose one of: {valid_sizes}.")
        self.kb_whisper_size = normalized_size
        self.kb_whisper_model_id = KB_WHISPER_MODEL_MAP[self.kb_whisper_size]

    def ensure_output_dir(self, output_dir: Path | None = None) -> Path:
        path = output_dir or self.output_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

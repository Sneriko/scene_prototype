from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import AppConfig
from .data_access import DataRepository
from .local_backend import LocalKBWhisperBackend
from .local_llm_backend import LocalEdgeBackend
from .models import CaseOutput
from .openai_client import OpenAIBackend
from .pdf_utils import extract_pdf_text


class AmbulanceCasePipeline:
    def __init__(self, config: AppConfig | None = None, backend: Any | None = None) -> None:
        self.config = config or AppConfig()
        self.repository = DataRepository(self.config)
        self.backend = backend or self._build_backend()

    def _build_backend(self) -> OpenAIBackend:
        if self.config.transcription_backend == "local_edge":
            return LocalEdgeBackend(self.config)
        if self.config.transcription_backend == "local_kb_whisper":
            return LocalKBWhisperBackend(self.config)
        return OpenAIBackend(self.config)

    def run_case(self, case_id: int, output_dir: Path | None = None) -> CaseOutput:
        case_assets = self.repository.get_case(case_id)
        reference_journals = self.repository.get_reference_journals(exclude_case_id=case_id)
        treatment_instructions = extract_pdf_text(self.config.treatment_pdf)

        raw_transcript = self.backend.transcribe_audio(case_assets.audio_path)
        try:
            diarized = self.backend.diarize_transcript(raw_transcript, audio_path=case_assets.audio_path)
        except TypeError:
            diarized = self.backend.diarize_transcript(raw_transcript)
        result = self.backend.generate_case_output(
            case_id=case_id,
            audio_path=case_assets.audio_path,
            raw_transcript=raw_transcript,
            diarized=diarized,
            treatment_instructions=treatment_instructions,
            reference_journals=reference_journals,
        )
        self.write_output(result, output_dir=output_dir)
        return result

    def run_audio_file(self, *, case_id: int, audio_path: Path, output_dir: Path | None = None) -> CaseOutput:
        reference_journals = self.repository.get_reference_journals(exclude_case_id=case_id)
        treatment_instructions = extract_pdf_text(self.config.treatment_pdf)

        raw_transcript = self.backend.transcribe_audio(audio_path)
        try:
            diarized = self.backend.diarize_transcript(raw_transcript, audio_path=audio_path)
        except TypeError:
            diarized = self.backend.diarize_transcript(raw_transcript)
        result = self.backend.generate_case_output(
            case_id=case_id,
            audio_path=audio_path,
            raw_transcript=raw_transcript,
            diarized=diarized,
            treatment_instructions=treatment_instructions,
            reference_journals=reference_journals,
        )
        self.write_output(result, output_dir=output_dir)
        return result

    def write_output(self, case_output: CaseOutput, output_dir: Path | None = None) -> Path:
        target_dir = self.config.ensure_output_dir(output_dir)
        target_path = target_dir / f"case_{case_output.case_id:02d}.json"
        target_path.write_text(case_output.to_json(indent=2), encoding="utf-8")
        return target_path

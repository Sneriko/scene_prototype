from __future__ import annotations

import json
from pathlib import Path

from .config import AppConfig
from .models import CaseOutput, DiarizedTranscript, TreatmentSuggestion
from .prompting import build_case_generation_prompt, build_diarization_prompt


class OpenAIBackend:
    def __init__(self, config: AppConfig) -> None:
        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "openai is required to run the transcription pipeline. Install package dependencies first."
            ) from exc
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)

    def transcribe_audio(self, audio_path: Path) -> str:
        with audio_path.open("rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                model=self.config.transcription_model,
                file=audio_file,
                prompt=(
                    "Transcribe this Swedish ambulance case recording accurately. "
                    "Preserve important medical terms, measurements, and speaker changes when possible."
                ),
            )
        return getattr(response, "text", str(response)).strip()

    def diarize_transcript(self, raw_transcript: str) -> DiarizedTranscript:
        response = self.client.chat.completions.create(
            model=self.config.reasoning_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You transform raw transcripts into structured JSON without adding unsupported facts.",
                },
                {"role": "user", "content": build_diarization_prompt(raw_transcript)},
            ],
        )
        payload = response.choices[0].message.content or "{}"
        return DiarizedTranscript.from_dict(json.loads(payload))

    def generate_case_output(
        self,
        *,
        case_id: int,
        audio_path: Path,
        raw_transcript: str,
        diarized: DiarizedTranscript,
        treatment_instructions: str,
        reference_journals: list[str],
    ) -> CaseOutput:
        response = self.client.chat.completions.create(
            model=self.config.reasoning_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You draft ambulance treatment suggestions and journals as structured JSON.",
                },
                {
                    "role": "user",
                    "content": build_case_generation_prompt(
                        case_id=case_id,
                        diarized=diarized,
                        treatment_instructions=treatment_instructions,
                        reference_journals=reference_journals,
                    ),
                },
            ],
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        suggestions = [TreatmentSuggestion.from_dict(item) for item in payload.get("treatment_suggestions", [])]
        return CaseOutput(
            case_id=case_id,
            audio_path=audio_path,
            raw_transcript=raw_transcript,
            diarized_transcript=diarized,
            treatment_suggestions=suggestions,
            drafted_journal=str(payload.get("drafted_journal", "")).strip(),
        )

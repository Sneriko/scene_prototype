from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json


@dataclass(slots=True)
class TranscriptSegment:
    speaker: str
    text: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptSegment":
        speaker = str(data.get("speaker", "unknown"))
        text_value = data.get("text")
        if text_value is None:
            text_value = data.get("utterance", data.get("content", ""))
        return cls(speaker=speaker, text=str(text_value))


@dataclass(slots=True)
class DiarizedTranscript:
    summary: str
    speakers: list[str]
    segments: list[TranscriptSegment]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiarizedTranscript":
        raw_segments = data.get("segments", [])
        segments: list[TranscriptSegment] = []
        for item in raw_segments:
            if isinstance(item, dict):
                segments.append(TranscriptSegment.from_dict(item))
            else:
                segments.append(TranscriptSegment(speaker="unknown", text=str(item)))

        return cls(
            summary=str(data.get("summary", "")),
            speakers=[str(item) for item in data.get("speakers", [])],
            segments=segments,
        )


@dataclass(slots=True)
class TreatmentSuggestion:
    title: str
    rationale: str
    urgency: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TreatmentSuggestion":
        return cls(
            title=str(data.get("title", "")),
            rationale=str(data.get("rationale", "")),
            urgency=str(data.get("urgency", "")),
        )


@dataclass(slots=True)
class CaseOutput:
    case_id: int
    audio_path: Path
    raw_transcript: str
    diarized_transcript: DiarizedTranscript
    treatment_suggestions: list[TreatmentSuggestion]
    drafted_journal: str

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=indent, default=str)

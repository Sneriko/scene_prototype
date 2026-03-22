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
        return cls(speaker=str(data["speaker"]), text=str(data["text"]))


@dataclass(slots=True)
class DiarizedTranscript:
    summary: str
    speakers: list[str]
    segments: list[TranscriptSegment]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiarizedTranscript":
        return cls(
            summary=str(data.get("summary", "")),
            speakers=[str(item) for item in data.get("speakers", [])],
            segments=[TranscriptSegment.from_dict(item) for item in data.get("segments", [])],
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

from pathlib import Path

from ambulance_case_backend.config import AppConfig
from ambulance_case_backend.local_llm_backend import LocalEdgeBackend
from ambulance_case_backend.models import DiarizedTranscript, TranscriptSegment


class FakeLocalClient:
    def chat_completion_json(self, **kwargs):
        return {
            "treatment_suggestions": [
                {"title": "Oxygen", "rationale": "Low saturation in transcript.", "urgency": "high"}
            ],
            "drafted_journal": "A local draft.",
        }


class LocalEdgeBackendForTest(LocalEdgeBackend):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.llm_client = FakeLocalClient()


def test_local_edge_backend_generates_case_output_without_openai_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = LocalEdgeBackendForTest(AppConfig(transcription_backend="local_edge"))

    result = backend.generate_case_output(
        case_id=123,
        audio_path=tmp_path / "audio.webm",
        raw_transcript="raw",
        diarized=DiarizedTranscript(
            summary="summary",
            speakers=["patient"],
            segments=[TranscriptSegment(speaker="patient", text="I cannot breathe")],
        ),
        treatment_instructions="oxygen guidance",
        reference_journals=[],
    )

    assert result.drafted_journal == "A local draft."
    assert result.treatment_suggestions[0].title == "Oxygen"

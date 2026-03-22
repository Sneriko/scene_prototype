from pathlib import Path

from ambulance_case_backend.config import AppConfig
from ambulance_case_backend.models import CaseOutput, DiarizedTranscript, TranscriptSegment, TreatmentSuggestion
from ambulance_case_backend.pipeline import AmbulanceCasePipeline


class FakeBackend:
    def transcribe_audio(self, audio_path: Path) -> str:
        return f"transcript for {audio_path.name}"

    def diarize_transcript(self, raw_transcript: str) -> DiarizedTranscript:
        return DiarizedTranscript(
            summary="test summary",
            speakers=["staff_1", "patient"],
            segments=[
                TranscriptSegment(speaker="staff_1", text="Hej, vad har hänt?"),
                TranscriptSegment(speaker="patient", text="Jag har ont i bröstet."),
            ],
        )

    def generate_case_output(self, **kwargs) -> CaseOutput:
        return CaseOutput(
            case_id=kwargs["case_id"],
            audio_path=kwargs["audio_path"],
            raw_transcript=kwargs["raw_transcript"],
            diarized_transcript=kwargs["diarized"],
            treatment_suggestions=[
                TreatmentSuggestion(title="Monitorering", rationale="Bröstsmärta kräver övervakning.", urgency="high")
            ],
            drafted_journal="Situation: Testfall.",
        )


def test_pipeline_writes_json_output(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("ambulance_case_backend.pipeline.extract_pdf_text", lambda _: "treatment guidance")
    pipeline = AmbulanceCasePipeline(config=AppConfig(), backend=FakeBackend())

    result = pipeline.run_case(case_id=1, output_dir=tmp_path)

    output_file = tmp_path / "case_01.json"
    assert result.case_id == 1
    assert output_file.exists()
    assert 'Situation: Testfall.' in output_file.read_text(encoding='utf-8')

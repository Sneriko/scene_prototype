from io import BytesIO
from pathlib import Path

from ambulance_case_backend.config import AppConfig
from ambulance_case_backend.edge_api import CaseStatus, EdgeCaseStore, demo_output_catalogs, demo_recording_media_type, load_demo_output, process_case
from ambulance_case_backend.pdf_export import journal_pdf, treatment_pdf


def test_edge_store_chunks_and_assembles_recording(tmp_path: Path) -> None:
    store = EdgeCaseStore(tmp_path)
    edge_case = store.create_case()

    store.save_chunk(edge_case.id, 2, BytesIO(b"second"))
    store.save_chunk(edge_case.id, 1, BytesIO(b"first"))
    assembled = store.assemble_chunks(edge_case.id)

    assert assembled.read_bytes() == b"firstsecond"
    assert store.get_case(edge_case.id).status == CaseStatus.QUEUED


def test_process_case_marks_case_failed_without_reraising_pipeline_errors(tmp_path: Path) -> None:
    store = EdgeCaseStore(tmp_path)
    edge_case = store.create_case()
    chunk_path = tmp_path / "chunk.webm"
    chunk_path.write_bytes(b"fake audio")
    with chunk_path.open("rb") as chunk_file:
        store.save_chunk(edge_case.id, 0, chunk_file)
    store.assemble_chunks(edge_case.id)

    result = process_case(
        store,
        AppConfig(transcription_backend="local_edge"),
        edge_case.id,
    )

    assert result is None
    assert edge_case.status == CaseStatus.FAILED
    assert edge_case.error is not None
    assert "optional local ASR dependencies" in edge_case.error


def test_process_case_reports_generating_status_after_transcription(tmp_path: Path, monkeypatch) -> None:
    store = EdgeCaseStore(tmp_path)
    edge_case = store.create_case()
    chunk_path = tmp_path / "chunk.webm"
    chunk_path.write_bytes(b"fake audio")
    with chunk_path.open("rb") as chunk_file:
        store.save_chunk(edge_case.id, 0, chunk_file)
    store.assemble_chunks(edge_case.id)
    observed_statuses: list[CaseStatus] = []

    class FakePipeline:
        def __init__(self, config: AppConfig) -> None:
            self.config = config

        def run_audio_file(self, **kwargs):
            kwargs["on_generation_start"]()
            observed_statuses.append(edge_case.status)
            raise RuntimeError("stop after generation marker")

    monkeypatch.setattr("ambulance_case_backend.edge_api.AmbulanceCasePipeline", FakePipeline)

    result = process_case(store, AppConfig(transcription_backend="local_edge"), edge_case.id)

    assert result is None
    assert observed_statuses == [CaseStatus.GENERATING]
    assert edge_case.status == CaseStatus.FAILED


def test_demo_output_can_be_loaded_from_outputs() -> None:
    result = load_demo_output(AppConfig(), 1)

    assert result.case_id == 1
    assert result.drafted_journal
    assert result.treatment_suggestions


def test_demo_pdf_endpoints_return_pdf_documents() -> None:
    result = load_demo_output(AppConfig(), 1)
    payload = treatment_pdf(result)

    assert payload.startswith(b"%PDF")
    assert b"%%EOF" in payload


def test_demo_output_catalogs_include_comparison_subdirectories(tmp_path: Path) -> None:
    config = AppConfig()
    config.output_dir = tmp_path
    (tmp_path / "local-qwen").mkdir()
    (tmp_path / "local-qwen" / "case_01.json").write_text("{}", encoding="utf-8")
    (tmp_path / "notes").mkdir()

    catalogs = demo_output_catalogs(config)

    assert [catalog["id"] for catalog in catalogs] == ["default", "local-qwen"]
    assert catalogs[1]["label"] == "Local Qwen"


def test_demo_output_can_be_loaded_from_comparison_catalog(tmp_path: Path) -> None:
    config = AppConfig()
    config.output_dir = tmp_path
    comparison_dir = tmp_path / "local-qwen"
    comparison_dir.mkdir()
    (comparison_dir / "case_01.json").write_text(
        """{
          "case_id": 1,
          "raw_transcript": "Local transcript",
          "diarized_transcript": {"summary": "", "speakers": [], "segments": []},
          "treatment_suggestions": [],
          "drafted_journal": "Local Qwen journal"
        }""",
        encoding="utf-8",
    )

    result = load_demo_output(config, 1, catalog="local-qwen")

    assert result.raw_transcript == "Local transcript"
    assert result.drafted_journal == "Local Qwen journal"


def test_demo_recording_media_type_supports_mp4_video() -> None:
    assert demo_recording_media_type(Path("Journal 1.mp4")) == "video/mp4"
    assert demo_recording_media_type(Path("Journal 1.m4a")) == "audio/mp4"


def test_journal_pdf_encodes_swedish_characters_for_builtin_pdf_fonts() -> None:
    result = load_demo_output(AppConfig(), 1)
    result.drafted_journal = "Patienten bor i Åkersberga och säger åäö ÅÄÖ."

    payload = journal_pdf(result)

    assert "Åkersberga".encode("cp1252") in payload
    assert "åäö ÅÄÖ".encode("cp1252") in payload
    assert "Åkersberga".encode("utf-8") not in payload

from io import BytesIO
from pathlib import Path

from ambulance_case_backend.config import AppConfig
from ambulance_case_backend.edge_api import CaseStatus, EdgeCaseStore, process_case


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

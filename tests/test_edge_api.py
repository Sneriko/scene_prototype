from io import BytesIO
from pathlib import Path

from ambulance_case_backend.edge_api import CaseStatus, EdgeCaseStore


def test_edge_store_chunks_and_assembles_recording(tmp_path: Path) -> None:
    store = EdgeCaseStore(tmp_path)
    edge_case = store.create_case()

    store.save_chunk(edge_case.id, 2, BytesIO(b"second"))
    store.save_chunk(edge_case.id, 1, BytesIO(b"first"))
    assembled = store.assemble_chunks(edge_case.id)

    assert assembled.read_bytes() == b"firstsecond"
    assert store.get_case(edge_case.id).status == CaseStatus.QUEUED

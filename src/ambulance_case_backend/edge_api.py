from dataclasses import dataclass, field
import json
from enum import Enum
from pathlib import Path
from shutil import copyfileobj
from threading import Lock
from typing import BinaryIO
from uuid import uuid4

from .config import AppConfig
from .data_access import DataRepository
from .models import CaseOutput
from .pdf_export import journal_pdf, treatment_pdf
from .pipeline import AmbulanceCasePipeline


class CaseStatus(str, Enum):
    RECORDING = "recording"
    QUEUED = "queued"
    TRANSCRIBING = "transcribing"
    GENERATING = "generating"
    READY = "ready"
    FAILED = "failed"


@dataclass(slots=True)
class EdgeCase:
    id: str
    numeric_case_id: int
    status: CaseStatus = CaseStatus.RECORDING
    audio_chunks: list[Path] = field(default_factory=list)
    assembled_audio: Path | None = None
    output: CaseOutput | None = None
    error: str | None = None


class EdgeCaseStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._cases: dict[str, EdgeCase] = {}
        self._next_numeric_case_id = 10_000

    def create_case(self) -> EdgeCase:
        with self._lock:
            case_id = uuid4().hex
            edge_case = EdgeCase(id=case_id, numeric_case_id=self._next_numeric_case_id)
            self._next_numeric_case_id += 1
            self._cases[case_id] = edge_case
        (self.root / case_id / "chunks").mkdir(parents=True, exist_ok=True)
        return edge_case

    def get_case(self, case_id: str) -> EdgeCase:
        try:
            return self._cases[case_id]
        except KeyError as exc:
            raise KeyError(f"Unknown case id: {case_id}") from exc

    def save_chunk(self, case_id: str, chunk_number: int, content: BinaryIO, suffix: str = ".webm") -> Path:
        edge_case = self.get_case(case_id)
        if edge_case.status != CaseStatus.RECORDING:
            raise ValueError(f"Cannot upload chunks while case is {edge_case.status}.")
        safe_suffix = suffix if suffix.startswith(".") and "/" not in suffix else ".webm"
        chunk_path = self.root / case_id / "chunks" / f"{chunk_number:06d}{safe_suffix}"
        with chunk_path.open("wb") as output_file:
            copyfileobj(content, output_file)
        with self._lock:
            edge_case.audio_chunks = sorted(set([*edge_case.audio_chunks, chunk_path]))
        return chunk_path

    def assemble_chunks(self, case_id: str, suffix: str = ".webm") -> Path:
        edge_case = self.get_case(case_id)
        if not edge_case.audio_chunks:
            raise ValueError("Cannot finish recording without at least one audio chunk.")
        output_path = self.root / case_id / f"recording{suffix}"
        with output_path.open("wb") as output_file:
            for chunk_path in sorted(edge_case.audio_chunks):
                with chunk_path.open("rb") as input_file:
                    copyfileobj(input_file, output_file)
        edge_case.assembled_audio = output_path
        edge_case.status = CaseStatus.QUEUED
        return output_path

    def set_failed(self, case_id: str, error: str) -> None:
        edge_case = self.get_case(case_id)
        edge_case.status = CaseStatus.FAILED
        edge_case.error = error


def case_output_from_dict(payload: dict, *, fallback_audio_path: Path) -> CaseOutput:
    from .models import DiarizedTranscript, TreatmentSuggestion

    return CaseOutput(
        case_id=int(payload["case_id"]),
        audio_path=Path(payload.get("audio_path") or fallback_audio_path),
        raw_transcript=str(payload.get("raw_transcript", "")),
        diarized_transcript=DiarizedTranscript.from_dict(payload.get("diarized_transcript", {})),
        treatment_suggestions=[
            TreatmentSuggestion.from_dict(item) for item in payload.get("treatment_suggestions", [])
        ],
        drafted_journal=str(payload.get("drafted_journal", "")),
    )


def load_demo_output(config: AppConfig, case_id: int) -> CaseOutput:
    output_path = config.output_dir / f"case_{case_id:02d}.json"
    if not output_path.exists():
        raise FileNotFoundError(f"No generated demo output exists for case {case_id}.")
    audio_path = config.audio_dir / f"Journal {case_id}.m4a"
    return case_output_from_dict(
        json.loads(output_path.read_text(encoding="utf-8")),
        fallback_audio_path=audio_path,
    )


def process_case(store: EdgeCaseStore, config: AppConfig, case_id: str) -> CaseOutput | None:
    edge_case = store.get_case(case_id)
    if edge_case.assembled_audio is None:
        raise ValueError("Case has no assembled audio file.")
    try:
        edge_case.status = CaseStatus.TRANSCRIBING
        pipeline = AmbulanceCasePipeline(config)
        result = pipeline.run_audio_file(
            case_id=edge_case.numeric_case_id,
            audio_path=edge_case.assembled_audio,
            output_dir=store.root / case_id,
            on_generation_start=lambda: setattr(edge_case, "status", CaseStatus.GENERATING),
        )
        edge_case.status = CaseStatus.READY
        edge_case.output = result
        return result
    except Exception as exc:
        store.set_failed(case_id, str(exc))
        return None


def create_app(config: AppConfig | None = None):
    from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import FileResponse, Response
    from fastapi.staticfiles import StaticFiles

    app_config = config or AppConfig(transcription_backend="local_edge")
    repository = DataRepository(app_config)
    store = EdgeCaseStore(app_config.project_root / "edge_cases")
    app = FastAPI(title="Ambulance Edge API")

    static_dir = Path(__file__).resolve().parent / "frontend"
    if static_dir.exists():
        app.mount("/app", StaticFiles(directory=static_dir, html=True), name="app")

    @app.get("/")
    def index():
        index_path = static_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return {"service": "ambulance-edge-api", "status": "ok"}

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "backend": app_config.transcription_backend,
            "local_llm_model": app_config.local_llm_model,
        }

    @app.get("/demo-cases")
    def demo_cases():
        cases = []
        for case_assets in repository.list_cases():
            output_path = app_config.output_dir / f"case_{case_assets.case_id:02d}.json"
            cases.append(
                {
                    "case_id": case_assets.case_id,
                    "label": f"Demo case {case_assets.case_id}",
                    "audio_file": case_assets.audio_path.name,
                    "journal_file": case_assets.journal_path.name,
                    "has_output": output_path.exists(),
                }
            )
        return {"cases": cases}

    @app.get("/demo-cases/{case_id}/output")
    def demo_case_output(case_id: int):
        try:
            return json.loads(load_demo_output(app_config, case_id).to_json())
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/demo-cases/{case_id}/treatment.pdf")
    def demo_treatment_pdf(case_id: int):
        try:
            payload = treatment_pdf(load_demo_output(app_config, case_id))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(
            payload,
            media_type="application/pdf",
            headers={"Content-Disposition": f'inline; filename="case-{case_id}-treatment.pdf"'},
        )

    @app.get("/demo-cases/{case_id}/journal.pdf")
    def demo_journal_pdf(case_id: int):
        try:
            payload = journal_pdf(load_demo_output(app_config, case_id))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(
            payload,
            media_type="application/pdf",
            headers={"Content-Disposition": f'inline; filename="case-{case_id}-journal.pdf"'},
        )

    @app.post("/cases")
    def create_case():
        edge_case = store.create_case()
        return {"case_id": edge_case.id, "status": edge_case.status}

    @app.post("/cases/{case_id}/audio-chunks")
    async def upload_chunk(case_id: str, chunk_number: int, file: UploadFile = File(...)):
        try:
            suffix = Path(file.filename or "chunk.webm").suffix or ".webm"
            store.save_chunk(case_id, chunk_number, file.file, suffix=suffix)
            return {"case_id": case_id, "chunk_number": chunk_number, "status": store.get_case(case_id).status}
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/cases/{case_id}/finish-recording")
    def finish_recording(case_id: str, background_tasks: BackgroundTasks):
        try:
            store.assemble_chunks(case_id)
            background_tasks.add_task(process_case, store, app_config, case_id)
            return {"case_id": case_id, "status": store.get_case(case_id).status}
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/cases/{case_id}/status")
    def case_status(case_id: str):
        try:
            edge_case = store.get_case(case_id)
            return {"case_id": case_id, "status": edge_case.status, "error": edge_case.error}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/cases/{case_id}/output")
    def case_output(case_id: str):
        try:
            edge_case = store.get_case(case_id)
            if edge_case.output is None:
                raise HTTPException(status_code=409, detail=f"Case is {edge_case.status}.")
            return json.loads(edge_case.output.to_json())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/cases/{case_id}/treatment.pdf")
    def case_treatment_pdf(case_id: str):
        try:
            edge_case = store.get_case(case_id)
            if edge_case.output is None:
                raise HTTPException(status_code=409, detail=f"Case is {edge_case.status}.")
            return Response(
                treatment_pdf(edge_case.output),
                media_type="application/pdf",
                headers={"Content-Disposition": f'inline; filename="{case_id}-treatment.pdf"'},
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/cases/{case_id}/journal.pdf")
    def case_journal_pdf(case_id: str):
        try:
            edge_case = store.get_case(case_id)
            if edge_case.output is None:
                raise HTTPException(status_code=409, detail=f"Case is {edge_case.status}.")
            return Response(
                journal_pdf(edge_case.output),
                media_type="application/pdf",
                headers={"Content-Disposition": f'inline; filename="{case_id}-journal.pdf"'},
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app

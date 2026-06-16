# Ambulance edge deployment plan

This document describes a privacy-preserving deployment where patient audio stays inside the ambulance. A Windows tablet/laptop records audio and talks only to an NVIDIA Spark device on the ambulance LAN. The Spark runs local ASR, diarization, treatment suggestion, and journal drafting services.

## Target architecture

```text
Windows device (crew UI)
  - browser/PWA or small desktop wrapper
  - microphone capture, pause/resume, case metadata form
  - uploads encrypted audio chunks over local HTTPS/WebSocket
  - displays transcript, treatment suggestions, and drafted journal
        |
        | ambulance-only Wi-Fi/Ethernet; no cloud dependency
        v
NVIDIA Spark edge server
  - FastAPI API gateway
  - local object storage for short-lived audio chunks
  - local transcription worker (Whisper-family model)
  - local diarization worker
  - local LLM worker (Qwen or Gemma via vLLM / llama.cpp / Ollama)
  - encrypted local audit/output database
```

## Recommended implementation path

1. **Keep this repository's pipeline contract, but split it behind an API.**
   The current pipeline already separates audio transcription, diarization, treatment-instruction retrieval, treatment suggestions, and journal drafting. Build a FastAPI service around those stages so the Windows client can create a case, stream/upload audio, request processing, poll or subscribe to status, and retrieve a final `CaseOutput` JSON payload.

2. **Replace cloud generation before deploying real patient data.**
   The repository currently supports local KB Whisper for transcription and local pyannote diarization, but the local backend still inherits OpenAI-backed journal/treatment generation. For an ambulance deployment, add a fully local generation backend whose `generate_case_output(...)` calls a local OpenAI-compatible endpoint such as vLLM, llama.cpp server, or Ollama running Qwen/Gemma.

3. **Run all model services on the Spark.**
   Prefer Docker Compose on the Spark with separate containers for:
   - `api`: FastAPI app, validation, auth, case lifecycle, model orchestration.
   - `asr`: faster-whisper / whisper.cpp / KB Whisper, depending on Swedish accuracy and Spark performance.
   - `diarization`: pyannote or an equivalent local speaker diarization service.
   - `llm`: Qwen or Gemma served through an OpenAI-compatible local API.
   - `storage`: encrypted volume or SQLite/Postgres on encrypted disk for temporary cases.

4. **Make the Windows UI thin.**
   Use a browser PWA first: it is easy to run on Windows, can capture microphone audio with `MediaRecorder`, can operate on a fixed local URL, and avoids native installation complexity. Add a native wrapper later only if you need OS-level device management, offline kiosk mode, or hardware buttons.

5. **Design for intermittent connectivity inside the ambulance.**
   The Windows client should buffer chunks locally until the Spark acknowledges receipt. The Spark should expose `/health`, `/cases/{id}/status`, and resumable chunk upload endpoints. The UI should show whether the data is still local, uploaded to Spark, being transcribed, or ready for review.

6. **Add clinical safety rails.**
   Treat generated instructions as decision support, not autonomous care. Return evidence snippets/section references from the treatment PDF with every suggestion, require crew review before journal export, and keep a clear audit trail of generated versus edited text.

## Suggested API shape

- `POST /cases` creates a case and returns a case id.
- `POST /cases/{case_id}/audio-chunks` uploads a numbered audio chunk.
- `POST /cases/{case_id}/finish-recording` tells the Spark to assemble and process audio.
- `GET /cases/{case_id}/status` returns recording, queued, transcribing, generating, ready, or failed.
- `GET /cases/{case_id}/transcript` returns raw and diarized transcript data.
- `GET /cases/{case_id}/output` returns treatment suggestions and drafted journal.
- `PATCH /cases/{case_id}/journal` stores crew edits to the journal draft.
- `POST /cases/{case_id}/export` exports the final journal to the ambulance record system or removable encrypted media.

## Model recommendations

- **ASR:** benchmark Swedish accuracy first. Start with KB Whisper because this repository already supports `KBLab/kb-whisper-*`, then compare faster-whisper large-v3/turbo and whisper.cpp quantized builds for latency.
- **Diarization:** keep pyannote for prototypes, but pre-download model weights onto the Spark. Do not require live Hugging Face access in production.
- **LLM:** start with a Qwen or Gemma instruct model served locally through an OpenAI-compatible API. Use quantized variants if memory is tight. Evaluate with real ambulance-style Swedish transcripts and treatment-richtlinjer prompts before choosing the model size.
- **Retrieval:** chunk the treatment PDF into a local vector or BM25 index and pass only relevant sections into the LLM prompt. This reduces latency and hallucination risk compared with injecting the entire PDF each time.

## Security and privacy requirements

- No outbound internet route from the ambulance inference network during production cases.
- Local TLS between Windows and Spark, with a pinned certificate in the PWA/native wrapper.
- Device-level authentication, short-lived operator sessions, and role separation for review/export.
- Full-disk encryption on Spark and Windows.
- Automatic deletion or retention policies for raw audio, transcripts, and generated drafts.
- Explicit redaction controls if any logs leave the vehicle for debugging.
- Disable telemetry in model-serving, frontend, and backend dependencies.

## How to test the full setup on a Dell XPS

1. **Run the Spark stack locally with Docker Compose.**
   Use the Dell as a single-node simulation: API, ASR, diarization, LLM, and frontend all on localhost. If the Dell has an NVIDIA GPU, use NVIDIA Container Toolkit; otherwise use CPU or smaller quantized models to validate workflows rather than performance.

2. **Use this repository's sample cases as golden tests.**
   The `data/ljudfiler` and `data/journaler` folders provide paired recordings and reference journals. Run the existing CLI against the sample cases, then compare transcript quality, suggestion grounding, and journal similarity before and after local-model changes.

3. **Add a mock-fast path for repeatable integration tests.**
   Provide mock ASR and mock LLM containers that return deterministic outputs. This lets you test chunk upload, job state transitions, UI behavior, and output rendering without waiting for large models.

4. **Emulate the ambulance LAN.**
   Put the Windows client, or a browser on the Dell, on the same Wi-Fi as the Spark-equivalent host. Test by disabling the Dell's internet route after model weights are downloaded. Confirm recording, upload, transcription, generation, and final journal review still work.

5. **Run end-to-end acceptance drills.**
   For each scenario, measure: time to first transcript, time to completed draft, GPU/CPU memory, audio loss after network interruption, recovery after Spark restart, output correctness, and whether any traffic attempts to leave the local network.

6. **Perform privacy verification.**
   Use firewall logs, DNS logs, and packet capture to verify that production-case audio and transcript data never leave the ambulance LAN. Include a test that fails if the backend tries to call OpenAI or Hugging Face during production mode.

## Milestones

1. Containerize the current backend and expose a local API.
2. Add a simple PWA frontend for recording and case review.
3. Add local LLM generation backend and remove OpenAI dependency from production mode.
4. Add local retrieval over treatment instructions.
5. Add deterministic mock services and end-to-end tests on the Dell XPS.
6. Benchmark and tune on the NVIDIA Spark.
7. Lock down networking, storage encryption, telemetry, and retention for ambulance deployment.

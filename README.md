# Ambulance AI Copilot POC

A proof-of-concept demo for turning ambulance conversations into structured treatment support and a draft ambulance journal.

The app now has two demo-friendly workflows:

1. **Finished demo cases** — load existing generated outputs from `outputs/` directly in the browser, bypassing recording. This is intended for coworker/investor demos where you want a reliable, repeatable walkthrough.
2. **Live recording** — record audio in the browser, upload chunks to the local edge API, and process the recording through the configured backend. This is the intended future ambulance workflow.

The backend still supports both a fully local edge path and an OpenAI-powered comparison path.

## What the POC shows

- A polished browser demo UI served by the local FastAPI edge server.
- Demo-case loading from the existing `data/` and `outputs/` folders.
- Live browser audio recording for future ambulance use.
- Suggested treatment instructions rendered in the UI and exposed as a displayable PDF.
- Draft ambulance journal rendered in the UI and exposed as a displayable PDF.
- Backend modes for:
  - fully local edge processing (`local_edge`),
  - local KB Whisper transcription with OpenAI generation (`local_kb_whisper`),
  - OpenAI transcription/generation (`openai`).

## Repository layout

```text
src/ambulance_case_backend/
├── cli.py                 # CLI entry point: run cases or serve the demo app
├── config.py              # paths, environment variables, model/backend settings
├── data_access.py         # discovers demo journals/audio from data/
├── edge_api.py            # FastAPI app, demo endpoints, recording endpoints, PDF endpoints
├── frontend/              # browser POC UI
├── local_backend.py       # local KB Whisper + optional pyannote diarization
├── local_llm_backend.py   # local OpenAI-compatible LLM generation
├── models.py              # dataclasses for transcripts, suggestions, and final output
├── openai_client.py       # OpenAI transcription, diarization, and generation backend
├── pdf_export.py          # lightweight PDF rendering for treatment/journal outputs
├── pdf_utils.py           # treatment-guideline PDF text extraction
├── pipeline.py            # orchestration pipeline
└── prompting.py           # shared prompts for diarization and case generation
```

## Installation

Install the base package:

```bash
pip install -e .
```

Install developer/test dependencies:

```bash
pip install -e '.[dev]'
```

Install local ASR/diarization dependencies:

```bash
pip install -e '.[local_asr]'
```

Install the full edge API/frontend stack:

```bash
pip install -e '.[edge]'
```

## Running the demo UI

Start the edge API and browser UI:

```bash
ambulance-case serve-edge --host 0.0.0.0 --port 8080 --transcription-backend local_edge
```

Then open:

```text
http://127.0.0.1:8080
```

The landing page automatically loads the first available finished demo case from `outputs/` and shows:

- suggested treatment instructions,
- draft journal text,
- an inline treatment PDF,
- an inline journal PDF,
- links to open each PDF in a new tab.

## Demo-case mode

Demo-case mode is for reliable presentations. It uses the checked-in/generated JSON files in `outputs/` and does not call a transcription or LLM backend.

The frontend calls:

- `GET /demo-cases` to list demo cases found in `data/`,
- `GET /demo-cases/{case_id}/output` to load the finished JSON output,
- `GET /demo-cases/{case_id}/treatment.pdf` to display treatment suggestions as a PDF,
- `GET /demo-cases/{case_id}/journal.pdf` to display the draft journal as a PDF.

This keeps demo playback stable even when model services are unavailable.

## Live recording mode

Live recording mode is the future production workflow. The browser records microphone audio using `MediaRecorder`, uploads chunks to the edge server, finishes the recording, and polls for output.

The frontend calls:

- `POST /cases` to create a recording case,
- `POST /cases/{case_id}/audio-chunks` to upload audio chunks,
- `POST /cases/{case_id}/finish-recording` to assemble chunks and start processing,
- `GET /cases/{case_id}/status` to poll processing status,
- `GET /cases/{case_id}/output` to fetch the final JSON,
- `GET /cases/{case_id}/treatment.pdf` to display the treatment PDF,
- `GET /cases/{case_id}/journal.pdf` to display the journal PDF.

## Backend modes

### Fully local edge mode

Use this for the intended privacy-preserving ambulance architecture:

```bash
export LOCAL_LLM_BASE_URL=http://127.0.0.1:11434/v1
export LOCAL_LLM_MODEL=qwen2.5:7b
ambulance-case serve-edge --host 0.0.0.0 --port 8080 --transcription-backend local_edge
```

`local_edge` uses:

- KB Whisper for transcription,
- pyannote for speaker diarization when `HUGGINGFACE_TOKEN` is configured,
- a local OpenAI-compatible LLM endpoint for treatment/journal generation.

If pyannote access is unavailable, transcription still works and segments are labeled as `speaker_unknown`.

### Local transcription with OpenAI generation

Use this for a hybrid comparison:

```bash
export OPENAI_API_KEY=your_key_here
ambulance-case serve-edge --host 0.0.0.0 --port 8080 --transcription-backend local_kb_whisper
```

### OpenAI comparison mode

Use this when you want the whole processing pipeline to run through OpenAI APIs:

```bash
export OPENAI_API_KEY=your_key_here
ambulance-case serve-edge --host 0.0.0.0 --port 8080 --transcription-backend openai
```

## Generating or refreshing demo outputs

Run the pipeline for an existing case:

```bash
ambulance-case run --case-id 3 --output-dir outputs --transcription-backend openai
```

Run with local KB Whisper transcription:

```bash
ambulance-case run --case-id 3 --output-dir outputs --transcription-backend local_kb_whisper --kb-whisper-size large
```

Local KB Whisper requires an `ffmpeg` executable on `PATH` to decode the bundled `.m4a` demo recordings.
The recordings are MPEG-4/M4A files, so a "malformed soundfile" error usually means `ffmpeg` is missing
or not visible to the Python environment, not that the demo audio is corrupt.

Speaker diarization also requires a Hugging Face token exported as `HUGGINGFACE_TOKEN`, `HF_TOKEN`, or
`HUGGINGFACE_HUB_TOKEN`, with accepted user conditions for both `pyannote/speaker-diarization-3.1` and
`pyannote/segmentation-3.0`.

Generated files are written as:

```text
outputs/case_03.json
```

The demo UI will pick them up automatically.

## Local LLM quick start with Ollama

Install and start Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

In a second terminal:

```bash
ollama pull qwen2.5:7b
curl http://127.0.0.1:11434/v1/models
```

Then run the edge app:

```bash
export LOCAL_LLM_BASE_URL=http://127.0.0.1:11434/v1
export LOCAL_LLM_MODEL=qwen2.5:7b
ambulance-case serve-edge --host 0.0.0.0 --port 8080 --transcription-backend local_edge
```

If `ollama serve` says the address is already in use, Ollama is already running.

## Tests

Run the test suite:

```bash
pytest
```

## Deployment direction

This POC is structured so the demo can be shown today while still pointing toward the intended product architecture:

- browser recording in the ambulance,
- processing on a local/edge server,
- local transcription and diarization,
- local LLM generation,
- PDFs for review, sharing, or export,
- OpenAI mode kept for demo comparison and benchmark purposes.

For broader deployment planning, see `docs/ambulance_edge_deployment.md`.

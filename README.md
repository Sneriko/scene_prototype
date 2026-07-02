# ambulance-case-backend

Python backend package for processing ambulance case recordings with OpenAI or a local KB Whisper transcription backend.

## Features

- Discovers paired audio recordings and ground-truth journals from the `data/` folder.
- Extracts treatment instructions from the supplied PDF.
- Transcribes ambulance case recordings.
- Supports:
  - OpenAI transcription (`gpt-4o-transcribe`), or
  - local KB Whisper transcription (`KBLab/kb-whisper-*`).
- Can diarize locally with `pyannote.audio` when using KB Whisper.
- Generates treatment suggestions grounded in the treatment instructions.
- Drafts a journal that matches the style of the provided example journals.
- Excludes the target journal from the in-context examples when generating output for its matching recording.

## Installation

```bash
pip install -e .
```

For local KB Whisper transcription + diarization support:

```bash
pip install -e .[local_asr]
```

Set environment variables:

```bash
export OPENAI_API_KEY=your_key_here
```

Speaker diarization uses the gated `pyannote/speaker-diarization-3.1` model when it is available. To enable real speaker labels, create a Hugging Face access token, accept/request access to the pyannote model terms on Hugging Face, and export the token before starting the backend:

```bash
export HUGGINGFACE_TOKEN=your_hf_token
```

If the token is missing or does not have access to the gated pyannote repo, local KB Whisper transcription still runs and the backend returns `speaker_unknown` transcript segments instead of failing the frontend request.

If you see a `torchcodec is not installed correctly` warning from pyannote, that is an FFmpeg/TorchCodec audio-decoder warning rather than a missing Hugging Face token. The browser UI converts recordings to WAV before upload, and the backend loads local ASR/diarization audio into memory with `soundfile` before passing it to Transformers or pyannote so uploaded browser recordings do not require server-side FFmpeg decoding.

## CLI usage

Run the full pipeline for one case number (OpenAI transcription):

```bash
ambulance-case run --case-id 3
```

Run with local KB Whisper transcription + speaker diarization:

```bash
ambulance-case run --case-id 3 --transcription-backend local_kb_whisper --kb-whisper-size large
```

Available KB Whisper size options: `tiny`, `base`, `small`, `medium`, `large`.

Write outputs to a custom directory:

```bash
ambulance-case run --case-id 3 --output-dir outputs
```

## Notes

- The package is structured as backend-only code so a frontend can be added later.
- The transcription and generation steps are intentionally separated so they can be swapped or cached later.
- `pypdf` is required at runtime to read the treatment-instruction PDF.
- Even in local transcription mode, the final treatment suggestions/journal drafting currently uses OpenAI chat completion models.

## Edge API and local ambulance UI

Install the API/frontend dependencies:

```bash
pip install -e '.[edge]'
```

Serve the local ambulance API and browser UI:

```bash
ambulance-case serve-edge --host 0.0.0.0 --port 8080 --transcription-backend local_edge
```

`local_edge` uses local KB Whisper/pyannote for transcription/diarization and a local OpenAI-compatible LLM endpoint configured with `LOCAL_LLM_BASE_URL`, `LOCAL_LLM_MODEL`, and `LOCAL_LLM_API_KEY`. The `edge` extra includes the local ASR/diarization packages (`transformers`, `torch`, and `pyannote.audio`) required by that mode.

## Deployment planning

For a privacy-preserving ambulance setup with a Windows recording UI and an NVIDIA Spark edge server, see [`docs/ambulance_edge_deployment.md`](docs/ambulance_edge_deployment.md).


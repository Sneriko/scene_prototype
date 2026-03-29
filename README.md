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

For local diarization model downloads (`pyannote/speaker-diarization-3.1`):

```bash
export HUGGINGFACE_TOKEN=your_hf_token
```

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

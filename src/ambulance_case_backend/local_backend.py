from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import shutil
import subprocess
import warnings
from typing import Any

from .config import AppConfig
from .models import DiarizedTranscript, TranscriptSegment
from .openai_client import OpenAIBackend


class LocalKBWhisperBackend(OpenAIBackend):
    """Backend that runs transcription + optional local diarization and reuses OpenAI for case generation."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._openai_generation_backend: OpenAIBackend | None = None
        self._asr_pipeline = self._build_asr_pipeline()
        self._diarization_pipeline = self._build_diarization_pipeline()

    def _build_asr_pipeline(self):
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        except ImportError as exc:
            missing_package = exc.name or "transformers/torch"
            raise RuntimeError(
                f"Local KB Whisper mode requires the optional local ASR dependencies; missing {missing_package!r}. "
                "Install them with: pip install -e '.[local_asr]' (or pip install -e '.[edge]' for the edge server)."
            ) from exc

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.config.kb_whisper_model_id,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(self.config.kb_whisper_model_id)

        model.to("cuda" if torch.cuda.is_available() else "cpu")
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
        )
        self._validate_local_audio_decoder()
        return asr_pipeline

    def _build_diarization_pipeline(self):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"\s*torchcodec is not installed correctly.*",
                    category=UserWarning,
                    module=r"pyannote\.audio\.core\.io",
                )
                from pyannote.audio import Pipeline
            from pyannote.core import Segment
        except ImportError as exc:
            missing_package = exc.name or "pyannote.audio"
            raise RuntimeError(
                f"Local diarization requires the optional local ASR dependencies; missing {missing_package!r}. "
                "Install them with: pip install -e '.[local_asr]' (or pip install -e '.[edge]' for the edge server)."
            ) from exc

        self._segment_type = Segment
        if not self.config.huggingface_token:
            warnings.warn(
                "HUGGINGFACE_TOKEN is not set; continuing without pyannote speaker diarization. "
                "Output will use a single 'speaker_unknown' speaker label.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        try:
            return Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=self.config.huggingface_token,
            )
        except Exception as exc:
            if self._is_huggingface_access_error(exc):
                warnings.warn(
                    "Cannot access gated pyannote speaker diarization models with the configured "
                    "Hugging Face token; continuing without speaker diarization. Visit and accept access "
                    "for https://huggingface.co/pyannote/speaker-diarization-3.1, "
                    "https://huggingface.co/pyannote/segmentation-3.0, and, with newer pyannote.audio "
                    "releases, https://huggingface.co/pyannote/speaker-diarization-community-1, "
                    "then restart the backend with an "
                    "authorized HUGGINGFACE_TOKEN, HF_TOKEN, or HUGGINGFACE_HUB_TOKEN to enable speaker labels.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None
            raise

    @staticmethod
    def _is_huggingface_access_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(
            marker in text
            for marker in ("401", "403", "gated repo", "private or gated", "restricted", "unauthorized", "could not download")
        )

    @staticmethod
    def _validate_local_audio_decoder() -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "Local KB Whisper requires the ffmpeg executable to decode demo .m4a recordings and browser audio. "
                "Install ffmpeg and make sure it is on PATH, then rerun the command. The bundled recordings are "
                "valid MPEG-4/M4A files; this error means the local decoder is unavailable, not that the repo audio "
                "is in the wrong format."
            )

    def transcribe_audio(self, audio_path: Path) -> str:
        result = self._asr_pipeline(
            self._load_audio_for_transformers(audio_path),
            return_timestamps=True,
            generate_kwargs={"language": "sw"},
        )
        text = result.get("text", "") if isinstance(result, dict) else str(result)
        return str(text).strip()

    def diarize_transcript(self, raw_transcript: str, audio_path: Path | None = None) -> DiarizedTranscript:  # type: ignore[override]
        if audio_path is None:
            raise ValueError("audio_path is required for local diarization backend.")

        asr_result = self._asr_pipeline(
            self._load_audio_for_transformers(audio_path),
            return_timestamps=True,
            generate_kwargs={"language": "sw"},
        )
        chunks = asr_result.get("chunks", []) if isinstance(asr_result, dict) else []

        if self._diarization_pipeline is None:
            return self._transcript_without_speaker_diarization(raw_transcript, chunks)

        diarization = self._diarization_pipeline(self._load_audio_for_pyannote(audio_path))
        speaker_aliases: dict[str, str] = {}
        alias_counter = 1
        segments: list[TranscriptSegment] = []

        for chunk in chunks:
            timestamp = chunk.get("timestamp", ()) if isinstance(chunk, dict) else ()
            if not isinstance(timestamp, (tuple, list)) or len(timestamp) != 2:
                continue
            start, end = timestamp
            if start is None or end is None:
                continue

            overlaps = diarization.crop(self._segment_type(float(start), float(end)), mode="intersection")
            duration_by_speaker: defaultdict[str, float] = defaultdict(float)
            for segment, _, speaker in overlaps.itertracks(yield_label=True):
                duration_by_speaker[str(speaker)] += float(segment.duration)

            if duration_by_speaker:
                speaker_label = max(duration_by_speaker.items(), key=lambda item: item[1])[0]
            else:
                speaker_label = "SPEAKER_UNKNOWN"

            if speaker_label not in speaker_aliases:
                speaker_aliases[speaker_label] = f"speaker_{alias_counter}"
                alias_counter += 1

            text = str(chunk.get("text", "")).strip()
            if text:
                segments.append(
                    TranscriptSegment(speaker=speaker_aliases[speaker_label], text=text)
                )

        summary = f"Diarized {len(segments)} segments with local KB Whisper + pyannote pipeline."
        return DiarizedTranscript(
            summary=summary,
            speakers=sorted(set(item.speaker for item in segments)),
            segments=segments,
        )

    def _transcript_without_speaker_diarization(self, raw_transcript: str, chunks: list[Any]) -> DiarizedTranscript:
        segments = [
            TranscriptSegment(speaker="speaker_unknown", text=str(chunk.get("text", "")).strip())
            for chunk in chunks
            if isinstance(chunk, dict) and str(chunk.get("text", "")).strip()
        ]
        if not segments and raw_transcript.strip():
            segments = [TranscriptSegment(speaker="speaker_unknown", text=raw_transcript.strip())]
        return DiarizedTranscript(
            summary=(
                f"Transcribed {len(segments)} segments with local KB Whisper. "
                "Speaker diarization was skipped because pyannote model access is not configured."
            ),
            speakers=["speaker_unknown"] if segments else [],
            segments=segments,
        )


    def _load_audio_for_transformers(self, audio_path: Path) -> dict[str, Any]:
        """Decode audio with ffmpeg and pass raw samples to Transformers.

        The Transformers ASR pipeline shells out to ffmpeg when given a path, but
        its generic "malformed soundfile" error hides the real decoder stderr.
        Decoding here makes local .m4a/.mp4 handling explicit and avoids path
        suffix quirks while still feeding Whisper 16 kHz mono audio.
        """
        return {"array": self._decode_audio_with_ffmpeg(audio_path), "sampling_rate": 16_000}

    @staticmethod
    def _decode_audio_with_ffmpeg(audio_path: Path) -> Any:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "Local KB Whisper audio decoding requires 'numpy'. Install local ASR dependencies with: "
                "pip install -e '.[local_asr]' (or pip install -e '.[edge]' for the edge server)."
            ) from exc

        command = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "-",
        ]
        try:
            completed = subprocess.run(command, check=True, capture_output=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Local KB Whisper requires the ffmpeg executable to decode demo .m4a recordings and browser audio. "
                "Install ffmpeg and make sure it is on PATH, then rerun the command."
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode(errors="replace").strip()
            detail = f" ffmpeg said: {stderr}" if stderr else ""
            raise RuntimeError(
                f"ffmpeg could not decode {audio_path}.{detail} The bundled recordings should be valid M4A files; "
                "if this is an uploaded file, convert it to wav, flac, mp3, m4a, or mp4 and try again."
            ) from exc

        if not completed.stdout:
            raise RuntimeError(f"ffmpeg decoded no audio samples from {audio_path}.")
        return np.frombuffer(completed.stdout, dtype=np.float32).copy()

    def _load_audio_for_pyannote(self, audio_path: Path) -> dict[str, Any]:
        """Load audio into memory so pyannote does not need torchcodec/FFmpeg decoding."""
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Local diarization audio loading requires 'torch'. "
                "Install local ASR dependencies with: pip install -e '.[local_asr]' (or pip install -e '.[edge]' for the edge server)"
            ) from exc

        waveform = torch.from_numpy(self._decode_audio_with_ffmpeg(audio_path)).unsqueeze(0)
        return {"waveform": waveform, "sample_rate": 16_000}

    def generate_case_output(self, **kwargs):
        if self._openai_generation_backend is None:
            self._openai_generation_backend = OpenAIBackend(self.config)
        return self._openai_generation_backend.generate_case_output(**kwargs)

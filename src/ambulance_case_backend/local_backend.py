from __future__ import annotations

from collections import defaultdict
from pathlib import Path
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
        return pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
        )

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
                    "Cannot access gated pyannote/speaker-diarization-3.1 model with the configured "
                    "HUGGINGFACE_TOKEN; continuing without speaker diarization. Visit "
                    "https://huggingface.co/pyannote/speaker-diarization-3.1, request/accept access, "
                    "then restart the backend with an authorized token to enable speaker labels.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None
            raise

    @staticmethod
    def _is_huggingface_access_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(marker in text for marker in ("401", "403", "gated repo", "restricted", "unauthorized"))

    def transcribe_audio(self, audio_path: Path) -> str:
        result = self._asr_pipeline(str(audio_path), generate_kwargs={"language": "sw"})
        text = result.get("text", "") if isinstance(result, dict) else str(result)
        return str(text).strip()

    def diarize_transcript(self, raw_transcript: str, audio_path: Path | None = None) -> DiarizedTranscript:  # type: ignore[override]
        if audio_path is None:
            raise ValueError("audio_path is required for local diarization backend.")

        asr_result = self._asr_pipeline(str(audio_path), return_timestamps=True, generate_kwargs={"language": "sw"})
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

    def _load_audio_for_pyannote(self, audio_path: Path) -> dict[str, Any]:
        """Load audio into memory so pyannote does not need torchcodec/FFmpeg decoding."""
        try:
            import soundfile as sf
            import torch
        except ImportError as exc:
            missing_package = exc.name or "soundfile/torch"
            raise RuntimeError(
                f"Local diarization audio loading requires {missing_package!r}. "
                "Install local ASR dependencies with: pip install -e '.[local_asr]' (or pip install -e '.[edge]' for the edge server)"
            ) from exc

        samples, sample_rate = sf.read(str(audio_path), always_2d=True, dtype="float32")
        waveform = torch.from_numpy(samples.T.copy())
        return {"waveform": waveform, "sample_rate": int(sample_rate)}

    def generate_case_output(self, **kwargs):
        if self._openai_generation_backend is None:
            self._openai_generation_backend = OpenAIBackend(self.config)
        return self._openai_generation_backend.generate_case_output(**kwargs)

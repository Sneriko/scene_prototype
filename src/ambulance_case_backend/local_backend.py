from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .config import AppConfig
from .models import DiarizedTranscript, TranscriptSegment
from .openai_client import OpenAIBackend


class LocalKBWhisperBackend(OpenAIBackend):
    """Backend that runs transcription + diarization locally and reuses OpenAI for case generation."""

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
            raise RuntimeError(
                "Local KB Whisper mode requires 'transformers' and 'torch'. "
                "Install with: pip install transformers torch"
            ) from exc

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.config.kb_whisper_model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(self.config.kb_whisper_model_id)

        model.to("cuda" if torch.cuda.is_available() else "cpu")
        return pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            return_timestamps=True,
        )

    def _build_diarization_pipeline(self):
        try:
            from pyannote.audio import Pipeline
            from pyannote.core import Segment
        except ImportError as exc:
            raise RuntimeError(
                "Local diarization requires 'pyannote.audio'. Install with: pip install pyannote.audio"
            ) from exc

        if not self.config.huggingface_token:
            raise ValueError(
                "HUGGINGFACE_TOKEN is required for diarization model downloads (pyannote/speaker-diarization-3.1)."
            )

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.config.huggingface_token,
        )
        self._segment_type = Segment
        return pipeline

    def transcribe_audio(self, audio_path: Path) -> str:
        result = self._asr_pipeline(str(audio_path), generate_kwargs={"language": "sw"})
        text = result.get("text", "") if isinstance(result, dict) else str(result)
        return str(text).strip()

    def diarize_transcript(self, raw_transcript: str, audio_path: Path | None = None) -> DiarizedTranscript:  # type: ignore[override]
        if audio_path is None:
            raise ValueError("audio_path is required for local diarization backend.")

        diarization = self._diarization_pipeline(str(audio_path))
        asr_result = self._asr_pipeline(str(audio_path), return_timestamps=True, generate_kwargs={"language": "sw"})
        chunks = asr_result.get("chunks", []) if isinstance(asr_result, dict) else []

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
    def generate_case_output(self, **kwargs):
        if self._openai_generation_backend is None:
            self._openai_generation_backend = OpenAIBackend(self.config)
        return self._openai_generation_backend.generate_case_output(**kwargs)


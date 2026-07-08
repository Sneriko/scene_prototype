import sys
import types

from ambulance_case_backend.config import AppConfig
from ambulance_case_backend.local_backend import LocalKBWhisperBackend


def test_diarization_pipeline_uses_pyannote_token_keyword(monkeypatch) -> None:
    captured_kwargs = {}

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            captured_kwargs["model_id"] = model_id
            captured_kwargs.update(kwargs)
            return "fake-pipeline"

    fake_audio = types.ModuleType("pyannote.audio")
    fake_audio.Pipeline = FakePipeline
    fake_core = types.ModuleType("pyannote.core")
    fake_core.Segment = object

    monkeypatch.setitem(sys.modules, "pyannote", types.ModuleType("pyannote"))
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio)
    monkeypatch.setitem(sys.modules, "pyannote.core", fake_core)

    backend = LocalKBWhisperBackend.__new__(LocalKBWhisperBackend)
    backend.config = AppConfig(huggingface_token="hf_test_token")

    pipeline = backend._build_diarization_pipeline()

    assert pipeline == "fake-pipeline"
    assert captured_kwargs == {
        "model_id": "pyannote/speaker-diarization-3.1",
        "token": "hf_test_token",
    }


def test_diarization_pipeline_falls_back_without_huggingface_token(monkeypatch) -> None:
    fake_audio = types.ModuleType("pyannote.audio")
    fake_audio.Pipeline = object
    fake_core = types.ModuleType("pyannote.core")
    fake_core.Segment = object

    monkeypatch.setitem(sys.modules, "pyannote", types.ModuleType("pyannote"))
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio)
    monkeypatch.setitem(sys.modules, "pyannote.core", fake_core)

    backend = LocalKBWhisperBackend.__new__(LocalKBWhisperBackend)
    backend.config = AppConfig(huggingface_token=None)

    assert backend._build_diarization_pipeline() is None


def test_diarization_pipeline_falls_back_on_gated_model_error(monkeypatch) -> None:
    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            raise RuntimeError("403 Client Error: Cannot access gated repo. Access is restricted.")

    fake_audio = types.ModuleType("pyannote.audio")
    fake_audio.Pipeline = FakePipeline
    fake_core = types.ModuleType("pyannote.core")
    fake_core.Segment = object

    monkeypatch.setitem(sys.modules, "pyannote", types.ModuleType("pyannote"))
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio)
    monkeypatch.setitem(sys.modules, "pyannote.core", fake_core)

    backend = LocalKBWhisperBackend.__new__(LocalKBWhisperBackend)
    backend.config = AppConfig(huggingface_token="hf_unauthorized")

    assert backend._build_diarization_pipeline() is None


def test_transcript_without_speaker_diarization_uses_chunks() -> None:
    backend = LocalKBWhisperBackend.__new__(LocalKBWhisperBackend)

    diarized = backend._transcript_without_speaker_diarization(
        "full transcript",
        [{"text": " Hej. "}, {"text": ""}, {"text": "Ont i bröstet."}],
    )

    assert diarized.summary == (
        "Transcribed 2 segments with local KB Whisper. "
        "Speaker diarization was skipped because pyannote model access is not configured."
    )
    assert diarized.speakers == ["speaker_unknown"]
    assert [segment.text for segment in diarized.segments] == ["Hej.", "Ont i bröstet."]


def test_local_audio_decoder_error_mentions_demo_m4a(monkeypatch) -> None:
    monkeypatch.setattr("ambulance_case_backend.local_backend.shutil.which", lambda name: None)

    try:
        LocalKBWhisperBackend._validate_local_audio_decoder()
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive assertion helper
        raise AssertionError("Expected missing ffmpeg to raise RuntimeError")

    assert "ffmpeg" in message
    assert "demo .m4a recordings" in message
    assert "not that the repo audio is in the wrong format" in message


def test_huggingface_access_error_detects_segmentation_download_failure() -> None:
    exc = RuntimeError("Could not download Model from pyannote/segmentation-3.0")

    assert LocalKBWhisperBackend._is_huggingface_access_error(exc)

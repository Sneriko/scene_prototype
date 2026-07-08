import sys
import types
from pathlib import Path

import pytest

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


def test_transcribe_audio_requests_timestamps_for_long_form_whisper(monkeypatch) -> None:
    captured_kwargs = {}

    class FakePipeline:
        def __call__(self, audio, **kwargs):
            captured_kwargs["audio"] = audio
            captured_kwargs.update(kwargs)
            return {"text": "  Hej från ambulansen.  "}

    backend = LocalKBWhisperBackend.__new__(LocalKBWhisperBackend)
    backend._asr_pipeline = FakePipeline()
    monkeypatch.setattr(
        backend,
        "_load_audio_for_transformers",
        lambda audio_path: {"array": [0.0], "sampling_rate": 16_000},
    )

    transcript = backend.transcribe_audio(Path("long-recording.m4a"))

    assert transcript == "Hej från ambulansen."
    assert captured_kwargs == {
        "audio": {"array": [0.0], "sampling_rate": 16_000},
        "return_timestamps": True,
        "generate_kwargs": {"language": "sw"},
    }

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


def test_huggingface_access_error_detects_community_pipeline_download_failure() -> None:
    exc = RuntimeError("Could not download xvec_transform.npz from pyannote/speaker-diarization-community-1")

    assert LocalKBWhisperBackend._is_huggingface_access_error(exc)


def test_transformers_audio_loading_decodes_with_ffmpeg(monkeypatch) -> None:
    captured_command = {}

    class FakeArray(list):
        def copy(self):
            return self

        def tolist(self):
            return list(self)

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.float32 = "float32"
    fake_numpy.frombuffer = lambda data, dtype: FakeArray([0.0, 1.0])
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)

    class Completed:
        stdout = b"\x00\x00\x00\x00\x00\x00\x80?"

    def fake_run(command, **kwargs):
        captured_command["command"] = command
        captured_command.update(kwargs)
        return Completed()

    monkeypatch.setattr("ambulance_case_backend.local_backend.subprocess.run", fake_run)

    backend = LocalKBWhisperBackend.__new__(LocalKBWhisperBackend)
    payload = backend._load_audio_for_transformers(Path("data/ljudfiler/Journal 1.m4a"))

    assert payload["sampling_rate"] == 16_000
    assert payload["array"].tolist() == [0.0, 1.0]
    assert captured_command["command"] == [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        "data/ljudfiler/Journal 1.m4a",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "f32le",
        "-",
    ]
    assert captured_command["check"] is True
    assert captured_command["capture_output"] is True


def test_transformers_audio_loading_reports_ffmpeg_stderr(monkeypatch) -> None:
    import subprocess

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.float32 = "float32"
    fake_numpy.frombuffer = lambda data, dtype: []
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)

    def fake_run(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr=b"bad atom")

    monkeypatch.setattr("ambulance_case_backend.local_backend.subprocess.run", fake_run)

    backend = LocalKBWhisperBackend.__new__(LocalKBWhisperBackend)
    with pytest.raises(RuntimeError, match="bad atom"):
        backend._load_audio_for_transformers(Path("broken.m4a"))

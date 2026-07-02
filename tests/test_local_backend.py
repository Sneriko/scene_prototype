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

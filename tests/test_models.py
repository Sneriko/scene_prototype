from ambulance_case_backend.models import DiarizedTranscript


def test_diarized_transcript_handles_missing_text_with_content_fallback() -> None:
    payload = {
        "summary": "short",
        "speakers": ["staff_1", "patient"],
        "segments": [
            {"speaker": "staff_1", "content": "Hej."},
            {"speaker": "patient", "utterance": "Ont i bröstet."},
        ],
    }

    diarized = DiarizedTranscript.from_dict(payload)

    assert diarized.segments[0].text == "Hej."
    assert diarized.segments[1].text == "Ont i bröstet."


def test_diarized_transcript_handles_non_dict_segments() -> None:
    payload = {
        "segments": ["raw line"],
    }

    diarized = DiarizedTranscript.from_dict(payload)

    assert diarized.segments[0].speaker == "unknown"
    assert diarized.segments[0].text == "raw line"

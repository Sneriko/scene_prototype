from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib import request

from .config import AppConfig
from .local_backend import LocalKBWhisperBackend
from .models import CaseOutput, DiarizedTranscript, TreatmentSuggestion
from .prompting import build_case_generation_prompt


class LocalOpenAICompatibleClient:
    """Tiny client for local OpenAI-compatible chat-completions servers."""

    def __init__(self, base_url: str, api_key: str = "local") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def chat_completion_json(self, *, model: str, messages: list[dict[str, str]], timeout_seconds: int = 300) -> dict[str, Any]:
        payload = json.dumps(
            {
                "model": model,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
            }
        ).encode("utf-8")
        http_request = request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(http_request, timeout=timeout_seconds) as response:  # noqa: S310 - local configured URL.
            response_payload = json.loads(response.read().decode("utf-8"))
        content = response_payload["choices"][0]["message"].get("content") or "{}"
        return json.loads(content)


class LocalEdgeBackend(LocalKBWhisperBackend):
    """Fully local backend: local ASR/diarization plus local OpenAI-compatible LLM generation."""

    def __init__(self, config: AppConfig, llm_client: LocalOpenAICompatibleClient | None = None) -> None:
        super().__init__(config)
        self.llm_client = llm_client or LocalOpenAICompatibleClient(
            base_url=config.local_llm_base_url,
            api_key=config.local_llm_api_key,
        )

    def generate_case_output(
        self,
        *,
        case_id: int,
        audio_path: Path,
        raw_transcript: str,
        diarized: DiarizedTranscript,
        treatment_instructions: str,
        reference_journals: list[str],
    ) -> CaseOutput:
        payload = self.llm_client.chat_completion_json(
            model=self.config.local_llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "You draft ambulance treatment suggestions and journals as structured JSON.",
                },
                {
                    "role": "user",
                    "content": build_case_generation_prompt(
                        case_id=case_id,
                        diarized=diarized,
                        treatment_instructions=treatment_instructions,
                        reference_journals=reference_journals,
                    ),
                },
            ],
        )
        suggestions = [TreatmentSuggestion.from_dict(item) for item in payload.get("treatment_suggestions", [])]
        return CaseOutput(
            case_id=case_id,
            audio_path=audio_path,
            raw_transcript=raw_transcript,
            diarized_transcript=diarized,
            treatment_suggestions=suggestions,
            drafted_journal=str(payload.get("drafted_journal", "")).strip(),
        )

from __future__ import annotations

from .models import DiarizedTranscript


def build_diarization_prompt(raw_transcript: str) -> str:
    return f"""
You are a medical-transcription post-processor.

Goal:
1. Clean up the transcript.
2. Infer speaker turns.
3. Normalize speakers using labels like staff_1, staff_2, patient, relative, caller, or bystander.
4. Return valid JSON with keys: summary, speakers, segments.
5. Keep the spoken content in the same language as the source material.
6. Do not invent clinical facts that are not present in the transcript.

Raw transcript:
{raw_transcript}
""".strip()


def render_diarized_transcript(diarized: DiarizedTranscript) -> str:
    return "\n".join(f"{segment.speaker}: {segment.text}" for segment in diarized.segments)


def build_case_generation_prompt(
    *,
    case_id: int,
    diarized: DiarizedTranscript,
    treatment_instructions: str,
    reference_journals: list[str],
) -> str:
    formatted_examples = "\n\n---\n\n".join(reference_journals)
    transcript_text = render_diarized_transcript(diarized)
    return f"""
You are helping draft structured ambulance documentation.

Case number: {case_id}

Tasks:
1. Review the diarized transcript.
2. Use the treatment instructions as the clinical grounding source.
3. Produce treatment suggestions that are explicitly tied to the case facts.
4. Write a draft journal in the same style and structure as the example journals.
5. Do not copy any example journal verbatim.
6. Do not mention that one journal was withheld.
7. If information is uncertain, state it carefully instead of inventing details.

Treatment instructions:
{treatment_instructions}

Reference journals:
{formatted_examples}

Diarized transcript:
{transcript_text}

Return valid JSON with this schema:
{{
  "treatment_suggestions": [
    {{"title": "...", "rationale": "...", "urgency": "low|medium|high"}}
  ],
  "drafted_journal": "..."
}}
""".strip()

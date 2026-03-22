from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .config import AppConfig

CASE_NUMBER_PATTERN = re.compile(r"Journal\s+(\d+)")


@dataclass(slots=True)
class CaseAssets:
    case_id: int
    audio_path: Path
    journal_path: Path
    ground_truth_journal: str


class DataRepository:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    @staticmethod
    def _extract_case_id(path: Path) -> int:
        match = CASE_NUMBER_PATTERN.search(path.name)
        if not match:
            raise ValueError(f"Could not extract case id from {path}")
        return int(match.group(1))

    def list_cases(self) -> list[CaseAssets]:
        journals = {
            self._extract_case_id(path): path
            for path in self.config.journals_dir.glob("*.txt")
        }
        audio_files = {
            self._extract_case_id(path): path
            for path in self.config.audio_dir.glob("*.m4a")
            if not path.name.endswith(":Zone.Identifier")
        }
        case_ids = sorted(set(journals) & set(audio_files))
        return [self.get_case(case_id) for case_id in case_ids]

    def get_case(self, case_id: int) -> CaseAssets:
        journal_path = next(path for path in self.config.journals_dir.glob("*.txt") if self._extract_case_id(path) == case_id)
        audio_path = next(
            path for path in self.config.audio_dir.glob("*.m4a")
            if not path.name.endswith(":Zone.Identifier") and self._extract_case_id(path) == case_id
        )
        return CaseAssets(
            case_id=case_id,
            audio_path=audio_path,
            journal_path=journal_path,
            ground_truth_journal=journal_path.read_text(encoding="utf-8").strip(),
        )

    def get_reference_journals(self, exclude_case_id: int) -> list[str]:
        references: list[tuple[int, str]] = []
        for journal_path in sorted(self.config.journals_dir.glob("*.txt")):
            case_id = self._extract_case_id(journal_path)
            if case_id == exclude_case_id:
                continue
            references.append((case_id, journal_path.read_text(encoding="utf-8").strip()))
        return [text for _, text in sorted(references)]

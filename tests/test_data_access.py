from ambulance_case_backend.config import AppConfig
from ambulance_case_backend.data_access import DataRepository


def test_reference_journals_exclude_target_case() -> None:
    repo = DataRepository(AppConfig())
    refs = repo.get_reference_journals(exclude_case_id=3)

    assert len(refs) == 4
    assert all("Journal 3" not in journal for journal in refs)


def test_list_cases_discovers_all_pairs() -> None:
    repo = DataRepository(AppConfig())
    case_ids = [case.case_id for case in repo.list_cases()]

    assert case_ids == [1, 2, 3, 4, 5]


def test_list_cases_includes_mp4_recordings(tmp_path) -> None:
    config = AppConfig()
    config.data_dir = tmp_path
    config.audio_dir = tmp_path / "ljudfiler"
    config.journals_dir = tmp_path / "journaler"
    config.audio_dir.mkdir(parents=True)
    config.journals_dir.mkdir(parents=True)
    (config.audio_dir / "Journal 9.mp4").write_bytes(b"fake mp4")
    (config.journals_dir / "Journal 9.txt").write_text("Journal 9 text", encoding="utf-8")

    [case] = DataRepository(config).list_cases()

    assert case.case_id == 9
    assert case.audio_path.name == "Journal 9.mp4"

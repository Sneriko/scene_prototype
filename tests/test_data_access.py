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

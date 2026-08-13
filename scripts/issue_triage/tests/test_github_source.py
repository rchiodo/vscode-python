from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

from vscode_python_issue_triage.github_source import (
    GitHubIssueSource,
    issue_event_number,
    normalize_timeline_event,
)


class _IssueWithoutHydratedPullRequest:
    def __init__(self, number: int, *, pull_request: bool = False) -> None:
        self.number = number
        self.raw_data: dict[str, object] = {"number": number}
        if pull_request:
            self.raw_data["pull_request"] = {"url": "https://example.test/pull/1"}

    @property
    def pull_request(self) -> object:
        raise AssertionError("The bulk issue list must not hydrate pull_request")


class _RepositoryWithIssues:
    def __init__(self, issues: list[_IssueWithoutHydratedPullRequest]) -> None:
        self._issues = issues

    def get_issues(self, **_kwargs: object) -> list[_IssueWithoutHydratedPullRequest]:
        return self._issues


class _CursorPages:
    def __init__(self) -> None:
        self._PaginatedList__nextUrl: str | None = "https://api.example.test/issues"
        self._PaginatedList__nextParams: dict[str, object] = {}
        self._recovered = False

    def _fetchNextPage(self) -> list[_IssueWithoutHydratedPullRequest]:  # noqa: N802
        if not self._recovered:
            assert self._PaginatedList__nextParams["page"] == 99
            self._PaginatedList__nextUrl = "https://api.example.test/issues?after=cursor"
            self._recovered = True
            return [_IssueWithoutHydratedPullRequest(98)]
        self._PaginatedList__nextUrl = None
        return [_IssueWithoutHydratedPullRequest(99)]


class _RepositoryWithCursor:
    def __init__(self) -> None:
        self.pages = _CursorPages()

    def get_issues(self, **_kwargs: object) -> _CursorPages:
        return self.pages


def test_list_closed_issues_filters_pull_requests_without_hydration() -> None:
    source = cast(Any, object.__new__(GitHubIssueSource))
    source._repository = _RepositoryWithIssues(
        [
            _IssueWithoutHydratedPullRequest(1),
            _IssueWithoutHydratedPullRequest(2, pull_request=True),
        ]
    )

    issues = GitHubIssueSource.list_closed_issues(source, limit=None)

    assert [issue.number for issue in issues] == [1]


def test_issue_pages_recover_cursor_from_legacy_page_checkpoint() -> None:
    source = cast(Any, object.__new__(GitHubIssueSource))
    source._repository = _RepositoryWithCursor()

    pages = tuple(GitHubIssueSource.iter_closed_issue_pages(source, completed_pages=99))

    assert len(pages) == 1
    assert pages[0][0] is None
    assert pages[0][1][0].number == 99


class _LazyRepositoryEvent:
    def __init__(self) -> None:
        self._rawData = {
            "event": "labeled",
            "created_at": "2026-01-01T00:00:00Z",
            "actor": {"login": "triage-bot"},
            "label": {"name": "~spam"},
            "issue": {"number": 42},
        }

    @property
    def raw_data(self) -> object:
        raise AssertionError("Repository events must not hydrate raw_data")


def test_repository_event_uses_stored_payload_without_hydration() -> None:
    event = _LazyRepositoryEvent()

    normalized = normalize_timeline_event(event)

    assert issue_event_number(event) == 42
    assert normalized.label == "~spam"


def test_normalize_event_reads_label_from_raw_timeline_data() -> None:
    event = SimpleNamespace(
        event="labeled",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        actor=SimpleNamespace(login="triage-bot"),
        raw_data={"label": {"name": "~spam", "color": "ededed"}},
    )

    normalized = normalize_timeline_event(event)

    assert normalized.label == "~spam"


def test_normalize_event_reads_previous_title() -> None:
    event = SimpleNamespace(
        event="renamed",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        actor=SimpleNamespace(login="reporter"),
        raw_data={"rename": {"from": "Original title", "to": "Current title"}},
    )

    normalized = normalize_timeline_event(event)

    assert normalized.renamed_from == "Original title"

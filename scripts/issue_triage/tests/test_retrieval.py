from __future__ import annotations

from vscode_python_issue_triage.ml import MLRecord
from vscode_python_issue_triage.retrieval import HistoricalIssueRetriever


def _record(
    number: int,
    *,
    created_at: str,
    title: str,
    body: str,
) -> MLRecord:
    return MLRecord(
        repository="microsoft/vscode-python",
        issue_number=number,
        title=title,
        body=body,
        url=f"https://github.com/microsoft/vscode-python/issues/{number}",
        created_at=created_at,
        content_source="github_current_snapshot",
        classification="bug",
        disposition="keep_open",
        labels=("bug",),
        information_request_ids=(),
    )


def test_retrieval_prefers_similar_issue_and_excludes_future_reports() -> None:
    similar_prior = _record(
        1,
        created_at="2020-01-01T00:00:00+00:00",
        title="Pytest discovery crashes",
        body="Test discovery fails with an exception.",
    )
    unrelated_prior = _record(
        2,
        created_at="2020-01-02T00:00:00+00:00",
        title="Select conda interpreter",
        body="Environment selection request.",
    )
    target = _record(
        3,
        created_at="2020-01-03T00:00:00+00:00",
        title="Pytest discovery crash",
        body="Test discovery fails and raises an exception.",
    )
    nearly_identical_future = _record(
        4,
        created_at="2020-01-04T00:00:00+00:00",
        title="Pytest discovery crash",
        body="Test discovery fails and raises an exception.",
    )
    retriever = HistoricalIssueRetriever(
        (similar_prior, unrelated_prior, target, nearly_identical_future)
    )

    evidence = retriever.retrieve(target, count=3)

    assert evidence[0].issue_number == similar_prior.issue_number
    assert {item.issue_number for item in evidence} == {1, 2}
    assert all(item.created_at < target.created_at for item in evidence)

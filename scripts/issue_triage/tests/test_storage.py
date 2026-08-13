from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from vscode_python_issue_triage.evaluation import compare_decision
from vscode_python_issue_triage.models import (
    ActualOutcome,
    Disposition,
    IssueSnapshot,
    TriageDecision,
)
from vscode_python_issue_triage.storage import ResultStore


def test_result_store_caches_and_exports(tmp_path: Path) -> None:
    database = tmp_path / "results.sqlite3"
    issue = IssueSnapshot(
        repository="microsoft/vscode-python",
        number=123,
        title="Test discovery fails",
        body="Tests are missing.",
        author="reporter",
        url="https://github.com/microsoft/vscode-python/issues/123",
        created_at="2026-01-01T00:00:00+00:00",
        closed_at="2026-01-02T00:00:00+00:00",
        state_reason="completed",
        final_labels=("bug", "area-testing"),
        timeline=(),
        comments=(),
    )
    decision = TriageDecision(
        classification="bug",
        labels=("bug", "area-testing"),
        disposition=Disposition.KEEP_OPEN,
        transfer_target=None,
        information_request_ids=(),
        confidence=0.9,
        rationale="The report identifies a test discovery failure.",
    )
    actual = ActualOutcome(
        classification="bug",
        labels=("area-testing", "bug"),
        disposition=Disposition.KEEP_OPEN,
        transfer_target=None,
        information_request_ids=(),
        information_request_comments=(),
    )

    with ResultStore(database) as store:
        store.begin_run("baseline", issue.repository, {"model": "test"})
        store.cache_issue(issue)
        store.set_cohort("baseline", issue.repository, (issue.number,))
        store.bind_run_snapshot("baseline", issue)
        store.record_result(
            run_id="baseline",
            issue=issue,
            prediction=decision,
            actual=actual,
            comparison=compare_decision(decision, actual),
            rendered_information_request=None,
        )
        store.cache_issue(replace(issue, body="A later global cache refresh."))
        store.finish_run("baseline")
        details, summary = store.export_run("baseline", tmp_path / "output")

        assert store.get_run_snapshot("baseline", issue.repository, issue.number) == issue

    detail = json.loads(details.read_text(encoding="utf-8"))
    aggregate = json.loads(summary.read_text(encoding="utf-8"))
    assert detail["issue"]["body"] == issue.body
    assert detail["comparison"]["overall_score"] == 1
    assert aggregate["classification_accuracy"] == 1


def test_result_store_rejects_changed_configuration(tmp_path: Path) -> None:
    with ResultStore(tmp_path / "results.sqlite3") as store:
        store.begin_run("baseline", "microsoft/vscode-python", {"agent_sha256": "first"})

        with pytest.raises(ValueError, match="different configuration"):
            store.begin_run(
                "baseline",
                "microsoft/vscode-python",
                {"agent_sha256": "second"},
            )


def test_result_store_preserves_ordered_cohort(tmp_path: Path) -> None:
    with ResultStore(tmp_path / "results.sqlite3") as store:
        store.begin_run("baseline", "microsoft/vscode-python", {"agent_sha256": "same"})
        store.set_cohort("baseline", "microsoft/vscode-python", (30, 20, 10))

        assert store.get_cohort("baseline") == (30, 20, 10)

        with pytest.raises(ValueError, match="different issue cohort"):
            store.set_cohort("baseline", "microsoft/vscode-python", (31, 20, 10))

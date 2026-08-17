from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from vscode_python_issue_triage.hosted import live_record_from_event, prepare_live_context
from vscode_python_issue_triage.ml import MLRecord, write_dataset


def _event() -> dict[str, object]:
    return {
        "issue": {
            "number": 20,
            "title": "Pytest discovery crashes",
            "body": "Discovery fails with an exception.",
            "html_url": "https://github.com/microsoft/vscode-python/issues/20",
            "created_at": "2024-02-01T00:00:00+00:00",
            "labels": [{"name": "triage-needed"}],
        },
        "repository": {"full_name": "microsoft/vscode-python"},
    }


def test_live_record_uses_issues_event_without_outcome_claims() -> None:
    record = live_record_from_event(_event())

    assert record.issue_number == 20
    assert record.classification == "unknown"
    assert record.labels == ("triage-needed",)


def test_prepare_live_context_retrieves_only_earlier_issues(tmp_path: Path) -> None:
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(_event()), encoding="utf-8")
    corpus_path = tmp_path / "corpus.jsonl"
    records = [
        MLRecord(
            repository="microsoft/vscode-python",
            issue_number=10,
            title="Pytest discovery failure",
            body="Discovery raises an exception.",
            url="https://github.com/microsoft/vscode-python/issues/10",
            created_at="2024-01-01T00:00:00+00:00",
            content_source="github_current_snapshot",
            classification="bug",
            disposition="keep_open",
            labels=("area-testing", "bug"),
            information_request_ids=(),
        ),
        MLRecord(
            repository="microsoft/vscode-python",
            issue_number=30,
            title="Pytest discovery failure",
            body="Discovery raises an exception.",
            url="https://github.com/microsoft/vscode-python/issues/30",
            created_at="2024-03-01T00:00:00+00:00",
            content_source="github_current_snapshot",
            classification="bug",
            disposition="keep_open",
            labels=("area-testing", "bug"),
            information_request_ids=(),
        ),
    ]
    write_dataset(corpus_path, records)
    output_path = tmp_path / "context.json"

    context = prepare_live_context(
        event_path=event_path,
        corpus_path=corpus_path,
        output_path=output_path,
        retrieval_count=5,
    )

    retrieved = cast(list[dict[str, object]], context["retrieval"]["issues"])
    assert [item["issue_number"] for item in retrieved] == [10]
    assert output_path.exists()

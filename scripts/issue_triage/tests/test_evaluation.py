from __future__ import annotations

from typing import Any

from vscode_python_issue_triage.evaluation import (
    compare_decision,
    compare_sets,
    summarize_results,
)
from vscode_python_issue_triage.models import ActualOutcome, Disposition, TriageDecision


def test_compare_sets_reports_distance() -> None:
    scores = compare_sets(("bug", "area-testing"), ("bug", "regression"))

    assert scores.precision == 0.5
    assert scores.recall == 0.5
    assert scores.f1 == 0.5
    assert scores.missing == ("regression",)
    assert scores.unexpected == ("area-testing",)


def test_compare_decision_scores_matching_empty_information_requests() -> None:
    predicted = TriageDecision(
        classification="bug",
        labels=("bug",),
        disposition=Disposition.KEEP_OPEN,
        transfer_target=None,
        information_request_ids=(),
        confidence=0.9,
        rationale="The report describes an extension failure.",
    )
    actual = ActualOutcome(
        classification="bug",
        labels=("bug",),
        disposition=Disposition.KEEP_OPEN,
        transfer_target=None,
        information_request_ids=(),
        information_request_comments=(),
    )

    comparison = compare_decision(predicted, actual)

    assert comparison.overall_score == 1
    assert comparison.information_exact


def test_transfer_is_unscored_without_historical_destination() -> None:
    predicted = TriageDecision(
        classification="bug",
        labels=("bug",),
        disposition=Disposition.TRANSFER,
        transfer_target="microsoft/debugpy",
        information_request_ids=(),
        confidence=0.9,
        rationale="The report is clearly owned by debugpy.",
    )
    actual = ActualOutcome(
        classification="bug",
        labels=("bug",),
        disposition=Disposition.KEEP_OPEN,
        transfer_target=None,
        information_request_ids=(),
        information_request_comments=(),
    )

    comparison = compare_decision(predicted, actual)

    assert comparison.disposition_match is None
    assert comparison.transfer_target_match is None
    assert comparison.overall_score == 1


def test_summary_information_metrics_only_include_relevant_issues() -> None:
    irrelevant: dict[str, Any] = {
        "prediction": {
            "classification": "bug",
            "disposition": "keep_open",
            "information_request_ids": [],
        },
        "actual": {
            "classification": "bug",
            "disposition": "keep_open",
            "information_request_ids": [],
        },
        "comparison": {
            "classification_match": True,
            "disposition_match": True,
            "labels_exact": True,
            "label_f1": 1.0,
            "information_exact": True,
            "information_f1": 1.0,
            "overall_score": 1.0,
        },
        "error": None,
    }
    relevant: dict[str, Any] = {
        "prediction": {
            "classification": "unknown",
            "disposition": "keep_open",
            "information_request_ids": [],
        },
        "actual": {
            "classification": "unknown",
            "disposition": "request_information",
            "information_request_ids": ["python_logs"],
        },
        "comparison": {
            "classification_match": True,
            "disposition_match": False,
            "labels_exact": True,
            "label_f1": 1.0,
            "information_exact": False,
            "information_f1": 0.0,
            "overall_score": 0.5,
        },
        "error": None,
    }

    summary = summarize_results([irrelevant, relevant])

    assert summary["information_evaluated"] == 1
    assert summary["information_exact_accuracy"] == 0
    assert summary["mean_information_f1"] == 0

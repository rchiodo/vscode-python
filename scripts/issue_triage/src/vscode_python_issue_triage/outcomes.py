"""Infer historical triage outcomes from labels, events, and comments."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .models import ActualOutcome, Disposition, IssueComment, IssueSnapshot

INFO_LABELS = frozenset({"info-needed", "needs more info", "~confirmation-needed", "~info-needed"})
CLASSIFICATION_LABELS = frozenset(
    {
        "*duplicate",
        "*out-of-scope",
        "*question",
        "bug",
        "feature-request",
        "important",
        "regression",
        "~spam",
    }
)

REQUEST_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "reproduction_steps": (
        re.compile(r"\bsteps? to repro(?:duce|duction)?\b", re.I),
        re.compile(r"\breproduc(?:e|ible|tion)\b", re.I),
    ),
    "expected_actual_behavior": (
        re.compile(r"\bexpected (?:behavior|behaviour|result)\b", re.I),
        re.compile(r"\bwhat (?:did|does) happen\b", re.I),
        re.compile(r"\bactual (?:behavior|behaviour|result)\b", re.I),
    ),
    "extension_version": (
        re.compile(r"\bpython extension version\b", re.I),
        re.compile(r"\bversion of (?:the )?(?:microsoft )?python extension\b", re.I),
    ),
    "vscode_version": (
        re.compile(r"\b(?:vs code|vscode) version\b", re.I),
        re.compile(r"\bhelp:\s*about\b", re.I),
    ),
    "operating_system": (
        re.compile(r"\boperating system\b", re.I),
        re.compile(r"\bos (?:version|details|information)\b", re.I),
    ),
    "python_version": (re.compile(r"\bpython (?:runtime )?version\b", re.I),),
    "environment_details": (
        re.compile(r"\binterpreter path\b", re.I),
        re.compile(r"\benvironment (?:type|details|information)\b", re.I),
        re.compile(r"\bconda|virtualenv|venv|pyenv|poetry\b", re.I),
    ),
    "python_logs": (
        re.compile(r"\bpython (?:output|log(?:ging|s)?)\b", re.I),
        re.compile(r"\bpython output (?:channel|panel)\b", re.I),
    ),
    "language_server_logs": (
        re.compile(r"\blanguage server (?:output|log(?:s)?)\b", re.I),
        re.compile(r"\bpylance (?:output|log(?:s)?)\b", re.I),
    ),
    "debugger_logs": (
        re.compile(r"\bdebug(?:ger)? (?:console|log(?:s)?)\b", re.I),
        re.compile(r"\blaunch\.json\b", re.I),
    ),
    "testing_logs": (
        re.compile(r"\btest(?:ing)? (?:output|log(?:s)?)\b", re.I),
        re.compile(r"\btest discovery\b", re.I),
    ),
    "minimal_reproduction": (
        re.compile(r"\bminimal (?:repro|reproduction|repository|project|example)\b", re.I),
        re.compile(r"\bsample (?:repository|project)\b", re.I),
    ),
    "extension_bisect": (re.compile(r"\bextension bisect\b", re.I),),
}


def infer_actual_outcome(issue: IssueSnapshot) -> ActualOutcome:
    """Normalize observable historical actions into the classifier's decision schema."""
    historical_labels = {
        event.label for event in issue.timeline if event.event == "labeled" and event.label
    }
    current_labels = set(issue.final_labels)
    observed_labels = current_labels | historical_labels
    effective_labels = current_labels | (historical_labels & {"~spam"})
    labels = tuple(sorted(label for label in effective_labels if _is_evaluated_label(label)))

    transfer_target = _infer_transfer_target(issue)
    disposition = _infer_disposition(
        effective_labels,
        observed_labels,
        transfer_target,
        issue.state_reason,
    )
    classification = _infer_classification(effective_labels)

    request_comments = _information_request_comments(issue, observed_labels)
    request_ids = tuple(
        request_id
        for request_id, patterns in REQUEST_PATTERNS.items()
        if any(pattern.search(comment) for pattern in patterns for comment in request_comments)
    )

    return ActualOutcome(
        classification=classification,
        labels=labels,
        disposition=disposition,
        transfer_target=transfer_target,
        information_request_ids=request_ids,
        information_request_comments=request_comments,
    )


def _is_evaluated_label(label: str) -> bool:
    return label.startswith("area-") or label in CLASSIFICATION_LABELS


def _infer_classification(labels: set[str]) -> str:
    if "~spam" in labels:
        return "spam"
    if "*out-of-scope" in labels:
        return "unrelated"
    if "bug" in labels:
        return "bug"
    if "feature-request" in labels:
        return "feature_request"
    if "*question" in labels:
        return "question"
    return "unknown"


def _infer_disposition(
    labels: set[str],
    observed_labels: set[str],
    transfer_target: str | None,
    state_reason: str | None,
) -> Disposition:
    if "~spam" in labels:
        return Disposition.CLOSE_SPAM
    if "*duplicate" in labels or state_reason == "duplicate":
        return Disposition.CLOSE_DUPLICATE
    if transfer_target:
        return Disposition.TRANSFER
    if "*out-of-scope" in labels:
        return Disposition.CLOSE_OUT_OF_SCOPE
    if observed_labels & INFO_LABELS:
        return Disposition.REQUEST_INFORMATION
    return Disposition.KEEP_OPEN


def _infer_transfer_target(issue: IssueSnapshot) -> str | None:
    # GitHub exposes only the repository an incoming transfer came from. It does
    # not expose a trustworthy outgoing destination on the issue timeline.
    return None


def _information_request_comments(issue: IssueSnapshot, all_labels: set[str]) -> tuple[str, ...]:
    if not all_labels & INFO_LABELS:
        return ()
    return tuple(
        comment.body
        for comment in issue.comments
        if _could_be_triager(comment, issue.author)
        and any(
            pattern.search(comment.body)
            for patterns in REQUEST_PATTERNS.values()
            for pattern in patterns
        )
    )


def _could_be_triager(comment: IssueComment, issue_author: str) -> bool:
    trusted_associations: Iterable[str] = ("COLLABORATOR", "MEMBER", "OWNER")
    return comment.author != issue_author and (
        comment.author_association in trusted_associations or comment.author.endswith("[bot]")
    )

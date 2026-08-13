from __future__ import annotations

from dataclasses import replace

from vscode_python_issue_triage.models import IssueComment, IssueSnapshot, TimelineEvent
from vscode_python_issue_triage.outcomes import infer_actual_outcome


def make_issue(
    *,
    labels: tuple[str, ...],
    events: tuple[TimelineEvent, ...] = (),
    comments: tuple[IssueComment, ...] = (),
) -> IssueSnapshot:
    return IssueSnapshot(
        repository="microsoft/vscode-python",
        number=123,
        title="Tests are not discovered",
        body="No tests appear.",
        author="reporter",
        url="https://github.com/microsoft/vscode-python/issues/123",
        created_at="2026-01-01T00:00:00+00:00",
        closed_at="2026-01-02T00:00:00+00:00",
        state_reason="not_planned",
        final_labels=labels,
        timeline=events,
        comments=comments,
    )


def test_transient_spam_label_is_ground_truth() -> None:
    issue = make_issue(
        labels=("triage-needed",),
        events=(
            TimelineEvent(
                event="labeled",
                created_at="2026-01-01T00:01:00+00:00",
                actor="bot",
                label="~spam",
            ),
            TimelineEvent(
                event="unlabeled",
                created_at="2026-01-01T00:02:00+00:00",
                actor="bot",
                label="~spam",
            ),
        ),
    )

    actual = infer_actual_outcome(issue)

    assert actual.classification == "spam"
    assert actual.disposition.value == "close_spam"
    assert actual.labels == ("~spam",)


def test_information_request_categories_are_inferred_from_triager_comment() -> None:
    issue = make_issue(
        labels=("bug", "area-testing", "info-needed"),
        comments=(
            IssueComment(
                author="maintainer",
                author_association="MEMBER",
                body=(
                    "Please provide steps to reproduce and the Python output channel logs "
                    "from test discovery."
                ),
                created_at="2026-01-01T01:00:00+00:00",
            ),
        ),
    )

    actual = infer_actual_outcome(issue)

    assert actual.disposition.value == "request_information"
    assert set(actual.information_request_ids) == {
        "python_logs",
        "reproduction_steps",
        "testing_logs",
    }


def test_removed_classification_label_is_not_ground_truth() -> None:
    issue = make_issue(
        labels=("info-needed",),
        events=(
            TimelineEvent(
                event="labeled",
                created_at="2026-01-01T00:01:00+00:00",
                actor="reporter",
                label="feature-request",
            ),
            TimelineEvent(
                event="unlabeled",
                created_at="2026-01-01T00:02:00+00:00",
                actor="maintainer",
                label="feature-request",
            ),
        ),
    )

    actual = infer_actual_outcome(issue)

    assert actual.classification == "unknown"
    assert actual.labels == ()


def test_native_duplicate_state_reason_is_ground_truth() -> None:
    issue = replace(make_issue(labels=("info-needed",)), state_reason="duplicate")

    actual = infer_actual_outcome(issue)

    assert actual.disposition.value == "close_duplicate"


def test_comment_link_is_not_treated_as_transfer() -> None:
    issue = make_issue(
        labels=("bug", "info-needed"),
        comments=(
            IssueComment(
                author="reporter",
                author_association="NONE",
                body="This looks like https://github.com/microsoft/debugpy/issues/123.",
                created_at="2026-01-01T01:00:00+00:00",
            ),
        ),
    )

    actual = infer_actual_outcome(issue)

    assert actual.transfer_target is None

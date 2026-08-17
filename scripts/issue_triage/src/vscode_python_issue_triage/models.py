"""Typed data exchanged by the issue-triage evaluator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Self, cast


class Disposition(StrEnum):
    """A first-pass action proposed for an issue."""

    KEEP_OPEN = "keep_open"
    REQUEST_INFORMATION = "request_information"
    CLOSE_SPAM = "close_spam"
    CLOSE_OUT_OF_SCOPE = "close_out_of_scope"
    CLOSE_DUPLICATE = "close_duplicate"
    TRANSFER = "transfer"


class TriageAction(StrEnum):
    """The user-facing first response to an issue."""

    CLOSE_SPAM = "close_spam"
    PROVIDE_GUIDANCE = "provide_guidance"
    ACKNOWLEDGE = "acknowledge"
    REQUEST_INFORMATION = "request_information"


CLASSIFICATIONS = frozenset({"bug", "feature_request", "question", "spam", "unrelated", "unknown"})


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array of strings")
    values = cast(list[object], value)
    items: list[str] = []
    for item in values:
        if not isinstance(item, str):
            raise ValueError(f"{field} must be an array of strings")
        items.append(item)
    return tuple(dict.fromkeys(items))


def _integer_tuple(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array of integers")
    values = cast(list[object], value)
    items: list[int] = []
    for item in values:
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"{field} must be an array of integers")
        items.append(item)
    return tuple(dict.fromkeys(items))


@dataclass(frozen=True)
class TimelineEvent:
    """A normalized GitHub issue timeline event."""

    event: str
    created_at: str | None
    actor: str | None
    label: str | None = None
    source_repository: str | None = None
    renamed_from: str | None = None


@dataclass(frozen=True)
class IssueComment:
    """A normalized GitHub issue comment."""

    author: str
    author_association: str
    body: str
    created_at: str


@dataclass(frozen=True)
class IssueSnapshot:
    """The initial report and later metadata needed to establish ground truth."""

    repository: str
    number: int
    title: str
    body: str
    author: str
    url: str
    created_at: str
    closed_at: str | None
    state_reason: str | None
    final_labels: tuple[str, ...]
    timeline: tuple[TimelineEvent, ...]
    comments: tuple[IssueComment, ...]
    content_source: str = "github_current_snapshot"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the snapshot for SQLite caching."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        """Deserialize a cached snapshot."""
        return cls(
            repository=str(value["repository"]),
            number=int(value["number"]),
            title=str(value["title"]),
            body=str(value["body"]),
            author=str(value["author"]),
            url=str(value["url"]),
            created_at=str(value["created_at"]),
            closed_at=str(value["closed_at"]) if value.get("closed_at") else None,
            state_reason=str(value["state_reason"]) if value.get("state_reason") else None,
            final_labels=tuple(str(label) for label in value["final_labels"]),
            timeline=tuple(TimelineEvent(**event) for event in value["timeline"]),
            comments=tuple(IssueComment(**comment) for comment in value["comments"]),
            content_source=str(value.get("content_source", "github_current_snapshot")),
        )


@dataclass(frozen=True)
class RetrievalEvidence:
    """One temporally eligible historical issue supplied as non-authoritative evidence."""

    issue_number: int
    issue_url: str
    title: str
    body_excerpt: str
    created_at: str
    similarity: float
    historical_classification: str
    historical_disposition: str
    historical_labels: tuple[str, ...]
    historical_information_request_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize evidence into prompts and durable result artifacts."""
        return asdict(self)


@dataclass(frozen=True)
class TriageDecision:
    """A validated decision returned by the classifier agent."""

    classification: str
    labels: tuple[str, ...]
    disposition: Disposition
    transfer_target: str | None
    information_request_ids: tuple[str, ...]
    confidence: float
    rationale: str
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize a decision."""
        value = asdict(self)
        value["disposition"] = self.disposition.value
        return value

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        *,
        allowed_labels: frozenset[str],
        allowed_request_ids: frozenset[str],
        allowed_transfer_targets: frozenset[str],
    ) -> Self:
        """Validate untrusted agent JSON."""
        classification = value.get("classification")
        if classification not in CLASSIFICATIONS:
            raise ValueError(f"Unsupported classification: {classification!r}")

        labels = _string_tuple(value.get("labels"), "labels")
        unknown_labels = set(labels) - allowed_labels
        if unknown_labels:
            raise ValueError(f"Agent returned unknown labels: {sorted(unknown_labels)}")

        try:
            disposition = Disposition(value.get("disposition"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported disposition: {value.get('disposition')!r}") from exc

        transfer_target_value = value.get("transfer_target")
        transfer_target = transfer_target_value if isinstance(transfer_target_value, str) else None
        if disposition is Disposition.TRANSFER:
            if transfer_target not in allowed_transfer_targets:
                raise ValueError(f"Unsupported transfer target: {transfer_target!r}")
        elif transfer_target is not None:
            raise ValueError("transfer_target is only valid for a transfer disposition")

        request_ids = _string_tuple(value.get("information_request_ids"), "information_request_ids")
        unknown_request_ids = set(request_ids) - allowed_request_ids
        if unknown_request_ids:
            raise ValueError(
                f"Agent returned unknown information request IDs: {sorted(unknown_request_ids)}"
            )
        if disposition is Disposition.REQUEST_INFORMATION:
            if not request_ids:
                raise ValueError("request_information requires at least one request ID")
        elif request_ids:
            raise ValueError("information_request_ids are only valid for request_information")

        confidence_value = value.get("confidence")
        if not isinstance(confidence_value, int | float) or isinstance(confidence_value, bool):
            raise ValueError("confidence must be a number")
        confidence = float(confidence_value)
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")

        rationale = value.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("rationale must be a non-empty string")

        return cls(
            classification=classification,
            labels=labels,
            disposition=disposition,
            transfer_target=transfer_target,
            information_request_ids=request_ids,
            confidence=confidence,
            rationale=rationale.strip(),
        )


@dataclass(frozen=True)
class ActionDecision:
    """A validated outcome-oriented triage decision."""

    action: TriageAction
    routing_target: str
    guidance_id: str | None
    information_request_ids: tuple[str, ...]
    supporting_issue_numbers: tuple[int, ...]
    confidence: float
    rationale: str
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize an action decision."""
        value = asdict(self)
        value["action"] = self.action.value
        return value

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        *,
        allowed_routing_targets: frozenset[str],
        allowed_guidance_ids: frozenset[str],
        allowed_request_ids: frozenset[str],
        allowed_supporting_issue_numbers: frozenset[int] = frozenset(),
    ) -> Self:
        """Validate an untrusted action response."""
        try:
            action = TriageAction(value.get("action"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported triage action: {value.get('action')!r}") from exc

        routing_target = value.get("routing_target")
        if not isinstance(routing_target, str) or routing_target not in allowed_routing_targets:
            raise ValueError(f"Unsupported routing target: {routing_target!r}")

        guidance_value = value.get("guidance_id")
        guidance_id = guidance_value if isinstance(guidance_value, str) else None
        request_ids = _string_tuple(value.get("information_request_ids"), "information_request_ids")
        unknown_request_ids = set(request_ids) - allowed_request_ids
        if unknown_request_ids:
            raise ValueError(
                f"Agent returned unknown information request IDs: {sorted(unknown_request_ids)}"
            )

        if action is TriageAction.PROVIDE_GUIDANCE:
            if guidance_id not in allowed_guidance_ids:
                raise ValueError(f"Unsupported guidance ID: {guidance_id!r}")
        elif guidance_id is not None:
            raise ValueError("guidance_id is only valid for provide_guidance")

        if action is TriageAction.REQUEST_INFORMATION:
            if not request_ids:
                raise ValueError("request_information requires at least one request ID")
        elif request_ids:
            raise ValueError("information_request_ids are only valid for request_information")

        supporting_issue_numbers = _integer_tuple(
            value.get("supporting_issue_numbers", []),
            "supporting_issue_numbers",
        )
        unknown_supporting_issues = set(supporting_issue_numbers) - allowed_supporting_issue_numbers
        if unknown_supporting_issues:
            raise ValueError(
                f"Agent cited issues that were not retrieved: {sorted(unknown_supporting_issues)}"
            )

        if action is TriageAction.CLOSE_SPAM and routing_target != "unknown":
            raise ValueError("close_spam must use the unknown routing target")

        confidence_value = value.get("confidence")
        if (
            not isinstance(confidence_value, int | float)
            or isinstance(confidence_value, bool)
            or not 0 <= float(confidence_value) <= 1
        ):
            raise ValueError("confidence must be a number between 0 and 1")
        rationale = value.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("rationale must be a non-empty string")

        return cls(
            action=action,
            routing_target=routing_target,
            guidance_id=guidance_id,
            information_request_ids=request_ids,
            supporting_issue_numbers=supporting_issue_numbers,
            confidence=float(confidence_value),
            rationale=rationale.strip(),
        )


@dataclass(frozen=True)
class ActualOutcome:
    """A normalized historical triage outcome."""

    classification: str
    labels: tuple[str, ...]
    disposition: Disposition
    transfer_target: str | None
    information_request_ids: tuple[str, ...]
    information_request_comments: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize an outcome."""
        value = asdict(self)
        value["disposition"] = self.disposition.value
        return value


@dataclass(frozen=True)
class Comparison:
    """Distance metrics between a predicted and historical decision."""

    classification_match: bool
    disposition_match: bool | None
    transfer_target_match: bool | None
    label_precision: float
    label_recall: float
    label_f1: float
    label_jaccard: float
    labels_exact: bool
    missing_labels: tuple[str, ...]
    unexpected_labels: tuple[str, ...]
    information_precision: float
    information_recall: float
    information_f1: float
    information_exact: bool
    missing_information_requests: tuple[str, ...]
    unexpected_information_requests: tuple[str, ...]
    overall_score: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize comparison metrics."""
        return asdict(self)

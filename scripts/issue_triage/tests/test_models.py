from __future__ import annotations

import pytest

from vscode_python_issue_triage.models import (
    ActionDecision,
    Disposition,
    TriageAction,
    TriageDecision,
)

ALLOWED_LABELS = frozenset({"bug", "area-testing", "info-needed"})
ALLOWED_REQUESTS = frozenset({"reproduction_steps"})
ALLOWED_TARGETS = frozenset({"microsoft/vscode"})


def test_decision_validates_agent_output() -> None:
    decision = TriageDecision.from_dict(
        {
            "classification": "bug",
            "labels": ["bug", "area-testing"],
            "disposition": "request_information",
            "transfer_target": None,
            "information_request_ids": ["reproduction_steps"],
            "confidence": 0.75,
            "rationale": "The report describes failed test discovery without reproduction steps.",
        },
        allowed_labels=ALLOWED_LABELS,
        allowed_request_ids=ALLOWED_REQUESTS,
        allowed_transfer_targets=ALLOWED_TARGETS,
    )

    assert decision.disposition is Disposition.REQUEST_INFORMATION
    assert decision.labels == ("bug", "area-testing")


def test_decision_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError, match="unknown labels"):
        TriageDecision.from_dict(
            {
                "classification": "bug",
                "labels": ["made-up"],
                "disposition": "keep_open",
                "transfer_target": None,
                "information_request_ids": [],
                "confidence": 0.5,
                "rationale": "A reason.",
            },
            allowed_labels=ALLOWED_LABELS,
            allowed_request_ids=ALLOWED_REQUESTS,
            allowed_transfer_targets=ALLOWED_TARGETS,
        )


def test_decision_rejects_boolean_confidence() -> None:
    with pytest.raises(ValueError, match="confidence must be a number"):
        TriageDecision.from_dict(
            {
                "classification": "bug",
                "labels": [],
                "disposition": "keep_open",
                "transfer_target": None,
                "information_request_ids": [],
                "confidence": True,
                "rationale": "Concrete failure.",
            },
            allowed_labels=frozenset(),
            allowed_request_ids=ALLOWED_REQUESTS,
            allowed_transfer_targets=frozenset(),
        )


def test_action_decision_validates_targeted_information_request() -> None:
    decision = ActionDecision.from_dict(
        {
            "action": "request_information",
            "routing_target": "pylance",
            "guidance_id": None,
            "information_request_ids": ["reproduction_steps"],
            "confidence": 0.6,
            "rationale": "The report may involve analysis but contains no reproduction.",
        },
        allowed_routing_targets=frozenset({"pylance", "unknown"}),
        allowed_guidance_ids=frozenset({"python-language"}),
        allowed_request_ids=ALLOWED_REQUESTS,
    )

    assert decision.action is TriageAction.REQUEST_INFORMATION
    assert decision.routing_target == "pylance"


def test_action_decision_rejects_guidance_without_catalog_id() -> None:
    with pytest.raises(ValueError, match="guidance ID"):
        ActionDecision.from_dict(
            {
                "action": "provide_guidance",
                "routing_target": "python",
                "guidance_id": None,
                "information_request_ids": [],
                "confidence": 0.8,
                "rationale": "Python behavior.",
            },
            allowed_routing_targets=frozenset({"python"}),
            allowed_guidance_ids=frozenset({"python-language"}),
            allowed_request_ids=ALLOWED_REQUESTS,
        )


def test_action_decision_rejects_issue_that_was_not_retrieved() -> None:
    with pytest.raises(ValueError, match="not retrieved"):
        ActionDecision.from_dict(
            {
                "action": "acknowledge",
                "routing_target": "vscode-python",
                "guidance_id": None,
                "information_request_ids": [],
                "supporting_issue_numbers": [999],
                "confidence": 0.8,
                "rationale": "This resembles a prior extension issue.",
            },
            allowed_routing_targets=frozenset({"vscode-python"}),
            allowed_guidance_ids=frozenset(),
            allowed_request_ids=ALLOWED_REQUESTS,
            allowed_supporting_issue_numbers=frozenset({123}),
        )

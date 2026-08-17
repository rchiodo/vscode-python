from __future__ import annotations

from vscode_python_issue_triage.action_triage import (
    render_action_response,
    summarize_action_results,
)
from vscode_python_issue_triage.configuration import Guidance, InformationRequest
from vscode_python_issue_triage.models import ActionDecision, RetrievalEvidence, TriageAction

REQUESTS = (
    InformationRequest(
        id="reproduction_steps",
        description="Missing steps",
        message="Provide reproduction steps.",
    ),
)
GUIDANCE = (
    Guidance(
        id="python-language",
        description="Python semantics",
        message="Read the Python language reference.",
    ),
)


def _decision(
    action: TriageAction,
    *,
    route: str = "vscode-python",
    guidance_id: str | None = None,
    request_ids: tuple[str, ...] = (),
) -> ActionDecision:
    return ActionDecision(
        action=action,
        routing_target=route,
        guidance_id=guidance_id,
        information_request_ids=request_ids,
        supporting_issue_numbers=(),
        confidence=0.8,
        rationale="Test rationale.",
    )


def test_render_action_response_explains_acknowledgement_decision() -> None:
    response = render_action_response(
        _decision(TriageAction.ACKNOWLEDGE),
        information_requests=REQUESTS,
        guidance=GUIDANCE,
    )

    assert response == (
        "Thanks for reporting this issue. Our initial automated triage routed this to "
        "**VS Code Python extension**.\n\n"
        "Test rationale.\n\n"
        "Someone from the team will look into it."
    )


def test_render_action_response_uses_readable_component_name() -> None:
    response = render_action_response(
        _decision(TriageAction.ACKNOWLEDGE, route="pylance"),
        information_requests=REQUESTS,
        guidance=GUIDANCE,
    )

    assert response is not None
    assert "**Pylance**" in response


def test_render_action_response_links_cited_retrieval_evidence() -> None:
    evidence = RetrievalEvidence(
        issue_number=123,
        issue_url="https://github.com/microsoft/vscode-python/issues/123",
        title="Similar issue",
        body_excerpt="Similar report",
        created_at="2020-01-01T00:00:00+00:00",
        similarity=0.8,
        historical_classification="bug",
        historical_disposition="keep_open",
        historical_labels=("bug",),
        historical_information_request_ids=(),
    )
    decision = ActionDecision(
        action=TriageAction.ACKNOWLEDGE,
        routing_target="vscode-python",
        guidance_id=None,
        information_request_ids=(),
        supporting_issue_numbers=(123,),
        confidence=0.8,
        rationale="This matches an earlier extension report.",
    )

    response = render_action_response(
        decision,
        information_requests=REQUESTS,
        guidance=GUIDANCE,
        retrieved_issues=(evidence,),
    )

    assert response is not None
    assert "[#123](https://github.com/microsoft/vscode-python/issues/123)" in response


def test_render_action_response_uses_guidance_catalog() -> None:
    response = render_action_response(
        _decision(
            TriageAction.PROVIDE_GUIDANCE,
            route="python",
            guidance_id="python-language",
        ),
        information_requests=REQUESTS,
        guidance=GUIDANCE,
    )

    assert response == "Read the Python language reference."


def test_action_summary_reports_distributions_without_historical_accuracy() -> None:
    rows = (
        {
            "decision": _decision(TriageAction.ACKNOWLEDGE).to_dict(),
            "error": None,
            "issue_number": 1,
        },
        {
            "decision": _decision(
                TriageAction.REQUEST_INFORMATION,
                route="unknown",
                request_ids=("reproduction_steps",),
            ).to_dict(),
            "error": None,
            "issue_number": 2,
        },
    )

    summary = summarize_action_results(
        rows=rows,
        cohort_size=2,
        model="test-model",
        seed=42,
    )

    assert summary["action_distribution"] == {
        "acknowledge": 1,
        "request_information": 1,
    }
    assert "accuracy" not in summary

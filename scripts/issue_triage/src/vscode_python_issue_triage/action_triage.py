"""Outcome-oriented, read-only issue triage over a fixed local cohort."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

from .classifier import CopilotClassifier, RecoverableClassificationError
from .configuration import (
    ACTION_TRIAGE_SKILLS,
    Guidance,
    InformationRequest,
    classification_skills_directory,
    load_agent_definition,
    load_guidance,
    load_information_requests,
    project_root,
    render_information_request,
)
from .ml import MLRecord, load_dataset
from .models import ActionDecision, IssueSnapshot, TriageAction

ROUTING_TARGET_NAMES = {
    "vscode-python": "VS Code Python extension",
    "pylance": "Pylance",
    "python-environments": "Python Environments",
    "linting": "Python linting integration",
    "python": "Python",
    "third-party": "a third-party project",
    "unknown": "an unknown component",
}


async def run_action_triage(
    *,
    dataset_path: Path,
    output_directory: Path,
    model: str,
    cohort_size: int,
    seed: int,
    concurrency: int,
    timeout_seconds: float,
    retries: int,
    agent_path: Path,
    information_requests_path: Path,
    guidance_path: Path,
) -> dict[str, Any]:
    """Generate resumable action decisions for a seeded local cohort."""
    records = load_dataset(dataset_path)
    ensure_directory(output_directory)
    cohort = load_or_create_action_cohort(
        records=records,
        dataset_path=dataset_path,
        output_directory=output_directory,
        cohort_size=cohort_size,
        seed=seed,
    )
    information_requests = load_information_requests(information_requests_path)
    guidance = load_guidance(guidance_path)
    agent = load_agent_definition(agent_path)
    results_path = output_directory / "action-results.jsonl"
    configuration = build_action_run_configuration(
        dataset_path=dataset_path,
        output_directory=output_directory,
        model=model,
        concurrency=concurrency,
        timeout_seconds=timeout_seconds,
        retries=retries,
        agent_path=agent_path,
        information_requests_path=information_requests_path,
        guidance_path=guidance_path,
    )
    validate_or_create_action_manifest(
        output_directory=output_directory,
        configuration=configuration,
        results_exist=results_path.exists(),
    )
    repair_trailing_jsonl(results_path)
    existing = load_action_results(results_path)
    pending = [
        record
        for record in cohort
        if record.issue_number not in existing
        or existing[record.issue_number].get("error") is not None
    ]
    classifier = CopilotClassifier(
        repository_root=project_root(),
        agent=agent,
        model=model,
        timeout=timeout_seconds,
        retries=retries,
        skill_names=ACTION_TRIAGE_SKILLS,
    )
    completed = len(existing)
    async with classifier:
        for start in range(0, len(pending), concurrency):
            rows = await asyncio.gather(
                *(
                    classify_action_record(
                        classifier=classifier,
                        record=record,
                        information_requests=information_requests,
                        guidance=guidance,
                    )
                    for record in pending[start : start + concurrency]
                )
            )
            append_synced_rows(results_path, rows)
            completed += len(rows)
            if completed % 25 < len(rows) or completed == len(cohort):
                print(f"Completed {completed}/{len(cohort)} action decisions.", flush=True)

    summary = summarize_action_results(
        rows=tuple(load_action_results(results_path).values()),
        cohort_size=len(cohort),
        model=model,
        seed=seed,
    )
    write_json_atomic(output_directory / "action-summary.json", summary)
    return summary


async def classify_action_record(
    *,
    classifier: CopilotClassifier,
    record: MLRecord,
    information_requests: tuple[InformationRequest, ...],
    guidance: tuple[Guidance, ...],
) -> dict[str, Any]:
    """Classify and render one local record without mutating GitHub."""
    issue = IssueSnapshot(
        repository=record.repository,
        number=record.issue_number,
        title=record.title,
        body=record.body,
        author="historical-reporter",
        url=record.url,
        created_at=record.created_at,
        closed_at=None,
        state_reason=None,
        final_labels=(),
        timeline=(),
        comments=(),
        content_source=record.content_source,
    )
    try:
        decision = await classifier.classify_action(
            issue,
            information_requests=information_requests,
            guidance=guidance,
        )
        return {
            "decision": decision.to_dict(),
            "error": None,
            "issue_number": issue.number,
            "issue_url": issue.url,
            "response": render_action_response(
                decision,
                information_requests=information_requests,
                guidance=guidance,
            ),
            "title": issue.title,
        }
    except RecoverableClassificationError as exc:
        return {
            "decision": None,
            "error": f"{type(exc).__name__}: {exc}",
            "issue_number": issue.number,
            "issue_url": issue.url,
            "response": None,
            "title": issue.title,
        }


def render_action_response(
    decision: ActionDecision,
    *,
    information_requests: tuple[InformationRequest, ...],
    guidance: tuple[Guidance, ...],
) -> str | None:
    """Render the exact proposed first response for review."""
    if decision.action is TriageAction.CLOSE_SPAM:
        return None
    if decision.action is TriageAction.ACKNOWLEDGE:
        routing_target = ROUTING_TARGET_NAMES[decision.routing_target]
        return (
            "Thanks for reporting this issue. Our initial automated triage routed this to "
            f"**{routing_target}**.\n\n{decision.rationale}\n\n"
            "Someone from the team will look into it."
        )
    if decision.action is TriageAction.REQUEST_INFORMATION:
        return render_information_request(decision.information_request_ids, information_requests)
    by_id = {entry.id: entry for entry in guidance}
    if decision.guidance_id is None:
        raise ValueError("provide_guidance decision does not contain a guidance ID")
    return by_id[decision.guidance_id].message


def load_or_create_action_cohort(
    *,
    records: tuple[MLRecord, ...],
    dataset_path: Path,
    output_directory: Path,
    cohort_size: int,
    seed: int,
) -> tuple[MLRecord, ...]:
    """Persist and reload one immutable random cohort."""
    if cohort_size > len(records):
        raise ValueError(f"--cohort-size cannot exceed dataset size {len(records)}")
    path = output_directory / "action-cohort.json"
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    by_number = {record.issue_number: record for record in records}
    if path.exists():
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError(f"Action cohort must be a JSON object: {path}")
        value = cast(dict[object, object], decoded)
        if (
            value.get("dataset_sha256") != dataset_sha256
            or value.get("cohort_size") != cohort_size
            or value.get("seed") != seed
        ):
            raise ValueError(f"Existing action cohort configuration differs: {path}")
        numbers = value.get("issue_numbers")
        if not isinstance(numbers, list) or not all(
            isinstance(number, int) and not isinstance(number, bool)
            for number in cast(list[object], numbers)
        ):
            raise ValueError(f"Action cohort issue_numbers must be integers: {path}")
        return tuple(by_number[cast(int, number)] for number in cast(list[object], numbers))

    cohort = tuple(random.Random(seed).sample(records, cohort_size))
    write_json_atomic(
        path,
        {
            "cohort_size": cohort_size,
            "dataset_sha256": dataset_sha256,
            "issue_numbers": [record.issue_number for record in cohort],
            "seed": seed,
        },
    )
    return cohort


def build_action_run_configuration(
    *,
    dataset_path: Path,
    output_directory: Path,
    model: str,
    concurrency: int,
    timeout_seconds: float,
    retries: int,
    agent_path: Path,
    information_requests_path: Path,
    guidance_path: Path,
) -> dict[str, Any]:
    """Bind all behavior-affecting inputs for safe resume."""
    implementation = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        implementation.update(path.name.encode("utf-8"))
        implementation.update(b"\0")
        implementation.update(path.read_bytes())
        implementation.update(b"\0")
    return {
        "agent_sha256": file_sha256(agent_path),
        "cohort_sha256": file_sha256(output_directory / "action-cohort.json"),
        "concurrency": concurrency,
        "dataset_sha256": file_sha256(dataset_path),
        "github_copilot_sdk_version": version("github-copilot-sdk"),
        "guidance_sha256": file_sha256(guidance_path),
        "implementation_sha256": implementation.hexdigest(),
        "information_requests_sha256": file_sha256(information_requests_path),
        "model": model,
        "retries": retries,
        "schema_version": 1,
        "skills": {
            name: file_sha256(classification_skills_directory() / name / "SKILL.md")
            for name in ACTION_TRIAGE_SKILLS
        },
        "timeout_seconds": timeout_seconds,
    }


def validate_or_create_action_manifest(
    *,
    output_directory: Path,
    configuration: dict[str, Any],
    results_exist: bool,
) -> None:
    """Reject resumes with different effective behavior."""
    path = output_directory / "run-manifest.json"
    if path.exists():
        existing: object = json.loads(path.read_text(encoding="utf-8"))
        if existing != configuration:
            raise ValueError(f"Existing action-triage run configuration differs: {path}")
        return
    if results_exist:
        raise ValueError(f"Cannot safely resume action results without a manifest: {path}")
    write_json_atomic(path, configuration)


def load_action_results(path: Path) -> dict[int, dict[str, Any]]:
    """Load the newest durable result for every issue."""
    results: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return results
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            decoded: object = json.loads(line)
            if not isinstance(decoded, dict):
                raise ValueError(f"Action result {path}:{line_number} must be an object")
            row = cast(dict[str, Any], decoded)
            results[int(row["issue_number"])] = row
    return results


def summarize_action_results(
    *,
    rows: tuple[dict[str, Any], ...],
    cohort_size: int,
    model: str,
    seed: int,
) -> dict[str, Any]:
    """Summarize predictions without pretending weak historical labels are truth."""
    completed = [row for row in rows if row.get("error") is None]
    actions = Counter(str(cast(dict[str, Any], row["decision"])["action"]) for row in completed)
    routes = Counter(
        str(cast(dict[str, Any], row["decision"])["routing_target"]) for row in completed
    )
    return {
        "action_distribution": dict(sorted(actions.items())),
        "cohort_size": cohort_size,
        "completed": len(completed),
        "errors": cohort_size - len(completed),
        "model": model,
        "routing_distribution": dict(sorted(routes.items())),
        "seed": seed,
    }


def append_synced_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append one completed batch before reporting progress."""
    with path.open("a", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace one JSON artifact."""
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def repair_trailing_jsonl(path: Path) -> None:
    """Discard an interrupted final JSONL record before resume."""
    if not path.exists() or path.stat().st_size == 0:
        return
    content = path.read_bytes()
    if content.endswith(b"\n"):
        return
    with path.open("r+b") as output:
        output.truncate(content.rfind(b"\n") + 1)
        output.flush()
        os.fsync(output.fileno())


def file_sha256(path: Path) -> str:
    """Hash one provenance input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_directory(path: Path) -> None:
    """Create an artifact directory outside the async I/O boundary."""
    path.mkdir(parents=True, exist_ok=True)

"""Classification-only LLM prompt benchmark over a fixed local issue cohort."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

from .classifier import CopilotClassifier, RecoverableClassificationError
from .configuration import (
    AgentDefinition,
    project_root,
)
from .ml import MLRecord, load_dataset
from .models import CLASSIFICATIONS


@dataclass(frozen=True)
class PromptVariant:
    """One classification prompt under comparison."""

    name: str
    instructions: str


PROMPT_VARIANTS = (
    PromptVariant(
        name="minimal",
        instructions="""You classify the primary type of vscode-python issue reports.

Choose exactly one:
- bug: existing behavior appears broken or incorrect
- feature_request: asks for new or changed behavior
- question: primarily asks how to use or configure something
- spam: advertising, abuse, nonsense, or no plausible software report
- unrelated: understandable content unrelated to Python development in VS Code
- unknown: the initial report is too ambiguous to choose another class

Return exactly one raw JSON object:
{"classification":"bug|feature_request|question|spam|unrelated|unknown",
"confidence":0.0,"rationale":"one concise sentence"}""",
    ),
    PromptVariant(
        name="repository-aware",
        instructions="""You predict the primary historical triage classification for an issue
filed in microsoft/vscode-python, using only its initial title and body.

vscode-python covers interpreter selection, environment activation, Python terminals, testing
integration, debugger entry points, and Python extension settings. Reports about Pylance,
Jupyter, VS Code core, debugpy, or Python Environments can still describe a bug or feature
request; do not call them unrelated merely because another Microsoft repository may own them.

Choose exactly one:
- bug: a concrete failure, regression, crash, incorrect result, or existing behavior not working
- feature_request: a request to add, enhance, or intentionally change behavior
- question: a usage or configuration question without a concrete product defect
- spam: unmistakable advertising, abuse, gibberish, or content with no plausible software intent
- unrelated: understandable content with no meaningful connection to Python development in VS Code
- unknown: insufficient or contradictory initial information prevents the above distinctions

Prefer bug over unknown when a concrete failure is described, even if reproduction details are
missing. Prefer feature_request when the desired behavior does not appear to exist. Use unknown
only as a genuine last resort.

Return exactly one raw JSON object:
{"classification":"bug|feature_request|question|spam|unrelated|unknown",
"confidence":0.0,"rationale":"one concise sentence"}""",
    ),
    PromptVariant(
        name="few-shot",
        instructions="""You predict the primary historical triage classification for an issue
filed in microsoft/vscode-python. Use only the initial title and body.

Classes: bug, feature_request, question, spam, unrelated, unknown.

Examples:
- "Pytest discovery crashes with TypeError after upgrading" -> bug
- "Add a command to copy the selected interpreter path" -> feature_request
- "How can I make the extension use my virtual environment?" -> question
- "Earn money fast at this link" -> spam
- "My printer leaves streaks on photographs" -> unrelated
- "Python issue" with no explanatory body -> unknown
- "Debugger ignores breakpoints in async code" -> bug
- "Support sorting environments by Python version" -> feature_request

A concrete malfunction is a bug even when details are missing or another Microsoft Python
component may ultimately own it. Reserve unknown for reports whose intent or failure cannot be
determined.

Return exactly one raw JSON object:
{"classification":"bug|feature_request|question|spam|unrelated|unknown",
"confidence":0.0,"rationale":"one concise sentence"}""",
    ),
)


async def run_classification_benchmark(
    *,
    dataset_path: Path,
    output_directory: Path,
    model: str,
    cohort_size: int,
    seed: int,
    concurrency: int,
    timeout_seconds: float,
    retries: int,
) -> dict[str, Any]:
    """Evaluate all prompt variants on one seeded random cohort."""
    records = load_dataset(dataset_path)
    cohort = load_or_create_cohort(
        records=records,
        dataset_path=dataset_path,
        output_directory=output_directory,
        cohort_size=cohort_size,
        seed=seed,
    )
    _ensure_directory(output_directory)
    results_path = output_directory / "classification-results.jsonl"
    run_configuration = build_run_configuration(
        dataset_path=dataset_path,
        output_directory=output_directory,
        model=model,
        concurrency=concurrency,
        timeout_seconds=timeout_seconds,
        retries=retries,
    )
    validate_or_create_run_manifest(
        output_directory=output_directory,
        configuration=run_configuration,
        results_exist=results_path.exists(),
    )
    _repair_trailing_line(results_path)
    existing = _load_results(results_path)
    pending = [
        (variant, record)
        for variant in PROMPT_VARIANTS
        for record in cohort
        if (variant.name, record.issue_number) not in existing
        or existing[(variant.name, record.issue_number)].get("error") is not None
    ]

    agent = AgentDefinition(
        name="python-issue-classifier",
        prompt=PROMPT_VARIANTS[0].instructions,
    )
    classifier = CopilotClassifier(
        repository_root=project_root(),
        agent=agent,
        model=model,
        timeout=timeout_seconds,
        retries=retries,
    )
    lock = asyncio.Lock()
    completed = len(existing)
    total = len(PROMPT_VARIANTS) * len(cohort)

    async with classifier:
        for start in range(0, len(pending), concurrency):
            batch = pending[start : start + concurrency]
            rows = await asyncio.gather(
                *(
                    _classify_one(
                        classifier=classifier,
                        variant=variant,
                        record=record,
                    )
                    for variant, record in batch
                )
            )
            async with lock:
                with results_path.open("a", encoding="utf-8", newline="\n") as output:
                    for row in rows:
                        output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    output.flush()
                    os.fsync(output.fileno())
            completed += len(rows)
            if completed % 25 < len(rows) or completed == total:
                print(f"Completed {completed}/{total} LLM classifications.", flush=True)

    summary = summarize_results(
        cohort=cohort,
        rows=tuple(_load_results(results_path).values()),
        model=model,
        seed=seed,
    )
    summary_path = output_directory / "classification-summary.json"
    temporary_path = summary_path.with_name(f"{summary_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(summary_path)
    return summary


async def _classify_one(
    *,
    classifier: CopilotClassifier,
    variant: PromptVariant,
    record: MLRecord,
) -> dict[str, Any]:
    try:
        prediction = await classifier.classify_label(
            title=record.title,
            body=record.body,
            instructions=variant.instructions,
        )
        return {
            "actual": record.classification,
            "confidence": prediction.confidence,
            "error": None,
            "issue_number": record.issue_number,
            "model": prediction.model,
            "predicted": prediction.classification,
            "prompt_sha256": hashlib.sha256(variant.instructions.encode("utf-8")).hexdigest(),
            "rationale": prediction.rationale,
            "variant": variant.name,
        }
    except RecoverableClassificationError as exc:
        return {
            "actual": record.classification,
            "confidence": None,
            "error": f"{type(exc).__name__}: {exc}",
            "issue_number": record.issue_number,
            "model": None,
            "predicted": None,
            "prompt_sha256": hashlib.sha256(variant.instructions.encode("utf-8")).hexdigest(),
            "rationale": None,
            "variant": variant.name,
        }


def load_or_create_cohort(
    *,
    records: tuple[MLRecord, ...],
    dataset_path: Path,
    output_directory: Path,
    cohort_size: int,
    seed: int,
) -> tuple[MLRecord, ...]:
    if cohort_size > len(records):
        raise ValueError(f"--cohort-size cannot exceed dataset size {len(records)}")
    cohort_path = output_directory / "classification-cohort.json"
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    by_number = {record.issue_number: record for record in records}
    if cohort_path.exists():
        decoded: object = json.loads(cohort_path.read_text(encoding="utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError(f"Cohort file must be a JSON object: {cohort_path}")
        value = cast(dict[object, object], decoded)
        if (
            value.get("dataset_sha256") != dataset_sha256
            or value.get("seed") != seed
            or value.get("cohort_size") != cohort_size
        ):
            raise ValueError(
                f"Existing cohort configuration differs: remove {cohort_path} "
                "or use the original dataset, seed, and cohort size"
            )
        issue_numbers = value.get("issue_numbers")
        if not isinstance(issue_numbers, list):
            raise ValueError(f"Cohort issue_numbers must be an array: {cohort_path}")
        typed_numbers = cast(list[object], issue_numbers)
        if not all(
            isinstance(number, int) and not isinstance(number, bool) for number in typed_numbers
        ):
            raise ValueError(f"Cohort issue_numbers must contain only integers: {cohort_path}")
        return tuple(by_number[cast(int, number)] for number in typed_numbers)

    cohort = tuple(random.Random(seed).sample(records, cohort_size))
    output_directory.mkdir(parents=True, exist_ok=True)
    cohort_path.write_text(
        json.dumps(
            {
                "cohort_size": cohort_size,
                "dataset_sha256": dataset_sha256,
                "issue_numbers": [record.issue_number for record in cohort],
                "seed": seed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return cohort


def _load_results(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    results: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.exists():
        return results
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            decoded: object = json.loads(line)
            if not isinstance(decoded, dict):
                raise ValueError(f"Result {path}:{line_number} must be a JSON object")
            row = cast(dict[str, Any], decoded)
            results[(str(row["variant"]), int(row["issue_number"]))] = row
    return results


def summarize_results(
    *,
    cohort: tuple[MLRecord, ...],
    rows: tuple[dict[str, Any], ...],
    model: str,
    seed: int,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for variant in PROMPT_VARIANTS:
        variant_rows = [row for row in rows if row["variant"] == variant.name]
        completed = [row for row in variant_rows if row["error"] is None]
        confusion = {
            actual: {
                predicted: sum(
                    row["actual"] == actual and row["predicted"] == predicted for row in completed
                )
                for predicted in sorted(CLASSIFICATIONS)
            }
            for actual in sorted(CLASSIFICATIONS)
        }
        per_class = _per_class_metrics(completed)
        supported_metrics = [metric for metric in per_class.values() if int(metric["support"]) > 0]
        reports[variant.name] = {
            "accuracy": (
                sum(row["actual"] == row["predicted"] for row in completed) / len(completed)
                if completed
                else None
            ),
            "completed": len(completed),
            "confusion": confusion,
            "errors": len(cohort) - len(completed),
            "macro_f1": _mean(float(metric["f1"]) for metric in supported_metrics),
            "mean_confidence": _mean(float(row["confidence"]) for row in completed),
            "per_class": per_class,
            "prompt_sha256": hashlib.sha256(variant.instructions.encode("utf-8")).hexdigest(),
        }
    return {
        "cohort_classification_distribution": _counts(record.classification for record in cohort),
        "cohort_size": len(cohort),
        "model": model,
        "seed": seed,
        "variants": reports,
    }


def _per_class_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    reports: dict[str, dict[str, float | int]] = {}
    for classification in sorted(CLASSIFICATIONS):
        true_positive = sum(
            row["actual"] == classification and row["predicted"] == classification for row in rows
        )
        false_positive = sum(
            row["actual"] != classification and row["predicted"] == classification for row in rows
        )
        false_negative = sum(
            row["actual"] == classification and row["predicted"] != classification for row in rows
        )
        support = true_positive + false_negative
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        reports[classification] = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "support": support,
        }
    return reports


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts = dict.fromkeys(sorted(CLASSIFICATIONS), 0)
    for value in values:
        counts[value] += 1
    return counts


def _mean(values: Iterable[float]) -> float:
    collected = tuple(values)
    return sum(collected) / len(collected) if collected else 0.0


def _repair_trailing_line(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    content = path.read_bytes()
    if content.endswith(b"\n"):
        return
    final_newline = content.rfind(b"\n")
    with path.open("r+b") as output:
        output.truncate(final_newline + 1)
        output.flush()
        os.fsync(output.fileno())


def validate_or_create_run_manifest(
    *,
    output_directory: Path,
    configuration: dict[str, Any],
    results_exist: bool,
) -> None:
    """Bind resumable predictions to one immutable model, prompt set, and cohort."""
    manifest_path = output_directory / "run-manifest.json"
    if manifest_path.exists():
        existing: object = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != configuration:
            raise ValueError(
                f"Existing LLM run configuration differs: use another --output-directory "
                f"or remove {manifest_path.parent}"
            )
        return
    if results_exist:
        raise ValueError(f"Cannot safely resume results without a run manifest: {manifest_path}")
    temporary_path = manifest_path.with_name(f"{manifest_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(configuration, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    temporary_path.replace(manifest_path)


def build_run_configuration(
    *,
    dataset_path: Path,
    output_directory: Path,
    model: str,
    concurrency: int,
    timeout_seconds: float,
    retries: int,
) -> dict[str, Any]:
    """Describe every input that must remain fixed when resuming a benchmark."""
    implementation_digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        implementation_digest.update(path.name.encode("utf-8"))
        implementation_digest.update(b"\0")
        implementation_digest.update(path.read_bytes())
        implementation_digest.update(b"\0")
    return {
        "cohort_sha256": hashlib.sha256(
            (output_directory / "classification-cohort.json").read_bytes()
        ).hexdigest(),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "github_copilot_sdk_version": version("github-copilot-sdk"),
        "implementation_sha256": implementation_digest.hexdigest(),
        "model": model,
        "runtime": {
            "concurrency": concurrency,
            "retries": retries,
            "timeout_seconds": timeout_seconds,
        },
        "prompts": {
            variant.name: hashlib.sha256(variant.instructions.encode("utf-8")).hexdigest()
            for variant in PROMPT_VARIANTS
        },
        "schema_version": 1,
    }


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

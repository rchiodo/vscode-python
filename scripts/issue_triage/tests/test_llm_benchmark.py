from __future__ import annotations

import hashlib
from pathlib import Path

from pytest import raises

from vscode_python_issue_triage.llm_benchmark import (
    PROMPT_VARIANTS,
    build_run_configuration,
    load_or_create_cohort,
    summarize_results,
    validate_or_create_run_manifest,
)
from vscode_python_issue_triage.ml import MLRecord, write_dataset


def _records(count: int = 20) -> tuple[MLRecord, ...]:
    classifications = ("bug", "feature_request", "unknown")
    return tuple(
        MLRecord(
            repository="microsoft/vscode-python",
            issue_number=index,
            title=f"Issue {index}",
            body="Initial report",
            url=f"https://github.com/microsoft/vscode-python/issues/{index}",
            created_at=f"2024-01-{index + 1:02d}T00:00:00+00:00",
            content_source="github_current_snapshot",
            classification=classifications[index % len(classifications)],
            disposition="keep_open",
            labels=(),
            information_request_ids=(),
        )
        for index in range(count)
    )


def test_random_cohort_is_persisted_and_reused(tmp_path: Path) -> None:
    records = _records()
    dataset = tmp_path / "dataset.jsonl"
    output = tmp_path / "llm"
    write_dataset(dataset, list(records))

    first = load_or_create_cohort(
        records=records,
        dataset_path=dataset,
        output_directory=output,
        cohort_size=10,
        seed=42,
    )
    second = load_or_create_cohort(
        records=tuple(reversed(records)),
        dataset_path=dataset,
        output_directory=output,
        cohort_size=10,
        seed=42,
    )

    assert [record.issue_number for record in first] == [record.issue_number for record in second]
    assert len({record.issue_number for record in first}) == 10


def test_summary_reports_accuracy_and_supported_macro_f1() -> None:
    cohort = _records(3)
    rows = tuple(
        {
            "actual": record.classification,
            "confidence": 0.8,
            "error": None,
            "issue_number": record.issue_number,
            "model": "test-model",
            "predicted": record.classification,
            "rationale": "Test",
            "variant": variant.name,
        }
        for variant in PROMPT_VARIANTS
        for record in cohort
    )

    summary = summarize_results(cohort=cohort, rows=rows, model="test-model", seed=42)

    assert summary["variants"]["minimal"]["accuracy"] == 1
    assert summary["variants"]["minimal"]["macro_f1"] == 1
    assert (
        summary["variants"]["minimal"]["prompt_sha256"]
        == hashlib.sha256(PROMPT_VARIANTS[0].instructions.encode("utf-8")).hexdigest()
    )


def test_summary_counts_missing_predictions_as_errors() -> None:
    summary = summarize_results(cohort=_records(3), rows=(), model="test-model", seed=42)

    assert summary["variants"]["minimal"]["accuracy"] is None
    assert summary["variants"]["minimal"]["errors"] == 3


def test_run_manifest_rejects_changed_configuration(tmp_path: Path) -> None:
    configuration = {"model": "gpt-5.4", "schema_version": 1}
    validate_or_create_run_manifest(
        output_directory=tmp_path,
        configuration=configuration,
        results_exist=False,
    )

    with raises(ValueError, match="configuration differs"):
        validate_or_create_run_manifest(
            output_directory=tmp_path,
            configuration={"model": "another-model", "schema_version": 1},
            results_exist=True,
        )


def test_run_configuration_records_runtime_settings(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    output = tmp_path / "llm"
    write_dataset(dataset, list(_records()))
    load_or_create_cohort(
        records=_records(),
        dataset_path=dataset,
        output_directory=output,
        cohort_size=10,
        seed=42,
    )

    configuration = build_run_configuration(
        dataset_path=dataset,
        output_directory=output,
        model="gpt-5.4",
        concurrency=3,
        timeout_seconds=90,
        retries=2,
    )

    assert configuration["runtime"] == {
        "concurrency": 3,
        "retries": 2,
        "timeout_seconds": 90,
    }

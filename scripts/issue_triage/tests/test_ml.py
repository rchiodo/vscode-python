from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pytest import MonkeyPatch, raises

import vscode_python_issue_triage.ml as ml_module
from vscode_python_issue_triage.ml import (
    MLRecord,
    chronological_split,
    collect_dataset_bulk,
    load_dataset,
    run_experiment,
    write_dataset,
)
from vscode_python_issue_triage.models import IssueSnapshot


def make_records(count: int = 30) -> tuple[MLRecord, ...]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    records: list[MLRecord] = []
    for index in range(count):
        is_bug = index % 2 == 0
        needs_information = index % 3 == 0
        records.append(
            MLRecord(
                repository="microsoft/vscode-python",
                issue_number=1000 + index,
                title=("Test discovery crashes" if is_bug else "Request a new environment command"),
                body=(
                    "pytest discovery fails with an exception"
                    if is_bug
                    else "Please add an environment selection feature"
                ),
                url=f"https://github.com/microsoft/vscode-python/issues/{1000 + index}",
                created_at=(start + timedelta(days=index)).isoformat(),
                content_source="github_current_snapshot",
                classification="bug" if is_bug else "feature_request",
                disposition=("request_information" if needs_information else "keep_open"),
                labels=(
                    ("area-testing", "bug") if is_bug else ("area-environments", "feature-request")
                ),
                information_request_ids=(("reproduction_steps",) if needs_information else ()),
            )
        )
    return tuple(records)


def test_chronological_split_keeps_newest_issues_in_holdout() -> None:
    records = tuple(reversed(make_records()))

    split = chronological_split(records)

    assert len(split.train) == 21
    assert len(split.tuning) == 4
    assert len(split.holdout) == 5
    assert split.train[-1].created_at < split.tuning[0].created_at
    assert split.tuning[-1].created_at < split.holdout[0].created_at


def test_dataset_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    records = make_records(12)

    write_dataset(path, list(records))

    assert load_dataset(path) == records
    assert path.with_suffix(".manifest.json").exists()


def test_dataset_rejects_content_that_does_not_match_manifest(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    write_dataset(path, list(make_records(12)))
    path.write_text(path.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")

    with raises(ValueError, match="manifest SHA-256"):
        load_dataset(path)


def test_tfidf_experiment_writes_model_metrics_and_predictions(tmp_path: Path) -> None:
    report = run_experiment(
        records=make_records(),
        approach="tfidf",
        output_directory=tmp_path,
        embedding_model="unused",
    )

    assert report["split"] == {"train": 21, "tuning": 4, "holdout": 5}
    assert 0 <= report["holdout"]["overall_score"] <= 1
    assert report["holdout"]["classification"]["accuracy"] == 1
    assert (tmp_path / "tfidf.model.joblib").exists()
    assert (tmp_path / "tfidf.metrics.json").exists()
    assert len((tmp_path / "tfidf.predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 5


class _CheckpointSource:
    def __init__(self, *, allow_download: bool) -> None:
        self.allow_download = allow_download
        self.download_calls = 0

    def iter_closed_issue_pages(
        self, *, start_cursor: str | None = None, completed_pages: int = 0
    ) -> Any:
        self._record_download()
        issue = SimpleNamespace(number=7, _rawData={"number": 7})
        yield None, (issue,)

    def iter_issue_event_pages(
        self, *, start_cursor: str | None = None, completed_pages: int = 0
    ) -> Any:
        self._record_download()
        return
        yield start_cursor, ()

    def iter_issue_comment_pages(
        self, *, start_cursor: str | None = None, completed_pages: int = 0
    ) -> Any:
        self._record_download()
        return
        yield start_cursor, ()

    def iter_issue_timeline(self, _issue_number: int) -> Any:
        self._record_download()
        return iter(())

    def build_snapshot(
        self,
        issue: Any,
        *,
        timeline: tuple[Any, ...],
        comments: tuple[Any, ...],
    ) -> IssueSnapshot:
        return IssueSnapshot(
            repository="microsoft/vscode-python",
            number=int(issue.number),
            title="Discovery fails",
            body="pytest discovery crashes",
            author="reporter",
            url="https://github.com/microsoft/vscode-python/issues/7",
            created_at="2024-01-01T00:00:00+00:00",
            closed_at="2024-01-02T00:00:00+00:00",
            state_reason="completed",
            final_labels=("bug",),
            timeline=timeline,
            comments=comments,
        )

    def close(self) -> None:
        pass

    def _record_download(self) -> None:
        if not self.allow_download:
            raise AssertionError("A completed checkpoint must not be downloaded again")
        self.download_calls += 1


def test_bulk_collection_reuses_completed_phase_checkpoints(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    active_source = _CheckpointSource(allow_download=True)

    def source_factory(_repository: str) -> _CheckpointSource:
        return active_source

    monkeypatch.setattr(ml_module, "GitHubIssueSource", source_factory)
    archive = tmp_path / "closed-issues.snapshot.jsonl"
    dataset = tmp_path / "dataset.jsonl"

    first = collect_dataset_bulk(
        repository="microsoft/vscode-python",
        dataset_path=dataset,
        snapshot_archive_path=archive,
    )
    assert active_source.download_calls == 2

    active_source = _CheckpointSource(allow_download=False)
    second = collect_dataset_bulk(
        repository="microsoft/vscode-python",
        dataset_path=dataset,
        snapshot_archive_path=archive,
    )

    assert second == first
    assert active_source.download_calls == 0


def test_bulk_collection_rejects_modified_completed_checkpoint(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    active_source = _CheckpointSource(allow_download=True)

    def source_factory(_repository: str) -> _CheckpointSource:
        return active_source

    monkeypatch.setattr(ml_module, "GitHubIssueSource", source_factory)
    archive = tmp_path / "closed-issues.snapshot.jsonl"
    dataset = tmp_path / "dataset.jsonl"
    collect_dataset_bulk(
        repository="microsoft/vscode-python",
        dataset_path=dataset,
        snapshot_archive_path=archive,
    )
    issue_checkpoint = archive.with_suffix(".issues.jsonl")
    issue_checkpoint.write_text("{}\n", encoding="utf-8")

    with raises(ValueError, match="manifest SHA-256"):
        collect_dataset_bulk(
            repository="microsoft/vscode-python",
            dataset_path=dataset,
            snapshot_archive_path=archive,
        )

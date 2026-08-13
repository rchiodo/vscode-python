"""Deterministic supervised-ML baselines for historical issue triage."""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import hashlib
import importlib
import json
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, Self, TextIO, cast

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from sklearn.pipeline import FeatureUnion

from .github_source import (
    GitHubIssueSource,
    is_pull_request,
    normalize_timeline_comment,
    normalize_timeline_event,
    timeline_event_type,
)
from .models import IssueComment, IssueSnapshot, TimelineEvent
from .outcomes import infer_actual_outcome
from .storage import ResultStore

FeatureMatrix = Any
RANDOM_STATE = 42
ZERO_DIVISION: Any = 0


@dataclass(frozen=True)
class MLRecord:
    """An immutable training example derived from one historical issue."""

    repository: str
    issue_number: int
    title: str
    body: str
    url: str
    created_at: str
    content_source: str
    classification: str
    disposition: str
    labels: tuple[str, ...]
    information_request_ids: tuple[str, ...]

    @property
    def text(self) -> str:
        """Return the only text features exposed to a model."""
        return f"{self.title}\n\n{self.body}".strip()

    def to_dict(self) -> dict[str, Any]:
        """Serialize a dataset row."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        """Deserialize a validated dataset row."""
        return cls(
            repository=str(value["repository"]),
            issue_number=int(value["issue_number"]),
            title=str(value["title"]),
            body=str(value["body"]),
            url=str(value["url"]),
            created_at=str(value["created_at"]),
            content_source=str(value["content_source"]),
            classification=str(value["classification"]),
            disposition=str(value["disposition"]),
            labels=tuple(str(item) for item in value["labels"]),
            information_request_ids=tuple(str(item) for item in value["information_request_ids"]),
        )

    @classmethod
    def from_issue(cls, issue: IssueSnapshot) -> Self:
        """Create a row from a source snapshot and inferred outcome."""
        actual = infer_actual_outcome(issue)
        return cls(
            repository=issue.repository,
            issue_number=issue.number,
            title=issue.title,
            body=issue.body,
            url=issue.url,
            created_at=issue.created_at,
            content_source=issue.content_source,
            classification=actual.classification,
            disposition=actual.disposition.value,
            labels=actual.labels,
            information_request_ids=actual.information_request_ids,
        )


@dataclass(frozen=True)
class DatasetSplit:
    """Chronological train, tuning, and holdout partitions."""

    train: tuple[MLRecord, ...]
    tuning: tuple[MLRecord, ...]
    holdout: tuple[MLRecord, ...]


class TextEncoder(Protocol):
    """Feature encoder persisted with an ML model bundle."""

    def fit_transform(self, texts: list[str]) -> FeatureMatrix:
        """Fit the encoder and transform training text."""

    def transform(self, texts: list[str]) -> FeatureMatrix:
        """Transform text without refitting."""


@dataclass
class TfidfEncoder:
    """Combined word and character TF-IDF features."""

    vectorizer: FeatureUnion = field(
        default_factory=lambda: FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        lowercase=True,
                        max_features=60_000,
                        ngram_range=(1, 2),
                        strip_accents="unicode",
                        sublinear_tf=True,
                    ),
                ),
                (
                    "character",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        lowercase=True,
                        max_features=80_000,
                        ngram_range=(3, 5),
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
    )

    def fit_transform(self, texts: list[str]) -> FeatureMatrix:
        """Fit and transform TF-IDF features."""
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: list[str]) -> FeatureMatrix:
        """Transform TF-IDF features."""
        return self.vectorizer.transform(texts)


@dataclass
class SentenceTransformerEncoder:
    """Sentence-transformer embeddings loaded lazily by model name."""

    model_name: str
    _model: Any = field(default=None, init=False, repr=False)

    def fit_transform(self, texts: list[str]) -> FeatureMatrix:
        """Encode training text; sentence transformers need no local fitting."""
        return self.transform(texts)

    def transform(self, texts: list[str]) -> FeatureMatrix:
        """Encode normalized dense embeddings."""
        model = self._load_model()
        return np.asarray(
            model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=len(texts) >= 100,
            )
        )

    def __getstate__(self) -> dict[str, object]:
        """Persist only the model name rather than transformer weights."""
        return {"model_name": self.model_name, "_model": None}

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                module = importlib.import_module("sentence_transformers")
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Sentence-transformer features require "
                    "`uv sync --project scripts\\issue_triage --extra embeddings`."
                ) from exc
            self._model = module.SentenceTransformer(self.model_name)
        return self._model


@dataclass
class SingleLabelHead:
    """A multiclass logistic or constant classifier."""

    estimator: Any

    def predict(self, features: FeatureMatrix) -> np.ndarray[Any, Any]:
        """Predict one class per record."""
        return np.asarray(self.estimator.predict(features))

    def confidence(self, features: FeatureMatrix) -> np.ndarray[Any, Any]:
        """Return maximum class probability per record."""
        probabilities = np.asarray(self.estimator.predict_proba(features))
        return probabilities.max(axis=1)


@dataclass
class MultiLabelHead:
    """Independent binary logistic heads with a tuning-selected threshold."""

    classes: tuple[str, ...]
    estimators: tuple[Any, ...]
    threshold: float = 0.5

    def encode(self, targets: list[tuple[str, ...]]) -> np.ndarray[Any, Any]:
        """Encode string label sets as a binary matrix."""
        class_indexes = {label: index for index, label in enumerate(self.classes)}
        encoded = np.zeros((len(targets), len(self.classes)), dtype=np.int8)
        for row, labels in enumerate(targets):
            for label in labels:
                index = class_indexes.get(label)
                if index is not None:
                    encoded[row, index] = 1
        return encoded

    def probabilities(self, features: FeatureMatrix) -> np.ndarray[Any, Any]:
        """Return positive-class probability for each independent label."""
        if not self.estimators:
            return np.empty((features.shape[0], 0))
        columns = [_positive_probability(estimator, features) for estimator in self.estimators]
        return np.column_stack(columns)

    def predict(self, features: FeatureMatrix) -> list[tuple[str, ...]]:
        """Decode thresholded probabilities to label tuples."""
        binary = self.probabilities(features) >= self.threshold
        return [
            tuple(label for index, label in enumerate(self.classes) if binary[row, index])
            for row in range(binary.shape[0])
        ]


@dataclass
class MLModelBundle:
    """Persisted feature encoder and prediction heads."""

    approach: str
    encoder: TextEncoder
    classification: SingleLabelHead
    disposition: SingleLabelHead
    labels: MultiLabelHead
    information_requests: MultiLabelHead
    metadata: dict[str, Any]


def collect_dataset(
    *,
    repository: str,
    limit: int,
    since: datetime | None,
    database: Path,
    dataset_path: Path,
    refresh_cache: bool,
) -> tuple[MLRecord, ...]:
    """Fetch and persist a fixed historical dataset."""
    source = GitHubIssueSource(repository)
    try:
        references = source.list_closed_issues(limit=limit, since=since)
        records: list[MLRecord] = []
        with ResultStore(database) as store:
            for index, reference in enumerate(references, start=1):
                issue = (
                    None if refresh_cache else store.get_cached_issue(repository, reference.number)
                )
                if issue is None:
                    issue = source.hydrate_issue(reference)
                    store.cache_issue(issue)
                records.append(MLRecord.from_issue(issue))
                if index % 25 == 0 or index == len(references):
                    print(f"Prepared {index}/{len(references)} ML examples.")
    finally:
        source.close()

    write_dataset(dataset_path, records)
    return tuple(records)


def collect_dataset_bulk(
    *,
    repository: str,
    dataset_path: Path,
    snapshot_archive_path: Path,
) -> tuple[MLRecord, ...]:
    """Download all closed issues using resumable repository-wide API streams."""
    source = GitHubIssueSource(repository)
    try:
        issue_checkpoint = snapshot_archive_path.with_suffix(".issues.jsonl")
        base_issues = _collect_issue_checkpoint(
            source=source,
            path=issue_checkpoint,
            repository=repository,
        )
        history_checkpoint = snapshot_archive_path.with_suffix(".history.jsonl")
        issues = _collect_history_checkpoint(
            source=source,
            path=history_checkpoint,
            repository=repository,
            base_issues=base_issues,
        )

        records: list[MLRecord] = []
        snapshot_archive_path.parent.mkdir(parents=True, exist_ok=True)
        with snapshot_archive_path.open("w", encoding="utf-8", newline="\n") as archive:
            for index, issue in enumerate(issues, start=1):
                archive.write(json.dumps(issue.to_dict(), sort_keys=True) + "\n")
                records.append(MLRecord.from_issue(issue))
                if index % 500 == 0 or index == len(issues):
                    print(f"Archived {index}/{len(issues)} closed issues.", flush=True)
    finally:
        source.close()

    write_dataset(dataset_path, records)
    _write_file_manifest(snapshot_archive_path, len(records), repository)
    return tuple(records)


def _collect_history_checkpoint(
    *,
    source: GitHubIssueSource,
    path: Path,
    repository: str,
    base_issues: tuple[IssueSnapshot, ...],
) -> tuple[IssueSnapshot, ...]:
    _repair_trailing_jsonl(path)
    expected_records = _completed_checkpoint_records(path, repository=repository)
    hydrated = {issue.number: issue for issue in _load_snapshot_checkpoint(path)}
    if expected_records is not None:
        _validate_checkpoint_record_count(path, expected_records, len(hydrated))
        print(f"Loaded {len(hydrated)} complete issue histories from {path}.", flush=True)
        return tuple(hydrated[issue.number] for issue in base_issues)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as output:
        for issue in base_issues:
            if issue.number in hydrated:
                continue
            timeline: list[TimelineEvent] = []
            comments: list[IssueComment] = []
            for event in source.iter_issue_timeline(issue.number):
                if timeline_event_type(event) == "commented":
                    comments.append(normalize_timeline_comment(event))
                else:
                    timeline.append(normalize_timeline_event(event))
            events = tuple(sorted(timeline, key=lambda item: item.created_at or ""))
            initial_title = next(
                (event.renamed_from for event in events if event.renamed_from),
                issue.title,
            )
            snapshot = replace(
                issue,
                title=initial_title,
                timeline=events,
                comments=tuple(comments),
                content_source=(
                    "github_current_body_reconstructed_initial_title"
                    if initial_title != issue.title
                    else "github_current_snapshot"
                ),
            )
            output.write(json.dumps(snapshot.to_dict(), sort_keys=True) + "\n")
            _sync_file(output)
            hydrated[issue.number] = snapshot
            if len(hydrated) % 100 == 0:
                print(
                    f"Checkpointed {len(hydrated)}/{len(base_issues)} issue histories.",
                    flush=True,
                )

    histories = tuple(hydrated[issue.number] for issue in base_issues)
    _write_file_manifest(path, len(histories), repository)
    print(f"Checkpointed all {len(histories)} issue histories.", flush=True)
    return histories


def _collect_issue_checkpoint(
    *,
    source: GitHubIssueSource,
    path: Path,
    repository: str,
) -> tuple[IssueSnapshot, ...]:
    _repair_trailing_jsonl(path)
    expected_records = _completed_checkpoint_records(path, repository=repository)
    issues = _load_snapshot_checkpoint(path)
    if expected_records is not None:
        _validate_checkpoint_record_count(path, expected_records, len(issues))
        print(f"Loaded {len(issues)} closed issues from {path}.", flush=True)
        return issues

    path.parent.mkdir(parents=True, exist_ok=True)
    issues_by_number = {issue.number: issue for issue in issues}
    state = _load_checkpoint_state(path)
    start_cursor = _state_optional_string(state, "next_cursor")
    completed_pages = _state_int(state, "next_page")
    with path.open("a", encoding="utf-8", newline="\n") as output:
        for next_cursor, page in source.iter_closed_issue_pages(
            start_cursor=start_cursor,
            completed_pages=completed_pages,
        ):
            for issue in page:
                if is_pull_request(issue) or issue.number in issues_by_number:
                    continue
                snapshot = source.build_snapshot(issue, timeline=(), comments=())
                issues_by_number[snapshot.number] = snapshot
                output.write(json.dumps(snapshot.to_dict(), sort_keys=True) + "\n")
            _sync_file(output)
            _write_checkpoint_state(path, {"next_cursor": next_cursor})
            if len(issues_by_number) % 1_000 < len(page):
                print(
                    f"Checkpointed {len(issues_by_number)} closed issue records.",
                    flush=True,
                )

    issues = tuple(issues_by_number.values())
    _write_file_manifest(path, len(issues), repository)
    print(f"Checkpointed all {len(issues)} closed issue records.", flush=True)
    return issues


def _load_snapshot_checkpoint(path: Path) -> tuple[IssueSnapshot, ...]:
    snapshots: dict[int, IssueSnapshot] = {}
    for row in _load_jsonl_objects(path):
        snapshot = IssueSnapshot.from_dict(row)
        snapshots[snapshot.number] = snapshot
    return tuple(snapshots.values())


def _load_jsonl_objects(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            decoded: object = json.loads(line)
            if not isinstance(decoded, dict):
                raise ValueError(f"Checkpoint {path}:{line_number} must be a JSON object")
            untyped = cast(dict[object, object], decoded)
            if not all(isinstance(key, str) for key in untyped):
                raise ValueError(f"Checkpoint {path}:{line_number} has a non-string key")
            yield cast(dict[str, Any], untyped)


def _completed_checkpoint_records(path: Path, *, repository: str) -> int | None:
    if not path.with_suffix(".manifest.json").exists():
        return None
    return _validate_file_manifest(path, expected_repository=repository)


def _validate_checkpoint_record_count(path: Path, expected: int, actual: int) -> None:
    if expected != actual:
        raise ValueError(
            f"Checkpoint {path} contains {actual} records but its manifest expects {expected}"
        )


def _load_checkpoint_state(path: Path) -> dict[str, object]:
    state_path = path.with_suffix(".state.json")
    if not state_path.exists():
        return {}
    decoded: object = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"Checkpoint state {state_path} must be a JSON object")
    untyped = cast(dict[object, object], decoded)
    return {str(key): value for key, value in untyped.items() if isinstance(key, str)}


def _write_checkpoint_state(path: Path, state: dict[str, object]) -> None:
    state_path = path.with_suffix(".state.json")
    temporary_path = state_path.with_name(f"{state_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(state, indent=2, sort_keys=True) + "\n")
        _sync_file(output)
    temporary_path.replace(state_path)


def _state_int(state: dict[str, object], key: str) -> int:
    value = state.get(key, 0)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Checkpoint state field {key!r} must be a non-negative integer")
    return value


def _state_optional_string(state: dict[str, object], key: str) -> str | None:
    value = state.get(key)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"Checkpoint state field {key!r} must be a string or null")
    return value


def _repair_trailing_jsonl(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    content = path.read_bytes()
    if content.endswith(b"\n"):
        return
    last_complete_line = content.rfind(b"\n")
    with path.open("r+b") as output:
        output.truncate(last_complete_line + 1)
        output.flush()
        os.fsync(output.fileno())


def _sync_file(output: TextIO) -> None:
    output.flush()
    os.fsync(output.fileno())


def write_dataset(path: Path, records: list[MLRecord]) -> None:
    """Write immutable JSONL records and a provenance manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        _sync_file(output)
    temporary_path.replace(path)

    _write_file_manifest(
        path,
        len(records),
        records[0].repository if records else None,
    )


def load_dataset(path: Path) -> tuple[MLRecord, ...]:
    """Load an existing immutable JSONL dataset."""
    expected_records = _validate_file_manifest(path)
    records: list[MLRecord] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            decoded: object = json.loads(line)
            if not isinstance(decoded, dict):
                raise ValueError(f"Dataset line {line_number} must be a JSON object")
            untyped = cast(dict[object, object], decoded)
            if not all(isinstance(key, str) for key in untyped):
                raise ValueError(f"Dataset line {line_number} has a non-string key")
            records.append(MLRecord.from_dict(cast(dict[str, Any], untyped)))
    if len(records) != expected_records:
        raise ValueError(
            f"Dataset {path} contains {len(records)} records; "
            f"its manifest declares {expected_records}"
        )
    return tuple(records)


def chronological_split(
    records: tuple[MLRecord, ...],
    *,
    train_fraction: float = 0.7,
    tuning_fraction: float = 0.15,
) -> DatasetSplit:
    """Split oldest-to-newest so holdout metrics expose temporal drift."""
    if len(records) < 12:
        raise ValueError("ML evaluation requires at least 12 historical issues")
    if train_fraction <= 0 or tuning_fraction <= 0:
        raise ValueError("Training and tuning fractions must be positive")
    if train_fraction + tuning_fraction >= 1:
        raise ValueError("Training and tuning fractions must leave a holdout set")

    ordered = tuple(sorted(records, key=lambda record: record.created_at))
    train_end = max(1, int(len(ordered) * train_fraction))
    tuning_end = max(train_end + 1, int(len(ordered) * (train_fraction + tuning_fraction)))
    tuning_end = min(tuning_end, len(ordered) - 1)
    return DatasetSplit(
        train=ordered[:train_end],
        tuning=ordered[train_end:tuning_end],
        holdout=ordered[tuning_end:],
    )


def run_experiment(
    *,
    records: tuple[MLRecord, ...],
    approach: str,
    output_directory: Path,
    embedding_model: str,
) -> dict[str, Any]:
    """Train one approach, tune thresholds, and evaluate untouched holdout data."""
    split = chronological_split(records)
    encoder: TextEncoder
    if approach == "tfidf":
        encoder = TfidfEncoder()
    elif approach == "sentence-transformer":
        encoder = SentenceTransformerEncoder(embedding_model)
    else:
        raise ValueError(f"Unsupported ML approach: {approach}")

    train_features = encoder.fit_transform([record.text for record in split.train])
    tuning_features = encoder.transform([record.text for record in split.tuning])
    holdout_features = encoder.transform([record.text for record in split.holdout])

    classification = SingleLabelHead(
        _fit_single_label(train_features, [record.classification for record in split.train])
    )
    disposition = SingleLabelHead(
        _fit_single_label(train_features, [record.disposition for record in split.train])
    )
    labels = _fit_multi_label(
        train_features,
        [record.labels for record in split.train],
        records,
        target="labels",
    )
    information_requests = _fit_multi_label(
        train_features,
        [record.information_request_ids for record in split.train],
        records,
        target="information_request_ids",
    )
    labels.threshold = _tune_threshold(
        labels, tuning_features, [record.labels for record in split.tuning]
    )
    information_requests.threshold = _tune_threshold(
        information_requests,
        tuning_features,
        [record.information_request_ids for record in split.tuning],
    )

    dataset_sha256 = _records_sha256(records)
    bundle = MLModelBundle(
        approach=approach,
        encoder=encoder,
        classification=classification,
        disposition=disposition,
        labels=labels,
        information_requests=information_requests,
        metadata={
            "dataset_sha256": dataset_sha256,
            "embedding_model": embedding_model if approach == "sentence-transformer" else None,
            "random_state": RANDOM_STATE,
            "train_issues": [record.issue_number for record in split.train],
            "tuning_issues": [record.issue_number for record in split.tuning],
            "holdout_issues": [record.issue_number for record in split.holdout],
        },
    )

    tuning_metrics, _ = _evaluate_records(bundle, split.tuning, tuning_features)
    holdout_metrics, predictions = _evaluate_records(bundle, split.holdout, holdout_features)
    report = {
        "approach": approach,
        "dataset_sha256": dataset_sha256,
        "records": len(records),
        "split": {
            "train": len(split.train),
            "tuning": len(split.tuning),
            "holdout": len(split.holdout),
        },
        "thresholds": {
            "labels": labels.threshold,
            "information_requests": information_requests.threshold,
        },
        "tuning": tuning_metrics,
        "holdout": holdout_metrics,
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_directory / f"{approach}.model.joblib")
    (output_directory / f"{approach}.metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_directory / f"{approach}.predictions.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as output:
        for prediction in predictions:
            output.write(json.dumps(prediction, sort_keys=True) + "\n")
    return report


def _fit_single_label(features: FeatureMatrix, targets: list[str]) -> Any:
    unique = sorted(set(targets))
    if len(unique) == 1:
        estimator = DummyClassifier(strategy="constant", constant=unique[0])
    else:
        estimator = LogisticRegression(
            class_weight="balanced",
            max_iter=2_000,
            random_state=RANDOM_STATE,
        )
    return estimator.fit(features, targets)


def _fit_multi_label(
    features: FeatureMatrix,
    train_targets: list[tuple[str, ...]],
    all_records: tuple[MLRecord, ...],
    *,
    target: str,
) -> MultiLabelHead:
    if target == "labels":
        classes = tuple(sorted({label for record in all_records for label in record.labels}))
    else:
        classes = tuple(
            sorted(
                {
                    request_id
                    for record in all_records
                    for request_id in record.information_request_ids
                }
            )
        )
    encoded = MultiLabelHead(classes, ()).encode(train_targets)
    estimators = tuple(
        _fit_binary_label(features, encoded[:, index]) for index in range(len(classes))
    )
    return MultiLabelHead(classes=classes, estimators=estimators)


def _fit_binary_label(features: FeatureMatrix, targets: np.ndarray[Any, Any]) -> Any:
    unique = np.unique(targets)
    if len(unique) == 1:
        estimator = DummyClassifier(strategy="constant", constant=int(unique[0]))
    else:
        estimator = LogisticRegression(
            class_weight="balanced",
            max_iter=2_000,
            random_state=RANDOM_STATE,
        )
    return estimator.fit(features, targets)


def _positive_probability(estimator: Any, features: FeatureMatrix) -> np.ndarray[Any, Any]:
    probabilities = np.asarray(estimator.predict_proba(features))
    classes = list(estimator.classes_)
    if 1 not in classes:
        return np.zeros(features.shape[0])
    return probabilities[:, classes.index(1)]


def _tune_threshold(
    head: MultiLabelHead,
    features: FeatureMatrix,
    targets: list[tuple[str, ...]],
) -> float:
    if not head.classes:
        return 0.5
    actual = head.encode(targets)
    if not actual.any():
        return 0.5
    probabilities = head.probabilities(features)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in (0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8):
        predicted = probabilities >= threshold
        score = float(
            f1_score(
                actual,
                predicted,
                average="micro",
                zero_division=ZERO_DIVISION,
            )
        )
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold


def _evaluate_records(
    bundle: MLModelBundle,
    records: tuple[MLRecord, ...],
    features: FeatureMatrix,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    actual_classification = [record.classification for record in records]
    actual_disposition = [record.disposition for record in records]
    predicted_classification = bundle.classification.predict(features)
    predicted_disposition = bundle.disposition.predict(features)
    predicted_labels = bundle.labels.predict(features)
    predicted_information = bundle.information_requests.predict(features)

    classification_metrics = _single_label_metrics(actual_classification, predicted_classification)
    disposition_metrics = _single_label_metrics(actual_disposition, predicted_disposition)
    label_metrics = _multi_label_metrics(
        bundle.labels,
        [record.labels for record in records],
        predicted_labels,
    )
    information_metrics = _multi_label_metrics(
        bundle.information_requests,
        [record.information_request_ids for record in records],
        predicted_information,
    )
    score_components = [
        classification_metrics["macro_f1"],
        disposition_metrics["macro_f1"],
    ]
    score_components.extend(
        metrics["micro_f1"]
        for metrics in (label_metrics, information_metrics)
        if metrics["micro_f1"] is not None
    )

    class_confidence = bundle.classification.confidence(features)
    disposition_confidence = bundle.disposition.confidence(features)
    predictions = [
        {
            "issue": {
                "number": record.issue_number,
                "url": record.url,
                "created_at": record.created_at,
            },
            "predicted": {
                "classification": str(predicted_classification[index]),
                "classification_confidence": float(class_confidence[index]),
                "disposition": str(predicted_disposition[index]),
                "disposition_confidence": float(disposition_confidence[index]),
                "labels": list(predicted_labels[index]),
                "information_request_ids": list(predicted_information[index]),
            },
            "actual": {
                "classification": record.classification,
                "disposition": record.disposition,
                "labels": list(record.labels),
                "information_request_ids": list(record.information_request_ids),
            },
        }
        for index, record in enumerate(records)
    ]
    return (
        {
            "issues": len(records),
            "classification": classification_metrics,
            "disposition": disposition_metrics,
            "labels": label_metrics,
            "information_requests": information_metrics,
            "overall_score": sum(score_components) / len(score_components),
        },
        predictions,
    )


def _single_label_metrics(actual: list[str], predicted: np.ndarray[Any, Any]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(
            f1_score(
                actual,
                predicted,
                average="macro",
                zero_division=ZERO_DIVISION,
            )
        ),
        "weighted_f1": float(
            f1_score(
                actual,
                predicted,
                average="weighted",
                zero_division=ZERO_DIVISION,
            )
        ),
    }


def _multi_label_metrics(
    head: MultiLabelHead,
    actual_targets: list[tuple[str, ...]],
    predicted_targets: list[tuple[str, ...]],
) -> dict[str, Any]:
    if not head.classes:
        return {
            "classes": 0,
            "exact_accuracy": 1.0,
            "evaluated_issues": 0,
            "jaccard": None,
            "macro_f1": None,
            "micro_f1": None,
        }
    actual = head.encode(actual_targets)
    predicted = head.encode(predicted_targets)
    relevant = np.any((actual == 1) | (predicted == 1), axis=1)
    if not relevant.any():
        return {
            "classes": len(head.classes),
            "exact_accuracy": float(accuracy_score(actual, predicted)),
            "evaluated_issues": 0,
            "jaccard": None,
            "macro_f1": None,
            "micro_f1": None,
        }
    relevant_actual = actual[relevant]
    relevant_predicted = predicted[relevant]
    if len(head.classes) == 1:
        jaccard = jaccard_score(
            relevant_actual.ravel(),
            relevant_predicted.ravel(),
            average="binary",
            zero_division=ZERO_DIVISION,
        )
    else:
        jaccard = jaccard_score(
            relevant_actual,
            relevant_predicted,
            average="samples",
            zero_division=ZERO_DIVISION,
        )
    return {
        "classes": len(head.classes),
        "exact_accuracy": float(accuracy_score(actual, predicted)),
        "evaluated_issues": int(relevant.sum()),
        "jaccard": float(jaccard),
        "macro_f1": float(
            f1_score(
                relevant_actual,
                relevant_predicted,
                average="macro",
                zero_division=ZERO_DIVISION,
            )
        ),
        "micro_f1": float(
            f1_score(
                relevant_actual,
                relevant_predicted,
                average="micro",
                zero_division=ZERO_DIVISION,
            )
        ),
    }


def _records_sha256(records: tuple[MLRecord, ...]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record.to_dict(), sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_file_manifest(path: Path, records: int, repository: str | None) -> None:
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "file": str(path.resolve()),
        "records": records,
        "repository": repository,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    manifest_path = path.with_suffix(".manifest.json")
    temporary_path = manifest_path.with_name(f"{manifest_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        _sync_file(output)
    temporary_path.replace(manifest_path)


def _validate_file_manifest(path: Path, *, expected_repository: str | None = None) -> int:
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise ValueError(f"Dataset manifest does not exist: {manifest_path}")
    decoded: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"Dataset manifest {manifest_path} must be a JSON object")
    manifest = cast(dict[object, object], decoded)
    records = manifest.get("records")
    repository = manifest.get("repository")
    expected_sha256 = manifest.get("sha256")
    if not isinstance(records, int) or isinstance(records, bool) or records < 0:
        raise ValueError(f"Dataset manifest {manifest_path} has an invalid record count")
    if not isinstance(expected_sha256, str):
        raise ValueError(f"Dataset manifest {manifest_path} has an invalid SHA-256")
    if expected_repository is not None and repository != expected_repository:
        raise ValueError(
            f"Dataset manifest {manifest_path} belongs to {repository!r}, "
            f"not {expected_repository!r}"
        )
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Dataset {path} does not match its manifest SHA-256")
    return records

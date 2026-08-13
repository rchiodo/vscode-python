"""Compare classifier decisions with normalized historical outcomes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .models import ActualOutcome, Comparison, TriageDecision


@dataclass(frozen=True)
class SetScores:
    """Precision and distance metrics for two sets."""

    precision: float
    recall: float
    f1: float
    jaccard: float
    exact: bool
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]


def compare_decision(predicted: TriageDecision, actual: ActualOutcome) -> Comparison:
    """Compare one prediction with its historical outcome."""
    label_scores = compare_sets(predicted.labels, actual.labels)
    information_scores = compare_sets(
        predicted.information_request_ids, actual.information_request_ids
    )
    disposition_match = (
        None
        if predicted.disposition.value == "transfer"
        and actual.disposition.value == "keep_open"
        and actual.transfer_target is None
        else predicted.disposition == actual.disposition
    )
    transfer_target_match = (
        predicted.transfer_target == actual.transfer_target
        if actual.transfer_target is not None
        else None
    )

    components = [
        float(predicted.classification == actual.classification),
        label_scores.f1,
    ]
    if disposition_match is not None:
        components.append(float(disposition_match))
    if (
        predicted.information_request_ids
        or actual.information_request_ids
        or predicted.disposition.value == "request_information"
        or actual.disposition.value == "request_information"
    ):
        components.append(information_scores.f1)
    if transfer_target_match is not None:
        components.append(float(transfer_target_match))

    return Comparison(
        classification_match=predicted.classification == actual.classification,
        disposition_match=disposition_match,
        transfer_target_match=transfer_target_match,
        label_precision=label_scores.precision,
        label_recall=label_scores.recall,
        label_f1=label_scores.f1,
        label_jaccard=label_scores.jaccard,
        labels_exact=label_scores.exact,
        missing_labels=label_scores.missing,
        unexpected_labels=label_scores.unexpected,
        information_precision=information_scores.precision,
        information_recall=information_scores.recall,
        information_f1=information_scores.f1,
        information_exact=information_scores.exact,
        missing_information_requests=information_scores.missing,
        unexpected_information_requests=information_scores.unexpected,
        overall_score=sum(components) / len(components),
    )


def compare_sets(predicted: Iterable[str], actual: Iterable[str]) -> SetScores:
    """Calculate metrics that remain defined when either set is empty."""
    predicted_set = set(predicted)
    actual_set = set(actual)
    intersection = predicted_set & actual_set
    union = predicted_set | actual_set

    precision = len(intersection) / len(predicted_set) if predicted_set else float(not actual_set)
    recall = len(intersection) / len(actual_set) if actual_set else float(not predicted_set)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return SetScores(
        precision=precision,
        recall=recall,
        f1=f1,
        jaccard=len(intersection) / len(union) if union else 1.0,
        exact=predicted_set == actual_set,
        missing=tuple(sorted(actual_set - predicted_set)),
        unexpected=tuple(sorted(predicted_set - actual_set)),
    )


def summarize_results(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate result records into a compact benchmark summary."""
    rows = list(results)
    completed = [row for row in rows if row.get("comparison") is not None]
    errors = [row for row in rows if row.get("error")]
    if not completed:
        return {"issues": len(rows), "completed": 0, "errors": len(errors)}

    comparisons = [row["comparison"] for row in completed]
    disposition_comparisons = [
        comparison for comparison in comparisons if comparison["disposition_match"] is not None
    ]
    information_comparisons = [
        row["comparison"] for row in completed if _has_information_decision(row)
    ]
    confusion: dict[str, dict[str, int]] = {}
    for row in completed:
        actual = str(row["actual"]["classification"])
        predicted = str(row["prediction"]["classification"])
        confusion.setdefault(actual, {})
        confusion[actual][predicted] = confusion[actual].get(predicted, 0) + 1

    return {
        "issues": len(rows),
        "completed": len(completed),
        "errors": len(errors),
        "classification_accuracy": _mean(
            float(comparison["classification_match"]) for comparison in comparisons
        ),
        "disposition_evaluated": len(disposition_comparisons),
        "disposition_accuracy": (
            _mean(float(comparison["disposition_match"]) for comparison in disposition_comparisons)
            if disposition_comparisons
            else None
        ),
        "labels_exact_accuracy": _mean(
            float(comparison["labels_exact"]) for comparison in comparisons
        ),
        "mean_label_f1": _mean(float(comparison["label_f1"]) for comparison in comparisons),
        "information_evaluated": len(information_comparisons),
        "information_exact_accuracy": (
            _mean(float(comparison["information_exact"]) for comparison in information_comparisons)
            if information_comparisons
            else None
        ),
        "mean_information_f1": (
            _mean(float(comparison["information_f1"]) for comparison in information_comparisons)
            if information_comparisons
            else None
        ),
        "mean_overall_score": _mean(
            float(comparison["overall_score"]) for comparison in comparisons
        ),
        "classification_confusion": confusion,
    }


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected)


def _has_information_decision(row: Mapping[str, Any]) -> bool:
    prediction = row["prediction"]
    actual = row["actual"]
    return bool(
        prediction["information_request_ids"]
        or actual["information_request_ids"]
        or prediction["disposition"] == "request_information"
        or actual["disposition"] == "request_information"
    )

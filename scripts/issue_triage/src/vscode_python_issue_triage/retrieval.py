"""Temporally safe retrieval of similar historical issue reports."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from datetime import datetime
from itertools import pairwise

from .ml import MLRecord
from .models import RetrievalEvidence

RETRIEVAL_ALGORITHM = "hashing-word-char-v1"
BODY_EXCERPT_LENGTH = 800
RETRIEVAL_BODY_LENGTH = 2_000
HASH_BUCKETS = 2**18
WORD_PATTERN = re.compile(r"[a-z0-9_.+-]+")


class HistoricalIssueRetriever:
    """Retrieve similar issues without fitting vocabulary or IDF on future reports."""

    def __init__(self, records: tuple[MLRecord, ...]) -> None:
        self._records = records
        self._created_at = tuple(_timestamp(record.created_at) for record in records)
        self._features = tuple(_hashed_features(_retrieval_text(record)) for record in records)

    def retrieve(self, target: MLRecord, *, count: int) -> tuple[RetrievalEvidence, ...]:
        """Return the strongest matches created strictly before the target issue."""
        if count < 1:
            return ()

        try:
            target_features = self._features[self._records.index(target)]
        except ValueError:
            target_features = _hashed_features(_retrieval_text(target))
        target_created_at = _timestamp(target.created_at)
        eligible_indices = [
            index
            for index, record in enumerate(self._records)
            if record.repository == target.repository
            and record.issue_number != target.issue_number
            and self._created_at[index] < target_created_at
        ]
        if not eligible_indices:
            return ()
        ranked = sorted(
            (
                (index, _cosine_similarity(target_features, self._features[index]))
                for index in eligible_indices
            ),
            key=lambda item: (-float(item[1]), self._records[item[0]].issue_number),
        )[:count]
        return tuple(
            _to_evidence(self._records[index], similarity=float(similarity))
            for index, similarity in ranked
        )


def _retrieval_text(record: MLRecord) -> str:
    return f"{record.title}\n{record.title}\n{record.body[:RETRIEVAL_BODY_LENGTH]}".strip()


def _hashed_features(text: str) -> dict[int, float]:
    normalized = " ".join(text.lower().split())
    words = WORD_PATTERN.findall(normalized)
    features: Counter[int] = Counter()
    for word in words:
        features[_hash_feature(f"w:{word}")] += 1
    for first, second in pairwise(words):
        features[_hash_feature(f"b:{first} {second}")] += 1
    for word in words:
        padded = f" {word} "
        for size in (3, 4):
            for start in range(len(padded) - size + 1):
                features[_hash_feature(f"c:{padded[start : start + size]}")] += 1
    norm = math.sqrt(sum(value * value for value in features.values()))
    return {key: value / norm for key, value in features.items()} if norm else {}


def _hash_feature(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest) % HASH_BUCKETS


def _cosine_similarity(left: dict[int, float], right: dict[int, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(key, 0.0) for key, value in left.items())


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Issue timestamp must include a timezone: {value!r}")
    return parsed


def _to_evidence(record: MLRecord, *, similarity: float) -> RetrievalEvidence:
    body_excerpt = record.body[:BODY_EXCERPT_LENGTH]
    if len(record.body) > BODY_EXCERPT_LENGTH:
        body_excerpt += "..."
    return RetrievalEvidence(
        issue_number=record.issue_number,
        issue_url=record.url,
        title=record.title,
        body_excerpt=body_excerpt,
        created_at=record.created_at,
        similarity=round(similarity, 6),
        historical_classification=record.classification,
        historical_disposition=record.disposition,
        historical_labels=record.labels,
        historical_information_request_ids=record.information_request_ids,
    )

"""SQLite persistence and portable JSON export for benchmark runs."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .evaluation import summarize_results
from .models import ActualOutcome, Comparison, IssueSnapshot, TriageDecision


class ResultStore:
    """Persist source snapshots and resumable evaluation runs."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.row_factory = sqlite3.Row
        self._create_schema()

    def __enter__(self) -> ResultStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the database."""
        self._connection.close()

    def begin_run(self, run_id: str, repository: str, configuration: dict[str, Any]) -> None:
        """Create a run or mark an existing run as active."""
        now = _now()
        serialized_configuration = json.dumps(configuration, sort_keys=True)
        existing = self._connection.execute(
            "SELECT repository, configuration_json FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if existing is not None and (
            existing["repository"] != repository
            or existing["configuration_json"] != serialized_configuration
        ):
            raise ValueError(
                f"Run {run_id!r} already exists with different configuration; use a new run ID."
            )
        self._connection.execute(
            """
            INSERT INTO runs(run_id, repository, configuration_json, status, started_at)
            VALUES (?, ?, ?, 'running', ?)
            ON CONFLICT(run_id) DO UPDATE SET
                configuration_json = excluded.configuration_json,
                status = 'running',
                finished_at = NULL
            """,
            (run_id, repository, serialized_configuration, now),
        )
        self._connection.commit()

    def finish_run(self, run_id: str, status: str = "completed") -> None:
        """Mark a run as completed or failed."""
        self._connection.execute(
            "UPDATE runs SET status = ?, finished_at = ? WHERE run_id = ?",
            (status, _now(), run_id),
        )
        self._connection.commit()

    def cache_issue(self, issue: IssueSnapshot) -> None:
        """Cache a fully hydrated historical issue."""
        self._connection.execute(
            """
            INSERT INTO issue_cache(repository, issue_number, snapshot_json, fetched_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(repository, issue_number) DO UPDATE SET
                snapshot_json = excluded.snapshot_json,
                fetched_at = excluded.fetched_at
            """,
            (
                issue.repository,
                issue.number,
                json.dumps(issue.to_dict(), sort_keys=True),
                _now(),
            ),
        )
        self._connection.commit()

    def get_cached_issue(self, repository: str, issue_number: int) -> IssueSnapshot | None:
        """Return a cached issue if present."""
        row = self._connection.execute(
            """
            SELECT snapshot_json FROM issue_cache
            WHERE repository = ? AND issue_number = ?
            """,
            (repository, issue_number),
        ).fetchone()
        if row is None:
            return None
        raw: dict[str, Any] = json.loads(row["snapshot_json"])
        return IssueSnapshot.from_dict(raw)

    def set_cohort(self, run_id: str, repository: str, issue_numbers: tuple[int, ...]) -> None:
        """Persist immutable ordered membership for a benchmark run."""
        existing = self.get_cohort(run_id)
        if existing:
            if existing != issue_numbers:
                raise ValueError(f"Run {run_id!r} already has a different issue cohort.")
            return
        self._connection.executemany(
            """
            INSERT INTO run_issues(run_id, repository, issue_number, position)
            VALUES (?, ?, ?, ?)
            """,
            (
                (run_id, repository, issue_number, position)
                for position, issue_number in enumerate(issue_numbers)
            ),
        )
        self._connection.commit()

    def get_cohort(self, run_id: str) -> tuple[int, ...]:
        """Return a run's issue numbers in their original order."""
        rows = self._connection.execute(
            """
            SELECT issue_number FROM run_issues
            WHERE run_id = ? ORDER BY position
            """,
            (run_id,),
        )
        return tuple(int(row["issue_number"]) for row in rows)

    def bind_run_snapshot(self, run_id: str, issue: IssueSnapshot) -> None:
        """Bind the exact classifier input snapshot to one run cohort member."""
        updated = self._connection.execute(
            """
            UPDATE run_issues SET snapshot_json = ?
            WHERE run_id = ? AND repository = ? AND issue_number = ?
            """,
            (
                json.dumps(issue.to_dict(), sort_keys=True),
                run_id,
                issue.repository,
                issue.number,
            ),
        )
        if updated.rowcount != 1:
            raise ValueError(f"Issue #{issue.number} is not a member of run {run_id!r}.")
        self._connection.commit()

    def get_run_snapshot(
        self, run_id: str, repository: str, issue_number: int
    ) -> IssueSnapshot | None:
        """Return the immutable snapshot used by a specific run."""
        row = self._connection.execute(
            """
            SELECT snapshot_json FROM run_issues
            WHERE run_id = ? AND repository = ? AND issue_number = ?
            """,
            (run_id, repository, issue_number),
        ).fetchone()
        if row is None or row["snapshot_json"] is None:
            return None
        raw: dict[str, Any] = json.loads(row["snapshot_json"])
        return IssueSnapshot.from_dict(raw)

    def has_result(self, run_id: str, repository: str, issue_number: int) -> bool:
        """Return whether a run already contains a successful result for an issue."""
        row = self._connection.execute(
            """
            SELECT 1 FROM results
            WHERE run_id = ? AND repository = ? AND issue_number = ? AND error IS NULL
            """,
            (run_id, repository, issue_number),
        ).fetchone()
        return row is not None

    def record_result(
        self,
        *,
        run_id: str,
        issue: IssueSnapshot,
        prediction: TriageDecision,
        actual: ActualOutcome,
        comparison: Comparison,
        rendered_information_request: str | None,
    ) -> None:
        """Record one successful classification and comparison."""
        self._connection.execute(
            """
            INSERT INTO results(
                run_id, repository, issue_number, issue_url, title,
                prediction_json, actual_json, comparison_json,
                rendered_information_request, error, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
            ON CONFLICT(run_id, repository, issue_number) DO UPDATE SET
                prediction_json = excluded.prediction_json,
                actual_json = excluded.actual_json,
                comparison_json = excluded.comparison_json,
                rendered_information_request = excluded.rendered_information_request,
                error = NULL,
                created_at = excluded.created_at
            """,
            (
                run_id,
                issue.repository,
                issue.number,
                issue.url,
                issue.title,
                json.dumps(prediction.to_dict(), sort_keys=True),
                json.dumps(actual.to_dict(), sort_keys=True),
                json.dumps(comparison.to_dict(), sort_keys=True),
                rendered_information_request,
                _now(),
            ),
        )
        self._connection.commit()

    def record_error(self, *, run_id: str, issue: IssueSnapshot, error: str) -> None:
        """Record an issue-level failure so it is visible and retryable."""
        self._connection.execute(
            """
            INSERT INTO results(
                run_id, repository, issue_number, issue_url, title,
                prediction_json, actual_json, comparison_json,
                rendered_information_request, error, created_at
            )
            VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?, ?)
            ON CONFLICT(run_id, repository, issue_number) DO UPDATE SET
                prediction_json = NULL,
                actual_json = NULL,
                comparison_json = NULL,
                rendered_information_request = NULL,
                error = excluded.error,
                created_at = excluded.created_at
            """,
            (
                run_id,
                issue.repository,
                issue.number,
                issue.url,
                issue.title,
                error,
                _now(),
            ),
        )
        self._connection.commit()

    def iter_results(self, run_id: str) -> Iterator[dict[str, Any]]:
        """Yield portable result dictionaries ordered by issue number."""
        rows = self._connection.execute(
            """
            SELECT r.issue_number, r.issue_url, r.title, r.prediction_json, r.actual_json,
                   r.comparison_json, r.rendered_information_request, r.error,
                   c.snapshot_json
            FROM results AS r
            LEFT JOIN run_issues AS c
                ON c.run_id = r.run_id
                AND c.repository = r.repository
                AND c.issue_number = r.issue_number
            WHERE r.run_id = ? ORDER BY r.issue_number
            """,
            (run_id,),
        )
        for row in rows:
            snapshot_json: object = row["snapshot_json"]
            snapshot = (
                _load_nullable_json(snapshot_json) if isinstance(snapshot_json, str) else None
            ) or {}
            yield {
                "issue": {
                    "number": row["issue_number"],
                    "url": row["issue_url"],
                    "title": row["title"],
                    "body": snapshot.get("body", ""),
                    "content_source": snapshot.get("content_source", "github_current_snapshot"),
                    "created_at": snapshot.get("created_at"),
                },
                "prediction": _load_nullable_json(row["prediction_json"]),
                "actual": _load_nullable_json(row["actual_json"]),
                "comparison": _load_nullable_json(row["comparison_json"]),
                "rendered_information_request": row["rendered_information_request"],
                "error": row["error"],
            }

    def export_run(self, run_id: str, output_directory: Path) -> tuple[Path, Path]:
        """Export detailed JSONL and aggregate JSON files."""
        output_directory.mkdir(parents=True, exist_ok=True)
        rows = list(self.iter_results(run_id))
        details_path = output_directory / f"{run_id}.jsonl"
        summary_path = output_directory / f"{run_id}.summary.json"
        with details_path.open("w", encoding="utf-8", newline="\n") as details:
            for row in rows:
                details.write(json.dumps(row, sort_keys=True) + "\n")
        summary_path.write_text(
            json.dumps(summarize_results(rows), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return details_path, summary_path

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            PRAGMA journal_mode = WAL;

            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                repository TEXT NOT NULL,
                configuration_json TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT
            );

            CREATE TABLE IF NOT EXISTS issue_cache (
                repository TEXT NOT NULL,
                issue_number INTEGER NOT NULL,
                snapshot_json TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                PRIMARY KEY(repository, issue_number)
            );

            CREATE TABLE IF NOT EXISTS results (
                run_id TEXT NOT NULL,
                repository TEXT NOT NULL,
                issue_number INTEGER NOT NULL,
                issue_url TEXT NOT NULL,
                title TEXT NOT NULL,
                prediction_json TEXT,
                actual_json TEXT,
                comparison_json TEXT,
                rendered_information_request TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY(run_id, repository, issue_number),
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS run_issues (
                run_id TEXT NOT NULL,
                repository TEXT NOT NULL,
                issue_number INTEGER NOT NULL,
                position INTEGER NOT NULL,
                snapshot_json TEXT,
                PRIMARY KEY(run_id, repository, issue_number),
                UNIQUE(run_id, position),
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            """
        )
        self._connection.commit()


def _load_nullable_json(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    result: dict[str, Any] = json.loads(value)
    return result


def _now() -> str:
    return datetime.now(UTC).isoformat()

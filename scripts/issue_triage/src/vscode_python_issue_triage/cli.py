"""Command-line entry point for historical issue triage evaluation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from .classifier import (
    CopilotClassifier,
    RecoverableClassificationError,
    classification_label_catalog,
)
from .configuration import (
    InformationRequest,
    default_action_agent_path,
    default_agent_path,
    default_catalog_path,
    default_guidance_path,
    load_agent_definition,
    load_information_requests,
    project_root,
    render_information_request,
)
from .evaluation import compare_decision
from .github_source import GitHubIssueSource
from .models import Disposition, IssueSnapshot
from .outcomes import infer_actual_outcome
from .storage import ResultStore


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the issue-triage command."""
    parser = _create_parser()
    options = parser.parse_args(arguments)

    try:
        if options.command == "evaluate":
            return asyncio.run(_evaluate(options))
        if options.command == "ml-evaluate":
            return _ml_evaluate(options)
        if options.command == "llm-classify":
            return asyncio.run(_llm_classify(options))
        if options.command == "action-triage":
            return asyncio.run(_action_triage(options))
        parser.error("a command is required")
    except KeyboardInterrupt:
        print("Interrupted; completed issue results remain resumable in SQLite.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


async def _evaluate(options: argparse.Namespace) -> int:
    _validate_options(options)
    repository_root = project_root()
    agent = load_agent_definition(options.agent)
    information_requests = load_information_requests(options.information_requests)
    since = _parse_since(options.since) if options.since else None

    source = GitHubIssueSource(options.repository)
    try:
        github_login = source.get_authenticated_login()
        label_catalog = classification_label_catalog(source.get_label_catalog())
        async with CopilotClassifier(
            repository_root=repository_root,
            agent=agent,
            model=options.model,
            timeout=options.timeout,
            retries=options.retries,
        ) as classifier:
            run_configuration = {
                "agent_name": agent.name,
                "agent_path": str(options.agent.resolve()),
                "agent_sha256": _sha256_file(options.agent),
                "copilot_login": classifier.authenticated_login,
                "copilot_sdk_version": version("github-copilot-sdk"),
                "github_login": github_login,
                "implementation_sha256": _implementation_sha256(),
                "information_requests_path": str(options.information_requests.resolve()),
                "information_requests_sha256": _sha256_file(options.information_requests),
                "label_catalog_sha256": _sha256_text(json.dumps(label_catalog, sort_keys=True)),
                "limit": options.limit,
                "concurrency": options.concurrency,
                "model": options.model,
                "pygithub_version": version("PyGithub"),
                "refresh_cache": options.refresh_cache,
                "repository": options.repository,
                "retries": options.retries,
                "since": since.isoformat() if since else None,
                "timeout": options.timeout,
            }
            print(
                f"GitHub identity: {github_login}; "
                f"Copilot identity: {classifier.authenticated_login or 'authenticated user'}"
            )
            with ResultStore(options.database) as store:
                store.begin_run(options.run_id, options.repository, run_configuration)
                try:
                    await _process_issues(
                        options=options,
                        source=source,
                        store=store,
                        classifier=classifier,
                        label_catalog=label_catalog,
                        information_requests=information_requests,
                        since=since,
                    )
                except BaseException:
                    store.finish_run(options.run_id, "failed")
                    raise
                store.finish_run(options.run_id)
                details_path, summary_path = store.export_run(
                    options.run_id, options.output_directory
                )
    finally:
        source.close()

    print(f"Detailed results: {details_path}")
    print(f"Summary: {summary_path}")
    return 0


def _ml_evaluate(options: argparse.Namespace) -> int:
    _validate_ml_options(options)
    from .ml import (
        collect_dataset,
        collect_dataset_bulk,
        load_dataset,
        run_experiment,
    )

    since = _parse_since(options.since) if options.since else None
    if options.dataset.exists() and not options.rebuild_dataset:
        records = load_dataset(options.dataset)
        print(f"Loaded {len(records)} examples from {options.dataset}.")
    elif options.all_issues:
        records = collect_dataset_bulk(
            repository=options.repository,
            dataset_path=options.dataset,
            snapshot_archive_path=options.snapshot_archive,
        )
    else:
        records = collect_dataset(
            repository=options.repository,
            limit=options.limit,
            since=since,
            database=options.database,
            dataset_path=options.dataset,
            refresh_cache=options.refresh_cache,
        )

    repositories = {record.repository for record in records}
    if repositories != {options.repository}:
        raise ValueError(
            f"Dataset repositories {sorted(repositories)} do not match "
            f"--repository {options.repository!r}."
        )

    approaches = (
        ("tfidf", "sentence-transformer") if options.approach == "both" else (options.approach,)
    )
    reports = {
        approach: run_experiment(
            records=records,
            approach=approach,
            output_directory=options.output_directory,
            embedding_model=options.embedding_model,
        )
        for approach in approaches
    }
    best_tuning = max(
        reports,
        key=lambda approach: float(reports[approach]["tuning"]["overall_score"]),
    )
    comparison = {
        "selection_basis": "tuning.overall_score",
        "selected_approach": best_tuning,
        "approaches": {
            approach: {
                "tuning_overall_score": report["tuning"]["overall_score"],
                "holdout_overall_score": report["holdout"]["overall_score"],
            }
            for approach, report in reports.items()
        },
    }
    comparison_path = options.output_directory / "comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Selected by tuning score: {best_tuning}")
    for approach, report in reports.items():
        print(
            f"{approach}: tuning={report['tuning']['overall_score']:.3f}, "
            f"holdout={report['holdout']['overall_score']:.3f}"
        )
    print(f"ML comparison: {comparison_path}")
    return 0


async def _llm_classify(options: argparse.Namespace) -> int:
    _validate_llm_options(options)
    from .llm_benchmark import run_classification_benchmark

    summary = await run_classification_benchmark(
        dataset_path=options.dataset,
        output_directory=options.output_directory,
        model=options.model,
        cohort_size=options.cohort_size,
        seed=options.seed,
        concurrency=options.concurrency,
        timeout_seconds=options.timeout,
        retries=options.retries,
    )
    print(f"LLM classification summary: {options.output_directory / 'classification-summary.json'}")
    for variant, report in summary["variants"].items():
        accuracy = report["accuracy"]
        accuracy_text = f"{accuracy:.3f}" if isinstance(accuracy, int | float) else "n/a"
        print(
            f"{variant}: accuracy={accuracy_text}, "
            f"macro_f1={report['macro_f1']:.3f}, errors={report['errors']}"
        )
    return 0


async def _action_triage(options: argparse.Namespace) -> int:
    _validate_llm_options(options)
    from .action_triage import run_action_triage

    summary = await run_action_triage(
        dataset_path=options.dataset,
        output_directory=options.output_directory,
        model=options.model,
        cohort_size=options.cohort_size,
        seed=options.seed,
        concurrency=options.concurrency,
        timeout_seconds=options.timeout,
        retries=options.retries,
        agent_path=options.agent,
        information_requests_path=options.information_requests,
        guidance_path=options.guidance,
    )
    print(f"Action triage summary: {options.output_directory / 'action-summary.json'}")
    print(
        f"completed={summary['completed']}, errors={summary['errors']}, "
        f"actions={summary['action_distribution']}"
    )
    return 0


async def _process_issues(
    *,
    options: argparse.Namespace,
    source: GitHubIssueSource,
    store: ResultStore,
    classifier: CopilotClassifier,
    label_catalog: dict[str, str],
    information_requests: tuple[InformationRequest, ...],
    since: datetime | None,
) -> None:
    batch: list[IssueSnapshot] = []
    seen = 0
    skipped = 0
    last_reported = 0
    cohort = store.get_cohort(options.run_id)
    references_by_number = {}
    if not cohort:
        references = source.list_closed_issues(limit=options.limit, since=since)
        references_by_number = {reference.number: reference for reference in references}
        cohort = tuple(references_by_number)
        store.set_cohort(options.run_id, options.repository, cohort)

    for issue_number in cohort:
        seen += 1
        if not options.force and store.has_result(options.run_id, options.repository, issue_number):
            skipped += 1
            continue

        issue = store.get_run_snapshot(options.run_id, options.repository, issue_number)
        if issue is None and not options.refresh_cache:
            issue = store.get_cached_issue(options.repository, issue_number)
        if issue is None:
            reference = references_by_number.get(issue_number) or source.get_issue(issue_number)
            issue = source.hydrate_issue(reference)
            store.cache_issue(issue)
        store.bind_run_snapshot(options.run_id, issue)
        batch.append(issue)

        if len(batch) >= options.concurrency:
            await _classify_batch(
                batch,
                options=options,
                store=store,
                classifier=classifier,
                label_catalog=label_catalog,
                information_requests=information_requests,
            )
            batch.clear()
            print(f"Processed {seen - skipped}/{seen} issues ({skipped} resumed).")
            last_reported = seen

    if batch:
        await _classify_batch(
            batch,
            options=options,
            store=store,
            classifier=classifier,
            label_catalog=label_catalog,
            information_requests=information_requests,
        )
    if seen != last_reported or seen == 0:
        print(f"Processed {seen - skipped}/{seen} issues ({skipped} resumed).")


async def _classify_batch(
    issues: list[IssueSnapshot],
    *,
    options: argparse.Namespace,
    store: ResultStore,
    classifier: CopilotClassifier,
    label_catalog: dict[str, str],
    information_requests: tuple[InformationRequest, ...],
) -> None:
    await asyncio.gather(
        *(
            _classify_one(
                issue,
                run_id=options.run_id,
                store=store,
                classifier=classifier,
                label_catalog=label_catalog,
                information_requests=information_requests,
            )
            for issue in issues
        )
    )


async def _classify_one(
    issue: IssueSnapshot,
    *,
    run_id: str,
    store: ResultStore,
    classifier: CopilotClassifier,
    label_catalog: dict[str, str],
    information_requests: tuple[InformationRequest, ...],
) -> None:
    try:
        prediction = await classifier.classify(
            issue,
            label_catalog=label_catalog,
            information_requests=information_requests,
        )
    except RecoverableClassificationError as exc:
        message = f"{type(exc).__name__}: {exc}"
        store.record_error(run_id=run_id, issue=issue, error=message)
        print(f"#{issue.number}: {message}", file=sys.stderr)
        return

    actual = infer_actual_outcome(issue)
    comparison = compare_decision(prediction, actual)
    rendered_request = (
        render_information_request(prediction.information_request_ids, information_requests)
        if prediction.disposition is Disposition.REQUEST_INFORMATION
        else None
    )
    store.record_result(
        run_id=run_id,
        issue=issue,
        prediction=prediction,
        actual=actual,
        comparison=comparison,
        rendered_information_request=rendered_request,
    )


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="issue-triage",
        description="Evaluate Copilot triage against historical GitHub issues.",
    )
    subparsers = parser.add_subparsers(dest="command")
    evaluate = subparsers.add_parser("evaluate", help="run a historical evaluation")
    default_output = project_root() / "scripts" / "issue_triage" / ".triage-results"
    evaluate.add_argument("--repository", default="microsoft/vscode-python")
    evaluate.add_argument("--limit", type=int, default=1000)
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--since")
    evaluate.add_argument(
        "--model",
        required=True,
        help="concrete Copilot model ID; auto is rejected for reproducible runs",
    )
    evaluate.add_argument("--concurrency", type=int, default=2)
    evaluate.add_argument("--timeout", type=float, default=180)
    evaluate.add_argument("--retries", type=int, default=1)
    evaluate.add_argument("--agent", type=Path, default=default_agent_path())
    evaluate.add_argument("--information-requests", type=Path, default=default_catalog_path())
    evaluate.add_argument("--database", type=Path, default=default_output / "triage.sqlite3")
    evaluate.add_argument("--output-directory", type=Path, default=default_output)
    evaluate.add_argument("--refresh-cache", action="store_true")
    evaluate.add_argument("--force", action="store_true")

    ml_evaluate = subparsers.add_parser(
        "ml-evaluate",
        help="train and evaluate deterministic supervised-ML baselines",
    )
    ml_output = default_output / "ml"
    ml_evaluate.add_argument("--repository", default="microsoft/vscode-python")
    ml_evaluate.add_argument("--limit", type=int, default=1000)
    ml_evaluate.add_argument(
        "--all",
        action="store_true",
        dest="all_issues",
        help="bulk-download every closed issue, event, and relevant comment",
    )
    ml_evaluate.add_argument("--since")
    ml_evaluate.add_argument(
        "--approach",
        choices=("tfidf", "sentence-transformer", "both"),
        default="tfidf",
    )
    ml_evaluate.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    ml_evaluate.add_argument("--dataset", type=Path, default=ml_output / "dataset.jsonl")
    ml_evaluate.add_argument(
        "--snapshot-archive",
        type=Path,
        default=ml_output / "closed-issues.snapshot.jsonl",
    )
    ml_evaluate.add_argument("--database", type=Path, default=default_output / "triage.sqlite3")
    ml_evaluate.add_argument("--output-directory", type=Path, default=ml_output)
    ml_evaluate.add_argument("--refresh-cache", action="store_true")
    ml_evaluate.add_argument("--rebuild-dataset", action="store_true")

    llm_classify = subparsers.add_parser(
        "llm-classify",
        help="compare classification-only LLM prompts on a seeded local cohort",
    )
    llm_output = default_output / "llm-classification"
    llm_classify.add_argument(
        "--model",
        required=True,
        help="concrete Copilot model ID; auto is rejected for reproducible runs",
    )
    llm_classify.add_argument("--cohort-size", type=int, default=300)
    llm_classify.add_argument("--seed", type=int, default=42)
    llm_classify.add_argument("--concurrency", type=int, default=4)
    llm_classify.add_argument("--timeout", type=float, default=180)
    llm_classify.add_argument("--retries", type=int, default=1)
    llm_classify.add_argument(
        "--dataset",
        type=Path,
        default=ml_output / "dataset.jsonl",
    )
    llm_classify.add_argument("--output-directory", type=Path, default=llm_output)

    action_triage = subparsers.add_parser(
        "action-triage",
        help="generate outcome-oriented first responses on a seeded local cohort",
    )
    action_output = default_output / "action-triage"
    action_triage.add_argument("--model", required=True)
    action_triage.add_argument("--cohort-size", type=int, default=300)
    action_triage.add_argument("--seed", type=int, default=42)
    action_triage.add_argument("--concurrency", type=int, default=4)
    action_triage.add_argument("--timeout", type=float, default=180)
    action_triage.add_argument("--retries", type=int, default=1)
    action_triage.add_argument("--dataset", type=Path, default=ml_output / "dataset.jsonl")
    action_triage.add_argument("--output-directory", type=Path, default=action_output)
    action_triage.add_argument("--agent", type=Path, default=default_action_agent_path())
    action_triage.add_argument(
        "--information-requests",
        type=Path,
        default=default_catalog_path(),
    )
    action_triage.add_argument("--guidance", type=Path, default=default_guidance_path())
    return parser


def _validate_options(options: argparse.Namespace) -> None:
    options.model = options.model.strip()
    if options.limit < 1:
        raise ValueError("--limit must be at least 1")
    if options.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if options.timeout <= 0:
        raise ValueError("--timeout must be greater than 0")
    if options.retries < 0:
        raise ValueError("--retries cannot be negative")
    if not options.model:
        raise ValueError("--model must be a non-empty concrete model ID")
    if options.model.lower() == "auto":
        raise ValueError("--model must be a concrete model ID, not auto")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", options.run_id):
        raise ValueError(
            "--run-id must start with an alphanumeric character and contain only "
            "letters, numbers, dots, underscores, or hyphens"
        )


def _validate_ml_options(options: argparse.Namespace) -> None:
    if not options.all_issues and options.limit < 12:
        raise ValueError("--limit must be at least 12 for chronological ML splits")


def _validate_llm_options(options: argparse.Namespace) -> None:
    options.model = options.model.strip()
    if not options.model or options.model.lower() == "auto":
        raise ValueError("--model must be a non-empty concrete model ID, not auto")
    if options.cohort_size < 1:
        raise ValueError("--cohort-size must be at least 1")
    if options.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if options.timeout <= 0:
        raise ValueError("--timeout must be greater than 0")
    if options.retries < 0:
        raise ValueError("--retries cannot be negative")


def _parse_since(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("--since must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("--since must include a timezone")
    return parsed


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    source_directory = Path(__file__).parent
    for path in sorted(source_directory.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()

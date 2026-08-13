"""Read-only GitHub issue ingestion backed by PyGithub."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import Any, cast

from github import Auth, Github
from github.Issue import Issue
from github.IssueComment import IssueComment as GitHubIssueComment
from github.IssueEvent import IssueEvent

from .models import IssueComment, IssueSnapshot, TimelineEvent


class GitHubIssueSource:
    """Retrieve closed issues and their historical context without mutating GitHub."""

    def __init__(self, repository: str, token: str | None = None) -> None:
        self.repository = repository
        self._github = Github(auth=Auth.Token(token or resolve_github_token()), per_page=100)
        self._repository = self._github.get_repo(repository)

    def close(self) -> None:
        """Release the underlying HTTP resources."""
        self._github.close()

    def get_label_catalog(self) -> dict[str, str]:
        """Return current repository labels and descriptions."""
        return {label.name: label.description or "" for label in self._repository.get_labels()}

    def get_authenticated_login(self) -> str:
        """Return the GitHub login whose token performs API reads."""
        return self._github.get_user().login

    def get_issue(self, issue_number: int) -> Issue:
        """Return one issue reference by number."""
        return self._repository.get_issue(issue_number)

    def list_closed_issues(
        self, *, limit: int | None, since: datetime | None = None
    ) -> list[Issue]:
        """Materialize a deduplicated issue cohort before slow classification starts."""
        if since is None:
            issues = self._repository.get_issues(
                state="closed",
                sort="created",
                direction="desc",
            )
        else:
            issues = self._repository.get_issues(
                state="closed",
                sort="created",
                direction="desc",
                since=since,
            )
        collected: list[Issue] = []
        seen_numbers: set[int] = set()
        for issue in issues:
            if is_pull_request(issue) or issue.number in seen_numbers:
                continue
            seen_numbers.add(issue.number)
            collected.append(issue)
            if limit is None and len(collected) % 1_000 == 0:
                print(f"Downloaded {len(collected)} closed issue records.", flush=True)
            if limit is not None and len(collected) >= limit:
                break
        return collected

    def iter_closed_issue_pages(
        self,
        *,
        start_cursor: str | None = None,
        completed_pages: int = 0,
    ) -> Iterator[tuple[str | None, tuple[Issue, ...]]]:
        """Yield pages from the closed-issue endpoint for resumable ingestion."""
        issues = self._repository.get_issues(
            state="closed",
            sort="created",
            direction="desc",
        )
        yield from _iter_rest_pages(
            issues,
            start_cursor=start_cursor,
            completed_pages=completed_pages,
        )

    def iter_issue_events(self) -> Iterator[IssueEvent]:
        """Yield repository-wide issue events through paginated bulk requests."""
        yield from self._repository.get_issues_events()

    def iter_issue_event_pages(
        self,
        *,
        start_cursor: str | None = None,
        completed_pages: int = 0,
    ) -> Iterator[tuple[str | None, tuple[IssueEvent, ...]]]:
        """Yield repository event pages for resumable ingestion."""
        events = self._repository.get_issues_events()
        yield from _iter_rest_pages(
            events,
            start_cursor=start_cursor,
            completed_pages=completed_pages,
        )

    def iter_issue_comments(self) -> Iterator[GitHubIssueComment]:
        """Yield repository-wide issue comments through paginated bulk requests."""
        yield from self._repository.get_issues_comments(
            sort="created",
            direction="asc",
        )

    def iter_issue_comment_pages(
        self,
        *,
        start_cursor: str | None = None,
        completed_pages: int = 0,
    ) -> Iterator[tuple[str | None, tuple[GitHubIssueComment, ...]]]:
        """Yield repository comment pages for resumable ingestion."""
        comments = self._repository.get_issues_comments(
            sort="created",
            direction="asc",
        )
        yield from _iter_rest_pages(
            comments,
            start_cursor=start_cursor,
            completed_pages=completed_pages,
        )

    def hydrate_issue(self, issue: Issue) -> IssueSnapshot:
        """Fetch timeline events and comments for a historical issue."""
        timeline = tuple(normalize_timeline_event(event) for event in issue.get_timeline())
        comments = tuple(normalize_issue_comment(comment) for comment in issue.get_comments())
        return self.build_snapshot(issue, timeline=timeline, comments=comments)

    def iter_issue_timeline(self, issue_number: int) -> Iterator[Any]:
        """Yield one issue's complete timeline, including comment events."""
        yield from self.get_issue(issue_number).get_timeline()

    def build_snapshot(
        self,
        issue: Issue,
        *,
        timeline: tuple[TimelineEvent, ...],
        comments: tuple[IssueComment, ...],
    ) -> IssueSnapshot:
        """Combine one issue with already-fetched bulk events and comments."""
        initial_title = next(
            (event.renamed_from for event in timeline if event.renamed_from),
            issue.title,
        )
        return IssueSnapshot(
            repository=self.repository,
            number=issue.number,
            title=initial_title,
            body=issue.body or "",
            author=issue.user.login if issue.user else "ghost",
            url=issue.html_url,
            created_at=_isoformat(issue.created_at),
            closed_at=_isoformat(issue.closed_at) if issue.closed_at else None,
            state_reason=issue.state_reason,
            final_labels=tuple(sorted(label.name for label in issue.labels)),
            timeline=timeline,
            comments=comments,
            content_source=(
                "github_current_body_reconstructed_initial_title"
                if initial_title != issue.title
                else "github_current_snapshot"
            ),
        )


def resolve_github_token() -> str:
    """Use environment or the active GitHub CLI identity without echoing its token."""
    for variable in ("GH_TOKEN", "GITHUB_TOKEN"):
        token = os.environ.get(variable)
        if token:
            return token

    try:
        completed = subprocess.run(
            ["gh", "auth", "token"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "GitHub CLI is not installed; install it and run `gh auth login`."
        ) from exc
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            "Could not read the active GitHub CLI identity; run `gh auth login`."
        ) from exc

    token = completed.stdout.strip()
    if not token:
        raise RuntimeError("GitHub CLI returned an empty authentication token.")
    return token


def normalize_timeline_event(event: Any) -> TimelineEvent:
    """Normalize a PyGithub timeline object, including fields exposed only as raw data."""
    raw_data = _stored_raw_data(event)
    raw_event = _string_keyed_dict(raw_data) or {}
    raw_actor = _string_keyed_dict(raw_event.get("actor")) or {}
    raw_label = _string_keyed_dict(raw_event.get("label")) or {}
    rename = _string_keyed_dict(raw_event.get("rename")) or {}
    label_name = raw_label.get("name")
    renamed_from = rename.get("from")
    return TimelineEvent(
        event=str(raw_event.get("event", "")),
        created_at=_normalized_timestamp(raw_event.get("created_at")),
        actor=str(raw_actor["login"]) if raw_actor.get("login") else None,
        label=str(label_name) if label_name else None,
        source_repository=_source_repository(raw_data),
        renamed_from=str(renamed_from) if renamed_from else None,
    )


def normalize_issue_comment(comment: Any) -> IssueComment:
    """Normalize a PyGithub issue comment."""
    user = getattr(comment, "user", None)
    created_at = comment.created_at
    return IssueComment(
        author=str(getattr(user, "login", None) or "ghost"),
        author_association=str(getattr(comment, "author_association", None) or "NONE"),
        body=str(getattr(comment, "body", None) or ""),
        created_at=_isoformat(created_at),
    )


def normalize_timeline_comment(event: Any) -> IssueComment:
    """Normalize a comment embedded in an issue timeline."""
    raw_event = _stored_raw_data(event)
    user = _string_keyed_dict(raw_event.get("user")) or {}
    created_at = _normalized_timestamp(raw_event.get("created_at"))
    if created_at is None:
        raise ValueError("Timeline comment does not contain a created_at timestamp")
    return IssueComment(
        author=str(user.get("login") or "ghost"),
        author_association=str(raw_event.get("author_association") or "NONE"),
        body=str(raw_event.get("body") or ""),
        created_at=created_at,
    )


def timeline_event_type(event: Any) -> str:
    """Read an event type without completing its lazy API object."""
    return str(_stored_raw_data(event).get("event") or "")


def issue_event_number(event: Any) -> int | None:
    """Read an event's issue number without completing its lazy API object."""
    raw_event = _stored_raw_data(event)
    issue = _string_keyed_dict(raw_event.get("issue"))
    number = issue.get("number") if issue else None
    return number if isinstance(number, int) else None


def issue_event_id(event: Any) -> int:
    """Read an event ID without completing its lazy API object."""
    event_id = _stored_raw_data(event).get("id")
    if not isinstance(event_id, int):
        raise ValueError("Repository issue event does not contain an ID")
    return event_id


def issue_comment_number(comment: Any) -> int:
    """Read a repository comment's issue number from its populated URL."""
    issue_url = str(getattr(comment, "issue_url", ""))
    try:
        return int(issue_url.rsplit("/", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid issue comment URL: {issue_url!r}") from exc


def issue_comment_id(comment: Any) -> int:
    """Read a populated repository comment ID."""
    return int(comment.id)


def is_pull_request(issue: Any) -> bool:
    """Identify a pull request from its list payload without completing it."""
    raw_data = _stored_raw_data(issue)
    return "pull_request" in raw_data


def _stored_raw_data(value: Any) -> dict[str, object]:
    # PyGithub's public raw_data property completes list objects with an extra request.
    attributes = cast(dict[str, object], vars(value))
    raw_data = attributes.get("_rawData", attributes.get("raw_data", {}))
    return _string_keyed_dict(raw_data) or {}


def _iter_rest_pages(
    pages: object,
    *,
    start_cursor: str | None,
    completed_pages: int,
) -> Iterator[tuple[str | None, tuple[Any, ...]]]:
    attributes = cast(dict[str, object], vars(pages))
    fetch_value = getattr(pages, "_fetchNextPage", None)
    if not callable(fetch_value):
        raise RuntimeError("PyGithub paginated list does not expose page fetching")
    fetch_next_page = cast(Callable[[], list[Any]], fetch_value)
    if start_cursor:
        attributes["_PaginatedList__nextUrl"] = start_cursor
        attributes["_PaginatedList__nextParams"] = {}
    elif completed_pages:
        raw_parameters = attributes.get("_PaginatedList__nextParams")
        if not isinstance(raw_parameters, dict):
            raise RuntimeError("PyGithub paginated list does not expose request parameters")
        next_parameters = cast(dict[str, object], raw_parameters)
        next_parameters["page"] = completed_pages
        fetch_next_page()

    while page := tuple(fetch_next_page()):
        next_cursor_value = attributes.get("_PaginatedList__nextUrl")
        next_cursor = next_cursor_value if isinstance(next_cursor_value, str) else None
        yield next_cursor, page
        if next_cursor is None:
            return


def _source_repository(raw_data: object) -> str | None:
    raw = _string_keyed_dict(raw_data)
    if raw is None:
        return None
    source = _string_keyed_dict(raw.get("source"))
    if source is None:
        return None
    issue = _string_keyed_dict(source.get("issue"))
    if issue is None:
        return None

    repository = _string_keyed_dict(issue.get("repository"))
    if repository is not None:
        full_name = repository.get("full_name")
        if isinstance(full_name, str):
            return full_name

    repository_url = issue.get("repository_url")
    marker = "/repos/"
    if isinstance(repository_url, str) and marker in repository_url:
        return repository_url.split(marker, 1)[1]
    return None


def _string_keyed_dict(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    untyped_values = cast(dict[object, object], value)
    return {key: entry for key, entry in untyped_values.items() if isinstance(key, str)}


def _isoformat(value: datetime) -> str:
    return value.isoformat()


def _normalized_timestamp(value: object) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return value if isinstance(value, str) else None

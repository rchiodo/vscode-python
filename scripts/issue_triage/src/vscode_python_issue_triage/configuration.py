"""Load evaluator configuration files and agent instructions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

ALLOWED_TRANSFER_TARGETS = frozenset(
    {
        "microsoft/debugpy",
        "microsoft/vscode",
        "microsoft/vscode-jupyter",
        "microsoft/vscode-pylance-release",
        "microsoft/vscode-python-environments",
    }
)
ACTION_TRIAGE_SKILLS = (
    "python-triage-spam-detection",
    "python-triage-upstream-guidance",
    "python-triage-ecosystem-routing",
    "python-triage-information-request",
)
ACTION_ROUTING_TARGETS = frozenset(
    {
        "linting",
        "pylance",
        "python",
        "python-environments",
        "third-party",
        "unknown",
        "vscode-python",
    }
)


@dataclass(frozen=True)
class InformationRequest:
    """A reusable information request the classifier can select."""

    id: str
    description: str
    message: str


@dataclass(frozen=True)
class Guidance:
    """A reusable response for non-product behavior."""

    id: str
    description: str
    message: str


@dataclass(frozen=True)
class AgentDefinition:
    """A custom agent name and authored prompt loaded from Markdown."""

    name: str
    prompt: str


def project_root() -> Path:
    """Return the repository root from an editable or source checkout."""
    return Path(__file__).resolve().parents[4]


def default_agent_path() -> Path:
    """Return the repository's triage-agent definition."""
    return project_root() / ".github" / "agents" / "python-issue-triage.agent.md"


def default_action_agent_path() -> Path:
    """Return the outcome-oriented triage-agent definition."""
    return project_root() / ".github" / "agents" / "python-issue-action-triage.agent.md"


def default_catalog_path() -> Path:
    """Return the reusable information-request catalog."""
    return project_root() / "scripts" / "issue_triage" / "config" / "information-requests.json"


def default_guidance_path() -> Path:
    """Return the reusable upstream-guidance catalog."""
    return project_root() / "scripts" / "issue_triage" / "config" / "guidance.json"


def classification_skills_directory() -> Path:
    """Return the repository skill directory used by classification sessions."""
    return project_root() / ".github" / "skills"


def load_agent_definition(path: Path) -> AgentDefinition:
    """Load the name and Markdown body from an agent file."""
    content = path.read_text(encoding="utf-8")
    if not content.startswith("---\n"):
        return AgentDefinition(name=_agent_name_from_path(path), prompt=content.strip())
    marker = content.find("\n---\n", 4)
    if marker == -1:
        raise ValueError(f"Agent file has unterminated frontmatter: {path}")
    frontmatter = content[4:marker]
    name = next(
        (
            line.partition(":")[2].strip()
            for line in frontmatter.splitlines()
            if line.partition(":")[0].strip() == "name"
        ),
        _agent_name_from_path(path),
    )
    if not name:
        raise ValueError(f"Agent file has an empty name: {path}")
    prompt = content[marker + 5 :].strip()
    if not prompt:
        raise ValueError(f"Agent file has an empty prompt: {path}")
    return AgentDefinition(name=name, prompt=prompt)


def load_information_requests(path: Path) -> tuple[InformationRequest, ...]:
    """Load and validate reusable information requests."""
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Information request catalog must be an array: {path}")
    raw_entries = cast(list[object], raw)

    requests: list[InformationRequest] = []
    seen_ids: set[str] = set()
    for untyped_entry in raw_entries:
        if not isinstance(untyped_entry, dict):
            raise ValueError("Each information request must be an object")
        untyped_values = cast(dict[object, object], untyped_entry)
        entry: dict[str, object] = {}
        for key, value in untyped_values.items():
            if not isinstance(key, str):
                raise ValueError("Information request keys must be strings")
            entry[key] = value
        request = InformationRequest(
            id=_required_string(entry, "id"),
            description=_required_string(entry, "description"),
            message=_required_string(entry, "message"),
        )
        if request.id in seen_ids:
            raise ValueError(f"Duplicate information request ID: {request.id}")
        seen_ids.add(request.id)
        requests.append(request)
    return tuple(requests)


def load_guidance(path: Path) -> tuple[Guidance, ...]:
    """Load and validate reusable non-product guidance."""
    entries = _load_catalog_entries(path)
    return tuple(
        Guidance(
            id=_required_string(entry, "id"),
            description=_required_string(entry, "description"),
            message=_required_string(entry, "message"),
        )
        for entry in entries
    )


def render_information_request(
    request_ids: tuple[str, ...], catalog: tuple[InformationRequest, ...]
) -> str:
    """Render the exact comment that a predicted information request would post."""
    by_id = {request.id: request for request in catalog}
    messages = [by_id[request_id].message for request_id in request_ids]
    return (
        "Thanks for reporting this issue. To classify and investigate it, please provide:\n\n"
        + "\n".join(f"- {message}" for message in messages)
    )


def _load_catalog_entries(path: Path) -> tuple[dict[str, object], ...]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Catalog must be an array: {path}")
    entries: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for untyped_entry in cast(list[object], raw):
        if not isinstance(untyped_entry, dict):
            raise ValueError(f"Each catalog entry must be an object: {path}")
        values = {
            key: value
            for key, value in cast(dict[object, object], untyped_entry).items()
            if isinstance(key, str)
        }
        entry_id = _required_string(values, "id")
        if entry_id in seen_ids:
            raise ValueError(f"Duplicate catalog ID: {entry_id}")
        seen_ids.add(entry_id)
        entries.append(values)
    return tuple(entries)


def _required_string(entry: dict[str, object], key: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Information request {key!r} must be a non-empty string")
    return value.strip()


def _agent_name_from_path(path: Path) -> str:
    name = path.name
    return name[: -len(".agent.md")] if name.endswith(".agent.md") else path.stem

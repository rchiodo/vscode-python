from __future__ import annotations

from vscode_python_issue_triage.classifier import (
    build_custom_agent_config,
    classification_label_catalog,
)
from vscode_python_issue_triage.configuration import (
    ACTION_TRIAGE_SKILLS,
    AgentDefinition,
    classification_skills_directory,
)


def test_classification_label_catalog_excludes_operational_labels() -> None:
    labels = classification_label_catalog(
        {
            "bug": "Probable bug",
            "area-testing": "Testing",
            "info-needed": "Needs information",
            "skip tests": "PR-only",
            "~spam": "Transient spam marker",
        }
    )

    assert labels == {
        "bug": "Probable bug",
        "area-testing": "Testing",
        "~spam": "Transient spam marker",
    }


def test_custom_agent_preloads_all_classification_skills() -> None:
    agent = build_custom_agent_config(
        agent=AgentDefinition(name="test-agent", prompt="Classify."),
        agent_prompt=None,
        skill_names=ACTION_TRIAGE_SKILLS,
    )
    skills = agent.get("skills")
    prompt = agent.get("prompt")

    assert skills == list(ACTION_TRIAGE_SKILLS)
    assert isinstance(prompt, str)
    assert all(name in prompt for name in ACTION_TRIAGE_SKILLS)
    assert all(
        (classification_skills_directory() / name / "SKILL.md").exists()
        for name in ACTION_TRIAGE_SKILLS
    )

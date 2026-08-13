from __future__ import annotations

import pytest

from vscode_python_issue_triage.cli import main


@pytest.mark.parametrize("model", ["", "   ", "auto", " AUTO "])
def test_cli_rejects_non_concrete_model(model: str, capsys: pytest.CaptureFixture[str]) -> None:
    result = main(["evaluate", "--run-id", "test", "--model", model])

    assert result == 1
    assert "concrete model" in capsys.readouterr().err


def test_ml_cli_rejects_too_small_cohort(capsys: pytest.CaptureFixture[str]) -> None:
    result = main(["ml-evaluate", "--limit", "11"])

    assert result == 1
    assert "at least 12" in capsys.readouterr().err


@pytest.mark.parametrize("model", ["", "  ", "auto"])
def test_llm_cli_rejects_non_concrete_model(model: str, capsys: pytest.CaptureFixture[str]) -> None:
    result = main(["llm-classify", "--model", model])

    assert result == 1
    assert "concrete model" in capsys.readouterr().err


def test_action_cli_rejects_non_concrete_model(capsys: pytest.CaptureFixture[str]) -> None:
    result = main(["action-triage", "--model", "auto"])

    assert result == 1
    assert "concrete model" in capsys.readouterr().err

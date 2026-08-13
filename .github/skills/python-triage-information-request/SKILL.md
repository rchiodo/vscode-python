---
name: python-triage-information-request
description: Select the minimum formulaic information needed to choose a triage action and route.
---

# Targeted information requests

Choose `request_information` only when missing facts prevent distinguishing spam, upstream guidance,
or a plausible Microsoft Python-tooling issue.

Select the smallest set of catalog requests that resolves the uncertainty:

- `expected_actual_behavior` when the claimed problem or desired outcome is unclear.
- `reproduction_steps` when no repeatable path to the behavior is described.
- `environment_details` and `python_version` for interpreter/environment ambiguity.
- `python_logs` for activation, interpreter, terminal, or general extension integration.
- `language_server_logs` for completion, navigation, analysis, or Pylance behavior.
- `testing_logs` for discovery/execution behavior.
- `debugger_logs` for debug-session behavior.
- `minimal_reproduction` when user code or a package may be responsible.
- `extension_bisect` when another extension may own the behavior.

Do not request every missing field. If the action is already clear, acknowledge or provide guidance
instead of blocking classification on investigation details.

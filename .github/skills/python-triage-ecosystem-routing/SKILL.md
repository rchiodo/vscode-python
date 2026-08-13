---
name: python-triage-ecosystem-routing
description: Route actionable and guidance cases within the Python tooling ecosystem.
---

# Ecosystem routing

Select one internal route independently from the user-facing action:

- `vscode-python`: interpreter selection, activation, terminals, testing integration, extension
  commands/settings, and general Python extension behavior.
- `pylance`: completion, navigation, type checking, semantic analysis, and Pylance diagnostics.
- `python-environments`: environment discovery/management behavior owned by the standalone Python
  Environments extension.
- `linting`: linter/formatter integration or an upstream linting/formatting tool.
- `python`: Python language, runtime, standard library, packaging fundamentals, or interpreter
  semantics.
- `third-party`: third-party packages, environment managers, tools, or user dependencies.
- `unknown`: spam or insufficient evidence to infer a route.

An issue can be actionable even if another Microsoft repository ultimately owns it. Route it
internally and still acknowledge it. User declarations such as "bug," "feature request," template
headings, or proposed labels are useful intent hints but are not authoritative and never override
the described behavior.

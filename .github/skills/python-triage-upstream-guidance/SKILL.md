---
name: python-triage-upstream-guidance
description: Distinguish Microsoft Python-tooling problems from Python, package, and user-code behavior.
---

# Product issue versus guidance

Choose `provide_guidance` when the described behavior is expected outside Microsoft Python tooling
and no extension integration failure is alleged.

Common guidance cases:

- Python syntax, scoping, imports, exceptions, standard-library semantics, or runtime behavior.
- A third-party package's API, installation, dependency resolution, or package-specific failure.
- User code producing the reported result independently of VS Code.
- A linter or formatter enforcing its own documented rule correctly.

Choose `acknowledge` when a plausible integration, UI, configuration, process-launch, interpreter
selection, diagnostics transport, testing, debugging, Pylance, or environment-management behavior
may require a Microsoft product change or investigation.

Do not require proof of a product defect before acknowledging. If the report could be either
upstream behavior or product integration and one targeted fact would decide, use
`request_information`.

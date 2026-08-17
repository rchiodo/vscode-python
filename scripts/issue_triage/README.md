# vscode-python issue triage evaluator

This read-only prototype asks a repository-defined Copilot agent to classify closed issues using
only each issue's title and body snapshot. It then reconstructs the historical triage outcome from
labels, timeline events, and maintainer comments, and measures the distance between the two.

The prototype does not comment on, label, close, or transfer GitHub issues.

## Prerequisites

- Python 3.11 or newer and [uv](https://docs.astral.sh/uv/)
- [GitHub CLI](https://cli.github.com/) authenticated with `gh auth login`
- GitHub Copilot CLI installed and authenticated for the same developer

PyGithub reads the token from `GH_TOKEN`, `GITHUB_TOKEN`, or `gh auth token`, in that order. The
Copilot SDK uses the developer's existing Copilot CLI authentication. Each run validates both
identities, prints the selected logins, and stores them with the run configuration.

## Run a benchmark

From the repository root:

```powershell
uv sync --project scripts\issue_triage
uv run --project scripts\issue_triage issue-triage evaluate `
    --repository microsoft/vscode-python `
    --model gpt-5.4 `
    --limit 25 `
    --run-id baseline-25
```

Start with a small run before increasing `--limit` into the thousands. A run ID is resumable:
successful issues already recorded under that ID are skipped. Use `--force` to classify them again.
The issue cohort is materialized and deduplicated before Copilot processing starts so a long run
does not interleave GitHub pagination with model calls. Its ordered membership and exact input
snapshots are persisted with the run, so later resumes and cache refreshes cannot change the sample
or the text behind an existing prediction.

Useful options:

- `--since 2025-01-01T00:00:00Z` limits the closed-issue listing by update time.
- `--model MODEL` selects a concrete Copilot model and is required. `auto` is rejected so a resumed
  benchmark cannot silently switch models; each prediction also records the SDK response model.
- `--agent PATH` tests a revised `.agent.md` without replacing the repository default.
- `--database PATH` and `--output-directory PATH` relocate persistent artifacts.
- `--refresh-cache` refetches issue timelines and comments from GitHub.

## Outputs

The default `.triage-results` directory contains:

- `triage.sqlite3`: cached issue snapshots, run configuration, predictions, actual outcomes, errors,
  and comparisons. This makes large runs resumable and keeps historical inputs stable while prompts
  are tuned. Prompt, information-catalog, label-catalog, implementation, SDK, and dependency
  versions are recorded; a run ID cannot resume if its effective configuration changed.
- `<run-id>.jsonl`: one reviewable record per issue, including the proposed labels and disposition,
  the exact information-request comment that would be posted, historical outcome, and distance
  metrics.
- `<run-id>.summary.json`: aggregate classification and disposition accuracy, label and information
  request F1, exact-match rates, overall score, and a classification confusion matrix.

The historical extractor recognizes transient labels such as `~spam`, which are removed when an
issue is closed and therefore cannot be recovered from final labels alone. Information requests are
mapped to the reusable catalog in `config/information-requests.json`; the classifier selects catalog
IDs rather than inventing one-off responses.

GitHub's issue API returns the current body, not prior body revisions. The evaluator reconstructs an
original title when the timeline contains rename events, but marks every exported record with
`content_source` because it cannot prove that the body was never edited after opening. A strict
opening-text experiment should use an archived `issues.opened` payload as its source or exclude
reports without such an archive.

GitHub's transferred timeline event identifies where an incoming issue came from, but not a
trustworthy outgoing destination. Historical transfer outcomes are therefore left unknown rather
than inferred from repository links in comments. Transfer predictions are still emitted for manual
review and can be evaluated once an authoritative transfer archive is supplied.

Each classification attempt uses an isolated, tool-free Copilot session. The session and its local
history are permanently deleted after the result is captured.

## Tune the system

Edit `.github/agents/python-issue-triage.agent.md` for classification policy and
`config/information-requests.json` for formulaic follow-up questions. Use a new run ID for each
prompt revision, then compare the summary and per-issue errors. Keep a held-out issue range that is
not used while tuning to avoid optimizing only for known historical decisions.

## Supervised ML baselines

The ML workflow builds one immutable dataset and compares deterministic classifiers on identical
chronological splits:

- Oldest 70%: model training.
- Next 15%: multi-label threshold tuning and approach selection.
- Newest 15%: untouched holdout reporting.

Install the lightweight TF-IDF stack and run a baseline:

```powershell
uv sync --project scripts\issue_triage --extra ml
uv run --project scripts\issue_triage issue-triage ml-evaluate `
    --repository microsoft/vscode-python `
    --all `
    --approach tfidf `
    --rebuild-dataset
```

The dataset is reused on later commands unless `--rebuild-dataset` is passed. To compare TF-IDF
with sentence-transformer embeddings on exactly the same records and splits:

```powershell
uv sync --project scripts\issue_triage --extra embeddings
uv run --project scripts\issue_triage issue-triage ml-evaluate `
    --repository microsoft/vscode-python `
    --approach both
```

ML artifacts are written under `.triage-results/ml`:

- `dataset.jsonl` and `dataset.manifest.json`: immutable examples and provenance.
- `closed-issues.snapshot.jsonl`: the complete local source archive with normalized label events
  and relevant comments used to infer historical outcomes.
- `closed-issues.snapshot.issues.jsonl`: the resumable base-issue checkpoint. Each API page is
  synced before its atomic cursor advances.
- `closed-issues.snapshot.history.jsonl`: complete per-issue timelines, including comments,
  synced after every issue. Repository-wide event and comment endpoints are not used because
  GitHub truncates them at 30,000 records. Re-running resumes at the first missing issue.
- `<approach>.model.joblib`: fitted encoder, classifiers, thresholds, and split membership.
- `<approach>.predictions.jsonl`: holdout predictions alongside historical outcomes.
- `<approach>.metrics.json`: tuning and holdout classification, disposition, label, and information
  request metrics.

### Classification-only LLM prompt comparison

Compare three fixed prompts on one seed-42 random sample from the manifest-verified local dataset:

```powershell
uv run --project scripts\issue_triage issue-triage llm-classify `
    --model gpt-5.4 `
    --cohort-size 300 `
    --seed 42
```

This command preserves the earlier bug-versus-feature experiment for comparison only. It does not
represent the recommended user-facing policy.

The command compares minimal definitions, repository-aware rules, and synthetic few-shot examples.
It writes the immutable cohort, resumable per-issue predictions, confusion matrices, per-class
metrics, and aggregate accuracy under `.triage-results/llm-classification`.

### Outcome-oriented action triage

Generate reviewable first responses without mutating GitHub:

```powershell
uv run --project scripts\issue_triage issue-triage action-triage `
    --model gpt-5.4 `
    --cohort-size 300 `
    --seed 42
```

The primary action is one of `close_spam`, `provide_guidance`, `acknowledge`, or
`request_information`. Bug versus feature is intentionally not part of this decision. A separate
internal route identifies `vscode-python`, `pylance`, `python-environments`, `linting`, `python`,
`third-party`, or `unknown`.

Every action session preloads these repository skills:

- `python-triage-spam-detection`
- `python-triage-upstream-guidance`
- `python-triage-ecosystem-routing`
- `python-triage-information-request`

The output includes an immutable cohort, exact decisions, rendered proposed responses, action and
route distributions, and a manifest binding model, skills, catalogs, runtime settings, SDK, dataset,
agent, and implementation. It deliberately does not report historical accuracy because existing
bug/feature labels do not provide trustworthy ground truth for these actions.

Acknowledgement responses identify the initial automated route and include the model's concise
rationale so a maintainer reviewing the issue can see why it was accepted.

To compare retrieval-augmented decisions on the same seeded cohort, add:

```powershell
uv run --project scripts\issue_triage issue-triage action-triage `
    --model gpt-5.4 `
    --cohort-size 20 `
    --seed 20260813 `
    --retrieval-count 5 `
    --output-directory scripts\issue_triage\.triage-results\action-triage-rag
```

Retrieval uses stateless hashed word and character features, so its representation is not fitted on
future reports. For each target, only issues created strictly earlier are eligible. Retrieved issues
and similarity scores are saved with every result. The agent may cite only those retrieved issue
numbers, and proposed responses link any citations it actually used.

- `comparison.json`: approach selected exclusively by tuning score, with holdout scores reported
  separately.

TF-IDF combines word and character n-grams. The embedding approach defaults to
`sentence-transformers/all-MiniLM-L6-v2`; use `--embedding-model` to choose another model. Both use
class-balanced logistic regression heads. Labels and information requests use independently trained
binary heads with thresholds chosen only from the tuning partition.

## Development

```powershell
uv sync --project scripts\issue_triage --extra dev
uv run --project scripts\issue_triage pytest
uv run --project scripts\issue_triage ruff check .
uv run --project scripts\issue_triage pyright
```

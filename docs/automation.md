# Automation: GitHub Actions + OpenRouter

This document describes the two GitHub Actions workflows that automate
the Bayesian reader pipeline and how to set them up. Everything here
is **additive** to the manual Claude Code flow — you can still run
every step locally the old way.

## Status matrix

| Phase | Workflow | LLM used? | Triggered how | Writes to main? |
|---|---|---|---|---|
| A — ingest | `.github/workflows/ingest.yml` | No | `issues.labeled: article` or manual | Yes (ingest commit) |
| B — draft claims | `.github/workflows/draft-claims.yml` | **OpenRouter** | Manual only | Yes (claims commit) |
| C — draft verification + apply | (not built yet) | Yes | Manual | No (will stage only) |

Phase C, D, E are designed in `docs/automation.md` but **not implemented**.
Each phase is independently operable.

## Phase A — ingest

**What it does**: when you open (or label) an issue with the `article`
label on mobile, GitHub runs this workflow and:

1. Calls `sync-issues` to import the URL into `knowledge_state/articles/<id>/record.json`
2. Calls `fetch-pending` to HTTP-fetch the page and write `canonical_text.txt`
3. Commits the result to `main` with a `chore(ingest)` message
4. Comments on the issue with the current acquisition status

**Secrets needed**: none. It uses the built-in `GITHUB_TOKEN` which
the `gh` CLI on the Ubuntu runner picks up automatically.

**LLM calls**: zero. Phase A is pure Python / stdlib.

**First run**: open any existing article issue, add/remove the
`article` label, and watch the Actions tab. Or use the "Run workflow"
button on the ingest workflow for a manual dispatch.

## Phase B — draft claims

**What it does**: given an article that Phase A has already ingested
(i.e. `canonical_text.txt` exists), this workflow:

1. Calls `scripts/auto_draft_claims.py`, which:
   - Reads `canonical_text.txt` and the active hypotheses (including
     `domain` and `meta_tags`)
   - Sends them to OpenRouter via `scripts/llm_client.py`
   - Receives a JSON claims draft from the model
   - Re-validates the draft against `bayesian_reader.validate_claims_payload`
   - Writes it to `knowledge_state/articles/<id>/_proposed_claims.json`
     (a gitignored scratch file)
2. Runs `bayesian_reader.py save-claims` to promote the draft into the
   tracked `claims.json`
3. Commits `claims.json` + `record.json` + `change_log.jsonl`
4. Optionally posts the claims as a comment on an issue you specify

**Secrets needed**: `OPENROUTER_API_KEY`. See setup below.

**Trigger**: **manual only**. You go to the Actions tab, pick
"draft-claims", click "Run workflow", and fill in:
- `article_id` (required, e.g. `06eb3b8e8732`)
- `issue_number` (optional, for the post-draft comment)
- `model` (optional, e.g. `anthropic/claude-3.5-sonnet`; defaults to
  the model chain in `llm_client.py`)
- `commit_claims` (optional, set to `false` to only produce the
  gitignored draft without running `save-claims`)

**LLM calls**: one per run. Cost varies by model; Claude 3.5 Sonnet
on a typical article is roughly $0.01–0.03.

**Anti-metaphor-collapse**: hard-enforced by `scripts/prompts/auto_draft_claims.md`:
- the drafter must only pick `hypothesis_candidates` from the list
  passed in the user message
- it must never invent new hypothesis IDs
- it must honor domain separation — e.g. an economy article cannot be
  attached to an AI hypothesis that shares a word like "grounding"
  (because "grounding" means different things in different domains)
- when in doubt the drafter is instructed to leave
  `hypothesis_candidates: []` and let a human decide later

The output is also re-validated server-side by `validate_claims_payload`
— the same validator the manual `save-claims` CLI uses — so any schema
drift causes the workflow to fail closed with
`_proposed_claims.broken.json` saved for inspection.

## Setup: one-time steps

### 1. Create an OpenRouter account

1. Visit https://openrouter.ai/
2. Sign in with GitHub/Google
3. Add credits ($5 is plenty for months of use at this volume)
4. Create an API key at https://openrouter.ai/keys
5. Copy the `sk-or-v1-...` key

### 2. Add the key to the repo

1. Go to `https://github.com/<owner>/<repo>/settings/secrets/actions`
2. Click **New repository secret**
3. Name: `OPENROUTER_API_KEY`
4. Value: the `sk-or-v1-...` key from step 1
5. Save

That's the only secret Phase B needs.

### 3. (Optional) Adjust the model chain

Edit `scripts/llm_client.py` → `DEFAULT_MODEL_CHAIN`. The default is:
- `anthropic/claude-3.5-sonnet`
- `openai/gpt-4o-mini` (fallback)
- `google/gemini-flash-1.5` (fallback)

You can also override per-run by passing the `model` input to the
workflow, which becomes `OPENROUTER_MODEL` inside the script.

### 4. (Optional) Local smoke test

Before running the workflow on a real article, you can verify the
OpenRouter path works locally:

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-v1-...
python scripts/auto_draft_claims.py --article-id <existing_article_id> --dry-run
```

The `--dry-run` flag prints the full system + user prompt without
spending any API credits. Once that looks right, run without
`--dry-run` to make a real call and get a draft.

## Operational notes

- **Concurrency**: both workflows use named concurrency groups so two
  simultaneous runs on the same files can't race each other.
- **Failure behavior**: Phase A tolerates fetch failures — the record
  still gets persisted with `fetch_failed` or `blocked` status and you
  can recover via `attach-manual` locally. Phase B fails loud: if the
  drafter returns invalid JSON or produces a schema-breaking draft,
  the workflow exits non-zero and the broken draft is uploaded as an
  artifact.
- **Drift from manual runs**: both workflows commit straight to `main`
  under the `github-actions[bot]` user. If you and the bot make
  conflicting changes, you'll get a normal merge conflict and resolve
  it the usual way.
- **Cost ceiling**: Phase B is manual-only to keep the spend visible.
  If you want a hard cap, create a low-budget API key on OpenRouter
  (they support per-key limits) and use only that key.

## What is explicitly NOT automated (yet)

- Creating new hypotheses: always human
- Choosing `meta_tags` for a new hypothesis: always human
- First-of-domain cold-start calibration: always human
- Cross-domain claim attachment: always human (Phase C will enforce
  this as a hard escalation)
- Running `apply-verification`: Phase C territory, still manual
- Merging to main on any band-boundary crossing: Phase C territory

These are the guardrails from the Phase 0–5 plan. Automation
inherits the caution; it does not dissolve it.

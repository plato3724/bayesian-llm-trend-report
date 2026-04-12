# Automation: GitHub Actions + OpenRouter

This document describes the two GitHub Actions workflows that automate
the Bayesian reader pipeline and how to set them up. Everything here
is **additive** to the manual Claude Code flow — you can still run
every step locally the old way.

## Status matrix

| Phase | Workflow | LLM used? | Triggered how | Writes to main? |
|---|---|---|---|---|
| A — ingest | `.github/workflows/ingest.yml` | No | `issues.labeled: article` or manual | Yes (ingest commit) |
| B — draft claims | `.github/workflows/draft-claims.yml` | **OpenRouter** | **Auto**: `workflow_run` after ingest; also manual dispatch | Yes (claims commit) |
| C — draft + stage + apply verification | `.github/workflows/draft-verification.yml` | **OpenRouter** | **Auto**: `workflow_run` after draft-claims; also manual dispatch | Yes (stage + apply commit when guard passes) |

All three phases chain automatically end-to-end: labeling an issue on
mobile kicks off `ingest` → on success `draft-claims` → on success
`draft-verification`. Each phase is also independently operable via
manual `workflow_dispatch` for re-runs, backfills, or model overrides.

Phase D and E are designed in this doc but **not implemented**.

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

1. **Resolves targets** — either the single `article_id` from a manual
   dispatch, or every article currently at stage `extract_claims`
   (returned by `bayesian_reader.py queue --stage extract_claims`) on
   a chained run
2. For each target, calls `scripts/auto_draft_claims.py`, which:
   - Reads `canonical_text.txt` and the active hypotheses (including
     `domain` and `meta_tags`)
   - Sends them to OpenRouter via `scripts/llm_client.py`
   - Receives a JSON claims draft from the model
   - Re-validates the draft against `bayesian_reader.validate_claims_payload`
   - Writes it to `knowledge_state/articles/<id>/_proposed_claims.json`
     (a gitignored scratch file)
3. Runs `bayesian_reader.py save-claims` to promote the draft into the
   tracked `claims.json`
4. After the loop, commits the batch (`claims.json` + `record.json` +
   `change_log.jsonl`) under `github-actions[bot]`

**Secrets needed**: `OPENROUTER_API_KEY`. See setup below.

**Triggers**:

- **Auto (chained)** — a `workflow_run` trigger fires this workflow
  whenever the `ingest-article` workflow completes successfully. In
  chained mode the workflow scans the queue and drafts claims for
  *every* article waiting at `extract_claims`, one at a time. A single
  failing article is logged as a warning and the batch continues; the
  workflow only exits non-zero if nothing succeeded.
- **Manual dispatch** — you can still pick a single `article_id`
  yourself from the Actions tab for re-runs or model overrides.
  Manual mode fails fast on the first error.

**Manual inputs**:
- `article_id` (required, e.g. `06eb3b8e8732`)
- `issue_number` (optional, for the post-draft comment)
- `model` (optional, e.g. `openai/gpt-5.4`; overrides the configured
  default model for this run only)
- `commit_claims` (optional, set to `false` to only produce the
  gitignored draft without running `save-claims`)

**LLM calls**: one per article. Cost varies by model. A chained batch
spends one call per queued article, so the ceiling is bounded by how
many articles you label in a given window.

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

## Phase C — draft + stage verification (optional apply)

**What it does**: given an article that Phase B has already drafted
claims for (i.e. `claims.json` exists with
`claim_extraction_status: completed`), this workflow:

1. Calls `scripts/auto_draft_verification.py`, which:
   - Reads `canonical_text.txt`, `claims.json`, and the active
     hypotheses (including `domain`, `meta_tags`, `posterior_log_odds`,
     and `supporting_item_count`, so the model can self-assess cold
     starts and borderline hypotheses)
   - Sends them to OpenRouter via `scripts/llm_client.py`
   - Expects a single JSON object with two keys:
     `verification_draft` and `approval`
   - Re-validates both halves against
     `bayesian_reader.validate_verification_payload` and
     `validate_approval_payload` — the same validators
     `stage-verification` uses, so anything that passes here will pass
     staging
   - Runs `compute_phase_c_escalations(draft, approval, hypotheses)`,
     which deterministically checks for:
     - **cross_domain**: any item whose domain differs from its
       attached hypothesis's domain
     - **band_crossing**: any hypothesis whose simulated posterior
       (current `posterior_log_odds` plus the sum of ordinal
       contributions from verified / partially_verified items) would
       land in a different `POSTERIOR_BANDS` bucket
   - If any trigger fires and `approval.decision` is `auto_approved`,
     the script downgrades it to `needs_human` in place and appends a
     plain-text audit suffix to `overall_rationale` so the audit trail
     survives `stage-verification`'s field-filtering normalization
   - Writes `_proposed_draft.json` and `_proposed_approval.json` to the
     article folder (both gitignored under the `_proposed_*.json` glob)
2. Runs `bayesian_reader.py stage-verification` to promote the scratch
   pair into the tracked `verification_draft.json` and `approval.json`,
   appending a `stage_verification` entry to `change_log.jsonl`
3. Commits `verification_draft.json`, `approval.json`, `record.json`,
   `change_log.jsonl`
4. **Optionally**, if `auto_apply=true` AND the Phase C guard is clean
   AND `approval.decision == auto_approved`, runs `apply-verification`
   and rebuilds the HTML report, then commits `verification.json`,
   `hypotheses.json`, `synthesis_state.json`, `change_log.jsonl`, and
   `docs/index.html`
5. Optionally posts a concise status comment on a triggering issue

**Secrets needed**: `OPENROUTER_API_KEY`, same key as Phase B.

**Trigger**: **manual only**. Actions tab → "draft-verification" →
"Run workflow", with:
- `article_id` (required)
- `issue_number` (optional)
- `model` (optional)
- `commit_staged` (default `true`) — set to `false` to only produce the
  gitignored scratch files without touching tracked state
- `auto_apply` (default `false`) — opt-in to run apply-verification.
  Even when set to `true`, the apply step is gated by the guard: it
  only fires if the LLM marked the approval `auto_approved` AND the
  Python guard returned zero triggers

**LLM calls**: one per run. Phase C's prompt is longer than Phase B's
(the verification draft carries more fields) so a typical run is
roughly 2× the token cost of a Phase B call.

**Why two gates on apply**: the Phase C guard is deterministic Python,
and it has the last word. The LLM is allowed to self-select
`needs_human`, and Python additionally enforces:

1. **Cross-domain hard escalation** — catching the "grounding in AI
   means sensor/causal grounding, not emotional grounding in an
   economy article" class of failure. Phase 2 is still warn-only in
   `verification_cross_domain_warnings`, but Phase C hardens it to a
   forced downgrade for anything that went through the drafter.
2. **Band-boundary crossing** — the project's posterior bands
   (`very_unlikely` → `near_certain`) are the unit of editorial
   judgment. A single run that would move a hypothesis across a band
   boundary is always worth a human second look, even if the draft
   looks clean on every individual item.

## What Phase C still does NOT automate

- Creating new hypotheses: always human
- Picking `meta_tags` when a new hypothesis is created: always human
- Overriding a `needs_human` decision: always human
  (`override-approval` CLI remains local-only)
- First-of-domain cold-start calibration: the prompt steers the LLM
  toward `slight` and `partially_verified`, but a human should still
  audit the first 2–3 claims in any brand-new domain
- Phase D (review UI over staged approvals) and Phase E (auto-apply on
  human re-approve) are still unbuilt

The guardrails from the Phase 0–5 plan are all still in force. Phase C
automation inherits the caution; it does not dissolve it.

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

### 3. (Optional) Adjust the default model

The repository default is `openai/gpt-5.4`.

To change it globally in GitHub Actions without editing code, create a
repository variable named `OPENROUTER_DEFAULT_MODEL`.

For one-off runs, pass the `model` input to the workflow; this becomes
`OPENROUTER_MODEL` inside the script and overrides the default only for
that run.

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

- **Concurrency**: all three workflows use named concurrency groups so
  two simultaneous runs on the same article can't race each other.
- **Failure behavior**: Phase A tolerates fetch failures — the record
  still gets persisted with `fetch_failed` or `blocked` status and you
  can recover via `attach-manual` locally. Phase B and Phase C fail
  loud: if the drafter returns invalid JSON or produces a
  schema-breaking draft, the workflow exits non-zero and the broken
  draft is uploaded as an artifact
  (`_proposed_claims.broken.json` for Phase B,
  `_proposed_verification.broken.json` for Phase C).
- **Drift from manual runs**: all workflows commit straight to `main`
  under the `github-actions[bot]` user. If you and the bot make
  conflicting changes, you'll get a normal merge conflict and resolve
  it the usual way.
- **Cost ceiling**: Phase B and Phase C are manual-only to keep the
  spend visible. If you want a hard cap, create a low-budget API key
  on OpenRouter (they support per-key limits) and use only that key.

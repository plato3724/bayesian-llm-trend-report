"""Phase C: draft a (verification, approval) pair for an article using OpenRouter.

This is the Phase C counterpart to scripts/auto_draft_claims.py. It calls an
LLM to produce both a verification_draft and an approval record for an
article whose claims.json has already been saved, validates both against
the same schemas the CLI uses, runs the Phase C escalation guard, and
writes the results as gitignored scratch files so that a human (or the
draft-verification GitHub Actions workflow) can promote them into tracked
state via `bayesian_reader.py stage-verification` next.

What this script REFUSES to do:
  - write verification_draft.json / approval.json / verification.json
    directly (that's the job of stage-verification and apply-verification)
  - invent new hypotheses
  - log to change_log.jsonl (stage-verification does that)
  - touch hypotheses.json or run recompute_posteriors

Usage:
    python scripts/auto_draft_verification.py --article-id 719a74e44194
    python scripts/auto_draft_verification.py --article-id 719a74e44194 \
        --output-draft /tmp/draft.json --output-approval /tmp/approval.json
    python scripts/auto_draft_verification.py --article-id 719a74e44194 --dry-run

Environment:
    OPENROUTER_API_KEY   required, see scripts/llm_client.py
    OPENROUTER_MODEL     optional, overrides the default model chain
                          to a single model id, e.g.
                          'anthropic/claude-3.5-sonnet'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Make `scripts/` importable so we can reuse bayesian_reader helpers.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import bayesian_reader as br  # noqa: E402
from llm_client import complete_json, DEFAULT_MODEL_CHAIN  # noqa: E402


PROMPT_PATH = SCRIPT_DIR / "prompts" / "auto_draft_verification.md"


def load_system_prompt() -> str:
    if not PROMPT_PATH.exists():
        raise SystemExit(
            f"System prompt not found at {PROMPT_PATH}. "
            "Phase C expects this file to exist."
        )
    return PROMPT_PATH.read_text(encoding="utf-8")


def collect_active_hypotheses() -> list[dict[str, Any]]:
    """Return a trimmed view of active hypotheses for the Phase C prompt.

    Includes `posterior_log_odds` in addition to the Phase B fields so the
    LLM can self-assess whether a proposed contribution would visibly move
    a borderline hypothesis across a band boundary.
    """

    hypothesis_index = br.load_hypothesis_index()
    summaries: list[dict[str, Any]] = []
    for hypothesis in hypothesis_index.values():
        if hypothesis.get("status", "active") != "active":
            continue
        summaries.append(
            {
                "id": hypothesis.get("id"),
                "domain": br.item_domain(hypothesis),
                "meta_tags": br.item_meta_tags(hypothesis),
                "statement": hypothesis.get("statement"),
                "rationale": hypothesis.get("rationale"),
                "posterior_band": hypothesis.get("posterior_band"),
                "posterior_probability": hypothesis.get("posterior_probability"),
                "posterior_log_odds": hypothesis.get("posterior_log_odds"),
                "supporting_item_count": len(
                    hypothesis.get("supporting_items", []) or []
                ),
            }
        )
    return summaries


def build_user_message(
    *,
    article_id: str,
    record: dict[str, Any],
    canonical_text: str,
    claims_doc: dict[str, Any],
    active_hypotheses: list[dict[str, Any]],
    model_id: str,
) -> str:
    """Assemble the user-role message for the drafting call.

    Packaged as a single well-labeled JSON object so the LLM cannot
    hallucinate the structure. The article's own domain comes from its
    record.json so Phase C can enforce the cross-domain guard with a
    known value rather than trusting the LLM to echo it back correctly.
    """

    article_domain = br.item_domain(record)
    payload = {
        "article_id": article_id,
        "article_title": record.get("title"),
        "article_url": record.get("url"),
        "article_domain": article_domain,
        "model_id": model_id,
        "canonical_text": canonical_text,
        "claims": claims_doc.get("claims", []),
        "active_hypotheses": active_hypotheses,
        "instructions": (
            "Return a JSON object with exactly two top-level keys: "
            "'verification_draft' and 'approval'. Every claim in 'claims' "
            "must appear exactly once in verification_draft.items. Every "
            "draft item must also appear exactly once in "
            "approval.per_claim_decisions. Do not wrap the JSON in "
            "markdown fences. Do not emit any prose outside the JSON."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _write_broken(
    article_id: str,
    *,
    draft: Any,
    approval: Any,
    errors: dict[str, Any],
) -> Path:
    """Dump a broken draft to <article>/_proposed_verification.broken.json.

    One file rather than two because if either half is broken the pair is
    useless, and keeping them together makes the debug artifact easier to
    inspect. The `_proposed_*` glob is already gitignored.
    """
    broken_path = br.article_dir(article_id) / "_proposed_verification.broken.json"
    broken_path.write_text(
        json.dumps(
            {
                "article_id": article_id,
                "errors": errors,
                "draft": draft,
                "approval": approval,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return broken_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Use OpenRouter to draft a (verification_draft, approval) pair "
            "for an article whose claims.json is already saved. This does "
            "NOT commit anything to knowledge_state — it only writes the "
            "gitignored scratch files that stage-verification consumes next."
        ),
    )
    parser.add_argument("--article-id", required=True)
    parser.add_argument(
        "--output-draft",
        help=(
            "Where to write the verification draft. Defaults to "
            "knowledge_state/articles/<id>/_proposed_draft.json, "
            "already in .gitignore."
        ),
    )
    parser.add_argument(
        "--output-approval",
        help=(
            "Where to write the approval record. Defaults to "
            "knowledge_state/articles/<id>/_proposed_approval.json, "
            "already in .gitignore."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build the prompts and print them to stdout without calling "
            "the API. Useful for debugging the context that would be sent."
        ),
    )
    parser.add_argument(
        "--model",
        help=(
            "Override the default model chain with a single model id "
            "(e.g. 'anthropic/claude-3.5-sonnet'). Also readable from "
            "the OPENROUTER_MODEL env var."
        ),
    )
    args = parser.parse_args()

    article_id = args.article_id
    record_path = br.article_record_path(article_id)
    if not record_path.exists():
        print(f"No record for article_id {article_id!r}", file=sys.stderr)
        return 2
    record = br.read_json(record_path, default=None)
    if record is None:
        print(f"Empty record.json for {article_id!r}", file=sys.stderr)
        return 2

    canonical_file = br.article_dir(article_id) / "canonical_text.txt"
    if not canonical_file.exists():
        print(
            f"canonical_text.txt missing for {article_id!r}. "
            "Run fetch-pending or attach-manual first.",
            file=sys.stderr,
        )
        return 2
    canonical_text = canonical_file.read_text(encoding="utf-8")
    if not canonical_text.strip():
        print(f"canonical_text.txt for {article_id!r} is empty.", file=sys.stderr)
        return 2

    claims_doc = br.load_claims(article_id)
    if claims_doc.get("claim_extraction_status") != "completed":
        print(
            f"claims.json for {article_id!r} is not completed. "
            "Run Phase B (auto_draft_claims + save-claims) first.",
            file=sys.stderr,
        )
        return 2
    if not claims_doc.get("claims"):
        print(
            f"claims.json for {article_id!r} has no claims to verify.",
            file=sys.stderr,
        )
        return 2

    # Resolve the model up front so we can stamp it into the user message
    # (the prompt asks the LLM to use it as the approver suffix).
    model_override = args.model or os.environ.get("OPENROUTER_MODEL")
    if model_override:
        model_chain = ((model_override, model_override),)
        model_id_for_prompt = model_override
    else:
        model_chain = DEFAULT_MODEL_CHAIN
        model_id_for_prompt = DEFAULT_MODEL_CHAIN[0][0]

    system_prompt = load_system_prompt()
    active_hypotheses = collect_active_hypotheses()
    user_message = build_user_message(
        article_id=article_id,
        record=record,
        canonical_text=canonical_text,
        claims_doc=claims_doc,
        active_hypotheses=active_hypotheses,
        model_id=model_id_for_prompt,
    )

    if args.dry_run:
        print("=== SYSTEM ===")
        print(system_prompt)
        print("=== USER ===")
        print(user_message)
        return 0

    completion = complete_json(
        system=system_prompt,
        user=user_message,
        model_chain=model_chain,
    )

    response_data = completion.data
    if not isinstance(response_data, dict):
        print(
            f"Model returned non-object JSON: {type(response_data).__name__}. "
            f"Raw text was: {completion.raw_text[:500]}",
            file=sys.stderr,
        )
        return 3

    draft = response_data.get("verification_draft")
    approval = response_data.get("approval")
    if not isinstance(draft, dict) or not isinstance(approval, dict):
        errors = {
            "shape": [
                "Model reply must have top-level 'verification_draft' and "
                "'approval' objects. Got keys: "
                f"{sorted(response_data.keys())}"
            ],
        }
        broken = _write_broken(
            article_id,
            draft=draft,
            approval=approval,
            errors=errors,
        )
        print(
            json.dumps(
                {"ok": False, "errors": errors, "broken_path": str(broken)},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 4

    # Force structural fields the LLM might forget. Both are safe to
    # override because they are identity fields, not interpretive.
    draft["article_id"] = article_id
    approval["article_id"] = article_id
    approval.setdefault("approver", f"llm:{completion.model}")

    # Domain on the draft must match the article's record.json domain;
    # Phase C treats the article's canonical domain as authoritative.
    article_domain = br.item_domain(record)
    draft.setdefault("domain", article_domain)

    # Validate both halves against the exact schemas the CLI uses.
    draft_errors = br.validate_verification_payload(draft, article_id, claims_doc)
    draft_items = draft.get("items", []) if isinstance(draft, dict) else []
    approval_errors = br.validate_approval_payload(approval, article_id, draft_items)

    if draft_errors or approval_errors:
        errors = {
            "verification_draft": draft_errors,
            "approval": approval_errors,
        }
        broken = _write_broken(
            article_id,
            draft=draft,
            approval=approval,
            errors=errors,
        )
        print(
            json.dumps(
                {"ok": False, "errors": errors, "broken_path": str(broken)},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 5

    # Run the Phase C guard. If any trigger fires, downgrade the approval
    # decision to needs_human and append a plain-text suffix to
    # overall_rationale so the audit trail survives stage-verification's
    # field-filtering normalization.
    hypothesis_index = br.load_hypothesis_index()
    triggers = br.compute_phase_c_escalations(
        draft_payload=draft,
        approval_payload=approval,
        hypothesis_index=hypothesis_index,
    )
    downgraded = False
    if triggers and approval.get("decision") == "auto_approved":
        approval["decision"] = "needs_human"
        suffix = br.format_phase_c_escalation_suffix(triggers)
        approval["overall_rationale"] = (
            (approval.get("overall_rationale") or "").rstrip() + suffix
        )
        downgraded = True

    # Expose triggers in the scratch approval file for debugging. This
    # field is dropped by stage_verification's normalization, so it only
    # lives in _proposed_approval.json and the CI artifact.
    if triggers:
        approval["escalation_triggers"] = triggers

    if args.output_draft:
        draft_path = Path(args.output_draft)
    else:
        draft_path = br.article_dir(article_id) / "_proposed_draft.json"
    if args.output_approval:
        approval_out = Path(args.output_approval)
    else:
        approval_out = br.article_dir(article_id) / "_proposed_approval.json"

    draft_path.parent.mkdir(parents=True, exist_ok=True)
    approval_out.parent.mkdir(parents=True, exist_ok=True)
    draft_path.write_text(
        json.dumps(draft, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    approval_out.write_text(
        json.dumps(approval, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "ok": True,
        "article_id": article_id,
        "draft_path": str(draft_path),
        "approval_path": str(approval_out),
        "model": completion.model,
        "attempts": completion.attempts,
        "item_count": len(draft_items),
        "decision": approval.get("decision"),
        "downgraded_by_guard": downgraded,
        "escalation_triggers": triggers,
        "usage": completion.usage,
        "next_step": (
            f"python scripts/bayesian_reader.py stage-verification "
            f"--article-id {article_id} "
            f"--draft {draft_path} "
            f"--approval {approval_out}"
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

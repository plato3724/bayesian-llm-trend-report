"""Phase B: draft a `_proposed_claims.json` for an article using OpenRouter.

This script is the LLM-facing half of the auto-draft pipeline. It is
deliberately small: everything domain-specific lives in the prompt file
at scripts/prompts/auto_draft_claims.md, and all of the LLM plumbing
lives in scripts/llm_client.py. What remains here is:

1. Gather the context a model needs for a single article:
     - the canonical_text
     - the list of existing active hypotheses (including domain +
       meta_tags, so the model can honor the anti-metaphor-collapse
       rule in the prompt)
2. Build system + user messages, call OpenRouter via llm_client.
3. Validate the reply against the claims.json schema the same way the
   bayesian_reader.save-claims CLI does (so we fail fast on schema
   drift without touching state).
4. Write the result to knowledge_state/articles/<id>/_proposed_claims.json
   — which is gitignored, so the file is a scratch artifact until a
   human (or a later Phase B+ workflow) runs `save-claims` against it.

What this script REFUSES to do:
  - write claims.json directly (that's save-claims' job and it does
    state cascade + change-log append)
  - invent new hypotheses
  - touch hypotheses.json / verification.json / posteriors

Usage:
    python scripts/auto_draft_claims.py --article-id 06eb3b8e8732
    python scripts/auto_draft_claims.py --article-id 06eb3b8e8732 \
        --output-file /tmp/draft.json
    python scripts/auto_draft_claims.py --article-id 06eb3b8e8732 \
        --dry-run

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

# Make `scripts/` importable so we can reuse bayesian_reader helpers
# without having to duplicate file paths.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# NOTE: importing bayesian_reader touches the knowledge_state directory
# via its bootstrap. That's fine — we're a read-only consumer here.
import bayesian_reader as br  # noqa: E402
from llm_client import complete_json, DEFAULT_MODEL_CHAIN  # noqa: E402


PROMPT_PATH = SCRIPT_DIR / "prompts" / "auto_draft_claims.md"


def load_system_prompt() -> str:
    if not PROMPT_PATH.exists():
        raise SystemExit(
            f"System prompt not found at {PROMPT_PATH}. "
            "Phase B expects this file to exist."
        )
    return PROMPT_PATH.read_text(encoding="utf-8")


def collect_active_hypotheses() -> list[dict[str, Any]]:
    """Return a trimmed view of active hypotheses for the prompt.

    We expose only the fields the model needs to (a) pick valid IDs and
    (b) honor the anti-metaphor-collapse rule. In particular we include
    `domain` and `meta_tags` so the model can see which hypotheses live
    in which concept space.
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
            }
        )
    return summaries


def build_user_message(
    *,
    article_id: str,
    record: dict[str, Any],
    canonical_text: str,
    active_hypotheses: list[dict[str, Any]],
) -> str:
    """Assemble the user-role message for the drafting call.

    We hand the model everything it needs as one well-labeled JSON
    object so it can't hallucinate the structure. This keeps the prompt
    deterministic and makes dry-run debugging easy.
    """

    payload = {
        "article_id": article_id,
        "article_title": record.get("title"),
        "article_url": record.get("url"),
        "canonical_text": canonical_text,
        "active_hypotheses": active_hypotheses,
        "instructions": (
            "Return a JSON object matching the claims schema described in the system prompt. "
            "Reject the temptation to invent hypothesis IDs; leave hypothesis_candidates empty "
            "when unsure. Do not wrap the JSON in markdown fences."
        ),
    }
    # Pretty-print so humans debugging a --dry-run can read it.
    return json.dumps(payload, ensure_ascii=False, indent=2)


def validate_draft_against_schema(
    draft: Any,
    *,
    article_id: str,
) -> list[str]:
    """Re-use bayesian_reader.validate_claims_payload.

    That validator is the same one save-claims uses, so anything that
    passes here will also pass save-claims. This is the hard guarantee
    that Phase B output stays in schema.
    """

    return br.validate_claims_payload(draft, article_id)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Use OpenRouter to draft a _proposed_claims.json for an article. "
            "This does NOT commit anything to knowledge_state — it only writes "
            "the gitignored scratch file that save-claims consumes next."
        ),
    )
    parser.add_argument("--article-id", required=True)
    parser.add_argument(
        "--output-file",
        help=(
            "Where to write the draft. Defaults to "
            "knowledge_state/articles/<id>/_proposed_claims.json, which is "
            "already in .gitignore."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build the prompts and print them to stdout without calling the API. "
            "Useful for debugging the context that would be sent."
        ),
    )
    parser.add_argument(
        "--model",
        help=(
            "Override the default model chain with a single model id "
            "(e.g. 'anthropic/claude-3.5-sonnet' or 'openai/gpt-4o-mini'). "
            "Also readable from the OPENROUTER_MODEL env var."
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
        print(
            f"canonical_text.txt for {article_id!r} is empty.",
            file=sys.stderr,
        )
        return 2

    system_prompt = load_system_prompt()
    active_hypotheses = collect_active_hypotheses()
    user_message = build_user_message(
        article_id=article_id,
        record=record,
        canonical_text=canonical_text,
        active_hypotheses=active_hypotheses,
    )

    if args.dry_run:
        print("=== SYSTEM ===")
        print(system_prompt)
        print("=== USER ===")
        print(user_message)
        return 0

    # Resolve the model chain: CLI flag > env var > built-in default.
    model_override = args.model or os.environ.get("OPENROUTER_MODEL")
    if model_override:
        model_chain = ((model_override, model_override),)
    else:
        model_chain = DEFAULT_MODEL_CHAIN

    completion = complete_json(
        system=system_prompt,
        user=user_message,
        model_chain=model_chain,
    )

    draft = completion.data
    if not isinstance(draft, dict):
        print(
            f"Model returned non-object JSON: {type(draft).__name__}. "
            f"Raw text was: {completion.raw_text[:500]}",
            file=sys.stderr,
        )
        return 3

    # Force the article_id even if the model forgot. Also force the
    # status string that save-claims requires. These are structural,
    # not interpretive, so overriding them is safe.
    draft["article_id"] = article_id
    draft.setdefault("claim_extraction_status", "completed")

    errors = validate_draft_against_schema(draft, article_id=article_id)
    if errors:
        print("Draft failed schema validation:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        # Still write the draft file so a human can inspect it, but
        # with a suffix that makes it obvious it's broken.
        broken_path = (
            br.article_dir(article_id) / "_proposed_claims.broken.json"
        )
        broken_path.write_text(
            json.dumps(draft, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"Wrote broken draft to {broken_path} for debugging.",
            file=sys.stderr,
        )
        return 4

    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = br.article_dir(article_id) / "_proposed_claims.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(draft, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "ok": True,
        "article_id": article_id,
        "output_file": str(output_path),
        "model": completion.model,
        "attempts": completion.attempts,
        "claim_count": len(draft.get("claims", [])),
        "usage": completion.usage,
        "next_step": (
            f"python scripts/bayesian_reader.py save-claims "
            f"--article-id {article_id} --file {output_path}"
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

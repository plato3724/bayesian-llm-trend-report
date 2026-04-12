"""Thin OpenRouter client wrapper.

OpenRouter (https://openrouter.ai/) exposes an OpenAI-compatible API
that proxies many upstream providers (Anthropic Claude, OpenAI GPT-4,
Google Gemini, Meta Llama, DeepSeek, Qwen, ...). We use the official
`openai` SDK with `base_url` overridden, which means we can swap the
underlying model by changing a single string without touching any
transport code.

The wrapper exists for three reasons:

1. Centralize auth / base_url / default headers in one place so every
   automation script shares the same setup.
2. Offer a `complete_json()` helper that takes a (system, user) pair
   plus a schema hint, calls the chat API with response_format=json,
   strips any markdown fences the model accidentally leaks, and
   returns parsed Python data. This removes every script's need to
   re-implement JSON-from-LLM plumbing.
3. Keep a narrow retry surface so a transient 429 does not break an
   automation run.

Environment variables this module reads:

    OPENROUTER_API_KEY   required, your sk-or-... key
    OPENROUTER_DEFAULT_MODEL
                          optional, repository-wide default model id;
                          defaults to 'openai/gpt-5.4'
    OPENROUTER_REFERER   optional, OpenRouter uses this for attribution
                          and leaderboard stats; defaults to the repo URL
    OPENROUTER_TITLE     optional, displayed in OpenRouter activity log;
                          defaults to 'tec_trade bayesian reader'

Nothing else in this file is environment-specific, so it can be imported
from scripts run locally, from GitHub Actions, or from Claude Code.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

try:
    from openai import OpenAI
    from openai import APIError, APIStatusError, RateLimitError
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The 'openai' package is required for scripts/llm_client.py. "
        "Install it with: pip install -r requirements.txt"
    ) from exc


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default to a single model for predictable automation behavior. The
# repository can still override it via OPENROUTER_DEFAULT_MODEL, and
# callers can still pass a one-off OPENROUTER_MODEL / --model override.
DEFAULT_MODEL_FALLBACK = "openai/gpt-5.4"


def _configured_default_model() -> str:
    """Return the configured default model id.

    Priority:
      1. OPENROUTER_DEFAULT_MODEL
      2. built-in fallback (`openai/gpt-5.4`)
    """

    value = (os.environ.get("OPENROUTER_DEFAULT_MODEL") or "").strip()
    return value or DEFAULT_MODEL_FALLBACK


DEFAULT_MODEL = _configured_default_model()

# The client now defaults to a single model. We keep the tuple shape so
# callers can still pass a custom chain explicitly if they want to.
DEFAULT_MODEL_CHAIN: tuple[tuple[str, str], ...] = (
    (DEFAULT_MODEL, DEFAULT_MODEL),
)


@dataclass
class CompletionResult:
    """Return value of `complete_json`."""

    data: Any           # parsed JSON
    raw_text: str       # raw assistant message content
    model: str          # model that actually answered (after fallback)
    attempts: int       # how many HTTP calls it took
    usage: dict[str, Any] | None  # token usage, if the provider reported it


def _get_required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"Missing required environment variable: {name}. "
            "Set it in GitHub Actions secrets or your local shell."
        )
    return value


def _strip_markdown_fences(text: str) -> str:
    """Some models wrap JSON in ```json ... ``` even when asked not to."""
    text = text.strip()
    if text.startswith("```"):
        # Drop the opening fence and optional language tag.
        text = re.sub(r"^```[a-zA-Z0-9]*\s*\n?", "", text)
        # Drop the trailing fence.
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def _parse_json_loose(text: str) -> Any:
    """Parse a model reply that is SUPPOSED to be JSON.

    We first try strict json.loads. If that fails, we look for the first
    top-level {...} block and try again. If THAT fails we re-raise the
    original error so the caller sees the real problem.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to locate a top-level object by bracket matching. This covers
    # the common failure mode where a model prepends a sentence like
    # "Here is the JSON you requested:" before the actual object.
    start = text.find("{")
    if start >= 0:
        depth = 0
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    # Give up; let json.loads raise the canonical error.
    return json.loads(text)


def build_client() -> OpenAI:
    """Construct an openai.OpenAI configured for OpenRouter."""

    api_key = _get_required_env("OPENROUTER_API_KEY")
    referer = os.environ.get(
        "OPENROUTER_REFERER",
        "https://github.com/plato3724/bayesian-llm-trend-report",
    )
    title = os.environ.get("OPENROUTER_TITLE", "tec_trade bayesian reader")

    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers={
            # OpenRouter-specific headers used for attribution and the
            # public leaderboard. Harmless to other providers since
            # OpenRouter terminates them at the proxy.
            "HTTP-Referer": referer,
            "X-Title": title,
        },
    )


def complete_json(
    system: str,
    user: str,
    *,
    model_chain: Iterable[tuple[str, str]] | None = None,
    temperature: float = 0.2,
    max_tokens: int = 4000,
    max_retries_per_model: int = 2,
    force_json_mode: bool = True,
) -> CompletionResult:
    """Call OpenRouter chat completions and parse the reply as JSON.

    Behavior:
      - Iterates `model_chain` (defaults to DEFAULT_MODEL_CHAIN).
      - For each model: up to `max_retries_per_model` attempts on
        retryable errors (429 / 5xx / transient network).
      - Non-retryable errors (4xx other than 429) bubble immediately.
      - Once a model responds with text, we strip markdown fences and
        parse. If parsing fails, we move on to the next model — the
        expectation is that the schema_hint in `system` is strict
        enough that a well-behaved model will comply.

    Args:
      system: The system prompt. Should describe the output schema
              explicitly and include any guardrails.
      user:   The user prompt, typically article context + question.
      model_chain: optional override for DEFAULT_MODEL_CHAIN.
      temperature: low by default because we want determinism, not
                   creativity, in JSON drafting.
      max_tokens: generation cap. 4k is enough for ~20 claims or a
                  verification draft with ~10 items.
      force_json_mode: whether to request response_format=json_object.
                       Most models on OpenRouter honor this; the few
                       that don't will just return plain text that we
                       still parse leniently.

    Returns a CompletionResult.
    """

    client = build_client()
    chain = tuple(model_chain) if model_chain is not None else DEFAULT_MODEL_CHAIN

    last_error: Exception | None = None
    attempts_total = 0

    for model_id, _display in chain:
        for attempt in range(max_retries_per_model):
            attempts_total += 1
            try:
                kwargs: dict[str, Any] = {
                    "model": model_id,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                }
                if force_json_mode:
                    # Not every upstream supports this; OpenRouter
                    # silently ignores it when the underlying model
                    # doesn't honor response_format.
                    kwargs["response_format"] = {"type": "json_object"}

                response = client.chat.completions.create(**kwargs)
            except RateLimitError as exc:
                last_error = exc
                # Exponential backoff: 2s, 4s, 8s ...
                time.sleep(2 ** (attempt + 1))
                continue
            except APIStatusError as exc:
                last_error = exc
                status = getattr(exc, "status_code", None)
                # Retry server errors; do NOT retry 4xx other than 429.
                if status is not None and 500 <= status < 600:
                    time.sleep(2 ** (attempt + 1))
                    continue
                # Try the next model in the chain for 4xx other than 429.
                break
            except APIError as exc:
                last_error = exc
                time.sleep(2 ** (attempt + 1))
                continue

            message = response.choices[0].message
            raw_text = (message.content or "").strip()
            stripped = _strip_markdown_fences(raw_text)
            if not stripped:
                last_error = RuntimeError(
                    f"Model {model_id} returned an empty message."
                )
                break

            try:
                data = _parse_json_loose(stripped)
            except json.JSONDecodeError as exc:
                last_error = RuntimeError(
                    f"Model {model_id} returned non-JSON: {exc}"
                )
                # Move on to the next model.
                break

            usage_raw = getattr(response, "usage", None)
            usage_dict: dict[str, Any] | None = None
            if usage_raw is not None:
                usage_dict = {
                    "prompt_tokens": getattr(usage_raw, "prompt_tokens", None),
                    "completion_tokens": getattr(usage_raw, "completion_tokens", None),
                    "total_tokens": getattr(usage_raw, "total_tokens", None),
                }

            return CompletionResult(
                data=data,
                raw_text=raw_text,
                model=model_id,
                attempts=attempts_total,
                usage=usage_dict,
            )

    # All models in the chain failed.
    raise RuntimeError(
        f"OpenRouter completion failed after {attempts_total} attempt(s) "
        f"across {len(chain)} model(s). Last error: {last_error!r}"
    )


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_MODEL_CHAIN",
    "CompletionResult",
    "build_client",
    "complete_json",
]

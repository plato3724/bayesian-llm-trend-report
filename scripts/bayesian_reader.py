from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import Message
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "knowledge_state"
ARTICLES_DIR = STATE_DIR / "articles"
REPORT_DIR = ROOT / "docs"
REPORT_PATH = REPORT_DIR / "index.html"
FRAMEWORK_PATH = STATE_DIR / "framework.json"
HYPOTHESES_PATH = STATE_DIR / "hypotheses.json"
SYNTHESIS_PATH = STATE_DIR / "synthesis_state.json"
CHANGE_LOG_PATH = STATE_DIR / "change_log.jsonl"
GITHUB_CONFIG_PATH = STATE_DIR / "github_config.json"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_state_dirs() -> None:
    ARTICLES_DIR.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def bootstrap_state_files() -> None:
    ensure_state_dirs()

    if not FRAMEWORK_PATH.exists():
        write_json(
            FRAMEWORK_PATH,
            {
                "version": 1,
                "created_at": utc_now(),
                "last_updated_at": utc_now(),
                "principles": [
                    "No full text, no article-level analysis.",
                    "Article claims are not facts until verified against primary or near-primary sources.",
                    "Only verified or partially verified claims may affect trend posteriors.",
                    "New evidence may weaken, split, retire, or replace existing hypotheses.",
                ],
                "pipeline_stages": [
                    "link_queue",
                    "full_text_acquisition",
                    "claim_extraction",
                    "verification",
                    "bayesian_synthesis",
                ],
                "claim_statuses": [
                    "verified",
                    "partially_verified",
                    "conflicted",
                    "unverified",
                ],
                "bayesian_update": {
                    "formula": "posterior_log_odds = prior_log_odds + sum(weight * log(likelihood_ratio))",
                    "weight_components": {
                        "source_quality": 0.35,
                        "directness": 0.25,
                        "reproducibility": 0.2,
                        "recency": 0.2,
                    },
                    "framework_update_policy": {
                        "add_hypothesis_when": "repeated verified evidence is not explained by current hypotheses",
                        "split_hypothesis_when": "one hypothesis accumulates recurring conflicting verified evidence",
                        "retire_hypothesis_when": "posterior remains weak and unsupported across multiple update cycles",
                    },
                },
            },
        )

    if not HYPOTHESES_PATH.exists():
        write_json(
            HYPOTHESES_PATH,
            {
                "created_at": utc_now(),
                "last_recomputed_at": None,
                "hypotheses": [],
            },
        )

    if not SYNTHESIS_PATH.exists():
        write_json(
            SYNTHESIS_PATH,
            {
                "created_at": utc_now(),
                "last_recomputed_at": None,
                "included_articles": [],
                "active_hypotheses": [],
                "retired_hypotheses": [],
                "notes": [
                    "This file is derived only from verified claims.",
                    "Articles without full text or without verification remain excluded.",
                ],
            },
        )

    if not CHANGE_LOG_PATH.exists():
        CHANGE_LOG_PATH.write_text("", encoding="utf-8")

    if not GITHUB_CONFIG_PATH.exists():
        write_json(
            GITHUB_CONFIG_PATH,
            {
                "repo": None,
                "pages_url": None,
                "issue_label": "article",
            },
        )


def relpath_from_root(path: Path | str) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return candidate.as_posix()


def article_id_from_url(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def article_dir(article_id: str) -> Path:
    return ARTICLES_DIR / article_id


def article_record_path(article_id: str) -> Path:
    return article_dir(article_id) / "record.json"


def claims_path(article_id: str) -> Path:
    return article_dir(article_id) / "claims.json"


def verification_path(article_id: str) -> Path:
    return article_dir(article_id) / "verification.json"


def save_default_article_files(article_id: str) -> None:
    base_dir = article_dir(article_id)
    (base_dir / "raw").mkdir(parents=True, exist_ok=True)
    (base_dir / "manual").mkdir(parents=True, exist_ok=True)

    if not claims_path(article_id).exists():
        write_json(
            claims_path(article_id),
            {
                "article_id": article_id,
                "claim_extraction_status": "not_started",
                "claims": [],
            },
        )

    if not verification_path(article_id).exists():
        write_json(
            verification_path(article_id),
            {
                "article_id": article_id,
                "verification_status": "not_started",
                "items": [],
            },
        )


def parse_url_md(path: Path) -> list[tuple[str, str]]:
    raw_lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in raw_lines if line]
    if len(lines) % 2 != 0:
        raise ValueError(f"Expected title/url pairs in {path}, found an odd number of non-empty lines.")

    pairs: list[tuple[str, str]] = []
    for index in range(0, len(lines), 2):
        title, url = lines[index], lines[index + 1]
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Invalid URL at pair {index // 2 + 1}: {url}")
        pairs.append((title, url))
    return pairs


def upsert_article(
    title: str,
    url: str,
    source_name: str,
    source_ref: str | None = None,
    source_note: str | None = None,
) -> tuple[str, bool]:
    article_id = article_id_from_url(url)
    save_default_article_files(article_id)
    record_path = article_record_path(article_id)

    record = read_json(record_path, default=None)
    created = record is None
    if record is None:
        record = {
            "article_id": article_id,
            "title": title,
            "url": url,
            "ingested_from": source_name,
            "ingested_at": utc_now(),
            "last_updated_at": utc_now(),
            "content_state": {
                "full_text_status": "missing",
                "canonical_text_source": None,
                "fetch_attempts": [],
                "manual_artifacts": [],
                "next_action": "attempt_fetch",
            },
            "analysis_state": {
                "claim_extraction_status": "not_started",
                "verification_status": "not_started",
                "bayesian_status": "excluded_until_verified",
            },
            "article_summary": {
                "events": [],
                "techniques": [],
                "tools": [],
            },
            "hypothesis_impacts": [],
            "ingest_sources": [],
        }
    else:
        record["title"] = title
        record["url"] = url
        record.setdefault("content_state", {})
        record["content_state"].setdefault("fetch_attempts", [])
        record["content_state"].setdefault("manual_artifacts", [])
        record["content_state"].setdefault("full_text_status", "missing")
        record["content_state"].setdefault("next_action", "attempt_fetch")
        record.setdefault("analysis_state", {})
        record["analysis_state"].setdefault("claim_extraction_status", "not_started")
        record["analysis_state"].setdefault("verification_status", "not_started")
        record["analysis_state"].setdefault("bayesian_status", "excluded_until_verified")
        record.setdefault("article_summary", {"events": [], "techniques": [], "tools": []})
        record.setdefault("hypothesis_impacts", [])
        record.setdefault("ingest_sources", [])

    source_entry = {
        "source_name": source_name,
        "source_ref": source_ref,
        "source_note": source_note,
        "ingested_at": utc_now(),
    }
    existing_keys = {
        (item.get("source_name"), item.get("source_ref"))
        for item in record.get("ingest_sources", [])
    }
    source_key = (source_entry["source_name"], source_entry["source_ref"])
    if source_key not in existing_keys:
        record["ingest_sources"].append(source_entry)

    write_json(record_path, record)
    return article_id, created


def extract_urls(text: str) -> list[str]:
    candidates = re.findall(r"https?://[^\s<>\]\)]+", text or "")
    cleaned: list[str] = []
    for candidate in candidates:
        url = candidate.rstrip(".,;:")
        if url not in cleaned:
            cleaned.append(url)
    return cleaned


def read_github_config() -> dict[str, Any]:
    bootstrap_state_files()
    return read_json(
        GITHUB_CONFIG_PATH,
        default={"repo": None, "pages_url": None, "issue_label": "article"},
    )


def write_github_config(
    repo: str | None = None,
    pages_url: str | None = None,
    issue_label: str | None = None,
) -> dict[str, Any]:
    config = read_github_config()
    if repo is not None:
        config["repo"] = repo
    if pages_url is not None:
        config["pages_url"] = pages_url
    if issue_label is not None:
        config["issue_label"] = issue_label
    write_json(GITHUB_CONFIG_PATH, config)
    return config


def resolve_repo_slug(repo: str | None) -> str:
    if repo:
        return repo
    config = read_github_config()
    repo_slug = config.get("repo")
    if repo_slug:
        return str(repo_slug)
    raise ValueError("GitHub repo is not configured. Pass --repo or set github_config.json first.")


def run_gh_json(args: list[str]) -> Any:
    result = subprocess.run(
        ["gh", *args],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "gh command failed")
    return json.loads(result.stdout)


def bootstrap_from_url_md(source_path: Path) -> dict[str, Any]:
    bootstrap_state_files()
    created = 0
    updated = 0
    imported_ids: list[str] = []

    for title, url in parse_url_md(source_path):
        article_id, was_created = upsert_article(
            title=title,
            url=url,
            source_name=str(source_path.name),
            source_ref=str(source_path),
        )
        if was_created:
            created += 1
        else:
            updated += 1
        imported_ids.append(article_id)

    append_jsonl(
        CHANGE_LOG_PATH,
        {
            "timestamp": utc_now(),
            "event": "bootstrap_from_url_md",
            "source_path": str(source_path),
            "created": created,
            "updated": updated,
            "article_ids": imported_ids,
        },
    )

    return {"created": created, "updated": updated, "article_ids": imported_ids}


def sync_github_issues(
    repo: str | None = None,
    label: str | None = None,
    state: str = "open",
    limit: int = 100,
    write_config_flag: bool = False,
) -> dict[str, Any]:
    repo_slug = resolve_repo_slug(repo)
    effective_label = label if label is not None else read_github_config().get("issue_label")
    base_command = [
        "issue",
        "list",
        "--repo",
        repo_slug,
        "--state",
        state,
        "--limit",
        str(limit),
        "--json",
        "number,title,body,url,labels,createdAt",
    ]
    fallback_used = False
    if effective_label:
        command = [*base_command, "--label", str(effective_label)]
        issues = run_gh_json(command)
        if not issues:
            issues = run_gh_json(base_command)
            fallback_used = True
    else:
        issues = run_gh_json(base_command)
    imported: list[dict[str, Any]] = []
    created = 0
    updated = 0
    skipped = 0

    for issue in issues:
        urls = extract_urls(issue.get("body") or "")
        if not urls:
            skipped += 1
            continue
        notes = issue.get("body", "").strip()
        for url in urls:
            article_id, was_created = upsert_article(
                title=(issue.get("title") or url).strip(),
                url=url,
                source_name="github_issue",
                source_ref=issue.get("url"),
                source_note=notes,
            )
            if was_created:
                created += 1
            else:
                updated += 1
            imported.append(
                {
                    "article_id": article_id,
                    "issue_number": issue.get("number"),
                    "issue_url": issue.get("url"),
                    "title": issue.get("title"),
                    "url": url,
                }
            )

    if write_config_flag:
        write_github_config(repo=repo_slug, issue_label=effective_label)

    append_jsonl(
        CHANGE_LOG_PATH,
        {
            "timestamp": utc_now(),
            "event": "sync_github_issues",
            "repo": repo_slug,
            "label": effective_label,
            "state": state,
            "fallback_used": fallback_used,
            "created": created,
            "updated": updated,
            "skipped": skipped,
        },
    )
    return {
        "repo": repo_slug,
        "label": effective_label,
        "fallback_used": fallback_used,
        "created": created,
        "updated": updated,
        "skipped": skipped,
        "imported": imported,
    }


class HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_stack = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_stack += 1
        if tag in {"p", "div", "section", "article", "br", "li", "h1", "h2", "h3"} and not self._skip_stack:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_stack:
            self._skip_stack -= 1
        if tag in {"p", "div", "section", "article", "li", "h1", "h2", "h3"} and not self._skip_stack:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_stack:
            self.parts.append(data)

    def text(self) -> str:
        raw = html.unescape("".join(self.parts))
        raw = re.sub(r"\r\n?", "\n", raw)
        raw = re.sub(r"[ \t\f\v]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


@dataclass
class FetchResult:
    status: str
    status_code: int | None
    final_url: str | None
    content_type: str | None
    text_length: int
    raw_path: str | None
    text_path: str | None
    note: str


def response_charset(headers: Message) -> str | None:
    content_type = headers.get("Content-Type")
    if not content_type:
        return None
    match = re.search(r"charset=([\w-]+)", content_type, re.IGNORECASE)
    return match.group(1) if match else None


def decode_bytes(data: bytes, headers: Message) -> str:
    candidates = [response_charset(headers), "utf-8", "gb18030", "gbk", "latin-1"]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return data.decode(candidate)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def extract_text_from_html(payload: str) -> str:
    parser = HTMLTextExtractor()
    parser.feed(payload)
    return parser.text()


def clean_extracted_text(text: str) -> str:
    blocked_exact = {
        "在小说阅读器读本章",
        "去阅读",
        "在小说阅读器中沉浸阅读",
    }
    blocked_contains = [
        "小说阅读器",
    ]

    cleaned_lines: list[str] = []
    previous_line = None
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if line in blocked_exact:
            continue
        if any(fragment in line for fragment in blocked_contains):
            continue
        if line == previous_line:
            continue
        cleaned_lines.append(line)
        previous_line = line
    return "\n".join(cleaned_lines).strip()


def detect_blocked_fetch(url: str, text: str) -> bool:
    text_lower = text.lower()
    return (
        "wappoc_appmsgcaptcha" in url
        or "appmsgcaptcha" in text_lower
        or ("mp.weixin.qq.com" in url and "验证码" in text)
    )


def fetch_url(url: str, destination_dir: Path) -> FetchResult:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path: Path | None = None
    text_path: Path | None = None

    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            body = response.read()
            final_url = response.geturl()
            content_type = response.headers.get_content_type()
            decoded = decode_bytes(body, response.headers)
            extension = ".html" if content_type == "text/html" else ".txt"
            raw_path = destination_dir / f"fetch_{timestamp}{extension}"
            raw_path.write_text(decoded, encoding="utf-8")

            extracted = extract_text_from_html(decoded) if content_type == "text/html" else decoded
            extracted = clean_extracted_text(extracted)
            blocked = detect_blocked_fetch(final_url, extracted)

            if extracted:
                text_path = destination_dir / f"extract_{timestamp}.txt"
                text_path.write_text(extracted, encoding="utf-8")

            if blocked:
                return FetchResult(
                    status="blocked",
                    status_code=response.status,
                    final_url=final_url,
                    content_type=content_type,
                    text_length=len(extracted),
                    raw_path=str(raw_path),
                    text_path=str(text_path) if text_path else None,
                    note="Fetch hit a WeChat captcha or equivalent block page.",
                )

            if len(extracted) >= 1200:
                status = "acquired"
                note = "Full text looks sufficient for claim extraction."
            elif len(extracted) >= 250:
                status = "partial"
                note = "Text was fetched but looks incomplete. Verify manually before analysis."
            else:
                status = "fetch_failed"
                note = "Response body was too short to trust as article text."

            return FetchResult(
                status=status,
                status_code=response.status,
                final_url=final_url,
                content_type=content_type,
                text_length=len(extracted),
                raw_path=str(raw_path),
                text_path=str(text_path) if text_path else None,
                note=note,
            )
    except urllib.error.HTTPError as exc:
        return FetchResult(
            status="fetch_failed",
            status_code=exc.code,
            final_url=url,
            content_type=None,
            text_length=0,
            raw_path=None,
            text_path=None,
            note=f"HTTPError: {exc.reason}",
        )
    except Exception as exc:  # pragma: no cover - best effort network path
        return FetchResult(
            status="fetch_failed",
            status_code=None,
            final_url=url,
            content_type=None,
            text_length=0,
            raw_path=None,
            text_path=None,
            note=f"{type(exc).__name__}: {exc}",
        )


def load_all_article_records() -> list[dict[str, Any]]:
    bootstrap_state_files()
    records: list[dict[str, Any]] = []
    for path in sorted(ARTICLES_DIR.glob("*/record.json")):
        records.append(read_json(path, default={}))
    return records


def save_article_record(record: dict[str, Any]) -> None:
    record["last_updated_at"] = utc_now()
    write_json(article_record_path(record["article_id"]), record)


def next_action_for_status(status: str) -> str:
    mapping = {
        "acquired": "extract_claims",
        "partial": "review_text_then_extract",
        "blocked": "search_mirror_or_request_manual_text",
        "fetch_failed": "retry_or_request_manual_text",
        "missing": "attempt_fetch",
    }
    return mapping.get(status, "review")


def fetch_pending(limit: int, force: bool = False) -> dict[str, Any]:
    records = load_all_article_records()
    if force:
        targets = records[:limit]
    else:
        targets = [
            record
            for record in records
            if record.get("content_state", {}).get("full_text_status") in {"missing", "fetch_failed", "partial"}
        ][:limit]

    results: list[dict[str, Any]] = []
    for record in targets:
        raw_dir = article_dir(record["article_id"]) / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        fetched = fetch_url(record["url"], raw_dir)

        attempt = {
            "attempted_at": utc_now(),
            "status": fetched.status,
            "status_code": fetched.status_code,
            "final_url": fetched.final_url,
            "content_type": fetched.content_type,
            "text_length": fetched.text_length,
            "raw_path": fetched.raw_path,
            "text_path": fetched.text_path,
            "note": fetched.note,
        }
        record["content_state"]["fetch_attempts"].append(attempt)
        record["content_state"]["full_text_status"] = fetched.status
        record["content_state"]["next_action"] = next_action_for_status(fetched.status)

        if fetched.status in {"acquired", "partial"} and fetched.text_path:
            canonical_path = article_dir(record["article_id"]) / "canonical_text.txt"
            shutil.copyfile(fetched.text_path, canonical_path)
            record["content_state"]["canonical_text_source"] = relpath_from_root(canonical_path)
        elif fetched.status == "blocked":
            record["content_state"]["canonical_text_source"] = None

        save_article_record(record)
        results.append(
            {
                "article_id": record["article_id"],
                "title": record["title"],
                "status": fetched.status,
                "final_url": fetched.final_url,
                "text_length": fetched.text_length,
                "note": fetched.note,
            }
        )

    append_jsonl(
        CHANGE_LOG_PATH,
        {
            "timestamp": utc_now(),
            "event": "fetch_pending",
            "limit": limit,
            "force": force,
            "results": results,
        },
    )

    return {"attempted": len(results), "results": results}


def summarize_status() -> dict[str, Any]:
    counts: dict[str, int] = {}
    records = load_all_article_records()
    for record in records:
        status = record.get("content_state", {}).get("full_text_status", "missing")
        counts[status] = counts.get(status, 0) + 1

    return {
        "total_articles": len(records),
        "content_status_counts": counts,
        "eligible_for_claim_extraction": sum(1 for record in records if record.get("content_state", {}).get("full_text_status") == "acquired"),
        "waiting_for_manual_or_search": sum(
            1
            for record in records
            if record.get("content_state", {}).get("full_text_status") in {"blocked", "fetch_failed"}
        ),
    }


def list_articles() -> dict[str, Any]:
    records = load_all_article_records()
    return {
        "articles": [
            {
                "article_id": record.get("article_id"),
                "title": record.get("title"),
                "url": record.get("url"),
                "full_text_status": record.get("content_state", {}).get("full_text_status"),
                "next_action": record.get("content_state", {}).get("next_action"),
                "canonical_text_source": record.get("content_state", {}).get("canonical_text_source"),
            }
            for record in records
        ]
    }


def refresh_record_states() -> dict[str, Any]:
    records = load_all_article_records()
    refreshed = 0

    for record in records:
        article_id = record["article_id"]
        claims = read_json(claims_path(article_id), default={"claims": [], "claim_extraction_status": "not_started"})
        verification = read_json(verification_path(article_id), default={"items": [], "verification_status": "not_started"})

        claim_status = claims.get("claim_extraction_status", "not_started")
        verification_status = verification.get("verification_status", "not_started")

        verified_items = [
            item for item in verification.get("items", [])
            if item.get("status") in {"verified", "partially_verified"} and item.get("hypothesis_id")
        ]
        conflicted_items = [
            item for item in verification.get("items", [])
            if item.get("status") in {"conflicted", "unverified"}
        ]

        events = [claim["text"] for claim in claims.get("claims", []) if claim.get("type") == "event"]
        techniques = [claim["text"] for claim in claims.get("claims", []) if claim.get("type") == "technique"]
        tools = [claim["text"] for claim in claims.get("claims", []) if claim.get("type") == "tool"]

        if claim_status != "completed":
            bayesian_status = "excluded_until_verified"
            next_action = "extract_claims"
        elif verified_items:
            bayesian_status = "included"
            next_action = "monitor_for_new_evidence"
        elif verification_status == "completed":
            bayesian_status = "excluded_after_verification"
            next_action = "wait_for_stronger_sources"
        elif conflicted_items:
            bayesian_status = "held_out_pending_better_evidence"
            next_action = "collect_better_sources"
        else:
            bayesian_status = "excluded_until_verified"
            next_action = "verify_claims"

        record.setdefault("analysis_state", {})
        record["analysis_state"]["claim_extraction_status"] = claim_status
        record["analysis_state"]["verification_status"] = verification_status
        record["analysis_state"]["bayesian_status"] = bayesian_status
        record.setdefault("content_state", {})
        record["content_state"]["next_action"] = next_action
        record["article_summary"] = {
            "events": events,
            "techniques": techniques,
            "tools": tools,
        }
        record["hypothesis_impacts"] = [
            {
                "hypothesis_id": item.get("hypothesis_id"),
                "status": item.get("status"),
                "weight": item.get("weight"),
                "likelihood_ratio": item.get("likelihood_ratio"),
            }
            for item in verified_items
        ]
        save_article_record(record)
        refreshed += 1

    append_jsonl(
        CHANGE_LOG_PATH,
        {
            "timestamp": utc_now(),
            "event": "refresh_record_states",
            "refreshed": refreshed,
        },
    )
    return {"refreshed": refreshed}


def recompute_posteriors() -> dict[str, Any]:
    """Recompute posteriors from all articles' verification files.

    Supports two evidence schemas in parallel:
      - Ordinal (new): source_trust / evidence_direction / evidence_strength
      - Legacy (old): weight / likelihood_ratio floats

    Legacy items are kept readable so that pre-existing verification.json
    files (written before the fuzzy refactor) continue to contribute to
    posteriors without a full migration. New payloads are validated to
    reject the legacy schema in validate_verification_payload.
    """
    bootstrap_state_files()
    hypotheses = read_json(HYPOTHESES_PATH, default={"hypotheses": []})
    article_records = load_all_article_records()
    included_articles: list[str] = []

    for hypothesis in hypotheses.get("hypotheses", []):
        prior_log_odds = float(hypothesis.get("prior_log_odds", 0.0))
        posterior_log_odds = prior_log_odds
        supporting_items: list[dict[str, Any]] = []

        for record in article_records:
            verification = read_json(verification_path(record["article_id"]), default={"items": []})
            for item in verification.get("items", []):
                if item.get("hypothesis_id") != hypothesis.get("id"):
                    continue
                if item.get("status") not in {"verified", "partially_verified"}:
                    continue

                contribution = 0.0
                supporting_entry: dict[str, Any] = {
                    "article_id": record["article_id"],
                    "claim_id": item.get("claim_id"),
                    "status": item.get("status"),
                }

                if is_ordinal_item(item):
                    try:
                        contribution = ordinal_contribution(
                            source_trust=item["source_trust"],
                            evidence_direction=item["evidence_direction"],
                            evidence_strength=item["evidence_strength"],
                        )
                    except KeyError:
                        # Malformed ordinal item — skip it rather than
                        # crash. validate_verification_payload should have
                        # caught this upstream.
                        continue
                    supporting_entry.update(
                        {
                            "source_trust": item["source_trust"],
                            "evidence_direction": item["evidence_direction"],
                            "evidence_strength": item["evidence_strength"],
                        }
                    )
                else:
                    weight = float(item.get("weight", 0.0))
                    lr = float(item.get("likelihood_ratio", 1.0))
                    if lr <= 0:
                        continue
                    contribution = weight * math.log(lr)
                    supporting_entry.update(
                        {
                            "weight": weight,
                            "likelihood_ratio": lr,
                        }
                    )

                posterior_log_odds += contribution
                supporting_items.append(supporting_entry)
                if record["article_id"] not in included_articles:
                    included_articles.append(record["article_id"])

        probability = 1 / (1 + math.exp(-posterior_log_odds))
        hypothesis["posterior_log_odds"] = posterior_log_odds
        hypothesis["posterior_probability"] = probability
        hypothesis["posterior_band"] = posterior_band(probability)
        hypothesis["last_recomputed_at"] = utc_now()
        hypothesis["supporting_items"] = supporting_items

    # Also fold in verified+null-hypothesis articles — they count as
    # included evidence even though they don't touch any posterior.
    for record in article_records:
        verification = read_json(verification_path(record["article_id"]), default={"items": []})
        for item in verification.get("items", []):
            if item.get("status") in STRONG_VERIFICATION_STATUSES:
                if record["article_id"] not in included_articles:
                    included_articles.append(record["article_id"])
                break

    hypotheses["last_recomputed_at"] = utc_now()
    write_json(HYPOTHESES_PATH, hypotheses)

    synthesis = read_json(SYNTHESIS_PATH, default={})
    synthesis["last_recomputed_at"] = utc_now()
    synthesis["included_articles"] = included_articles
    synthesis["active_hypotheses"] = [
        {
            "id": item.get("id"),
            "statement": item.get("statement"),
            "posterior_probability": item.get("posterior_probability"),
            "posterior_band": item.get("posterior_band"),
        }
        for item in hypotheses.get("hypotheses", [])
    ]
    write_json(SYNTHESIS_PATH, synthesis)

    append_jsonl(
        CHANGE_LOG_PATH,
        {
            "timestamp": utc_now(),
            "event": "recompute_posteriors",
            "included_articles": included_articles,
            "hypothesis_count": len(hypotheses.get("hypotheses", [])),
        },
    )
    return {
        "hypothesis_count": len(hypotheses.get("hypotheses", [])),
        "included_articles": included_articles,
    }


def attach_manual_file(article_id: str, source_file: Path) -> dict[str, Any]:
    if not source_file.exists():
        raise FileNotFoundError(source_file)

    record = read_json(article_record_path(article_id), default=None)
    if record is None:
        raise ValueError(f"Unknown article_id: {article_id}")

    manual_dir = article_dir(article_id) / "manual"
    manual_dir.mkdir(parents=True, exist_ok=True)
    destination = manual_dir / source_file.name
    shutil.copyfile(source_file, destination)

    record["content_state"]["manual_artifacts"].append(
        {
            "attached_at": utc_now(),
            "path": str(destination),
        }
    )
    record["content_state"]["next_action"] = "review_manual_material"
    save_article_record(record)

    append_jsonl(
        CHANGE_LOG_PATH,
        {
            "timestamp": utc_now(),
            "event": "attach_manual_file",
            "article_id": article_id,
            "source_file": str(source_file),
            "destination_file": str(destination),
        },
    )
    return {"article_id": article_id, "destination": str(destination)}


# ---------------------------------------------------------------------------
# Phase 1 affordances: queue / next / save-* / state diff
# ---------------------------------------------------------------------------


CLAIM_TYPES = {"event", "technique", "tool"}
VERIFICATION_STATUSES = {"verified", "partially_verified", "conflicted", "unverified"}
STRONG_VERIFICATION_STATUSES = {"verified", "partially_verified"}


# ---------------------------------------------------------------------------
# Fuzzy (ordinal) Bayesian model
# ---------------------------------------------------------------------------
# Rationale: all four inputs to the old continuous weight formula were
# subjective guesses, so producing posteriors like 0.8741 was false precision.
# The new model keeps the Bayesian core (log-odds accumulation) but:
#   - only exposes ordinal inputs to humans / LLMs
#   - only exposes band names (plus a direction arrow) to the report
# Continuous floats still exist internally as the single source of truth for
# math, but they must never appear in user-visible output.
#
# An evidence item contributes to log-odds as:
#     sign(direction) * trust_multiplier * strength_logit
# All three factors are ordinal lookups (weak/moderate/strong, etc.)
#
# Legacy verification items using the old weight/likelihood_ratio floats are
# still accepted by recompute_posteriors so we don't have to re-annotate every
# existing hypothesis supporting_item. New verification payloads use the
# ordinal schema exclusively.

POSTERIOR_BANDS: list[tuple[str, float, float]] = [
    ("very_unlikely", 0.00, 0.15),
    ("unlikely", 0.15, 0.35),
    ("uncertain", 0.35, 0.65),
    ("likely", 0.65, 0.93),
    ("very_likely", 0.93, 0.98),
    ("near_certain", 0.98, 1.01),
]

POSTERIOR_BAND_LABELS: dict[str, str] = {
    "very_unlikely": "几乎可以否定",
    "unlikely": "证据偏弱",
    "uncertain": "拿不准",
    "likely": "证据偏强",
    "very_likely": "证据很强",
    "near_certain": "几乎可以确认",
    "unknown": "无数据",
}

SOURCE_TRUST_LEVELS: dict[str, float] = {
    "weak": 0.3,
    "moderate": 0.6,
    "strong": 0.9,
}

EVIDENCE_STRENGTHS: dict[str, float] = {
    "slight": 0.4,
    "moderate": 1.0,
    "strong": 1.8,
}

EVIDENCE_DIRECTIONS: dict[str, int] = {
    "support": 1,
    "against": -1,
}

# Minimum log-odds delta that still counts as "moved" when deciding whether
# to draw an up/down arrow next to a band label.
BAND_DIFF_EPSILON = 0.01


def posterior_band(probability: float | None) -> str:
    if probability is None:
        return "unknown"
    for name, low, high in POSTERIOR_BANDS:
        if low <= probability < high:
            return name
    return POSTERIOR_BANDS[-1][0]


def posterior_band_arrow(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return "·"
    delta = after - before
    if delta > BAND_DIFF_EPSILON:
        return "↑"
    if delta < -BAND_DIFF_EPSILON:
        return "↓"
    return "·"


def format_band(probability: float | None) -> str:
    """Return 'very_likely' style label for a probability, no number."""
    return posterior_band(probability)


def format_band_transition(before: float | None, after: float | None) -> str:
    """Return 'likely → very_likely ↑' style transition string, no numbers."""
    before_band = posterior_band(before)
    after_band = posterior_band(after)
    arrow = posterior_band_arrow(before, after)
    if before_band == after_band:
        return f"{after_band} {arrow}"
    return f"{before_band} → {after_band} {arrow}"


def ordinal_contribution(
    source_trust: str,
    evidence_direction: str,
    evidence_strength: str,
) -> float:
    """Map an ordinal evidence tuple to a log-odds contribution.

    Raises KeyError if any field is not in the allowed vocabulary; callers
    should validate first using validate_verification_payload.
    """
    trust = SOURCE_TRUST_LEVELS[source_trust]
    strength = EVIDENCE_STRENGTHS[evidence_strength]
    sign = EVIDENCE_DIRECTIONS[evidence_direction]
    return sign * trust * strength


def is_ordinal_item(item: dict[str, Any]) -> bool:
    """True when a verification item uses the new ordinal schema."""
    return "evidence_strength" in item or "source_trust" in item


def draft_verification_path(article_id: str) -> Path:
    return article_dir(article_id) / "verification_draft.json"


def approval_path(article_id: str) -> Path:
    return article_dir(article_id) / "approval.json"


def load_claims(article_id: str) -> dict[str, Any]:
    return read_json(
        claims_path(article_id),
        default={"article_id": article_id, "claim_extraction_status": "not_started", "claims": []},
    )


def load_verification(article_id: str) -> dict[str, Any]:
    return read_json(
        verification_path(article_id),
        default={"article_id": article_id, "verification_status": "not_started", "items": []},
    )


def load_hypothesis_index() -> dict[str, dict[str, Any]]:
    hypotheses = read_json(HYPOTHESES_PATH, default={"hypotheses": []})
    return {item["id"]: item for item in hypotheses.get("hypotheses", []) if item.get("id")}


def snapshot_state() -> dict[str, Any]:
    """Capture a lightweight snapshot for state diffs.

    Stores raw probabilities internally because we need them to detect
    sub-band motion, but format_state_diff should only ever print bands.
    """
    hypotheses = read_json(HYPOTHESES_PATH, default={"hypotheses": []})
    synthesis = read_json(SYNTHESIS_PATH, default={})
    return {
        "hypotheses": {
            item.get("id"): item.get("posterior_probability")
            for item in hypotheses.get("hypotheses", [])
            if item.get("id")
        },
        "included_articles": list(synthesis.get("included_articles", [])),
    }


def format_state_diff(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Render a band-based diff.

    Deliberately never shows a decimal probability. A hypothesis whose
    underlying log-odds moved but stayed inside the same band shows only
    the arrow (↑/↓/·); a band transition shows 'before → after arrow'.
    """
    lines: list[str] = []

    before_hyp = before.get("hypotheses", {})
    after_hyp = after.get("hypotheses", {})
    for hypothesis_id in sorted(set(before_hyp) | set(after_hyp)):
        before_val = before_hyp.get(hypothesis_id)
        after_val = after_hyp.get(hypothesis_id)
        if before_val is None and after_val is None:
            continue
        if before_val is None:
            lines.append(f"{hypothesis_id}: (new) → {posterior_band(after_val)}")
            continue
        if after_val is None:
            lines.append(f"{hypothesis_id}: {posterior_band(before_val)} → (removed)")
            continue
        arrow = posterior_band_arrow(before_val, after_val)
        if arrow == "·":
            # Stayed in place within its band and no meaningful sub-band
            # motion either. Don't clutter the diff.
            continue
        lines.append(f"{hypothesis_id}: {format_band_transition(before_val, after_val)}")

    before_set = set(before.get("included_articles") or [])
    after_set = set(after.get("included_articles") or [])
    added = sorted(after_set - before_set)
    removed = sorted(before_set - after_set)
    if added:
        lines.append(f"included_articles +{len(added)}: {', '.join(added)}")
    if removed:
        lines.append(f"included_articles -{len(removed)}: {', '.join(removed)}")
    if not added and not removed and len(before_set) != len(after_set):
        lines.append(f"included_articles: {len(before_set)} → {len(after_set)}")

    return lines


def validate_claims_payload(payload: Any, article_id: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["claims payload must be a JSON object"]

    payload_article_id = payload.get("article_id")
    if payload_article_id and payload_article_id != article_id:
        errors.append(
            f"article_id mismatch: payload says {payload_article_id!r}, target is {article_id!r}"
        )

    status = payload.get("claim_extraction_status")
    if status and status != "completed":
        errors.append(
            f"claim_extraction_status must be 'completed' for save-claims (got {status!r})"
        )

    claims = payload.get("claims")
    if not isinstance(claims, list):
        errors.append("'claims' must be a list")
        return errors
    if not claims:
        errors.append("'claims' list is empty; refuse to mark extraction completed with no claims")

    hypothesis_ids = set(load_hypothesis_index().keys())
    seen_ids: set[str] = set()

    for index, claim in enumerate(claims):
        prefix = f"claims[{index}]"
        if not isinstance(claim, dict):
            errors.append(f"{prefix} must be an object")
            continue

        claim_id = claim.get("id")
        if not isinstance(claim_id, str) or not claim_id.strip():
            errors.append(f"{prefix}.id must be a non-empty string")
        elif claim_id in seen_ids:
            errors.append(f"{prefix}.id duplicate: {claim_id!r}")
        else:
            seen_ids.add(claim_id)

        claim_type = claim.get("type")
        if claim_type not in CLAIM_TYPES:
            errors.append(
                f"{prefix}.type must be one of {sorted(CLAIM_TYPES)} (got {claim_type!r})"
            )

        text = claim.get("text")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{prefix}.text must be a non-empty string")

        candidates = claim.get("hypothesis_candidates", [])
        if not isinstance(candidates, list):
            errors.append(f"{prefix}.hypothesis_candidates must be a list")
        else:
            for candidate in candidates:
                if not isinstance(candidate, str):
                    errors.append(
                        f"{prefix}.hypothesis_candidates must contain strings (got {candidate!r})"
                    )
                elif candidate not in hypothesis_ids:
                    errors.append(
                        f"{prefix}.hypothesis_candidates references unknown hypothesis_id {candidate!r}"
                    )

    return errors


def validate_verification_payload(
    payload: Any,
    article_id: str,
    claims_doc: dict[str, Any],
) -> list[str]:
    """Validate a verification payload.

    Rules:
      - verified / partially_verified → source_url and source_title required
        (regardless of whether a hypothesis is attached)
      - hypothesis_id may be null for a 'verified orthogonal fact' claim that
        doesn't update any hypothesis this round
      - when hypothesis_id is non-null, the ordinal evidence triple
        (source_trust, evidence_direction, evidence_strength) is required
        and must be in the allowed vocabulary
      - legacy items carrying weight/likelihood_ratio floats are rejected in
        new payloads to prevent reintroducing false-precision schemas
    """
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["verification payload must be a JSON object"]

    payload_article_id = payload.get("article_id")
    if payload_article_id and payload_article_id != article_id:
        errors.append(
            f"article_id mismatch: payload says {payload_article_id!r}, target is {article_id!r}"
        )

    if claims_doc.get("claim_extraction_status") != "completed":
        errors.append(
            "cannot save verification before claim extraction is completed for this article"
        )

    status = payload.get("verification_status")
    if status and status not in {"completed", "drafted"}:
        errors.append(
            f"verification_status must be 'completed' or 'drafted' (got {status!r})"
        )

    items = payload.get("items")
    if not isinstance(items, list):
        errors.append("'items' must be a list")
        return errors

    known_claim_ids = {claim.get("id") for claim in claims_doc.get("claims", []) if claim.get("id")}
    hypothesis_ids = set(load_hypothesis_index().keys())

    for index, item in enumerate(items):
        prefix = f"items[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be an object")
            continue

        # Reject legacy float-based schema outright for new payloads.
        if "weight" in item or "likelihood_ratio" in item:
            errors.append(
                f"{prefix}: legacy 'weight'/'likelihood_ratio' floats are no longer accepted; "
                f"use source_trust / evidence_direction / evidence_strength instead"
            )

        claim_id = item.get("claim_id")
        if claim_id not in known_claim_ids:
            errors.append(
                f"{prefix}.claim_id {claim_id!r} not found in claims.json for {article_id}"
            )

        item_status = item.get("status")
        if item_status not in VERIFICATION_STATUSES:
            errors.append(
                f"{prefix}.status must be one of {sorted(VERIFICATION_STATUSES)} (got {item_status!r})"
            )

        hypothesis_id = item.get("hypothesis_id")
        if hypothesis_id is not None:
            if not isinstance(hypothesis_id, str):
                errors.append(f"{prefix}.hypothesis_id must be a string or null")
            elif hypothesis_id not in hypothesis_ids:
                errors.append(
                    f"{prefix}.hypothesis_id {hypothesis_id!r} not found in hypotheses.json"
                )

        assessment = item.get("assessment")
        if not isinstance(assessment, str) or not assessment.strip():
            errors.append(f"{prefix}.assessment must be a non-empty string")

        if item_status in STRONG_VERIFICATION_STATUSES:
            source_url = item.get("source_url")
            if not isinstance(source_url, str) or not source_url.strip():
                errors.append(
                    f"{prefix}.source_url is required when status is {item_status!r}"
                )
            source_title = item.get("source_title")
            if not isinstance(source_title, str) or not source_title.strip():
                errors.append(
                    f"{prefix}.source_title is required when status is {item_status!r}"
                )

            if hypothesis_id:
                # A claim attached to a hypothesis must supply the ordinal
                # evidence triple; otherwise there's nothing to contribute
                # to the posterior.
                trust = item.get("source_trust")
                if trust not in SOURCE_TRUST_LEVELS:
                    errors.append(
                        f"{prefix}.source_trust must be one of "
                        f"{sorted(SOURCE_TRUST_LEVELS)} when hypothesis_id is set "
                        f"(got {trust!r})"
                    )
                direction = item.get("evidence_direction")
                if direction not in EVIDENCE_DIRECTIONS:
                    errors.append(
                        f"{prefix}.evidence_direction must be one of "
                        f"{sorted(EVIDENCE_DIRECTIONS)} when hypothesis_id is set "
                        f"(got {direction!r})"
                    )
                strength = item.get("evidence_strength")
                if strength not in EVIDENCE_STRENGTHS:
                    errors.append(
                        f"{prefix}.evidence_strength must be one of "
                        f"{sorted(EVIDENCE_STRENGTHS)} when hypothesis_id is set "
                        f"(got {strength!r})"
                    )
            else:
                # Orthogonal fact: shouldn't carry evidence fields because
                # they wouldn't be used, and leaving them in would be
                # misleading.
                for stray in ("source_trust", "evidence_direction", "evidence_strength"):
                    if stray in item and item.get(stray) is not None:
                        errors.append(
                            f"{prefix}.{stray} must be omitted when hypothesis_id is null"
                        )

    return errors


def _load_save_payload(file_arg: str | None) -> Any:
    if file_arg is None or file_arg == "-":
        raw = sys.stdin.read()
        origin = "<stdin>"
    else:
        source_path = Path(file_arg)
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        raw = source_path.read_text(encoding="utf-8")
        origin = str(source_path)
    if not raw.strip():
        raise ValueError(f"No JSON payload supplied from {origin}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON from {origin}: {exc}") from exc


def save_claims(article_id: str, payload: Any) -> dict[str, Any]:
    bootstrap_state_files()
    record = read_json(article_record_path(article_id), default=None)
    if record is None:
        raise ValueError(f"Unknown article_id: {article_id}")

    errors = validate_claims_payload(payload, article_id)
    if errors:
        return {"ok": False, "article_id": article_id, "errors": errors}

    before = snapshot_state()

    normalized = {
        "article_id": article_id,
        "claim_extraction_status": "completed",
        "extracted_at": utc_now(),
        "claims": payload.get("claims", []),
    }
    save_default_article_files(article_id)
    write_json(claims_path(article_id), normalized)

    refresh_record_states()
    recompute_posteriors()
    after = snapshot_state()

    append_jsonl(
        CHANGE_LOG_PATH,
        {
            "timestamp": utc_now(),
            "event": "save_claims",
            "article_id": article_id,
            "claim_count": len(normalized["claims"]),
        },
    )

    next_task = get_next_task()
    return {
        "ok": True,
        "article_id": article_id,
        "claim_count": len(normalized["claims"]),
        "state_diff": format_state_diff(before, after),
        "next_task": next_task,
    }


def save_verification(article_id: str, payload: Any) -> dict[str, Any]:
    bootstrap_state_files()
    record = read_json(article_record_path(article_id), default=None)
    if record is None:
        raise ValueError(f"Unknown article_id: {article_id}")

    claims_doc = load_claims(article_id)
    errors = validate_verification_payload(payload, article_id, claims_doc)
    if errors:
        return {"ok": False, "article_id": article_id, "errors": errors}

    before = snapshot_state()

    normalized = {
        "article_id": article_id,
        "verification_status": "completed",
        "verified_at": utc_now(),
        "items": payload.get("items", []),
    }
    save_default_article_files(article_id)
    write_json(verification_path(article_id), normalized)

    refresh_record_states()
    recompute_posteriors()
    after = snapshot_state()

    append_jsonl(
        CHANGE_LOG_PATH,
        {
            "timestamp": utc_now(),
            "event": "save_verification",
            "article_id": article_id,
            "item_count": len(normalized["items"]),
        },
    )

    next_task = get_next_task()
    return {
        "ok": True,
        "article_id": article_id,
        "item_count": len(normalized["items"]),
        "state_diff": format_state_diff(before, after),
        "next_task": next_task,
    }


# ---------------------------------------------------------------------------
# Staged approval workflow: stage → (LLM self-approve or human override) → apply
# ---------------------------------------------------------------------------
# The old save-verification path committed verification.json and recomputed
# posteriors in one shot, which gave a human no window to intercept what the
# LLM had decided. The staged workflow splits that into:
#
#   1. stage-verification  — writes verification_draft.json and approval.json
#      to disk, prints both. No posterior change.
#   2. (optional) override-approval — user rewrites approval.json to refuse,
#      detach a claim, or mark for manual review.
#   3. apply-verification  — reads draft + approval, refuses unless the
#      approval decision is auto_approved / human_approved / human_overridden,
#      then writes verification.json and recomputes.
#
# approval.json is the permanent record of who decided what. When the LLM
# self-approves, it leaves a rationale the user can audit; when the human
# overrides, the override is recorded alongside the original LLM proposal.


APPROVAL_DECISIONS = {
    "auto_approved",      # LLM drafted and LLM decided it was safe to apply
    "needs_human",        # LLM explicitly wants the human to look before apply
    "human_approved",     # Human read it and signed off
    "human_overridden",   # Human changed the decision and/or draft
}

APPLY_ALLOWED_DECISIONS = {"auto_approved", "human_approved", "human_overridden"}


def validate_approval_payload(
    payload: Any,
    article_id: str,
    draft_items: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["approval payload must be a JSON object"]

    if payload.get("article_id") not in (None, article_id):
        errors.append(
            f"approval.article_id mismatch: payload says {payload.get('article_id')!r}, "
            f"target is {article_id!r}"
        )

    decision = payload.get("decision")
    if decision not in APPROVAL_DECISIONS:
        errors.append(
            f"approval.decision must be one of {sorted(APPROVAL_DECISIONS)} (got {decision!r})"
        )

    approver = payload.get("approver")
    if not isinstance(approver, str) or not approver.strip():
        errors.append("approval.approver must be a non-empty string (e.g. 'llm:claude' or 'human:byh21')")

    overall = payload.get("overall_rationale")
    if not isinstance(overall, str) or not overall.strip():
        errors.append("approval.overall_rationale must be a non-empty string")

    per_claim = payload.get("per_claim_decisions")
    if not isinstance(per_claim, list):
        errors.append("approval.per_claim_decisions must be a list")
        return errors

    draft_claim_ids = [item.get("claim_id") for item in draft_items]
    seen_claim_ids: set[str] = set()
    for index, entry in enumerate(per_claim):
        prefix = f"per_claim_decisions[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{prefix} must be an object")
            continue
        claim_id = entry.get("claim_id")
        if not isinstance(claim_id, str):
            errors.append(f"{prefix}.claim_id must be a string")
            continue
        if claim_id not in draft_claim_ids:
            errors.append(
                f"{prefix}.claim_id {claim_id!r} not found in the staged verification draft"
            )
        if claim_id in seen_claim_ids:
            errors.append(f"{prefix}.claim_id duplicate: {claim_id!r}")
        seen_claim_ids.add(claim_id)
        entry_decision = entry.get("decision")
        if entry_decision not in {"accept", "accept_as_fact", "reject", "defer"}:
            errors.append(
                f"{prefix}.decision must be one of "
                f"['accept', 'accept_as_fact', 'reject', 'defer'] (got {entry_decision!r})"
            )
        reasoning = entry.get("reasoning")
        if not isinstance(reasoning, str) or not reasoning.strip():
            errors.append(f"{prefix}.reasoning must be a non-empty string")

    # Warn if the approval did not cover every draft claim. This is not a
    # hard error so humans can override with a coarser rationale, but it
    # catches the common LLM mistake of skipping claims silently.
    missing = [cid for cid in draft_claim_ids if cid not in seen_claim_ids]
    if missing:
        errors.append(
            f"approval does not cover draft claim ids: {missing}. "
            f"Every staged claim needs a per-claim decision (accept/accept_as_fact/reject/defer)."
        )

    return errors


def stage_verification(
    article_id: str,
    draft_payload: Any,
    approval_payload: Any,
) -> dict[str, Any]:
    """Write verification_draft.json + approval.json, do NOT touch posteriors."""
    bootstrap_state_files()
    record = read_json(article_record_path(article_id), default=None)
    if record is None:
        raise ValueError(f"Unknown article_id: {article_id}")

    claims_doc = load_claims(article_id)
    draft_errors = validate_verification_payload(draft_payload, article_id, claims_doc)

    if not isinstance(draft_payload, dict):
        return {
            "ok": False,
            "article_id": article_id,
            "stage": "verification_draft",
            "errors": draft_errors or ["draft payload must be a JSON object"],
        }

    draft_items = draft_payload.get("items", []) if isinstance(draft_payload, dict) else []
    approval_errors = validate_approval_payload(approval_payload, article_id, draft_items)

    if draft_errors or approval_errors:
        return {
            "ok": False,
            "article_id": article_id,
            "stage": "verification_draft",
            "errors": {
                "verification_draft": draft_errors,
                "approval": approval_errors,
            },
        }

    normalized_draft = {
        "article_id": article_id,
        "verification_status": "drafted",
        "drafted_at": utc_now(),
        "items": draft_items,
    }
    normalized_approval = {
        "article_id": article_id,
        "timestamp": utc_now(),
        "approver": approval_payload["approver"],
        "decision": approval_payload["decision"],
        "overall_rationale": approval_payload["overall_rationale"],
        "per_claim_decisions": approval_payload["per_claim_decisions"],
        "human_override": approval_payload.get("human_override"),
    }

    write_json(draft_verification_path(article_id), normalized_draft)
    write_json(approval_path(article_id), normalized_approval)

    append_jsonl(
        CHANGE_LOG_PATH,
        {
            "timestamp": utc_now(),
            "event": "stage_verification",
            "article_id": article_id,
            "approver": normalized_approval["approver"],
            "decision": normalized_approval["decision"],
            "item_count": len(draft_items),
        },
    )

    return {
        "ok": True,
        "article_id": article_id,
        "draft_path": relpath_from_root(draft_verification_path(article_id)),
        "approval_path": relpath_from_root(approval_path(article_id)),
        "decision": normalized_approval["decision"],
        "approver": normalized_approval["approver"],
        "item_count": len(draft_items),
        "overall_rationale": normalized_approval["overall_rationale"],
        "per_claim_decisions": normalized_approval["per_claim_decisions"],
        "ready_to_apply": normalized_approval["decision"] in APPLY_ALLOWED_DECISIONS,
        "next_step": (
            "python scripts/bayesian_reader.py apply-verification "
            f"--article-id {article_id}"
            if normalized_approval["decision"] in APPLY_ALLOWED_DECISIONS
            else (
                "approval.decision is 'needs_human'; edit the files or call "
                "override-approval before apply-verification"
            )
        ),
    }


def apply_verification(article_id: str) -> dict[str, Any]:
    """Read staged draft + approval, commit to verification.json, recompute."""
    bootstrap_state_files()
    record = read_json(article_record_path(article_id), default=None)
    if record is None:
        raise ValueError(f"Unknown article_id: {article_id}")

    draft_path = draft_verification_path(article_id)
    approval_file = approval_path(article_id)
    if not draft_path.exists() or not approval_file.exists():
        return {
            "ok": False,
            "article_id": article_id,
            "errors": [
                "no staged verification draft found; run stage-verification first"
            ],
        }

    draft = read_json(draft_path, default={})
    approval = read_json(approval_file, default={})

    decision = approval.get("decision")
    if decision not in APPLY_ALLOWED_DECISIONS:
        return {
            "ok": False,
            "article_id": article_id,
            "decision": decision,
            "errors": [
                f"approval.decision is {decision!r}; apply requires one of "
                f"{sorted(APPLY_ALLOWED_DECISIONS)}. Call override-approval to "
                f"change it, or edit approval.json manually."
            ],
        }

    claims_doc = load_claims(article_id)
    draft_items = draft.get("items", [])
    draft_errors = validate_verification_payload(
        {
            "article_id": article_id,
            "verification_status": "completed",
            "items": draft_items,
        },
        article_id,
        claims_doc,
    )
    if draft_errors:
        return {
            "ok": False,
            "article_id": article_id,
            "errors": {"verification_draft": draft_errors},
        }

    before = snapshot_state()

    normalized = {
        "article_id": article_id,
        "verification_status": "completed",
        "verified_at": utc_now(),
        "items": draft_items,
        "approval_ref": {
            "approver": approval.get("approver"),
            "decision": decision,
            "timestamp": approval.get("timestamp"),
        },
    }
    save_default_article_files(article_id)
    write_json(verification_path(article_id), normalized)

    refresh_record_states()
    recompute_posteriors()
    after = snapshot_state()

    append_jsonl(
        CHANGE_LOG_PATH,
        {
            "timestamp": utc_now(),
            "event": "apply_verification",
            "article_id": article_id,
            "item_count": len(draft_items),
            "approver": approval.get("approver"),
            "decision": decision,
        },
    )

    next_task = get_next_task()
    return {
        "ok": True,
        "article_id": article_id,
        "item_count": len(draft_items),
        "approver": approval.get("approver"),
        "decision": decision,
        "state_diff": format_state_diff(before, after),
        "next_task": next_task,
    }


def override_approval(
    article_id: str,
    decision: str,
    reason: str,
    approver: str = "human:cli",
    detach_claim_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Rewrite an existing approval.json with a human override.

    `detach_claim_ids` removes the hypothesis attachment of each listed claim
    in verification_draft.json (setting hypothesis_id and the ordinal fields
    to null), which is the most common override: "keep the fact, drop the
    hypothesis mapping."
    """
    bootstrap_state_files()
    if decision not in APPROVAL_DECISIONS:
        raise ValueError(
            f"decision must be one of {sorted(APPROVAL_DECISIONS)} (got {decision!r})"
        )
    if not reason.strip():
        raise ValueError("override reason must be non-empty")

    draft_path = draft_verification_path(article_id)
    approval_file = approval_path(article_id)
    if not draft_path.exists() or not approval_file.exists():
        return {
            "ok": False,
            "article_id": article_id,
            "errors": ["no staged verification draft to override"],
        }

    draft = read_json(draft_path, default={})
    approval = read_json(approval_file, default={})

    edits: list[dict[str, Any]] = []
    if detach_claim_ids:
        for item in draft.get("items", []):
            if item.get("claim_id") in detach_claim_ids and item.get("hypothesis_id"):
                edits.append(
                    {
                        "claim_id": item["claim_id"],
                        "action": "detach_hypothesis",
                        "previous_hypothesis_id": item.get("hypothesis_id"),
                    }
                )
                item["hypothesis_id"] = None
                for key in ("source_trust", "evidence_direction", "evidence_strength"):
                    item.pop(key, None)
        draft["verification_status"] = "drafted"
        draft["last_modified_at"] = utc_now()
        write_json(draft_path, draft)

    approval["human_override"] = {
        "timestamp": utc_now(),
        "approver": approver,
        "previous_decision": approval.get("decision"),
        "reason": reason,
        "edits": edits,
    }
    approval["decision"] = decision
    approval["approver"] = approver
    write_json(approval_file, approval)

    append_jsonl(
        CHANGE_LOG_PATH,
        {
            "timestamp": utc_now(),
            "event": "override_approval",
            "article_id": article_id,
            "decision": decision,
            "detach": detach_claim_ids or [],
        },
    )

    return {
        "ok": True,
        "article_id": article_id,
        "decision": decision,
        "approver": approver,
        "detached": detach_claim_ids or [],
        "edits": edits,
    }


def _pending_record_sort_key(record: dict[str, Any]) -> str:
    return str(record.get("ingested_at") or record.get("article_id") or "")


def list_pending_tasks(stage: str | None = None) -> list[dict[str, Any]]:
    records = load_all_article_records()
    tasks: list[dict[str, Any]] = []

    for record in sorted(records, key=_pending_record_sort_key):
        article_id = record.get("article_id")
        if not article_id:
            continue
        content_status = record.get("content_state", {}).get("full_text_status")
        if content_status not in {"acquired", "partial"}:
            continue

        claims_doc = load_claims(article_id)
        verification_doc = load_verification(article_id)
        claim_status = claims_doc.get("claim_extraction_status", "not_started")
        verification_status = verification_doc.get("verification_status", "not_started")

        if claim_status != "completed":
            task_stage = "extract_claims"
        elif verification_status != "completed":
            task_stage = "verify_claims"
        else:
            continue

        if stage and task_stage != stage:
            continue

        tasks.append(
            {
                "article_id": article_id,
                "stage": task_stage,
                "title": record.get("title"),
                "url": record.get("url"),
                "content_status": content_status,
                "ingested_at": record.get("ingested_at"),
            }
        )

    return tasks


def get_next_task(stage: str | None = None) -> dict[str, Any] | None:
    tasks = list_pending_tasks(stage=stage)
    return tasks[0] if tasks else None


CLAIMS_SCHEMA_EXAMPLE = {
    "article_id": "<same as target article_id>",
    "claim_extraction_status": "completed",
    "claims": [
        {
            "id": "<short_snake_case_id, unique within this file>",
            "type": "event | technique | tool",
            "text": "<one-sentence statement the article actually makes>",
            "hypothesis_candidates": ["<hypothesis_id from hypotheses.json, or empty list>"],
        }
    ],
}

VERIFICATION_SCHEMA_EXAMPLE = {
    "article_id": "<same as target article_id>",
    "verification_status": "completed",
    "items": [
        {
            "claim_id": "<id from claims.json>",
            "hypothesis_id": "<hypothesis_id or null>",
            "status": "verified | partially_verified | conflicted | unverified",
            "source_type": "<free-form tag, e.g. official_engineering_post>",
            "source_title": "<primary source title (required for verified/partially_verified)>",
            "source_url": "<primary source url (required for verified/partially_verified)>",
            "assessment": "<why this evidence supports or fails to support the hypothesis>",
            "source_trust": "weak | moderate | strong (required iff hypothesis_id is set)",
            "evidence_direction": "support | against (required iff hypothesis_id is set)",
            "evidence_strength": "slight | moderate | strong (required iff hypothesis_id is set)",
        }
    ],
}


def build_next_task_payload(
    article_id: str | None = None,
    stage: str | None = None,
    include_canonical_text: bool = True,
) -> dict[str, Any]:
    bootstrap_state_files()

    if article_id is None:
        task = get_next_task(stage=stage)
        if task is None:
            return {"pending": False, "reason": "no pending tasks"}
        article_id = task["article_id"]
        task_stage = task["stage"]
    else:
        task_candidates = [t for t in list_pending_tasks() if t["article_id"] == article_id]
        if not task_candidates:
            return {
                "pending": False,
                "article_id": article_id,
                "reason": "article is not pending extraction or verification",
            }
        task_stage = task_candidates[0]["stage"]
        if stage and stage != task_stage:
            return {
                "pending": False,
                "article_id": article_id,
                "reason": f"requested stage {stage!r} but current stage is {task_stage!r}",
            }

    record = read_json(article_record_path(article_id), default=None)
    if record is None:
        return {"pending": False, "article_id": article_id, "reason": "record not found"}

    canonical_file = article_dir(article_id) / "canonical_text.txt"
    canonical_text = ""
    if include_canonical_text and canonical_file.exists():
        canonical_text = canonical_file.read_text(encoding="utf-8")

    hypothesis_index = load_hypothesis_index()
    active_hypotheses = [
        {
            "id": item.get("id"),
            "statement": item.get("statement"),
            "rationale": item.get("rationale"),
            "posterior_probability": item.get("posterior_probability"),
        }
        for item in hypothesis_index.values()
        if item.get("status", "active") == "active"
    ]

    claims_doc = load_claims(article_id)
    verification_doc = load_verification(article_id)

    payload: dict[str, Any] = {
        "pending": True,
        "article_id": article_id,
        "stage": task_stage,
        "title": record.get("title"),
        "url": record.get("url"),
        "ingested_at": record.get("ingested_at"),
        "content_status": record.get("content_state", {}).get("full_text_status"),
        "canonical_text_path": relpath_from_root(canonical_file),
        "canonical_text": canonical_text if include_canonical_text else None,
        "active_hypotheses": active_hypotheses,
        "existing_claims": claims_doc if task_stage == "verify_claims" else None,
        "existing_verification": verification_doc if verification_doc.get("items") else None,
        "write_command": (
            f"python scripts/bayesian_reader.py save-claims --article-id {article_id} --file -"
            if task_stage == "extract_claims"
            else f"python scripts/bayesian_reader.py save-verification --article-id {article_id} --file -"
        ),
        "expected_schema_example": (
            CLAIMS_SCHEMA_EXAMPLE if task_stage == "extract_claims" else VERIFICATION_SCHEMA_EXAMPLE
        ),
    }
    return payload


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def html_escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def probability_percent(value: float | None) -> str:
    """Deprecated: kept only for backward-compat. Prefer band_label()."""
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def band_label(value: float | None) -> str:
    """Return a bilingual band label for display in reports, e.g.
    'very_likely · 几乎可以确认'. No numeric probability."""
    band_name = posterior_band(value)
    chinese = POSTERIOR_BAND_LABELS.get(band_name, "")
    if chinese:
        return f"{band_name} · {chinese}"
    return band_name


def band_fill_fraction(value: float | None) -> float:
    """Return 0.0..1.0 fill fraction for the progress bar, snapped to the
    midpoint of the enclosing band so the visual also coarsens."""
    if value is None:
        return 0.0
    band_name = posterior_band(value)
    for name, low, high in POSTERIOR_BANDS:
        if name == band_name:
            # Snap to midpoint, but clamp the last band (1.01) to 0.95
            midpoint = (low + min(high, 1.0)) / 2
            return max(0.0, min(1.0, midpoint))
    return 0.0


def card_list(items: list[str], empty_text: str = "暂无") -> str:
    if not items:
        return f"<p class='muted'>{html_escape(empty_text)}</p>"
    return "<ul class='pill-list'>" + "".join(
        f"<li>{html_escape(item)}</li>" for item in items
    ) + "</ul>"


def build_article_detail_html(record: dict[str, Any], hypothesis_index: dict[str, dict[str, Any]]) -> str:
    article_id = record["article_id"]
    claims = read_json(claims_path(article_id), default={"claims": []})
    verification = read_json(verification_path(article_id), default={"items": []})

    verified_rows: list[str] = []
    for item in verification.get("items", []):
        claim_text = next(
            (claim["text"] for claim in claims.get("claims", []) if claim.get("id") == item.get("claim_id")),
            item.get("claim_id", ""),
        )
        hypothesis_statement = ""
        if item.get("hypothesis_id") and item["hypothesis_id"] in hypothesis_index:
            hypothesis_statement = hypothesis_index[item["hypothesis_id"]]["statement"]

        source_html = ""
        if item.get("source_url"):
            source_html = (
                f"<a href='{html_escape(item['source_url'])}' target='_blank' rel='noreferrer'>"
                f"{html_escape(item.get('source_title') or item['source_url'])}</a>"
            )
        elif item.get("source_title"):
            source_html = html_escape(item["source_title"])
        else:
            source_html = "<span class='muted'>无可展示来源</span>"

        metrics: list[str] = []
        if is_ordinal_item(item):
            trust = item.get("source_trust")
            direction = item.get("evidence_direction")
            strength = item.get("evidence_strength")
            if trust and strength and direction:
                metrics.append(f"{direction}: {strength} from {trust} source")
            elif not item.get("hypothesis_id"):
                metrics.append("verified fact, no hypothesis attached")
        else:
            # Legacy float-based items: summarize without the raw numbers
            # so the report still obeys the "no false precision" rule.
            if item.get("weight") is not None or item.get("likelihood_ratio") is not None:
                metrics.append("legacy weighted evidence")

        verified_rows.append(
            "".join(
                [
                    "<div class='evidence-row'>",
                    f"<div class='evidence-top'><span class='status status-{html_escape(item.get('status', 'unknown'))}'>{html_escape(item.get('status', 'unknown'))}</span>",
                    f"<span class='muted'>{html_escape(' · '.join(metrics))}</span></div>",
                    f"<p class='claim'>{html_escape(claim_text)}</p>",
                    f"<p class='muted'>{html_escape(hypothesis_statement) if hypothesis_statement else '未映射到趋势假设'}</p>",
                    f"<p class='muted'>{source_html}</p>",
                    f"<p>{html_escape(item.get('assessment', ''))}</p>",
                    "</div>",
                ]
            )
        )

    return "".join(
        [
            f"<details class='article-card' data-status='{html_escape(record.get('analysis_state', {}).get('bayesian_status', 'unknown'))}' data-article-id='{html_escape(record['article_id'])}'>",
            f"<summary><span>{html_escape(record['title'])}</span><span class='status status-{html_escape(record.get('analysis_state', {}).get('bayesian_status', 'unknown'))}'>{html_escape(record.get('analysis_state', {}).get('bayesian_status', 'unknown'))}</span></summary>",
            "<div class='article-body'>",
            f"<p><a href='{html_escape(record['url'])}' target='_blank' rel='noreferrer'>{html_escape(record['url'])}</a></p>",
            "<div class='article-grid'>",
            "<section>",
            "<h4>事件</h4>",
            card_list(record.get("article_summary", {}).get("events", []), "没有纳入事件摘要"),
            "</section>",
            "<section>",
            "<h4>技术</h4>",
            card_list(record.get("article_summary", {}).get("techniques", []), "没有纳入技术摘要"),
            "</section>",
            "<section>",
            "<h4>工具</h4>",
            card_list(record.get("article_summary", {}).get("tools", []), "没有纳入工具摘要"),
            "</section>",
            "</div>",
            "<section>",
            "<h4>核实结果</h4>",
            "".join(verified_rows) if verified_rows else "<p class='muted'>暂无核实条目</p>",
            "</section>",
            "</div>",
            "</details>",
        ]
    )


def build_report_html() -> str:
    framework = read_json(FRAMEWORK_PATH, default={})
    hypotheses = read_json(HYPOTHESES_PATH, default={"hypotheses": []})
    synthesis = read_json(SYNTHESIS_PATH, default={})
    github_config = read_github_config()
    records = load_all_article_records()

    hypothesis_index = {item["id"]: item for item in hypotheses.get("hypotheses", [])}
    included_ids = set(synthesis.get("included_articles", []))
    included_records = [record for record in records if record.get("article_id") in included_ids]
    pending_records = [
        record
        for record in records
        if record.get("article_id") not in included_ids
        and record.get("analysis_state", {}).get("claim_extraction_status") != "completed"
    ]
    excluded_records = [record for record in records if record.get("article_id") not in included_ids]

    active_hypotheses = sorted(
        hypotheses.get("hypotheses", []),
        key=lambda item: item.get("posterior_probability", 0.0),
        reverse=True,
    )

    hypothesis_cards = "".join(
        [
            "".join(
                [
                    "<article class='hypothesis-card'>",
                    f"<div class='score' style='--pct:{band_fill_fraction(item.get('posterior_probability')) * 100:.1f}%'>",
                    "<div class='score-bar'></div>",
                    f"<span>{html_escape(band_label(item.get('posterior_probability')))}</span>",
                    "</div>",
                    f"<h3>{html_escape(item.get('statement', ''))}</h3>",
                    f"<p>{html_escape(item.get('rationale', ''))}</p>",
                    f"<p class='muted'>证据条数: {len(item.get('supporting_items', []))}</p>",
                    "</article>",
                ]
            )
            for item in active_hypotheses
        ]
    )

    narrative_html = "".join(
        f"<li>{html_escape(item)}</li>" for item in synthesis.get("trend_narrative", [])
    )

    tool_cards = "".join(
        [
            "".join(
                [
                    "<article class='tool-card'>",
                    f"<p class='tool-category'>{html_escape(item.get('category', ''))}</p>",
                    f"<h3>{html_escape(item.get('name', ''))}</h3>",
                    f"<p><a href='{html_escape(item.get('url', ''))}' target='_blank' rel='noreferrer'>{html_escape(item.get('url', ''))}</a></p>",
                    "</article>",
                ]
            )
            for item in synthesis.get("tool_index", [])
        ]
    )

    excluded_html = "".join(
        [
            "".join(
                [
                    "<article class='excluded-card'>",
                    f"<h3>{html_escape(next((record['title'] for record in records if record['article_id'] == item.get('article_id')), item.get('article_id', '')))}</h3>",
                    f"<p class='muted'>{html_escape(item.get('reason', ''))}</p>",
                    "</article>",
                ]
            )
            for item in synthesis.get("excluded_articles", [])
        ]
    )

    included_articles_html = "".join(
        build_article_detail_html(record, hypothesis_index)
        for record in sorted(included_records, key=lambda item: item.get("title", ""))
    )

    pending_articles_html = "".join(
        build_article_detail_html(record, hypothesis_index)
        for record in sorted(pending_records, key=lambda item: item.get("title", ""))
    )

    excluded_articles_html = "".join(
        build_article_detail_html(record, hypothesis_index)
        for record in sorted(excluded_records, key=lambda item: item.get("title", ""))
        if record.get("analysis_state", {}).get("claim_extraction_status") == "completed"
    )

    principles_html = "".join(
        f"<li>{html_escape(item)}</li>" for item in framework.get("principles", [])
    )

    note_html = "".join(
        f"<li>{html_escape(item)}</li>" for item in synthesis.get("notes", [])
    )

    summary_stats = [
        ("纳入文章", len(included_records)),
        ("待处理文章", len(pending_records)),
        ("排除文章", len(synthesis.get("excluded_articles", []))),
        ("活跃趋势", len(active_hypotheses)),
        ("最近更新", synthesis.get("last_recomputed_at", "N/A")),
    ]
    stats_html = "".join(
        f"<article class='stat-card'><p class='stat-label'>{html_escape(label)}</p><p class='stat-value'>{html_escape(value)}</p></article>"
        for label, value in summary_stats
    )
    pages_link_html = ""
    if github_config.get("pages_url"):
        pages_link_html = (
            "<p class='muted'>对外访问地址："
            f"<a href='{html_escape(github_config['pages_url'])}' target='_blank' rel='noreferrer'>{html_escape(github_config['pages_url'])}</a>"
            "</p>"
        )
    repo_link_html = ""
    issue_link_html = ""
    if github_config.get("repo"):
        repo_url = f"https://github.com/{github_config['repo']}"
        repo_link_html = (
            "<p class='muted'>GitHub 仓库："
            f"<a href='{html_escape(repo_url)}' target='_blank' rel='noreferrer'>{html_escape(github_config['repo'])}</a>"
            "</p>"
        )
        issue_link_html = (
            "<p class='muted'>手机投递入口："
            f"<a href='{html_escape(repo_url + '/issues/new/choose')}' target='_blank' rel='noreferrer'>新建文章 Issue</a>"
            "</p>"
        )

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>大模型趋势核实报告</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --paper: #fffdf8;
      --ink: #1f2430;
      --muted: #69707d;
      --line: #ddd2bf;
      --accent: #0f766e;
      --accent-soft: #d7f0eb;
      --warm: #9a3412;
      --shadow: 0 10px 30px rgba(31, 36, 48, 0.08);
      --radius: 18px;
    }}
    * {{ box-sizing: border-box; scroll-behavior: smooth; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.14), transparent 30%),
        radial-gradient(circle at top right, rgba(154, 52, 18, 0.08), transparent 28%),
        var(--bg);
      line-height: 1.6;
    }}
    a {{ color: var(--accent); }}
    .page {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(255, 253, 248, 0.94), rgba(250, 246, 238, 0.96));
      border: 1px solid var(--line);
      border-radius: 28px;
      padding: 28px;
      box-shadow: var(--shadow);
    }}
    .eyebrow {{
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-size: 0.82rem;
      color: var(--warm);
      margin: 0 0 8px;
    }}
    h1, h2, h3, h4 {{
      margin: 0 0 12px;
      line-height: 1.25;
    }}
    h1 {{
      font-size: clamp(2rem, 5vw, 3.6rem);
      max-width: 12ch;
    }}
    .hero p {{
      max-width: 68ch;
      margin: 0 0 6px;
    }}
    .section {{
      margin-top: 28px;
      background: rgba(255, 253, 248, 0.76);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 22px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(8px);
    }}
    .section-nav {{
      position: sticky;
      top: 12px;
      z-index: 20;
      padding: 14px 16px;
    }}
    .nav-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .nav-row a {{
      display: inline-flex;
      align-items: center;
      padding: 8px 12px;
      border-radius: 999px;
      text-decoration: none;
      color: var(--ink);
      background: var(--paper);
      border: 1px solid var(--line);
      font-size: 0.92rem;
    }}
    .stats,
    .hypothesis-grid,
    .tool-grid,
    .article-grid {{
      display: grid;
      gap: 16px;
    }}
    .stats {{
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      margin-top: 22px;
    }}
    .stat-card, .hypothesis-card, .tool-card, .excluded-card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
    }}
    .stat-label, .tool-category {{
      color: var(--muted);
      margin: 0 0 8px;
      font-size: 0.92rem;
    }}
    .stat-value {{
      margin: 0;
      font-size: 1.08rem;
      font-weight: 700;
    }}
    .hypothesis-grid {{
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    }}
    .score {{
      position: relative;
      margin-bottom: 14px;
    }}
    .score-bar {{
      height: 10px;
      background: #ebe5d8;
      border-radius: 999px;
      overflow: hidden;
      position: relative;
    }}
    .score-bar::before {{
      content: "";
      position: absolute;
      inset: 0;
      width: var(--pct);
      background: linear-gradient(90deg, var(--accent), #14b8a6);
      border-radius: inherit;
    }}
    .score span {{
      display: block;
      margin-top: 8px;
      font-size: 0.85rem;
      color: var(--muted);
      letter-spacing: 0.02em;
    }}
    .muted {{
      color: var(--muted);
    }}
    ol, ul {{
      margin: 0;
      padding-left: 1.2rem;
    }}
    .tool-grid {{
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }}
    .controls {{
      display: grid;
      gap: 12px;
      margin: 14px 0 18px;
    }}
    .controls input {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 14px;
      font: inherit;
      background: var(--paper);
    }}
    .filter-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .filter-btn {{
      border: 1px solid var(--line);
      background: var(--paper);
      border-radius: 999px;
      padding: 8px 12px;
      font: inherit;
      cursor: pointer;
      color: var(--ink);
    }}
    .filter-btn.is-active {{
      background: var(--accent-soft);
      border-color: #7dd3c8;
      color: #0b5b55;
    }}
    .article-card {{
      border: 1px solid var(--line);
      border-radius: 16px;
      background: var(--paper);
      margin-bottom: 14px;
      overflow: hidden;
    }}
    .article-card summary {{
      list-style: none;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      padding: 16px 18px;
      font-weight: 700;
    }}
    .article-card summary::-webkit-details-marker {{
      display: none;
    }}
    .article-body {{
      border-top: 1px solid var(--line);
      padding: 18px;
    }}
    .article-grid {{
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      margin-bottom: 18px;
    }}
    .pill-list {{
      list-style: none;
      padding: 0;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .pill-list li {{
      padding: 8px 10px;
      border-radius: 999px;
      background: #f1ece1;
      border: 1px solid #e3d9c8;
      font-size: 0.92rem;
    }}
    .evidence-row {{
      border-top: 1px solid #eee5d6;
      padding: 14px 0;
    }}
    .evidence-row:first-of-type {{
      border-top: 0;
      padding-top: 0;
    }}
    .evidence-top {{
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 8px;
    }}
    .status {{
      display: inline-flex;
      align-items: center;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.82rem;
      font-weight: 700;
      border: 1px solid transparent;
      white-space: nowrap;
    }}
    .status-verified, .status-included {{
      background: #dcfce7;
      color: #166534;
      border-color: #bbf7d0;
    }}
    .status-partially_verified {{
      background: #fef3c7;
      color: #92400e;
      border-color: #fde68a;
    }}
    .status-unverified, .status-excluded_after_verification {{
      background: #fee2e2;
      color: #991b1b;
      border-color: #fecaca;
    }}
    .status-held_out_pending_better_evidence, .status-excluded_until_verified {{
      background: #e0e7ff;
      color: #3730a3;
      border-color: #c7d2fe;
    }}
    .claim {{
      font-weight: 600;
      margin: 0 0 8px;
    }}
    .split {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 18px;
    }}
    @media (max-width: 900px) {{
      .split {{
        grid-template-columns: 1fr;
      }}
      .page {{
        padding: 16px;
      }}
      .hero {{
        padding: 22px;
      }}
      .article-card summary {{
        flex-direction: column;
        align-items: flex-start;
      }}
      .section-nav {{
        top: 8px;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <p class="eyebrow">Verified Bayesian Synthesis</p>
      <h1>现代大模型趋势核实报告</h1>
      <p>这不是文章摘要页，而是“全文抓取 -> 命题提取 -> 主源核实 -> 贝叶斯更新”后的阅读版结果。未核实内容不会直接进入趋势判断。</p>
      {repo_link_html}
      {issue_link_html}
      {pages_link_html}
      <div class="stats">{stats_html}</div>
    </section>

    <nav class="section section-nav">
      <div class="nav-row">
        <a href="#trends">核心趋势</a>
        <a href="#narrative">本轮结论</a>
        <a href="#tools">工具索引</a>
        <a href="#pending">待处理</a>
        <a href="#articles">文章证据</a>
        <a href="#excluded">排除项</a>
      </div>
    </nav>

    <section class="section" id="trends">
      <h2>核心趋势</h2>
      <p class="muted">后验概率越高，说明当前证据越支持这个趋势；不是“绝对为真”，而是“在现有样本下更值得作为默认判断”。</p>
      <div class="hypothesis-grid">{hypothesis_cards}</div>
    </section>

    <section class="section split" id="narrative">
      <div>
        <h2>本轮结论</h2>
        <ol>{narrative_html}</ol>
      </div>
      <div>
        <h2>方法约束</h2>
        <ol>{principles_html}</ol>
      </div>
    </section>

    <section class="section" id="tools">
      <h2>工具与项目索引</h2>
      <div class="tool-grid">{tool_cards}</div>
    </section>

    <section class="section" id="pending">
      <h2>待处理文章</h2>
      <p class="muted">这里展示已经进入状态库、且全文已抓到或等待后续处理的文章。它们还没有进入趋势判断。</p>
      {pending_articles_html if pending_articles_html else "<p class='muted'>暂无</p>"}
    </section>

    <section class="section" id="articles">
      <h2>文章证据</h2>
      <p class="muted">每篇文章只保留高价值、可被核实的 claim。营销语句和弱来源内容会被保留在原始快照里，但不会进入趋势更新。</p>
      <div class="controls">
        <input id="articleSearch" type="search" placeholder="搜索文章标题、事件、技术、工具">
        <div class="filter-row">
          <button type="button" class="filter-btn is-active" data-filter="all">全部</button>
          <button type="button" class="filter-btn" data-filter="included">已纳入</button>
          <button type="button" class="filter-btn" data-filter="excluded_after_verification">已排除</button>
        </div>
      </div>
      {included_articles_html}
    </section>

    <section class="section split" id="excluded">
      <div>
        <h2>已排除文章</h2>
        {excluded_html if excluded_html else "<p class='muted'>暂无</p>"}
      </div>
      <div>
        <h2>排除但保留记录</h2>
        {excluded_articles_html if excluded_articles_html else "<p class='muted'>暂无</p>"}
      </div>
    </section>

    <section class="section">
      <h2>阅读说明</h2>
      <ol>{note_html}</ol>
    </section>
  </main>
  <script>
    (() => {{
      const cards = [...document.querySelectorAll('.article-card')];
      const search = document.getElementById('articleSearch');
      const buttons = [...document.querySelectorAll('.filter-btn')];
      let activeFilter = 'all';

      function applyFilters() {{
        const q = (search?.value || '').trim().toLowerCase();
        cards.forEach((card) => {{
          const text = card.textContent.toLowerCase();
          const status = card.dataset.status || '';
          const matchesQuery = !q || text.includes(q);
          const matchesFilter = activeFilter === 'all' || status === activeFilter;
          card.style.display = matchesQuery && matchesFilter ? '' : 'none';
        }});
      }}

      search?.addEventListener('input', applyFilters);
      buttons.forEach((button) => {{
        button.addEventListener('click', () => {{
          activeFilter = button.dataset.filter || 'all';
          buttons.forEach((node) => node.classList.toggle('is-active', node === button));
          applyFilters();
        }});
      }});
      applyFilters();
    }})();
  </script>
</body>
</html>
"""
    return html_content


def build_report(output_path: Path = REPORT_PATH) -> dict[str, Any]:
    bootstrap_state_files()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_content = build_report_html()
    output_path.write_text(html_content, encoding="utf-8")

    append_jsonl(
        CHANGE_LOG_PATH,
        {
            "timestamp": utc_now(),
            "event": "build_report",
            "output_path": str(output_path),
        },
    )
    return {"output_path": str(output_path)}


def run_pipeline(
    repo: str | None = None,
    label: str | None = None,
    state: str = "open",
    issue_limit: int = 100,
    fetch_limit: int = 20,
    force_fetch: bool = False,
    output_path: Path = REPORT_PATH,
    skip_sync_issues: bool = False,
    write_config_flag: bool = False,
) -> dict[str, Any]:
    bootstrap_state_files()

    sync_result: dict[str, Any] | None = None
    if not skip_sync_issues:
        sync_result = sync_github_issues(
            repo=repo,
            label=label,
            state=state,
            limit=issue_limit,
            write_config_flag=write_config_flag,
        )

    fetch_result = fetch_pending(limit=fetch_limit, force=force_fetch)
    refresh_result = refresh_record_states()
    recompute_result = recompute_posteriors()
    report_result = build_report(output_path=output_path)
    status_result = summarize_status()

    pending_tasks = list_pending_tasks()
    pending_summary = {
        "total": len(pending_tasks),
        "extract_claims": sum(1 for t in pending_tasks if t["stage"] == "extract_claims"),
        "verify_claims": sum(1 for t in pending_tasks if t["stage"] == "verify_claims"),
        "next_task": pending_tasks[0] if pending_tasks else None,
    }

    result = {
        "synced_issues": sync_result,
        "fetch": fetch_result,
        "refresh": refresh_result,
        "recompute": recompute_result,
        "report": report_result,
        "status": status_result,
        "pending": pending_summary,
    }

    append_jsonl(
        CHANGE_LOG_PATH,
        {
            "timestamp": utc_now(),
            "event": "run_pipeline",
            "repo": repo,
            "label": label,
            "state": state,
            "issue_limit": issue_limit,
            "fetch_limit": fetch_limit,
            "force_fetch": force_fetch,
            "skip_sync_issues": skip_sync_issues,
            "output_path": str(output_path),
            "result_summary": {
                "synced": 0 if sync_result is None else len(sync_result.get("imported", [])),
                "fetch_attempted": fetch_result.get("attempted"),
                "refreshed": refresh_result.get("refreshed"),
                "included_articles": len(recompute_result.get("included_articles", [])),
            },
        },
    )
    return result


def set_github_config(
    repo: str,
    pages_url: str | None = None,
    issue_label: str = "article",
) -> dict[str, Any]:
    config = write_github_config(repo=repo, pages_url=pages_url, issue_label=issue_label)
    append_jsonl(
        CHANGE_LOG_PATH,
        {
            "timestamp": utc_now(),
            "event": "set_github_config",
            "repo": repo,
            "pages_url": pages_url,
            "issue_label": issue_label,
        },
    )
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bayesian article analysis workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser("bootstrap", help="Import title/url pairs from url.md")
    bootstrap.add_argument("--source", default=str(ROOT / "url.md"))

    fetch = subparsers.add_parser("fetch-pending", help="Attempt to fetch pending articles")
    fetch.add_argument("--limit", type=int, default=20)
    fetch.add_argument("--force", action="store_true")

    subparsers.add_parser("status", help="Show content acquisition status")
    subparsers.add_parser("list-articles", help="List imported articles and their workflow state")
    refresh = subparsers.add_parser(
        "refresh-records",
        help="Refresh article record states from claims and verification files",
    )
    refresh.add_argument(
        "--recompute",
        action="store_true",
        help="After refreshing record states, also recompute hypothesis posteriors.",
    )
    subparsers.add_parser("recompute", help="Recompute hypothesis posteriors from verified claims")

    next_cmd = subparsers.add_parser(
        "next",
        help="Print the next pending task (extract or verify) with full context",
    )
    next_cmd.add_argument("--stage", choices=["extract_claims", "verify_claims"])
    next_cmd.add_argument("--article-id", help="Force a specific article instead of the queue head")
    next_cmd.add_argument(
        "--no-text",
        action="store_true",
        help="Omit canonical_text from the output (useful when piping to a listing)",
    )

    queue_cmd = subparsers.add_parser(
        "queue",
        help="List all pending extract/verify tasks (oldest first)",
    )
    queue_cmd.add_argument("--stage", choices=["extract_claims", "verify_claims"])

    save_claims_cmd = subparsers.add_parser(
        "save-claims",
        help="Validate and write claims.json for an article, then cascade refresh+recompute",
    )
    save_claims_cmd.add_argument("--article-id", required=True)
    save_claims_cmd.add_argument(
        "--file",
        help="Path to a JSON file, or '-' for stdin (default: stdin)",
    )

    save_verification_cmd = subparsers.add_parser(
        "save-verification",
        help="[advanced] Validate and write verification.json in one shot, skipping the approval stage",
    )
    save_verification_cmd.add_argument("--article-id", required=True)
    save_verification_cmd.add_argument(
        "--file",
        help="Path to a JSON file, or '-' for stdin (default: stdin)",
    )

    stage_cmd = subparsers.add_parser(
        "stage-verification",
        help="Stage a verification draft + LLM approval record without touching posteriors",
    )
    stage_cmd.add_argument("--article-id", required=True)
    stage_cmd.add_argument(
        "--draft",
        required=True,
        help="Path to the verification draft JSON file, or '-' for stdin",
    )
    stage_cmd.add_argument(
        "--approval",
        required=True,
        help="Path to the approval JSON file",
    )

    apply_cmd = subparsers.add_parser(
        "apply-verification",
        help="Read the staged draft + approval, write verification.json, recompute posteriors",
    )
    apply_cmd.add_argument("--article-id", required=True)

    override_cmd = subparsers.add_parser(
        "override-approval",
        help="Human override of a staged LLM approval decision",
    )
    override_cmd.add_argument("--article-id", required=True)
    override_cmd.add_argument(
        "--decision",
        required=True,
        choices=sorted(APPROVAL_DECISIONS),
    )
    override_cmd.add_argument("--reason", required=True)
    override_cmd.add_argument(
        "--approver",
        default="human:cli",
        help="Identifier for the human performing the override (default: human:cli)",
    )
    override_cmd.add_argument(
        "--detach",
        action="append",
        default=[],
        metavar="CLAIM_ID",
        help="Detach this claim's hypothesis attachment in the staged draft (repeatable)",
    )

    report = subparsers.add_parser("build-report", help="Generate a readable static HTML report")
    report.add_argument("--output", default=str(REPORT_PATH))
    sync = subparsers.add_parser("sync-issues", help="Import article links from GitHub Issues")
    sync.add_argument("--repo")
    sync.add_argument("--label")
    sync.add_argument("--state", default="open")
    sync.add_argument("--limit", type=int, default=100)
    sync.add_argument("--write-config", action="store_true")
    pipeline = subparsers.add_parser("run-pipeline", help="Run issue sync, fetch, recompute, and report build in one step")
    pipeline.add_argument("--repo")
    pipeline.add_argument("--label")
    pipeline.add_argument("--state", default="open")
    pipeline.add_argument("--issue-limit", type=int, default=100)
    pipeline.add_argument("--fetch-limit", type=int, default=20)
    pipeline.add_argument("--force-fetch", action="store_true")
    pipeline.add_argument("--output", default=str(REPORT_PATH))
    pipeline.add_argument("--skip-sync-issues", action="store_true")
    pipeline.add_argument("--write-config", action="store_true")
    config = subparsers.add_parser("set-github-config", help="Persist repo and Pages settings")
    config.add_argument("--repo", required=True)
    config.add_argument("--pages-url")
    config.add_argument("--issue-label", default="article")

    attach = subparsers.add_parser("attach-manual", help="Copy a manual artifact into an article folder")
    attach.add_argument("--article-id", required=True)
    attach.add_argument("--file", required=True)

    return parser


def main(argv: list[str]) -> int:
    bootstrap_state_files()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "bootstrap":
        result = bootstrap_from_url_md(Path(args.source))
        print_json(result)
        return 0

    if args.command == "fetch-pending":
        result = fetch_pending(limit=args.limit, force=args.force)
        print_json(result)
        return 0

    if args.command == "status":
        result = summarize_status()
        print_json(result)
        return 0

    if args.command == "list-articles":
        result = list_articles()
        print_json(result)
        return 0

    if args.command == "refresh-records":
        result = refresh_record_states()
        if getattr(args, "recompute", False):
            result["recompute"] = recompute_posteriors()
        print_json(result)
        return 0

    if args.command == "recompute":
        result = recompute_posteriors()
        print_json(result)
        return 0

    if args.command == "next":
        payload = build_next_task_payload(
            article_id=args.article_id,
            stage=args.stage,
            include_canonical_text=not args.no_text,
        )
        print_json(payload)
        return 0 if payload.get("pending") else 1

    if args.command == "queue":
        tasks = list_pending_tasks(stage=args.stage)
        print_json({"pending_count": len(tasks), "tasks": tasks})
        return 0

    if args.command == "save-claims":
        try:
            payload = _load_save_payload(args.file)
        except (FileNotFoundError, ValueError) as exc:
            print_json({"ok": False, "error": str(exc)})
            return 2
        result = save_claims(article_id=args.article_id, payload=payload)
        print_json(result)
        return 0 if result.get("ok") else 2

    if args.command == "save-verification":
        try:
            payload = _load_save_payload(args.file)
        except (FileNotFoundError, ValueError) as exc:
            print_json({"ok": False, "error": str(exc)})
            return 2
        result = save_verification(article_id=args.article_id, payload=payload)
        print_json(result)
        return 0 if result.get("ok") else 2

    if args.command == "stage-verification":
        try:
            draft_payload = _load_save_payload(args.draft)
            approval_file_arg = args.approval
            if approval_file_arg == "-":
                print_json({"ok": False, "error": "approval cannot come from stdin when draft also uses stdin"})
                return 2
            approval_payload = _load_save_payload(approval_file_arg)
        except (FileNotFoundError, ValueError) as exc:
            print_json({"ok": False, "error": str(exc)})
            return 2
        result = stage_verification(
            article_id=args.article_id,
            draft_payload=draft_payload,
            approval_payload=approval_payload,
        )
        print_json(result)
        return 0 if result.get("ok") else 2

    if args.command == "apply-verification":
        result = apply_verification(article_id=args.article_id)
        print_json(result)
        return 0 if result.get("ok") else 2

    if args.command == "override-approval":
        try:
            result = override_approval(
                article_id=args.article_id,
                decision=args.decision,
                reason=args.reason,
                approver=args.approver,
                detach_claim_ids=args.detach or None,
            )
        except ValueError as exc:
            print_json({"ok": False, "error": str(exc)})
            return 2
        print_json(result)
        return 0 if result.get("ok") else 2

    if args.command == "build-report":
        result = build_report(output_path=Path(args.output))
        print_json(result)
        return 0

    if args.command == "sync-issues":
        result = sync_github_issues(
            repo=args.repo,
            label=args.label,
            state=args.state,
            limit=args.limit,
            write_config_flag=args.write_config,
        )
        print_json(result)
        return 0

    if args.command == "run-pipeline":
        result = run_pipeline(
            repo=args.repo,
            label=args.label,
            state=args.state,
            issue_limit=args.issue_limit,
            fetch_limit=args.fetch_limit,
            force_fetch=args.force_fetch,
            output_path=Path(args.output),
            skip_sync_issues=args.skip_sync_issues,
            write_config_flag=args.write_config,
        )
        print_json(result)
        return 0

    if args.command == "set-github-config":
        result = set_github_config(
            repo=args.repo,
            pages_url=args.pages_url,
            issue_label=args.issue_label,
        )
        print_json(result)
        return 0

    if args.command == "attach-manual":
        result = attach_manual_file(article_id=args.article_id, source_file=Path(args.file))
        print_json(result)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

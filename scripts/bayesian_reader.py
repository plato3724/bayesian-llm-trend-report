from __future__ import annotations

import argparse
import hashlib
import html
import http.cookiejar
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
HYPOTHESIS_DETAIL_DIR = REPORT_DIR / "hypotheses"
FRAMEWORK_PATH = STATE_DIR / "framework.json"
HYPOTHESES_PATH = STATE_DIR / "hypotheses.json"
CANDIDATE_HYPOTHESES_PATH = STATE_DIR / "candidate_hypotheses.json"
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            entries.append(payload)
    return entries


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

    if not CANDIDATE_HYPOTHESES_PATH.exists():
        write_json(
            CANDIDATE_HYPOTHESES_PATH,
            {
                "created_at": utc_now(),
                "last_built_at": None,
                "candidates": [],
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
        # Accept URLs in either the issue body OR the title. Users frequently
        # paste the URL into the title and use the body for a short domain
        # hint (e.g. "怀旧经济"), so we must check both places and dedupe.
        body_text = issue.get("body") or ""
        title_text = issue.get("title") or ""
        urls_seen: set[str] = set()
        urls: list[str] = []
        for url in extract_urls(body_text) + extract_urls(title_text):
            if url not in urls_seen:
                urls_seen.add(url)
                urls.append(url)
        if not urls:
            skipped += 1
            continue
        notes = body_text.strip()
        for url in urls:
            # Prefer a title that isn't itself just the URL, otherwise fall
            # back to the URL so upsert_article still has something to show.
            display_title = title_text.strip()
            if not display_title or display_title == url:
                display_title = url
            article_id, was_created = upsert_article(
                title=display_title,
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


class HTMLLinkExtractor(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.links: list[dict[str, str]] = []
        self._current_href: str | None = None
        self._current_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attr_map = {key.lower(): value for key, value in attrs}
        href = attr_map.get("href")
        if not href:
            return
        self._current_href = href
        self._current_parts = []

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._current_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self._current_href is None:
            return
        anchor_text = re.sub(r"\s+", " ", "".join(self._current_parts)).strip()
        resolved = urllib.parse.urljoin(self.base_url, self._current_href)
        self.links.append(
            {
                "url": resolved,
                "anchor_text": anchor_text,
            }
        )
        self._current_href = None
        self._current_parts = []


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


def extract_title_from_html(payload: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", payload, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    title = html.unescape(match.group(1))
    return re.sub(r"\s+", " ", title).strip()


def normalize_source_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    normalized = parsed._replace(fragment="")
    return urllib.parse.urlunparse(normalized)


def guess_source_type(url: str, anchor_text: str = "", context_snippet: str = "") -> str:
    parsed = urllib.parse.urlparse(url)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    hint = f"{anchor_text} {context_snippet}".lower()
    if any(token in host for token in ("arxiv.org", "doi.org", "openreview.net", "aclanthology.org")):
        return "paper"
    if "github.com" in host:
        return "github"
    if (
        host.startswith("docs.")
        or "readthedocs" in host
        or "developer." in host
        or "/docs" in path
        or "documentation" in path
    ):
        return "docs"
    if any(token in hint for token in ("paper", "preprint", "arxiv", "doi")):
        return "paper"
    if any(token in hint for token in ("github", "repo", "repository")):
        return "github"
    if any(token in hint for token in ("docs", "documentation", "guide", "api")):
        return "docs"
    if any(token in hint for token in ("product", "launch", "release", "official")):
        return "product"
    return "unknown"


def should_keep_source_link(url: str, article_url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    if normalize_source_url(url) == normalize_source_url(article_url):
        return False

    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if any(host.endswith(blocked) for blocked in ("twitter.com", "x.com", "facebook.com", "linkedin.com")):
        return False
    if any(token in path for token in ("/share", "/login", "/signup", "/subscribe", "/privacy", "/terms")):
        return False
    return True


def latest_raw_fetch_html(article_id: str) -> Path | None:
    raw_dir = article_dir(article_id) / "raw"
    candidates = sorted(raw_dir.glob("fetch_*.html"))
    return candidates[-1] if candidates else None


def extract_article_source_links(article_id: str) -> dict[str, Any]:
    record = read_json(article_record_path(article_id), default=None)
    if record is None:
        raise ValueError(f"Unknown article_id: {article_id}")

    article_url = record.get("url") or ""
    links: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    raw_html = latest_raw_fetch_html(article_id)
    if raw_html is not None:
        payload = raw_html.read_text(encoding="utf-8", errors="replace")
        parser = HTMLLinkExtractor(base_url=article_url)
        parser.feed(payload)
        for entry in parser.links:
            normalized_url = normalize_source_url(entry["url"])
            if normalized_url in seen_urls:
                continue
            if not should_keep_source_link(normalized_url, article_url):
                continue
            seen_urls.add(normalized_url)
            anchor_text = entry.get("anchor_text", "")
            links.append(
                {
                    "url": normalized_url,
                    "anchor_text": anchor_text,
                    "context_snippet": anchor_text,
                    "link_type_guess": guess_source_type(normalized_url, anchor_text, anchor_text),
                    "selected_for_fetch": False,
                }
            )

    canonical_file = article_dir(article_id) / "canonical_text.txt"
    if canonical_file.exists():
        canonical_text = canonical_file.read_text(encoding="utf-8")
        for extracted_url in extract_urls(canonical_text):
            normalized_url = normalize_source_url(extracted_url)
            if normalized_url in seen_urls:
                continue
            if not should_keep_source_link(normalized_url, article_url):
                continue
            seen_urls.add(normalized_url)
            links.append(
                {
                    "url": normalized_url,
                    "anchor_text": "",
                    "context_snippet": "",
                    "link_type_guess": guess_source_type(normalized_url),
                    "selected_for_fetch": False,
                }
            )

    return {
        "article_id": article_id,
        "extracted_at": utc_now(),
        "source_html_path": relpath_from_root(raw_html) if raw_html else None,
        "links": links,
    }


def source_type_priority(source_type: str) -> int:
    priorities = {
        "paper": 0,
        "github": 1,
        "docs": 2,
        "product": 3,
        "news": 4,
        "unknown": 5,
    }
    return priorities.get(source_type, 9)


def _extract_input_value(payload: str, input_id: str) -> str | None:
    pattern = (
        r"<input\b[^>]*\bid=[\"']"
        + re.escape(input_id)
        + r"[\"'][^>]*\bvalue=[\"'](.*?)[\"'][^>]*>"
    )
    match = re.search(pattern, payload, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return html.unescape(match.group(1))


def extract_text_from_embedded_state(payload: str) -> str:
    """Best-effort extraction for SPA/H5 pages whose meaningful text lives in embedded JSON."""
    snippets: list[str] = []

    init_data_raw = _extract_input_value(payload, "initData")
    if init_data_raw:
        try:
            init_data = json.loads(init_data_raw)
        except json.JSONDecodeError:
            init_data = None
        if isinstance(init_data, dict):
            goods_data_raw = init_data.get("goods_data")
            if isinstance(goods_data_raw, str):
                try:
                    goods_data = json.loads(goods_data_raw)
                except json.JSONDecodeError:
                    goods_data = None
                if isinstance(goods_data, dict):
                    for key in ("goods_name", "goods_brief_text"):
                        value = goods_data.get(key)
                        if isinstance(value, str) and value.strip():
                            snippets.append(value.strip())
                    detail_html = goods_data.get("goods_detail_text")
                    if isinstance(detail_html, str) and detail_html.strip():
                        detail_text = extract_text_from_html(detail_html)
                        if detail_text:
                            snippets.append(detail_text)

    combined = "\n\n".join(part for part in snippets if part)
    return clean_extracted_text(combined)


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


def fetch_url(
    url: str,
    destination_dir: Path,
    *,
    acquired_threshold: int = 1200,
    partial_threshold: int = 250,
) -> FetchResult:
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path: Path | None = None
    text_path: Path | None = None

    try:
        with opener.open(request, timeout=25) as response:
            body = response.read()
            final_url = response.geturl()
            content_type = response.headers.get_content_type()
            decoded = decode_bytes(body, response.headers)
            extension = ".html" if content_type == "text/html" else ".txt"
            raw_path = destination_dir / f"fetch_{timestamp}{extension}"
            raw_path.write_text(decoded, encoding="utf-8")

            extracted = extract_text_from_html(decoded) if content_type == "text/html" else decoded
            if content_type == "text/html" and len(extracted.strip()) < 250:
                embedded_text = extract_text_from_embedded_state(decoded)
                if len(embedded_text) > len(extracted):
                    extracted = embedded_text
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

            if len(extracted) >= acquired_threshold:
                status = "acquired"
                note = "Full text looks sufficient for claim extraction."
            elif len(extracted) >= partial_threshold:
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


def build_source_context(
    article_id: str,
    limit: int = 5,
    excerpt_chars: int = 3000,
) -> dict[str, Any]:
    bootstrap_state_files()
    record = read_json(article_record_path(article_id), default=None)
    if record is None:
        raise ValueError(f"Unknown article_id: {article_id}")

    links_doc = extract_article_source_links(article_id)
    links = links_doc.get("links", [])
    prioritized_links = sorted(
        links,
        key=lambda item: (
            source_type_priority(str(item.get("link_type_guess") or "unknown")),
            str(item.get("url") or ""),
        ),
    )
    selected = prioritized_links[: max(limit, 0)]
    selected_urls = {entry.get("url") for entry in selected}
    for entry in links:
        entry["selected_for_fetch"] = entry.get("url") in selected_urls

    write_json(source_links_path(article_id), links_doc)

    sources_dir = article_sources_dir(article_id)
    raw_dir = sources_dir / "raw"
    text_dir = sources_dir / "text"
    raw_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    sources: list[dict[str, Any]] = []
    status_summary = {"acquired": 0, "partial": 0, "blocked": 0, "failed": 0}

    for entry in selected:
        url = str(entry.get("url") or "").strip()
        if not url:
            continue
        fetched = fetch_url(
            url,
            raw_dir,
            acquired_threshold=300,
            partial_threshold=80,
        )
        title = entry.get("anchor_text") or entry.get("context_snippet") or url
        text_path_value = fetched.text_path
        excerpt = ""
        if text_path_value:
            fetched_text_path = Path(text_path_value)
            if fetched_text_path.exists():
                target_path = text_dir / f"{fetched_text_path.stem}_{len(sources) + 1}.txt"
                shutil.copyfile(fetched_text_path, target_path)
                text_path_value = str(target_path)
                source_text = target_path.read_text(encoding="utf-8")
                excerpt = source_text[:excerpt_chars].strip()

        raw_path_value = fetched.raw_path
        if raw_path_value:
            raw_path = Path(raw_path_value)
            if raw_path.exists() and raw_path.suffix.lower() == ".html":
                html_payload = raw_path.read_text(encoding="utf-8", errors="replace")
                html_title = extract_title_from_html(html_payload)
                if html_title:
                    title = html_title

        status_key = fetched.status if fetched.status in status_summary else "failed"
        status_summary[status_key] += 1
        sources.append(
            {
                "url": url,
                "title": title,
                "source_type": entry.get("link_type_guess") or "unknown",
                "status": fetched.status,
                "raw_path": raw_path_value,
                "text_path": text_path_value,
                "text_length": fetched.text_length,
                "text_excerpt": excerpt,
                "content_type": fetched.content_type,
                "note": fetched.note,
            }
        )

    context_doc = {
        "article_id": article_id,
        "built_at": utc_now(),
        "selected_limit": limit,
        "source_html_path": links_doc.get("source_html_path"),
        "sources": sources,
        "status_summary": status_summary,
    }
    write_json(source_context_path(article_id), context_doc)
    append_jsonl(
        CHANGE_LOG_PATH,
        {
            "timestamp": utc_now(),
            "event": "build_source_context",
            "article_id": article_id,
            "link_count": len(links),
            "selected_count": len(selected),
            "source_count": len(sources),
            "status_summary": status_summary,
        },
    )
    return {
        "ok": True,
        "article_id": article_id,
        "link_count": len(links),
        "selected_count": len(selected),
        "source_count": len(sources),
        "source_links_path": relpath_from_root(source_links_path(article_id)),
        "source_context_path": relpath_from_root(source_context_path(article_id)),
        "status_summary": status_summary,
    }


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
        verification_draft = read_json(
            draft_verification_path(article_id),
            default={"items": [], "verification_status": "not_started"},
        )
        approval = read_json(approval_path(article_id), default={})

        claim_status = claims.get("claim_extraction_status", "not_started")
        verification_status = verification.get("verification_status", "not_started")
        draft_status = verification_draft.get("verification_status", "not_started")
        approval_decision = approval.get("decision")

        if verification_status != "completed" and draft_status == "drafted":
            verification_status = "drafted"

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
        elif verification_status == "drafted":
            if approval_decision == "auto_approved":
                bayesian_status = "verification_staged"
                next_action = "apply_staged_verification"
            else:
                bayesian_status = "held_for_review"
                next_action = "review_staged_verification"
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
        max_verified_at: str | None = None

        for record in article_records:
            verification = read_json(verification_path(record["article_id"]), default={"items": []})
            article_verified_at = verification.get("verified_at")
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
                if article_verified_at and (
                    max_verified_at is None or article_verified_at > max_verified_at
                ):
                    max_verified_at = article_verified_at

        probability = 1 / (1 + math.exp(-posterior_log_odds))
        hypothesis["posterior_log_odds"] = posterior_log_odds
        hypothesis["posterior_probability"] = probability
        hypothesis["posterior_band"] = posterior_band(probability)
        hypothesis["last_recomputed_at"] = utc_now()
        hypothesis["supporting_items"] = supporting_items
        hypothesis["newest_evidence_at"] = max_verified_at

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
    synthesis["tool_index"] = build_tool_index(
        article_records,
        existing_items=synthesis.get("tool_index", []),
    )
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

# Default domain label used when a hypothesis, article, or verification item
# was written before the multi-domain schema landed. Phase 0 adds the
# `domain` field additively: read paths fall back to this default, so
# existing files continue to work without migration.
DEFAULT_DOMAIN = "ai"


def item_domain(item: dict[str, Any]) -> str:
    """Return the domain of a hypothesis / article / verification item.

    Backward-compatible: items written before Phase 0 have no `domain`
    field and are treated as belonging to the default domain.
    """
    value = item.get("domain") if isinstance(item, dict) else None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return DEFAULT_DOMAIN


def item_meta_tags(item: dict[str, Any]) -> list[str]:
    """Return the meta_tags list for a hypothesis, with backward-compat."""
    value = item.get("meta_tags") if isinstance(item, dict) else None
    if not isinstance(value, list):
        return []
    return [tag for tag in value if isinstance(tag, str) and tag.strip()]


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


def article_sources_dir(article_id: str) -> Path:
    return article_dir(article_id) / "sources"


def source_links_path(article_id: str) -> Path:
    return article_dir(article_id) / "source_links.json"


def source_context_path(article_id: str) -> Path:
    return article_dir(article_id) / "source_context.json"


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


def load_source_context(article_id: str) -> dict[str, Any]:
    return read_json(
        source_context_path(article_id),
        default={"article_id": article_id, "built_at": None, "sources": [], "status_summary": {}},
    )


def load_hypothesis_index() -> dict[str, dict[str, Any]]:
    hypotheses = read_json(HYPOTHESES_PATH, default={"hypotheses": []})
    return {item["id"]: item for item in hypotheses.get("hypotheses", []) if item.get("id")}


def meta_scan(domain_filter: str | None = None) -> dict[str, Any]:
    """Read-only cross-hypothesis tag scan.

    Groups active hypotheses by their `meta_tags` and reports:
      - tag → list of (hypothesis_id, domain, band)
      - clusters (tags shared by 2+ hypotheses)
      - cross-domain clusters (tags that span multiple domains)
      - singletons (tags used by only one hypothesis)

    When `domain_filter` is provided, only hypotheses in that domain are
    included. `cross_domain_cluster_count` will therefore always be 0
    under a domain filter — it's meaningless in a single-domain slice.

    Phase 3-lite: this command NEVER writes to hypotheses.json,
    synthesis_state.json, or change_log.jsonl. It is a pure describe-only
    lens on the current state, safe to run at any time.
    """
    hypotheses_doc = read_json(HYPOTHESES_PATH, default={"hypotheses": []})
    hypotheses = hypotheses_doc.get("hypotheses", [])

    tag_to_members: dict[str, list[dict[str, Any]]] = {}
    untagged: list[dict[str, Any]] = []
    all_domains: set[str] = set()

    for hypothesis in hypotheses:
        if hypothesis.get("status") not in (None, "active"):
            continue
        hypothesis_id = hypothesis.get("id")
        if not hypothesis_id:
            continue

        domain = item_domain(hypothesis)
        if domain_filter is not None and domain != domain_filter:
            continue
        all_domains.add(domain)
        tags = item_meta_tags(hypothesis)

        member = {
            "hypothesis_id": hypothesis_id,
            "domain": domain,
            "band": hypothesis.get("posterior_band") or "unknown",
            "statement": hypothesis.get("statement", ""),
        }

        if not tags:
            untagged.append(member)
            continue

        for tag in tags:
            tag_to_members.setdefault(tag, []).append(member)

    clusters: list[dict[str, Any]] = []
    singletons: list[dict[str, Any]] = []
    cross_domain_clusters: list[dict[str, Any]] = []

    for tag in sorted(tag_to_members):
        members = tag_to_members[tag]
        domains_in_tag = sorted({m["domain"] for m in members})
        entry = {
            "tag": tag,
            "member_count": len(members),
            "domains": domains_in_tag,
            "cross_domain": len(domains_in_tag) > 1,
            "members": members,
        }
        if len(members) >= 2:
            clusters.append(entry)
            if entry["cross_domain"]:
                cross_domain_clusters.append(entry)
        else:
            singletons.append(entry)

    clusters.sort(
        key=lambda e: (not e["cross_domain"], -e["member_count"], e["tag"])
    )

    return {
        "scanned_at": utc_now(),
        "domain_filter": domain_filter,
        "hypothesis_count": len(hypotheses),
        "domain_count": len(all_domains),
        "domains": sorted(all_domains),
        "tag_count": len(tag_to_members),
        "cluster_count": len(clusters),
        "cross_domain_cluster_count": len(cross_domain_clusters),
        "clusters": clusters,
        "singletons": singletons,
        "untagged_hypotheses": untagged,
    }


def format_meta_scan(report: dict[str, Any]) -> list[str]:
    """Pretty-print a meta_scan report for console output.

    Deliberately dense but readable. No probability numbers; only bands.
    Uses ASCII-only output characters so the report prints cleanly on
    Windows consoles that default to cp936/cp1252.
    """
    lines: list[str] = []
    filter_tag = (
        f" [filter: domain={report['domain_filter']}]"
        if report.get("domain_filter")
        else ""
    )
    lines.append(
        f"meta-scan{filter_tag}: {report['hypothesis_count']} hypotheses | "
        f"{report['domain_count']} domain(s) [{', '.join(report['domains'])}] | "
        f"{report['tag_count']} tag(s) | "
        f"{report['cluster_count']} cluster(s) "
        f"({report['cross_domain_cluster_count']} cross-domain)"
    )

    if report["clusters"]:
        lines.append("")
        lines.append("-- clusters (tags shared by 2+ hypotheses) --")
        for entry in report["clusters"]:
            marker = " [CROSS-DOMAIN]" if entry["cross_domain"] else ""
            lines.append(
                f"  [{entry['tag']}] x {entry['member_count']}"
                f" (domains: {', '.join(entry['domains'])}){marker}"
            )
            for member in entry["members"]:
                lines.append(
                    f"      - {member['hypothesis_id']}"
                    f"  [{member['domain']} / {member['band']}]"
                )
    else:
        lines.append("")
        lines.append("-- no clusters yet (every tag is a singleton) --")

    if report["singletons"]:
        lines.append("")
        lines.append("-- singleton tags --")
        for entry in report["singletons"]:
            member = entry["members"][0]
            lines.append(
                f"  [{entry['tag']}] -> {member['hypothesis_id']}"
                f"  [{member['domain']} / {member['band']}]"
            )

    if report["untagged_hypotheses"]:
        lines.append("")
        lines.append("-- hypotheses with no meta_tags (candidates for tagging) --")
        for member in report["untagged_hypotheses"]:
            lines.append(
                f"  - {member['hypothesis_id']}"
                f"  [{member['domain']} / {member['band']}]"
            )

    return lines


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

    # Optional top-level domain (Phase 0, additive). Absent = default domain.
    payload_domain = payload.get("domain")
    if payload_domain is not None:
        if not isinstance(payload_domain, str) or not payload_domain.strip():
            errors.append("domain, if provided, must be a non-empty string")

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

        # Optional per-item domain (Phase 0, additive). Missing = inherit the
        # payload-level domain, which itself falls back to DEFAULT_DOMAIN.
        item_domain_value = item.get("domain")
        if item_domain_value is not None:
            if not isinstance(item_domain_value, str) or not item_domain_value.strip():
                errors.append(
                    f"{prefix}.domain, if provided, must be a non-empty string"
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


def verification_cross_domain_warnings(
    payload: Any,
    article_id: str,
) -> list[str]:
    """Return non-fatal warnings about cross-domain hypothesis attachments.

    The goal is to surface — without blocking — cases where a verification
    item attaches a claim to a hypothesis that lives in a different domain.
    This is the earliest tripwire against 'metaphor collapse': if an economy
    article tries to feed evidence into an AI hypothesis, we want a visible
    flag in the staging output even though the save itself still succeeds.

    Phase 2: warn only, never fail. Phase 4 (meta-hypothesis layer) may
    tighten this into a hard error that requires an explicit
    `allow_cross_domain_attach: true` opt-in per item.
    """
    warnings: list[str] = []
    if not isinstance(payload, dict):
        return warnings

    payload_domain = payload.get("domain")
    if not isinstance(payload_domain, str) or not payload_domain.strip():
        payload_domain = DEFAULT_DOMAIN

    hypothesis_index = load_hypothesis_index()
    items = payload.get("items", [])
    if not isinstance(items, list):
        return warnings

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        hypothesis_id = item.get("hypothesis_id")
        if not isinstance(hypothesis_id, str):
            continue
        hypothesis = hypothesis_index.get(hypothesis_id)
        if hypothesis is None:
            continue

        item_domain_value = item.get("domain")
        if isinstance(item_domain_value, str) and item_domain_value.strip():
            effective_item_domain = item_domain_value.strip()
        else:
            effective_item_domain = payload_domain

        hypothesis_domain = item_domain(hypothesis)
        if effective_item_domain != hypothesis_domain:
            warnings.append(
                f"items[{index}]: cross-domain attach — claim domain "
                f"{effective_item_domain!r} but hypothesis {hypothesis_id!r} "
                f"lives in domain {hypothesis_domain!r}. This is allowed for "
                f"Phase 2 but flagged as a potential metaphor-collapse risk. "
                f"Review the assessment before apply."
            )

    return warnings


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

    warnings = verification_cross_domain_warnings(payload, article_id)

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
            "cross_domain_warnings": warnings,
        },
    )

    next_task = get_next_task()
    response: dict[str, Any] = {
        "ok": True,
        "article_id": article_id,
        "item_count": len(normalized["items"]),
        "state_diff": format_state_diff(before, after),
        "next_task": next_task,
    }
    if warnings:
        response["warnings"] = warnings
    return response


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
REVIEW_ACTIONS = {"approve", "approve-safe", "reject"}


def compute_phase_c_escalations(
    draft_payload: dict[str, Any],
    approval_payload: dict[str, Any],
    hypothesis_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return Phase C escalation triggers for an LLM-drafted (draft, approval).

    Phase C is the automated verification drafter. Before a drafted
    approval can stay at ``auto_approved``, two deterministic guards run:

      A. cross_domain — any item whose effective domain differs from the
         attached hypothesis's domain. The LLM was asked to leave such
         attachments at ``hypothesis_id=null``, but Python enforces it.

      B. band_crossing — simulate the additive log-odds update from every
         verified / partially_verified item in the draft, grouped by
         hypothesis_id. If the simulated posterior would land in a
         different ``POSTERIOR_BANDS`` bucket than the current one, the
         draft is flagged. This mirrors what ``recompute_posteriors``
         would compute at apply time without any filesystem IO.

    The function is pure: it reads nothing from disk, mutates none of its
    inputs, and returns an empty list when the draft is safe to auto-apply.
    Callers downgrade ``approval.decision`` to ``needs_human`` and append an
    audit trail to ``overall_rationale`` when the list is non-empty.

    Trigger shape:
        {
          "trigger": "cross_domain" | "band_crossing",
          "hypothesis_id": <str>,
          "detail": <human-readable string>,
          # cross_domain-only:
          "item_index": <int>,
          "item_domain": <str>,
          "hypothesis_domain": <str>,
          # band_crossing-only:
          "before_band": <str>,
          "after_band": <str>,
          "delta_log_odds": <float>,
          "claim_ids": [<str>, ...],
        }
    """

    triggers: list[dict[str, Any]] = []
    if not isinstance(draft_payload, dict):
        return triggers

    items = draft_payload.get("items")
    if not isinstance(items, list):
        return triggers

    payload_domain_raw = draft_payload.get("domain")
    if isinstance(payload_domain_raw, str) and payload_domain_raw.strip():
        payload_domain = payload_domain_raw.strip()
    else:
        payload_domain = DEFAULT_DOMAIN

    # --- Trigger A: cross-domain attach -----------------------------------
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        hypothesis_id = item.get("hypothesis_id")
        if not isinstance(hypothesis_id, str):
            continue
        hypothesis = hypothesis_index.get(hypothesis_id)
        if hypothesis is None:
            continue

        item_domain_raw = item.get("domain")
        if isinstance(item_domain_raw, str) and item_domain_raw.strip():
            effective_item_domain = item_domain_raw.strip()
        else:
            effective_item_domain = payload_domain

        hypothesis_domain = item_domain(hypothesis)
        if effective_item_domain != hypothesis_domain:
            triggers.append(
                {
                    "trigger": "cross_domain",
                    "hypothesis_id": hypothesis_id,
                    "item_index": index,
                    "item_domain": effective_item_domain,
                    "hypothesis_domain": hypothesis_domain,
                    "detail": (
                        f"items[{index}] domain={effective_item_domain!r} "
                        f"attached to hypothesis {hypothesis_id!r} "
                        f"(domain={hypothesis_domain!r})"
                    ),
                }
            )

    # --- Trigger B: band-crossing simulation ------------------------------
    # Group contributions per hypothesis_id. Only verified / partially
    # verified items contribute, matching recompute_posteriors at line 906.
    per_hypothesis_deltas: dict[str, float] = {}
    per_hypothesis_claims: dict[str, list[str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("status") not in STRONG_VERIFICATION_STATUSES:
            continue
        hypothesis_id = item.get("hypothesis_id")
        if not isinstance(hypothesis_id, str):
            continue
        if hypothesis_id not in hypothesis_index:
            continue
        if not is_ordinal_item(item):
            # Phase C payloads must be ordinal; anything else already
            # failed validate_verification_payload. Skip defensively.
            continue
        try:
            contribution = ordinal_contribution(
                source_trust=item["source_trust"],
                evidence_direction=item["evidence_direction"],
                evidence_strength=item["evidence_strength"],
            )
        except KeyError:
            continue
        per_hypothesis_deltas[hypothesis_id] = (
            per_hypothesis_deltas.get(hypothesis_id, 0.0) + contribution
        )
        claim_id = item.get("claim_id")
        if isinstance(claim_id, str):
            per_hypothesis_claims.setdefault(hypothesis_id, []).append(claim_id)

    for hypothesis_id, delta in per_hypothesis_deltas.items():
        hypothesis = hypothesis_index.get(hypothesis_id)
        if hypothesis is None:
            continue
        try:
            current_log_odds = float(hypothesis.get("posterior_log_odds", 0.0))
        except (TypeError, ValueError):
            current_log_odds = 0.0
        try:
            current_probability = float(hypothesis.get("posterior_probability", 0.5))
        except (TypeError, ValueError):
            current_probability = 1 / (1 + math.exp(-current_log_odds))

        new_log_odds = current_log_odds + delta
        new_probability = 1 / (1 + math.exp(-new_log_odds))

        before_band = posterior_band(current_probability)
        after_band = posterior_band(new_probability)
        if before_band != after_band:
            triggers.append(
                {
                    "trigger": "band_crossing",
                    "hypothesis_id": hypothesis_id,
                    "before_band": before_band,
                    "after_band": after_band,
                    "delta_log_odds": delta,
                    "claim_ids": per_hypothesis_claims.get(hypothesis_id, []),
                    "detail": (
                        f"hypothesis {hypothesis_id!r} would move "
                        f"{format_band_transition(current_probability, new_probability)} "
                        f"(Δlog-odds={delta:+.3f}) via items "
                        f"{per_hypothesis_claims.get(hypothesis_id, [])}"
                    ),
                }
            )

    # TODO: Phase D may add trigger D (per-run absolute Δlog-odds ceiling)
    # as a backstop for the rare case where a big contribution stays inside
    # one band but still represents more movement than one article should be
    # allowed to cause without a human. Leaving unimplemented until a real
    # run exposes the gap.

    return triggers


def format_phase_c_escalation_suffix(
    triggers: list[dict[str, Any]],
) -> str:
    """Render escalation triggers as a plain-text suffix for overall_rationale.

    The suffix is appended verbatim to ``approval.overall_rationale`` so the
    audit trail survives ``stage_verification``'s field-filtering
    normalization. Format is stable and grep-friendly.
    """
    if not triggers:
        return ""
    lines = [
        "",
        "",
        "---",
        "[PHASE_C_ESCALATION] Automatically downgraded from auto_approved to "
        "needs_human by the Phase C guard. Triggers:",
    ]
    for entry in triggers:
        kind = entry.get("trigger", "unknown")
        detail = entry.get("detail", "")
        lines.append(f"- {kind}: {detail}")
    return "\n".join(lines)


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

    cross_domain_warnings = verification_cross_domain_warnings(draft_payload, article_id)

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

    response: dict[str, Any] = {
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
    if cross_domain_warnings:
        response["warnings"] = cross_domain_warnings
    return response


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
    draft_domain = draft.get("domain")
    revalidation_payload = {
        "article_id": article_id,
        "verification_status": "completed",
        "items": draft_items,
    }
    if isinstance(draft_domain, str) and draft_domain.strip():
        revalidation_payload["domain"] = draft_domain.strip()
    draft_errors = validate_verification_payload(
        revalidation_payload,
        article_id,
        claims_doc,
    )
    if draft_errors:
        return {
            "ok": False,
            "article_id": article_id,
            "errors": {"verification_draft": draft_errors},
        }

    # Run cross-domain warnings at apply time too, not just stage time. A
    # user who stages on Monday and applies on Friday should still see the
    # metaphor-collapse tripwire fire before the posterior actually moves.
    cross_domain_warnings = verification_cross_domain_warnings(
        revalidation_payload, article_id
    )

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
    if isinstance(draft_domain, str) and draft_domain.strip():
        normalized["domain"] = draft_domain.strip()
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
            "cross_domain_warnings": cross_domain_warnings,
        },
    )

    next_task = get_next_task()
    response: dict[str, Any] = {
        "ok": True,
        "article_id": article_id,
        "item_count": len(draft_items),
        "approver": approval.get("approver"),
        "decision": decision,
        "state_diff": format_state_diff(before, after),
        "next_task": next_task,
    }
    if cross_domain_warnings:
        response["warnings"] = cross_domain_warnings
    return response


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


def github_issue_url_for_record(record: dict[str, Any]) -> str | None:
    for source in reversed(record.get("ingest_sources", [])):
        if source.get("source_name") != "github_issue":
            continue
        source_ref = source.get("source_ref")
        if isinstance(source_ref, str) and source_ref.strip():
            return source_ref.strip()
    return None


def find_article_id_by_issue_number(issue_number: int | str) -> str | None:
    issue_suffix = f"/issues/{issue_number}"
    for record in load_all_article_records():
        issue_url = github_issue_url_for_record(record)
        if issue_url and issue_url.endswith(issue_suffix):
            article_id = record.get("article_id")
            if isinstance(article_id, str) and article_id:
                return article_id
    return None


def article_summary_lines(record: dict[str, Any], limit: int = 4) -> list[str]:
    summary = record.get("article_summary", {})
    grouped_items = [
        ("事件", summary.get("events", [])),
        ("技术", summary.get("techniques", [])),
        ("产品", summary.get("tools", [])),
    ]
    lines: list[str] = []
    offsets = [0 for _ in grouped_items]
    while len(lines) < limit:
        appended = False
        for index, (label, items) in enumerate(grouped_items):
            if offsets[index] >= len(items):
                continue
            lines.append(f"{label}: {items[offsets[index]]}")
            offsets[index] += 1
            appended = True
            if len(lines) >= limit:
                break
        if not appended:
            break
    return lines


def held_review_guidance(article_id: str) -> dict[str, Any]:
    bootstrap_state_files()
    record = read_json(article_record_path(article_id), default=None)
    if record is None:
        raise ValueError(f"Unknown article_id: {article_id}")

    draft = read_json(
        draft_verification_path(article_id),
        default={"items": [], "verification_status": "not_started"},
    )
    approval = read_json(approval_path(article_id), default={})
    hypotheses = read_json(HYPOTHESES_PATH, default={"hypotheses": []})
    hypothesis_index = {
        item["id"]: item
        for item in hypotheses.get("hypotheses", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }

    triggers = compute_phase_c_escalations(draft, approval, hypothesis_index)
    per_claim = approval.get("per_claim_decisions", [])
    if not isinstance(per_claim, list):
        per_claim = []
    decisions = [
        entry.get("decision")
        for entry in per_claim
        if isinstance(entry, dict) and isinstance(entry.get("decision"), str)
    ]
    overall_rationale = approval.get("overall_rationale")
    overall_text = overall_rationale.lower() if isinstance(overall_rationale, str) else ""

    reasons: list[dict[str, str]] = []
    seen_codes: set[str] = set()

    def add_reason(code: str, detail: str) -> None:
        if code in seen_codes:
            return
        reasons.append({"code": code, "detail": detail})
        seen_codes.add(code)

    if any(entry.get("trigger") == "cross_domain" for entry in triggers):
        add_reason("cross_domain", "存在跨领域的 hypothesis 挂接，不能直接放行。")
    if any(entry.get("trigger") == "band_crossing" for entry in triggers):
        add_reason("band_crossing", "这次 apply 会把趋势推过一个 posterior 档位，值得人工确认。")
    if any(decision == "reject" for decision in decisions):
        add_reason("draft_conflict", "草稿里有至少一条 claim 被明确标记为 reject。")
    elif any(decision == "defer" for decision in decisions):
        add_reason("draft_conflict", "草稿里有至少一条关键 claim 被标记为 defer。")
    if any(decision == "accept_as_fact" for decision in decisions):
        add_reason("mapping_risk", "有些内容更适合作为事实保留，而不是直接推动 hypothesis。")
    if any(
        token in overall_text
        for token in (
            "manufacturer",
            "secondary article",
            "secondary-source",
            "secondary source",
            "primary-source",
            "primary source",
            "specification",
            "specifications",
            "superlative",
            "not independently substantiated",
        )
    ):
        add_reason("source_risk", "当前依据更像厂商口径或二手转述，独立主源支撑不够强。")
    if "cold-start" in overall_text or "cold start" in overall_text:
        add_reason("mapping_risk", "这次挂接的 hypothesis 仍处于冷启动阶段，直接 apply 风险偏高。")

    accept_count = sum(decision == "accept" for decision in decisions)
    accept_fact_count = sum(decision == "accept_as_fact" for decision in decisions)
    reject_count = sum(decision == "reject" for decision in decisions)
    defer_count = sum(decision == "defer" for decision in decisions)

    recommended_action = "approve"
    why = "当前 hold 更像模型保守处理；没有明显的结构性风险，可以人工放行。"
    if reject_count > 0 or (not triggers and accept_count == 0 and accept_fact_count == 0):
        recommended_action = "reject"
        why = "关键 claim 本身不够稳，直接 apply 的收益低于误判风险。"
    elif triggers or accept_fact_count > 0 or defer_count > 0 or "source_risk" in seen_codes or "mapping_risk" in seen_codes:
        recommended_action = "approve-safe"
        why = "事实可以保留，但当前不适合把 hypothesis 挂接一并放进 posterior。"

    return {
        "article_id": article_id,
        "issue_url": github_issue_url_for_record(record),
        "summary_lines": article_summary_lines(record),
        "reasons": reasons,
        "recommended_action": recommended_action,
        "why": why,
        "overall_rationale": approval.get("overall_rationale", ""),
        "decision": approval.get("decision"),
        "trigger_count": len(triggers),
    }


def build_held_issue_comment(article_id: str) -> str:
    guidance = held_review_guidance(article_id)
    lines = [
        f"Phase C held `{article_id}` for human review.",
        "",
        "Article summary:",
    ]
    summary_lines = guidance.get("summary_lines") or []
    if summary_lines:
        lines.extend(f"- {line}" for line in summary_lines)
    else:
        lines.append("- 暂无可展示摘要")
    lines.extend(["", "Held reason:"])
    reasons = guidance.get("reasons") or []
    if reasons:
        for entry in reasons:
            lines.append(f"- `{entry['code']}`: {entry['detail']}")
    else:
        lines.append("- `manual_review`: 模型没有自动放行，建议人工确认后再决定。")
    lines.extend(
        [
            "",
            "Recommended action:",
            f"- `/{guidance['recommended_action']}`",
            "",
            "Why:",
            f"- {guidance['why']}",
            "",
            "Reply with one command:",
            "- `/approve` — apply the staged verification",
            "- `/approve-safe` — apply only factual verification and detach all hypothesis links",
            "- `/reject` — keep this article held and do not apply",
        ]
    )
    return "\n".join(lines)


def review_held(
    article_id: str,
    action: str,
    approver: str = "human:github",
    reason: str | None = None,
) -> dict[str, Any]:
    bootstrap_state_files()
    if action not in REVIEW_ACTIONS:
        raise ValueError(f"action must be one of {sorted(REVIEW_ACTIONS)} (got {action!r})")

    draft = read_json(
        draft_verification_path(article_id),
        default={"items": [], "verification_status": "not_started"},
    )
    approval = read_json(approval_path(article_id), default={})
    if draft.get("verification_status") != "drafted" or not approval:
        return {
            "ok": False,
            "article_id": article_id,
            "action": action,
            "errors": ["no held/staged verification draft found for this article"],
        }

    verification = read_json(
        verification_path(article_id),
        default={"items": [], "verification_status": "not_started"},
    )
    if verification.get("verification_status") == "completed":
        return {
            "ok": False,
            "article_id": article_id,
            "action": action,
            "errors": ["verification has already been applied for this article"],
        }

    before = snapshot_state()
    detached_claim_ids: list[str] = []
    if action == "approve":
        override_reason = reason or "Approved from GitHub issue review command."
        override_result = override_approval(
            article_id=article_id,
            decision="human_approved",
            reason=override_reason,
            approver=approver,
        )
        if not override_result.get("ok"):
            return {**override_result, "action": action}
        apply_result = apply_verification(article_id=article_id)
        return {
            **apply_result,
            "action": action,
            "applied": bool(apply_result.get("ok")),
            "detached_claim_ids": detached_claim_ids,
        }

    if action == "approve-safe":
        detached_claim_ids = [
            item["claim_id"]
            for item in draft.get("items", [])
            if isinstance(item, dict)
            and isinstance(item.get("claim_id"), str)
            and item.get("hypothesis_id")
        ]
        override_reason = reason or (
            "Applied from GitHub issue review command after detaching all hypothesis links."
        )
        override_result = override_approval(
            article_id=article_id,
            decision="human_overridden",
            reason=override_reason,
            approver=approver,
            detach_claim_ids=detached_claim_ids,
        )
        if not override_result.get("ok"):
            return {**override_result, "action": action}
        apply_result = apply_verification(article_id=article_id)
        return {
            **apply_result,
            "action": action,
            "applied": bool(apply_result.get("ok")),
            "detached_claim_ids": detached_claim_ids,
        }

    override_reason = reason or "Rejected from GitHub issue review command; keeping article held."
    override_result = override_approval(
        article_id=article_id,
        decision="needs_human",
        reason=override_reason,
        approver=approver,
    )
    if not override_result.get("ok"):
        return {**override_result, "action": action}
    refresh_record_states()
    after = snapshot_state()
    return {
        "ok": True,
        "article_id": article_id,
        "action": action,
        "applied": False,
        "decision": "needs_human",
        "detached_claim_ids": detached_claim_ids,
        "state_diff": format_state_diff(before, after),
        "next_step": "kept_held_for_review",
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
    "domain": "<optional domain label, defaults to 'ai' if omitted>",
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
            "domain": "<optional, inherits the payload-level domain if omitted>",
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


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def hypothesis_theory_text(item: dict[str, Any]) -> str:
    theory = (item.get("theory") or "").strip()
    if theory:
        return theory
    rationale = (item.get("rationale") or "").strip()
    statement = (item.get("statement") or "").strip()
    if rationale and statement:
        return f"{statement}\n\n{rationale}"
    return rationale or statement


def hypothesis_detail_href(hypothesis_id: str) -> str:
    return f"hypotheses/{hypothesis_id}.html"


REPORT_STATUS_LABELS = {
    "included": "已纳入",
    "excluded_until_verified": "待提取/待核实",
    "verification_staged": "待应用",
    "held_for_review": "待人工复核",
    "held_out_pending_better_evidence": "待更强证据",
    "excluded_after_verification": "已排除",
}

REPORT_STATUS_DESCRIPTIONS = {
    "included": "该文章的核实结果已经进入当前趋势判断。",
    "excluded_until_verified": "文章已入库，但还没有完成 claim 提取或主源核实，暂不进入趋势判断。",
    "verification_staged": "核实草稿已暂存并通过自动门禁，等待写入 verification.json 后才会影响后验。",
    "held_for_review": "核实草稿已暂存，但审批结论或 Guard 要求人工复核，当前不会自动改写后验。",
    "held_out_pending_better_evidence": "已发现部分冲突或证据不足，当前保留记录，等待更强来源。",
    "excluded_after_verification": "文章已完成核实，但当前证据不足以支持任何趋势更新，因此被排除在后验之外。",
}

EVIDENCE_STATUS_LABELS = {
    "verified": "已核实",
    "partially_verified": "部分核实",
    "conflicted": "存在冲突",
    "unverified": "未核实",
}

APPROVAL_DECISION_LABELS = {
    "auto_approved": "自动通过",
    "needs_human": "需要人工确认",
    "human_approved": "人工通过",
    "human_overridden": "人工改写后通过",
}


def report_status_label(status: str | None) -> str:
    if not status:
        return "未知状态"
    return REPORT_STATUS_LABELS.get(status, status)


def evidence_status_label(status: str | None) -> str:
    if not status:
        return "未知"
    return EVIDENCE_STATUS_LABELS.get(status, status)


def approval_decision_label(decision: str | None) -> str:
    if not decision:
        return ""
    return APPROVAL_DECISION_LABELS.get(decision, decision)


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
    verification_draft = read_json(
        draft_verification_path(article_id),
        default={"items": [], "verification_status": "not_started"},
    )
    approval = read_json(approval_path(article_id), default={})
    bayesian_status = record.get("analysis_state", {}).get("bayesian_status", "unknown")
    rendered_verification = verification
    evidence_heading = "核实结果"
    evidence_empty_text = "暂无核实条目"
    if not verification.get("items") and verification_draft.get("verification_status") == "drafted":
        rendered_verification = verification_draft
        evidence_heading = "核实草稿"
        evidence_empty_text = "暂无核实草稿"

    state_notes: list[str] = []
    status_description = REPORT_STATUS_DESCRIPTIONS.get(bayesian_status)
    if status_description:
        state_notes.append(status_description)
    if verification_draft.get("verification_status") == "drafted":
        decision_label = approval_decision_label(approval.get("decision"))
        if decision_label:
            state_notes.append(f"审批结论：{decision_label}")
        if bayesian_status == "held_for_review":
            issue_url = github_issue_url_for_record(record)
            if issue_url:
                state_notes.append(
                    "可在对应 GitHub issue 中回复 /approve、/approve-safe 或 /reject 继续处理。"
                )
        overall_rationale = approval.get("overall_rationale")
        if overall_rationale:
            state_notes.append(f"审批说明：{overall_rationale}")
        human_override = approval.get("human_override") or {}
        if human_override.get("reason"):
            state_notes.append(f"人工修订：{human_override['reason']}")

    state_note_html = ""
    if state_notes:
        state_note_html = "<div class='status-note'>" + "".join(
            f"<p>{html_escape(line)}</p>" for line in state_notes
        ) + "</div>"

    verified_rows: list[str] = []
    for item in rendered_verification.get("items", []):
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
                metrics.append("已核实事实，未挂接趋势假设")
        else:
            # Legacy float-based items: summarize without the raw numbers
            # so the report still obeys the "no false precision" rule.
            if item.get("weight") is not None or item.get("likelihood_ratio") is not None:
                metrics.append("历史证据权重（旧格式）")

        item_status = item.get("status", "unknown")

        verified_rows.append(
            "".join(
                [
                    "<div class='evidence-row'>",
                    f"<div class='evidence-top'><span class='status status-{html_escape(item_status)}'>{html_escape(evidence_status_label(item_status))}</span>",
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
            f"<details class='article-card' data-status='{html_escape(bayesian_status)}' data-article-id='{html_escape(record['article_id'])}'>",
            f"<summary><span>{html_escape(record['title'])}</span><span class='status status-{html_escape(bayesian_status)}'>{html_escape(report_status_label(bayesian_status))}</span></summary>",
            "<div class='article-body'>",
            f"<p><a href='{html_escape(record['url'])}' target='_blank' rel='noreferrer'>{html_escape(record['url'])}</a></p>",
            state_note_html,
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
            f"<h4>{html_escape(evidence_heading)}</h4>",
            "".join(verified_rows) if verified_rows else f"<p class='muted'>{html_escape(evidence_empty_text)}</p>",
            "</section>",
            "</div>",
            "</details>",
        ]
    )


RECENCY_WINDOW_HOURS = 24


def _build_evidence_lookup_maps(
    records: list[dict[str, Any]],
) -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[tuple[str, str], str],
    dict[tuple[str, str], dict[str, Any]],
]:
    """Build lookup tables for hypothesis card evidence drill-down.

    Returns
    (
        article_title_map,
        article_url_map,
        article_verified_at_map,
        claim_text_map,
        verification_map,
    ).
    """
    article_title_map: dict[str, str] = {}
    article_url_map: dict[str, str] = {}
    article_verified_at_map: dict[str, str] = {}
    claim_text_map: dict[tuple[str, str], str] = {}
    verification_map: dict[tuple[str, str], dict[str, Any]] = {}

    for record in records:
        aid = record.get("article_id")
        if not aid:
            continue
        article_title_map[aid] = record.get("title") or aid
        article_url_map[aid] = record.get("url") or ""

        claims_doc = read_json(claims_path(aid), default={"claims": []})
        for claim in claims_doc.get("claims", []):
            cid = claim.get("id")
            if cid:
                claim_text_map[(aid, cid)] = claim.get("text", "")

        verif = read_json(verification_path(aid), default={"items": []})
        verified_at = verif.get("verified_at")
        if verified_at:
            article_verified_at_map[aid] = verified_at
        for item in verif.get("items", []):
            cid = item.get("claim_id")
            if cid:
                verification_map[(aid, cid)] = item

    return (
        article_title_map,
        article_url_map,
        article_verified_at_map,
        claim_text_map,
        verification_map,
    )


def latest_applied_article_batch() -> tuple[str | None, set[str]]:
    latest_timestamp: str | None = None
    latest_article_ids: set[str] = set()
    for entry in read_jsonl(CHANGE_LOG_PATH):
        if entry.get("event") != "apply_verification":
            continue
        timestamp = entry.get("timestamp")
        article_id = entry.get("article_id")
        if not timestamp or not article_id:
            continue
        if latest_timestamp is None or timestamp > latest_timestamp:
            latest_timestamp = timestamp
            latest_article_ids = {article_id}
        elif timestamp == latest_timestamp:
            latest_article_ids.add(article_id)
    return latest_timestamp, latest_article_ids


TOOL_CATEGORY_BY_HYPOTHESIS = {
    "agent_harness_decoupling": "agent_runtime",
    "structured_external_memory": "memory_system",
    "token_efficiency_as_architecture": "token_compression",
    "speech_from_tts_to_scene_generation": "speech_generation",
    "software_engineering_shifts_to_orchestration_and_testing": "developer_tool",
    "nostalgia_economy_mainstreaming": "consumer_product",
    "embodied_models_add_tactile_grounding": "embodied_tactile",
}


def normalize_tool_name(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    return value.strip("“”\"'`.,;:，。：；（）()[]")


def infer_tool_name_from_claim(text: str) -> str | None:
    working = normalize_tool_name(text)
    if not working:
        return None

    prefixes = [
        "The article states that ",
        "The article claims that ",
        "The article says that ",
        "The article describes ",
        "文章称",
        "文章指出",
        "文章提到",
    ]
    lower_working = working.lower()
    for prefix in prefixes:
        if lower_working.startswith(prefix.lower()):
            working = working[len(prefix) :].strip()
            break

    candidates: list[str] = []
    delimiters = [
        " 是",
        " is ",
        " 将",
        " uses ",
        " 使用",
        " has ",
        " combines ",
        " has been adapted",
        " adopts ",
        " supports ",
        " 提供",
        " 支持",
    ]
    for delimiter in delimiters:
        if delimiter in working:
            candidates.append(normalize_tool_name(working.split(delimiter, 1)[0]))

    regex_patterns = [
        r"\b([A-Z][A-Za-z0-9.+/_-]*(?:\s+[A-Z0-9][A-Za-z0-9.+/_-]*){0,3})\b",
        r"\b([A-Za-z][A-Za-z0-9.+/_-]*\s\d(?:\.\d+)?)\b",
    ]
    for pattern in regex_patterns:
        for match in re.finditer(pattern, working):
            candidates.append(normalize_tool_name(match.group(1)))

    generic = {
        "",
        "the article",
        "the hand shown in the demo",
        "this article",
        "article",
    }

    def score(candidate: str) -> int:
        lower = candidate.lower()
        if lower in generic:
            return -100
        if candidate.startswith("the "):
            return -50
        value = 0
        if re.search(r"[A-Z]", candidate):
            value += 2
        if any(ch.isdigit() for ch in candidate):
            value += 2
        if len(candidate.split()) <= 4:
            value += 1
        if len(candidate) <= 32:
            value += 1
        return value

    ranked = sorted(
        ((score(candidate), candidate) for candidate in candidates if candidate),
        key=lambda item: (item[0], len(item[1])),
        reverse=True,
    )
    if not ranked or ranked[0][0] <= 0:
        return None
    return ranked[0][1]


def infer_tool_category(claim: dict[str, Any], verification_item: dict[str, Any]) -> str:
    hypothesis_ids = [
        verification_item.get("hypothesis_id"),
        *(claim.get("hypothesis_candidates") or []),
    ]
    for hypothesis_id in hypothesis_ids:
        if hypothesis_id in TOOL_CATEGORY_BY_HYPOTHESIS:
            return TOOL_CATEGORY_BY_HYPOTHESIS[hypothesis_id]

    text = (claim.get("text") or "").lower()
    if "dataset" in text:
        return "dataset"
    if "assistant" in text or "agent" in text:
        return "agent"
    if "speech" in text or "tts" in text or "voice" in text:
        return "speech_generation"
    if "simulation" in text or "platform" in text:
        return "simulation_tool"
    return "tooling"


def build_tool_index(
    records: list[dict[str, Any]],
    *,
    existing_items: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    manual_items = [
        dict(item)
        for item in (existing_items or [])
        if item.get("source") != "auto"
    ]
    manual_article_ids = {
        item.get("source_article_id")
        for item in manual_items
        if item.get("source_article_id")
    }
    auto_items: list[dict[str, Any]] = []
    seen_keys = {
        (
            item.get("source_article_id"),
            normalize_tool_name(item.get("name", "")),
            item.get("url", ""),
        )
        for item in manual_items
    }

    records_by_time = sorted(
        records,
        key=lambda record: (
            read_json(verification_path(record["article_id"]), default={}).get("verified_at") or "",
            record.get("article_id", ""),
        ),
        reverse=True,
    )

    for record in records_by_time:
        article_id = record.get("article_id")
        if not article_id or article_id in manual_article_ids:
            continue

        claims_doc = read_json(claims_path(article_id), default={"claims": []})
        claims_by_id = {
            claim.get("id"): claim
            for claim in claims_doc.get("claims", [])
            if claim.get("id")
        }
        verification_doc = read_json(verification_path(article_id), default={"items": []})

        for verification_item in verification_doc.get("items", []):
            if verification_item.get("status") not in STRONG_VERIFICATION_STATUSES:
                continue
            claim = claims_by_id.get(verification_item.get("claim_id"))
            if not claim or claim.get("type") != "tool":
                continue

            tool_name = infer_tool_name_from_claim(claim.get("text", ""))
            if not tool_name:
                continue
            url = verification_item.get("source_url") or record.get("url") or ""
            key = (article_id, normalize_tool_name(tool_name), url)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            auto_items.append(
                {
                    "name": tool_name,
                    "category": infer_tool_category(claim, verification_item),
                    "source_article_id": article_id,
                    "source_claim_id": claim.get("id"),
                    "url": url,
                    "source": "auto",
                }
            )

    return manual_items + auto_items


_CANDIDATE_STOPWORDS = {
    "the",
    "and",
    "that",
    "with",
    "from",
    "this",
    "article",
    "claims",
    "claim",
    "states",
    "shows",
    "using",
    "into",
    "for",
    "its",
    "their",
    "have",
    "has",
    "are",
    "was",
    "were",
    "will",
    "can",
    "said",
    "says",
    "which",
    "about",
    "through",
    "based",
    "system",
    "model",
    "models",
    "paper",
    "dataset",
    "company",
    "product",
    "platform",
}

CANDIDATE_VISIBLE_STATUSES = {"observed", "emerging"}
CANDIDATE_TERMINAL_STATUSES = {"promoted", "rejected"}
CANDIDATE_REVIEWABLE_STATUSES = CANDIDATE_VISIBLE_STATUSES
CANDIDATE_MIN_SUPPORT_COUNT = 3
CANDIDATE_MIN_ARTICLE_COUNT = 2
CANDIDATE_MIN_SOURCE_DIVERSITY = 2

CANDIDATE_THEME_LIBRARY: dict[str, dict[str, Any]] = {
    "medical_vertical_semantic_alignment": {
        "statement": "医疗垂直视觉语言模型正在从通用图文对齐转向强领域语义对齐与专用数据资源驱动。",
        "rationale_template": "多篇文章中的未映射证据都在强调医疗垂直场景里的专用数据集、语义软标签、异质关系建模和临床语义对齐，这比“某个项目发布了”更像同一类方法论迁移。",
        "theory": "如果这条候选成立，说明多模态模型在医疗等高语义密度行业里，竞争焦点会从通用图文预训练迁移到更强的领域语义约束、结构化关系建模和专用数据闭环。那时真正有价值的不是“把医学图像也喂给通用 VLM”，而是让模型在临床语义空间里建立稳定对齐。",
        "cluster_signal_template": "以 {top_keyword} 为代表的证据显示，医疗垂直多模态系统开始依赖专用数据和更强领域语义对齐，而不是沿用通用图文预训练范式。",
        "cluster_rationale_template": "这里的关键信号不是单个项目存在，而是数据集建设、语义软标签和结构化临床关系建模被同时拉进同一条方法链。",
        "theme_keywords": ["医疗", "超声", "语义对齐", "专用数据", "垂直VLM"],
        "match_keywords": [
            "ultrasound",
            "clinical",
            "semantic",
            "image-text",
            "graph encoder",
            "soft labels",
            "diagnostic",
            "medical",
        ],
        "min_hits": 2,
    },
    "embodied_hardware_stack_productization": {
        "statement": "具身硬件正在把灵巧手本体、触觉感知、仿真适配和数据采集栈一起产品化，而不是只卖单点部件。",
        "rationale_template": "这类未映射证据反复指向的不是单一硬件参数，而是具身硬件开始把本体设计、触觉、仿真平台兼容和采数链路打包成完整交付栈。",
        "theory": "如果这条候选成立，具身领域下一阶段的竞争将不只发生在机械结构或单个传感器指标上，而会转向谁能把硬件、感知、仿真和数据回流组成更完整的训练闭环。产品化边界会从“部件能力”上移到“整套具身基础设施能力”。",
        "cluster_signal_template": "以 {top_keyword} 为代表的证据显示，具身硬件正在从单点部件能力转向硬件、触觉、仿真和采数链路一体化交付。",
        "cluster_rationale_template": "这些事实放在一起时，强调的不再是某个机械参数，而是整套具身训练基础设施开始被当成产品边界来定义。",
        "theme_keywords": ["具身硬件", "触觉", "仿真适配", "数据采集", "产品化"],
        "match_keywords": [
            "tactile",
            "visuotactile",
            "teleoperation",
            "simulation",
            "mujoco",
            "isaac",
            "omniverse",
            "dexterous",
            "direct-drive",
        ],
        "min_hits": 2,
    },
    "browser_rendering_runtime_shift": {
        "statement": "浏览器前端渲染正在从传统 DOM 路径外溢到更靠近图形运行时的混合渲染栈。",
        "rationale_template": "这些未映射证据共同描述的不是某个前端技巧，而是浏览器开始把 HTML 能力延伸到 canvas/图形运行时边界，意味着渲染模型本身在发生变化。",
        "theory": "如果这条候选成立，未来前端视觉体验的差异化会越来越多地来自浏览器图形运行时和渲染管线层，而不是单纯的组件层样式堆叠。Web 页面会更像一个可编排图形场景，而不是传统 DOM 树的静态排布。",
        "cluster_signal_template": "以 {top_keyword} 为代表的证据显示，浏览器渲染能力正在突破传统 DOM 边界，开始向更接近图形运行时的混合模型延伸。",
        "cluster_rationale_template": "这些信号共同指向的是渲染模型在变化，而不是某个前端技巧本身在流行。",
        "theme_keywords": ["浏览器运行时", "混合渲染", "Canvas", "前端图形化"],
        "match_keywords": [
            "html-in-canvas",
            "canvas",
            "browser",
            "rendering",
            "chrome flag",
            "wicg",
            "draw-element",
        ],
        "min_hits": 2,
    },
    "open_embodied_release_practice": {
        "statement": "具身模型发布正在从单纯论文公开转向代码、权重和训练要素一并开放的完整 release practice。",
        "rationale_template": "这类未映射证据指向的不是单个模型能力，而是具身方向开始把开放发布当成可复现和可落地的一部分工程规范。",
        "theory": "如果这条候选成立，具身领域的竞争门槛会越来越体现为谁能提供完整可复现的 release stack，包括权重、代码、训练细节和部署路径，而不只是论文中的模型指标。",
        "cluster_signal_template": "以 {top_keyword} 为代表的证据显示，具身方向开始把代码、权重和训练要素一并开放，形成更完整的 release practice。",
        "cluster_rationale_template": "这里的重点不是某个模型开源了，而是开放交付本身正在变成具身项目的工程规范。",
        "theme_keywords": ["具身开源", "权重开放", "可复现", "release practice"],
        "match_keywords": [
            "fully open source",
            "weights",
            "open source",
            "release page",
            "training data",
            "code",
        ],
        "min_hits": 2,
    },
}


def slugify(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return text or "cluster"


def candidate_cluster_keywords(claim_text: str, claim_type: str) -> list[str]:
    text = normalize_tool_name(claim_text)
    if not text:
        return []

    keywords: list[str] = []
    if claim_type == "tool":
        tool_name = infer_tool_name_from_claim(text)
        if tool_name:
            keywords.append(tool_name)

    phrase_patterns = [
        r"\b([A-Z][A-Za-z0-9.+/_-]*(?:\s+[A-Z0-9][A-Za-z0-9.+/_-]*){0,3})\b",
        r"\b([A-Za-z][A-Za-z0-9.+/_-]*\s\d(?:\.\d+)?)\b",
        r"\b([A-Za-z]+(?:-[A-Za-z0-9]+)+)\b",
    ]
    for pattern in phrase_patterns:
        for match in re.finditer(pattern, text):
            candidate = normalize_tool_name(match.group(1))
            if candidate and candidate not in keywords:
                keywords.append(candidate)

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9.+/_-]{2,}", text.lower())
    for token in tokens:
        if token in _CANDIDATE_STOPWORDS:
            continue
        if any(ch.isdigit() for ch in token) or "-" in token or len(token) >= 8:
            if token not in keywords:
                keywords.append(token)

    generic_words = {
        "the",
        "this",
        "that",
        "these",
        "those",
        "article",
        "brainco",
        "according",
    }

    def score(keyword: str) -> tuple[int, int]:
        lower = keyword.lower()
        score_value = 0
        if lower in generic_words or lower in _CANDIDATE_STOPWORDS:
            score_value -= 5
        if any(ch.isdigit() for ch in keyword):
            score_value += 3
        if "-" in keyword or "/" in keyword:
            score_value += 2
        if len(keyword) >= 10:
            score_value += 2
        if len(keyword.split()) >= 2:
            score_value += 2
        elif re.search(r"[A-Z]", keyword) and len(keyword) >= 4:
            score_value += 1
        return (score_value, len(keyword))

    ranked = sorted(
        {keyword for keyword in keywords if keyword and keyword.lower() not in _CANDIDATE_STOPWORDS},
        key=score,
        reverse=True,
    )
    return ranked[:5]


def candidate_keywords_from_claim_id(claim_id: str) -> list[str]:
    keywords: list[str] = []
    for token in claim_id.split("_"):
        token = token.strip().lower()
        if not token or token in _CANDIDATE_STOPWORDS:
            continue
        if len(token) <= 2 and not any(ch.isdigit() for ch in token):
            continue
        keywords.append(token)
    return keywords[:5]


def fallback_candidate_cluster_label(claim_text: str, claim_id: str | None = None) -> str:
    if claim_id:
        claim_id_keywords = candidate_keywords_from_claim_id(claim_id)
        if claim_id_keywords:
            return "_".join(claim_id_keywords[:3])
    tokens = re.findall(r"[a-z0-9]+", claim_text.lower())
    filtered = [token for token in tokens if token not in _CANDIDATE_STOPWORDS and len(token) >= 4]
    if filtered:
        return "_".join(filtered[:3])
    return "unmapped_signal"


def candidate_theme_matches(text: str) -> dict[str, int]:
    lower = text.lower()
    matches: dict[str, int] = {}
    for theme_id, config in CANDIDATE_THEME_LIBRARY.items():
        keywords = config.get("match_keywords", [])
        hit_count = sum(1 for keyword in keywords if keyword in lower)
        if hit_count >= int(config.get("min_hits", 2)):
            matches[theme_id] = hit_count
    return matches


def candidate_theme_keywords(theme_id: str) -> list[str]:
    config = CANDIDATE_THEME_LIBRARY.get(theme_id, {})
    keywords = config.get("theme_keywords", [])
    return keywords[:5] if isinstance(keywords, list) else []


def candidate_primary_theme(theme_scores: dict[str, int]) -> str | None:
    if not theme_scores:
        return None
    return max(theme_scores.items(), key=lambda entry: entry[1])[0]


def candidate_statement(theme_id: str, domain: str, fallback_keyword: str) -> str:
    config = CANDIDATE_THEME_LIBRARY.get(theme_id)
    if config and config.get("statement"):
        return str(config["statement"])
    return f"{domain} 方向围绕 {fallback_keyword} 的未映射证据正在汇成一个新的趋势命题。"


def candidate_cluster_statement(theme_id: str | None, top_keyword: str, domain: str) -> str:
    if theme_id:
        config = CANDIDATE_THEME_LIBRARY.get(theme_id, {})
        template = config.get("cluster_signal_template")
        if template:
            return str(template).format(top_keyword=top_keyword)
    return f"以 {top_keyword} 为代表的一组 {domain} 未映射证据，显示这里可能存在尚未被正式 hypothesis 吸收的结构性变化。"


def candidate_cluster_rationale(
    *,
    theme_id: str | None,
    support_count: int,
    article_count: int,
    source_diversity: int,
) -> str:
    if theme_id:
        config = CANDIDATE_THEME_LIBRARY.get(theme_id, {})
        template = config.get("cluster_rationale_template")
        if template:
            base = str(template)
        else:
            base = "这些证据共享同一类更高层变化方向。"
    else:
        base = "这些证据虽然尚未能并入现有 hypothesis，但它们之间已经出现了初步的内容共性。"
    return f"{base} 当前簇内已有 {article_count} 篇文章、{support_count} 条证据、{source_diversity} 个来源。"


def candidate_rationale(
    *,
    theme_id: str,
    article_count: int,
    support_count: int,
    source_cluster_labels: list[str],
    source_cluster_statements: list[str] | None = None,
) -> str:
    config = CANDIDATE_THEME_LIBRARY.get(theme_id, {})
    base = str(config.get("rationale_template", "这些未映射证据在更高层上共享同一个变化方向。"))
    cluster_text = "、".join(source_cluster_labels[:3]) if source_cluster_labels else "多个来源簇"
    statement_text = ""
    if source_cluster_statements:
        statement_text = " ".join(source_cluster_statements[:2])
    if statement_text:
        return (
            f"{base} {statement_text} 当前已汇聚 {article_count} 篇文章、{support_count} 条证据，"
            f"底层来源簇包括 {cluster_text}。"
        )
    return f"{base} 当前已汇聚 {article_count} 篇文章、{support_count} 条证据，底层来源簇包括 {cluster_text}。"


def candidate_theory(theme_id: str) -> str:
    config = CANDIDATE_THEME_LIBRARY.get(theme_id, {})
    return str(config.get("theory", ""))


def candidate_confidence_note(article_count: int, source_diversity: int) -> str:
    if article_count >= 3 and source_diversity >= 3:
        return "该候选已具备跨文章、跨来源的初步稳定性，适合进入优先评审。"
    return "该候选已跨过单篇文章层，但当前仍属于早期信号，适合继续观察或人工审阅。"


def candidate_sort_key(item: dict[str, Any]) -> tuple[int, int, int, str]:
    return (
        int(item.get("source_diversity", 0)),
        int(item.get("article_count", 0)),
        int(item.get("support_count", 0)),
        item.get("last_supported_at") or "",
    )


def candidate_cluster_sort_key(item: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(item.get("article_count", 0)),
        int(item.get("support_count", 0)),
        item.get("last_supported_at") or "",
    )


def load_candidate_hypotheses_doc() -> dict[str, Any]:
    bootstrap_state_files()
    return read_json(
        CANDIDATE_HYPOTHESES_PATH,
        default={"created_at": utc_now(), "last_built_at": None, "clusters": [], "candidates": []},
    )


def write_candidate_hypotheses_doc(payload: dict[str, Any]) -> None:
    payload.setdefault("created_at", utc_now())
    payload.setdefault("last_built_at", None)
    payload.setdefault("clusters", [])
    payload.setdefault("candidates", [])
    write_json(CANDIDATE_HYPOTHESES_PATH, payload)


def find_candidate_entry(candidate_id: str) -> tuple[dict[str, Any], int, dict[str, Any]]:
    payload = load_candidate_hypotheses_doc()
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("candidate_hypotheses.json is malformed: candidates must be a list")
    for index, item in enumerate(candidates):
        if isinstance(item, dict) and item.get("id") == candidate_id:
            return payload, index, item
    raise ValueError(f"Unknown candidate_id: {candidate_id}")


def candidate_hypothesis_id(candidate_id: str) -> str:
    if candidate_id.startswith("candidate_"):
        return candidate_id[len("candidate_") :]
    return candidate_id


def ensure_unique_hypothesis_id(base_id: str, hypotheses_doc: dict[str, Any]) -> str:
    hypothesis_ids = {
        item.get("id")
        for item in hypotheses_doc.get("hypotheses", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if base_id not in hypothesis_ids:
        return base_id
    suffix = 2
    while f"{base_id}_{suffix}" in hypothesis_ids:
        suffix += 1
    return f"{base_id}_{suffix}"


def candidate_review_ref(action: str, approver: str, reason: str | None) -> dict[str, Any]:
    return {
        "action": action,
        "reviewed_at": utc_now(),
        "reviewed_by": approver,
        "reason": reason or "",
    }


def build_candidate_hypotheses(records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    bootstrap_state_files()
    article_records = records or load_all_article_records()
    existing_payload = load_candidate_hypotheses_doc()
    existing_candidates = existing_payload.get("candidates", [])
    existing_by_id = {
        item.get("id"): item
        for item in existing_candidates
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }

    base_clusters: dict[tuple[str, str], dict[str, Any]] = {}
    for record in article_records:
        article_id = record.get("article_id")
        if not article_id:
            continue

        verification_doc = read_json(verification_path(article_id), default={"items": []})
        claims_doc = read_json(claims_path(article_id), default={"claims": []})
        claims_by_id = {
            claim.get("id"): claim
            for claim in claims_doc.get("claims", [])
            if claim.get("id")
        }
        verified_at = verification_doc.get("verified_at")

        for item in verification_doc.get("items", []):
            if item.get("status") not in STRONG_VERIFICATION_STATUSES:
                continue
            if item.get("hypothesis_id"):
                continue

            claim = claims_by_id.get(item.get("claim_id"))
            if not claim:
                continue
            claim_type = claim.get("type")
            if claim_type not in CLAIM_TYPES:
                continue

            claim_text = (claim.get("text") or "").strip()
            if not claim_text:
                continue

            domain = item_domain(item)
            claim_id = claim.get("id")
            seed_keywords = candidate_cluster_keywords(claim_text, claim_type)
            claim_id_keywords = candidate_keywords_from_claim_id(claim_id or "")
            for keyword in claim_id_keywords:
                if keyword not in seed_keywords:
                    seed_keywords.append(keyword)
            cluster_label = (
                slugify(seed_keywords[0].lower())
                if seed_keywords
                else fallback_candidate_cluster_label(claim_text, claim_id)
            )
            cluster_key = (domain, cluster_label)
            cluster = base_clusters.setdefault(
                cluster_key,
                {
                    "domain": domain,
                    "cluster_label": cluster_label,
                    "seed_keywords": [],
                    "support_article_ids": [],
                    "support_claim_ids": [],
                    "source_urls": [],
                    "evidence_items": [],
                    "last_supported_at": None,
                },
            )

            for keyword in seed_keywords:
                if keyword not in cluster["seed_keywords"]:
                    cluster["seed_keywords"].append(keyword)

            if article_id not in cluster["support_article_ids"]:
                cluster["support_article_ids"].append(article_id)
            if claim_id and claim_id not in cluster["support_claim_ids"]:
                cluster["support_claim_ids"].append(claim_id)

            source_url = item.get("source_url") or record.get("url") or ""
            if source_url and source_url not in cluster["source_urls"]:
                cluster["source_urls"].append(source_url)

            cluster["evidence_items"].append(
                {
                    "article_id": article_id,
                    "claim_id": claim_id,
                    "claim_type": claim_type,
                    "status": item.get("status"),
                    "claim_text": claim_text,
                    "assessment": item.get("assessment") or "",
                    "source_url": source_url,
                    "source_title": item.get("source_title") or record.get("title") or "",
                    "verified_at": verified_at,
                }
            )

            if verified_at and (
                cluster["last_supported_at"] is None or verified_at > cluster["last_supported_at"]
            ):
                cluster["last_supported_at"] = verified_at

    cluster_rows: list[dict[str, Any]] = []
    theme_groups: dict[tuple[str, str], dict[str, Any]] = {}
    for cluster in base_clusters.values():
        support_count = len(cluster["evidence_items"])
        if support_count < 2:
            continue

        article_count = len(cluster["support_article_ids"])
        source_diversity = len(cluster["source_urls"])
        seed_keywords = cluster["seed_keywords"][:5]
        top_keyword = seed_keywords[0] if seed_keywords else cluster["cluster_label"].replace("_", " ")
        theme_scores: dict[str, int] = {}
        for item in cluster["evidence_items"]:
            text = " ".join(
                [
                    item.get("claim_text") or "",
                    item.get("assessment") or "",
                    item.get("source_title") or "",
                ]
            )
            for theme_id, score in candidate_theme_matches(text).items():
                theme_scores[theme_id] = theme_scores.get(theme_id, 0) + score

        matched_themes = [
            theme_id
            for theme_id, _score in sorted(theme_scores.items(), key=lambda entry: entry[1], reverse=True)
        ]
        primary_theme = candidate_primary_theme(theme_scores)
        cluster_row = {
            "id": f"cluster_{cluster['domain']}_{slugify(cluster['cluster_label'])}"[:80],
            "domain": cluster["domain"],
            "cluster_label": cluster["cluster_label"],
            "top_keyword": top_keyword,
            "seed_keywords": seed_keywords,
            "support_article_ids": cluster["support_article_ids"],
            "support_claim_ids": cluster["support_claim_ids"],
            "support_count": support_count,
            "article_count": article_count,
            "source_diversity": source_diversity,
            "matched_themes": matched_themes,
            "primary_theme": primary_theme,
            "theme_scores": theme_scores,
            "abstract_statement": candidate_cluster_statement(primary_theme, top_keyword, cluster["domain"]),
            "abstract_rationale": candidate_cluster_rationale(
                theme_id=primary_theme,
                support_count=support_count,
                article_count=article_count,
                source_diversity=source_diversity,
            ),
            "theme_keywords": candidate_theme_keywords(primary_theme) if primary_theme else seed_keywords,
            "last_supported_at": cluster["last_supported_at"],
            "source_urls": cluster["source_urls"],
            "evidence_items": sorted(
                cluster["evidence_items"],
                key=lambda item: (item.get("verified_at") or "", item.get("article_id") or ""),
                reverse=True,
            ),
        }
        cluster_rows.append(cluster_row)

        for theme_id in matched_themes:
            group_key = (cluster["domain"], theme_id)
            group = theme_groups.setdefault(
                group_key,
                {
                    "domain": cluster["domain"],
                    "theme_id": theme_id,
                    "source_cluster_ids": [],
                    "source_cluster_labels": [],
                    "source_cluster_statements": [],
                    "support_article_ids": [],
                    "support_claim_ids": [],
                    "source_urls": [],
                    "evidence_items": [],
                    "last_supported_at": None,
                    "theme_keywords": candidate_theme_keywords(theme_id),
                },
            )
            if cluster_row["id"] not in group["source_cluster_ids"]:
                group["source_cluster_ids"].append(cluster_row["id"])
            if top_keyword not in group["source_cluster_labels"]:
                group["source_cluster_labels"].append(top_keyword)
            cluster_statement = cluster_row["abstract_statement"]
            if cluster_statement not in group["source_cluster_statements"]:
                group["source_cluster_statements"].append(cluster_statement)
            for article_id in cluster["support_article_ids"]:
                if article_id not in group["support_article_ids"]:
                    group["support_article_ids"].append(article_id)
            for claim_id in cluster["support_claim_ids"]:
                if claim_id not in group["support_claim_ids"]:
                    group["support_claim_ids"].append(claim_id)
            for source_url in cluster["source_urls"]:
                if source_url and source_url not in group["source_urls"]:
                    group["source_urls"].append(source_url)
            group["evidence_items"].extend(cluster_row["evidence_items"])
            last_supported_at = cluster["last_supported_at"]
            if last_supported_at and (
                group["last_supported_at"] is None or last_supported_at > group["last_supported_at"]
            ):
                group["last_supported_at"] = last_supported_at

    candidates: list[dict[str, Any]] = []
    for group in theme_groups.values():
        support_count = len(group["evidence_items"])
        article_count = len(group["support_article_ids"])
        source_diversity = len(group["source_urls"])
        if support_count < CANDIDATE_MIN_SUPPORT_COUNT:
            continue
        if article_count < CANDIDATE_MIN_ARTICLE_COUNT:
            continue
        if source_diversity < CANDIDATE_MIN_SOURCE_DIVERSITY:
            continue

        theme_id = group["theme_id"]
        status = "emerging" if article_count >= 3 and source_diversity >= 3 else "observed"
        candidate_id = f"candidate_{group['domain']}_{theme_id}"[:80]
        existing_item = existing_by_id.get(candidate_id)
        if existing_item and existing_item.get("status") in CANDIDATE_TERMINAL_STATUSES:
            candidates.append(dict(existing_item))
            continue

        fallback_keyword = group["source_cluster_labels"][0] if group["source_cluster_labels"] else theme_id
        generated_item = {
            "id": candidate_id,
            "domain": group["domain"],
            "status": status,
            "theme_id": theme_id,
            "statement": candidate_statement(theme_id, group["domain"], fallback_keyword),
            "rationale": candidate_rationale(
                theme_id=theme_id,
                article_count=article_count,
                support_count=support_count,
                source_cluster_labels=group["source_cluster_labels"],
                source_cluster_statements=group["source_cluster_statements"],
            ),
            "theory": candidate_theory(theme_id),
            "theme_keywords": group["theme_keywords"],
            "source_cluster_ids": group["source_cluster_ids"],
            "source_cluster_labels": group["source_cluster_labels"],
            "source_cluster_statements": group["source_cluster_statements"],
            "support_article_ids": group["support_article_ids"],
            "support_claim_ids": group["support_claim_ids"],
            "support_count": support_count,
            "article_count": article_count,
            "source_diversity": source_diversity,
            "last_supported_at": group["last_supported_at"],
            "confidence_note": candidate_confidence_note(article_count, source_diversity),
            "evidence_items": sorted(
                group["evidence_items"],
                key=lambda item: (item.get("verified_at") or "", item.get("article_id") or ""),
                reverse=True,
            ),
        }
        if existing_item and existing_item.get("status") in CANDIDATE_VISIBLE_STATUSES:
            if existing_item.get("theory"):
                generated_item["theory"] = existing_item["theory"]
            if existing_item.get("review_ref"):
                generated_item["review_ref"] = existing_item["review_ref"]
        candidates.append(generated_item)

    candidates.sort(key=candidate_sort_key, reverse=True)
    cluster_rows.sort(key=candidate_cluster_sort_key, reverse=True)
    seen_ids = {
        item.get("id")
        for item in candidates
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    for existing_item in existing_candidates:
        if not isinstance(existing_item, dict):
            continue
        candidate_id = existing_item.get("id")
        if (
            isinstance(candidate_id, str)
            and candidate_id not in seen_ids
            and existing_item.get("status") in CANDIDATE_TERMINAL_STATUSES
        ):
            candidates.append(dict(existing_item))

    candidates.sort(key=candidate_sort_key, reverse=True)
    payload = existing_payload
    payload["last_built_at"] = utc_now()
    payload["clusters"] = cluster_rows
    payload["candidates"] = candidates
    write_candidate_hypotheses_doc(payload)

    visible_count = sum(
        1
        for item in candidates
        if isinstance(item, dict) and item.get("status") in CANDIDATE_VISIBLE_STATUSES
    )
    append_jsonl(
        CHANGE_LOG_PATH,
        {
            "timestamp": utc_now(),
            "event": "build_candidate_hypotheses",
            "cluster_count": len(cluster_rows),
            "candidate_count": visible_count,
            "candidate_total_count": len(candidates),
        },
    )
    return {
        "candidate_count": visible_count,
        "cluster_count": len(cluster_rows),
        "candidate_total_count": len(candidates),
        "clusters": cluster_rows,
        "candidates": candidates,
    }


def build_candidate_evidence_rows(
    candidate: dict[str, Any],
    *,
    article_title_map: dict[str, str],
) -> list[str]:
    rows: list[str] = []
    for item in candidate.get("evidence_items", []):
        article_id = item.get("article_id", "")
        title = html_escape(article_title_map.get(article_id, article_id))
        source_html = f"<a href='#article-{html_escape(article_id)}'>{title}</a>"
        rows.append(
            "<div class='evidence-row'>"
            "<div class='evidence-top'>"
            f"<span class='status status-{html_escape(item.get('status', 'unknown'))}'>"
            f"{html_escape(evidence_status_label(item.get('status', 'unknown')))}"
            "</span>"
            f"{source_html}"
            f"<span class='muted'>{html_escape(item.get('claim_type', ''))}</span>"
            "</div>"
            f"<p class='claim'>{html_escape(item.get('claim_text', ''))}</p>"
            f"<p class='muted'>{html_escape(item.get('assessment') or '暂无核实说明')}</p>"
            "</div>"
        )
    return rows


def build_candidate_card_html(
    candidate: dict[str, Any],
    *,
    article_title_map: dict[str, str],
) -> str:
    theme_keywords = candidate.get("theme_keywords") or candidate.get("seed_keywords") or []
    keywords_html = (
        "<ul class='pill-list'>"
        + "".join(f"<li>{html_escape(keyword)}</li>" for keyword in theme_keywords)
        + "</ul>"
        if theme_keywords
        else "<p class='muted'>暂无关键词</p>"
    )
    evidence_rows = build_candidate_evidence_rows(candidate, article_title_map=article_title_map)
    evidence_html = "".join(evidence_rows) if evidence_rows else "<p class='muted'>暂无证据条目</p>"
    status = candidate.get("status", "observed")
    status_label = "跨文章浮现" if status == "emerging" else "持续观察"
    source_clusters = candidate.get("source_cluster_labels") or []
    source_cluster_text = "、".join(str(item) for item in source_clusters[:4]) if source_clusters else "暂无"
    confidence_note = candidate.get("confidence_note") or ""
    theory = candidate.get("theory") or ""
    parts = [
        "<article class='candidate-card'>",
        f"<p class='candidate-status'>{html_escape(status_label)}</p>",
        f"<h3>{html_escape(candidate.get('statement', ''))}</h3>",
        f"<p class='muted'>candidate_id: <code>{html_escape(candidate.get('id', ''))}</code></p>",
        f"<p>{html_escape(candidate.get('rationale', ''))}</p>",
        f"<p class='muted'>来源簇: {html_escape(source_cluster_text)}</p>",
        "<div class='candidate-metrics'>",
        f"<span>支持文章: {html_escape(candidate.get('article_count', 0))}</span>",
        f"<span>证据条数: {html_escape(candidate.get('support_count', 0))}</span>",
        f"<span>来源多样性: {html_escape(candidate.get('source_diversity', 0))}</span>",
        "</div>",
        keywords_html,
    ]
    if confidence_note:
        parts.append(f"<p class='muted'>{html_escape(confidence_note)}</p>")
    if theory:
        parts.append(f"<p>{html_escape(theory)}</p>")
    parts.extend(
        [
            "<details class='hyp-evidence'>",
            f"<summary>查看候选证据 ({html_escape(candidate.get('support_count', 0))})</summary>",
            "<div class='hyp-evidence-body'>",
            "<p class='muted'>这些证据已被归纳成一个候选趋势命题，但尚未并入正式 hypothesis，也不参与当前后验计算。</p>",
            evidence_html,
            "</div>",
            "</details>",
            "</article>",
        ]
    )
    return "".join(parts)


def build_cluster_card_html(
    cluster: dict[str, Any],
    *,
    article_title_map: dict[str, str],
) -> str:
    theme_keywords = cluster.get("theme_keywords") or cluster.get("seed_keywords") or []
    keywords_html = (
        "<ul class='pill-list'>"
        + "".join(f"<li>{html_escape(keyword)}</li>" for keyword in theme_keywords)
        + "</ul>"
        if theme_keywords
        else "<p class='muted'>暂无关键词</p>"
    )
    evidence_rows = build_candidate_evidence_rows(cluster, article_title_map=article_title_map)
    evidence_html = "".join(evidence_rows) if evidence_rows else "<p class='muted'>暂无证据条目</p>"
    matched_themes = cluster.get("matched_themes") or []
    theme_text = "、".join(str(item) for item in matched_themes[:3]) if matched_themes else "未识别稳定主题"
    parts = [
        "<article class='candidate-card'>",
        "<p class='candidate-status'>潜在主题簇</p>",
        f"<h3>{html_escape(cluster.get('abstract_statement', ''))}</h3>",
        f"<p class='muted'>cluster_id: <code>{html_escape(cluster.get('id', ''))}</code></p>",
        f"<p>{html_escape(cluster.get('abstract_rationale', ''))}</p>",
        f"<p class='muted'>主题识别: {html_escape(theme_text)}</p>",
        "<div class='candidate-metrics'>",
        f"<span>支持文章: {html_escape(cluster.get('article_count', 0))}</span>",
        f"<span>证据条数: {html_escape(cluster.get('support_count', 0))}</span>",
        f"<span>来源多样性: {html_escape(cluster.get('source_diversity', 0))}</span>",
        "</div>",
        keywords_html,
        "<details class='hyp-evidence'>",
        f"<summary>查看底层证据簇 ({html_escape(cluster.get('support_count', 0))})</summary>",
        "<div class='hyp-evidence-body'>",
        "<p class='muted'>这些内容已经出现清晰的主题方向，但目前仍停留在单篇或弱独立性的早期阶段，因此只展示为潜在主题簇。</p>",
        evidence_html,
        "</div>",
        "</details>",
        "</article>",
    ]
    return "".join(parts)


def build_hypothesis_evidence_rows(
    supporting: list[dict[str, Any]],
    *,
    article_title_map: dict[str, str],
    article_url_map: dict[str, str],
    claim_text_map: dict[tuple[str, str], str],
    verification_map: dict[tuple[str, str], dict[str, Any]],
    truncate_assessment: bool,
) -> list[str]:
    rows: list[str] = []
    for si in supporting:
        aid = si.get("article_id", "")
        cid = si.get("claim_id", "")
        title = html_escape(article_title_map.get(aid, aid))
        article_url = article_url_map.get(aid, "")
        claim_text = html_escape(claim_text_map.get((aid, cid), cid))
        status = si.get("status", "unknown")
        status_label = evidence_status_label(status)

        verif_item = verification_map.get((aid, cid), {})
        assessment = (verif_item.get("assessment") or "").strip()
        if truncate_assessment and assessment and len(assessment) > 120:
            cut = assessment.find("。")
            if cut > 0 and cut < 200:
                assessment = assessment[: cut + 1]
            else:
                assessment = assessment[:120] + "…"

        source_html = (
            f"<a href='#article-{html_escape(aid)}'>{title}</a>"
            if article_url
            else f"<span>{title}</span>"
        )
        open_html = (
            f"<a class='mini-link' href='#article-{html_escape(aid)}'>查看文章</a>"
            if article_url
            else ""
        )

        rows.append(
            "<div class='evidence-row'>"
            f"<div class='evidence-top'>"
            f"<span class='status status-{html_escape(status)}'>{html_escape(status_label)}</span>"
            f"{source_html}"
            f"{open_html}"
            "</div>"
            f"<p class='claim'>{claim_text}</p>"
            f"<p class='muted'>{html_escape(assessment or '暂无核实说明')}</p>"
            "</div>"
        )
    return rows


def build_hypothesis_card_html(
    item: dict[str, Any],
    *,
    article_title_map: dict[str, str],
    article_url_map: dict[str, str],
    claim_text_map: dict[tuple[str, str], str],
    verification_map: dict[tuple[str, str], dict[str, Any]],
    highlighted_article_ids: set[str],
) -> str:
    """Render a single hypothesis card with evidence drill-down, new-badge, and theory."""
    supporting = item.get("supporting_items") or []
    latest_supporting = [
        si for si in supporting if si.get("article_id") in highlighted_article_ids
    ]
    is_recent = bool(latest_supporting)

    card_class = "hypothesis-card has-new-evidence" if is_recent else "hypothesis-card"
    new_badge = "<span class='new-badge'>本次新增</span>" if is_recent else ""

    # --- Score bar ---
    pct = band_fill_fraction(item.get("posterior_probability")) * 100
    label = band_label(item.get("posterior_probability"))
    score_html = (
        f"<div class='score' style='--pct:{pct:.1f}%'>"
        "<div class='score-bar'></div>"
        f"<span>{html_escape(label)}</span>"
        "</div>"
    )

    # --- Theory expand (改动 3) ---
    theory_text = hypothesis_theory_text(item)
    theory_preview = ""
    if theory_text:
        theory_preview = theory_text.splitlines()[0].strip()
    if len(theory_preview) > 110:
        theory_preview = theory_preview[:110].rstrip() + "…"
    detail_href = html_escape(hypothesis_detail_href(item.get("id", "")))
    theory_preview_html = ""
    if theory_preview:
        theory_preview_html = (
            f"<p class='muted theory-preview'>{html_escape(theory_preview)}</p>"
        )
    theory_html = (
        "<div class='hypothesis-links'>"
        f"<a class='detail-link' href='{detail_href}'>查看理论</a>"
        f"{theory_preview_html}"
        "</div>"
    )

    # --- Evidence drill-down (改动 1) ---
    evidence_rows = build_hypothesis_evidence_rows(
        supporting,
        article_title_map=article_title_map,
        article_url_map=article_url_map,
        claim_text_map=claim_text_map,
        verification_map=verification_map,
        truncate_assessment=True,
    )

    recent_titles: list[str] = []
    for si in latest_supporting:
        aid = si.get("article_id", "")
        title = article_title_map.get(aid, aid)
        if title and title not in recent_titles:
            recent_titles.append(title)

    recent_html = ""
    if recent_titles:
        preview = "；".join(html_escape(title) for title in recent_titles[:2])
        extra = ""
        if len(recent_titles) > 2:
            extra = f" 等 {len(recent_titles)} 篇"
        recent_html = (
            "<p class='recent-evidence'>"
            f"本次新增证据来自：{preview}{extra}"
            "</p>"
        )

    evidence_body = "".join(evidence_rows) if evidence_rows else "<p class='muted'>暂无证据条目</p>"
    evidence_html = (
        "<details class='hyp-evidence'>"
        f"<summary>证据条数: {len(supporting)} — 查看支撑证据</summary>"
        f"<div class='hyp-evidence-body'>{evidence_body}</div>"
        "</details>"
    )

    return (
        f"<article class='{card_class}'>"
        f"{new_badge}"
        f"{score_html}"
        f"<h3>{html_escape(item.get('statement', ''))}</h3>"
        f"<p>{html_escape(item.get('rationale', ''))}</p>"
        f"{recent_html}"
        f"{theory_html}"
        f"{evidence_html}"
        "</article>"
    )


def build_hypothesis_detail_html(
    item: dict[str, Any],
    *,
    article_title_map: dict[str, str],
    article_url_map: dict[str, str],
    claim_text_map: dict[tuple[str, str], str],
    verification_map: dict[tuple[str, str], dict[str, Any]],
) -> str:
    supporting = item.get("supporting_items") or []
    evidence_rows = build_hypothesis_evidence_rows(
        supporting,
        article_title_map=article_title_map,
        article_url_map=article_url_map,
        claim_text_map=claim_text_map,
        verification_map=verification_map,
        truncate_assessment=False,
    )
    theory_text = hypothesis_theory_text(item)
    theory_html = "".join(
        f"<p>{html_escape(paragraph)}</p>"
        for paragraph in theory_text.splitlines()
        if paragraph.strip()
    ) or "<p class='muted'>暂无更详细的理论说明。</p>"
    tags = item.get("meta_tags") or []
    tags_html = (
        "<ul class='pill-list'>"
        + "".join(f"<li>{html_escape(tag)}</li>" for tag in tags)
        + "</ul>"
        if tags
        else "<p class='muted'>暂无 meta tags</p>"
    )
    statement = html_escape(item.get("statement", ""))
    rationale = html_escape(item.get("rationale", ""))
    posterior = html_escape(band_label(item.get("posterior_probability")))
    probability = item.get("posterior_probability")
    probability_html = f"{probability * 100:.1f}%" if isinstance(probability, (int, float)) else "N/A"
    newest = item.get("newest_evidence_at")
    newest_html = html_escape(newest) if newest else "暂无"
    evidence_html = "".join(evidence_rows) if evidence_rows else "<p class='muted'>暂无支撑证据。</p>"

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{statement}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5efe5;
      --paper: rgba(255, 252, 246, 0.94);
      --line: #d7c8b1;
      --ink: #1f2937;
      --muted: #6b7280;
      --accent: #d97706;
      --accent-soft: rgba(217, 119, 6, 0.12);
      --radius: 24px;
      --shadow: 0 20px 45px rgba(94, 71, 38, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(217,119,6,0.12), transparent 32%),
        linear-gradient(180deg, #fbf7ef 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .page {{
      width: min(980px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0 48px;
    }}
    .panel {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 24px;
      box-shadow: var(--shadow);
      margin-top: 18px;
    }}
    .back-link {{
      display: inline-flex;
      text-decoration: none;
      color: #92400e;
      margin-bottom: 10px;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 18px 0;
    }}
    .meta-card {{
      background: #fffaf1;
      border: 1px solid #ead9bc;
      border-radius: 16px;
      padding: 14px;
    }}
    .meta-label, .muted {{ color: var(--muted); }}
    .meta-label {{ margin: 0 0 6px; font-size: 0.9rem; }}
    .meta-value {{ margin: 0; font-weight: 700; }}
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
    }}
    .status-verified {{
      background: #dcfce7;
      color: #166534;
      border-color: #bbf7d0;
    }}
    .status-partially_verified {{
      background: #fef3c7;
      color: #92400e;
      border-color: #fde68a;
    }}
    .claim {{ font-weight: 600; margin: 0 0 8px; }}
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
    .mini-link {{ color: #92400e; text-decoration: none; }}
  </style>
</head>
<body>
  <main class="page">
    <section class="panel">
      <a class="back-link" href="../index.html#trends">返回核心趋势</a>
      <p class="muted">假设详情</p>
      <h1>{statement}</h1>
      <p>{rationale}</p>
      <div class="meta">
        <article class="meta-card">
          <p class="meta-label">当前后验</p>
          <p class="meta-value">{posterior}</p>
        </article>
        <article class="meta-card">
          <p class="meta-label">概率</p>
          <p class="meta-value">{probability_html}</p>
        </article>
        <article class="meta-card">
          <p class="meta-label">证据条数</p>
          <p class="meta-value">{len(supporting)}</p>
        </article>
        <article class="meta-card">
          <p class="meta-label">最近证据时间</p>
          <p class="meta-value">{newest_html}</p>
        </article>
      </div>
      <h2>理论展开</h2>
      {theory_html}
      <h2>Meta Tags</h2>
      {tags_html}
    </section>
    <section class="panel">
      <h2>支撑证据</h2>
      {evidence_html}
    </section>
  </main>
</body>
</html>
"""


def build_report_html() -> str:
    framework = read_json(FRAMEWORK_PATH, default={})
    hypotheses = read_json(HYPOTHESES_PATH, default={"hypotheses": []})
    candidate_doc = read_json(CANDIDATE_HYPOTHESES_PATH, default={"candidates": []})
    synthesis = read_json(SYNTHESIS_PATH, default={})
    github_config = read_github_config()
    records = load_all_article_records()

    hypothesis_index = {item["id"]: item for item in hypotheses.get("hypotheses", [])}
    included_ids = set(synthesis.get("included_articles", []))
    included_records = [record for record in records if record.get("article_id") in included_ids]
    pending_statuses = {
        "excluded_until_verified",
        "verification_staged",
        "held_for_review",
        "held_out_pending_better_evidence",
    }
    pending_records = [
        record
        for record in records
        if record.get("article_id") not in included_ids
        and record.get("analysis_state", {}).get("bayesian_status") in pending_statuses
    ]
    excluded_records = [
        record
        for record in records
        if record.get("article_id") not in included_ids
        and record.get("analysis_state", {}).get("bayesian_status") == "excluded_after_verification"
    ]

    (
        article_title_map,
        article_url_map,
        _article_verified_at_map,
        claim_text_map,
        verification_map,
    ) = _build_evidence_lookup_maps(records)
    _latest_apply_timestamp, latest_article_ids = latest_applied_article_batch()

    active_hypotheses = sorted(
        hypotheses.get("hypotheses", []),
        key=lambda item: item.get("posterior_probability", 0.0),
        reverse=True,
    )
    candidate_hypotheses = sorted(
        [
            item
            for item in candidate_doc.get("candidates", [])
            if isinstance(item, dict) and item.get("status") in CANDIDATE_VISIBLE_STATUSES
        ],
        key=candidate_sort_key,
        reverse=True,
    )
    potential_clusters = sorted(
        [
            item
            for item in candidate_doc.get("clusters", [])
            if isinstance(item, dict) and item.get("primary_theme")
        ],
        key=candidate_cluster_sort_key,
        reverse=True,
    )
    tool_index = build_tool_index(
        records,
        existing_items=synthesis.get("tool_index", []),
    )

    hypothesis_cards = "".join(
        build_hypothesis_card_html(
            item,
            article_title_map=article_title_map,
            article_url_map=article_url_map,
            claim_text_map=claim_text_map,
            verification_map=verification_map,
            highlighted_article_ids=latest_article_ids,
        )
        for item in active_hypotheses
    )
    candidate_cards = "".join(
        build_candidate_card_html(
            item,
            article_title_map=article_title_map,
        )
        for item in candidate_hypotheses
    )
    cluster_cards = "".join(
        build_cluster_card_html(
            item,
            article_title_map=article_title_map,
        )
        for item in potential_clusters
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
            for item in tool_index
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
        ("候选假设", len(candidate_hypotheses)),
        ("潜在主题簇", len(potential_clusters)),
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
    .stat-card, .hypothesis-card, .tool-card, .excluded-card, .candidate-card {{
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
    .hypothesis-card {{
      position: relative;
    }}
    .candidate-card {{
      border-style: dashed;
      background: rgba(255, 251, 243, 0.92);
    }}
    .candidate-status {{
      margin: 0 0 8px;
      color: #92400e;
      font-size: 0.9rem;
      font-weight: 700;
    }}
    .candidate-metrics {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 12px 0;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .candidate-metrics span {{
      padding: 6px 10px;
      border-radius: 999px;
      background: #f7efe1;
      border: 1px solid #eadcc7;
    }}
    .has-new-evidence {{
      border-color: #f59e0b;
      box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.15);
    }}
    .new-badge {{
      position: absolute;
      top: 14px;
      right: 14px;
      display: inline-flex;
      align-items: center;
      padding: 5px 10px;
      border-radius: 999px;
      background: #f59e0b;
      color: #fffaf0;
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.02em;
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
    .recent-evidence {{
      margin: 0 0 10px;
      color: #92400e;
      font-weight: 600;
    }}
    .hypothesis-links {{
      margin: 12px 0;
    }}
    .detail-link, .mini-link {{
      color: #92400e;
      text-decoration: none;
      font-weight: 600;
    }}
    .detail-link:hover, .mini-link:hover {{
      text-decoration: underline;
    }}
    .theory-preview {{
      margin: 8px 0 0;
    }}
    .hyp-evidence {{
      margin-top: 14px;
      border-top: 1px solid #e7dece;
      padding-top: 12px;
    }}
    .hyp-evidence summary {{
      cursor: pointer;
      font-weight: 700;
      color: #7c4a12;
    }}
    .hyp-evidence-body {{
      margin-top: 12px;
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
    .status-verification_staged {{
      background: #fef3c7;
      color: #92400e;
      border-color: #fde68a;
    }}
    .status-unverified, .status-excluded_after_verification {{
      background: #fee2e2;
      color: #991b1b;
      border-color: #fecaca;
    }}
    .status-held_for_review, .status-held_out_pending_better_evidence, .status-excluded_until_verified {{
      background: #e0e7ff;
      color: #3730a3;
      border-color: #c7d2fe;
    }}
    .status-note {{
      margin: 0 0 18px;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid #eadcc7;
      background: #f7f2e8;
    }}
    .status-note p {{
      margin: 0 0 6px;
    }}
    .status-note p:last-child {{
      margin-bottom: 0;
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
        <a href="#candidates">候选假设</a>
        <a href="#clusters">潜在主题簇</a>
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

    <section class="section" id="candidates">
      <h2>候选假设</h2>
      <p class="muted">这里展示的是从多篇、相对独立的未映射证据中抽象出来的候选趋势命题。它们不会进入当前后验计算，只用于提示框架可能需要生长。</p>
      <div class="hypothesis-grid">{candidate_cards if candidate_cards else "<p class='muted'>暂无候选假设</p>"}</div>
    </section>

    <section class="section" id="clusters">
      <h2>潜在主题簇</h2>
      <p class="muted">这里展示的是已经出现明确主题方向、但尚未跨过 candidate 门槛的早期信号。它们通常仍偏单篇或弱独立性，因此先作为观察层，不进入 posterior，也不开放 promote/reject。</p>
      <div class="hypothesis-grid">{cluster_cards if cluster_cards else "<p class='muted'>暂无潜在主题簇</p>"}</div>
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
      <p class="muted">这里展示已经入库、但尚未进入后验的文章，包括待提取、待核实、已暂存待应用、以及等待人工复核的状态。</p>
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

    hypotheses_doc = read_json(HYPOTHESES_PATH, default={"hypotheses": []})
    records = load_all_article_records()
    (
        article_title_map,
        article_url_map,
        _article_verified_at_map,
        claim_text_map,
        verification_map,
    ) = _build_evidence_lookup_maps(records)
    HYPOTHESIS_DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in HYPOTHESIS_DETAIL_DIR.glob("*.html"):
        old_file.unlink()
    for item in hypotheses_doc.get("hypotheses", []):
        hypothesis_id = item.get("id")
        if not hypothesis_id:
            continue
        detail_html = build_hypothesis_detail_html(
            item,
            article_title_map=article_title_map,
            article_url_map=article_url_map,
            claim_text_map=claim_text_map,
            verification_map=verification_map,
        )
        (HYPOTHESIS_DETAIL_DIR / f"{hypothesis_id}.html").write_text(detail_html, encoding="utf-8")

    append_jsonl(
        CHANGE_LOG_PATH,
        {
            "timestamp": utc_now(),
            "event": "build_report",
            "output_path": str(output_path),
        },
    )
    return {"output_path": str(output_path)}


def build_candidates() -> dict[str, Any]:
    bootstrap_state_files()
    result = build_candidate_hypotheses(load_all_article_records())
    return {
        "candidate_count": result.get("candidate_count", 0),
        "candidate_path": str(CANDIDATE_HYPOTHESES_PATH),
    }


def promote_candidate(
    candidate_id: str,
    *,
    approver: str = "human:github",
    reason: str | None = None,
) -> dict[str, Any]:
    payload, index, candidate = find_candidate_entry(candidate_id)
    status = candidate.get("status")
    if status not in CANDIDATE_REVIEWABLE_STATUSES:
        return {
            "ok": False,
            "candidate_id": candidate_id,
            "error": f"candidate status must be one of {sorted(CANDIDATE_REVIEWABLE_STATUSES)} (got {status!r})",
        }

    hypotheses_doc = read_json(HYPOTHESES_PATH, default={"hypotheses": []})
    hypothesis_id = ensure_unique_hypothesis_id(candidate_hypothesis_id(candidate_id), hypotheses_doc)
    new_hypothesis = {
        "id": hypothesis_id,
        "domain": candidate.get("domain") or DEFAULT_DOMAIN,
        "meta_tags": [],
        "statement": candidate.get("statement", ""),
        "prior_log_odds": 0.0,
        "status": "active",
        "rationale": candidate.get("rationale", ""),
        "theory": candidate.get("theory") or candidate.get("rationale", ""),
        "posterior_log_odds": 0.0,
        "posterior_probability": 0.5,
        "posterior_band": posterior_band(0.5),
        "last_recomputed_at": None,
        "supporting_items": [],
        "newest_evidence_at": None,
    }
    hypotheses_doc.setdefault("hypotheses", []).append(new_hypothesis)
    write_json(HYPOTHESES_PATH, hypotheses_doc)

    updated_items = 0
    touched_articles: set[str] = set()
    for evidence_item in candidate.get("evidence_items", []):
        article_id = evidence_item.get("article_id")
        claim_id = evidence_item.get("claim_id")
        if not isinstance(article_id, str) or not isinstance(claim_id, str):
            continue
        verification_doc = read_json(verification_path(article_id), default={"items": []})
        changed = False
        for item in verification_doc.get("items", []):
            if item.get("claim_id") != claim_id:
                continue
            if item.get("status") not in STRONG_VERIFICATION_STATUSES:
                continue
            if item.get("hypothesis_id"):
                continue
            item["hypothesis_id"] = hypothesis_id
            changed = True
            updated_items += 1
        if changed:
            write_json(verification_path(article_id), verification_doc)
            touched_articles.add(article_id)

    payload["candidates"][index]["status"] = "promoted"
    payload["candidates"][index]["promoted_hypothesis_id"] = hypothesis_id
    payload["candidates"][index]["review_ref"] = candidate_review_ref("promote", approver, reason)
    write_candidate_hypotheses_doc(payload)

    refresh_record_states()
    recompute_posteriors()
    append_jsonl(
        CHANGE_LOG_PATH,
        {
            "timestamp": utc_now(),
            "event": "promote_candidate",
            "candidate_id": candidate_id,
            "hypothesis_id": hypothesis_id,
            "updated_items": updated_items,
            "approver": approver,
        },
    )
    return {
        "ok": True,
        "candidate_id": candidate_id,
        "hypothesis_id": hypothesis_id,
        "updated_items": updated_items,
        "touched_articles": sorted(touched_articles),
        "status": "promoted",
    }


def reject_candidate(
    candidate_id: str,
    *,
    approver: str = "human:github",
    reason: str | None = None,
) -> dict[str, Any]:
    payload, index, candidate = find_candidate_entry(candidate_id)
    status = candidate.get("status")
    if status not in CANDIDATE_REVIEWABLE_STATUSES:
        return {
            "ok": False,
            "candidate_id": candidate_id,
            "error": f"candidate status must be one of {sorted(CANDIDATE_REVIEWABLE_STATUSES)} (got {status!r})",
        }

    payload["candidates"][index]["status"] = "rejected"
    payload["candidates"][index]["review_ref"] = candidate_review_ref("reject", approver, reason)
    write_candidate_hypotheses_doc(payload)
    append_jsonl(
        CHANGE_LOG_PATH,
        {
            "timestamp": utc_now(),
            "event": "reject_candidate",
            "candidate_id": candidate_id,
            "approver": approver,
        },
    )
    return {
        "ok": True,
        "candidate_id": candidate_id,
        "status": "rejected",
    }


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

    review_cmd = subparsers.add_parser(
        "review-held",
        help="Apply a one-shot human review action for a held/staged verification draft",
    )
    review_cmd.add_argument("--article-id", required=True)
    review_cmd.add_argument(
        "--action",
        required=True,
        choices=sorted(REVIEW_ACTIONS),
    )
    review_cmd.add_argument(
        "--approver",
        default="human:github",
        help="Identifier for the human review actor (default: human:github)",
    )
    review_cmd.add_argument(
        "--reason",
        help="Optional human review rationale; defaults to a built-in message per action",
    )

    source_context_cmd = subparsers.add_parser(
        "build-source-context",
        help="Extract cited source links from article HTML and fetch a small source context cache",
    )
    source_context_cmd.add_argument("--article-id", required=True)
    source_context_cmd.add_argument("--limit", type=int, default=5)

    subparsers.add_parser(
        "build-candidates",
        help="Build candidate hypotheses from verified but unmapped evidence",
    )
    promote_candidate_cmd = subparsers.add_parser(
        "promote-candidate",
        help="Promote a candidate hypothesis into a formal hypothesis and attach its evidence",
    )
    promote_candidate_cmd.add_argument("--candidate-id", required=True)
    promote_candidate_cmd.add_argument("--approver", default="human:github")
    promote_candidate_cmd.add_argument("--reason")

    reject_candidate_cmd = subparsers.add_parser(
        "reject-candidate",
        help="Reject a candidate hypothesis and remove it from the open candidate queue",
    )
    reject_candidate_cmd.add_argument("--candidate-id", required=True)
    reject_candidate_cmd.add_argument("--approver", default="human:github")
    reject_candidate_cmd.add_argument("--reason")

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

    meta_scan_cmd = subparsers.add_parser(
        "meta-scan",
        help="Read-only cross-hypothesis meta_tag cluster scan (never writes state)",
    )
    meta_scan_cmd.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the formatted text report",
    )
    meta_scan_cmd.add_argument(
        "--domain",
        help="Restrict the scan to a single domain (e.g. 'ai'). Omit to scan all domains.",
    )

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

    if args.command == "review-held":
        try:
            result = review_held(
                article_id=args.article_id,
                action=args.action,
                approver=args.approver,
                reason=args.reason,
            )
        except ValueError as exc:
            print_json({"ok": False, "error": str(exc)})
            return 2
        print_json(result)
        return 0 if result.get("ok") else 2

    if args.command == "build-source-context":
        try:
            result = build_source_context(
                article_id=args.article_id,
                limit=args.limit,
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

    if args.command == "build-candidates":
        result = build_candidates()
        print_json(result)
        return 0

    if args.command == "promote-candidate":
        try:
            result = promote_candidate(
                candidate_id=args.candidate_id,
                approver=args.approver,
                reason=args.reason,
            )
        except ValueError as exc:
            print_json({"ok": False, "error": str(exc)})
            return 2
        print_json(result)
        return 0 if result.get("ok") else 2

    if args.command == "reject-candidate":
        try:
            result = reject_candidate(
                candidate_id=args.candidate_id,
                approver=args.approver,
                reason=args.reason,
            )
        except ValueError as exc:
            print_json({"ok": False, "error": str(exc)})
            return 2
        print_json(result)
        return 0 if result.get("ok") else 2

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

    if args.command == "meta-scan":
        report = meta_scan(domain_filter=args.domain)
        if args.json:
            print_json(report)
        else:
            for line in format_meta_scan(report):
                print(line)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

"""Post/update GitHub Issue comments for the knowledge ingest workflow."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTICLE_ROOT = ROOT / "knowledge_state" / "articles"


def run_json(command: list[str]) -> Any:
    raw = subprocess.check_output(command, cwd=ROOT, text=True, encoding="utf-8")
    return json.loads(raw or "null")


def run_text(command: list[str]) -> str:
    return subprocess.check_output(command, cwd=ROOT, text=True, encoding="utf-8")


def issue_comments(issue_number: str) -> list[dict[str, Any]]:
    repository = os.environ["GITHUB_REPOSITORY"]
    return run_json(
        [
            "gh",
            "api",
            f"repos/{repository}/issues/{issue_number}/comments?per_page=100",
        ]
    )


def issue_data(issue_number: str) -> dict[str, Any]:
    return run_json(["gh", "issue", "view", issue_number, "--json", "body,url"])


def first_comment_with_marker(issue_number: str, marker: str) -> dict[str, Any] | None:
    for comment in issue_comments(issue_number):
        if marker in (comment.get("body") or ""):
            return comment
    return None


def post_or_update(issue_number: str, marker: str, body: str) -> None:
    existing = first_comment_with_marker(issue_number, marker)
    if existing:
        repository = os.environ["GITHUB_REPOSITORY"]
        comment_id = str(existing["id"])
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
            json.dump({"body": body}, handle, ensure_ascii=False)
            input_path = handle.name
        try:
            subprocess.check_call(
                [
                    "gh",
                    "api",
                    "-X",
                    "PATCH",
                    f"repos/{repository}/issues/comments/{comment_id}",
                    "--input",
                    input_path,
                ],
                cwd=ROOT,
            )
        finally:
            Path(input_path).unlink(missing_ok=True)
        return

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".md", delete=False) as handle:
        handle.write(body)
        input_path = handle.name
    try:
        subprocess.check_call(
            ["gh", "issue", "comment", issue_number, "--body-file", input_path],
            cwd=ROOT,
        )
    finally:
        Path(input_path).unlink(missing_ok=True)


def read_json(path: Path, default: Any | None = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def knowledge_status() -> dict[str, Any]:
    try:
        raw = run_text(["python", "scripts/knowledge_pipeline.py", "status"])
        return json.loads(raw)
    except Exception as exc:
        return {"status_error": str(exc)}


def matched_articles(issue_number: str) -> list[dict[str, Any]]:
    issue = issue_data(issue_number)
    issue_body = issue.get("body") or ""
    issue_url = issue.get("url") or ""
    matched: list[dict[str, Any]] = []

    for article_dir in sorted(ARTICLE_ROOT.glob("*")):
        if not article_dir.is_dir():
            continue
        record = read_json(article_dir / "record.json")
        if not record:
            continue

        record_blob = json.dumps(record, ensure_ascii=False)
        article_url = record.get("url") or ""
        issue_match = bool(issue_url and issue_url in record_blob)
        body_match = bool(article_url and article_url in issue_body)
        if not issue_match and not body_match:
            continue

        classification = read_json(article_dir / "classification.json")
        extraction = read_json(article_dir / "extraction.json")
        concepts = read_json(article_dir / "concepts.json")
        matched.append(
            {
                "article_id": article_dir.name,
                "title": record.get("title") or article_url or article_dir.name,
                "url": article_url,
                "full_text_status": (record.get("content_state") or {}).get(
                    "full_text_status", "unknown"
                ),
                "classification_method": classification.get("method", "unknown"),
                "primary_category": classification.get("primary_category", "unknown"),
                "secondary_categories": classification.get("secondary_categories") or [],
                "tags": classification.get("tags") or [],
                "importance": classification.get("importance", "unknown"),
                "extraction_status": extraction.get("extraction_status", "unknown"),
                "concept_count": len(
                    concepts.get("concept_candidates") or concepts.get("concepts") or []
                ),
            }
        )

    return matched


def workflow_url() -> str:
    server_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    if repository and run_id:
        return f"{server_url}/{repository}/actions/runs/{run_id}"
    return ""


def pages_url() -> str:
    return os.environ.get("PAGES_URL") or "docs/knowledge.html"


def ack_body(issue_number: str) -> str:
    lines = [
        f"<!-- knowledge-ingest-ack:{issue_number} -->",
        "已收到这篇文章，开始进入知识系统处理。",
        "",
        "- 当前阶段：排队、抓取全文、分类、入库",
        "- 后续回复：先确认文章入库，再确认深度综述和表达框架刷新完成",
        "- 旧贝叶斯系统：保留为历史页，不会被这个流程更新",
    ]
    url = workflow_url()
    if url:
        lines.append(f"- Workflow：{url}")
    return "\n".join(lines)


def render_articles(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return [
            "没有找到与这个 issue 匹配的文章记录。可能原因是链接被编辑、抓取被阻止，或文章已经通过其他 issue 入库。",
        ]

    lines = ["本次匹配到的文章："]
    for item in items:
        lines.append(f"- `{item['article_id']}` {item['title']}")
        if item["url"]:
            lines.append(f"  - 来源：{item['url']}")
        lines.append(f"  - 全文抓取：`{item['full_text_status']}`")
        lines.append(f"  - 内容提取：`{item['extraction_status']}`")
        lines.append(f"  - 主分类：`{item['primary_category']}`")
        if item["secondary_categories"]:
            secondary = ", ".join(f"`{value}`" for value in item["secondary_categories"])
            lines.append(f"  - 次级分类：{secondary}")
        lines.append(f"  - 分类方式：`{item['classification_method']}`")
        lines.append(f"  - 重要性：`{item['importance']}`")
        if item["tags"]:
            tags = ", ".join(f"`{value}`" for value in item["tags"][:8])
            lines.append(f"  - 标签：{tags}")
        lines.append(f"  - 页面：{pages_url()}#articles")
    return lines


def result_body(issue_number: str, phase: str) -> str:
    if phase == "quick":
        marker = f"<!-- knowledge-ingest-quick:{issue_number} -->"
        title = "文章已进入知识系统。"
        details = [
            "分类文章库和基础知识页面已经完成一次快速更新。",
            "深度综述、判断层和表达框架正在继续刷新；完成后会再更新一次评论。",
        ]
    elif phase == "final":
        marker = f"<!-- knowledge-ingest-final:{issue_number} -->"
        title = "知识系统更新完成。"
        details = [
            "文章库、分类、深度综述、判断层、表达框架和静态页面已经刷新。",
            "旧贝叶斯系统保留为历史页，不参与这次更新。",
        ]
    else:
        raise ValueError(f"Unsupported phase: {phase}")

    lines = [marker, title, "", *details, "", *render_articles(matched_articles(issue_number))]
    lines.extend(
        [
            "",
            "知识系统覆盖状态：",
            "```json",
            json.dumps(knowledge_status(), ensure_ascii=False, indent=2),
            "```",
        ]
    )
    url = workflow_url()
    if url:
        lines.extend(["", f"Workflow：{url}"])
    return "\n".join(lines)


def command_ack(args: argparse.Namespace) -> None:
    issue_number = str(args.issue_number)
    post_or_update(
        issue_number,
        f"<!-- knowledge-ingest-ack:{issue_number} -->",
        ack_body(issue_number),
    )


def command_result(args: argparse.Namespace) -> None:
    issue_number = str(args.issue_number)
    post_or_update(
        issue_number,
        f"<!-- knowledge-ingest-{args.phase}:{issue_number} -->",
        result_body(issue_number, args.phase),
    )


def command_should_ingest(args: argparse.Namespace) -> None:
    issue_number = str(args.issue_number)
    action = str(args.action or "")
    final_marker = f"<!-- knowledge-ingest-final:{issue_number} -->"
    try:
        already_finished = first_comment_with_marker(issue_number, final_marker) is not None
    except Exception as exc:
        print(f"Comment lookup failed; ingest will continue: {exc}", file=sys.stderr)
        already_finished = False
    should_ingest = not (action in {"opened", "labeled"} and already_finished)
    print(f"should_ingest={'true' if should_ingest else 'false'}")
    print(f"already_finished={'true' if already_finished else 'false'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    ack = subparsers.add_parser("ack")
    ack.add_argument("--issue-number", required=True)
    ack.set_defaults(func=command_ack)

    result = subparsers.add_parser("result")
    result.add_argument("--issue-number", required=True)
    result.add_argument("--phase", choices=["quick", "final"], required=True)
    result.set_defaults(func=command_result)

    should_ingest = subparsers.add_parser("should-ingest")
    should_ingest.add_argument("--issue-number", required=True)
    should_ingest.add_argument("--action", required=True)
    should_ingest.set_defaults(func=command_should_ingest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

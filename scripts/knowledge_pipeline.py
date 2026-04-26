"""Additive knowledge abstraction pipeline.

This script starts the non-Bayesian redesign alongside the existing
bayesian_reader.py workflow. It reuses article ingest/fetch artifacts, then
adds four MVP layers:

1. article extraction
2. classification and tags
3. internal concept abstraction updates
4. topic review generation

The implementation keeps deterministic fallbacks, but uses an LLM when a local
OpenRouter key is available for higher-quality classification and review
drafting. The state shape and review loop should still work without a live API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import bayesian_reader as br  # noqa: E402


STATE_DIR = br.STATE_DIR
ARTICLES_DIR = STATE_DIR / "articles"
TAXONOMY_DIR = STATE_DIR / "taxonomy"
CONCEPTS_DIR = STATE_DIR / "concepts"
REVIEWS_DIR = STATE_DIR / "reviews"
JUDGMENTS_DIR = STATE_DIR / "judgments"
THEORIES_DIR = STATE_DIR / "theories"
FRAMEWORKS_DIR = STATE_DIR / "frameworks"
REPORT_PATH = br.REPORT_DIR / "knowledge.html"

GLOBAL_CONCEPTS_PATH = CONCEPTS_DIR / "concepts.json"
CONCEPT_RELATIONS_PATH = CONCEPTS_DIR / "concept_relations.json"
TOPIC_REVIEWS_PATH = REVIEWS_DIR / "topic_reviews.json"
JUDGMENTS_PATH = JUDGMENTS_DIR / "judgments.json"
THEORIES_PATH = THEORIES_DIR / "theories.json"
THINKING_FRAMEWORKS_PATH = FRAMEWORKS_DIR / "thinking_frameworks.json"
EXPRESSION_TEMPLATES_PATH = FRAMEWORKS_DIR / "expression_templates.json"

ACTIVE_ARTICLE_STATE = "active"
INACTIVE_ARTICLE_STATES = {"archived", "excluded"}
VALID_ARTICLE_STATES = {ACTIVE_ARTICLE_STATE, *INACTIVE_ARTICLE_STATES}


CATEGORY_DEFINITIONS: list[dict[str, Any]] = [
    {
        "id": "ai_agent",
        "label": "AI Agent",
        "description": "Agent 行为、工作流、运行时、工具调用、自主性和长任务系统。",
        "keywords": [
            "agent",
            "agents",
            "autonomous",
            "workflow",
            "harness",
            "session",
            "tool use",
            "browser use",
            "multi-agent",
            "claude code",
            "coding agent",
            "智能体",
            "多智能体",
            "工作流",
            "工具调用",
            "自主任务",
        ],
        "default_tags": ["agent-runtime", "workflow-orchestration"],
    },
    {
        "id": "ai_infrastructure",
        "label": "AI 基础设施",
        "description": "模型服务、编排、模型运维、沙箱、运行时系统和平台层。",
        "keywords": [
            "infrastructure",
            "runtime",
            "sandbox",
            "serving",
            "deployment",
            "api",
            "latency",
            "provider",
            "platform",
            "orchestration",
            "基础设施",
            "运行时",
            "沙箱",
            "部署",
            "平台层",
        ],
        "default_tags": ["runtime-layer", "infrastructure"],
    },
    {
        "id": "memory_context",
        "label": "记忆 / Context Engineering",
        "description": "记忆系统、上下文压缩、检索、图记忆和长上下文设计。",
        "keywords": [
            "memory",
            "context",
            "retrieval",
            "graph",
            "knowledge graph",
            "rag",
            "long context",
            "token",
            "compression",
            "session state",
            "记忆",
            "上下文",
            "知识库",
            "检索",
            "压缩",
            "长上下文",
        ],
        "default_tags": ["long-term-memory", "context-engineering"],
    },
    {
        "id": "developer_tools",
        "label": "开发者工具",
        "description": "软件工程工具、编码助手、测试、代码仓库和开发工作流。",
        "keywords": [
            "developer",
            "coding",
            "software engineering",
            "github",
            "repo",
            "repository",
            "testing",
            "ci",
            "code",
            "debug",
            "开发者",
            "程序员",
            "代码",
            "编程",
            "软件工程",
            "测试",
        ],
        "default_tags": ["developer-experience", "software-engineering"],
    },
    {
        "id": "multimodal_ai",
        "label": "多模态 AI",
        "description": "视觉、语音、音频、视频、图像和跨模态模型系统。",
        "keywords": [
            "multimodal",
            "vision",
            "image",
            "video",
            "audio",
            "speech",
            "tts",
            "voice",
            "clip",
            "text-to-image",
            "多模态",
            "图像",
            "视频",
            "语音",
            "音频",
            "声音",
            "视觉",
        ],
        "default_tags": ["multimodal", "cross-modal"],
    },
    {
        "id": "robotics_embodied_ai",
        "label": "机器人 / 具身智能",
        "description": "机器人、具身智能、触觉 grounding、传感器和物理世界 agent。",
        "keywords": [
            "robot",
            "robotics",
            "embodied",
            "tactile",
            "touch",
            "egocentric",
            "sensor",
            "physical",
            "grounding",
            "机器人",
            "具身",
            "触觉",
            "传感器",
            "物理世界",
        ],
        "default_tags": ["embodied-ai", "physical-grounding"],
    },
    {
        "id": "consumer_economy",
        "label": "消费经济",
        "description": "消费市场、产品、文化、怀旧、品牌和商业行为。",
        "keywords": [
            "consumer",
            "market",
            "brand",
            "nostalgia",
            "economy",
            "user",
            "users",
            "commerce",
            "retail",
            "platform",
            "消费",
            "市场",
            "品牌",
            "怀旧",
            "复古",
            "二手",
            "用户",
            "电商",
            "文旅",
            "经济",
        ],
        "default_tags": ["consumer-behavior", "market-signal"],
    },
    {
        "id": "research_benchmark",
        "label": "研究 / Benchmark",
        "description": "论文、数据集、benchmark、实验、学术结果和模型评测。",
        "keywords": [
            "paper",
            "arxiv",
            "benchmark",
            "dataset",
            "evaluation",
            "accuracy",
            "experiment",
            "cvpr",
            "openreview",
            "leaderboard",
            "论文",
            "数据集",
            "评测",
            "实验",
            "准确率",
            "基准",
        ],
        "default_tags": ["benchmark", "research-result"],
    },
    {
        "id": "open_source_ecosystem",
        "label": "开源生态",
        "description": "开源项目、代码仓库、维护者、stars、版本发布和社区采用。",
        "keywords": [
            "open source",
            "open-source",
            "github",
            "stars",
            "repository",
            "license",
            "maintainer",
            "release",
            "community",
            "开源",
            "仓库",
            "社区",
            "维护者",
            "发布",
        ],
        "default_tags": ["open-source", "ecosystem"],
    },
    {
        "id": "policy_regulation",
        "label": "政策 / 监管",
        "description": "政策、监管、法律约束、治理、安全规则和机构决策。",
        "keywords": [
            "policy",
            "regulation",
            "law",
            "legal",
            "compliance",
            "ban",
        ],
        "default_tags": ["policy", "governance"],
    },
    {
        "id": "product_business",
        "label": "产品 / 商业",
        "description": "产品发布、商业模式、变现、公司战略和市场定位。",
        "keywords": [
            "product",
            "launch",
            "pricing",
            "business",
            "company",
            "startup",
            "revenue",
            "customer",
            "enterprise",
            "commercial",
            "产品",
            "发布",
            "商业模式",
            "公司",
            "创业",
            "收入",
            "客户",
            "企业",
            "定价",
        ],
        "default_tags": ["product-launch", "business-model"],
    },
]


TAG_KEYWORDS: dict[str, list[str]] = {
    "agent-runtime": ["agent", "runtime", "harness", "session", "sandbox", "智能体", "运行时", "沙箱"],
    "workflow-orchestration": ["workflow", "orchestration", "multi-agent", "pipeline", "工作流", "编排", "多智能体"],
    "long-term-memory": ["memory", "long-term", "session state", "graph memory", "记忆", "长期记忆", "图记忆"],
    "context-engineering": ["context", "token", "compression", "retrieval", "上下文", "token", "压缩", "检索"],
    "developer-experience": ["developer", "coding", "debug", "ide", "github", "开发者", "编程", "调试", "程序员"],
    "software-engineering": ["software engineering", "testing", "ci", "repository", "软件工程", "测试", "代码仓库"],
    "multimodal": ["multimodal", "vision", "image", "video", "audio", "多模态", "视觉", "图像", "视频", "音频"],
    "speech-generation": ["speech", "voice", "tts", "audio", "语音", "声音", "音色", "音频"],
    "embodied-ai": ["robot", "robotics", "embodied", "egocentric", "机器人", "具身", "第一人称"],
    "physical-grounding": ["tactile", "touch", "sensor", "grounding", "触觉", "传感器", "物理", "grounding"],
    "consumer-behavior": ["consumer", "user", "market", "brand", "消费", "用户", "品牌", "怀旧", "复古"],
    "market-signal": ["market", "growth", "economy", "revenue", "市场", "增长", "经济", "收入", "规模"],
    "benchmark": ["benchmark", "evaluation", "accuracy", "leaderboard", "评测", "准确率", "排行榜", "基准"],
    "research-result": ["paper", "arxiv", "dataset", "experiment", "cvpr", "论文", "数据集", "实验", "研究"],
    "open-source": ["open source", "open-source", "github", "repository", "stars", "开源", "仓库", "星标"],
    "ecosystem": ["community", "maintainer", "release", "ecosystem", "社区", "维护者", "发布", "生态"],
    "policy": ["policy", "regulation", "law", "legal", "政策", "监管", "法律", "合规"],
    "governance": ["governance", "safety", "compliance", "standard", "治理", "安全", "标准"],
    "product-launch": ["launch", "released", "product", "service", "发布", "产品", "服务", "上线"],
    "business-model": ["pricing", "business", "revenue", "commercial", "enterprise", "定价", "商业", "收入", "企业"],
}


CONCEPT_SEEDS: dict[str, dict[str, Any]] = {
    "agent-runtime": {
        "name": "Agent Runtime",
        "definition": "面向 agent 的持久执行层，包括 session、工具、沙箱、恢复机制和审计轨迹。",
    },
    "workflow-orchestration": {
        "name": "Workflow Orchestration",
        "definition": "组织工具、模型、agent 和状态之间多步骤工作的协调层。",
    },
    "long-term-memory": {
        "name": "Long-Term Memory",
        "definition": "让 AI 系统能跨任务或 session 保存、检索有用状态的外部化记忆。",
    },
    "context-engineering": {
        "name": "Context Engineering",
        "definition": "设计哪些信息进入模型上下文、何时检索、如何压缩和组织。",
    },
    "developer-experience": {
        "name": "Developer Experience",
        "definition": "工具如何降低构建、测试、调试和维护软件的摩擦。",
    },
    "software-engineering": {
        "name": "Software Engineering Shift",
        "definition": "软件规划、实现、测试、评审和运行方式的变化。",
    },
    "multimodal": {
        "name": "Multimodal AI",
        "definition": "在文本、图像、视频、音频、语音或其他模态之间组合、对齐或转换的 AI 系统。",
    },
    "speech-generation": {
        "name": "Speech Generation",
        "definition": "生成语音、声音、音频场景或可控口语内容的系统。",
    },
    "embodied-ai": {
        "name": "Embodied AI",
        "definition": "植根于物理世界行动、感知、机器人或第一人称交互的 AI 系统。",
    },
    "physical-grounding": {
        "name": "Physical Grounding",
        "definition": "把模型行为连接到物理信号、传感器、触觉或动作的证据与机制。",
    },
    "consumer-behavior": {
        "name": "Consumer Behavior",
        "definition": "用户、消费者或文化群体采用产品、分配注意力和支出的行为模式。",
    },
    "market-signal": {
        "name": "Market Signal",
        "definition": "显示某个市场、品类或用户行为可能变化的可测量或反复出现的信号。",
    },
    "benchmark": {
        "name": "Benchmark Evidence",
        "definition": "用于比较系统的评测结果，通常需要仔细核对来源、任务设置和实验条件。",
    },
    "research-result": {
        "name": "Research Result",
        "definition": "可能支撑更大技术判断的论文、数据集、实验或学术发现。",
    },
    "open-source": {
        "name": "Open Source Signal",
        "definition": "来自公开仓库、发布、采用、贡献者或可审查实现的证据。",
    },
    "ecosystem": {
        "name": "Ecosystem Formation",
        "definition": "围绕一个主题形成的项目、用户、维护者、服务和实践群。",
    },
    "policy": {
        "name": "Policy Constraint",
        "definition": "改变行动者能构建、购买或部署什么的法律、监管或机构规则。",
    },
    "governance": {
        "name": "Governance Layer",
        "definition": "塑造系统行为和采用路径的规则、审查流程、安全控制或机构安排。",
    },
    "product-launch": {
        "name": "Product Launch",
        "definition": "让某种能力以产品、服务或明确包装触达用户的公开发布事件。",
    },
    "business-model": {
        "name": "Business Model",
        "definition": "产品、公司或生态如何捕获价值并组织经济激励。",
    },
}


def ensure_dirs() -> None:
    for directory in (
        TAXONOMY_DIR,
        CONCEPTS_DIR,
        REVIEWS_DIR,
        JUDGMENTS_DIR,
        THEORIES_DIR,
        FRAMEWORKS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def now() -> str:
    return br.utc_now()


def article_extraction_path(article_id: str) -> Path:
    return br.article_dir(article_id) / "extraction.json"


def article_classification_path(article_id: str) -> Path:
    return br.article_dir(article_id) / "classification.json"


def article_concepts_path(article_id: str) -> Path:
    return br.article_dir(article_id) / "concepts.json"


def write_if_missing(path: Path, payload: dict[str, Any], force: bool = False) -> bool:
    if path.exists() and not force:
        return False
    br.write_json(path, payload)
    return True


def init_state(force: bool = False) -> dict[str, Any]:
    ensure_dirs()
    created: list[str] = []

    categories_path = TAXONOMY_DIR / "categories.json"
    tags_path = TAXONOMY_DIR / "tags.json"

    if write_if_missing(
        categories_path,
        {
            "created_at": now(),
            "last_updated_at": now(),
            "version": 1,
            "categories": CATEGORY_DEFINITIONS,
            "governance": {
                "primary_category": "每篇文章使用一个稳定主分类，作为综述分组依据。",
                "secondary_categories": "副分类用于表达强重叠关系，不重复生成综述。",
                "changes": "分类重命名、拆分、合并应通过明确的人审步骤完成。",
            },
        },
        force=force,
    ):
        created.append(br.relpath_from_root(categories_path))

    if write_if_missing(
        tags_path,
        {
            "created_at": now(),
            "last_updated_at": now(),
            "version": 1,
            "tags": [
                {
                    "id": tag,
                    "label": tag.replace("-", " ").title(),
                    "keywords": keywords,
                    "status": "active",
                }
                for tag, keywords in sorted(TAG_KEYWORDS.items())
            ],
            "governance": {
                "rule": "标签主要是检索入口。只有稳定、反复出现的标签才应提升为概念。",
                "cleanup": "人审时合并别名，清理一次性措辞。",
            },
        },
        force=force,
    ):
        created.append(br.relpath_from_root(tags_path))

    if write_if_missing(
        GLOBAL_CONCEPTS_PATH,
        {
            "created_at": now(),
            "last_updated_at": now(),
            "version": 1,
            "concepts": [],
            "governance": {
                "statuses": ["drafted", "approved", "merged", "deprecated"],
                "rule": "概念是可复用的思考对象，不只是标签。",
            },
        },
        force=force,
    ):
        created.append(br.relpath_from_root(GLOBAL_CONCEPTS_PATH))

    if write_if_missing(
        CONCEPT_RELATIONS_PATH,
        {
            "created_at": now(),
            "last_updated_at": now(),
            "version": 1,
            "relations": [],
        },
        force=force,
    ):
        created.append(br.relpath_from_root(CONCEPT_RELATIONS_PATH))

    if write_if_missing(
        TOPIC_REVIEWS_PATH,
        {
            "created_at": now(),
            "last_updated_at": now(),
            "version": 1,
            "reviews": [],
            "review_policy": {
                "unit": "按主分类组织的动态主题综述。",
                "status_flow": ["drafted", "reviewed", "archived"],
            },
        },
        force=force,
    ):
        created.append(br.relpath_from_root(TOPIC_REVIEWS_PATH))

    if write_if_missing(
        JUDGMENTS_PATH,
        {
            "created_at": now(),
            "last_updated_at": now(),
            "version": 1,
            "judgments": [],
            "confidence_levels": ["emerging", "plausible", "strong", "settled", "contested"],
        },
        force=force,
    ):
        created.append(br.relpath_from_root(JUDGMENTS_PATH))

    if write_if_missing(
        THEORIES_PATH,
        {
            "created_at": now(),
            "last_updated_at": now(),
            "version": 1,
            "theories": [],
        },
        force=force,
    ):
        created.append(br.relpath_from_root(THEORIES_PATH))

    if write_if_missing(
        THINKING_FRAMEWORKS_PATH,
        {
            "created_at": now(),
            "last_updated_at": now(),
            "version": 1,
            "frameworks": [
                {
                    "framework_id": "from_article_to_abstraction",
                    "name": "从文章到抽象",
                    "purpose": "把单篇文章推进到概念、主题综述、判断和理论层。",
                    "dimensions": [
                        "这篇文章声称或呈现了什么？",
                        "它属于哪个稳定分类？",
                        "它触及哪些可复用概念？",
                        "它强化或削弱了哪个模式？",
                        "它可能改变哪一个更高层判断？",
                    ],
                    "status": "drafted",
                }
            ],
        },
        force=force,
    ):
        created.append(br.relpath_from_root(THINKING_FRAMEWORKS_PATH))

    if write_if_missing(
        EXPRESSION_TEMPLATES_PATH,
        {
            "created_at": now(),
            "last_updated_at": now(),
            "version": 1,
            "templates": [
                {
                    "template_id": "review_to_argument",
                    "name": "从综述到论点",
                    "structure": [
                        "现象",
                        "代表案例",
                        "重复模式",
                        "结构性解释",
                        "反向信号",
                        "我的判断",
                    ],
                    "status": "drafted",
                }
            ],
        },
        force=force,
    ):
        created.append(br.relpath_from_root(EXPRESSION_TEMPLATES_PATH))

    br.append_jsonl(
        br.CHANGE_LOG_PATH,
        {
            "timestamp": now(),
            "event": "knowledge_init",
            "created": created,
            "force": force,
        },
    )
    return {"ok": True, "created": created}


def reset_global_concepts() -> None:
    """Reset generated concept index while preserving the schema contract."""
    init_state()
    existing = br.read_json(GLOBAL_CONCEPTS_PATH, default={})
    br.write_json(
        GLOBAL_CONCEPTS_PATH,
        {
            "created_at": existing.get("created_at") or now()
            if isinstance(existing, dict)
            else now(),
            "last_updated_at": now(),
            "version": existing.get("version", 1) if isinstance(existing, dict) else 1,
            "concepts": [],
            "governance": (
                existing.get("governance")
                if isinstance(existing, dict) and isinstance(existing.get("governance"), dict)
                else {
                    "statuses": ["drafted", "approved", "merged", "deprecated"],
                    "rule": "概念是可复用的思考对象，不只是标签。",
                }
            ),
        },
    )


def compact_text(value: str, limit: int = 5000) -> str:
    text = re.sub(r"\s+", " ", value or "").strip()
    return text[:limit]


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def localize_article_text(text: str) -> str:
    """Keep source terms, but prefer Chinese framing in generated views."""
    value = text or ""
    replacements = [
        ("The article states that ", "文章称，"),
        ("The article claims that ", "文章称，"),
        ("The article says that ", "文章称，"),
        ("The paper states that ", "论文称，"),
        ("The paper claims that ", "论文称，"),
        ("According to the article, ", "据文章称，"),
    ]
    for source, target in replacements:
        value = value.replace(source, target)
    return value


def sentence_candidates(text: str, limit: int = 8) -> list[str]:
    raw = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    sentences: list[str] = []
    for item in raw:
        cleaned = re.sub(r"\s+", " ", item).strip()
        if len(cleaned) < 20:
            continue
        if cleaned in sentences:
            continue
        sentences.append(localize_article_text(cleaned[:280]))
        if len(sentences) >= limit:
            break
    return sentences


def category_definition(category_id: str | None) -> dict[str, Any]:
    for category in CATEGORY_DEFINITIONS:
        if category.get("id") == category_id:
            return category
    return {}


def review_keywords(category_id: str | None, tags: list[str]) -> list[str]:
    keywords: list[str] = []
    category = category_definition(category_id)
    for item in category.get("keywords", []):
        if isinstance(item, str):
            keywords.append(item)
    for tag in tags:
        keywords.extend(TAG_KEYWORDS.get(tag, []))
        keywords.extend(part for part in re.split(r"[-_]", tag) if len(part) > 2)
    seen: set[str] = set()
    unique: list[str] = []
    for keyword in keywords:
        normalized = keyword.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(keyword.strip())
    return unique


def split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?。！？；;])\s+|\n+", text or "")
    sentences: list[str] = []
    for item in raw:
        cleaned = re.sub(r"\s+", " ", item).strip()
        if 28 <= len(cleaned) <= 360:
            sentences.append(cleaned)
    return sentences


def select_evidence_sentences(text: str, keywords: list[str], limit: int = 4) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return sentence_candidates(text, limit=limit)
    scored: list[tuple[int, int, str]] = []
    lowered_keywords = [keyword.lower() for keyword in keywords if keyword.strip()]
    for index, sentence in enumerate(sentences):
        lower = sentence.lower()
        score = 0
        for keyword in lowered_keywords:
            if keyword and keyword in lower:
                score += 4
        if re.search(r"\d", sentence):
            score += 2
        if any(mark in sentence for mark in ["因此", "说明", "意味着", "关键", "核心", "不是", "而是", "增长", "成本", "问题"]):
            score += 2
        if "点击" in sentence or "关注" in sentence or "预览时标签" in sentence:
            score -= 6
        if "http://" in lower or "https://" in lower:
            score -= 6
        if score > 0:
            scored.append((score, -index, sentence))
    scored.sort(reverse=True)
    selected: list[str] = []
    for _score, _neg_index, sentence in scored:
        if sentence not in selected:
            selected.append(localize_article_text(sentence))
        if len(selected) >= limit:
            break
    if selected:
        return selected
    return sentence_candidates(text, limit=limit)


def load_article_text(article_id: str) -> str:
    canonical = br.article_dir(article_id) / "canonical_text.txt"
    if not canonical.exists():
        return ""
    return canonical.read_text(encoding="utf-8", errors="replace")


def claim_texts(article_id: str) -> list[dict[str, Any]]:
    claims_doc = br.read_json(
        br.claims_path(article_id),
        default={"claim_extraction_status": "not_started", "claims": []},
    )
    if not isinstance(claims_doc, dict):
        return []
    claims = claims_doc.get("claims", [])
    if not isinstance(claims, list):
        return []
    return [claim for claim in claims if isinstance(claim, dict)]


def extract_entities(text: str, title: str, limit: int = 20) -> list[str]:
    combined = f"{title}\n{text}"
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9]*(?:[- ][A-Z0-9][A-Za-z0-9]*){0,4}\b", combined)
    noisy = {"The", "This", "That", "And", "For", "With", "From", "JSON", "URL"}
    counter: Counter[str] = Counter()
    for candidate in candidates:
        cleaned = candidate.strip()
        if cleaned in noisy or len(cleaned) < 2:
            continue
        counter[cleaned] += 1
    return [item for item, _count in counter.most_common(limit)]


def extract_article(article_id: str, force: bool = False) -> dict[str, Any]:
    init_state()
    record = br.read_json(br.article_record_path(article_id), default=None)
    if record is None:
        raise ValueError(f"Unknown article_id: {article_id}")

    output_path = article_extraction_path(article_id)
    if output_path.exists() and not force:
        return {
            "ok": True,
            "article_id": article_id,
            "skipped": True,
            "path": br.relpath_from_root(output_path),
        }

    canonical_text = load_article_text(article_id)
    claims = claim_texts(article_id)
    summary_doc = record.get("article_summary", {}) if isinstance(record, dict) else {}
    article_summary_items: list[str] = []
    if isinstance(summary_doc, dict):
        for key in ("events", "techniques", "tools"):
            value = summary_doc.get(key)
            if isinstance(value, list):
                article_summary_items.extend(str(item) for item in value if item)

    if article_summary_items:
        main_points = [localize_article_text(item) for item in article_summary_items[:8]]
    elif claims:
        main_points = [
            localize_article_text(str(claim.get("text", "")).strip())
            for claim in claims[:8]
            if claim.get("text")
        ]
    else:
        main_points = sentence_candidates(canonical_text, limit=8)

    summary = " ".join(main_points[:3]).strip()
    if not summary:
        summary = compact_text(canonical_text, limit=500)

    structured_claims = []
    if claims:
        for claim in claims:
            structured_claims.append(
                {
                    "id": claim.get("id"),
                    "type": claim.get("type"),
                    "text": localize_article_text(str(claim.get("text") or "")),
                    "source": "legacy_claims_json",
                }
            )
    else:
        for index, sentence in enumerate(sentence_candidates(canonical_text, limit=6), start=1):
            structured_claims.append(
                {
                    "id": f"auto_point_{index}",
                    "type": "event",
                    "text": sentence,
                    "source": "canonical_text_heuristic",
                }
            )

    payload = {
        "article_id": article_id,
        "extraction_status": "completed" if canonical_text.strip() or structured_claims else "needs_text",
        "generated_at": now(),
        "method": "deterministic_mvp",
        "summary": summary,
        "main_points": main_points,
        "entities": extract_entities(canonical_text, str(record.get("title") or "")),
        "claims": structured_claims,
        "important_quotes": [],
        "open_questions": [],
        "source_trace": {
            "record_path": br.relpath_from_root(br.article_record_path(article_id)),
            "canonical_text_path": br.relpath_from_root(br.article_dir(article_id) / "canonical_text.txt"),
            "legacy_claims_path": br.relpath_from_root(br.claims_path(article_id)),
        },
    }
    br.write_json(output_path, payload)
    br.append_jsonl(
        br.CHANGE_LOG_PATH,
        {
            "timestamp": now(),
            "event": "knowledge_extract_article",
            "article_id": article_id,
            "claim_count": len(structured_claims),
            "main_point_count": len(main_points),
        },
    )
    return {
        "ok": True,
        "article_id": article_id,
        "path": br.relpath_from_root(output_path),
        "claim_count": len(structured_claims),
    }


def score_keywords(text: str, keywords: list[str]) -> int:
    text_lower = text.lower()
    score = 0
    for keyword in keywords:
        keyword_lower = keyword.lower()
        if not keyword_lower:
            continue
        if re.fullmatch(r"[a-z0-9][a-z0-9 \-]*", keyword_lower):
            pattern = (
                r"(?<![a-z0-9])"
                + re.escape(keyword_lower).replace(r"\ ", r"\s+")
                + r"(?![a-z0-9])"
            )
            occurrences = len(re.findall(pattern, text_lower))
        else:
            occurrences = text_lower.count(keyword_lower)
        if occurrences:
            score += occurrences
    return score


def infer_source_type(url: str, text: str) -> str:
    lower_url = (url or "").lower()
    lower_text = text.lower()
    if any(token in lower_url for token in ("arxiv.org", "openreview.net", "aclanthology.org", "doi.org")):
        return "paper"
    if "github.com" in lower_url:
        return "repository"
    if "mp.weixin.qq.com" in lower_url:
        return "media"
    if any(token in lower_url for token in ("docs.", "/docs", "readthedocs")):
        return "docs"
    if any(token in lower_text for token in ("paper", "benchmark", "dataset", "experiment")):
        return "research"
    return "unknown"


def category_options_for_llm() -> list[dict[str, Any]]:
    return [
        {
            "id": category["id"],
            "label": category.get("label", category["id"]),
            "description": category.get("description", ""),
            "default_tags": category.get("default_tags", []),
        }
        for category in CATEGORY_DEFINITIONS
    ]


def normalize_category_id(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    valid = {category["id"] for category in CATEGORY_DEFINITIONS}
    if raw in valid:
        return raw
    lowered = raw.lower()
    for category in CATEGORY_DEFINITIONS:
        if lowered == str(category.get("label", "")).lower():
            return category["id"]
    return None


def normalize_tag(value: Any) -> str | None:
    tag = str(value or "").strip().lower()
    if not tag:
        return None
    tag = re.sub(r"[^a-z0-9]+", "-", tag).strip("-")
    return tag or None


def normalize_llm_classification(
    raw: Any,
    *,
    fallback: dict[str, Any],
    article_id: str,
    model: str,
    attempts: int,
    category_scores: dict[str, int],
) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    primary = normalize_category_id(raw.get("primary_category"))
    if primary is None:
        return None

    secondary: list[str] = []
    raw_secondary = raw.get("secondary_categories", [])
    if isinstance(raw_secondary, list):
        for item in raw_secondary:
            category_id = normalize_category_id(item)
            if category_id and category_id != primary and category_id not in secondary:
                secondary.append(category_id)
            if len(secondary) >= 3:
                break

    tags: list[str] = []
    raw_tags = raw.get("tags", [])
    if isinstance(raw_tags, list):
        for item in raw_tags:
            tag = normalize_tag(item)
            if tag and tag not in tags:
                tags.append(tag)
            if len(tags) >= 12:
                break
    if not tags:
        tags = list(fallback.get("tags", []))[:8]

    source_type = str(raw.get("source_type") or "").strip().lower()
    if source_type not in {"paper", "repository", "docs", "media", "research", "product", "market", "unknown"}:
        source_type = str(fallback.get("source_type") or "unknown")

    importance = str(raw.get("importance") or "").strip().lower()
    if importance not in {"high", "medium", "low"}:
        importance = str(fallback.get("importance") or "medium")

    rationale = compact_text(str(raw.get("rationale") or ""), limit=700)
    if not rationale:
        rationale = "LLM 根据全文摘录、抽取要点和分类体系生成的分类草稿。"

    return {
        "article_id": article_id,
        "classification_status": "drafted",
        "generated_at": now(),
        "method": "llm_openrouter",
        "llm_model": model,
        "llm_attempts": attempts,
        "primary_category": primary,
        "secondary_categories": secondary,
        "tags": tags,
        "source_type": source_type,
        "importance": importance,
        "rationale": rationale,
        "category_scores": category_scores,
        "fallback_suggestion": {
            "primary_category": fallback.get("primary_category"),
            "secondary_categories": fallback.get("secondary_categories", []),
            "tags": fallback.get("tags", []),
        },
    }


def classify_article_with_llm(
    article_id: str,
    record: dict[str, Any],
    extraction: dict[str, Any],
    canonical_text: str,
    fallback: dict[str, Any],
    category_scores: dict[str, int],
) -> dict[str, Any] | None:
    try:
        from llm_client import complete_json, load_project_env
    except SystemExit:
        return None

    load_project_env()
    if not os.environ.get("OPENROUTER_API_KEY"):
        return None

    claims = []
    for claim in extraction.get("claims", []):
        if isinstance(claim, dict) and claim.get("text"):
            claims.append(str(claim.get("text")))
        if len(claims) >= 12:
            break

    system = """
你是一个中文知识系统的文章分类器。你要把单篇文章放入一个主分类，让同类文章聚合在一起，服务后续深度综述。

只输出 JSON，不要输出 Markdown。Schema:
{
  "primary_category": "必须是给定分类 id 之一，且只能有一个",
  "secondary_categories": ["可选，最多 3 个分类 id"],
  "tags": ["英文 kebab-case 标签，保留 AI Agent、LLM、GPU 等必要英文概念"],
  "source_type": "paper | repository | docs | media | research | product | market | unknown",
  "importance": "high | medium | low",
  "rationale": "中文说明，解释为什么归入该主分类，不要提内部 article_id"
}

分类原则：
1. 主分类看文章讨论的核心机制，而不是出现频率最高的词。
2. 不确定时选择最能支撑未来综述比较的分类。
3. 标签要描述可复用证据线索，不要写泛泛的 news、article、ai。
""".strip()

    user = json.dumps(
        {
            "categories": category_options_for_llm(),
            "deterministic_suggestion": fallback,
            "article": {
                "title": record.get("title"),
                "url": record.get("url"),
                "summary": extraction.get("summary"),
                "main_points": extraction.get("main_points", []),
                "claims": claims,
                "canonical_excerpt": compact_text(canonical_text, limit=7000),
            },
        },
        ensure_ascii=False,
        indent=2,
    )

    try:
        result = complete_json(system, user, temperature=0.1, max_tokens=1800)
    except Exception as exc:
        br.append_jsonl(
            br.CHANGE_LOG_PATH,
            {
                "timestamp": now(),
                "event": "knowledge_classify_article_llm_failed",
                "article_id": article_id,
                "error": compact_text(str(exc), limit=500),
            },
        )
        return None

    return normalize_llm_classification(
        result.data,
        fallback=fallback,
        article_id=article_id,
        model=result.model,
        attempts=result.attempts,
        category_scores=category_scores,
    )


def classify_article(article_id: str, force: bool = False) -> dict[str, Any]:
    init_state()
    record = br.read_json(br.article_record_path(article_id), default=None)
    if record is None:
        raise ValueError(f"Unknown article_id: {article_id}")

    extraction_result = extract_article(article_id, force=False)
    output_path = article_classification_path(article_id)
    if output_path.exists() and not force:
        return {
            "ok": True,
            "article_id": article_id,
            "skipped": True,
            "path": br.relpath_from_root(output_path),
            "extraction": extraction_result,
        }

    extraction = br.read_json(article_extraction_path(article_id), default={})
    canonical_text = load_article_text(article_id)
    claims_joined = " ".join(
        str(item.get("text") or "")
        for item in extraction.get("claims", [])
        if isinstance(item, dict)
    )
    text = " ".join(
        [
            str(record.get("title") or ""),
            str(extraction.get("summary") or ""),
            claims_joined,
            compact_text(canonical_text, limit=4000),
        ]
    )

    category_scores = {
        category["id"]: score_keywords(text, category.get("keywords", []))
        for category in CATEGORY_DEFINITIONS
    }
    sorted_categories = sorted(
        CATEGORY_DEFINITIONS,
        key=lambda item: (category_scores[item["id"]], item["id"]),
        reverse=True,
    )
    primary = sorted_categories[0]["id"] if sorted_categories else "product_business"
    if category_scores.get(primary, 0) == 0:
        primary = "product_business"

    primary_score = category_scores.get(primary, 0)
    secondary_threshold = max(5, int(primary_score * 0.25))
    secondary = [
        category["id"]
        for category in sorted_categories
        if category["id"] != primary
        and category_scores.get(category["id"], 0) >= secondary_threshold
    ][:3]

    tags: set[str] = set()
    scoped_tags: set[str] = set()
    for category in CATEGORY_DEFINITIONS:
        if category["id"] == primary or category["id"] in secondary:
            scoped_tags.update(category.get("default_tags", []))
    tags.update(scoped_tags)
    for tag, keywords in TAG_KEYWORDS.items():
        tag_score = score_keywords(text, keywords)
        if tag in scoped_tags and tag_score >= 1:
            tags.add(tag)
        elif tag_score >= max(6, int(primary_score * 0.15)):
            tags.add(tag)

    if not tags:
        tags.add("market-signal")

    source_type = infer_source_type(str(record.get("url") or ""), text)
    claim_count = len(extraction.get("claims", [])) if isinstance(extraction, dict) else 0
    importance = "high" if claim_count >= 6 else "medium" if claim_count >= 3 else "low"

    fallback_payload = {
        "article_id": article_id,
        "classification_status": "drafted",
        "generated_at": now(),
        "method": "deterministic_keyword_mvp",
        "primary_category": primary,
        "secondary_categories": secondary,
        "tags": sorted(tags),
        "source_type": source_type,
        "importance": importance,
        "rationale": (
            "关键词规则生成的分类草稿。需要人审；后续可由 LLM 分类替换，但保留相同 schema。"
        ),
        "category_scores": category_scores,
    }
    payload = classify_article_with_llm(
        article_id=article_id,
        record=record,
        extraction=extraction,
        canonical_text=canonical_text,
        fallback=fallback_payload,
        category_scores=category_scores,
    ) or fallback_payload

    br.write_json(output_path, payload)
    br.append_jsonl(
        br.CHANGE_LOG_PATH,
        {
            "timestamp": now(),
            "event": "knowledge_classify_article",
            "article_id": article_id,
            "primary_category": primary,
            "tags": sorted(tags),
        },
    )
    return {
        "ok": True,
        "article_id": article_id,
        "path": br.relpath_from_root(output_path),
        "primary_category": primary,
        "tags": sorted(tags),
    }


def concept_id_from_tag(tag: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", tag.lower()).strip("_")


def default_concept_for_tag(tag: str) -> dict[str, Any]:
    seed = CONCEPT_SEEDS.get(tag, {})
    concept_id = concept_id_from_tag(tag)
    return {
        "concept_id": concept_id,
        "name": seed.get("name") or tag.replace("-", " ").title(),
        "definition": seed.get("definition")
        or "从反复出现的文章标签中生成的草稿概念。",
        "aliases": [tag],
        "status": "drafted",
        "created_at": now(),
        "last_updated_at": now(),
        "related_categories": [],
        "related_tags": [tag],
        "supporting_articles": [],
        "notes": [],
    }


def update_concepts(article_id: str, force: bool = False) -> dict[str, Any]:
    init_state()
    classify_result = classify_article(article_id, force=False)
    output_path = article_concepts_path(article_id)
    if output_path.exists() and not force:
        return {
            "ok": True,
            "article_id": article_id,
            "skipped": True,
            "path": br.relpath_from_root(output_path),
            "classification": classify_result,
        }

    classification = br.read_json(article_classification_path(article_id), default={})
    extraction = br.read_json(article_extraction_path(article_id), default={})
    tags = [
        tag
        for tag in classification.get("tags", [])
        if isinstance(tag, str) and tag.strip()
    ]
    primary_category = classification.get("primary_category")
    secondary_categories = classification.get("secondary_categories", [])
    if not isinstance(secondary_categories, list):
        secondary_categories = []
    categories = [
        item
        for item in [primary_category, *secondary_categories]
        if isinstance(item, str) and item
    ]

    global_doc = br.read_json(GLOBAL_CONCEPTS_PATH, default={"concepts": []})
    concepts = global_doc.setdefault("concepts", [])
    by_id = {
        item.get("concept_id"): item
        for item in concepts
        if isinstance(item, dict) and item.get("concept_id")
    }

    matched_concepts: list[str] = []
    created_concepts: list[str] = []
    for tag in tags:
        concept_id = concept_id_from_tag(tag)
        concept = by_id.get(concept_id)
        if concept is None:
            concept = default_concept_for_tag(tag)
            concepts.append(concept)
            by_id[concept_id] = concept
            created_concepts.append(concept_id)

        for category in categories:
            if category not in concept.setdefault("related_categories", []):
                concept["related_categories"].append(category)
        if tag not in concept.setdefault("related_tags", []):
            concept["related_tags"].append(tag)
        article_refs = concept.setdefault("supporting_articles", [])
        if article_id not in article_refs:
            article_refs.append(article_id)
        concept["last_updated_at"] = now()
        matched_concepts.append(concept_id)

    global_doc["last_updated_at"] = now()
    br.write_json(GLOBAL_CONCEPTS_PATH, global_doc)

    article_doc = {
        "article_id": article_id,
        "concept_status": "drafted",
        "generated_at": now(),
        "method": "tag_to_concept_mvp",
        "matched_concepts": sorted(set(matched_concepts)),
        "created_concepts": created_concepts,
        "candidate_concepts": [],
        "source_trace": {
            "classification_path": br.relpath_from_root(article_classification_path(article_id)),
            "extraction_path": br.relpath_from_root(article_extraction_path(article_id)),
        },
        "article_entities": extraction.get("entities", []) if isinstance(extraction, dict) else [],
    }
    br.write_json(output_path, article_doc)
    br.append_jsonl(
        br.CHANGE_LOG_PATH,
        {
            "timestamp": now(),
            "event": "knowledge_update_concepts",
            "article_id": article_id,
            "matched_concepts": sorted(set(matched_concepts)),
            "created_concepts": created_concepts,
        },
    )
    return {
        "ok": True,
        "article_id": article_id,
        "path": br.relpath_from_root(output_path),
        "matched_concepts": sorted(set(matched_concepts)),
        "created_concepts": created_concepts,
    }


def load_ready_articles() -> list[dict[str, Any]]:
    records = br.load_all_article_records()
    ready: list[dict[str, Any]] = []
    for record in records:
        article_id = record.get("article_id")
        if not isinstance(article_id, str):
            continue
        if not article_is_active(record):
            continue
        if not article_extraction_path(article_id).exists():
            continue
        if not article_classification_path(article_id).exists():
            continue
        ready.append(record)
    return ready


def article_lifecycle_state(record: dict[str, Any] | None) -> str:
    if not isinstance(record, dict):
        return ACTIVE_ARTICLE_STATE
    state = record.get("lifecycle_state")
    if not isinstance(state, str) or not state.strip():
        return ACTIVE_ARTICLE_STATE
    return state.strip().lower()


def article_is_active(record: dict[str, Any] | None) -> bool:
    return article_lifecycle_state(record) == ACTIVE_ARTICLE_STATE


def issue_url_for_record(record: dict[str, Any]) -> str:
    for source in reversed(record.get("ingest_sources", [])):
        if not isinstance(source, dict):
            continue
        source_ref = source.get("source_ref")
        if isinstance(source_ref, str) and "/issues/" in source_ref:
            return source_ref.strip()
    return ""


def github_repo_url() -> str:
    config = br.read_github_config()
    repo = config.get("repo")
    if isinstance(repo, str) and repo.strip():
        return f"https://github.com/{repo.strip()}"
    return "https://github.com/plato3724/bayesian-llm-trend-report"


def article_management_issue_url(row: dict[str, Any]) -> str:
    article_id = str(row.get("article_id") or "")
    title = str(row.get("title") or article_id)
    source_url = str(row.get("url") or "")
    body = (
        "这是一篇早期导入文章的管理 issue。它没有原始 GitHub issue，因此从知识页新建此管理入口。\n\n"
        f"- 文章 ID: `{article_id}`\n"
        f"- 标题: {title}\n"
        f"- 来源: {source_url}\n\n"
        "在下面评论其中一条命令即可触发知识系统更新：\n\n"
        f"```text\n/archive {article_id} 原因：\n/exclude {article_id} 原因：\n/restore {article_id} 原因：\n```"
    )
    query = urllib.parse.urlencode(
        {
            "template": "manage-article.md",
            "title": f"[Manage Article] {article_id}",
            "labels": "knowledge-management",
            "body": body,
        }
    )
    return f"{github_repo_url()}/issues/new?{query}"


def top_items(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    return [
        {"id": item, "count": count}
        for item, count in counter.most_common(limit)
    ]


def representative_articles(article_ids: list[str], limit: int = 5) -> list[dict[str, Any]]:
    reps: list[dict[str, Any]] = []
    for article_id in article_ids[:limit]:
        record = br.read_json(br.article_record_path(article_id), default={})
        extraction = br.read_json(article_extraction_path(article_id), default={})
        reps.append(
            {
                "article_id": article_id,
                "title": record.get("title"),
                "url": record.get("url"),
                "summary": extraction.get("summary"),
            }
        )
    return reps


def category_label(category_id: str | None) -> str:
    if not category_id:
        return "Uncategorized"
    for category in CATEGORY_DEFINITIONS:
        if category.get("id") == category_id:
            return str(category.get("label") or category_id)
    return category_id.replace("_", " ").title()


def category_description(category_id: str | None) -> str:
    if not category_id:
        return "尚未归入稳定主题的文章，会在后续分类后移动到对应分类。"
    for category in CATEGORY_DEFINITIONS:
        if category.get("id") == category_id:
            return str(category.get("description") or "")
    return "该分类下的文章会被合并进入同一条综述线索。"


PATTERN_INTERPRETATIONS: dict[str, str] = {
    "agent-runtime": "这组文章把 agent 能力视为运行时问题：持久 session、工具、状态和执行边界，与基础模型本身同样重要。",
    "workflow-orchestration": "反复出现的信号是编排：有用系统需要协调多步骤、多个行动者、工具或审查门，而不是依赖一次模型回答。",
    "context-engineering": "共同问题不只是上下文窗口变大，而是如何选择、压缩、检索和结构化 context。",
    "long-term-memory": "这些文章指向一种外部系统对象：memory 可以跨 session 查询、维护和审计。",
    "developer-experience": "材料显示，采用率取决于系统是否改变日常工作流，而不只是模型孤立能力是否足够强。",
    "software-engineering": "重复关切是：当实现变便宜以后，软件工程的重心会向测试、评审、编排和判断迁移。",
    "cross-modal": "共同线索是跨模态转换和对齐，价值来自不同输入输出形式的连接，而不是单一媒介的增强。",
    "multimodal": "主题正在从单点模型 demo 转向把图像、音频、视频、文本或传感数据组合起来的产品和研究系统。",
    "speech-generation": "信号是语音生成正在变成可控的场景/声音设计，而不仅是句子级 TTS。",
    "embodied-ai": "材料把物理交互视为独立智能问题：感知与行动本身就是系统边界的一部分。",
    "physical-grounding": "重复主张是，触觉、第一人称视频或传感器等物理信号可以提供文本无法提供的 grounding。",
    "benchmark": "该主题依赖测量型 claim，因此综述必须区分真实比较证据与厂商/文章层面的 benchmark 叙事。",
    "research-result": "证据基础偏研究，关键问题是论文和数据集能否转化为稳定产品或生态变化。",
    "open-source": "公开仓库让信号可检查、可复用，但热度仍需和真实能力、真实采用区分开。",
    "ecosystem": "主题已经不只是孤立项目；证据指向围绕某个方向形成项目、用户、维护者和实践群。",
    "product-launch": "证据来自产品包装：某种能力变重要，往往发生在它被包装成产品、服务或用户工作流之后。",
    "business-model": "综述应观察注意力和使用量是否能转化为可重复的价值捕获机制。",
    "infrastructure": "共同信号是：这个品类依赖可见应用之下的平台层或系统层。",
    "runtime-layer": "文章组指向 runtime 层：可靠性、状态、权限和可重复性都在这里被处理。",
}


def pattern_interpretation(pattern: str) -> str:
    return PATTERN_INTERPRETATIONS.get(
        pattern,
        "这个模式在文章组中反复出现，应作为潜在组织概念审查，而不是只当作松散标签。",
    )


def category_central_question(category_id: str, article_count: int) -> str:
    label = category_label(category_id)
    if article_count <= 1:
        return (
            f"这篇单独的「{label}」文章揭示了什么？还缺什么证据，才能形成稳定主题综述？"
        )
    return (
        f"在这 {article_count} 篇「{label}」文章中，超越单篇 claim 的重复结构是什么？"
    )


def category_implication(category_id: str, top_patterns: list[str]) -> str:
    label = category_label(category_id)
    pattern_text = ", ".join(top_patterns[:3]) if top_patterns else "the repeated concepts"
    return (
        f"如果「{label}」继续围绕 {pattern_text} 增长，后续分析应把它看作系统层变化，而不是孤立新闻。"
    )


def article_case_analysis(article_id: str, top_patterns: list[str]) -> dict[str, Any]:
    record = br.read_json(br.article_record_path(article_id), default={})
    extraction = br.read_json(article_extraction_path(article_id), default={})
    classification = br.read_json(article_classification_path(article_id), default={})
    tags = [
        tag
        for tag in classification.get("tags", [])
        if isinstance(tag, str)
    ]
    overlap = [tag for tag in top_patterns if tag in tags]
    role = "anchor_case" if overlap else "supporting_case"
    pattern_text = ", ".join(overlap[:3]) if overlap else ", ".join(tags[:3])
    summary = extraction.get("summary") or ""
    return {
        "article_id": article_id,
        "title": record.get("title"),
        "url": record.get("url"),
        "role": role,
        "what_it_shows": (
            f"这个案例为 {pattern_text or '该分类'} 提供了具体的文章级证据。"
        ),
        "limits": (
            "它不能单独构成结论；需要和本综述中的其他案例比较后才能判断其结构性意义。"
        ),
        "summary": summary,
    }


DEEP_REVIEW_FRAMES: dict[str, dict[str, str]] = {
    "ai_agent": {
        "object": "Agent 系统",
        "surface": "模型能否完成单次任务",
        "depth": "任务如何被拆解、编排、校验并在失败后恢复",
        "shift": "竞争重心正在从回答质量转向可运行的工作流能力",
    },
    "memory_context": {
        "object": "Context 与 Memory",
        "surface": "上下文窗口是否足够大",
        "depth": "系统如何选择、压缩、检索、更新和审计信息",
        "shift": "真正的瓶颈从容量转向信息组织和长期状态管理",
    },
    "multimodal_ai": {
        "object": "多模态 AI",
        "surface": "模型是否支持更多输入输出格式",
        "depth": "不同模态如何改变任务边界、证据结构和产品形态",
        "shift": "多模态价值不在媒介叠加，而在重新定义什么任务可以被机器处理",
    },
    "developer_tools": {
        "object": "开发者工具",
        "surface": "AI 是否能生成更多代码",
        "depth": "工程判断、验证、协作和交付链路如何被重新分配",
        "shift": "软件生产的稀缺性从实现速度转向约束表达和结果审查",
    },
    "research_benchmark": {
        "object": "研究与 Benchmark",
        "surface": "指标是否刷新或论文是否发布",
        "depth": "测量方式是否真实解释了能力边界和可迁移价值",
        "shift": "benchmark 应被当作证据材料，而不是结论本身",
    },
    "ecosystem_business": {
        "object": "生态与商业化",
        "surface": "产品、融资、发布和增长叙事",
        "depth": "能力如何被包装成可重复采用、可付费、可维护的生产关系",
        "shift": "真正重要的是技术信号能否变成稳定的分发和价值捕获机制",
    },
    "consumer_economy": {
        "object": "消费文化与市场情绪",
        "surface": "某个爆款、怀旧话题或消费品类突然走红",
        "depth": "情绪、媒介记忆、线下场景和交易机制如何共同把文化记忆转化为消费行为",
        "shift": "关键变化不是单个品类走红，而是情绪价值正在被平台、品牌和交易市场重新组织",
    },
    "product_business": {
        "object": "产品与商业化",
        "surface": "产品发布、公司叙事或增长数字",
        "depth": "能力如何被包装成可重复采用、可付费、可维护的生产关系",
        "shift": "真正重要的是技术信号能否变成稳定的分发和价值捕获机制",
    },
    "ai_infrastructure": {
        "object": "基础设施",
        "surface": "底层平台、runtime 或工具链更新",
        "depth": "可靠性、状态、权限、成本和可复现性如何支撑上层应用",
        "shift": "越成熟的 AI 应用，越依赖不可见的系统层能力",
    },
    "open_source_ecosystem": {
        "object": "开源生态",
        "surface": "项目 star、开源发布或社区热度",
        "depth": "项目、维护者、用户和实践群如何把单点能力变成可复用生态",
        "shift": "开源价值不只在代码公开，而在可验证、可复用和可共同演化",
    },
    "robotics_embodied_ai": {
        "object": "机器人与具身智能",
        "surface": "机器人 demo、传感器或单项任务效果",
        "depth": "感知、物理交互、控制和任务泛化如何共同决定智能边界",
        "shift": "具身智能的核心不是把语言模型搬进机器人，而是让智能接受物理世界约束",
    },
}


def deep_review_frame(category_id: str) -> dict[str, str]:
    return DEEP_REVIEW_FRAMES.get(
        category_id,
        {
            "object": category_label(category_id),
            "surface": "单篇文章中的新鲜 claim",
            "depth": "多篇文章共同指向的底层机制、约束和变化方向",
            "shift": "关键变化在于多个零散事件开始呈现共同的结构性方向",
        },
    )


def collect_review_material(
    article_ids: list[str],
    *,
    category_id: str | None = None,
    limit: int = 8,
) -> list[dict[str, Any]]:
    material: list[dict[str, Any]] = []
    for article_id in article_ids[:limit]:
        record = br.read_json(br.article_record_path(article_id), default={})
        extraction = br.read_json(article_extraction_path(article_id), default={})
        classification = br.read_json(article_classification_path(article_id), default={})
        tags = [
            item
            for item in classification.get("tags", [])
            if isinstance(item, str) and item.strip()
        ][:8]
        canonical_text = load_article_text(article_id)
        keywords = review_keywords(category_id or classification.get("primary_category"), tags)
        claims: list[str] = []
        for item in extraction.get("claims", []):
            if isinstance(item, str) and item.strip():
                claims.append(item.strip())
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                claims.append(item["text"].strip())
        material.append(
            {
                "article_id": article_id,
                "title": record.get("title") or article_id,
                "url": record.get("url") or "",
                "text_length": len(canonical_text),
                "full_text_excerpt": compact_text(canonical_text, limit=2200),
                "evidence_sentences": select_evidence_sentences(canonical_text, keywords, limit=5),
                "summary": extraction.get("summary") or "",
                "main_points": [
                    item
                    for item in extraction.get("main_points", [])
                    if isinstance(item, str) and item.strip()
                ][:4],
                "claims": claims[:4],
                "tags": tags,
            }
        )
    return material


def evidence_phrase(material: list[dict[str, Any]]) -> str:
    titles = [str(item.get("title") or item.get("article_id")) for item in material[:3]]
    if not titles:
        return "当前材料"
    if len(titles) == 1:
        return f"《{titles[0]}》"
    return "、".join(f"《{title}》" for title in titles[:-1]) + f" 和《{titles[-1]}》"


def short_title(title: Any, limit: int = 26) -> str:
    text = str(title or "").strip()
    if not text:
        return "代表性文章"
    if text.startswith("http"):
        return text
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def evidence_summary(material: list[dict[str, Any]], limit: int = 3) -> list[str]:
    lines: list[str] = []
    for item in material[:limit]:
        title = short_title(item.get("title"))
        sentence = ""
        evidence = item.get("evidence_sentences")
        if isinstance(evidence, list) and evidence:
            sentence = str(evidence[0])
        elif item.get("summary"):
            sentence = str(item.get("summary"))
        if sentence:
            lines.append(f"《{title}》显示：{sentence}")
    return lines


def evidence_paragraph(material: list[dict[str, Any]], limit: int = 3) -> str:
    lines = evidence_summary(material, limit=limit)
    if not lines:
        return "当前材料还缺少足够清晰的全文证据句，后续需要补充抓取或人工校正。"
    return "；".join(lines) + "。"


def case_relation_paragraph(material: list[dict[str, Any]], case_count: int) -> str:
    if not material:
        return "当前还没有足够的案例材料支撑跨文章比较。"
    if len(material) == 1:
        item = material[0]
        title = short_title(item.get("title"))
        tags = item.get("tags", [])
        tag_text = "、".join(tags[:3]) if isinstance(tags, list) and tags else "该主题"
        evidence = item.get("evidence_sentences", [])
        sentence = str(evidence[0]) if isinstance(evidence, list) and evidence else str(item.get("summary") or "")
        return f"《{title}》目前是这个主题下的唯一案例，主要承担 {tag_text} 的证据角色：{sentence}。"
    parts: list[str] = []
    for item in material[:4]:
        title = short_title(item.get("title"))
        tags = item.get("tags", [])
        tag_text = "、".join(tags[:3]) if isinstance(tags, list) and tags else "该主题"
        evidence = item.get("evidence_sentences", [])
        sentence = str(evidence[0]) if isinstance(evidence, list) and evidence else str(item.get("summary") or "")
        if sentence:
            parts.append(f"《{title}》承担的是 {tag_text} 的证据角色：{sentence}")
    if parts:
        return "；".join(parts[:3]) + "。"
    return f"这些案例至少形成了 {case_count} 个观察点，但仍需要更精确的全文证据来区分机制、产品和市场信号。"


def review_headings(
    *,
    label: str,
    frame: dict[str, str],
    repeated_pattern_name: str,
    concept_text: str,
    first_case_title: Any,
) -> list[str]:
    return [
        f"一、{label}的真实议题：{frame['depth']}",
        f"二、{repeated_pattern_name} 暴露的瓶颈",
        f"三、《{short_title(first_case_title)}》等案例如何互相解释",
        f"四、从 {concept_text} 推出的判断",
        "五、哪些证据会改变这篇综述的结论",
    ]


def build_deep_review_article(
    *,
    category_id: str,
    article_ids: list[str],
    top_patterns: list[str],
    top_concepts: list[str],
    dominant_patterns: list[dict[str, Any]],
    case_analysis: list[dict[str, Any]],
    key_findings: list[dict[str, Any]],
    open_questions: list[str],
) -> dict[str, Any]:
    label = category_label(category_id)
    frame = deep_review_frame(category_id)
    material = collect_review_material(article_ids, category_id=category_id)
    evidence = evidence_phrase(material)
    pattern_text = "、".join(top_patterns[:3]) if top_patterns else "尚未稳定成形的重复模式"
    concept_text = "、".join(top_concepts[:3]) if top_concepts else "尚未稳定成形的概念"
    article_count = len(article_ids)
    case_count = len(case_analysis)

    core_thesis = (
        f"这组 {label} 材料真正值得关注的不是{frame['surface']}，而是{frame['depth']}。"
        f"{frame['shift']}。"
    )
    dek = (
        f"基于 {article_count} 篇文章，当前综述把 {pattern_text} 视为主要证据线索，"
        f"并把 {concept_text} 作为概念层的组织入口。"
    )

    first_case = case_analysis[0] if case_analysis else {}
    first_case_title = first_case.get("title") or (material[0].get("title") if material else "代表性文章")
    first_case_point = first_case.get("what_it_shows") or (
        material[0].get("summary") if material else "它提供了理解该主题的初始证据。"
    )
    repeated_pattern = dominant_patterns[0] if dominant_patterns else {}
    repeated_pattern_name = repeated_pattern.get("pattern") or (top_patterns[0] if top_patterns else "核心模式")
    repeated_pattern_interpretation = repeated_pattern.get("interpretation") or (
        "这个模式需要在更多文章中继续验证。"
    )

    sections = [
        {
            "heading": "一、真正的问题不是新闻，而是结构变化",
            "paragraphs": [
                (
                    f"{label} 这一组文章如果只按发布时间或单篇观点阅读，很容易变成信息罗列。"
                    f"更有效的读法是把每篇文章当作一个观察窗口：它们分别显示 {frame['object']} "
                    f"在能力、产品、工程和组织采用之间的关系正在怎样移动。"
                ),
                (
                    f"{evidence} 等材料共同说明，表层事件背后反复出现的是 {pattern_text}。"
                    f"这使问题从“发生了什么”转向“为什么这些事情会同时出现”。"
                ),
            ],
        },
        {
            "heading": "二、反复出现的模式说明了系统瓶颈",
            "paragraphs": [
                (
                    f"当前最强的模式是 {repeated_pattern_name}。{repeated_pattern_interpretation}"
                    f"当同一模式以不同形态在多篇文章中重复出现时，它就不再只是单篇文章的修辞，"
                    f"而是开始暴露这一主题的系统瓶颈。"
                ),
                (
                    f"这里更本质的认知是：{frame['depth']}。"
                    f"也就是说，判断这类材料的价值，不应只看单点能力强弱，"
                    f"而要看它是否改变了系统的约束条件、协作方式或可规模化路径。"
                ),
            ],
        },
        {
            "heading": "三、案例之间不是并列关系，而是互相解释",
            "paragraphs": [
                (
                    f"以《{first_case_title}》为例，{first_case_point}"
                    f"这个案例的意义不在于它单独证明了结论，而在于它能和另外 {max(case_count - 1, 0)} "
                    f"个代表案例一起构成对同一机制的交叉观察。"
                ),
                (
                    "这些案例之间并不是简单并列：有的提供机制解释，有的提供产品化证据，"
                    "有的只是热度信号，也有的暴露了当前叙事的薄弱处。"
                    "它们的价值来自相互解释，而不是各自作为孤立事件存在。"
                ),
            ],
        },
        {
            "heading": "四、从材料中可以推出的判断",
            "paragraphs": [
                (
                    f"第一，{label} 的核心变量不是单个技术点，而是技术点进入系统后的组织方式。"
                    f"第二，{concept_text} 这些概念之间需要被看作一组相互牵引的解释框架，"
                    f"而不是彼此独立的标签。"
                ),
                (
                    f"第三，{frame['shift']}。这使得这一类文章的意义超出了单次发布或单篇论文，"
                    f"它们共同描绘的是一个正在变化的系统边界。"
                ),
            ],
        },
        {
            "heading": "五、这组判断的边界",
            "paragraphs": [
                (
                    "这组材料仍然不足以把趋势判断变成确定结论。真正可能削弱上述判断的，"
                    "不是又一篇相似案例，而是失败案例、低采用率案例、成本约束、组织阻力，"
                    "以及指标表现与真实价值不一致的证据。"
                ),
                (
                    "因此，这篇综述的结论应被理解为可检验的判断：它解释了当前材料之间的共同结构，"
                    "但也必须接受后续材料对其边界条件的修正。"
                ),
            ],
        },
    ]

    headings = review_headings(
        label=label,
        frame=frame,
        repeated_pattern_name=str(repeated_pattern_name),
        concept_text=concept_text,
        first_case_title=first_case_title,
    )
    if article_count <= 1:
        headings[2] = f"三、《{short_title(first_case_title)}》这篇文章提供了什么证据"
    full_text_evidence = evidence_paragraph(material, limit=3)
    case_relations = case_relation_paragraph(material, case_count=case_count)
    sections = [
        {
            "heading": headings[0],
            "paragraphs": [
                (
                    f"{label}这一组文章不能只按发布时间或单篇观点阅读。"
                    f"更重要的是看它们如何从不同角度指向同一个问题：{frame['depth']}。"
                ),
                full_text_evidence,
            ],
        },
        {
            "heading": headings[1],
            "paragraphs": [
                (
                    f"当前最强的重复模式是 {repeated_pattern_name}。{repeated_pattern_interpretation}"
                    "这个模式不是从标题或标签直接推出来的，而是需要回到文章中的具体材料："
                    "它反复出现在产品叙事、技术细节、使用场景或市场反应里。"
                ),
                (
                    f"因此，{label}的判断重点不应是单点能力强弱，而是它是否改变了"
                    f"{frame['object']} 的约束条件、协作方式或规模化路径。"
                ),
            ],
        },
        {
            "heading": headings[2],
            "paragraphs": [
                case_relations,
                (
                    (
                        "由于当前只有一篇文章，这里还不能形成跨案例综合；它只能作为主题的初始证据点。"
                        if article_count <= 1
                        else f"这说明 {article_count} 篇文章之间不是简单相加。有些文章提供机制解释，"
                        "有些文章提供案例和场景，有些文章只提供热度信号；只有区分这些证据角色，综述才可能形成判断。"
                    )
                ),
            ],
        },
        {
            "heading": headings[3],
            "paragraphs": [
                (
                    f"把这些材料放在一起后，可以形成一个暂时判断：{frame['shift']}。"
                    f"{concept_text} 不是装饰性标签，而是把多篇文章连接起来的解释入口。"
                ),
                (
                    f"如果这个判断成立，后续文章库的重点就不是继续堆积同类新闻，"
                    f"而是观察哪些新材料会增强、修正或削弱这一解释。"
                ),
            ],
        },
        {
            "heading": headings[4],
            "paragraphs": [
                (
                    "这组材料仍然不足以把趋势判断变成确定结论。真正可能削弱上述判断的，"
                    "不是又一篇相似案例，而是失败案例、低采用率案例、成本约束、组织阻力，"
                    "以及指标表现与真实价值不一致的证据。"
                ),
                (
                    "因此，这篇综述的结论应被理解为可检验的判断：它解释了当前材料之间的共同结构，"
                    "但也必须接受后续全文材料对其边界条件的修正。"
                ),
            ],
        },
    ]

    insights = [
        {
            "claim": f"{label} 的关键变化是从“能力展示”转向“系统约束重组”。",
            "why_it_matters": "这能避免把每次发布或论文指标误读为结构性变化。",
            "evidence_article_ids": article_ids[:5],
        },
        {
            "claim": f"{pattern_text} 是当前最值得继续追踪的证据线索。",
            "why_it_matters": "重复模式比单篇观点更适合沉淀为分类综述和判断层。",
            "evidence_article_ids": sorted(
                {
                    aid
                    for pattern in dominant_patterns[:3]
                    for aid in pattern.get("evidence_article_ids", [])
                    if isinstance(aid, str)
                }
            )[:8],
        },
        {
            "claim": "综述的质量取决于反证，而不是摘要数量。",
            "why_it_matters": "只有持续纳入边界、失败和冲突材料，综述才会形成可更新的认知。",
            "evidence_article_ids": article_ids[:3],
        },
    ]

    return {
        "title": f"{label}深度分析",
        "dek": dek,
        "core_thesis": core_thesis,
        "sections": sections,
        "insights": insights,
        "essential_cognition": (
            f"把 {label} 看成一个证据网络：文章提供观察，标签提供归类，概念提供抽象，"
            "综述负责形成可被反驳和更新的判断。"
        ),
        "misreadings_to_avoid": [
            "把多篇文章压缩成并列摘要，而不比较它们之间的机制关系。",
            "把高频标签直接当作结论，而不检查它是否有跨案例证据。",
            "只收集支持材料，忽略失败、约束和反例。",
        ],
        "writing_hooks": [
            f"{label} 的表层故事是新能力，深层故事是系统边界改变。",
            f"判断 {label} 是否重要，要看它是否改变了约束，而不是是否制造了新鲜感。",
        ],
        "source_trace": {
            "material_basis": "canonical_text + extraction + classification + concepts",
            "article_text_lengths": {
                str(item.get("article_id")): item.get("text_length", 0)
                for item in material
            },
            "evidence_sentence_count": sum(
                len(item.get("evidence_sentences", []))
                for item in material
                if isinstance(item.get("evidence_sentences"), list)
            ),
        },
        "generated_by": "deterministic_deep_review_article_v1",
    }


def build_review_analysis(
    *,
    category_id: str,
    article_ids: list[str],
    tag_counter: Counter[str],
    tag_articles: dict[str, list[str]],
    concept_counter: Counter[str],
    key_findings: list[dict[str, Any]],
    open_questions: list[str],
) -> dict[str, Any]:
    article_count = len(article_ids)
    top_patterns = [item["id"] for item in top_items(tag_counter, 5)]
    top_concepts = [item["id"] for item in top_items(concept_counter, 5)]
    maturity = "single_case" if article_count == 1 else "emerging" if article_count < 4 else "recurring"
    maturity_label = {
        "single_case": "单案例",
        "emerging": "早期成形",
        "recurring": "反复出现",
    }[maturity]
    label = category_label(category_id)
    pattern_phrase = "、".join(top_patterns[:3]) if top_patterns else "尚无稳定重复模式"
    concept_phrase = "、".join(top_concepts[:3]) if top_concepts else "尚无稳定概念簇"

    synthesis = (
        f"这组「{label}」综述目前处于「{maturity_label}」阶段。最强的重复模式是 {pattern_phrase}；"
        f"概念层正在围绕 {concept_phrase} 聚集。因此综述重点不应只是罗列文章 claim，"
        f"而应判断这些案例是否在描述同一个底层机制。"
    )

    dominant_patterns = []
    for item in top_items(tag_counter, 6):
        pattern = item["id"]
        supporting = tag_articles.get(pattern, [])
        dominant_patterns.append(
            {
                "pattern": pattern,
                "count": item["count"],
                "interpretation": pattern_interpretation(pattern),
                "evidence_article_ids": supporting[:8],
                "strength": "recurring" if item["count"] >= 3 else "emerging" if item["count"] == 2 else "single_case",
            }
        )

    case_analysis = [
        article_case_analysis(article_id, top_patterns)
        for article_id in article_ids[:5]
    ]

    tensions: list[dict[str, Any]] = []
    if article_count <= 1:
        tensions.append(
            {
                "tension": "Single-case risk",
                "why_it_matters": "单篇文章可以识别信号，但还不能区分结构性模式和一次性故事。",
                "evidence_article_ids": article_ids,
            }
        )
    if len(top_patterns) >= 2:
        tensions.append(
            {
                "tension": "Pattern overlap",
                "why_it_matters": (
                    f"当前综述同时混合了 {top_patterns[0]} 和 {top_patterns[1]}。下一轮应判断它们是同一机制，还是两个相邻主题。"
                ),
                "evidence_article_ids": sorted(set(tag_articles.get(top_patterns[0], []) + tag_articles.get(top_patterns[1], [])))[:8],
            }
        )

    next_questions = list(open_questions[:5])
    next_questions.append(
        f"哪类文章最能反驳当前对「{label}」的解释？"
    )
    next_questions.append(
        "本综述中的哪个概念应该在人审后提升、合并或重命名？"
    )

    analytical_takeaways = []
    for pattern in dominant_patterns[:3]:
        analytical_takeaways.append(
            {
                "takeaway": f"{pattern['pattern']} 是当前的组织性模式。",
                "why_it_matters": pattern["interpretation"],
                "evidence_article_ids": pattern["evidence_article_ids"],
            }
        )

    deep_review = build_deep_review_article(
        category_id=category_id,
        article_ids=article_ids,
        top_patterns=top_patterns,
        top_concepts=top_concepts,
        dominant_patterns=dominant_patterns,
        case_analysis=case_analysis,
        key_findings=key_findings,
        open_questions=open_questions,
    )

    return {
        "central_question": category_central_question(category_id, article_count),
        "maturity": maturity,
        "synthesis": synthesis,
        "deep_review": deep_review,
        "analytical_takeaways": analytical_takeaways,
        "dominant_patterns": dominant_patterns,
        "case_analysis": case_analysis,
        "tensions": tensions,
        "implications": [
            category_implication(category_id, top_patterns),
            "下一轮综述应比较案例之间的关系，而不只是分别总结每篇文章。",
        ],
        "next_questions": next_questions[:8],
    }


def build_topic_reviews(category: str | None = None) -> dict[str, Any]:
    init_state()
    records = load_ready_articles()
    grouped: dict[str, list[str]] = defaultdict(list)

    for record in records:
        article_id = record["article_id"]
        classification = br.read_json(article_classification_path(article_id), default={})
        primary = classification.get("primary_category")
        if not isinstance(primary, str) or not primary:
            continue
        if category and primary != category:
            continue
        grouped[primary].append(article_id)

    existing_doc = br.read_json(TOPIC_REVIEWS_PATH, default={"reviews": []})
    existing_reviews = {
        item.get("review_id"): item
        for item in existing_doc.get("reviews", [])
        if isinstance(item, dict) and item.get("review_id")
    }

    reviews: list[dict[str, Any]] = []
    for category_id, article_ids in sorted(grouped.items()):
        tag_counter: Counter[str] = Counter()
        tag_articles: dict[str, list[str]] = defaultdict(list)
        concept_counter: Counter[str] = Counter()
        key_findings: list[dict[str, Any]] = []
        open_questions: list[str] = []

        for article_id in article_ids:
            classification = br.read_json(article_classification_path(article_id), default={})
            extraction = br.read_json(article_extraction_path(article_id), default={})
            concepts = br.read_json(article_concepts_path(article_id), default={})

            for tag in classification.get("tags", []):
                if isinstance(tag, str):
                    tag_counter[tag] += 1
                    if article_id not in tag_articles[tag]:
                        tag_articles[tag].append(article_id)
            for concept_id in concepts.get("matched_concepts", []):
                if isinstance(concept_id, str):
                    concept_counter[concept_id] += 1

            for point in extraction.get("main_points", [])[:3]:
                if isinstance(point, str) and point.strip():
                    key_findings.append(
                        {
                            "article_id": article_id,
                            "text": point.strip(),
                        }
                    )
            for question in extraction.get("open_questions", []):
                if isinstance(question, str) and question.strip():
                    open_questions.append(question.strip())

        review_id = f"{category_id}_living_review"
        existing = existing_reviews.get(review_id, {})
        analysis = build_review_analysis(
            category_id=category_id,
            article_ids=article_ids,
            tag_counter=tag_counter,
            tag_articles=tag_articles,
            concept_counter=concept_counter,
            key_findings=key_findings,
            open_questions=open_questions,
        )
        existing_source_trace = existing.get("source_trace", {}) if isinstance(existing, dict) else {}
        preserve_existing_deep_review = (
            isinstance(existing.get("deep_review"), dict)
            and isinstance(existing_source_trace, dict)
            and existing_source_trace.get("deep_review_method") == "llm"
            and sorted(existing.get("included_articles", [])) == sorted(article_ids)
        )
        review = {
            "review_id": review_id,
            "title": f"{category_label(category_id)}动态综述",
            "status": existing.get("status", "drafted"),
            "version": int(existing.get("version", 0)) + 1 if existing else 1,
            "last_updated_at": now(),
            "scope": {
                "primary_category": category_id,
                "tags": [item["id"] for item in top_items(tag_counter, 12)],
            },
            "included_articles": article_ids,
            "article_count": len(article_ids),
            "central_question": analysis["central_question"],
            "maturity": analysis["maturity"],
            "synthesis": analysis["synthesis"],
            "deep_review": existing["deep_review"] if preserve_existing_deep_review else analysis["deep_review"],
            "analytical_takeaways": analysis["analytical_takeaways"],
            "key_findings": key_findings[:20],
            "recurring_patterns": analysis["dominant_patterns"],
            "dominant_patterns": analysis["dominant_patterns"],
            "case_analysis": analysis["case_analysis"],
            "disagreements": analysis["tensions"],
            "tensions": analysis["tensions"],
            "implications": analysis["implications"],
            "representative_cases": representative_articles(article_ids, limit=5),
            "concepts_used": top_items(concept_counter, 12),
            "open_questions": open_questions[:10],
            "next_questions": analysis["next_questions"],
            "source_trace": {
                "generated_from": "article extraction, classification, and concepts files",
                "method": "deterministic_review_analysis_mvp",
                **(
                    {
                        "deep_review_method": existing_source_trace.get("deep_review_method"),
                        "deep_review_model": existing_source_trace.get("deep_review_model"),
                        "deep_review_attempts": existing_source_trace.get("deep_review_attempts"),
                    }
                    if preserve_existing_deep_review
                    else {}
                ),
            },
        }
        reviews.append(review)

    if category:
        untouched = [
            item
            for item in existing_doc.get("reviews", [])
            if isinstance(item, dict) and item.get("scope", {}).get("primary_category") != category
        ]
        final_reviews = untouched + reviews
    else:
        final_reviews = reviews

    payload = {
        "created_at": existing_doc.get("created_at") or now(),
        "last_updated_at": now(),
        "version": int(existing_doc.get("version", 1)),
        "reviews": sorted(final_reviews, key=lambda item: item.get("review_id", "")),
        "review_policy": existing_doc.get(
            "review_policy",
            {
                "unit": "Living topic review grouped by primary category.",
                "status_flow": ["drafted", "reviewed", "archived"],
            },
        ),
    }
    br.write_json(TOPIC_REVIEWS_PATH, payload)
    br.append_jsonl(
        br.CHANGE_LOG_PATH,
        {
            "timestamp": now(),
            "event": "knowledge_build_topic_reviews",
            "category": category,
            "review_count": len(reviews),
        },
    )
    return {
        "ok": True,
        "review_count": len(reviews),
        "path": br.relpath_from_root(TOPIC_REVIEWS_PATH),
        "categories": sorted(grouped.keys()),
    }


def deep_review_llm_system_prompt() -> str:
    return """
你是一个中文知识系统的主题综述作者。你的任务不是总结多篇文章，而是写一篇深度分析文章。

必须遵守：
- 以中文为主，必要英文概念可以保留，例如 Agent Runtime、Context Engineering、Benchmark。
- 不要逐篇罗列摘要。
- 先形成核心判断，再解释证据之间的机制关系。
- 必须明确哪些材料是机制证据、哪些是案例证据、哪些只是热度或叙事信号。
- 必须提出反证、边界条件和下一轮阅读要验证的问题。
- 文章要有洞见，但不能脱离输入证据。
- sections 是最终给用户阅读的文章正文，不要写“写作原则”“分析方法”“如何做综述”等过程性文字。
- insights、essential_cognition、misreadings_to_avoid 是内部结构化元数据，不要把它们写成正文里的清单。
- 正文中引用文章时，只使用文章标题、简称或“材料一/材料二/材料三”。不要输出 article_id、哈希 ID、文件名或内部路径。
- 如果输入里出现 article_id，它只用于 evidence_article_ids 字段，不得出现在 title、dek、core_thesis 或 sections.paragraphs 中。

只输出 JSON object，schema 如下：
{
  "title": "string",
  "dek": "string",
  "core_thesis": "string",
  "sections": [
    {"heading": "string", "paragraphs": ["string"]}
  ],
  "insights": [
    {
      "claim": "string",
      "why_it_matters": "string",
      "evidence_article_ids": ["string"]
    }
  ],
  "essential_cognition": "string",
  "misreadings_to_avoid": ["string"],
  "writing_hooks": ["string"],
  "generated_by": "llm_deep_review_article_v1"
}
""".strip()


def normalize_deep_review(raw: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("LLM deep review must be a JSON object")
    deep = dict(fallback)
    for key in [
        "title",
        "dek",
        "core_thesis",
        "essential_cognition",
        "generated_by",
    ]:
        if isinstance(raw.get(key), str) and raw.get(key, "").strip():
            deep[key] = raw[key].strip()
    for key in ["sections", "insights", "misreadings_to_avoid", "writing_hooks"]:
        if isinstance(raw.get(key), list):
            deep[key] = raw[key]
    deep["generated_by"] = raw.get("generated_by") or "llm_deep_review_article_v1"
    return deep


def replace_ids_in_text(text: str, replacements: dict[str, str]) -> str:
    value = text
    for article_id, title in replacements.items():
        if article_id and title:
            value = value.replace(article_id, title)
    return value


def scrub_article_ids_from_deep_review(
    deep_review: dict[str, Any],
    material: list[dict[str, Any]],
) -> dict[str, Any]:
    replacements = {
        str(item.get("article_id")): str(item.get("title") or item.get("article_id"))
        for item in material
        if item.get("article_id")
    }
    if not replacements:
        return deep_review

    for key in ["title", "dek", "core_thesis", "essential_cognition"]:
        if isinstance(deep_review.get(key), str):
            deep_review[key] = replace_ids_in_text(deep_review[key], replacements)

    for section in deep_review.get("sections", []):
        if not isinstance(section, dict):
            continue
        if isinstance(section.get("heading"), str):
            section["heading"] = replace_ids_in_text(section["heading"], replacements)
        section["paragraphs"] = [
            replace_ids_in_text(paragraph, replacements)
            for paragraph in section.get("paragraphs", [])
            if isinstance(paragraph, str)
        ]

    return deep_review


def find_review_for_category(reviews_doc: dict[str, Any], category: str) -> dict[str, Any]:
    for review in reviews_doc.get("reviews", []):
        if not isinstance(review, dict):
            continue
        scope = review.get("scope", {}) if isinstance(review.get("scope"), dict) else {}
        if scope.get("primary_category") == category:
            return review
    raise ValueError(f"No topic review found for category: {category}")


def draft_deep_review(category: str, model: str | None = None) -> dict[str, Any]:
    from llm_client import complete_json, load_project_env

    loaded_env_files = load_project_env()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "Missing required environment variable: OPENROUTER_API_KEY. "
            "Set it in your shell or in a project .env file."
        )
    reviews_doc = br.read_json(TOPIC_REVIEWS_PATH, default={"reviews": []})
    try:
        review = find_review_for_category(reviews_doc, category)
    except ValueError:
        build_topic_reviews(category=category)
        reviews_doc = br.read_json(TOPIC_REVIEWS_PATH, default={"reviews": []})
        review = find_review_for_category(reviews_doc, category)
    material = collect_review_material(review.get("included_articles", []), category_id=category, limit=12)
    public_material: list[dict[str, Any]] = []
    for index, item in enumerate(material, start=1):
        public_material.append(
            {
                "ref": f"材料{index}",
                "title": item.get("title"),
                "url": item.get("url"),
                "summary": item.get("summary"),
                "main_points": item.get("main_points", []),
                "claims": item.get("claims", []),
                "tags": item.get("tags", []),
                "text_length": item.get("text_length"),
                "full_text_excerpt": item.get("full_text_excerpt"),
                "evidence_sentences": item.get("evidence_sentences", []),
            }
        )
    context = {
        "category": {
            "id": category,
            "label": category_label(category),
        },
        "article_count": review.get("article_count"),
        "included_articles": public_material,
        "article_ref_map": [
            {
                "ref": f"材料{index}",
                "article_id": item.get("article_id"),
                "title": item.get("title"),
            }
            for index, item in enumerate(material, start=1)
        ],
        "existing_deep_review": review.get("deep_review", {}),
        "dominant_patterns": review.get("dominant_patterns", []),
        "case_analysis": review.get("case_analysis", []),
        "tensions": review.get("tensions", []),
        "concepts_used": review.get("concepts_used", []),
        "next_questions": review.get("next_questions", []),
    }
    user_prompt = (
        "请基于以下主题综述证据，重写 deep_review。输出必须是符合 schema 的 JSON object。\n\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )

    model_chain = ((model, model),) if model else None
    result = complete_json(
        deep_review_llm_system_prompt(),
        user_prompt,
        model_chain=model_chain,
        temperature=0.25,
        max_tokens=7000,
    )
    review["deep_review"] = scrub_article_ids_from_deep_review(
        normalize_deep_review(
        result.data,
        fallback=review.get("deep_review", {}) if isinstance(review.get("deep_review"), dict) else {},
        ),
        material,
    )
    review.setdefault("source_trace", {})
    if isinstance(review["source_trace"], dict):
        review["source_trace"]["deep_review_method"] = "llm"
        review["source_trace"]["deep_review_model"] = result.model
        review["source_trace"]["deep_review_attempts"] = result.attempts
    reviews_doc["last_updated_at"] = now()
    br.write_json(TOPIC_REVIEWS_PATH, reviews_doc)
    report = build_report()
    return {
        "ok": True,
        "category": category,
        "model": result.model,
        "loaded_env_files": loaded_env_files,
        "path": br.relpath_from_root(TOPIC_REVIEWS_PATH),
        "report": report,
    }


FRAMEWORK_PROMPT_VERSION = "frameworks_from_reviews_v1"


def framework_source_reviews(reviews_doc: dict[str, Any]) -> list[dict[str, Any]]:
    reviews = reviews_doc.get("reviews", []) if isinstance(reviews_doc, dict) else []
    source: list[dict[str, Any]] = []
    for review in reviews:
        if not isinstance(review, dict):
            continue
        deep = review.get("deep_review", {}) if isinstance(review.get("deep_review"), dict) else {}
        insights = []
        for item in deep.get("insights", []):
            if isinstance(item, dict):
                insights.append(
                    {
                        "claim": item.get("claim"),
                        "why_it_matters": item.get("why_it_matters"),
                    }
                )
        source.append(
            {
                "review_id": review.get("review_id"),
                "category": review.get("scope", {}).get("primary_category")
                if isinstance(review.get("scope"), dict)
                else None,
                "title": review.get("title"),
                "article_count": review.get("article_count"),
                "central_question": review.get("central_question"),
                "synthesis": review.get("synthesis"),
                "core_thesis": deep.get("core_thesis"),
                "insights": insights[:5],
                "misreadings_to_avoid": deep.get("misreadings_to_avoid", [])[:5]
                if isinstance(deep.get("misreadings_to_avoid"), list)
                else [],
                "tensions": review.get("tensions", [])[:5]
                if isinstance(review.get("tensions"), list)
                else [],
                "next_questions": review.get("next_questions", [])[:5]
                if isinstance(review.get("next_questions"), list)
                else [],
            }
        )
    return source


def framework_source_hash(source_reviews: list[dict[str, Any]]) -> str:
    return stable_hash(
        {
            "prompt_version": FRAMEWORK_PROMPT_VERSION,
            "reviews": source_reviews,
        }
    )


def slug_id(value: Any, fallback: str) -> str:
    text = str(value or "").strip().lower()
    slugged = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return slugged or fallback


def normalize_string_list(value: Any, limit: int, fallback: list[str]) -> list[str]:
    items: list[str] = []
    if isinstance(value, list):
        for item in value:
            text = compact_text(str(item or ""), limit=260)
            if text and text not in items:
                items.append(text)
            if len(items) >= limit:
                break
    return items or fallback[:limit]


def deterministic_frameworks_from_reviews(
    source_reviews: list[dict[str, Any]],
    source_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    categories = [
        str(item.get("category") or item.get("review_id") or "")
        for item in source_reviews
        if item.get("category") or item.get("review_id")
    ]
    review_ids = [
        str(item.get("review_id"))
        for item in source_reviews
        if item.get("review_id")
    ]
    generated_at = now()
    framework_doc = {
        "created_at": generated_at,
        "last_updated_at": generated_at,
        "version": 1,
        "source_hash": source_hash,
        "prompt_version": FRAMEWORK_PROMPT_VERSION,
        "generated_by": "deterministic_framework_fallback_v1",
        "frameworks": [
            {
                "framework_id": "mechanism_constraint_counterevidence",
                "name": "机制-约束-反证框架",
                "purpose": "把分类综述中的观察推进到可讨论的判断：先解释机制，再找约束和反证，最后给出可更新结论。",
                "dimensions": [
                    "这个分类的核心现象是什么？",
                    "多篇文章共同指向了什么底层机制？",
                    "哪些边界条件会让这个机制失效？",
                    "已有材料里有哪些反向信号或失败线索？",
                    "当前最稳妥的判断是什么，后续应如何更新？",
                ],
                "best_for": categories[:6],
                "source_review_ids": review_ids,
                "status": "drafted",
            },
            {
                "framework_id": "evidence_network_mapping",
                "name": "证据网络映射框架",
                "purpose": "把文章、分类、标签、综述之间的关系画成证据网络，避免把单篇文章误读成结构性结论。",
                "dimensions": [
                    "哪些文章提供机制证据，哪些只提供热度信号？",
                    "哪些标签反复出现，哪些只是单篇材料的噪声？",
                    "不同分类综述之间是否共享同一个问题结构？",
                    "哪些材料支持当前判断，哪些材料削弱当前判断？",
                    "下一篇新文章应该验证网络中的哪个薄弱环节？",
                ],
                "best_for": categories[:6],
                "source_review_ids": review_ids,
                "status": "drafted",
            },
        ],
    }
    template_doc = {
        "created_at": generated_at,
        "last_updated_at": generated_at,
        "version": 1,
        "source_hash": source_hash,
        "prompt_version": FRAMEWORK_PROMPT_VERSION,
        "generated_by": "deterministic_framework_fallback_v1",
        "templates": [
            {
                "template_id": "review_to_argument",
                "name": "从综述到论点",
                "use_case": "把一个分类综述改写成一篇分析文章或观点笔记。",
                "structure": [
                    "一句话判断",
                    "代表性材料",
                    "重复模式",
                    "机制解释",
                    "约束与反证",
                    "可更新结论",
                ],
                "source_review_ids": review_ids,
                "status": "drafted",
            },
            {
                "template_id": "cross_review_theory_note",
                "name": "跨综述理论笔记",
                "use_case": "把多个分类综述抽象成更高层的判断或理论雏形。",
                "structure": [
                    "多个分类共同暴露的问题",
                    "表层差异",
                    "共享机制",
                    "边界条件",
                    "理论化表达",
                    "后续验证材料",
                ],
                "source_review_ids": review_ids,
                "status": "drafted",
            },
        ],
    }
    return framework_doc, template_doc


def framework_llm_system_prompt() -> str:
    return """
你是一个中文知识系统的高层抽象设计器。你要基于多个主题综述，生成可复用的“思考框架”和“表达模板”。

只输出 JSON，不要输出 Markdown。Schema:
{
  "frameworks": [
    {
      "framework_id": "英文 snake_case",
      "name": "中文名称，可保留必要英文术语",
      "purpose": "这个框架解决什么思考问题",
      "dimensions": ["4-7 个问题或分析维度"],
      "best_for": ["适用的分类或场景"],
      "source_review_ids": ["来自输入的 review_id"],
      "status": "drafted"
    }
  ],
  "templates": [
    {
      "template_id": "英文 snake_case",
      "name": "中文名称",
      "use_case": "这个模板用于写什么",
      "structure": ["5-8 个表达段落或论证步骤"],
      "source_review_ids": ["来自输入的 review_id"],
      "status": "drafted"
    }
  ]
}

要求：
1. 不要罗列综述摘要，要抽象出可迁移的思考和表达结构。
2. 框架必须能处理反证、失败、边界条件，不能只支持正向材料。
3. 中文为主，AI Agent、LLM、Context Engineering 等必要英文术语可保留。
4. 不要输出内部文章 id 或无意义哈希。
""".strip()


def normalize_framework_generation(
    raw: Any,
    *,
    fallback_frameworks: dict[str, Any],
    fallback_templates: dict[str, Any],
    source_hash: str,
    model: str | None,
    attempts: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    generated_at = now()
    if not isinstance(raw, dict):
        return fallback_frameworks, fallback_templates

    frameworks: list[dict[str, Any]] = []
    for index, item in enumerate(raw.get("frameworks", []), start=1):
        if not isinstance(item, dict):
            continue
        name = compact_text(str(item.get("name") or ""), limit=120)
        purpose = compact_text(str(item.get("purpose") or ""), limit=420)
        dimensions = normalize_string_list(item.get("dimensions"), 7, [])
        if not name or not purpose or len(dimensions) < 3:
            continue
        frameworks.append(
            {
                "framework_id": slug_id(item.get("framework_id") or name, f"framework_{index}"),
                "name": name,
                "purpose": purpose,
                "dimensions": dimensions,
                "best_for": normalize_string_list(item.get("best_for"), 8, []),
                "source_review_ids": normalize_string_list(item.get("source_review_ids"), 12, []),
                "status": "llm_draft" if model else "drafted",
            }
        )
        if len(frameworks) >= 4:
            break

    templates: list[dict[str, Any]] = []
    for index, item in enumerate(raw.get("templates", []), start=1):
        if not isinstance(item, dict):
            continue
        name = compact_text(str(item.get("name") or ""), limit=120)
        use_case = compact_text(str(item.get("use_case") or ""), limit=360)
        structure = normalize_string_list(item.get("structure"), 8, [])
        if not name or len(structure) < 4:
            continue
        templates.append(
            {
                "template_id": slug_id(item.get("template_id") or name, f"template_{index}"),
                "name": name,
                "use_case": use_case,
                "structure": structure,
                "source_review_ids": normalize_string_list(item.get("source_review_ids"), 12, []),
                "status": "llm_draft" if model else "drafted",
            }
        )
        if len(templates) >= 4:
            break

    if not frameworks:
        frameworks = fallback_frameworks.get("frameworks", [])
    if not templates:
        templates = fallback_templates.get("templates", [])

    framework_doc = {
        "created_at": fallback_frameworks.get("created_at") or generated_at,
        "last_updated_at": generated_at,
        "version": int(fallback_frameworks.get("version", 1)),
        "source_hash": source_hash,
        "prompt_version": FRAMEWORK_PROMPT_VERSION,
        "generated_by": "llm_framework_generation_v1" if model else "deterministic_framework_fallback_v1",
        "llm_model": model,
        "llm_attempts": attempts,
        "frameworks": frameworks,
    }
    template_doc = {
        "created_at": fallback_templates.get("created_at") or generated_at,
        "last_updated_at": generated_at,
        "version": int(fallback_templates.get("version", 1)),
        "source_hash": source_hash,
        "prompt_version": FRAMEWORK_PROMPT_VERSION,
        "generated_by": "llm_framework_generation_v1" if model else "deterministic_framework_fallback_v1",
        "llm_model": model,
        "llm_attempts": attempts,
        "templates": templates,
    }
    return framework_doc, template_doc


def build_frameworks(force: bool = False, model: str | None = None) -> dict[str, Any]:
    init_state()
    reviews_doc = br.read_json(TOPIC_REVIEWS_PATH, default={"reviews": []})
    source_reviews = framework_source_reviews(reviews_doc if isinstance(reviews_doc, dict) else {})
    source_hash = framework_source_hash(source_reviews)
    existing_frameworks = br.read_json(THINKING_FRAMEWORKS_PATH, default={})
    existing_templates = br.read_json(EXPRESSION_TEMPLATES_PATH, default={})
    if (
        not force
        and isinstance(existing_frameworks, dict)
        and isinstance(existing_templates, dict)
        and existing_frameworks.get("source_hash") == source_hash
        and existing_templates.get("source_hash") == source_hash
        and existing_frameworks.get("frameworks")
        and existing_templates.get("templates")
    ):
        return {
            "ok": True,
            "skipped": True,
            "reason": "source_hash unchanged",
            "source_hash": source_hash,
            "framework_count": len(existing_frameworks.get("frameworks", [])),
            "template_count": len(existing_templates.get("templates", [])),
        }

    fallback_frameworks, fallback_templates = deterministic_frameworks_from_reviews(source_reviews, source_hash)
    raw: Any = None
    llm_model: str | None = None
    llm_attempts: int | None = None
    loaded_env_files: list[str] = []
    method = "deterministic"

    try:
        from llm_client import complete_json, load_project_env

        loaded_env_files = load_project_env()
        if os.environ.get("OPENROUTER_API_KEY"):
            model_chain = ((model, model),) if model else None
            user_prompt = (
                "请基于以下主题综述，生成思考框架和表达模板。输出必须是符合 schema 的 JSON object。\n\n"
                + json.dumps({"reviews": source_reviews}, ensure_ascii=False, indent=2)
            )
            result = complete_json(
                framework_llm_system_prompt(),
                user_prompt,
                model_chain=model_chain,
                temperature=0.25,
                max_tokens=9000,
            )
            raw = result.data
            llm_model = result.model
            llm_attempts = result.attempts
            method = "llm"
    except Exception as exc:
        br.append_jsonl(
            br.CHANGE_LOG_PATH,
            {
                "timestamp": now(),
                "event": "knowledge_build_frameworks_llm_failed",
                "error": compact_text(str(exc), limit=500),
            },
        )

    framework_doc, template_doc = normalize_framework_generation(
        raw,
        fallback_frameworks=fallback_frameworks,
        fallback_templates=fallback_templates,
        source_hash=source_hash,
        model=llm_model,
        attempts=llm_attempts,
    )
    br.write_json(THINKING_FRAMEWORKS_PATH, framework_doc)
    br.write_json(EXPRESSION_TEMPLATES_PATH, template_doc)
    br.append_jsonl(
        br.CHANGE_LOG_PATH,
        {
            "timestamp": now(),
            "event": "knowledge_build_frameworks",
            "method": method,
            "source_hash": source_hash,
            "framework_count": len(framework_doc.get("frameworks", [])),
            "template_count": len(template_doc.get("templates", [])),
        },
    )
    return {
        "ok": True,
        "method": method,
        "model": llm_model,
        "loaded_env_files": loaded_env_files,
        "source_hash": source_hash,
        "framework_count": len(framework_doc.get("frameworks", [])),
        "template_count": len(template_doc.get("templates", [])),
        "paths": [
            br.relpath_from_root(THINKING_FRAMEWORKS_PATH),
            br.relpath_from_root(EXPRESSION_TEMPLATES_PATH),
        ],
    }


def rebuild_concepts() -> dict[str, Any]:
    reset_global_concepts()
    updated: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for article_id in iter_article_ids():
        if not article_classification_path(article_id).exists():
            skipped.append({"article_id": article_id, "reason": "missing classification.json"})
            continue
        result = update_concepts(article_id, force=True)
        updated.append(
            {
                "article_id": article_id,
                "matched_concepts": result.get("matched_concepts", []),
            }
        )
    concepts_doc = br.read_json(GLOBAL_CONCEPTS_PATH, default={"concepts": []})
    return {
        "ok": True,
        "updated_articles": len(updated),
        "skipped_articles": len(skipped),
        "concept_count": len(concepts_doc.get("concepts", []))
        if isinstance(concepts_doc, dict)
        else 0,
        "skipped": skipped,
    }


def iter_article_ids(limit: int | None = None, active_only: bool = True) -> list[str]:
    ids = []
    for record in br.load_all_article_records():
        article_id = record.get("article_id")
        if not isinstance(article_id, str):
            continue
        if active_only and not article_is_active(record):
            continue
        ids.append(article_id)
    return ids[:limit] if limit is not None else ids


def run_mvp(
    limit: int | None = None,
    force: bool = False,
    skip_frameworks: bool = False,
) -> dict[str, Any]:
    init_state(force=force)
    if force:
        reset_global_concepts()
    article_ids = iter_article_ids(limit=limit)
    results: list[dict[str, Any]] = []
    for article_id in article_ids:
        record = br.read_json(br.article_record_path(article_id), default={})
        content_status = record.get("content_state", {}).get("full_text_status")
        if content_status not in {"acquired", "partial"}:
            results.append(
                {
                    "article_id": article_id,
                    "skipped": True,
                    "reason": f"content_status={content_status!r}",
                }
            )
            continue
        extraction = extract_article(article_id, force=force)
        classification = classify_article(article_id, force=force)
        concepts = update_concepts(article_id, force=force)
        results.append(
            {
                "article_id": article_id,
                "extraction": extraction.get("path"),
                "classification": classification.get("path"),
                "concepts": concepts.get("path"),
            }
        )
    reviews = build_topic_reviews()
    frameworks = None if skip_frameworks else build_frameworks()
    report = build_report()
    return {
        "ok": True,
        "processed": sum(1 for item in results if not item.get("skipped")),
        "skipped": sum(1 for item in results if item.get("skipped")),
        "results": results,
        "reviews": reviews,
        "frameworks": frameworks,
        "report": report,
    }


def refresh_llm_reviews(
    force: bool = False,
    model: str | None = None,
) -> dict[str, Any]:
    reviews_doc = br.read_json(TOPIC_REVIEWS_PATH, default={"reviews": []})
    categories: list[str] = []
    for review in reviews_doc.get("reviews", []):
        if not isinstance(review, dict):
            continue
        scope = review.get("scope", {}) if isinstance(review.get("scope"), dict) else {}
        category = scope.get("primary_category")
        if not isinstance(category, str) or not category:
            continue
        trace = review.get("source_trace", {}) if isinstance(review.get("source_trace"), dict) else {}
        if force or trace.get("deep_review_method") != "llm":
            categories.append(category)

    refreshed: list[dict[str, Any]] = []
    skipped = 0
    for category in categories:
        result = draft_deep_review(category=category, model=model)
        refreshed.append(
            {
                "category": category,
                "model": result.get("model"),
                "path": result.get("path"),
            }
        )

    if not categories:
        skipped = len(reviews_doc.get("reviews", [])) if isinstance(reviews_doc, dict) else 0

    return {
        "ok": True,
        "refreshed_count": len(refreshed),
        "skipped_count": skipped,
        "refreshed": refreshed,
    }


def knowledge_status() -> dict[str, Any]:
    all_article_ids = iter_article_ids(active_only=False)
    article_ids = iter_article_ids(active_only=True)
    inactive_records = [
        record
        for record in br.load_all_article_records()
        if isinstance(record.get("article_id"), str) and not article_is_active(record)
    ]
    extracted = sum(1 for aid in article_ids if article_extraction_path(aid).exists())
    classified = sum(1 for aid in article_ids if article_classification_path(aid).exists())
    concepted = sum(1 for aid in article_ids if article_concepts_path(aid).exists())
    concepts_doc = br.read_json(GLOBAL_CONCEPTS_PATH, default={"concepts": []})
    reviews_doc = br.read_json(TOPIC_REVIEWS_PATH, default={"reviews": []})
    return {
        "article_count": len(article_ids),
        "total_article_count": len(all_article_ids),
        "inactive_article_count": len(inactive_records),
        "articles_with_extraction": extracted,
        "articles_with_classification": classified,
        "articles_with_concepts": concepted,
        "concept_count": len(concepts_doc.get("concepts", []))
        if isinstance(concepts_doc, dict)
        else 0,
        "review_count": len(reviews_doc.get("reviews", []))
        if isinstance(reviews_doc, dict)
        else 0,
    }


def set_article_lifecycle(
    article_id: str,
    state: str,
    reason: str = "",
    actor: str = "",
) -> dict[str, Any]:
    state = state.strip().lower()
    if state not in VALID_ARTICLE_STATES:
        raise ValueError(f"Unsupported article lifecycle state: {state}")

    record_path = br.article_record_path(article_id)
    record = br.read_json(record_path, default=None)
    if not isinstance(record, dict):
        raise FileNotFoundError(f"No record.json found for article_id {article_id!r}")

    previous_state = article_lifecycle_state(record)
    timestamp = now()
    record["lifecycle_state"] = state
    record["last_updated_at"] = timestamp
    record.setdefault("lifecycle_events", []).append(
        {
            "timestamp": timestamp,
            "from": previous_state,
            "to": state,
            "reason": reason,
            "actor": actor,
        }
    )
    if state in INACTIVE_ARTICLE_STATES:
        record["exclude_reason"] = reason
        record["archived_at"] = timestamp
        record["archived_by"] = actor
    else:
        record["exclude_reason"] = None
        record["restored_at"] = timestamp
        record["restored_by"] = actor

    br.write_json(record_path, record)
    br.append_jsonl(
        br.CHANGE_LOG_PATH,
        {
            "timestamp": timestamp,
            "event": "knowledge_article_lifecycle_update",
            "article_id": article_id,
            "from": previous_state,
            "to": state,
            "reason": reason,
            "actor": actor,
        },
    )
    return {
        "ok": True,
        "article_id": article_id,
        "previous_state": previous_state,
        "state": state,
        "record_path": br.relpath_from_root(record_path),
    }


def html_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "item"


def pill(label: Any, class_name: str = "") -> str:
    extra = f" {class_name}" if class_name else ""
    return f"<span class='pill{extra}'>{html_escape(label)}</span>"


STATUS_LABELS: dict[str, str] = {
    "drafted": "草稿",
    "llm_draft": "LLM 初稿",
    "approved": "已确认",
    "merged": "已合并",
    "deprecated": "已废弃",
    "reviewed": "已审阅",
    "archived": "已归档",
    "single_case": "单案例",
    "emerging": "早期成形",
    "recurring": "反复出现",
    "low": "低",
    "medium": "中",
    "high": "高",
    "acquired": "已获取",
    "partial": "部分获取",
    "blocked": "被阻断",
    "fetch_failed": "抓取失败",
    "missing": "缺失",
}


TENSION_LABELS: dict[str, str] = {
    "Single-case risk": "单案例风险",
    "Pattern overlap": "模式重叠",
}


def display_status(value: Any) -> str:
    text = "" if value is None else str(value)
    return STATUS_LABELS.get(text, text)


def display_tension(value: Any) -> str:
    text = "" if value is None else str(value)
    return TENSION_LABELS.get(text, text)


def article_title_map() -> dict[str, str]:
    titles: dict[str, str] = {}
    for record in br.load_all_article_records():
        article_id = record.get("article_id")
        if isinstance(article_id, str):
            titles[article_id] = str(record.get("title") or article_id)
    return titles


def load_article_view_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in br.load_all_article_records():
        article_id = record.get("article_id")
        if not isinstance(article_id, str):
            continue
        if not article_is_active(record):
            continue
        extraction = br.read_json(article_extraction_path(article_id), default={})
        classification = br.read_json(article_classification_path(article_id), default={})
        concepts = br.read_json(article_concepts_path(article_id), default={})
        rows.append(
            {
                "article_id": article_id,
                "title": record.get("title") or article_id,
                "url": record.get("url") or "",
                "content_status": record.get("content_state", {}).get("full_text_status", "unknown"),
                "primary_category": classification.get("primary_category"),
                "secondary_categories": classification.get("secondary_categories", []),
                "tags": classification.get("tags", []),
                "importance": classification.get("importance", "unknown"),
                "summary": extraction.get("summary", ""),
                "concepts": concepts.get("matched_concepts", []),
                "issue_url": issue_url_for_record(record),
                "lifecycle_state": article_lifecycle_state(record),
            }
        )
    return sorted(rows, key=lambda item: (str(item.get("primary_category")), str(item.get("title"))))


def render_deep_review(deep_review: Any) -> str:
    if not isinstance(deep_review, dict):
        return ""

    sections_html = ""
    for section in deep_review.get("sections", []):
        if not isinstance(section, dict):
            continue
        paragraphs = "".join(
            f"<p>{html_escape(paragraph)}</p>"
            for paragraph in section.get("paragraphs", [])
            if isinstance(paragraph, str) and paragraph.strip()
        )
        if paragraphs:
            sections_html += (
                f"<section class='deep-section'>"
                f"<h4>{html_escape(section.get('heading', ''))}</h4>"
                f"{paragraphs}</section>"
            )

    return f"""
      <div class="deep-review">
        <p class="eyebrow">深度分析文章</p>
        <h3>{html_escape(deep_review.get("title", ""))}</h3>
        <p class="lead">{html_escape(deep_review.get("dek", ""))}</p>
        <div class="thesis">
          <strong>核心判断</strong>
          <p>{html_escape(deep_review.get("core_thesis", ""))}</p>
        </div>
        {sections_html}
      </div>
    """


def render_review_card(review: dict[str, Any], titles: dict[str, str]) -> str:
    scope = review.get("scope", {}) if isinstance(review.get("scope"), dict) else {}
    category_id = scope.get("primary_category")
    patterns = review.get("dominant_patterns") or review.get("recurring_patterns", [])
    pattern_html = "".join(
        pill(item.get("pattern", ""), "pattern")
        for item in patterns[:8]
        if isinstance(item, dict)
    )
    cases = review.get("case_analysis") or review.get("representative_cases", [])
    case_items = []
    for case in cases[:4]:
        if not isinstance(case, dict):
            continue
        article_id = str(case.get("article_id") or "")
        title = case.get("title") or titles.get(article_id) or article_id
        url = case.get("url") or ""
        if url:
            link = (
                "<a href=\""
                + html_escape(url)
                + "\" target=\"_blank\" rel=\"noreferrer\">"
                + html_escape(title)
                + "</a>"
            )
        else:
            link = html_escape(title)
        what_it_shows = case.get("what_it_shows") or case.get("summary") or ""
        limits = case.get("limits") or ""
        case_items.append(
            f"<li><strong>{link}</strong><p>{html_escape(what_it_shows)}</p><p class='muted'>{html_escape(limits)}</p></li>"
        )
    takeaways = [
        item
        for item in review.get("analytical_takeaways", [])
        if isinstance(item, dict)
    ][:4]
    takeaway_html = "".join(
        (
            f"<li><strong>{html_escape(item.get('takeaway', ''))}</strong>"
            f"<p>{html_escape(item.get('why_it_matters', ''))}</p></li>"
        )
        for item in takeaways
    )
    pattern_rows = "".join(
        (
            f"<li><strong>{html_escape(item.get('pattern', ''))}</strong> "
            f"<span class='muted'>({html_escape(display_status(item.get('strength', '')))}, {html_escape(item.get('count', ''))} 篇)</span>"
            f"<p>{html_escape(item.get('interpretation', ''))}</p></li>"
        )
        for item in patterns[:4]
        if isinstance(item, dict)
    )
    tensions = [
        item
        for item in (review.get("tensions") or review.get("disagreements") or [])
        if isinstance(item, dict)
    ][:3]
    tension_html = "".join(
        f"<li><strong>{html_escape(display_tension(item.get('tension', '')))}</strong><p>{html_escape(item.get('why_it_matters', ''))}</p></li>"
        for item in tensions
    )
    next_questions = [
        item
        for item in review.get("next_questions", [])
        if isinstance(item, str)
    ][:4]
    question_html = "".join(f"<li>{html_escape(item)}</li>" for item in next_questions)
    concepts = review.get("concepts_used", [])
    concept_html = "".join(
        pill(item.get("id", ""), "concept")
        for item in concepts[:8]
        if isinstance(item, dict)
    )
    deep_review_html = render_deep_review(review.get("deep_review"))
    deep = review.get("deep_review", {}) if isinstance(review.get("deep_review"), dict) else {}
    trace = review.get("source_trace", {}) if isinstance(review.get("source_trace"), dict) else {}
    review_method_label = "LLM 深度分析" if trace.get("deep_review_method") == "llm" else "规则草稿"
    preview = deep.get("core_thesis") or review.get("central_question", "")
    return f"""
      <details class="card review-card" id="{html_escape(slug(str(review.get('review_id', 'review'))))}">
        <summary class="review-summary">
          <span class="review-summary-main">
            <span>
              <span class="eyebrow">{html_escape(category_label(str(category_id)))}</span>
              <span class="review-title">{html_escape(review.get("title", ""))}</span>
            </span>
            <span class="review-meta">
              <span class="type-badge thinking">{html_escape(review_method_label)}</span>
              <span class="count">{html_escape(review.get("article_count", 0))} 篇文章</span>
            </span>
          </span>
          <span class="review-preview">{html_escape(preview)}</span>
          <span class="pill-row review-preview-tags">{pattern_html}</span>
        </summary>
        <div class="review-body">
        {deep_review_html}
        <details class="evidence-detail">
          <summary>证据附录：问题、模式、案例与下一轮问题</summary>
        <p class="review-question">{html_escape(review.get("central_question", ""))}</p>
        <p>{html_escape(review.get("synthesis", ""))}</p>
        <div class="split">
          <div>
            <h4>分析性结论</h4>
            <ul>{takeaway_html if takeaway_html else '<li>暂无分析性结论</li>'}</ul>
          </div>
          <div>
            <h4>主导模式</h4>
            <ul>{pattern_rows if pattern_rows else '<li>暂无主导模式</li>'}</ul>
          </div>
        </div>
        <div class="split">
          <div>
            <h4>案例分析</h4>
            <ul>{''.join(case_items) if case_items else '<li>暂无案例分析</li>'}</ul>
          </div>
          <div>
            <h4>张力与下一步问题</h4>
            <ul>{tension_html if tension_html else '<li>暂无记录的张力</li>'}</ul>
            <ul>{question_html if question_html else ''}</ul>
          </div>
        </div>
        </details>
        <div class="concept-strip">{concept_html}</div>
        </div>
      </details>
    """


def render_concept_card(concept: dict[str, Any], titles: dict[str, str]) -> str:
    article_ids = [
        item
        for item in concept.get("supporting_articles", [])
        if isinstance(item, str)
    ]
    article_links = "".join(
        f"<li>{html_escape(titles.get(article_id, article_id))}</li>"
        for article_id in article_ids[:5]
    )
    category_html = "".join(
        pill(category_label(category), "category")
        for category in concept.get("related_categories", [])
        if isinstance(category, str)
    )
    tag_html = "".join(
        pill(tag, "pattern")
        for tag in concept.get("related_tags", [])
        if isinstance(tag, str)
    )
    return f"""
      <article class="card concept-card">
        <div class="card-head">
          <div>
            <p class="eyebrow">{html_escape(display_status(concept.get("status", "drafted")))}</p>
            <h3>{html_escape(concept.get("name", concept.get("concept_id", "")))}</h3>
          </div>
          <span class="count">{len(article_ids)} 篇文章</span>
        </div>
        <p>{html_escape(concept.get("definition", ""))}</p>
        <div class="pill-row">{category_html}{tag_html}</div>
        <details>
          <summary>支撑文章</summary>
          <ul>{article_links if article_links else '<li>暂无支撑文章</li>'}</ul>
        </details>
      </article>
    """


def render_article_row(row: dict[str, Any]) -> str:
    url = row.get("url") or ""
    title = row.get("title") or row.get("article_id")
    title_html = (
        f"<a href='{html_escape(url)}' target='_blank' rel='noreferrer'>{html_escape(title)}</a>"
        if url
        else html_escape(title)
    )
    tags = row.get("tags", []) if isinstance(row.get("tags"), list) else []
    tag_html = "".join(pill(tag, "pattern") for tag in tags[:8])
    concepts = row.get("concepts", []) if isinstance(row.get("concepts"), list) else []
    concept_html = "".join(pill(concept, "concept") for concept in concepts[:8])
    summary = row.get("summary") or ""
    return f"""
      <article class="article-row">
        <div>
          <p class="eyebrow">{html_escape(category_label(row.get("primary_category")))}</p>
          <h3>{title_html}</h3>
          <p>{html_escape(summary)}</p>
          <div class="pill-row">{tag_html}{concept_html}</div>
        </div>
        <div class="article-meta">
          <span>{html_escape(category_label(row.get("primary_category")))}</span>
          <span>{html_escape(display_status(row.get("importance", "unknown")))}</span>
          <span>{html_escape(display_status(row.get("content_status", "unknown")))}</span>
        </div>
      </article>
    """


def group_article_rows_by_category(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        category = row.get("primary_category")
        key = category if isinstance(category, str) and category else "uncategorized"
        grouped[key].append(row)
    return {
        key: sorted(items, key=lambda item: str(item.get("title") or ""))
        for key, items in sorted(grouped.items(), key=lambda pair: category_label(pair[0]))
    }


def render_article_group(category_id: str, rows: list[dict[str, Any]]) -> str:
    items_html = ""
    tag_counter: Counter[str] = Counter()
    for row in rows:
        url = row.get("url") or ""
        title = row.get("title") or row.get("article_id")
        title_html = (
            f"<a href='{html_escape(url)}' target='_blank' rel='noreferrer'>{html_escape(title)}</a>"
            if url
            else html_escape(title)
        )
        tags = row.get("tags", []) if isinstance(row.get("tags"), list) else []
        for tag in tags:
            if isinstance(tag, str) and tag:
                tag_counter[tag] += 1
        tag_html = "".join(pill(tag, "pattern") for tag in tags[:6])
        summary = row.get("summary") or ""
        article_id = str(row.get("article_id") or "")
        issue_url = row.get("issue_url") or ""
        manage_html = (
            f"<a class='article-action-link' href='{html_escape(issue_url)}' target='_blank' rel='noreferrer'>管理 issue</a>"
            if issue_url
            else f"<a class='article-action-link' href='{html_escape(article_management_issue_url(row))}' target='_blank' rel='noreferrer'>新建管理 issue</a>"
        )
        items_html += f"""
          <li class="category-article">
            <div>
              <h4>{title_html}</h4>
              <p>{html_escape(summary)}</p>
              <div class="pill-row">{tag_html}</div>
              <div class="article-actions">
                <button type="button" class="article-action" data-copy="/archive {html_escape(article_id)} 原因：">归档请求</button>
                <button type="button" class="article-action" data-copy="/exclude {html_escape(article_id)} 原因：">排除这篇</button>
                {manage_html}
              </div>
            </div>
            <div class="article-meta">
              <span>{html_escape(article_id)}</span>
              <span>{html_escape(display_status(row.get("importance", "unknown")))}</span>
              <span>{html_escape(display_status(row.get("content_status", "unknown")))}</span>
            </div>
          </li>
        """
    preview_tag_html = "".join(
        pill(item["id"], "pattern")
        for item in top_items(tag_counter, 5)
    )
    return f"""
      <details class="article-group" id="articles-{html_escape(slug(category_id))}">
        <summary class="category-summary">
          <span class="category-summary-main">
            <span>
              <span class="eyebrow">分类文章</span>
              <span class="category-title">{html_escape(category_label(category_id))}</span>
            </span>
            <span class="count">{len(rows)} 篇文章</span>
          </span>
          <span class="category-intro">{html_escape(category_description(category_id))}</span>
          <span class="pill-row category-preview-tags">{preview_tag_html}</span>
        </summary>
        <ul class="category-article-list">{items_html}</ul>
      </details>
    """


def build_report_html() -> str:
    init_state()
    status = knowledge_status()
    reviews_doc = br.read_json(TOPIC_REVIEWS_PATH, default={"reviews": []})
    frameworks_doc = br.read_json(THINKING_FRAMEWORKS_PATH, default={"frameworks": []})
    templates_doc = br.read_json(EXPRESSION_TEMPLATES_PATH, default={"templates": []})

    titles = article_title_map()
    reviews = [
        item
        for item in reviews_doc.get("reviews", [])
        if isinstance(item, dict)
    ]
    articles = load_article_view_rows()
    frameworks = [
        item
        for item in frameworks_doc.get("frameworks", [])
        if isinstance(item, dict)
    ]
    templates = [
        item
        for item in templates_doc.get("templates", [])
        if isinstance(item, dict)
    ]

    review_cards = "".join(render_review_card(review, titles) for review in reviews)
    article_groups = group_article_rows_by_category(articles)
    article_group_html = "".join(
        render_article_group(category_id, rows)
        for category_id, rows in article_groups.items()
    )

    framework_cards = []
    for item in frameworks:
        dims = item.get("dimensions", [])
        dim_html = "".join(f"<li>{html_escape(dim)}</li>" for dim in dims if isinstance(dim, str))
        best_for = item.get("best_for", []) if isinstance(item.get("best_for"), list) else []
        best_for_html = "".join(pill(category_label(value), "category") for value in best_for[:6])
        framework_cards.append(
            f"""
            <article class="card framework-card thinking-framework">
              <div class="type-row">
                <span class="type-badge thinking">思考框架</span>
                <span class="status-badge">{html_escape(display_status(item.get("status", "drafted")))}</span>
              </div>
              <h3>{html_escape(item.get("name", ""))}</h3>
              <p>{html_escape(item.get("purpose", ""))}</p>
              <div class="pill-row">{best_for_html}</div>
              <ul>{dim_html}</ul>
            </article>
            """
        )
    for item in templates:
        steps = item.get("structure", [])
        step_html = "".join(f"<li>{html_escape(step)}</li>" for step in steps if isinstance(step, str))
        framework_cards.append(
            f"""
            <article class="card framework-card expression-template">
              <div class="type-row">
                <span class="type-badge expression">表达模板</span>
                <span class="status-badge">{html_escape(display_status(item.get("status", "drafted")))}</span>
              </div>
              <h3>{html_escape(item.get("name", ""))}</h3>
              <p>{html_escape(item.get("use_case", ""))}</p>
              <ul>{step_html}</ul>
            </article>
            """
        )

    stats = [
        ("文章", status["article_count"]),
        ("已抽取", status["articles_with_extraction"]),
        ("已分类", status["articles_with_classification"]),
        ("综述", status["review_count"]),
    ]
    stat_html = "".join(
        f"<div class='stat'><span>{html_escape(label)}</span><strong>{html_escape(value)}</strong></div>"
        for label, value in stats
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>知识系统</title>
  <style>
    :root {{
      --ink: #202124;
      --muted: #5f6368;
      --line: #d9dde3;
      --surface: #ffffff;
      --wash: #f6f7f9;
      --accent: #0f766e;
      --accent-2: #9a3412;
      --accent-3: #1d4ed8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--wash);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }}
    a {{ color: var(--accent-3); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .shell {{ max-width: 1180px; margin: 0 auto; padding: 28px 20px 56px; }}
    header {{ padding: 18px 0 20px; }}
    h1 {{ font-size: 42px; line-height: 1.08; margin: 0 0 12px; letter-spacing: 0; }}
    h2 {{ font-size: 26px; margin: 44px 0 16px; letter-spacing: 0; }}
    h3 {{ font-size: 18px; margin: 0 0 10px; letter-spacing: 0; }}
    h4 {{ margin: 12px 0 8px; font-size: 14px; }}
    p {{ margin: 0 0 12px; }}
    ul {{ margin: 0; padding-left: 20px; }}
    nav {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0 0; }}
    nav a {{
      border: 1px solid var(--line);
      background: var(--surface);
      border-radius: 6px;
      padding: 8px 11px;
      color: var(--ink);
      font-size: 14px;
    }}
    .lede {{ max-width: 820px; color: var(--muted); font-size: 17px; }}
    .stats {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 24px 0; }}
    .stat {{ background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .stat span {{ display: block; color: var(--muted); font-size: 13px; }}
    .stat strong {{ display: block; font-size: 27px; margin-top: 3px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    .card, .article-row, .article-group {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      overflow-wrap: anywhere;
    }}
    .card-head {{ display: flex; justify-content: space-between; gap: 14px; align-items: flex-start; }}
    .eyebrow {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0; margin-bottom: 4px; }}
    .count {{ white-space: nowrap; color: var(--accent); font-weight: 700; font-size: 13px; }}
    .muted {{ color: var(--muted); }}
    .review-question {{ color: var(--accent-2); font-weight: 700; }}
    .review-card > summary {{
      list-style: none;
      color: inherit;
      font-weight: inherit;
      cursor: pointer;
    }}
    .review-card > summary::-webkit-details-marker {{ display: none; }}
    .review-summary {{
      display: grid;
      gap: 10px;
      position: relative;
      padding-right: 76px;
    }}
    .review-summary::after {{
      content: "展开";
      position: absolute;
      top: 0;
      right: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 4px 9px;
      color: var(--accent-3);
      background: #fafafa;
      font-size: 12px;
      font-weight: 700;
    }}
    .review-card[open] .review-summary::after {{ content: "收起"; }}
    .review-summary-main {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }}
    .review-title {{ display: block; font-size: 18px; font-weight: 800; color: var(--ink); }}
    .review-meta {{ display: inline-flex; align-items: center; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
    .review-preview {{ display: block; color: var(--muted); }}
    .review-preview-tags {{ margin: 0; }}
    .review-body {{ margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--line); }}
    .deep-review {{
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      margin: 14px 0 16px;
      padding: 16px 0 6px;
    }}
    .deep-review .lead {{ color: var(--muted); font-size: 15px; }}
    .thesis {{
      border-left: 3px solid var(--accent);
      background: #f1f8f6;
      padding: 10px 12px;
      margin: 12px 0 16px;
    }}
    .thesis strong {{ display: block; margin-bottom: 4px; }}
    .deep-section {{ margin: 16px 0; }}
    .deep-section h4 {{ font-size: 15px; color: var(--accent-2); }}
    .pill-row, .concept-strip {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0; }}
    .pill {{ display: inline-flex; align-items: center; border-radius: 999px; border: 1px solid var(--line); padding: 3px 8px; font-size: 12px; background: #fafafa; }}
    .pill.pattern {{ border-color: #bfd4d1; color: var(--accent); }}
    .pill.concept {{ border-color: #c7d2fe; color: var(--accent-3); }}
    .pill.category {{ border-color: #fed7aa; color: var(--accent-2); }}
    .framework-card {{ border-top: 4px solid var(--line); }}
    .thinking-framework {{ border-top-color: var(--accent); }}
    .expression-template {{ border-top-color: var(--accent-3); }}
    .type-row {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 10px; }}
    .type-badge, .status-badge {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      border: 1px solid var(--line);
      padding: 3px 9px;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .type-badge.thinking {{ color: var(--accent); border-color: #99d2c8; background: #eef9f7; }}
    .type-badge.expression {{ color: var(--accent-3); border-color: #bfdbfe; background: #eff6ff; }}
    .status-badge {{ color: var(--muted); background: #fafafa; }}
    .split {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    details {{ margin-top: 12px; }}
    summary {{ cursor: pointer; color: var(--accent-3); font-weight: 600; }}
    .article-list {{ display: grid; gap: 12px; }}
    .article-group {{ margin-bottom: 14px; }}
    .article-group > summary {{
      list-style: none;
      color: inherit;
      font-weight: inherit;
    }}
    .article-group > summary::-webkit-details-marker {{ display: none; }}
    .category-summary {{
      display: grid;
      gap: 8px;
      position: relative;
      padding-right: 76px;
    }}
    .category-summary::after {{
      content: "展开";
      position: absolute;
      top: 0;
      right: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 4px 9px;
      color: var(--accent-3);
      background: #fafafa;
      font-size: 12px;
      font-weight: 700;
    }}
    .article-group[open] .category-summary::after {{ content: "收起"; }}
    .category-summary-main {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }}
    .category-title {{ display: block; font-size: 18px; font-weight: 800; color: var(--ink); }}
    .category-intro {{ display: block; color: var(--muted); max-width: 820px; }}
    .category-preview-tags {{ margin: 0; }}
    .category-article-list {{ list-style: none; padding-left: 0; display: grid; gap: 12px; margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--line); }}
    .category-article {{ display: grid; grid-template-columns: minmax(0, 1fr) 130px; gap: 16px; border-top: 1px solid var(--line); padding-top: 12px; }}
    .category-article:first-child {{ border-top: 0; padding-top: 0; }}
    .article-row {{ display: grid; grid-template-columns: minmax(0, 1fr) 160px; gap: 18px; }}
    .article-meta {{ display: flex; flex-direction: column; gap: 8px; color: var(--muted); font-size: 13px; }}
    .article-actions {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; align-items: center; }}
    .article-action, .article-action-link {{
      appearance: none;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fafafa;
      color: var(--accent-3);
      padding: 5px 9px;
      font: inherit;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
    }}
    .article-action:hover, .article-action-link:hover {{ background: #eef4ff; text-decoration: none; }}
    .article-action.copied {{ color: var(--accent); border-color: #99d2c8; background: #eef9f7; }}
    .article-action-muted {{ color: var(--muted); font-size: 12px; }}
    .note {{ color: var(--muted); max-width: 840px; }}
    @media (max-width: 860px) {{
      .stats, .grid, .split, .article-row, .category-article {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 32px; }}
      .review-summary {{ padding-right: 0; padding-bottom: 34px; }}
      .review-summary::after {{ top: auto; bottom: 0; left: 0; right: auto; }}
      .review-summary-main {{ flex-direction: column; }}
      .review-meta {{ justify-content: flex-start; }}
      .category-summary {{ padding-right: 0; padding-bottom: 34px; }}
      .category-summary::after {{ top: auto; bottom: 0; left: 0; right: auto; }}
      .category-summary-main {{ flex-direction: column; }}
      .article-meta {{ flex-direction: row; flex-wrap: wrap; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <p class="eyebrow">文章驱动的知识抽象</p>
      <h1>知识系统</h1>
      <p class="lede">这是一个非贝叶斯的文章库视图：文章先进入分类、标签和概念层，再形成动态主题综述，最后服务于判断、理论和表达框架。</p>
      <nav>
        <a href="#reviews">主题综述</a>
        <a href="#articles">分类文章</a>
        <a href="#frameworks">表达框架</a>
      </nav>
    </header>

    <section class="stats">{stat_html}</section>

    <section id="reviews">
      <h2>主题综述</h2>
      <p class="note">主题综述按主分类组织文章，并尝试回答该类材料背后的重复结构、张力和下一步问题。当前内容仍是草稿，等待分类与文章归组人审。</p>
      <div class="grid">{review_cards if review_cards else '<p>暂无综述。</p>'}</div>
    </section>

    <section id="articles">
      <h2>分类文章</h2>
      <p class="note">文章按主分类聚合；新增文章完成抽取后先由 LLM 分到一个主分类，再进入对应分类组。</p>
      <div class="article-list">{article_group_html if article_group_html else '<p>暂无文章。</p>'}</div>
    </section>

    <section id="frameworks">
      <h2>表达框架</h2>
      <p class="note">这一层负责把积累的知识转化成思考、写作和论证结构。</p>
      <div class="grid">{''.join(framework_cards) if framework_cards else '<p>暂无框架。</p>'}</div>
    </section>
  </main>
  <script>
    document.querySelectorAll('[data-copy]').forEach((button) => {{
      button.addEventListener('click', async () => {{
        const text = button.getAttribute('data-copy') || '';
        try {{
          await navigator.clipboard.writeText(text);
          const oldText = button.textContent;
          button.textContent = '已复制命令';
          button.classList.add('copied');
          window.setTimeout(() => {{
            button.textContent = oldText;
            button.classList.remove('copied');
          }}, 1600);
        }} catch (error) {{
          window.prompt('复制这条命令到文章 issue 评论中执行：', text);
        }}
      }});
    }});
  </script>
</body>
</html>
"""


def build_report(output_path: Path = REPORT_PATH) -> dict[str, Any]:
    init_state()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = build_report_html()
    output_path.write_text(html, encoding="utf-8")
    br.append_jsonl(
        br.CHANGE_LOG_PATH,
        {
            "timestamp": now(),
            "event": "knowledge_build_report",
            "output_path": br.relpath_from_root(output_path),
        },
    )
    return {
        "ok": True,
        "path": br.relpath_from_root(output_path),
        "bytes": len(html.encode("utf-8")),
    }


def print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Non-Bayesian article-to-knowledge abstraction pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_cmd = subparsers.add_parser("init", help="Create taxonomy, concept, review, judgment, theory, and framework files")
    init_cmd.add_argument("--force", action="store_true", help="Overwrite baseline taxonomy/framework files")

    extract = subparsers.add_parser("extract-article", help="Create extraction.json for one article")
    extract.add_argument("--article-id", required=True)
    extract.add_argument("--force", action="store_true")

    classify = subparsers.add_parser("classify-article", help="Create classification.json for one article")
    classify.add_argument("--article-id", required=True)
    classify.add_argument("--force", action="store_true")

    concepts = subparsers.add_parser("update-concepts", help="Update article and global concept files")
    concepts.add_argument("--article-id", required=True)
    concepts.add_argument("--force", action="store_true")

    subparsers.add_parser("rebuild-concepts", help="Rebuild global concepts from existing classifications")

    reviews = subparsers.add_parser("build-reviews", help="Build living topic reviews")
    reviews.add_argument("--category")

    deep_review = subparsers.add_parser("draft-deep-review", help="Use an LLM to rewrite one topic review as a deep analysis article")
    deep_review.add_argument("--category", required=True)
    deep_review.add_argument("--model")

    frameworks = subparsers.add_parser("build-frameworks", help="Generate thinking frameworks and expression templates from topic reviews")
    frameworks.add_argument("--force", action="store_true")
    frameworks.add_argument("--model")

    refresh_reviews = subparsers.add_parser("refresh-llm-reviews", help="Rewrite non-LLM topic reviews with the configured LLM")
    refresh_reviews.add_argument("--force", action="store_true")
    refresh_reviews.add_argument("--model")

    report = subparsers.add_parser("build-report", help="Generate the static non-Bayesian knowledge report")
    report.add_argument("--output", default=str(REPORT_PATH))

    lifecycle = subparsers.add_parser("set-article-state", help="Set article lifecycle state: active, archived, or excluded")
    lifecycle.add_argument("--article-id", required=True)
    lifecycle.add_argument("--state", required=True, choices=sorted(VALID_ARTICLE_STATES))
    lifecycle.add_argument("--reason", default="")
    lifecycle.add_argument("--actor", default="")

    run = subparsers.add_parser("run-mvp", help="Run extraction, classification, concept update, and reviews")
    run.add_argument("--limit", type=int)
    run.add_argument("--force", action="store_true")
    run.add_argument("--skip-frameworks", action="store_true")

    subparsers.add_parser("status", help="Show non-Bayesian knowledge pipeline coverage")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        print_json(init_state(force=args.force))
        return 0
    if args.command == "extract-article":
        print_json(extract_article(args.article_id, force=args.force))
        return 0
    if args.command == "classify-article":
        print_json(classify_article(args.article_id, force=args.force))
        return 0
    if args.command == "update-concepts":
        print_json(update_concepts(args.article_id, force=args.force))
        return 0
    if args.command == "rebuild-concepts":
        print_json(rebuild_concepts())
        return 0
    if args.command == "build-reviews":
        print_json(build_topic_reviews(category=args.category))
        return 0
    if args.command == "draft-deep-review":
        print_json(draft_deep_review(category=args.category, model=args.model))
        return 0
    if args.command == "build-frameworks":
        print_json(build_frameworks(force=args.force, model=args.model))
        return 0
    if args.command == "refresh-llm-reviews":
        print_json(refresh_llm_reviews(force=args.force, model=args.model))
        return 0
    if args.command == "build-report":
        print_json(build_report(output_path=Path(args.output)))
        return 0
    if args.command == "set-article-state":
        print_json(
            set_article_lifecycle(
                article_id=args.article_id,
                state=args.state,
                reason=args.reason,
                actor=args.actor,
            )
        )
        return 0
    if args.command == "run-mvp":
        print_json(run_mvp(limit=args.limit, force=args.force, skip_frameworks=args.skip_frameworks))
        return 0
    if args.command == "status":
        print_json(knowledge_status())
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

# 非贝叶斯知识系统

这份文档描述新的知识流水线。它和原来的 Bayesian reader 并行存在，不删除旧的 `hypotheses.json`、`verification.json` 或 posterior 计算逻辑。

新的中心不是“证据更新概率”，而是：

```text
GitHub issue / URL inbox
  -> 文章库
  -> 抽取、LLM 分类、标签
  -> 动态主题综述
  -> 判断与理论
  -> 思考 / 表达框架
```

主要界面是：

```text
docs/knowledge.html
```

## 当前 MVP

入口脚本是：

```bash
python scripts/knowledge_pipeline.py
```

目前已经实现五层：

1. **文章抽取**
   - 写入 `knowledge_state/articles/<article_id>/extraction.json`
   - 优先复用已有 `claims.json`
   - 如果没有 claims，则从 `canonical_text.txt` 里做规则化句子抽取

2. **分类与标签**
   - 写入 `knowledge_state/articles/<article_id>/classification.json`
   - 生成 `primary_category`、`secondary_categories`、`tags`、`source_type`、`importance`
   - 当前是关键词规则草稿，后续可替换成 LLM 草稿 + 人审

3. **后台概念抽象**
   - 写入单篇文章的 `knowledge_state/articles/<article_id>/concepts.json`
   - 更新全局 `knowledge_state/concepts/concepts.json`
   - 只作为综述和判断层的内部特征，不再作为前台“概念库”展示

4. **动态主题综述**
   - 写入 `knowledge_state/reviews/topic_reviews.json`
   - 按 `primary_category` 聚合文章
   - 输出 `central_question`、`synthesis`、`analytical_takeaways`、`dominant_patterns`、`case_analysis`、`tensions`、`next_questions`

5. **静态知识报告**
   - 写入 `docs/knowledge.html`
   - 包含主题综述、分类文章、表达框架三个前台视图
   - 不覆盖旧的 Bayesian 报告 `docs/index.html`

## 目录结构

全局状态：

```text
knowledge_state/
  taxonomy/
    categories.json
    tags.json

  concepts/
    concepts.json
    concept_relations.json

  reviews/
    topic_reviews.json

  judgments/
    judgments.json

  theories/
    theories.json

  frameworks/
    thinking_frameworks.json
    expression_templates.json
```

单篇文章新增文件：

```text
knowledge_state/articles/<article_id>/
  extraction.json
  classification.json
  concepts.json
```

旧文件继续保留：

```text
record.json
canonical_text.txt
claims.json
verification.json
```

## 常用命令

初始化新状态文件：

```bash
python scripts/knowledge_pipeline.py init
```

强制刷新 taxonomy、frameworks 等基线文件：

```bash
python scripts/knowledge_pipeline.py init --force
```

处理单篇文章：

```bash
python scripts/knowledge_pipeline.py extract-article --article-id <id>
python scripts/knowledge_pipeline.py classify-article --article-id <id>
python scripts/knowledge_pipeline.py update-concepts --article-id <id>
```

跑完整 MVP：

```bash
python scripts/knowledge_pipeline.py run-mvp
```

强制全量重建，包括概念索引、主题综述和报告：

```bash
python scripts/knowledge_pipeline.py run-mvp --force
```

只重建后台概念抽象：

```bash
python scripts/knowledge_pipeline.py rebuild-concepts
```

只重建主题综述：

```bash
python scripts/knowledge_pipeline.py build-reviews
```

使用 LLM 把某个主题综述改写成深度分析文章：

```bash
python scripts/knowledge_pipeline.py draft-deep-review --category ai_agent
```

这个命令需要 `OPENROUTER_API_KEY`。没有 key 时，系统仍会使用 deterministic fallback 生成可运行的 `deep_review` 结构。
脚本会自动读取当前目录、项目根目录或项目上一级目录中的 `.env`，格式为：

```bash
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_DEFAULT_MODEL=openai/gpt-5.4
```

从主题综述生成思考框架和表达模板：
```bash
python scripts/knowledge_pipeline.py build-frameworks
```

`build-frameworks` 会读取 `topic_reviews.json`，优先用 LLM 生成 `thinking_frameworks.json` 和 `expression_templates.json`。结果会记录 `source_hash` 与 `prompt_version`；综述输入没有变化时会直接复用，避免重复调用 LLM。需要强制重写时使用：
```bash
python scripts/knowledge_pipeline.py build-frameworks --force
```

只重建静态报告：

```bash
python scripts/knowledge_pipeline.py build-report
```

查看覆盖率：

```bash
python scripts/knowledge_pipeline.py status
```

## 状态语义

### Article Extraction

`extraction.json` 是文章级知识快照：

```json
{
  "article_id": "...",
  "extraction_status": "completed",
  "summary": "...",
  "main_points": [],
  "entities": [],
  "claims": [],
  "important_quotes": [],
  "open_questions": []
}
```

这一层不是事实核验层。它记录“文章说了什么”和系统从文章中抽出了什么。

### Classification

`classification.json` 决定文章进入哪个主题综述：

```json
{
  "article_id": "...",
  "classification_status": "drafted",
  "primary_category": "ai_agent",
  "secondary_categories": [],
  "tags": ["agent-runtime", "workflow-orchestration"],
  "source_type": "media",
  "importance": "medium"
}
```

当前分类是草稿。后续应加入人工 approve / override，避免分类和概念污染。

### 后台概念抽象

概念由稳定标签和重复证据提升而来，但不再作为独立前台概念库。它只服务于综述生成、判断抽象和后续人工治理：

```json
{
  "concept_id": "agent_runtime",
  "name": "Agent Runtime",
  "definition": "面向 agent 的持久执行层，包括 session、工具、沙箱、恢复机制和审计轨迹。",
  "status": "drafted",
  "related_categories": ["ai_agent"],
  "related_tags": ["agent-runtime"],
  "supporting_articles": ["..."]
}
```

后台概念治理后续应支持：

```text
merge
split
rename
alias
deprecate
approve
```

### Topic Reviews

`topic_reviews.json` 是当前最重要的综合层：

```json
{
  "review_id": "ai_agent_living_review",
  "central_question": "...",
  "synthesis": "...",
  "deep_review": {
    "title": "...",
    "dek": "...",
    "core_thesis": "...",
    "sections": [],
    "insights": [],
    "essential_cognition": "...",
    "misreadings_to_avoid": []
  },
  "analytical_takeaways": [],
  "dominant_patterns": [],
  "case_analysis": [],
  "tensions": [],
  "implications": [],
  "next_questions": []
}
```

主题综述不应该只是罗列文章观点。它要回答：

- 这一类文章共同指向什么结构？
- 重复模式是什么？
- 哪些案例只是单点信号？
- 哪些概念需要提升、合并或重命名？
- 当前解释有什么反证、张力和不确定性？

当前综述的主输出是 `deep_review`：前台只渲染 `title`、`dek`、`core_thesis` 和 `sections`，它们共同构成一篇可阅读的深度分析文章。`insights`、`essential_cognition`、`misreadings_to_avoid` 是内部结构化元数据，用于后续判断层和表达层，不应该直接作为最终文章正文展示。`analytical_takeaways`、`dominant_patterns`、`case_analysis` 等字段只作为证据附录，不再作为用户阅读综述的主体。

规则版 `deep_review` 必须读取每篇文章的 `canonical_text.txt`，并在 `source_trace` 中记录全文长度和证据句数量。它只能生成“全文证据驱动的草稿综述”；真正的深度分析应使用 `draft-deep-review` 调用 LLM 或人工编辑完成。

## 后续设计步骤

1. **LLM 草稿抽取与分类**
   - 保留当前 deterministic fallback
   - 增加 schema 校验
   - 输出仍然是 `extraction.json` 和 `classification.json`

2. **人工校正层**
   - 支持分类、标签、后台概念抽象的 approve / override
   - 人审后再进入更高层 judgment / theory

3. **Judgment 层**
   - 从一个或多个 topic review 中形成明确判断
   - 使用 `emerging`、`plausible`、`strong`、`settled`、`contested` 等解释性置信等级
   - 不再使用 posterior probability 作为中心指标

4. **Theory 层**
   - 把多个判断组织成更高层解释模型
   - 保留适用边界、反例和不确定性

5. **Expression 层**
   - 从 review / judgment / theory 生成写作提纲、论证框架和表达模板
   - 当前由 `build-frameworks` 从主题综述生成，LLM 输出后仍是 `drafted`，等待人工确认后再提升为 `reviewed` 或 `approved`

6. **报告 UI**
   - 当前已有 `docs/knowledge.html`
   - 后续可增加筛选、搜索、单分类详情页、单综述详情页

## 与旧 Bayesian 系统的关系

旧系统不是废弃，而是降级为历史实现和证据来源之一。

| 旧对象 | 新角色 |
|---|---|
| GitHub issue ingest | 保留，作为文章入口 |
| `canonical_text.txt` | 保留，作为文章库基础文本 |
| `claims.json` | 复用为 article extraction 输入 |
| `verification.json` | 可作为 source grounding / evidence trace |
| `hypotheses.json` | 可迁移为 judgments 或 theories 的素材 |
| `posterior_probability` | 不再作为核心组织原则 |
| `synthesis_state.json` | 由 reviews、judgments、theories 取代 |

新系统仍然重视可追溯性，只是不再以贝叶斯后验计算作为知识组织中心。

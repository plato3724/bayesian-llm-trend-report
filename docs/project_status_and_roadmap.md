# 项目状态与后续优化路线图

更新日期：2026-04-13

本文档用于记录当前项目已经完成的能力、现阶段设计边界，以及后续建议推进的优化阶段，避免后续迭代时反复重新梳理上下文。

## 1. 项目当前目标

项目的核心目标是把“文章收集 -> 全文抓取 -> claim 提取 -> 主源核实 -> 趋势后验更新 -> 报告展示”做成一条尽可能自动化、但仍保留必要风控的知识生产链路。

当前系统不是单纯的文章摘要器，而是：

- 先抓取全文
- 再抽取结构化 claims
- 再做 verification
- 只有通过验证并映射到趋势假设的证据，才进入 Bayesian posterior
- 最终在静态报告中呈现趋势、证据、工具索引和候选假设

## 2. 当前已完成的阶段

### 2.1 基础数据与框架层

已完成：

- 建立 `knowledge_state/` 作为统一状态目录
- 建立 article record / claims / verification / hypotheses / synthesis 等核心状态文件
- 建立 `scripts/bayesian_reader.py` 作为主入口 CLI
- 建立趋势假设 `hypotheses.json`
- 建立后验重算、状态刷新、报告构建、变更日志等基础流程

意义：

- 项目已经不是一次性脚本，而是具备持续运行能力的状态机
- 后续新增自动化、候选假设、报告优化，都是建立在这套状态模型上

### 2.2 GitHub Actions 自动化主链路

已完成：

- Phase A: `ingest.yml`
  - 从 GitHub issue 导入文章链接
  - 抓取网页全文
  - 写入 `canonical_text.txt` 和 `record.json`
- Phase B: `draft-claims.yml`
  - 调用 OpenRouter / `openai/gpt-5.4`
  - 自动生成 claims 草稿并保存为正式 `claims.json`
- Phase C: `draft-verification.yml`
  - 自动生成 `verification_draft.json` 和 `approval.json`
  - 根据 guard 决定自动 apply 或进入 held
- Held review: `review-held.yml`
  - 支持在 GitHub issue 评论里执行 `/approve`、`/approve-safe`、`/reject`
  - 无需本地手改 `approval.json`

意义：

- 已实现“issue 加文章 -> 自动推进到 verification/apply/report”的主链路
- 人工复核已从本地 CLI 迁移到 GitHub comment，降低了操作复杂度

### 2.3 LLM 使用与模型管理

已完成：

- 统一通过 OpenRouter 调用模型
- 默认模型固定为 `openai/gpt-5.4`
- 支持通过 GitHub repo variable / workflow input 覆盖模型
- claims / verification 使用同一套客户端逻辑与 JSON 校验

意义：

- 默认模型稳定
- 模型配置仍可修改，不会把流程写死在单一 provider 细节上

### 2.4 报告展示优化

已完成：

- 趋势假设卡片支持展开查看具体支撑证据
- 卡片可跳转到对应文章证据区
- 每个核心假设都有独立详情页 `docs/hypotheses/*.html`
- `hypotheses.json` 已支持 `theory` 字段
- 首页可展示“本次新增”高亮，且高亮已改为按最新 `apply_verification` 精准定位，而不是粗糙时间窗口
- 首页已支持展示 held / staged / pending 等状态

意义：

- 报告从“概率卡片展示”升级成“可追踪证据来源的阅读界面”
- 用户可以直接看到新上传文章推动了哪个趋势假设

### 2.5 工具索引自动汇总

已完成：

- 工具索引不再只依赖手工维护的 `tool_index`
- 系统会从已核实文章中的 `tool` 类 claim 自动补充工具条目
- 手工维护条目优先，自动汇总作为增量补全

意义：

- 新文章里出现的工具/项目可以自动出现在首页工具索引中
- 减少手工同步工具卡的成本

### 2.6 假设生长机制：第一阶段

已完成：

- 新增 `knowledge_state/candidate_hypotheses.json`
- 新增 `build-candidates` 流程
- 从“已核实但未映射到正式 hypothesis”的 evidence 中生成候选假设簇
- 首页新增“候选假设”区
- 候选卡展示 `candidate_id`

当前设计原则：

- candidate 只是观察层，不进入 posterior
- candidate 不会自动变成正式 hypothesis
- 目的是提示“框架可能需要生长”

### 2.7 假设生长机制：第二阶段

已完成：

- 新增 GitHub comment review 流程：`review-candidate.yml`
- 支持命令：
  - `/promote-candidate <candidate_id>`
  - `/reject-candidate <candidate_id>`
- 候选状态已支持：
  - `observed`
  - `emerging`
  - `promoted`
  - `rejected`
- `promote` 可以：
  - 创建正式 hypothesis
  - 把 candidate 对应 evidence 挂到新 hypothesis
  - 重算 posterior 并重建报告
- `reject` 可以：
  - 拒绝该 candidate
  - 避免其反复作为开放候选重新出现

意义：

- 系统已经具备“从文章事实中长出新假设，并通过 GitHub 完成人工升格”的闭环

## 3. 当前系统的关键边界

这些边界是当前设计刻意保留的，不是缺陷。

### 3.1 正式 hypothesis 不会自动创建

当前 candidate 只能自动生成，不能自动变成正式 hypothesis。

原因：

- 单篇文章或单一来源很容易带来噪音
- 正式 hypothesis 一旦进入 posterior，会影响整个趋势层
- 候选层的作用就是把“可能的新趋势”先放到观察层

### 3.2 held 仍然存在

即使主链路已经自动化，Phase C 的 `held` 仍是必要机制。

原因：

- 某些文章只适合保留事实，不适合直接推动趋势判断
- `cross_domain`、`band_crossing`、弱主源、营销口径等情况仍应人工确认

### 3.3 候选假设目前更像“证据簇”，不是成熟理论

当前 candidate 的生成逻辑主要基于“未映射但已核实证据”的聚类与关键词抽取。

因此：

- 它适合发现框架空白
- 但还不适合自动生成高质量理论阐述

## 4. 当前项目完成度判断

如果把项目拆成三层：

1. 基础状态机与 Bayesian 主链路
2. GitHub 自动化与人工复核闭环
3. 假设生长与框架扩张

那么当前完成度可以判断为：

- 第 1 层：已完成
- 第 2 层：已完成可用版本
- 第 3 层：已完成前两阶段，进入可用但仍需增强的状态

也就是说：

- 这个项目已经不是原型，而是可持续运行的系统
- 但离“长期可维护、可扩张、可观察”的成熟系统还有后续阶段要做

## 5. 建议的后续优化阶段

以下阶段按优先级排序。

### 阶段 3：候选假设推荐与提醒机制

目标：

- 不再要求用户先打开报告找 `candidate_id`
- 系统应主动提醒哪些 candidate 值得 review

建议内容：

- 在报告候选卡片中直接展示可复制命令
  - `/promote-candidate <id>`
  - `/reject-candidate <id>`
- 在候选卡中增加推荐动作
  - 推荐 promote
  - 推荐 reject
- 在检测到新 candidate 时，自动在 GitHub issue 或固定 review issue 中评论提示

价值：

- 降低 candidate review 的操作成本
- 让假设生长流程更接近“被系统提示，再由人判断”

### 阶段 4：候选假设质量增强

目标：

- 提升 candidate 的可读性与解释质量
- 避免 candidate 只是一组关键词堆叠

建议内容：

- 为 candidate 增加更稳定的聚类与归一化策略
- 对重复出现的实体、技术路线、产品族做更好的命名归并
- 为 candidate 生成更高质量的 `statement` / `rationale`
- 让 `observed` 与 `emerging` 的升级条件更明确，例如：
  - 支持文章数
  - 来源多样性
  - 跨时间重复出现程度

价值：

- 让 candidate 更像“待升格假设”，而不是临时 evidence bucket

### 阶段 5：候选假设详情页

目标：

- 让 candidate 也具备可展开阅读、可追踪证据、可理解理论方向的详情页

建议内容：

- 为 candidate 生成独立静态详情页
- 展示：
  - candidate statement
  - candidate rationale
  - 支撑文章
  - 核实条目
  - 推荐动作
  - 与现有正式 hypotheses 的区别

价值：

- 降低 promote 决策的信息切换成本
- 让用户在 GitHub review 前先通过报告阅读 candidate 全貌

### 阶段 6：候选 review 的操作体验优化

目标：

- 进一步减少 GitHub comment 的记忆负担

建议内容：

- 在 candidate 提示中直接给出复制即用的命令
- 支持从固定 issue 管理 candidate review
- 对错误命令给出更清楚的回写提示
- 当 `candidate_id` 不存在时，返回接近匹配项

价值：

- 让 GitHub review 体验从“可用”提升到“顺手”

### 阶段 7：报告与状态可观察性增强

目标：

- 让系统更容易排查当前停在哪一步、为什么停、影响了哪些趋势

建议内容：

- 增加自动化状态面板
  - 今日新 ingest 数量
  - 待 claims 数量
  - 待 verification 数量
  - held 数量
  - open candidate 数量
- 给趋势卡补充“最近一次被哪篇文章推动”
- 给 candidate 补充“最近一次支持来源”
- 给 held 增加更统一的 reason taxonomy

价值：

- 降低维护与排障成本
- 让项目逐步从“功能链路”升级为“可运营系统”

### 阶段 8：更严格的数据质量控制

目标：

- 控制 candidate 和 hypothesis 的质量漂移

建议内容：

- 对 evidence 聚类与 hypothesis promote 增加更明确的阈值
- 对低质量主源、营销稿、二手转述做更严格过滤
- 在 promote 前要求最小支持条件
  - 至少 N 条 evidence
  - 至少 M 篇文章
  - 至少 K 个独立来源

价值：

- 防止 formal hypothesis 被噪音污染
- 保持趋势层的可信度

## 6. 当前不建议优先做的事情

以下事项目前不建议优先推进。

### 6.1 不建议直接让系统自动 promote candidate

原因：

- 风险太高
- 一旦自动写入 formal hypothesis，会直接污染 posterior
- 当前 candidate 的质量还不够稳定到可以无人审批

### 6.2 不建议现在就做 candidate merge into existing hypothesis

原因：

- 当前 candidate 机制的目标是发现框架空白
- 过早引入 merge，容易让很多本应独立的新趋势被并回旧 hypothesis
- 会削弱“框架真正生长”的意义

如果后续真要做 merge，应该作为单独阶段，在明确规则后谨慎引入。

### 6.3 不建议把页面改成复杂前端应用

原因：

- 当前静态 HTML 已能覆盖大多数阅读与 review 场景
- 项目瓶颈仍在知识建模与流程控制，不在前端框架能力

## 7. 建议的近期行动顺序

如果继续优化，建议按下面顺序推进：

1. 先做“候选假设推荐与提醒机制”
2. 再做“candidate 质量增强”
3. 再做“candidate 详情页”
4. 再做“状态面板与可观察性”
5. 最后再考虑更复杂的 review 语义，例如 merge

## 8. 当前项目结论

截至 2026-04-13，项目已经完成：

- 自动 ingest
- 自动 claims
- 自动 verification/stage/apply
- GitHub 内人工 held review
- 报告证据可视化
- 工具索引自动汇总
- 候选假设生成
- GitHub 内 candidate promote/reject

因此当前系统已经具备：

- 自动吸收文章
- 自动更新趋势
- 在发现框架空白时产生候选假设
- 通过 GitHub 评论完成关键人工决策

后续优化重点不再是“把主链路做通”，而是：

- 让候选假设更好用
- 让系统提示更明确
- 让状态更可观察
- 让框架生长更稳、更可维护

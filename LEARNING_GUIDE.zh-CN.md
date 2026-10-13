# GPT Researcher 源码学习路线

这份指南面向希望从源码理解“研究型 Agent”设计的人。建议先掌握主链路，再逐步进入深度研究、多 Agent、MCP 和前端；不要一开始就通读所有检索器与抓取器实现。

> 本地源码快照：`main` 分支，提交 `0957c301ed06c2a5857b834358c7227c739041d4`。上游仓库后续更新后，文件和默认模型可能发生变化。

## 1. 先建立全局认知

GPT Researcher 的核心不是“让一个大模型直接写答案”，而是把研究任务拆成一条可观察、可替换的流水线：

```text
用户问题
  -> 选择研究角色
  -> 规划搜索问题
  -> 多路检索
  -> 抓取网页或读取本地文档
  -> 过滤、压缩与整理上下文
  -> 生成带来源的报告
  -> 可选：导出、审校、可视化和多 Agent 协作
```

读代码时始终追踪四类数据：

1. `query`：用户真正想研究什么。
2. `context`：从外部来源收集并压缩后的证据。
3. `report`：模型基于证据生成的最终内容。
4. `costs / logs / sources`：用于评估、追踪和调试研究过程的元数据。

## 2. 仓库地图

| 路径 | 作用 | 建议优先级 |
| --- | --- | --- |
| `gpt_researcher/agent.py` | 单 Agent 总编排器 `GPTResearcher` | 最高 |
| `gpt_researcher/skills/researcher.py` | 研究规划、检索、抓取、上下文构建 | 最高 |
| `gpt_researcher/skills/writer.py` | 将研究上下文写成报告 | 最高 |
| `gpt_researcher/prompts.py` | Prompt 家族入口 | 高 |
| `gpt_researcher/actions/` | 检索、查询处理、报告生成等具体动作 | 高 |
| `gpt_researcher/config/` | 环境变量、默认值及配置解析 | 高 |
| `gpt_researcher/retrievers/` | Tavily、DuckDuckGo、学术搜索、MCP 等检索适配器 | 中 |
| `gpt_researcher/scraper/` | 网页、PDF 与第三方抓取实现 | 中 |
| `gpt_researcher/context/` | 上下文过滤策略 | 中 |
| `gpt_researcher/document/` | 本地与在线文档加载 | 中 |
| `gpt_researcher/skills/deep_research.py` | 递归式深度研究 | 中高 |
| `multi_agents/` | 基于 LangGraph 的研究团队 | 进阶 |
| `deep_agents/` | 基于 Deep Agents 的主编—研究员委派模式 | 进阶 |
| `backend/server/` | FastAPI、WebSocket、任务执行与报告存储 | 应用层 |
| `frontend/` | 原生 Web 前端 | 应用层 |
| `frontend/nextjs/` | 独立的 Next.js、React、TypeScript 前端 | 应用层 |
| `tests/` | 行为边界、回归场景与扩展范例 | 全程配合阅读 |
| `docs/docs/gpt-researcher/` | 配置、LLM、检索器、MCP、前端与部署说明 | 随查随用 |

`README-zh_CN.md` 适合快速了解项目背景，但部分配置说明落后于当前源码。涉及 Python 版本、默认模型、启动命令和依赖时，应以 `README.md`、`pyproject.toml`、`.env.example` 和实际代码为准。

## 3. 推荐学习路线

### 阶段 0：准备基础知识

先确保你能解释以下概念，不要求精通：

- Python `async` / `await`、`asyncio.gather` 与并发限制；
- LLM 的 system/user prompt、上下文窗口和 token 成本；
- 搜索、抓取、RAG、embedding 与向量检索的区别；
- FastAPI、WebSocket 的基本请求与流式事件模型；
- LangGraph 的节点、边、状态和条件路由。

验收：能用自己的话说明“检索到 URL”为什么不等于“已经获得可供模型引用的可靠上下文”。

### 阶段 1：跑通最小 API 心智模型

按顺序读：

1. `README.md` 中的 PIP package 示例；
2. `gpt_researcher/__init__.py`；
3. `gpt_researcher/agent.py` 中的 `GPTResearcher.__init__`、`conduct_research()`、`write_report()`；
4. `tests/research_test.py` 和 `tests/test_quick_search.py`。

最小调用只有两个核心步骤：

```python
from gpt_researcher import GPTResearcher

researcher = GPTResearcher(query="你的研究问题")
context = await researcher.conduct_research()
report = await researcher.write_report()
```

重点回答：

- 初始化阶段创建了哪些协作对象？
- 为什么“收集证据”和“撰写报告”被拆成两个公开方法？
- `report_type`、`report_source`、`tone` 会在哪些分支影响流程？

验收：画出 `GPTResearcher -> ResearchConductor -> ReportGenerator` 的三层调用图。

### 阶段 2：追踪一次完整研究

以 `gpt_researcher/skills/researcher.py` 的 `conduct_research()` 为主线，继续阅读：

1. `gpt_researcher/actions/agent_creator.py`：如何选择研究角色；
2. `gpt_researcher/actions/query_processing.py`：如何规划查询并取得搜索结果；
3. `gpt_researcher/actions/retriever.py`：如何把配置映射为检索器类；
4. `gpt_researcher/skills/browser.py` 与 `gpt_researcher/scraper/`：URL 如何变成正文；
5. `gpt_researcher/skills/context_manager.py` 与 `gpt_researcher/context/`：如何过滤上下文；
6. `gpt_researcher/skills/curator.py`：可选的来源整理。

建议在纸上维护一张数据形态表：每一步的输入类型、输出类型、是否调用网络、是否调用 LLM、是否产生费用。

验收：能定位“搜索为空”“抓取失败”“上下文为空”各自应从哪里开始排查。

### 阶段 3：理解 Prompt 与报告生成

按顺序读：

1. `gpt_researcher/prompts.py`；
2. `gpt_researcher/actions/report_generation.py`；
3. `gpt_researcher/skills/writer.py`；
4. `gpt_researcher/actions/markdown_processing.py`；
5. `tests/test_add_references_order.py`、`tests/test_report_generation_available_images.py`。

阅读时区分三件事：

- Prompt 决定模型被要求做什么；
- `context` 决定模型能够依据什么；
- 后处理决定引用、标题和格式最终如何呈现。

特别关注 `ReportGenerator.write_report()` 的空上下文保护：没有检索到材料时，系统应拒绝生成看似可信的“有来源报告”。

验收：选择一个报告类型，写出它从上下文到最终 Markdown 的完整调用链。

### 阶段 4：理解配置与可替换组件

先读 `gpt_researcher/config/variables/default.py`，再读 `gpt_researcher/config/config.py`。当前版本的重要默认项包括：

- `RETRIEVER=tavily`；
- `FAST_LLM`、`SMART_LLM`、`STRATEGIC_LLM` 三种模型职责；
- `CONTEXT_FILTER`、`MAX_SEARCH_RESULTS_PER_QUERY`；
- `REPORT_SOURCE`、`REPORT_FORMAT`、`LANGUAGE`；
- 深度研究的 breadth、depth、concurrency；
- MCP 与图片生成的可选配置。

然后任选一个内置检索器做纵向阅读，例如：

```text
actions/retriever.py
  -> retrievers/tavily/
  -> actions/query_processing.py
  -> scraper/
  -> context/
```

不要同时读完所有检索器。它们大多实现同一边界，学习价值主要在统一接口、结果标准化、异常处理与插件入口。

验收：能说明增加一个新检索器最少需要实现和注册哪些内容，并找到对应测试模式。

### 阶段 5：深度研究

阅读 `gpt_researcher/skills/deep_research.py`，重点关注：

- breadth 与 depth 如何控制问题树；
- 后续问题如何根据已有 learning 生成；
- 并发控制如何避免无限扩张；
- 上下文裁剪如何控制 token；
- JSON 修复和备用解析如何对抗不稳定模型输出；
- 引用如何随 learning 保留下来。

配套测试：

- `tests/test_deep_research_parsing.py`；
- `tests/skills/test_deep_research_empty_results.py`；
- `tests/skills/test_deep_research_zero_queries.py`。

验收：手工模拟 breadth=2、depth=2 的查询树，并估算最多会触发多少次研究分支。

### 阶段 6：多 Agent 架构

先读 `multi_agents/agents/orchestrator.py`，它明确展示了 LangGraph 工作流：

```text
browser -> planner -> human
                      | accept
                      v
researcher -> writer -> fact_checker -> visualizer -> publisher -> END
              ^              |
              |---- revise ---|
```

随后阅读 `multi_agents/memory/research.py` 了解共享状态，再按图中的节点依次阅读 `multi_agents/agents/`。注意规划审阅和事实审阅循环都有次数上限，这是让 Agent 工作流可终止的重要工程约束。

最后再读 `deep_agents/agent.py`，比较另一种模式：主编先规划，以子 Agent 并行研究各章节，将草稿写入文件，再统一审校和组装。

验收：比较单 Agent、LangGraph 多 Agent、Deep Agents 三种设计在状态管理、并发、可控性和成本上的差异。

### 阶段 7：应用层与实时交互

按顺序读：

1. `main.py`：服务入口；
2. `backend/server/app.py`：HTTP 和 WebSocket 路由；
3. `backend/server/server_utils.py`：运行研究、流式日志与文件导出；
4. `backend/server/websocket_manager.py`：连接生命周期；
5. `frontend/`：浏览器如何消费事件和展示报告。

验收：从前端发起研究开始，追踪到 `GPTResearcher.conduct_research()`，再追踪一条进度事件如何回到浏览器。

### 阶段 8：MCP、可观测性与评估

最后处理横切能力：

- `gpt_researcher/mcp/` 与 `gpt_researcher/retrievers/mcp/`；
- `gpt_researcher/utils/logging_config.py`；
- `docs/docs/gpt-researcher/handling-logs/`；
- `deep_agents/benchmark*.py` 与 `deep_agents/benchmark_results/`；
- `tests/test_mcp*.py`、`tests/test_costs.py`、`tests/test_logging*.py`。

验收：能给一次研究任务列出质量、覆盖度、引用可靠性、延迟和成本五类指标，而不是只看最终文字是否“像一份报告”。

## 4. 本地运行建议

项目当前要求 Python 3.12 或更高版本。为了避免污染你的默认 Python，学习时建议使用项目级虚拟环境；下面只是一套可选命令，不会因阅读本指南而自动执行：

```bash
cd /Users/ernurat/Desktop/agent学习/gpt_research
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

至少需要为所选模型和检索器配置密钥。默认组合通常需要：

```bash
export OPENAI_API_KEY="..."
export TAVILY_API_KEY="..."
```

不要把真实密钥提交进 Git。启动完整 Web 应用：

```bash
python -m uvicorn main:app --reload
```

然后访问 `http://localhost:8000`。首次运行前先检查 `gpt_researcher/config/variables/default.py`，因为模型、价格和供应商能力会随版本变化。

还可以通过 CLI 运行：

```bash
python cli.py "你的研究问题" --report_type research_report --tone analytical
```

如果要单独研究 Next.js 前端，请进入 `frontend/nextjs/`，按照该目录的说明安装 Node.js 依赖并启动开发服务器。它不是理解 Python 核心 Agent 的前置条件。

## 5. 测试与调试方法

先运行与正在阅读的模块最接近的单测，再运行完整测试：

```bash
python -m pytest tests/test_quick_search.py
python -m pytest tests/test_deep_research_parsing.py
python -m pytest tests/
```

调试一次研究时，建议记录以下检查点：

1. 配置最终解析成了什么，而不是你以为环境变量是什么；
2. 实际启用了哪些 retriever；
3. 每个 query 返回了哪些 URL；
4. 抓取结果是否有正文、标题和来源；
5. 过滤后 context 是否仍保留关键证据；
6. 报告中的每个关键主张能否回到来源；
7. 每一步耗时与 token/API 成本。

涉及外部 API 的测试可能产生费用或受网络波动影响。先读测试内容，确认它是纯单元测试还是集成测试，再决定是否运行。

## 6. 建议完成的四个练习

### 练习一：做一次只读调用追踪

不修改源码，为 `conduct_research()` 的主要步骤画时序图，并标出所有网络和 LLM 调用。目标是建立整体心智模型。

### 练习二：增加一个 Prompt 家族

复制现有 Prompt 家族的最小实现，只改变报告风格，并为选择逻辑补测试。不要同时修改检索、抓取或后端。

### 练习三：实现一个简单检索器插件

选择一个你熟悉的数据源，复用插件注册机制，统一输出字段并为“空结果、字段缺失、请求失败”各写一个测试。目标是理解边界适配，而不是追求功能复杂度。

### 练习四：比较三种编排模式

让单 Agent、`multi_agents/` 和 `deep_agents/` 研究同一问题，记录来源数、重复来源、耗时、成本、事实错误和报告结构。目标是用数据理解“多 Agent”是否真的改善结果。

## 7. 阅读源码时最值得追问的问题

- 哪些步骤必须调用 LLM，哪些可以用确定性代码完成？
- 并发提升速度时，是否造成重复搜索、重复抓取或更高成本？
- 来源数量增加是否真的提高了观点覆盖与事实可靠性？
- Prompt 输出不是合法 JSON 时，系统如何降级？
- 抓取失败、空上下文和部分成功时，系统是否诚实暴露不确定性？
- `visited_urls`、上下文长度和递归深度如何阻止任务失控？
- 引用是在研究阶段保存，还是在写作阶段重新拼装？会不会错配？
- 多 Agent 的共享状态中，谁有权覆盖谁的结果？循环如何终止？
- 如何为 Agent 写稳定测试，而不是依赖某次模型恰好输出正确文本？

## 8. 一个可执行的四周计划

| 周次 | 学习内容 | 产出 |
| --- | --- | --- |
| 第 1 周 | 阶段 0–2：单 Agent 与研究主链路 | 调用图、数据形态表、一次运行日志注释 |
| 第 2 周 | 阶段 3–4：Prompt、报告、配置与检索器 | 一个小型 Prompt 实验或检索器插件 |
| 第 3 周 | 阶段 5–6：深度研究与多 Agent | 查询树推演、三种编排方式对比表 |
| 第 4 周 | 阶段 7–8：服务层、MCP、评估与测试 | 端到端时序图、质量与成本评估报告 |

学习完成的标准不是“读完所有文件”，而是你能从一个研究结果反向定位：它用了什么配置、调用了哪些数据源、经过什么过滤、由哪个 Prompt 生成、如何验证，以及失败时该看哪里。

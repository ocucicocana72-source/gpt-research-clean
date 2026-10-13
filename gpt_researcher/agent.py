"""GPT Researcher Agent 模块。

本模块提供 GPTResearcher 主类，使用大语言模型和网络搜索协调自主研究与报告生成。
"""

import asyncio
import json
import os
import logging
from typing import Any, Optional

from .actions import (
    add_references,
    choose_agent,
    extract_headers,
    extract_sections,
    get_retrievers,
    get_search_results,
    table_of_contents,
)
from .config import Config
from .llm_provider import GenericLLMProvider
from .memory import Memory
from .prompts import get_prompt_family
from .skills.browser import BrowserManager
from .skills.context_manager import ContextManager
from .skills.curator import SourceCurator
from .skills.deep_research import DeepResearchSkill
from .skills.image_generator import ImageGenerator
from .skills.researcher import ResearchConductor
from .skills.writer import ReportGenerator
from .utils.enum import ReportSource, ReportType, Tone
from .utils.llm import create_chat_completion
from .vector_store import VectorStoreWrapper
import json_repair


class GPTResearcher:
    """GPT Researcher 主 Agent 类。

    该类负责协调完整的研究流程，包括网络搜索、内容抓取、上下文管理以及
    使用大语言模型生成报告。

    属性：
        query：研究查询或问题。
        report_type：要生成的报告类型。
        cfg：配置对象。
        context：累积的研究上下文。
        research_costs：累积的 API 成本。
        step_costs：按步骤统计的成本字典。
    """

    def __init__(
        self,
        query: str,
        report_type: str = ReportType.ResearchReport.value,
        report_format: str = "markdown",
        report_source: str = ReportSource.Web.value,
        tone: Tone = Tone.Objective,
        source_urls: list[str] | None = None,
        document_urls: list[str] | None = None,
        complement_source_urls: bool = False,
        query_domains: list[str] | None = None,
        documents=None,
        vector_store=None,
        vector_store_filter=None,
        config_path=None,
        websocket=None,
        agent=None,
        role=None,
        parent_query: str = "",
        subtopics: list | None = None,
        visited_urls: set | None = None,
        verbose: bool = True,
        context=None,
        headers: dict | None = None,
        max_subtopics: int = 5,
        log_handler=None,
        prompt_family: str | None = None,
        mcp_configs: list[dict] | None = None,
        mcp_max_iterations: int | None = None,
        mcp_strategy: str | None = None,
        knowledge_check: bool = True,
        needs_knowledge_base: bool | None = None,
        **kwargs
    ):
        """
        初始化 GPT Researcher 实例。
        
        参数：
            query（str）：研究查询或问题。
            report_type（str）：要生成的报告类型。
            report_format（str）：报告格式（markdown、pdf 等）。
            report_source（str）：报告信息来源（web、local 等）。
            tone（Tone）：报告语气。
            source_urls（list[str]，可选）：要使用的指定来源 URL 列表。
            document_urls（list[str]，可选）：要使用的文档 URL 列表。
            complement_source_urls（bool）：是否通过网络搜索补充来源 URL。
            query_domains（list[str]，可选）：限制搜索范围的域名列表。
            documents：用于 LangChain 集成的文档对象。
            vector_store：用于文档检索的向量存储。
            vector_store_filter：向量存储查询过滤器。
            config_path：配置文件路径。
            websocket：用于流式输出的 WebSocket。
            agent：预定义的 Agent 类型。
            role：预定义的 Agent 角色。
            parent_query：子主题报告的父查询。
            subtopics：待研究的子主题列表。
            visited_urls：已访问 URL 的集合。
            verbose（bool）：是否输出详细日志。
            context：预加载的研究上下文。
            headers（dict，可选）：请求和配置使用的额外请求头。
            max_subtopics（int）：要生成的最大子主题数量。
            log_handler：日志事件处理器。
            prompt_family：要使用的提示词族。
            mcp_configs（list[dict]，可选）：MCP 服务器配置列表。
                每个字典可以包含：
                - name（str）：MCP 服务器名称
                - command（str）：启动服务器的命令
                - args（list[str]）：服务器命令的参数
                - tool_name（str）：MCP 服务器上要使用的特定工具
                - env（dict）：服务器环境变量
                - connection_url（str）：WebSocket 或 HTTP 连接 URL
                - connection_type（str）：连接类型（stdio、websocket、http）
                - connection_token（str）：远程连接的认证令牌
                
                示例：
                ```python
                mcp_configs=[{
                    "command": "python",
                    "args": ["my_mcp_server.py"],
                    "name": "search"
                }]
                ```
            mcp_strategy（str，可选）：MCP 执行策略，可选值：
                - "fast"（默认）：仅使用原始查询运行一次 MCP，以获得最佳性能
                - "deep"：对所有子查询运行 MCP，以获得最大完整性
                - "disabled"：完全跳过 MCP，仅使用网络检索器
        """
        self.kwargs = kwargs
        self.query = query
        self.report_type = report_type
        self.cfg = Config(config_path)
        self.cfg.set_verbose(verbose)
        self.report_source = report_source if report_source else getattr(self.cfg, 'report_source', None)
        self.report_format = report_format
        self.max_subtopics = max_subtopics
        self.tone = tone if isinstance(tone, Tone) else Tone.Objective
        self.source_urls = source_urls
        self.document_urls = document_urls
        self.complement_source_urls = complement_source_urls
        self.query_domains = query_domains or []
        self.research_sources = []  # 抓取到的来源列表，包括标题、内容和图片
        self.research_images = []  # 已选中的研究图片列表
        self.documents = documents
        self.vector_store = (
            VectorStoreWrapper(vector_store, chunking_strategy=self.cfg.vector_store_chunking)
            if vector_store else None
        )
        self.vector_store_filter = vector_store_filter
        self.websocket = websocket
        self.agent = agent
        self.role = role
        self.parent_query = parent_query
        self.subtopics = subtopics or []
        self.visited_urls = visited_urls if visited_urls is not None else set()
        self.verbose = verbose
        self.context = context or []
        self.headers = headers or {}
        self.research_costs = 0.0
        self.step_costs: dict[str, float] = {}
        self._current_step: str = "general"
        self.knowledge_check = knowledge_check
        self.needs_knowledge_base: bool | None = needs_knowledge_base
        self.direct_answer: str | None = None
        self.log_handler = log_handler
        self.prompt_family = get_prompt_family(prompt_family or self.cfg.prompt_family, self.cfg)
        
        # 如果提供了 MCP 配置，则进行处理。
        self.mcp_configs = mcp_configs
        if mcp_configs:
            self._process_mcp_configs(mcp_configs)
        
        self.retrievers = get_retrievers(self.headers, self.cfg)
        # 首次使用时再构建：标准研究流程使用 Jev 或空上下文过滤器时，
        # 不需要提前加载嵌入模型或凭据。
        self._memory = None
        
        # 设置默认编码为 UTF-8。
        self.encoding = kwargs.get('encoding', 'utf-8')
        self.kwargs.pop('encoding', None)  # 移除 encoding，避免将其传递给大语言模型调用

        # 初始化各个组件。
        self.research_conductor: ResearchConductor = ResearchConductor(self)
        self.report_generator: ReportGenerator = ReportGenerator(self)
        self.context_manager: ContextManager = ContextManager(self)
        self.scraper_manager: BrowserManager = BrowserManager(self)
        self.source_curator: SourceCurator = SourceCurator(self)
        self.deep_researcher: Optional[DeepResearchSkill] = None
        if report_type == ReportType.DeepResearch.value:
            self.deep_researcher = DeepResearchSkill(self)

        # 初始化图片生成器（可选，仅在配置后启用）。
        self.image_generator: Optional[ImageGenerator] = ImageGenerator(self)
        self.available_images: list = []  # 已预生成、可嵌入报告的图片
        self._research_id: str = ""  # 本次研究会话的唯一 ID

        # 处理 MCP 策略配置，并兼容旧配置。
        self.mcp_strategy = self._resolve_mcp_strategy(mcp_strategy, mcp_max_iterations)
    
    def _generate_research_id(self) -> str:
        """为本次会话生成唯一的研究 ID。
        
        返回：
            本次研究会话的唯一字符串标识符。
        """
        if not self._research_id:
            import hashlib
            import time
            # 根据查询和时间戳生成唯一 ID。
            unique_str = f"{self.query}_{time.time()}"
            self._research_id = f"research_{hashlib.md5(unique_str.encode()).hexdigest()[:12]}"
        return self._research_id

    def _resolve_mcp_strategy(self, mcp_strategy: str | None, mcp_max_iterations: int | None) -> str:
        """
        从多个来源解析 MCP 策略，并兼容旧配置。
        
        优先级：
        1. 参数 mcp_strategy（新方式）
        2. 参数 mcp_max_iterations（兼容旧方式）
        3. 配置项 MCP_STRATEGY
        4. 默认值 "fast"
        
        参数：
            mcp_strategy：新的策略参数。
            mcp_max_iterations：用于兼容旧版本的参数。
            
        返回：
            str：解析后的策略（"fast"、"deep" 或 "disabled"）。
        """
        # 优先级 1：如果传入 mcp_strategy，则使用它。
        if mcp_strategy is not None:
            # 支持新的策略名称。
            if mcp_strategy in ["fast", "deep", "disabled"]:
                return mcp_strategy
            # 兼容旧策略名称。
            elif mcp_strategy == "optimized":
                import logging
                logging.getLogger(__name__).warning("mcp_strategy 'optimized' is deprecated, use 'fast' instead")
                return "fast"
            elif mcp_strategy == "comprehensive":
                import logging
                logging.getLogger(__name__).warning("mcp_strategy 'comprehensive' is deprecated, use 'deep' instead")
                return "deep"
            else:
                import logging
                logging.getLogger(__name__).warning(f"Invalid mcp_strategy '{mcp_strategy}', defaulting to 'fast'")
                return "fast"
        
        # 优先级 2：转换 mcp_max_iterations，以兼容旧参数。
        if mcp_max_iterations is not None:
            import logging
            logging.getLogger(__name__).warning("mcp_max_iterations is deprecated, use mcp_strategy instead")
            
            if mcp_max_iterations == 0:
                return "disabled"
            elif mcp_max_iterations == 1:
                return "fast"
            elif mcp_max_iterations == -1:
                return "deep"
            else:
                # 其他数值均按快速模式处理。
                return "fast"
        
        # 优先级 3：使用配置文件中的设置。
        if hasattr(self.cfg, 'mcp_strategy'):
            config_strategy = self.cfg.mcp_strategy
            # 支持新的策略名称。
            if config_strategy in ["fast", "deep", "disabled"]:
                return config_strategy
            # 兼容旧策略名称。
            elif config_strategy == "optimized":
                return "fast"
            elif config_strategy == "comprehensive":
                return "deep"
            
        # 优先级 4：默认使用快速模式。
        return "fast"

    @property
    def memory(self) -> Memory:
        """按需创建并返回向量记忆，避免普通流程提前初始化嵌入模型。"""
        if self._memory is None:
            self._memory = Memory(
                self.cfg.embedding_provider, self.cfg.embedding_model, **self.cfg.embedding_kwargs
            )
        return self._memory

    @memory.setter
    def memory(self, value: Memory) -> None:
        """注入外部构造的向量记忆实例。"""
        self._memory = value

    def _process_mcp_configs(self, mcp_configs: list[dict]) -> None:
        """
        处理配置字典列表中的 MCP 配置。把 MCP 工具包装成符合项目检索器接口的组件。
        MCP 被当作一种“研究信息来源”，所以接入到了统一的检索器接口中。
        这里的“检索器”并不只表示搜索引擎，而是泛指：接收查询并返回研究资料的组件。

        通过直接修改 self.cfg.retrievers 将 MCP 检索器加入当前检索器列表。
        有意避免修改 os.environ，从而避免本次会话的 MCP 设置影响并发或后续请求
       （修复 issue #1676：进程级环境变量污染）。

        参数：
            mcp_configs（list[dict]）：MCP 服务器配置字典列表。
        """
        # 通过 cfg 而不是 os.environ 添加 MCP，避免污染进程环境。
        if hasattr(self.cfg, 'retrievers') and self.cfg.retrievers:
            current_retrievers = (
                list(self.cfg.retrievers)
                if isinstance(self.cfg.retrievers, list)
                else [r.strip() for r in str(self.cfg.retrievers).split(",") if r.strip()]
            )
            if "mcp" not in current_retrievers:
                current_retrievers.append("mcp")
                self.cfg.retrievers = current_retrievers
        else:
            self.cfg.retrievers = ["mcp"]

        # 保存 mcp_configs，供 MCP 检索器使用。
        self.mcp_configs = mcp_configs

    async def _log_event(self, event_type: str, **kwargs):
        """处理日志事件的辅助方法。"""
        if self.log_handler:
            try:
                if event_type == "tool":
                    await self.log_handler.on_tool_start(kwargs.get('tool_name', ''), **kwargs)
                elif event_type == "action":
                    await self.log_handler.on_agent_action(kwargs.get('action', ''), **kwargs)
                elif event_type == "research":
                    await self.log_handler.on_research_step(kwargs.get('step', ''), kwargs.get('details', {}))

                # 同时写入普通日志，作为备用记录。
                import logging
                research_logger = logging.getLogger('research')
                research_logger.info(f"{event_type}: {json.dumps(kwargs, default=str)}")

            except Exception as e:
                import logging
                logging.getLogger('research').error(f"Error in _log_event: {e}", exc_info=True)

    async def _answer_without_retrieval(self) -> str:
        """Answer directly when the routing model determines retrieval is unnecessary."""
        self._current_step = "direct_answer"
        prompt = (
            "直接回答用户问题。不要声称查阅了资料，不要编造引用；如果问题本身缺少必要条件，"
            "明确说明假设。回答简洁、准确，并使用用户提问的语言。\n\n用户问题：\n"
            f"{self.query}"
        )
        return await create_chat_completion(
            model=self.cfg.smart_llm_model,
            messages=[{"role": "user", "content": prompt}],
            llm_provider=self.cfg.smart_llm_provider,
            max_tokens=self.cfg.smart_token_limit,
            llm_kwargs=self.cfg.llm_kwargs,
            cost_callback=self.add_costs,
        )

    async def _route_knowledge_requirement(self) -> bool:
        """Use one LLM call to decide whether this root task needs retrieval."""
        if self.needs_knowledge_base is not None:
            return self.needs_knowledge_base
        if not self.knowledge_check or self.parent_query:
            return True
        prompt = self.prompt_family.knowledge_base_routing_prompt(self.query)
        try:
            response = await create_chat_completion(
                model=self.cfg.fast_llm_model,
                messages=[{"role": "user", "content": prompt}],
                llm_provider=self.cfg.fast_llm_provider,
                temperature=0,
                max_tokens=300,
                llm_kwargs=self.cfg.llm_kwargs,
                cost_callback=self.add_costs,
            )
            payload = json_repair.loads(response)
            if isinstance(payload, dict) and isinstance(payload.get("needs_knowledge_base"), bool):
                return payload["needs_knowledge_base"]
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Knowledge-base routing failed; defaulting to retrieval: %s", exc
            )
        return True

    async def conduct_research(self, on_progress=None):
        """执行研究流程。

        该方法协调主要研究流程，包括 Agent 选择、网络搜索和上下文收集。

        参数：
            on_progress：深度研究期间用于更新进度的可选回调。

        返回：
            累积的研究上下文。
        """
        await self._log_event("research", step="start", details={
            "query": self.query,
            "report_type": self.report_type,
            "agent": self.agent,
            "role": self.role
        })

        # Route before agent selection, web search, local loading, or deep research.
        self.needs_knowledge_base = await self._route_knowledge_requirement()
        await self._log_event("research", step="knowledge_route", details={
            "needs_knowledge_base": self.needs_knowledge_base,
        })
        if not self.needs_knowledge_base:
            self.direct_answer = await self._answer_without_retrieval()
            self.context = self.direct_answer
            return self.context

        # 深度研究使用独立流程处理。
        if self.report_type == ReportType.DeepResearch.value and self.deep_researcher:
            self._current_step = "deep_research"
            return await self._handle_deep_research(on_progress)

        if not (self.agent and self.role):
            self._current_step = "agent_selection"
            await self._log_event("action", action="choose_agent")
            # 过滤 LLM API 不支持的 encoding 参数。
            # filtered_kwargs = {k: v for k, v in self.kwargs.items() if k != 'encoding'}
            self.agent, self.role = await choose_agent(
                query=self.query,
                cfg=self.cfg,
                parent_query=self.parent_query,
                cost_callback=self.add_costs,
                headers=self.headers,
                prompt_family=self.prompt_family,
                **self.kwargs,
            # **filtered_kwargs
            )
            await self._log_event("action", action="agent_selected", details={
                "agent": self.agent,
                "role": self.role
            })

        await self._log_event("research", step="conducting_research", details={
            "agent": self.agent,
            "role": self.role
        })
        self._current_step = "research"
        # ResearchConductor 负责把“选角色”之后的检索、抓取和上下文压缩串起来。
        self.context = await self.research_conductor.conduct_research()

        await self._log_event("research", step="research_completed", details={
            "context_length": len(self.context)
        })
        
        # 如果启用图片，则提前生成（在报告写作前完成，以改善交互体验）。
        self.available_images = []
        if self.image_generator and self.image_generator.is_enabled():
            await self._log_event("research", step="planning_images")
            # 将上下文列表转换为字符串，供分析使用。
            context_str = "\n\n".join(self.context) if isinstance(self.context, list) else str(self.context)
            self.available_images = await self.image_generator.plan_and_generate_images(
                context=context_str,
                query=self.query,
                research_id=self._generate_research_id(),
            )
            await self._log_event("research", step="images_pre_generated", details={
                "images_count": len(self.available_images)
            })
        
        return self.context

    async def _handle_deep_research(self, on_progress=None):
        """处理深度研究的执行和日志记录。

        参数：
            on_progress：用于更新进度的可选回调。

        返回：
            深度研究产生的累积研究上下文。
        """
        # 记录深度研究配置。
        await self._log_event("research", step="deep_research_initialize", details={
            "type": "deep_research",
            "breadth": self.deep_researcher.breadth,
            "depth": self.deep_researcher.depth,
            "concurrency": self.deep_researcher.concurrency_limit
        })

        # 记录深度研究开始。
        await self._log_event("research", step="deep_research_start", details={
            "query": self.query,
            "breadth": self.deep_researcher.breadth,
            "depth": self.deep_researcher.depth,
            "concurrency": self.deep_researcher.concurrency_limit
        })

        # 执行深度研究并取得上下文。
        self.context = await self.deep_researcher.run(on_progress=on_progress)

        # 获取研究总成本。
        total_costs = self.get_costs()

        # 记录深度研究完成及成本。
        await self._log_event("research", step="deep_research_complete", details={
            "context_length": len(self.context),
            "visited_urls": len(self.visited_urls),
            "total_costs": total_costs
        })

        # 记录最终成本更新。
        await self._log_event("research", step="cost_update", details={
            "cost": total_costs,
            "total_cost": total_costs,
            "research_type": "deep_research"
        })

        # 返回研究上下文。
        return self.context

    async def write_report(
        self,
        existing_headers: list = [],
        relevant_written_contents: list = [],
        ext_context=None,
        custom_prompt="",
    ) -> str:
        """撰写研究报告。

        参数：
            existing_headers：已有标题列表，用于避免重复。
            relevant_written_contents：此前已写内容列表，用作上下文。
            ext_context：替代内部上下文使用的外部上下文。
            custom_prompt：指导报告生成的自定义提示词。

        返回：
            生成的报告字符串。
        """
        if self.direct_answer is not None:
            return self.direct_answer

        # 如果有预生成图片则直接使用（图片在 conduct_research 阶段生成）。
        has_available_images = bool(self.available_images)
        
        self._current_step = "report_writing"
        await self._log_event("research", step="writing_report", details={
            "existing_headers": existing_headers,
            "context_source": "external" if ext_context else "internal",
            "available_images_count": len(self.available_images),
        })

        # 生成并嵌入可用图片的报告。
        # 报告生成器只消费已经收集好的上下文，不再重新执行检索。
        report = await self.report_generator.write_report(
            existing_headers=existing_headers,
            relevant_written_contents=relevant_written_contents,
            ext_context=ext_context or self.context,
            custom_prompt=custom_prompt,
            available_images=self.available_images,  # 传入预生成的图片
        )

        await self._log_event("research", step="report_completed", details={
            "report_length": len(report),
            "images_embedded": len(self.available_images) if has_available_images else 0,
        })
        return report

    async def write_report_conclusion(self, report_body: str) -> str:
        """撰写报告的结论部分。

        参数：
            report_body：待总结的报告正文。

        返回：
            生成的结论文本。
        """
        await self._log_event("research", step="writing_conclusion")
        conclusion = await self.report_generator.write_report_conclusion(report_body)
        await self._log_event("research", step="conclusion_completed")
        return conclusion

    async def write_introduction(self) -> str:
        """撰写报告的引言部分。

        返回：
            生成的引言文本。
        """
        await self._log_event("research", step="writing_introduction")
        intro = await self.report_generator.write_introduction()
        await self._log_event("research", step="introduction_completed")
        return intro

    async def quick_search(
        self,
        query: str,
        query_domains: list[str] = None,
        aggregated_summary: bool = False,
        all_retrievers: bool = False,
    ) -> list[Any] | str:
        """不执行完整研究流程，直接进行快速搜索。

        参数：
            query：搜索查询。
            query_domains：限制搜索范围的可选域名列表。
            aggregated_summary：是否返回搜索结果的聚合摘要。
            all_retrievers：为 True 时，并发查询所有已配置的检索器并合并结果
                （按 URL 去重）。默认为 False，仅使用主检索器以兼容旧行为。

        返回：
            搜索结果列表或生成的摘要字符串。
        """
        if all_retrievers and len(self.retrievers) > 1:
            search_results = await self._search_all_retrievers(query, query_domains)
        else:
            search_results = await get_search_results(
                query, self.retrievers[0], query_domains=query_domains, researcher=self
            )

        if not aggregated_summary:
            return search_results

        # 整理摘要所需的结果。搜索检索器通常使用 href（URL）和 body（内容）字段；
        # 同时兼容已预归一化记录使用的备用字段。
        context = ""
        for i, result in enumerate(search_results, 1):
            title = result.get("title", "")
            body = result.get("body") or result.get("content", "")
            url = result.get("href") or result.get("url", "")
            context += f"[{i}] {title}: {body} ({url})\n\n"

        prompt = self.prompt_family.generate_quick_summary_prompt(query, context)

        summary = await create_chat_completion(
            model=self.cfg.smart_llm_model,
            messages=[{"role": "user", "content": prompt}],
            llm_provider=self.cfg.smart_llm_provider,
            max_tokens=self.cfg.smart_token_limit,
            llm_kwargs=self.cfg.llm_kwargs,
            cost_callback=self.add_costs
        )

        return summary

    async def _search_all_retrievers(
        self, query: str, query_domains: list[str] = None
    ) -> list[dict[str, Any]]:
        """并发查询所有已配置的检索器并合并结果。

        结果按 URL 去重（同时检查不同检索器使用的 ``url`` 和 ``href`` 字段）。
        出错的检索器会被跳过，避免单个服务商失败导致整个搜索中止。

        参数：
            query：搜索查询。
            query_domains：限制搜索范围的可选域名列表。

        返回：
            合并且去重后的经过检索器搜索后的合并后结果列表。
        """
        tasks = [
            get_search_results(query, retriever, query_domains=query_domains, researcher=self)
            for retriever in self.retrievers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        merged: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for result in results:
            if isinstance(result, Exception) or not result:
                continue
            for item in result:
                url = item.get("url") or item.get("href") or ""
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                merged.append(item)
        return merged

    async def get_subtopics(self):
        """为研究查询生成子主题。

        返回：
            生成的子主题列表。
        """
        return await self.report_generator.get_subtopics()

    async def get_draft_section_titles(self, current_subtopic: str) -> list[str]:
        """为子主题生成章节标题草稿。

        参数：
            current_subtopic：要生成章节的子主题。

        返回：
            章节标题字符串列表。
        """
        return await self.report_generator.get_draft_section_titles(current_subtopic)

    async def get_similar_written_contents_by_draft_section_titles(
        self,
        current_subtopic: str,
        draft_section_titles: list[str],
        written_contents: list[dict],
        max_results: int = 10
    ) -> list[str]:
        """根据章节标题查找相似的既有内容。

        参数：
            current_subtopic：当前正在撰写的子主题。
            draft_section_titles：章节标题草稿列表。
            written_contents：待搜索的既有内容。
            max_results：返回结果的最大数量。

        返回：
            相似内容字符串列表。
        """
        return await self.context_manager.get_similar_written_contents_by_draft_section_titles(
            current_subtopic,
            draft_section_titles,
            written_contents,
            max_results
        )

    # 工具方法。
    def get_research_images(self, top_k: int = 10) -> list[dict[str, Any]]:
        """获取研究过程中收集的排名靠前的研究图片。

        参数：
            top_k：返回图片的最大数量。

        返回：
            图片字典列表。
        """
        return self.research_images[:top_k]

    def add_research_images(self, images: list[dict[str, Any]]) -> None:
        """将图片添加到研究图片集合中。

        参数：
            images：要添加的图片字典列表。
        """
        self.research_images.extend(images)

    def get_research_sources(self) -> list[dict[str, Any]]:
        """获取研究过程中收集的全部研究来源。

        返回：
            包含标题、内容和图片的来源字典列表。
        """
        return self.research_sources

    def add_research_sources(self, sources: list[dict[str, Any]]) -> None:
        """将来源添加到研究来源集合中。

        参数：
            sources：要添加的来源字典列表。
        """
        self.research_sources.extend(sources)

    def add_references(self, report_markdown: str, visited_urls: set) -> str:
        """向 Markdown 报告添加参考文献部分。

        参数：
            report_markdown：Markdown 报告文本。
            visited_urls：要作为参考文献加入的 URL 集合。

        返回：
            添加参考文献后的报告。
        """
        return add_references(report_markdown, visited_urls)

    def extract_headers(self, markdown_text: str) -> list[dict]:
        """从 Markdown 文本中提取标题。

        参数：
            markdown_text：待解析的 Markdown 文本。

        返回：
            标题字典列表。
        """
        return extract_headers(markdown_text)

    def extract_sections(self, markdown_text: str) -> list[dict]:
        """从 Markdown 文本中提取章节。

        参数：
            markdown_text：待解析的 Markdown 文本。

        返回：
            章节字典列表。
        """
        return extract_sections(markdown_text)

    def table_of_contents(self, markdown_text: str) -> str:
        """为 Markdown 文本生成目录。

        参数：
            markdown_text：要生成目录的 Markdown 文本。

        返回：
            Markdown 格式的目录字符串。
        """
        return table_of_contents(markdown_text)

    def get_source_urls(self) -> list:
        """获取所有已访问的来源 URL。

        返回：
            已访问 URL 字符串列表。
        """
        return list(self.visited_urls)

    def get_research_context(self) -> list:
        """获取累积的研究上下文。

        返回：
            研究过程中收集的上下文条目列表。
        """
        return self.context

    def get_costs(self) -> float:
        """获取累积的 API 总成本。

        返回：
            以美元计的总成本。
        """
        return self.research_costs

    def get_step_costs(self) -> dict[str, float]:
        """获取按研究步骤划分的 API 成本明细。

        返回：
            将步骤名称映射到美元成本的字典。
        """
        return dict(self.step_costs)

    def set_verbose(self, verbose: bool) -> None:
        """设置详细输出模式。

        参数：
            verbose：是否启用详细输出。
        """
        self.verbose = verbose

    def add_costs(self, cost: float) -> None:
        """增加累积的 API 成本。

        成本会归入通过 ``_current_step`` 设置的当前步骤。

        参数：
            cost：要增加的美元成本。

        异常：
            ValueError：当 cost 不是数字时抛出。
        """
        if not isinstance(cost, (float, int)):
            raise ValueError("Cost must be an integer or float")
        self.research_costs += cost
        step = self._current_step
        self.step_costs[step] = self.step_costs.get(step, 0.0) + cost
        if self.log_handler:
            self._log_event("research", step="cost_update", details={
                "cost": cost,
                "total_cost": self.research_costs,
                "step_name": step,
            })

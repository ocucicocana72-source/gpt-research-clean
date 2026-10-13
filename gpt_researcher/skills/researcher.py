"""Research conductor skill for GPT Researcher.

This module provides the ResearchConductor class that manages and
coordinates the research process including query planning, web searching,
and context gathering.
                      ┌─ Tavily
                      ├─ DuckDuckGo
用户问题 → 子查询规划 ├─ 本地文档
                      ├─ 向量数据库
                      └─ MCP 工具
                            ↓
                     统一研究上下文
                            ↓
                          LLM
"""

import asyncio
import logging
import os
import random

from ..actions.agent_creator import choose_agent
from ..actions.query_processing import get_search_results, plan_research_outline, generate_sub_queries
from ..actions.utils import stream_output
from ..document import DocumentLoader, LangChainDocumentLoader, OnlineDocumentLoader
from ..utils.enum import ReportSource, ReportType
from ..utils.logging_config import get_json_handler


class ResearchConductor:
    """管理并协调研究流程。

    该类负责处理主要的研究工作流，包括规划研究查询、执行网络搜索、
    管理 MCP 检索器以及从各种来源收集上下文信息。

    属性：
    researcher：父级 GPTResearcher 实例。
    logger：用于记录研究事件的日志记录器。
    json_handler：用于 JSON 日志记录的处理程序。
    """

    def __init__(self, researcher):
        """Initialize the ResearchConductor.

        Args:
            researcher: The GPTResearcher instance that owns this conductor.
        """
        self.researcher = researcher
        self.logger = logging.getLogger('research')
        self.json_handler = get_json_handler()
        # 缓存 MCP 结果，避免重复调用。
        self._mcp_results_cache = None
        # 保护并发研究流程中的缓存填充。
        self._mcp_cache_lock = asyncio.Lock()
        # 记录平衡模式下的 MCP 查询次数。
        self._mcp_query_count = 0

    async def _get_initial_search_results(self, query, query_domains=None):
        """获取研究规划阶段使用的首批搜索结果。"""
        await stream_output(
            "logs",
            "planning_research",
            f"🌐 Browsing the web to learn more about the task: {query}...",
            self.researcher.websocket,
        )

        search_results = await get_search_results(
            query,
            self.researcher.retrievers[0],
            query_domains,
            researcher=self.researcher,
            max_results=self.researcher.cfg.max_search_results_per_query,
        )
        self.logger.info(f"Initial search results obtained: {len(search_results)} results")
        return search_results

    async def plan_research(self, query, query_domains=None, *, search_results=None):
        """Plan sub-queries, optionally using results already obtained for this pass."""
        if search_results is None:
            search_results = await self._get_initial_search_results(query, query_domains)

        await stream_output(
            "logs",
            "planning_research",
            f"🤔 Planning the research strategy and subtasks...",
            self.researcher.websocket,
        )

        retriever_names = [r.__name__ for r in self.researcher.retrievers]
        # 移除重复日志；该信息会在 conduct_research 中统一记录。

        outline = await plan_research_outline(
            query=query,
            search_results=search_results,
            agent_role_prompt=self.researcher.role,
            cfg=self.researcher.cfg,
            parent_query=self.researcher.parent_query,
            report_type=self.researcher.report_type,
            cost_callback=self.researcher.add_costs,
            retriever_names=retriever_names,  # Pass retriever names for MCP optimization
            **self.researcher.kwargs
        )
        self.logger.info(f"Research outline planned: {outline}")
        return outline

    async def conduct_research(self):
        """Runs the GPT Researcher to conduct research"""
        if self.json_handler:
            self.json_handler.update_content("query", self.researcher.query)
        
        self.logger.info(f"Starting research for query: {self.researcher.query}")
        
        # 在研究开始时统一记录当前启用的检索器。
        retriever_names = [r.__name__ for r in self.researcher.retrievers]
        self.logger.info(f"Active retrievers: {retriever_names}")
        
        # 注意：这里故意不清空 visited_urls。它可能与父研究器共享，
        # 例如详细报告会把累计 URL 传给子主题研究器，避免重复抓取。
        research_data = []

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "starting_research",
                f"🔍 Starting the research task for '{self.researcher.query}'...",
                self.researcher.websocket,
            )
            await stream_output(
                "logs",
                "agent_generated",
                self.researcher.agent,
                self.researcher.websocket
            )

        # 如果尚未定义，则选择 Agent 和角色。
        if not (self.researcher.agent and self.researcher.role):
            self.researcher.agent, self.researcher.role = await choose_agent(
                query=self.researcher.query,
                cfg=self.researcher.cfg,
                parent_query=self.researcher.parent_query,
                cost_callback=self.researcher.add_costs,
                headers=self.researcher.headers,
                prompt_family=self.researcher.prompt_family
            )
                
        # 检查是否配置了 MCP 检索器。
        has_mcp_retriever = any("mcpretriever" in r.__name__.lower() for r in self.researcher.retrievers)
        if has_mcp_retriever:
            self.logger.info("MCP retrievers configured and will be used with standard research flow")

        # 这是资料来源的总分发点：URL、Web、本地、混合等模式最终都归一化为 context。
        if self.researcher.source_urls:
            self.logger.info("Using provided source URLs")
            research_data = await self._get_context_by_urls(self.researcher.source_urls)
            # `research_data and len(research_data) == 0` 不可能为真：
            # 真值对象不会为空，因此原通知不会触发。
            if not research_data and self.researcher.verbose:
                await stream_output(
                    "logs",
                    "answering_from_memory",
                    f"🧐 I was unable to find relevant context in the provided sources...",
                    self.researcher.websocket,
                )
            if self.researcher.complement_source_urls:
                self.logger.info("Complementing with web search")
                additional_research = await self._get_context_by_web_search(self.researcher.query, [], self.researcher.query_domains)
                research_data = " ".join(
                    part for part in (research_data, additional_research) if part
                )
        elif self.researcher.report_source == ReportSource.Web.value:
            self.logger.info("Using web search with all configured retrievers")
            research_data = await self._get_context_by_web_search(self.researcher.query, [], self.researcher.query_domains)
        elif self.researcher.report_source == ReportSource.Local.value:
            self.logger.info("Using local search")
            document_data = await DocumentLoader(self.researcher.cfg.doc_path).load()
            self.logger.info(f"Loaded {len(document_data)} documents")
            if self.researcher.vector_store:
                self.researcher.vector_store.load(document_data)
                research_data = await self._get_context_by_vectorstore(
                    self.researcher.query, self.researcher.vector_store_filter
                )
            else:
                research_data = await self._get_context_by_web_search(
                    self.researcher.query, document_data, self.researcher.query_domains
                )
        # 混合搜索，同时使用本地文档和 Web 来源。
        elif self.researcher.report_source == ReportSource.Hybrid.value:
            if self.researcher.document_urls:
                document_data = await OnlineDocumentLoader(self.researcher.document_urls).load()
            else:
                document_data = await DocumentLoader(self.researcher.cfg.doc_path).load()
            if self.researcher.vector_store:
                self.researcher.vector_store.load(document_data)
            # 本地文档和 Web 搜索彼此独立，因此并发执行；visited_urls 仍会跨来源去重。
            local_context = (
                self._get_context_by_vectorstore(self.researcher.query, self.researcher.vector_store_filter)
                if self.researcher.vector_store
                else self._get_context_by_web_search(self.researcher.query, document_data, self.researcher.query_domains)
            )
            docs_context, web_context = await asyncio.gather(
                local_context,
                self._get_context_by_web_search(self.researcher.query, [], self.researcher.query_domains),
            )
            research_data = self.researcher.prompt_family.join_local_web_documents(docs_context, web_context)
        elif self.researcher.report_source == ReportSource.Azure.value:
            from ..document.azure_document_loader import AzureDocumentLoader
            azure_loader = AzureDocumentLoader(
                container_name=os.getenv("AZURE_CONTAINER_NAME"),
                connection_string=os.getenv("AZURE_CONNECTION_STRING")
            )
            azure_files = await azure_loader.load()
            document_data = await DocumentLoader(azure_files).load()  # Reuse existing loader
            research_data = await self._get_context_by_web_search(self.researcher.query, document_data)
            
        elif self.researcher.report_source == ReportSource.LangChainDocuments.value:
            langchain_documents_data = await LangChainDocumentLoader(
                self.researcher.documents
            ).load()
            if self.researcher.vector_store:
                self.researcher.vector_store.load(langchain_documents_data)
            research_data = await self._get_context_by_web_search(
                self.researcher.query, langchain_documents_data, self.researcher.query_domains
            )
        elif self.researcher.report_source == ReportSource.LangChainVectorStore.value:
            research_data = await self._get_context_by_vectorstore(self.researcher.query, self.researcher.vector_store_filter)

        # 对来源排序并进行整理。
        self.researcher.context = research_data
        if self.researcher.cfg.curate_sources:
            self.logger.info("Curating sources")
            curated = await self.researcher.source_curator.curate_sources(research_data)
            # curate_sources() 返回包含 Title/Content/Source 字段的字典列表。
            # 归一化为字符串，避免下游按字符串使用 researcher.context 时出错。
            if isinstance(curated, list):
                self.researcher.context = "\n\n".join(
                    "Title: {title}\nContent: {content}\nSource: {source}".format(
                        title=s.get("Title", ""),
                        content=s.get("Content", ""),
                        source=s.get("Source", ""),
                    ) if isinstance(s, dict) else str(s)
                    for s in curated
                )
            else:
                self.researcher.context = curated

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "research_step_finalized",
                f"Finalized research step.\n💸 Total Research Costs: ${self.researcher.get_costs()}",
                self.researcher.websocket,
            )
            if self.json_handler:
                self.json_handler.update_content("costs", self.researcher.get_costs())
                self.json_handler.update_content("context", self.researcher.context)

        self.logger.info(f"Research completed. Context size: {len(str(self.researcher.context))}")
        return self.researcher.context

    async def _get_context_by_urls(self, urls):
        """Scrapes and compresses the context from the given urls"""
        self.logger.info(f"Getting context from URLs: {urls}")
        
        new_search_urls = await self._get_new_urls(urls)
        self.logger.info(f"New URLs to process: {new_search_urls}")

        scraped_content = await self.researcher.scraper_manager.browse_urls(new_search_urls)
        self.logger.info(f"Scraped content from {len(scraped_content)} URLs")

        if self.researcher.vector_store:
            self.researcher.vector_store.load(scraped_content)

        context = await self.researcher.context_manager.get_similar_content_by_query(
            self.researcher.query, scraped_content
        )
        return context

    # 其他方法也按同样方式补充日志……

    async def _get_context_by_vectorstore(self, query, filter: dict | None = None):
        """
        Generates the context for the research task by searching the vectorstore
        Returns:
            context: List of context
        """
        self.logger.info(f"Starting vectorstore search for query: {query}")
        context = []
        # 本地向量检索先将问题拆成三个互补子查询，不触发网页初搜。
        sub_queries = await generate_sub_queries(
            query=query,
            parent_query=self.researcher.parent_query,
            report_type=self.researcher.report_type,
            context=[],
            cfg=self.researcher.cfg,
            cost_callback=self.researcher.add_costs,
            prompt_family=self.researcher.prompt_family,
            max_iterations=3,
        )
        sub_queries = (sub_queries or [query])[:3]

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "subqueries",
                f"🗂️  I will conduct my research based on the following queries: {sub_queries}...",
                self.researcher.websocket,
                True,
                sub_queries,
            )

        # 使用 asyncio.gather 异步处理子查询。
        context = await asyncio.gather(
            *[
                self._process_sub_query_with_vectorstore(sub_query, filter)
                for sub_query in sub_queries
            ]
        )
        return context

    async def _get_context_by_web_search(self, query, scraped_data: list | None = None, query_domains: list | None = None):
        """
        Generates the context for the research task by searching the query and scraping the results
        Returns:
            context: List of context
        """
        self.logger.info(f"Starting web search for query: {query}")
        
        if scraped_data is None:
            scraped_data = []
        if query_domains is None:
            query_domains = []

        # 可配置的 MCP 优化：控制 MCP 策略。
        mcp_retrievers = [r for r in self.researcher.retrievers if "mcpretriever" in r.__name__.lower()]
        
        # 获取 MCP 策略配置。
        mcp_strategy = self._get_mcp_strategy()
        
        # 加锁，确保并发研究流程（例如混合模式）能够
        # 只填充一次 MCP 缓存，避免重复执行相同的 MCP 研究。
        async with self._mcp_cache_lock:
            if mcp_retrievers and self._mcp_results_cache is None:
                if mcp_strategy == "disabled":
                    # MCP 已禁用，完全跳过 MCP 研究。
                    self.logger.info("MCP disabled by strategy, skipping MCP research")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_disabled",
                            f"⚡ MCP research disabled by configuration",
                            self.researcher.websocket,
                        )
                elif mcp_strategy == "fast":
                    # 快速模式：使用原始查询运行一次 MCP。
                    self.logger.info("MCP fast strategy: Running once with original query")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_optimization",
                            f"🚀 MCP Fast: Running once for main query (performance mode)",
                            self.researcher.websocket,
                        )

                    # 使用原始查询执行一次 MCP 研究。
                    mcp_context = await self._execute_mcp_research_for_queries([query], mcp_retrievers)
                    self._mcp_results_cache = mcp_context
                    self.logger.info(f"MCP results cached: {len(mcp_context)} total context entries")
                elif mcp_strategy == "deep":
                    # 深度模式：为所有查询运行 MCP，交给逐查询流程执行。
                    self.logger.info("MCP deep strategy: Will run for all queries")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_comprehensive",
                            f"🔍 MCP Deep: Will run for each sub-query (thorough mode)",
                            self.researcher.websocket,
                        )
                    # 不缓存，让每个子查询单独运行 MCP。
                else:
                    # 未知策略，默认使用快速模式。
                    self.logger.warning(f"Unknown MCP strategy '{mcp_strategy}', defaulting to fast")
                    mcp_context = await self._execute_mcp_research_for_queries([query], mcp_retrievers)
                    self._mcp_results_cache = mcp_context
                    self.logger.info(f"MCP results cached: {len(mcp_context)} total context entries")

        # 生成包含原始查询的子查询。
        initial_results = await self._get_initial_search_results(query, query_domains)
        sub_queries = await self.plan_research(query, query_domains, search_results=initial_results)
        self.logger.info(f"Generated sub-queries: {sub_queries}")
        
        # 如果不是子研究器，则加入原始查询以获得更完整的结果。
        if self.researcher.report_type != "subtopic_report":
            sub_queries.append(query)

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "subqueries",
                f"🗂️ I will conduct my research based on the following queries: {sub_queries}...",
                self.researcher.websocket,
                True,
                sub_queries,
            )

        # 使用 asyncio.gather 异步处理子查询。
        try:
            # 在后续搜索可能丢失规划依据前读取来源；保留已提供文档和仅 MCP 路径不变。
            initial_context = ""
            if not scraped_data and initial_results:
                initial_context = await self._get_context_from_initial_results(
                    query, query_domains, initial_results
                )
            context = await asyncio.gather(
                *[
                    self._process_sub_query(sub_query, scraped_data, query_domains)
                    for sub_query in sub_queries
                ]
            )
            self.logger.info(f"Gathered context from {len(context)} sub-queries")
        # 过滤空结果并拼接上下文。
            context = [c for c in [initial_context, *context] if c]
            if context:
                combined_context = " ".join(context)
                self.logger.info(f"Combined context size: {len(combined_context)}")
                return combined_context
            return []
        except Exception as e:
            self.logger.error(f"Error during web search: {e}", exc_info=True)
            return []

    def _get_mcp_strategy(self) -> str:
        """
        Get the MCP strategy configuration.
        
        Priority:
        1. Instance-level setting (self.researcher.mcp_strategy)
        2. Config file setting (self.researcher.cfg.mcp_strategy) 
        3. Default value ("fast")
        
        Returns:
            str: MCP strategy
                "disabled" = Skip MCP entirely
                "fast" = Run MCP once with original query (default)
                "deep" = Run MCP for all sub-queries
        """
        # 先检查实例级设置。
        if hasattr(self.researcher, 'mcp_strategy') and self.researcher.mcp_strategy is not None:
            return self.researcher.mcp_strategy
        
        # 再检查配置设置。
        if hasattr(self.researcher.cfg, 'mcp_strategy'):
            return self.researcher.cfg.mcp_strategy
        
        # 默认使用快速模式。
        return "fast"

    async def _execute_mcp_research_for_queries(self, queries: list, mcp_retrievers: list) -> list:
        """
        Execute MCP research for a list of queries.
        
        Args:
            queries: List of queries to research
            mcp_retrievers: List of MCP retriever classes
            
        Returns:
            list: Combined MCP context entries from all queries
        """
        all_mcp_context = []
        
        for i, query in enumerate(queries, 1):
            self.logger.info(f"Executing MCP research for query {i}/{len(queries)}: {query}")
            
            for retriever in mcp_retrievers:
                try:
                    mcp_results = await self._execute_mcp_research(retriever, query)
                    if mcp_results:
                        for result in mcp_results:
                            content = result.get("body", "")
                            url = result.get("href", "")
                            title = result.get("title", "")
                            
                            if content:
                                context_entry = {
                                    "content": content,
                                    "url": url,
                                    "title": title,
                                    "query": query,
                                    "source_type": "mcp"
                                }
                                all_mcp_context.append(context_entry)
                        
                        self.logger.info(f"Added {len(mcp_results)} MCP results for query: {query}")
                        
                        if self.researcher.verbose:
                            await stream_output(
                                "logs",
                                "mcp_results_cached",
                                f"✅ Cached {len(mcp_results)} MCP results from query {i}/{len(queries)}",
                                self.researcher.websocket,
                            )
                except Exception as e:
                    self.logger.error(f"Error in MCP research for query '{query}': {e}")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_cache_error",
                            f"⚠️ MCP research error for query {i}, continuing with other sources",
                            self.researcher.websocket,
                        )
        
        return all_mcp_context

    def _tavily_mcp_redundant_with_direct(self, mcp_retrievers, non_mcp_retrievers) -> bool:
        """True when MCP would only re-query Tavily while direct Tavily is active.

        The frontend Tavily Web Search MCP preset hits the same API as
        `TavilySearch` and adds extra LLM tool-selection cost for no new data
        when both run together (#1875).
        """
        if not mcp_retrievers or not non_mcp_retrievers:
            return False
        has_direct_tavily = any(
            getattr(r, "__name__", "").lower() == "tavilysearch" for r in non_mcp_retrievers
        )
        if not has_direct_tavily:
            return False
        configs = getattr(self.researcher, "mcp_configs", None) or []
        if not configs:
            return False
        # 如果所有 MCP 服务都是 Tavily MCP，则视为重复配置。
        def _is_tavily_mcp(cfg: dict) -> bool:
            """判断单个 MCP 配置是否指向 Tavily 服务。"""
            name = str(cfg.get("name", "")).lower()
            args = " ".join(str(a) for a in (cfg.get("args") or [])).lower()
            command = str(cfg.get("command", "")).lower()
            blob = f"{name} {args} {command}"
            return "tavily" in blob

        return all(isinstance(c, dict) and _is_tavily_mcp(c) for c in configs)


    async def _process_sub_query(self, sub_query: str, scraped_data: list = [], query_domains: list = []):
        """Takes in a sub query and scrapes urls based on it and gathers context."""
        if self.json_handler:
            self.json_handler.log_event("sub_query", {
                "query": sub_query,
                "scraped_data_size": len(scraped_data)
            })
        
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "running_subquery_research",
                f"\n🔍 Running research for '{sub_query}'...",
                self.researcher.websocket,
            )

        try:
            # 识别 MCP 检索器。
            mcp_retrievers = [r for r in self.researcher.retrievers if "mcpretriever" in r.__name__.lower()]
            non_mcp_retrievers = [r for r in self.researcher.retrievers if "mcpretriever" not in r.__name__.lower()]

            # 在默认 RETRIEVER=tavily 时，避免直连检索器和 tavily-mcp 重复查询。
            if self._tavily_mcp_redundant_with_direct(mcp_retrievers, non_mcp_retrievers):
                self.logger.warning(
                    "Skipping LLM MCP Tavily path because TavilySearch is already configured as a direct retriever; set RETRIEVER without tavily or use non-Tavily MCP servers to keep MCP."
                )
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "mcp_tavily_deduped",
                        "⚠️ Skipping Tavily MCP (redundant with direct Tavily retriever) to avoid double API cost",
                        self.researcher.websocket,
                    )
                mcp_retrievers = []
            
            # 初始化上下文组件。
            mcp_context = []
            web_context = ""
            
            # 获取 MCP 策略配置。
            mcp_strategy = self._get_mcp_strategy()
            
            # 可配置的 MCP 处理流程。
            if mcp_retrievers:
                if mcp_strategy == "disabled":
                    # MCP 已禁用，完全跳过。
                    self.logger.info(f"MCP disabled for sub-query: {sub_query}")
                elif mcp_strategy == "fast" and self._mcp_results_cache is not None:
                    # 快速模式：使用缓存结果。
                    mcp_context = self._mcp_results_cache.copy()
                    
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_cache_reuse",
                            f"♻️ Reusing cached MCP results ({len(mcp_context)} sources) for: {sub_query}",
                            self.researcher.websocket,
                        )
                    
                    self.logger.info(f"Reused {len(mcp_context)} cached MCP results for sub-query: {sub_query}")
                elif mcp_strategy == "deep":
                    # 深度模式：为每个子查询运行 MCP。
                    self.logger.info(f"Running deep MCP research for: {sub_query}")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_comprehensive_run",
                            f"🔍 Running deep MCP research for: {sub_query}",
                            self.researcher.websocket,
                        )
                    
                    mcp_context = await self._execute_mcp_research_for_queries([sub_query], mcp_retrievers)
                else:
                    # 回退逻辑：没有缓存且不是深度模式时，为当前查询运行 MCP。
                    self.logger.warning("MCP cache not available, falling back to per-sub-query execution")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_fallback",
                            f"🔌 MCP cache unavailable, running MCP research for: {sub_query}",
                            self.researcher.websocket,
                        )
                    
                    mcp_context = await self._execute_mcp_research_for_queries([sub_query], mcp_retrievers)
            
            # 使用非 MCP 检索器获取 Web 上下文（未提供抓取数据时）。
            if not scraped_data:
                scraped_data = await self._scrape_data_by_urls(sub_query, query_domains)
                self.logger.info(f"Scraped data size: {len(scraped_data)}")

            # 根据抓取数据获取相似内容。
            if scraped_data:
                web_context = await self.researcher.context_manager.get_similar_content_by_query(sub_query, scraped_data)
                self.logger.info(f"Web content found for sub-query: {len(str(web_context)) if web_context else 0} chars")

            # 智能合并 MCP 上下文与 Web 上下文。
            combined_context = self._combine_mcp_and_web_context(mcp_context, web_context, sub_query)
            
            # 记录上下文合并结果。
            if combined_context:
                context_length = len(str(combined_context))
                self.logger.info(f"Combined context for '{sub_query}': {context_length} chars")
                
                if self.researcher.verbose:
                    mcp_count = len(mcp_context)
                    web_available = bool(web_context)
                    cache_used = self._mcp_results_cache is not None and mcp_retrievers and mcp_strategy != "deep"
                    cache_status = " (cached)" if cache_used else ""
                    await stream_output(
                        "logs",
                        "context_combined",
                        f"📚 Combined research context: {mcp_count} MCP sources{cache_status}, {'web content' if web_available else 'no web content'}",
                        self.researcher.websocket,
                    )
            else:
                self.logger.warning(f"No combined context found for sub-query: {sub_query}")
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "subquery_context_not_found",
                        f"🤷 No content found for '{sub_query}'...",
                        self.researcher.websocket,
                    )
            
            if combined_context and self.json_handler:
                self.json_handler.log_event("content_found", {
                    "sub_query": sub_query,
                    "content_size": len(str(combined_context)),
                    "mcp_sources": len(mcp_context),
                    "web_content": bool(web_context)
                })
                
            return combined_context
            
        except Exception as e:
            self.logger.error(f"Error processing sub-query {sub_query}: {e}", exc_info=True)
            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "subquery_error",
                    f"❌ Error processing '{sub_query}': {str(e)}",
                    self.researcher.websocket,
                )
            return ""

    async def _execute_mcp_research(self, retriever, query):
        """
        Execute MCP research using the new two-stage approach.
        
        Args:
            retriever: The MCP retriever class
            query: The search query
            
        Returns:
            list: MCP research results
        """
        retriever_name = retriever.__name__
        
        self.logger.info(f"Executing MCP research with {retriever_name} for query: {query}")
        
        try:
            # 使用正确参数实例化 MCP 检索器。
            # 传入同时包含 cfg 和 mcp_configs 的研究器实例。
            retriever_instance = retriever(
                query=query, 
                headers=self.researcher.headers,
                query_domains=self.researcher.query_domains,
                websocket=self.researcher.websocket,
                researcher=self.researcher  # Pass the entire researcher instance
            )
            
            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "mcp_retrieval_stage1",
                    f"🧠 Stage 1: Selecting optimal MCP tools for: {query}",
                    self.researcher.websocket,
                )
            
            # 执行两阶段 MCP 搜索。
            results = retriever_instance.search(
                max_results=self.researcher.cfg.max_search_results_per_query
            )
            
            if results:
                result_count = len(results)
                self.logger.info(f"MCP research completed: {result_count} results from {retriever_name}")
                
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "mcp_research_complete",
                        f"🎯 MCP research completed: {result_count} intelligent results obtained",
                        self.researcher.websocket,
                    )
                
                return results
            else:
                self.logger.info(f"No results returned from MCP research with {retriever_name}")
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "mcp_no_results",
                        f"ℹ️ No relevant information found via MCP for: {query}",
                        self.researcher.websocket,
                    )
                return []
                
        except Exception as e:
            self.logger.error(f"Error in MCP research with {retriever_name}: {str(e)}")
            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "mcp_research_error",
                    f"⚠️ MCP research error: {str(e)} - continuing with other sources",
                    self.researcher.websocket,
                )
            return []

    def _combine_mcp_and_web_context(self, mcp_context: list, web_context: str, sub_query: str) -> str:
        """
        Intelligently combine MCP and web research context.
        
        Args:
            mcp_context: List of MCP context entries
            web_context: Web research context string  
            sub_query: The sub-query being processed
            
        Returns:
            str: Combined context string
        """
        combined_parts = []
        
        # 如果有 Web 上下文，先加入它。
        if web_context and web_context.strip():
            combined_parts.append(web_context.strip())
            self.logger.debug(f"Added web context: {len(web_context)} chars")
        
        # 按统一格式加入 MCP 上下文。
        if mcp_context:
            mcp_formatted = []
            
            for i, item in enumerate(mcp_context):
                content = item.get("content", "")
                url = item.get("url", "")
                title = item.get("title", f"MCP Result {i+1}")
                
                if content and content.strip():
                    # 创建格式规范的上下文条目。
                    if url and url != f"mcp://llm_analysis":
                        citation = f"\n\n*Source: {title} ({url})*"
                    else:
                        citation = f"\n\n*Source: {title}*"
                    
                    formatted_content = f"{content.strip()}{citation}"
                    mcp_formatted.append(formatted_content)
            
            if mcp_formatted:
                # 使用清晰分隔符拼接 MCP 结果。
                mcp_section = "\n\n---\n\n".join(mcp_formatted)
                combined_parts.append(mcp_section)
                self.logger.debug(f"Added {len(mcp_context)} MCP context entries")
        
        # 合并所有部分。
        if combined_parts:
            final_context = "\n\n".join(combined_parts)
            self.logger.info(f"Combined context for '{sub_query}': {len(final_context)} total chars")
            return final_context
        else:
            self.logger.warning(f"No context to combine for sub-query: {sub_query}")
            return ""

    async def _process_sub_query_with_vectorstore(self, sub_query: str, filter: dict | None = None):
        """Takes in a sub query and gathers context from the user provided vector store

        Args:
            sub_query (str): The sub-query generated from the original query

        Returns:
            str: The context gathered from search
        """
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "running_subquery_with_vectorstore_research",
                f"\n🔍 Running research for '{sub_query}'...",
                self.researcher.websocket,
            )

        context = await self.researcher.context_manager.get_similar_content_by_query_with_vectorstore(sub_query, filter)

        return context

    async def _get_new_urls(self, url_set_input):
        """Gets the new urls from the given url set.
        Args: url_set_input (set[str]): The url set to get the new urls from
        Returns: list[str]: The new urls from the given url set
        """

        new_urls = []
        for url in url_set_input:
            if url not in self.researcher.visited_urls:
                self.researcher.visited_urls.add(url)
                new_urls.append(url)
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "added_source_url",
                        f"✅ Added source url to research: {url}\n",
                        self.researcher.websocket,
                        True,
                        url,
                    )

        return new_urls

    async def _get_context_from_initial_results(self, query, query_domains, initial_results):
        """Fetch planning sources through normal retrieval; never treat snippets as pages."""
        if "mcpretriever" in self.researcher.retrievers[0].__name__.lower():
            return ""
        try:
            data = await self._scrape_data_by_urls(
                query, query_domains, initial_results=initial_results
            )
            if data:
                return await self.researcher.context_manager.get_similar_content_by_query(query, data)
        except Exception as e:
            # 规划来源不可用时，不应阻断后续研究。
            self.logger.warning(f"Error reading initial search results: {e}")
        return ""

    async def _search_relevant_source_urls(self, query, query_domains: list | None = None, *, initial_results=None):
        """汇总各检索器的新 URL，并分离可直接使用的预取正文。】需要抓取的 URL
        → 浏览器抓取
        → scraped_content
                       ↘
                         合并 → 所有可用正文
                       ↗
        检索器预取的正文
        → prefetched_content"""
        new_search_urls = []
        prefetched_content = []
        if query_domains is None:
            query_domains = []

        # 遍历当前配置的检索器，兼容检索器被临时修改的情况。
        # 初始结果只来自第一个检索器，复用其内容契约，避免重复搜索或错误归因。
        retrievers = self.researcher.retrievers if initial_results is None else self.researcher.retrievers[:1]
        for retriever_class in retrievers:
            # 跳过 MCP 检索器，因为它们不提供待抓取 URL。
            # MCP 工具返回的不一定是网页 URL，可能是：
            # - 数据库查询结果；
            # - GitHub 仓库内容；
            # - 企业内部知识库；
            # - 文件系统内容；
            # - API 数据；
            # - LLM 整理后的工具结果。
            # 所以 _search_relevant_source_urls() 会跳过 MCP：
            if "mcpretriever" in retriever_class.__name__.lower():
                continue

            try:
                # 使用子查询实例化检索器。
                retriever = retriever_class(query, query_domains=query_domains)

                # 使用当前检索器执行搜索。
                if initial_results is None:
                    search_results = await asyncio.to_thread(
                        retriever.search, max_results=self.researcher.cfg.max_search_results_per_query
                    )
                else:
                    search_results = initial_results

                if not search_results:
                    continue

                # 判断检索器返回的是待抓取 URL 还是已获取正文。
                # 优先使用 BaseRetriever.requires_scraping，未声明时再使用旧的长度启发式规则。
                requires_scraping = getattr(retriever, "requires_scraping", None)

                # 分离已有正文的结果和仍需抓取的结果。
                for result in search_results:
                    url = result.get("href") or result.get("url")
                    raw_content = result.get("raw_content")

                    if not url:
                        continue

                    if requires_scraping is True:
                        # 已声明需要抓取：URL 旁的内容都是预览，无论长短都仍需抓取页面，
                        # 避免把长摘要误当成正文并丢失引用。
                        new_search_urls.append(url)
                    elif requires_scraping is False:
                        # 已声明无需抓取：检索器已经自行获取正文。
                        if raw_content:
                            prefetched_content.append({
                                "url": url,
                                "raw_content": raw_content,
                            })
                            self.researcher.add_research_sources([{"url": url}])
                            # #2100：预取 URL 必须像 _get_new_urls 中的抓取 URL 一样加入 visited_urls，
                            # 否则重复合并时会丢失引用。
                            self.researcher.visited_urls.add(url)
                        else:
                            new_search_urls.append(url)
                    elif raw_content and len(raw_content) > 100:
                        # 未声明时保持旧行为不变。
                        prefetched_content.append({
                            "url": url,
                            "raw_content": raw_content,
                        })
                        self.researcher.add_research_sources([{"url": url}])
                            # #2100：使用与上面相同的 visited_urls 记录方式。
                        self.researcher.visited_urls.add(url)
                    else:
                        new_search_urls.append(url)
            except Exception as e:
                self.logger.error(f"Error searching with {retriever_class.__name__}: {e}")

        # 获取去重后的 URL。
        new_search_urls = await self._get_new_urls(new_search_urls)
        random.shuffle(new_search_urls)

        return new_search_urls, prefetched_content

    async def _scrape_data_by_urls(self, sub_query, query_domains: list | None = None, *, initial_results=None):
        """
        Runs a sub-query across multiple retrievers and scrapes the resulting URLs.
        Retrievers that already provide full content (e.g. PubMed Central) have their
        content passed through directly without re-scraping.

        Args:
            sub_query (str): The sub-query to search for.

        Returns:
            list: A list of scraped content results.
        """
        if query_domains is None:
            query_domains = []

        new_search_urls, prefetched_content = await self._search_relevant_source_urls(
            sub_query, query_domains, initial_results=initial_results
        )

        # 启用详细模式时记录研究过程。
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "researching",
                f"🤔 Researching for relevant information across multiple sources...\n",
                self.researcher.websocket,
            )

        # 抓取仍需获取的 URL，跳过检索器已提供的内容。
        scraped_content = await self.researcher.scraper_manager.browse_urls(new_search_urls)

        # 合并已提供完整正文的检索器返回的预取内容。
        scraped_content.extend(prefetched_content)

        if self.researcher.vector_store:
            self.researcher.vector_store.load(scraped_content)

        return scraped_content

    async def _search(self, retriever, query):
        """
        Perform a search using the specified retriever.
        
        Args:
            retriever: The retriever class to use
            query: The search query
            
        Returns:
            list: Search results
        """
        retriever_name = retriever.__name__
        is_mcp_retriever = "mcpretriever" in retriever_name.lower()
        
        self.logger.info(f"Searching with {retriever_name} for query: {query}")
        
        try:
            # 实例化检索器。
            retriever_instance = retriever(
                query=query, 
                headers=self.researcher.headers,
                query_domains=self.researcher.query_domains,
                websocket=self.researcher.websocket if is_mcp_retriever else None,
                researcher=self.researcher if is_mcp_retriever else None
            )
            
            # 使用 MCP 检索器时记录 MCP 服务配置。
            if is_mcp_retriever and self.researcher.verbose:
                await stream_output(
                    "logs",
                    "mcp_retrieval",
                    f"🔌 Consulting MCP server(s) for information on: {query}",
                    self.researcher.websocket,
                )
            
            # 执行搜索。
            if hasattr(retriever_instance, 'search'):
                results = retriever_instance.search(
                    max_results=self.researcher.cfg.max_search_results_per_query
                )
                
                # 记录结果信息。
                if results:
                    result_count = len(results)
                    self.logger.info(f"Received {result_count} results from {retriever_name}")
                    
                    # 记录 MCP 检索器的特殊信息。
                    if is_mcp_retriever:
                        if self.researcher.verbose:
                            await stream_output(
                                "logs",
                                "mcp_results",
                                f"✓ Retrieved {result_count} results from MCP server",
                                self.researcher.websocket,
                            )
                        
                        # 记录结果详情。
                        for i, result in enumerate(results[:3]):  # Log first 3 results
                            title = result.get("title", "No title")
                            url = result.get("href", "No URL")
                            content_length = len(result.get("body", "")) if result.get("body") else 0
                            self.logger.info(f"MCP result {i+1}: '{title}' from {url} ({content_length} chars)")
                            
                        if result_count > 3:
                            self.logger.info(f"... and {result_count - 3} more MCP results")
                else:
                    self.logger.info(f"No results returned from {retriever_name}")
                    if is_mcp_retriever and self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_no_results",
                            f"ℹ️ No relevant information found from MCP server for: {query}",
                            self.researcher.websocket,
                        )
                
                return results
            else:
                self.logger.error(f"Retriever {retriever_name} does not have a search method")
                return []
        except Exception as e:
            self.logger.error(f"Error searching with {retriever_name}: {str(e)}")
            if is_mcp_retriever and self.researcher.verbose:
                await stream_output(
                    "logs",
                    "mcp_error",
                    f"❌ Error retrieving information from MCP server: {str(e)}",
                    self.researcher.websocket,
                )
            return []
            
    async def _extract_content(self, results):
        """
        Extract content from search results using the browser manager.
        
        Args:
            results: Search results
            
        Returns:
            list: Extracted content
        """
        self.logger.info(f"Extracting content from {len(results)} search results")
        
        # 从搜索结果中获取 URL。
        urls = []
        for result in results:
            if isinstance(result, dict) and "href" in result:
                urls.append(result["href"])
        
        # 没有 URL 时跳过。
        if not urls:
            return []
            
        # 确保不重复访问已经访问过的 URL。
        new_urls = [url for url in urls if url not in self.researcher.visited_urls]
        
        # 没有新 URL 时返回空结果。
        if not new_urls:
            return []
            
        # 抓取 URL 对应的内容。
        scraped_content = await self.researcher.scraper_manager.browse_urls(new_urls)
        
        # 将 URL 加入 visited_urls。
        self.researcher.visited_urls.update(new_urls)
        
        return scraped_content
        
    async def _summarize_content(self, query, content):
        """
        Summarize the extracted content.
        
        Args:
            query: The search query
            content: The extracted content
            
        Returns:
            str: Summarized content
        """
        self.logger.info(f"Summarizing content for query: {query}")
        
        # 没有内容时跳过。
        if not content:
            return ""
            
        # 使用上下文管理器总结内容。
        summary = await self.researcher.context_manager.get_similar_content_by_query(
            query, content
        )
        
        return summary
        
    async def _update_search_progress(self, current, total):
        """
        Update the search progress.
        
        Args:
            current: Current number of sub-queries processed
            total: Total number of sub-queries
        """
        if self.researcher.verbose and self.researcher.websocket:
            progress = int((current / total) * 100)
            await stream_output(
                "logs",
                "research_progress",
                f"📊 Research Progress: {progress}%",
                self.researcher.websocket,
                True,
                {
                    "current": current,
                    "total": total,
                    "progress": progress
                }
            )

import asyncio
import hashlib
import time
from typing import List, Dict, Set, Optional, Any
from fastapi import WebSocket

from gpt_researcher import GPTResearcher


class DetailedReport:
    def __init__(
        self,
        query: str,
        report_type: str,
        report_source: str,
        source_urls: List[str] = [],
        document_urls: List[str] = [],
        query_domains: List[str] = [],
        config_path: str = None,
        tone: Any = "",
        websocket: WebSocket = None,
        subtopics: List[Dict] = [],
        headers: Optional[Dict] = None,
        complement_source_urls: bool = False,
        mcp_configs=None,
        mcp_strategy=None,
        max_search_results=None,
    ):
        """初始化详细报告编排器及跨子主题共享的上下文状态。"""
        self.query = query
        self.report_type = report_type
        self.report_source = report_source
        self.source_urls = source_urls
        self.document_urls = document_urls
        self.query_domains = query_domains
        self.config_path = config_path
        self.tone = tone
        self.websocket = websocket
        self.subtopics = subtopics
        self.headers = headers or {}
        self.complement_source_urls = complement_source_urls
        self.max_search_results = max_search_results
        
        # 为本报告生成唯一研究 ID。
        self.research_id = self._generate_research_id(query)
        
        # 使用可选 MCP 参数初始化研究器。
        gpt_researcher_params = {
            "query": self.query,
            "query_domains": self.query_domains,
            "report_type": "research_report",
            "report_source": self.report_source,
            "source_urls": self.source_urls,
            "document_urls": self.document_urls,
            "config_path": self.config_path,
            "tone": self.tone,
            "websocket": self.websocket,
            "headers": self.headers,
            "complement_source_urls": self.complement_source_urls,
        }

        # 如果提供了 MCP 参数，则加入研究器配置。
        if mcp_configs is not None:
            gpt_researcher_params["mcp_configs"] = mcp_configs
        if mcp_strategy is not None:
            gpt_researcher_params["mcp_strategy"] = mcp_strategy

        self.gpt_researcher = GPTResearcher(**gpt_researcher_params)

        # 如果用户提供了搜索结果数，则覆盖默认值。
        if max_search_results is not None:
            self.gpt_researcher.cfg.max_search_results_per_query = int(max_search_results)
        self.existing_headers: List[Dict] = []
        self.global_context: List[str] = []
        self.global_written_sections: List[str] = []
        self.global_urls: Set[str] = set(
            self.source_urls) if self.source_urls else set()

    def _generate_research_id(self, query: str) -> str:
        """Generate a unique research ID from query and timestamp."""
        timestamp = str(int(time.time()))
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        return f"detailed_{timestamp}_{query_hash}"

    async def run(self) -> str:
        """执行初始研究、逐子主题写作，并组装完整详细报告。"""
        # 先建立全局上下文，再拆分子主题，避免各章节彼此孤立。
        await self._initial_research()
        subtopics = await self._get_all_subtopics()
        report_introduction = await self.gpt_researcher.write_introduction()
        _, report_body = await self._generate_subtopic_reports(subtopics)
        self.gpt_researcher.visited_urls.update(self.global_urls)
        report = await self._construct_detailed_report(report_introduction, report_body)
        return report

    async def _initial_research(self) -> None:
        """完成主问题研究，并保存后续子主题共享的上下文和 URL。"""
        await self.gpt_researcher.conduct_research()
        self.global_context = self.gpt_researcher.context
        self.global_urls = self.gpt_researcher.visited_urls

    async def _get_all_subtopics(self) -> List[Dict]:
        """让主研究器生成子主题，并归一化为任务字典列表。"""
        subtopics_data = await self.gpt_researcher.get_subtopics()

        all_subtopics = []
        if subtopics_data and subtopics_data.subtopics:
            for subtopic in subtopics_data.subtopics:
                all_subtopics.append({"task": subtopic.task})
        else:
            print(f"Unexpected subtopics data format: {subtopics_data}")

        return all_subtopics

    async def _generate_subtopic_reports(self, subtopics: List[Dict]) -> tuple:
        """顺序生成各子主题报告，同时返回结构化结果和拼接正文。"""
        subtopic_reports = []
        subtopics_report_body = ""

        for subtopic in subtopics:
            result = await self._get_subtopic_report(subtopic)
            if result["report"]:
                subtopic_reports.append(result)
                subtopics_report_body += f"\n\n\n{result['report']}"

        return subtopic_reports, subtopics_report_body

    def _hashable_context(self, input_context: List[str] | List[dict]):
        """把字符串或字典上下文统一转成可去重的字符串列表。"""
        # 将上下文转换为字符串，确保可哈希并兼容 MCP 返回的字典。
        context_items = []
        
        for item in input_context:
            if isinstance(item, dict):
                # 将字典上下文转换为字符串格式。
                title = item.get("title", "No title")
                content = item.get("body", item.get("content", ""))
                context_str = f"Title: {title}\nContent: {content}"
                context_items.append(context_str)
            else:
                context_items.append(str(item))
        
        return context_items

    async def _get_subtopic_report(self, subtopic: Dict) -> Dict[str, str]:
        """为单个子主题复用全局资料生成报告，并回写共享状态。"""
        current_subtopic_task = subtopic.get("task")
        subtopic_assistant = GPTResearcher(
            query=current_subtopic_task,
            query_domains=self.query_domains,
            report_type="subtopic_report",
            report_source=self.report_source,
            websocket=self.websocket,
            headers=self.headers,
            parent_query=self.query,
            subtopics=self.subtopics,
            visited_urls=self.global_urls,
            agent=self.gpt_researcher.agent,
            role=self.gpt_researcher.role,
            tone=self.tone,
            complement_source_urls=self.complement_source_urls,
            source_urls=self.source_urls,
            # 传递 MCP 配置，使后续子主题研究器也能使用 MCP。
            mcp_configs=self.gpt_researcher.mcp_configs,
            mcp_strategy=self.gpt_researcher.mcp_strategy
        )

        # 将搜索结果数覆盖值传递给子主题研究器。
        if self.max_search_results is not None:
            subtopic_assistant.cfg.max_search_results_per_query = int(self.max_search_results)

        subtopic_assistant.context = list(set(self._hashable_context(self.global_context)))
        await subtopic_assistant.conduct_research()

        draft_section_titles = await subtopic_assistant.get_draft_section_titles(current_subtopic_task)

        if not isinstance(draft_section_titles, str):
            draft_section_titles = str(draft_section_titles)

        parse_draft_section_titles = self.gpt_researcher.extract_headers(draft_section_titles)
        parse_draft_section_titles_text = [header.get(
            "text", "") for header in parse_draft_section_titles]

        relevant_contents = await subtopic_assistant.get_similar_written_contents_by_draft_section_titles(
            current_subtopic_task, parse_draft_section_titles_text, self.global_written_sections
        )

        # 撰写子主题报告（图片已在主研究阶段预生成）。
        subtopic_report = await subtopic_assistant.write_report(
            existing_headers=self.existing_headers,
            relevant_written_contents=relevant_contents,
        )

        self.global_written_sections.extend(self.gpt_researcher.extract_sections(subtopic_report))
        self.global_context = list(set(self._hashable_context(subtopic_assistant.context)))
        self.global_urls.update(subtopic_assistant.visited_urls)

        self.existing_headers.append({
            "subtopic task": current_subtopic_task,
            "headers": self.gpt_researcher.extract_headers(subtopic_report),
        })

        return {"topic": subtopic, "report": subtopic_report}

    async def _construct_detailed_report(self, introduction: str, report_body: str) -> str:
        """将引言、目录、正文、结论和引用组装为最终报告。"""
        toc = self.gpt_researcher.table_of_contents(report_body)
        conclusion = await self.gpt_researcher.write_report_conclusion(report_body)
        conclusion_with_references = self.gpt_researcher.add_references(
            conclusion, self.gpt_researcher.visited_urls)
        report = f"{introduction}\n\n{toc}\n\n{report_body}\n\n{conclusion_with_references}"
        
        # 图片在 conduct_research() 阶段预生成，并在 write_report() 阶段嵌入。
        return report

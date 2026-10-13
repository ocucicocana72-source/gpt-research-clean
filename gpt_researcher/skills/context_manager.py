"""GPT Researcher 的上下文管理模块。

本模块提供 ContextManager 类，负责研究查询所需的上下文检索、压缩和相似度匹配。
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set

from ..actions.utils import stream_output
from ..context.compression import VectorstoreCompressor, WrittenContentCompressor
from ..context.lexical import rank_written_sections
from ..context.select import select_context

logger = logging.getLogger(__name__)


class ContextManager:
    """管理研究过程中的上下文检索与压缩。

    该类根据查询查找相似内容，统一处理不同来源的上下文，并压缩内容以便后续处理。

    属性：
        researcher：所属的 GPTResearcher 实例。
    """

    def __init__(self, researcher):
        """初始化上下文管理器。

        参数：
            researcher：拥有该管理器的 GPTResearcher 实例。
        """
        self.researcher = researcher

    async def get_similar_content_by_query(self, query: str, pages: list) -> str:
        """根据查询从页面内容中提取相关上下文。

        参数：
            query：用于匹配相关内容的搜索查询。
            pages：待检索的页面内容列表。

        返回：
            由相关内容压缩得到的上下文字符串。
        """
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "fetching_query_content",
                f"📚 Getting relevant content based on query: {query}...",
                self.researcher.websocket,
            )

        # select_context 会根据配置选择具体的过滤/压缩策略；embeddings 使用 lambda 延迟创建，
        # 只有选中的策略确实需要向量模型时才会初始化。
        return await select_context(
            query,
            pages,
            self.researcher.cfg,
            max_results=10,
            prompt_family=self.researcher.prompt_family,
            cost_callback=self.researcher.add_costs,
            embeddings=lambda: self.researcher.memory.get_embeddings(),
        )

    async def get_similar_content_by_query_with_vectorstore(self, query: str, filter: dict | None) -> str:
        """根据查询从向量存储中获取相关上下文。

        参数：
            query：用于匹配相关内容的搜索查询。
            filter：向量存储查询使用的可选过滤条件。

            vector_store_filter
            = 搜索范围过滤
            = “去哪些文档里找？”        在所有向量库中的所有信息中 筛选要查询的向量库

            CONTEXT_FILTER
            = 上下文筛选算法
            = “拿到网页后，如何挑选有用段落？”    在向量库中根据评分标准筛选信息

        返回：
            从向量存储检索并压缩得到的上下文字符串。
        """
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "fetching_query_format",
                f" Getting relevant content based on query: {query}...",
                self.researcher.websocket,
                )
        # 将向量存储及过滤条件封装到压缩器中，再异步检索最相关的内容。
        vectorstore_compressor = VectorstoreCompressor(
            self.researcher.vector_store, filter=filter, prompt_family=self.researcher.prompt_family,
            config=self.researcher.cfg,
            cost_callback=self.researcher.add_costs,
            **self.researcher.kwargs
        )
        return await vectorstore_compressor.async_get_context(query=query, max_results=8)

    async def get_similar_written_contents_by_draft_section_titles(
        self,
        current_subtopic: str,
        draft_section_titles: List[str],
        written_contents: List[Dict],
        max_results: int = 10
    ) -> List[str]:
        """根据当前子主题和章节标题草稿查找相关的已写内容。

        分别使用当前子主题和每个章节标题发起检索，再合并、去重结果。

        参数：
            current_subtopic：当前正在撰写的子主题。
            draft_section_titles：章节标题草稿列表。
            written_contents：之前已经写好的内容字典列表。
            max_results：返回结果的最大数量。

        返回：
            相关的已写内容字符串列表。
        """
        # 当前子主题与各章节标题都作为独立查询，用于扩大相关内容的召回范围。
        all_queries = [current_subtopic] + draft_section_titles

        async def process_query(query: str) -> Set[str]:
            return set(await self.__get_similar_written_contents_by_query(query, written_contents, **self.researcher.kwargs))

        # 并发执行全部查询，然后借助集合合并并去除不同查询返回的重复内容。
        results = await asyncio.gather(*[process_query(query) for query in all_queries])
        relevant_contents = set().union(*results)
        relevant_contents = list(relevant_contents)[:max_results]

        return relevant_contents

    async def __get_similar_written_contents_by_query(
        self,
        query: str,
        written_contents: List[Dict],
        similarity_threshold: float = 0.5,
        max_results: int = 10
    ) -> List[str]:
        """根据单个查询查找相关的已写内容。

        参数：
            query：用于查找相似内容的查询。
            written_contents：已写内容字典列表。
            similarity_threshold：相似度分数的最低阈值。
            max_results：返回结果的最大数量。

        返回：
            相似的已写内容字符串列表。
        """
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "fetching_relevant_written_content",
                f"🔎 Getting relevant written content based on query: {query}...",
                self.researcher.websocket,
            )

        try:
            embeddings = self.researcher.memory.get_embeddings()
        except Exception as e:
            # 没有可用的嵌入模型（未配置，或缺少依赖包/密钥）时，
            # 改用关键词为已写章节排序，保证检索流程仍能继续。
            logger.info(f"Embeddings unavailable ({e}); ranking written content by keywords")
            return rank_written_sections(query, written_contents, max_results=max_results)

        # 嵌入模型可用时，按向量相似度筛选并压缩已写内容。
        written_content_compressor = WrittenContentCompressor(
            documents=written_contents,
            embeddings=embeddings,
            similarity_threshold=similarity_threshold,
            **self.researcher.kwargs
        )
        return await written_content_compressor.async_get_context(
            query=query, max_results=max_results, cost_callback=self.researcher.add_costs
        )

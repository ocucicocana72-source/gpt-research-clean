"""GPT Researcher 的来源整理技能。

本模块提供 SourceCurator 类，根据相关性、可信度和可靠性
评估并排序研究来源。
source_data 原始来源
        ↓
curate_sources() 把评估标准写进 Prompt
        ↓
create_chat_completion() 将 Prompt 发送给 LLM
        ↓
LLM 判断相关性、可信度等，并重新排序、筛选
        ↓
parse_json_markdown() 解析 LLM 返回的 JSON
        ↓
curated_sources
"""

import logging
from typing import Dict, List, Optional

import json_repair
from langchain_core.utils.json import parse_json_markdown

from ..actions import stream_output
from ..config.config import Config
from ..utils.llm import create_chat_completion

logger = logging.getLogger(__name__)


class SourceCurator:
    """根据相关性、可信度和可靠性对来源排序并筛选。

    该类使用基于 LLM 的评估来审查研究来源，
    并选出最适合用于生成报告的来源。

    属性:
        researcher: 持有该来源整理器的 GPTResearcher 实例。
    """

    def __init__(self, researcher):
        """初始化来源整理器。

        参数:
            researcher: 持有该整理器的 GPTResearcher 实例。
        """
        self.researcher = researcher

    async def curate_sources(
        self,
        source_data: List,
        max_results: int = 10,
    ) -> List:
        """根据研究任务和评估准则对来源排序并筛选。

        参数:
            source_data: 待排序的来源文档列表。
            max_results: 最多返回的高质量来源数量。

        返回:
            排序并筛选后的来源列表；发生异常时返回原始列表。
        """
        logger.debug(f"Curating {len(source_data)} sources")
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "research_plan",
                f"⚖️ Evaluating and curating sources by credibility and relevance...",
                self.researcher.websocket,
            )

        response = ""
        try:
            # 使用配置的高能力模型评估来源；prompt 中会带入查询、原始来源和数量上限。
            response = await create_chat_completion(
                model=self.researcher.cfg.smart_llm_model,
                messages=[
                    {"role": "system", "content": f"{self.researcher.role}"},
                    {"role": "user", "content": self.researcher.prompt_family.curate_sources(
                        self.researcher.query, source_data, max_results)},
                ],
                temperature=0.2,
                max_tokens=8000,
                llm_provider=self.researcher.cfg.smart_llm_provider,
                llm_kwargs=self.researcher.cfg.llm_kwargs,
                cost_callback=self.researcher.add_costs,
            )
            #LLM 返回的 response 是字符串，不是列表。
            # parse_json_markdown() 才负责把字符串转换为 Python 列表。
            # LLM 即使收到只返回 JSON 的指令，仍可能添加 ```json 围栏或额外文字，
            # 这些内容无法被普通 json.loads 直接解析。此处沿用项目其他模块的容错方式
            # （参见 actions/query_processing.py 和 multi_agents/agents/utils/llms.py），
            # 避免丢弃内容合法但带有围栏的响应。
            curated_sources = parse_json_markdown(response, parser=json_repair.loads)
            # json_repair 不会因无法使用的输出抛出异常，而可能返回 "" 或字典。
            # 显式要求结果为列表，以便异常结果进入下方的回退逻辑，而不是清空研究上下文。
            if not isinstance(curated_sources, list):
                raise ValueError(
                    f"expected a JSON list of sources, got "
                    f"{type(curated_sources).__name__}"
                )
            logger.debug(f"Curated {len(curated_sources)} of {len(source_data)} sources")

            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "research_plan",
                    f"🏅 Verified and ranked top {len(curated_sources)} most reliable sources",
                    self.researcher.websocket,
                )

            return curated_sources

        except Exception as e:
            # 模型调用或响应解析失败时，保留全部原始来源，确保下游仍有可用上下文。
            logger.error(f"Error in curate_sources: {e}")
            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "research_plan",
                    f"🚫 Source verification failed: {str(e)}",
                    self.researcher.websocket,
                )
            return source_data

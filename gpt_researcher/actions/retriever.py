"""Retriever factory and utilities for GPT Researcher.

This module provides functions to instantiate and manage various
search retriever implementations.
"""

import logging
from importlib.metadata import entry_points

logger = logging.getLogger(__name__)

#: Entry-point group that third-party packages use to register retrievers.
RETRIEVER_ENTRY_POINT_GROUP = "gpt_researcher.retrievers"


def get_retriever(retriever: str):
    """根据名称获取检索器（retriever）类。

    参数:
    retriever: 要获取的检索器名称（例如：'google'、'tavily'、'duckduckgo'）。

    返回:
    如果找到，则返回检索器类；否则返回 None。

    支持的检索器:
    - google: Google 自定义搜索
    - searx: SearX 搜索引擎
    - searchapi: SearchAPI 服务
    - serpapi: SerpAPI 服务
    - serper: Serper API
    - duckduckgo: DuckDuckGo 搜索
    - bing: Bing 搜索
    - brave: Brave Search API
    - arxiv: arXiv 学术搜索
    - tavily: Tavily 搜索 API
    - exa: Exa 搜索
    - crw: fastCRW 搜索（兼容 Firecrawl 的网页抓取工具）
    - semantic_scholar: Semantic Sc​​holar 学术搜索
    - pubmed_central: PubMed Central 医学文献
    - openalex: OpenAlex 学术著作目录
    - custom: 用户自定义检索器
    - mcp: Model Context Protocol (MCP) 检索器
    - xquik: Xquik X/Twitter 搜索
    - getxapi: GetXAPI X/Twitter 搜索

    对于其他名称，系统会在注册于 ``gpt_researcher.retrievers`` 入口点组（entry-point group）下的已安装插件中进行查找。内置名称具有最高优先级。
    """
    match retriever:
        case "google":
            from gpt_researcher.retrievers import GoogleSearch

            return GoogleSearch
        case "searx":
            from gpt_researcher.retrievers import SearxSearch

            return SearxSearch
        case "searchapi":
            from gpt_researcher.retrievers import SearchApiSearch

            return SearchApiSearch
        case "serpapi":
            from gpt_researcher.retrievers import SerpApiSearch

            return SerpApiSearch
        case "serper":
            from gpt_researcher.retrievers import SerperSearch

            return SerperSearch
        case "duckduckgo":
            from gpt_researcher.retrievers import Duckduckgo

            return Duckduckgo
        case "bing":
            from gpt_researcher.retrievers import BingSearch

            return BingSearch
        case "brave":
            from gpt_researcher.retrievers import BraveSearch

            return BraveSearch
        case "bocha":
            from gpt_researcher.retrievers import BoChaSearch

            return BoChaSearch
        case "arxiv":
            from gpt_researcher.retrievers import ArxivSearch

            return ArxivSearch
        case "tavily":
            from gpt_researcher.retrievers import TavilySearch

            return TavilySearch
        case "groundroute":
            from gpt_researcher.retrievers import GroundRouteSearch

            return GroundRouteSearch
        case "exa":
            from gpt_researcher.retrievers import ExaSearch

            return ExaSearch
        case "crw":
            from gpt_researcher.retrievers import CRWRetriever

            return CRWRetriever
        case "semantic_scholar":
            from gpt_researcher.retrievers import SemanticScholarSearch

            return SemanticScholarSearch
        case "pubmed_central":
            from gpt_researcher.retrievers import PubMedCentralSearch

            return PubMedCentralSearch
        case "custom":
            from gpt_researcher.retrievers import CustomRetriever

            return CustomRetriever
        case "mcp":
            from gpt_researcher.retrievers import MCPRetriever

            return MCPRetriever
        case "xquik":
            from gpt_researcher.retrievers import XquikSearch

            return XquikSearch
        case "openalex":
            from gpt_researcher.retrievers import OpenAlexSearch

            return OpenAlexSearch
        case "getxapi":
            from gpt_researcher.retrievers import GetXAPISearch

            return GetXAPISearch

        case _:
            return _load_plugin_retriever(retriever)


def _load_plugin_retriever(name: str):
    """Load a retriever class that an installed package registered by name."""
    for entry_point in entry_points(group=RETRIEVER_ENTRY_POINT_GROUP, name=name):
        try:
            return entry_point.load()
        except Exception as exc:
            logger.warning(f"Failed to load retriever plugin '{name}' ({entry_point.value}): {exc}")
            return None
    return None


def get_retrievers(headers: dict[str, str], cfg):
    """
    根据头部信息（headers）、配置（config）或默认设置，确定要使用的检索器（retriever）。

    参数：
    headers (dict): 头部信息字典
    cfg: 配置对象

    返回：
    list: 待用于搜索的检索器类列表。
    """
    # Check headers first for multiple retrievers
    if headers.get("retrievers"):
        retrievers = headers.get("retrievers").split(",")
    # If not found, check headers for a single retriever
    elif headers.get("retriever"):
        retrievers = [headers.get("retriever")]
    # If not in headers, check config for multiple retrievers
    elif cfg.retrievers:
        # Handle both list and string formats for config retrievers
        if isinstance(cfg.retrievers, str):
            retrievers = cfg.retrievers.split(",")
        else:
            retrievers = cfg.retrievers
    # If not found, check config for a single retriever
    elif cfg.retriever:
        retrievers = [cfg.retriever]
    # If still not set, use default retriever
    else:
        retrievers = [get_default_retriever().__name__]

    # 去除每个检索器名称两端的空白字符，
    # 这样即使逗号分隔的列表中包含空格（例如来自标头或配置的 "tavily, exa"），
    # 也能正确解析，而不会静默回退到默认检索器。
    retrievers = [r.strip() for r in retrievers if r and r.strip()]

    if not retrievers:
        return [get_default_retriever()]

    # Convert retriever names to actual retriever classes
    # Use get_default_retriever() as a fallback for any invalid retriever names
    retriever_classes = [get_retriever(r) or get_default_retriever() for r in retrievers]
    
    return retriever_classes


def get_default_retriever():
    """Get the default retriever class.

    Returns:
        The TavilySearch retriever class as the default search provider.
    """
    from gpt_researcher.retrievers import TavilySearch

    return TavilySearch

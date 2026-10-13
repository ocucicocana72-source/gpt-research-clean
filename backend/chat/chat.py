import logging
import os
import json
from fastapi import WebSocket
from typing import List, Dict, Any

from gpt_researcher.config.config import Config
from gpt_researcher.context.select import select_context
from gpt_researcher.utils.llm import create_chat_completion
from gpt_researcher.utils.tools import create_chat_completion_with_tools, create_search_tool
try:
    from tavily import TavilyClient
except ImportError:  # optional dependency for chat web search
    TavilyClient = None
from datetime import datetime

# 配置日志。
# 获取 logger 实例。
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Only log to console
    ]
)

# LLM 客户端由 GPT Researcher 统一的 LLM 系统处理。
# 该系统支持所有已配置的提供商（OpenAI、Google Gemini、Anthropic 等）。

def get_tools():
    """Define tools for LLM function calling (primarily for OpenAI-compatible providers)"""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "quick_search",
                "description": "Search for current events or online information when you need new knowledge that doesn't exist in the current context",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    return tools

class ChatAgentWithMemory:
    def __init__(
        self,
        report: str,
        config_path="default",
        headers=None,
        vector_store=None
    ):
        """初始化报告问答 Agent，并保存检索与模型配置。"""
        self.report = report
        self.headers = headers
        self.config = Config(config_path)
        self.vector_store = vector_store
        self.retriever = None
        self.search_metadata = None
        
        # 初始化 Tavily 客户端（可选，仅在存在 API 密钥时启用）。
        tavily_api_key = os.environ.get("TAVILY_API_KEY")
        if tavily_api_key and TavilyClient is not None:
            self.tavily_client = TavilyClient(api_key=tavily_api_key)
        else:
            self.tavily_client = None
            if TavilyClient is None:
                logger.warning("tavily package not installed - web search in chat will be disabled")
            else:
                logger.warning("TAVILY_API_KEY not set - web search in chat will be disabled")
        
        # 调用方提供的向量存储由调用方自行查询；否则报告使用与研究相同的上下文过滤器。
        if self.vector_store is not None:
            try:
                self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
            except TypeError:
                self.retriever = self.vector_store.as_retriever()

    def quick_search(self, query):
        """Perform a web search for current information using Tavily"""
        try:
            # 检查 Tavily 客户端是否可用。
            if self.tavily_client is None:
                logger.warning(f"Tavily client not available, skipping web search for: {query}")
                self.search_metadata = {
                    "query": query,
                    "sources": [],
                    "error": "Web search is disabled - TAVILY_API_KEY not configured"
                }
                return {
                    "error": "Web search is disabled - TAVILY_API_KEY not configured",
                    "results": []
                }
            
            logger.info(f"Performing web search for: {query}")
            results = self.tavily_client.search(query=query, max_results=5)
            
            # 保存供前端使用的搜索元数据。
            self.search_metadata = self._build_search_metadata(query, results)
            
            return results
        except Exception as e:
            logger.error(f"Error performing web search: {str(e)}", exc_info=True)
            results = {
                "error": str(e),
                "results": []
            }
            self.search_metadata = self._build_search_metadata(query, results)
            return results

    @staticmethod
    def _build_search_metadata(query, results):
        """Describe one search result without making another provider request."""
        metadata = {"query": query, "sources": []}
        for result in results.get("results", []):
            content = result.get("content", "")
            metadata["sources"].append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": content[:200] + "..." if len(content) > 200 else content,
            })
        if "error" in results:
            metadata["error"] = results["error"]
        return metadata


    async def process_chat_completion(self, messages: List[Dict[str, str]]):
        """Process chat completion using configured LLM provider with tool calling support"""
        processed_metadata = []

        def search_with_metadata(query):
            """执行一次快速检索，同时记录本轮工具调用的来源元数据。"""
            results = self.quick_search(query)
            # 将来源限制在本轮完成过程，并与返回给模型的证据绑定，包括重复查询和失败记录。
            processed_metadata.append({
                "tool": "quick_search",
                "query": query,
                "search_metadata": self._build_search_metadata(query, results),
            })
            return results

        search_tool = create_search_tool(search_with_metadata)
        
        # 使用支持工具调用的聊天补全工具。
        response, tool_calls_metadata = await create_chat_completion_with_tools(
            messages=messages,
            tools=[search_tool],
            model=self.config.smart_llm_model,
            llm_provider=self.config.smart_llm_provider,
            llm_kwargs=self.config.llm_kwargs,
        )
        
        # 回退到普通补全时工具不会返回元数据；此时答案不是基于这些搜索结果生成的。
        return response, processed_metadata if tool_calls_metadata else []



    async def _retrieve_context(self, user_message: str) -> str:
        """The parts of the report relevant to the latest user message.

        Uses the injected vector store when there is one, otherwise the same
        context filter research uses. Falls back to the full report, so chat
        always has something to answer from.
        """
        if not self.report or not user_message:
            return self.report or ""
        if self.retriever is not None:
            try:
                docs = self.retriever.invoke(user_message)
                chunks = [str(getattr(d, "page_content", "") or "") for d in docs or []]
                if any(chunks):
                    return "\n\n".join(c for c in chunks if c)
            except Exception as exc:  # noqa: BLE001 - retrieval must not break chat
                logger.warning(f"Report retrieval failed, using the context filter: {exc}")
        try:
            context = await select_context(
                user_message,
                [{"url": "report", "title": "Research report", "raw_content": self.report}],
                self.config,
            )
        except Exception as exc:  # noqa: BLE001 - chat must keep working
            logger.warning(f"Report context selection failed, using full report: {exc}")
            return self.report
        return context or self.report

    async def chat(self, messages, websocket=None):
        """Chat with configured LLM provider (supports OpenAI, Google Gemini, Anthropic, etc.)
        
        Args:
            messages: List of chat messages with role and content
            websocket: Optional websocket for streaming responses
        
        Returns:
            tuple: (str: The AI response message, dict: metadata about tool usage)
        """
        try:
            
            # 优先使用检索到的报告片段，避免每轮都填入整份报告。
            last_user = ""
            for msg in reversed(messages or []):
                if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                    last_user = str(msg.get("content"))
                    break
            report_context = await self._retrieve_context(last_user)

            # 使用报告上下文组装系统提示词。
            system_prompt = f"""
            You are GPT Researcher, an autonomous research agent created by an open source community at https://github.com/assafelovic/gpt-researcher, homepage: https://gptr.dev. 
            To learn more about GPT Researcher you can suggest to check out: https://docs.gptr.dev.
            
            This is a chat about a research report that you created. Answer based on the given context and report.
            You must include citations to your answer based on the report.
            
            You may use the quick_search tool when the user asks about information that might require current data 
            not found in the report, such as recent events, updated statistics, or news. If there's no report available,
            you can use the quick_search tool to find information online.
            
            You must respond in markdown format. You must make it readable with paragraphs, tables, etc when possible. 
            Remember that you're answering in a chat not a report.
            
            Assume the current time is: {datetime.now()}.
            
            Report: {report_context}
            
            """
            
            # 将消息历史整理为 OpenAI 输入格式。
            formatted_messages = []
            
            # 先加入系统消息。
            formatted_messages.append({
                "role": "system", 
                "content": system_prompt
            })
            
            # 加入用户/助手消息历史，并过滤非必要字段。
            for msg in messages:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                else:
                    logger.warning(f"Skipping message with missing role or content: {msg}")
            
            # 使用已配置的 LLM 提供商处理对话。
            ai_message, tool_calls_metadata = await self.process_chat_completion(formatted_messages)
            
            # 消息为空时提供兜底响应。
            if not ai_message:
                logger.warning("No AI message content found in response, using fallback message")
                ai_message = "I apologize, but I couldn't generate a proper response. Please try asking your question again."
            
            logger.info(f"Generated response: {ai_message[:100]}..." if len(ai_message) > 100 else f"Generated response: {ai_message}")
            
            # 同时返回消息和工具调用元数据。
            return ai_message, tool_calls_metadata
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}", exc_info=True)
            raise

    def get_context(self):
        """return the current context of the chat"""
        return self.report

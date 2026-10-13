import asyncio
import json
import os
import re
import time
import shutil
import traceback
from typing import Awaitable, Dict, List, Any
from fastapi.responses import JSONResponse, FileResponse
from gpt_researcher.document.document import DocumentLoader
from gpt_researcher import GPTResearcher
from gpt_researcher.actions import stream_output
# 此模块可能以两个包名导入：
# `backend.server.server_utils`（main.py / Procfile 入口）或
# `server.server_utils`（backend/server/app.py 会把 backend/ 加入 sys.path）。
# 只有前者能解析 `backend.utils`，因此保留裸模块名作为回退。
try:
    from backend.utils import write_md_to_pdf, write_md_to_word, write_text_to_md
except ImportError:  # pragma: no cover - legacy sys.path-shimmed import
    from utils import write_md_to_pdf, write_md_to_word, write_text_to_md
from pathlib import Path
from datetime import datetime
from fastapi import HTTPException, WebSocketDisconnect
import logging
import hashlib

from .multi_agent_runner import run_multi_agent_task

# 导入聊天 Agent。
try:
    import sys
    backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    from chat.chat import ChatAgentWithMemory
except ImportError:
    ChatAgentWithMemory = None

logger = logging.getLogger(__name__)

class CustomLogsHandler:
    """Custom handler to capture streaming logs from the research process"""
    def __init__(self, websocket, task: str):
        """初始化实时日志处理器，并为当前任务创建结构化日志文件。"""
        self.logs = []
        self.websocket = websocket
        sanitized_filename = sanitize_filename(f"task_{int(time.time())}_{task}")
        self.log_file = os.path.join("outputs", f"{sanitized_filename}.json")
        self.timestamp = datetime.now().isoformat()
        # 使用元数据初始化日志文件。
        os.makedirs("outputs", exist_ok=True)
        with open(self.log_file, 'w') as f:
            json.dump({
                "timestamp": self.timestamp,
                "events": [],
                "content": {
                    "query": "",
                    "sources": [],
                    "context": [],
                    "report": "",
                    "costs": 0.0
                }
            }, f, indent=2)

    async def send_json(self, data: Dict[str, Any]) -> None:
        """Store log data and send to websocket"""
        # 发送到 WebSocket，以便实时显示。
        if self.websocket:
            await self.websocket.send_json(data)
            
        # 读取当前日志文件。
        with open(self.log_file, 'r') as f:
            log_data = json.load(f)
            
        # 根据数据类型更新对应区块。
        if data.get('type') == 'logs':
            log_data['events'].append({
                "timestamp": datetime.now().isoformat(),
                "type": "event",
                "data": data
            })
        else:
            # 其他数据更新内容区块。
            log_data['content'].update(data)
            
        # 保存更新后的日志文件。
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)


class Researcher:
    def __init__(self, query: str, report_type: str = "research_report"):
        """创建一个适合后端调用的 GPTResearcher 包装对象。"""
        self.query = query
        self.report_type = report_type
        # 为本次研究任务生成唯一 ID。
        self.research_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(query)}"
        # 使用研究 ID 初始化日志处理器。
        self.logs_handler = CustomLogsHandler(None, self.research_id)
        self.researcher = GPTResearcher(
            query=query,
            report_type=report_type,
            websocket=self.logs_handler
        )

    async def research(self) -> dict:
        """Conduct research and return paths to generated files"""
        await self.researcher.conduct_research()
        report = await self.researcher.write_report()
        
        # 生成报告文件。
        sanitized_filename = sanitize_filename(f"task_{int(time.time())}_{self.query}")
        file_paths = await generate_report_files(report, sanitized_filename)
        
        # 获取 CustomLogsHandler 创建的 JSON 日志路径。
        json_relative_path = os.path.relpath(self.logs_handler.log_file)
        
        return {
            "output": {
                **file_paths,  # Include PDF, DOCX, and MD paths
                "json": json_relative_path
            }
        }

def sanitize_filename(filename: str) -> str:
    """把任务标题压缩为不暴露原始查询内容的安全文件名。"""
    # 拆分文件名组成部分。
    prefix, timestamp, *task_parts = filename.split('_')
    task = '_'.join(task_parts)
    task_hash = hashlib.md5(task.encode('utf-8', errors='ignore')).hexdigest()[:10]
            
    # 重新组合并清理文件名。
    sanitized = f"{prefix}_{timestamp}_{task_hash}"
    return re.sub(r"[^\w\s-]", "", sanitized).strip()


async def handle_start_command(websocket, data: str, manager):
    """解析 WebSocket 的 ``start`` 命令并启动研究与报告导出。"""
    json_data = json.loads(data[6:])
    (
        task,
        report_type,
        source_urls,
        document_urls,
        tone,
        headers,
        report_source,
        query_domains,
        mcp_enabled,
        mcp_strategy,
        mcp_configs,
        max_search_results,
    ) = extract_command_data(json_data)

    if not task or not report_type:
        print("Error: Missing task or report_type")
        return

    # 使用 WebSocket 和任务创建日志处理器。
    logs_handler = CustomLogsHandler(websocket, task)
    # 使用查询初始化日志内容。
    await logs_handler.send_json({
        "query": task,
        "sources": [],
        "context": [],
        "report": ""
    })

    sanitized_filename = sanitize_filename(f"task_{int(time.time())}_{task}")

    # WebSocket 管理器在这里把协议层参数转交给核心研究编排器。
    report = await manager.start_streaming(
        task,
        report_type,
        report_source,
        source_urls,
        document_urls,
        tone,
        websocket,
        headers,
        query_domains,
        mcp_enabled,
        mcp_strategy,
        mcp_configs,
        max_search_results,
    )
    report = str(report)
    file_paths = await generate_report_files(report, sanitized_filename)
    # 将 JSON 日志路径加入文件路径字典。
    file_paths["json"] = os.path.relpath(logs_handler.log_file)
    await send_file_paths(websocket, file_paths)


async def handle_human_feedback(data: str):
    """解析并记录客户端发送的人工反馈消息。"""
    feedback_data = json.loads(data[14:])  # Remove "human_feedback" prefix
    print(f"Received human feedback: {feedback_data}")
    # TODO：增加将反馈转发给对应 Agent 或更新研究状态的逻辑。


async def handle_chat_command(websocket, data: str):
    """Handle chat command from WebSocket."""
    try:
        # 解析聊天数据，格式为 "chat {json_data}"。
        json_str = data[5:].strip()  # Remove "chat " prefix
        chat_data = json.loads(json_str)
        
        message = chat_data.get("message", "")
        report = chat_data.get("report", "")
        messages = chat_data.get("messages", [])
        
        # 如果只提供 message，则转换为 messages 格式。
        if message and not messages:
            messages = [{"role": "user", "content": message}]
        
        if not messages:
            await websocket.send_json({
                "type": "chat",
                "content": "No message provided.",
                "role": "assistant"
            })
            return
        
        # 检查 ChatAgentWithMemory 是否可用。
        if ChatAgentWithMemory is None:
            await websocket.send_json({
                "type": "chat",
                "content": "Chat functionality is not available. Please check the server configuration.",
                "role": "assistant"
            })
            return
        
        # 使用报告上下文创建聊天 Agent。
        chat_agent = ChatAgentWithMemory(
            report=report,
            config_path="default",
            headers=None
        )
        
        # 处理聊天请求。
        response_content, tool_calls_metadata = await chat_agent.chat(messages, websocket)
        
        # 通过 WebSocket 发送响应。
        await websocket.send_json({
            "type": "chat",
            "content": response_content,
            "role": "assistant",
            "metadata": {
                "tool_calls": tool_calls_metadata
            } if tool_calls_metadata else None
        })
        
        logger.info(f"Chat response sent successfully")
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse chat data: {e}")
        await websocket.send_json({
            "type": "chat",
            "content": f"Error: Invalid message format - {str(e)}",
            "role": "assistant"
        })
    except Exception as e:
        logger.error(f"Error handling chat command: {e}\n{traceback.format_exc()}")
        await websocket.send_json({
            "type": "chat",
            "content": f"Error processing your message: {str(e)}",
            "role": "assistant"
        })

async def generate_report_files(report: str, filename: str) -> Dict[str, str]:
    """把同一份 Markdown 报告导出为 PDF、DOCX 和 MD 文件。"""
    pdf_path = await write_md_to_pdf(report, filename)
    docx_path = await write_md_to_word(report, filename)
    md_path = await write_text_to_md(report, filename)
    return {"pdf": pdf_path, "docx": docx_path, "md": md_path}


async def send_file_paths(websocket, file_paths: Dict[str, str]):
    """通过 WebSocket 把已生成报告的下载路径发送给客户端。"""
    await websocket.send_json({"type": "path", "output": file_paths})


def get_config_dict(
    langchain_api_key: str, openai_api_key: str, tavily_api_key: str,
    google_api_key: str, google_cx_key: str, bing_api_key: str,
    searchapi_api_key: str, serpapi_api_key: str, serper_api_key: str, searx_url: str
) -> Dict[str, str]:
    """用显式参数优先、进程环境兜底的方式组装服务配置。"""
    return {
        "LANGCHAIN_API_KEY": langchain_api_key or os.getenv("LANGCHAIN_API_KEY", ""),
        "OPENAI_API_KEY": openai_api_key or os.getenv("OPENAI_API_KEY", ""),
        "TAVILY_API_KEY": tavily_api_key or os.getenv("TAVILY_API_KEY", ""),
        "GOOGLE_API_KEY": google_api_key or os.getenv("GOOGLE_API_KEY", ""),
        "GOOGLE_CX_KEY": google_cx_key or os.getenv("GOOGLE_CX_KEY", ""),
        "BING_API_KEY": bing_api_key or os.getenv("BING_API_KEY", ""),
        "SEARCHAPI_API_KEY": searchapi_api_key or os.getenv("SEARCHAPI_API_KEY", ""),
        "SERPAPI_API_KEY": serpapi_api_key or os.getenv("SERPAPI_API_KEY", ""),
        "SERPER_API_KEY": serper_api_key or os.getenv("SERPER_API_KEY", ""),
        "SEARX_URL": searx_url or os.getenv("SEARX_URL", ""),
        "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2", "true"),
        "DOC_PATH": os.getenv("DOC_PATH", "./my-docs"),
        "RETRIEVER": os.getenv("RETRIEVER", ""),
        "EMBEDDING_MODEL": os.getenv("OPENAI_EMBEDDING_MODEL", "")
    }


def update_environment_variables(config: Dict[str, str]):
    """把配置字典写入当前服务进程的环境变量。"""
    for key, value in config.items():
        os.environ[key] = value


async def handle_file_upload(file, DOC_PATH: str) -> Dict[str, str]:
    """安全保存上传文件，并重新加载本地文档数据源。"""
    file_path = os.path.join(DOC_PATH, os.path.basename(file.filename))
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f"File uploaded to {file_path}")

    document_loader = DocumentLoader(DOC_PATH)
    await document_loader.load()

    return {"filename": file.filename, "path": file_path}


async def handle_file_deletion(filename: str, DOC_PATH: str) -> JSONResponse:
    """删除知识库目录中的文件，并返回对应 HTTP JSON 响应。"""
    file_path = os.path.join(DOC_PATH, os.path.basename(filename))
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File deleted: {file_path}")
        return JSONResponse(content={"message": "File deleted successfully"})
    else:
        print(f"File not found: {file_path}")
        return JSONResponse(status_code=404, content={"message": "File not found"})


async def execute_multi_agents(manager) -> Any:
    """使用首个活动 WebSocket 运行多 Agent 研究任务。"""
    websocket = manager.active_connections[0] if manager.active_connections else None
    if websocket:
        report = await run_multi_agent_task("Is AI in a hype cycle?", websocket, stream_output)
        return {"report": report}
    else:
        return JSONResponse(status_code=400, content={"message": "No active WebSocket connection"})


async def handle_websocket_communication(websocket, manager):
    """持续接收 WebSocket 命令，并管理单个连接上的后台研究任务。"""
    running_task: asyncio.Task | None = None

    def run_long_running_task(awaitable: Awaitable) -> asyncio.Task:
        """把长任务包装成可追踪、可取消且会向客户端报告错误的 Task。"""
        async def safe_run():
            """执行协程，并统一处理取消和异常消息。"""
            try:
                await awaitable
            except asyncio.CancelledError:
                logger.info("Task cancelled.")
                raise
            except Exception as e:
                logger.error(f"Error running task: {e}\n{traceback.format_exc()}")
                await websocket.send_json(
                    {
                        "type": "logs",
                        "content": "error",
                        "output": f"Error: {e}",
                    }
                )

        return asyncio.create_task(safe_run())

    try:
        while True:
            try:
                data = await websocket.receive_text()
                logger.info(f"Received WebSocket message: {data[:50]}..." if len(data) > 50 else data)
                
                if data == "ping":
                    await websocket.send_text("pong")
                elif running_task and not running_task.done():
                    # 如果已有任务运行，则丢弃新请求。
                    logger.warning(
                        f"Received request while task is already running. Request data preview: {data[: min(20, len(data))]}..."
                    )
                    await websocket.send_json(
                        {
                            "type": "logs",
                            "content": "warning",
                            "output": "Task already running. Please wait.",
                        }
                    )
                # 去除空白后使用 startswith 统一判断命令类型。
                elif data.strip().startswith("start"):
                    logger.info(f"Processing start command")
                    running_task = run_long_running_task(
                        handle_start_command(websocket, data, manager)
                    )
                elif data.strip().startswith("human_feedback"):
                    logger.info(f"Processing human_feedback command")
                    running_task = run_long_running_task(handle_human_feedback(data))
                elif data.strip().startswith("chat"):
                    logger.info(f"Processing chat command")
                    running_task = run_long_running_task(handle_chat_command(websocket, data))
                else:
                    error_msg = f"Error: Unknown command or not enough parameters provided. Received: '{data[:100]}...'" if len(data) > 100 else f"Error: Unknown command or not enough parameters provided. Received: '{data}'"
                    logger.error(error_msg)
                    print(error_msg)
                    await websocket.send_json({
                        "type": "error",
                        "content": "error",
                        "output": "Unknown command received by server"
                    })
            except WebSocketDisconnect:
                # 标签页关闭或新研究任务关闭旧连接属于正常断开，不是错误；交给端点清理。
                raise
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}\n{traceback.format_exc()}")
                print(f"WebSocket error: {e}")
                break
    finally:
        if running_task and not running_task.done():
            running_task.cancel()

def extract_command_data(json_data: Dict) -> tuple:
    """按研究启动函数需要的固定顺序提取命令字段。"""
    return (
        json_data.get("task"),
        json_data.get("report_type"),
        json_data.get("source_urls"),
        json_data.get("document_urls"),
        json_data.get("tone"),
        json_data.get("headers", {}),
        json_data.get("report_source"),
        json_data.get("query_domains", []),
        json_data.get("mcp_enabled", False),
        json_data.get("mcp_strategy", "fast"),
        json_data.get("mcp_configs", []),
        json_data.get("max_search_results"),
    )

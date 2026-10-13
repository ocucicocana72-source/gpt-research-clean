import json
import os
from typing import Dict, List, Any
import time
import logging
import sys
import warnings
from pathlib import Path

# 忽略 Pydantic V2 迁移警告。
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, File, UploadFile, BackgroundTasks, HTTPException
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel, ConfigDict

# 将父目录加入 sys.path，确保可以导入 server 包。
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from server.websocket_manager import WebSocketManager
from server.server_utils import (
    get_config_dict, sanitize_filename,
    update_environment_variables, handle_file_upload, handle_file_deletion,
    execute_multi_agents, handle_websocket_communication
)
from server.agent_discovery import build_agent_discovery_document

from server.websocket_manager import run_agent
from utils import write_md_to_word, write_md_to_pdf
from gpt_researcher.utils.enum import Tone
from chat.chat import ChatAgentWithMemory

from server.report_store import ReportStore

# 已移除 MongoDB 服务，当前不需要数据库持久化。

# 配置日志。
logger = logging.getLogger(__name__)

# 不覆盖上层 logger 配置。
logger.propagate = True

# 降低 uvicorn 热重载日志噪声。
logging.getLogger("uvicorn.supervisors.ChangeReload").setLevel(logging.WARNING)

# 请求模型。


class ResearchRequest(BaseModel):
    task: str
    report_type: str
    report_source: str
    tone: str
    headers: dict | None = None
    repo_name: str
    branch_name: str
    generate_in_background: bool = True


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # Allow extra fields in the request
    
    report: str
    messages: List[Dict[str, Any]]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理 FastAPI 启动与关闭阶段，并挂载报告和前端静态目录。"""
    # 启动阶段。
    os.makedirs("outputs", exist_ok=True)
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
    
    # 挂载前端静态文件。
    frontend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")
    if os.path.exists(frontend_path):
        app.mount("/site", StaticFiles(directory=frontend_path), name="frontend")
        logger.debug(f"Frontend mounted from: {frontend_path}")
        
        # 同时直接挂载 static 目录，支持 /static/ 资源引用。
        static_path = os.path.join(frontend_path, "static")
        if os.path.exists(static_path):
            app.mount("/static", StaticFiles(directory=static_path), name="static")
            logger.debug(f"Static assets mounted from: {static_path}")
    else:
        logger.warning(f"Frontend directory not found: {frontend_path}")
    
    logger.info("GPT Researcher API ready - local mode (no database persistence)")
    yield
    # 关闭阶段。
    logger.info("Research API shutting down")

# 初始化应用。
app = FastAPI(lifespan=lifespan)

# 配置 CORS 允许的来源。
allowed_origins_env = os.getenv("CORS_ALLOW_ORIGINS")
ALLOWED_ORIGINS = (
    [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
    if allowed_origins_env
    else [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://app.gptr.dev",
    ]
)

# 使用标准 JSON 响应，无需 MongoDB 自定义编码。

# 添加 CORS 中间件。
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 使用默认 JSON 响应类。

# 挂载前端静态文件，并取得前端目录的绝对路径。
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))

# 挂载静态目录。
app.mount("/static", StaticFiles(directory=os.path.join(frontend_dir, "static")), name="static")
app.mount("/site", StaticFiles(directory=frontend_dir), name="site")

# WebSocket 管理器。
manager = WebSocketManager()

report_store = ReportStore(Path(os.getenv('REPORT_STORE_PATH', os.path.join('data', 'reports.json'))))

# 常量。
DOC_PATH = os.getenv("DOC_PATH", "./my-docs")

# 启动事件。


# 生命周期事件已由上面的 lifespan 上下文管理器处理。


# 路由。
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend HTML page."""
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
    index_path = os.path.join(frontend_dir, "index.html")
    
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Frontend index.html not found")
    
    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    return HTMLResponse(content=content)


@app.get("/.well-known/agent-discovery.json")
async def agent_discovery(request: Request):
    """Advertise GPT Researcher services via the Agent Discovery Protocol."""
    origin = str(request.base_url).rstrip("/")
    domain = request.url.hostname or request.headers.get("host", "")
    contact = os.getenv("AGENT_DISCOVERY_CONTACT")

    document = build_agent_discovery_document(origin=origin, domain=domain, contact=contact)
    response = JSONResponse(content=document)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

@app.get("/report/{research_id}")
async def read_report(request: Request, research_id: str):
    """按研究任务 ID 返回已生成的 DOCX 报告。"""
    docx_path = os.path.join('outputs', f"{research_id}.docx")
    if not os.path.exists(docx_path):
        return {"message": "Report not found."}
    return FileResponse(docx_path)


# 简化的 API 路由，不使用数据库持久化。
@app.get("/api/reports")
async def get_all_reports(report_ids: str = None):
    """列出全部报告，或按逗号分隔的 ID 列表筛选报告。"""
    report_ids_list = report_ids.split(",") if report_ids else None
    reports = await report_store.list_reports(report_ids_list)
    return {"reports": reports}


@app.get("/api/reports/{research_id}")
async def get_report_by_id(research_id: str):
    """读取单份持久化报告；不存在时返回 404。"""
    report = await report_store.get_report(research_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"report": report}


@app.post("/api/reports")
async def create_or_update_report(request: Request):
    """根据请求体中的 ID 新建或覆盖报告，并保持时间戳单调递增。"""
    try:
        data = await request.json()
        research_id = data.get("id", "temp_id")

        now_ms = int(time.time() * 1000)
        existing = await report_store.get_report(research_id)
        incoming_timestamp = data.get("timestamp")
        timestamp = incoming_timestamp if isinstance(incoming_timestamp, int) else now_ms
        if existing and isinstance(existing.get("timestamp"), int):
            timestamp = max(timestamp, existing["timestamp"])

        report = {
            "id": research_id,
            "question": data.get("question"),
            "answer": data.get("answer"),
            "orderedData": data.get("orderedData") or [],
            "chatMessages": data.get("chatMessages") or [],
            "timestamp": timestamp,
        }

        await report_store.upsert_report(research_id, report)
        return {"success": True, "id": research_id}
    except Exception as e:
        logger.error(f"Error processing report creation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/reports/{research_id}")
async def update_report(research_id: str, request: Request):
    """合并更新指定报告，只用非空请求字段覆盖现有数据。"""
    existing = await report_store.get_report(research_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Report not found")

    data = await request.json()
    now_ms = int(time.time() * 1000)

    updated = {
        **existing,
        **{k: v for k, v in data.items() if v is not None},
        "id": research_id,
        "timestamp": now_ms,
    }

    await report_store.upsert_report(research_id, updated)
    return {"success": True, "id": research_id}


@app.delete("/api/reports/{research_id}")
async def delete_report(research_id: str):
    """删除指定报告；目标不存在时返回 404。"""
    existed = await report_store.delete_report(research_id)
    if not existed:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"success": True}


@app.get("/api/reports/{research_id}/chat")
async def get_report_chat(research_id: str):
    """返回指定报告已经持久化的对话历史。"""
    report = await report_store.get_report(research_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"chatMessages": report.get("chatMessages") or []}


@app.post("/api/reports/{research_id}/chat")
async def research_report_chat(research_id: str, request: Request):
    """Chat against a stored report: LLM reply + persist user/assistant messages."""
    report = await report_store.get_report(research_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")

    try:
        data = await request.json()
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}

    # 同时支持单条消息对象和 {messages, report, message} 格式。
    incoming_messages = data.get("messages")
    if not isinstance(incoming_messages, list):
        if any(key in data for key in ("role", "content")):
            incoming_messages = [data]
        elif isinstance(data.get("message"), dict):
            incoming_messages = [data["message"]]
        else:
            incoming_messages = []

    report_text = data.get("report") or report.get("answer") or ""
    history = report.get("chatMessages") or []
    if not isinstance(history, list):
        history = []

    messages_for_llm = list(history)
    for msg in incoming_messages:
        if isinstance(msg, dict):
            messages_for_llm.append(msg)

    try:
        chat_agent = ChatAgentWithMemory(
            report=report_text if isinstance(report_text, str) else str(report_text),
            config_path="default",
            headers=None,
        )
        response_content, tool_calls_metadata = await chat_agent.chat(messages_for_llm, None)
    except Exception as e:
        logger.error(f"Error in research report chat: {str(e)}", exc_info=True)
        return {"error": str(e)}

    now_ms = int(time.time() * 1000)
    response_message = {
        "role": "assistant",
        "content": response_content,
        "timestamp": now_ms,
        "metadata": {"tool_calls": tool_calls_metadata} if tool_calls_metadata else None,
    }

    new_chat = list(history)
    for msg in incoming_messages:
        if isinstance(msg, dict):
            new_chat.append(msg)
    new_chat.append(response_message)

    updated = {
        **report,
        "chatMessages": new_chat,
        "timestamp": now_ms,
    }
    await report_store.upsert_report(research_id, updated)
    return {"success": True, "id": research_id, "response": response_message}




async def write_report(research_request: ResearchRequest, research_id: str = None):
    """执行研究任务、导出 DOCX/PDF，并组装供 HTTP 接口返回的结果。"""
    # run_agent 是 HTTP 层进入核心 GPTResearcher 编排流程的边界。
    report_information = await run_agent(
        task=research_request.task,
        report_type=research_request.report_type,
        report_source=research_request.report_source,
        source_urls=[],
        document_urls=[],
        tone=Tone[research_request.tone],
        websocket=None,
        stream_output=None,
        headers=research_request.headers,
        query_domains=[],
        config_path="",
        return_researcher=True
    )

    docx_path = await write_md_to_word(report_information[0], research_id)
    pdf_path = await write_md_to_pdf(report_information[0], research_id)
    if research_request.report_type != "multi_agents":
        report, researcher = report_information
        response = {
            "research_id": research_id,
            "research_information": {
                "source_urls": researcher.get_source_urls(),
                "research_costs": researcher.get_costs(),
                "visited_urls": list(researcher.visited_urls),
                "research_images": researcher.get_research_images(),
                # "research_sources": researcher.get_research_sources(),  # 原始来源内容可能很大。
            },
            "report": report,
            "docx_path": docx_path,
            "pdf_path": pdf_path
        }
    else:
        response = { "research_id": research_id, "report": "", "docx_path": docx_path, "pdf_path": pdf_path }

    return response

@app.post("/report/")
async def generate_report(research_request: ResearchRequest, background_tasks: BackgroundTasks):
    """创建研究任务，并按请求选择后台执行或等待完整结果。"""
    research_id = sanitize_filename(f"task_{int(time.time())}_{research_request.task}")

    # 后台模式立即返回任务 ID；同步模式则等待研究和文件导出全部完成。
    if research_request.generate_in_background:
        background_tasks.add_task(write_report, research_request=research_request, research_id=research_id)
        return {"message": "Your report is being generated in the background. Please check back later.",
                "research_id": research_id}
    else:
        response = await write_report(research_request, research_id)
        return response


@app.get("/files/")
async def list_files():
    """列出本地文档知识库目录中的文件。"""
    if not os.path.exists(DOC_PATH):
        os.makedirs(DOC_PATH, exist_ok=True)
    files = os.listdir(DOC_PATH)
    print(f"Files in {DOC_PATH}: {files}")
    return {"files": files}


@app.post("/api/multi_agents")
async def run_multi_agents():
    """通过当前 WebSocket 连接启动多 Agent 研究流程。"""
    return await execute_multi_agents(manager)


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """保存上传文件并触发本地文档索引加载。"""
    return await handle_file_upload(file, DOC_PATH)


@app.delete("/files/{filename}")
async def delete_file(filename: str):
    """从本地文档知识库目录删除指定文件。"""
    return await handle_file_deletion(filename, DOC_PATH)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """维护研究任务的 WebSocket 生命周期并分发客户端命令。"""
    await manager.connect(websocket)
    try:
        await handle_websocket_communication(websocket, manager)
    except WebSocketDisconnect as e:
        logger.info(f"WebSocket disconnected with code {e.code} and reason: '{e.reason}'")
    except Exception as e:
        logger.error(f"Unexpected WebSocket error: {str(e)}")
    finally:
        # 无论循环如何结束，都释放连接队列和发送任务。
        await manager.disconnect(websocket)

@app.post("/api/chat")
async def chat(chat_request: ChatRequest):
    """Process a chat request with a report and message history.

    Args:
        chat_request: ChatRequest object containing report text and message history

    Returns:
        JSON response with the assistant's message and any tool usage metadata
    """
    try:
        logger.info(f"Received chat request with {len(chat_request.messages)} messages")

        # 使用报告内容创建聊天 Agent。
        chat_agent = ChatAgentWithMemory(
            report=chat_request.report,
            config_path="default",
            headers=None
        )

        # 处理对话并取得响应及工具调用元数据。
        response_content, tool_calls_metadata = await chat_agent.chat(chat_request.messages, None)
        logger.info(f"response_content: {response_content}")
        logger.info(f"Got chat response of length: {len(response_content) if response_content else 0}")
        
        if tool_calls_metadata:
            logger.info(f"Tool calls used: {json.dumps(tool_calls_metadata)}")

        # 将响应整理为包含角色、内容、时间戳和元数据的消息对象。
        response_message = {
            "role": "assistant",
            "content": response_content,
            "timestamp": int(time.time() * 1000),  # Current time in milliseconds
            "metadata": {
                "tool_calls": tool_calls_metadata
            } if tool_calls_metadata else None
        }

        logger.info(f"Returning formatted response: {json.dumps(response_message)[:100]}...")
        return {"response": response_message}
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        return {"error": str(e)}

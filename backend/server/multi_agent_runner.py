import os
import sys
from typing import Any, Awaitable, Callable

RunResearchTask = Callable[..., Awaitable[Any]]


def _ensure_repo_root_on_path() -> None:
    """Ensure top-level repo root is importable for multi-agent modules."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _resolve_run_research_task() -> RunResearchTask:
    """解析可用的多 Agent 入口，优先使用主实现并兼容 AG2 实现。"""
    _ensure_repo_root_on_path()

    try:
        from multi_agents.main import run_research_task
        return run_research_task
    except Exception:
        try:
            from multi_agents.ag2.main import run_research_task
            return run_research_task
        except Exception as ag2_error:
            raise ImportError(
                "Could not import run_research_task from multi_agents or multi_agents/ag2"
            ) from ag2_error


async def run_multi_agent_task(*args, **kwargs) -> Any:
    """延迟解析并调用多 Agent 研究入口。"""
    run_research_task = _resolve_run_research_task()
    return await run_research_task(*args, **kwargs)

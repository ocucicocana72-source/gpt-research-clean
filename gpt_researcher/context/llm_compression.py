"""LLM compression used when retrieved context exceeds the token budget."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from langchain_core.documents import Document

from ..prompts import PromptFamily
from ..utils.llm import create_chat_completion


def estimate_tokens(text: str) -> int:
    """Conservative token estimate that does not require a tokenizer package."""
    return max(1, (len(text) + 3) // 4)


def _format_sources(documents: Iterable[Document | dict]) -> str:
    blocks = []
    for index, document in enumerate(documents, 1):
        if isinstance(document, Document):
            content = document.page_content
            metadata = document.metadata or {}
            source = metadata.get("source") or metadata.get("url") or "unknown"
        else:
            content = document.get("raw_content") or document.get("page_content") or ""
            source = document.get("source") or document.get("url") or "unknown"
        blocks.append(f"SOURCE {index} ({source})\n{content}")
    return "\n\n".join(blocks)


async def compress_documents(
    query: str,
    documents: list[Document | dict],
    cfg: Any,
    *,
    prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
    cost_callback: Callable[[float], None] | None = None,
) -> str:
    """Summarize retrieved documents while retaining source markers."""
    source_text = _format_sources(documents)
    prompt = prompt_family.context_compression_prompt(query, source_text)
    return await create_chat_completion(
        model=cfg.smart_llm_model,
        messages=[{"role": "user", "content": prompt}],
        llm_provider=cfg.smart_llm_provider,
        max_tokens=getattr(cfg, "context_summary_token_limit", cfg.smart_token_limit),
        llm_kwargs=cfg.llm_kwargs,
        cost_callback=cost_callback,
    )

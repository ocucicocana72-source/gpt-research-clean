"""GPT Researcher 的 LLM 工具函数。

本模块通过统一接口创建不同厂商的 LLM 实例，并提供聊天补全与子主题生成能力。
"""
from __future__ import annotations

import logging
import os
from typing import Any
import asyncio

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from gpt_researcher.llm_provider.generic.base import (
    NO_SUPPORT_TEMPERATURE_MODELS,
    SUPPORT_REASONING_EFFORT_MODELS,
    ReasoningEfforts,
)

from ..prompts import PromptFamily
from .costs import calculate_llm_cost
from .validators import Subtopics


def get_llm(llm_provider: str, **kwargs):
    """根据厂商名称创建 LLM 提供商实例。

    参数：
        llm_provider: LLM 厂商名称，例如 ``openai``、``anthropic``。
        **kwargs: 传给具体厂商实现的额外关键字参数。

    返回：
        已按指定厂商完成配置的 ``GenericLLMProvider`` 实例。
    """
    # 延迟导入可避免模块加载阶段过早初始化各厂商的依赖。
    from gpt_researcher.llm_provider import GenericLLMProvider
    return GenericLLMProvider.from_provider(llm_provider, **kwargs)


async def create_chat_completion(
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = 0.4,
        max_tokens: int | None = 4000,
        llm_provider: str | None = None,
        stream: bool = False,
        websocket: Any | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        cost_callback: callable = None,
        reasoning_effort: str | None = ReasoningEfforts.Medium.value,
        **kwargs
) -> str:
    """调用指定厂商的模型生成聊天补全结果。

    参数：
        messages: 发给模型的消息列表，每项包含角色与消息内容。
        model: 要调用的模型名称。
        temperature: 生成温度，默认值为 0.4。
        max_tokens: 最大输出 token 数，默认值为 4000。
        llm_provider: LLM 厂商名称。
        stream: 是否以流式方式返回响应。
        websocket: 当前请求用于传输流式内容的 WebSocket。
        llm_kwargs: 直接传给 LLM 实例的额外参数。
        cost_callback: 用于累计本次调用成本的回调函数。
        reasoning_effort: 推理模型的推理强度，默认值为 ``medium``。
        **kwargs: 传给厂商响应方法的额外参数。

    返回：
        模型生成的文本响应。
    """
    # 模型是创建厂商实例时必需的参数，因此先在统一入口处校验。
    if model is None:
        raise ValueError("Model cannot be None")
    # 防止环境变量拼写错误等情况产生离谱的 token 上限；各模型的真实限制仍由上游厂商校验。
    if max_tokens is not None and max_tokens > 200_000:
        raise ValueError(
            f"max_tokens={max_tokens} exceeds the largest output limit of "
            "any currently available model (128k as of late 2025). "
            "Check your FAST_TOKEN_LIMIT / SMART_TOKEN_LIMIT / "
            "STRATEGIC_TOKEN_LIMIT env vars for typos."
        )

    # 先组装通用参数，再根据模型能力有选择地加入推理强度或温度。
    provider_kwargs = {'model': model}

    if model in SUPPORT_REASONING_EFFORT_MODELS:
        provider_kwargs['reasoning_effort'] = reasoning_effort

    if model not in NO_SUPPORT_TEMPERATURE_MODELS:
        provider_kwargs['temperature'] = temperature
    else:
        # 这类模型固定使用默认温度，但输出限制仍然生效；对推理模型而言，该限制也包含推理 token。
        provider_kwargs['temperature'] = None
    provider_kwargs['max_tokens'] = max_tokens

    # 调用方或环境变量中的配置优先级最高，所以最后合并；这样不会覆盖通过 ``llm_kwargs`` 或
    # ``LLM_KWARGS`` 明确传入的温度等参数。
    if llm_kwargs:
        provider_kwargs.update(llm_kwargs)
    elif os.environ.get("LLM_KWARGS"):
        import json
        try:
            provider_kwargs.update(json.loads(os.environ["LLM_KWARGS"]))
        except json.JSONDecodeError:
            pass

    if llm_provider == "openai":
        base_url = os.environ.get("OPENAI_BASE_URL", None)
        if base_url:
            provider_kwargs['openai_api_base'] = base_url

    provider = get_llm(llm_provider, **provider_kwargs)
    response = ""
    # WebSocket 流式响应一旦开始便不宜重试，否则可能向客户端重复发送部分内容。
    max_attempts = 1 if (stream and websocket is not None) else 10
    last_exception: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = await provider.get_chat_response(
                messages, stream, websocket, **kwargs
            )
        except Exception as exc:
            last_exception = exc
            logging.getLogger(__name__).warning(
                f"LLM request failed (attempt {attempt}/{max_attempts}): {exc}"
            )
            if attempt < max_attempts:
                # 指数退避时间依次为 1、2、4、8 秒，之后保持 8 秒上限。
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
                continue
            break

        if not response:
            last_exception = RuntimeError("Empty response from LLM provider")
            logging.getLogger(__name__).warning(
                f"LLM returned empty response (attempt {attempt}/{max_attempts})"
            )
            if attempt < max_attempts:
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
                continue
            break

        if cost_callback:
            # 只有成功取得非空响应时才计算费用，并通过回调交给上层累计。
            llm_costs = calculate_llm_cost(
                llm_provider=llm_provider,
                model=model,
                input_content=str(messages),
                output_content=response,
                response_metadata=provider.last_response_metadata,
                usage_metadata=provider.last_usage_metadata,
                request_options=provider_kwargs,
            )
            cost_callback(llm_costs)

        return response

    logging.error(f"Failed to get response from {llm_provider} API")
    raise RuntimeError(f"Failed to get response from {llm_provider} API") from last_exception


async def construct_subtopics(
    task: str,
    data: str,
    config,
    subtopics: list = [],
    prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
    **kwargs
) -> list:
    """根据研究任务和已有资料生成结构化子主题。

    参数：
        task: 主要研究任务或主题。
        data: 用作上下文的补充资料。
        config: 模型、token 上限和子主题数量等配置。
        subtopics: 已有子主题；生成失败时也会作为回退结果返回。
        prompt_family: 负责生成子主题提示词的提示词系列。
        **kwargs: 传给 LangChain 链异步调用的额外参数。

    返回：
        解析后的 ``Subtopics`` 结果；发生异常时返回传入的 ``subtopics``。
    """
    try:
        # 解析器既提供输出格式说明，也负责把模型响应校验为 Subtopics 结构。
        parser = PydanticOutputParser(pydantic_object=Subtopics)

        prompt = PromptTemplate(
            template=prompt_family.generate_subtopics_prompt(),
            input_variables=["task", "data", "subtopics", "max_subtopics"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()},
        )

        provider_kwargs = {'model': config.smart_llm_model}

        if config.llm_kwargs:
            provider_kwargs.update(config.llm_kwargs)

        if config.smart_llm_model in SUPPORT_REASONING_EFFORT_MODELS:
            provider_kwargs['reasoning_effort'] = ReasoningEfforts.High.value
        else:
            provider_kwargs['temperature'] = config.temperature
        provider_kwargs['max_tokens'] = config.smart_token_limit

        provider = get_llm(config.smart_llm_provider, **provider_kwargs)

        model = provider.llm

        # 数据依次经过提示词模板、LLM 和 Pydantic 解析器。
        chain = prompt | model | parser

        output = await chain.ainvoke({
            "task": task,
            "data": data,
            "subtopics": subtopics,
            "max_subtopics": config.max_subtopics
        }, **kwargs)

        return output

    except Exception as e:
        # 子主题生成失败不阻断报告流程，记录完整堆栈后退回已有子主题。
        logging.getLogger(__name__).error(
            "Exception in parsing subtopics: %s", e, exc_info=True
        )
        return subtopics

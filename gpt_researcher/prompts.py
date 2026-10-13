import warnings
from datetime import date, datetime, timezone

from langchain_core.documents import Document

from .config import Config
from .utils.enum import ReportSource, ReportType, Tone
from .utils.enum import PromptFamily as PromptFamilyEnum
from typing import Callable, List, Dict, Any


## Prompt 家族

class PromptFamily:
    """用于格式化 prompt 的通用类。

    该类可以由针对特定模型的派生类覆盖。方法分为两组：

    1. Prompt 生成器：遵循标准格式，并与 ReportType 枚举对应，
       应通过 get_prompt_by_report_type 访问。

    2. Prompt 方法：面向特定场景，不采用统一签名，由 Agent 代码直接调用。

    所有派生类都必须保留相同的方法名集合，但可以覆盖其中的具体方法。
    """

    def __init__(self, config: Config):
        """使用配置实例初始化，可供派生类根据模型或服务提供商选择 prompt。"""
        self.cfg = config

    # MCP 专用 prompt
    @staticmethod
    def generate_mcp_tool_selection_prompt(query: str, tools_info: List[Dict], max_tools: int = 3) -> str:
        """生成用于让 LLM 选择 MCP 工具的 prompt。
        
        参数：
            query：研究问题。
            tools_info：包含元数据的可用工具列表。
            max_tools：最多选择的工具数量。
            
        返回值：
            str：工具选择 prompt。
        """
        import json
        
        return f"""你是一名研究助手，需要为研究问题选择最相关的工具。

研究问题：“{query}”

可用工具：
{json.dumps(tools_info, indent=2)}

任务：分析这些工具，严格选择 {max_tools} 个最适合研究该问题的工具。

选择标准：
- 选择能够提供与问题相关的信息、数据或洞察的工具。
- 优先选择能够搜索、检索或访问相关内容的工具。
- 考虑工具之间的互补性，例如使用不同的数据源。
- 排除与研究主题明显无关的工具。

请严格按照以下格式返回 JSON 对象：
{{
  "selected_tools": [
    {{
      "index": 0,
      "name": "tool_name",
      "relevance_score": 9,
      "reason": "详细说明该工具为何相关"
    }}
  ],
  "selection_reasoning": "对整体选择策略的说明"
}}

严格选择 {max_tools} 个工具，并按照与研究问题的相关性排序。
"""

    @staticmethod
    def generate_mcp_research_prompt(query: str, selected_tools: List) -> str:
        """生成使用已选 MCP 工具执行研究的 prompt。
        
        参数：
            query：研究问题。
            selected_tools：已选 MCP 工具列表。
            
        返回值：
            str：研究执行 prompt。
        """
        # 兼容字符串工具，以及带有 .name 属性的工具对象。
        tool_names = []
        for tool in selected_tools:
            if hasattr(tool, 'name'):
                tool_names.append(tool.name)
            else:
                tool_names.append(str(tool))
        
        return f"""你是一名可以使用专业工具的研究助手。你的任务是研究下面的问题，并提供全面、准确的信息。

研究问题：“{query}”

执行要求：
1. 使用可用工具收集与问题相关的信息。
2. 如有需要，调用多个工具以获得全面覆盖。
3. 如果工具调用失败或返回空结果，尝试其他方法。
4. 尽可能综合多个来源的信息。
5. 聚焦于直接回答问题的事实性和相关性信息。

可用工具：{tool_names}

请进行充分研究并给出结果，合理使用工具以收集最相关、最全面的信息。"""

    @staticmethod
    def knowledge_base_routing_prompt(query: str) -> str:
        """判断用户问题是否需要检索本地知识库或网页资料。"""
        return f"""你是一个检索路由器。判断下面的问题是否必须查阅外部资料才能可靠回答。

用户问题：
{query}

需要知识库的情况包括：询问具体技术事实、版本差异、错误排查、代码/接口用法、多个资料的比较，或问题可能超出模型稳定知识。
不需要知识库的情况包括：纯粹的写作改写、翻译、格式整理、简单数学运算、闲聊，以及用户明确要求直接给出通用解释且不依赖具体版本或外部事实。

只返回 JSON，不要 Markdown：
{{"needs_knowledge_base": true, "reason": "一句话说明理由"}}
"""

    @staticmethod
    def context_compression_prompt(query: str, context: str) -> str:
        """生成超预算上下文的压缩提示，并保留来源编号。"""
        return f"""你是检索上下文压缩器。请围绕问题提炼支持答案所必需的事实，删除重复、无关和营销性内容。

问题：{query}

上下文（每段以 SOURCE 编号开头）：
{context}

要求：
- 只保留能直接帮助回答问题的事实、定义、条件、版本和步骤。
- 保留每条事实对应的 SOURCE 编号；不能编造来源或补充上下文之外的知识。
- 如果来源互相矛盾，分别保留并标记冲突。
- 输出简洁的 Markdown，使用 [SOURCE n] 标记。
"""

    # 图像生成 prompt
    @staticmethod
    def generate_image_analysis_prompt(
        query: str,
        sections: List[Dict[str, Any]],
        max_images: int = 3,
    ) -> str:
        """生成用于分析报告哪些章节需要配图的 prompt。
        
        参数：
            query：研究问题。
            sections：包含标题和内容的报告章节列表。
            max_images：最多建议的图片数量。
            
        返回值：
            str：分析 prompt。
        """
        sections_text = "\n\n".join([
            f"### 章节 {i+1}： {s['header']}\n{s['content'][:500]}..."
            for i, s in enumerate(sections)
        ])
        
        return f"""分析以下研究报告章节，找出最需要视觉插图或示意图的 {max_images} 个章节。

研究主题：{query}

报告章节：
{sections_text}

对每个推荐章节提供：
1. 章节编号（从 1 开始）。
2. 能生成信息丰富插图的具体、详细图像 prompt。
3. 简要说明该章节为何适合可视化。

重要指导原则：
- 选择视觉呈现确实有助于理解的章节。
- 聚焦本身适合可视化的概念、流程、比较、数据流或统计信息。
- 避免选择纯文本分析、引言或结论章节。
- 图像 prompt 应足够具体，以生成相关且专业的插图。
- 图片应具有信息性和教育性，而非仅用于装饰。
- 可考虑示意图、流程图、对比图或概念图。

请以 JSON 格式回答：
{{
    "suggestions": [
        {{
            "section_number": 1,
            "section_header": "章节标题",
            "image_prompt": "用于生成信息丰富插图的详细 prompt……",
            "image_type": "diagram|flowchart|comparison|concept|data_visualization",
            "reason": "该章节适合可视化的原因"
        }}
    ]
}}

只返回 JSON，不要添加任何其他文本。"""

    @staticmethod
    def generate_image_prompt_enhancement(
        base_prompt: str,
        section_content: str,
        research_topic: str,
    ) -> str:
        """结合上下文增强图像 prompt，以获得更好的生成效果。
        
        参数：
            base_prompt：基础图像生成 prompt。
            section_content：报告章节内容。
            research_topic：研究主题。
            
        返回值：
            str：增强后的图像 prompt。
        """
        return f"""为研究报告创建一张专业且信息丰富的插图。

研究主题：{research_topic}

图像描述：{base_prompt}

报告上下文：
{section_content[:800]}

风格要求：
- 采用适合学术或商业报告的专业、整洁设计。
- 使用清晰、易于理解的视觉元素。
- 采用现代、简约的美学风格。
- 使用专业配色（蓝色、青绿色、灰色）。
- 避免在图片中放置过多文字。
- 使用高对比度以保证可读性。
- 如果展示数据或比较，请使用清晰的标签和图例。
- 同时适合数字阅读和打印。"""

    @staticmethod
    def generate_search_queries_prompt(
        question: str,
        parent_query: str,
        report_type: str,
        max_iterations: int = 3,
        context: List[Dict[str, Any]] = [],
    ):
        """生成针对给定问题的搜索查询 prompt。

        参数：问题、父问题、报告类型、最多生成的查询数量和实时网页上下文。
        返回值：搜索查询 prompt。
        """

        if (
            report_type == ReportType.DetailedReport.value
            or report_type == ReportType.SubtopicReport.value
        ):
            task = f"{parent_query} - {question}"
        else:
            task = question

        context_prompt = f"""
你是一名经验丰富的研究助手，需要为以下任务生成搜索查询，以查找相关信息：“{task}”。
上下文：{context}

请利用这些上下文来改进和细化搜索查询。上下文包含实时网页信息，可以帮助你生成更具体、更相关的查询；请考虑其中提到的时事、最新进展或具体细节。
""" if context else ""

        dynamic_example = ", ".join([f'"query {i+1}"' for i in range(max_iterations)])

        return f"""为以下任务生成恰好 {max_iterations} 条互补的搜索查询：“{task}”

每条查询都必须是自然语言短语。不要使用 site:、filetype:、inurl:、intitle:、OR、AND 或 NOT 等搜索运算符语法，因为这些运算符并非所有搜索后端都支持，可能导致许多后端返回空结果。

如有需要，假定当前日期为 {datetime.now(timezone.utc).strftime('%B %d, %Y')}。

{context_prompt}
必须按照以下格式返回字符串列表：[{dynamic_example}]。
必须返回恰好 {max_iterations} 条；如果问题简单，也要从不同角度拆分，不能返回 1 条或超过 {max_iterations} 条。
回答中只能包含该列表。
"""

    @staticmethod
    def generate_report_prompt(
        question: str,
        context,
        report_source: str,
        report_format="apa",
        total_words=1000,
        tone=None,
        language="english",
    ):
        """生成针对给定问题和研究摘要的报告 prompt。"""

        reference_prompt = ""
        if report_source == ReportSource.Web.value:
            reference_prompt = f"""
你必须在报告末尾将所有使用过的来源 URL 列为参考文献，并确保每个来源只出现一次，不要重复添加。
每个 URL 都应使用超链接：[url website](url)
此外，在报告正文引用来源的位置，也必须加入相关 URL 的超链接：

例如：作者。（年月日）。网页标题。网站名称。[url website](url)
"""
        else:
            reference_prompt = f"""
你必须在报告末尾将所有使用过的来源文档名称列为参考文献，并确保每个来源只出现一次，不要重复添加。"
"""

        tone_prompt = f"请使用{tone.value}语气撰写报告。" if tone else ""

        return f"""
信息：“{context}”
---
根据以上信息，对以下问题或任务“{question}”撰写详细报告。
报告应聚焦问题的答案，结构清晰、信息丰富、深入全面；如有可用事实和数据，应包含这些内容，且至少包含 {total_words} 个词。
请尽可能利用所提供的全部相关且必要的信息，将报告写得充分完整。

请遵循以下报告写作要求：
- 你必须根据给定信息形成具体、有效的独立观点，不要回避问题或给出空泛、无意义的结论。
- 必须使用 Markdown 语法和 {report_format} 格式撰写报告。
- 使用清晰的 Markdown 标题组织报告：主标题使用 #，主要章节使用 ##，小节使用 ###。
- 展示结构化数据或比较时，使用 Markdown 表格提高可读性。
- 必须优先考虑来源的相关性、可靠性和重要性，选择可信来源而非不可靠来源。
- 在来源可信的前提下，优先使用较新的文章。
- 不得添加目录，但必须使用规范的 Markdown 标题（# ## ###）清晰组织报告。
- 使用 {report_format} 格式的文内引用，并在引用句或段落末尾添加 Markdown 超链接，例如：([文内引用](url))。
- 每个重要论断、数据或引文都必须标注其来源的文内引用。不要引用所提供信息中不存在的来源。
- 不要忘记在报告末尾按照 {report_format} 格式添加参考文献列表。
- {reference_prompt}
- {tone_prompt}
必须使用以下语言撰写报告：{language}。
假定当前日期为 {date.today()}。
"""

    @staticmethod
    def curate_sources(query, sources, max_results=10):
        return f"""你的目标是评估并整理为研究任务“{query}”提供的抓取内容，同时优先保留相关且高质量的信息，尤其是包含统计数据、数字或具体数据的来源。

最终整理后的列表将作为研究报告的上下文，因此请优先考虑：
- 尽可能保留原始信息，尤其重视包含定量数据或独特洞察的来源。
- 纳入广泛的观点和洞察。
- 只过滤明显无关或不可用的内容。

评估指导原则：
1. 根据以下标准评估每个来源：
   - 相关性：纳入与研究问题直接或部分相关的来源，宁可保留，也不要过度过滤。
   - 可信度：优先选择权威来源，但除非明显不可信，否则保留其他来源。
   - 时效性：优先选择较新的信息，除非旧数据具有必要性或价值。
   - 客观性：如果带有偏见的来源能提供独特或互补视角，也应保留。
   - 定量价值：优先选择包含统计数据、数字或其他具体数据的来源。
2. 来源选择：
   - 尽可能纳入相关来源，最多 {max_results} 个，兼顾覆盖范围和多样性。
   - 优先选择包含统计数据、数值数据或可验证事实的来源。
   - 如果重叠内容能够增加深度，尤其是涉及数据时，可以保留。
   - 只有在来源完全无关、严重过时或因内容质量差而不可用时才排除。
3. 内容保留：
   - 不要改写、总结或压缩任何来源内容。
   - 保留所有可用信息，只清理明显的垃圾内容或格式问题。
   - 如果边缘相关或不完整的来源包含有价值的数据或洞察，也应保留。

待评估的来源列表：
{sources}

必须严格按照原始来源列表的 JSON 格式返回结果。
回答不得包含任何 Markdown 格式或其他文本（例如 ```json），只能返回 JSON 列表！
"""

    @staticmethod
    def generate_resource_report_prompt(
        question, context, report_source: str, report_format="apa", tone=None, total_words=1000, language="english"
    ):
        """生成针对给定问题和研究摘要的资源报告 prompt。"""

        reference_prompt = ""
        if report_source == ReportSource.Web.value:
            reference_prompt = f"""
            必须包含所有相关来源 URL。
            每个 URL 都应使用超链接：[url website](url)
            """
        else:
            reference_prompt = f"""
            必须在报告末尾将所有使用过的来源文档名称列为参考文献，并确保每个来源只出现一次，不要重复添加。"
        """

        return (
            f'"""{context}"""\n\n根据以上信息，为以下问题或主题“{question}”生成资源推荐报告。报告应详细分析每个推荐资源，'
            "说明每个来源如何帮助回答研究问题。\n"
            "重点关注每个来源的相关性、可靠性和重要性。\n"
            "确保报告结构清晰、信息丰富、深入，并遵循 Markdown 语法。\n"
            "适当使用 Markdown 表格和其他格式，以清晰地组织和呈现信息。\n"
            "尽可能包含相关事实、数据和数字。\n"
            f"报告至少包含 {total_words} 个词。\n"
            f"必须使用以下语言撰写报告：{language}。\n"
            "必须包含所有相关来源 URL。"
            "每个 URL 都应使用超链接：[url website](url)"
            f"{reference_prompt}"
        )

    @staticmethod
    def generate_custom_report_prompt(
        query_prompt, context, report_source: str, report_format="apa", tone=None, total_words=1000, language: str = "english"
    ):
        return f'"{context}"\n\n{query_prompt}'

    @staticmethod
    def generate_outline_report_prompt(
        question, context, report_source: str, report_format="apa", tone=None,  total_words=1000, language: str = "english"
    ):
        """生成针对给定问题和研究摘要的报告大纲 prompt。"""

        return (
            f'"""{context}""" 根据以上信息，使用 Markdown 语法为以下问题或主题“{question}”生成研究报告大纲。'
            "大纲应为研究报告提供结构清晰的框架，包括需要覆盖的主要章节、小节和关键点。"
            f"研究报告应详细、信息丰富、深入，且至少包含 {total_words} 个词。"
            "使用合适的 Markdown 语法格式化大纲，确保可读性。"
            "适当使用 Markdown 表格和其他格式，以提升信息呈现效果。"
        )

    @staticmethod
    def generate_deep_research_prompt(
        question: str,
        context: str,
        report_source: str,
        report_format="apa",
        tone=None,
        total_words=2000,
        language: str = "english"
    ):
        """生成用于处理分层研究结果的深度研究报告 prompt。"""
        reference_prompt = ""
        if report_source == ReportSource.Web.value:
            reference_prompt = f"""
你必须在报告末尾将所有使用过的来源 URL 列为参考文献，并确保每个来源只出现一次，不要重复添加。
每个 URL 都应使用超链接：[url website](url)
此外，在报告正文引用来源的位置，也必须加入相关 URL 的超链接：

例如：作者。（年月日）。网页标题。网站名称。[url website](url)
"""
        else:
            reference_prompt = f"""
你必须在报告末尾将所有使用过的来源文档名称列为参考文献，并确保每个来源只出现一次，不要重复添加。"
"""

        tone_prompt = f"请使用{tone.value}语气撰写报告。" if tone else ""

        return f"""
以下是按层级整理的研究信息和引用：

"{context}"

请撰写一份全面的研究报告，回答以下问题：“{question}”

报告应：
1. 综合多个研究深度层级的信息。
2. 整合不同研究分支的发现。
3. 构建从基础认识到高级洞察的连贯叙事。
4. 在全文正确引用来源。
5. 结构清晰，包含明确的章节和小节。
6. 至少包含 {total_words} 个词。
7. 使用 Markdown 语法并遵循 {report_format} 格式。
8. 展示比较数据、统计数据或结构化信息时，使用 Markdown 表格、列表及其他格式。

其他要求：
- 优先呈现来自更深研究层级的洞察。
- 突出不同研究分支之间的联系。
- 包含相关统计数据、具体数据和实例。
- 必须根据给定信息形成具体、有效的独立观点，不要回避问题或给出空泛、无意义的结论。
- 必须优先考虑来源的相关性、可靠性和重要性，选择可信来源而非不可靠来源。
- 在来源可信的前提下，优先使用较新的文章。
- 使用 {report_format} 格式的文内引用，并在引用句或段落末尾添加 Markdown 超链接，例如：([文内引用](url))。
- {tone_prompt}
- 使用 {language} 撰写。

{reference_prompt}

请撰写一份充分研究、内容严谨的报告，将所有收集到的信息综合为一个连贯整体。
假定当前日期为 {datetime.now(timezone.utc).strftime('%B %d, %Y')}。
"""

    @staticmethod
    def auto_agent_instructions():
        return """
这项任务需要研究给定主题，不论主题的复杂程度或是否存在确定答案。研究由特定服务器执行，服务器由类型和角色定义，不同服务器需要不同的指令。
Agent
服务器由主题所属领域以及可用于研究该主题的具体服务器名称决定。Agent 按专业领域分类，每种服务器类型对应一个表情符号。

示例：
任务：“我应该投资苹果公司的股票吗？”
回答：
{
    "server": "💰 Finance Agent",
    "agent_role_prompt": "你是一名经验丰富的金融分析 AI 助手。你的主要目标是根据给定的数据和趋势，撰写全面、敏锐、公正且结构严谨的金融报告。"
}
任务：“转售运动鞋能赚钱吗？”
回答：
{
    "server":  "📈 Business Analyst Agent",
    "agent_role_prompt": "你是一名经验丰富的 AI 商业分析助手。你的主要目标是根据给定的商业数据、市场趋势和战略分析，撰写全面、有洞察力、公正且结构系统的商业报告。"
}
任务：“特拉维夫最有趣的景点有哪些？”
回答：
{
    "server":  "🌍 Travel Agent",
    "agent_role_prompt": "你是一名见多识广的 AI 旅行向导助手。你的主要职责是围绕给定地点撰写引人入胜、有洞察力、公正且结构清晰的旅行报告，包括历史、景点和文化信息。"
}
"""

    @staticmethod
    def generate_summary_prompt(query, data):
        """生成针对给定问题和文本的摘要 prompt。"""

        return (
            f'{data}\n 根据以上文本，围绕以下任务或问题进行总结：“{query}”。\n 如果无法使用该文本回答问题，必须简要总结文本内容。\n 尽可能包含所有事实信息，例如数字、统计数据和引文。'
        )

    @staticmethod
    def generate_quick_summary_prompt(query: str, context: str) -> str:
        """生成针对给定问题和上下文的快速摘要 prompt。"""
        return f"""
仅根据提供的搜索结果，综合回答以下问题。
问题：“{query}”

搜索结果：
{context}

要求：
1. 提供一段连续的叙事性摘要。
2. 使用与搜索结果对应的编号 [1]、[2] 等引用来源。
3. 如果结果不足以回答问题，请明确说明。
4. 聚焦准确性和相关性。
"""

    @staticmethod
    def pretty_print_docs(docs: list[Document], top_n: int | None = None) -> str:
        """将文档列表压缩为上下文字符串。"""
        return f"\n".join(f"来源：{d.metadata.get('source')}\n"
                          f"标题：{d.metadata.get('title')}\n"
                          f"内容：{d.page_content}\n"
                          for i, d in enumerate(docs)
                          if top_n is None or i < top_n)

    @staticmethod
    def join_local_web_documents(docs_context: str, web_context: str) -> str:
        """将本地网页文档与从互联网抓取的上下文拼接起来。"""
        return f"本地文档上下文：{docs_context}\n\n网页来源上下文：{web_context}"

    ################################################################################################

    # 详细报告 prompt

    @staticmethod
    def generate_subtopics_prompt() -> str:
        return """
主主题：

{task}

研究数据：

{data}

- 构建一个子主题列表，作为该任务报告的章节标题。
- 可参考的子主题列表：{subtopics}。
- 子主题之间不得重复。
- 子主题数量最多为 {max_subtopics} 个。
- 最后按照任务相关性和意义排序，使其适合呈现在详细报告中。

“重要！”：
- 每个子主题都必须只与主主题和所提供的研究数据相关！

{format_instructions}
"""

    @staticmethod
    def generate_subtopic_report_prompt(
        current_subtopic,
        existing_headers: list,
        relevant_written_contents: list,
        main_topic: str,
        context,
        report_format: str = "apa",
        max_subsections=5,
        total_words=800,
        tone: Tone = Tone.Objective,
        language: str = "english",
    ) -> str:
        return f"""
上下文：
"{context}"

主主题和子主题：
使用最新可用信息，围绕主主题“{main_topic}”下的子主题“{current_subtopic}”撰写详细报告。
小节数量最多为 {max_subsections} 个。

内容重点：
- 报告应聚焦回答问题，结构清晰、信息丰富、深入，并在可用时包含事实和数据。
- 使用 Markdown 语法并遵循 {report_format.upper()} 格式。
- 展示数据、比较或结构化信息时，使用 Markdown 表格提高可读性。

重要：内容和章节必须具有唯一性：
- 这一要求对于确保内容独特且不与已有报告重叠至关重要。
- 撰写新小节前，仔细检查下方提供的已有标题和内容。
- 不得写入已有内容已经覆盖的信息。
- 不得使用已有标题作为新小节标题。
- 不要重复已有内容或其高度相似的变体，以避免重复。
- 如果包含嵌套小节，确保其独特且未被已有内容覆盖。
- 确保内容完全新颖，不与之前子主题报告中的任何信息重叠。

“已有子主题报告”：
- 已有子主题报告及其章节标题：

    {existing_headers}

- 之前子主题报告中的已有内容：

    {relevant_written_contents}

“结构和格式”：
- 由于本报告是更大报告的一部分，只包含按合适子主题划分的正文，不要包含引言或结论章节。

- 在报告引用相关来源 URL 的位置，必须加入 Markdown 超链接，例如：

    ### Section Header

    这是示例文本（[文内引用](url)）。

- 主子主题标题使用 H2（##），小节使用 H3（###）。
- 使用较小的 Markdown 标题（例如 H2 或 H3）组织内容，避免使用最大的标题 H1，因为它将用于整个报告的标题。
- 将内容组织为彼此独立的章节，使其与已有报告互补但不重叠。
- 如果添加与已有小节相似或相同的内容，应清楚说明新内容与之前子主题报告已有内容的区别。例如：

    ### 新标题（与已有标题相似）

    前一章节讨论了[主题 A]，本章节将探讨[主题 B]。"

“日期”：
如有需要，假定当前日期为 {datetime.now(timezone.utc).strftime('%B %d, %Y')}。

“重要！”：
- 必须使用以下语言撰写报告：{language}。
- 必须聚焦主主题，排除所有无关信息！
- 不得包含引言、结论、摘要或参考文献章节。
- 必须使用 {report_format.upper()} 格式的文内引用，并在引用句或段落末尾添加 Markdown 超链接，例如：([文内引用](url))。
- 如果添加相似或相同的小节，必须在必要时说明已有内容与新内容的区别。
- 报告至少包含 {total_words} 个词。
- 全文使用{tone.value}语气。

不要添加结论章节。
"""

    @staticmethod
    def generate_draft_titles_prompt(
        current_subtopic: str,
        main_topic: str,
        context: str,
        max_subsections: int = 5
    ) -> str:
        return f"""
“上下文”：
"{context}"

“主主题和子主题”：
使用最新可用信息，为主主题“{main_topic}”下的子主题“{current_subtopic}”的详细报告拟定章节标题草案。

“任务”：
1. 为子主题报告创建章节标题草案列表。
2. 每个标题都应简洁并与子主题相关。
3. 标题不应过于宏观，但要足够详细，能够覆盖子主题的主要方面。
4. 使用 Markdown 语法编写标题，使用 H3（###），因为 H1 和 H2 将用于更大报告的标题。
5. 确保标题覆盖子主题的主要方面。

“结构和格式”：
使用 Markdown 语法以列表形式提供标题草案，例如：

### 标题 1
### 标题 2
### 标题 3

“重要！”：
- 必须聚焦主主题，排除所有无关信息！
- 不得包含引言、结论、摘要或参考文献章节。
- 只创建标题，不要撰写内容。
"""

    @staticmethod
    def generate_report_introduction(question: str, research_summary: str = "", language: str = "english", report_format: str = "apa") -> str:
        return f"""{research_summary}\n
根据以上最新信息，围绕主题“{question}”撰写详细的报告引言。
- 引言应简洁、结构清晰、信息丰富，并使用 Markdown 语法。
- 由于引言是更大报告的一部分，不要包含报告中通常出现的其他章节。
- 引言前应使用一个适合整个报告主题的 H1 标题。
- 必须使用 {report_format.upper()} 格式的文内引用，并在引用句或段落末尾添加 Markdown 超链接，例如：([文内引用](url))。
如有需要，假定当前日期为 {datetime.now(timezone.utc).strftime('%B %d, %Y')}。
- 输出必须使用 {language} 语言。
"""


    @staticmethod
    def generate_report_conclusion(query: str, report_content: str, language: str = "english", report_format: str = "apa") -> str:
        """生成简洁的结论，总结研究报告的主要发现及其影响。"""
        prompt = f"""
    请根据下面的研究报告和研究任务撰写简洁结论，总结主要发现及其影响：

    研究任务：{query}

    研究报告：{report_content}

    结论应：
    1. 回顾研究的主要内容。
    2. 突出最重要的发现。
    3. 讨论影响或后续步骤。
    4. 长度约为 2 至 3 段。

    如果报告末尾没有“## 结论”章节标题，请在结论开头添加该标题。
    必须使用 {report_format.upper()} 格式的文内引用，并在引用句或段落末尾添加 Markdown 超链接，例如：([文内引用](url))。

    重要：整个结论必须使用 {language} 语言撰写。

    撰写结论：
    """

        return prompt


class GranitePromptFamily(PromptFamily):
    """IBM Granite 模型的 prompt 家族。"""


    def _get_granite_class(self) -> type[PromptFamily]:
        """根据版本号获取对应的 Granite prompt 家族。"""
        if "3.3" in self.cfg.smart_llm:
            return Granite33PromptFamily
        if "3" in self.cfg.smart_llm:
            return Granite3PromptFamily
        # 如果不是已知版本，则返回默认 prompt 家族。
        return PromptFamily

    def pretty_print_docs(self, *args, **kwargs) -> str:
        return self._get_granite_class().pretty_print_docs(*args, **kwargs)

    def join_local_web_documents(self, *args, **kwargs) -> str:
        return self._get_granite_class().join_local_web_documents(*args, **kwargs)


class Granite3PromptFamily(PromptFamily):
    """IBM Granite 3.X 模型（3.3 之前版本）的 prompt 家族。"""

    _DOCUMENTS_PREFIX = "<|start_of_role|>documents<|end_of_role|>\n"
    _DOCUMENTS_SUFFIX = "\n<|end_of_text|>"

    @classmethod
    def pretty_print_docs(cls, docs: list[Document], top_n: int | None = None) -> str:
        if not docs:
            return ""
        all_documents = "\n\n".join([
            f"文档 {doc.metadata.get('source', i)}\n" + \
            f"标题：{doc.metadata.get('title')}\n" + \
            doc.page_content
            for i, doc in enumerate(docs)
            if top_n is None or i < top_n
        ])
        return "".join([cls._DOCUMENTS_PREFIX, all_documents, cls._DOCUMENTS_SUFFIX])

    @classmethod
    def join_local_web_documents(cls, docs_context: str | list, web_context: str | list) -> str:
        """使用 Granite 偏好的格式拼接本地网页文档。"""
        if isinstance(docs_context, str) and docs_context.startswith(cls._DOCUMENTS_PREFIX):
            docs_context = docs_context[len(cls._DOCUMENTS_PREFIX):]
        if isinstance(web_context, str) and web_context.endswith(cls._DOCUMENTS_SUFFIX):
            web_context = web_context[:-len(cls._DOCUMENTS_SUFFIX)]
        all_documents = "\n\n".join([docs_context, web_context])
        return "".join([cls._DOCUMENTS_PREFIX, all_documents, cls._DOCUMENTS_SUFFIX])


class Granite33PromptFamily(PromptFamily):
    """IBM Granite 3.3 模型的 prompt 家族。"""

    _DOCUMENT_TEMPLATE = """<|start_of_role|>document {{"document_id": "{document_id}"}}<|end_of_role|>
{document_content}<|end_of_text|>
"""

    @staticmethod
    def _get_content(doc: Document) -> str:
        doc_content = doc.page_content
        if title := doc.metadata.get("title"):
            doc_content = f"标题：{title}\n{doc_content}"
        return doc_content.strip()

    @classmethod
    def pretty_print_docs(cls, docs: list[Document], top_n: int | None = None) -> str:
        return "\n".join([
            cls._DOCUMENT_TEMPLATE.format(
                document_id=doc.metadata.get("source", i),
                document_content=cls._get_content(doc),
            )
            for i, doc in enumerate(docs)
            if top_n is None or i < top_n
        ])

    @classmethod
    def join_local_web_documents(cls, docs_context: str | list, web_context: str | list) -> str:
        """使用 Granite 偏好的格式拼接本地网页文档。"""
        return "\n\n".join([docs_context, web_context])

## 工厂 #########################################################################

# 各类 prompt 生成函数使用的统一函数签名。
PROMPT_GENERATOR = Callable[
    [
        str,        # 问题
        str,        # 上下文
        str,        # 报告来源
        str,        # 报告格式
        str | None, # 语气
        int,        # 总词数
        str,        # 语言
    ],
    str,
]

report_type_mapping = {
    ReportType.ResearchReport.value: "generate_report_prompt",
    ReportType.ResourceReport.value: "generate_resource_report_prompt",
    ReportType.OutlineReport.value: "generate_outline_report_prompt",
    ReportType.CustomReport.value: "generate_custom_report_prompt",
    ReportType.SubtopicReport.value: "generate_subtopic_report_prompt",
    ReportType.DeepResearch.value: "generate_deep_research_prompt",
}


def get_prompt_by_report_type(
    report_type: str,
    prompt_family: type[PromptFamily] | PromptFamily,
):
    prompt_by_type = getattr(prompt_family, report_type_mapping.get(report_type, ""), None)
    default_report_type = ReportType.ResearchReport.value
    if not prompt_by_type:
        warnings.warn(
            f"Invalid report type: {report_type}.\n"
            f"Please use one of the following: {', '.join([enum_value for enum_value in report_type_mapping.keys()])}\n"
            f"Using default report type: {default_report_type} prompt.",
            UserWarning,
        )
        prompt_by_type = getattr(prompt_family, report_type_mapping.get(default_report_type))
    return prompt_by_type


prompt_family_mapping = {
    PromptFamilyEnum.Default.value: PromptFamily,
    PromptFamilyEnum.Granite.value: GranitePromptFamily,
    PromptFamilyEnum.Granite3.value: Granite3PromptFamily,
    PromptFamilyEnum.Granite31.value: Granite3PromptFamily,
    PromptFamilyEnum.Granite32.value: Granite3PromptFamily,
    PromptFamilyEnum.Granite33.value: Granite33PromptFamily,
}


def get_prompt_family(
    prompt_family_name: PromptFamilyEnum | str, config: Config,
) -> PromptFamily:
    """根据名称或值获取 prompt 家族。"""
    if isinstance(prompt_family_name, PromptFamilyEnum):
        prompt_family_name = prompt_family_name.value
    if prompt_family := prompt_family_mapping.get(prompt_family_name):
        return prompt_family(config)
    warnings.warn(
        f"Invalid prompt family: {prompt_family_name}.\n"
        f"Please use one of the following: {', '.join([enum_value for enum_value in prompt_family_mapping.keys()])}\n"
        f"Using default prompt family: {PromptFamilyEnum.Default.value} prompt.",
        UserWarning,
    )
    return PromptFamily()

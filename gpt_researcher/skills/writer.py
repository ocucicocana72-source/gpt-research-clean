"""Report generator skill for GPT Researcher.

This module provides the ReportGenerator class that handles report
writing, including introductions, conclusions, and subtopic management.
"""

import json
from typing import Dict, Optional

from ..actions import (
    generate_draft_section_titles,
    generate_report,
    stream_output,
    write_conclusion,
    write_report_introduction,
)
from ..utils.llm import construct_subtopics


class ReportGenerator:
    """根据研究数据生成报告。

    该类负责处理报告生成的各个方面，包括
    撰写引言和结论，以及管理报告结构。

    属性：
    researcher：父级 GPTResearcher 实例。
    research_params：包含报告生成参数的字典。
    """

    def __init__(self, researcher):
        """Initialize the ReportGenerator.

        Args:
            researcher: The GPTResearcher instance that owns this generator.
        """
        self.researcher = researcher
        self.research_params = {
            "query": self.researcher.query,
            "agent_role_prompt": self.researcher.cfg.agent_role or self.researcher.role,
            "report_type": self.researcher.report_type,
            "report_source": self.researcher.report_source,
            "tone": self.researcher.tone,
            "websocket": self.researcher.websocket,
            "cfg": self.researcher.cfg,
            "headers": self.researcher.headers,
        }

    async def write_report(self, existing_headers: list = [], relevant_written_contents: list = [], ext_context=None, custom_prompt="", available_images: list = None) -> str:
        """
        根据现有标题和相关内容生成报告。

        参数：

        existing_headers（列表）：现有标题列表。

        relevant_written_contents（列表）：相关内容列表。

        ext_context（可选）：外部上下文（如有）。

        custom_prompt（字符串）：报告的自定义提示。

        available_images（列表）：可用于嵌入的预生成图像。

        返回值：

        字符串：生成的报告。
        """
        available_images = available_images or []
        
        # 在写报告前发送选中的图片。
        research_images = self.researcher.get_research_images()
        if research_images:
            await stream_output(
                "images",
                "selected_images",
                json.dumps(research_images),
                self.researcher.websocket,
                True,
                research_images
            )

        # 外部上下文用于详细报告的子章节；否则使用本轮研究器累计的资料。
        context = ext_context or self.researcher.context

        # 防止在没有资料时编造报告：如果所有检索器都为空、被阻断或限流，
        # 不要悄悄生成看似有来源的确定性报告，而要明确返回无法生成的结果。
        # 因此保护的关键并不是在 Prompt 里告诉模型“不要编造”，而是：
        # 在调用大模型之前直接结束函数，从程序控制流上阻止报告生成。
        _ctx = "\n".join(context) if isinstance(context, list) else str(context or "")
        if not _ctx.strip():
            return (
                f'I could not gather any source material for "{self.researcher.query}". '
                "No sources were retrieved (searches may have returned nothing or been "
                "blocked), so I am not able to produce a reliable, sourced report."
            )
        
        # 记录图片可用情况。
        if available_images and self.researcher.verbose:
            await stream_output(
                "logs",
                "images_available",
                f"🖼️ {len(available_images)} pre-generated images available for embedding",
                self.researcher.websocket,
            )
        
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "writing_report",
                f"✍️ Writing report for '{self.researcher.query}'...",
                self.researcher.websocket,
            )

        report_params = self.research_params.copy()
        if not report_params["agent_role_prompt"]:
            report_params["agent_role_prompt"] = self.researcher.cfg.agent_role or self.researcher.role
        report_params["context"] = context
        report_params["custom_prompt"] = custom_prompt
        report_params["available_images"] = available_images  # Pass pre-generated images

        # 子主题报告需要主问题和已写章节，避免不同章节重复相同内容。
        if self.researcher.report_type == "subtopic_report":
            report_params.update({
                "main_topic": self.researcher.parent_query,
                "existing_headers": existing_headers,
                "relevant_written_contents": relevant_written_contents,
                "cost_callback": self.researcher.add_costs,
            })
        else:
            report_params["cost_callback"] = self.researcher.add_costs

        report = await generate_report(**report_params, **self.researcher.kwargs)

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "report_written",
                f"📝 Report written for '{self.researcher.query}'",
                self.researcher.websocket,
            )

        return report

    async def write_report_conclusion(self, report_content: str) -> str:
        """
        Write the conclusion for the report.

        Args:
            report_content (str): The content of the report.

        Returns:
            str: The generated conclusion.
        """
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "writing_conclusion",
                f"✍️ Writing conclusion for '{self.researcher.query}'...",
                self.researcher.websocket,
            )

        conclusion = await write_conclusion(
            query=self.researcher.query,
            context=report_content,
            config=self.researcher.cfg,
            agent_role_prompt=self.researcher.cfg.agent_role or self.researcher.role,
            cost_callback=self.researcher.add_costs,
            websocket=self.researcher.websocket,
            prompt_family=self.researcher.prompt_family,
            **self.researcher.kwargs
        )

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "conclusion_written",
                f"📝 Conclusion written for '{self.researcher.query}'",
                self.researcher.websocket,
            )

        return conclusion

    async def write_introduction(self):
        """Write the introduction section of the report."""
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "writing_introduction",
                f"✍️ Writing introduction for '{self.researcher.query}'...",
                self.researcher.websocket,
            )

        introduction = await write_report_introduction(
            query=self.researcher.query,
            context=self.researcher.context,
            agent_role_prompt=self.researcher.cfg.agent_role or self.researcher.role,
            config=self.researcher.cfg,
            websocket=self.researcher.websocket,
            cost_callback=self.researcher.add_costs,
            prompt_family=self.researcher.prompt_family,
            **self.researcher.kwargs
        )

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "introduction_written",
                f"📝 Introduction written for '{self.researcher.query}'",
                self.researcher.websocket,
            )

        return introduction

    async def get_subtopics(self):
        """Retrieve subtopics for the research."""
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "generating_subtopics",
                f"🌳 Generating subtopics for '{self.researcher.query}'...",
                self.researcher.websocket,
            )

        subtopics = await construct_subtopics(
            task=self.researcher.query,
            data=self.researcher.context,
            config=self.researcher.cfg,
            subtopics=self.researcher.subtopics,
            prompt_family=self.researcher.prompt_family,
            **self.researcher.kwargs
        )

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "subtopics_generated",
                f"📊 Subtopics generated for '{self.researcher.query}'",
                self.researcher.websocket,
            )

        return subtopics

    async def get_draft_section_titles(self, current_subtopic: str):
        """Generate draft section titles for the report."""
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "generating_draft_sections",
                f"📑 Generating draft section titles for '{self.researcher.query}'...",
                self.researcher.websocket,
            )

        draft_section_titles = await generate_draft_section_titles(
            query=self.researcher.query,
            current_subtopic=current_subtopic,
            context=self.researcher.context,
            role=self.researcher.cfg.agent_role or self.researcher.role,
            websocket=self.researcher.websocket,
            config=self.researcher.cfg,
            cost_callback=self.researcher.add_costs,
            prompt_family=self.researcher.prompt_family,
            **self.researcher.kwargs
        )

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "draft_sections_generated",
                f"🗂️ Draft section titles generated for '{self.researcher.query}'",
                self.researcher.websocket,
            )

        return draft_section_titles

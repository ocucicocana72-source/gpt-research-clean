"""
Provides a command line interface for the GPTResearcher class.

Usage:

```shell
python cli.py "<query>" --report_type <report_type> --tone <tone> --query_domains <foo.com,bar.com>
```

"""
import argparse
import asyncio
import re
from argparse import RawTextHelpFormatter
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from backend.report_type import DetailedReport
from backend.utils import write_md_to_pdf, write_md_to_word
from gpt_researcher import GPTResearcher
from gpt_researcher.utils.enum import ReportSource, ReportType, Tone
from gpt_researcher.utils.llm import create_chat_completion

# =============================================================================
# CLI
# =============================================================================

# cli.add_argument()：定义支持哪些参数
#         ↓
# 用户在终端传入参数
#         ↓
# cli.parse_args()：解析参数
#         ↓
# args.xxx：在 Python 代码中取得参数值

cli = argparse.ArgumentParser(
    description="Generate a research report.",
    # Enables the use of newlines in the help message
    formatter_class=RawTextHelpFormatter)

# =====================================
# Arg: Query
# =====================================

cli.add_argument(
    # Position 0 argument
    "query",
    type=str,
    help="The query to conduct research on.")

# =====================================
# Arg: Report Type
# =====================================

# 直接从枚举生成合法值，使 CLI 的参数校验与核心报告类型保持一致。
choices = [report_type.value for report_type in ReportType]

report_type_descriptions = {
    ReportType.ResearchReport.value: "Summary - Short and fast (~2 min)",
    ReportType.DetailedReport.value: "Detailed - In depth and longer (~5 min)",
    ReportType.ResourceReport.value: "",
    ReportType.OutlineReport.value: "",
    ReportType.CustomReport.value: "",
    ReportType.SubtopicReport.value: "",
    ReportType.DeepResearch.value: "Deep Research"
}

cli.add_argument(
    "--report_type",
    type=str,
    help="The type of report to generate. Options:\n" + "\n".join(
        f"  {choice}: {report_type_descriptions[choice]}" for choice in choices
    ),
    # Deserialize ReportType as a List of strings:
    choices=choices,
    required=True)

# =====================================
# Arg: Tone
# =====================================

cli.add_argument(
    "--tone",
    type=str,
    help="The tone of the report (optional).",
    choices=["objective", "formal", "analytical", "persuasive", "informative",
            "explanatory", "descriptive", "critical", "comparative", "speculative",
            "reflective", "narrative", "humorous", "optimistic", "pessimistic"],
    default="objective"
)

# =====================================
# Arg: Encoding
# =====================================

cli.add_argument(
    "--encoding",
    type=str,
    help="The encoding to use for the output file (default: utf-8).",
    default="utf-8"
)

# =====================================
# Arg: Query Domains
# =====================================

cli.add_argument(
    "--query_domains",
    type=str,
    help="A comma-separated list of domains to search for the query.",
    default=""
)

# =====================================
# Arg: Report Source
# =====================================

cli.add_argument(
    "--report_source",
    type=str,
    help="The source of information for the report.",
    choices=["web", "local", "hybrid", "azure", "langchain_documents",
             "langchain_vectorstore", "static"],
    default="web"
)

# =====================================
# Arg: Output Format Flags
# =====================================

cli.add_argument(
    "--no-pdf",
    action="store_true",
    help="Skip PDF generation (generate markdown and DOCX only)."
)

cli.add_argument(
    "--no-docx",
    action="store_true",
    help="Skip DOCX generation (generate markdown and PDF only)."
)

# =============================================================================
# Output helpers: LLM-based filename + sanitization + YAML frontmatter
# =============================================================================

# 同时排除 Windows 非法字符和控制字符，避免 LLM 生成的标题无法作为文件名。
_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _sanitize_filename(name: str, max_len: int = 60) -> str:
    """Turn an arbitrary string into a safe filename stem.

    - Strips characters that are illegal on Windows/Linux filesystems.
    - Collapses runs of whitespace into a single underscore.
    - Trims leading/trailing dots and spaces (Windows rejects them).
    - Truncates to ``max_len`` characters (counted as unicode code points,
      so CJK characters count as 1).
      将任意字符串转换为安全的文件名主干。

    - 移除 Windows/Linux 文件系统中不允许使用的字符。
    - 将连续的空白字符合并为单个下划线。
    - 移除开头和结尾的点号及空格（Windows 不允许这些字符出现在这些位置）。
    - 截断至 ``max_len`` 个字符（按 Unicode 码点计数，即中日韩字符计为 1 个字符）。
    """
    cleaned = _INVALID_FILENAME_CHARS.sub("", name).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = cleaned.strip("._ ")
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip("._ ")
    return cleaned or "untitled"


async def _generate_task_title(
    query: str,
    report: str,
    researcher: GPTResearcher | None,
) -> str:
    """请已配置的快速大语言模型（LLM）为报告生成一个简短的标题。

    该标题将用作文件名主体，因此要求简短（不超过 20 个字符）
    且不包含标点符号。若因故调用 LLM 失败，系统将回退使用原始查询内容，
    以确保整体流程不中断。
    """
    # 标题只需要主题信息；截断正文可避免为命名再次传入整篇报告。
    report_preview = (report or "")[:600]
    prompt = (
        "You are a librarian. Produce a concise title (at most 20 characters, "
        "same language as the user's query) for the following research report. "
        "The title will be used as a filename, so:\n"
        "  - do not wrap it in quotes, brackets or punctuation\n"
        "  - do not add any prefix, suffix or explanation\n"
        "  - output the title on a single line, nothing else\n\n"
        f"[User query]\n{query}\n\n"
        f"[Report preview]\n{report_preview}\n\n"
        "Title:"
    )

    try:
        # 复用本次研究的快速模型配置，并将这次调用记入同一成本统计。
        cfg = researcher.cfg if researcher else None
        title = await create_chat_completion(
            model=(cfg.fast_llm_model if cfg else None),
            llm_provider=(cfg.fast_llm_provider if cfg else None),
            messages=[
                {"role": "system", "content": "You output only a short title, nothing else."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=64,
            llm_kwargs=(cfg.llm_kwargs if cfg else None),
            cost_callback=(researcher.add_costs if researcher else None),
        )
        # LLM 偶尔会加引号或多行说明；只保留第一行纯标题。
        title = title.strip().strip('"').strip("'").strip("《》").splitlines()[0].strip()
        return title or query
    except Exception as e:
        print(f"Warning: LLM title generation failed ({e}); falling back to query.")        # 调用大模型失败
        return query


def _build_frontmatter(
    task_id: str,
    title: str,
    args: argparse.Namespace,
    researcher: GPTResearcher | None,
) -> str:
    """构建一个置于 Markdown 报告开头的 YAML Frontmatter 块。

    字段包括：task_id、title、query、report_type、report_source、tone、
    query_domains（非空时）、created_at、sources_count、total_cost_usd。
    """
    def _yaml_quote(v: str) -> str:
        # Double-quoted YAML string with backslash/quote escaping.
        return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'

    # 这些运行数据被写入 Markdown 头部，便于后续检索、审计和成本分析。 添加一些任务id 标题名字 报告类型 消耗的token query_domains 存储的是“允许搜索的域名列表”，用于限制研究器从哪些网站获取资料。
    query_domains = [d for d in (args.query_domains.split(",") if args.query_domains else []) if d]
    sources_count = len(researcher.visited_urls) if researcher else 0
    total_cost = round(researcher.get_costs(), 6) if researcher else 0.0

    lines = [
        "---",
        f"task_id: {_yaml_quote(task_id)}",
        f"title: {_yaml_quote(title)}",
        f"query: {_yaml_quote(args.query)}",
        f"report_type: {_yaml_quote(args.report_type)}",
        f"report_source: {_yaml_quote(args.report_source)}",
        f"tone: {_yaml_quote(args.tone)}",
    ]
    if query_domains:
        lines.append("query_domains:")
        lines.extend(f"  - {_yaml_quote(d)}" for d in query_domains)
    lines.append(f"created_at: {_yaml_quote(datetime.now().isoformat(timespec='seconds'))}")
    lines.append(f"sources_count: {sources_count}")
    lines.append(f"total_cost_usd: {total_cost}")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _resolve_unique_path(output_dir: Path, stem: str, ext: str = ".md") -> Path:
    """Return ``output_dir/stem.ext``, suffixing ``_2``, ``_3``... on collisions."""
    # 不覆盖同名历史报告，而是为新文件递增后缀。
    candidate = output_dir / f"{stem}{ext}"
    if not candidate.exists():
        return candidate
    i = 2
    while True:
        candidate = output_dir / f"{stem}_{i}{ext}"
        if not candidate.exists():
            return candidate
        i += 1


# =============================================================================
# Main
# =============================================================================

async def main(args):
    """
    针对给定查询进行研究，生成报告，并将其以 Markdown 文件形式写入输出目录。
    """
    # argparse 保留原始字符串；研究器内部需要域名列表。
    query_domains = args.query_domains.split(",") if args.query_domains else []

    researcher: GPTResearcher | None = None

    if args.report_type == 'detailed_report':
        # 详细报告由专用编排器拆分子主题、分段研究，再组装引言、正文和结论。
        # 注意：此分支目前固定使用 web_search，也未传入 args.tone。
        detailed_report = DetailedReport(
            query=args.query,
            query_domains=query_domains,
            report_type="research_report",
            report_source="web_search",
        )

        report = await detailed_report.run()
        # DetailedReport owns an internal GPTResearcher; reuse it so we can
        # surface sources_count / total_cost and reuse the fast LLM for the title.
        researcher = getattr(detailed_report, "gpt_researcher", None)
    else:
        # CLI 接收简短关键词，核心类则需要携带完整写作描述的 Tone 枚举。
        tone_map = {
            "objective": Tone.Objective,
            "formal": Tone.Formal,
            "analytical": Tone.Analytical,
            "persuasive": Tone.Persuasive,
            "informative": Tone.Informative,
            "explanatory": Tone.Explanatory,
            "descriptive": Tone.Descriptive,
            "critical": Tone.Critical,
            "comparative": Tone.Comparative,
            "speculative": Tone.Speculative,
            "reflective": Tone.Reflective,
            "narrative": Tone.Narrative,
            "humorous": Tone.Humorous,
            "optimistic": Tone.Optimistic,
            "pessimistic": Tone.Pessimistic
        }

        researcher = GPTResearcher(
            query=args.query,
            query_domains=query_domains,
            report_type=args.report_type,
            report_source=args.report_source,
            tone=tone_map[args.tone],
            encoding=args.encoding
        )

        # 研究和写作是两个阶段：先检索/抓取并积累上下文，再基于上下文生成报告。
        await researcher.conduct_research()

        report = await researcher.write_report()

    # ------------------------------------------------------------------
    # Write the report: LLM-generated title -> safe filename stem,
    # prepended with a YAML frontmatter block describing the run.
    # ------------------------------------------------------------------
    # 每次运行有独立 ID；文件名则使用 LLM 标题并再做跨平台清洗。
    task_id = str(uuid4())

    raw_title = await _generate_task_title(args.query, report, researcher)
    safe_stem = _sanitize_filename(raw_title)
    print(f"Task title: {raw_title}  ->  filename stem: {safe_stem}")

    # YAML frontmatter 与报告正文共存于同一 Markdown，转换格式时也使用同份内容。
    frontmatter = _build_frontmatter(task_id, raw_title, args, researcher)
    final_markdown = frontmatter + report

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = _resolve_unique_path(output_dir, safe_stem, ".md")
    md_path.write_text(final_markdown, encoding="utf-8")
    print(f"Report written to '{md_path}'")

    # PDF/DOCX share the same stem so the three files stay grouped together.
    shared_stem = md_path.stem

    # Generate PDF if not disabled
    if not args.no_pdf:
        try:
            pdf_path = await write_md_to_pdf(final_markdown, shared_stem)
            if pdf_path:
                print(f"PDF written to '{pdf_path}'")
        except Exception as e:
            print(f"Warning: PDF generation failed: {e}")

    # Generate DOCX if not disabled
    if not args.no_docx:
        try:
            docx_path = await write_md_to_word(final_markdown, shared_stem)
            if docx_path:
                print(f"DOCX written to '{docx_path}'")
        except Exception as e:
            print(f"Warning: DOCX generation failed: {e}")

if __name__ == "__main__":
    # 仅在直接执行 cli.py 时加载 .env、解析参数并启动异步事件循环。
    load_dotenv()
    args = cli.parse_args()
    asyncio.run(main(args))

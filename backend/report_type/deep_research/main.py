from gpt_researcher import GPTResearcher
from backend.utils import write_md_to_pdf
import asyncio


async def main(task: str):
    """运行深度研究示例，显示进度并把最终报告导出为 PDF。"""
    # 定义进度回调。
    def on_progress(progress):
        """在命令行打印深度、广度和查询完成度。"""
        print(f"Depth: {progress.current_depth}/{progress.total_depth}")
        print(f"Breadth: {progress.current_breadth}/{progress.total_breadth}")
        print(f"Queries: {progress.completed_queries}/{progress.total_queries}")
        if progress.current_query:
            print(f"Current query: {progress.current_query}")
    
    # 使用深度研究类型初始化研究器。
    researcher = GPTResearcher(
        query=task,
        report_type="deep",  # This will trigger deep research
    )
    
    # 执行研究并跟踪进度。
    print("Starting deep research...")
    context = await researcher.conduct_research(on_progress=on_progress)
    print("\nResearch completed. Generating report...")
    
    # 生成最终报告。
    report = await researcher.write_report()
    await write_md_to_pdf(report, "deep_research_report")
    print(f"\nFinal Report: {report}")

if __name__ == "__main__":
    query = "What are the most effective ways for beginners to start investing?"
    asyncio.run(main(query))

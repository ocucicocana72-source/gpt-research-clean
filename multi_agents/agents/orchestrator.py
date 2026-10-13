import os
import time
import datetime
from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import MemorySaver
from .utils.views import print_agent_output
from ..memory.research import ResearchState
from .utils.utils import sanitize_filename
from .plan_review import (
    DEFAULT_MAX_PLAN_REVISIONS,
    route_human_feedback,
)
from .fact_review import (
    DEFAULT_MAX_FACT_CHECK_REVISIONS,
    MaxFactCheckRevisionsExceededError,
    route_fact_check,
)

# Import agent classes
from . import \
    WriterAgent, \
    EditorAgent, \
    PublisherAgent, \
    ResearchAgent, \
    HumanAgent, \
    FactCheckerAgent, \
    VisualizerAgent


class ChiefEditorAgent:
    """Agent responsible for managing and coordinating editing tasks."""

    def __init__(self, task: dict, websocket=None, stream_output=None, tone=None, headers=None):
        self.task = task
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers or {}
        self.tone = tone
        self.task_id = self._generate_task_id()
        self.output_dir = self._create_output_directory()

    def _generate_task_id(self):
        # Currently time based, but can be any unique identifier
        return int(time.time())

    def _create_output_directory(self):
        output_dir = "./outputs/" + \
            sanitize_filename(
                f"run_{self.task_id}_{self.task.get('query')[0:40].strip()}")

        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _initialize_agents(self):
        return {
            "writer": WriterAgent(self.websocket, self.stream_output, self.headers),
            "editor": EditorAgent(self.websocket, self.stream_output, self.tone, self.headers),
            "research": ResearchAgent(self.websocket, self.stream_output, self.tone, self.headers),
            "publisher": PublisherAgent(self.output_dir, self.websocket, self.stream_output, self.headers),
            "human": HumanAgent(self.websocket, self.stream_output, self.headers),
            "fact_checker": FactCheckerAgent(self.websocket, self.stream_output, self.headers),
            "visualizer": VisualizerAgent(self.websocket, self.stream_output, self.headers)
        }

    def _create_workflow(self, agents):
        workflow = StateGraph(ResearchState)

        # Add nodes for each agent
        workflow.add_node("browser", agents["research"].run_initial_research)
        workflow.add_node("planner", agents["editor"].plan_research)
        workflow.add_node("researcher", agents["editor"].run_parallel_research)
        workflow.add_node("writer", agents["writer"].run)
        workflow.add_node("fact_checker", agents["fact_checker"].run)
        workflow.add_node("visualizer", agents["visualizer"].run)
        workflow.add_node("publisher", agents["publisher"].run)
        workflow.add_node("human", agents["human"].review_plan)

        # Add edges
        self._add_workflow_edges(workflow)

        return workflow

    def _add_workflow_edges(self, workflow):
        workflow.add_edge('browser', 'planner')
        workflow.add_edge('planner', 'human')
        workflow.add_edge('researcher', 'writer')
        workflow.add_edge('writer', 'fact_checker')
        workflow.add_edge('visualizer', 'publisher')
        workflow.set_entry_point("browser")
        workflow.add_edge('publisher', END)

        # Human loop: exact "no" approval (human agent) + bounded plan revisions
        # via plan_review.route_human_feedback (task.max_plan_revisions).
        workflow.add_conditional_edges(
            'human',
            self._route_human_feedback,
            {"accept": "researcher", "revise": "planner"},
        )

        # Fact-checker loop — bounded via task.max_fact_check_revisions
        workflow.add_conditional_edges(
            'fact_checker',
            self._route_fact_check,
            {"accept": "visualizer", "revise": "writer"}
        )

    def _route_human_feedback(self, review):
        """
        处理人工计划反馈；达到最大计划修订次数（`max_plan_revisions`）后强制接受。

        当超出设定的上限时，`route_human_feedback` 会引发异常。
        对于图中的边，我们将其视为“接受”操作，从而使执行流程继续进入
        “研究”（research）阶段，而不是因达到 LangGraph 默认的递归限制（`recursion_limit`）而终止。
        """
        from .plan_review import MaxPlanRevisionsExceededError

        max_plan_revisions = self.task.get(
            "max_plan_revisions", DEFAULT_MAX_PLAN_REVISIONS)
        try:
            return route_human_feedback(review, max_plan_revisions)
        except MaxPlanRevisionsExceededError:
            return "accept"

    def _route_fact_check(self, state):
        """对路由事实核查结果进行处理；一旦达到上限，则强制接受。

            当超出 `max_fact_check_revisions` 限制时，`route_fact_check` 会引发异常。
            与“人在回路”（human-in-the-loop）模式类似，边缘（edge）会将此情况视为“接受”，从而使运行流程得以继续，避免因触及 LangGraph 的递归限制（recursion_limit）而终止。
        """
        max_fact_check_revisions = self.task.get(
            "max_fact_check_revisions", DEFAULT_MAX_FACT_CHECK_REVISIONS)
        try:
            return route_fact_check(state, max_fact_check_revisions)
        except MaxFactCheckRevisionsExceededError:
            return "accept"

    def init_research_team(self):
        """Initialize and create a workflow for the research team."""
        agents = self._initialize_agents()
        return self._create_workflow(agents)

    async def _log_research_start(self):
        message = f"Starting the research process for query '{self.task.get('query')}'..."
        if self.websocket and self.stream_output:
            await self.stream_output("logs", "starting_research", message, self.websocket)
        else:
            print_agent_output(message, "MASTER")

    async def run_research_task(self, task_id=None):
        """
        Run a research task with the initialized research team.

        Args:
            task_id (optional): The ID of the task to run.

        Returns:
            The result of the research task.
        """
        research_team = self.init_research_team()
        chain = research_team.compile()

        await self._log_research_start()

        config = {
            "configurable": {
                "thread_id": task_id,
                "thread_ts": datetime.datetime.utcnow()
            }
        }

        result = await chain.ainvoke({"task": self.task}, config=config)
        return result

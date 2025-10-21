"""DeepAgents implemented as Middleware"""

from deepagents.events import get_event_bus
from deepagents.context import AgentContext
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, SummarizationMiddleware
from langchain.agents.middleware.prompt_caching import AnthropicPromptCachingMiddleware
from langchain_core.tools import BaseTool, tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langchain.chat_models import init_chat_model
from langgraph.types import Command
from langgraph.runtime import Runtime
from langchain.tools.tool_node import InjectedState
from typing import Annotated, TYPE_CHECKING
from deepagents.state import PlanningState, FilesystemState
from deepagents.tools import write_todos, ls, read_file, write_file, edit_file
from deepagents.prompts import WRITE_TODOS_SYSTEM_PROMPT, TASK_SYSTEM_PROMPT, FILESYSTEM_SYSTEM_PROMPT, TASK_TOOL_DESCRIPTION, BASE_AGENT_PROMPT

if TYPE_CHECKING:
    from deepagents.agent import Agent
else:
    # Import at runtime for _get_agents (no circular dependency)
    from deepagents.agent import ToolAgent

###########################
# Observer Middleware
###########################

class ObserverMiddleware(AgentMiddleware):
    """Middleware that emits all agent events to the event bus.

    Emits events for:
    - Messages (AI thinking, tool calls, tool responses)
    - Todos updates
    - Files updates (future)

    Agent name is determined from runtime.context.agent_name (no hardcoding).
    """

    def __init__(self):
        super().__init__()
        self._last_message_count = 0
        self._last_todos = None

    def after_model(self, state: AgentState, runtime: Runtime[AgentContext]) -> dict | None:
        """Emit events after each model interaction."""
        event_bus = get_event_bus()

        # Get agent info from runtime context (dynamic, not hardcoded)
        if runtime.context:
            agent_name = runtime.context.agent_name
            agent_fg_color = runtime.context.agent_fg_color
            agent_bg_color = runtime.context.agent_bg_color
            agent_level = runtime.context.agent_level
        else:
            agent_name = "unknown"
            agent_fg_color = "#000000"
            agent_bg_color = "#ffffff"
            agent_level = 0

        # Emit NEW messages only (to maintain sequence)
        if "messages" in state and state["messages"]:
            current_count = len(state["messages"])
            if current_count > self._last_message_count:
                # Emit only new messages
                for message in state["messages"][self._last_message_count:]:
                    event_bus.emit("message", agent_name, agent_fg_color, agent_bg_color, agent_level, message)
                self._last_message_count = current_count

        # Emit todo updates (only if changed)
        if "todos" in state:
            current_todos = state["todos"]
            if current_todos != self._last_todos:
                event_bus.emit("todos", agent_name, agent_fg_color, agent_bg_color, agent_level, current_todos)
                self._last_todos = current_todos

        # Emit file updates (if needed)
        if "files" in state:
            event_bus.emit("files", agent_name, agent_fg_color, agent_bg_color, agent_level, state["files"])

        return None


###########################
# Planning Middleware
###########################

class PlanningMiddleware(AgentMiddleware):
    state_schema = PlanningState
    tools = [write_todos]

    def modify_model_request(self, request: ModelRequest, agent_state: PlanningState, runtime: Runtime) -> ModelRequest:
        request.system_prompt = request.system_prompt + "\n\n" + WRITE_TODOS_SYSTEM_PROMPT
        return request

###########################
# Filesystem Middleware
###########################

class FilesystemMiddleware(AgentMiddleware):
    state_schema = FilesystemState
    tools = [ls, read_file, write_file, edit_file]

    def modify_model_request(self, request: ModelRequest, agent_state: FilesystemState, runtime: Runtime) -> ModelRequest:
        request.system_prompt = request.system_prompt + "\n\n" + FILESYSTEM_SYSTEM_PROMPT
        return request

###########################
# SubAgent Middleware
###########################

class SubAgentMiddleware(AgentMiddleware):
    def __init__(
        self,
        default_subagent_tools: list[BaseTool] = [],
        subagents: list["Agent"] = [],
        model=None,
        is_async=False,
    ) -> None:
        super().__init__()
        task_tool = create_task_tool(
            default_subagent_tools=default_subagent_tools,
            subagents=subagents,
            model=model,
            is_async=is_async,
        )
        self.tools = [task_tool]

    def modify_model_request(self, request: ModelRequest, agent_state: AgentState, runtime: Runtime) -> ModelRequest:
        request.system_prompt = request.system_prompt + "\n\n" + TASK_SYSTEM_PROMPT
        return request

def _get_agents(
    default_subagent_tools: list[BaseTool],
    subagents: list["Agent"],
    model
):
    """Build agents dict mapping name -> Agent object.

    Creates general-purpose agent and includes all provided subagents.
    """
    # General purpose subagent middleware
    general_purpose_middleware = [
        ObserverMiddleware(),  # Agent name from runtime.context.agent_name
        PlanningMiddleware(),
        FilesystemMiddleware(),
        SummarizationMiddleware(
            model=model,
            max_tokens_before_summary=120000,
            messages_to_keep=20,
        ),
        AnthropicPromptCachingMiddleware(ttl="5m", unsupported_model_behavior="ignore"),
    ]

    # Create general-purpose agent graph
    general_purpose_graph = create_agent(
        model,
        prompt=BASE_AGENT_PROMPT,
        tools=default_subagent_tools,
        checkpointer=False,
        context_schema=AgentContext,
        middleware=general_purpose_middleware
    )

    # Wrap in ToolAgent for consistent interface
    agents = {
        "general-purpose": ToolAgent(
            name="general-purpose",
            description="General purpose agent for complex tasks",
            graph=general_purpose_graph,
            fg_color="#000000",
            bg_color="#ffffff",
        )
    }

    # Add all subagents (already built Agent objects)
    for agent in subagents:
        agents[agent.name] = agent

    return agents


def _get_subagent_description(subagents: list["Agent"]):
    """Build description list for subagents."""
    return [f"- {agent.name}: {agent.description}" for agent in subagents]


def create_task_tool(
    default_subagent_tools: list[BaseTool],
    subagents: list["Agent"],
    model,
    is_async: bool = False,
):
    agents = _get_agents(
        default_subagent_tools, subagents, model
    )
    other_agents_string = _get_subagent_description(subagents)

    if is_async:
        @tool(
            description=TASK_TOOL_DESCRIPTION.format(other_agents=other_agents_string)
        )
        async def task(
            description: str,
            subagent_type: str,
            state: Annotated[dict, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ):
            if subagent_type not in agents:
                return f"Error: invoked agent of type {subagent_type}, the only allowed types are {[f'`{k}`' for k in agents]}"
            sub_agent = agents[subagent_type]

            state["messages"] = [{"role": "user", "content": description}]

            # Agent automatically enters tree and calculates level
            result = await sub_agent.ainvoke(state)
            state_update = {}
            for k, v in result.items():
                if k not in ["todos", "messages"]:
                    state_update[k] = v
            return Command(
                update={
                    **state_update,
                    "messages": [
                        ToolMessage(
                            result["messages"][-1].content, tool_call_id=tool_call_id
                        )
                    ],
                }
            )
    else:
        @tool(
            description=TASK_TOOL_DESCRIPTION.format(other_agents=other_agents_string)
        )
        def task(
            description: str,
            subagent_type: str,
            state: Annotated[dict, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ):
            if subagent_type not in agents:
                return f"Error: invoked agent of type {subagent_type}, the only allowed types are {[f'`{k}`' for k in agents]}"
            sub_agent = agents[subagent_type]

            state["messages"] = [{"role": "user", "content": description}]

            # Agent automatically enters tree and calculates level
            result = sub_agent.invoke(state)
            state_update = {}
            for k, v in result.items():
                if k not in ["todos", "messages"]:
                    state_update[k] = v
            return Command(
                update={
                    **state_update,
                    "messages": [
                        ToolMessage(
                            result["messages"][-1].content, tool_call_id=tool_call_id
                        )
                    ],
                }
            )
    return task

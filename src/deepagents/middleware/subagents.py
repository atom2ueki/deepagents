"""Subagent middleware for spawning task-specific subagents."""

from typing import Annotated, TYPE_CHECKING
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, SummarizationMiddleware, TodoListMiddleware
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.tools import BaseTool, tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langchain.tools.tool_node import InjectedState

from deepagents.context import AgentContext

if TYPE_CHECKING:
    from deepagents.agent import Agent
else:
    # Import at runtime for _get_agents (no circular dependency)
    from deepagents.agent import ToolAgent

# Import other middleware
from deepagents.middleware.observer import ObserverMiddleware
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware

BASE_AGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools."

TASK_SYSTEM_PROMPT = """## `task` (subagent spawner)

You have access to a `task` tool to launch short-lived subagents that handle isolated tasks. These agents are ephemeral — they live only for the duration of the task and return a single result.

When to use the task tool:
- When a task is complex and multi-step, and can be fully delegated in isolation
- When a task is independent of other tasks and can run in parallel
- When a task requires focused reasoning or heavy token/context usage that would bloat the orchestrator thread
- When sandboxing improves reliability (e.g. code execution, structured searches, data formatting)
- When you only care about the output of the subagent, and not the intermediate steps (ex. performing a lot of research and then returned a synthesized report, performing a series of computations or lookups to achieve a concise, relevant answer.)

Subagent lifecycle:
1. **Spawn** → Provide clear role, instructions, and expected output
2. **Run** → The subagent completes the task autonomously
3. **Return** → The subagent provides a single structured result
4. **Reconcile** → Incorporate or synthesize the result into the main thread

When NOT to use the task tool:
- If you need to see the intermediate reasoning or steps after the subagent has completed (the task tool hides them)
- If the task is trivial (a few tool calls or simple lookup)
- If delegating does not reduce token usage, complexity, or context switching
- If splitting would add latency without benefit

## Important Task Tool Usage Notes to Remember
- Whenever possible, parallelize the work that you do. This is true for both tool_calls, and for tasks. Whenever you have independent steps to complete - make tool_calls, or kick off tasks (subagents) in parallel to accomplish them faster. This saves time for the user, which is incredibly important.
- Remember to use the `task` tool to silo independent tasks within a multi-part objective.
- You should use the `task` tool whenever you have a complex task that will take multiple steps, and is independent from other tasks that the agent needs to complete. These agents are highly competent and efficient."""

TASK_TOOL_DESCRIPTION = """Launch an ephemeral subagent to handle complex, multi-step independent tasks with isolated context windows.

Available agent types and the tools they have access to:
- general-purpose: General-purpose agent for researching complex questions, searching for files and content, and executing multi-step tasks. When you are searching for a keyword or file and are not confident that you will find the right match in the first few tries use this agent to perform the search for you. This agent has access to all tools as the main agent.
{other_agents}"""


class SubAgentMiddleware(AgentMiddleware):
    """Middleware that provides subagent spawning capabilities."""

    def __init__(
        self,
        default_subagent_tools: list[BaseTool] = [],
        subagents: list["Agent"] = [],
        model=None,
        is_async=False,
        use_longterm_memory: bool = False,
    ) -> None:
        super().__init__()
        task_tool = create_task_tool(
            default_subagent_tools=default_subagent_tools,
            subagents=subagents,
            model=model,
            is_async=is_async,
            use_longterm_memory=use_longterm_memory,
        )
        self.tools = [task_tool]

    def modify_model_request(self, request: ModelRequest, agent_state: AgentState) -> ModelRequest:
        """Add subagent system prompt to the model request."""
        request.system_prompt = request.system_prompt + "\n\n" + TASK_SYSTEM_PROMPT
        return request


def _get_agents(
    default_subagent_tools: list[BaseTool],
    subagents: list["Agent"],
    model,
    use_longterm_memory: bool = False,
):
    """Build agents dict mapping name -> Agent object.

    Creates general-purpose agent and includes all provided subagents.
    """
    # General purpose subagent middleware
    general_purpose_middleware = [
        ObserverMiddleware(),  # Agent name from runtime.context.agent_name
        TodoListMiddleware(),
        FilesystemMiddleware(long_term_memory=use_longterm_memory),
        SummarizationMiddleware(
            model=model,
            max_tokens_before_summary=170000,
            messages_to_keep=6,
        ),
        AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
        PatchToolCallsMiddleware(),
    ]

    # Create general-purpose agent graph
    general_purpose_graph = create_agent(
        model,
        system_prompt=BASE_AGENT_PROMPT,
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
    use_longterm_memory: bool = False,
):
    """Create the task tool for spawning subagents."""
    agents = _get_agents(
        default_subagent_tools, subagents, model, use_longterm_memory
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

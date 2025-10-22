"""Middleware for providing subagents to an agent via a `task` tool."""

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TYPE_CHECKING

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse, TodoListMiddleware
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.tools import BaseTool, ToolRuntime
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command

from deepagents.context import AgentContext
from deepagents.middleware.observer import ObserverMiddleware
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware

if TYPE_CHECKING:
    from deepagents.agent import Agent

BASE_AGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools."

DEFAULT_GENERAL_PURPOSE_DESCRIPTION = "General-purpose agent for researching complex questions, searching for files and content, and executing multi-step tasks. When you are searching for a keyword or file and are not confident that you will find the right match in the first few tries use this agent to perform the search for you. This agent has access to all tools as the main agent."  # noqa: E501

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
{available_agents}"""

# State keys that should be excluded when passing state to subagents
_EXCLUDED_STATE_KEYS = ("messages", "todos")


def _get_subagents(
    default_subagent_tools: Sequence[BaseTool | Callable | dict[str, Any]] | None,
    subagents: list["Agent"],
    model: str | BaseChatModel,
    use_longterm_memory: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """Build agents dict mapping name -> Agent object.

    Creates general-purpose agent and includes all provided subagents.

    Returns:
        Tuple of (agent_dict, description_list) where agent_dict maps agent names
        to graph instances and description_list contains formatted descriptions.
    """
    from deepagents.agent import ToolAgent

    # General purpose subagent middleware
    # Note: Subagents use shortterm memory only (no store access)
    general_purpose_middleware = [
        ObserverMiddleware(),
        TodoListMiddleware(),
        FilesystemMiddleware(long_term_memory=False),  # Subagents don't have store access
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
        tools=default_subagent_tools or [],
        checkpointer=False,
        context_schema=AgentContext,
        middleware=general_purpose_middleware,
    )

    # Wrap in ToolAgent for consistent interface
    agents: dict[str, Any] = {
        "general-purpose": ToolAgent(
            name="general-purpose",
            description="General purpose agent for complex tasks",
            graph=general_purpose_graph,
            fg_color="#000000",
            bg_color="#ffffff",
        )
    }

    # Add all subagents (must be Agent objects in OOP fork)
    subagent_descriptions = []
    for agent in subagents:
        agents[agent.name] = agent
        subagent_descriptions.append(f"- {agent.name}: {agent.description}")

    # Add general-purpose agent description
    subagent_descriptions.insert(0, f"- general-purpose: {DEFAULT_GENERAL_PURPOSE_DESCRIPTION}")

    return agents, subagent_descriptions


def _create_task_tool(
    default_subagent_tools: Sequence[BaseTool | Callable | dict[str, Any]] | None,
    subagents: list["Agent"],
    model: str | BaseChatModel,
    use_longterm_memory: bool = False,
) -> BaseTool:
    """Create the task tool for spawning subagents."""
    agents, subagent_descriptions = _get_subagents(
        default_subagent_tools, subagents, model, use_longterm_memory
    )
    available_agents_string = "\n".join(subagent_descriptions)

    def _return_command_with_state_update(result: dict, tool_call_id: str) -> Command:
        """Helper to create Command with state update."""
        state_update = {k: v for k, v in result.items() if k not in _EXCLUDED_STATE_KEYS}
        return Command(
            update={
                **state_update,
                "messages": [ToolMessage(result["messages"][-1].content, tool_call_id=tool_call_id)],
            }
        )

    def _validate_and_prepare_state(
        subagent_type: str, description: str, runtime: ToolRuntime
    ) -> tuple[Any, dict]:
        """Validate subagent type and prepare state for invocation."""
        if subagent_type not in agents:
            msg = f"Error: invoked agent of type {subagent_type}, the only allowed types are {[f'`{k}`' for k in agents]}"
            raise ValueError(msg)
        subagent = agents[subagent_type]
        # Create a new state dict to avoid mutating the original
        subagent_state = {k: v for k, v in runtime.state.items() if k not in _EXCLUDED_STATE_KEYS}
        subagent_state["messages"] = [HumanMessage(content=description)]
        return subagent, subagent_state

    def task(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        """Launch an ephemeral subagent to handle a complex task (sync)."""
        subagent, subagent_state = _validate_and_prepare_state(subagent_type, description, runtime)
        result = subagent.invoke(subagent_state)
        if not runtime.tool_call_id:
            raise ValueError("Tool call ID is required for subagent invocation")
        return _return_command_with_state_update(result, runtime.tool_call_id)

    async def atask(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        """Launch an ephemeral subagent to handle a complex task (async)."""
        subagent, subagent_state = _validate_and_prepare_state(subagent_type, description, runtime)
        result = await subagent.ainvoke(subagent_state)
        if not runtime.tool_call_id:
            raise ValueError("Tool call ID is required for subagent invocation")
        return _return_command_with_state_update(result, runtime.tool_call_id)

    return StructuredTool.from_function(
        name="task",
        func=task,
        coroutine=atask,
        description=TASK_TOOL_DESCRIPTION.format(available_agents=available_agents_string),
    )


class SubAgentMiddleware(AgentMiddleware):
    """Middleware for providing subagents to an agent via a `task` tool.

    This middleware adds a `task` tool to the agent that can be used to invoke subagents.
    Subagents are useful for handling complex tasks that require multiple steps, or tasks
    that require a lot of context to resolve.

    A chief benefit of subagents is that they can handle multi-step tasks, and then return
    a clean, concise response to the main agent.

    Args:
        default_subagent_tools: The tools to use for the default general-purpose subagent.
        subagents: A list of Agent instances this agent can delegate to.
        model: The model to use for subagents.
        use_longterm_memory: Whether to enable long-term memory for subagents.
    """

    def __init__(
        self,
        default_model: str | BaseChatModel,
        default_tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
        subagents: list["Agent"] | None = None,
        use_longterm_memory: bool = False,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize the SubAgentMiddleware."""
        super().__init__()
        self.system_prompt = system_prompt if system_prompt is not None else TASK_SYSTEM_PROMPT
        task_tool = _create_task_tool(
            default_subagent_tools=default_tools,
            subagents=subagents or [],
            model=default_model,
            use_longterm_memory=use_longterm_memory,
        )
        self.tools = [task_tool]

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Update the system prompt to include instructions on using subagents."""
        if self.system_prompt is not None:
            request.system_prompt = (
                request.system_prompt + "\n\n" + self.system_prompt
                if request.system_prompt
                else self.system_prompt
            )
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Update the system prompt to include instructions on using subagents."""
        if self.system_prompt is not None:
            request.system_prompt = (
                request.system_prompt + "\n\n" + self.system_prompt
                if request.system_prompt
                else self.system_prompt
            )
        return await handler(request)

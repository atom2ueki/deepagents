"""Deepagents come with planning, filesystem, and subagents."""

from collections.abc import Callable, Sequence
from typing import Any, TYPE_CHECKING

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
    TodoListMiddleware,
)
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.agents.structured_output import ResponseFormat
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.cache.base import BaseCache
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from deepagents.context import AgentContext
from deepagents.middleware import (
    ObserverMiddleware,
    FilesystemMiddleware,
    SubAgentMiddleware,
    PatchToolCallsMiddleware,
)

if TYPE_CHECKING:
    from deepagents.agent import Agent, ToolAgent

BASE_AGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools."


def get_default_model() -> ChatAnthropic:
    """Get the default model for deep agents.

    Returns:
        ChatAnthropic instance configured with Claude Sonnet 4.
    """
    return ChatAnthropic(  # pyright: ignore[reportCallIssue]  # Pydantic field aliases
        model_name="claude-sonnet-4-5-20250929",
        max_tokens=20000,  # pyright: ignore[reportCallIssue]  # aliased to max_tokens_to_sample
    )


def create_deep_agent(
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: list["Agent"] | None = None,
    response_format: ResponseFormat | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    use_longterm_memory: bool = False,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    handle_tool_errors: bool = True,
    debug: bool = False,
    name: str | None = None,
    description: str = "",
    fg_color: str = "#000000",
    bg_color: str = "#ffffff",
    cache: BaseCache | None = None,
) -> "ToolAgent":
    """Create a deep agent.

    This agent will by default have access to a tool to write todos (write_todos),
    four file editing tools: write_file, ls, read_file, edit_file, and a tool to call
    subagents.

    Args:
        model: The model to use.
        tools: The tools the agent should have access to.
        system_prompt: The additional instructions the agent should have. Will go in
            the system prompt.
        middleware: Additional middleware to apply after standard middleware.
        subagents: List of Agent instances this agent can delegate to.
        response_format: A structured output response format to use for the agent.
        context_schema: The schema of the deep agent.
        checkpointer: Optional checkpointer for persisting agent state between runs.
        store: Optional store for persisting longterm memories.
        use_longterm_memory: Whether to use longterm memory - you must provide a store
            in order to use longterm memory.
        interrupt_on: Optional Dict[str, bool | InterruptOnConfig] mapping tool names to
            interrupt configs.
        handle_tool_errors: If True, tool errors become messages instead of exceptions.
        debug: Whether to enable debug mode. Passed through to create_agent.
        name: The name of the agent (used for event emissions). Defaults to "main".
        description: Agent description (used by parent agents for delegation).
        fg_color: Foreground color for UI events.
        bg_color: Background color for UI events.
        cache: The cache to use for the agent. Passed through to create_agent.

    Returns:
        A configured deep agent (ToolAgent instance).
    """
    from deepagents.agent import ToolAgent

    if model is None:
        model = get_default_model()

    # Wrap tools with error handling if requested
    if handle_tool_errors and tools:
        wrapped_tools = []
        for tool in tools:
            if isinstance(tool, BaseTool):
                tool.handle_tool_error = True
                wrapped_tools.append(tool)
            else:
                wrapped_tools.append(tool)
        tools = wrapped_tools

    deepagent_middleware = [
        ObserverMiddleware(),  # Custom middleware for UI event emissions
        TodoListMiddleware(),
        FilesystemMiddleware(long_term_memory=use_longterm_memory),
        SubAgentMiddleware(
            default_model=model,
            default_tools=tools,
            subagents=subagents if subagents is not None else [],
            use_longterm_memory=use_longterm_memory,
        ),
        SummarizationMiddleware(
            model=model,
            max_tokens_before_summary=170000,
            messages_to_keep=6,
        ),
        AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
        PatchToolCallsMiddleware(),
    ]

    if interrupt_on is not None:
        deepagent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    if middleware:
        deepagent_middleware.extend(middleware)

    graph = create_agent(
        model,
        system_prompt=(
            system_prompt + "\n\n" + BASE_AGENT_PROMPT
            if system_prompt
            else BASE_AGENT_PROMPT
        ),
        tools=tools,
        middleware=deepagent_middleware,
        response_format=response_format,
        context_schema=context_schema or AgentContext,
        checkpointer=checkpointer,
        store=store,
        debug=debug,
        name=name,
        cache=cache,
    ).with_config({"recursion_limit": 1000})

    # Wrap in ToolAgent for OOP interface with auto-context injection
    agent = ToolAgent(
        name=name or "main",
        description=description,
        graph=graph,
        fg_color=fg_color,
        bg_color=bg_color,
    )
    agent.subagents = subagents if subagents is not None else []
    return agent

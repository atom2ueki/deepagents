from typing import Sequence, Union, Callable, Any, Type, Optional, TYPE_CHECKING
from langchain_core.tools import BaseTool
from langchain_core.language_models import LanguageModelLike
from langgraph.types import Checkpointer
from langgraph.store.base import BaseStore
from langchain.agents import create_agent as langchain_create_agent
from langchain.agents.middleware import AgentMiddleware, SummarizationMiddleware, HumanInTheLoopMiddleware, InterruptOnConfig, TodoListMiddleware
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from deepagents.middleware import ObserverMiddleware, FilesystemMiddleware, SubAgentMiddleware, PatchToolCallsMiddleware
from deepagents.context import AgentContext

BASE_AGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools."

def get_default_model() -> ChatAnthropic:
    """Get the default model for deep agents.

    Returns:
        ChatAnthropic instance configured with Claude Sonnet 4.5.
    """
    return ChatAnthropic(model_name="claude-sonnet-4-5-20250929", max_tokens=20000)

if TYPE_CHECKING:
    from deepagents.agent import Agent

from deepagents.agent import ToolAgent

def agent_builder(
    tools: Sequence[Union[BaseTool, Callable, dict[str, Any]]],
    instructions: str,
    name: str,
    description: str,
    fg_color: str,
    bg_color: str,
    middleware: Optional[list[AgentMiddleware]] = None,
    tool_configs: Optional[dict[str, bool | InterruptOnConfig]] = None,
    model: Optional[Union[str, LanguageModelLike]] = None,
    subagents: Optional[list["Agent"]] = None,
    context_schema: Optional[Type[Any]] = None,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    use_longterm_memory: bool = False,
    is_async: bool = False,
    handle_tool_errors: bool = True,
) -> "ToolAgent":
    if model is None:
        model = get_default_model()

    deepagent_middleware = [
        ObserverMiddleware(),  # Agent name from runtime.context.agent_name
        TodoListMiddleware(),
        FilesystemMiddleware(long_term_memory=use_longterm_memory),
        SubAgentMiddleware(
            default_subagent_tools=tools,   # NOTE: These tools are piped to the general-purpose subagent.
            subagents=subagents if subagents is not None else [],
            model=model,
            is_async=is_async,
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
    # Add tool interrupt config if provided
    if tool_configs is not None:
        deepagent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=tool_configs))

    if middleware is not None:
        deepagent_middleware.extend(middleware)

    # Wrap tools with error handling if requested
    if handle_tool_errors:
        wrapped_tools = []
        for tool in tools:
            if isinstance(tool, BaseTool):
                # Set handle_tool_error on BaseTool instances
                tool.handle_tool_error = True
                wrapped_tools.append(tool)
            else:
                wrapped_tools.append(tool)
        tools = wrapped_tools

    graph = langchain_create_agent(
        model,
        system_prompt=instructions + "\n\n" + BASE_AGENT_PROMPT if instructions else BASE_AGENT_PROMPT,
        tools=tools,
        middleware=deepagent_middleware,
        context_schema=context_schema or AgentContext,
        checkpointer=checkpointer,
        store=store,
    )

    # Wrap in ToolAgent for OOP interface with auto-context injection
    agent = ToolAgent(
        name=name,
        description=description,
        graph=graph,
        fg_color=fg_color,
        bg_color=bg_color,
    )
    # Store subagents for access (e.g., registering colors)
    agent.subagents = subagents if subagents is not None else []
    return agent

def create_deep_agent(
    tools: Sequence[Union[BaseTool, Callable, dict[str, Any]]] = [],
    instructions: str = "",
    name: str = "main",
    description: str = "",
    fg_color: str = "#000000",
    bg_color: str = "#ffffff",
    middleware: Optional[list[AgentMiddleware]] = None,
    model: Optional[Union[str, LanguageModelLike]] = None,
    subagents: Optional[list["Agent"]] = None,
    context_schema: Optional[Type[Any]] = None,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    use_longterm_memory: bool = False,
    tool_configs: Optional[dict[str, bool | InterruptOnConfig]] = None,
    handle_tool_errors: bool = True,
) -> "ToolAgent":
    """Create a ToolAgent (agent built from tools and instructions).

    Returns a ToolAgent instance with auto-context injection.

    Args:
        tools: Tools available to the agent
        instructions: System prompt/instructions for the agent
        name: Agent name (used for event emissions)
        description: Agent description (used by parent agents for delegation)
        model: Model to use
        subagents: List of Agent instances this agent can delegate to
        context_schema: Custom context schema
        checkpointer: Optional checkpointer for persistence
        store: Optional store for persistent long-term memory
        use_longterm_memory: Whether to enable long-term memory (requires store)
        tool_configs: Tool interrupt configurations
        handle_tool_errors: If True, tool errors become messages instead of exceptions

    Returns:
        ToolAgent instance with auto-context injection
    """
    return agent_builder(
        tools=tools,
        instructions=instructions,
        name=name,
        description=description,
        fg_color=fg_color,
        bg_color=bg_color,
        middleware=middleware,
        model=model,
        subagents=subagents,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        use_longterm_memory=use_longterm_memory,
        tool_configs=tool_configs,
        is_async=False,
        handle_tool_errors=handle_tool_errors,
    )

def async_create_deep_agent(
    tools: Sequence[Union[BaseTool, Callable, dict[str, Any]]] = [],
    instructions: str = "",
    name: str = "main",
    description: str = "",
    fg_color: str = "#000000",
    bg_color: str = "#ffffff",
    middleware: Optional[list[AgentMiddleware]] = None,
    model: Optional[Union[str, LanguageModelLike]] = None,
    subagents: Optional[list["Agent"]] = None,
    context_schema: Optional[Type[Any]] = None,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    use_longterm_memory: bool = False,
    tool_configs: Optional[dict[str, bool | InterruptOnConfig]] = None,
    handle_tool_errors: bool = True,
) -> "ToolAgent":
    """Create a ToolAgent (async version).

    Returns a ToolAgent instance with auto-context injection.

    Args:
        tools: Tools available to the agent
        instructions: System prompt/instructions for the agent
        name: Agent name (used for event emissions)
        description: Agent description (used by parent agents for delegation)
        model: Model to use
        subagents: List of Agent instances this agent can delegate to
        context_schema: Custom context schema
        checkpointer: Optional checkpointer for persistence
        store: Optional store for persistent long-term memory
        use_longterm_memory: Whether to enable long-term memory (requires store)
        tool_configs: Tool interrupt configurations
        handle_tool_errors: If True, tool errors become messages instead of exceptions

    Returns:
        ToolAgent instance with auto-context injection
    """
    return agent_builder(
        tools=tools,
        instructions=instructions,
        name=name,
        description=description,
        fg_color=fg_color,
        bg_color=bg_color,
        middleware=middleware,
        model=model,
        subagents=subagents,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        use_longterm_memory=use_longterm_memory,
        tool_configs=tool_configs,
        is_async=True,
        handle_tool_errors=handle_tool_errors,
    )
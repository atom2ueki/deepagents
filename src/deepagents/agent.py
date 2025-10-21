"""Agent abstraction layer for DeepAgents.

Provides a clean OOP interface where all agents (root or nested) are Agent instances.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, Union, Callable, Type
from langchain_core.tools import BaseTool
from langchain_core.language_models import LanguageModelLike
from langgraph.types import Checkpointer
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.human_in_the_loop import ToolConfig

from deepagents.context import AgentContext
from deepagents.tree import get_agent_tree

class Agent(ABC):
    """Abstract base class for all agents (root or nested).

    All agents share the same interface - the only difference is whether
    they have subagents (tree relationship, not type difference).
    """

    def __init__(
        self,
        name: str,
        description: str,
        fg_color: str,
        bg_color: str,
    ):
        """Initialize agent with name and description.

        Args:
            name: Agent name (used for event emissions and routing)
            description: Agent description (used by parent agents for delegation)
            fg_color: Foreground color for agent label
            bg_color: Background color for agent label
        """
        self.name = name
        self.description = description
        self.fg_color = fg_color
        self.bg_color = bg_color

    @abstractmethod
    async def ainvoke(self, state: dict, config: Optional[dict] = None, **kwargs) -> dict:
        """Invoke agent asynchronously with auto-context injection.

        Args:
            state: Input state
            config: Optional runtime config
            **kwargs: Additional arguments

        Returns:
            Output state
        """
        pass

    @abstractmethod
    def invoke(self, state: dict, config: Optional[dict] = None, **kwargs) -> dict:
        """Invoke agent synchronously with auto-context injection.

        Args:
            state: Input state
            config: Optional runtime config
            **kwargs: Additional arguments

        Returns:
            Output state
        """
        pass


class ToolAgent(Agent):
    """Agent built from tools, instructions, and model.

    This is the standard agent created by create_deep_agent().
    """

    def __init__(
        self,
        name: str,
        description: str,
        graph,  # The compiled LangGraph agent
        fg_color: str,
        bg_color: str,
    ):
        """Initialize ToolAgent.

        Args:
            name: Agent name
            description: Agent description
            graph: Compiled LangGraph agent graph
            fg_color: Foreground color for agent label
            bg_color: Background color for agent label
        """
        super().__init__(name, description, fg_color, bg_color)
        self._graph = graph

    async def ainvoke(self, state: dict, config: Optional[dict] = None, **kwargs) -> dict:
        """Invoke agent with auto-context injection."""
        config = config or {}
        tree = get_agent_tree()

        # Auto-inject context as kwarg if not provided (for Runtime.context in LangGraph)
        if "context" not in kwargs:
            # Enter tree and get level automatically
            level = tree.enter_agent(self.name, self.fg_color, self.bg_color)

            kwargs["context"] = AgentContext(
                agent_name=self.name,
                agent_fg_color=self.fg_color,
                agent_bg_color=self.bg_color,
                agent_level=level,
            )

        try:
            return await self._graph.ainvoke(state, config, **kwargs)
        finally:
            # Exit tree when done
            tree.exit_agent()

    def invoke(self, state: dict, config: Optional[dict] = None, **kwargs) -> dict:
        """Invoke agent with auto-context injection."""
        config = config or {}
        tree = get_agent_tree()

        # Auto-inject context as kwarg if not provided (for Runtime.context in LangGraph)
        if "context" not in kwargs:
            # Enter tree and get level automatically
            level = tree.enter_agent(self.name, self.fg_color, self.bg_color)

            kwargs["context"] = AgentContext(
                agent_name=self.name,
                agent_fg_color=self.fg_color,
                agent_bg_color=self.bg_color,
                agent_level=level,
            )

        try:
            return self._graph.invoke(state, config, **kwargs)
        finally:
            # Exit tree when done
            tree.exit_agent()

    def with_config(self, config: dict) -> "ToolAgent":
        """Return a new ToolAgent with updated config (delegates to graph)."""
        new_agent = ToolAgent(
            name=self.name,
            description=self.description,
            graph=self._graph.with_config(config),
            fg_color=self.fg_color,
            bg_color=self.bg_color,
        )
        return new_agent


class CustomAgent(Agent):
    """Agent with pre-built custom graph.

    Used when you have a custom LangGraph workflow (e.g., vision_agent).
    """

    def __init__(
        self,
        name: str,
        description: str,
        graph,  # Pre-built compiled graph
        fg_color: str,
        bg_color: str,
    ):
        """Initialize CustomAgent.

        Args:
            name: Agent name
            description: Agent description
            graph: Pre-built compiled LangGraph graph
            fg_color: Foreground color for agent label
            bg_color: Background color for agent label
        """
        super().__init__(name, description, fg_color, bg_color)
        self._graph = graph

    async def ainvoke(self, state: dict, config: Optional[dict] = None, **kwargs) -> dict:
        """Invoke agent with auto-context injection."""
        config = config or {}
        tree = get_agent_tree()

        # Auto-inject context into config["configurable"] if not provided
        if "configurable" not in config:
            config["configurable"] = {}
        if "context" not in config["configurable"]:
            # Enter tree and get level automatically
            level = tree.enter_agent(self.name, self.fg_color, self.bg_color)

            config["configurable"]["context"] = AgentContext(
                agent_name=self.name,
                agent_fg_color=self.fg_color,
                agent_bg_color=self.bg_color,
                agent_level=level,
            )

        try:
            return await self._graph.ainvoke(state, config, **kwargs)
        finally:
            # Exit tree when done
            tree.exit_agent()

    def invoke(self, state: dict, config: Optional[dict] = None, **kwargs) -> dict:
        """Invoke agent with auto-context injection."""
        config = config or {}
        tree = get_agent_tree()

        # Auto-inject context into config["configurable"] if not provided
        if "configurable" not in config:
            config["configurable"] = {}
        if "context" not in config["configurable"]:
            # Enter tree and get level automatically
            level = tree.enter_agent(self.name, self.fg_color, self.bg_color)

            config["configurable"]["context"] = AgentContext(
                agent_name=self.name,
                agent_fg_color=self.fg_color,
                agent_bg_color=self.bg_color,
                agent_level=level,
            )

        try:
            return self._graph.invoke(state, config, **kwargs)
        finally:
            # Exit tree when done
            tree.exit_agent()

    def with_config(self, config: dict) -> "CustomAgent":
        """Return a new CustomAgent with updated config (delegates to graph)."""
        new_agent = CustomAgent(
            name=self.name,
            description=self.description,
            graph=self._graph.with_config(config),
            fg_color=self.fg_color,
            bg_color=self.bg_color,
        )
        return new_agent

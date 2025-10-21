"""Runtime context for DeepAgents execution.

Provides agent-scoped information accessible to all middleware and nodes.
"""

from dataclasses import dataclass


@dataclass
class AgentContext:
    """Runtime context for agent execution.

    Attributes:
        agent_name: Name of the currently executing agent (e.g., "main", "vision_agent")
        agent_fg_color: Foreground color for agent display
        agent_bg_color: Background color for agent display
        agent_level: Nesting level in agent hierarchy (0=root, 1=first subagent, etc.)
    """

    agent_name: str = "main"
    agent_fg_color: str = "#000000"
    agent_bg_color: str = "#ffffff"
    agent_level: int = 0

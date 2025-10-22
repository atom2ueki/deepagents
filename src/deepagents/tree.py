"""Agent execution tree for tracking agent hierarchy.

Uses parent-pointer tree pattern to automatically track agent nesting depth
without manual level calculation.
"""

from typing import Optional, List
from contextvars import ContextVar


class AgentNode:
    """Node in the agent execution tree.

    Represents a single agent invocation with automatic depth tracking
    through parent-child relationships.
    """

    def __init__(self, name: str, fg_color: str, bg_color: str):
        """Initialize agent node.

        Args:
            name: Agent name
            fg_color: Foreground color for display
            bg_color: Background color for display
        """
        self.name = name
        self.fg_color = fg_color
        self.bg_color = bg_color
        self.parent: Optional[AgentNode] = None
        self.children: List[AgentNode] = []

    @property
    def level(self) -> int:
        """Calculate depth level by walking up parent chain.

        Returns:
            0 for root, 1 for direct children, 2 for grandchildren, etc.
        """
        level = 0
        node = self.parent
        while node:
            level += 1
            node = node.parent
        return level


class AgentTree:
    """Manages agent execution tree with current pointer pattern.

    Tracks the currently executing agent and maintains the call hierarchy
    without explicit level tracking or stack management.
    """

    def __init__(self):
        """Initialize empty agent tree."""
        self._root: Optional[AgentNode] = None
        self._current: Optional[AgentNode] = None

    def enter_agent(self, name: str, fg_color: str, bg_color: str) -> int:
        """Enter a new agent context.

        Creates a new agent node and makes it the current agent.
        If there's already a current agent, the new node becomes its child.

        Args:
            name: Agent name
            fg_color: Foreground color
            bg_color: Background color

        Returns:
            Level of the new agent (0 for root, 1 for subagent, etc.)
        """
        node = AgentNode(name, fg_color, bg_color)

        if self._root is None:
            # First agent - becomes root
            self._root = node
            self._current = node
        elif self._current is not None:
            # Child of current agent
            node.parent = self._current
            self._current.children.append(node)
            self._current = node  # Move pointer to new child

        return node.level

    def exit_agent(self) -> Optional[int]:
        """Exit current agent context.

        Moves the current pointer back to the parent agent.

        Returns:
            Level of the parent agent, or None if at root
        """
        if self._current and self._current.parent:
            self._current = self._current.parent
            return self._current.level
        return None

    @property
    def current_level(self) -> int:
        """Get current agent's level.

        Returns:
            Current depth level (0 for root)
        """
        return self._current.level if self._current else 0

    @property
    def current_agent(self) -> Optional[AgentNode]:
        """Get current agent node.

        Returns:
            Current agent node or None if tree is empty
        """
        return self._current

    def reset(self):
        """Reset the tree to empty state."""
        self._root = None
        self._current = None


# Context-local agent tree instance (thread-safe and async-safe)
_agent_tree_ctx: ContextVar[AgentTree] = ContextVar('_agent_tree_ctx')


def get_agent_tree() -> AgentTree:
    """Get the context-local agent tree instance.

    Each async task or thread gets its own isolated tree for thread-safety.

    Returns:
        Context-local AgentTree instance
    """
    tree = _agent_tree_ctx.get(None)
    if tree is None:
        tree = AgentTree()
        _agent_tree_ctx.set(tree)
    return tree

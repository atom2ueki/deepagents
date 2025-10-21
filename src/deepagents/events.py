"""Event system for DeepAgents."""

from typing import Callable, Any


class EventBus:
    """Global event bus for agent events."""

    def __init__(self):
        self._listeners = []

    def subscribe(self, listener: Callable[[str, str, str, str, int, Any], None]):
        """Subscribe to agent events.

        Args:
            listener: Callback function(event_type, agent_name, agent_fg_color, agent_bg_color, agent_level, data)
                event_type: "message", "todos", "files"
                agent_name: Name of the agent ("main", "dom_agent", "vision_agent", etc.)
                agent_fg_color: Foreground color for the agent
                agent_bg_color: Background color for the agent
                agent_level: Nesting level in agent hierarchy (0=root, 1=subagent, etc.)
                data: Event data (message object, todos list, files dict, etc.)
        """
        self._listeners.append(listener)

    def unsubscribe(self, listener: Callable[[str, str, str, str, int, Any], None]):
        """Unsubscribe from agent events."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def emit(self, event_type: str, agent_name: str, agent_fg_color: str, agent_bg_color: str, agent_level: int, data: Any):
        """Emit an event to all listeners."""
        for listener in self._listeners:
            try:
                listener(event_type, agent_name, agent_fg_color, agent_bg_color, agent_level, data)
            except Exception:
                # Silently ignore listener errors
                pass

    def clear(self):
        """Clear all listeners."""
        self._listeners.clear()


# Global event bus instance
_event_bus = EventBus()


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    return _event_bus

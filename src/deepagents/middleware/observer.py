"""Observer middleware for emitting agent events."""

from deepagents.events import get_event_bus
from deepagents.context import AgentContext
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime


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

"""Observer middleware for emitting agent events."""

from typing import Any
from collections import deque
from deepagents.events import get_event_bus
from deepagents.context import AgentContext
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime


class ObserverMiddleware(AgentMiddleware[AgentState, AgentContext]):
    """Middleware that emits all agent events to the event bus.

    Emits events for:
    - Messages: Deduplicated by message ID (tracks last 1000 IDs)
    - Todos: Only emitted when changed

    Agent name is determined from runtime.context.agent_name (no hardcoding).

    Memory-bounded: Uses deque with maxlen=1000 for message IDs to prevent memory leaks.
    """

    def __init__(self):
        super().__init__()
        # Track last 1000 emitted message IDs (bounded to prevent memory leak)
        self._emitted_message_ids: deque[str] = deque(maxlen=1000)
        # Track last todos to only emit when changed
        self._last_todos = None

    def after_model(
        self, state: AgentState, runtime: Runtime[AgentContext]
    ) -> dict | None:
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

        # Emit NEW messages only (track by message ID)
        if "messages" in state and state["messages"]:
            for message in state["messages"]:
                # Each LangChain message has a unique ID
                msg_id = message.id
                if msg_id and msg_id not in self._emitted_message_ids:
                    event_bus.emit(
                        "message",
                        agent_name,
                        agent_fg_color,
                        agent_bg_color,
                        agent_level,
                        message,
                    )
                    self._emitted_message_ids.append(msg_id)

        # Emit todos only when changed
        if "todos" in state:
            current_todos = state["todos"]
            if current_todos != self._last_todos:
                event_bus.emit(
                    "todos",
                    agent_name,
                    agent_fg_color,
                    agent_bg_color,
                    agent_level,
                    current_todos,
                )
                self._last_todos = current_todos

        return None

"""Observer middleware for emitting agent events."""

import json
import sys
from typing import Any, Optional
from collections import deque
from datetime import datetime
from deepagents.events import get_event_bus
from deepagents.context import AgentContext
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime


class ObserverMiddleware(AgentMiddleware[AgentState, AgentContext]):
    """Middleware that emits all agent events to the event bus and JSON stdout.

    Emits events for:
    - Messages: Deduplicated by message ID (tracks last 1000 IDs)
    - Todos: Only emitted when changed

    Also emits structured JSON events to stdout for CLI consumption:
    - agent_change: When switching between agents
    - thinking: AI content/reasoning
    - tool_call: Tool invocations
    - tool_result: Tool execution results
    - completion: Agent completion messages
    - todos: Todo list updates

    Agent name is determined from runtime.context.agent_name (no hardcoding).

    Memory-bounded: Uses deque with maxlen=1000 for message IDs to prevent memory leaks.
    """

    def __init__(self):
        super().__init__()
        # Track last 1000 emitted message IDs (bounded to prevent memory leak)
        self._emitted_message_ids: deque[str] = deque(maxlen=1000)
        # Track last todos to only emit when changed
        self._last_todos = None
        # Track current agent for change detection (name + level)
        self._current_agent: Optional[tuple[str, int]] = None

    def _emit_json(
        self,
        event_type: str,
        agent_name: str,
        agent_level: int,
        agent_fg_color: str,
        agent_bg_color: str,
        data: dict,
    ):
        """Emit a JSON event to stdout."""
        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "agent_level": agent_level,
            "agent_fg_color": agent_fg_color,
            "agent_bg_color": agent_bg_color,
            **data,
        }
        sys.stdout.write(json.dumps(event) + "\n")
        sys.stdout.flush()

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

        # Emit agent change event (track both name and level)
        current_state = (agent_name, agent_level)
        if current_state != self._current_agent:
            self._emit_json(
                "agent_change",
                agent_name,
                agent_level,
                agent_fg_color,
                agent_bg_color,
                {"new_agent": agent_name},
            )
            self._current_agent = current_state

        # Emit NEW messages only (track by message ID)
        if "messages" in state and state["messages"]:
            # Track if we've seen subagent completion in this batch
            seen_task_completion = False

            for message in state["messages"]:
                # Each LangChain message has a unique ID
                msg_id = message.id
                if msg_id and msg_id not in self._emitted_message_ids:
                    # Check if this is a task tool_result (subagent completion)
                    msg_type = getattr(message, "type", None)
                    if msg_type == "tool":
                        tool_name = getattr(message, "name", "unknown")
                        if tool_name == "task":
                            seen_task_completion = True
                            # Reset agent tracking so next after_model detects parent
                            self._current_agent = None

                    # After seeing task completion, subsequent messages are from parent
                    # We can't change runtime.context, but we don't emit those messages yet
                    # They'll be emitted in the next after_model() call with correct context
                    if seen_task_completion and msg_type != "tool":
                        # Skip this message, DON'T mark as emitted so it processes next time
                        continue

                    # Mark as emitted
                    self._emitted_message_ids.append(msg_id)

                    # Emit to event bus (for backward compatibility)
                    event_bus.emit(
                        "message",
                        agent_name,
                        agent_fg_color,
                        agent_bg_color,
                        agent_level,
                        message,
                    )

                    # Emit JSON events to stdout
                    self._handle_message_json(
                        message, agent_name, agent_level, agent_fg_color, agent_bg_color
                    )

        # Emit todos only when changed
        if "todos" in state:
            current_todos = state["todos"]
            if current_todos != self._last_todos:
                # Emit to event bus
                event_bus.emit(
                    "todos",
                    agent_name,
                    agent_fg_color,
                    agent_bg_color,
                    agent_level,
                    current_todos,
                )
                # Emit JSON to stdout
                self._emit_json(
                    "todos",
                    agent_name,
                    agent_level,
                    agent_fg_color,
                    agent_bg_color,
                    {"todos": current_todos if isinstance(current_todos, list) else []},
                )
                self._last_todos = current_todos

        return None

    def _handle_message_json(
        self, message: Any, agent_name: str, agent_level: int,
        agent_fg_color: str, agent_bg_color: str
    ):
        """Parse message and emit JSON events."""
        # Get message type
        msg_type = getattr(message, "type", None)

        # Skip human messages
        if msg_type == "human":
            return

        # Handle AI messages
        if msg_type == "ai":
            metadata = getattr(message, "metadata", {}) or {}
            msg_subtype = metadata.get("type")

            # Completion - emit and return (don't also emit as thinking)
            if msg_subtype == "completion":
                self._emit_json(
                    "completion",
                    agent_name,
                    agent_level,
                    agent_fg_color,
                    agent_bg_color,
                    {
                        "status": metadata.get("status", "success"),
                        "message": getattr(message, "content", ""),
                    },
                )
                return

            # Regular thinking - emit if has content
            content = getattr(message, "content", "")
            if content and isinstance(content, str):
                self._emit_json(
                    "thinking",
                    agent_name,
                    agent_level,
                    agent_fg_color,
                    agent_bg_color,
                    {"content": content},
                )

            # Tool calls
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    # Handle both dict and object formats
                    if isinstance(tc, dict):
                        tool_name = tc.get("name", "unknown")
                        tool_call_id = tc.get("id", "")
                        args = tc.get("args", {})
                    else:
                        tool_name = getattr(tc, "name", "unknown")
                        tool_call_id = getattr(tc, "id", "")
                        args = getattr(tc, "args", {})

                    self._emit_json(
                        "tool_call",
                        agent_name,
                        agent_level,
                        agent_fg_color,
                        agent_bg_color,
                        {
                            "tool_name": tool_name,
                            "tool_call_id": tool_call_id,
                            "args": args,
                        },
                    )

        # Handle tool results
        elif msg_type == "tool":
            tool_name = getattr(message, "name", "unknown")
            self._emit_json(
                "tool_result",
                agent_name,
                agent_level,
                agent_fg_color,
                agent_bg_color,
                {
                    "tool_name": tool_name,
                    "tool_call_id": getattr(message, "tool_call_id", ""),
                    "success": True,
                },
            )

            # If this is a subagent delegation result, force reset
            # This ensures next after_model() call will detect parent agent change
            if tool_name == "task":
                self._current_agent = None

# 🧠🤖Deep Agents (OOP Fork)

> **Note**: This is a modified fork of [langchain-ai/deepagents](https://github.com/langchain-ai/deepagents) that adds an **object-oriented agent abstraction layer** with event emission for UI integration.

Using an LLM to call tools in a loop is the simplest form of an agent.
This architecture, however, can yield agents that are "shallow" and fail to plan and act over longer, more complex tasks.
Applications like "Deep Research", "Manus", and "Claude Code" have gotten around this limitation by implementing a combination of four things:
a **planning tool**, **sub agents**, access to a **file system**, and a **detailed prompt**.

<img src="deep_agents.png" alt="deep agent" width="600"/>

`deepagents` is a Python package that implements these in a general purpose way so that you can easily create a Deep Agent for your application.

**Acknowledgements: This project was primarily inspired by Claude Code, and initially was largely an attempt to see what made Claude Code general purpose, and make it even more so.**

## Key Features

This version includes:

### 🎯 **OOP Agent Abstraction**
- **Agent hierarchy**: `Agent` (base) → `ToolAgent` / `CustomAgent` (implementations)
- Agents are **objects** instead of raw LangGraph graphs
- Consistent interface: `agent.invoke()`, `agent.ainvoke()`, `agent.stream()`
- Built-in metadata: `name`, `description`, `fg_color`, `bg_color`

### 📡 **Event System**
- Real-time event emission via `ObserverMiddleware`
- Event bus for UI integration: `get_event_bus()`
- Events include: messages, todos, agent hierarchy (tree levels)
- Perfect for building interactive UIs on top of deep agents

### 🎨 **Visual Agent Tracking**
- Each agent has customizable colors (`fg_color`, `bg_color`)
- Automatic tree level tracking for nested subagents
- `AgentContext` passed through execution for debugging

### 🔧 **Simple, Expressive API**
```python
agent = create_deep_agent(
    tools=[tool1, tool2],
    instructions="Your system prompt here...",
    name="my-agent",
    description="What this agent does",
    subagents=[other_agent1, other_agent2],  # OOP agent objects
    fg_color="#4A90E2",
    bg_color="#E3F2FD",
)
# Returns: ToolAgent (wraps CompiledStateGraph with rich metadata)
```

## Installation

```bash
# From GitHub (this fork)
pip install git+https://github.com/atom2ueki/deepagents.git

# For development
git clone https://github.com/atom2ueki/deepagents.git
cd deepagents
uv sync
```

## Quick Start

```python
import os
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# Web search tool
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
):
    """Run a web search"""
    return tavily_client.search(query, max_results=max_results, topic=topic)

# Create the deep agent (returns ToolAgent object)
agent = create_deep_agent(
    name="researcher",
    description="Expert research agent",
    tools=[internet_search],
    instructions="""You are an expert researcher.
    Conduct thorough research and write polished reports.""",
    fg_color="#2E7D32",
    bg_color="#E8F5E9",
)

# Invoke the agent
result = agent.invoke({"messages": [{"role": "user", "content": "What is langgraph?"}]})
```

See [examples/research/research_agent.py](examples/research/research_agent.py) for a more complex example with subagents.

## Core Capabilities

### 📋 **Planning & Task Decomposition**
Built-in `write_todos` tool enables agents to break down complex tasks, track progress, and adapt plans.

### 💾 **Context Management**
File system tools (`ls`, `read_file`, `write_file`, `edit_file`) allow agents to offload context to memory.

### 🔄 **Subagent Spawning**
Built-in `task` tool enables spawning specialized subagents for context isolation.

### 💿 **Long-term Memory**
Persistent memory across conversations using LangGraph's Store:

```python
from langgraph.store.memory import InMemoryStore
from deepagents import create_deep_agent

store = InMemoryStore()
agent = create_deep_agent(
    tools=[...],
    instructions="...",
    store=store,
    use_longterm_memory=True  # Enable persistent storage
)

# Files prefixed with /memories/ persist across threads
agent.invoke({"messages": [{
    "role": "user",
    "content": "Save important data to /memories/notes.txt"
}]})
```

## API Reference

### `create_deep_agent()`

```python
def create_deep_agent(
    tools: Sequence[BaseTool | Callable] = [],
    instructions: str = "",
    name: str = "main",
    description: str = "",
    fg_color: str = "#000000",
    bg_color: str = "#ffffff",
    middleware: Optional[list[AgentMiddleware]] = None,
    model: Optional[LanguageModelLike] = None,
    subagents: Optional[list[Agent]] = None,
    context_schema: Optional[Type[Any]] = None,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    use_longterm_memory: bool = False,
    tool_configs: Optional[dict[str, bool | InterruptOnConfig]] = None,
    handle_tool_errors: bool = True,
) -> ToolAgent
```

**Parameters:**
- `tools`: List of tools available to the agent
- `instructions`: System prompt for the agent
- `name`: Agent name (for event emissions and debugging)
- `description`: What this agent does (used by parent agents)
- `fg_color`, `bg_color`: Visual identification colors
- `model`: LangChain model (defaults to Claude Sonnet 4.5)
- `subagents`: List of ToolAgent objects this agent can delegate to
- `store`: BaseStore for long-term memory persistence
- `use_longterm_memory`: Enable `/memories/` path for persistent storage

**Returns:** `ToolAgent` instance with auto-context injection

### Creating Subagents

```python
from deepagents import create_deep_agent

# Create specialized subagents
research_agent = create_deep_agent(
    name="researcher",
    description="Conducts deep research on topics",
    instructions="You are a thorough researcher...",
    tools=[search_tool],
    fg_color="#4A90E2",
    bg_color="#E3F2FD",
)

critic_agent = create_deep_agent(
    name="critic",
    description="Critiques and improves content",
    instructions="You are a critical editor...",
    tools=[],
    fg_color="#E91E63",
    bg_color="#FCE4EC",
)

# Main orchestrator delegates to subagents
main_agent = create_deep_agent(
    name="orchestrator",
    description="Main coordination agent",
    tools=[...],
    instructions="Coordinate research and critique...",
    subagents=[research_agent, critic_agent],  # OOP objects!
    fg_color="#2E7D32",
    bg_color="#E8F5E9",
)
```

## Event System

```python
from deepagents import get_event_bus

# Get the global event bus
event_bus = get_event_bus()

# Register event handlers
def on_message(agent_name, fg_color, bg_color, level, message):
    print(f"[{agent_name} L{level}] {message.content}")

def on_todo_update(agent_name, fg_color, bg_color, level, todos):
    print(f"[{agent_name}] Todos: {todos}")

event_bus.on("message", on_message)
event_bus.on("todos", on_todo_update)

# Run agent - events fire automatically
agent.invoke({"messages": [...]})
```

**Event Types:**
- `"message"`: AI messages, tool calls, tool responses
- `"todos"`: Todo list updates
- `"files"`: File system changes (future)

## Agent Hierarchy & Tree Levels

```python
# Agents automatically track their depth in execution tree
main_agent.invoke(...)
# main_agent: level 0
#   ├─ research_agent (via task tool): level 1
#   │   └─ general-purpose: level 2
#   └─ critic_agent (via task tool): level 1
```

The `level` is automatically calculated and passed through `AgentContext`.

## Advanced Usage

### Accessing the Underlying Graph

If you need direct access to the LangGraph `CompiledStateGraph`:

```python
agent = create_deep_agent(...)
graph = agent.graph  # CompiledStateGraph for advanced LangGraph features
```

This gives you full LangGraph capabilities like custom state inspection, checkpointing, and streaming.

## Testing

```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_deepagents.py -v

# Run examples
cd examples/research
python test_agent.py
python test_longterm_memory.py
```

## Contributing

Contributions are welcome! Please submit PRs to:
- https://github.com/atom2ueki/deepagents

For issues or feature requests, open an issue on GitHub.

## License

MIT License

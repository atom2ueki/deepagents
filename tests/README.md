# Test Suite

## Current Status

The test suite has been **updated to use the OOP API** for all subagent tests.

### What Works
- ✅ Basic agent creation tests
- ✅ Tool and middleware tests
- ✅ `ToolAgent` transparently delegates to underlying `CompiledStateGraph`
- ✅ Tests can access `agent.nodes`, `agent.stream_channels`, etc.
- ✅ **Subagent tests** - Now use OOP format with `create_deep_agent()`

### Changes Made

All subagent tests have been converted from dictionary format to OOP API:

**Before (Dictionary API):**
```python
subagents = [{
    "name": "weather_agent",
    "description": "Use this agent to get the weather",
    "prompt": "You are a weather agent.",
    "tools": [get_weather],
    "model": SAMPLE_MODEL,
}]
agent = create_deep_agent(tools=[sample_tool], subagents=subagents)
```

**After (OOP API):**
```python
weather_agent = create_deep_agent(
    name="weather_agent",
    description="Use this agent to get the weather",
    instructions="You are a weather agent.",
    tools=[get_weather],
    model=SAMPLE_MODEL,
    fg_color="#4A90E2",
    bg_color="#E3F2FD",
)
agent = create_deep_agent(tools=[sample_tool], subagents=[weather_agent])
```

### Key API Changes
- `"prompt"` → `instructions` parameter
- Dictionary → `ToolAgent` object (returned by `create_deep_agent()`)
- Added `fg_color` and `bg_color` for visual identification
- Subagents list now contains Agent objects instead of dictionaries

## Running Tests

Run all tests:
```bash
pytest tests/ -v
```

Run specific test files:
```bash
pytest tests/test_deepagents.py -v  # Main deepagents tests
pytest tests/test_middleware.py -v  # Middleware tests
pytest tests/test_hitl.py -v        # HITL tests
```

## Test Structure

```
tests/
├── README.md                 # This file
├── test_deepagents.py       # Main deepagents tests (needs subagent updates)
├── test_hitl.py            # Human-in-the-loop tests
├── test_middleware.py      # Middleware tests
└── utils.py                # Test utilities
```

Upstream has:
```
tests/
├── __init__.py
├── integration_tests/      # Separated integration tests
│   ├── test_deepagents.py
│   ├── test_filesystem_middleware.py
│   ├── test_hitl.py
│   └── test_subagent_middleware.py
├── test_middleware.py
└── utils.py
```

## Next Steps

1. Decide on backward compatibility approach
2. Update subagent tests to use OOP API
3. Consider adopting upstream's test structure (integration_tests/)
4. Add tests for OOP-specific features (events, tree tracking, colors)

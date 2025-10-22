from dotenv import load_dotenv
load_dotenv()

from deepagents.graph import create_deep_agent
from langchain.agents import create_agent
from tests.utils import assert_all_deepagent_qualities, SAMPLE_MODEL, sample_tool, get_weather, get_soccer_scores, SampleMiddlewareWithTools, SampleMiddlewareWithToolsAndState, WeatherToolMiddleware, ResearchMiddleware, ResearchMiddlewareWithTools, TOY_BASKETBALL_RESEARCH

class TestDeepAgents:
    def test_base_deep_agent(self):
        agent = create_deep_agent()
        assert_all_deepagent_qualities(agent)

    def test_deep_agent_with_tool(self):
        agent = create_deep_agent(tools=[sample_tool])
        assert_all_deepagent_qualities(agent)
        assert "sample_tool" in agent.nodes["tools"].bound._tools_by_name.keys()

    def test_deep_agent_with_middleware_with_tool(self):
        agent = create_deep_agent(middleware=[SampleMiddlewareWithTools()])
        assert_all_deepagent_qualities(agent)
        assert "sample_tool" in agent.nodes["tools"].bound._tools_by_name.keys()

    def test_deep_agent_with_middleware_with_tool_and_state(self):
        agent = create_deep_agent(middleware=[SampleMiddlewareWithToolsAndState()])
        assert_all_deepagent_qualities(agent)
        assert "sample_tool" in agent.nodes["tools"].bound._tools_by_name.keys()
        assert "sample_input" in agent.stream_channels

    def test_deep_agent_with_subagents(self):
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
        assert_all_deepagent_qualities(agent)
        result = agent.invoke({"messages": [{"role": "user", "content": "What is the weather in Tokyo?"}]})
        agent_messages = [msg for msg in result.get("messages", []) if msg.type == "ai"]
        tool_calls = [tool_call for msg in agent_messages for tool_call in msg.tool_calls]
        assert any([tool_call["name"] == "task" and tool_call["args"].get("subagent_type") == "weather_agent" for tool_call in tool_calls])

    def test_deep_agent_with_subagents_gen_purpose(self):
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
        assert_all_deepagent_qualities(agent)
        result = agent.invoke({"messages": [{"role": "user", "content": "Use the general purpose subagent to call the sample tool"}]})
        agent_messages = [msg for msg in result.get("messages", []) if msg.type == "ai"]
        tool_calls = [tool_call for msg in agent_messages for tool_call in msg.tool_calls]
        assert any([tool_call["name"] == "task" and tool_call["args"].get("subagent_type") == "general-purpose" for tool_call in tool_calls])

    def test_deep_agent_with_subagents_with_middleware(self):
        weather_agent = create_deep_agent(
            name="weather_agent",
            description="Use this agent to get the weather",
            instructions="You are a weather agent.",
            tools=[],
            model=SAMPLE_MODEL,
            middleware=[WeatherToolMiddleware()],
            fg_color="#4A90E2",
            bg_color="#E3F2FD",
        )
        agent = create_deep_agent(tools=[sample_tool], subagents=[weather_agent])
        assert_all_deepagent_qualities(agent)
        result = agent.invoke({"messages": [{"role": "user", "content": "What is the weather in Tokyo?"}]})
        agent_messages = [msg for msg in result.get("messages", []) if msg.type == "ai"]
        tool_calls = [tool_call for msg in agent_messages for tool_call in msg.tool_calls]
        assert any([tool_call["name"] == "task" and tool_call["args"].get("subagent_type") == "weather_agent" for tool_call in tool_calls])

    def test_deep_agent_with_custom_subagents(self):
        weather_agent = create_deep_agent(
            name="weather_agent",
            description="Use this agent to get the weather",
            instructions="You are a weather agent.",
            tools=[get_weather],
            model=SAMPLE_MODEL,
            fg_color="#4A90E2",
            bg_color="#E3F2FD",
        )
        soccer_agent = create_deep_agent(
            name="soccer_agent",
            description="Use this agent to get the latest soccer scores",
            instructions="You are a soccer agent.",
            tools=[get_soccer_scores],
            model=SAMPLE_MODEL,
            fg_color="#FF9800",
            bg_color="#FFF3E0",
        )
        agent = create_deep_agent(tools=[sample_tool], subagents=[weather_agent, soccer_agent])
        assert_all_deepagent_qualities(agent)
        result = agent.invoke({"messages": [{"role": "user", "content": "Look up the weather in Tokyo, and the latest scores for Manchester City!"}]})
        agent_messages = [msg for msg in result.get("messages", []) if msg.type == "ai"]
        tool_calls = [tool_call for msg in agent_messages for tool_call in msg.tool_calls]
        assert any([tool_call["name"] == "task" and tool_call["args"].get("subagent_type") == "weather_agent" for tool_call in tool_calls])
        assert any([tool_call["name"] == "task" and tool_call["args"].get("subagent_type") == "soccer_agent" for tool_call in tool_calls])

    def test_deep_agent_with_extended_state_and_subagents(self):
        basketball_agent = create_deep_agent(
            name="basketball_info_agent",
            description="Use this agent to get surface level info on any basketball topic",
            instructions="You are a basketball info agent.",
            middleware=[ResearchMiddlewareWithTools()],
            fg_color="#9C27B0",
            bg_color="#F3E5F5",
        )
        agent = create_deep_agent(tools=[sample_tool], subagents=[basketball_agent], middleware=[ResearchMiddleware()])
        assert_all_deepagent_qualities(agent)
        assert "research" in agent.stream_channels
        result = agent.invoke({"messages": [{"role": "user", "content": "Get surface level info on lebron james"}]}, config={"recursion_limit": 100})
        agent_messages = [msg for msg in result.get("messages", []) if msg.type == "ai"]
        tool_calls = [tool_call for msg in agent_messages for tool_call in msg.tool_calls]
        assert any([tool_call["name"] == "task" and tool_call["args"].get("subagent_type") == "basketball_info_agent" for tool_call in tool_calls])
        assert TOY_BASKETBALL_RESEARCH in result["research"]

    def test_deep_agent_with_subagents_no_tools(self):
        basketball_agent = create_deep_agent(
            name="basketball_info_agent",
            description="Use this agent to get surface level info on any basketball topic",
            instructions="You are a basketball info agent.",
            fg_color="#9C27B0",
            bg_color="#F3E5F5",
        )
        agent = create_deep_agent(tools=[sample_tool], subagents=[basketball_agent])
        assert_all_deepagent_qualities(agent)
        result = agent.invoke({"messages": [{"role": "user", "content": "Use the basketball info subagent to call the sample tool"}]}, config={"recursion_limit": 100})
        agent_messages = [msg for msg in result.get("messages", []) if msg.type == "ai"]
        tool_calls = [tool_call for msg in agent_messages for tool_call in msg.tool_calls]
        assert any([tool_call["name"] == "task" and tool_call["args"].get("subagent_type") == "basketball_info_agent" for tool_call in tool_calls])

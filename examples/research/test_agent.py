"""Simple test script for the research agent."""
import os
from dotenv import load_dotenv

load_dotenv()

# Import the agent from research_agent.py
from research_agent import agent

def test_basic_functionality():
    """Test basic agent invocation."""
    print("Testing research agent with a simple question...")
    print("=" * 60)

    # Simple test query
    result = agent.invoke({
        "messages": [{"role": "user", "content": "What is LangGraph in 2 sentences?"}]
    })

    # Print the final message
    final_message = result["messages"][-1].content
    print("\nAgent Response:")
    print("-" * 60)
    print(final_message)
    print("-" * 60)
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    # Verify environment variables
    required_vars = ["ANTHROPIC_API_KEY", "TAVILY_API_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]

    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file")
        exit(1)

    print("✅ Environment variables loaded")
    test_basic_functionality()

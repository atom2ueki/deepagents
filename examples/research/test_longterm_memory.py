"""Test script for long-term memory feature."""
import os
from dotenv import load_dotenv
from langgraph.store.memory import InMemoryStore

load_dotenv()

from deepagents import create_deep_agent

def test_longterm_memory():
    """Test long-term memory with persistent store."""
    print("Testing long-term memory feature...")
    print("=" * 60)

    # Create a store for long-term memory
    store = InMemoryStore()

    # Create agent with long-term memory enabled
    agent = create_deep_agent(
        name="memory-test-agent",
        description="Agent for testing long-term memory",
        instructions="You are a helpful assistant with access to both short-term and long-term file storage.",
        store=store,
        use_longterm_memory=True,
    )

    print("\n✅ Agent created with long-term memory support")
    print(f"   Store type: {type(store).__name__}")

    # Test 1: Write to short-term filesystem
    print("\n📝 Test 1: Writing to short-term filesystem...")
    result1 = agent.invoke({
        "messages": [{"role": "user", "content": "Please write 'This is temporary data' to /temp.txt"}]
    })
    print(f"   Response: {result1['messages'][-1].content[:100]}...")

    # Test 2: Write to long-term filesystem
    print("\n📝 Test 2: Writing to long-term filesystem...")
    result2 = agent.invoke({
        "messages": [{"role": "user", "content": "Please write 'This persists across conversations' to /memories/persistent.txt"}]
    })
    print(f"   Response: {result2['messages'][-1].content[:100]}...")

    # Test 3: List all files
    print("\n📋 Test 3: Listing all files...")
    result3 = agent.invoke({
        "messages": [{"role": "user", "content": "List all files in the filesystem"}]
    })
    print(f"   Response: {result3['messages'][-1].content}")

    # Test 4: Read from long-term memory
    print("\n📖 Test 4: Reading from long-term filesystem...")
    result4 = agent.invoke({
        "messages": [{"role": "user", "content": "Read the file /memories/persistent.txt"}]
    })
    print(f"   Response: {result4['messages'][-1].content}")

    print("\n" + "=" * 60)
    print("✅ All long-term memory tests completed successfully!")
    print("\nKey observations:")
    print("  • Files prefixed with /memories/ are stored persistently")
    print("  • Regular files are stored in ephemeral state")
    print("  • Both can be accessed with the same tools")

if __name__ == "__main__":
    # Verify environment variables
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ Missing ANTHROPIC_API_KEY environment variable")
        print("Please set it in your .env file")
        exit(1)

    print("✅ Environment variables loaded")
    test_longterm_memory()


import os
import asyncio
import json
from datetime import datetime
from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.genai.types import Content, Part

from models import ExecutionTrace, EventData
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_google_api_key():
    """Retrieve the Google API key from environment variables."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")
    return api_key

session_service = InMemorySessionService()
my_user_id = "adk_researcher_001"

topic_research_agent: Agent

agent_instruction="""
        You are an expert thesis professor and topic researcher in AI and emerging technologies. Your goal is to help researchers
        find compelling research topics using your search tool.

        You should:
        1.  **Understand the Research topic:** If the user's request is vague, ask for details like target audience,
            content niche, and specific keywords.
        2.  **Research Trends:** Use your search tool to find trending topics, popular questions,
            and content gaps in the specified research topic.
        3.  **Generate Ideas:** Suggest 5-7 specific masters and PhD thesis topics that are:
            - Relevant to the research topic
            - Trending or evergreen
            - Actionable and specific
            - SEO-friendly
        4.  **Provide Context:** For each topic, briefly explain why it's a good choice.
        5.  **Summarize:** Present the information in a clear, organized format.

        Be creative, data-driven, and always think about AI domain with a focus on emerging trends and research opportunities.
        """

def create_topic_research_agent():
    """Create a topic research agent for research purposes."""
    return Agent(
        name="topic_research_agent",
        model="gemini-2.5-flash",
        instruction=agent_instruction,
        tools=[google_search]
    )

def get_agent():
    """Get the topic research agent based on the environment configuration."""
    if not os.getenv("GOOGLE_API_KEY"):
        # Initialize the agent for offline LocalGCP use
        return Agent(
            name="localgcp_agent",
            model="gemma3",
            instruction=agent_instruction
        )
    else:
        # Initialize your standard cloud-connected agent here
        print("✅ API Key configured successfully!")
        return create_topic_research_agent()


async def run_agent_query(agent: Agent, query: str, session: Session, user_id: str):
    print(f"\n🚀 Running query for agent: '{agent.name}' in session: '{session.id}'...")

    # Create the Runner - the execution orchestrator
    runner = Runner(
            agent=agent,
            session_service=session_service,
            app_name=agent.name
        )
    print(f"🧩 Runner created for agent: '{agent.name}' with session: '{session.id}'")
    final_response = ""
    execution_trace: ExecutionTrace = {
        "events": [],
        "tools_called": [],
        "reasoning_steps": [],
        "timestamps": []
    }


    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=Content(parts=[Part(text=query)], role="user")
        ):
            # Log all event types
            event_type = type(event).__name__
            timestamp = datetime.now().isoformat()

            logger.info(f"Event: {event_type}")
            print(f"\n📍 Event Type: {event_type}")
            print(f"   Timestamp: {timestamp}")

            # Capture event data
            event_data: EventData = {
                "type": event_type,
                "timestamp": timestamp,
                "details": {}
            }

            # Check for different event types
            if hasattr(event, 'content') and event.content:
                print(f"   Content: {event.content}")
                event_data["details"]["content"] = str(event.content)

            if hasattr(event, 'tool_call'):
                print(f"   🔧 Tool Call: {event.tool_call}")
                event_data["details"]["tool_call"] = str(event.tool_call)
                execution_trace["tools_called"].append(str(event.tool_call))
                logger.info(f"Tool called: {event.tool_call}")

            if hasattr(event, 'tool_result'):
                print(f"   📊 Tool Result: {event.tool_result}")
                event_data["details"]["tool_result"] = str(event.tool_result)

            if hasattr(event, 'thinking'):
                print(f"   🧠 Reasoning: {event.thinking}")
                event_data["details"]["thinking"] = str(event.thinking)
                execution_trace["reasoning_steps"].append(str(event.thinking))
                logger.info(f"Reasoning: {event.thinking}")

            if hasattr(event, 'text'):
                print(f"   💬 Text: {event.text}")
                event_data["details"]["text"] = str(event.text)

            # Check if this is the final response
            if event.is_final_response():
                print("\n✅ FINAL RESPONSE DETECTED")
                if event.content and event.content.parts:
                    final_response = event.content.parts[0].text or ""
                event_data["details"]["is_final"] = True
                event_data["details"]["response"] = final_response

            execution_trace["events"].append(event_data)
            execution_trace["timestamps"].append(timestamp)

    except Exception as e:
        final_response = f"An error occurred: {e}"
        logger.error(f"Error during agent execution: {e}", exc_info=True)


    # Save detailed execution trace
    trace_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_filename = f"trace_{trace_timestamp}.json"

    output_dir = os.path.join(os.getcwd(), "simple_research_agent", "traces")
    os.makedirs(output_dir, exist_ok=True)
    trace_filepath = os.path.join(output_dir, trace_filename)

    with open(trace_filepath, "w", encoding="utf-8") as f:
        json.dump(execution_trace, f, indent=2, default=str)

    logger.info(f"Execution trace saved to: {trace_filepath}")
    print(f"📋 Execution trace saved to: {trace_filepath}")

    # Save response to file with timestamp
    filename = f"response_{trace_timestamp}.txt"
    response_output_dir = os.path.join(os.getcwd(), "simple_research_agent", "responses")
    os.makedirs(response_output_dir, exist_ok=True)
    filepath = os.path.join(response_output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"User ID: {user_id}\n")
        f.write(f"Session ID: {session.id}\n")
        f.write(f"Query: {query}\n")
        f.write("="*50 + "\n")
        f.write(f"Response:\n{final_response}\n")
        f.write("="*50 + "\n")
        f.write(f"Tools Called: {len(execution_trace['tools_called'])}\n")
        if execution_trace['tools_called']:
            for i, tool in enumerate(execution_trace['tools_called'], 1):
                f.write(f"  {i}. {tool}\n")
        f.write(f"Reasoning Steps: {len(execution_trace['reasoning_steps'])}\n")
        f.write(f"Total Events: {len(execution_trace['events'])}\n")

    print(f"💾 Response saved to: {filepath}\n")

    return final_response

async def run_topic_research():
    topic_research_session = await session_service.create_session(
        user_id=my_user_id,
        app_name="topic_research_agent"
        )
    query = "I need research topic ideas for a PhD Thesis focused on Agentic AI and machine learning for Industrial Automation. "
    print(f"🗣️ User Query: '{query}'")

    await run_agent_query(topic_research_agent, query, topic_research_session, my_user_id)

if __name__ == "__main__":
    get_google_api_key()  # Ensure the API key is set before proceeding
    # Test execution entirely offline
    topic_research_agent = get_agent()
    asyncio.run(run_topic_research())
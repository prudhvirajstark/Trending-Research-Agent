
import os
import asyncio
from datetime import datetime
from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.genai.types import Content, Part

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
    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=Content(parts=[Part(text=query)], role="user")
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response = event.content.parts[0].text or ""
    except Exception as e:
        final_response = f"An error occurred: {e}"


    # Save response to file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"response_{timestamp}.txt"

    output_dir = os.path.join(os.getcwd(), "simple_research_agent", "responses")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"User ID: {user_id}\n")
        f.write(f"Session ID: {session.id}\n")
        f.write(f"Query: {query}\n")
        f.write("="*50 + "\n")
        f.write(f"Response:\n{final_response}\n")

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
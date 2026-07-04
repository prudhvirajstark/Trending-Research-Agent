"""
PhD Thesis Multi-Agent Evaluation System - Main Entry Point

Production-ready system using Google GenAI SDK with three specialized agents:
- Professor: Provides mentoring guidance
- Critic: Challenges assumptions and finds flaws
- Reviewer: Evaluates against journal standards using rubrics

Supports multiple communication patterns:
- Sequential: Linear evaluation through all agents
- Debate: Professor and Critic iterate on sections
- Group Chat: Round-robin discussion until consensus
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

from src.agents.professor import ProfessorAgent
from src.agents.critic import CriticAgent
from src.agents.reviewer import ReviewerAgent
from src.agents.coordinator import CoordinatorAgent
from src.orchestration.router import ThesisRouter
from src.utils.config_loader import ConfigLoader
from src.utils.telemetry import setup_logging


# Load environment variables
load_dotenv()

# Configure logging
setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_format=os.getenv("LOG_FORMAT", "text"),
)

logger = logging.getLogger(__name__)


async def main():
    """Main entry point for the PhD thesis evaluation system."""

    logger.info("=" * 80)
    logger.info("PhD Thesis Multi-Agent Evaluation System")
    logger.info("=" * 80)

    # Load configurations
    config_loader = ConfigLoader(config_dir="config")
    agent_configs = config_loader.load_agent_prompts()
    model_configs = config_loader.load_model_config()

    if not agent_configs or not model_configs:
        logger.error("Failed to load configuration files")
        sys.exit(1)

    # Initialize agents with configurations
    logger.info("Initializing agents...")

    professor_config = agent_configs.get("professor", {})
    professor = ProfessorAgent(
        system_prompt=professor_config.get("system_prompt", "")
    )

    critic_config = agent_configs.get("critic", {})
    critic = CriticAgent(
        system_prompt=critic_config.get("system_prompt", "")
    )

    reviewer_config = agent_configs.get("reviewer", {})
    reviewer = ReviewerAgent(
        system_prompt=reviewer_config.get("system_prompt", "")
    )

    # Initialize Coordinator for intelligent routing and state management
    coordinator_prompt = """
    You are a task coordinator for a PhD thesis evaluation system. Your role is to:
    1. Intelligently route tasks to appropriate agents
    2. Manage evaluation sessions and state
    3. Optimize for cost and latency
    4. Call tools for document operations

    Be strategic in your routing decisions based on task type and resource constraints.
    """
    coordinator = CoordinatorAgent(system_prompt=coordinator_prompt)

    # Register built-in tools
    coordinator.register_builtin_tools()

    logger.info("✅ All agents initialized successfully (including Coordinator)")

    # Initialize router with coordinator
    router = ThesisRouter(
        professor=professor,
        critic=critic,
        reviewer=reviewer,
        coordinator=coordinator,
    )

    # Example: Evaluate a thesis section
    sample_thesis = """
    Abstract:
    This thesis explores the application of transformers to long-form document understanding.
    We propose a novel attention mechanism that reduces computational complexity from O(n²) to O(n log n).
    Our experiments on BERT-scale models show 23% improvement in efficiency with minimal accuracy loss.

    Introduction:
    Long-form documents present a fundamental challenge to modern language models due to their quadratic
    attention complexity. Our approach leverages sparse attention patterns to achieve linear complexity.

    Methodology:
    We implemented a sliding window attention mechanism with local-to-global patterns.
    The model was trained on 500GB of academic papers and evaluated on standard benchmarks.
    """

    logger.info("\nCoordinator Analysis:")
    logger.info("-" * 80)

    # 1. Get coordinator routing decision
    logger.info("🔄 Getting coordinator routing decision...")
    routing = await router.coordinator_route(sample_thesis)
    logger.info(f"Primary Agent: {routing.get('primary_agent')}")
    logger.info(f"Secondary Agents: {routing.get('secondary_agents', [])}")
    logger.info(f"Reasoning: {routing.get('reasoning')}")

    # 2. Get cost estimate
    logger.info("\n💰 Cost Estimation:")
    cost_estimate = router.get_cost_estimate(["professor", "critic", "reviewer"])
    logger.info(f"Total Estimated Cost: ${cost_estimate['total_estimated_cost']}")
    for agent, info in cost_estimate.get("agents", {}).items():
        logger.info(f"  {agent}: ${info['estimated_cost']} ({info['estimated_latency']}s latency)")

    # 3. Create evaluation session
    session_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    coordinator.create_session(session_id, {"thesis_topic": "Transformers"})
    logger.info(f"\n📋 Created evaluation session: {session_id}")

    logger.info("\nStarting sequential evaluation...")
    logger.info(f"Evaluating: {sample_thesis[:100]}...\n")

    # Run sequential evaluation
    results = await router.sequential_evaluation(sample_thesis)

    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)

    if "professor" in results.get("evaluations", {}):
        prof_eval = results["evaluations"]["professor"]
        logger.info("\n📚 PROFESSOR FEEDBACK:")
        if "evaluation" in prof_eval:
            logger.info(prof_eval["evaluation"][:500] + "...")

    if "critic" in results.get("evaluations", {}):
        critic_eval = results["evaluations"]["critic"]
        logger.info("\n⚡ CRITIC ANALYSIS:")
        if "critical_analysis" in critic_eval:
            logger.info(critic_eval["critical_analysis"][:500] + "...")

    if "reviewer" in results.get("evaluations", {}):
        reviewer_eval = results["evaluations"]["reviewer"]
        logger.info("\n📊 REVIEWER VERDICT:")
        if "review" in reviewer_eval:
            review = reviewer_eval["review"]
            if "scores" in review:
                logger.info(f"Scores: {review['scores']}")
            if "average_score" in review:
                logger.info(f"Average Score: {review['average_score']}/10")
            if "verdict" in review:
                logger.info(f"Verdict: {review['verdict']}")

    logger.info("\n" + "=" * 80)
    logger.info("Evaluation complete!")
    logger.info("=" * 80)

    # Log evaluation to coordinator session
    if "professor" in results.get("evaluations", {}):
        coordinator.log_evaluation(session_id, "professor", results["evaluations"]["professor"])
    if "critic" in results.get("evaluations", {}):
        coordinator.log_evaluation(session_id, "critic", results["evaluations"]["critic"])
    if "reviewer" in results.get("evaluations", {}):
        coordinator.log_evaluation(session_id, "reviewer", results["evaluations"]["reviewer"])

    # Display coordinator session summary
    session_summary = coordinator.get_session_summary(session_id)
    logger.info("\n🔗 Coordinator Session Summary:")
    logger.info(f"Session ID: {session_summary['session_id']}")
    logger.info(f"Evaluations Logged: {session_summary['evaluation_count']}")
    logger.info(f"Agents Involved: {', '.join(session_summary['agents_involved'])}")

    # Display metrics
    logger.info("\n📈 Execution Metrics:")
    metrics = router.get_metrics()
    logger.info(f"Router Metrics:\n{metrics}")

    # Rate limiting and token budget metrics
    logger.info("\n⏱️  Rate Limiting & Token Budget:")
    agent_metrics = professor.get_metrics()
    if "rate_limiter" in agent_metrics:
        rate_metrics = agent_metrics["rate_limiter"]
        logger.info(
            f"Rate Limit: {rate_metrics['requests_last_minute']}/{rate_metrics['capacity']} "
            f"req/min ({rate_metrics['utilization']*100:.1f}% utilized)"
        )
    if "token_budget" in agent_metrics:
        token_metrics = agent_metrics["token_budget"]
        logger.info(
            f"Token Usage: {token_metrics['tokens_used']:,}/{token_metrics['daily_limit']:,} "
            f"({token_metrics['usage_percent']:.1f}%) | "
            f"Reset: {token_metrics['reset_time']}"
        )

    # Coordinator-specific metrics
    if hasattr(coordinator, "get_tool_metrics"):
        tool_metrics = coordinator.get_tool_metrics()
        if tool_metrics:
            logger.info(f"Tool Usage: {tool_metrics}")



async def interactive_mode():
    """Interactive mode for real-time thesis evaluation."""

    logger.info("Entering interactive mode...")

    config_loader = ConfigLoader(config_dir="config")
    agent_configs = config_loader.load_agent_prompts()

    professor = ProfessorAgent(
        system_prompt=agent_configs.get("professor", {}).get("system_prompt", "")
    )
    critic = CriticAgent(
        system_prompt=agent_configs.get("critic", {}).get("system_prompt", "")
    )
    reviewer = ReviewerAgent(
        system_prompt=agent_configs.get("reviewer", {}).get("system_prompt", "")
    )

    router = ThesisRouter(professor=professor, critic=critic, reviewer=reviewer)

    print("\n" + "=" * 80)
    print("PhD Thesis Multi-Agent Evaluation System - Interactive Mode")
    print("=" * 80)
    print("\nCommands:")
    print("  1: Sequential evaluation")
    print("  2: Debate (Professor vs Critic)")
    print("  3: Group chat")
    print("  4: Exit")
    print("=" * 80 + "\n")

    while True:
        choice = input("\nSelect mode (1-4): ").strip()

        if choice == "1":
            content = input("Paste your thesis content (or 'done' to finish):\n")
            if content.lower() != "done":
                results = await router.sequential_evaluation(content)
                print("\n✅ Sequential evaluation completed")
                print(f"Agents executed: {results['agents_executed']}")

        elif choice == "2":
            section = input("Paste the section to debate:\n")
            title = input("Section title: ")
            results = await router.debate_loop(section, title, num_rounds=2)
            print(f"\n✅ Debate completed with {len(results['debate_history'])} exchanges")

        elif choice == "3":
            proposal = input("Paste the initial proposal:\n")
            results = await router.group_chat_until_consensus(proposal)
            print("\n✅ Group chat completed")
            print(f"Consensus reached: {results['consensus_reached']}")

        elif choice == "4":
            print("\nExiting...")
            break

        else:
            print("Invalid choice. Please select 1-4.")


if __name__ == "__main__":
    # Run main evaluation by default, or use --interactive for interactive mode
    if "--interactive" in sys.argv:
        asyncio.run(interactive_mode())
    else:
        asyncio.run(main())


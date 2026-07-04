"""
Router / Orchestrator - Coordinates multi-agent workflow

Manages communication flow between agents using different communication patterns:
- Sequential: Linear flow through all agents
- Debate: Professor and Critic iterate on a section
- Group Chat: Round-robin discussion until consensus

Also manages tool registry and integrates the Coordinator for intelligent routing.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from datetime import datetime

from src.agents.base_agent import BaseAgent


logger = logging.getLogger(__name__)


class CommunicationPattern(str, Enum):
    """Supported agent communication patterns."""

    SEQUENTIAL = "sequential"
    DEBATE = "debate"
    GROUP_CHAT = "group_chat"


class ThesisRouter:
    """Orchestrates communication between all agents with coordinator support."""

    def __init__(
        self,
        professor: BaseAgent,
        critic: BaseAgent,
        reviewer: BaseAgent,
        coordinator: Optional[BaseAgent] = None,
    ):
        """
        Initialize the router with agents and optional coordinator.

        Args:
            professor: The Professor agent
            critic: The Critic agent
            reviewer: The Reviewer agent
            coordinator: Optional Coordinator agent for intelligent routing
        """
        self.professor = professor
        self.critic = critic
        self.reviewer = reviewer
        self.coordinator = coordinator

        self.execution_history: List[Dict[str, Any]] = []

        # Tool registry for document operations and system tasks
        self.tools: Dict[str, Callable] = {}

        logger.info("ThesisRouter initialized with coordinator support")

    def register_tool(self, tool_name: str, handler: Callable) -> None:
        """
        Register a tool with the router.

        Args:
            tool_name: Name of the tool
            handler: Async callable that implements the tool
        """
        self.tools[tool_name] = handler

        # Also register with coordinator if available
        if self.coordinator and hasattr(self.coordinator, "register_tool"):
            self.coordinator.register_tool(
                name=tool_name,
                description=f"Tool: {tool_name}",
                parameters={"type": "object"},
                handler=handler,
            )

        logger.info(f"Registered tool: {tool_name}")

    async def coordinator_route(
        self,
        task_input: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Use coordinator to intelligently route a task.

        Args:
            task_input: The task to route
            session_id: Optional session ID

        Returns:
            Routing decision from coordinator
        """
        if not self.coordinator:
            logger.warning("Coordinator not available for routing")
            return {"primary_agent": "professor", "secondary_agents": []}

        routing = await self.coordinator.route_task(
            task_input=task_input,
            session_id=session_id,
        )

        return routing

    def get_cost_estimate(self, agents: List[str]) -> Dict[str, Any]:
        """
        Get cost estimate for running specific agents.

        Args:
            agents: List of agent names

        Returns:
            Cost estimate dictionary
        """
        if not self.coordinator or not hasattr(self.coordinator, "get_cost_estimate"):
            return {"total_estimated_cost": 0.0, "breakdown": {}}

        return self.coordinator.get_cost_estimate(agents)

    async def sequential_evaluation(
        self,
        thesis_content: str,
        skip_agents: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run sequential evaluation: Professor → Critic → Reviewer.

        Args:
            thesis_content: The thesis section or full content
            skip_agents: List of agent names to skip (e.g., ["reviewer"])

        Returns:
            Aggregated evaluation results from all agents
        """
        skip_agents = skip_agents or []
        results = {
            "pattern": CommunicationPattern.SEQUENTIAL.value,
            "timestamp": datetime.now().isoformat(),
            "thesis_excerpt": thesis_content[:200] + "..." if len(thesis_content) > 200 else thesis_content,
            "agents_executed": [],
            "evaluations": {},
        }

        # Run Professor evaluation
        if "professor" not in skip_agents:
            logger.info("Running Professor evaluation...")
            try:
                prof_result = await self.professor.evaluate(thesis_content)
                results["evaluations"]["professor"] = prof_result
                results["agents_executed"].append("professor")
                self.execution_history.append(
                    {
                        "agent": "professor",
                        "pattern": CommunicationPattern.SEQUENTIAL.value,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            except Exception as e:
                logger.error(f"Professor evaluation failed: {str(e)}")
                results["evaluations"]["professor"] = {"error": str(e)}

        # Run Critic evaluation
        if "critic" not in skip_agents:
            logger.info("Running Critic evaluation...")
            try:
                critic_result = await self.critic.evaluate(thesis_content)
                results["evaluations"]["critic"] = critic_result
                results["agents_executed"].append("critic")
                self.execution_history.append(
                    {
                        "agent": "critic",
                        "pattern": CommunicationPattern.SEQUENTIAL.value,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            except Exception as e:
                logger.error(f"Critic evaluation failed: {str(e)}")
                results["evaluations"]["critic"] = {"error": str(e)}

        # Run Reviewer evaluation
        if "reviewer" not in skip_agents:
            logger.info("Running Reviewer evaluation...")
            try:
                reviewer_result = await self.reviewer.evaluate(thesis_content)
                results["evaluations"]["reviewer"] = reviewer_result
                results["agents_executed"].append("reviewer")
                self.execution_history.append(
                    {
                        "agent": "reviewer",
                        "pattern": CommunicationPattern.SEQUENTIAL.value,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            except Exception as e:
                logger.error(f"Reviewer evaluation failed: {str(e)}")
                results["evaluations"]["reviewer"] = {"error": str(e)}

        return results

    async def debate_loop(
        self,
        section_content: str,
        section_title: str,
        num_rounds: int = 3,
    ) -> Dict[str, Any]:
        """
        Run debate pattern: Professor and Critic iterate on a section.

        Args:
            section_content: The thesis section to debate
            section_title: Title of the section
            num_rounds: Number of debate rounds (each agent goes twice per round)

        Returns:
            Debate history and final consensus
        """
        self.professor.clear_history()
        self.critic.clear_history()

        results = {
            "pattern": CommunicationPattern.DEBATE.value,
            "section_title": section_title,
            "timestamp": datetime.now().isoformat(),
            "num_rounds": num_rounds,
            "debate_history": [],
        }

        current_content = section_content

        for round_num in range(num_rounds):
            logger.info(f"Debate round {round_num + 1}/{num_rounds}")

            # Professor's turn
            logger.info(f"Round {round_num + 1}: Professor's critique...")
            try:
                prof_response = await self.professor.suggest_improvements(
                    current_content, section_title
                )
                results["debate_history"].append(
                    {
                        "round": round_num + 1,
                        "speaker": "professor",
                        "content": prof_response,
                    }
                )
            except Exception as e:
                logger.error(f"Professor response failed: {str(e)}")

            # Critic's turn
            logger.info(f"Round {round_num + 1}: Critic's response...")
            try:
                critic_response = await self.critic.challenge_claims(
                    section_content, prof_response
                )
                results["debate_history"].append(
                    {
                        "round": round_num + 1,
                        "speaker": "critic",
                        "content": critic_response,
                    }
                )
            except Exception as e:
                logger.error(f"Critic response failed: {str(e)}")

        return results

    async def group_chat_until_consensus(
        self,
        initial_proposal: str,
        consensus_threshold: float = 0.8,
        max_iterations: int = 5,
    ) -> Dict[str, Any]:
        """
        Run group chat pattern: Round-robin discussion until consensus.

        Args:
            initial_proposal: Initial thesis proposal
            consensus_threshold: Score threshold for consensus (0-1)
            max_iterations: Maximum number of round-robin iterations

        Returns:
            Group chat history and consensus result
        """
        self.professor.clear_history()
        self.critic.clear_history()
        self.reviewer.clear_history()

        results = {
            "pattern": CommunicationPattern.GROUP_CHAT.value,
            "timestamp": datetime.now().isoformat(),
            "iterations": [],
            "consensus_reached": False,
            "final_scores": {},
        }

        conversation_context = initial_proposal

        for iteration in range(max_iterations):
            logger.info(f"Group chat iteration {iteration + 1}/{max_iterations}")

            iteration_data = {
                "iteration": iteration + 1,
                "messages": [],
            }

            # Round-robin: Professor → Critic → Reviewer
            agents = [
                ("professor", self.professor),
                ("critic", self.critic),
                ("reviewer", self.reviewer),
            ]

            for agent_name, agent in agents:
                logger.info(f"  {agent_name.capitalize()} speaking...")
                try:
                    response = await agent.generate_response(
                        f"Given this context:\n{conversation_context}\n\nProvide your perspective:"
                    )
                    iteration_data["messages"].append(
                        {
                            "speaker": agent_name,
                            "message": response,
                        }
                    )
                    conversation_context += f"\n\n{agent_name.upper()}: {response}"
                except Exception as e:
                    logger.error(f"{agent_name} response failed: {str(e)}")

            results["iterations"].append(iteration_data)

            # Check for consensus (simplified: reviewer score)
            if iteration > 1:  # After at least 2 iterations
                try:
                    review_result = await self.reviewer.evaluate(conversation_context)
                    if "review" in review_result and "average_score" in review_result["review"]:
                        avg_score = review_result["review"]["average_score"]
                        if avg_score / 100 >= consensus_threshold:
                            results["consensus_reached"] = True
                            results["final_scores"] = review_result["review"]["scores"]
                            logger.info("Consensus reached!")
                            break
                except Exception as e:
                    logger.error(f"Consensus check failed: {str(e)}")

        return results

    async def evaluate_thesis_section(
        self,
        section_content: str,
        section_title: str = "Thesis Section",
        pattern: CommunicationPattern = CommunicationPattern.SEQUENTIAL,
    ) -> Dict[str, Any]:
        """
        Evaluate a thesis section using specified communication pattern.

        Args:
            section_content: The thesis section to evaluate
            section_title: Title of the section
            pattern: Which communication pattern to use

        Returns:
            Comprehensive evaluation results
        """
        logger.info(f"Starting evaluation: {section_title} using {pattern.value} pattern")

        if pattern == CommunicationPattern.SEQUENTIAL:
            return await self.sequential_evaluation(section_content)

        elif pattern == CommunicationPattern.DEBATE:
            return await self.debate_loop(section_content, section_title)

        elif pattern == CommunicationPattern.GROUP_CHAT:
            return await self.group_chat_until_consensus(section_content)

        else:
            raise ValueError(f"Unknown communication pattern: {pattern}")

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get the history of all agent executions."""
        return self.execution_history

    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregate metrics across all agents."""
        return {
            "professor_metrics": self.professor.get_metrics(),
            "critic_metrics": self.critic.get_metrics(),
            "reviewer_metrics": self.reviewer.get_metrics(),
            "total_evaluations": len(self.execution_history),
        }

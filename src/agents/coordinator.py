"""
Coordinator Agent - The Orchestrator

Intelligently routes tasks to appropriate agents, manages state, and coordinates
tool-calling for document operations and system tasks.

Uses Gemini 2.0 Flash for fast, low-latency orchestration and exceptional tool-calling.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum

from .base_agent import BaseAgent
from src.utils.document_parsers import DocumentParser


logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Types of tasks the coordinator can route."""

    MENTORING = "mentoring"  # Route to Professor
    CRITICAL_ANALYSIS = "critical_analysis"  # Route to Critic
    PEER_REVIEW = "peer_review"  # Route to Reviewer
    MULTI_AGENT = "multi_agent"  # Run all agents
    DEBATE = "debate"  # Professor vs Critic debate
    DOCUMENT_PARSING = "document_parsing"  # Parse PDF/LaTeX
    STATE_MANAGEMENT = "state_management"  # Manage session state
    COST_OPTIMIZATION = "cost_optimization"  # Select optimal agents


class AgentCostModel:
    """Model for calculating agent costs and latency."""

    # Approximate costs per 1M tokens (in USD)
    COSTS = {
        "gemini-2.0-flash": 0.075,
        "gemini-2.5-pro": 0.60,
    }

    # Approximate latency (in seconds)
    LATENCIES = {
        "gemini-2.0-flash": 0.5,
        "gemini-2.5-pro": 1.5,
    }

    @classmethod
    def estimate_cost(cls, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost for an API call.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        cost_per_mtok = cls.COSTS.get(model, 0.10)
        total_tokens = input_tokens + output_tokens
        return (total_tokens / 1_000_000) * cost_per_mtok

    @classmethod
    def estimate_latency(cls, model: str) -> float:
        """Get estimated latency for a model."""
        return cls.LATENCIES.get(model, 1.0)


class ToolDefinition:
    """Definition of a callable tool for agents."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable,
    ):
        """
        Initialize tool definition.

        Args:
            name: Tool name
            description: Tool description
            parameters: Parameter schema
            handler: Async callable that implements the tool
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler
        self.call_count = 0
        self.total_time = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class CoordinatorAgent(BaseAgent):
    """Orchestrates multi-agent system with intelligent routing and tool management."""

    def __init__(self, system_prompt: str):
        """Initialize the Coordinator with Gemini 2.0 Flash."""
        super().__init__(
            name="Coordinator",
            model="gemini-2.0-flash",
            system_prompt=system_prompt,
            temperature=0.3,  # Slightly higher for intelligent routing decisions
            max_output_tokens=1000,
            reasoning_effort=None,
        )

        # Tool registry
        self.tools: Dict[str, ToolDefinition] = {}

        # State management
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.execution_log: List[Dict[str, Any]] = []

        # Routing rules: task type -> agent names
        self.routing_rules: Dict[TaskType, List[str]] = {
            TaskType.MENTORING: ["professor"],
            TaskType.CRITICAL_ANALYSIS: ["critic"],
            TaskType.PEER_REVIEW: ["reviewer"],
            TaskType.MULTI_AGENT: ["professor", "critic", "reviewer"],
            TaskType.DEBATE: ["professor", "critic"],
        }

        logger.info("Coordinator agent initialized with Gemini 2.0 Flash")

    def register_builtin_tools(self) -> None:
        """Register built-in tools for document parsing and system operations."""
        # Document parsing tools
        self.register_tool(
            name="parse_document",
            description="Parse a document file (PDF, LaTeX, Markdown, TXT) and extract content",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the document file to parse",
                    }
                },
                "required": ["file_path"],
            },
            handler=DocumentParser.parse_file,
        )

        # Session management tool
        self.register_tool(
            name="save_session",
            description="Save the current session state for persistence",
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Unique session ID"},
                    "state": {
                        "type": "object",
                        "description": "State data to save",
                    },
                },
                "required": ["session_id", "state"],
            },
            handler=self._save_session_handler,
        )

        logger.info("Built-in tools registered")

    async def _save_session_handler(
        self, session_id: str, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handler for saving session state."""
        self.update_session_state(session_id, state)
        return {"status": "success", "session_id": session_id}

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable,
    ) -> None:
        """
        Register a tool for the coordinator.

        Args:
            name: Tool name
            description: Tool description
            parameters: Parameter schema (JSON Schema format)
            handler: Async callable that implements the tool
        """
        tool = ToolDefinition(name, description, parameters, handler)
        self.tools[name] = tool
        logger.info(f"Registered tool: {name}")

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get all registered tools in API schema format."""
        return [tool.to_dict() for tool in self.tools.values()]

    async def route_task(
        self,
        task_input: str,
        task_type: TaskType = TaskType.MULTI_AGENT,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Intelligently route a task to appropriate agents.

        Args:
            task_input: The task description or content
            task_type: Type of task
            session_id: Optional session ID for state tracking

        Returns:
            Routing decision with selected agents and context
        """
        routing_prompt = f"""
        You are a task router. Analyze this task and determine optimal routing:

        TASK TYPE: {task_type.value}
        TASK INPUT: {task_input[:300]}...

        Available agents and their strengths:
        - professor: Mentoring, structure, methodology (Gemini 2.5 Pro, $0.60/M tokens)
        - critic: Critical analysis, flaws, assumptions (Gemini 2.5 Pro, $0.60/M tokens)
        - reviewer: Scoring, rubrics, verdicts (Gemini 2.0 Flash, $0.075/M tokens)

        Provide routing decision in JSON:
        {{
            "primary_agent": "name",
            "secondary_agents": ["names"],
            "reasoning": "why this routing",
            "estimated_cost": 0.00,
            "priority": "high|medium|low"
        }}
        """

        response = await self.generate_response(routing_prompt)

        # Parse routing decision
        routing = self._parse_routing_response(response)

        # Log execution
        self.execution_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type.value,
                "routing": routing,
                "session_id": session_id,
            }
        )

        return routing

    def _parse_routing_response(self, response: str) -> Dict[str, Any]:
        """Parse routing decision from response."""
        try:
            # Extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse routing response: {e}")

        # Fallback routing
        return {
            "primary_agent": "professor",
            "secondary_agents": [],
            "reasoning": "Default routing due to parse error",
            "estimated_cost": 0.10,
            "priority": "medium",
        }

    async def evaluate(self, thesis_content: str) -> Dict[str, Any]:
        """
        Coordinator evaluation: routes and orchestrates other agents.

        Args:
            thesis_content: Content to evaluate

        Returns:
            Coordination results
        """
        coordination_prompt = f"""
        As a coordinator, analyze this thesis and generate a high-level evaluation strategy:

        {thesis_content[:500]}...

        Provide:
        1. Recommended evaluation order
        2. Priority agents to engage
        3. Key areas to focus on
        4. Estimated effort level

        Format as JSON.
        """

        response = await self.generate_response(coordination_prompt)

        return {
            "agent": self.name,
            "timestamp": datetime.now().isoformat(),
            "coordination_strategy": response,
            "type": "coordination",
            "model_used": self.model,
            "metrics": self.get_metrics(),
        }

    def create_session(self, session_id: str, metadata: Dict[str, Any]) -> None:
        """
        Create a new tracking session.

        Args:
            session_id: Unique session identifier
            metadata: Optional metadata for the session
        """
        self.sessions[session_id] = {
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "evaluations": [],
            "state": {},
        }
        logger.info(f"Created session: {session_id}")

    def update_session_state(
        self, session_id: str, state_updates: Dict[str, Any]
    ) -> None:
        """
        Update state for a session.

        Args:
            session_id: Session identifier
            state_updates: Dictionary of state updates
        """
        if session_id not in self.sessions:
            self.create_session(session_id)

        self.sessions[session_id]["state"].update(state_updates)

    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Get current session state."""
        if session_id not in self.sessions:
            return {}
        return self.sessions[session_id]["state"]

    def log_evaluation(
        self, session_id: str, agent_name: str, result: Dict[str, Any]
    ) -> None:
        """
        Log an evaluation result to a session.

        Args:
            session_id: Session identifier
            agent_name: Name of agent that performed evaluation
            result: Evaluation result
        """
        if session_id not in self.sessions:
            self.create_session(session_id, metadata={})

        self.sessions[session_id]["evaluations"].append(
            {
                "agent": agent_name,
                "result": result,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of a session's evaluations."""
        if session_id not in self.sessions:
            return {}

        session = self.sessions[session_id]
        return {
            "session_id": session_id,
            "created_at": session["created_at"],
            "evaluation_count": len(session["evaluations"]),
            "agents_involved": list(
                set(e["agent"] for e in session["evaluations"])
            ),
            "metadata": session["metadata"],
        }

    async def call_tool(
        self, tool_name: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Call a registered tool.

        Args:
            tool_name: Name of the tool to call
            **kwargs: Tool parameters

        Returns:
            Tool result
        """
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        tool = self.tools[tool_name]
        start_time = datetime.now()

        try:
            result = await tool.handler(**kwargs)
            tool.call_count += 1
            tool.total_time += (datetime.now() - start_time).total_seconds()

            logger.info(f"Tool '{tool_name}' executed successfully")
            return {
                "tool": tool_name,
                "result": result,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Tool '{tool_name}' failed: {str(e)}")
            return {
                "tool": tool_name,
                "error": str(e),
                "success": False,
            }

    def get_tool_metrics(self) -> Dict[str, Any]:
        """Get metrics for all registered tools."""
        return {
            tool_name: {
                "call_count": tool.call_count,
                "total_time_seconds": tool.total_time,
                "average_time_seconds": (
                    tool.total_time / tool.call_count if tool.call_count > 0 else 0
                ),
            }
            for tool_name, tool in self.tools.items()
        }

    def get_execution_log(
        self, session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get execution log, optionally filtered by session.

        Args:
            session_id: Optional session filter

        Returns:
            List of execution log entries
        """
        if session_id:
            return [e for e in self.execution_log if e.get("session_id") == session_id]
        return self.execution_log

    def get_cost_estimate(self, agents_to_run: List[str]) -> Dict[str, Any]:
        """
        Estimate total cost for running specified agents.

        Args:
            agents_to_run: List of agent names

        Returns:
            Cost estimate dictionary
        """
        # Map agent names to models
        agent_models = {
            "professor": "gemini-2.5-pro",
            "critic": "gemini-2.5-pro",
            "reviewer": "gemini-2.0-flash",
        }

        # Estimate: 4000 output tokens per evaluation
        output_tokens = 4000
        input_tokens = 2000  # Average input

        total_cost = 0.0
        breakdown = {}

        for agent_name in agents_to_run:
            model = agent_models.get(agent_name, "gemini-2.0-flash")
            cost = AgentCostModel.estimate_cost(model, input_tokens, output_tokens)
            latency = AgentCostModel.estimate_latency(model)

            breakdown[agent_name] = {
                "model": model,
                "estimated_cost": round(cost, 4),
                "estimated_latency": latency,
            }

            total_cost += cost

        return {
            "total_estimated_cost": round(total_cost, 4),
            "agents": breakdown,
            "currency": "USD",
        }

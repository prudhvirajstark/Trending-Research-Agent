"""
Base agent wrapper around Google GenAI SDK

Provides abstract base class for all agents with common functionality:
- Model initialization with config
- Request/response handling
- Streaming support
- Error handling and retries
- Logging and telemetry hooks
"""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any, List, AsyncGenerator

import google.genai
from google.genai.types import Content, Part
from google.genai.errors import ClientError

from src.utils.rate_limiter import RateLimiter, TokenBudgetTracker
from src.models import HybridProvider, ModelProvider


logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all thesis evaluation agents."""

    # Shared rate limiter across all agents (free tier: 60 req/min, use conservative 30)
    _rate_limiter = RateLimiter(requests_per_minute=30, min_delay_seconds=2.0)

    # Shared token budget tracker for daily limits
    _token_budget = TokenBudgetTracker(daily_limit=1_500_000)

    # Shared model provider (initialized once, reused across agents)
    _model_provider: Optional[ModelProvider] = None

    def __init__(
        self,
        name: str,
        model: str,
        system_prompt: str,
        temperature: float = 0.7,
        max_output_tokens: int = 4000,
        reasoning_effort: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        """
        Initialize a base agent.

        Args:
            name: Agent identifier (e.g., "professor", "critic", "reviewer")
            model: Model to use (e.g., "gemini-2.5-pro" or "gemma3")
            system_prompt: System prompt defining agent behavior
            temperature: Sampling temperature (0.0-1.0)
            max_output_tokens: Maximum tokens in response
            reasoning_effort: Level of reasoning ("low", "medium", "high", or None)
            provider: Provider to use ("gemini", "ollama", or "hybrid"). Defaults to hybrid.
        """
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort

        # Initialize model provider (shared across all agents)
        if BaseAgent._model_provider is None:
            provider_type = provider or os.getenv("MODEL_PROVIDER", "hybrid")
            logger.info(f"Initializing model provider: {provider_type}")

            if provider_type == "hybrid":
                BaseAgent._model_provider = HybridProvider(
                    primary=os.getenv("PRIMARY_PROVIDER", "gemini"),
                    fallback_to_ollama=os.getenv("FALLBACK_TO_OLLAMA", "true").lower() == "true",
                    ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
                )
            else:
                raise ValueError(f"Unknown provider type: {provider_type}")

        self.provider = BaseAgent._model_provider

        # Execution tracking
        self.message_history: List[Dict[str, str]] = []
        self.last_execution_time: Optional[float] = None

        # Log model provider info
        available_models = self.provider.get_available_models()
        logger.info(
            f"Initialized agent '{self.name}' with model '{self.model}' "
            f"(temp={self.temperature}, reasoning={self.reasoning_effort}). "
            f"Available models: {', '.join(available_models[:3])}{'...' if len(available_models) > 3 else ''}"
        )

    async def generate_response(
        self,
        user_message: str,
        include_thinking: bool = False,
    ) -> str:
        """
        Generate a response from the agent.

        Args:
            user_message: Input message to the agent
            include_thinking: Whether to include thinking steps in response

        Returns:
            Generated response text
        """
        start_time = datetime.now()

        try:
            # Check token budget before making request
            remaining_budget = self._token_budget.get_remaining_budget()
            if remaining_budget < 100:
                logger.error(
                    f"Token budget exhausted: {remaining_budget} tokens remaining. "
                    f"Daily limit will reset at {self._token_budget.reset_time.isoformat()}"
                )
                raise RuntimeError(
                    f"Daily token budget exceeded. Remaining: {remaining_budget} tokens. "
                    f"Reset at {self._token_budget.reset_time.isoformat()}"
                )

            # Apply rate limiting (respects free tier request limits)
            logger.debug(f"Applying rate limit for {self.name}...")
            await self._rate_limiter.acquire()
            logger.debug(f"Rate limit acquired for {self.name}")

            # Build messages list with history
            messages = self._build_message_history(user_message)

            # Make API call using model provider (Gemini or Ollama)
            response_text = await self.provider.generate_content(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                reasoning_effort=self.reasoning_effort,
            )

            if not response_text:
                logger.warning(f"Empty response from {self.name}")
                return ""

            # Track token usage (estimate: ~1 token per 4 chars, simplified)
            estimated_tokens = len(user_message) // 4 + len(response_text) // 4
            self._token_budget.add_tokens(estimated_tokens)

            # Update message history
            self.message_history.append({"role": "user", "content": user_message})
            self.message_history.append({"role": "assistant", "content": response_text})

            # Track execution metrics
            self.last_execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Agent '{self.name}' generated response in {self.last_execution_time:.2f}s"
            )

            return response_text

        except ClientError as e:
            # Handle quota exhaustion (429) with 60-second wait and retry
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                logger.warning(
                    f"\n{'='*80}"
                )
                logger.warning(
                    "QUOTA EXHAUSTED - Free Tier Limit Hit (429 RESOURCE_EXHAUSTED)"
                )
                logger.warning(
                    f"Agent: {self.name}"
                )
                logger.warning(
                    f"Time: {datetime.now().isoformat()}"
                )
                logger.warning(
                    "Waiting 60 seconds for quota reset..."
                )
                logger.warning(
                    f"{'='*80}\n"
                )

                # Wait 60 seconds for quota to reset
                for i in range(60, 0, -1):
                    if i % 10 == 0 or i <= 5:
                        logger.info(f"Waiting: {i} seconds remaining...")
                    await asyncio.sleep(1)

                logger.info(
                    f"Quota wait complete. Retrying request for {self.name}..."
                )

                # Recursive retry after quota reset
                return await self.generate_response(user_message, include_thinking)
            else:
                # Other client errors
                logger.error(
                    f"API Error from {self.name}: {str(e)}"
                )
                raise

        except Exception as e:
            logger.error(
                f"Error generating response from {self.name}: {str(e)}"
            )
            raise

    async def generate_streaming_response(
        self,
        user_message: str,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from the agent.

        Args:
            user_message: Input message to the agent

        Yields:
            Chunks of generated text
        """
        try:
            messages = self._build_message_history(user_message)

            config = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
            }

            # Stream the response
            response = self.client.models.generate_content_stream(
                model=f"models/{self.model}",
                contents=messages,
                config=google.genai.types.GenerateContentConfig(**config),
            )

            full_response = ""
            async for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    yield chunk.text

            # Update history after streaming completes
            self.message_history.append({"role": "user", "content": user_message})
            self.message_history.append({"role": "assistant", "content": full_response})

        except Exception as e:
            logger.error(f"Error in streaming response from {self.name}: {str(e)}")
            raise

    def _build_message_history(self, user_message: str) -> List[Content]:
        """
        Build message list including system prompt and conversation history.

        Args:
            user_message: Current user message

        Returns:
            List of Content objects for the API
        """
        messages = []

        # Add system prompt as first message
        messages.append(
            Content(
                role="user",
                parts=[Part.from_text(text=self.system_prompt)],
            )
        )
        messages.append(
            Content(
                role="model",
                parts=[Part.from_text(text="Understood. I will follow these instructions.")],
            )
        )

        # Add conversation history
        for msg in self.message_history[-10:]:  # Keep last 10 messages
            role = "user" if msg["role"] == "user" else "model"
            messages.append(
                Content(
                    role=role,
                    parts=[Part.from_text(text=msg["content"])],
                )
            )

        # Add current message
        messages.append(
            Content(
                role="user",
                parts=[Part.from_text(text=user_message)],
            )
        )

        return messages

    def clear_history(self):
        """Clear conversation history."""
        self.message_history = []
        logger.info(f"Cleared message history for agent '{self.name}'")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for this agent.

        Returns:
            Dictionary with metrics including rate limiting info
        """
        return {
            "agent_name": self.name,
            "model": self.model,
            "last_execution_time_seconds": self.last_execution_time,
            "message_count": len(self.message_history),
            "temperature": self.temperature,
            "rate_limiter": self._rate_limiter.get_metrics(),
            "token_budget": self._token_budget.get_metrics(),
        }

    async def route_task(self, task: str) -> Dict[str, Any]:
        """
        Route a task to the appropriate agent based on task type.

        Args:
            task: Task description

        Returns:
            Dictionary with routing result
        """
        # Placeholder routing logic (to be implemented in subclasses)
        logger.info(f"Routing task '{task}' for agent '{self.name}'")
        return {"agent": self.name, "task": task, "status": "routed"}

    @abstractmethod
    async def evaluate(self, content: str) -> Dict[str, Any]:
        """
        Abstract method for agent-specific evaluation logic.

        Args:
            content: Content to evaluate

        Returns:
            Dictionary with evaluation results
        """
        pass

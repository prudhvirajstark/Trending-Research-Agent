"""
Model Provider Abstraction Layer

Supports multiple LLM providers (Gemini, Ollama, etc.) with unified interface.
Allows seamless switching between cloud and local models.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, List

import google.genai
from google.genai.types import Content


logger = logging.getLogger(__name__)


class ModelProvider(ABC):
    """Abstract base class for model providers."""

    @abstractmethod
    async def generate_content(
        self,
        model: str,
        messages: List[Content],
        temperature: float,
        max_output_tokens: int,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """
        Generate content using the provider's model.

        Args:
            model: Model identifier
            messages: List of Content messages
            temperature: Sampling temperature
            max_output_tokens: Maximum output tokens
            reasoning_effort: Level of reasoning effort (if supported)

        Returns:
            Generated response text
        """
        pass

    @abstractmethod
    def validate_model(self, model: str) -> bool:
        """Check if model is available."""
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        pass


class GeminiProvider(ModelProvider):
    """Google Gemini API provider."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini provider."""
        import os

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")

        self.client = google.genai.Client(api_key=self.api_key)
        logger.info("Initialized Gemini provider")

    async def generate_content(
        self,
        model: str,
        messages: List[Content],
        temperature: float,
        max_output_tokens: int,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """Generate content using Gemini."""
        config = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }

        # Add reasoning effort if specified
        if reasoning_effort:
            config["thinking"] = {
                "type": "enabled",
                "budget_tokens": 5000,
            }

        response = self.client.models.generate_content(
            model=f"models/{model}",
            contents=messages,
            config=google.genai.types.GenerateContentConfig(**config),
        )

        return response.text

    def validate_model(self, model: str) -> bool:
        """Check if Gemini model exists."""
        valid_models = ["gemini-2.0-flash", "gemini-2.5-pro", "gemini-1.5-pro"]
        return model in valid_models

    def get_available_models(self) -> List[str]:
        """Get available Gemini models."""
        return ["gemini-2.0-flash", "gemini-2.5-pro", "gemini-1.5-pro"]


class OllamaProvider(ModelProvider):
    """Ollama local model provider."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama provider.

        Args:
            base_url: Ollama server base URL
        """
        self.base_url = base_url
        self.available_models = []
        self._fetch_models()
        logger.info(f"Initialized Ollama provider at {base_url}")

    def _fetch_models(self) -> None:
        """Fetch available models from Ollama."""
        try:
            import requests

            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                data = response.json()
                self.available_models = [
                    model["name"] for model in data.get("models", [])
                ]
                logger.info(f"Found Ollama models: {self.available_models}")
            else:
                logger.warning("Could not fetch Ollama models")
        except Exception as e:
            logger.warning(f"Ollama connection error: {str(e)}")

    async def generate_content(
        self,
        model: str,
        messages: List[Content],
        temperature: float,
        max_output_tokens: int,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """Generate content using Ollama."""
        import aiohttp

        # Convert Content/Part messages to Ollama format
        ollama_messages = []
        for content in messages:
            role = "user" if content.role == "user" else "assistant"
            text = content.parts[0].text if content.parts else ""
            ollama_messages.append({"role": role, "content": text})

        payload = {
            "model": model,
            "messages": ollama_messages,
            "temperature": temperature,
            "stream": False,
        }

        # Ollama doesn't support max_output_tokens directly
        # It uses num_predict for max tokens
        if max_output_tokens:
            payload["options"] = {"num_predict": max_output_tokens}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("message", {}).get("content", "")
                    else:
                        error_text = await response.text()
                        raise RuntimeError(f"Ollama error: {response.status} - {error_text}")
        except asyncio.TimeoutError:
            raise RuntimeError("Ollama request timeout")

    def validate_model(self, model: str) -> bool:
        """Check if Ollama model is available."""
        # Handle both exact names and base names (e.g., "gemma3" vs "gemma3:latest")
        return any(
            model in m or m.startswith(model.split(":")[0])
            for m in self.available_models
        )

    def get_available_models(self) -> List[str]:
        """Get available Ollama models."""
        return self.available_models


class HybridProvider(ModelProvider):
    """Hybrid provider that routes to Gemini or Ollama based on configuration."""

    def __init__(
        self,
        primary: str = "gemini",
        fallback_to_ollama: bool = True,
        ollama_url: str = "http://localhost:11434",
    ):
        """
        Initialize hybrid provider.

        Args:
            primary: Primary provider ("gemini" or "ollama")
            fallback_to_ollama: Fallback to Ollama if Gemini fails
            ollama_url: Ollama server URL
        """
        self.primary = primary
        self.fallback_to_ollama = fallback_to_ollama

        self.providers = {}

        # Initialize Gemini provider
        try:
            self.providers["gemini"] = GeminiProvider()
        except ValueError as e:
            logger.warning(f"Gemini provider unavailable: {str(e)}")

        # Initialize Ollama provider
        try:
            self.providers["ollama"] = OllamaProvider(ollama_url)
        except Exception as e:
            logger.warning(f"Ollama provider unavailable: {str(e)}")

        if primary not in self.providers:
            if self.providers:
                self.primary = next(iter(self.providers.keys()))
                logger.info(f"Primary provider not available, using {self.primary}")
            else:
                raise RuntimeError("No model providers available")

        logger.info(f"Hybrid provider initialized. Primary: {self.primary}")

    async def generate_content(
        self,
        model: str,
        messages: List[Content],
        temperature: float,
        max_output_tokens: int,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """Generate content with fallback logic and intelligent model selection."""
        provider_name = self.primary
        last_error = None
        selected_model = model

        try:
            if provider_name in self.providers:
                provider = self.providers[provider_name]

                # Check if model exists in primary provider
                if not provider.validate_model(model):
                    # Model not available in primary provider, pick first available model
                    available = provider.get_available_models()
                    if available:
                        selected_model = available[0]
                        logger.warning(
                            f"Model {model} not available in {provider_name}. "
                            f"Using {selected_model} instead."
                        )
                    else:
                        logger.warning(f"No models available in {provider_name}")
                        last_error = f"No models available in {provider_name}"
                        raise ValueError(last_error)

                logger.info(f"Using {provider_name} for model {selected_model}")
                return await provider.generate_content(
                    selected_model, messages, temperature, max_output_tokens, reasoning_effort
                )
        except Exception as e:
            logger.warning(f"Error with {provider_name}: {str(e)}")
            last_error = str(e)

        # Try fallback provider
        if self.fallback_to_ollama:
            for alt_provider_name, provider in self.providers.items():
                if alt_provider_name == provider_name:
                    continue
                try:
                    # Try original model first in fallback
                    if provider.validate_model(model):
                        logger.info(
                            f"Falling back to {alt_provider_name} for model {model}"
                        )
                        return await provider.generate_content(
                            model,
                            messages,
                            temperature,
                            max_output_tokens,
                            reasoning_effort,
                        )
                    else:
                        # Model not available, pick first available from fallback
                        available = provider.get_available_models()
                        if available:
                            fallback_model = available[0]
                            logger.info(
                                f"Falling back to {alt_provider_name} with model {fallback_model}"
                            )
                            return await provider.generate_content(
                                fallback_model,
                                messages,
                                temperature,
                                max_output_tokens,
                                reasoning_effort,
                            )
                except Exception as e:
                    logger.warning(f"Error with fallback {alt_provider_name}: {str(e)}")
                    last_error = str(e)

        raise RuntimeError(
            f"Failed to generate content. Last error: {last_error}. "
            f"Available providers: {list(self.providers.keys())}"
        )

    def validate_model(self, model: str) -> bool:
        """Check if model is available in any provider."""
        return any(provider.validate_model(model) for provider in self.providers.values())

    def get_available_models(self) -> List[str]:
        """Get all available models from all providers."""
        models = {}
        for provider_name, provider in self.providers.items():
            for model in provider.get_available_models():
                if model not in models:
                    models[model] = []
                models[model].append(provider_name)

        result = []
        for model, providers in models.items():
            providers_str = "/".join(providers)
            result.append(f"{model} ({providers_str})")

        return result

    def set_primary_provider(self, provider: str) -> None:
        """Switch primary provider."""
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not available")
        self.primary = provider
        logger.info(f"Primary provider switched to {provider}")

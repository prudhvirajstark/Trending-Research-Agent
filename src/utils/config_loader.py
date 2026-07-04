"""
Configuration utilities for loading YAML configs and managing agent setup.
"""

import yaml
import logging
from typing import Dict, Any
from pathlib import Path


logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and manages configuration from YAML files."""

    def __init__(self, config_dir: str = "config"):
        """
        Initialize config loader.

        Args:
            config_dir: Path to configuration directory
        """
        self.config_dir = Path(config_dir)
        self.agent_prompts = None
        self.model_config = None

    def load_agent_prompts(self) -> Dict[str, Any]:
        """
        Load agent prompts configuration.

        Returns:
            Dictionary with agent configurations
        """
        if self.agent_prompts:
            return self.agent_prompts

        prompts_file = self.config_dir / "agent_prompts.yaml"

        if not prompts_file.exists():
            logger.warning(f"Agent prompts file not found: {prompts_file}")
            return {}

        with open(prompts_file, "r") as f:
            config = yaml.safe_load(f)

        self.agent_prompts = config.get("agents", {})
        logger.info(f"Loaded agent prompts for: {list(self.agent_prompts.keys())}")

        return self.agent_prompts

    def load_model_config(self) -> Dict[str, Any]:
        """
        Load model configuration.

        Returns:
            Dictionary with model configurations
        """
        if self.model_config:
            return self.model_config

        model_file = self.config_dir / "model_config.yaml"

        if not model_file.exists():
            logger.warning(f"Model config file not found: {model_file}")
            return {}

        with open(model_file, "r") as f:
            config = yaml.safe_load(f)

        self.model_config = config.get("models", {})
        logger.info(f"Loaded model configs for: {list(self.model_config.keys())}")

        return self.model_config

    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific agent.

        Args:
            agent_name: Name of the agent (professor, critic, reviewer)

        Returns:
            Agent configuration dictionary
        """
        prompts = self.load_agent_prompts()
        return prompts.get(agent_name, {})

    def get_model_config(self, model_alias: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model.

        Args:
            model_alias: Model alias (professor, critic, reviewer, coordinator)

        Returns:
            Model configuration dictionary
        """
        models = self.load_model_config()
        return models.get(model_alias, {})

    def get_all_configs(self) -> Dict[str, Any]:
        """
        Get all configurations.

        Returns:
            Combined configuration dictionary
        """
        return {
            "agents": self.load_agent_prompts(),
            "models": self.load_model_config(),
        }

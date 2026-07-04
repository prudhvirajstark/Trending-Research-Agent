"""Initialization for agents module"""

from .base_agent import BaseAgent
from .professor import ProfessorAgent
from .critic import CriticAgent
from .reviewer import ReviewerAgent
from .coordinator import CoordinatorAgent

__all__ = ["BaseAgent", "ProfessorAgent", "CriticAgent", "ReviewerAgent", "CoordinatorAgent"]


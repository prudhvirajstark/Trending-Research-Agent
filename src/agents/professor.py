"""
Professor Agent - The Mentor

Provides high-level guidance on structure, methodology, and academic rigor.
Uses Gemini 2.5 Pro for deep reasoning and large context windows.
"""

import logging
from typing import Dict, Any
from datetime import datetime

from .base_agent import BaseAgent


logger = logging.getLogger(__name__)


class ProfessorAgent(BaseAgent):
    """Agent that provides mentoring guidance on thesis structure and methodology."""

    def __init__(self, system_prompt: str):
        """Initialize the Professor agent with Gemini 2.5 Pro."""
        super().__init__(
            name="Professor",
            model="gemini-2.5-pro",
            system_prompt=system_prompt,
            temperature=0.2,  # Low temperature for consistent logical guidance
            max_output_tokens=4000,
            reasoning_effort=None,  # Standard reasoning sufficient for mentoring
        )

    async def evaluate(self, thesis_content: str) -> Dict[str, Any]:
        """
        Evaluate thesis content and provide mentoring guidance.

        Args:
            thesis_content: The thesis section or full content to evaluate

        Returns:
            Dictionary with guidance, suggestions, and structural feedback
        """
        evaluation_prompt = f"""
        Please review the following thesis content and provide comprehensive mentoring guidance:

        ---THESIS CONTENT---
        {thesis_content}
        ---END CONTENT---

        Provide your feedback in the following structured format:

        **STRENGTHS**:
        - [List key strengths]

        **AREAS FOR IMPROVEMENT**:
        - [List specific improvement areas]

        **STRUCTURAL SUGGESTIONS**:
        - [Suggest structural improvements]

        **LITERATURE GAPS**:
        - [Identify missing foundational works or citations]

        **METHODOLOGY VALIDATION**:
        - [Assess if methodology is sound and well-motivated]

        **HIGH-LEVEL GUIDANCE**:
        - [Provide high-level direction for improvement]

        **NEXT STEPS**:
        - [Recommend specific next steps for the student]
        """

        response_text = await self.generate_response(evaluation_prompt)

        return {
            "agent": self.name,
            "timestamp": datetime.now().isoformat(),
            "evaluation": response_text,
            "type": "mentoring_guidance",
            "model_used": self.model,
            "metrics": self.get_metrics(),
        }

    async def suggest_improvements(self, section: str, section_title: str) -> str:
        """
        Provide specific improvement suggestions for a thesis section.

        Args:
            section: The thesis section content
            section_title: Title/name of the section

        Returns:
            Structured improvement suggestions
        """
        prompt = f"""
        As a thesis professor, review this section: "{section_title}"

        {section}

        Provide:
        1. Key feedback on this section's contribution
        2. How it fits into the larger thesis narrative
        3. Specific citations or references that should be added
        4. Structural recommendations
        5. Questions the author should answer
        """

        return await self.generate_response(prompt)

    async def validate_methodology(self, methodology: str) -> Dict[str, Any]:
        """
        Validate the research methodology.

        Args:
            methodology: Description of the research methodology

        Returns:
            Validation results and suggestions
        """
        prompt = f"""
        Validate this research methodology as a thesis advisor:

        {methodology}

        Assess:
        1. Is the methodology sound and well-motivated?
        2. Are there potential threats to validity?
        3. Is the methodology appropriate for the research question?
        4. What assumptions are being made?
        5. Are there alternative methodologies to consider?

        Provide actionable feedback.
        """

        response = await self.generate_response(prompt)

        return {
            "agent": self.name,
            "validation_type": "methodology",
            "feedback": response,
            "timestamp": datetime.now().isoformat(),
        }

    async def identify_research_gap(self, literature_review: str) -> Dict[str, Any]:
        """
        Help identify and validate the research gap from the literature review.

        Args:
            literature_review: The literature review section

        Returns:
            Analysis of the research gap and its significance
        """
        prompt = f"""
        Based on this literature review, help identify the research gap:

        {literature_review}

        Analyze:
        1. What is the clear research gap being highlighted?
        2. Is this gap significant and worth researching?
        3. Is it appropriately motivated by the literature?
        4. What prior work is most relevant to this gap?
        5. How should the student position their contribution relative to existing work?

        Provide constructive guidance.
        """

        response = await self.generate_response(prompt)

        return {
            "agent": self.name,
            "analysis_type": "research_gap",
            "gap_analysis": response,
            "timestamp": datetime.now().isoformat(),
        }

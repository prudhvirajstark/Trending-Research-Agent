"""
Critic Agent - The Adversary

Finds flaws, challenges assumptions, and tests thesis limitations.
Uses Gemini 2.5 Pro with extended thinking for advanced reasoning.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from .base_agent import BaseAgent


logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """Agent that provides critical analysis and challenges thesis assumptions."""

    def __init__(self, system_prompt: str):
        """Initialize the Critic agent with Gemini 2.5 Pro + Thinking."""
        super().__init__(
            name="Critic",
            model="gemini-2.5-pro",
            system_prompt=system_prompt,
            temperature=0.5,  # Medium temperature for diverse counter-arguments
            max_output_tokens=4000,
            reasoning_effort="medium",  # Enable thinking for logical deconstruction
        )

    async def evaluate(self, thesis_content: str) -> Dict[str, Any]:
        """
        Perform critical analysis of thesis content.

        Args:
            thesis_content: The thesis section or full content to critique

        Returns:
            Dictionary with critical findings and challenges
        """
        evaluation_prompt = f"""
        Perform a rigorous critical analysis of this thesis content:

        ---THESIS CONTENT---
        {thesis_content}
        ---END CONTENT---

        Provide a critical analysis in this format:

        **ASSUMPTIONS CHALLENGED**:
        - [List each assumption and why it should be questioned]

        **LOGICAL FLAWS**:
        - [Identify gaps, inconsistencies, or weak reasoning]

        **EVIDENCE GAPS**:
        - [What claims lack sufficient evidence?]

        **METHODOLOGICAL CONCERNS**:
        - [Identify potential methodology problems]

        **COUNTERARGUMENTS**:
        - [Propose alternative interpretations or competing explanations]

        **UNADDRESSED LIMITATIONS**:
        - [What limitations does the thesis not acknowledge?]

        **CRITICAL VERDICT**:
        - [Summary of the severity and impact of critical issues found]

        Be sharp but fair. Force rigorous thinking about the work.
        """

        response_text = await self.generate_response(evaluation_prompt)

        return {
            "agent": self.name,
            "timestamp": datetime.now().isoformat(),
            "critical_analysis": response_text,
            "type": "critical_review",
            "model_used": self.model,
            "reasoning_enabled": True,
            "metrics": self.get_metrics(),
        }

    async def challenge_claims(self, claim: str, supporting_evidence: str) -> str:
        """
        Challenge a specific claim and its supporting evidence.

        Args:
            claim: The thesis claim to challenge
            supporting_evidence: Evidence provided for the claim

        Returns:
            Critical challenge and alternative interpretations
        """
        prompt = f"""
        I'm presenting this claim in my thesis:

        CLAIM: {claim}

        SUPPORTING EVIDENCE:
        {supporting_evidence}

        As a critical reviewer, challenge this claim by:
        1. Questioning the assumptions behind the claim
        2. Identifying alternative interpretations of the evidence
        3. Pointing out what evidence might contradict this claim
        4. Asking what would need to be true for this claim to be false
        5. Suggesting what evidence would strengthen or weaken the claim

        Be rigorous and specific.
        """

        return await self.generate_response(prompt)

    async def find_methodological_flaws(self, methodology: str) -> Dict[str, Any]:
        """
        Identify potential flaws in research methodology.

        Args:
            methodology: Description of the research methodology

        Returns:
            List of potential methodological issues
        """
        prompt = f"""
        As a critical peer reviewer, identify potential flaws in this methodology:

        {methodology}

        Analyze and challenge:
        1. Validity threats - What could make results invalid?
        2. Reliability concerns - Could results be replicated?
        3. Bias risks - What biases could affect the study?
        4. Generalizability - What are the limits of generalization?
        5. Design alternatives - What design choices are questionable?
        6. Missing controls - What variables should be controlled?

        Be specific about each flaw and its implications.
        """

        response = await self.generate_response(prompt)

        return {
            "agent": self.name,
            "analysis_type": "methodological_critique",
            "flaws_identified": response,
            "timestamp": datetime.now().isoformat(),
        }

    async def test_robustness(
        self,
        conclusion: str,
        evidence: str,
        assumptions: str,
    ) -> Dict[str, Any]:
        """
        Test the robustness of thesis conclusions against challenges.

        Args:
            conclusion: The thesis conclusion to test
            evidence: Supporting evidence for the conclusion
            assumptions: Stated or implied assumptions

        Returns:
            Robustness test results
        """
        prompt = f"""
        Test the robustness of this thesis conclusion:

        CONCLUSION: {conclusion}

        EVIDENCE: {evidence}

        ASSUMPTIONS: {assumptions}

        Robustness testing:
        1. What would falsify this conclusion?
        2. How sensitive is the conclusion to changes in assumptions?
        3. What alternative conclusions could equally explain the evidence?
        4. What additional evidence would strengthen the conclusion?
        5. Under what conditions would this conclusion NOT hold?
        6. Rate the conclusion's robustness (Fragile/Moderate/Robust) and explain why.

        Provide a realistic assessment.
        """

        response = await self.generate_response(prompt)

        return {
            "agent": self.name,
            "test_type": "robustness_test",
            "results": response,
            "timestamp": datetime.now().isoformat(),
        }

    async def identify_hidden_assumptions(self, text: str) -> List[Dict[str, str]]:
        """
        Identify hidden or implicit assumptions in thesis text.

        Args:
            text: Thesis section or full content

        Returns:
            List of identified assumptions with explanations
        """
        prompt = f"""
        Identify all hidden or implicit assumptions in this text:

        {text}

        For each assumption, provide:
        1. The assumption (state it explicitly)
        2. Where it appears or is implied
        3. Why it matters
        4. How it affects the conclusions
        5. Whether it's reasonable or needs challenging

        Focus on non-obvious, foundational assumptions.
        """

        response = await self.generate_response(prompt)

        return [{
            "agent": self.name,
            "analysis_type": "hidden_assumptions",
            "assumptions_found": response,
            "timestamp": datetime.now().isoformat(),
        }]

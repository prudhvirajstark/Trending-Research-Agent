"""
Reviewer Agent - The Peer Evaluator

Scores thesis based on journal/conference standards using structured rubrics.
Uses Gemini 2.0 Flash for fast, deterministic evaluation.
"""

import logging
from typing import Dict, Any
from datetime import datetime
from enum import Enum

from .base_agent import BaseAgent


logger = logging.getLogger(__name__)


class ReviewVerdict(str, Enum):
    """Possible review verdicts."""

    ACCEPT = "Accept"
    MAJOR_REVISION = "Major Revision"
    REJECT = "Reject"


class ReviewerAgent(BaseAgent):
    """Agent that evaluates thesis against journal/conference standards."""

    # Rubric scoring criteria
    RUBRIC_CRITERIA = {
        "originality": "Does the work present novel insights or methods?",
        "clarity": "Is the writing clear, well-organized, and accessible?",
        "methodology": "Is the research design sound and reproducible?",
        "references": "Are citations appropriate and comprehensive?",
        "impact": "Will this work advance the field?",
        "ethics": "Are ethical guidelines followed?",
    }

    # Verdict thresholds
    VERDICT_THRESHOLDS = {
        "accept": 80,  # Average score >= 80
        "major_revision": 60,  # Average score 60-79
        "reject": 0,  # Average score < 60
    }

    def __init__(self, system_prompt: str):
        """Initialize the Reviewer agent with Gemini 2.0 Flash."""
        super().__init__(
            name="Reviewer",
            model="gemini-2.0-flash",
            system_prompt=system_prompt,
            temperature=0.0,  # Zero temperature for consistent, deterministic scoring
            max_output_tokens=2000,
            reasoning_effort=None,
        )

    async def evaluate(self, thesis_content: str) -> Dict[str, Any]:
        """
        Perform comprehensive peer review evaluation.

        Args:
            thesis_content: The thesis section or full content to review

        Returns:
            Structured review with scores and verdict
        """
        evaluation_prompt = f"""
        You are performing a peer review. Evaluate this thesis content:

        ---THESIS CONTENT---
        {thesis_content}
        ---END CONTENT---

        Score each criterion on a scale of 0-10:

        1. Originality (0-10): {self.RUBRIC_CRITERIA['originality']}
        2. Clarity (0-10): {self.RUBRIC_CRITERIA['clarity']}
        3. Methodology (0-10): {self.RUBRIC_CRITERIA['methodology']}
        4. References (0-10): {self.RUBRIC_CRITERIA['references']}
        5. Impact (0-10): {self.RUBRIC_CRITERIA['impact']}
        6. Ethics (0-10): {self.RUBRIC_CRITERIA['ethics']}

        Format your response EXACTLY as:
        ORIGINALITY: [score]
        CLARITY: [score]
        METHODOLOGY: [score]
        REFERENCES: [score]
        IMPACT: [score]
        ETHICS: [score]
        FEEDBACK: [detailed feedback on each criterion]

        Then provide a VERDICT: (Accept / Major Revision / Reject)
        with JUSTIFICATION: [explain the verdict]
        """

        response_text = await self.generate_response(evaluation_prompt)

        # Parse the structured response
        review_data = self._parse_review_response(response_text)

        return {
            "agent": self.name,
            "timestamp": datetime.now().isoformat(),
            "review": review_data,
            "type": "peer_review",
            "model_used": self.model,
            "metrics": self.get_metrics(),
        }

    def _parse_review_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse structured review response into scores and verdict.

        Args:
            response_text: Raw response from the reviewer

        Returns:
            Structured review data
        """
        scores = {}
        feedback = ""
        verdict = ReviewVerdict.MAJOR_REVISION
        justification = ""

        lines = response_text.split("\n")

        for line in lines:
            line = line.strip()

            if line.startswith("ORIGINALITY:"):
                try:
                    scores["originality"] = int(
                        line.split(":")[1].strip().split()[0]
                    )
                except (IndexError, ValueError):
                    scores["originality"] = 5

            elif line.startswith("CLARITY:"):
                try:
                    scores["clarity"] = int(line.split(":")[1].strip().split()[0])
                except (IndexError, ValueError):
                    scores["clarity"] = 5

            elif line.startswith("METHODOLOGY:"):
                try:
                    scores["methodology"] = int(
                        line.split(":")[1].strip().split()[0]
                    )
                except (IndexError, ValueError):
                    scores["methodology"] = 5

            elif line.startswith("REFERENCES:"):
                try:
                    scores["references"] = int(
                        line.split(":")[1].strip().split()[0]
                    )
                except (IndexError, ValueError):
                    scores["references"] = 5

            elif line.startswith("IMPACT:"):
                try:
                    scores["impact"] = int(line.split(":")[1].strip().split()[0])
                except (IndexError, ValueError):
                    scores["impact"] = 5

            elif line.startswith("ETHICS:"):
                try:
                    scores["ethics"] = int(line.split(":")[1].strip().split()[0])
                except (IndexError, ValueError):
                    scores["ethics"] = 5

            elif line.startswith("FEEDBACK:"):
                feedback = "\n".join(lines[lines.index(line) :])
                feedback = feedback.replace("FEEDBACK:", "").strip()
                break

            elif line.startswith("VERDICT:"):
                verdict_text = line.split(":")[1].strip()
                if "accept" in verdict_text.lower():
                    verdict = ReviewVerdict.ACCEPT
                elif "reject" in verdict_text.lower():
                    verdict = ReviewVerdict.REJECT
                else:
                    verdict = ReviewVerdict.MAJOR_REVISION

            elif line.startswith("JUSTIFICATION:"):
                justification = line.split(":", 1)[1].strip()

        # Calculate average score
        if scores:
            average_score = sum(scores.values()) / len(scores)
        else:
            average_score = 50

        # Determine verdict based on score if not explicitly stated
        if not verdict or verdict == ReviewVerdict.MAJOR_REVISION:
            if average_score >= self.VERDICT_THRESHOLDS["accept"]:
                verdict = ReviewVerdict.ACCEPT
            elif average_score >= self.VERDICT_THRESHOLDS["major_revision"]:
                verdict = ReviewVerdict.MAJOR_REVISION
            else:
                verdict = ReviewVerdict.REJECT

        return {
            "scores": scores,
            "average_score": round(average_score, 2),
            "verdict": verdict.value,
            "feedback": feedback,
            "justification": justification,
        }

    async def score_criterion(self, content: str, criterion: str) -> Dict[str, Any]:
        """
        Score a specific evaluation criterion.

        Args:
            content: Thesis content to evaluate
            criterion: Which criterion to score (from RUBRIC_CRITERIA)

        Returns:
            Score and detailed feedback for the criterion
        """
        if criterion not in self.RUBRIC_CRITERIA:
            raise ValueError(
                f"Unknown criterion: {criterion}. Must be one of {list(self.RUBRIC_CRITERIA.keys())}"
            )

        prompt = f"""
        Score this thesis content on the criterion: "{criterion}"

        Definition: {self.RUBRIC_CRITERIA[criterion]}

        CONTENT:
        {content}

        Score on 0-10 scale where:
        0-3: Major deficiencies
        4-6: Moderate quality with significant issues
        7-8: Good quality with minor issues
        9-10: Excellent quality

        Provide your response in format:
        SCORE: [0-10]
        JUSTIFICATION: [explain the score]
        SPECIFIC_FEEDBACK: [actionable feedback]
        """

        response = await self.generate_response(prompt)

        # Parse response
        score = 5  # Default
        justification = ""
        feedback = ""

        lines = response.split("\n")
        for line in lines:
            if line.startswith("SCORE:"):
                try:
                    score = int(line.split(":")[1].strip().split()[0])
                except (IndexError, ValueError):
                    pass
            elif line.startswith("JUSTIFICATION:"):
                justification = line.split(":", 1)[1].strip()
            elif line.startswith("SPECIFIC_FEEDBACK:"):
                feedback = line.split(":", 1)[1].strip()

        return {
            "agent": self.name,
            "criterion": criterion,
            "score": score,
            "justification": justification,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat(),
        }

    async def assess_ethics_compliance(self, methodology: str) -> Dict[str, Any]:
        """
        Assess ethical compliance of research methodology.

        Args:
            methodology: Description of the research methodology

        Returns:
            Ethics assessment with compliance score and issues
        """
        prompt = f"""
        Assess ethical compliance for this research methodology:

        {methodology}

        Check for:
        1. Informed consent procedures
        2. Data privacy and protection measures
        3. Potential harm to participants
        4. Conflict of interest disclosures
        5. Institutional review board (IRB) compliance
        6. Responsible research practices
        7. Data retention and deletion policies

        Compliance Score (0-10): [score]
        Issues Found: [list any ethical issues]
        Recommendations: [specific recommendations to improve ethics]
        """

        response = await self.generate_response(prompt)

        return {
            "agent": self.name,
            "assessment_type": "ethics_compliance",
            "assessment": response,
            "timestamp": datetime.now().isoformat(),
        }

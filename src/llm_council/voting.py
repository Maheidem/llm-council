"""Deterministic voting machine for LLM Council.

This module provides structured vote parsing and tallying with NO LLM interpretation.
All vote processing is done via regex parsing and pure Python computation.
"""

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .models import Vote, VoteChoice, ConsensusType

logger = logging.getLogger(__name__)


class VoteParseError(Exception):
    """Raised when a vote cannot be parsed."""
    pass


@dataclass
class StructuredVote:
    """A vote parsed from structured format.

    Expected input format:
        [VOTE] AGREE
        [CONFIDENCE] 0.8
        [REASONING] The proposal addresses core concerns.

    Or simplified:
        VOTE: AGREE
        CONFIDENCE: 0.8
        REASON: The proposal addresses core concerns.
    """
    persona_name: str
    choice: VoteChoice
    confidence: float = 0.5  # 0.0 to 1.0
    reasoning: str = ""
    raw_response: str = ""
    parse_success: bool = True
    parse_errors: list[str] = field(default_factory=list)


class VoteParser:
    """Parses structured votes from LLM responses.

    Supports multiple formats for robustness:
    - [VOTE] AGREE / [CONFIDENCE] 0.8 / [REASONING] ...
    - VOTE: AGREE / CONFIDENCE: 0.8 / REASON: ...
    - Simple keywords: AGREE, DISAGREE, ABSTAIN anywhere in text
    """

    # Patterns for structured format
    VOTE_PATTERNS = [
        # [VOTE] AGREE format
        r'\[VOTE\]\s*(AGREE|DISAGREE|ABSTAIN)',
        # VOTE: AGREE format
        r'VOTE:\s*(AGREE|DISAGREE|ABSTAIN)',
        # **VOTE:** format (markdown)
        r'\*\*VOTE:?\*\*\s*(AGREE|DISAGREE|ABSTAIN)',
        # Standalone keywords (fallback)
        r'^(AGREE|DISAGREE|ABSTAIN)\b',
    ]

    CONFIDENCE_PATTERNS = [
        r'\[CONFIDENCE\]\s*([\d.]+)',
        r'CONFIDENCE:\s*([\d.]+)',
        r'\*\*CONFIDENCE:?\*\*\s*([\d.]+)',
    ]

    REASONING_PATTERNS = [
        r'\[REASONING\]\s*(.+?)(?=\[|$)',
        r'REASON(?:ING)?:\s*(.+?)(?=\n[A-Z\[\*]|$)',
        r'\*\*REASON(?:ING)?:?\*\*\s*(.+?)(?=\n\*\*|$)',
    ]

    @classmethod
    def parse(cls, persona_name: str, response: str) -> StructuredVote:
        """Parse a vote from LLM response text.

        Args:
            persona_name: Name of the voting persona
            response: Raw LLM response text

        Returns:
            StructuredVote with parsed data or defaults on failure
        """
        response_upper = response.upper()
        errors = []

        # Parse vote choice
        choice = None
        for pattern in cls.VOTE_PATTERNS:
            match = re.search(pattern, response_upper, re.MULTILINE | re.IGNORECASE)
            if match:
                choice_str = match.group(1).upper()
                choice = VoteChoice[choice_str]
                logger.debug(f"Parsed vote choice '{choice_str}' using pattern: {pattern}")
                break

        # Fallback: look for keywords anywhere
        if choice is None:
            if 'DISAGREE' in response_upper:
                choice = VoteChoice.DISAGREE
            elif 'AGREE' in response_upper:
                choice = VoteChoice.AGREE
            elif 'ABSTAIN' in response_upper:
                choice = VoteChoice.ABSTAIN
            else:
                choice = VoteChoice.ABSTAIN
                errors.append("Could not parse vote choice, defaulting to ABSTAIN")

        # Parse confidence
        confidence = 0.5  # Default
        for pattern in cls.CONFIDENCE_PATTERNS:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    confidence = float(match.group(1))
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                    break
                except ValueError:
                    errors.append(f"Could not parse confidence value: {match.group(1)}")

        # Parse reasoning
        reasoning = ""
        for pattern in cls.REASONING_PATTERNS:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                reasoning = match.group(1).strip()[:500]  # Limit length
                break

        # Fallback: use everything after the vote choice
        if not reasoning and choice:
            # Take text after the vote keyword
            vote_idx = response_upper.find(choice.value.upper())
            if vote_idx >= 0:
                reasoning = response[vote_idx + len(choice.value):].strip()[:500]

        return StructuredVote(
            persona_name=persona_name,
            choice=choice,
            confidence=confidence,
            reasoning=reasoning,
            raw_response=response,
            parse_success=len(errors) == 0,
            parse_errors=errors,
        )

    @classmethod
    def to_legacy_vote(cls, structured: StructuredVote) -> Vote:
        """Convert StructuredVote to legacy Vote model."""
        return Vote(
            persona_name=structured.persona_name,
            choice=structured.choice,
            reasoning=structured.reasoning,
        )


@dataclass
class VoteTally:
    """Result of deterministic vote counting."""
    agree_count: int = 0
    disagree_count: int = 0
    abstain_count: int = 0
    total_votes: int = 0
    total_voting: int = 0  # Excludes abstentions
    agree_ratio: float = 0.0
    weighted_agree: float = 0.0  # Confidence-weighted
    weighted_disagree: float = 0.0
    consensus_reached: bool = False
    consensus_type: Optional[ConsensusType] = None
    winning_choice: Optional[VoteChoice] = None


class VotingMachine:
    """Deterministic voting machine.

    Tallies votes using pure Python computation with no LLM interpretation.
    All thresholds and rules are explicitly defined.
    """

    # Consensus thresholds
    THRESHOLDS = {
        ConsensusType.UNANIMOUS: 1.0,
        ConsensusType.SUPERMAJORITY: 2/3,
        ConsensusType.MAJORITY: 0.5,
        ConsensusType.PLURALITY: 0.0,
    }

    def __init__(self, consensus_type: ConsensusType = ConsensusType.MAJORITY):
        """Initialize voting machine.

        Args:
            consensus_type: Type of consensus required for approval
        """
        self.consensus_type = consensus_type
        self.threshold = self.THRESHOLDS[consensus_type]

    def tally(self, votes: list[StructuredVote]) -> VoteTally:
        """Tally votes and determine outcome.

        Args:
            votes: List of structured votes

        Returns:
            VoteTally with deterministic results
        """
        tally = VoteTally(
            total_votes=len(votes),
            consensus_type=self.consensus_type,
        )

        for vote in votes:
            if vote.choice == VoteChoice.AGREE:
                tally.agree_count += 1
                tally.weighted_agree += vote.confidence
            elif vote.choice == VoteChoice.DISAGREE:
                tally.disagree_count += 1
                tally.weighted_disagree += vote.confidence
            else:  # ABSTAIN
                tally.abstain_count += 1

        tally.total_voting = tally.agree_count + tally.disagree_count

        # Calculate ratio (avoid division by zero)
        if tally.total_voting > 0:
            tally.agree_ratio = tally.agree_count / tally.total_voting

        # Determine winning choice
        if tally.agree_count > tally.disagree_count:
            tally.winning_choice = VoteChoice.AGREE
        elif tally.disagree_count > tally.agree_count:
            tally.winning_choice = VoteChoice.DISAGREE
        else:
            tally.winning_choice = None  # Tie

        # Check consensus based on type
        if tally.total_voting == 0:
            tally.consensus_reached = False
        elif self.consensus_type == ConsensusType.UNANIMOUS:
            tally.consensus_reached = tally.agree_ratio == 1.0
        elif self.consensus_type == ConsensusType.PLURALITY:
            # Plurality: most votes wins (agree > disagree)
            tally.consensus_reached = tally.winning_choice == VoteChoice.AGREE
        else:
            # Majority/Supermajority: agree ratio exceeds threshold
            tally.consensus_reached = tally.agree_ratio > self.threshold

        logger.info(
            f"Vote tally: {tally.agree_count} agree, {tally.disagree_count} disagree, "
            f"{tally.abstain_count} abstain. Ratio: {tally.agree_ratio:.2%}. "
            f"Consensus ({self.consensus_type.value}): {tally.consensus_reached}"
        )

        return tally

    def to_dict(self, tally: VoteTally) -> dict:
        """Convert tally to dictionary for JSON output."""
        return {
            "agree_count": tally.agree_count,
            "disagree_count": tally.disagree_count,
            "abstain_count": tally.abstain_count,
            "total_votes": tally.total_votes,
            "total_voting": tally.total_voting,
            "agree_ratio": round(tally.agree_ratio, 4),
            "weighted_agree": round(tally.weighted_agree, 4),
            "weighted_disagree": round(tally.weighted_disagree, 4),
            "consensus_reached": tally.consensus_reached,
            "consensus_type": tally.consensus_type.value if tally.consensus_type else None,
            "winning_choice": tally.winning_choice.value if tally.winning_choice else None,
        }


# Vote prompt template for instructing LLMs to output structured votes
VOTE_PROMPT_TEMPLATE = """Based on your perspective and the discussion, cast your vote on the proposal.

PROPOSAL: {proposal}

You MUST respond in this EXACT format:
[VOTE] AGREE or DISAGREE or ABSTAIN
[CONFIDENCE] 0.0 to 1.0 (how confident you are)
[REASONING] Your reasoning in 1-2 sentences

Example:
[VOTE] AGREE
[CONFIDENCE] 0.85
[REASONING] The proposal aligns with practical implementation concerns and addresses key risks.
"""

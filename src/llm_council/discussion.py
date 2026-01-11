"""Discussion protocol types for LLM Council.

This module provides structured response parsing for persona contributions,
including PASS mechanism and response type classification.
"""

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ResponseType(Enum):
    """Type of response from a persona."""
    CONTRIBUTION = "contribution"  # Active discussion contribution
    PASS = "pass"                  # Persona defers, no new input
    CALL_VOTE = "call_vote"        # Request to initiate voting
    QUESTION = "question"          # Asking for clarification


class DiscussionPhase(Enum):
    """Current phase of the discussion."""
    OPENING = "opening"            # Initial statements
    DELIBERATION = "deliberation"  # Active discussion
    SYNTHESIS = "synthesis"        # Working toward agreement
    VOTING = "voting"              # Formal vote in progress
    CLOSED = "closed"              # Discussion complete


@dataclass
class PersonaResponse:
    """A parsed response from a persona.

    Expected formats:
        [PASS] I concur with the previous speakers.
        [CONTRIBUTION] My analysis shows that...
        [CALL_VOTE] I believe we've reached sufficient agreement.

    Or naturally embedded:
        I'll pass on this round as I agree with what's been said.
        Let's call a vote - I think we have consensus.
    """
    persona_name: str
    response_type: ResponseType
    content: str
    raw_response: str = ""
    is_mediator: bool = False
    pass_reason: Optional[str] = None
    vote_proposal: Optional[str] = None


class ResponseParser:
    """Parses persona responses to detect PASS and other directives."""

    # Patterns for explicit markers
    PASS_PATTERNS = [
        r'^\s*\[PASS\]',
        r'^\s*PASS:',
        r'\bI(?:\'ll| will)? pass\b',
        r'\bpassing on this\b',
        r'\bnothing (?:more )?to add\b',
        r'\bI concur\b.*\bnothing\b',
        r'\bdefer(?:ring)? to\b',
    ]

    CALL_VOTE_PATTERNS = [
        r'\[CALL_VOTE\]',
        r'\[ACTION\]\s*call_vote',
        r'call(?:ing)? (?:a |for )?vote',
        r'let\'?s? vote\b',
        r'ready to vote\b',
        r'move to vote\b',
    ]

    QUESTION_PATTERNS = [
        r'\[QUESTION\]',
        r'(?:can|could|would) (?:you|someone) (?:please )?(?:clarify|explain)',
        r'I(?:\'d| would) like to (?:ask|understand)',
    ]

    @classmethod
    def parse(cls, persona_name: str, response: str, is_mediator: bool = False) -> PersonaResponse:
        """Parse a persona response to determine type and extract content.

        Args:
            persona_name: Name of the responding persona
            response: Raw response text
            is_mediator: Whether this persona is the mediator

        Returns:
            PersonaResponse with parsed data
        """
        response_stripped = response.strip()
        response_lower = response.lower()

        # Check for PASS
        for pattern in cls.PASS_PATTERNS:
            if re.search(pattern, response_stripped, re.IGNORECASE):
                # Extract reason after PASS marker
                reason = re.sub(r'^\s*\[?PASS\]?:?\s*', '', response_stripped, flags=re.IGNORECASE)
                logger.debug(f"{persona_name} is passing: {reason[:50]}...")
                return PersonaResponse(
                    persona_name=persona_name,
                    response_type=ResponseType.PASS,
                    content=reason,
                    raw_response=response,
                    is_mediator=is_mediator,
                    pass_reason=reason,
                )

        # Check for CALL_VOTE (mediators more likely)
        for pattern in cls.CALL_VOTE_PATTERNS:
            if re.search(pattern, response_stripped, re.IGNORECASE):
                # Extract proposal if present
                proposal_match = re.search(
                    r'\[PROPOSAL\]\s*(.+?)(?=\[|$)',
                    response,
                    re.IGNORECASE | re.DOTALL
                )
                proposal = proposal_match.group(1).strip() if proposal_match else None
                logger.debug(f"{persona_name} calling vote: {proposal[:50] if proposal else 'no proposal'}...")
                return PersonaResponse(
                    persona_name=persona_name,
                    response_type=ResponseType.CALL_VOTE,
                    content=response_stripped,
                    raw_response=response,
                    is_mediator=is_mediator,
                    vote_proposal=proposal,
                )

        # Check for QUESTION
        for pattern in cls.QUESTION_PATTERNS:
            if re.search(pattern, response_stripped, re.IGNORECASE):
                return PersonaResponse(
                    persona_name=persona_name,
                    response_type=ResponseType.QUESTION,
                    content=response_stripped,
                    raw_response=response,
                    is_mediator=is_mediator,
                )

        # Default: regular contribution
        return PersonaResponse(
            persona_name=persona_name,
            response_type=ResponseType.CONTRIBUTION,
            content=response_stripped,
            raw_response=response,
            is_mediator=is_mediator,
        )


@dataclass
class DiscussionState:
    """Tracks the current state of the discussion."""
    phase: DiscussionPhase = DiscussionPhase.OPENING
    round_number: int = 0
    consecutive_passes: dict[str, int] = field(default_factory=dict)
    total_passes: int = 0
    total_contributions: int = 0
    vote_called: bool = False
    vote_caller: Optional[str] = None
    current_proposal: Optional[str] = None

    def record_response(self, response: PersonaResponse):
        """Record a response and update state."""
        if response.response_type == ResponseType.PASS:
            self.consecutive_passes[response.persona_name] = \
                self.consecutive_passes.get(response.persona_name, 0) + 1
            self.total_passes += 1
        else:
            self.consecutive_passes[response.persona_name] = 0
            self.total_contributions += 1

        if response.response_type == ResponseType.CALL_VOTE:
            self.vote_called = True
            self.vote_caller = response.persona_name
            if response.vote_proposal:
                self.current_proposal = response.vote_proposal
            self.phase = DiscussionPhase.VOTING

    def should_auto_vote(self, total_personas: int, pass_threshold: float = 0.5) -> bool:
        """Check if we should automatically call a vote based on PASS rate."""
        if self.total_passes + self.total_contributions == 0:
            return False
        pass_rate = self.total_passes / (self.total_passes + self.total_contributions)
        return pass_rate >= pass_threshold

    def advance_round(self):
        """Advance to next round."""
        self.round_number += 1
        if self.round_number == 1:
            self.phase = DiscussionPhase.OPENING
        elif self.round_number <= 2:
            self.phase = DiscussionPhase.DELIBERATION
        else:
            self.phase = DiscussionPhase.SYNTHESIS


# Prompt additions for PASS mechanism
PASS_INSTRUCTION = """You may respond with [PASS] if you have nothing new to add and agree with what's been said.
Format: [PASS] <brief reason>

Example:
[PASS] I concur with The Pragmatist's risk assessment and have nothing to add."""

CONTRIBUTION_INSTRUCTION = """Respond with your perspective on the discussion. Stay in character.
If you're calling for a vote, use: [CALL_VOTE] <your reasoning>"""

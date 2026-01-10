"""Data models for LLM Council."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ConsensusType(Enum):
    """Type of consensus required."""
    UNANIMOUS = "unanimous"
    SUPERMAJORITY = "supermajority"  # 2/3
    MAJORITY = "majority"  # >50%
    PLURALITY = "plurality"  # Most votes wins


class VoteChoice(Enum):
    """Voting options."""
    AGREE = "agree"
    DISAGREE = "disagree"
    ABSTAIN = "abstain"


@dataclass
class Persona:
    """Represents an AI persona in the council."""
    name: str
    role: str
    expertise: list[str]
    personality_traits: list[str]
    perspective: str  # General viewpoint/bias this persona brings

    def to_system_prompt(self) -> str:
        """Generate system prompt for this persona."""
        traits = ", ".join(self.personality_traits)
        expertise = ", ".join(self.expertise)
        return f"""You are {self.name}, a {self.role}.

Your areas of expertise: {expertise}
Your personality traits: {traits}
Your perspective: {self.perspective}

You are participating in a council discussion. Stay in character and provide insights based on your unique perspective and expertise. Be constructive but also challenge ideas when appropriate based on your role."""


@dataclass
class Message:
    """A message in the discussion."""
    persona_name: str
    content: str
    round_number: int
    message_type: str = "discussion"  # discussion, vote, summary


@dataclass
class Vote:
    """A vote cast by a persona."""
    persona_name: str
    choice: VoteChoice
    reasoning: str


@dataclass
class RoundResult:
    """Result of a discussion round."""
    round_number: int
    messages: list[Message]
    consensus_reached: bool
    consensus_position: Optional[str] = None
    votes: list[Vote] = field(default_factory=list)


@dataclass
class CouncilSession:
    """A complete council session."""
    topic: str
    objective: str
    personas: list[Persona]
    rounds: list[RoundResult] = field(default_factory=list)
    final_consensus: Optional[str] = None
    consensus_reached: bool = False

    def to_dict(self) -> dict:
        """Convert session to dictionary for JSON output."""
        return {
            "topic": self.topic,
            "objective": self.objective,
            "personas": [
                {
                    "name": p.name,
                    "role": p.role,
                    "expertise": p.expertise,
                    "personality_traits": p.personality_traits,
                    "perspective": p.perspective,
                }
                for p in self.personas
            ],
            "rounds": [
                {
                    "round_number": r.round_number,
                    "messages": [
                        {
                            "persona_name": m.persona_name,
                            "content": m.content,
                            "round_number": m.round_number,
                            "message_type": m.message_type,
                        }
                        for m in r.messages
                    ],
                    "consensus_reached": r.consensus_reached,
                    "consensus_position": r.consensus_position,
                    "votes": [
                        {
                            "persona_name": v.persona_name,
                            "choice": v.choice.value,
                            "reasoning": v.reasoning,
                        }
                        for v in r.votes
                    ],
                }
                for r in self.rounds
            ],
            "final_consensus": self.final_consensus,
            "consensus_reached": self.consensus_reached,
        }


# Default personas for common use cases
DEFAULT_PERSONAS = [
    Persona(
        name="The Pragmatist",
        role="Practical Implementation Expert",
        expertise=["project management", "resource optimization", "risk assessment"],
        personality_traits=["practical", "results-oriented", "cautious"],
        perspective="Focus on what's achievable with current resources and constraints",
    ),
    Persona(
        name="The Innovator",
        role="Creative Solutions Architect",
        expertise=["emerging technologies", "creative problem-solving", "disruption"],
        personality_traits=["visionary", "optimistic", "unconventional"],
        perspective="Push boundaries and explore novel approaches",
    ),
    Persona(
        name="The Critic",
        role="Devil's Advocate",
        expertise=["risk analysis", "failure modes", "quality assurance"],
        personality_traits=["skeptical", "analytical", "thorough"],
        perspective="Identify weaknesses, risks, and potential failures",
    ),
    Persona(
        name="The Diplomat",
        role="Consensus Builder",
        expertise=["stakeholder management", "communication", "conflict resolution"],
        personality_traits=["empathetic", "balanced", "inclusive"],
        perspective="Find common ground and ensure all viewpoints are heard",
    ),
    Persona(
        name="The Specialist",
        role="Domain Expert",
        expertise=["technical depth", "best practices", "industry standards"],
        personality_traits=["precise", "knowledgeable", "methodical"],
        perspective="Ensure technical accuracy and adherence to standards",
    ),
]

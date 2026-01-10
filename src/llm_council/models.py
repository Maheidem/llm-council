"""Data models for LLM Council."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any


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
class PersonaProviderConfig:
    """Provider configuration specific to a persona.

    All fields are optional - unset fields inherit from global defaults.
    """
    model: Optional[str] = None
    provider: Optional[str] = None  # Named provider reference
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    context_size: Optional[int] = None
    response_size: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in {
            "model": self.model,
            "provider": self.provider,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "context_size": self.context_size,
            "response_size": self.response_size,
        }.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonaProviderConfig':
        """Create from dictionary."""
        return cls(
            model=data.get("model"),
            provider=data.get("provider"),
            api_base=data.get("api_base"),
            api_key=data.get("api_key"),
            temperature=data.get("temperature"),
            top_p=data.get("top_p"),
            top_k=data.get("top_k"),
            max_tokens=data.get("max_tokens"),
            context_size=data.get("context_size"),
            response_size=data.get("response_size"),
        )


@dataclass
class Persona:
    """Represents an AI persona in the council."""
    name: str
    role: str
    expertise: list[str]
    personality_traits: list[str]
    perspective: str  # General viewpoint/bias this persona brings
    provider_config: Optional[PersonaProviderConfig] = None  # Per-persona provider settings

    def to_system_prompt(self) -> str:
        """Generate system prompt for this persona."""
        traits = ", ".join(self.personality_traits)
        expertise = ", ".join(self.expertise)
        return f"""You are {self.name}, a {self.role}.

Your areas of expertise: {expertise}
Your personality traits: {traits}
Your perspective: {self.perspective}

You are participating in a council discussion. Stay in character and provide insights based on your unique perspective and expertise. Be constructive but also challenge ideas when appropriate based on your role."""

    def with_provider_config(self, config: PersonaProviderConfig) -> 'Persona':
        """Return a new Persona with the given provider config."""
        return Persona(
            name=self.name,
            role=self.role,
            expertise=self.expertise,
            personality_traits=self.personality_traits,
            perspective=self.perspective,
            provider_config=config,
        )


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

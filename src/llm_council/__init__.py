"""LLM Council - Multi-persona AI deliberation and consensus tool."""

__version__ = "0.1.0"

from .models import (
    Persona,
    Message,
    Vote,
    RoundResult,
    CouncilSession,
    ConsensusType,
    VoteChoice,
)
from .assertions import assert_council, CouncilAssertions, AssertionReport, ValidationResult
from .providers import create_provider, LiteLLMProvider, ProviderConfig
from .personas import PersonaManager
from .council import CouncilEngine

__all__ = [
    # Models
    "Persona",
    "Message",
    "Vote",
    "RoundResult",
    "CouncilSession",
    "ConsensusType",
    "VoteChoice",
    # Assertions (US-01)
    "assert_council",
    "CouncilAssertions",
    "AssertionReport",
    "ValidationResult",
    # Providers
    "create_provider",
    "LiteLLMProvider",
    "ProviderConfig",
    # Personas
    "PersonaManager",
    # Council
    "CouncilEngine",
]

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
from .schemas import (
    DiscussionRequestSchema,
    PersonaTemplateSchema,
    SessionOutputSchema,
    SchemaValidationError,
    ValidationErrors,
    ValidationErrorCode,
    validate_discussion_request,
    validate_persona_template,
    validate_session_output,
    SchemaValidator,
)
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
    # Schemas (US-02)
    "DiscussionRequestSchema",
    "PersonaTemplateSchema",
    "SessionOutputSchema",
    "SchemaValidationError",
    "ValidationErrors",
    "ValidationErrorCode",
    "validate_discussion_request",
    "validate_persona_template",
    "validate_session_output",
    "SchemaValidator",
    # Providers
    "create_provider",
    "LiteLLMProvider",
    "ProviderConfig",
    # Personas
    "PersonaManager",
    # Council
    "CouncilEngine",
]

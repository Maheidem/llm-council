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
from .contracts import (
    INTERFACE_VERSION,
    ErrorCode,
    ErrorSeverity,
    InterfaceError,
    InterfaceContract,
    ContractRegistry,
    ErrorHandler,
    FailureRecovery,
    RecoveryAction,
    RecoveryResult,
    get_contract_registry,
    get_error_handler,
    get_interface_version,
)
from .templates import (
    PersonaTemplate,
    PersonaTemplateLibrary,
    TemplateLoader,
    get_template_library,
    create_persona_from_template,
    list_builtin_templates,
    get_builtin_template,
)
from .persistence import (
    RetentionPolicy,
    StoredSession,
    SessionStorage,
    SQLiteStorage,
    SessionExporter,
    SessionManager,
    get_session_manager,
    save_session,
    load_session,
)
from .metrics import (
    MetricType,
    MetricPoint,
    AggregatedMetric,
    MetricsCollector,
    MetricsAggregator,
    Timer,
    SessionMetrics,
    MetricsReporter,
    get_metrics_collector,
    get_metrics_reporter,
    time_operation,
    record_latency,
    record_tokens,
    record_session_metrics,
    get_metrics_summary,
)
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
    # Contracts (US-03)
    "INTERFACE_VERSION",
    "ErrorCode",
    "ErrorSeverity",
    "InterfaceError",
    "InterfaceContract",
    "ContractRegistry",
    "ErrorHandler",
    "FailureRecovery",
    "RecoveryAction",
    "RecoveryResult",
    "get_contract_registry",
    "get_error_handler",
    "get_interface_version",
    # Templates (US-04)
    "PersonaTemplate",
    "PersonaTemplateLibrary",
    "TemplateLoader",
    "get_template_library",
    "create_persona_from_template",
    "list_builtin_templates",
    "get_builtin_template",
    # Persistence (US-05)
    "RetentionPolicy",
    "StoredSession",
    "SessionStorage",
    "SQLiteStorage",
    "SessionExporter",
    "SessionManager",
    "get_session_manager",
    "save_session",
    "load_session",
    # Metrics (US-06)
    "MetricType",
    "MetricPoint",
    "AggregatedMetric",
    "MetricsCollector",
    "MetricsAggregator",
    "Timer",
    "SessionMetrics",
    "MetricsReporter",
    "get_metrics_collector",
    "get_metrics_reporter",
    "time_operation",
    "record_latency",
    "record_tokens",
    "record_session_metrics",
    "get_metrics_summary",
    # Providers
    "create_provider",
    "LiteLLMProvider",
    "ProviderConfig",
    # Personas
    "PersonaManager",
    # Council
    "CouncilEngine",
]

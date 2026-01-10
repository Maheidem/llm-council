"""US-03: Versioned interface contracts for persona communication.

Provides standardized error handling, failure recovery, and edge case handling.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any, Callable
from datetime import datetime
import time


# Interface version using semantic versioning
INTERFACE_VERSION = "1.0.0"
INTERFACE_VERSION_MAJOR = 1
INTERFACE_VERSION_MINOR = 0
INTERFACE_VERSION_PATCH = 0


class ErrorSeverity(str, Enum):
    """Severity levels for errors."""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCode(str, Enum):
    """Standardized error codes for persona interactions."""
    # Communication errors
    TIMEOUT = "E001"
    CONNECTION_FAILED = "E002"
    MALFORMED_RESPONSE = "E003"
    EMPTY_RESPONSE = "E004"

    # Consensus errors
    CONSENSUS_DEADLOCK = "E010"
    MAX_ROUNDS_EXCEEDED = "E011"
    INSUFFICIENT_PERSONAS = "E012"

    # Persona errors
    PERSONA_UNAVAILABLE = "E020"
    INVALID_PERSONA_RESPONSE = "E021"
    PERSONA_CONFLICT = "E022"

    # System errors
    PROVIDER_ERROR = "E030"
    RATE_LIMIT_EXCEEDED = "E031"
    QUOTA_EXCEEDED = "E032"

    # Recovery errors
    RECOVERY_FAILED = "E040"
    FALLBACK_EXHAUSTED = "E041"


@dataclass
class InterfaceError:
    """Structured error for interface contracts."""
    code: ErrorCode
    message: str
    severity: ErrorSeverity
    timestamp: datetime = field(default_factory=datetime.now)
    context: dict = field(default_factory=dict)
    recoverable: bool = True

    def to_dict(self) -> dict:
        return {
            "code": self.code.value,
            "message": self.message,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "recoverable": self.recoverable,
        }


@dataclass
class RecoveryAction:
    """Action to take for error recovery."""
    name: str
    description: str
    executor: Optional[Callable[[], Any]] = None
    max_retries: int = 3
    retry_delay_ms: int = 1000


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    success: bool
    action_taken: str
    attempts: int
    final_error: Optional[InterfaceError] = None
    recovered_value: Optional[Any] = None


class FailureRecovery:
    """Failure recovery mechanisms for edge cases."""

    def __init__(self):
        self._strategies: dict[ErrorCode, list[RecoveryAction]] = {}
        self._register_default_strategies()

    def _register_default_strategies(self):
        """Register default recovery strategies."""
        # Timeout recovery
        self._strategies[ErrorCode.TIMEOUT] = [
            RecoveryAction(
                name="retry_with_backoff",
                description="Retry with exponential backoff",
                max_retries=3,
                retry_delay_ms=1000,
            ),
            RecoveryAction(
                name="reduce_complexity",
                description="Retry with simpler prompt",
                max_retries=2,
            ),
        ]

        # Malformed response recovery
        self._strategies[ErrorCode.MALFORMED_RESPONSE] = [
            RecoveryAction(
                name="retry_request",
                description="Retry the same request",
                max_retries=2,
            ),
            RecoveryAction(
                name="request_structured_output",
                description="Request JSON-formatted output",
                max_retries=1,
            ),
        ]

        # Consensus deadlock recovery
        self._strategies[ErrorCode.CONSENSUS_DEADLOCK] = [
            RecoveryAction(
                name="force_vote",
                description="Force a voting round to break deadlock",
                max_retries=1,
            ),
            RecoveryAction(
                name="introduce_tiebreaker",
                description="Add a tiebreaker persona",
                max_retries=1,
            ),
        ]

        # Empty response recovery
        self._strategies[ErrorCode.EMPTY_RESPONSE] = [
            RecoveryAction(
                name="retry_with_prompt",
                description="Retry with clearer prompt",
                max_retries=2,
            ),
        ]

    def get_strategies(self, error_code: ErrorCode) -> list[RecoveryAction]:
        """Get recovery strategies for an error code."""
        return self._strategies.get(error_code, [])

    def register_strategy(self, error_code: ErrorCode, action: RecoveryAction):
        """Register a custom recovery strategy."""
        if error_code not in self._strategies:
            self._strategies[error_code] = []
        self._strategies[error_code].append(action)

    def attempt_recovery(
        self,
        error: InterfaceError,
        retry_func: Optional[Callable[[], Any]] = None,
    ) -> RecoveryResult:
        """Attempt to recover from an error."""
        if not error.recoverable:
            return RecoveryResult(
                success=False,
                action_taken="none",
                attempts=0,
                final_error=error,
            )

        strategies = self.get_strategies(error.code)
        if not strategies:
            return RecoveryResult(
                success=False,
                action_taken="no_strategy_available",
                attempts=0,
                final_error=error,
            )

        total_attempts = 0
        for strategy in strategies:
            for attempt in range(strategy.max_retries):
                total_attempts += 1

                # Apply retry delay with backoff
                if attempt > 0:
                    delay_ms = strategy.retry_delay_ms * (2 ** (attempt - 1))
                    time.sleep(delay_ms / 1000)

                # Execute recovery
                executor = strategy.executor or retry_func
                if executor:
                    try:
                        result = executor()
                        return RecoveryResult(
                            success=True,
                            action_taken=strategy.name,
                            attempts=total_attempts,
                            recovered_value=result,
                        )
                    except Exception:
                        continue

        return RecoveryResult(
            success=False,
            action_taken="all_strategies_exhausted",
            attempts=total_attempts,
            final_error=InterfaceError(
                code=ErrorCode.FALLBACK_EXHAUSTED,
                message="All recovery strategies exhausted",
                severity=ErrorSeverity.CRITICAL,
                recoverable=False,
            ),
        )


@dataclass
class InterfaceContract:
    """Contract definition for persona communication."""
    version: str
    operation: str
    input_schema: Optional[dict] = None
    output_schema: Optional[dict] = None
    timeout_ms: int = 30000
    max_retries: int = 3
    error_codes: list[ErrorCode] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "operation": self.operation,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "timeout_ms": self.timeout_ms,
            "max_retries": self.max_retries,
            "error_codes": [e.value for e in self.error_codes],
        }


class ContractRegistry:
    """Registry of interface contracts."""

    def __init__(self):
        self._contracts: dict[str, InterfaceContract] = {}
        self._register_default_contracts()

    def _register_default_contracts(self):
        """Register default contracts."""
        # Persona discussion contract
        self.register(InterfaceContract(
            version=INTERFACE_VERSION,
            operation="persona_discuss",
            input_schema={
                "type": "object",
                "required": ["topic", "context", "persona"],
                "properties": {
                    "topic": {"type": "string"},
                    "context": {"type": "string"},
                    "persona": {"type": "object"},
                },
            },
            output_schema={
                "type": "object",
                "required": ["response"],
                "properties": {
                    "response": {"type": "string"},
                    "position": {"type": "string"},
                },
            },
            timeout_ms=30000,
            error_codes=[
                ErrorCode.TIMEOUT,
                ErrorCode.MALFORMED_RESPONSE,
                ErrorCode.EMPTY_RESPONSE,
            ],
        ))

        # Consensus check contract
        self.register(InterfaceContract(
            version=INTERFACE_VERSION,
            operation="check_consensus",
            input_schema={
                "type": "object",
                "required": ["responses"],
                "properties": {
                    "responses": {"type": "array"},
                },
            },
            output_schema={
                "type": "object",
                "required": ["consensus_reached"],
                "properties": {
                    "consensus_reached": {"type": "boolean"},
                    "consensus_position": {"type": "string"},
                },
            },
            timeout_ms=15000,
            error_codes=[
                ErrorCode.TIMEOUT,
                ErrorCode.CONSENSUS_DEADLOCK,
            ],
        ))

        # Vote contract
        self.register(InterfaceContract(
            version=INTERFACE_VERSION,
            operation="persona_vote",
            input_schema={
                "type": "object",
                "required": ["options", "persona"],
            },
            output_schema={
                "type": "object",
                "required": ["choice", "reasoning"],
            },
            timeout_ms=15000,
            error_codes=[
                ErrorCode.TIMEOUT,
                ErrorCode.INVALID_PERSONA_RESPONSE,
            ],
        ))

    def register(self, contract: InterfaceContract):
        """Register a contract."""
        self._contracts[contract.operation] = contract

    def get(self, operation: str) -> Optional[InterfaceContract]:
        """Get a contract by operation name."""
        return self._contracts.get(operation)

    def list_operations(self) -> list[str]:
        """List all registered operations."""
        return list(self._contracts.keys())

    def get_version(self) -> str:
        """Get interface version."""
        return INTERFACE_VERSION

    def is_compatible(self, version: str) -> bool:
        """Check if a version is compatible with current interface."""
        try:
            parts = version.split(".")
            major = int(parts[0])
            # Major version must match for compatibility
            return major == INTERFACE_VERSION_MAJOR
        except (ValueError, IndexError):
            return False


class ErrorHandler:
    """Standardized error handling across personas."""

    def __init__(self):
        self._handlers: dict[ErrorCode, Callable[[InterfaceError], None]] = {}
        self._recovery = FailureRecovery()
        self._error_log: list[InterfaceError] = []

    def handle(self, error: InterfaceError) -> RecoveryResult:
        """Handle an error with recovery attempt."""
        self._error_log.append(error)

        # Call custom handler if registered
        if error.code in self._handlers:
            self._handlers[error.code](error)

        # Attempt recovery
        return self._recovery.attempt_recovery(error)

    def register_handler(self, code: ErrorCode, handler: Callable[[InterfaceError], None]):
        """Register a custom error handler."""
        self._handlers[code] = handler

    def create_error(
        self,
        code: ErrorCode,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[dict] = None,
        recoverable: bool = True,
    ) -> InterfaceError:
        """Create a standardized error."""
        return InterfaceError(
            code=code,
            message=message,
            severity=severity,
            context=context or {},
            recoverable=recoverable,
        )

    def get_error_log(self) -> list[InterfaceError]:
        """Get the error log."""
        return self._error_log.copy()

    def clear_error_log(self):
        """Clear the error log."""
        self._error_log.clear()


# Default instances
_contract_registry = ContractRegistry()
_error_handler = ErrorHandler()


def get_contract_registry() -> ContractRegistry:
    """Get the default contract registry."""
    return _contract_registry


def get_error_handler() -> ErrorHandler:
    """Get the default error handler."""
    return _error_handler


def get_interface_version() -> str:
    """Get the interface version."""
    return INTERFACE_VERSION

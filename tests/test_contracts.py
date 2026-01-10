"""Tests for US-03: Versioned interface contracts."""

import pytest
from llm_council.contracts import (
    INTERFACE_VERSION,
    ErrorSeverity,
    ErrorCode,
    InterfaceError,
    RecoveryAction,
    RecoveryResult,
    FailureRecovery,
    InterfaceContract,
    ContractRegistry,
    ErrorHandler,
    get_contract_registry,
    get_error_handler,
    get_interface_version,
)


class TestInterfaceVersion:
    """Tests for interface versioning."""

    def test_version_format(self):
        assert INTERFACE_VERSION == "1.0.0"
        parts = INTERFACE_VERSION.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_get_interface_version(self):
        assert get_interface_version() == INTERFACE_VERSION


class TestErrorCodes:
    """Tests for error codes."""

    def test_error_code_values(self):
        assert ErrorCode.TIMEOUT.value == "E001"
        assert ErrorCode.CONSENSUS_DEADLOCK.value == "E010"
        assert ErrorCode.PERSONA_UNAVAILABLE.value == "E020"
        assert ErrorCode.PROVIDER_ERROR.value == "E030"

    def test_all_error_codes_unique(self):
        values = [e.value for e in ErrorCode]
        assert len(values) == len(set(values))


class TestInterfaceError:
    """Tests for interface errors."""

    def test_error_creation(self):
        error = InterfaceError(
            code=ErrorCode.TIMEOUT,
            message="Request timed out",
            severity=ErrorSeverity.ERROR,
            context={"timeout_ms": 30000},
        )
        assert error.code == ErrorCode.TIMEOUT
        assert error.message == "Request timed out"
        assert error.severity == ErrorSeverity.ERROR
        assert error.recoverable is True

    def test_error_to_dict(self):
        error = InterfaceError(
            code=ErrorCode.TIMEOUT,
            message="Timeout",
            severity=ErrorSeverity.WARNING,
        )
        d = error.to_dict()
        assert d["code"] == "E001"
        assert d["message"] == "Timeout"
        assert d["severity"] == "warning"
        assert "timestamp" in d
        assert d["recoverable"] is True

    def test_non_recoverable_error(self):
        error = InterfaceError(
            code=ErrorCode.QUOTA_EXCEEDED,
            message="Quota exceeded",
            severity=ErrorSeverity.CRITICAL,
            recoverable=False,
        )
        assert error.recoverable is False


class TestRecoveryAction:
    """Tests for recovery actions."""

    def test_recovery_action_creation(self):
        action = RecoveryAction(
            name="retry",
            description="Retry the request",
            max_retries=3,
            retry_delay_ms=1000,
        )
        assert action.name == "retry"
        assert action.max_retries == 3
        assert action.retry_delay_ms == 1000


class TestFailureRecovery:
    """Tests for failure recovery mechanisms."""

    def test_default_strategies_registered(self):
        recovery = FailureRecovery()
        strategies = recovery.get_strategies(ErrorCode.TIMEOUT)
        assert len(strategies) > 0

    def test_get_strategies_for_unknown_error(self):
        recovery = FailureRecovery()
        strategies = recovery.get_strategies(ErrorCode.RATE_LIMIT_EXCEEDED)
        # May or may not have strategies
        assert isinstance(strategies, list)

    def test_register_custom_strategy(self):
        recovery = FailureRecovery()
        action = RecoveryAction(
            name="custom_action",
            description="Custom recovery",
        )
        recovery.register_strategy(ErrorCode.PROVIDER_ERROR, action)
        strategies = recovery.get_strategies(ErrorCode.PROVIDER_ERROR)
        assert any(s.name == "custom_action" for s in strategies)

    def test_recovery_non_recoverable_error(self):
        recovery = FailureRecovery()
        error = InterfaceError(
            code=ErrorCode.TIMEOUT,
            message="Timeout",
            severity=ErrorSeverity.CRITICAL,
            recoverable=False,
        )
        result = recovery.attempt_recovery(error)
        assert result.success is False
        assert result.action_taken == "none"

    def test_recovery_with_executor(self):
        recovery = FailureRecovery()
        error = InterfaceError(
            code=ErrorCode.TIMEOUT,
            message="Timeout",
            severity=ErrorSeverity.ERROR,
        )

        call_count = [0]

        def retry_func():
            call_count[0] += 1
            if call_count[0] >= 2:
                return "success"
            raise Exception("Retry")

        result = recovery.attempt_recovery(error, retry_func)
        assert result.success is True
        assert result.recovered_value == "success"


class TestInterfaceContract:
    """Tests for interface contracts."""

    def test_contract_creation(self):
        contract = InterfaceContract(
            version="1.0.0",
            operation="test_op",
            timeout_ms=5000,
        )
        assert contract.version == "1.0.0"
        assert contract.operation == "test_op"
        assert contract.timeout_ms == 5000

    def test_contract_to_dict(self):
        contract = InterfaceContract(
            version="1.0.0",
            operation="test_op",
            error_codes=[ErrorCode.TIMEOUT],
        )
        d = contract.to_dict()
        assert d["version"] == "1.0.0"
        assert d["operation"] == "test_op"
        assert "E001" in d["error_codes"]


class TestContractRegistry:
    """Tests for contract registry."""

    def test_default_contracts_registered(self):
        registry = ContractRegistry()
        operations = registry.list_operations()
        assert "persona_discuss" in operations
        assert "check_consensus" in operations
        assert "persona_vote" in operations

    def test_get_contract(self):
        registry = ContractRegistry()
        contract = registry.get("persona_discuss")
        assert contract is not None
        assert contract.operation == "persona_discuss"

    def test_get_nonexistent_contract(self):
        registry = ContractRegistry()
        contract = registry.get("nonexistent")
        assert contract is None

    def test_register_custom_contract(self):
        registry = ContractRegistry()
        contract = InterfaceContract(
            version="1.0.0",
            operation="custom_op",
        )
        registry.register(contract)
        assert registry.get("custom_op") is not None

    def test_version_compatibility_same_major(self):
        registry = ContractRegistry()
        assert registry.is_compatible("1.0.0")
        assert registry.is_compatible("1.1.0")
        assert registry.is_compatible("1.99.99")

    def test_version_compatibility_different_major(self):
        registry = ContractRegistry()
        assert not registry.is_compatible("2.0.0")
        assert not registry.is_compatible("0.9.0")

    def test_version_compatibility_invalid_format(self):
        registry = ContractRegistry()
        assert not registry.is_compatible("invalid")
        assert not registry.is_compatible("")


class TestErrorHandler:
    """Tests for error handler."""

    def test_create_error(self):
        handler = ErrorHandler()
        error = handler.create_error(
            code=ErrorCode.TIMEOUT,
            message="Request timed out",
        )
        assert error.code == ErrorCode.TIMEOUT
        assert error.message == "Request timed out"

    def test_handle_error_logs(self):
        handler = ErrorHandler()
        error = handler.create_error(
            code=ErrorCode.TIMEOUT,
            message="Timeout",
        )
        handler.handle(error)
        log = handler.get_error_log()
        assert len(log) == 1
        assert log[0].code == ErrorCode.TIMEOUT

    def test_clear_error_log(self):
        handler = ErrorHandler()
        error = handler.create_error(ErrorCode.TIMEOUT, "Timeout")
        handler.handle(error)
        assert len(handler.get_error_log()) == 1
        handler.clear_error_log()
        assert len(handler.get_error_log()) == 0

    def test_register_custom_handler(self):
        handler = ErrorHandler()
        handled_errors = []

        def custom_handler(error):
            handled_errors.append(error)

        handler.register_handler(ErrorCode.TIMEOUT, custom_handler)

        error = handler.create_error(ErrorCode.TIMEOUT, "Timeout")
        handler.handle(error)

        assert len(handled_errors) == 1


class TestGlobalInstances:
    """Tests for global instances."""

    def test_get_contract_registry_singleton(self):
        registry1 = get_contract_registry()
        registry2 = get_contract_registry()
        assert registry1 is registry2

    def test_get_error_handler_singleton(self):
        handler1 = get_error_handler()
        handler2 = get_error_handler()
        assert handler1 is handler2


class TestEdgeCaseHandling:
    """Tests for edge case handling."""

    def test_timeout_recovery_strategies(self):
        recovery = FailureRecovery()
        strategies = recovery.get_strategies(ErrorCode.TIMEOUT)
        assert any(s.name == "retry_with_backoff" for s in strategies)

    def test_malformed_response_recovery(self):
        recovery = FailureRecovery()
        strategies = recovery.get_strategies(ErrorCode.MALFORMED_RESPONSE)
        assert len(strategies) > 0

    def test_consensus_deadlock_recovery(self):
        recovery = FailureRecovery()
        strategies = recovery.get_strategies(ErrorCode.CONSENSUS_DEADLOCK)
        assert any(s.name == "force_vote" for s in strategies)

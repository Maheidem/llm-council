"""US-01: Test Assertion API for validating council outputs.

Provides constraint validators for CI/CD integration testing.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional
import re

from .models import CouncilSession


@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    constraint_name: str
    message: str
    actual_value: Any = None
    expected_value: Any = None


@dataclass
class AssertionReport:
    """Report containing all validation results."""
    results: list[ValidationResult]

    @property
    def passed(self) -> bool:
        """Check if all validations passed."""
        return all(r.passed for r in self.results)

    @property
    def failed_count(self) -> int:
        """Count of failed validations."""
        return sum(1 for r in self.results if not r.passed)

    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            "passed": self.passed,
            "total": len(self.results),
            "failed": self.failed_count,
            "results": [
                {
                    "constraint": r.constraint_name,
                    "passed": r.passed,
                    "message": r.message,
                    "actual": r.actual_value,
                    "expected": r.expected_value,
                }
                for r in self.results
            ]
        }


class CouncilAssertions:
    """Assertion API for validating council session outputs."""

    def __init__(self, session: CouncilSession):
        """Initialize with a council session."""
        self.session = session
        self._results: list[ValidationResult] = []

    def _add_result(self, result: ValidationResult) -> "CouncilAssertions":
        """Add a validation result and return self for chaining."""
        self._results.append(result)
        return self

    # Constraint 1: Consensus Validator
    def assert_consensus_reached(self) -> "CouncilAssertions":
        """Validate that consensus was reached."""
        passed = self.session.consensus_reached
        return self._add_result(ValidationResult(
            passed=passed,
            constraint_name="consensus_reached",
            message="Consensus was reached" if passed else "Consensus was NOT reached",
            actual_value=passed,
            expected_value=True,
        ))

    def assert_consensus_not_reached(self) -> "CouncilAssertions":
        """Validate that consensus was NOT reached (for testing edge cases)."""
        passed = not self.session.consensus_reached
        return self._add_result(ValidationResult(
            passed=passed,
            constraint_name="consensus_not_reached",
            message="Consensus was not reached as expected" if passed else "Consensus WAS reached unexpectedly",
            actual_value=self.session.consensus_reached,
            expected_value=False,
        ))

    # Constraint 2: Round Count Validator
    def assert_max_rounds(self, max_rounds: int) -> "CouncilAssertions":
        """Validate that rounds completed is within limit."""
        actual = len(self.session.rounds)
        passed = actual <= max_rounds
        return self._add_result(ValidationResult(
            passed=passed,
            constraint_name="max_rounds",
            message=f"Rounds ({actual}) within limit ({max_rounds})" if passed else f"Rounds ({actual}) exceeded limit ({max_rounds})",
            actual_value=actual,
            expected_value=max_rounds,
        ))

    def assert_min_rounds(self, min_rounds: int) -> "CouncilAssertions":
        """Validate that at least N rounds were completed."""
        actual = len(self.session.rounds)
        passed = actual >= min_rounds
        return self._add_result(ValidationResult(
            passed=passed,
            constraint_name="min_rounds",
            message=f"Rounds ({actual}) meets minimum ({min_rounds})" if passed else f"Rounds ({actual}) below minimum ({min_rounds})",
            actual_value=actual,
            expected_value=min_rounds,
        ))

    # Constraint 3: Content Validator
    def assert_consensus_contains(self, substring: str, case_sensitive: bool = False) -> "CouncilAssertions":
        """Validate that final consensus contains a substring."""
        consensus = self.session.final_consensus or ""
        if case_sensitive:
            passed = substring in consensus
        else:
            passed = substring.lower() in consensus.lower()
        return self._add_result(ValidationResult(
            passed=passed,
            constraint_name="consensus_contains",
            message=f"Consensus contains '{substring}'" if passed else f"Consensus does NOT contain '{substring}'",
            actual_value=consensus[:100] + "..." if len(consensus) > 100 else consensus,
            expected_value=substring,
        ))

    def assert_consensus_matches(self, pattern: str) -> "CouncilAssertions":
        """Validate that final consensus matches a regex pattern."""
        consensus = self.session.final_consensus or ""
        passed = bool(re.search(pattern, consensus, re.IGNORECASE))
        return self._add_result(ValidationResult(
            passed=passed,
            constraint_name="consensus_matches",
            message=f"Consensus matches pattern '{pattern}'" if passed else f"Consensus does NOT match pattern '{pattern}'",
            actual_value=consensus[:100] + "..." if len(consensus) > 100 else consensus,
            expected_value=pattern,
        ))

    # Constraint 4: Decision Boundary Validator
    def assert_decision_in_options(self, options: list[str]) -> "CouncilAssertions":
        """Validate that consensus mentions one of the expected options."""
        consensus = self.session.final_consensus or ""
        consensus_lower = consensus.lower()
        found_options = [opt for opt in options if opt.lower() in consensus_lower]
        passed = len(found_options) > 0
        return self._add_result(ValidationResult(
            passed=passed,
            constraint_name="decision_in_options",
            message=f"Decision includes option(s): {found_options}" if passed else f"Decision does not include any of: {options}",
            actual_value=found_options if found_options else None,
            expected_value=options,
        ))

    # Constraint 5: Persona Participation Validator
    def assert_persona_count(self, expected_count: int) -> "CouncilAssertions":
        """Validate the number of personas that participated."""
        actual = len(self.session.personas) if self.session.personas else 0
        passed = actual == expected_count
        return self._add_result(ValidationResult(
            passed=passed,
            constraint_name="persona_count",
            message=f"Persona count ({actual}) matches expected ({expected_count})" if passed else f"Persona count ({actual}) does not match expected ({expected_count})",
            actual_value=actual,
            expected_value=expected_count,
        ))

    def assert_min_personas(self, min_count: int) -> "CouncilAssertions":
        """Validate minimum number of participating personas."""
        actual = len(self.session.personas) if self.session.personas else 0
        passed = actual >= min_count
        return self._add_result(ValidationResult(
            passed=passed,
            constraint_name="min_personas",
            message=f"Persona count ({actual}) meets minimum ({min_count})" if passed else f"Persona count ({actual}) below minimum ({min_count})",
            actual_value=actual,
            expected_value=min_count,
        ))

    # Constraint 6: Custom Validator
    def assert_custom(
        self,
        name: str,
        validator: Callable[[CouncilSession], bool],
        message_pass: str = "Custom validation passed",
        message_fail: str = "Custom validation failed",
    ) -> "CouncilAssertions":
        """Apply a custom validation function."""
        passed = validator(self.session)
        return self._add_result(ValidationResult(
            passed=passed,
            constraint_name=name,
            message=message_pass if passed else message_fail,
            actual_value=None,
            expected_value=None,
        ))

    def report(self) -> AssertionReport:
        """Generate validation report."""
        return AssertionReport(results=self._results)

    def validate(self) -> bool:
        """Run all assertions and return True if all pass."""
        return all(r.passed for r in self._results)

    def raise_on_failure(self) -> None:
        """Raise AssertionError if any validation failed."""
        failed = [r for r in self._results if not r.passed]
        if failed:
            messages = [f"  - {r.constraint_name}: {r.message}" for r in failed]
            raise AssertionError(f"Council validation failed:\n" + "\n".join(messages))


def assert_council(session: CouncilSession) -> CouncilAssertions:
    """Create an assertion chain for a council session.

    Example:
        report = (
            assert_council(session)
            .assert_consensus_reached()
            .assert_max_rounds(3)
            .assert_decision_in_options(["React", "Vue", "Angular"])
            .report()
        )

        if not report.passed:
            print(report.to_dict())
    """
    return CouncilAssertions(session)

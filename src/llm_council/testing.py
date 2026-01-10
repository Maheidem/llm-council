"""US-07: Parameterized testing framework for multi-provider variance tracking.

Provides tools for testing council sessions across different providers
and tracking behavioral variance.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any, Callable
from pathlib import Path


class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class ProviderConfig:
    """Configuration for a test provider."""
    name: str
    model: str
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    extra_params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "api_base": self.api_base,
            "extra_params": self.extra_params,
        }


@dataclass
class TestCase:
    """A parameterized test case."""
    name: str
    topic: str
    objective: str
    expected_consensus: Optional[bool] = None
    max_rounds: int = 3
    personas_count: int = 3
    timeout_ms: int = 60000
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "topic": self.topic,
            "objective": self.objective,
            "expected_consensus": self.expected_consensus,
            "max_rounds": self.max_rounds,
            "personas_count": self.personas_count,
            "timeout_ms": self.timeout_ms,
            "tags": self.tags,
            "metadata": self.metadata,
        }


@dataclass
class TestResult:
    """Result of executing a test case."""
    test_case: TestCase
    provider: ProviderConfig
    status: TestStatus
    duration_ms: float
    consensus_reached: Optional[bool] = None
    rounds_count: int = 0
    tokens_used: int = 0
    error_message: Optional[str] = None
    session_data: Optional[dict] = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def passed(self) -> bool:
        return self.status == TestStatus.PASSED

    def to_dict(self) -> dict:
        return {
            "test_case": self.test_case.name,
            "provider": self.provider.name,
            "status": self.status.value,
            "duration_ms": round(self.duration_ms, 2),
            "consensus_reached": self.consensus_reached,
            "rounds_count": self.rounds_count,
            "tokens_used": self.tokens_used,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class VarianceMetric:
    """Metric tracking variance across providers."""
    metric_name: str
    values: dict[str, float]  # provider_name -> value
    mean: float
    std_dev: float
    min_value: float
    max_value: float
    variance_ratio: float  # max/min ratio

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "values": self.values,
            "mean": round(self.mean, 4),
            "std_dev": round(self.std_dev, 4),
            "min": round(self.min_value, 4),
            "max": round(self.max_value, 4),
            "variance_ratio": round(self.variance_ratio, 4),
        }


class TestSuite:
    """Collection of parameterized test cases."""

    def __init__(self, name: str):
        self.name = name
        self._test_cases: list[TestCase] = []
        self._providers: list[ProviderConfig] = []

    def add_test_case(self, test_case: TestCase):
        """Add a test case to the suite."""
        self._test_cases.append(test_case)

    def add_provider(self, provider: ProviderConfig):
        """Add a provider to test against."""
        self._providers.append(provider)

    def get_test_cases(self, tags: Optional[list[str]] = None) -> list[TestCase]:
        """Get test cases, optionally filtered by tags."""
        if tags:
            return [tc for tc in self._test_cases if any(t in tc.tags for t in tags)]
        return self._test_cases.copy()

    def get_providers(self) -> list[ProviderConfig]:
        """Get all configured providers."""
        return self._providers.copy()

    def count_tests(self) -> int:
        """Get total number of test cases."""
        return len(self._test_cases)

    def count_providers(self) -> int:
        """Get number of providers."""
        return len(self._providers)

    def total_executions(self) -> int:
        """Get total number of test executions (tests * providers)."""
        return len(self._test_cases) * len(self._providers)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "test_cases": [tc.to_dict() for tc in self._test_cases],
            "providers": [p.to_dict() for p in self._providers],
            "total_executions": self.total_executions(),
        }


class TestExecutor:
    """Executes parameterized tests against multiple providers."""

    def __init__(self, suite: TestSuite):
        self.suite = suite
        self._results: list[TestResult] = []
        self._executor: Optional[Callable] = None

    def set_executor(self, executor: Callable[[TestCase, ProviderConfig], TestResult]):
        """Set the test execution function."""
        self._executor = executor

    def run_all(self, tags: Optional[list[str]] = None) -> list[TestResult]:
        """Run all tests against all providers."""
        results = []
        test_cases = self.suite.get_test_cases(tags)
        providers = self.suite.get_providers()

        for test_case in test_cases:
            for provider in providers:
                result = self._execute_test(test_case, provider)
                results.append(result)
                self._results.append(result)

        return results

    def run_test(self, test_name: str, provider_name: Optional[str] = None) -> list[TestResult]:
        """Run a specific test, optionally against a specific provider."""
        results = []
        test_cases = [tc for tc in self.suite.get_test_cases() if tc.name == test_name]
        providers = self.suite.get_providers()

        if provider_name:
            providers = [p for p in providers if p.name == provider_name]

        for test_case in test_cases:
            for provider in providers:
                result = self._execute_test(test_case, provider)
                results.append(result)
                self._results.append(result)

        return results

    def _execute_test(self, test_case: TestCase, provider: ProviderConfig) -> TestResult:
        """Execute a single test case against a provider."""
        if self._executor is None:
            return self._create_mock_result(test_case, provider)

        start_time = time.perf_counter()
        try:
            result = self._executor(test_case, provider)
            return result
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return TestResult(
                test_case=test_case,
                provider=provider,
                status=TestStatus.ERROR,
                duration_ms=duration_ms,
                error_message=str(e),
            )

    def _create_mock_result(self, test_case: TestCase, provider: ProviderConfig) -> TestResult:
        """Create a mock result when no executor is set."""
        return TestResult(
            test_case=test_case,
            provider=provider,
            status=TestStatus.SKIPPED,
            duration_ms=0.0,
            error_message="No executor configured",
        )

    def get_results(self) -> list[TestResult]:
        """Get all test results."""
        return self._results.copy()

    def clear_results(self):
        """Clear all stored results."""
        self._results.clear()


class VarianceTracker:
    """Tracks variance in test results across providers."""

    def __init__(self):
        self._results_by_test: dict[str, list[TestResult]] = {}

    def add_results(self, results: list[TestResult]):
        """Add test results for variance analysis."""
        for result in results:
            test_name = result.test_case.name
            if test_name not in self._results_by_test:
                self._results_by_test[test_name] = []
            self._results_by_test[test_name].append(result)

    def calculate_variance(self, test_name: str) -> dict[str, VarianceMetric]:
        """Calculate variance metrics for a specific test."""
        results = self._results_by_test.get(test_name, [])
        if not results:
            return {}

        metrics = {}

        # Duration variance
        duration_values = {r.provider.name: r.duration_ms for r in results if r.passed}
        if duration_values:
            metrics["duration_ms"] = self._create_variance_metric("duration_ms", duration_values)

        # Rounds variance
        rounds_values = {r.provider.name: float(r.rounds_count) for r in results if r.passed}
        if rounds_values:
            metrics["rounds_count"] = self._create_variance_metric("rounds_count", rounds_values)

        # Tokens variance
        tokens_values = {r.provider.name: float(r.tokens_used) for r in results if r.passed}
        if tokens_values:
            metrics["tokens_used"] = self._create_variance_metric("tokens_used", tokens_values)

        # Consensus rate by provider
        consensus_values = {}
        for result in results:
            provider = result.provider.name
            if provider not in consensus_values:
                consensus_values[provider] = []
            if result.consensus_reached is not None:
                consensus_values[provider].append(1.0 if result.consensus_reached else 0.0)

        consensus_rates = {
            p: sum(v) / len(v) if v else 0.0
            for p, v in consensus_values.items()
        }
        if consensus_rates:
            metrics["consensus_rate"] = self._create_variance_metric("consensus_rate", consensus_rates)

        return metrics

    def _create_variance_metric(self, name: str, values: dict[str, float]) -> VarianceMetric:
        """Create a variance metric from provider values."""
        import statistics

        vals = list(values.values())
        if not vals:
            return VarianceMetric(
                metric_name=name,
                values=values,
                mean=0.0,
                std_dev=0.0,
                min_value=0.0,
                max_value=0.0,
                variance_ratio=1.0,
            )

        mean = statistics.mean(vals)
        std_dev = statistics.stdev(vals) if len(vals) > 1 else 0.0
        min_val = min(vals)
        max_val = max(vals)
        variance_ratio = max_val / min_val if min_val > 0 else float('inf')

        return VarianceMetric(
            metric_name=name,
            values=values,
            mean=mean,
            std_dev=std_dev,
            min_value=min_val,
            max_value=max_val,
            variance_ratio=variance_ratio,
        )

    def get_all_variance(self) -> dict[str, dict[str, VarianceMetric]]:
        """Get variance metrics for all tests."""
        return {
            test_name: self.calculate_variance(test_name)
            for test_name in self._results_by_test.keys()
        }

    def get_summary(self) -> dict:
        """Get a summary of variance across all tests."""
        all_variance = self.get_all_variance()

        if not all_variance:
            return {
                "total_tests": 0,
                "avg_duration_variance_ratio": 0.0,
                "avg_rounds_variance_ratio": 0.0,
                "avg_tokens_variance_ratio": 0.0,
            }

        duration_ratios = []
        rounds_ratios = []
        tokens_ratios = []

        for metrics in all_variance.values():
            if "duration_ms" in metrics:
                duration_ratios.append(metrics["duration_ms"].variance_ratio)
            if "rounds_count" in metrics:
                rounds_ratios.append(metrics["rounds_count"].variance_ratio)
            if "tokens_used" in metrics:
                tokens_ratios.append(metrics["tokens_used"].variance_ratio)

        return {
            "total_tests": len(all_variance),
            "avg_duration_variance_ratio": (
                sum(duration_ratios) / len(duration_ratios) if duration_ratios else 0.0
            ),
            "avg_rounds_variance_ratio": (
                sum(rounds_ratios) / len(rounds_ratios) if rounds_ratios else 0.0
            ),
            "avg_tokens_variance_ratio": (
                sum(tokens_ratios) / len(tokens_ratios) if tokens_ratios else 0.0
            ),
        }

    def clear(self):
        """Clear all tracked results."""
        self._results_by_test.clear()


class TestReporter:
    """Generates test reports."""

    def __init__(self, executor: TestExecutor, variance_tracker: VarianceTracker):
        self.executor = executor
        self.variance_tracker = variance_tracker

    def generate_summary(self) -> dict:
        """Generate a test summary report."""
        results = self.executor.get_results()
        total = len(results)
        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in results if r.status == TestStatus.FAILED)
        errors = sum(1 for r in results if r.status == TestStatus.ERROR)
        skipped = sum(1 for r in results if r.status == TestStatus.SKIPPED)

        return {
            "suite_name": self.executor.suite.name,
            "total_executions": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "skipped": skipped,
            "pass_rate": passed / total if total > 0 else 0.0,
            "providers_tested": len(self.executor.suite.get_providers()),
            "test_cases": len(self.executor.suite.get_test_cases()),
        }

    def generate_detailed_report(self) -> dict:
        """Generate a detailed test report."""
        results = self.executor.get_results()
        variance_summary = self.variance_tracker.get_summary()

        # Group results by test case
        by_test: dict[str, list[dict]] = {}
        for result in results:
            test_name = result.test_case.name
            if test_name not in by_test:
                by_test[test_name] = []
            by_test[test_name].append(result.to_dict())

        # Group results by provider
        by_provider: dict[str, list[dict]] = {}
        for result in results:
            provider_name = result.provider.name
            if provider_name not in by_provider:
                by_provider[provider_name] = []
            by_provider[provider_name].append(result.to_dict())

        return {
            "summary": self.generate_summary(),
            "variance": variance_summary,
            "by_test": by_test,
            "by_provider": by_provider,
            "all_results": [r.to_dict() for r in results],
        }

    def export_json(self, path: str):
        """Export report to JSON file."""
        report = self.generate_detailed_report()
        Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")

    def export_junit_xml(self, path: str):
        """Export report to JUnit XML format for CI integration."""
        results = self.executor.get_results()

        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append(f'<testsuite name="{self.executor.suite.name}" tests="{len(results)}">')

        for result in results:
            status = "pass" if result.passed else "fail"
            lines.append(
                f'  <testcase name="{result.test_case.name}" '
                f'classname="{result.provider.name}" '
                f'time="{result.duration_ms / 1000:.3f}">'
            )
            if result.status == TestStatus.FAILED:
                lines.append(f'    <failure message="{result.error_message or "Test failed"}" />')
            elif result.status == TestStatus.ERROR:
                lines.append(f'    <error message="{result.error_message or "Test error"}" />')
            elif result.status == TestStatus.SKIPPED:
                lines.append('    <skipped />')
            lines.append('  </testcase>')

        lines.append('</testsuite>')

        Path(path).write_text('\n'.join(lines), encoding="utf-8")


# Factory functions
def create_test_suite(name: str) -> TestSuite:
    """Create a new test suite."""
    return TestSuite(name)


def create_provider(
    name: str,
    model: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    **extra_params,
) -> ProviderConfig:
    """Create a provider configuration."""
    return ProviderConfig(
        name=name,
        model=model,
        api_base=api_base,
        api_key=api_key,
        extra_params=extra_params,
    )


def create_test_case(
    name: str,
    topic: str,
    objective: str,
    expected_consensus: Optional[bool] = None,
    max_rounds: int = 3,
    personas_count: int = 3,
    timeout_ms: int = 60000,
    tags: Optional[list[str]] = None,
    **metadata,
) -> TestCase:
    """Create a test case."""
    return TestCase(
        name=name,
        topic=topic,
        objective=objective,
        expected_consensus=expected_consensus,
        max_rounds=max_rounds,
        personas_count=personas_count,
        timeout_ms=timeout_ms,
        tags=tags or [],
        metadata=metadata,
    )


# Predefined test suites
def get_standard_test_suite() -> TestSuite:
    """Get a standard test suite with common test cases."""
    suite = TestSuite("standard")

    # Basic consensus tests
    suite.add_test_case(TestCase(
        name="simple_consensus",
        topic="Best programming language for beginners",
        objective="Reach consensus on the best language for teaching programming",
        expected_consensus=True,
        max_rounds=3,
        tags=["consensus", "simple"],
    ))

    suite.add_test_case(TestCase(
        name="technical_decision",
        topic="Database choice for a new web application",
        objective="Decide between SQL and NoSQL for a medium-scale web app",
        expected_consensus=True,
        max_rounds=4,
        tags=["consensus", "technical"],
    ))

    suite.add_test_case(TestCase(
        name="controversial_topic",
        topic="Tabs vs Spaces in code formatting",
        objective="Attempt to reach consensus on tabs vs spaces",
        expected_consensus=None,  # May not reach consensus
        max_rounds=3,
        tags=["consensus", "controversial"],
    ))

    suite.add_test_case(TestCase(
        name="multi_option_decision",
        topic="Cloud provider selection",
        objective="Choose between AWS, GCP, and Azure for a startup",
        expected_consensus=True,
        max_rounds=5,
        tags=["consensus", "complex"],
    ))

    suite.add_test_case(TestCase(
        name="architecture_review",
        topic="Microservices vs Monolith architecture",
        objective="Recommend architecture for a growing e-commerce platform",
        expected_consensus=True,
        max_rounds=4,
        tags=["consensus", "architecture"],
    ))

    return suite

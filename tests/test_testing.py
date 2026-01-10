"""Tests for US-07: Parameterized testing framework."""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime

from llm_council.testing import (
    TestStatus,
    ProviderConfig,
    TestCase,
    TestResult,
    VarianceMetric,
    TestSuite,
    TestExecutor,
    VarianceTracker,
    TestReporter,
    create_test_suite,
    create_provider,
    create_test_case,
    get_standard_test_suite,
)


class TestTestStatus:
    """Tests for TestStatus enum."""

    def test_status_values(self):
        assert TestStatus.PENDING.value == "pending"
        assert TestStatus.RUNNING.value == "running"
        assert TestStatus.PASSED.value == "passed"
        assert TestStatus.FAILED.value == "failed"
        assert TestStatus.SKIPPED.value == "skipped"
        assert TestStatus.ERROR.value == "error"


class TestProviderConfig:
    """Tests for ProviderConfig dataclass."""

    def test_provider_creation(self):
        provider = ProviderConfig(
            name="test-provider",
            model="gpt-4",
            api_base="http://localhost:1234/v1",
        )
        assert provider.name == "test-provider"
        assert provider.model == "gpt-4"
        assert provider.api_base == "http://localhost:1234/v1"

    def test_provider_to_dict(self):
        provider = ProviderConfig(
            name="test",
            model="model-1",
            extra_params={"temperature": 0.7},
        )
        d = provider.to_dict()
        assert d["name"] == "test"
        assert d["model"] == "model-1"
        assert d["extra_params"] == {"temperature": 0.7}


class TestTestCase:
    """Tests for TestCase dataclass."""

    def test_test_case_creation(self):
        tc = TestCase(
            name="test-1",
            topic="Test Topic",
            objective="Test Objective",
            expected_consensus=True,
        )
        assert tc.name == "test-1"
        assert tc.expected_consensus is True
        assert tc.max_rounds == 3
        assert tc.personas_count == 3

    def test_test_case_with_tags(self):
        tc = TestCase(
            name="test-1",
            topic="Topic",
            objective="Objective",
            tags=["fast", "simple"],
        )
        assert tc.tags == ["fast", "simple"]

    def test_test_case_to_dict(self):
        tc = TestCase(
            name="test-1",
            topic="Topic",
            objective="Objective",
            max_rounds=5,
            tags=["test"],
            metadata={"key": "value"},
        )
        d = tc.to_dict()
        assert d["name"] == "test-1"
        assert d["max_rounds"] == 5
        assert d["tags"] == ["test"]
        assert d["metadata"] == {"key": "value"}


class TestTestResult:
    """Tests for TestResult dataclass."""

    def test_result_creation(self):
        tc = TestCase(name="t1", topic="T", objective="O")
        provider = ProviderConfig(name="p1", model="m1")
        result = TestResult(
            test_case=tc,
            provider=provider,
            status=TestStatus.PASSED,
            duration_ms=1500.0,
            consensus_reached=True,
            rounds_count=2,
        )
        assert result.status == TestStatus.PASSED
        assert result.passed is True
        assert result.duration_ms == 1500.0

    def test_result_failed(self):
        tc = TestCase(name="t1", topic="T", objective="O")
        provider = ProviderConfig(name="p1", model="m1")
        result = TestResult(
            test_case=tc,
            provider=provider,
            status=TestStatus.FAILED,
            duration_ms=500.0,
            error_message="Assertion failed",
        )
        assert result.passed is False
        assert result.error_message == "Assertion failed"

    def test_result_to_dict(self):
        tc = TestCase(name="t1", topic="T", objective="O")
        provider = ProviderConfig(name="p1", model="m1")
        result = TestResult(
            test_case=tc,
            provider=provider,
            status=TestStatus.PASSED,
            duration_ms=1500.0,
            consensus_reached=True,
            rounds_count=2,
            tokens_used=1000,
        )
        d = result.to_dict()
        assert d["test_case"] == "t1"
        assert d["provider"] == "p1"
        assert d["status"] == "passed"
        assert d["rounds_count"] == 2


class TestVarianceMetric:
    """Tests for VarianceMetric dataclass."""

    def test_variance_metric_creation(self):
        metric = VarianceMetric(
            metric_name="duration",
            values={"p1": 100.0, "p2": 150.0},
            mean=125.0,
            std_dev=35.36,
            min_value=100.0,
            max_value=150.0,
            variance_ratio=1.5,
        )
        assert metric.metric_name == "duration"
        assert metric.variance_ratio == 1.5

    def test_variance_metric_to_dict(self):
        metric = VarianceMetric(
            metric_name="tokens",
            values={"p1": 1000.0, "p2": 1200.0},
            mean=1100.0,
            std_dev=141.42,
            min_value=1000.0,
            max_value=1200.0,
            variance_ratio=1.2,
        )
        d = metric.to_dict()
        assert d["metric_name"] == "tokens"
        assert d["mean"] == 1100.0


class TestTestSuite:
    """Tests for TestSuite class."""

    def test_suite_creation(self):
        suite = TestSuite("test-suite")
        assert suite.name == "test-suite"
        assert suite.count_tests() == 0
        assert suite.count_providers() == 0

    def test_add_test_case(self):
        suite = TestSuite("test")
        tc = TestCase(name="t1", topic="T", objective="O")
        suite.add_test_case(tc)
        assert suite.count_tests() == 1

    def test_add_provider(self):
        suite = TestSuite("test")
        provider = ProviderConfig(name="p1", model="m1")
        suite.add_provider(provider)
        assert suite.count_providers() == 1

    def test_get_test_cases(self):
        suite = TestSuite("test")
        suite.add_test_case(TestCase(name="t1", topic="T", objective="O", tags=["fast"]))
        suite.add_test_case(TestCase(name="t2", topic="T", objective="O", tags=["slow"]))

        all_tests = suite.get_test_cases()
        assert len(all_tests) == 2

        fast_tests = suite.get_test_cases(tags=["fast"])
        assert len(fast_tests) == 1
        assert fast_tests[0].name == "t1"

    def test_total_executions(self):
        suite = TestSuite("test")
        suite.add_test_case(TestCase(name="t1", topic="T", objective="O"))
        suite.add_test_case(TestCase(name="t2", topic="T", objective="O"))
        suite.add_provider(ProviderConfig(name="p1", model="m1"))
        suite.add_provider(ProviderConfig(name="p2", model="m2"))

        assert suite.total_executions() == 4  # 2 tests * 2 providers

    def test_suite_to_dict(self):
        suite = TestSuite("test")
        suite.add_test_case(TestCase(name="t1", topic="T", objective="O"))
        suite.add_provider(ProviderConfig(name="p1", model="m1"))

        d = suite.to_dict()
        assert d["name"] == "test"
        assert len(d["test_cases"]) == 1
        assert len(d["providers"]) == 1
        assert d["total_executions"] == 1


class TestTestExecutor:
    """Tests for TestExecutor class."""

    def test_executor_creation(self):
        suite = TestSuite("test")
        executor = TestExecutor(suite)
        assert executor.suite is suite
        assert len(executor.get_results()) == 0

    def test_run_without_executor(self):
        suite = TestSuite("test")
        suite.add_test_case(TestCase(name="t1", topic="T", objective="O"))
        suite.add_provider(ProviderConfig(name="p1", model="m1"))

        executor = TestExecutor(suite)
        results = executor.run_all()

        assert len(results) == 1
        assert results[0].status == TestStatus.SKIPPED

    def test_run_with_mock_executor(self):
        suite = TestSuite("test")
        suite.add_test_case(TestCase(name="t1", topic="T", objective="O"))
        suite.add_provider(ProviderConfig(name="p1", model="m1"))

        def mock_executor(tc, provider):
            return TestResult(
                test_case=tc,
                provider=provider,
                status=TestStatus.PASSED,
                duration_ms=100.0,
                consensus_reached=True,
                rounds_count=2,
            )

        executor = TestExecutor(suite)
        executor.set_executor(mock_executor)
        results = executor.run_all()

        assert len(results) == 1
        assert results[0].status == TestStatus.PASSED

    def test_run_multiple_tests_providers(self):
        suite = TestSuite("test")
        suite.add_test_case(TestCase(name="t1", topic="T1", objective="O1"))
        suite.add_test_case(TestCase(name="t2", topic="T2", objective="O2"))
        suite.add_provider(ProviderConfig(name="p1", model="m1"))
        suite.add_provider(ProviderConfig(name="p2", model="m2"))

        def mock_executor(tc, provider):
            return TestResult(
                test_case=tc,
                provider=provider,
                status=TestStatus.PASSED,
                duration_ms=100.0,
            )

        executor = TestExecutor(suite)
        executor.set_executor(mock_executor)
        results = executor.run_all()

        assert len(results) == 4  # 2 tests * 2 providers

    def test_run_with_tags(self):
        suite = TestSuite("test")
        suite.add_test_case(TestCase(name="t1", topic="T", objective="O", tags=["fast"]))
        suite.add_test_case(TestCase(name="t2", topic="T", objective="O", tags=["slow"]))
        suite.add_provider(ProviderConfig(name="p1", model="m1"))

        def mock_executor(tc, provider):
            return TestResult(
                test_case=tc,
                provider=provider,
                status=TestStatus.PASSED,
                duration_ms=100.0,
            )

        executor = TestExecutor(suite)
        executor.set_executor(mock_executor)
        results = executor.run_all(tags=["fast"])

        assert len(results) == 1
        assert results[0].test_case.name == "t1"

    def test_run_specific_test(self):
        suite = TestSuite("test")
        suite.add_test_case(TestCase(name="t1", topic="T1", objective="O1"))
        suite.add_test_case(TestCase(name="t2", topic="T2", objective="O2"))
        suite.add_provider(ProviderConfig(name="p1", model="m1"))
        suite.add_provider(ProviderConfig(name="p2", model="m2"))

        def mock_executor(tc, provider):
            return TestResult(
                test_case=tc,
                provider=provider,
                status=TestStatus.PASSED,
                duration_ms=100.0,
            )

        executor = TestExecutor(suite)
        executor.set_executor(mock_executor)
        results = executor.run_test("t1", "p1")

        assert len(results) == 1
        assert results[0].test_case.name == "t1"
        assert results[0].provider.name == "p1"

    def test_executor_handles_exception(self):
        suite = TestSuite("test")
        suite.add_test_case(TestCase(name="t1", topic="T", objective="O"))
        suite.add_provider(ProviderConfig(name="p1", model="m1"))

        def failing_executor(tc, provider):
            raise RuntimeError("Test error")

        executor = TestExecutor(suite)
        executor.set_executor(failing_executor)
        results = executor.run_all()

        assert len(results) == 1
        assert results[0].status == TestStatus.ERROR
        assert "Test error" in results[0].error_message

    def test_clear_results(self):
        suite = TestSuite("test")
        suite.add_test_case(TestCase(name="t1", topic="T", objective="O"))
        suite.add_provider(ProviderConfig(name="p1", model="m1"))

        executor = TestExecutor(suite)
        executor.run_all()
        assert len(executor.get_results()) == 1

        executor.clear_results()
        assert len(executor.get_results()) == 0


class TestVarianceTracker:
    """Tests for VarianceTracker class."""

    def test_tracker_creation(self):
        tracker = VarianceTracker()
        assert len(tracker.get_all_variance()) == 0

    def test_add_results(self):
        tracker = VarianceTracker()
        tc = TestCase(name="t1", topic="T", objective="O")
        results = [
            TestResult(
                test_case=tc,
                provider=ProviderConfig(name="p1", model="m1"),
                status=TestStatus.PASSED,
                duration_ms=100.0,
                rounds_count=2,
                tokens_used=1000,
                consensus_reached=True,
            ),
            TestResult(
                test_case=tc,
                provider=ProviderConfig(name="p2", model="m2"),
                status=TestStatus.PASSED,
                duration_ms=150.0,
                rounds_count=3,
                tokens_used=1200,
                consensus_reached=True,
            ),
        ]
        tracker.add_results(results)

        variance = tracker.calculate_variance("t1")
        assert "duration_ms" in variance
        assert "rounds_count" in variance
        assert "tokens_used" in variance

    def test_variance_calculation(self):
        tracker = VarianceTracker()
        tc = TestCase(name="t1", topic="T", objective="O")
        results = [
            TestResult(
                test_case=tc,
                provider=ProviderConfig(name="p1", model="m1"),
                status=TestStatus.PASSED,
                duration_ms=100.0,
            ),
            TestResult(
                test_case=tc,
                provider=ProviderConfig(name="p2", model="m2"),
                status=TestStatus.PASSED,
                duration_ms=200.0,
            ),
        ]
        tracker.add_results(results)

        variance = tracker.calculate_variance("t1")
        duration = variance["duration_ms"]

        assert duration.min_value == 100.0
        assert duration.max_value == 200.0
        assert duration.mean == 150.0
        assert duration.variance_ratio == 2.0

    def test_get_summary(self):
        tracker = VarianceTracker()
        tc = TestCase(name="t1", topic="T", objective="O")
        results = [
            TestResult(
                test_case=tc,
                provider=ProviderConfig(name="p1", model="m1"),
                status=TestStatus.PASSED,
                duration_ms=100.0,
                rounds_count=2,
                tokens_used=1000,
            ),
            TestResult(
                test_case=tc,
                provider=ProviderConfig(name="p2", model="m2"),
                status=TestStatus.PASSED,
                duration_ms=200.0,
                rounds_count=4,
                tokens_used=2000,
            ),
        ]
        tracker.add_results(results)

        summary = tracker.get_summary()
        assert summary["total_tests"] == 1
        assert summary["avg_duration_variance_ratio"] == 2.0

    def test_clear(self):
        tracker = VarianceTracker()
        tc = TestCase(name="t1", topic="T", objective="O")
        results = [
            TestResult(
                test_case=tc,
                provider=ProviderConfig(name="p1", model="m1"),
                status=TestStatus.PASSED,
                duration_ms=100.0,
            ),
        ]
        tracker.add_results(results)
        assert len(tracker.get_all_variance()) == 1

        tracker.clear()
        assert len(tracker.get_all_variance()) == 0


class TestTestReporter:
    """Tests for TestReporter class."""

    def test_generate_summary(self):
        suite = TestSuite("test")
        suite.add_test_case(TestCase(name="t1", topic="T", objective="O"))
        suite.add_provider(ProviderConfig(name="p1", model="m1"))

        def mock_executor(tc, provider):
            return TestResult(
                test_case=tc,
                provider=provider,
                status=TestStatus.PASSED,
                duration_ms=100.0,
            )

        executor = TestExecutor(suite)
        executor.set_executor(mock_executor)
        executor.run_all()

        tracker = VarianceTracker()
        reporter = TestReporter(executor, tracker)

        summary = reporter.generate_summary()
        assert summary["suite_name"] == "test"
        assert summary["total_executions"] == 1
        assert summary["passed"] == 1
        assert summary["pass_rate"] == 1.0

    def test_generate_detailed_report(self):
        suite = TestSuite("test")
        suite.add_test_case(TestCase(name="t1", topic="T", objective="O"))
        suite.add_provider(ProviderConfig(name="p1", model="m1"))

        def mock_executor(tc, provider):
            return TestResult(
                test_case=tc,
                provider=provider,
                status=TestStatus.PASSED,
                duration_ms=100.0,
            )

        executor = TestExecutor(suite)
        executor.set_executor(mock_executor)
        executor.run_all()

        tracker = VarianceTracker()
        reporter = TestReporter(executor, tracker)

        report = reporter.generate_detailed_report()
        assert "summary" in report
        assert "variance" in report
        assert "by_test" in report
        assert "by_provider" in report
        assert "all_results" in report

    def test_export_json(self):
        suite = TestSuite("test")
        suite.add_test_case(TestCase(name="t1", topic="T", objective="O"))
        suite.add_provider(ProviderConfig(name="p1", model="m1"))

        executor = TestExecutor(suite)
        executor.run_all()

        tracker = VarianceTracker()
        reporter = TestReporter(executor, tracker)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            reporter.export_json(path)
            content = Path(path).read_text(encoding="utf-8")
            data = json.loads(content)
            assert "summary" in data
        finally:
            Path(path).unlink(missing_ok=True)

    def test_export_junit_xml(self):
        suite = TestSuite("test")
        suite.add_test_case(TestCase(name="t1", topic="T", objective="O"))
        suite.add_provider(ProviderConfig(name="p1", model="m1"))

        executor = TestExecutor(suite)
        executor.run_all()

        tracker = VarianceTracker()
        reporter = TestReporter(executor, tracker)

        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            path = f.name

        try:
            reporter.export_junit_xml(path)
            content = Path(path).read_text(encoding="utf-8")
            assert '<?xml version="1.0"' in content
            assert '<testsuite' in content
            assert '<testcase' in content
        finally:
            Path(path).unlink(missing_ok=True)


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_test_suite(self):
        suite = create_test_suite("my-suite")
        assert suite.name == "my-suite"
        assert isinstance(suite, TestSuite)

    def test_create_provider(self):
        provider = create_provider(
            name="test",
            model="gpt-4",
            api_base="http://localhost:1234",
            temperature=0.7,
        )
        assert provider.name == "test"
        assert provider.model == "gpt-4"
        assert provider.extra_params == {"temperature": 0.7}

    def test_create_test_case(self):
        tc = create_test_case(
            name="test-1",
            topic="Topic",
            objective="Objective",
            expected_consensus=True,
            tags=["fast"],
            custom_key="custom_value",
        )
        assert tc.name == "test-1"
        assert tc.expected_consensus is True
        assert tc.tags == ["fast"]
        assert tc.metadata == {"custom_key": "custom_value"}


class TestStandardTestSuite:
    """Tests for the standard test suite."""

    def test_get_standard_suite(self):
        suite = get_standard_test_suite()
        assert suite.name == "standard"
        assert suite.count_tests() >= 5

    def test_standard_suite_has_tagged_tests(self):
        suite = get_standard_test_suite()
        consensus_tests = suite.get_test_cases(tags=["consensus"])
        assert len(consensus_tests) >= 5

    def test_standard_suite_test_cases_complete(self):
        suite = get_standard_test_suite()
        for tc in suite.get_test_cases():
            assert tc.name
            assert tc.topic
            assert tc.objective
            assert tc.max_rounds > 0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_suite_summary(self):
        suite = TestSuite("empty")
        executor = TestExecutor(suite)
        tracker = VarianceTracker()
        reporter = TestReporter(executor, tracker)

        summary = reporter.generate_summary()
        assert summary["total_executions"] == 0
        assert summary["pass_rate"] == 0.0

    def test_variance_with_single_provider(self):
        tracker = VarianceTracker()
        tc = TestCase(name="t1", topic="T", objective="O")
        results = [
            TestResult(
                test_case=tc,
                provider=ProviderConfig(name="p1", model="m1"),
                status=TestStatus.PASSED,
                duration_ms=100.0,
            ),
        ]
        tracker.add_results(results)

        variance = tracker.calculate_variance("t1")
        assert variance["duration_ms"].std_dev == 0.0
        assert variance["duration_ms"].variance_ratio == 1.0

    def test_mixed_status_results(self):
        suite = TestSuite("test")
        suite.add_test_case(TestCase(name="t1", topic="T", objective="O"))
        suite.add_provider(ProviderConfig(name="p1", model="m1"))
        suite.add_provider(ProviderConfig(name="p2", model="m2"))

        call_count = [0]

        def mock_executor(tc, provider):
            call_count[0] += 1
            if call_count[0] == 1:
                return TestResult(
                    test_case=tc,
                    provider=provider,
                    status=TestStatus.PASSED,
                    duration_ms=100.0,
                )
            else:
                return TestResult(
                    test_case=tc,
                    provider=provider,
                    status=TestStatus.FAILED,
                    duration_ms=50.0,
                    error_message="Failed",
                )

        executor = TestExecutor(suite)
        executor.set_executor(mock_executor)
        results = executor.run_all()

        tracker = VarianceTracker()
        reporter = TestReporter(executor, tracker)

        summary = reporter.generate_summary()
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["pass_rate"] == 0.5

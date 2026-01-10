"""Tests for US-06: Performance metrics module."""

import pytest
import time
from datetime import datetime, timedelta

from llm_council.metrics import (
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


class TestMetricType:
    """Tests for MetricType enum."""

    def test_metric_types(self):
        assert MetricType.LATENCY.value == "latency"
        assert MetricType.TOKENS.value == "tokens"
        assert MetricType.ROUNDS.value == "rounds"
        assert MetricType.CONSENSUS.value == "consensus"
        assert MetricType.CUSTOM.value == "custom"


class TestMetricPoint:
    """Tests for MetricPoint dataclass."""

    def test_metric_point_creation(self):
        point = MetricPoint(
            name="test_metric",
            value=42.5,
            timestamp=datetime.now(),
            metric_type=MetricType.LATENCY,
        )
        assert point.name == "test_metric"
        assert point.value == 42.5
        assert point.metric_type == MetricType.LATENCY
        assert point.tags == {}

    def test_metric_point_to_dict(self):
        now = datetime.now()
        point = MetricPoint(
            name="test",
            value=100.0,
            timestamp=now,
            metric_type=MetricType.TOKENS,
            tags={"session": "123"},
        )
        d = point.to_dict()
        assert d["name"] == "test"
        assert d["value"] == 100.0
        assert d["timestamp"] == now.isoformat()
        assert d["metric_type"] == "tokens"
        assert d["tags"] == {"session": "123"}


class TestAggregatedMetric:
    """Tests for AggregatedMetric dataclass."""

    def test_aggregated_metric_to_dict(self):
        agg = AggregatedMetric(
            name="latency",
            count=10,
            min_value=50.0,
            max_value=150.0,
            mean=100.0,
            median=95.0,
            p95=140.0,
            p99=148.0,
            std_dev=25.5,
            total=1000.0,
        )
        d = agg.to_dict()
        assert d["name"] == "latency"
        assert d["count"] == 10
        assert d["min"] == 50.0
        assert d["max"] == 150.0
        assert d["mean"] == 100.0
        assert d["p95"] == 140.0


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_record_basic(self):
        collector = MetricsCollector()
        point = collector.record("test", 42.0)

        assert point.name == "test"
        assert point.value == 42.0
        assert collector.count() == 1

    def test_record_latency(self):
        collector = MetricsCollector()
        point = collector.record_latency("api_call", 150.5)

        assert point.metric_type == MetricType.LATENCY
        assert point.value == 150.5

    def test_record_tokens(self):
        collector = MetricsCollector()
        point = collector.record_tokens("session", 1500)

        assert point.metric_type == MetricType.TOKENS
        assert point.value == 1500.0

    def test_record_rounds(self):
        collector = MetricsCollector()
        point = collector.record_rounds("discussion", 3)

        assert point.metric_type == MetricType.ROUNDS
        assert point.value == 3.0

    def test_record_consensus_reached(self):
        collector = MetricsCollector()
        point = collector.record_consensus("session", True)

        assert point.metric_type == MetricType.CONSENSUS
        assert point.value == 1.0

    def test_record_consensus_not_reached(self):
        collector = MetricsCollector()
        point = collector.record_consensus("session", False)

        assert point.value == 0.0

    def test_record_with_tags(self):
        collector = MetricsCollector()
        point = collector.record("test", 100.0, tags={"env": "prod"})

        assert point.tags == {"env": "prod"}

    def test_get_points_all(self):
        collector = MetricsCollector()
        collector.record("a", 1.0)
        collector.record("b", 2.0)
        collector.record("a", 3.0)

        points = collector.get_points()
        assert len(points) == 3

    def test_get_points_by_name(self):
        collector = MetricsCollector()
        collector.record("a", 1.0)
        collector.record("b", 2.0)
        collector.record("a", 3.0)

        points = collector.get_points(name="a")
        assert len(points) == 2

    def test_get_points_by_type(self):
        collector = MetricsCollector()
        collector.record_latency("lat", 100.0)
        collector.record_tokens("tok", 500)
        collector.record_latency("lat", 200.0)

        points = collector.get_points(metric_type=MetricType.LATENCY)
        assert len(points) == 2

    def test_get_points_since(self):
        collector = MetricsCollector()
        collector.record("old", 1.0)
        cutoff = datetime.now()
        time.sleep(0.01)
        collector.record("new", 2.0)

        points = collector.get_points(since=cutoff)
        assert len(points) == 1
        assert points[0].name == "new"

    def test_max_points_trimming(self):
        collector = MetricsCollector(max_points=10)

        for i in range(20):
            collector.record("test", float(i))

        assert collector.count() == 10

    def test_clear(self):
        collector = MetricsCollector()
        collector.record("test", 1.0)
        collector.record("test", 2.0)
        assert collector.count() == 2

        collector.clear()
        assert collector.count() == 0


class TestMetricsAggregator:
    """Tests for MetricsAggregator."""

    def test_aggregate_empty(self):
        result = MetricsAggregator.aggregate([])
        assert result is None

    def test_aggregate_single_point(self):
        point = MetricPoint(
            name="test",
            value=100.0,
            timestamp=datetime.now(),
            metric_type=MetricType.LATENCY,
        )
        result = MetricsAggregator.aggregate([point])

        assert result is not None
        assert result.count == 1
        assert result.mean == 100.0
        assert result.p95 == 100.0
        assert result.std_dev == 0.0

    def test_aggregate_multiple_points(self):
        collector = MetricsCollector()
        for v in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            collector.record("test", float(v))

        points = collector.get_points()
        result = MetricsAggregator.aggregate(points)

        assert result is not None
        assert result.count == 10
        assert result.min_value == 10.0
        assert result.max_value == 100.0
        assert result.mean == 55.0
        assert result.total == 550.0

    def test_aggregate_p95(self):
        collector = MetricsCollector()
        # Create 100 points: 1-100
        for i in range(1, 101):
            collector.record("test", float(i))

        points = collector.get_points()
        result = MetricsAggregator.aggregate(points)

        # P95 should be around 95
        assert result is not None
        assert result.p95 >= 90

    def test_aggregate_by_name(self):
        collector = MetricsCollector()
        collector.record("a", 10.0)
        collector.record("b", 20.0)
        collector.record("a", 30.0)
        collector.record("b", 40.0)

        points = collector.get_points()
        by_name = MetricsAggregator.aggregate_by_name(points)

        assert "a" in by_name
        assert "b" in by_name
        assert by_name["a"].mean == 20.0  # (10+30)/2
        assert by_name["b"].mean == 30.0  # (20+40)/2


class TestTimer:
    """Tests for Timer context manager."""

    def test_timer_records_latency(self):
        collector = MetricsCollector()

        with Timer(collector, "test_op"):
            time.sleep(0.01)  # 10ms

        assert collector.count() == 1
        points = collector.get_points()
        assert points[0].name == "test_op"
        assert points[0].metric_type == MetricType.LATENCY
        assert points[0].value >= 10  # At least 10ms

    def test_timer_elapsed_property(self):
        collector = MetricsCollector()

        with Timer(collector, "test") as t:
            time.sleep(0.01)

        assert t.elapsed_ms is not None
        assert t.elapsed_ms >= 10

    def test_timer_with_tags(self):
        collector = MetricsCollector()

        with Timer(collector, "test", tags={"op": "test"}):
            pass

        points = collector.get_points()
        assert points[0].tags == {"op": "test"}


class TestSessionMetrics:
    """Tests for SessionMetrics dataclass."""

    def test_session_metrics_creation(self):
        metrics = SessionMetrics(
            session_id="test-123",
            topic="Test Topic",
            total_latency_ms=1500.0,
            round_latencies_ms=[500.0, 500.0, 500.0],
            tokens_used=3000,
            rounds_count=3,
            consensus_reached=True,
            personas_count=4,
        )
        assert metrics.session_id == "test-123"
        assert metrics.consensus_reached is True

    def test_avg_round_latency(self):
        metrics = SessionMetrics(
            session_id="test",
            topic="Test",
            total_latency_ms=1500.0,
            round_latencies_ms=[400.0, 500.0, 600.0],
            tokens_used=3000,
            rounds_count=3,
            consensus_reached=True,
            personas_count=4,
        )
        assert metrics.avg_round_latency_ms == 500.0

    def test_avg_round_latency_empty(self):
        metrics = SessionMetrics(
            session_id="test",
            topic="Test",
            total_latency_ms=0.0,
            round_latencies_ms=[],
            tokens_used=0,
            rounds_count=0,
            consensus_reached=False,
            personas_count=4,
        )
        assert metrics.avg_round_latency_ms == 0.0

    def test_tokens_per_round(self):
        metrics = SessionMetrics(
            session_id="test",
            topic="Test",
            total_latency_ms=1500.0,
            round_latencies_ms=[500.0, 500.0, 500.0],
            tokens_used=3000,
            rounds_count=3,
            consensus_reached=True,
            personas_count=4,
        )
        assert metrics.tokens_per_round == 1000.0

    def test_to_dict(self):
        metrics = SessionMetrics(
            session_id="test-123",
            topic="Test Topic",
            total_latency_ms=1500.0,
            round_latencies_ms=[500.0, 500.0, 500.0],
            tokens_used=3000,
            rounds_count=3,
            consensus_reached=True,
            personas_count=4,
        )
        d = metrics.to_dict()
        assert d["session_id"] == "test-123"
        assert d["total_latency_ms"] == 1500.0
        assert d["avg_round_latency_ms"] == 500.0
        assert d["tokens_per_round"] == 1000.0


class TestMetricsReporter:
    """Tests for MetricsReporter."""

    def test_record_session(self):
        collector = MetricsCollector()
        reporter = MetricsReporter(collector)

        metrics = SessionMetrics(
            session_id="test-1",
            topic="Test",
            total_latency_ms=1000.0,
            round_latencies_ms=[500.0, 500.0],
            tokens_used=2000,
            rounds_count=2,
            consensus_reached=True,
            personas_count=3,
        )
        reporter.record_session(metrics)

        # Check collector has entries
        assert collector.count() == 4  # latency, tokens, rounds, consensus

    def test_get_summary_empty(self):
        collector = MetricsCollector()
        reporter = MetricsReporter(collector)

        summary = reporter.get_summary()
        assert summary["total_sessions"] == 0
        assert summary["consensus_rate"] == 0.0

    def test_get_summary_with_data(self):
        collector = MetricsCollector()
        reporter = MetricsReporter(collector)

        # Add two sessions, one with consensus, one without
        reporter.record_session(SessionMetrics(
            session_id="s1", topic="T1", total_latency_ms=1000.0,
            round_latencies_ms=[500.0, 500.0], tokens_used=2000,
            rounds_count=2, consensus_reached=True, personas_count=3,
        ))
        reporter.record_session(SessionMetrics(
            session_id="s2", topic="T2", total_latency_ms=1500.0,
            round_latencies_ms=[500.0, 500.0, 500.0], tokens_used=3000,
            rounds_count=3, consensus_reached=False, personas_count=3,
        ))

        summary = reporter.get_summary()
        assert summary["total_sessions"] == 2
        assert summary["consensus_rate"] == 0.5
        assert summary["total_tokens_used"] == 5000

    def test_get_latency_report(self):
        collector = MetricsCollector()
        reporter = MetricsReporter(collector)

        reporter.record_session(SessionMetrics(
            session_id="s1", topic="T1", total_latency_ms=1000.0,
            round_latencies_ms=[1000.0], tokens_used=1000,
            rounds_count=1, consensus_reached=True, personas_count=3,
        ))

        report = reporter.get_latency_report()
        assert "session_latency" in report

    def test_get_consensus_report(self):
        collector = MetricsCollector()
        reporter = MetricsReporter(collector)

        reporter.record_session(SessionMetrics(
            session_id="s1", topic="T1", total_latency_ms=1000.0,
            round_latencies_ms=[500.0, 500.0], tokens_used=2000,
            rounds_count=2, consensus_reached=True, personas_count=3,
        ))
        reporter.record_session(SessionMetrics(
            session_id="s2", topic="T2", total_latency_ms=1500.0,
            round_latencies_ms=[500.0, 500.0, 500.0], tokens_used=3000,
            rounds_count=3, consensus_reached=False, personas_count=3,
        ))

        report = reporter.get_consensus_report()
        assert report["total_sessions"] == 2
        assert report["consensus_reached"] == 1
        assert report["consensus_failed"] == 1
        assert report["consensus_rate"] == 0.5
        assert report["avg_rounds_when_reached"] == 2.0
        assert report["avg_rounds_when_failed"] == 3.0

    def test_get_full_report(self):
        collector = MetricsCollector()
        reporter = MetricsReporter(collector)

        reporter.record_session(SessionMetrics(
            session_id="s1", topic="T1", total_latency_ms=1000.0,
            round_latencies_ms=[1000.0], tokens_used=1000,
            rounds_count=1, consensus_reached=True, personas_count=3,
        ))

        report = reporter.get_full_report()
        assert "summary" in report
        assert "latency" in report
        assert "tokens" in report
        assert "consensus" in report
        assert "sessions" in report
        assert len(report["sessions"]) == 1

    def test_clear(self):
        collector = MetricsCollector()
        reporter = MetricsReporter(collector)

        reporter.record_session(SessionMetrics(
            session_id="s1", topic="T1", total_latency_ms=1000.0,
            round_latencies_ms=[1000.0], tokens_used=1000,
            rounds_count=1, consensus_reached=True, personas_count=3,
        ))

        reporter.clear()
        assert reporter.get_summary()["total_sessions"] == 0
        assert collector.count() == 0


class TestGlobalFunctions:
    """Tests for global helper functions."""

    def test_get_metrics_collector(self):
        # Reset global state
        import llm_council.metrics as m
        m._metrics_collector = None

        collector = get_metrics_collector()
        assert collector is not None
        assert isinstance(collector, MetricsCollector)

        # Should return same instance
        collector2 = get_metrics_collector()
        assert collector is collector2

    def test_get_metrics_reporter(self):
        import llm_council.metrics as m
        m._metrics_collector = None
        m._metrics_reporter = None

        reporter = get_metrics_reporter()
        assert reporter is not None
        assert isinstance(reporter, MetricsReporter)

    def test_time_operation(self):
        import llm_council.metrics as m
        m._metrics_collector = None

        with time_operation("test_op"):
            time.sleep(0.01)

        collector = get_metrics_collector()
        points = collector.get_points(name="test_op")
        assert len(points) == 1
        assert points[0].value >= 10

    def test_record_latency_global(self):
        import llm_council.metrics as m
        m._metrics_collector = None

        point = record_latency("api", 150.0)
        assert point.value == 150.0

    def test_record_tokens_global(self):
        import llm_council.metrics as m
        m._metrics_collector = None

        point = record_tokens("session", 1500)
        assert point.value == 1500.0

    def test_record_session_metrics_global(self):
        import llm_council.metrics as m
        m._metrics_collector = None
        m._metrics_reporter = None

        metrics = SessionMetrics(
            session_id="test",
            topic="Test",
            total_latency_ms=1000.0,
            round_latencies_ms=[1000.0],
            tokens_used=1000,
            rounds_count=1,
            consensus_reached=True,
            personas_count=3,
        )
        record_session_metrics(metrics)

        summary = get_metrics_summary()
        assert summary["total_sessions"] == 1


class TestMetricsPerformance:
    """Tests for metrics performance requirements."""

    def test_record_1000_points_under_100ms(self):
        collector = MetricsCollector()

        start = time.perf_counter()
        for i in range(1000):
            collector.record("test", float(i))
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, f"Recording 1000 points took {elapsed_ms:.2f}ms"

    def test_aggregate_1000_points_under_50ms(self):
        collector = MetricsCollector()
        for i in range(1000):
            collector.record("test", float(i))

        points = collector.get_points()

        start = time.perf_counter()
        MetricsAggregator.aggregate(points)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, f"Aggregating 1000 points took {elapsed_ms:.2f}ms"

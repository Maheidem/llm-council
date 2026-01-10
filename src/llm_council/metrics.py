"""US-06: Performance metrics and analytics module.

Provides metrics collection, aggregation, and reporting for council sessions.
"""

import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, Any
from enum import Enum


class MetricType(Enum):
    """Types of metrics."""
    LATENCY = "latency"
    TOKENS = "tokens"
    ROUNDS = "rounds"
    CONSENSUS = "consensus"
    CUSTOM = "custom"


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: datetime
    metric_type: MetricType
    tags: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "metric_type": self.metric_type.value,
            "tags": self.tags,
        }


@dataclass
class AggregatedMetric:
    """Aggregated statistics for a metric."""
    name: str
    count: int
    min_value: float
    max_value: float
    mean: float
    median: float
    p95: float
    p99: float
    std_dev: float
    total: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "count": self.count,
            "min": self.min_value,
            "max": self.max_value,
            "mean": round(self.mean, 4),
            "median": round(self.median, 4),
            "p95": round(self.p95, 4),
            "p99": round(self.p99, 4),
            "std_dev": round(self.std_dev, 4),
            "total": round(self.total, 4),
        }


class MetricsCollector:
    """Collects and stores metrics data points."""

    def __init__(self, max_points: int = 10000):
        self._points: list[MetricPoint] = []
        self._max_points = max_points

    def record(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.CUSTOM,
        tags: Optional[dict] = None,
    ) -> MetricPoint:
        """Record a metric data point."""
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(),
            metric_type=metric_type,
            tags=tags or {},
        )
        self._points.append(point)

        # Trim if over capacity
        if len(self._points) > self._max_points:
            self._points = self._points[-self._max_points:]

        return point

    def record_latency(self, name: str, latency_ms: float, tags: Optional[dict] = None) -> MetricPoint:
        """Record a latency metric in milliseconds."""
        return self.record(name, latency_ms, MetricType.LATENCY, tags)

    def record_tokens(self, name: str, tokens: int, tags: Optional[dict] = None) -> MetricPoint:
        """Record token usage."""
        return self.record(name, float(tokens), MetricType.TOKENS, tags)

    def record_rounds(self, name: str, rounds: int, tags: Optional[dict] = None) -> MetricPoint:
        """Record rounds count."""
        return self.record(name, float(rounds), MetricType.ROUNDS, tags)

    def record_consensus(self, name: str, reached: bool, tags: Optional[dict] = None) -> MetricPoint:
        """Record consensus outcome (1.0 = reached, 0.0 = not reached)."""
        return self.record(name, 1.0 if reached else 0.0, MetricType.CONSENSUS, tags)

    def get_points(
        self,
        name: Optional[str] = None,
        metric_type: Optional[MetricType] = None,
        since: Optional[datetime] = None,
    ) -> list[MetricPoint]:
        """Get filtered metric points."""
        points = self._points
        if name:
            points = [p for p in points if p.name == name]
        if metric_type:
            points = [p for p in points if p.metric_type == metric_type]
        if since:
            points = [p for p in points if p.timestamp >= since]
        return points

    def clear(self):
        """Clear all metric points."""
        self._points.clear()

    def count(self) -> int:
        """Get total number of metric points."""
        return len(self._points)


class MetricsAggregator:
    """Aggregates metrics into statistics."""

    @staticmethod
    def aggregate(points: list[MetricPoint]) -> Optional[AggregatedMetric]:
        """Aggregate metric points into statistics."""
        if not points:
            return None

        values = [p.value for p in points]
        name = points[0].name

        if len(values) == 1:
            val = values[0]
            return AggregatedMetric(
                name=name,
                count=1,
                min_value=val,
                max_value=val,
                mean=val,
                median=val,
                p95=val,
                p99=val,
                std_dev=0.0,
                total=val,
            )

        sorted_values = sorted(values)
        n = len(sorted_values)

        # Calculate percentiles
        p95_idx = int(n * 0.95)
        p99_idx = int(n * 0.99)
        p95 = sorted_values[min(p95_idx, n - 1)]
        p99 = sorted_values[min(p99_idx, n - 1)]

        return AggregatedMetric(
            name=name,
            count=n,
            min_value=min(values),
            max_value=max(values),
            mean=statistics.mean(values),
            median=statistics.median(values),
            p95=p95,
            p99=p99,
            std_dev=statistics.stdev(values) if n > 1 else 0.0,
            total=sum(values),
        )

    @staticmethod
    def aggregate_by_name(points: list[MetricPoint]) -> dict[str, AggregatedMetric]:
        """Aggregate points grouped by name."""
        by_name: dict[str, list[MetricPoint]] = {}
        for p in points:
            if p.name not in by_name:
                by_name[p.name] = []
            by_name[p.name].append(p)

        return {
            name: MetricsAggregator.aggregate(pts)
            for name, pts in by_name.items()
            if MetricsAggregator.aggregate(pts) is not None
        }


class Timer:
    """Context manager for timing operations."""

    def __init__(self, collector: MetricsCollector, name: str, tags: Optional[dict] = None):
        self.collector = collector
        self.name = name
        self.tags = tags
        self._start: Optional[float] = None
        self._elapsed_ms: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._start is not None:
            self._elapsed_ms = (time.perf_counter() - self._start) * 1000
            self.collector.record_latency(self.name, self._elapsed_ms, self.tags)

    @property
    def elapsed_ms(self) -> Optional[float]:
        """Get elapsed time in milliseconds (available after exiting context)."""
        return self._elapsed_ms


@dataclass
class SessionMetrics:
    """Metrics for a single council session."""
    session_id: str
    topic: str
    total_latency_ms: float
    round_latencies_ms: list[float]
    tokens_used: int
    rounds_count: int
    consensus_reached: bool
    personas_count: int
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def avg_round_latency_ms(self) -> float:
        """Average latency per round."""
        if not self.round_latencies_ms:
            return 0.0
        return sum(self.round_latencies_ms) / len(self.round_latencies_ms)

    @property
    def tokens_per_round(self) -> float:
        """Average tokens per round."""
        if self.rounds_count == 0:
            return 0.0
        return self.tokens_used / self.rounds_count

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "topic": self.topic,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "round_latencies_ms": [round(l, 2) for l in self.round_latencies_ms],
            "avg_round_latency_ms": round(self.avg_round_latency_ms, 2),
            "tokens_used": self.tokens_used,
            "tokens_per_round": round(self.tokens_per_round, 2),
            "rounds_count": self.rounds_count,
            "consensus_reached": self.consensus_reached,
            "personas_count": self.personas_count,
            "timestamp": self.timestamp.isoformat(),
        }


class MetricsReporter:
    """Generates metrics reports."""

    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self._session_metrics: list[SessionMetrics] = []

    def record_session(self, session_metrics: SessionMetrics):
        """Record a session's metrics."""
        self._session_metrics.append(session_metrics)

        # Also record to collector for aggregation
        self.collector.record_latency(
            "session_latency",
            session_metrics.total_latency_ms,
            {"session_id": session_metrics.session_id}
        )
        self.collector.record_tokens(
            "session_tokens",
            session_metrics.tokens_used,
            {"session_id": session_metrics.session_id}
        )
        self.collector.record_rounds(
            "rounds_to_consensus",
            session_metrics.rounds_count,
            {"session_id": session_metrics.session_id}
        )
        self.collector.record_consensus(
            "consensus_reached",
            session_metrics.consensus_reached,
            {"session_id": session_metrics.session_id}
        )

    def get_summary(self) -> dict:
        """Get a summary of all collected metrics."""
        if not self._session_metrics:
            return {
                "total_sessions": 0,
                "consensus_rate": 0.0,
                "latency_p95_ms": 0.0,
                "avg_rounds_to_consensus": 0.0,
                "total_tokens_used": 0,
            }

        sessions = self._session_metrics
        latencies = [s.total_latency_ms for s in sessions]
        consensus_count = sum(1 for s in sessions if s.consensus_reached)
        rounds_list = [s.rounds_count for s in sessions]

        # Calculate p95 latency
        sorted_latencies = sorted(latencies)
        p95_idx = int(len(sorted_latencies) * 0.95)
        latency_p95 = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)]

        return {
            "total_sessions": len(sessions),
            "consensus_rate": consensus_count / len(sessions),
            "latency_p95_ms": round(latency_p95, 2),
            "avg_rounds_to_consensus": sum(rounds_list) / len(rounds_list),
            "total_tokens_used": sum(s.tokens_used for s in sessions),
            "avg_tokens_per_session": sum(s.tokens_used for s in sessions) / len(sessions),
            "avg_latency_ms": sum(latencies) / len(latencies),
        }

    def get_latency_report(self) -> dict:
        """Get detailed latency report."""
        latency_points = self.collector.get_points(metric_type=MetricType.LATENCY)
        aggregated = MetricsAggregator.aggregate_by_name(latency_points)
        return {name: agg.to_dict() for name, agg in aggregated.items()}

    def get_token_report(self) -> dict:
        """Get detailed token usage report."""
        token_points = self.collector.get_points(metric_type=MetricType.TOKENS)
        aggregated = MetricsAggregator.aggregate_by_name(token_points)
        return {name: agg.to_dict() for name, agg in aggregated.items()}

    def get_consensus_report(self) -> dict:
        """Get consensus statistics."""
        if not self._session_metrics:
            return {
                "total_sessions": 0,
                "consensus_reached": 0,
                "consensus_failed": 0,
                "consensus_rate": 0.0,
                "avg_rounds_when_reached": 0.0,
                "avg_rounds_when_failed": 0.0,
            }

        sessions = self._session_metrics
        reached = [s for s in sessions if s.consensus_reached]
        failed = [s for s in sessions if not s.consensus_reached]

        return {
            "total_sessions": len(sessions),
            "consensus_reached": len(reached),
            "consensus_failed": len(failed),
            "consensus_rate": len(reached) / len(sessions) if sessions else 0.0,
            "avg_rounds_when_reached": (
                sum(s.rounds_count for s in reached) / len(reached) if reached else 0.0
            ),
            "avg_rounds_when_failed": (
                sum(s.rounds_count for s in failed) / len(failed) if failed else 0.0
            ),
        }

    def get_full_report(self) -> dict:
        """Get complete metrics report."""
        return {
            "summary": self.get_summary(),
            "latency": self.get_latency_report(),
            "tokens": self.get_token_report(),
            "consensus": self.get_consensus_report(),
            "sessions": [s.to_dict() for s in self._session_metrics],
        }

    def clear(self):
        """Clear all recorded metrics."""
        self._session_metrics.clear()
        self.collector.clear()


# Global instances
_metrics_collector: Optional[MetricsCollector] = None
_metrics_reporter: Optional[MetricsReporter] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_metrics_reporter() -> MetricsReporter:
    """Get or create the global metrics reporter."""
    global _metrics_reporter
    if _metrics_reporter is None:
        _metrics_reporter = MetricsReporter(get_metrics_collector())
    return _metrics_reporter


def time_operation(name: str, tags: Optional[dict] = None) -> Timer:
    """Create a timer for measuring operation duration."""
    return Timer(get_metrics_collector(), name, tags)


def record_latency(name: str, latency_ms: float, tags: Optional[dict] = None) -> MetricPoint:
    """Record a latency metric."""
    return get_metrics_collector().record_latency(name, latency_ms, tags)


def record_tokens(name: str, tokens: int, tags: Optional[dict] = None) -> MetricPoint:
    """Record token usage."""
    return get_metrics_collector().record_tokens(name, tokens, tags)


def record_session_metrics(metrics: SessionMetrics):
    """Record a session's metrics to the global reporter."""
    get_metrics_reporter().record_session(metrics)


def get_metrics_summary() -> dict:
    """Get summary of all metrics."""
    return get_metrics_reporter().get_summary()

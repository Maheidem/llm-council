"""Tests for US-05: Session persistence with SQLite."""

import pytest
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from llm_council.persistence import (
    RetentionPolicy,
    StoredSession,
    SQLiteStorage,
    SessionExporter,
    SessionManager,
    get_session_manager,
    save_session,
    load_session,
)
from llm_council.models import (
    Persona,
    Message,
    Vote,
    RoundResult,
    CouncilSession,
    VoteChoice,
)


def create_test_persona(name: str = "Test Persona") -> Persona:
    """Create a test persona."""
    return Persona(
        name=name,
        role="Test Role",
        expertise=["testing"],
        personality_traits=["analytical"],
        perspective="Test perspective",
    )


def create_test_session(
    topic: str = "Test Topic",
    consensus_reached: bool = True,
    rounds: int = 2,
) -> CouncilSession:
    """Create a test council session."""
    personas = [create_test_persona(f"Persona {i}") for i in range(3)]

    round_results = []
    for r in range(rounds):
        messages = [
            Message(
                persona_name=f"Persona {i}",
                content=f"Message from persona {i} in round {r}",
                round_number=r,
            )
            for i in range(3)
        ]
        votes = [
            Vote(
                persona_name=f"Persona {i}",
                choice=VoteChoice.AGREE if i < 2 else VoteChoice.DISAGREE,
                reasoning=f"Reasoning from persona {i}",
            )
            for i in range(3)
        ]
        round_results.append(RoundResult(
            round_number=r,
            messages=messages,
            votes=votes,
            consensus_reached=consensus_reached and r == rounds - 1,
        ))

    return CouncilSession(
        topic=topic,
        objective="Test objective",
        personas=personas,
        rounds=round_results,
        consensus_reached=consensus_reached,
        final_consensus="Test final consensus" if consensus_reached else None,
    )


class TestRetentionPolicy:
    """Tests for retention policy enum."""

    def test_retention_values(self):
        assert RetentionPolicy.DAYS_7.value == 7
        assert RetentionPolicy.DAYS_30.value == 30
        assert RetentionPolicy.DAYS_90.value == 90
        assert RetentionPolicy.FOREVER.value == -1


class TestStoredSession:
    """Tests for StoredSession dataclass."""

    def test_stored_session_creation(self):
        session = StoredSession(
            id="test-id",
            topic="Test Topic",
            objective="Test Objective",
            created_at=datetime.now(),
            consensus_reached=True,
            rounds_count=3,
            compressed_size=1024,
        )
        assert session.id == "test-id"
        assert session.topic == "Test Topic"
        assert session.data is None

    def test_stored_session_to_dict(self):
        now = datetime.now()
        session = StoredSession(
            id="test-id",
            topic="Test Topic",
            objective="Test Objective",
            created_at=now,
            consensus_reached=True,
            rounds_count=3,
            compressed_size=1024,
            data={"key": "value"},
        )
        d = session.to_dict()
        assert d["id"] == "test-id"
        assert d["topic"] == "Test Topic"
        assert d["created_at"] == now.isoformat()
        assert d["data"] == {"key": "value"}


class TestSQLiteStorage:
    """Tests for SQLite storage backend."""

    def test_memory_storage(self):
        storage = SQLiteStorage()  # Uses :memory: by default
        assert storage.db_path == ":memory:"

    def test_file_storage(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            storage = SQLiteStorage(db_path)
            assert storage.db_path == db_path
            # Close any connections by letting storage go out of scope
            del storage
        finally:
            # On Windows, may need a small delay for file handle release
            import gc
            gc.collect()
            try:
                Path(db_path).unlink(missing_ok=True)
            except PermissionError:
                pass  # File locked, will be cleaned up later

    def test_save_and_load(self):
        storage = SQLiteStorage()
        session = create_test_session()

        storage.save("test-session", session)
        loaded = storage.load("test-session")

        assert loaded is not None
        assert loaded.id == "test-session"
        assert loaded.topic == "Test Topic"
        assert loaded.consensus_reached is True
        assert loaded.rounds_count == 2
        assert loaded.data is not None

    def test_load_nonexistent(self):
        storage = SQLiteStorage()
        loaded = storage.load("nonexistent")
        assert loaded is None

    def test_delete(self):
        storage = SQLiteStorage()
        session = create_test_session()

        storage.save("test-session", session)
        assert storage.load("test-session") is not None

        result = storage.delete("test-session")
        assert result is True
        assert storage.load("test-session") is None

    def test_delete_nonexistent(self):
        storage = SQLiteStorage()
        result = storage.delete("nonexistent")
        assert result is False

    def test_list_sessions(self):
        storage = SQLiteStorage()

        # Save multiple sessions
        for i in range(5):
            session = create_test_session(topic=f"Topic {i}")
            storage.save(f"session-{i}", session)

        sessions = storage.list_sessions()
        assert len(sessions) == 5

    def test_list_sessions_with_limit(self):
        storage = SQLiteStorage()

        for i in range(10):
            session = create_test_session(topic=f"Topic {i}")
            storage.save(f"session-{i}", session)

        sessions = storage.list_sessions(limit=5)
        assert len(sessions) == 5

    def test_list_sessions_with_offset(self):
        storage = SQLiteStorage()

        for i in range(10):
            session = create_test_session(topic=f"Topic {i}")
            storage.save(f"session-{i}", session)

        sessions = storage.list_sessions(limit=5, offset=5)
        assert len(sessions) == 5

    def test_search_by_topic(self):
        storage = SQLiteStorage()

        storage.save("s1", create_test_session(topic="Python Development"))
        storage.save("s2", create_test_session(topic="JavaScript Development"))
        storage.save("s3", create_test_session(topic="Data Analysis"))

        results = storage.search("Development")
        assert len(results) == 2

    def test_search_no_results(self):
        storage = SQLiteStorage()
        storage.save("s1", create_test_session(topic="Test Topic"))

        results = storage.search("Nonexistent")
        assert len(results) == 0

    def test_compression(self):
        storage = SQLiteStorage()
        session = create_test_session()

        storage.save("test-session", session)
        loaded = storage.load("test-session")

        # Compressed data should be smaller than raw JSON
        raw_json = json.dumps(session.to_dict())
        assert loaded.compressed_size < len(raw_json.encode("utf-8"))

    def test_retention_policy_days_7(self):
        storage = SQLiteStorage(retention_policy=RetentionPolicy.DAYS_7)
        session = create_test_session()

        storage.save("test-session", session)

        # Manually update created_at to 8 days ago
        with storage._connect() as conn:
            old_date = (datetime.now() - timedelta(days=8)).isoformat()
            conn.execute(
                "UPDATE sessions SET created_at = ? WHERE id = ?",
                (old_date, "test-session")
            )
            conn.commit()

        deleted = storage.apply_retention()
        assert deleted == 1
        assert storage.load("test-session") is None

    def test_retention_policy_forever(self):
        storage = SQLiteStorage(retention_policy=RetentionPolicy.FOREVER)
        session = create_test_session()

        storage.save("test-session", session)

        # Even with old date, should not be deleted
        with storage._connect() as conn:
            old_date = (datetime.now() - timedelta(days=365)).isoformat()
            conn.execute(
                "UPDATE sessions SET created_at = ? WHERE id = ?",
                (old_date, "test-session")
            )
            conn.commit()

        deleted = storage.apply_retention()
        assert deleted == 0
        assert storage.load("test-session") is not None

    def test_get_stats_empty(self):
        storage = SQLiteStorage()
        stats = storage.get_stats()

        assert stats["total_sessions"] == 0
        assert stats["total_size_bytes"] == 0
        assert stats["avg_size_bytes"] == 0
        assert stats["oldest_session"] is None
        assert stats["newest_session"] is None

    def test_get_stats_with_data(self):
        storage = SQLiteStorage()

        for i in range(5):
            session = create_test_session(topic=f"Topic {i}")
            storage.save(f"session-{i}", session)

        stats = storage.get_stats()

        assert stats["total_sessions"] == 5
        assert stats["total_size_bytes"] > 0
        assert stats["avg_size_bytes"] > 0
        assert stats["oldest_session"] is not None
        assert stats["newest_session"] is not None
        assert stats["retention_policy"] == "DAYS_30"


class TestSessionExporter:
    """Tests for session export functionality."""

    def test_export_json(self):
        storage = SQLiteStorage()
        storage.save("s1", create_test_session(topic="Topic 1"))
        storage.save("s2", create_test_session(topic="Topic 2"))

        exporter = SessionExporter(storage)
        json_str = exporter.export_json()

        data = json.loads(json_str)
        assert "exported_at" in data
        assert data["session_count"] == 2
        assert len(data["sessions"]) == 2

    def test_export_json_specific_sessions(self):
        storage = SQLiteStorage()
        storage.save("s1", create_test_session(topic="Topic 1"))
        storage.save("s2", create_test_session(topic="Topic 2"))
        storage.save("s3", create_test_session(topic="Topic 3"))

        exporter = SessionExporter(storage)
        json_str = exporter.export_json(session_ids=["s1", "s3"])

        data = json.loads(json_str)
        assert data["session_count"] == 2

    def test_export_csv(self):
        storage = SQLiteStorage()
        storage.save("s1", create_test_session(topic="Topic 1"))
        storage.save("s2", create_test_session(topic="Topic 2"))

        exporter = SessionExporter(storage)
        csv_str = exporter.export_csv()

        lines = csv_str.strip().split("\n")
        assert len(lines) == 3  # Header + 2 rows
        assert "id,topic,objective" in lines[0]

    def test_export_to_file_json(self):
        storage = SQLiteStorage()
        storage.save("s1", create_test_session(topic="Topic 1"))

        exporter = SessionExporter(storage)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            exporter.export_to_file(path, format="json")
            content = Path(path).read_text(encoding="utf-8")
            data = json.loads(content)
            assert data["session_count"] == 1
        finally:
            Path(path).unlink(missing_ok=True)

    def test_export_to_file_csv(self):
        storage = SQLiteStorage()
        storage.save("s1", create_test_session(topic="Topic 1"))

        exporter = SessionExporter(storage)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name

        try:
            exporter.export_to_file(path, format="csv")
            content = Path(path).read_text(encoding="utf-8")
            assert "id,topic,objective" in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_export_unsupported_format(self):
        storage = SQLiteStorage()
        exporter = SessionExporter(storage)

        with pytest.raises(ValueError) as exc_info:
            exporter.export_to_file("test.xml", format="xml")
        assert "Unsupported format" in str(exc_info.value)


class TestSessionManager:
    """Tests for high-level session manager."""

    def test_save_and_load_session(self):
        manager = SessionManager()
        session = create_test_session()

        session_id = manager.save_session(session)
        assert session_id is not None

        loaded = manager.load_session(session_id)
        assert loaded is not None
        assert loaded.topic == "Test Topic"

    def test_save_with_custom_id(self):
        manager = SessionManager()
        session = create_test_session()

        session_id = manager.save_session(session, session_id="custom-id")
        assert session_id == "custom-id"

        loaded = manager.load_session("custom-id")
        assert loaded is not None

    def test_delete_session(self):
        manager = SessionManager()
        session = create_test_session()

        session_id = manager.save_session(session)
        assert manager.delete_session(session_id) is True
        assert manager.load_session(session_id) is None

    def test_list_sessions(self):
        manager = SessionManager()

        for i in range(3):
            manager.save_session(create_test_session(topic=f"Topic {i}"))

        sessions = manager.list_sessions()
        assert len(sessions) == 3

    def test_search_sessions(self):
        manager = SessionManager()

        manager.save_session(create_test_session(topic="Python Development"))
        manager.save_session(create_test_session(topic="JavaScript Development"))
        manager.save_session(create_test_session(topic="Data Analysis"))

        results = manager.search_sessions("Development")
        assert len(results) == 2

    def test_export_json(self):
        manager = SessionManager()
        manager.save_session(create_test_session())

        json_str = manager.export_json()
        data = json.loads(json_str)
        assert data["session_count"] == 1

    def test_export_csv(self):
        manager = SessionManager()
        manager.save_session(create_test_session())

        csv_str = manager.export_csv()
        assert "id,topic,objective" in csv_str

    def test_apply_retention(self):
        manager = SessionManager(retention_policy=RetentionPolicy.DAYS_7)
        manager.save_session(create_test_session())

        # New session should not be deleted
        deleted = manager.apply_retention()
        assert deleted == 0

    def test_get_stats(self):
        manager = SessionManager()
        manager.save_session(create_test_session())
        manager.save_session(create_test_session())

        stats = manager.get_stats()
        assert stats["total_sessions"] == 2


class TestGlobalFunctions:
    """Tests for global helper functions."""

    def test_get_session_manager(self):
        # Reset global manager for clean test
        import llm_council.persistence as p
        p._session_manager = None

        manager = get_session_manager()
        assert manager is not None
        assert isinstance(manager, SessionManager)

    def test_save_and_load_session_globals(self):
        import llm_council.persistence as p
        p._session_manager = None

        session = create_test_session()
        session_id = save_session(session)

        loaded = load_session(session_id)
        assert loaded is not None
        assert loaded.topic == "Test Topic"


class TestPersistencePerformance:
    """Tests for persistence performance requirements."""

    def test_save_under_100ms(self):
        import time

        storage = SQLiteStorage()
        session = create_test_session()

        start = time.perf_counter()
        storage.save("perf-test", session)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, f"Save took {elapsed_ms:.2f}ms, expected < 100ms"

    def test_load_under_50ms(self):
        import time

        storage = SQLiteStorage()
        session = create_test_session()
        storage.save("perf-test", session)

        start = time.perf_counter()
        storage.load("perf-test")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, f"Load took {elapsed_ms:.2f}ms, expected < 50ms"

    def test_list_100_sessions_under_200ms(self):
        import time

        storage = SQLiteStorage()

        # Create 100 sessions
        for i in range(100):
            session = create_test_session(topic=f"Topic {i}")
            storage.save(f"session-{i}", session)

        start = time.perf_counter()
        storage.list_sessions(limit=100)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 200, f"List took {elapsed_ms:.2f}ms, expected < 200ms"

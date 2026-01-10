"""US-05: Session persistence with SQLite.

Provides session storage with configurable retention and export capabilities.
"""

import json
import sqlite3
import csv
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Any, Iterator
import zlib

from .models import CouncilSession


class RetentionPolicy(Enum):
    """Retention policy options."""
    DAYS_7 = 7
    DAYS_30 = 30
    DAYS_90 = 90
    FOREVER = -1


@dataclass
class StoredSession:
    """A stored session with metadata."""
    id: str
    topic: str
    objective: str
    created_at: datetime
    consensus_reached: bool
    rounds_count: int
    compressed_size: int
    data: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "topic": self.topic,
            "objective": self.objective,
            "created_at": self.created_at.isoformat(),
            "consensus_reached": self.consensus_reached,
            "rounds_count": self.rounds_count,
            "compressed_size": self.compressed_size,
            "data": self.data,
        }


class SessionStorage(ABC):
    """Abstract base class for session storage backends."""

    @abstractmethod
    def save(self, session_id: str, session: CouncilSession) -> None:
        """Save a session."""
        pass

    @abstractmethod
    def load(self, session_id: str) -> Optional[StoredSession]:
        """Load a session by ID."""
        pass

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """Delete a session by ID."""
        pass

    @abstractmethod
    def list_sessions(
        self,
        limit: int = 100,
        offset: int = 0,
        since: Optional[datetime] = None,
    ) -> list[StoredSession]:
        """List sessions with pagination."""
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 100) -> list[StoredSession]:
        """Search sessions by topic or objective."""
        pass

    @abstractmethod
    def apply_retention(self, policy: RetentionPolicy) -> int:
        """Apply retention policy and return count of deleted sessions."""
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        """Get storage statistics."""
        pass


class SQLiteStorage(SessionStorage):
    """SQLite-based session storage."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        retention_policy: RetentionPolicy = RetentionPolicy.DAYS_30,
    ):
        self.db_path = db_path or ":memory:"
        self.retention_policy = retention_policy
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                objective TEXT NOT NULL,
                created_at TEXT NOT NULL,
                consensus_reached INTEGER NOT NULL,
                rounds_count INTEGER NOT NULL,
                compressed_data BLOB NOT NULL,
                compressed_size INTEGER NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON sessions(created_at)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_topic ON sessions(topic)
        """)
        conn.commit()

    def _connect(self) -> sqlite3.Connection:
        """Get or create database connection."""
        # For in-memory databases, we need to keep a persistent connection
        # Otherwise each connect() creates a new separate database
        if self.db_path == ":memory:":
            if self._conn is None:
                self._conn = sqlite3.connect(":memory:")
            return self._conn
        return sqlite3.connect(self.db_path)

    def _compress(self, data: dict) -> bytes:
        """Compress session data."""
        json_str = json.dumps(data)
        return zlib.compress(json_str.encode("utf-8"))

    def _decompress(self, data: bytes) -> dict:
        """Decompress session data."""
        json_str = zlib.decompress(data).decode("utf-8")
        return json.loads(json_str)

    def save(self, session_id: str, session: CouncilSession) -> None:
        """Save a session."""
        data = session.to_dict()
        compressed = self._compress(data)

        conn = self._connect()
        conn.execute("""
            INSERT OR REPLACE INTO sessions
            (id, topic, objective, created_at, consensus_reached, rounds_count, compressed_data, compressed_size)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            session.topic,
            session.objective,
            datetime.now().isoformat(),
            1 if session.consensus_reached else 0,
            len(session.rounds),
            compressed,
            len(compressed),
        ))
        conn.commit()

    def load(self, session_id: str) -> Optional[StoredSession]:
        """Load a session by ID."""
        conn = self._connect()
        cursor = conn.execute("""
            SELECT id, topic, objective, created_at, consensus_reached,
                   rounds_count, compressed_data, compressed_size
            FROM sessions WHERE id = ?
        """, (session_id,))
        row = cursor.fetchone()

        if not row:
            return None

        data = self._decompress(row[6])
        return StoredSession(
            id=row[0],
            topic=row[1],
            objective=row[2],
            created_at=datetime.fromisoformat(row[3]),
            consensus_reached=bool(row[4]),
            rounds_count=row[5],
            compressed_size=row[7],
            data=data,
        )

    def delete(self, session_id: str) -> bool:
        """Delete a session by ID."""
        conn = self._connect()
        cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        return cursor.rowcount > 0

    def list_sessions(
        self,
        limit: int = 100,
        offset: int = 0,
        since: Optional[datetime] = None,
    ) -> list[StoredSession]:
        """List sessions with pagination."""
        conn = self._connect()
        if since:
            cursor = conn.execute("""
                SELECT id, topic, objective, created_at, consensus_reached,
                       rounds_count, compressed_size
                FROM sessions
                WHERE created_at >= ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (since.isoformat(), limit, offset))
        else:
            cursor = conn.execute("""
                SELECT id, topic, objective, created_at, consensus_reached,
                       rounds_count, compressed_size
                FROM sessions
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))

        return [
            StoredSession(
                id=row[0],
                topic=row[1],
                objective=row[2],
                created_at=datetime.fromisoformat(row[3]),
                consensus_reached=bool(row[4]),
                rounds_count=row[5],
                compressed_size=row[6],
            )
            for row in cursor.fetchall()
        ]

    def search(self, query: str, limit: int = 100) -> list[StoredSession]:
        """Search sessions by topic or objective."""
        conn = self._connect()
        cursor = conn.execute("""
            SELECT id, topic, objective, created_at, consensus_reached,
                   rounds_count, compressed_size
            FROM sessions
            WHERE topic LIKE ? OR objective LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (f"%{query}%", f"%{query}%", limit))

        return [
            StoredSession(
                id=row[0],
                topic=row[1],
                objective=row[2],
                created_at=datetime.fromisoformat(row[3]),
                consensus_reached=bool(row[4]),
                rounds_count=row[5],
                compressed_size=row[6],
            )
            for row in cursor.fetchall()
        ]

    def apply_retention(self, policy: Optional[RetentionPolicy] = None) -> int:
        """Apply retention policy and return count of deleted sessions."""
        policy = policy or self.retention_policy
        if policy == RetentionPolicy.FOREVER:
            return 0

        cutoff = datetime.now() - timedelta(days=policy.value)
        conn = self._connect()
        cursor = conn.execute(
            "DELETE FROM sessions WHERE created_at < ?",
            (cutoff.isoformat(),)
        )
        conn.commit()
        return cursor.rowcount

    def get_stats(self) -> dict:
        """Get storage statistics."""
        conn = self._connect()
        cursor = conn.execute("""
            SELECT
                COUNT(*) as total_sessions,
                SUM(compressed_size) as total_size,
                AVG(compressed_size) as avg_size,
                MIN(created_at) as oldest,
                MAX(created_at) as newest
            FROM sessions
        """)
        row = cursor.fetchone()

        return {
            "total_sessions": row[0] or 0,
            "total_size_bytes": row[1] or 0,
            "avg_size_bytes": int(row[2] or 0),
            "oldest_session": row[3],
            "newest_session": row[4],
            "retention_policy": self.retention_policy.name,
        }


class SessionExporter:
    """Export sessions to JSON/CSV formats."""

    def __init__(self, storage: SessionStorage):
        self.storage = storage

    def export_json(
        self,
        session_ids: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        include_data: bool = True,
    ) -> str:
        """Export sessions to JSON format."""
        sessions = self._get_sessions(session_ids, since, include_data)
        return json.dumps({
            "exported_at": datetime.now().isoformat(),
            "session_count": len(sessions),
            "sessions": [s.to_dict() for s in sessions],
        }, indent=2)

    def export_csv(
        self,
        session_ids: Optional[list[str]] = None,
        since: Optional[datetime] = None,
    ) -> str:
        """Export sessions to CSV format (metadata only)."""
        sessions = self._get_sessions(session_ids, since, include_data=False)

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "id", "topic", "objective", "created_at",
            "consensus_reached", "rounds_count", "compressed_size"
        ])

        for s in sessions:
            writer.writerow([
                s.id, s.topic, s.objective, s.created_at.isoformat(),
                s.consensus_reached, s.rounds_count, s.compressed_size
            ])

        return output.getvalue()

    def export_to_file(
        self,
        path: str,
        format: str = "json",
        session_ids: Optional[list[str]] = None,
        since: Optional[datetime] = None,
    ) -> None:
        """Export sessions to a file."""
        if format == "json":
            content = self.export_json(session_ids, since)
        elif format == "csv":
            content = self.export_csv(session_ids, since)
        else:
            raise ValueError(f"Unsupported format: {format}")

        Path(path).write_text(content, encoding="utf-8")

    def _get_sessions(
        self,
        session_ids: Optional[list[str]],
        since: Optional[datetime],
        include_data: bool,
    ) -> list[StoredSession]:
        """Get sessions for export."""
        if session_ids:
            sessions = []
            for sid in session_ids:
                session = self.storage.load(sid)
                if session:
                    if not include_data:
                        session.data = None
                    sessions.append(session)
            return sessions
        else:
            sessions = self.storage.list_sessions(limit=10000, since=since)
            if include_data:
                # Load full data for each session
                return [
                    self.storage.load(s.id) or s
                    for s in sessions
                ]
            return sessions


class SessionManager:
    """High-level session management interface."""

    def __init__(
        self,
        storage: Optional[SessionStorage] = None,
        retention_policy: RetentionPolicy = RetentionPolicy.DAYS_30,
    ):
        self.storage = storage or SQLiteStorage(retention_policy=retention_policy)
        self.exporter = SessionExporter(self.storage)
        self._session_counter = 0

    def save_session(self, session: CouncilSession, session_id: Optional[str] = None) -> str:
        """Save a council session and return the session ID."""
        if not session_id:
            self._session_counter += 1
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._session_counter}"

        self.storage.save(session_id, session)
        return session_id

    def load_session(self, session_id: str) -> Optional[StoredSession]:
        """Load a session by ID."""
        return self.storage.load(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        return self.storage.delete(session_id)

    def list_sessions(
        self,
        limit: int = 100,
        offset: int = 0,
        since: Optional[datetime] = None,
    ) -> list[StoredSession]:
        """List sessions."""
        return self.storage.list_sessions(limit, offset, since)

    def search_sessions(self, query: str, limit: int = 100) -> list[StoredSession]:
        """Search sessions."""
        return self.storage.search(query, limit)

    def export_json(self, **kwargs) -> str:
        """Export sessions to JSON."""
        return self.exporter.export_json(**kwargs)

    def export_csv(self, **kwargs) -> str:
        """Export sessions to CSV."""
        return self.exporter.export_csv(**kwargs)

    def apply_retention(self) -> int:
        """Apply retention policy."""
        return self.storage.apply_retention()

    def get_stats(self) -> dict:
        """Get storage statistics."""
        return self.storage.get_stats()


# Default manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager(
    db_path: Optional[str] = None,
    retention_policy: RetentionPolicy = RetentionPolicy.DAYS_30,
) -> SessionManager:
    """Get or create the default session manager."""
    global _session_manager
    if _session_manager is None:
        storage = SQLiteStorage(db_path, retention_policy)
        _session_manager = SessionManager(storage, retention_policy)
    return _session_manager


def save_session(session: CouncilSession, session_id: Optional[str] = None) -> str:
    """Save a session using the default manager."""
    return get_session_manager().save_session(session, session_id)


def load_session(session_id: str) -> Optional[StoredSession]:
    """Load a session using the default manager."""
    return get_session_manager().load_session(session_id)

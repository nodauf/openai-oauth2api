from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import time


@dataclass(slots=True)
class OAuthState:
    state: str
    code_verifier: str
    created_at: int


@dataclass(slots=True)
class SessionRecord:
    session_id: str
    access_token: str
    refresh_token: str | None
    token_type: str
    expires_at: int | None
    created_at: int


@dataclass(slots=True)
class CredentialsBundle:
    version: int
    exported_at: int
    session: SessionRecord


class Store:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS oauth_states (
                    state TEXT PRIMARY KEY,
                    code_verifier TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    access_token TEXT NOT NULL,
                    refresh_token TEXT,
                    token_type TEXT NOT NULL,
                    expires_at INTEGER,
                    created_at INTEGER NOT NULL
                );
                """
            )

    def save_state(self, state: str, code_verifier: str) -> None:
        now = int(time.time())
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO oauth_states(state, code_verifier, created_at) VALUES (?, ?, ?)",
                (state, code_verifier, now),
            )

    def pop_state(self, state: str) -> OAuthState | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT state, code_verifier, created_at FROM oauth_states WHERE state = ?",
                (state,),
            ).fetchone()
            if row is None:
                return None
            connection.execute("DELETE FROM oauth_states WHERE state = ?", (state,))
            return OAuthState(
                state=row["state"],
                code_verifier=row["code_verifier"],
                created_at=row["created_at"],
            )

    def upsert_session(
        self,
        *,
        session_id: str,
        access_token: str,
        refresh_token: str | None,
        token_type: str,
        expires_at: int | None,
        created_at: int | None = None,
    ) -> SessionRecord:
        now = created_at if created_at is not None else int(time.time())
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO sessions(session_id, access_token, refresh_token, token_type, expires_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, access_token, refresh_token, token_type, expires_at, now),
            )
        return SessionRecord(session_id, access_token, refresh_token, token_type, expires_at, now)

    def create_session(
        self,
        *,
        session_id: str,
        access_token: str,
        refresh_token: str | None,
        token_type: str,
        expires_at: int | None,
    ) -> SessionRecord:
        return self.upsert_session(
            session_id=session_id,
            access_token=access_token,
            refresh_token=refresh_token,
            token_type=token_type,
            expires_at=expires_at,
        )

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT session_id, access_token, refresh_token, token_type, expires_at, created_at
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return SessionRecord(
            session_id=row["session_id"],
            access_token=row["access_token"],
            refresh_token=row["refresh_token"],
            token_type=row["token_type"],
            expires_at=row["expires_at"],
            created_at=row["created_at"],
        )

    def list_sessions(self) -> list[SessionRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT session_id, access_token, refresh_token, token_type, expires_at, created_at
                FROM sessions
                ORDER BY created_at DESC
                """
            ).fetchall()
        return [
            SessionRecord(
                session_id=row["session_id"],
                access_token=row["access_token"],
                refresh_token=row["refresh_token"],
                token_type=row["token_type"],
                expires_at=row["expires_at"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def get_latest_session(self) -> SessionRecord | None:
        sessions = self.list_sessions()
        return sessions[0] if sessions else None

    def update_session_tokens(
        self,
        session_id: str,
        *,
        access_token: str,
        refresh_token: str | None,
        token_type: str,
        expires_at: int | None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE sessions
                SET access_token = ?, refresh_token = ?, token_type = ?, expires_at = ?
                WHERE session_id = ?
                """,
                (access_token, refresh_token, token_type, expires_at, session_id),
            )

    def delete_session(self, session_id: str) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    def export_session(self, session_id: str) -> CredentialsBundle:
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(f"unknown session_id: {session_id}")
        return CredentialsBundle(version=1, exported_at=int(time.time()), session=session)

    def import_session(self, bundle: CredentialsBundle | dict) -> SessionRecord:
        if isinstance(bundle, dict):
            version = int(bundle.get("version", 1))
            if version != 1:
                raise ValueError(f"unsupported bundle version: {version}")
            session_data = bundle["session"]
            exported_at = int(bundle.get("exported_at", int(time.time())))
            bundle = CredentialsBundle(
                version=version,
                exported_at=exported_at,
                session=SessionRecord(
                    session_id=session_data["session_id"],
                    access_token=session_data["access_token"],
                    refresh_token=session_data.get("refresh_token"),
                    token_type=session_data.get("token_type", "bearer"),
                    expires_at=session_data.get("expires_at"),
                    created_at=session_data.get("created_at", exported_at),
                ),
            )

        return self.upsert_session(
            session_id=bundle.session.session_id,
            access_token=bundle.session.access_token,
            refresh_token=bundle.session.refresh_token,
            token_type=bundle.session.token_type,
            expires_at=bundle.session.expires_at,
            created_at=bundle.session.created_at,
        )

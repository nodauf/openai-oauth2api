from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openai_oauth_proxy.store import Store


class CredentialsTests(unittest.TestCase):
    def test_export_import_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "proxy.sqlite3"
            store = Store(db_path)
            session = store.create_session(
                session_id="session-123",
                access_token="access-token",
                refresh_token="refresh-token",
                token_type="bearer",
                expires_at=123456,
            )

            bundle = store.export_session(session.session_id)
            self.assertEqual(bundle.session.session_id, "session-123")

            imported_store = Store(Path(tmp) / "imported.sqlite3")
            imported = imported_store.import_session(
                {
                    "version": 1,
                    "exported_at": bundle.exported_at,
                    "session": {
                        "session_id": bundle.session.session_id,
                        "access_token": bundle.session.access_token,
                        "refresh_token": bundle.session.refresh_token,
                        "token_type": bundle.session.token_type,
                        "expires_at": bundle.session.expires_at,
                        "created_at": bundle.session.created_at,
                    },
                }
            )
            self.assertEqual(imported.session_id, "session-123")
            self.assertEqual(imported_store.get_session("session-123").access_token, "access-token")


if __name__ == "__main__":
    unittest.main()

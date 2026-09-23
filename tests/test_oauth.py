from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openai_oauth_proxy.config import Settings
from openai_oauth_proxy.oauth import build_authorize_url, generate_pkce_pair


class OAuthTests(unittest.TestCase):
    def test_build_authorize_url_contains_pkce(self) -> None:
        settings = Settings()
        _, challenge = generate_pkce_pair()
        url = build_authorize_url(
            state="state",
            code_challenge=challenge,
            redirect_uri="http://127.0.0.1:1455/auth/callback",
        )
        self.assertIn("response_type=code", url)
        self.assertIn("client_id=app_EMoamEEZ73f0CkXaXp7hrann", url)
        self.assertIn("state=state", url)
        self.assertIn("code_challenge_method=S256", url)

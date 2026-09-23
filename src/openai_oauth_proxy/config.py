from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(slots=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8000
    data_dir: Path = Path("./data")
    openai_api_base: str = "https://chatgpt.com/backend-api/codex"
    openai_api_key: str = ""
    upstream_auth_mode: str = "oauth_token"
    default_model: str = "gpt-5.3-codex"
    openai_organization: str = ""
    openai_project: str = ""
    request_timeout_seconds: int = 60
    cookie_name: str = "openai_proxy_session"

    @property
    def database_path(self) -> Path:
        return self.data_dir / "proxy.sqlite3"

    @property
    def callback_url(self) -> str:
        return "http://localhost:1455/auth/callback"


def load_settings() -> Settings:
    data_dir = Path(os.getenv("DATA_DIR", "./data"))
    return Settings(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        data_dir=data_dir,
        openai_api_base=os.getenv("OPENAI_API_BASE", "https://chatgpt.com/backend-api/codex"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        upstream_auth_mode=os.getenv("OPENAI_UPSTREAM_AUTH_MODE", "oauth_token"),
        default_model=os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5.3-codex"),
        openai_organization=os.getenv("OPENAI_ORGANIZATION", ""),
        openai_project=os.getenv("OPENAI_PROJECT", ""),
    )

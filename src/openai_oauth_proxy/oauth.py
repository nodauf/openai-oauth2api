from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import json
import secrets
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .auth_config import AUTHORIZE_EXTRA_PARAMS, AUTHORIZE_URL, CLIENT_ID, SCOPE, TOKEN_URL
from .config import Settings


@dataclass(slots=True)
class TokenResponse:
    access_token: str
    refresh_token: str | None
    token_type: str
    expires_in: int | None
    raw: dict


class OAuthError(RuntimeError):
    pass


def generate_pkce_pair() -> tuple[str, str]:
    code_verifier = secrets.token_urlsafe(96)
    challenge = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    code_challenge = base64.urlsafe_b64encode(challenge).rstrip(b"=").decode("ascii")
    return code_verifier, code_challenge


def build_authorize_url(*, state: str, code_challenge: str, redirect_uri: str) -> str:
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": SCOPE,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    params.update(AUTHORIZE_EXTRA_PARAMS)
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


def _exchange_token(
    settings: Settings,
    *,
    grant_type: str,
    extra_fields: dict[str, str],
) -> TokenResponse:
    payload = {
        "grant_type": grant_type,
        "client_id": CLIENT_ID,
        **extra_fields,
    }
    request = Request(
        TOKEN_URL,
        data=urlencode(payload).encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urlopen(request, timeout=settings.request_timeout_seconds) as response:
        raw = json.loads(response.read().decode("utf-8"))

    access_token = raw.get("access_token")
    if not access_token:
        raise OAuthError("token response missing access_token")

    expires_in = raw.get("expires_in")
    return TokenResponse(
        access_token=access_token,
        refresh_token=raw.get("refresh_token"),
        token_type=raw.get("token_type", "bearer"),
        expires_in=int(expires_in) if expires_in is not None else None,
        raw=raw,
    )


def exchange_code_for_token(
    settings: Settings,
    *,
    code: str,
    code_verifier: str,
    redirect_uri: str,
) -> TokenResponse:
    return _exchange_token(
        settings,
        grant_type="authorization_code",
        extra_fields={
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
        },
    )


def refresh_access_token(
    settings: Settings,
    *,
    refresh_token: str,
) -> TokenResponse:
    return _exchange_token(
        settings,
        grant_type="refresh_token",
        extra_fields={"refresh_token": refresh_token},
    )


def expires_at_from_now(expires_in: int | None) -> int | None:
    if expires_in is None:
        return None
    return int(time.time()) + max(0, expires_in)

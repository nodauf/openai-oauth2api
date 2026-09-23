from __future__ import annotations

import json
import secrets
import time
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.parse import parse_qs, urlparse

from .config import Settings, load_settings
from .oauth import (
    OAuthError,
    build_authorize_url,
    exchange_code_for_token,
    expires_at_from_now,
    generate_pkce_pair,
)
from .proxy import (
    ProxyError,
    call_openai_raw,
    convert_chat_to_responses_payload,
    convert_responses_to_chat_completion,
    iter_chat_completion_chunks,
    iter_codex_sse,
    open_openai_raw,
    parse_codex_sse,
    resolve_session_token,
)
from .store import Store


def json_bytes(data: object) -> bytes:
    return json.dumps(data, ensure_ascii=True, indent=2).encode("utf-8")


def html_page(title: str, body: str) -> bytes:
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
      body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 900px; line-height: 1.5; }}
      code, pre {{ background: #f6f8fa; padding: 0.2rem 0.35rem; border-radius: 4px; }}
      pre {{ padding: 1rem; overflow-x: auto; }}
    </style>
  </head>
  <body>
    {body}
  </body>
</html>""".encode("utf-8")


def read_json_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def bearer_token_from_request(handler: BaseHTTPRequestHandler, settings: Settings) -> str | None:
    auth = handler.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth.removeprefix("Bearer ").strip()

    cookie_header = handler.headers.get("Cookie", "")
    if not cookie_header:
        return None

    cookie = SimpleCookie()
    cookie.load(cookie_header)
    morsel = cookie.get(settings.cookie_name)
    if morsel:
        return morsel.value
    return None


def make_cookie(settings: Settings, session_id: str) -> str:
    cookie = SimpleCookie()
    cookie[settings.cookie_name] = session_id
    cookie[settings.cookie_name]["httponly"] = True
    cookie[settings.cookie_name]["path"] = "/"
    cookie[settings.cookie_name]["samesite"] = "Lax"
    return cookie.output(header="").strip()


def _callback_host_port(callback_url: str) -> tuple[str, int]:
    parsed = urlparse(callback_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 1455
    return host, port


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "openai-oauth-proxy/0.1"

    @property
    def settings(self) -> Settings:
        return self.server.settings  # type: ignore[attr-defined]

    @property
    def store(self) -> Store:
        return self.server.store  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:
        return

    def send_json(self, status: int, payload: object, headers: dict[str, str] | None = None) -> None:
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, status: int, body: bytes, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def send_redirect(self, location: str, headers: dict[str, str] | None = None) -> None:
        self.send_response(HTTPStatus.FOUND)
        self.send_header("Location", location)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.handle_index()
            return
        if parsed.path == "/login":
            self.handle_login()
            return
        if parsed.path == "/healthz":
            self.send_json(HTTPStatus.OK, {"status": "ok"})
            return
        if parsed.path == "/v1/models":
            self.handle_models()
            return
        self.send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/v1/responses":
            self.handle_responses_proxy()
            return
        if parsed.path == "/v1/chat/completions":
            self.handle_chat_completions_proxy()
            return
        self.send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})

    def handle_index(self) -> None:
        callback_url = self.settings.callback_url
        body = html_page(
            "OpenAI OAuth Proxy",
            f"""
            <h1>OpenAI OAuth Proxy</h1>
            <p>Login: <a href="/login">/login</a></p>
            <p>Health: <a href="/healthz">/healthz</a></p>
            <p>Default model: <code>{self.settings.default_model}</code></p>
            <p>Callback URL: <code>{callback_url}</code></p>
            <pre>curl -H 'Authorization: Bearer &lt;session_id&gt;' \\
  -H 'Content-Type: application/json' \\
  -d '{{"model":"{self.settings.default_model}","messages":[{{"role":"user","content":"Hello"}}]}}' \\
  http://{self.settings.host}:{self.settings.port}/v1/chat/completions</pre>
            """,
        )
        self.send_html(HTTPStatus.OK, body)

    def handle_login(self) -> None:
        state = secrets.token_urlsafe(32)
        code_verifier, code_challenge = generate_pkce_pair()
        self.store.save_state(state, code_verifier)
        self.send_redirect(
            build_authorize_url(
                state=state,
                code_challenge=code_challenge,
                redirect_uri=self.settings.callback_url,
            )
        )

    def handle_models(self) -> None:
        self.send_json(
            HTTPStatus.OK,
            {
                "object": "list",
                "data": [
                    {
                        "id": self.settings.default_model,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "openai",
                    }
                ],
            },
        )

    def _get_session_or_401(self):
        session_id = bearer_token_from_request(self, self.settings)
        if not session_id:
            self.send_json(HTTPStatus.UNAUTHORIZED, {"error": {"message": "missing bearer token or session cookie"}})
            return None

        session = self.store.get_session(session_id)
        if session is None:
            self.send_json(HTTPStatus.UNAUTHORIZED, {"error": {"message": "unknown session"}})
            return None
        return session

    def handle_responses_proxy(self) -> None:
        session = self._get_session_or_401()
        if session is None:
            return

        body = read_json_body(self)
        # ``verbosity`` is a proxy-only flag.  The Responses API does not
        # accept it, and this endpoint already returns the complete upstream
        # response (or event stream).
        body.pop("verbosity", None)
        if "model" not in body or not body["model"]:
            body["model"] = self.settings.default_model

        try:
            token = resolve_session_token(self.settings, self.store, session)
        except ProxyError as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": {"message": str(exc)}})
            return

        wants_stream = body.get("stream") is True
        with open_openai_raw(
            self.settings,
            token=token,
            path="/responses",
            body=json.dumps(body).encode("utf-8"),
            stream=wants_stream,
        ) as upstream:
            status = upstream.status
            headers = dict(upstream.headers.items())
            if status >= 400 or not wants_stream:
                self._forward_upstream_response(status, headers, upstream.read())
                return
            self._relay_upstream_stream(status, headers, upstream)

    def handle_chat_completions_proxy(self) -> None:
        session = self._get_session_or_401()
        if session is None:
            return

        body = read_json_body(self)
        verbose = body.pop("verbosity", False) is True
        responses_payload = convert_chat_to_responses_payload(body, self.settings.default_model)
        try:
            token = resolve_session_token(self.settings, self.store, session)
        except ProxyError as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": {"message": str(exc)}})
            return

        if body.get("stream") is True:
            self._stream_chat_completion(token, body, responses_payload, verbose=verbose)
            return

        status, headers, upstream_body = call_openai_raw(
            self.settings,
            token=token,
            path="/responses",
            body=json.dumps(responses_payload).encode("utf-8"),
            stream=True,
        )
        if status >= 400:
            self._forward_upstream_response(status, headers, upstream_body)
            return

        text, final_response = parse_codex_sse(upstream_body)
        if verbose and final_response:
            self.send_json(HTTPStatus.OK, final_response)
            return

        if not text and final_response:
            self.send_json(
                HTTPStatus.OK,
                convert_responses_to_chat_completion(final_response, self.settings.default_model),
            )
            return

        self.send_json(
            HTTPStatus.OK,
            {
                "id": final_response.get("id", "chatcmpl_proxy"),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": final_response.get("model") or self.settings.default_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": final_response.get("usage", {}),
            },
        )

    def _stream_chat_completion(
        self,
        token: str,
        body: dict,
        responses_payload: dict,
        *,
        verbose: bool = False,
    ) -> None:
        with open_openai_raw(
            self.settings,
            token=token,
            path="/responses",
            body=json.dumps(responses_payload).encode("utf-8"),
            stream=True,
        ) as upstream:
            status = upstream.status
            headers = dict(upstream.headers.items())
            if status >= 400:
                self._forward_upstream_response(status, headers, upstream.read())
                return

            if verbose:
                self._relay_upstream_stream(status, headers, upstream)
                return

            self._begin_event_stream(status, headers)
            stream_options = body.get("stream_options")
            include_usage = isinstance(stream_options, dict) and stream_options.get("include_usage") is True
            try:
                events = iter_codex_sse(upstream)
                for chunk in iter_chat_completion_chunks(
                    events,
                    self.settings.default_model,
                    include_usage=include_usage,
                ):
                    self.wfile.write(b"data: " + json.dumps(chunk, separators=(",", ":")).encode("utf-8") + b"\n\n")
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionError):
                return

    def _begin_event_stream(self, status: int, headers: dict[str, str]) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        for key in ("x-request-id", "openai-processing-ms", "openai-organization", "openai-version"):
            if key in headers:
                self.send_header(key, headers[key])
        self.end_headers()
        self.close_connection = True

    def _relay_upstream_stream(self, status: int, headers: dict[str, str], upstream) -> None:
        self._begin_event_stream(status, headers)
        read_chunk = getattr(upstream, "read1", upstream.read)
        try:
            while chunk := read_chunk(8192):
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionError):
            return

    def _forward_upstream_response(self, status: int, headers: dict[str, str], body: bytes) -> None:
        self.send_response(status)
        content_type = headers.get("Content-Type", "application/json")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key in ("x-request-id", "openai-processing-ms", "openai-organization", "openai-version"):
            if key in headers:
                self.send_header(key, headers[key])
        self.end_headers()
        self.wfile.write(body)


class CallbackHandler(BaseHTTPRequestHandler):
    server_version = "openai-oauth-proxy-callback/0.1"

    @property
    def settings(self) -> Settings:
        return self.server.settings  # type: ignore[attr-defined]

    @property
    def store(self) -> Store:
        return self.server.store  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:
        return

    def send_json(self, status: int, payload: object) -> None:
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, status: int, body: bytes, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/auth/callback":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})
            return

        params = parse_qs(parsed.query)
        code = params.get("code", [None])[0]
        state = params.get("state", [None])[0]
        error = params.get("error", [None])[0]

        if error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": {"message": error}})
            return
        if not code or not state:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": {"message": "missing code or state"}})
            return

        saved = self.store.pop_state(state)
        if saved is None:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": {"message": "invalid or expired state"}})
            return

        callback_url = self.server.callback_redirect_uri  # type: ignore[attr-defined]
        try:
            token = exchange_code_for_token(
                self.settings,
                code=code,
                code_verifier=saved.code_verifier,
                redirect_uri=callback_url,
            )
        except OAuthError as exc:
            self.send_json(HTTPStatus.BAD_GATEWAY, {"error": {"message": str(exc)}})
            return

        session_id = secrets.token_urlsafe(32)
        self.store.create_session(
            session_id=session_id,
            access_token=token.access_token,
            refresh_token=token.refresh_token,
            token_type=token.token_type,
            expires_at=expires_at_from_now(token.expires_in),
        )
        body = html_page(
            "Login complete",
            f"""
            <h1>Login complete</h1>
            <p>Your API session token is:</p>
            <pre>{session_id}</pre>
            <p>Use it as <code>Authorization: Bearer {session_id}</code> on <code>/v1/*</code> requests.</p>
            <p>You can close this tab now.</p>
            """,
        )
        self.send_html(HTTPStatus.OK, body, headers={"Set-Cookie": make_cookie(self.settings, session_id)})


class ProxyHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, RequestHandlerClass, settings: Settings, store: Store):
        super().__init__(server_address, RequestHandlerClass)
        self.settings = settings
        self.store = store


class CallbackHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, RequestHandlerClass, settings: Settings, store: Store):
        super().__init__(server_address, RequestHandlerClass)
        self.settings = settings
        self.store = store
        self.callback_redirect_uri = settings.callback_url


def main(port: int | None = None, host: str | None = None) -> None:
    settings = load_settings()
    if port is not None:
        settings.port = port
    if host is not None:
        settings.host = host

    callback_host, callback_port = _callback_host_port(settings.callback_url)

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    store = Store(settings.database_path)
    proxy_server = ProxyHTTPServer((settings.host, settings.port), RequestHandler, settings, store)
    callback_server = CallbackHTTPServer((callback_host, callback_port), CallbackHandler, settings, store)

    print(f"Proxy listening on http://{settings.host}:{settings.port}")
    print(f"Callback listening on {settings.callback_url}")

    callback_thread = Thread(target=callback_server.serve_forever, daemon=True)
    callback_thread.start()
    try:
        proxy_server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        proxy_server.server_close()
        callback_server.server_close()

from __future__ import annotations

import json
import sys
import tempfile
from io import BytesIO
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openai_oauth_proxy.config import Settings
from openai_oauth_proxy.oauth import TokenResponse
from openai_oauth_proxy.server import CallbackHandler, CallbackHTTPServer, RequestHandler
from openai_oauth_proxy.store import Store


class CallbackServerTests(unittest.TestCase):
    def test_callback_exchanges_code_and_creates_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(data_dir=Path(tmp))
            store = Store(settings.database_path)
            store.save_state("valid-state", "verifier")
            with patch.object(ThreadingHTTPServer, "__init__", return_value=None):
                server = CallbackHTTPServer(("127.0.0.1", 0), CallbackHandler, settings, store)

            handler = CallbackHandler.__new__(CallbackHandler)
            handler.server = server
            handler.path = "/auth/callback?code=auth-code&state=valid-state"
            handler.send_html = Mock()
            handler.send_json = Mock()

            token = TokenResponse(
                access_token="access-token",
                refresh_token="refresh-token",
                token_type="bearer",
                expires_in=3600,
                raw={},
            )
            with patch("openai_oauth_proxy.server.exchange_code_for_token", return_value=token) as exchange:
                handler.do_GET()

            exchange.assert_called_once_with(
                settings,
                code="auth-code",
                code_verifier="verifier",
                redirect_uri=settings.callback_url,
            )
            handler.send_json.assert_not_called()
            status, body = handler.send_html.call_args.args
            headers = handler.send_html.call_args.kwargs["headers"]
            self.assertEqual(status, HTTPStatus.OK)
            self.assertIn(b"Login complete", body)
            self.assertIn(settings.cookie_name, headers["Set-Cookie"])

            sessions = store.list_sessions()
            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0].access_token, "access-token")


class StreamingServerTests(unittest.TestCase):
    def test_chat_completion_stream_writes_sse_chunks_and_done(self) -> None:
        class FakeUpstream(BytesIO):
            status = HTTPStatus.OK
            headers = {"Content-Type": "text/event-stream"}

        upstream = FakeUpstream(
            b'event: response.created\ndata: {"response":{"id":"resp_123","model":"gpt-test"}}\n\n'
            b'event: response.output_text.delta\ndata: {"delta":"Hello"}\n\n'
            b'event: response.completed\ndata: {"response":{"id":"resp_123","model":"gpt-test"}}\n\n'
        )
        handler = RequestHandler.__new__(RequestHandler)
        handler.server = SimpleNamespace(settings=Settings(default_model="gpt-test"), store=Mock())
        handler.wfile = BytesIO()
        handler._begin_event_stream = Mock()

        with patch("openai_oauth_proxy.server.open_openai_raw", return_value=upstream):
            handler._stream_chat_completion("token", {"stream": True}, {"stream": True})

        handler._begin_event_stream.assert_called_once()
        frames = handler.wfile.getvalue().decode("utf-8")
        self.assertIn('"object":"chat.completion.chunk"', frames)
        self.assertIn('"content":"Hello"', frames)
        self.assertTrue(frames.endswith("data: [DONE]\n\n"))

    def test_chat_completion_stream_writes_function_call_chunks(self) -> None:
        class FakeUpstream(BytesIO):
            status = HTTPStatus.OK
            headers = {"Content-Type": "text/event-stream"}

        upstream = FakeUpstream(
            b'event: response.created\ndata: {"response":{"id":"resp_123","model":"gpt-test"}}\n\n'
            b'event: response.output_item.added\ndata: {"output_index":0,"item":'
            b'{"type":"function_call","id":"fc_123","call_id":"call_123",'
            b'"name":"get_file_info","arguments":""}}\n\n'
            b'event: response.function_call_arguments.delta\ndata: {"item_id":"fc_123",'
            b'"output_index":0,"delta":"{\\"path\\":\\"/tmp/test.bin\\"}"}\n\n'
            b'event: response.completed\ndata: {"response":{"id":"resp_123","model":"gpt-test"}}\n\n'
        )
        handler = RequestHandler.__new__(RequestHandler)
        handler.server = SimpleNamespace(settings=Settings(default_model="gpt-test"), store=Mock())
        handler.wfile = BytesIO()
        handler._begin_event_stream = Mock()

        with patch("openai_oauth_proxy.server.open_openai_raw", return_value=upstream):
            handler._stream_chat_completion("token", {"stream": True}, {"stream": True})

        frames = [
            json.loads(line.removeprefix("data: "))
            for line in handler.wfile.getvalue().decode("utf-8").splitlines()
            if line.startswith("data: {")
        ]
        first_tool_call = frames[0]["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual(first_tool_call["id"], "call_123")
        self.assertEqual(first_tool_call["index"], 0)
        self.assertEqual(first_tool_call["function"]["name"], "get_file_info")
        self.assertEqual(
            frames[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
            '{"path":"/tmp/test.bin"}',
        )
        self.assertEqual(frames[2]["choices"][0]["finish_reason"], "tool_calls")

    def test_verbose_stream_relays_complete_upstream_events(self) -> None:
        class FakeUpstream(BytesIO):
            status = HTTPStatus.OK
            headers = {"Content-Type": "text/event-stream"}

        raw_events = (
            b'event: response.completed\ndata: {"response":{"id":"resp_123",'
            b'"status":"completed","usage":{"input_tokens":1}}}\n\n'
        )
        upstream = FakeUpstream(raw_events)
        handler = RequestHandler.__new__(RequestHandler)
        handler.server = SimpleNamespace(settings=Settings(default_model="gpt-test"), store=Mock())
        handler._relay_upstream_stream = Mock()

        with patch("openai_oauth_proxy.server.open_openai_raw", return_value=upstream):
            handler._stream_chat_completion(
                "token",
                {"stream": True},
                {"stream": True},
                verbose=True,
            )

        handler._relay_upstream_stream.assert_called_once_with(
            HTTPStatus.OK,
            {"Content-Type": "text/event-stream"},
            upstream,
        )


class ChatCompletionServerTests(unittest.TestCase):
    def test_verbosity_returns_raw_response_without_forwarding_flag(self) -> None:
        final_response = {
            "id": "resp_123",
            "object": "response",
            "status": "completed",
            "model": "gpt-test",
            "output": [],
            "usage": {"input_tokens": 1, "output_tokens": 2},
            "metadata": {"source": "upstream"},
        }
        upstream_body = (
            b"event: response.completed\n"
            + b"data: "
            + json.dumps({"response": final_response}).encode("utf-8")
            + b"\n\n"
        )

        handler = RequestHandler.__new__(RequestHandler)
        handler.server = SimpleNamespace(settings=Settings(default_model="gpt-test"), store=Mock())
        handler._get_session_or_401 = Mock(return_value=object())
        handler.send_json = Mock()

        request_body = {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "Hello"}],
            "verbosity": True,
        }
        with (
            patch("openai_oauth_proxy.server.read_json_body", return_value=request_body),
            patch("openai_oauth_proxy.server.resolve_session_token", return_value="token"),
            patch(
                "openai_oauth_proxy.server.call_openai_raw",
                return_value=(HTTPStatus.OK, {}, upstream_body),
            ) as call_upstream,
        ):
            handler.handle_chat_completions_proxy()

        forwarded = json.loads(call_upstream.call_args.kwargs["body"])
        self.assertNotIn("verbosity", forwarded)
        handler.send_json.assert_called_once_with(HTTPStatus.OK, final_response)

    def test_non_streaming_function_call_returns_chat_completion_tool_call(self) -> None:
        final_response = {
            "id": "resp_123",
            "status": "completed",
            "model": "gpt-test",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_123",
                    "call_id": "call_123",
                    "name": "get_file_info",
                    "arguments": '{"path":"/tmp/test.bin"}',
                }
            ],
        }
        upstream_body = (
            b"event: response.completed\n"
            + b"data: "
            + json.dumps({"response": final_response}).encode("utf-8")
            + b"\n\n"
        )
        handler = RequestHandler.__new__(RequestHandler)
        handler.server = SimpleNamespace(settings=Settings(default_model="gpt-test"), store=Mock())
        handler._get_session_or_401 = Mock(return_value=object())
        handler.send_json = Mock()

        with (
            patch(
                "openai_oauth_proxy.server.read_json_body",
                return_value={"messages": [{"role": "user", "content": "Inspect it"}]},
            ),
            patch("openai_oauth_proxy.server.resolve_session_token", return_value="token"),
            patch(
                "openai_oauth_proxy.server.call_openai_raw",
                return_value=(HTTPStatus.OK, {}, upstream_body),
            ),
        ):
            handler.handle_chat_completions_proxy()

        response = handler.send_json.call_args.args[1]
        choice = response["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["tool_calls"][0]["id"], "call_123")


class ResponsesServerTests(unittest.TestCase):
    def test_verbosity_is_not_forwarded_to_responses_api(self) -> None:
        class FakeUpstream(BytesIO):
            status = HTTPStatus.OK
            headers = {"Content-Type": "application/json"}

        upstream = FakeUpstream(b'{"id":"resp_123"}')
        handler = RequestHandler.__new__(RequestHandler)
        handler.server = SimpleNamespace(settings=Settings(default_model="gpt-test"), store=Mock())
        handler._get_session_or_401 = Mock(return_value=object())
        handler._forward_upstream_response = Mock()

        request_body = {
            "model": "gpt-test",
            "input": "Hello",
            "verbosity": True,
        }
        with (
            patch("openai_oauth_proxy.server.read_json_body", return_value=request_body),
            patch("openai_oauth_proxy.server.resolve_session_token", return_value="token"),
            patch("openai_oauth_proxy.server.open_openai_raw", return_value=upstream) as open_upstream,
        ):
            handler.handle_responses_proxy()

        forwarded = json.loads(open_upstream.call_args.kwargs["body"])
        self.assertNotIn("verbosity", forwarded)
        handler._forward_upstream_response.assert_called_once_with(
            HTTPStatus.OK,
            {"Content-Type": "application/json"},
            b'{"id":"resp_123"}',
        )


if __name__ == "__main__":
    unittest.main()

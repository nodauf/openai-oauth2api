from __future__ import annotations

import json
import tempfile
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openai_oauth_proxy.config import Settings
from openai_oauth_proxy.proxy import (
    convert_chat_to_responses_payload,
    convert_responses_to_chat_completion,
    extract_output_text,
    iter_chat_completion_chunks,
    iter_codex_sse,
)
from openai_oauth_proxy.store import Store


class ProxyTests(unittest.TestCase):
    def test_convert_chat_to_responses_payload(self) -> None:
        payload = {
            "model": "gpt-5.3-codex",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hello"},
            ],
            "max_completion_tokens": 42,
            "reasoning_effort": "high",
            "verbosity": True,
            "stream_options": {"include_usage": True},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_weather"},
            },
        }
        converted = convert_chat_to_responses_payload(payload, "gpt-5.3-codex")
        self.assertEqual(converted["model"], "gpt-5.3-codex")
        self.assertEqual(converted["instructions"], "Be concise.")
        self.assertEqual(converted["input"][0]["role"], "user")
        self.assertEqual(converted["input"][0]["content"], "Hello")
        self.assertNotIn("max_completion_tokens", converted)
        self.assertNotIn("max_output_tokens", converted)
        self.assertNotIn("stream_options", converted)
        self.assertNotIn("reasoning_effort", converted)
        self.assertNotIn("verbosity", converted)
        self.assertEqual(converted["reasoning"], {"effort": "high"})
        self.assertEqual(
            converted["tools"],
            [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get the weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            ],
        )
        self.assertEqual(
            converted["tool_choice"],
            {"type": "function", "name": "get_weather"},
        )

    def test_convert_chat_to_responses_payload_preserves_simple_tool_choices(self) -> None:
        for tool_choice in ("auto", "none", "required"):
            with self.subTest(tool_choice=tool_choice):
                converted = convert_chat_to_responses_payload(
                    {"messages": [], "tool_choice": tool_choice},
                    "gpt-5.3-codex",
                )
                self.assertEqual(converted["tool_choice"], tool_choice)

    def test_convert_chat_to_responses_payload_preserves_reasoning_fields(self) -> None:
        payload = {
            "messages": [],
            "reasoning": {"summary": "auto"},
            "reasoning_effort": "medium",
        }

        converted = convert_chat_to_responses_payload(payload, "gpt-5.3-codex")

        self.assertEqual(converted["reasoning"], {"summary": "auto", "effort": "medium"})
        self.assertEqual(payload["reasoning"], {"summary": "auto"})

    def test_convert_chat_tool_calls_and_outputs_to_responses_items(self) -> None:
        payload = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_123", "content": "Sunny"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_456",
                            "type": "function",
                            "function": {"name": "get_time", "arguments": '{"city":"Paris"}'},
                        },
                        {
                            "id": "call_789",
                            "type": "function",
                            "function": {"name": "get_time", "arguments": '{"city":"Tokyo"}'},
                        },
                    ],
                },
            ]
        }

        converted = convert_chat_to_responses_payload(payload, "gpt-test")

        self.assertEqual(
            converted["input"],
            [
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "get_weather",
                    "arguments": '{"city":"Paris"}',
                },
                {"type": "function_call_output", "call_id": "call_123", "output": "Sunny"},
                {
                    "type": "function_call",
                    "call_id": "call_456",
                    "name": "get_time",
                    "arguments": '{"city":"Paris"}',
                },
                {
                    "type": "function_call",
                    "call_id": "call_789",
                    "name": "get_time",
                    "arguments": '{"city":"Tokyo"}',
                },
            ],
        )

    def test_extract_output_text(self) -> None:
        response = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "Hello"},
                        {"type": "output_text", "text": " world"},
                    ],
                }
            ]
        }
        self.assertEqual(extract_output_text(response), "Hello world")

    def test_convert_responses_to_chat_completion(self) -> None:
        response = {
            "id": "resp_123",
            "status": "completed",
            "model": "gpt-5.3-codex",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Answer"}],
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 2},
        }
        converted = convert_responses_to_chat_completion(response, "gpt-5.3-codex")
        self.assertEqual(converted["object"], "chat.completion")
        self.assertEqual(converted["choices"][0]["message"]["content"], "Answer")
        self.assertEqual(converted["usage"]["output_tokens"], 2)

    def test_convert_responses_function_calls_to_chat_completion(self) -> None:
        response = {
            "id": "resp_123",
            "status": "completed",
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

        converted = convert_responses_to_chat_completion(response, "gpt-test")

        choice = converted["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIsNone(choice["message"]["content"])
        self.assertEqual(
            choice["message"]["tool_calls"],
            [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_file_info",
                        "arguments": '{"path":"/tmp/test.bin"}',
                    },
                }
            ],
        )

    def test_iter_codex_sse_handles_incremental_events(self) -> None:
        from io import BytesIO

        raw = BytesIO(
            b'event: response.output_text.delta\ndata: {"delta":"Hello"}\n\n'
            b'event: response.completed\ndata: {"response":{"id":"resp_123"}}\n\n'
        )
        events = list(iter_codex_sse(raw))
        self.assertEqual(events[0]["type"], "response.output_text.delta")
        self.assertEqual(events[0]["delta"], "Hello")
        self.assertEqual(events[1]["response"]["id"], "resp_123")

    def test_iter_chat_completion_chunks(self) -> None:
        events = iter(
            [
                {
                    "type": "response.created",
                    "response": {"id": "resp_123", "model": "gpt-test", "created_at": 123},
                },
                {"type": "response.output_text.delta", "delta": "Hello"},
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_123",
                        "model": "gpt-test",
                        "usage": {"input_tokens": 1, "output_tokens": 2},
                    },
                },
            ]
        )
        chunks = list(iter_chat_completion_chunks(events, "default", include_usage=True))
        self.assertEqual(chunks[0]["object"], "chat.completion.chunk")
        self.assertEqual(chunks[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual(chunks[1]["choices"][0]["delta"]["content"], "Hello")
        self.assertEqual(chunks[2]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(chunks[3]["choices"], [])
        self.assertEqual(chunks[3]["usage"]["completion_tokens"], 2)
        self.assertEqual(chunks[3]["usage"]["prompt_tokens"], 1)

    def test_iter_chat_completion_chunks_streams_multiple_function_calls(self) -> None:
        events = iter(
            [
                {
                    "type": "response.created",
                    "response": {"id": "resp_123", "model": "gpt-test", "created_at": 123},
                },
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_123",
                        "call_id": "call_123",
                        "name": "get_file_info",
                        "arguments": "",
                    },
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_123",
                    "output_index": 0,
                    "delta": '{"path":"/tmp/',
                },
                {
                    "type": "response.output_item.added",
                    "output_index": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_456",
                        "call_id": "call_456",
                        "name": "get_owner",
                        "arguments": "",
                    },
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_456",
                    "output_index": 1,
                    "delta": '{"path":"/tmp/test.bin"}',
                },
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": "fc_123",
                    "output_index": 0,
                    "arguments": '{"path":"/tmp/test.bin"}',
                },
                {"type": "response.completed", "response": {"id": "resp_123"}},
            ]
        )

        chunks = list(iter_chat_completion_chunks(events, "default"))

        self.assertEqual(
            chunks[0]["choices"][0]["delta"],
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_file_info", "arguments": ""},
                    }
                ]
            },
        )
        self.assertEqual(
            chunks[1]["choices"][0]["delta"],
            {"tool_calls": [{"index": 0, "function": {"arguments": '{"path":"/tmp/'}}]},
        )
        self.assertEqual(chunks[2]["choices"][0]["delta"]["tool_calls"][0]["index"], 1)
        self.assertEqual(chunks[2]["choices"][0]["delta"]["tool_calls"][0]["id"], "call_456")
        self.assertEqual(chunks[3]["choices"][0]["delta"]["tool_calls"][0]["index"], 1)
        self.assertEqual(
            chunks[4]["choices"][0]["delta"],
            {"tool_calls": [{"index": 0, "function": {"arguments": 'test.bin"}'}}]},
        )
        self.assertEqual(chunks[5]["choices"][0]["finish_reason"], "tool_calls")

    def test_store_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "db.sqlite3")
            store.save_state("state", "verifier")
            saved = store.pop_state("state")
            self.assertIsNotNone(saved)
            assert saved is not None
            self.assertEqual(saved.code_verifier, "verifier")
            session = store.create_session(
                session_id="session",
                access_token="token",
                refresh_token="refresh",
                token_type="bearer",
                expires_at=123,
            )
            self.assertEqual(store.get_session("session").access_token, "token")
            store.update_session_tokens(
                "session",
                access_token="new-token",
                refresh_token=None,
                token_type="bearer",
                expires_at=None,
            )
            self.assertEqual(store.get_session("session").access_token, "new-token")


if __name__ == "__main__":
    unittest.main()

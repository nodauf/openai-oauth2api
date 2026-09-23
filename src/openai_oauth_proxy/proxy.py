from __future__ import annotations

import json
import time
from io import BytesIO
from typing import BinaryIO, Iterator
from urllib.error import HTTPError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from .config import Settings
from .oauth import refresh_access_token
from .store import SessionRecord, Store


class ProxyError(RuntimeError):
    pass


def build_upstream_headers(settings: Settings, token: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "responses=v1",
    }
    if settings.openai_organization:
        headers["OpenAI-Organization"] = settings.openai_organization
    if settings.openai_project:
        headers["OpenAI-Project"] = settings.openai_project
    return headers


def resolve_upstream_token(settings: Settings, store: Store, session: SessionRecord) -> str:
    if settings.upstream_auth_mode == "api_key":
        if not settings.openai_api_key:
            raise ProxyError("OPENAI_API_KEY is required when OPENAI_UPSTREAM_AUTH_MODE=api_key")
        return settings.openai_api_key

    if session.expires_at is None:
        return session.access_token
    if session.expires_at - int(time.time()) > 60:
        return session.access_token
    if not session.refresh_token:
        return session.access_token

    refreshed = refresh_access_token(settings, refresh_token=session.refresh_token)
    store.update_session_tokens(
        session.session_id,
        access_token=refreshed.access_token,
        refresh_token=refreshed.refresh_token,
        token_type=refreshed.token_type,
        expires_at=None if refreshed.expires_in is None else int(time.time()) + refreshed.expires_in,
    )
    return refreshed.access_token


def resolve_session_token(settings: Settings, store: Store, session: SessionRecord) -> str:
    return resolve_upstream_token(settings, store, session)


def call_openai_raw(
    settings: Settings,
    *,
    token: str,
    path: str,
    body: bytes,
    stream: bool = False,
) -> tuple[int, dict[str, str], bytes]:
    with open_openai_raw(settings, token=token, path=path, body=body, stream=stream) as response:
        return response.status, dict(response.headers.items()), response.read()


def open_openai_raw(
    settings: Settings,
    *,
    token: str,
    path: str,
    body: bytes,
    stream: bool = False,
):
    headers = build_upstream_headers(settings, token)
    if stream:
        headers["Accept"] = "text/event-stream"
    request = Request(
        urljoin(settings.openai_api_base.rstrip("/") + "/", path.lstrip("/")),
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        return urlopen(request, timeout=settings.request_timeout_seconds)
    except HTTPError as exc:
        return exc


def convert_chat_to_responses_payload(payload: dict, default_model: str) -> dict:
    messages = payload.get("messages", [])
    input_messages = []
    instructions = []

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role in {"system", "developer"} and isinstance(content, str):
            instructions.append(content)
            continue

        if role == "tool":
            input_messages.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id", ""),
                    "output": content,
                }
            )
            continue

        if isinstance(content, list):
            text_chunks = []
            for item in content:
                if isinstance(item, dict) and item.get("type") in {"text", "input_text", "output_text"}:
                    text_chunks.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_chunks.append(item)
            content = "\n".join(chunk for chunk in text_chunks if chunk)

        tool_calls = message.get("tool_calls") if role == "assistant" else None
        if isinstance(content, str):
            input_messages.append({"role": role, "content": content})
        elif not tool_calls:
            input_messages.append(message)

        if role == "assistant" and isinstance(tool_calls, list):
            for tool_call in tool_calls:
                function = tool_call.get("function") if isinstance(tool_call, dict) else None
                if not isinstance(function, dict):
                    continue
                input_messages.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call.get("id", ""),
                        "name": function.get("name", ""),
                        "arguments": function.get("arguments", ""),
                    }
                )

    responses_payload = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "messages",
            "model",
            "stream",
            "stream_options",
            "max_tokens",
            "max_completion_tokens",
            "max_output_tokens",
            "reasoning_effort",
            "verbosity",
        }
    }

    if "reasoning_effort" in payload:
        reasoning = responses_payload.get("reasoning")
        responses_payload["reasoning"] = dict(reasoning) if isinstance(reasoning, dict) else {}
        responses_payload["reasoning"]["effort"] = payload["reasoning_effort"]

    tools = responses_payload.get("tools")
    if isinstance(tools, list):
        converted_tools = []
        for tool in tools:
            if (
                not isinstance(tool, dict)
                or tool.get("type") != "function"
                or not isinstance(tool.get("function"), dict)
            ):
                converted_tools.append(tool)
                continue

            converted_tool = {key: value for key, value in tool.items() if key != "function"}
            converted_tool.update(tool["function"])
            converted_tool["type"] = "function"
            converted_tools.append(converted_tool)
        responses_payload["tools"] = converted_tools

    tool_choice = responses_payload.get("tool_choice")
    if (
        isinstance(tool_choice, dict)
        and tool_choice.get("type") == "function"
        and isinstance(tool_choice.get("function"), dict)
    ):
        converted_tool_choice = {
            key: value for key, value in tool_choice.items() if key != "function"
        }
        converted_tool_choice.update(tool_choice["function"])
        converted_tool_choice["type"] = "function"
        responses_payload["tool_choice"] = converted_tool_choice

    responses_payload["model"] = payload.get("model") or default_model
    responses_payload["input"] = input_messages
    responses_payload["instructions"] = "\n\n".join(instructions) if instructions else "You are a helpful assistant."
    responses_payload["store"] = False
    responses_payload["stream"] = True
    return responses_payload


def extract_output_text(data: object) -> str:
    parts: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            node_type = node.get("type")
            if node_type in {"output_text", "text"} and isinstance(node.get("text"), str):
                parts.append(node["text"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data)
    return "".join(parts)


def convert_responses_to_chat_completion(payload: dict, default_model: str) -> dict:
    text = extract_output_text(payload.get("output", []))
    tool_calls = []
    output = payload.get("output", [])
    if isinstance(output, list):
        for index, item in enumerate(output):
            if not isinstance(item, dict) or item.get("type") != "function_call":
                continue
            call_id = item.get("call_id") or item.get("id") or f"call_proxy_{index}"
            arguments = item.get("arguments", "")
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": arguments if isinstance(arguments, str) else "",
                    },
                }
            )

    model = payload.get("model") or default_model
    created = int(time.time())
    status = payload.get("status")
    finish_reason = (
        "tool_calls"
        if tool_calls
        else ("stop" if status in {None, "completed"} else "length")
    )
    message: dict[str, object] = {
        "role": "assistant",
        "content": None if tool_calls and not text else text,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": payload.get("id", "chatcmpl_proxy"),
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": payload.get("usage", {}),
    }


def parse_codex_sse(raw_body: bytes) -> tuple[str, dict]:
    text_parts: list[str] = []
    final_response: dict = {}
    for event in iter_codex_sse(BytesIO(raw_body)):
        event_type = event.get("type")
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                text_parts.append(delta)
        elif event_type == "response.completed":
            response = event.get("response")
            if isinstance(response, dict):
                final_response = response

    if not text_parts and final_response:
        text_parts.append(extract_output_text(final_response.get("output", [])))

    return "".join(text_parts), final_response


def iter_codex_sse(stream: BinaryIO) -> Iterator[dict]:
    current_event: str | None = None
    data_lines: list[str] = []

    def flush() -> dict | None:
        if not data_lines:
            return None
        data_text = "\n".join(data_lines).strip()
        if not data_text:
            return None
        try:
            event_payload = json.loads(data_text)
        except json.JSONDecodeError:
            event_payload = {"data": data_text}
        if not isinstance(event_payload, dict):
            event_payload = {"data": event_payload}
        if current_event:
            event_payload["type"] = current_event
        return event_payload

    for raw_line in stream:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            event_payload = flush()
            if event_payload is not None:
                yield event_payload
            current_event = None
            data_lines = []
            continue
        if line.startswith("event:"):
            current_event = line.split(":", 1)[1].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].lstrip())
            continue

    event_payload = flush()
    if event_payload is not None:
        yield event_payload


def iter_chat_completion_chunks(
    events: Iterator[dict],
    default_model: str,
    *,
    include_usage: bool = False,
) -> Iterator[dict]:
    completion_id = "chatcmpl_proxy"
    model = default_model
    created = int(time.time())
    started = False
    finished = False
    final_usage: dict = {}
    function_calls: dict[int, dict[str, str]] = {}
    function_call_indices_by_item_id: dict[str, int] = {}
    saw_function_call = False

    def chat_usage(usage: dict) -> dict:
        converted = {
            "prompt_tokens": usage.get("input_tokens", usage.get("prompt_tokens", 0)),
            "completion_tokens": usage.get("output_tokens", usage.get("completion_tokens", 0)),
            "total_tokens": usage.get("total_tokens", 0),
        }
        input_details = usage.get("input_tokens_details")
        if isinstance(input_details, dict):
            converted["prompt_tokens_details"] = input_details
        output_details = usage.get("output_tokens_details")
        if isinstance(output_details, dict):
            converted["completion_tokens_details"] = output_details
        return converted

    def chunk(delta: dict, finish_reason: str | None = None) -> dict:
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }

    def output_index(event: dict) -> int | None:
        index = event.get("output_index")
        if isinstance(index, int) and not isinstance(index, bool):
            return index
        item_id = event.get("item_id")
        if isinstance(item_id, str):
            return function_call_indices_by_item_id.get(item_id)
        return None

    def function_call_added(event: dict, item: dict) -> dict | None:
        nonlocal saw_function_call, started

        index = output_index(event)
        if index is None:
            index = len(function_calls)
        if index in function_calls:
            return None

        call_id = item.get("call_id") or item.get("id") or f"call_proxy_{index}"
        name = item.get("name", "")
        arguments = item.get("arguments", "")
        state = {
            "id": call_id if isinstance(call_id, str) else f"call_proxy_{index}",
            "name": name if isinstance(name, str) else "",
            "arguments": arguments if isinstance(arguments, str) else "",
        }
        function_calls[index] = state
        item_id = item.get("id")
        if isinstance(item_id, str):
            function_call_indices_by_item_id[item_id] = index
        saw_function_call = True
        started = True
        return chunk(
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": state["id"],
                        "type": "function",
                        "function": {
                            "name": state["name"],
                            "arguments": state["arguments"],
                        },
                    }
                ]
            }
        )

    def missing_arguments_delta(event: dict, arguments: object) -> dict | None:
        index = output_index(event)
        if index is None or index not in function_calls or not isinstance(arguments, str):
            return None
        state = function_calls[index]
        emitted_arguments = state["arguments"]
        if not arguments.startswith(emitted_arguments):
            return None
        delta = arguments[len(emitted_arguments) :]
        if not delta:
            return None
        state["arguments"] += delta
        return chunk(
            {
                "tool_calls": [
                    {
                        "index": index,
                        "function": {"arguments": delta},
                    }
                ]
            }
        )

    for event in events:
        event_type = event.get("type")
        response_id = event.get("response_id")
        if isinstance(response_id, str):
            completion_id = response_id
        response = event.get("response")
        if isinstance(response, dict):
            completion_id = response.get("id") or completion_id
            model = response.get("model") or model
            response_created = response.get("created_at")
            if isinstance(response_created, (int, float)):
                created = int(response_created)

        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if not isinstance(delta, str):
                continue
            if not started:
                yield chunk({"role": "assistant", "content": ""})
                started = True
            yield chunk({"content": delta})
        elif event_type == "response.output_item.added":
            item = event.get("item")
            if not isinstance(item, dict) or item.get("type") != "function_call":
                continue
            tool_call_chunk = function_call_added(event, item)
            if tool_call_chunk is not None:
                yield tool_call_chunk
        elif event_type == "response.function_call_arguments.delta":
            index = output_index(event)
            delta = event.get("delta")
            if index is None or index not in function_calls or not isinstance(delta, str):
                continue
            function_calls[index]["arguments"] += delta
            yield chunk(
                {
                    "tool_calls": [
                        {
                            "index": index,
                            "function": {"arguments": delta},
                        }
                    ]
                }
            )
        elif event_type == "response.function_call_arguments.done":
            arguments_chunk = missing_arguments_delta(event, event.get("arguments"))
            if arguments_chunk is not None:
                yield arguments_chunk
        elif event_type == "response.output_item.done":
            item = event.get("item")
            if not isinstance(item, dict) or item.get("type") != "function_call":
                continue
            tool_call_chunk = function_call_added(event, item)
            if tool_call_chunk is not None:
                yield tool_call_chunk
            arguments_chunk = missing_arguments_delta(event, item.get("arguments"))
            if arguments_chunk is not None:
                yield arguments_chunk
        elif event_type == "response.completed":
            if isinstance(response, dict):
                usage = response.get("usage")
                if isinstance(usage, dict):
                    final_usage = usage
            if not started:
                yield chunk({"role": "assistant", "content": ""})
                started = True
            yield chunk({}, "tool_calls" if saw_function_call else "stop")
            finished = True
        elif event_type == "response.incomplete":
            if isinstance(response, dict):
                usage = response.get("usage")
                if isinstance(usage, dict):
                    final_usage = usage
            if not started:
                yield chunk({"role": "assistant", "content": ""})
                started = True
            yield chunk({}, "length")
            finished = True
        elif event_type in {"response.failed", "error"}:
            error = event.get("error")
            if not isinstance(error, dict) and isinstance(response, dict):
                error = response.get("error")
            if not isinstance(error, dict):
                error = {"message": "upstream streaming request failed"}
            yield {"error": error}
            finished = True

    if not finished:
        if not started:
            yield chunk({"role": "assistant", "content": ""})
        yield chunk({}, "tool_calls" if saw_function_call else "stop")

    if include_usage and final_usage:
        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [],
            "usage": chat_usage(final_usage),
        }

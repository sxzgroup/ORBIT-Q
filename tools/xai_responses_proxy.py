#!/usr/bin/env python3
"""Narrow compatibility proxy for Codex against the xAI Responses API.

The proxy never logs authorization headers or prompt/tool content.  Besides the
encrypted-reasoning replay repair, it fixes one representation mismatch seen in
xAI tool calls: integer-valued arguments can be serialized as JSON floats even
when Codex's tool router requires an integer.  Normalization is schema-scoped
and value-preserving (for example, ``6594.0`` becomes ``6594``); non-integral
values and fields that are not integer tool parameters are left untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


# Codex 0.146 advertises some integer router fields as generic JSON numbers to
# Responses-compatible providers.  Keep the repair narrow and explicit.  The
# request schema is also inspected, so future correctly-declared integer fields
# are normalized without being added here.
_KNOWN_INTEGER_FIELDS: dict[str, set[str]] = {
    "exec_command": {"max_output_tokens", "yield_time_ms"},
    "write_stdin": {"max_output_tokens", "session_id", "yield_time_ms"},
    "wait": {"max_tokens", "yield_time_ms"},
}


def _tool_parts(tool: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    """Return a tool name and parameter schema for either Responses shape."""
    function = tool.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        parameters = function.get("parameters")
    else:
        name = tool.get("name")
        parameters = tool.get("parameters")
    return (
        name if isinstance(name, str) else None,
        parameters if isinstance(parameters, dict) else None,
    )


def _set_schema_type_integer(schema: dict[str, Any]) -> bool:
    """Change a numeric property schema to integer without broadening it."""
    changed = False
    if schema.get("type") == "number":
        schema["type"] = "integer"
        changed = True
    for keyword in ("anyOf", "oneOf"):
        variants = schema.get(keyword)
        if isinstance(variants, list):
            for variant in variants:
                if isinstance(variant, dict) and variant.get("type") == "number":
                    variant["type"] = "integer"
                    changed = True
    return changed


def _prepare_integer_tool_schemas(
    tools: list[Any],
) -> tuple[dict[str, set[str]], list[str]]:
    """Discover integer fields and repair known Codex integer schemas."""
    integer_fields: dict[str, set[str]] = defaultdict(set)
    schema_repairs: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name, parameters = _tool_parts(tool)
        if name is None or parameters is None:
            continue
        properties = parameters.get("properties")
        if not isinstance(properties, dict):
            continue
        known = _KNOWN_INTEGER_FIELDS.get(name, set())
        for field, field_schema in properties.items():
            if not isinstance(field, str) or not isinstance(field_schema, dict):
                continue
            if field in known and _set_schema_type_integer(field_schema):
                schema_repairs.append(f"{name}.{field}")
            declared_types = {field_schema.get("type")}
            for keyword in ("anyOf", "oneOf"):
                variants = field_schema.get(keyword)
                if isinstance(variants, list):
                    declared_types.update(
                        variant.get("type")
                        for variant in variants
                        if isinstance(variant, dict)
                    )
            if "integer" in declared_types or field in known:
                integer_fields[name].add(field)
    return dict(integer_fields), sorted(schema_repairs)


def _normalize_arguments(arguments: str, integer_fields: set[str]) -> tuple[str, int]:
    """Normalize integral floats in selected top-level function arguments."""
    try:
        values = json.loads(arguments)
    except (TypeError, json.JSONDecodeError):
        return arguments, 0
    if not isinstance(values, dict):
        return arguments, 0
    changed = 0
    for field in integer_fields:
        value = values.get(field)
        if isinstance(value, float) and value.is_integer():
            values[field] = int(value)
            changed += 1
    if changed == 0:
        return arguments, 0
    return json.dumps(values, separators=(",", ":")), changed


def _normalize_function_call_stream(
    response_body: bytes, integer_fields: dict[str, set[str]]
) -> tuple[bytes, int]:
    """Normalize complete function arguments and their streamed delta sequence."""
    lines = response_body.splitlines(keepends=True)
    events: dict[int, dict[str, Any]] = {}
    item_names: dict[str, str] = {}
    delta_lines: dict[str, list[int]] = defaultdict(list)
    delta_text: dict[str, list[str]] = defaultdict(list)

    for index, line in enumerate(lines):
        raw = line.rstrip(b"\r\n")
        if not raw.startswith(b"data: ") or raw == b"data: [DONE]":
            continue
        try:
            event = json.loads(raw[6:])
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(event, dict):
            continue
        events[index] = event
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") == "function_call":
            item_id = item.get("id")
            name = item.get("name")
            if isinstance(item_id, str) and isinstance(name, str):
                item_names[item_id] = name
        if event.get("type") == "response.function_call_arguments.delta":
            item_id = event.get("item_id")
            delta = event.get("delta")
            if isinstance(item_id, str) and isinstance(delta, str):
                delta_lines[item_id].append(index)
                delta_text[item_id].append(delta)

    normalized_by_item: dict[str, str] = {}
    total_changes = 0
    item_ids = set(delta_text)
    for event in events.values():
        item_id = event.get("item_id")
        if isinstance(item_id, str):
            item_ids.add(item_id)
        item = event.get("item")
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            item_ids.add(item["id"])

    for item_id in item_ids:
        name = item_names.get(item_id)
        fields = integer_fields.get(name or "", set())
        if not fields:
            continue
        arguments = "".join(delta_text.get(item_id, []))
        if not arguments:
            for event in events.values():
                if event.get("item_id") == item_id and isinstance(event.get("arguments"), str):
                    arguments = event["arguments"]
                    break
                item = event.get("item")
                if (
                    isinstance(item, dict)
                    and item.get("id") == item_id
                    and isinstance(item.get("arguments"), str)
                ):
                    arguments = item["arguments"]
                    break
        normalized, changes = _normalize_arguments(arguments, fields)
        if changes:
            normalized_by_item[item_id] = normalized
            total_changes += changes

    if not normalized_by_item:
        return response_body, 0

    first_delta: set[str] = set()
    for index, event in events.items():
        item_id = event.get("item_id")
        if isinstance(item_id, str) and item_id in normalized_by_item:
            if event.get("type") == "response.function_call_arguments.delta":
                event["delta"] = (
                    normalized_by_item[item_id] if item_id not in first_delta else ""
                )
                first_delta.add(item_id)
            if isinstance(event.get("arguments"), str):
                event["arguments"] = normalized_by_item[item_id]
        item = event.get("item")
        if isinstance(item, dict):
            nested_id = item.get("id")
            if (
                isinstance(nested_id, str)
                and nested_id in normalized_by_item
                and isinstance(item.get("arguments"), str)
            ):
                item["arguments"] = normalized_by_item[nested_id]
        ending = b"\r\n" if lines[index].endswith(b"\r\n") else b"\n"
        lines[index] = b"data: " + json.dumps(event, separators=(",", ":")).encode() + ending
    return b"".join(lines), total_changes


def _item_shape(item: object) -> object:
    if not isinstance(item, dict):
        return type(item).__name__
    encrypted = item.get("encrypted_content")
    shape: dict[str, object] = {
        "type": item.get("type"),
        "role": item.get("role"),
        "id": item.get("id"),
        "call_id": item.get("call_id"),
        "name": item.get("name"),
        "status": item.get("status"),
        "keys": sorted(item),
    }
    if isinstance(encrypted, str):
        shape["encrypted_len"] = len(encrypted)
        shape["encrypted_sha256"] = hashlib.sha256(encrypted.encode()).hexdigest()
    return {key: value for key, value in shape.items() if value is not None}


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream = "https://api.x.ai"
    request_index = 0
    drop_reasoning_input = False
    restore_reasoning_status = False
    normalize_integral_tool_arguments = False

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        ProxyHandler.request_index += 1
        integer_fields: dict[str, set[str]] = {}
        try:
            payload = json.loads(body)
            payload_changed = False
            inputs = payload.get("input", [])
            if self.drop_reasoning_input and any(
                isinstance(item, dict) and item.get("type") == "function_call_output"
                for item in inputs
            ):
                payload["input"] = [
                    item
                    for item in inputs
                    if not (isinstance(item, dict) and item.get("type") == "reasoning")
                ]
                payload_changed = True
            elif self.restore_reasoning_status:
                changed = False
                for item in inputs:
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "reasoning"
                        and isinstance(item.get("encrypted_content"), str)
                        and "status" not in item
                    ):
                        item["status"] = "completed"
                        changed = True
                    if isinstance(item, dict) and item.get("type") == "reasoning" and "content" in item:
                        item.pop("content")
                        changed = True
                if changed:
                    payload_changed = True
            tools = payload.get("tools", [])
            schema_repairs: list[str] = []
            if self.normalize_integral_tool_arguments and isinstance(tools, list):
                integer_fields, schema_repairs = _prepare_integer_tool_schemas(tools)
                payload_changed = payload_changed or bool(schema_repairs)
            if payload_changed:
                body = json.dumps(payload, separators=(",", ":")).encode()
            inputs = payload.get("input", [])
            record = {
                "request_index": ProxyHandler.request_index,
                "path": self.path,
                "top_level_keys": sorted(payload),
                "tool_types": [tool.get("type") for tool in tools if isinstance(tool, dict)],
                "input": [_item_shape(item) for item in inputs],
                "include": payload.get("include"),
                "store": payload.get("store"),
                "reasoning": payload.get("reasoning"),
                "integer_schema_repairs": schema_repairs,
            }
            print(json.dumps(record, separators=(",", ":")), flush=True)
        except (UnicodeDecodeError, json.JSONDecodeError):
            print(json.dumps({"request_index": ProxyHandler.request_index, "invalid_json": True}), flush=True)

        headers = {
            "Authorization": self.headers.get("Authorization", ""),
            "Content-Type": self.headers.get("Content-Type", "application/json"),
            "Accept": self.headers.get("Accept", "text/event-stream"),
        }
        request = urllib.request.Request(
            f"{self.upstream}{self.path}", data=body, headers=headers, method="POST"
        )
        try:
            response = urllib.request.urlopen(request, timeout=3600)
        except urllib.error.HTTPError as error:
            response = error

        self.send_response(response.status)
        for name, value in response.headers.items():
            if name.lower() not in {"content-length", "connection", "transfer-encoding"}:
                self.send_header(name, value)
        self.send_header("Connection", "close")
        self.end_headers()
        response_body = response.read()
        normalized_arguments = 0
        if self.normalize_integral_tool_arguments and integer_fields:
            response_body, normalized_arguments = _normalize_function_call_stream(
                response_body, integer_fields
            )
        if normalized_arguments:
            print(
                json.dumps(
                    {
                        "response_to_request": ProxyHandler.request_index,
                        "normalized_integral_tool_arguments": normalized_arguments,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
        for line in response_body.splitlines():
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            try:
                event = json.loads(line[6:])
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            item = event.get("item")
            if isinstance(item, dict) and isinstance(item.get("encrypted_content"), str):
                encrypted = item["encrypted_content"]
                print(
                    json.dumps(
                        {
                            "response_to_request": ProxyHandler.request_index,
                            "event_type": event.get("type"),
                            "item": _item_shape(item),
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
        self.wfile.write(response_body)
        self.wfile.flush()
        self.close_connection = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--drop-reasoning-input", action="store_true")
    parser.add_argument("--restore-reasoning-status", action="store_true")
    parser.add_argument("--normalize-integral-tool-arguments", action="store_true")
    args = parser.parse_args()
    ProxyHandler.drop_reasoning_input = args.drop_reasoning_input
    ProxyHandler.restore_reasoning_status = args.restore_reasoning_status
    ProxyHandler.normalize_integral_tool_arguments = args.normalize_integral_tool_arguments
    server = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    print(json.dumps({"listening": f"{args.host}:{args.port}"}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
        sys.exit(0)


if __name__ == "__main__":
    main()

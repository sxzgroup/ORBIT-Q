#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import xai_responses_proxy as proxy


def _sse(*events: dict) -> bytes:
    return b"".join(
        b"data: " + json.dumps(event, separators=(",", ":")).encode() + b"\n\n"
        for event in events
    ) + b"data: [DONE]\n\n"


def _events(body: bytes) -> list[dict]:
    answer = []
    for line in body.splitlines():
        if line.startswith(b"data: {"):
            answer.append(json.loads(line[6:]))
    return answer


class IntegerToolCompatibilityTests(unittest.TestCase):
    def test_known_integer_schema_is_narrowly_repaired(self) -> None:
        tools = [
            {
                "type": "function",
                "name": "write_stdin",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "number"},
                        "yield_time_ms": {"type": "number"},
                        "ratio": {"type": "number"},
                    },
                },
            }
        ]
        fields, repairs = proxy._prepare_integer_tool_schemas(tools)
        properties = tools[0]["parameters"]["properties"]
        self.assertEqual(properties["session_id"]["type"], "integer")
        self.assertEqual(properties["yield_time_ms"]["type"], "integer")
        self.assertEqual(properties["ratio"]["type"], "number")
        expected_fields = {
            field for field, schema in properties.items() if schema["type"] == "integer"
        }
        self.assertEqual(fields, {"write_stdin": expected_fields})
        self.assertEqual(
            repairs, sorted(f"write_stdin.{field}" for field in expected_fields)
        )

    def test_split_streamed_arguments_are_reassembled_and_normalized(self) -> None:
        item = "fc_1"
        raw = _sse(
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": item,
                    "name": "write_stdin",
                    "arguments": "",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": item,
                "delta": '{"session_id":6594.',
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": item,
                "delta": '0,"yield_time_ms":90000.0,"chars":""}',
            },
            {
                "type": "response.function_call_arguments.done",
                "item_id": item,
                "arguments": '{"session_id":6594.0,"yield_time_ms":90000.0,"chars":""}',
            },
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": item,
                    "name": "write_stdin",
                    "arguments": '{"session_id":6594.0,"yield_time_ms":90000.0,"chars":""}',
                },
            },
        )
        cooked, changes = proxy._normalize_function_call_stream(
            raw, {"write_stdin": {"session_id", "yield_time_ms"}}
        )
        self.assertEqual(changes, 2)
        events = _events(cooked)
        deltas = [
            event["delta"]
            for event in events
            if event["type"] == "response.function_call_arguments.delta"
        ]
        arguments = json.loads("".join(deltas))
        self.assertEqual(arguments["session_id"], 6594)
        self.assertIsInstance(arguments["session_id"], int)
        self.assertEqual(arguments["yield_time_ms"], 90000)
        self.assertIsInstance(arguments["yield_time_ms"], int)
        done = next(
            event
            for event in events
            if event["type"] == "response.function_call_arguments.done"
        )
        self.assertEqual(json.loads(done["arguments"]), arguments)

    def test_non_integral_or_unrelated_values_are_not_changed(self) -> None:
        arguments = '{"session_id":1.5,"temperature":0.5}'
        cooked, changes = proxy._normalize_arguments(arguments, {"session_id"})
        self.assertEqual(cooked, arguments)
        self.assertEqual(changes, 0)


if __name__ == "__main__":
    unittest.main()

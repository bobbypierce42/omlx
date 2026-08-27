# SPDX-License-Identifier: Apache-2.0
"""Protocol tests for the OpenAI/NVIDIA-compatible VoiceChat transport."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.api.realtime_routes import _serialize_model_event, realtime_router
from omlx.engine.voicechat import VoiceChatBusyError, VoiceChatEngine, VoiceChatEvent
from omlx.server_metrics import ServerMetrics


class _StubStreamingSession:
    input_sample_rate = 16_000
    output_sample_rate = 22_050
    frame_samples = 1_280

    def __init__(self, push_events=None):
        self.push_events = list(push_events or [])
        self.push_lengths: list[int] = []
        self.push_rates: list[int] = []
        self.closed = False
        self.flush_calls: list[bool] = []
        self.cancel_calls = 0

    async def push_audio(self, samples, *, sample_rate):
        self.push_lengths.append(len(samples))
        self.push_rates.append(sample_rate)
        return list(self.push_events)

    async def flush(self, *, pad_partial=True):
        self.flush_calls.append(pad_partial)
        self.closed = True
        return [VoiceChatEvent(kind="done", frame_index=1)]

    async def cancel(self):
        self.cancel_calls += 1
        self.closed = True
        return [VoiceChatEvent(kind="cancelled", frame_index=0)]

    async def close(self):
        self.closed = True


def _client(session: _StubStreamingSession, *, engine=None):
    app = FastAPI()
    app.include_router(realtime_router)
    engine = engine or MagicMock(spec=VoiceChatEngine)
    engine.create_realtime_session = AsyncMock(return_value=session)
    pool = MagicMock()
    pool.get_entry.return_value = None
    pool.get_engine = AsyncMock(return_value=engine)
    patches = (
        patch("omlx.api.realtime_routes._get_engine_pool", return_value=pool),
        patch(
            "omlx.api.realtime_routes._resolve_model", side_effect=lambda value: value
        ),
        patch("omlx.api.realtime_routes._verify_ws_api_key", return_value=True),
    )
    return patches, TestClient(app), engine


def _connect_and_configure(client: TestClient, **session_fields):
    websocket = client.websocket_connect("/v1/realtime")
    websocket.__enter__()
    created = websocket.receive_json()
    config = {"model": "fake-voicechat", **session_fields}
    websocket.send_json({"type": "session.update", "session": config})
    updated = websocket.receive_json()
    return websocket, created, updated


def test_official_style_fixture_streams_voicechat_events():
    session = _StubStreamingSession(
        [
            VoiceChatEvent(
                kind="assistant_text_delta",
                frame_index=0,
                token_id=42,
                delta="hello",
                text="hello",
            ),
            VoiceChatEvent(
                kind="function_delta",
                frame_index=0,
                token_id=43,
                delta="{",
                text="{",
                call_id="call_lookup",
                function_name="lookup",
            ),
            VoiceChatEvent(
                kind="audio",
                frame_index=0,
                samples=np.zeros(1_764, dtype=np.float32),
                sample_rate=22_050,
                audio_codes=np.zeros(31, dtype=np.int32),
            ),
        ]
    )
    patches, client, engine = _client(session)
    with patches[0], patches[1], patches[2], client:
        websocket, created, updated = _connect_and_configure(client, seed=7)
        try:
            assert created["type"] == "session.created"
            assert created["session"]["state"] == "configuring"
            assert updated["type"] == "session.updated"
            assert updated["session"]["state"] == "ready"
            assert updated["session"]["frame_samples"] == 1_280

            pcm = np.zeros(1_280, dtype="<i2").tobytes()
            websocket.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode("ascii"),
                    "sample_rate": 16_000,
                }
            )
            assert websocket.receive_json()["type"] == "response.text.delta"
            function = websocket.receive_json()
            assert function["type"] == "response.function_call_arguments.delta"
            assert function["call_id"] == "call_lookup"
            assert function["name"] == "lookup"
            assert function["arguments"] == "{"
            audio = websocket.receive_json()
            assert audio["type"] == "response.audio.delta"
            assert audio["sample_rate"] == 22_050
            assert audio["format"] == "pcm16"
            assert audio["channels"] == 1
            assert len(base64.b64decode(audio["delta"])) == 1_764 * 2

            websocket.send_json({"type": "input_audio_buffer.commit"})
            assert websocket.receive_json()["type"] == "input_audio_buffer.committed"
            assert (
                websocket.receive_json()["type"]
                == "response.function_call_arguments.done"
            )
            assert websocket.receive_json()["type"] == "response.done"
        finally:
            websocket.__exit__(None, None, None)
    engine.create_realtime_session.assert_awaited_once_with(
        system_prompt=None,
        seed=7,
        max_streaming_seconds=None,
    )


def test_function_channel_emits_stable_structured_call_events():
    session = _StubStreamingSession(
        [
            SimpleNamespace(
                kind="function_delta",
                frame_index=0,
                token_id=1,
                delta="{",
                text="{",
                call_id="call_weather",
                function_name="weather",
            ),
            SimpleNamespace(
                kind="function_delta",
                frame_index=0,
                token_id=2,
                delta='"city":"Paris"}',
                text='{"city":"Paris"}',
                call_id="call_weather",
                function_name="weather",
            ),
        ]
    )
    patches, client, _ = _client(session)
    with patches[0], patches[1], patches[2], client:
        websocket, _, _ = _connect_and_configure(client)
        try:
            pcm = np.zeros(1_280, dtype="<i2").tobytes()
            websocket.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode("ascii"),
                }
            )
            first = websocket.receive_json()
            second = websocket.receive_json()
            assert first["type"] == "response.function_call_arguments.delta"
            assert second["type"] == "response.function_call_arguments.delta"
            assert first["call_id"] == second["call_id"] == "call_weather"
            assert first["item_id"] == second["item_id"]
            assert first["response_id"] == second["response_id"]
            assert first["name"] == second["name"] == "weather"
            assert second["arguments"] == '{"city":"Paris"}'

            websocket.send_json({"type": "input_audio_buffer.commit"})
            assert websocket.receive_json()["type"] == "input_audio_buffer.committed"
            done = websocket.receive_json()
            assert done["type"] == "response.function_call_arguments.done"
            assert done["call_id"] == "call_weather"
            assert done["name"] == "weather"
            assert done["arguments"] == '{"city":"Paris"}'
            assert websocket.receive_json()["type"] == "response.done"
        finally:
            websocket.__exit__(None, None, None)


def test_parallel_function_call_emits_unsupported_operation():
    session = _StubStreamingSession(
        [
            VoiceChatEvent(
                kind="function_delta",
                delta="{",
                parallel=True,
            )
        ]
    )
    patches, client, _ = _client(session)
    with patches[0], patches[1], patches[2], client:
        websocket, _, _ = _connect_and_configure(client)
        try:
            pcm = np.zeros(1_280, dtype="<i2").tobytes()
            websocket.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode("ascii"),
                }
            )
            error = websocket.receive_json()
            assert error["type"] == "error"
            assert error["error"]["code"] == "unsupported_operation"
        finally:
            websocket.__exit__(None, None, None)


def test_invalid_bearer_auth_is_rejected_without_starting_model():
    session = _StubStreamingSession()
    patches, client, engine = _client(session)
    patches = (
        *patches[:2],
        patch("omlx.api.realtime_routes._verify_ws_api_key", return_value=False),
    )
    with (
        patches[0],
        patches[1],
        patches[2],
        client,
        client.websocket_connect(
            "/v1/realtime", headers={"Authorization": "Bearer bad"}
        ) as websocket,
    ):
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["error"]["code"] == "invalid_api_key"
    engine.create_realtime_session.assert_not_awaited()


@pytest.mark.parametrize("sample_rate", [0, 12_345])
def test_invalid_append_sample_rates_are_rejected_before_model_work(sample_rate):
    session = _StubStreamingSession()
    patches, client, _ = _client(session)
    with patches[0], patches[1], patches[2], client:
        websocket, _, _ = _connect_and_configure(client)
        try:
            pcm = np.zeros(1_280, dtype="<i2").tobytes()
            websocket.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode("ascii"),
                    "sample_rate": sample_rate,
                }
            )
            error = websocket.receive_json()
            assert error["type"] == "error"
            assert error["error"]["param"] == "sample_rate"
            assert session.push_lengths == []
        finally:
            websocket.send_json({"type": "response.cancel"})
            assert websocket.receive_json()["type"] == "response.cancelled"
            websocket.__exit__(None, None, None)


def test_oversized_text_is_rejected_before_json_parsing():
    import omlx.api.realtime_routes as routes

    oversized = "{" + ("x" * routes._MAX_INBOUND_TEXT_BYTES)
    with (
        patch.object(routes.json, "loads", side_effect=AssertionError("parsed")),
        pytest.raises(routes._RealtimeProtocolError, match="maximum"),
    ):
        routes._parse_json_text(oversized)


def test_tool_extras_are_canonicalized_and_bounded():
    import omlx.api.realtime_routes as routes

    valid = routes._validate_tools(
        [
            {
                "type": "function",
                "name": "lookup",
                "parameters": {"type": "object"},
                "unbounded_extra": {"nested": {"ignored": True}},
            }
        ]
    )
    assert valid == [
        {"type": "function", "name": "lookup", "parameters": {"type": "object"}}
    ]
    with pytest.raises(routes._RealtimeProtocolError, match="too large"):
        routes._validate_tools(
            [
                {
                    "type": "function",
                    "name": "lookup",
                    "parameters": {
                        "description": "x" * (routes._MAX_TOOL_STRING_BYTES + 1)
                    },
                }
            ]
        )


def test_tools_are_rejected_when_engine_cannot_apply_them():
    import omlx.api.realtime_routes as routes

    tool = {
        "type": "function",
        "name": "lookup",
        "parameters": {"type": "object"},
    }
    settings = routes._validate_session_settings(
        {"model": "fake-voicechat", "tools": [tool]}, model_hint=None
    )

    class _NoToolsEngine:
        def __init__(self):
            self.calls = 0

        async def create_realtime_session(
            self, *, system_prompt=None, seed=0, max_streaming_seconds=None
        ):
            del system_prompt, seed, max_streaming_seconds
            self.calls += 1
            return _StubStreamingSession()

    engine = _NoToolsEngine()
    pool = SimpleNamespace(
        get_entry=lambda _model: None,
        get_engine=AsyncMock(return_value=engine),
    )
    with (
        patch(
            "omlx.api.realtime_routes._resolve_model", side_effect=lambda value: value
        ),
        pytest.raises(routes._RealtimeProtocolError, match="cannot apply") as exc_info,
    ):
        asyncio.run(routes._open_runtime(pool, settings, "sess_test"))
    assert exc_info.value.code == "unsupported_operation"
    assert engine.calls == 0


def test_tools_are_passed_when_engine_exposes_a_tools_parameter():
    import omlx.api.realtime_routes as routes

    tool = {
        "type": "function",
        "name": "lookup",
        "description": "Find a record",
        "parameters": {"type": "object"},
        "ignored": {"deep": {"value": True}},
    }
    settings = routes._validate_session_settings(
        {"model": "fake-voicechat", "tools": [tool]}, model_hint=None
    )

    class _ToolsEngine:
        def __init__(self):
            self.tools = None

        async def create_realtime_session(
            self, *, system_prompt=None, seed=0, max_streaming_seconds=None, tools=None
        ):
            del system_prompt, seed, max_streaming_seconds
            self.tools = tools
            return _StubStreamingSession()

    engine = _ToolsEngine()
    pool = SimpleNamespace(
        get_entry=lambda _model: None,
        get_engine=AsyncMock(return_value=engine),
    )
    with patch(
        "omlx.api.realtime_routes._resolve_model", side_effect=lambda value: value
    ):
        runtime = asyncio.run(routes._open_runtime(pool, settings, "sess_test"))
        asyncio.run(routes._close_runtime(runtime))
    assert engine.tools == [
        {
            "type": "function",
            "name": "lookup",
            "description": "Find a record",
            "parameters": {"type": "object"},
        }
    ]


def test_response_create_is_rejected_instead_of_being_silent():
    session = _StubStreamingSession()
    patches, client, _ = _client(session)
    with patches[0], patches[1], patches[2], client:
        websocket, _, _ = _connect_and_configure(client)
        try:
            websocket.send_json({"type": "response.create"})
            error = websocket.receive_json()
            assert error["type"] == "error"
            assert error["error"]["code"] == "unsupported_operation"
            assert session.push_lengths == []
            assert session.flush_calls == []
        finally:
            websocket.send_json({"type": "response.cancel"})
            assert websocket.receive_json()["type"] == "response.cancelled"
            websocket.__exit__(None, None, None)


def test_invalid_session_update_can_be_corrected_before_model_work():
    session = _StubStreamingSession()
    patches, client, engine = _client(session)
    with patches[0], patches[1], patches[2], client:
        websocket, _, validation_error = _connect_and_configure(
            client, input_audio_format="opus"
        )
        try:
            # The helper has already consumed the first response, which is the
            # validation error for the unsupported format.
            assert validation_error["type"] == "error"
            websocket.send_json(
                {
                    "type": "session.update",
                    "event_id": "valid-update",
                    "session": {"model": "fake-voicechat", "frame_duration_ms": 80},
                }
            )
            updated = websocket.receive_json()
            assert updated["type"] == "session.updated"
        finally:
            websocket.send_json({"type": "response.cancel"})
            assert websocket.receive_json()["type"] == "response.cancelled"
            websocket.__exit__(None, None, None)
    engine.create_realtime_session.assert_awaited_once()


def test_large_resampled_chunk_is_split_into_native_frames():
    session = _StubStreamingSession()
    patches, client, _ = _client(session)
    with patches[0], patches[1], patches[2], client:
        websocket, _, updated = _connect_and_configure(
            client,
            audio={
                "input": {"format": {"type": "audio/pcm", "rate": 24_000}},
                "output": {"format": {"type": "audio/pcm", "rate": 22_050}},
            },
        )
        try:
            assert updated["session"]["input_audio_format"]["sample_rate"] == 24_000
            pcm = np.zeros(5_760, dtype="<i2").tobytes()
            websocket.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode("ascii"),
                    "sample_rate": 24_000,
                }
            )
            deadline = time.monotonic() + 1
            while len(session.push_lengths) < 3 and time.monotonic() < deadline:
                time.sleep(0.01)
            websocket.send_json({"type": "response.cancel"})
            assert websocket.receive_json()["type"] == "response.cancelled"
        finally:
            websocket.__exit__(None, None, None)
    assert session.push_lengths == [1_280, 1_280, 1_280]
    assert session.push_rates == [16_000, 16_000, 16_000]


def test_response_cancel_preempts_a_large_append():
    started = threading.Event()
    cancelled = threading.Event()
    push_calls: list[int] = []

    class _SlowSession(_StubStreamingSession):
        async def push_audio(self, samples, *, sample_rate):
            push_calls.append(len(samples))
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return []

    session = _SlowSession()
    patches, client, _ = _client(session)
    with patches[0], patches[1], patches[2], client:
        websocket, _, _ = _connect_and_configure(client)
        try:
            pcm = np.zeros(1_280 * 100, dtype="<i2").tobytes()
            websocket.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode("ascii"),
                }
            )
            assert started.wait(1)
            websocket.send_json({"type": "response.cancel"})
            assert websocket.receive_json()["type"] == "response.cancelled"
        finally:
            websocket.__exit__(None, None, None)
    assert cancelled.is_set()
    assert push_calls == [1_280]
    assert session.cancel_calls == 1


def test_duplicate_event_id_and_tool_result_are_rejected():
    session = _StubStreamingSession()
    patches, client, _ = _client(session)
    with patches[0], patches[1], patches[2], client:
        websocket, _, _ = _connect_and_configure(client)
        try:
            websocket.send_json({"type": "session.ping", "event_id": "duplicate"})
            assert websocket.receive_json()["type"] == "session.pong"
            websocket.send_json({"type": "session.ping", "event_id": "duplicate"})
            error = websocket.receive_json()
            assert error["error"]["code"] == "invalid_request"
            assert error["error"]["param"] == "event_id"

            websocket.send_json(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "ok",
                    },
                }
            )
            error = websocket.receive_json()
            assert error["error"]["code"] == "unsupported_operation"
        finally:
            websocket.send_json({"type": "response.cancel"})
            assert websocket.receive_json()["type"] == "response.cancelled"
            websocket.__exit__(None, None, None)


def test_busy_session_closes_with_retryable_code():
    first_session = _StubStreamingSession()
    patches, client, engine = _client(first_session)
    engine.create_realtime_session.side_effect = [
        first_session,
        VoiceChatBusyError(),
    ]
    with patches[0], patches[1], patches[2], client:
        first, _, _ = _connect_and_configure(client)
        try:
            second = client.websocket_connect("/v1/realtime")
            second.__enter__()
            try:
                assert second.receive_json()["type"] == "session.created"
                second.send_json(
                    {"type": "session.update", "session": {"model": "fake-voicechat"}}
                )
                error = second.receive_json()
                assert error["error"]["code"] == "server_busy"
                try:
                    second.receive_json()
                except Exception as exc:
                    assert getattr(exc, "code", None) == 1013
            finally:
                second.__exit__(None, None, None)
        finally:
            first.send_json({"type": "response.cancel"})
            assert first.receive_json()["type"] == "response.cancelled"
            first.__exit__(None, None, None)


def test_route_disconnect_increments_server_metrics():
    metrics = ServerMetrics()
    patches, client, _ = _client(_StubStreamingSession())
    with (
        patches[0],
        patches[1],
        patches[2],
        patch("omlx.server_metrics.get_server_metrics", return_value=metrics),
        client,
    ):
        websocket, _, _ = _connect_and_configure(client)
        websocket.__exit__(None, None, None)

    snapshot = metrics.get_voicechat_snapshot()
    assert snapshot["session_terminal_states"] == {
        "done": 0,
        "cancelled": 0,
        "disconnected": 1,
        "busy": 0,
        "error": 0,
    }


def test_route_cancel_increments_server_metrics():
    metrics = ServerMetrics()
    patches, client, _ = _client(_StubStreamingSession())
    with (
        patches[0],
        patches[1],
        patches[2],
        patch("omlx.server_metrics.get_server_metrics", return_value=metrics),
        client,
    ):
        websocket, _, _ = _connect_and_configure(client)
        websocket.send_json({"type": "response.cancel"})
        assert websocket.receive_json()["type"] == "response.cancelled"
        websocket.__exit__(None, None, None)

    snapshot = metrics.get_voicechat_snapshot()
    assert snapshot["session_terminal_states"]["cancelled"] == 1
    assert snapshot["cancellations"] == 1


def test_route_busy_increments_server_metrics():
    metrics = ServerMetrics()
    patches, client, engine = _client(_StubStreamingSession())
    engine.create_realtime_session.side_effect = VoiceChatBusyError()
    with (
        patches[0],
        patches[1],
        patches[2],
        patch("omlx.server_metrics.get_server_metrics", return_value=metrics),
        client,
    ):
        websocket = client.websocket_connect("/v1/realtime")
        websocket.__enter__()
        try:
            assert websocket.receive_json()["type"] == "session.created"
            websocket.send_json(
                {"type": "session.update", "session": {"model": "fake-voicechat"}}
            )
            assert websocket.receive_json()["error"]["code"] == "server_busy"
        finally:
            websocket.__exit__(None, None, None)

    snapshot = metrics.get_voicechat_snapshot()
    assert snapshot["session_terminal_states"]["busy"] == 1
    assert snapshot["busy_responses"] == 1


def test_route_error_increments_server_metrics():
    metrics = ServerMetrics()
    session = _StubStreamingSession(
        [VoiceChatEvent(kind="error", error="backend failure")]
    )
    patches, client, _ = _client(session)
    with (
        patches[0],
        patches[1],
        patches[2],
        patch("omlx.server_metrics.get_server_metrics", return_value=metrics),
        client,
    ):
        websocket, _, _ = _connect_and_configure(client)
        try:
            pcm = np.zeros(1_280, dtype="<i2").tobytes()
            websocket.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode("ascii"),
                }
            )
            assert websocket.receive_json()["type"] == "error"
        finally:
            websocket.__exit__(None, None, None)

    snapshot = metrics.get_voicechat_snapshot()
    assert snapshot["session_terminal_states"]["error"] == 1


def test_disconnect_during_session_startup_cancels_startup_task():
    from omlx.api.realtime_routes import realtime_voicechat

    started = threading.Event()
    cancelled = threading.Event()

    async def blocked_create(**_kwargs):
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    engine = MagicMock(spec=VoiceChatEngine)
    engine.create_realtime_session = blocked_create

    class _DisconnectedWebSocket:
        headers = {}
        query_params = SimpleNamespace(get=lambda _name: None)

        def __init__(self):
            self.sent = []
            self._first = True

        async def accept(self):
            pass

        async def send_json(self, payload):
            self.sent.append(payload)

        async def receive(self):
            if self._first:
                self._first = False
                return {
                    "type": "websocket.receive",
                    "text": '{"type":"session.update","session":{"model":"fake-voicechat"}}',
                }
            await asyncio.to_thread(started.wait, 1)
            return {"type": "websocket.disconnect", "code": 1000}

        async def close(self, code=1000):
            self.close_code = code

    websocket = _DisconnectedWebSocket()
    pool = MagicMock()
    pool.get_entry.return_value = None
    pool.get_engine = AsyncMock(return_value=engine)
    with (
        patch("omlx.api.realtime_routes._get_engine_pool", return_value=pool),
        patch(
            "omlx.api.realtime_routes._resolve_model", side_effect=lambda value: value
        ),
        patch("omlx.api.realtime_routes._verify_ws_api_key", return_value=True),
    ):
        asyncio.run(realtime_voicechat(websocket))

    assert started.is_set()
    assert cancelled.is_set()


def test_audio_serializer_preserves_native_output_metadata():
    event = VoiceChatEvent(
        kind="audio",
        samples=np.zeros(1_764, dtype=np.float32),
        sample_rate=22_050,
        audio_codes=np.arange(31, dtype=np.int32),
    )
    payload = _serialize_model_event(event)
    assert payload["sample_rate"] == 22_050
    assert payload["format"] == "pcm16"
    assert payload["channels"] == 1
    assert payload["audio_codes"] == list(range(31))


def test_backend_error_redacts_credentials_before_sending():
    import omlx.api.realtime_routes as routes

    class _WebSocket:
        def __init__(self):
            self.sent = []

        async def send_json(self, payload):
            self.sent.append(payload)

    websocket = _WebSocket()
    asyncio.run(
        routes._send_error(
            websocket,
            "backend failed Authorization: Bearer super-secret-token "
            "api_key=another-secret sk-1234567890abcdef",
        )
    )
    serialized = json.dumps(websocket.sent)
    assert "super-secret-token" not in serialized
    assert "another-secret" not in serialized
    assert "sk-1234567890abcdef" not in serialized
    assert "redacted" in serialized


def test_session_content_is_not_written_to_trace_logs(caplog):
    session = _StubStreamingSession()
    patches, client, _ = _client(session)
    secret = "private-instruction-should-not-be-logged"
    with patches[0], patches[1], patches[2], client, caplog.at_level(logging.DEBUG):
        websocket, _, updated = _connect_and_configure(client, instructions=secret)
        try:
            assert updated["session"]["instructions"] == secret
        finally:
            websocket.send_json({"type": "response.cancel"})
            assert websocket.receive_json()["type"] == "response.cancelled"
            websocket.__exit__(None, None, None)
    assert secret not in caplog.text


@pytest.mark.parametrize(
    "terminal_state",
    ["completed", "cancelled", "disconnected", "busy", "error"],
)
def test_realtime_terminal_finalizer_records_each_terminal_path(terminal_state):
    import omlx.api.realtime_routes as routes

    with patch.object(routes, "_record_voicechat_metric") as record:
        routes._record_realtime_terminal(None, terminal_state, "voice-model")

    record.assert_called_once_with(
        "record_voicechat_session_terminal",
        terminal_state,
        model_id="voice-model",
    )


def test_realtime_terminal_finalizer_records_only_operational_runtime_fields():
    import omlx.api.realtime_routes as routes

    runtime = SimpleNamespace(
        terminal_state="completed",
        model_id="voice-model",
        turns=2,
        interruptions=1,
        cancellations=1,
        input_audio_duration_seconds=1.5,
        output_audio_duration_seconds=2.5,
        first_transcript_latency_seconds=0.2,
        first_text_latency_seconds=0.4,
        first_audio_latency_seconds=0.6,
        turn_latency_seconds=1.0,
        tool_requests=1,
        tool_failures=1,
        tool_parallel_rejections=1,
    )
    with patch.object(routes, "_record_voicechat_metric") as record:
        routes._record_realtime_terminal(runtime, "disconnected")

    record.assert_called_once_with(
        "record_voicechat_session_terminal",
        "completed",
        model_id="voice-model",
        turns=2,
        interruptions=1,
        cancellations=1,
        audio_duration_seconds=1.5,
        output_audio_duration_seconds=2.5,
        first_transcript_latency_seconds=0.2,
        first_text_latency_seconds=0.4,
        first_audio_latency_seconds=0.6,
        turn_latency_seconds=1.0,
        tool_requests=1,
        tool_failures=1,
        tool_parallel_rejections=1,
    )

# SPDX-License-Identifier: Apache-2.0
"""Protocol tests for the OpenAI/NVIDIA-compatible VoiceChat transport."""

from __future__ import annotations

import asyncio
import base64
import logging
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.api.realtime_routes import _serialize_model_event, realtime_router
from omlx.engine.voicechat import VoiceChatBusyError, VoiceChatEngine, VoiceChatEvent


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
            assert websocket.receive_json()["type"] == "response.function.delta"
            audio = websocket.receive_json()
            assert audio["type"] == "response.audio.delta"
            assert audio["sample_rate"] == 22_050
            assert audio["format"] == "pcm16"
            assert audio["channels"] == 1
            assert len(base64.b64decode(audio["delta"])) == 1_764 * 2

            websocket.send_json({"type": "input_audio_buffer.commit"})
            assert websocket.receive_json()["type"] == "input_audio_buffer.committed"
            assert websocket.receive_json()["type"] == "response.done"
        finally:
            websocket.__exit__(None, None, None)
    engine.create_realtime_session.assert_awaited_once_with(
        system_prompt=None,
        seed=7,
        max_streaming_seconds=None,
    )


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
            websocket.send_json({"type": "response.cancel"})
            assert websocket.receive_json()["type"] == "response.cancelled"
        finally:
            websocket.__exit__(None, None, None)
    assert session.push_lengths == [1_280, 1_280, 1_280]
    assert session.push_rates == [16_000, 16_000, 16_000]


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

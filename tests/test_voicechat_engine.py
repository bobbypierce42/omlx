# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the dedicated VoiceChat engine lifecycle."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import mlx.core as mx
import numpy as np
import pytest

import omlx.engine.voicechat as voicechat_module
from omlx.engine.voicechat import (
    VoiceChatBusyError,
    VoiceChatEngine,
    VoiceChatEventKind,
    VoiceChatInferenceError,
    VoiceChatStatus,
    _safe_error_message,
    normalize_voicechat_events,
)
from omlx.engine_pool import EngineEntry, EnginePool


class _ExecutorOnlyArray:
    dtype = mx.float32

    def __init__(self, values):
        self._values = values
        self.size = len(values)

    def __len__(self):
        return self.size

    def __array__(self, dtype=None, copy=None):
        if not threading.current_thread().name.startswith("mlx-global"):
            raise AssertionError("MLX output was converted outside the executor")
        result = np.asarray(self._values, dtype=dtype)
        return result.copy() if copy else result


class FakeStreamingSession:
    input_sample_rate = 16_000
    output_sample_rate = 22_050
    frame_samples = 1_280

    def __init__(self):
        self.closed = False
        self.cancel_calls = 0
        self.block_push = False
        self.push_calls = 0
        self.push_lengths = []
        self.push_delay = 0.0
        self.push_started = threading.Event()
        self.push_release = threading.Event()
        self.events = [
            SimpleNamespace(
                kind="audio",
                frame_index=0,
                samples=[0.0, 0.25],
                sample_rate=22_050,
            )
        ]
        self.push_error: Exception | None = None

    def push_audio(self, samples, *, sample_rate=None):
        self.push_calls += 1
        self.push_lengths.append(len(samples))
        del samples, sample_rate
        if self.push_calls == 1:
            self.push_started.set()
        if self.push_delay:
            time.sleep(self.push_delay)
        if self.block_push:
            self.push_release.wait(timeout=2)
        if self.push_error is not None:
            raise self.push_error
        return list(self.events)

    def flush(self, *, pad_partial=True):
        del pad_partial
        self.closed = True
        return list(self.events) + [SimpleNamespace(kind="done", frame_index=1)]

    def cancel(self):
        self.cancel_calls += 1
        self.closed = True
        return [SimpleNamespace(kind="cancelled", frame_index=1)]


class FakeModelSession:
    def __init__(self, streaming_session: FakeStreamingSession):
        self.tokenizer = SimpleNamespace()
        self.streaming_session = streaming_session
        self.generate_result = SimpleNamespace(
            audio=[0.0, 0.25, -0.25], sample_rate=22_050
        )
        self.generate_error: Exception | None = None
        self.create_started = threading.Event()
        self.create_release: threading.Event | None = None

    def create_streaming_session(self, **kwargs):
        del kwargs
        self.create_started.set()
        if self.create_release is not None:
            self.create_release.wait(timeout=2)
        return self.streaming_session

    def generate(self, audio, **kwargs):
        del audio, kwargs
        if self.generate_error is not None:
            raise self.generate_error
        return self.generate_result


class FakeModel:
    def __init__(self, session: FakeModelSession):
        self.session = session
        self.config = SimpleNamespace(model_type="nemotron_voicechat")

    def create_session(self, processor):
        del processor
        return self.session


async def _make_engine(
    streaming_session: FakeStreamingSession | None = None,
    *,
    session: FakeModelSession | None = None,
) -> tuple[VoiceChatEngine, FakeModelSession, FakeStreamingSession]:
    raw = streaming_session or FakeStreamingSession()
    model_session = session or FakeModelSession(raw)
    model = FakeModel(model_session)
    engine = VoiceChatEngine("voicechat", loader=lambda _: (model, object()))
    await engine.start()
    return engine, model_session, raw


async def _wait_for_event(event: threading.Event) -> None:
    for _ in range(200):
        if event.is_set():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("worker event was not set")


@pytest.mark.asyncio
async def test_load_uses_strict_merged_mlx_vlm_loader():
    raw = FakeStreamingSession()
    model_session = FakeModelSession(raw)
    model = FakeModel(model_session)
    engine = VoiceChatEngine("voicechat")

    with patch("mlx_vlm.load", return_value=(model, object())) as load:
        await engine.start()

    load.assert_called_once_with(
        "voicechat",
        lazy=True,
        strict=True,
        trust_remote_code=False,
    )
    assert engine.status is VoiceChatStatus.QUEUED
    await engine.stop()


@pytest.mark.asyncio
async def test_load_failure_is_strict_and_status_is_failed():
    engine = VoiceChatEngine("voicechat", loader=lambda _: (object(), object()))

    with pytest.raises(TypeError, match="session API"):
        await engine.start()

    assert engine.status is VoiceChatStatus.FAILED
    assert engine.get_status()["loaded"] is False
    assert engine.get_status()["error"]


def test_safe_error_message_redacts_json_api_key():
    secret = "json-shaped-api-key"
    message = _safe_error_message(
        RuntimeError(f'backend failed: {{"api_key": "{secret}"}}')
    )

    assert secret not in message
    assert '"api_key": "<redacted>"' in message


@pytest.mark.asyncio
async def test_reservation_is_atomic_before_stream_startup_await():
    raw = FakeStreamingSession()
    model_session = FakeModelSession(raw)
    model_session.create_release = threading.Event()
    engine, _, _ = await _make_engine(raw, session=model_session)

    first = asyncio.create_task(engine.create_realtime_session())
    await _wait_for_event(model_session.create_started)

    with pytest.raises(VoiceChatBusyError) as exc_info:
        await engine.create_realtime_session()
    assert exc_info.value.retryable is True
    assert exc_info.value.code == "busy"

    model_session.create_release.set()
    session = await first
    await session.release()
    await engine.stop()


@pytest.mark.asyncio
async def test_disconnect_cleanup_releases_reservation_and_activity():
    engine, _, raw = await _make_engine()
    session = await engine.create_realtime_session()

    assert engine.has_active_requests() is True
    events = await session.release()

    assert [event.kind for event in events] == [VoiceChatEventKind.CANCELLED]
    assert raw.cancel_calls == 1
    assert engine.has_active_requests() is False
    assert engine.status is VoiceChatStatus.QUEUED
    await engine.stop()


@pytest.mark.asyncio
async def test_cancel_returns_before_huge_push_finishes():
    raw = FakeStreamingSession()
    raw.push_delay = 0.02
    engine, _, _ = await _make_engine(raw)
    session = await engine.create_realtime_session()

    push = asyncio.create_task(session.push_audio([0.0] * (1_280 * 64)))
    await _wait_for_event(raw.push_started)

    cancelled = await asyncio.wait_for(session.cancel(), timeout=1.0)

    assert [event.kind for event in cancelled] == [VoiceChatEventKind.CANCELLED]
    assert raw.push_calls < 64
    assert all(length <= 1_280 for length in raw.push_lengths)
    assert raw.cancel_calls == 1
    await push
    assert engine.has_active_requests() is False
    await engine.stop()


@pytest.mark.asyncio
async def test_cancel_during_model_work_drains_before_release():
    raw = FakeStreamingSession()
    raw.block_push = True
    engine, _, _ = await _make_engine(raw)
    session = await engine.create_realtime_session()

    push = asyncio.create_task(session.push_audio([0.0] * 1_280))
    await _wait_for_event(raw.push_started)
    push.cancel()
    with pytest.raises(asyncio.CancelledError):
        await push

    assert engine.has_active_requests() is True
    raw.push_release.set()
    for _ in range(200):
        if not engine.has_active_requests():
            break
        await asyncio.sleep(0.005)

    assert engine.has_active_requests() is False
    assert raw.cancel_calls == 1
    assert session.closed is True
    await engine.stop()


@pytest.mark.asyncio
async def test_close_detaches_raw_session_references():
    engine, _, raw = await _make_engine()
    session = await engine.create_realtime_session()
    reservation = session._reservation

    await session.close()

    assert raw.closed is True
    assert session._raw_session is None
    assert session._engine is None
    assert reservation.raw_session is None
    assert reservation.wrapper is None
    await engine.stop()
    assert engine._model is None


@pytest.mark.asyncio
async def test_offline_work_is_busy_while_realtime_owns_checkpoint():
    engine, _, _ = await _make_engine()
    session = await engine.create_realtime_session()

    with pytest.raises(VoiceChatBusyError):
        await engine.process("input.wav")

    await session.release()
    output = await engine.process("input.wav")
    assert output[:4] == b"RIFF"
    await engine.stop()


@pytest.mark.asyncio
async def test_offline_no_audio_and_backend_errors_are_structured():
    engine, model_session, _ = await _make_engine()
    model_session.generate_result = SimpleNamespace(audio=[], sample_rate=22_050)

    with pytest.raises(VoiceChatInferenceError, match="no audio"):
        await engine.process("input.wav")

    model_session.generate_error = RuntimeError("backend failed")
    with pytest.raises(VoiceChatInferenceError, match="backend failed"):
        await engine.process("input.wav")
    await engine.stop()


def test_event_normalization_does_not_convert_remaining_mlx_arrays():
    samples = _ExecutorOnlyArray([0.0, 0.25])

    normalized = normalize_voicechat_events(
        [SimpleNamespace(kind="audio", samples=samples)]
    )

    assert normalized[0].samples is samples


@pytest.mark.asyncio
async def test_realtime_audio_and_codes_are_materialized_on_mlx_executor():
    raw = FakeStreamingSession()
    raw.events = [
        SimpleNamespace(
            kind="audio",
            frame_index=0,
            samples=_ExecutorOnlyArray([0.0, 0.25]),
            sample_rate=22_050,
            audio_codes=_ExecutorOnlyArray([1, 2]),
        )
    ]
    engine, _, _ = await _make_engine(raw)

    with patch.object(voicechat_module.mx, "array", _ExecutorOnlyArray):
        try:
            session = await engine.create_realtime_session()
            events = await session.push_audio([0.0])
        finally:
            await engine.stop()

    assert isinstance(events[0].samples, np.ndarray)
    assert isinstance(events[0].audio_codes, np.ndarray)


@pytest.mark.asyncio
async def test_offline_audio_is_materialized_before_wav_encoding():
    raw = FakeStreamingSession()
    model_session = FakeModelSession(raw)
    model_session.generate_result = SimpleNamespace(
        audio=_ExecutorOnlyArray([0.0, 0.25, -0.25]),
        sample_rate=22_050,
    )
    engine, _, _ = await _make_engine(raw, session=model_session)

    with patch.object(voicechat_module.mx, "array", _ExecutorOnlyArray):
        try:
            output = await engine.process("input.wav")
        finally:
            await engine.stop()

    assert output[:4] == b"RIFF"


@pytest.mark.asyncio
async def test_realtime_backend_and_no_audio_events_are_normalized():
    no_audio = SimpleNamespace(kind="audio", frame_index=2, samples=[])
    backend_events = normalize_voicechat_events(
        [
            SimpleNamespace(kind="user_transcript_delta", delta="hi", text="hi"),
            SimpleNamespace(kind="assistant_text_delta", delta="hello", text="hello"),
            SimpleNamespace(kind="function_delta", delta="{}", text="{}"),
            no_audio,
        ]
    )

    assert [event.kind for event in backend_events] == [
        VoiceChatEventKind.TRANSCRIPT,
        VoiceChatEventKind.ASSISTANT_TEXT,
        VoiceChatEventKind.FUNCTION,
        VoiceChatEventKind.ERROR,
    ]
    assert backend_events[-1].code == "no_audio"

    raw = FakeStreamingSession()
    raw.push_error = RuntimeError("backend failed")
    engine, _, _ = await _make_engine(raw)
    session = await engine.create_realtime_session()
    events = await session.push_audio([0.0])
    assert events[0].kind is VoiceChatEventKind.ERROR
    assert events[0].error == "backend failed"
    assert raw.cancel_calls == 1
    await engine.stop()


@pytest.mark.asyncio
async def test_pool_tracks_voicechat_activity_and_pending_unload():
    raw = FakeStreamingSession()
    raw.block_push = True
    engine, _, _ = await _make_engine(raw)
    session = await engine.create_realtime_session()
    pool = EnginePool()
    entry = EngineEntry(
        model_id="voicechat",
        model_path="/models/voicechat",
        model_type="audio_sts",
        engine_type="voicechat",
        estimated_size=1,
        config_model_type="nemotron_voicechat",
        is_voicechat=True,
        engine=engine,
    )
    pool._entries["voicechat"] = entry
    pool._unload_engine = AsyncMock()

    push = asyncio.create_task(session.push_audio([0.0] * 1_280))
    await _wait_for_event(raw.push_started)
    unload = asyncio.create_task(pool.request_unload("voicechat"))
    for _ in range(200):
        if entry.pending_unload_reason:
            break
        await asyncio.sleep(0.005)
    assert entry.pending_unload_reason == "manual unload"
    assert pool.get_status()["models"][0]["voicechat_status"] == "active"

    raw.push_release.set()
    await push
    assert await unload is True
    pool._unload_engine.assert_awaited_once_with("voicechat")
    await engine.stop()

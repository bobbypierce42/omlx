# SPDX-License-Identifier: Apache-2.0
"""Dedicated engine for the model-specific Nemotron VoiceChat runtime."""

from __future__ import annotations

import asyncio
import gc
import inspect
import logging
import threading
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from functools import partial
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..engine_core import get_mlx_executor
from .audio_utils import audio_to_wav_bytes
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)

VOICECHAT_INPUT_SAMPLE_RATE = 16_000
VOICECHAT_OUTPUT_SAMPLE_RATE = 22_050
VOICECHAT_FRAME_DURATION_MS = 80
VOICECHAT_FRAME_SAMPLES = (
    VOICECHAT_INPUT_SAMPLE_RATE * VOICECHAT_FRAME_DURATION_MS // 1000
)


class VoiceChatStatus(StrEnum):
    """Safe lifecycle states exposed by a VoiceChat engine."""

    LOADING = "loading"
    ACTIVE = "active"
    QUEUED = "queued"
    FAILED = "failed"
    UNLOADED = "unloaded"


VoiceChatEngineStatus = VoiceChatStatus


class VoiceChatEventKind(StrEnum):
    """Internal event categories consumed by the future transport layer."""

    TRANSCRIPT = "transcript"
    ASSISTANT_TEXT = "assistant_text"
    FUNCTION = "function"
    AUDIO = "audio"
    DONE = "done"
    CANCELLED = "cancelled"
    BUSY = "busy"
    ERROR = "error"


VoiceChatEventType = VoiceChatEventKind


@dataclass
class VoiceChatEvent:
    """Normalized internal output from a realtime VoiceChat turn."""

    kind: VoiceChatEventKind | str
    frame_index: int | None = None
    token_id: int | None = None
    delta: str | None = None
    text: str | None = None
    samples: Any | None = None
    sample_rate: int | None = None
    audio_codes: Any | None = None
    error: str | None = None
    code: str | None = None
    retryable: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the internal event fields for a protocol adapter."""
        result: dict[str, Any] = {
            "kind": self.kind.value if isinstance(self.kind, Enum) else self.kind
        }
        for name in (
            "frame_index",
            "token_id",
            "delta",
            "text",
            "samples",
            "sample_rate",
            "audio_codes",
            "error",
            "code",
            "retryable",
        ):
            value = getattr(self, name)
            if value is not None and (name != "retryable" or value):
                result[name] = value
        return result

    @property
    def type(self) -> str:
        """Return the string event kind used by protocol adapters."""
        return self.kind.value if isinstance(self.kind, Enum) else str(self.kind)


class VoiceChatError(RuntimeError):
    """Base error carrying the fields needed by a later protocol adapter."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "voicechat_error",
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class VoiceChatBusyError(VoiceChatError):
    """The checkpoint is reserved by another operation."""

    def __init__(self) -> None:
        super().__init__(
            "VoiceChat checkpoint already has an active session; retry later",
            code="busy",
            retryable=True,
        )

    def as_event(self) -> VoiceChatEvent:
        return VoiceChatEvent(
            kind=VoiceChatEventKind.BUSY,
            error=str(self),
            code=self.code,
            retryable=True,
        )


class VoiceChatInferenceError(VoiceChatError):
    """The model failed to create or produce a VoiceChat result."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="inference_error")


def _safe_error_message(exc: BaseException) -> str:
    message = str(exc).replace("\r", " ").replace("\n", " ").strip()
    return message[:500] or type(exc).__name__


def _error_event(
    message: str,
    *,
    code: str = "inference_error",
    retryable: bool = False,
) -> VoiceChatEvent:
    return VoiceChatEvent(
        kind=VoiceChatEventKind.ERROR,
        error=message,
        code=code,
        retryable=retryable,
    )


def _event_value(event: Any, name: str, default: Any = None) -> Any:
    if isinstance(event, dict):
        return event.get(name, default)
    return getattr(event, name, default)


def _has_audio_samples(samples: Any) -> bool:
    if samples is None:
        return False
    size = getattr(samples, "size", None)
    if size is not None:
        try:
            return int(size) > 0
        except (TypeError, ValueError):
            pass
    try:
        return len(samples) > 0
    except TypeError:
        return True


_MODEL_EVENT_KIND_MAP = {
    "user_transcript_delta": VoiceChatEventKind.TRANSCRIPT,
    "transcript": VoiceChatEventKind.TRANSCRIPT,
    "assistant_text_delta": VoiceChatEventKind.ASSISTANT_TEXT,
    "assistant_text": VoiceChatEventKind.ASSISTANT_TEXT,
    "function_delta": VoiceChatEventKind.FUNCTION,
    "function": VoiceChatEventKind.FUNCTION,
    "audio": VoiceChatEventKind.AUDIO,
    "done": VoiceChatEventKind.DONE,
    "cancelled": VoiceChatEventKind.CANCELLED,
    "busy": VoiceChatEventKind.BUSY,
    "error": VoiceChatEventKind.ERROR,
}


def normalize_voicechat_event(event: Any) -> VoiceChatEvent:
    """Map one merged mlx-vlm event to oMLX's internal event vocabulary."""
    if isinstance(event, VoiceChatEvent):
        return event

    raw_kind = _event_value(event, "kind")
    if isinstance(raw_kind, Enum):
        raw_kind = raw_kind.value
    normalized_kind = _MODEL_EVENT_KIND_MAP.get(str(raw_kind))
    if normalized_kind is None:
        return _error_event(
            "VoiceChat model emitted an unsupported event", code="protocol"
        )

    samples = _event_value(event, "samples")
    if normalized_kind is VoiceChatEventKind.AUDIO and not _has_audio_samples(samples):
        return _error_event("VoiceChat model produced no audio output", code="no_audio")
    sample_rate = _event_value(event, "sample_rate")
    if normalized_kind is VoiceChatEventKind.AUDIO and sample_rate is None:
        sample_rate = VOICECHAT_OUTPUT_SAMPLE_RATE

    return VoiceChatEvent(
        kind=normalized_kind,
        frame_index=_event_value(event, "frame_index"),
        token_id=_event_value(event, "token_id"),
        delta=_event_value(event, "delta"),
        text=_event_value(event, "text", _event_value(event, "transcript")),
        samples=samples,
        sample_rate=sample_rate,
        audio_codes=_event_value(event, "audio_codes"),
        error=_event_value(event, "error", _event_value(event, "message")),
        code=_event_value(event, "code"),
        retryable=bool(_event_value(event, "retryable", False)),
    )


def normalize_voicechat_events(events: Any) -> list[VoiceChatEvent]:
    """Normalize a model event list without accepting tool-result mutations."""
    if events is None:
        return []
    if isinstance(events, (dict, VoiceChatEvent)) or hasattr(events, "kind"):
        events = [events]
    return [normalize_voicechat_event(event) for event in events]


def _load_voicechat_components(model_name: str):
    """Load the merged mlx-vlm model, processor, and model-specific session."""
    try:
        from mlx_vlm import load
    except ImportError as exc:
        raise ImportError(
            "mlx-vlm with merged VoiceChat support is required for VoiceChat inference"
        ) from exc

    model, processor = load(
        model_name,
        lazy=True,
        strict=True,
        trust_remote_code=False,
    )
    create_session = getattr(model, "create_session", None)
    if not callable(create_session):
        raise TypeError("loaded model does not expose the VoiceChat session API")
    return model, processor, create_session(processor)


def _coerce_loaded_components(loaded: Any):
    def validate(session: Any):
        missing = [
            name
            for name in ("generate", "create_streaming_session")
            if not callable(getattr(session, name, None))
        ]
        if missing:
            raise TypeError(
                "loaded model does not expose the VoiceChat session API: "
                + ", ".join(missing)
            )
        return session

    if isinstance(loaded, tuple):
        if len(loaded) == 3:
            model, processor, session = loaded
            return model, processor, validate(session)
        if len(loaded) == 2:
            model, processor = loaded
            create_session = getattr(model, "create_session", None)
            if not callable(create_session):
                raise TypeError(
                    "loaded model does not expose the VoiceChat session API"
                )
            return model, processor, validate(create_session(processor))
    session = loaded
    model = getattr(session, "model", None)
    processor = getattr(session, "tokenizer", None)
    return model, processor, validate(session)


@dataclass
class _Reservation:
    mode: str
    activity_id: str
    reservation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    raw_session: Any | None = None
    wrapper: VoiceChatRealtimeSession | None = None
    operation: asyncio.Future[Any] | None = None
    released: bool = False
    close_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class VoiceChatEngine(BaseNonStreamingEngine):
    """One model-specific VoiceChat runtime with checkpoint-wide admission."""

    def __init__(
        self,
        model_name: str,
        *,
        loader: Callable[[str], Any] | None = None,
        input_sample_rate: int = VOICECHAT_INPUT_SAMPLE_RATE,
        output_sample_rate: int = VOICECHAT_OUTPUT_SAMPLE_RATE,
        frame_duration_ms: int = VOICECHAT_FRAME_DURATION_MS,
    ) -> None:
        super().__init__()
        self._model_name = model_name
        self._loader = loader or _load_voicechat_components
        self._model: Any | None = None
        self._processor: Any | None = None
        self._session: Any | None = None
        self._state_lock = threading.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._reservation: _Reservation | None = None
        self._status = VoiceChatStatus.UNLOADED
        self._load_error: str | None = None
        self._cleanup_tasks: set[asyncio.Task[Any]] = set()
        self.input_sample_rate = int(input_sample_rate)
        self.output_sample_rate = int(output_sample_rate)
        self.frame_duration_ms = int(frame_duration_ms)
        self.frame_samples = self.input_sample_rate * self.frame_duration_ms // 1000

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def tokenizer(self) -> Any | None:
        return getattr(self._session, "tokenizer", self._processor)

    @property
    def model_type(self) -> str:
        config = getattr(self._model, "config", None)
        return str(getattr(config, "model_type", "nemotron_voicechat"))

    @property
    def status(self) -> VoiceChatStatus:
        with self._state_lock:
            return self._status

    def get_status(self) -> dict[str, Any]:
        with self._state_lock:
            reservation = self._reservation
            status = self._status
            loaded = self._session is not None
            return {
                "status": status.value,
                "loaded": loaded,
                "active_session": reservation is not None
                and reservation.mode == "realtime",
                "session_mode": reservation.mode if reservation else None,
                "input_sample_rate": self.input_sample_rate,
                "output_sample_rate": self.output_sample_rate,
                "frame_duration_ms": self.frame_duration_ms,
                "frame_samples": self.frame_samples,
                "error": self._load_error,
            }

    async def start(self) -> None:
        """Load merged mlx-vlm components on the process MLX executor."""
        async with self._lifecycle_lock:
            with self._state_lock:
                if self._session is not None:
                    self._status = VoiceChatStatus.QUEUED
                    return
                self._status = VoiceChatStatus.LOADING
                self._load_error = None

            loop = asyncio.get_running_loop()
            try:

                def load_sync():
                    return _coerce_loaded_components(self._loader(self._model_name))

                loaded = await loop.run_in_executor(
                    get_mlx_executor(),
                    load_sync,
                )
            except asyncio.CancelledError:
                with self._state_lock:
                    self._status = VoiceChatStatus.UNLOADED
                raise
            except Exception as exc:
                message = _safe_error_message(exc)
                with self._state_lock:
                    self._status = VoiceChatStatus.FAILED
                    self._load_error = message
                logger.error(
                    "VoiceChat model load failed: model=%s error_type=%s",
                    self._model_name,
                    type(exc).__name__,
                )
                raise

            model, processor, session = loaded
            with self._state_lock:
                self._model = model
                self._processor = processor
                self._session = session
                self._status = VoiceChatStatus.QUEUED
            logger.info("VoiceChat model loaded: model=%s", self._model_name)

    async def stop(self) -> None:
        """Cancel active work, release the reservation, and unload the model."""
        async with self._lifecycle_lock:
            reservation = self._reservation
            if reservation is not None:
                wrapper = reservation.wrapper
                if wrapper is not None:
                    with suppress(Exception):
                        await asyncio.shield(wrapper.close())
                else:
                    operation = reservation.operation
                    if operation is not None and not operation.done():
                        with suppress(BaseException):
                            await asyncio.shield(operation)
                    with suppress(Exception):
                        await asyncio.shield(self._release_reservation(reservation))

            if self._cleanup_tasks:
                await asyncio.gather(
                    *tuple(self._cleanup_tasks), return_exceptions=True
                )

            with self._state_lock:
                self._model = None
                self._processor = None
                self._session = None
                self._reservation = None
                self._status = VoiceChatStatus.UNLOADED

            gc.collect()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                get_mlx_executor(), lambda: (mx.synchronize(), mx.clear_cache())
            )
            logger.info("VoiceChat model unloaded: model=%s", self._model_name)

    def _reserve(self, mode: str) -> _Reservation:
        """Reserve the checkpoint without yielding to the event loop."""
        with self._state_lock:
            if self._session is None:
                raise RuntimeError("VoiceChat engine is not started")
            if self._reservation is not None:
                raise VoiceChatBusyError()
            activity_id = self._begin_activity(
                "voicechat",
                detail=f"VoiceChat {mode} session",
                metadata={
                    "mode": mode,
                    "input_sample_rate": self.input_sample_rate,
                    "output_sample_rate": self.output_sample_rate,
                    "frame_duration_ms": self.frame_duration_ms,
                },
            )
            reservation = _Reservation(mode=mode, activity_id=activity_id)
            self._reservation = reservation
            self._status = VoiceChatStatus.QUEUED
            return reservation

    def _mark_active(self, reservation: _Reservation) -> None:
        with self._state_lock:
            if self._reservation is reservation and not reservation.released:
                self._status = VoiceChatStatus.ACTIVE

    def _schedule_cleanup(self, coroutine: Any) -> None:
        task = asyncio.create_task(coroutine, name="voicechat-cleanup")
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)

    async def _finish_cancelled_operation(
        self,
        reservation: _Reservation,
        operation: asyncio.Future[Any],
        *,
        assign_raw_session: bool = False,
    ) -> None:
        try:
            result = await asyncio.shield(operation)
        except BaseException:
            result = None
        if assign_raw_session and result is not None:
            reservation.raw_session = result
        if reservation.operation is operation:
            reservation.operation = None
        with suppress(Exception):
            await self._release_reservation(reservation)

    async def _release_reservation(
        self, reservation: _Reservation
    ) -> list[VoiceChatEvent]:
        async with reservation.close_lock:
            with self._state_lock:
                if reservation.released:
                    return []

            operation = reservation.operation
            if operation is not None and not operation.done():
                with suppress(BaseException):
                    await asyncio.shield(operation)
            if reservation.operation is operation:
                reservation.operation = None

            raw_events: Any = []
            raw_session = reservation.raw_session
            cancel = getattr(raw_session, "cancel", None)
            if callable(cancel) and getattr(raw_session, "closed", False) is not True:
                loop = asyncio.get_running_loop()
                try:
                    raw_events = await loop.run_in_executor(get_mlx_executor(), cancel)
                    if inspect.isawaitable(raw_events):
                        raw_events = await raw_events
                except Exception as exc:
                    raw_events = [_error_event(_safe_error_message(exc))]

            with self._state_lock:
                if self._reservation is not reservation:
                    return normalize_voicechat_events(raw_events)
                self._reservation = None
                reservation.released = True
                self._status = (
                    VoiceChatStatus.QUEUED
                    if self._session is not None
                    else VoiceChatStatus.UNLOADED
                )

            with suppress(Exception):
                await self._finish_activity(reservation.activity_id)
            return normalize_voicechat_events(raw_events)

    async def create_realtime_session(
        self,
        *,
        system_prompt: str | None = None,
        seed: int = 0,
        max_streaming_seconds: float | None = None,
        use_language_cache: bool = True,
        use_perception_cache: bool = True,
        profile: bool = False,
    ) -> VoiceChatRealtimeSession:
        """Atomically reserve the checkpoint before creating model state."""
        reservation = self._reserve("realtime")
        session = self._session
        assert session is not None
        loop = asyncio.get_running_loop()
        operation = loop.run_in_executor(
            get_mlx_executor(),
            partial(
                session.create_streaming_session,
                system_prompt=system_prompt,
                seed=seed,
                max_streaming_seconds=max_streaming_seconds,
                use_language_cache=use_language_cache,
                use_perception_cache=use_perception_cache,
                profile=profile,
            ),
        )
        reservation.operation = operation
        try:
            raw_session = await asyncio.shield(operation)
        except asyncio.CancelledError:
            self._schedule_cleanup(
                self._finish_cancelled_operation(
                    reservation, operation, assign_raw_session=True
                )
            )
            raise
        except Exception as exc:
            reservation.operation = None
            await asyncio.shield(self._release_reservation(reservation))
            raise VoiceChatInferenceError(
                f"VoiceChat session initialization failed: {_safe_error_message(exc)}"
            ) from exc
        finally:
            if operation.done() and reservation.operation is operation:
                reservation.operation = None

        try:
            reservation.raw_session = raw_session
            wrapper = VoiceChatRealtimeSession(self, reservation, raw_session)
        except Exception as exc:
            await asyncio.shield(self._release_reservation(reservation))
            raise VoiceChatInferenceError(
                f"VoiceChat session initialization failed: {_safe_error_message(exc)}"
            ) from exc
        reservation.wrapper = wrapper
        self._mark_active(reservation)
        return wrapper

    async def create_session(self, **kwargs: Any) -> VoiceChatRealtimeSession:
        """Alias matching the model-specific session terminology."""
        return await self.create_realtime_session(**kwargs)

    async def process(self, audio_path: str | Path, **kwargs: Any) -> bytes:
        """Run one offline model-specific turn and return mono PCM WAV bytes."""
        reservation = self._reserve("offline")
        self._mark_active(reservation)
        session = self._session
        assert session is not None
        loop = asyncio.get_running_loop()
        operation = loop.run_in_executor(
            get_mlx_executor(), partial(session.generate, str(audio_path), **kwargs)
        )
        reservation.operation = operation
        try:
            result = await asyncio.shield(operation)
            reservation.operation = None
            audio = _event_value(result, "audio")
            if not _has_audio_samples(audio):
                raise VoiceChatInferenceError(
                    "VoiceChat model produced no audio output"
                )
            sample_rate = int(
                _event_value(result, "sample_rate", self.output_sample_rate)
                or self.output_sample_rate
            )
            output = audio_to_wav_bytes(audio, sample_rate)
            self._update_activity(
                reservation.activity_id,
                output_bytes=len(output),
            )
            return output
        except asyncio.CancelledError:
            self._schedule_cleanup(
                self._finish_cancelled_operation(reservation, operation)
            )
            raise
        except VoiceChatInferenceError:
            raise
        except Exception as exc:
            raise VoiceChatInferenceError(
                f"VoiceChat inference failed: {_safe_error_message(exc)}"
            ) from exc
        finally:
            if operation.done() and reservation.operation is operation:
                reservation.operation = None
            if operation.done() and not reservation.released:
                await asyncio.shield(self._release_reservation(reservation))

    async def abort_all_requests(
        self,
        *,
        reason: str | None = None,
        error_code: str | None = None,
    ) -> int:
        """Cancel the active realtime/offline operation without unloading."""
        del reason, error_code
        reservation = self._reservation
        if reservation is None:
            return 0
        wrapper = reservation.wrapper
        if wrapper is not None:
            await wrapper.cancel()
        else:
            operation = reservation.operation
            if operation is not None and not operation.done():
                with suppress(BaseException):
                    await asyncio.shield(operation)
            await self._release_reservation(reservation)
        return 1

    def get_stats(self) -> dict[str, Any]:
        status = self.get_status()
        return {
            "engine_type": "voicechat",
            "model_name": self._model_name,
            "loaded": status["loaded"],
            "status": status["status"],
            "active_session": status["active_session"],
            "session_mode": status["session_mode"],
            "input_sample_rate": self.input_sample_rate,
            "output_sample_rate": self.output_sample_rate,
            "frame_duration_ms": self.frame_duration_ms,
            "frame_samples": self.frame_samples,
        }

    def get_cache_stats(self) -> None:
        return None

    def __repr__(self) -> str:
        return f"<VoiceChatEngine model={self._model_name} status={self.status.value}>"


class VoiceChatRealtimeSession:
    """Async façade for one reserved merged mlx-vlm streaming session."""

    def __init__(
        self,
        engine: VoiceChatEngine,
        reservation: _Reservation,
        raw_session: Any,
    ) -> None:
        self._engine = engine
        self._reservation = reservation
        self._raw_session = raw_session
        self._operation_lock = asyncio.Lock()
        self._operation: asyncio.Future[Any] | None = None
        self._close_task: asyncio.Task[list[VoiceChatEvent]] | None = None
        self._closed = False
        self._cancel_requested = False
        self._audio_emitted = False
        self.input_sample_rate = int(
            getattr(raw_session, "input_sample_rate", engine.input_sample_rate)
            or engine.input_sample_rate
        )
        self.output_sample_rate = int(
            getattr(raw_session, "output_sample_rate", engine.output_sample_rate)
            or engine.output_sample_rate
        )
        self.frame_samples = int(
            getattr(raw_session, "frame_samples", engine.frame_samples)
            or engine.frame_samples
        )

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def session_id(self) -> str:
        return self._reservation.reservation_id

    async def _call_model(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        if self._closed:
            raise RuntimeError("VoiceChat realtime session is closed")
        async with self._operation_lock:
            if self._closed:
                raise RuntimeError("VoiceChat realtime session is closed")
            method = getattr(self._raw_session, method_name)
            loop = asyncio.get_running_loop()
            operation = loop.run_in_executor(
                get_mlx_executor(), partial(method, *args, **kwargs)
            )
            self._operation = operation
            self._reservation.operation = operation
            try:
                return await asyncio.shield(operation)
            except asyncio.CancelledError:
                self._closed = True
                self._engine._schedule_cleanup(
                    self._finish_cancelled_operation(operation)
                )
                raise
            finally:
                if operation.done():
                    if self._operation is operation:
                        self._operation = None
                    if self._reservation.operation is operation:
                        self._reservation.operation = None

    async def _finish_cancelled_operation(self, operation: asyncio.Future[Any]) -> None:
        with suppress(BaseException):
            await asyncio.shield(operation)
        if self._operation is operation:
            self._operation = None
        if self._reservation.operation is operation:
            self._reservation.operation = None
        with suppress(Exception):
            await self.close()

    async def _events_for_call(
        self, method_name: str, *args: Any, **kwargs: Any
    ) -> list[VoiceChatEvent]:
        try:
            raw_events = await self._call_model(method_name, *args, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            events = [_error_event(_safe_error_message(exc), code="backend_error")]
            with suppress(Exception):
                await self.close()
            return events

        try:
            events = normalize_voicechat_events(raw_events)
        except Exception as exc:
            events = [_error_event(_safe_error_message(exc), code="protocol")]
            with suppress(Exception):
                await self.close()
            return events
        self._audio_emitted = self._audio_emitted or any(
            event.kind is VoiceChatEventKind.AUDIO for event in events
        )
        self._engine._update_activity(
            self._reservation.activity_id,
            events_emitted=len(events),
        )
        if any(
            event.kind
            in (
                VoiceChatEventKind.ERROR,
                VoiceChatEventKind.DONE,
                VoiceChatEventKind.CANCELLED,
            )
            for event in events
        ):
            with suppress(Exception):
                await self.close()
        return events

    async def push_audio(
        self, samples: Any, *, sample_rate: int | None = None
    ) -> list[VoiceChatEvent]:
        return await self._events_for_call(
            "push_audio", samples, sample_rate=sample_rate
        )

    async def feed(self, samples: Any, *, sample_rate: int | None = None):
        return await self.push_audio(samples, sample_rate=sample_rate)

    async def push(self, samples: Any, *, sample_rate: int | None = None):
        return await self.push_audio(samples, sample_rate=sample_rate)

    async def flush(self, *, pad_partial: bool = True) -> list[VoiceChatEvent]:
        events = await self._events_for_call("flush", pad_partial=pad_partial)
        if not self._audio_emitted and not any(
            event.kind is VoiceChatEventKind.ERROR for event in events
        ):
            events.insert(
                0,
                _error_event(
                    "VoiceChat model produced no audio output", code="no_audio"
                ),
            )
        with suppress(Exception):
            await self.close()
        return events

    async def cancel(self) -> list[VoiceChatEvent]:
        self._cancel_requested = True
        return await self.close()

    async def close(self) -> list[VoiceChatEvent]:
        self._cancel_requested = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close_impl(), name="voicechat-session-close"
            )
        return await asyncio.shield(self._close_task)

    async def _close_impl(self) -> list[VoiceChatEvent]:
        operation = self._operation or self._reservation.operation
        if operation is not None and not operation.done():
            with suppress(BaseException):
                await asyncio.shield(operation)
        if self._operation is operation:
            self._operation = None
        if self._reservation.operation is operation:
            self._reservation.operation = None

        raw_events = await self._engine._release_reservation(self._reservation)
        self._closed = True
        events = normalize_voicechat_events(raw_events)
        if not events and not self._reservation.released:
            return events
        if not events and self._cancel_requested:
            events = [VoiceChatEvent(kind=VoiceChatEventKind.CANCELLED)]
        return events

    async def release(self) -> list[VoiceChatEvent]:
        return await self.close()


VoiceChatSession = VoiceChatRealtimeSession

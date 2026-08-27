# SPDX-License-Identifier: Apache-2.0
"""OpenAI/NVIDIA-compatible WebSocket transport for VoiceChat models."""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import inspect
import json
import math
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from ..engine.voicechat import (
    VOICECHAT_FRAME_DURATION_MS,
    VOICECHAT_INPUT_SAMPLE_RATE,
    VOICECHAT_OUTPUT_SAMPLE_RATE,
    VoiceChatBusyError,
    VoiceChatEngine,
    VoiceChatEvent,
    VoiceChatEventKind,
    normalize_voicechat_events,
)
from ..exceptions import ModelBusyError
from .audio_routes import _verify_ws_api_key
from .realtime_models import (
    RealtimeAudioAppend,
    RealtimeAudioCommit,
    RealtimeConversationItemCreate,
    RealtimeResponseCreate,
    RealtimeSessionUpdate,
)
from .safe_messages import safe_message as _safe_message

realtime_router = APIRouter()

_AUTH_TIMEOUT_SECONDS = 10.0
_MAX_AUDIO_BYTES = 15 * 1024 * 1024
_MAX_AUDIO_BASE64_BYTES = 20 * 1024 * 1024
_MAX_INBOUND_TEXT_BYTES = 22 * 1024 * 1024
_MAX_TOOL_PAYLOAD_BYTES = 64 * 1024
_MAX_TOOL_JSON_DEPTH = 24
_MAX_TOOL_CONTAINER_ITEMS = 1024
_MAX_TOOL_STRING_BYTES = 16 * 1024
_MAX_QUEUED_MESSAGES = 32
_MAX_RESAMPLE_INPUT_SAMPLES = 4096
_BUSY_CLOSE_CODE = 1013
_AUTH_CLOSE_CODE = 1008
_SERVER_CLOSE_CODE = 1011
_SUPPORTED_SAMPLE_RATES = frozenset(
    {8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000}
)
_SUPPORTED_PCM_FORMATS = frozenset({"pcm16", "audio/pcm", "audio/x-pcm", "s16le"})
_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class _RealtimeProtocolError(ValueError):
    """A safe, structured protocol rejection."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_request",
        param: str | None = None,
        close_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.param = param
        self.close_code = close_code


@dataclass(frozen=True)
class _SessionSettings:
    model: str
    system_prompt: str | None
    seed: int
    max_streaming_seconds: float | None
    input_sample_rate: int
    output_sample_rate: int
    frame_duration_ms: int
    tools: list[dict[str, Any]]


@dataclass
class _FunctionCallState:
    call_id: str
    item_id: str
    response_id: str
    name: str
    arguments: str = ""


@dataclass
class _RealtimeRuntime:
    session: Any
    engine: Any
    settings: _SessionSettings
    session_id: str
    model_id: str
    native_input_sample_rate: int
    native_output_sample_rate: int
    frame_samples: int
    input_resampler: _StreamingLinearResampler
    received_audio: bool = False
    pending_native_samples: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    function_call: _FunctionCallState | None = None
    first_input_at: float | None = None
    input_audio_duration_seconds: float = 0.0
    output_audio_duration_seconds: float = 0.0
    first_transcript_latency_seconds: float | None = None
    first_text_latency_seconds: float | None = None
    first_audio_latency_seconds: float | None = None
    turn_latency_seconds: float | None = None
    turns: int = 0
    interruptions: int = 0
    cancellations: int = 0
    tool_requests: int = 0
    tool_failures: int = 0
    tool_parallel_rejections: int = 0
    terminal_state: str | None = None


@dataclass(frozen=True)
class _InboundMessage:
    payload: dict[str, Any] | None = None
    binary: bytes | None = None
    error: str | None = None
    error_code: str = "invalid_request"
    error_param: str | None = None


@dataclass
class _RealtimeControl:
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    cancel_payload: dict[str, Any] | None = None


class _StreamingLinearResampler:
    """Preserve sample continuity while converting arbitrary WS chunks."""

    def __init__(self, source_rate: int, target_rate: int) -> None:
        self.source_rate = source_rate
        self.target_rate = target_rate
        self._step = source_rate / target_rate
        self._buffer = np.zeros(0, dtype=np.float32)
        self._position = 0.0

    def push(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return samples
        if self.source_rate == self.target_rate:
            return samples

        if self._buffer.size:
            source = np.concatenate((self._buffer, samples))
        else:
            source = samples

        output: list[float] = []
        position = self._position
        limit = source.size - 1
        while position < limit:
            index = int(position)
            fraction = position - index
            output.append(
                float(source[index] * (1.0 - fraction) + source[index + 1] * fraction)
            )
            position += self._step

        keep = min(int(position), source.size)
        self._buffer = source[keep:].copy()
        self._position = position - keep
        return np.asarray(output, dtype=np.float32)

    def finish(self) -> np.ndarray:
        """Emit endpoint samples when a client commits the input buffer."""
        if self.source_rate == self.target_rate or self._buffer.size == 0:
            return np.zeros(0, dtype=np.float32)

        output: list[float] = []
        position = self._position
        while position < self._buffer.size:
            index = int(position)
            if index >= self._buffer.size - 1:
                output.append(float(self._buffer[-1]))
            else:
                fraction = position - index
                output.append(
                    float(
                        self._buffer[index] * (1.0 - fraction)
                        + self._buffer[index + 1] * fraction
                    )
                )
            position += self._step
        return np.asarray(output, dtype=np.float32)


def _new_event_id() -> str:
    return f"event_{uuid.uuid4().hex[:16]}"


def _new_session_id() -> str:
    return f"sess_{uuid.uuid4().hex[:16]}"


def _record_voicechat_metric(method: str, *args: Any, **kwargs: Any) -> None:
    """Keep metrics fail-soft so persistence never interrupts a session."""
    try:
        from ..server_metrics import get_server_metrics

        recorder = getattr(get_server_metrics(), method, None)
        if callable(recorder):
            recorder(*args, **kwargs)
    except Exception:
        return


def _sample_count(samples: Any) -> int:
    size = getattr(samples, "size", None)
    if size is not None:
        try:
            return max(0, int(size))
        except (TypeError, ValueError, OverflowError):
            pass
    try:
        return max(0, len(samples))
    except (TypeError, ValueError):
        return 0


def _terminal_state_from_events(events: Any) -> str | None:
    for event in normalize_voicechat_events(events):
        kind = event.type
        if kind == VoiceChatEventKind.ERROR.value:
            return "error"
        if kind == VoiceChatEventKind.BUSY.value:
            return "busy"
        if kind == VoiceChatEventKind.CANCELLED.value:
            return "cancelled"
        if kind == VoiceChatEventKind.DONE.value:
            return "completed"
    return None


def _set_runtime_terminal(runtime: _RealtimeRuntime, state: str) -> None:
    if runtime.terminal_state is None:
        runtime.terminal_state = state


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _parse_json_text(text: Any) -> dict[str, Any]:
    if not isinstance(text, str):
        raise _RealtimeProtocolError("message must be a JSON object")
    if len(text.encode("utf-8")) > _MAX_INBOUND_TEXT_BYTES:
        raise _RealtimeProtocolError(
            "message exceeds the maximum allowed size", param="message"
        )
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, RecursionError, MemoryError) as exc:
        raise _RealtimeProtocolError("message must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise _RealtimeProtocolError("message must be a JSON object")
    return payload


def _check_json_bounds(
    value: Any,
    path: str,
    *,
    depth: int = 0,
    max_depth: int = _MAX_TOOL_JSON_DEPTH,
    max_items: int = _MAX_TOOL_CONTAINER_ITEMS,
    max_string_bytes: int = _MAX_TOOL_STRING_BYTES,
) -> None:
    if depth > max_depth:
        raise _RealtimeProtocolError("nested JSON payload is too deep", param=path)
    if isinstance(value, str):
        if len(value.encode("utf-8")) > max_string_bytes:
            raise _RealtimeProtocolError("nested JSON string is too large", param=path)
        return
    if isinstance(value, dict):
        if len(value) > max_items:
            raise _RealtimeProtocolError(
                "nested JSON object has too many fields", param=path
            )
        for index, item in enumerate(value.values()):
            _check_json_bounds(
                item,
                f"{path}[{index}]",
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_string_bytes=max_string_bytes,
            )
        return
    if isinstance(value, list):
        if len(value) > max_items:
            raise _RealtimeProtocolError(
                "nested JSON array has too many items", param=path
            )
        for index, item in enumerate(value):
            _check_json_bounds(
                item,
                f"{path}[{index}]",
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_string_bytes=max_string_bytes,
            )


def _get_engine_pool():
    """Return the active engine pool without importing server at module load."""
    from omlx.server import _server_state

    pool = _server_state.engine_pool
    if pool is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return pool


def _resolve_model(model_id: str) -> str:
    """Resolve aliases using the same resolver as the other API routes."""
    from omlx.server import resolve_model_id

    return resolve_model_id(model_id) or model_id


def _header_api_key(websocket: WebSocket) -> str | None:
    authorization = websocket.headers.get("authorization")
    if authorization:
        scheme, separator, value = authorization.partition(" ")
        if scheme.lower() == "bearer" and separator:
            return value
    api_key = websocket.headers.get("x-api-key")
    return api_key if api_key is not None else None


def _verify_ws_auth(websocket: WebSocket, in_band_key: str | None = None) -> bool:
    header_key = _header_api_key(websocket)
    if header_key is not None:
        return _verify_ws_api_key(header_key)
    return _verify_ws_api_key(in_band_key)


def _strict_int(value: Any, param: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _RealtimeProtocolError(f"{param} must be an integer", param=param)
    return value


def _validate_event_id(payload: dict[str, Any], seen: set[str]) -> str | None:
    value = payload.get("event_id")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise _RealtimeProtocolError(
            "event_id must be a non-empty string of at most 512 characters",
            param="event_id",
        )
    if value in seen:
        raise _RealtimeProtocolError(
            "event_id must be unique within the session", param="event_id"
        )
    if len(seen) >= 4096:
        raise _RealtimeProtocolError(
            "too many client event IDs in one session", param="event_id"
        )
    seen.add(value)
    return value


def _format_details(value: Any, param: str) -> tuple[str, int | None, int]:
    if value is None:
        return "pcm16", None, 1
    if isinstance(value, str):
        format_name = value.lower()
        rate = None
        channels = 1
    elif isinstance(value, dict):
        raw_type = value.get("type", value.get("format", "pcm16"))
        if not isinstance(raw_type, str):
            raise _RealtimeProtocolError(
                f"{param}.type must be a string", param=f"{param}.type"
            )
        format_name = raw_type.lower()
        rate = value.get("rate", value.get("sample_rate"))
        channels = value.get("channels", 1)
    else:
        raise _RealtimeProtocolError(
            f"{param} must be pcm16 or a structured PCM format", param=param
        )

    if format_name not in _SUPPORTED_PCM_FORMATS:
        raise _RealtimeProtocolError(
            "only mono PCM16 audio is supported",
            param=param,
        )
    if rate is not None:
        rate = _strict_int(rate, f"{param}.sample_rate")
    channels = _strict_int(channels, f"{param}.channels")
    if channels != 1:
        raise _RealtimeProtocolError(
            "only mono PCM16 audio is supported", param=f"{param}.channels"
        )
    return "pcm16", rate, channels


def _session_audio_side(
    config: dict[str, Any], direction: str
) -> tuple[str, int | None, int]:
    legacy_key = f"{direction}_audio_format"
    nested_format: Any = None
    audio = config.get("audio")
    if audio is not None and not isinstance(audio, dict):
        raise _RealtimeProtocolError(
            "session.audio must be an object", param="session.audio"
        )
    if isinstance(audio, dict):
        side = audio.get("input" if direction == "input" else "output")
        if side is not None and not isinstance(side, dict):
            raise _RealtimeProtocolError(
                f"session.audio.{direction} must be an object",
                param=f"session.audio.{direction}",
            )
        if isinstance(side, dict):
            nested_format = side.get("format")
            if nested_format is None and "type" in side:
                nested_format = side

    value = nested_format if nested_format is not None else config.get(legacy_key)
    format_name, rate, channels = _format_details(value, f"session.{legacy_key}")
    direct_rate = config.get(f"{direction}_sample_rate")
    if direct_rate is not None:
        direct_rate = _strict_int(direct_rate, f"session.{direction}_sample_rate")
        if rate is not None and rate != direct_rate:
            raise _RealtimeProtocolError(
                f"session.{direction}_sample_rate conflicts with the audio format",
                param=f"session.{direction}_sample_rate",
            )
        rate = direct_rate
    return format_name, rate, channels


def _validate_tools(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise _RealtimeProtocolError(
            "session.tools must be an array", param="session.tools"
        )
    if len(value) > 128:
        raise _RealtimeProtocolError(
            "session.tools may contain at most 128 tools", param="session.tools"
        )

    validated: list[dict[str, Any]] = []
    for index, tool in enumerate(value):
        param = f"session.tools[{index}]"
        if not isinstance(tool, dict):
            raise _RealtimeProtocolError("tool must be an object", param=param)
        try:
            tool_size = len(
                json.dumps(tool, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            )
        except (TypeError, ValueError, RecursionError, MemoryError) as exc:
            raise _RealtimeProtocolError(
                "tool definition must be a bounded JSON object", param=param
            ) from exc
        if tool_size > _MAX_TOOL_PAYLOAD_BYTES:
            raise _RealtimeProtocolError(
                "tool definition exceeds the maximum allowed size", param=param
            )
        _check_json_bounds(tool, param)
        if tool.get("type", "function") != "function":
            raise _RealtimeProtocolError(
                "only function tools are supported", param=f"{param}.type"
            )
        function = tool.get("function", tool)
        if not isinstance(function, dict):
            raise _RealtimeProtocolError(
                "tool.function must be an object", param=f"{param}.function"
            )
        name = function.get("name")
        if not isinstance(name, str) or _TOOL_NAME_RE.fullmatch(name) is None:
            raise _RealtimeProtocolError(
                "tool function names must contain 1-64 letters, digits, '_' or '-'",
                param=f"{param}.name",
            )
        parameters = function.get("parameters", {})
        if not isinstance(parameters, dict):
            raise _RealtimeProtocolError(
                "tool parameters must be a JSON object",
                param=f"{param}.parameters",
            )
        for field_name in ("description",):
            if field_name in function and not isinstance(function[field_name], str):
                raise _RealtimeProtocolError(
                    f"tool {field_name} must be a string",
                    param=f"{param}.{field_name}",
                )
        strict = function.get("strict")
        if strict is not None and not isinstance(strict, bool):
            raise _RealtimeProtocolError(
                "tool strict must be a boolean", param=f"{param}.strict"
            )
        normalized: dict[str, Any] = {
            "type": "function",
            "name": name,
            "parameters": parameters,
        }
        if "description" in function:
            normalized["description"] = function["description"]
        if strict is not None:
            normalized["strict"] = strict
        validated.append(normalized)
    return validated


def _validate_modalities(config: dict[str, Any]) -> None:
    for field_name in ("modalities", "output_modalities"):
        value = config.get(field_name)
        if value is None:
            continue
        if not isinstance(value, list) or any(
            item not in {"audio", "text"} for item in value
        ):
            raise _RealtimeProtocolError(
                f"session.{field_name} may contain only audio or text",
                param=f"session.{field_name}",
            )


def _validate_tool_choice(value: Any) -> None:
    if value is None or (
        isinstance(value, str) and value in {"auto", "none", "required"}
    ):
        return
    if not isinstance(value, dict):
        raise _RealtimeProtocolError(
            "session.tool_choice must be auto, none, required, or an object",
            param="session.tool_choice",
        )
    function = value.get("function")
    name = value.get("name")
    if name is None and isinstance(function, dict):
        name = function.get("name")
    if value.get("type") not in {"function", "function_call"} or not isinstance(
        name, str
    ):
        raise _RealtimeProtocolError(
            "session.tool_choice function name is invalid",
            param="session.tool_choice",
        )


def _validate_session_settings(
    config: Any,
    *,
    model_hint: str | None,
) -> _SessionSettings:
    if not isinstance(config, dict):
        raise _RealtimeProtocolError(
            "session.update.session must be an object", param="session"
        )

    _validate_modalities(config)
    _validate_tool_choice(config.get("tool_choice"))

    session_type = config.get("type")
    if session_type is not None and session_type != "realtime":
        raise _RealtimeProtocolError(
            "only realtime VoiceChat sessions are supported", param="session.type"
        )

    model = config.get("model") or model_hint
    if not isinstance(model, str) or not model.strip():
        raise _RealtimeProtocolError(
            "session.update.session.model is required", param="session.model"
        )
    model = model.strip()
    if model_hint and config.get("model") and model != model_hint:
        raise _RealtimeProtocolError(
            "session model does not match the WebSocket model", param="session.model"
        )

    system_prompt = config.get("system_prompt")
    instructions = config.get("instructions")
    for name, value in (
        ("system_prompt", system_prompt),
        ("instructions", instructions),
    ):
        if value is not None and not isinstance(value, str):
            raise _RealtimeProtocolError(
                f"session.{name} must be a string or null", param=f"session.{name}"
            )
    if (
        system_prompt is not None
        and instructions is not None
        and system_prompt != instructions
    ):
        raise _RealtimeProtocolError(
            "system_prompt and instructions must match when both are provided",
            param="session.instructions",
        )
    prompt = system_prompt if system_prompt is not None else instructions
    if prompt is not None and len(prompt) > 32_768:
        raise _RealtimeProtocolError(
            "session instructions are too long", param="session.instructions"
        )

    seed = _strict_int(config.get("seed", 0), "session.seed")
    if seed < 0:
        raise _RealtimeProtocolError(
            "session.seed must be non-negative", param="session.seed"
        )

    max_streaming_seconds = config.get("max_streaming_seconds")
    if max_streaming_seconds is not None:
        if isinstance(max_streaming_seconds, bool) or not isinstance(
            max_streaming_seconds, (int, float)
        ):
            raise _RealtimeProtocolError(
                "max_streaming_seconds must be a number or null",
                param="session.max_streaming_seconds",
            )
        max_streaming_seconds = float(max_streaming_seconds)
        if not math.isfinite(max_streaming_seconds) or max_streaming_seconds <= 0:
            raise _RealtimeProtocolError(
                "max_streaming_seconds must be positive and finite",
                param="session.max_streaming_seconds",
            )

    _, input_rate, _ = _session_audio_side(config, "input")
    _, output_rate, _ = _session_audio_side(config, "output")
    input_rate = VOICECHAT_INPUT_SAMPLE_RATE if input_rate is None else input_rate
    output_rate = VOICECHAT_OUTPUT_SAMPLE_RATE if output_rate is None else output_rate
    if input_rate not in _SUPPORTED_SAMPLE_RATES:
        raise _RealtimeProtocolError(
            "unsupported input sample rate",
            param="session.input_audio_format.sample_rate",
        )
    if output_rate not in _SUPPORTED_SAMPLE_RATES:
        raise _RealtimeProtocolError(
            "unsupported output sample rate",
            param="session.output_audio_format.sample_rate",
        )

    frame_duration_ms = config.get("frame_duration_ms")
    audio = config.get("audio")
    if frame_duration_ms is None and isinstance(audio, dict):
        input_audio = audio.get("input")
        if isinstance(input_audio, dict):
            frame_duration_ms = input_audio.get("frame_duration_ms")
    if frame_duration_ms is None:
        frame_duration_ms = VOICECHAT_FRAME_DURATION_MS
    frame_duration_ms = _strict_int(frame_duration_ms, "session.frame_duration_ms")
    if frame_duration_ms != VOICECHAT_FRAME_DURATION_MS:
        raise _RealtimeProtocolError(
            "VoiceChat requires 80 ms audio frames", param="session.frame_duration_ms"
        )
    frame_samples = config.get("frame_samples")
    if frame_samples is not None and _strict_int(
        frame_samples, "session.frame_samples"
    ) != (VOICECHAT_INPUT_SAMPLE_RATE * VOICECHAT_FRAME_DURATION_MS // 1000):
        raise _RealtimeProtocolError(
            "VoiceChat requires 1,280-sample native frames",
            param="session.frame_samples",
        )

    parallel_tool_calls = config.get("parallel_tool_calls", False)
    if not isinstance(parallel_tool_calls, bool):
        raise _RealtimeProtocolError(
            "parallel_tool_calls must be a boolean", param="session.parallel_tool_calls"
        )
    if parallel_tool_calls:
        raise _RealtimeProtocolError(
            "parallel tool calls are not supported",
            code="unsupported_operation",
            param="session.parallel_tool_calls",
        )

    turn_detection = config.get("turn_detection")
    if turn_detection is not None:
        raise _RealtimeProtocolError(
            "server turn detection is not supported; commit audio explicitly",
            code="unsupported_operation",
            param="session.turn_detection",
        )
    if isinstance(audio, dict):
        input_audio = audio.get("input")
        if (
            isinstance(input_audio, dict)
            and input_audio.get("turn_detection") is not None
        ):
            raise _RealtimeProtocolError(
                "server turn detection is not supported; commit audio explicitly",
                code="unsupported_operation",
                param="session.audio.input.turn_detection",
            )

    return _SessionSettings(
        model=model,
        system_prompt=prompt,
        seed=seed,
        max_streaming_seconds=max_streaming_seconds,
        input_sample_rate=input_rate,
        output_sample_rate=output_rate,
        frame_duration_ms=frame_duration_ms,
        tools=_validate_tools(config.get("tools")),
    )


def _validate_audio_event(payload: dict[str, Any]) -> RealtimeAudioAppend:
    audio = payload.get("audio")
    if not isinstance(audio, str) or not audio:
        raise _RealtimeProtocolError(
            "input_audio_buffer.append.audio must be a base64 string",
            param="audio",
        )
    sample_rate = payload.get("sample_rate")
    if sample_rate is not None:
        _strict_int(sample_rate, "sample_rate")
    try:
        return RealtimeAudioAppend.model_validate(payload)
    except Exception as exc:
        raise _RealtimeProtocolError(
            "invalid input_audio_buffer.append event", param="input_audio_buffer.append"
        ) from exc


def _decode_pcm16(value: str) -> np.ndarray:
    if len(value) > _MAX_AUDIO_BASE64_BYTES:
        raise _RealtimeProtocolError(
            "audio chunk exceeds the maximum allowed size", param="audio"
        )
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise _RealtimeProtocolError(
            "audio must be valid base64 PCM16", param="audio"
        ) from exc
    if len(raw) > _MAX_AUDIO_BYTES:
        raise _RealtimeProtocolError(
            "audio chunk exceeds the maximum allowed size", param="audio"
        )
    if len(raw) == 0:
        raise _RealtimeProtocolError("audio chunk must not be empty", param="audio")
    if len(raw) % 2:
        raise _RealtimeProtocolError(
            "PCM16 audio must contain an even number of bytes", param="audio"
        )
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def _decode_binary_pcm16(raw: bytes) -> np.ndarray:
    if not raw:
        raise _RealtimeProtocolError("audio chunk must not be empty", param="audio")
    if len(raw) > _MAX_AUDIO_BYTES:
        raise _RealtimeProtocolError(
            "audio chunk exceeds the maximum allowed size", param="audio"
        )
    if len(raw) % 2:
        raise _RealtimeProtocolError(
            "PCM16 audio must contain an even number of bytes", param="audio"
        )
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def _audio_to_base64(samples: Any) -> str:
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    pcm = np.round(np.clip(values, -1.0, 1.0) * 32767.0).astype("<i2")
    return base64.b64encode(pcm.tobytes()).decode("ascii")


_pcm16_from_base64 = _decode_pcm16


def _audio_codes_to_list(codes: Any) -> list[Any] | None:
    if codes is None:
        return None
    try:
        return np.asarray(codes).reshape(-1).tolist()
    except (TypeError, ValueError):
        return None


def _serialize_model_event(
    event: VoiceChatEvent,
    *,
    output_sample_rate: int = VOICECHAT_OUTPUT_SAMPLE_RATE,
) -> dict[str, Any]:
    """Map one normalized engine event to the merged VoiceChat wire format."""
    raw_kind = getattr(event, "type", getattr(event, "kind", None))
    if hasattr(raw_kind, "value"):
        raw_kind = raw_kind.value
    kind = {
        "user_transcript_delta": VoiceChatEventKind.TRANSCRIPT.value,
        "assistant_text_delta": VoiceChatEventKind.ASSISTANT_TEXT.value,
        "function_delta": VoiceChatEventKind.FUNCTION.value,
    }.get(str(raw_kind), str(raw_kind))
    common = {"frame_index": event.frame_index}
    if kind == VoiceChatEventKind.ASSISTANT_TEXT.value:
        return {
            "type": "response.text.delta",
            **common,
            "token_id": event.token_id,
            "delta": event.delta,
            "text": event.text,
        }
    if kind == VoiceChatEventKind.FUNCTION.value:
        call_id = getattr(event, "call_id", None)
        function_name = getattr(event, "function_name", None)
        arguments = getattr(event, "arguments", None)
        return {
            "type": "response.function_call_arguments.delta",
            **common,
            "token_id": event.token_id,
            "call_id": call_id or "call_unknown",
            "item_id": "item_unknown",
            "name": function_name or "voicechat_function",
            "delta": event.delta,
            "arguments": arguments or event.text or event.delta or "",
        }
    if kind == VoiceChatEventKind.TRANSCRIPT.value:
        return {
            "type": "conversation.item.input_audio_transcription.delta",
            **common,
            "delta": event.delta,
            "transcript": event.text,
        }
    if kind == VoiceChatEventKind.AUDIO.value:
        samples = np.asarray(event.samples, dtype=np.float32).reshape(-1)
        source_rate = int(
            event.sample_rate
            if event.sample_rate is not None
            else VOICECHAT_OUTPUT_SAMPLE_RATE
        )
        if source_rate not in _SUPPORTED_SAMPLE_RATES:
            raise _RealtimeProtocolError(
                "unsupported output sample rate", param="sample_rate"
            )
        if source_rate != output_sample_rate and samples.size:
            output_length = max(
                1, round(samples.size * output_sample_rate / source_rate)
            )
            positions = (
                np.arange(output_length, dtype=np.float32)
                * source_rate
                / output_sample_rate
            )
            samples = np.interp(
                positions,
                np.arange(samples.size, dtype=np.float32),
                samples,
            ).astype(np.float32)
        payload: dict[str, Any] = {
            "type": "response.audio.delta",
            **common,
            "delta": _audio_to_base64(samples),
            "format": "pcm16",
            "sample_rate": output_sample_rate,
            "channels": 1,
        }
        audio_codes = _audio_codes_to_list(event.audio_codes)
        if audio_codes is not None:
            payload["audio_codes"] = audio_codes
        return payload
    if kind == VoiceChatEventKind.DONE.value:
        return {"type": "response.done", **common}
    if kind == VoiceChatEventKind.CANCELLED.value:
        return {"type": "response.cancelled", **common}
    raise _RealtimeProtocolError(
        "VoiceChat emitted an unsupported event", code="protocol"
    )


def _is_voicechat_engine(engine: Any, entry: Any = None) -> bool:
    if isinstance(engine, VoiceChatEngine):
        return True
    if getattr(engine, "engine_type", None) == "voicechat":
        return True
    if getattr(engine, "realtime_voicechat", False) is True:
        return True
    # Test doubles and alternate pool implementations can expose the same
    # narrow session interface without inheriting the concrete engine class.
    if not callable(getattr(engine, "create_realtime_session", None)):
        return False
    if _entry_is_voicechat(entry) is True or entry is None:
        return True
    return getattr(engine, "_spec_class", None) is VoiceChatEngine


def _entry_is_voicechat(entry: Any) -> bool | None:
    if entry is None:
        return None
    engine_type = getattr(entry, "engine_type", None)
    is_voicechat = getattr(entry, "is_voicechat", None)
    capability = getattr(entry, "realtime_voicechat", None)
    known = (
        isinstance(engine_type, str)
        or isinstance(is_voicechat, bool)
        or isinstance(capability, bool)
    )
    if not known:
        return None
    return engine_type == "voicechat" or is_voicechat is True or capability is True


def _session_create_kwargs(
    engine: Any,
    settings: _SessionSettings,
) -> dict[str, Any]:
    create = getattr(engine, "create_realtime_session", None)
    if getattr(engine, "_spec_class", None) is VoiceChatEngine:
        create = VoiceChatEngine.create_realtime_session
    try:
        parameters = inspect.signature(create).parameters
    except (TypeError, ValueError):
        parameters = {}
        accepts_arbitrary = True
    else:
        accepts_arbitrary = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )

    kwargs: dict[str, Any] = {}
    if "system_prompt" in parameters or accepts_arbitrary:
        kwargs["system_prompt"] = settings.system_prompt
    elif "instructions" in parameters:
        kwargs["instructions"] = settings.system_prompt
    elif settings.system_prompt is not None:
        raise _RealtimeProtocolError(
            "this VoiceChat engine cannot apply session instructions",
            code="unsupported_operation",
            param="session.instructions",
            close_code=_AUTH_CLOSE_CODE,
        )

    for name, value in (
        ("seed", settings.seed),
        ("max_streaming_seconds", settings.max_streaming_seconds),
    ):
        if name in parameters or accepts_arbitrary:
            kwargs[name] = value

    if settings.tools:
        if "tools" not in parameters and not accepts_arbitrary:
            raise _RealtimeProtocolError(
                "this VoiceChat engine cannot apply session tools",
                code="unsupported_operation",
                param="session.tools",
                close_code=_AUTH_CLOSE_CODE,
            )
        kwargs["tools"] = settings.tools
    return kwargs


async def _open_runtime(
    pool: Any,
    settings: _SessionSettings,
    session_id: str,
) -> _RealtimeRuntime:
    resolved_model = _resolve_model(settings.model)
    get_entry = getattr(pool, "get_entry", None)
    entry = get_entry(resolved_model) if callable(get_entry) else None
    if _entry_is_voicechat(entry) is False:
        raise _RealtimeProtocolError(
            "the selected model does not advertise realtime_voicechat",
            code="unsupported_model",
            param="session.model",
            close_code=_AUTH_CLOSE_CODE,
        )

    try:
        engine = await pool.get_engine(resolved_model)
    except VoiceChatBusyError:
        raise
    except ModelBusyError as exc:
        raise VoiceChatBusyError() from exc
    except Exception as exc:
        raise _RealtimeProtocolError(
            "VoiceChat model initialization failed",
            code="session_initialization_failed",
            close_code=_SERVER_CLOSE_CODE,
        ) from exc

    if not _is_voicechat_engine(engine, entry):
        raise _RealtimeProtocolError(
            "the selected model does not support realtime VoiceChat",
            code="unsupported_model",
            param="session.model",
            close_code=_AUTH_CLOSE_CODE,
        )

    try:
        session_kwargs = _session_create_kwargs(engine, settings)
        session = await _maybe_await(engine.create_realtime_session(**session_kwargs))
    except _RealtimeProtocolError:
        raise
    except VoiceChatBusyError:
        raise
    except Exception as exc:
        raise _RealtimeProtocolError(
            "VoiceChat session initialization failed",
            code="session_initialization_failed",
            close_code=_SERVER_CLOSE_CODE,
        ) from exc

    if session is None:
        raise _RealtimeProtocolError(
            "VoiceChat session initialization returned no session",
            code="session_initialization_failed",
            close_code=_SERVER_CLOSE_CODE,
        )

    session_input_rate = getattr(session, "input_sample_rate", None)
    if session_input_rate is None:
        session_input_rate = getattr(engine, "input_sample_rate", None)
    native_input_rate = int(
        VOICECHAT_INPUT_SAMPLE_RATE
        if session_input_rate is None
        else session_input_rate
    )
    session_output_rate = getattr(session, "output_sample_rate", None)
    if session_output_rate is None:
        session_output_rate = getattr(engine, "output_sample_rate", None)
    native_output_rate = int(
        VOICECHAT_OUTPUT_SAMPLE_RATE
        if session_output_rate is None
        else session_output_rate
    )
    raw_frame_samples = getattr(session, "frame_samples", None)
    frame_samples = int(
        native_input_rate * VOICECHAT_FRAME_DURATION_MS // 1000
        if raw_frame_samples is None
        else raw_frame_samples
    )
    if native_input_rate != VOICECHAT_INPUT_SAMPLE_RATE:
        with contextlib.suppress(Exception):
            await session.close()
        raise _RealtimeProtocolError(
            "VoiceChat model exposes an unsupported native input sample rate",
            code="unsupported_model",
            close_code=_SERVER_CLOSE_CODE,
        )
    if native_output_rate != VOICECHAT_OUTPUT_SAMPLE_RATE:
        with contextlib.suppress(Exception):
            await session.close()
        raise _RealtimeProtocolError(
            "VoiceChat model exposes an unsupported native output sample rate",
            code="unsupported_model",
            close_code=_SERVER_CLOSE_CODE,
        )
    if frame_samples != native_input_rate * VOICECHAT_FRAME_DURATION_MS // 1000:
        with contextlib.suppress(Exception):
            await session.close()
        raise _RealtimeProtocolError(
            "VoiceChat model exposes an unsupported frame size",
            code="unsupported_model",
            close_code=_SERVER_CLOSE_CODE,
        )

    return _RealtimeRuntime(
        session=session,
        engine=engine,
        settings=settings,
        session_id=session_id,
        model_id=resolved_model,
        native_input_sample_rate=native_input_rate,
        native_output_sample_rate=native_output_rate,
        frame_samples=frame_samples,
        input_resampler=_StreamingLinearResampler(
            settings.input_sample_rate, native_input_rate
        ),
    )


async def _receive_payload(websocket: WebSocket) -> dict[str, Any]:
    try:
        message = await asyncio.wait_for(
            websocket.receive(), timeout=_AUTH_TIMEOUT_SECONDS
        )
    except TimeoutError as exc:
        raise _RealtimeProtocolError("handshake timeout", code="timeout") from exc
    if message.get("type") == "websocket.disconnect":
        raise WebSocketDisconnect(message.get("code", 1000))
    return _parse_json_text(message.get("text"))


async def _read_messages(
    websocket: WebSocket,
    queue: asyncio.Queue[_InboundMessage],
    control: _RealtimeControl | None = None,
) -> None:
    def enqueue(message: _InboundMessage) -> bool:
        try:
            queue.put_nowait(message)
        except asyncio.QueueFull:
            return False
        return True

    def enqueue_error(error: _RealtimeProtocolError) -> bool:
        return enqueue(
            _InboundMessage(
                error=str(error),
                error_code=error.code,
                error_param=error.param,
            )
        )

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return
            data = message.get("bytes")
            if data is not None:
                if not enqueue(_InboundMessage(binary=data)):
                    return
                continue
            text = message.get("text")
            if not text:
                if not enqueue(_InboundMessage(error="message must be a JSON object")):
                    return
                continue
            try:
                payload = _parse_json_text(text)
            except _RealtimeProtocolError as exc:
                if not enqueue_error(exc):
                    return
                continue
            if control is not None and payload.get("type") in {
                "session.cancel",
                "response.cancel",
            }:
                if control.cancel_payload is None:
                    control.cancel_payload = payload
                control.cancel_event.set()
                continue
            if not enqueue(_InboundMessage(payload=payload)):
                return
    except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
        return


async def _send_error(
    websocket: WebSocket,
    message: str,
    *,
    code: str = "invalid_request",
    param: str | None = None,
    client_event_id: str | None = None,
) -> None:
    error: dict[str, Any] = {"code": code, "message": _safe_message(message)}
    if param:
        error["param"] = param
    if client_event_id:
        error["event_id"] = client_event_id
    await websocket.send_json(
        {"event_id": _new_event_id(), "type": "error", "error": error}
    )


async def _send_event(websocket: WebSocket, payload: dict[str, Any]) -> None:
    await websocket.send_json({"event_id": _new_event_id(), **payload})


def _session_view(
    session_id: str,
    *,
    state: str,
    model: str | None,
    input_sample_rate: int,
    output_sample_rate: int,
    frame_samples: int,
    settings: _SessionSettings | None = None,
) -> dict[str, Any]:
    session: dict[str, Any] = {
        "id": session_id,
        "state": state,
        "input_audio_format": {
            "type": "pcm16",
            "sample_rate": input_sample_rate,
        },
        "output_audio_format": {
            "type": "pcm16",
            "sample_rate": output_sample_rate,
        },
        "frame_samples": frame_samples,
    }
    if model is not None:
        session["model"] = model
    if settings is not None:
        session.update(
            {
                "system_prompt": settings.system_prompt,
                "instructions": settings.system_prompt,
                "seed": settings.seed,
                "max_streaming_seconds": settings.max_streaming_seconds,
                "tools": settings.tools,
            }
        )
    return session


def _function_name(event: VoiceChatEvent, tools: list[dict[str, Any]]) -> str:
    name = getattr(event, "function_name", None)
    if isinstance(name, str) and _TOOL_NAME_RE.fullmatch(name):
        return name
    tool_names = [
        tool["name"]
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("name"), str)
    ]
    if len(tool_names) == 1:
        return tool_names[0]
    return "voicechat_function"


def _function_call_id(event: VoiceChatEvent) -> str | None:
    call_id = getattr(event, "call_id", None)
    if call_id is None:
        return None
    if not isinstance(call_id, str) or not call_id or len(call_id) > 128:
        raise _RealtimeProtocolError(
            "VoiceChat emitted an invalid function call ID",
            code="protocol",
            param="call_id",
        )
    return call_id


def _function_delta_payload(
    runtime: _RealtimeRuntime,
    event: VoiceChatEvent,
) -> dict[str, Any]:
    if getattr(event, "parallel", False):
        raise _RealtimeProtocolError(
            "parallel tool calls are not supported",
            code="unsupported_operation",
            param="function_call",
            close_code=_AUTH_CLOSE_CODE,
        )

    event_call_id = _function_call_id(event)
    event_name = getattr(event, "function_name", None)
    if event_name is not None and (
        not isinstance(event_name, str) or _TOOL_NAME_RE.fullmatch(event_name) is None
    ):
        event_name = None
    state = runtime.function_call
    if state is not None and (
        (event_call_id is not None and event_call_id != state.call_id)
        or (event_name is not None and event_name != state.name)
    ):
        raise _RealtimeProtocolError(
            "parallel tool calls are not supported",
            code="unsupported_operation",
            param="function_call",
            close_code=_AUTH_CLOSE_CODE,
        )
    if state is None:
        state = _FunctionCallState(
            call_id=event_call_id or f"call_{uuid.uuid4().hex[:16]}",
            item_id=f"item_{uuid.uuid4().hex[:16]}",
            response_id=f"resp_{uuid.uuid4().hex[:16]}",
            name=event_name or _function_name(event, runtime.settings.tools),
        )
        runtime.function_call = state

    delta = event.delta or ""
    full_arguments = getattr(event, "arguments", None)
    if full_arguments is not None:
        state.arguments = str(full_arguments)
    elif event.text is not None:
        state.arguments = str(event.text)
    else:
        state.arguments += delta
    return {
        "type": "response.function_call_arguments.delta",
        "frame_index": event.frame_index,
        "token_id": event.token_id,
        "response_id": state.response_id,
        "item_id": state.item_id,
        "output_index": 0,
        "call_id": state.call_id,
        "name": state.name,
        "delta": delta,
        "arguments": state.arguments,
    }


async def _send_function_done(
    websocket: WebSocket,
    runtime: _RealtimeRuntime,
    event: VoiceChatEvent,
) -> None:
    state = runtime.function_call
    if state is None:
        return
    await _send_event(
        websocket,
        {
            "type": "response.function_call_arguments.done",
            "frame_index": event.frame_index,
            "response_id": state.response_id,
            "item_id": state.item_id,
            "output_index": 0,
            "call_id": state.call_id,
            "name": state.name,
            "arguments": state.arguments,
        },
    )
    runtime.function_call = None


async def _send_model_events(
    websocket: WebSocket,
    events: Any,
    *,
    output_sample_rate: int,
    runtime: _RealtimeRuntime | None = None,
) -> bool:
    terminal = False
    for event in normalize_voicechat_events(events):
        kind = {
            "user_transcript_delta": VoiceChatEventKind.TRANSCRIPT.value,
            "assistant_text_delta": VoiceChatEventKind.ASSISTANT_TEXT.value,
            "function_delta": VoiceChatEventKind.FUNCTION.value,
        }.get(event.type, event.type)
        if runtime is not None and runtime.first_input_at is not None:
            elapsed = max(0.0, time.perf_counter() - runtime.first_input_at)
            if kind == VoiceChatEventKind.TRANSCRIPT.value and (
                event.delta or event.text
            ):
                if runtime.first_transcript_latency_seconds is None:
                    runtime.first_transcript_latency_seconds = elapsed
            elif kind == VoiceChatEventKind.ASSISTANT_TEXT.value and (
                event.delta or event.text
            ):
                if runtime.first_text_latency_seconds is None:
                    runtime.first_text_latency_seconds = elapsed
        if runtime is not None and kind == VoiceChatEventKind.AUDIO.value:
            count = _sample_count(event.samples)
            rate = event.sample_rate or runtime.native_output_sample_rate
            if count and rate > 0:
                runtime.output_audio_duration_seconds += count / rate
                if (
                    runtime.first_input_at is not None
                    and runtime.first_audio_latency_seconds is None
                ):
                    runtime.first_audio_latency_seconds = max(
                        0.0, time.perf_counter() - runtime.first_input_at
                    )
        if kind == VoiceChatEventKind.ERROR.value:
            if runtime is not None:
                _set_runtime_terminal(runtime, "error")
            await _send_error(
                websocket,
                event.error or "VoiceChat inference failed",
                code=event.code or "inference_error",
            )
            terminal = True
            continue
        if kind == VoiceChatEventKind.BUSY.value:
            if runtime is not None:
                _set_runtime_terminal(runtime, "busy")
            await _send_error(
                websocket,
                event.error or "another realtime VoiceChat session is active",
                code="server_busy",
            )
            terminal = True
            continue
        if kind == VoiceChatEventKind.FUNCTION.value:
            if runtime is None:
                raise _RealtimeProtocolError(
                    "function events require an active VoiceChat runtime",
                    code="protocol",
                )
            was_new_call = runtime.function_call is None
            if event.parallel:
                runtime.tool_parallel_rejections += 1
            try:
                function_payload = _function_delta_payload(runtime, event)
            except _RealtimeProtocolError:
                if not event.parallel:
                    runtime.tool_failures += 1
                raise
            if was_new_call and not event.parallel:
                runtime.tool_requests += 1
            await _send_event(
                websocket,
                function_payload,
            )
            continue
        if kind == VoiceChatEventKind.DONE.value and runtime is not None:
            await _send_function_done(websocket, runtime, event)
        await _send_event(
            websocket,
            _serialize_model_event(
                event,
                output_sample_rate=output_sample_rate,
            ),
        )
        if kind in {
            VoiceChatEventKind.DONE.value,
            VoiceChatEventKind.CANCELLED.value,
        }:
            if runtime is not None:
                _set_runtime_terminal(
                    runtime,
                    "completed"
                    if kind == VoiceChatEventKind.DONE.value
                    else "cancelled",
                )
            if kind == VoiceChatEventKind.CANCELLED.value and runtime is not None:
                runtime.function_call = None
            terminal = True
    return terminal


async def _push_native_samples(
    websocket: WebSocket,
    runtime: _RealtimeRuntime,
    native_samples: np.ndarray,
) -> bool:
    if native_samples.size == 0:
        return False
    runtime.received_audio = True
    for start in range(0, native_samples.size, runtime.frame_samples):
        frame = native_samples[start : start + runtime.frame_samples]
        events = await _maybe_await(
            runtime.session.push_audio(
                frame,
                sample_rate=runtime.native_input_sample_rate,
            )
        )
        if await _send_model_events(
            websocket,
            events,
            output_sample_rate=runtime.settings.output_sample_rate,
            runtime=runtime,
        ):
            return True
        await asyncio.sleep(0)
    return False


async def _push_samples(
    websocket: WebSocket,
    runtime: _RealtimeRuntime,
    samples: np.ndarray,
) -> bool:
    runtime.received_audio = runtime.received_audio or samples.size > 0
    if samples.size:
        if runtime.first_input_at is None:
            runtime.first_input_at = time.perf_counter()
            runtime.turns += 1
        runtime.input_audio_duration_seconds += (
            samples.size / runtime.settings.input_sample_rate
        )
    for start in range(0, samples.size, _MAX_RESAMPLE_INPUT_SAMPLES):
        native_samples = runtime.input_resampler.push(
            samples[start : start + _MAX_RESAMPLE_INPUT_SAMPLES]
        )
        if native_samples.size:
            if runtime.pending_native_samples.size:
                runtime.pending_native_samples = np.concatenate(
                    (runtime.pending_native_samples, native_samples)
                )
            else:
                runtime.pending_native_samples = native_samples
        complete_samples = (
            runtime.pending_native_samples.size // runtime.frame_samples
        ) * runtime.frame_samples
        if complete_samples:
            complete = runtime.pending_native_samples[:complete_samples]
            runtime.pending_native_samples = runtime.pending_native_samples[
                complete_samples:
            ]
            if await _push_native_samples(websocket, runtime, complete):
                return True
        await asyncio.sleep(0)
    return False


def _take_pending_native_samples(
    runtime: _RealtimeRuntime,
    *,
    pad_partial: bool,
) -> np.ndarray:
    pending = runtime.pending_native_samples
    runtime.pending_native_samples = np.zeros(0, dtype=np.float32)
    if not pad_partial:
        runtime.input_resampler.finish()
        return np.zeros(0, dtype=np.float32)
    tail = runtime.input_resampler.finish()
    if pending.size and tail.size:
        return np.concatenate((pending, tail))
    return pending if pending.size else tail


async def _handle_client_message(
    websocket: WebSocket,
    runtime: _RealtimeRuntime,
    inbound: _InboundMessage,
    seen_event_ids: set[str],
) -> bool:
    if inbound.error:
        await _send_error(
            websocket,
            inbound.error,
            code=inbound.error_code,
            param=inbound.error_param,
        )
        return False
    if inbound.binary is not None:
        try:
            samples = _decode_binary_pcm16(inbound.binary)
        except _RealtimeProtocolError as exc:
            await _send_error(
                websocket,
                str(exc),
                code=exc.code,
                param=exc.param,
            )
            return False
        return await _push_samples(websocket, runtime, samples)

    payload = inbound.payload or {}
    client_event_id = None
    try:
        client_event_id = _validate_event_id(payload, seen_event_ids)
    except _RealtimeProtocolError as exc:
        await _send_error(websocket, str(exc), code=exc.code, param=exc.param)
        return False

    message_type = payload.get("type")
    if message_type == "session.update":
        await _send_error(
            websocket,
            "the active session cannot be reconfigured",
            param="type",
            client_event_id=client_event_id,
        )
        return False
    if message_type == "session.ping":
        await _send_event(websocket, {"type": "session.pong"})
        return False
    if message_type in {"session.cancel", "response.cancel"}:
        _set_runtime_terminal(runtime, "cancelled")
        runtime.cancellations += 1
        if message_type == "response.cancel":
            runtime.interruptions += 1
        events = await _maybe_await(runtime.session.cancel())
        terminal = await _send_model_events(
            websocket,
            events,
            output_sample_rate=runtime.settings.output_sample_rate,
            runtime=runtime,
        )
        if not terminal:
            await _send_event(
                websocket, {"type": "response.cancelled", "frame_index": None}
            )
        return True
    if message_type == "input_audio_buffer.append":
        try:
            append = _validate_audio_event(payload)
            sample_rate = (
                append.sample_rate
                if append.sample_rate is not None
                else runtime.settings.input_sample_rate
            )
            if sample_rate != runtime.settings.input_sample_rate:
                raise _RealtimeProtocolError(
                    "audio sample rate does not match the negotiated session rate",
                    param="sample_rate",
                )
            samples = _decode_pcm16(append.audio)
        except _RealtimeProtocolError as exc:
            await _send_error(
                websocket,
                str(exc),
                code=exc.code,
                param=exc.param,
                client_event_id=client_event_id,
            )
            return False
        return await _push_samples(websocket, runtime, samples)
    if message_type == "input_audio_buffer.commit":
        try:
            if "pad_partial" in payload and not isinstance(
                payload["pad_partial"], bool
            ):
                raise _RealtimeProtocolError(
                    "pad_partial must be a boolean", param="pad_partial"
                )
            commit = RealtimeAudioCommit.model_validate(payload)
        except _RealtimeProtocolError as exc:
            await _send_error(
                websocket,
                str(exc),
                code=exc.code,
                param=exc.param,
                client_event_id=client_event_id,
            )
            return False
        except Exception:
            await _send_error(
                websocket,
                "invalid input_audio_buffer.commit event",
                param="input_audio_buffer.commit",
                client_event_id=client_event_id,
            )
            return False
        if not runtime.received_audio:
            await _send_error(
                websocket,
                "cannot commit an empty input audio buffer",
                param="input_audio_buffer",
                client_event_id=client_event_id,
            )
            return False
        if runtime.first_input_at is not None:
            runtime.turn_latency_seconds = max(
                0.0, time.perf_counter() - runtime.first_input_at
            )
        await _send_event(websocket, {"type": "input_audio_buffer.committed"})
        if await _push_native_samples(
            websocket,
            runtime,
            _take_pending_native_samples(runtime, pad_partial=commit.pad_partial),
        ):
            return True
        events = await _maybe_await(
            runtime.session.flush(pad_partial=commit.pad_partial)
        )
        terminal = await _send_model_events(
            websocket,
            events,
            output_sample_rate=runtime.settings.output_sample_rate,
            runtime=runtime,
        )
        if not terminal and runtime.terminal_state is None:
            _set_runtime_terminal(runtime, "completed")
        return True
    if message_type == "response.create":
        try:
            response = RealtimeResponseCreate.model_validate(payload)
            response_config = response.response or {}
            if response_config.get("parallel_tool_calls"):
                runtime.tool_parallel_rejections += 1
                raise _RealtimeProtocolError(
                    "parallel tool calls are not supported",
                    code="unsupported_operation",
                    param="response.parallel_tool_calls",
                )
            if "tools" in response_config:
                _validate_tools(response_config["tools"])
            output_modalities = response_config.get("output_modalities")
            if output_modalities is not None and (
                not isinstance(output_modalities, list)
                or any(item not in {"audio", "text"} for item in output_modalities)
            ):
                raise _RealtimeProtocolError(
                    "response.output_modalities may contain only audio or text",
                    param="response.output_modalities",
                )
        except _RealtimeProtocolError as exc:
            await _send_error(
                websocket,
                str(exc),
                code=exc.code,
                param=exc.param,
                client_event_id=client_event_id,
            )
            return False
        except Exception:
            await _send_error(
                websocket,
                "invalid response.create event",
                code="invalid_request",
                param="response.create",
                client_event_id=client_event_id,
            )
            return False
        await _send_error(
            websocket,
            "response.create is not supported; commit input audio instead",
            code="unsupported_operation",
            param="type",
            client_event_id=client_event_id,
        )
        return False
    if message_type == "conversation.item.create":
        try:
            item_event = RealtimeConversationItemCreate.model_validate(payload)
            item = item_event.item
            if item.get("type") == "function_call_output" or "output" in item:
                runtime.tool_failures += 1
                raise _RealtimeProtocolError(
                    "tool results are not supported by this VoiceChat backend",
                    code="unsupported_operation",
                    param="item",
                )
            raise _RealtimeProtocolError(
                "conversation item input is not supported by this VoiceChat backend",
                code="unsupported_operation",
                param="item",
            )
        except _RealtimeProtocolError as exc:
            await _send_error(
                websocket,
                str(exc),
                code=exc.code,
                param=exc.param,
                client_event_id=client_event_id,
            )
        except Exception:
            await _send_error(
                websocket,
                "invalid conversation.item.create event",
                client_event_id=client_event_id,
            )
        return False
    if message_type in {"input_audio_buffer.clear", "session.close"}:
        await _send_error(
            websocket,
            "the requested realtime operation is not supported",
            code="unsupported_operation",
            param="type",
            client_event_id=client_event_id,
        )
        return False
    if message_type == "stop":
        if not runtime.received_audio:
            await _send_error(websocket, "cannot commit an empty input audio buffer")
            return False
        if runtime.first_input_at is not None:
            runtime.turn_latency_seconds = max(
                0.0, time.perf_counter() - runtime.first_input_at
            )
        await _send_event(websocket, {"type": "input_audio_buffer.committed"})
        if await _push_native_samples(
            websocket,
            runtime,
            _take_pending_native_samples(runtime, pad_partial=True),
        ):
            return True
        events = await _maybe_await(runtime.session.flush(pad_partial=True))
        terminal = await _send_model_events(
            websocket,
            events,
            output_sample_rate=runtime.settings.output_sample_rate,
            runtime=runtime,
        )
        if not terminal and runtime.terminal_state is None:
            _set_runtime_terminal(runtime, "completed")
        return True

    await _send_error(
        websocket,
        f"unsupported event type {message_type!r}",
        client_event_id=client_event_id,
    )
    return False


async def _close_runtime(runtime: _RealtimeRuntime | None) -> None:
    if runtime is None:
        return
    close = getattr(runtime.session, "close", None)
    if not callable(close):
        close = getattr(runtime.session, "release", None)
    if callable(close):
        with contextlib.suppress(BaseException):
            result = close()
            if inspect.isawaitable(result):
                await asyncio.shield(result)


def _record_realtime_terminal(
    runtime: _RealtimeRuntime | None,
    terminal_state: str,
    model_id: str = "",
) -> None:
    if runtime is None:
        _record_voicechat_metric(
            "record_voicechat_session_terminal",
            terminal_state,
            model_id=model_id,
        )
        return
    turn_latency = runtime.turn_latency_seconds
    if turn_latency is None and runtime.first_input_at is not None:
        turn_latency = max(0.0, time.perf_counter() - runtime.first_input_at)
    _record_voicechat_metric(
        "record_voicechat_session_terminal",
        runtime.terminal_state or terminal_state,
        model_id=runtime.model_id,
        turns=runtime.turns,
        interruptions=runtime.interruptions,
        cancellations=runtime.cancellations,
        audio_duration_seconds=runtime.input_audio_duration_seconds,
        output_audio_duration_seconds=runtime.output_audio_duration_seconds,
        first_transcript_latency_seconds=runtime.first_transcript_latency_seconds,
        first_text_latency_seconds=runtime.first_text_latency_seconds,
        first_audio_latency_seconds=runtime.first_audio_latency_seconds,
        turn_latency_seconds=turn_latency,
        tool_requests=runtime.tool_requests,
        tool_failures=runtime.tool_failures,
        tool_parallel_rejections=runtime.tool_parallel_rejections,
    )


async def _send_session_created(
    websocket: WebSocket,
    session_id: str,
    model: str | None,
) -> None:
    await _send_event(
        websocket,
        {
            "type": "session.created",
            "session": _session_view(
                session_id,
                state="configuring",
                model=model,
                input_sample_rate=VOICECHAT_INPUT_SAMPLE_RATE,
                output_sample_rate=VOICECHAT_OUTPUT_SAMPLE_RATE,
                frame_samples=VOICECHAT_INPUT_SAMPLE_RATE
                * VOICECHAT_FRAME_DURATION_MS
                // 1000,
            ),
        },
    )


@realtime_router.websocket("/v1/realtime")
async def realtime_voicechat(websocket: WebSocket) -> None:
    """Serve one full-duplex VoiceChat session over a JSON WebSocket."""
    await websocket.accept()
    session_id = _new_session_id()
    runtime: _RealtimeRuntime | None = None
    reader_task: asyncio.Task[None] | None = None
    startup_task: asyncio.Task[_RealtimeRuntime] | None = None
    processor_task: asyncio.Task[None] | None = None
    seen_event_ids: set[str] = set()
    control = _RealtimeControl()
    socket_closed = False
    terminal_state: str | None = None
    metric_model_id = ""

    async def close_socket(code: int = 1000) -> None:
        nonlocal socket_closed
        if socket_closed:
            return
        socket_closed = True
        with contextlib.suppress(Exception):
            await websocket.close(code=code)

    try:
        first: dict[str, Any] | None = None
        query_model = websocket.query_params.get("model") or None
        start_model = query_model
        metric_model_id = start_model or ""
        header_key = _header_api_key(websocket)
        if header_key is not None:
            if not _verify_ws_auth(websocket):
                terminal_state = "error"
                await _send_error(
                    websocket,
                    "invalid API key",
                    code="invalid_api_key",
                )
                await close_socket(_AUTH_CLOSE_CODE)
                return
        elif not _verify_ws_api_key(None):
            try:
                first = await _receive_payload(websocket)
            except WebSocketDisconnect:
                terminal_state = "disconnected"
                return
            except _RealtimeProtocolError:
                terminal_state = "error"
                await _send_error(
                    websocket,
                    "invalid API key",
                    code="invalid_api_key",
                )
                await close_socket(_AUTH_CLOSE_CODE)
                return
            if isinstance(first.get("model"), str):
                metric_model_id = first["model"].strip()
            if not _verify_ws_auth(websocket, first.get("api_key")):
                terminal_state = "error"
                await _send_error(
                    websocket,
                    "invalid API key",
                    code="invalid_api_key",
                    client_event_id=(
                        first.get("event_id")
                        if isinstance(first.get("event_id"), str)
                        else None
                    ),
                )
                await close_socket(_AUTH_CLOSE_CODE)
                return

        await _send_session_created(websocket, session_id, start_model)

        if first is not None and first.get("type") == "start":
            if isinstance(first.get("model"), str) and first["model"].strip():
                start_model = first["model"].strip()
                metric_model_id = start_model
            first = None

        while True:
            if first is None:
                try:
                    first = await _receive_payload(websocket)
                except WebSocketDisconnect:
                    terminal_state = "disconnected"
                    return
                except _RealtimeProtocolError as exc:
                    await _send_error(
                        websocket, str(exc), code=exc.code, param=exc.param
                    )
                    if exc.code == "timeout":
                        terminal_state = "error"
                        await close_socket(_AUTH_CLOSE_CODE)
                        return
                    continue

            try:
                client_event_id = _validate_event_id(first, seen_event_ids)
            except _RealtimeProtocolError as exc:
                await _send_error(websocket, str(exc), code=exc.code, param=exc.param)
                first = None
                continue
            if first.get("type") != "session.update":
                await _send_error(
                    websocket,
                    "send session.update before audio",
                    param="type",
                    client_event_id=client_event_id,
                )
                first = None
                continue
            try:
                update = RealtimeSessionUpdate.model_validate(first)
                settings = _validate_session_settings(
                    first.get("session"), model_hint=start_model
                )
            except _RealtimeProtocolError as exc:
                await _send_error(
                    websocket,
                    str(exc),
                    code=exc.code,
                    param=exc.param,
                    client_event_id=client_event_id,
                )
                first = None
                continue
            except Exception:
                await _send_error(
                    websocket,
                    "invalid session.update event",
                    param="session.update",
                    client_event_id=client_event_id,
                )
                first = None
                continue
            del update
            metric_model_id = settings.model
            with contextlib.suppress(Exception):
                metric_model_id = _resolve_model(settings.model)
            break

        queue: asyncio.Queue[_InboundMessage] = asyncio.Queue(
            maxsize=_MAX_QUEUED_MESSAGES
        )
        reader_task = asyncio.create_task(
            _read_messages(websocket, queue, control), name="voicechat-realtime-reader"
        )
        try:
            pool = _get_engine_pool()
        except HTTPException as exc:
            terminal_state = "error"
            await _send_error(
                websocket,
                str(exc.detail),
                code="server_unavailable",
            )
            await close_socket(_SERVER_CLOSE_CODE)
            return
        startup_task = asyncio.create_task(
            _open_runtime(pool, settings, session_id),
            name="voicechat-realtime-startup",
        )
        done, _ = await asyncio.wait(
            {reader_task, startup_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if reader_task in done and not startup_task.done():
            terminal_state = "disconnected"
            startup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await startup_task
            return
        try:
            runtime = startup_task.result()
        except VoiceChatBusyError:
            terminal_state = "busy"
            await _send_error(
                websocket,
                "another realtime VoiceChat session is already active",
                code="server_busy",
            )
            await close_socket(_BUSY_CLOSE_CODE)
            return
        except _RealtimeProtocolError as exc:
            terminal_state = "error"
            await _send_error(
                websocket,
                str(exc),
                code=exc.code,
                param=exc.param,
            )
            await close_socket(exc.close_code or _SERVER_CLOSE_CODE)
            return
        except asyncio.CancelledError:
            terminal_state = "cancelled"
            return
        except Exception:
            terminal_state = "error"
            await _send_error(
                websocket,
                "VoiceChat session initialization failed",
                code="session_initialization_failed",
            )
            await close_socket(_SERVER_CLOSE_CODE)
            return

        _record_voicechat_metric(
            "record_voicechat_session_started",
            model_id=runtime.model_id,
        )

        await _send_event(
            websocket,
            {
                "type": "session.updated",
                "session": _session_view(
                    session_id,
                    state="ready",
                    model=settings.model,
                    input_sample_rate=settings.input_sample_rate,
                    output_sample_rate=settings.output_sample_rate,
                    frame_samples=runtime.frame_samples,
                    settings=settings,
                ),
            },
        )

        processor_task = asyncio.create_task(
            _process_realtime_queue(
                websocket,
                runtime,
                queue,
                seen_event_ids,
                close_socket,
                control,
            ),
            name="voicechat-realtime-processor",
        )
        done, _ = await asyncio.wait(
            {reader_task, processor_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if reader_task in done and not processor_task.done():
            processor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await processor_task
    except WebSocketDisconnect:
        terminal_state = "disconnected"
        return
    except asyncio.CancelledError:
        terminal_state = "cancelled"
        if runtime is not None:
            _set_runtime_terminal(runtime, "cancelled")
            runtime.cancellations += 1
        return
    except RuntimeError:
        terminal_state = "disconnected"
        return
    except Exception:
        terminal_state = "error"
        raise
    finally:
        for task in (reader_task, processor_task):
            if task is not None and not task.done():
                task.cancel()
        for task in (reader_task, processor_task):
            if task is not None:
                with contextlib.suppress(BaseException):
                    await task
        if startup_task is not None and not startup_task.done():
            startup_task.cancel()
            with contextlib.suppress(BaseException):
                await startup_task
        await _close_runtime(runtime)
        _record_realtime_terminal(
            runtime,
            terminal_state or "disconnected",
            metric_model_id,
        )
        await close_socket()


async def _cancel_runtime(
    websocket: WebSocket,
    runtime: _RealtimeRuntime,
    control: _RealtimeControl,
    seen_event_ids: set[str],
) -> None:
    payload = control.cancel_payload
    control.cancel_payload = None
    control.cancel_event.clear()
    if payload is not None:
        try:
            _validate_event_id(payload, seen_event_ids)
        except _RealtimeProtocolError as exc:
            await _send_error(websocket, str(exc), code=exc.code, param=exc.param)
    _set_runtime_terminal(runtime, "cancelled")
    runtime.cancellations += 1
    if payload is not None and payload.get("type") == "response.cancel":
        runtime.interruptions += 1
    events = await _maybe_await(runtime.session.cancel())
    terminal = await _send_model_events(
        websocket,
        events,
        output_sample_rate=runtime.settings.output_sample_rate,
        runtime=runtime,
    )
    if not terminal:
        await _send_event(
            websocket, {"type": "response.cancelled", "frame_index": None}
        )


async def _handle_message_with_cancel(
    websocket: WebSocket,
    runtime: _RealtimeRuntime,
    inbound: _InboundMessage,
    seen_event_ids: set[str],
    control: _RealtimeControl,
) -> bool:
    operation = asyncio.create_task(
        _handle_client_message(websocket, runtime, inbound, seen_event_ids),
        name="voicechat-realtime-message",
    )
    cancellation = asyncio.create_task(
        control.cancel_event.wait(), name="voicechat-realtime-cancel-waiter"
    )
    try:
        done, _ = await asyncio.wait(
            {operation, cancellation}, return_when=asyncio.FIRST_COMPLETED
        )
        if cancellation in done:
            operation.cancel()
            with contextlib.suppress(BaseException):
                await operation
            await _cancel_runtime(websocket, runtime, control, seen_event_ids)
            return True
        return operation.result()
    finally:
        if not cancellation.done():
            cancellation.cancel()
        with contextlib.suppress(BaseException):
            await cancellation


async def _process_realtime_queue(
    websocket: WebSocket,
    runtime: _RealtimeRuntime,
    queue: asyncio.Queue[_InboundMessage],
    seen_event_ids: set[str],
    close_socket: Callable[[int], Awaitable[None]] | None = None,
    control: _RealtimeControl | None = None,
) -> None:
    while True:
        if control is None:
            inbound = await queue.get()
        else:
            next_message = asyncio.create_task(
                queue.get(), name="voicechat-realtime-message-waiter"
            )
            cancel_waiter = asyncio.create_task(
                control.cancel_event.wait(), name="voicechat-realtime-cancel-waiter"
            )
            done, _ = await asyncio.wait(
                {next_message, cancel_waiter}, return_when=asyncio.FIRST_COMPLETED
            )
            if cancel_waiter in done:
                next_message.cancel()
                with contextlib.suppress(BaseException):
                    await next_message
                cancel_waiter.cancel()
                with contextlib.suppress(BaseException):
                    await cancel_waiter
                await _cancel_runtime(websocket, runtime, control, seen_event_ids)
                return
            cancel_waiter.cancel()
            with contextlib.suppress(BaseException):
                await cancel_waiter
            inbound = next_message.result()
        try:
            if control is None:
                should_close = await _handle_client_message(
                    websocket,
                    runtime,
                    inbound,
                    seen_event_ids,
                )
            else:
                should_close = await _handle_message_with_cancel(
                    websocket,
                    runtime,
                    inbound,
                    seen_event_ids,
                    control,
                )
        except VoiceChatBusyError:
            _set_runtime_terminal(runtime, "busy")
            await _send_error(
                websocket,
                "another realtime VoiceChat session is already active",
                code="server_busy",
            )
            with contextlib.suppress(Exception):
                if close_socket is None:
                    await websocket.close(code=_BUSY_CLOSE_CODE)
                else:
                    await close_socket(_BUSY_CLOSE_CODE)
            return
        except _RealtimeProtocolError as exc:
            if exc.close_code is not None:
                _set_runtime_terminal(runtime, "error")
            await _send_error(
                websocket,
                str(exc),
                code=exc.code,
                param=exc.param,
            )
            should_close = exc.close_code is not None
            if exc.close_code is not None:
                with contextlib.suppress(Exception):
                    if close_socket is None:
                        await websocket.close(code=exc.close_code)
                    else:
                        await close_socket(exc.close_code)
        except (WebSocketDisconnect, RuntimeError):
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            _set_runtime_terminal(runtime, "error")
            await _send_error(
                websocket,
                "VoiceChat inference failed",
                code="inference_error",
            )
            return
        finally:
            queue.task_done()
        if should_close:
            if close_socket is None:
                with contextlib.suppress(Exception):
                    await websocket.close(code=1000)
            else:
                await close_socket(1000)
            return


router = realtime_router
realtime_voicechat_endpoint = realtime_voicechat


__all__ = [
    "router",
    "realtime_router",
    "realtime_voicechat",
    "realtime_voicechat_endpoint",
    "_audio_to_base64",
    "_decode_pcm16",
    "_pcm16_from_base64",
    "_serialize_model_event",
    "_verify_ws_api_key",
]

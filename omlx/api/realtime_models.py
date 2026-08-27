# SPDX-License-Identifier: Apache-2.0
"""Request and event models for the VoiceChat realtime WebSocket API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _validate_event_id(value: str | None) -> str | None:
    if value is not None and (not value.strip() or len(value) > 512):
        raise ValueError(
            "event_id must be a non-empty string of at most 512 characters"
        )
    return value


class RealtimeAudioFormat(BaseModel):
    """Structured PCM format used by the OpenAI/NVIDIA session contract."""

    model_config = ConfigDict(extra="ignore")

    type: str = "pcm16"
    rate: int | None = None
    sample_rate: int | None = None
    channels: int | None = None


class RealtimeTool(BaseModel):
    """A function tool definition accepted in a realtime session."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["function"] = "function"
    name: str | None = None
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    function: dict[str, Any] | None = None


class RealtimeSessionConfig(BaseModel):
    """Client-provided VoiceChat session settings.

    Both the model-specific ``system_prompt``/legacy format fields and the
    OpenAI GA ``instructions``/nested ``audio`` fields are represented. The
    route performs the cross-field capability checks because those checks need
    the loaded engine's native rates.
    """

    model_config = ConfigDict(extra="ignore")

    type: str | None = None
    model: str | None = None
    instructions: str | None = None
    system_prompt: str | None = None
    seed: int = 0
    max_streaming_seconds: float | None = None
    max_output_tokens: int | str | None = None
    input_audio_format: str | RealtimeAudioFormat | None = None
    output_audio_format: str | RealtimeAudioFormat | None = None
    input_sample_rate: int | None = None
    output_sample_rate: int | None = None
    frame_duration_ms: int | None = None
    audio: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    turn_detection: dict[str, Any] | None = None


class RealtimeSessionUpdate(BaseModel):
    """The first client event that configures a realtime session."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["session.update"]
    event_id: str | None = None
    session: RealtimeSessionConfig

    _event_id = field_validator("event_id")(_validate_event_id)


class RealtimeAudioAppend(BaseModel):
    """A base64-encoded PCM16 input chunk."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["input_audio_buffer.append"]
    event_id: str | None = None
    audio: str
    sample_rate: int | None = None

    _event_id = field_validator("event_id")(_validate_event_id)


class RealtimeAudioCommit(BaseModel):
    """Commit buffered audio and finish the current model response."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["input_audio_buffer.commit"]
    event_id: str | None = None
    pad_partial: bool = True

    _event_id = field_validator("event_id")(_validate_event_id)


class RealtimeResponseCreate(BaseModel):
    """Optional OpenAI response trigger accepted for client compatibility."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["response.create"]
    event_id: str | None = None
    response: dict[str, Any] | None = None

    _event_id = field_validator("event_id")(_validate_event_id)


class RealtimeResponseCancel(BaseModel):
    """Cancel a live VoiceChat response."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["response.cancel", "session.cancel"]
    event_id: str | None = None

    _event_id = field_validator("event_id")(_validate_event_id)


class RealtimeConversationItemCreate(BaseModel):
    """Conversation item event, including rejected tool-result items."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["conversation.item.create"]
    event_id: str | None = None
    item: dict[str, Any]

    _event_id = field_validator("event_id")(_validate_event_id)


class RealtimeError(BaseModel):
    """Structured error body used by server error events."""

    model_config = ConfigDict(extra="ignore")

    code: str
    message: str
    param: str | None = None
    event_id: str | None = None


class RealtimeServerEvent(BaseModel):
    """Base shape for a server event sent over the realtime socket."""

    model_config = ConfigDict(extra="allow")

    type: str
    event_id: str
    error: RealtimeError | None = None


# Names used by callers that mirror the OpenAI type names.
RealtimeInputAudioBufferAppend = RealtimeAudioAppend
RealtimeInputAudioBufferCommit = RealtimeAudioCommit
RealtimeSessionUpdateRequest = RealtimeSessionUpdate


__all__ = [
    "RealtimeAudioAppend",
    "RealtimeAudioCommit",
    "RealtimeAudioFormat",
    "RealtimeConversationItemCreate",
    "RealtimeError",
    "RealtimeInputAudioBufferAppend",
    "RealtimeInputAudioBufferCommit",
    "RealtimeResponseCancel",
    "RealtimeResponseCreate",
    "RealtimeServerEvent",
    "RealtimeSessionConfig",
    "RealtimeSessionUpdate",
    "RealtimeSessionUpdateRequest",
    "RealtimeTool",
]

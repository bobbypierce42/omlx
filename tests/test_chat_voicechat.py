# SPDX-License-Identifier: Apache-2.0
"""Static contracts for the web VoiceChat operator surface."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAT_TEMPLATE = ROOT / "omlx/admin/templates/chat.html"
I18N_DIR = ROOT / "omlx/admin/i18n"

VOICECHAT_I18N_KEYS = {
    "chat.voicechat.label",
    "chat.voicechat.start",
    "chat.voicechat.stop",
    "chat.voicechat.retry",
    "chat.voicechat.connecting",
    "chat.voicechat.listening",
    "chat.voicechat.active",
    "chat.voicechat.ending",
    "chat.voicechat.interrupted",
    "chat.voicechat.ended",
    "chat.voicechat.disconnected",
    "chat.voicechat.busy",
    "chat.voicechat.loading",
    "chat.voicechat.ready",
    "chat.voicechat.failed",
    "chat.voicechat.unloaded",
    "chat.voicechat.stale_models",
    "chat.voicechat.model_failed",
    "chat.voicechat.transcript",
    "chat.voicechat.assistant",
    "chat.voicechat.tools",
    "chat.voicechat.tool_pending",
    "chat.voicechat.tool_completed",
    "chat.voicechat.tool_failed",
    "chat.voicechat.tool_timed_out",
    "chat.voicechat.tool_unsupported",
    "chat.voicechat.input_format",
    "chat.voicechat.output_format",
    "chat.voicechat.input_placeholder",
    "chat.voicechat.no_audio",
    "chat.voicechat.mic_denied",
    "chat.voicechat.model_list_loading",
    "chat.voicechat.model_list_failed",
}


def _template() -> str:
    return CHAT_TEMPLATE.read_text(encoding="utf-8")


def _section(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


def _voicechat_section(source: str) -> str:
    return _section(
        source,
        "// ===== Realtime VoiceChat operator session =====",
        "// ===== Realtime microphone transcription =====",
    )


def test_voicechat_controls_are_capability_gated_and_picker_visible():
    source = _template()

    assert "realtimeVoiceChatMap" in source
    assert "hasVoiceChatSupport()" in source
    assert 'x-show="hasVoiceChatSupport()"' in source
    assert "realtime_voicechat" in source
    assert "t !== 'audio_tts' && t !== 'audio_sts'" in source
    assert "|| isVoiceChat" in source


def test_voicechat_uses_task_four_handshake_without_tools_or_start_event():
    voicechat = _voicechat_section(_template())

    assert "type: 'session.update'" in voicechat
    assert "api_key: this.getApiKey()" in voicechat
    assert "type: 'start'" not in voicechat
    assert "session.tools" not in voicechat
    assert "activeTools()" not in voicechat


def test_voicechat_negotiates_audio_and_cleans_up_microphone():
    source = _template()
    lifecycle = _section(source, "startVoiceChat()", "streamResponse(")

    assert "/v1/realtime?model=" in lifecycle
    assert "type: 'session.update'" in lifecycle
    assert "input_audio_format" in lifecycle
    assert "output_audio_format" in lifecycle
    assert "AudioWorklet" in lifecycle
    assert "input_audio_buffer.append" in lifecycle
    assert "input_audio_buffer.commit" in lifecycle
    assert "response.cancel" in lifecycle
    assert "getUserMedia" in lifecycle
    assert "getTracks().forEach(t => t.stop())" in lifecycle
    assert "_resampleVoiceChatSamples" in lifecycle


def test_voicechat_flushes_audio_worklet_tail_before_transport_stop():
    voicechat = _voicechat_section(_template())

    assert "this._emit()" in voicechat
    assert "{ type: 'flush' }" in voicechat
    assert "{ type: 'flushed' }" in voicechat
    assert "flushVoiceChatCapture(session)" in voicechat
    assert "session.flushingCapture" in voicechat
    assert "await this.flushVoiceChatCapture(session)" in voicechat


def test_voicechat_renders_transcript_text_playback_and_interruption():
    source = _template()

    assert "conversation.item.input_audio_transcription.delta" in source
    assert "response.text.delta" in source
    assert "response.audio.delta" in source
    assert "queueVoiceChatAudio" in source
    assert "voiceChatUserTranscript" in source
    assert "voiceChatAssistantText" in source
    assert "interrupted" in source
    assert "stopVoiceChatPlayback" in source


def test_voicechat_tool_rendering_is_metadata_only_and_never_continues():
    source = _template()
    tools = _section(source, "handleVoiceChatToolDelta(data)", "handleVoiceChatError(data)")

    for state in ("pending", "completed", "failed", "timed_out"):
        assert state in source
    assert "data.arguments" not in tools
    assert "data.result" not in tools
    assert "conversation.item.create" not in tools
    assert "tool_unsupported" in source


def test_voicechat_statuses_include_busy_stale_and_unloaded_paths():
    source = _template()

    for status in ("loading", "failed", "active", "busy", "stale", "unloaded"):
        assert status in source
    assert "model_list_loading" in source
    assert "model_list_failed" in source
    assert "retryVoiceChat()" in source
    select_model = _section(source, "async selectModel(modelId)", "loadChatHistory()")
    assert "stopVoiceChat({ discard: true })" in select_model


def test_voicechat_locale_keys_exist_in_every_locale():
    for locale_path in sorted(I18N_DIR.glob("*.json")):
        translations = json.loads(locale_path.read_text(encoding="utf-8"))
        missing = VOICECHAT_I18N_KEYS - translations.keys()
        assert not missing, f"{locale_path.name} is missing {sorted(missing)}"
        empty = {key for key in VOICECHAT_I18N_KEYS if not translations[key]}
        assert not empty, f"{locale_path.name} has empty values {sorted(empty)}"

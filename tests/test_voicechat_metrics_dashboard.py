# SPDX-License-Identifier: Apache-2.0
"""Static contracts for VoiceChat metrics and dashboard states."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_admin_stats_exposes_a_dedicated_voicechat_metrics_snapshot():
    source = (ROOT / "omlx/admin/routes.py").read_text()

    assert "get_voicechat_snapshot" in source
    assert '"voicechat_metrics"' in source
    assert "model_id=resolved_model" in source
    assert "scope=scope" in source


def test_dashboard_renders_all_voicechat_metrics_states():
    template = (ROOT / "omlx/admin/templates/dashboard/_status.html").read_text()
    javascript = (ROOT / "omlx/admin/static/js/dashboard.js").read_text()

    for state in ("loading", "error", "empty", "success"):
        assert f"voicechatMetricsStateForScope() === '{state}'" in template
    assert "voicechat_metrics" in javascript
    assert "voicechatStatsModels" in javascript
    assert 'x-for="m in statsModels"' in template
    assert "setVoicechatMetrics('session'" in javascript
    assert "setVoicechatMetrics('alltime'" in javascript
    assert "voicechatMetric" in template
    assert "first_audio_latency_seconds" in javascript
    assert "tool_outcomes" in template


def test_voicechat_dashboard_metric_surface_has_no_content_properties():
    template = (ROOT / "omlx/admin/templates/dashboard/_status.html").read_text()
    javascript = (ROOT / "omlx/admin/static/js/dashboard.js").read_text()
    surface = template + javascript

    for forbidden in (
        "raw_audio",
        "prompt_text",
        "tool_arguments",
        "tool_results",
        "credentials",
    ):
        assert forbidden not in surface

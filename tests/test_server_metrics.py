# SPDX-License-Identifier: Apache-2.0
"""Tests for server_metrics module."""

import json
import threading
from pathlib import Path

import pytest

from omlx.server_metrics import (
    VOICECHAT_ALLOWED_PROPERTIES,
    ServerMetrics,
    get_server_metrics,
    reset_server_metrics,
)


class TestServerMetrics:
    """Tests for ServerMetrics class."""

    def test_initial_snapshot(self):
        """Test that initial snapshot has all zero values."""
        metrics = ServerMetrics()
        snapshot = metrics.get_snapshot()

        assert snapshot["total_tokens_served"] == 0
        assert snapshot["total_cached_tokens"] == 0
        assert snapshot["cache_efficiency"] == 0.0
        assert snapshot["total_prompt_tokens"] == 0
        assert snapshot["total_completion_tokens"] == 0
        assert snapshot["total_requests"] == 0
        assert snapshot["avg_prefill_tps"] == 0.0
        assert snapshot["avg_generation_tps"] == 0.0
        assert snapshot["uptime_seconds"] >= 0

    def test_record_request_complete(self):
        """Test recording a single completed request."""
        metrics = ServerMetrics()
        metrics.record_request_complete(
            prompt_tokens=100,
            completion_tokens=50,
            cached_tokens=30,
            prefill_duration=0.5,
            generation_duration=1.0,
        )

        snapshot = metrics.get_snapshot()
        assert snapshot["total_tokens_served"] == 150
        assert snapshot["total_cached_tokens"] == 30
        assert snapshot["total_prompt_tokens"] == 100
        assert snapshot["total_completion_tokens"] == 50
        assert snapshot["total_requests"] == 1

    def test_multiple_requests(self):
        """Test accumulation across multiple requests."""
        metrics = ServerMetrics()

        for _ in range(5):
            metrics.record_request_complete(
                prompt_tokens=100,
                completion_tokens=50,
                cached_tokens=20,
                prefill_duration=0.2,
                generation_duration=0.5,
            )

        snapshot = metrics.get_snapshot()
        assert snapshot["total_tokens_served"] == 750  # (100+50)*5
        assert snapshot["total_cached_tokens"] == 100  # 20*5
        assert snapshot["total_requests"] == 5

    def test_cache_efficiency(self):
        """Test cache efficiency calculation (cached / prompt tokens)."""
        metrics = ServerMetrics()

        # 300 prompt tokens, 100 cached
        metrics.record_request_complete(
            prompt_tokens=200,
            completion_tokens=50,
            cached_tokens=60,
            prefill_duration=0.5,
            generation_duration=1.0,
        )
        metrics.record_request_complete(
            prompt_tokens=100,
            completion_tokens=50,
            cached_tokens=40,
            prefill_duration=0.3,
            generation_duration=0.5,
        )

        snapshot = metrics.get_snapshot()
        # cached_tokens=100, prompt_tokens=300 -> 33.3%
        assert snapshot["cache_efficiency"] == pytest.approx(33.3, abs=0.1)

    def test_cache_efficiency_zero_prompts(self):
        """Test cache efficiency when no prompts have been processed."""
        metrics = ServerMetrics()
        snapshot = metrics.get_snapshot()
        assert snapshot["cache_efficiency"] == 0.0

    def test_average_speed(self):
        """Test average speed calculation."""
        metrics = ServerMetrics()

        metrics.record_request_complete(
            prompt_tokens=1000,
            completion_tokens=100,
            prefill_duration=2.0,  # 500 tok/s
            generation_duration=2.0,  # 50 tok/s
        )
        metrics.record_request_complete(
            prompt_tokens=1000,
            completion_tokens=100,
            prefill_duration=2.0,
            generation_duration=2.0,
        )

        snapshot = metrics.get_snapshot()
        # total_prompt=2000, total_prefill_duration=4.0 -> 500 tok/s
        assert snapshot["avg_prefill_tps"] == pytest.approx(500.0, abs=0.1)
        # total_completion=200, total_gen_duration=4.0 -> 50 tok/s
        assert snapshot["avg_generation_tps"] == pytest.approx(50.0, abs=0.1)

    def test_average_prefill_speed_excludes_cached(self):
        """Test that average prefill speed excludes cached tokens."""
        metrics = ServerMetrics()
        metrics.record_request_complete(
            prompt_tokens=1000,
            completion_tokens=100,
            cached_tokens=400,
            prefill_duration=2.0,
            generation_duration=2.0,
        )

        snapshot = metrics.get_snapshot()
        # actual processed = 1000 - 400 = 600, duration = 2.0 -> 300 tok/s
        assert snapshot["avg_prefill_tps"] == pytest.approx(300.0, abs=0.1)
        # generation speed unchanged: 100 / 2.0 = 50 tok/s
        assert snapshot["avg_generation_tps"] == pytest.approx(50.0, abs=0.1)

    def test_average_speed_zero_duration(self):
        """Test average speed when duration is zero."""
        metrics = ServerMetrics()
        metrics.record_request_complete(
            prompt_tokens=100,
            completion_tokens=50,
            prefill_duration=0.0,
            generation_duration=0.0,
        )
        snapshot = metrics.get_snapshot()
        assert snapshot["avg_prefill_tps"] == 0.0
        assert snapshot["avg_generation_tps"] == 0.0

    def test_thread_safety(self):
        """Test concurrent recording from multiple threads."""
        metrics = ServerMetrics()
        num_threads = 10
        records_per_thread = 100

        def record_batch():
            for _ in range(records_per_thread):
                metrics.record_request_complete(
                    prompt_tokens=10,
                    completion_tokens=5,
                    cached_tokens=3,
                    prefill_duration=0.01,
                    generation_duration=0.01,
                )

        threads = [threading.Thread(target=record_batch) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snapshot = metrics.get_snapshot()
        total_expected = num_threads * records_per_thread
        assert snapshot["total_requests"] == total_expected
        assert snapshot["total_tokens_served"] == total_expected * 15  # 10+5
        assert snapshot["total_cached_tokens"] == total_expected * 3

    def test_snapshot_fields(self):
        """Test that snapshot returns all expected fields."""
        metrics = ServerMetrics()
        snapshot = metrics.get_snapshot()

        expected_fields = {
            "total_tokens_served",
            "total_cached_tokens",
            "cache_efficiency",
            "total_prompt_tokens",
            "total_completion_tokens",
            "total_requests",
            "avg_prefill_tps",
            "avg_generation_tps",
            "uptime_seconds",
        }
        assert set(snapshot.keys()) == expected_fields

    def test_default_cached_tokens(self):
        """Test that cached_tokens defaults to 0."""
        metrics = ServerMetrics()
        metrics.record_request_complete(
            prompt_tokens=100,
            completion_tokens=50,
        )
        snapshot = metrics.get_snapshot()
        assert snapshot["total_cached_tokens"] == 0

    def test_per_model_tracking(self):
        """Test that per-model counters track independently."""
        metrics = ServerMetrics()
        metrics.record_request_complete(
            prompt_tokens=100, completion_tokens=50, model_id="model-a"
        )
        metrics.record_request_complete(
            prompt_tokens=200, completion_tokens=80, model_id="model-b"
        )

        # Global should have both
        snapshot_all = metrics.get_snapshot()
        assert snapshot_all["total_prompt_tokens"] == 300
        assert snapshot_all["total_completion_tokens"] == 130

        # Per-model should be isolated
        snapshot_a = metrics.get_snapshot(model_id="model-a")
        assert snapshot_a["total_prompt_tokens"] == 100
        assert snapshot_a["total_completion_tokens"] == 50

        snapshot_b = metrics.get_snapshot(model_id="model-b")
        assert snapshot_b["total_prompt_tokens"] == 200
        assert snapshot_b["total_completion_tokens"] == 80

    def test_per_model_snapshot_calculations(self):
        """Test that per-model snapshot computes derived values correctly."""
        metrics = ServerMetrics()
        metrics.record_request_complete(
            prompt_tokens=1000,
            completion_tokens=100,
            cached_tokens=400,
            prefill_duration=2.0,
            generation_duration=2.0,
            model_id="fast-model",
        )
        metrics.record_request_complete(
            prompt_tokens=500,
            completion_tokens=50,
            cached_tokens=0,
            prefill_duration=1.0,
            generation_duration=1.0,
            model_id="slow-model",
        )

        snapshot = metrics.get_snapshot(model_id="fast-model")
        # (1000 - 400) / 2.0 = 300 tok/s
        assert snapshot["avg_prefill_tps"] == pytest.approx(300.0, abs=0.1)
        assert snapshot["cache_efficiency"] == pytest.approx(40.0, abs=0.1)
        assert snapshot["total_requests"] == 1

    def test_per_model_unknown_returns_zeros(self):
        """Test that unknown model_id returns zero metrics."""
        metrics = ServerMetrics()
        metrics.record_request_complete(
            prompt_tokens=100, completion_tokens=50, model_id="model-a"
        )

        snapshot = metrics.get_snapshot(model_id="nonexistent")
        assert snapshot["total_prompt_tokens"] == 0
        assert snapshot["total_completion_tokens"] == 0
        assert snapshot["total_requests"] == 0

    def test_clear_metrics(self):
        """Test that clear_metrics resets all counters."""
        metrics = ServerMetrics()
        metrics.record_request_complete(
            prompt_tokens=100,
            completion_tokens=50,
            cached_tokens=30,
            prefill_duration=0.5,
            generation_duration=1.0,
            model_id="model-a",
        )

        metrics.clear_metrics()
        snapshot = metrics.get_snapshot()
        assert snapshot["total_prompt_tokens"] == 0
        assert snapshot["total_completion_tokens"] == 0
        assert snapshot["total_cached_tokens"] == 0
        assert snapshot["total_requests"] == 0

        # Per-model should also be cleared
        snapshot_a = metrics.get_snapshot(model_id="model-a")
        # Falls back to global (empty) since per-model was cleared
        assert snapshot_a["total_prompt_tokens"] == 0

    def test_clear_metrics_does_not_affect_alltime(self):
        """Test that clear_metrics only resets session, not all-time."""
        metrics = ServerMetrics()
        metrics.record_request_complete(
            prompt_tokens=100, completion_tokens=50, model_id="model-a"
        )

        metrics.clear_metrics()

        # Session should be zero
        session = metrics.get_snapshot(scope="session")
        assert session["total_prompt_tokens"] == 0

        # All-time should still have the data
        alltime = metrics.get_snapshot(scope="alltime")
        assert alltime["total_prompt_tokens"] == 100
        assert alltime["total_completion_tokens"] == 50


class TestAlltimePersistence:
    """Tests for all-time stats persistence."""

    def test_alltime_snapshot(self):
        """Test that alltime scope returns cumulative data."""
        metrics = ServerMetrics()
        metrics.record_request_complete(prompt_tokens=100, completion_tokens=50)

        alltime = metrics.get_snapshot(scope="alltime")
        assert alltime["total_prompt_tokens"] == 100
        assert alltime["total_completion_tokens"] == 50
        assert alltime["total_requests"] == 1

    def test_alltime_per_model(self):
        """Test alltime per-model tracking."""
        metrics = ServerMetrics()
        metrics.record_request_complete(
            prompt_tokens=100, completion_tokens=50, model_id="model-a"
        )
        metrics.record_request_complete(
            prompt_tokens=200, completion_tokens=80, model_id="model-b"
        )

        alltime_a = metrics.get_snapshot(model_id="model-a", scope="alltime")
        assert alltime_a["total_prompt_tokens"] == 100

        alltime_b = metrics.get_snapshot(model_id="model-b", scope="alltime")
        assert alltime_b["total_prompt_tokens"] == 200

    def test_alltime_persistence_save_load(self, tmp_path):
        """Test save/load round-trip for all-time stats."""
        stats_path = tmp_path / "stats.json"

        # Create and populate metrics
        m1 = ServerMetrics(stats_path=stats_path)
        m1.record_request_complete(
            prompt_tokens=500,
            completion_tokens=100,
            cached_tokens=50,
            prefill_duration=1.0,
            generation_duration=2.0,
            model_id="test-model",
        )
        m1.save_alltime()

        # Verify file exists
        assert stats_path.exists()
        data = json.loads(stats_path.read_text())
        assert data["total_prompt_tokens"] == 500
        assert data["total_completion_tokens"] == 100
        assert data["total_cached_tokens"] == 50
        assert "test-model" in data["per_model"]

        # Load into new instance
        m2 = ServerMetrics(stats_path=stats_path)
        alltime = m2.get_snapshot(scope="alltime")
        assert alltime["total_prompt_tokens"] == 500
        assert alltime["total_completion_tokens"] == 100
        assert alltime["total_cached_tokens"] == 50
        assert alltime["total_requests"] == 1

        # Per-model should also be restored
        alltime_model = m2.get_snapshot(model_id="test-model", scope="alltime")
        assert alltime_model["total_prompt_tokens"] == 500

    def test_alltime_accumulates_across_resets(self, tmp_path):
        """Test that all-time stats accumulate across reset cycles."""
        stats_path = tmp_path / "stats.json"

        # Session 1
        m1 = ServerMetrics(stats_path=stats_path)
        m1.record_request_complete(prompt_tokens=100, completion_tokens=50)
        m1.save_alltime()

        # Session 2
        m2 = ServerMetrics(stats_path=stats_path)
        m2.record_request_complete(prompt_tokens=200, completion_tokens=80)
        m2.save_alltime()

        # Session 3: verify accumulation
        m3 = ServerMetrics(stats_path=stats_path)
        alltime = m3.get_snapshot(scope="alltime")
        assert alltime["total_prompt_tokens"] == 300  # 100 + 200
        assert alltime["total_completion_tokens"] == 130  # 50 + 80
        assert alltime["total_requests"] == 2

        # Session metrics should start fresh
        session = m3.get_snapshot(scope="session")
        assert session["total_prompt_tokens"] == 0
        assert session["total_requests"] == 0

    def test_clear_alltime_metrics(self, tmp_path):
        """Test that clear_alltime_metrics resets and deletes file."""
        stats_path = tmp_path / "stats.json"

        metrics = ServerMetrics(stats_path=stats_path)
        metrics.record_request_complete(prompt_tokens=100, completion_tokens=50)
        metrics.save_alltime()
        assert stats_path.exists()

        metrics.clear_alltime_metrics()

        alltime = metrics.get_snapshot(scope="alltime")
        assert alltime["total_prompt_tokens"] == 0
        assert alltime["total_requests"] == 0
        assert not stats_path.exists()

    def test_corrupted_stats_file(self, tmp_path):
        """Test graceful handling of corrupted stats file."""
        stats_path = tmp_path / "stats.json"
        stats_path.write_text("not valid json {{{")

        # Should not raise, should start with zeros
        metrics = ServerMetrics(stats_path=stats_path)
        alltime = metrics.get_snapshot(scope="alltime")
        assert alltime["total_prompt_tokens"] == 0
        assert alltime["total_requests"] == 0

    def test_missing_stats_file(self, tmp_path):
        """Test that missing stats file starts with zeros."""
        stats_path = tmp_path / "nonexistent" / "stats.json"

        metrics = ServerMetrics(stats_path=stats_path)
        alltime = metrics.get_snapshot(scope="alltime")
        assert alltime["total_prompt_tokens"] == 0

    def test_periodic_save_timing(self, tmp_path):
        """Test that periodic save respects the interval."""
        stats_path = tmp_path / "stats.json"
        metrics = ServerMetrics(stats_path=stats_path)

        # Record request - should not trigger save (interval not elapsed)
        metrics.record_request_complete(prompt_tokens=100, completion_tokens=50)
        assert not stats_path.exists()

        # Force save time to be in the past
        metrics._last_save_time = 0

        # Record again - should now trigger periodic save
        metrics.record_request_complete(prompt_tokens=100, completion_tokens=50)
        assert stats_path.exists()

    def test_reset_server_metrics_saves_alltime(self, tmp_path):
        """Test that reset_server_metrics saves before resetting."""
        stats_path = tmp_path / "stats.json"
        reset_server_metrics(stats_path=stats_path)
        m1 = get_server_metrics()
        m1.record_request_complete(prompt_tokens=100, completion_tokens=50)

        # Reset should save the all-time data
        reset_server_metrics(stats_path=stats_path)
        assert stats_path.exists()

        m2 = get_server_metrics()
        alltime = m2.get_snapshot(scope="alltime")
        assert alltime["total_prompt_tokens"] == 100

    def test_save_uses_atomic_write(self, tmp_path):
        """Test that save writes atomically via tmp file."""
        stats_path = tmp_path / "stats.json"
        metrics = ServerMetrics(stats_path=stats_path)
        metrics.record_request_complete(prompt_tokens=100, completion_tokens=50)
        metrics.save_alltime()

        # Verify no leftover tmp file
        tmp_file = stats_path.with_suffix(".json.tmp")
        assert not tmp_file.exists()
        assert stats_path.exists()


class TestVoiceChatMetrics:
    """Tests for local VoiceChat operational metrics."""

    def test_allowlist_excludes_content_and_credentials(self):
        forbidden = {
            "raw_audio",
            "transcript",
            "prompt",
            "instructions",
            "tools",
            "arguments",
            "results",
            "keys",
            "credential",
        }
        assert not forbidden & {
            property_name.lower() for property_name in VOICECHAT_ALLOWED_PROPERTIES
        }

        metrics = ServerMetrics()
        metrics.record_voicechat_event("transcript", model_id="voice-model")
        assert (
            metrics.get_voicechat_snapshot(model_id="voice-model")["sessions_started"]
            == 0
        )

    def test_terminal_paths_and_operational_counters(self):
        metrics = ServerMetrics()
        for terminal_state in ("done", "cancelled", "disconnected", "busy", "error"):
            metrics.record_voicechat_session_terminal(
                terminal_state,
                model_id="voice-model",
            )
        metrics.record_voicechat_session_started(model_id="voice-model")
        metrics.record_voicechat_offline_request(
            model_id="voice-model",
            outcome="completed",
            audio_duration_seconds=1.5,
        )
        metrics.record_voicechat_tool("requested", model_id="voice-model")
        metrics.record_voicechat_tool("failed", model_id="voice-model")
        metrics.record_voicechat_tool("parallel_rejected", model_id="voice-model")
        metrics.record_voicechat_tool("completed", model_id="voice-model")
        metrics.record_voicechat_tool("timeout", model_id="voice-model")

        snapshot = metrics.get_voicechat_snapshot(model_id="voice-model")

        assert snapshot["sessions_started"] == 1
        assert snapshot["session_terminal_states"] == {
            "done": 1,
            "cancelled": 1,
            "disconnected": 1,
            "busy": 1,
            "error": 1,
        }
        assert snapshot["offline_sts"]["requests"] == 1
        assert snapshot["offline_sts"]["completed"] == 1
        assert snapshot["audio"]["input_duration_seconds"] == pytest.approx(1.5)
        assert snapshot["busy_responses"] == 1
        assert snapshot["tool_outcomes"] == {
            "requested": 1,
            "failed": 1,
            "parallel_rejected": 1,
        }

    def test_latency_and_turns_are_aggregated(self):
        metrics = ServerMetrics()
        metrics.record_voicechat_session_terminal(
            "done",
            model_id="model-a",
            turns=2,
            interruptions=1,
            cancellations=1,
            audio_duration_seconds=2.0,
            output_audio_duration_seconds=3.0,
            first_transcript_latency_seconds=0.2,
            first_text_latency_seconds=0.4,
            first_audio_latency_seconds=0.6,
            turn_latency_seconds=1.2,
        )
        metrics.record_voicechat_session_terminal(
            "done",
            model_id="model-a",
            first_audio_latency_seconds=0.8,
        )

        snapshot = metrics.get_voicechat_snapshot(model_id="model-a")

        assert snapshot["turns"] == 2
        assert snapshot["interruptions"] == 1
        assert snapshot["cancellations"] == 1
        assert snapshot["audio"]["duration_seconds"] == pytest.approx(5.0)
        assert snapshot["latency"]["first_transcript_seconds"] == pytest.approx(0.2)
        assert snapshot["latency"]["first_text_seconds"] == pytest.approx(0.4)
        assert snapshot["latency"]["first_audio_seconds"] == pytest.approx(0.7)
        assert snapshot["latency"]["turn_seconds"] == pytest.approx(1.2)

    def test_per_model_scope_and_unknown_models_are_isolated(self):
        metrics = ServerMetrics()
        metrics.record_voicechat_session_started(model_id="model-a")
        metrics.record_voicechat_session_started(model_id="model-b")
        metrics.record_voicechat_session_terminal("done", model_id="model-a")

        model_a = metrics.get_voicechat_snapshot(model_id="model-a")
        model_b = metrics.get_voicechat_snapshot(model_id="model-b")
        global_snapshot = metrics.get_voicechat_snapshot()
        unknown = metrics.get_voicechat_snapshot(model_id="missing")

        assert model_a["sessions_started"] == 1
        assert model_a["session_terminal_states"]["done"] == 1
        assert model_b["sessions_started"] == 1
        assert model_b["session_terminal_states"]["done"] == 0
        assert global_snapshot["sessions_started"] == 2
        assert global_snapshot["session_terminal_states"]["done"] == 1
        assert unknown["sessions_started"] == 0
        assert unknown["session_terminal_states"]["done"] == 0

    def test_persistence_and_legacy_missing_fields(self, tmp_path):
        stats_path = tmp_path / "stats.json"
        metrics = ServerMetrics(stats_path=stats_path)
        metrics.record_voicechat_session_terminal(
            "done",
            model_id="voice-model",
            first_audio_latency_seconds=0.75,
        )
        metrics.save_alltime()

        persisted = json.loads(stats_path.read_text())
        assert persisted["voicechat"]["per_model"]["voice-model"]
        persisted_keys = set(persisted["voicechat"]["aggregate"])
        assert not persisted_keys & {
            "raw_audio",
            "transcript",
            "prompt",
            "instructions",
            "tools",
            "arguments",
            "results",
            "api_key",
            "credentials",
        }

        restored = ServerMetrics(stats_path=stats_path)
        assert (
            restored.get_voicechat_snapshot(scope="alltime")["session_terminal_states"][
                "done"
            ]
            == 1
        )
        assert restored.get_voicechat_snapshot(model_id="voice-model", scope="alltime")[
            "latency"
        ]["first_audio_seconds"] == pytest.approx(0.75)

        legacy_path = tmp_path / "legacy.json"
        legacy_path.write_text(json.dumps({"total_requests": 9}))
        legacy = ServerMetrics(stats_path=legacy_path)
        assert legacy.get_voicechat_snapshot(scope="alltime")["sessions_started"] == 0
        assert (
            legacy.get_voicechat_snapshot(scope="alltime")["offline_sts"]["requests"]
            == 0
        )

    def test_clear_session_and_alltime_voicechat_metrics(self, tmp_path):
        stats_path = tmp_path / "stats.json"
        metrics = ServerMetrics(stats_path=stats_path)
        metrics.record_voicechat_session_started(model_id="voice-model")
        metrics.record_voicechat_session_terminal("done", model_id="voice-model")
        metrics.save_alltime()

        metrics.clear_metrics()
        assert metrics.get_voicechat_snapshot()["sessions_started"] == 0
        assert metrics.get_voicechat_snapshot(scope="alltime")["sessions_started"] == 1

        metrics.clear_alltime_metrics()
        assert metrics.get_voicechat_snapshot(scope="alltime")["sessions_started"] == 0
        assert not stats_path.exists()


class TestServerMetricsSingleton:
    """Tests for global singleton functions."""

    def test_get_server_metrics_returns_instance(self):
        """Test that get_server_metrics returns a ServerMetrics instance."""
        reset_server_metrics()
        metrics = get_server_metrics()
        assert isinstance(metrics, ServerMetrics)

    def test_get_server_metrics_returns_same_instance(self):
        """Test that get_server_metrics returns the same instance."""
        reset_server_metrics()
        m1 = get_server_metrics()
        m2 = get_server_metrics()
        assert m1 is m2

    def test_reset_server_metrics(self):
        """Test that reset creates a fresh instance."""
        reset_server_metrics()
        m1 = get_server_metrics()
        m1.record_request_complete(prompt_tokens=100, completion_tokens=50)

        reset_server_metrics()
        m2 = get_server_metrics()

        assert m1 is not m2
        snapshot = m2.get_snapshot()
        assert snapshot["total_tokens_served"] == 0

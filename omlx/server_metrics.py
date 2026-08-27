# SPDX-License-Identifier: Apache-2.0
"""
Server-level metrics for the oMLX admin dashboard.

Provides a thread-safe singleton that aggregates serving metrics
across all engines/models. Session metrics reset on server start,
while all-time metrics persist across restarts via JSON file.
"""

import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Interval between periodic saves of all-time stats (seconds)
_SAVE_INTERVAL = 300

_VOICECHAT_COUNTER_FIELDS = (
    "sessions_started",
    "sessions_completed",
    "sessions_cancelled",
    "sessions_disconnected",
    "sessions_busy",
    "sessions_failed",
    "offline_requests",
    "offline_completed",
    "offline_failed",
    "offline_cancelled",
    "offline_busy",
    "turns",
    "interruptions",
    "cancellations",
    "busy_responses",
    "tool_requests",
    "tool_failures",
    "tool_parallel_rejections",
)
_VOICECHAT_DURATION_FIELDS = (
    "audio_duration_seconds",
    "output_audio_duration_seconds",
)
_VOICECHAT_LATENCY_FIELDS = (
    "first_transcript_latency",
    "first_text_latency",
    "first_audio_latency",
    "turn_latency",
)

# Only operational aggregates are accepted by the VoiceChat metrics surface.
# Content-bearing request fields intentionally have no representation here.
VOICECHAT_ALLOWED_PROPERTIES = frozenset(
    {
        "model_id",
        "terminal_state",
        "offline_outcome",
        "audio_duration_seconds",
        "output_audio_duration_seconds",
        "first_transcript_latency_seconds",
        "first_text_latency_seconds",
        "first_audio_latency_seconds",
        "turn_latency_seconds",
        "tool_outcome",
    }
)
VOICECHAT_PROPERTY_ALLOWLIST = VOICECHAT_ALLOWED_PROPERTIES


class ServerMetrics:
    """
    Global server-level metrics for the Status dashboard.

    Thread-safe: uses threading.Lock since scheduler runs in ThreadPoolExecutor.
    Tracks cumulative totals and average speeds across all requests,
    with optional per-model breakdown.

    Supports two scopes:
    - "session": resets on server restart (default, backward-compatible)
    - "alltime": persisted across restarts via stats_path JSON file
    """

    def __init__(self, stats_path: Optional[Path] = None):
        self._lock = threading.Lock()
        self._stats_path = stats_path

        # Session totals (reset on server restart or clear)
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_cached_tokens: int = 0
        self.total_requests: int = 0
        self.total_prefill_duration: float = 0.0
        self.total_generation_duration: float = 0.0
        self._per_model: Dict[str, Dict[str, Any]] = {}

        # Preflight rejections — split by reason so operators can tell
        # "user fed an oversize prompt" (hard_limit) apart from "system
        # under memory pressure, admission paused" (admission_paused).
        # This is the only operator-visible signal for the prefill-peak
        # memory guard; the guard was previously dead and shipping it
        # without telemetry repeats that mistake.
        self.preflight_rejections: Dict[str, int] = {
            "hard_limit": 0,
            "admission_paused": 0,
        }

        # All-time totals (persisted across restarts)
        self._alltime_prompt_tokens: int = 0
        self._alltime_completion_tokens: int = 0
        self._alltime_cached_tokens: int = 0
        self._alltime_requests: int = 0
        self._alltime_prefill_duration: float = 0.0
        self._alltime_generation_duration: float = 0.0
        self._alltime_per_model: Dict[str, Dict[str, Any]] = {}

        # VoiceChat aggregates use the same session/all-time persistence as
        # serving metrics, but stay in a separate safe namespace.
        self._voicechat = self._new_voicechat_counters()
        self._voicechat_per_model: Dict[str, Dict[str, Any]] = {}
        self._alltime_voicechat = self._new_voicechat_counters()
        self._alltime_voicechat_per_model: Dict[str, Dict[str, Any]] = {}

        self._start_time = time.time()
        self._last_save_time = time.time()

        # Load persisted all-time stats
        if stats_path:
            self._load_alltime()

    @staticmethod
    def _new_model_counters() -> Dict[str, Any]:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "requests": 0,
            "prefill_duration": 0.0,
            "generation_duration": 0.0,
        }

    @staticmethod
    def _new_voicechat_counters() -> Dict[str, Any]:
        counters: Dict[str, Any] = {field: 0 for field in _VOICECHAT_COUNTER_FIELDS}
        counters.update({field: 0.0 for field in _VOICECHAT_DURATION_FIELDS})
        for field in _VOICECHAT_LATENCY_FIELDS:
            counters[f"{field}_total_seconds"] = 0.0
            counters[f"{field}_count"] = 0
        return counters

    @staticmethod
    def _load_voicechat_bucket(target: Dict[str, Any], data: Any) -> None:
        """Load only known aggregate fields so old or foreign data is ignored."""
        if not isinstance(data, dict):
            return
        for field in _VOICECHAT_COUNTER_FIELDS:
            try:
                value = int(data.get(field, 0))
            except (TypeError, ValueError, OverflowError):
                value = 0
            target[field] = max(0, value)
        for field in _VOICECHAT_DURATION_FIELDS:
            try:
                value = float(data.get(field, 0.0))
            except (TypeError, ValueError, OverflowError):
                value = 0.0
            target[field] = value if math.isfinite(value) and value >= 0 else 0.0
        for field in _VOICECHAT_LATENCY_FIELDS:
            total_field = f"{field}_total_seconds"
            count_field = f"{field}_count"
            try:
                total = float(data.get(total_field, 0.0))
            except (TypeError, ValueError, OverflowError):
                total = 0.0
            try:
                count = int(data.get(count_field, 0))
            except (TypeError, ValueError, OverflowError):
                count = 0
            target[total_field] = total if math.isfinite(total) and total >= 0 else 0.0
            target[count_field] = max(0, count)

    def _load_alltime(self) -> None:
        """Load all-time stats from disk. Called once during __init__."""
        if not self._stats_path or not self._stats_path.exists():
            return
        try:
            with open(self._stats_path) as f:
                data = json.load(f)
            self._alltime_prompt_tokens = int(data.get("total_prompt_tokens", 0))
            self._alltime_completion_tokens = int(
                data.get("total_completion_tokens", 0)
            )
            self._alltime_cached_tokens = int(data.get("total_cached_tokens", 0))
            self._alltime_requests = int(data.get("total_requests", 0))
            self._alltime_prefill_duration = float(
                data.get("total_prefill_duration", 0.0)
            )
            self._alltime_generation_duration = float(
                data.get("total_generation_duration", 0.0)
            )
            per_model = data.get("per_model", {})
            for model_id, counters in per_model.items():
                self._alltime_per_model[model_id] = {
                    "prompt_tokens": int(counters.get("prompt_tokens", 0)),
                    "completion_tokens": int(counters.get("completion_tokens", 0)),
                    "cached_tokens": int(counters.get("cached_tokens", 0)),
                    "requests": int(counters.get("requests", 0)),
                    "prefill_duration": float(counters.get("prefill_duration", 0.0)),
                    "generation_duration": float(
                        counters.get("generation_duration", 0.0)
                    ),
                }
            voicechat_data = data.get("voicechat", data.get("voicechat_metrics", {}))
            if isinstance(voicechat_data, dict):
                aggregate = voicechat_data.get("aggregate", voicechat_data)
                self._load_voicechat_bucket(self._alltime_voicechat, aggregate)
                voicechat_per_model = voicechat_data.get("per_model", {})
                if isinstance(voicechat_per_model, dict):
                    for model_id, counters in voicechat_per_model.items():
                        if isinstance(model_id, str):
                            bucket = self._new_voicechat_counters()
                            self._load_voicechat_bucket(bucket, counters)
                            self._alltime_voicechat_per_model[model_id] = bucket
            logger.info("Loaded all-time stats from %s", self._stats_path)
        except (json.JSONDecodeError, TypeError, KeyError, ValueError, OSError) as e:
            logger.warning("Failed to load all-time stats from %s: %s", self._stats_path, e)

    def save_alltime(self) -> None:
        """Save all-time stats to disk. Thread-safe."""
        if not self._stats_path:
            return
        with self._lock:
            data = {
                "total_prompt_tokens": self._alltime_prompt_tokens,
                "total_completion_tokens": self._alltime_completion_tokens,
                "total_cached_tokens": self._alltime_cached_tokens,
                "total_requests": self._alltime_requests,
                "total_prefill_duration": self._alltime_prefill_duration,
                "total_generation_duration": self._alltime_generation_duration,
                "per_model": dict(self._alltime_per_model),
                "voicechat": {
                    "aggregate": dict(self._alltime_voicechat),
                    "per_model": {
                        model_id: dict(counters)
                        for model_id, counters in self._alltime_voicechat_per_model.items()
                    },
                },
            }
            self._last_save_time = time.time()
        try:
            self._stats_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._stats_path.with_suffix(".json.tmp")
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            tmp_path.replace(self._stats_path)
        except OSError as e:
            logger.warning("Failed to save all-time stats to %s: %s", self._stats_path, e)

    def _maybe_save_alltime(self) -> None:
        """Save all-time stats if enough time has passed. Called within lock."""
        if not self._stats_path:
            return
        now = time.time()
        if now - self._last_save_time >= _SAVE_INTERVAL:
            # Release lock before I/O
            self._lock.release()
            try:
                self.save_alltime()
            finally:
                self._lock.acquire()

    def record_request_complete(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        prefill_duration: float = 0.0,
        generation_duration: float = 0.0,
        model_id: str = "",
    ) -> None:
        """Record a completed request. Thread-safe."""
        with self._lock:
            # Session counters
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_cached_tokens += cached_tokens
            self.total_requests += 1
            self.total_prefill_duration += prefill_duration
            self.total_generation_duration += generation_duration

            # All-time counters
            self._alltime_prompt_tokens += prompt_tokens
            self._alltime_completion_tokens += completion_tokens
            self._alltime_cached_tokens += cached_tokens
            self._alltime_requests += 1
            self._alltime_prefill_duration += prefill_duration
            self._alltime_generation_duration += generation_duration

            # Per-model counters (session)
            if model_id:
                if model_id not in self._per_model:
                    self._per_model[model_id] = self._new_model_counters()
                m = self._per_model[model_id]
                m["prompt_tokens"] += prompt_tokens
                m["completion_tokens"] += completion_tokens
                m["cached_tokens"] += cached_tokens
                m["requests"] += 1
                m["prefill_duration"] += prefill_duration
                m["generation_duration"] += generation_duration

                # Per-model counters (all-time)
                if model_id not in self._alltime_per_model:
                    self._alltime_per_model[model_id] = self._new_model_counters()
                am = self._alltime_per_model[model_id]
                am["prompt_tokens"] += prompt_tokens
                am["completion_tokens"] += completion_tokens
                am["cached_tokens"] += cached_tokens
                am["requests"] += 1
                am["prefill_duration"] += prefill_duration
                am["generation_duration"] += generation_duration

            # Periodic save
            self._maybe_save_alltime()

    def record_preflight_rejection(self, reason: str) -> None:
        """Increment the preflight-rejection counter for ``reason``.

        Unknown reasons are bucketed under ``"other"`` rather than
        silently dropped so an operator notices when a new reject path
        is added without updating the metric.
        """
        with self._lock:
            if reason not in self.preflight_rejections:
                reason = "other"
                self.preflight_rejections.setdefault("other", 0)
            self.preflight_rejections[reason] += 1

    @staticmethod
    def _safe_voicechat_model_id(model_id: str) -> str:
        if not isinstance(model_id, str):
            return ""
        return model_id.strip()[:512]

    def _record_voicechat_update(
        self,
        model_id: str,
        updates: Dict[str, int | float],
    ) -> None:
        """Apply an allowlisted VoiceChat aggregate update to both scopes."""
        model_id = self._safe_voicechat_model_id(model_id)
        with self._lock:
            buckets = [self._voicechat, self._alltime_voicechat]
            if model_id:
                if model_id not in self._voicechat_per_model:
                    self._voicechat_per_model[model_id] = self._new_voicechat_counters()
                if model_id not in self._alltime_voicechat_per_model:
                    self._alltime_voicechat_per_model[model_id] = (
                        self._new_voicechat_counters()
                    )
                buckets.extend(
                    (
                        self._voicechat_per_model[model_id],
                        self._alltime_voicechat_per_model[model_id],
                    )
                )

            for bucket in buckets:
                for field, value in updates.items():
                    if field in bucket:
                        bucket[field] += value
            self._maybe_save_alltime()

    def record_voicechat_session_started(self, model_id: str = "") -> None:
        """Record a successfully initialized realtime VoiceChat session."""
        self._record_voicechat_update(model_id, {"sessions_started": 1})

    def record_voicechat_session_terminal(
        self,
        terminal_state: str,
        *,
        model_id: str = "",
        turns: int = 0,
        interruptions: int = 0,
        cancellations: int = 0,
        audio_duration_seconds: float = 0.0,
        output_audio_duration_seconds: float = 0.0,
        first_transcript_latency_seconds: float | None = None,
        first_text_latency_seconds: float | None = None,
        first_audio_latency_seconds: float | None = None,
        turn_latency_seconds: float | None = None,
        tool_requests: int = 0,
        tool_failures: int = 0,
        tool_parallel_rejections: int = 0,
    ) -> None:
        """Record one terminal realtime session and its safe aggregates."""
        state = str(terminal_state).strip().lower()
        state_field = {
            "done": "sessions_completed",
            "completed": "sessions_completed",
            "cancel": "sessions_cancelled",
            "cancelled": "sessions_cancelled",
            "canceled": "sessions_cancelled",
            "disconnected": "sessions_disconnected",
            "disconnect": "sessions_disconnected",
            "busy": "sessions_busy",
            "error": "sessions_failed",
            "failed": "sessions_failed",
        }.get(state, "sessions_failed")
        updates: Dict[str, int | float] = {state_field: 1}
        if state == "busy":
            updates["busy_responses"] = 1
        for field, value in (
            ("turns", turns),
            ("interruptions", interruptions),
            ("cancellations", cancellations),
            ("tool_requests", tool_requests),
            ("tool_failures", tool_failures),
            ("tool_parallel_rejections", tool_parallel_rejections),
        ):
            try:
                updates[field] = max(0, int(value))
            except (TypeError, ValueError, OverflowError):
                updates[field] = 0
        for field, value in (
            ("audio_duration_seconds", audio_duration_seconds),
            ("output_audio_duration_seconds", output_audio_duration_seconds),
        ):
            try:
                duration = float(value)
            except (TypeError, ValueError, OverflowError):
                duration = 0.0
            updates[field] = (
                duration if math.isfinite(duration) and duration >= 0 else 0.0
            )

        for field, value in (
            ("first_transcript_latency", first_transcript_latency_seconds),
            ("first_text_latency", first_text_latency_seconds),
            ("first_audio_latency", first_audio_latency_seconds),
            ("turn_latency", turn_latency_seconds),
        ):
            if value is None:
                continue
            try:
                latency = float(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(latency) and latency >= 0:
                updates[f"{field}_total_seconds"] = latency
                updates[f"{field}_count"] = 1
        self._record_voicechat_update(model_id, updates)

    def record_voicechat_offline_request(
        self,
        *,
        model_id: str = "",
        outcome: str = "failed",
        audio_duration_seconds: float = 0.0,
    ) -> None:
        """Record one VoiceChat offline STS request without retaining audio."""
        outcome = str(outcome).strip().lower().replace("-", "_").replace(" ", "_")
        outcome_field = {
            "completed": "offline_completed",
            "success": "offline_completed",
            "failed": "offline_failed",
            "error": "offline_failed",
            "cancel": "offline_cancelled",
            "cancelled": "offline_cancelled",
            "canceled": "offline_cancelled",
            "busy": "offline_busy",
        }.get(outcome, "offline_failed")
        updates: Dict[str, int | float] = {
            "offline_requests": 1,
            outcome_field: 1,
        }
        if outcome == "busy":
            updates["busy_responses"] = 1
        try:
            duration = float(audio_duration_seconds)
        except (TypeError, ValueError, OverflowError):
            duration = 0.0
        updates["audio_duration_seconds"] = (
            duration if math.isfinite(duration) and duration >= 0 else 0.0
        )
        self._record_voicechat_update(model_id, updates)

    def record_voicechat_tool(self, outcome: str, *, model_id: str = "") -> None:
        """Record a tool lifecycle outcome supported by the local transport."""
        outcome_field = {
            "requested": "tool_requests",
            "tool_requested": "tool_requests",
            "failed": "tool_failures",
            "tool_failed": "tool_failures",
            "parallel_rejected": "tool_parallel_rejections",
            "parallel_reject": "tool_parallel_rejections",
            "tool_parallel_rejected": "tool_parallel_rejections",
            "tool_parallel_reject": "tool_parallel_rejections",
        }.get(str(outcome).strip().lower().replace("-", "_").replace(" ", "_"))
        # A tool result cannot complete or time out in this backend because
        # same-session continuation is not available. Do not manufacture it.
        if outcome_field is None:
            return
        self._record_voicechat_update(model_id, {outcome_field: 1})

    def record_voicechat_event(self, event: str, *, model_id: str = "") -> None:
        """Record one named operational event; unknown/content events are ignored."""
        event_name = str(event).strip().lower().replace("-", "_").replace(" ", "_")
        if event_name == "session_started":
            self.record_voicechat_session_started(model_id)
        elif event_name in {
            "done",
            "completed",
            "cancelled",
            "canceled",
            "disconnected",
            "disconnect",
            "busy",
            "error",
            "failed",
        }:
            self.record_voicechat_session_terminal(event_name, model_id=model_id)
        elif event_name in {
            "requested",
            "tool_requested",
            "failed_tool",
            "tool_failed",
            "parallel_rejected",
            "parallel_reject",
            "tool_parallel_rejected",
            "tool_parallel_reject",
        }:
            self.record_voicechat_tool(
                "failed"
                if event_name in {"failed_tool", "tool_failed"}
                else event_name,
                model_id=model_id,
            )

    def _build_snapshot(
        self,
        prompt: int,
        completion: int,
        cached: int,
        requests: int,
        prefill_dur: float,
        gen_dur: float,
        uptime: float,
    ) -> Dict[str, Any]:
        """Build a metrics snapshot dict from raw values."""
        actual_processed = prompt - cached
        avg_prefill_tps = (
            actual_processed / prefill_dur if prefill_dur > 0 else 0.0
        )
        avg_generation_tps = completion / gen_dur if gen_dur > 0 else 0.0
        cache_efficiency = (cached / prompt * 100) if prompt > 0 else 0.0

        return {
            "total_tokens_served": prompt + completion,
            "total_cached_tokens": cached,
            "cache_efficiency": round(cache_efficiency, 1),
            "total_prompt_tokens": prompt,
            "total_completion_tokens": completion,
            "total_requests": requests,
            "avg_prefill_tps": round(avg_prefill_tps, 1),
            "avg_generation_tps": round(avg_generation_tps, 1),
            "uptime_seconds": round(uptime, 1),
        }

    def get_snapshot(
        self, model_id: str = "", scope: str = "session"
    ) -> Dict[str, Any]:
        """Get current metrics snapshot. Thread-safe.

        Args:
            model_id: If provided and tracked, return per-model metrics.
                      Otherwise return global aggregate.
            scope: "session" for current session, "alltime" for persisted totals.
        """
        with self._lock:
            now = time.time()
            uptime = now - self._start_time

            if scope == "alltime":
                if model_id:
                    m = self._alltime_per_model.get(model_id)
                    if m:
                        return self._build_snapshot(
                            m["prompt_tokens"],
                            m["completion_tokens"],
                            m["cached_tokens"],
                            m["requests"],
                            m["prefill_duration"],
                            m["generation_duration"],
                            uptime,
                        )
                    return self._build_snapshot(0, 0, 0, 0, 0.0, 0.0, uptime)
                return self._build_snapshot(
                    self._alltime_prompt_tokens,
                    self._alltime_completion_tokens,
                    self._alltime_cached_tokens,
                    self._alltime_requests,
                    self._alltime_prefill_duration,
                    self._alltime_generation_duration,
                    uptime,
                )

            # scope == "session" (default)
            if model_id:
                m = self._per_model.get(model_id)
                if m:
                    return self._build_snapshot(
                        m["prompt_tokens"],
                        m["completion_tokens"],
                        m["cached_tokens"],
                        m["requests"],
                        m["prefill_duration"],
                        m["generation_duration"],
                        uptime,
                    )
                return self._build_snapshot(0, 0, 0, 0, 0.0, 0.0, uptime)

            return self._build_snapshot(
                self.total_prompt_tokens,
                self.total_completion_tokens,
                self.total_cached_tokens,
                self.total_requests,
                self.total_prefill_duration,
                self.total_generation_duration,
                uptime,
            )

    @staticmethod
    def _build_voicechat_snapshot(bucket: Dict[str, Any], scope: str) -> Dict[str, Any]:
        """Build a JSON-safe VoiceChat dashboard snapshot from known fields."""
        result: Dict[str, Any] = {
            field: bucket[field] for field in _VOICECHAT_COUNTER_FIELDS
        }
        result["audio_duration_seconds"] = round(
            float(bucket["audio_duration_seconds"]), 3
        )
        result["output_audio_duration_seconds"] = round(
            float(bucket["output_audio_duration_seconds"]), 3
        )
        result["session_terminal_states"] = {
            "done": result["sessions_completed"],
            "cancelled": result["sessions_cancelled"],
            "disconnected": result["sessions_disconnected"],
            "busy": result["sessions_busy"],
            "error": result["sessions_failed"],
        }
        result["offline_sts"] = {
            "requests": result["offline_requests"],
            "completed": result["offline_completed"],
            "failed": result["offline_failed"],
            "cancelled": result["offline_cancelled"],
            "busy": result["offline_busy"],
        }
        result["tool_outcomes"] = {
            "requested": result["tool_requests"],
            "failed": result["tool_failures"],
            "parallel_rejected": result["tool_parallel_rejections"],
        }
        result["audio"] = {
            "input_duration_seconds": result["audio_duration_seconds"],
            "output_duration_seconds": result["output_audio_duration_seconds"],
            "duration_seconds": round(
                result["audio_duration_seconds"]
                + result["output_audio_duration_seconds"],
                3,
            ),
        }
        latency: Dict[str, float] = {}
        for field, public_name in (
            ("first_transcript_latency", "first_transcript_seconds"),
            ("first_text_latency", "first_text_seconds"),
            ("first_audio_latency", "first_audio_seconds"),
            ("turn_latency", "turn_seconds"),
        ):
            count = int(bucket[f"{field}_count"])
            average = float(bucket[f"{field}_total_seconds"]) / count if count else 0.0
            latency[public_name] = round(average, 3)
            result[f"{field}_latency_seconds"] = round(average, 3)
            result[f"{field}_latency_observations"] = count
        result["latency"] = latency
        result["scope"] = "alltime" if scope == "alltime" else "session"
        return result

    def get_voicechat_snapshot(
        self, model_id: str = "", scope: str = "session"
    ) -> Dict[str, Any]:
        """Get safe VoiceChat metrics for a session or persisted all-time scope."""
        model_id = self._safe_voicechat_model_id(model_id)
        with self._lock:
            if scope == "alltime":
                bucket = (
                    self._alltime_voicechat_per_model.get(model_id)
                    if model_id
                    else self._alltime_voicechat
                )
            else:
                bucket = (
                    self._voicechat_per_model.get(model_id)
                    if model_id
                    else self._voicechat
                )
            if bucket is None:
                bucket = self._new_voicechat_counters()
            return self._build_voicechat_snapshot(bucket, scope)

    get_voicechat_metrics = get_voicechat_snapshot

    def clear_metrics(self) -> None:
        """Clear session metrics. Thread-safe."""
        with self._lock:
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
            self.total_cached_tokens = 0
            self.total_requests = 0
            self.total_prefill_duration = 0.0
            self.total_generation_duration = 0.0
            self._per_model.clear()
            self._voicechat = self._new_voicechat_counters()
            self._voicechat_per_model.clear()

    def clear_alltime_metrics(self) -> None:
        """Clear all-time metrics and delete the persisted file. Thread-safe."""
        with self._lock:
            self._alltime_prompt_tokens = 0
            self._alltime_completion_tokens = 0
            self._alltime_cached_tokens = 0
            self._alltime_requests = 0
            self._alltime_prefill_duration = 0.0
            self._alltime_generation_duration = 0.0
            self._alltime_per_model.clear()
            self._alltime_voicechat = self._new_voicechat_counters()
            self._alltime_voicechat_per_model.clear()
        if self._stats_path and self._stats_path.exists():
            try:
                self._stats_path.unlink()
            except OSError as e:
                logger.warning(
                    "Failed to delete stats file %s: %s", self._stats_path, e
                )


# Global singleton
_server_metrics: Optional[ServerMetrics] = None


def get_server_metrics() -> ServerMetrics:
    """Get the global ServerMetrics singleton."""
    global _server_metrics
    if _server_metrics is None:
        _server_metrics = ServerMetrics()
    return _server_metrics


def reset_server_metrics(stats_path: Optional[Path] = None) -> None:
    """Reset metrics (called on server start).

    If a previous instance exists and has a stats_path, save before resetting.
    """
    global _server_metrics
    if _server_metrics is not None:
        _server_metrics.save_alltime()
    _server_metrics = ServerMetrics(stats_path=stats_path)

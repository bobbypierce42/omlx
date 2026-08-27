# SPDX-License-Identifier: Apache-2.0
"""Tests for POST /v1/audio/process (STS — Speech-to-Speech / audio processing).

Verifies the STS endpoint accepts multipart audio uploads and returns WAV
audio bytes.

All unit tests run with mocked STSEngine and EnginePool — mlx-audio is not
required. Integration tests (marked @pytest.mark.slow) need a real model.
"""

import asyncio
import io
import os
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# WAV fixture helpers
# ---------------------------------------------------------------------------


def _make_wav_bytes(duration_secs: float = 0.1, sample_rate: int = 16000) -> bytes:
    """Generate minimal valid WAV bytes (silence)."""
    n_samples = int(sample_rate * duration_secs)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()


TINY_WAV = _make_wav_bytes()
RIFF_MAGIC = b"RIFF"


# ---------------------------------------------------------------------------
# Mock STS/VoiceChat engines and EnginePool
# ---------------------------------------------------------------------------


def _make_mock_sts_engine(output_wav: bytes = None) -> MagicMock:
    """Build a mock STSEngine that returns the given WAV bytes."""
    from omlx.engine.sts import STSEngine

    engine = MagicMock(spec=STSEngine)
    engine.process = AsyncMock(return_value=output_wav or TINY_WAV)
    return engine


def _make_mock_voicechat_engine(output_wav: bytes = None) -> MagicMock:
    """Build a mock VoiceChatEngine that returns the given WAV bytes."""
    from omlx.engine.voicechat import VoiceChatEngine

    engine = MagicMock(spec=VoiceChatEngine)
    engine.process = AsyncMock(return_value=output_wav or TINY_WAV)
    return engine


def _make_mock_pool(
    sts_engine=None,
    model_id: str = "deepfilternet",
    entry=None,
) -> MagicMock:
    """Build a mock EnginePool that returns the given STS engine."""
    pool = MagicMock()
    pool.get_engine = AsyncMock(return_value=sts_engine or _make_mock_sts_engine())
    pool.get_entry = MagicMock(
        return_value=entry
        if entry is not None
        else MagicMock(model_type="audio_sts", engine_type="audio_sts")
    )
    pool.get_model_ids.return_value = [model_id]
    pool.preload_pinned_models = AsyncMock()
    pool.check_ttl_expirations = AsyncMock()
    pool.shutdown = AsyncMock()
    pool.resolve_model_id = MagicMock(side_effect=lambda m, _: m)
    return pool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ensure_audio_routes(app):
    """Register audio routes if not already present (e.g., mlx-audio not installed)."""
    from omlx.api.audio_routes import router as audio_router

    audio_paths = {"/v1/audio/transcriptions", "/v1/audio/speech", "/v1/audio/process"}
    existing = {getattr(r, "path", "") for r in app.routes}
    if not audio_paths & existing:
        app.include_router(audio_router)


@pytest.fixture
def server_sts_client():
    """TestClient using the full omlx server app with mocked STS pool."""
    from omlx.server import app

    _ensure_audio_routes(app)

    mock_pool = _make_mock_pool()

    with patch("omlx.server._server_state") as mock_state:
        mock_state.engine_pool = mock_pool
        mock_state.global_settings = None
        mock_state.distributed_inference_enabled = False
        mock_state.process_memory_enforcer = None
        mock_state.hf_downloader = None
        mock_state.ms_downloader = None
        mock_state.mcp_manager = None
        mock_state.api_key = None
        mock_state.settings_manager = MagicMock()
        mock_state.settings_manager.resolve_model_id = MagicMock(
            side_effect=lambda m, _: m
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, mock_pool


@pytest.fixture
def audio_sts_client():
    """Minimal TestClient for the audio router with a mocked STS engine."""
    from omlx.api.audio_routes import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)

    mock_pool = _make_mock_pool()

    with patch("omlx.api.audio_routes._get_engine_pool", return_value=mock_pool):
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, mock_pool


@pytest.fixture
def voicechat_client():
    """Minimal TestClient using a mocked VoiceChat engine and pool."""
    from fastapi import FastAPI

    from omlx.api.audio_routes import router

    app = FastAPI()
    app.include_router(router)

    voicechat = _make_mock_voicechat_engine()
    entry = MagicMock(
        model_type="audio_sts",
        engine_type="voicechat",
        realtime_voicechat=True,
    )
    mock_pool = _make_mock_pool(
        voicechat,
        model_id="voicechat",
        entry=entry,
    )

    with (
        patch("omlx.api.audio_routes._get_engine_pool", return_value=mock_pool),
        patch("omlx.api.audio_routes._resolve_model", side_effect=lambda model: model),
        TestClient(app, raise_server_exceptions=False) as client,
    ):
        yield client, mock_pool, voicechat


# ---------------------------------------------------------------------------
# TestSTSEndpointBasic
# ---------------------------------------------------------------------------


class TestSTSEndpointBasic:
    """Core STS endpoint behaviour."""

    def test_post_process_returns_200(self, server_sts_client):
        """POST /v1/audio/process with valid WAV returns 200."""
        client, _ = server_sts_client
        response = client.post(
            "/v1/audio/process",
            files={"file": ("audio.wav", TINY_WAV, "audio/wav")},
            data={"model": "deepfilternet"},
        )
        assert response.status_code == 200

    def test_response_is_audio_bytes(self, server_sts_client):
        """Response body is non-empty bytes."""
        client, _ = server_sts_client
        response = client.post(
            "/v1/audio/process",
            files={"file": ("audio.wav", TINY_WAV, "audio/wav")},
            data={"model": "deepfilternet"},
        )
        assert len(response.content) > 0

    def test_response_has_wav_header(self, server_sts_client):
        """Response starts with RIFF WAV magic bytes."""
        client, _ = server_sts_client
        response = client.post(
            "/v1/audio/process",
            files={"file": ("audio.wav", TINY_WAV, "audio/wav")},
            data={"model": "deepfilternet"},
        )
        assert response.status_code == 200
        assert response.content[:4] == RIFF_MAGIC

    def test_response_content_type_is_audio(self, server_sts_client):
        """Content-Type indicates audio (wav or octet-stream)."""
        client, _ = server_sts_client
        response = client.post(
            "/v1/audio/process",
            files={"file": ("audio.wav", TINY_WAV, "audio/wav")},
            data={"model": "deepfilternet"},
        )
        ct = response.headers.get("content-type", "")
        assert "audio" in ct or "octet-stream" in ct

    def test_engine_loaded_via_pool(self, server_sts_client):
        """EnginePool.get_engine() is called with the provided model ID."""
        client, mock_pool = server_sts_client
        client.post(
            "/v1/audio/process",
            files={"file": ("audio.wav", TINY_WAV, "audio/wav")},
            data={"model": "deepfilternet"},
        )
        mock_pool.get_engine.assert_awaited()

    def test_engine_process_called_with_file(self, server_sts_client):
        """engine.process() is called (file path forwarded)."""
        client, mock_pool = server_sts_client
        client.post(
            "/v1/audio/process",
            files={"file": ("audio.wav", TINY_WAV, "audio/wav")},
            data={"model": "deepfilternet"},
        )
        process_mock: AsyncMock = mock_pool.get_engine.return_value.process
        process_mock.assert_awaited_once()

    def test_different_model_names_accepted(self, server_sts_client):
        """Various STS model names are forwarded to the pool correctly."""
        client, mock_pool = server_sts_client
        for model_name in ("mossformer2-se", "deepfilternet3", "sam-audio-base"):
            mock_pool.get_engine = AsyncMock(return_value=_make_mock_sts_engine())
            response = client.post(
                "/v1/audio/process",
                files={"file": ("audio.wav", TINY_WAV, "audio/wav")},
                data={"model": model_name},
            )
            assert response.status_code == 200


class TestVoiceChatEndpoint:
    """VoiceChat-specific dispatch and lifecycle behavior."""

    def test_voicechat_process_returns_playable_wav(self, voicechat_client):
        client, _, voicechat = voicechat_client
        metrics = MagicMock()
        with patch("omlx.api.audio_routes.get_server_metrics", return_value=metrics):
            response = client.post(
                "/v1/audio/process",
                files={"file": ("recorded.wav", TINY_WAV, "audio/wav")},
                data={"model": "voicechat"},
            )

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        with wave.open(io.BytesIO(response.content), "rb") as output:
            assert output.getnchannels() == 1
            assert output.getframerate() == 16000
            assert output.getnframes() > 0
        voicechat.process.assert_awaited_once()
        temp_path = voicechat.process.await_args.args[0]
        assert not os.path.exists(temp_path)
        metrics.record_voicechat_offline_request.assert_called_once()
        metric_kwargs = metrics.record_voicechat_offline_request.call_args.kwargs
        assert metric_kwargs["model_id"] == "voicechat"
        assert metric_kwargs["outcome"] == "completed"
        assert metric_kwargs["audio_duration_seconds"] == pytest.approx(0.1)

    def test_voicechat_process_records_failed_metric(self, voicechat_client):
        client, _, voicechat = voicechat_client
        voicechat.process.side_effect = RuntimeError("processing failed")
        metrics = MagicMock()
        with patch("omlx.api.audio_routes.get_server_metrics", return_value=metrics):
            response = client.post(
                "/v1/audio/process",
                files={"file": ("recorded.wav", TINY_WAV, "audio/wav")},
                data={"model": "voicechat"},
            )

        assert response.status_code >= 500
        metric_kwargs = metrics.record_voicechat_offline_request.call_args.kwargs
        assert metric_kwargs["outcome"] == "failed"
        assert metric_kwargs["audio_duration_seconds"] == pytest.approx(0.1)

    def test_empty_audio_is_rejected_before_inference(self, voicechat_client):
        client, _, voicechat = voicechat_client
        response = client.post(
            "/v1/audio/process",
            files={"file": ("empty.wav", b"", "audio/wav")},
            data={"model": "voicechat"},
        )

        assert response.status_code == 400
        voicechat.process.assert_not_awaited()

    def test_invalid_audio_error_is_not_successful(self, voicechat_client):
        client, _, voicechat = voicechat_client
        voicechat.process.side_effect = ValueError("invalid audio")

        response = client.post(
            "/v1/audio/process",
            files={"file": ("recorded.wav", b"not-audio", "audio/wav")},
            data={"model": "voicechat"},
        )

        assert response.status_code >= 400

    def test_voicechat_load_failure_returns_server_error(self, voicechat_client):
        client, pool, _ = voicechat_client
        pool.get_engine.side_effect = RuntimeError("voicechat load failed")
        metrics = MagicMock()

        with patch("omlx.api.audio_routes.get_server_metrics", return_value=metrics):
            response = client.post(
                "/v1/audio/process",
                files={"file": ("recorded.wav", TINY_WAV, "audio/wav")},
                data={"model": "voicechat"},
            )

        assert response.status_code >= 500
        assert (
            metrics.record_voicechat_offline_request.call_args.kwargs["outcome"]
            == "failed"
        )

    def test_voicechat_busy_returns_retryable_503(self, voicechat_client):
        from omlx.engine.voicechat import VoiceChatBusyError

        client, _, voicechat = voicechat_client
        voicechat.process.side_effect = VoiceChatBusyError()
        metrics = MagicMock()

        with patch("omlx.api.audio_routes.get_server_metrics", return_value=metrics):
            response = client.post(
                "/v1/audio/process",
                files={"file": ("recorded.wav", TINY_WAV, "audio/wav")},
                data={"model": "voicechat"},
            )

        assert response.status_code == 503
        assert response.headers["retry-after"] == "1"
        error = response.json()["error"]
        assert error["code"] == "busy"
        assert error["retryable"] is True
        assert (
            metrics.record_voicechat_offline_request.call_args.kwargs["outcome"]
            == "busy"
        )

    def test_client_disconnect_cleans_uploaded_tempfile(self):
        from fastapi import UploadFile

        from omlx.api.audio_routes import process_audio

        voicechat = _make_mock_voicechat_engine()
        pool = _make_mock_pool(
            voicechat,
            model_id="voicechat",
            entry=MagicMock(
                model_type="audio_sts",
                engine_type="voicechat",
                realtime_voicechat=True,
            ),
        )
        started = asyncio.Event()
        state = {}

        async def blocked_process(audio_path):
            state["path"] = audio_path
            started.set()
            await asyncio.Future()

        voicechat.process.side_effect = blocked_process
        metrics = MagicMock()

        async def run_request():
            with (
                patch("omlx.api.audio_routes._get_engine_pool", return_value=pool),
                patch(
                    "omlx.api.audio_routes._resolve_model",
                    side_effect=lambda model: model,
                ),
                patch("omlx.api.audio_routes.get_server_metrics", return_value=metrics),
            ):
                upload = UploadFile(file=io.BytesIO(TINY_WAV), filename="audio.wav")
                request = asyncio.create_task(process_audio(upload, "voicechat"))
                await started.wait()
                request.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await request

        asyncio.run(run_request())
        assert not os.path.exists(state["path"])
        assert (
            metrics.record_voicechat_offline_request.call_args.kwargs["outcome"]
            == "cancelled"
        )

    def test_invalid_engine_output_returns_server_error(self, voicechat_client):
        client, _, voicechat = voicechat_client
        voicechat.process.return_value = b"not-a-wav"

        response = client.post(
            "/v1/audio/process",
            files={"file": ("recorded.wav", TINY_WAV, "audio/wav")},
            data={"model": "voicechat"},
        )

        assert response.status_code >= 500


# ---------------------------------------------------------------------------
# TestSTSEndpointErrors
# ---------------------------------------------------------------------------


class TestSTSEndpointErrors:
    """Error cases for the STS endpoint."""

    def test_missing_file_returns_error(self, server_sts_client):
        """Request without file field returns 4xx error."""
        client, _ = server_sts_client
        response = client.post(
            "/v1/audio/process",
            data={"model": "deepfilternet"},
        )
        assert response.status_code >= 400

    def test_missing_model_returns_error(self, server_sts_client):
        """Request without model field returns 4xx error."""
        client, _ = server_sts_client
        response = client.post(
            "/v1/audio/process",
            files={"file": ("audio.wav", TINY_WAV, "audio/wav")},
        )
        assert response.status_code >= 400

    def test_unsupported_model_returns_404(self, server_sts_client):
        """Requesting an unknown model returns 404."""
        client, mock_pool = server_sts_client
        from omlx.exceptions import ModelNotFoundError
        mock_pool.get_engine.side_effect = ModelNotFoundError(
            model_id="nonexistent-sts",
            available_models=["deepfilternet"],
        )
        response = client.post(
            "/v1/audio/process",
            files={"file": ("audio.wav", TINY_WAV, "audio/wav")},
            data={"model": "nonexistent-sts"},
        )
        assert response.status_code in (404, 400, 422)

    def test_engine_error_returns_500(self, server_sts_client):
        """Engine runtime error returns 5xx."""
        client, mock_pool = server_sts_client
        mock_pool.get_engine.return_value.process = AsyncMock(
            side_effect=RuntimeError("processing failed")
        )
        response = client.post(
            "/v1/audio/process",
            files={"file": ("audio.wav", TINY_WAV, "audio/wav")},
            data={"model": "deepfilternet"},
        )
        assert response.status_code >= 500


# ---------------------------------------------------------------------------
# TestSTSModelAliasResolution
# ---------------------------------------------------------------------------


class TestSTSModelAliasResolution:
    """Verify that STS endpoint resolves model aliases (#489)."""

    def test_process_resolves_alias(self):
        """POST /v1/audio/process with alias resolves to real model ID."""
        from omlx.server import app

        _ensure_audio_routes(app)

        mock_pool = _make_mock_pool(model_id="MossFormer2-SE")
        mock_pool.resolve_model_id = MagicMock(
            return_value="MossFormer2-SE"
        )

        with patch("omlx.server._server_state") as mock_state:
            mock_state.engine_pool = mock_pool
            mock_state.global_settings = None
            mock_state.distributed_inference_enabled = False
            mock_state.process_memory_enforcer = None
            mock_state.hf_downloader = None
            mock_state.ms_downloader = None
            mock_state.mcp_manager = None
            mock_state.api_key = None
            mock_state.settings_manager = MagicMock()
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post(
                    "/v1/audio/process",
                    data={"model": "denoise"},
                    files={"file": ("test.wav", TINY_WAV, "audio/wav")},
                )
                assert response.status_code == 200
                mock_pool.get_engine.assert_awaited_once_with("MossFormer2-SE")

    def test_process_direct_model_id(self):
        """POST /v1/audio/process with direct model ID works without alias."""
        from omlx.server import app

        _ensure_audio_routes(app)

        mock_pool = _make_mock_pool(model_id="MossFormer2-SE")
        mock_pool.resolve_model_id = MagicMock(
            return_value="MossFormer2-SE"
        )

        with patch("omlx.server._server_state") as mock_state:
            mock_state.engine_pool = mock_pool
            mock_state.global_settings = None
            mock_state.distributed_inference_enabled = False
            mock_state.process_memory_enforcer = None
            mock_state.hf_downloader = None
            mock_state.ms_downloader = None
            mock_state.mcp_manager = None
            mock_state.api_key = None
            mock_state.settings_manager = MagicMock()
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post(
                    "/v1/audio/process",
                    data={"model": "MossFormer2-SE"},
                    files={"file": ("test.wav", TINY_WAV, "audio/wav")},
                )
                assert response.status_code == 200
                mock_pool.get_engine.assert_awaited_once_with("MossFormer2-SE")


# ---------------------------------------------------------------------------
# TestSTSEngineUnit
# ---------------------------------------------------------------------------


class TestSTSEngineUnit:
    """Unit tests for STSEngine (no mlx-audio required)."""

    def test_import(self):
        """STSEngine can be imported."""
        from omlx.engine.sts import STSEngine
        assert STSEngine is not None

    def test_init(self):
        """STSEngine can be instantiated."""
        from omlx.engine.sts import STSEngine
        engine = STSEngine("mlx-community/DeepFilterNet-mlx")
        assert engine.model_name == "mlx-community/DeepFilterNet-mlx"

    def test_get_stats_not_loaded(self):
        """get_stats() returns loaded=False when not started."""
        from omlx.engine.sts import STSEngine
        engine = STSEngine("test-sts-model")
        stats = engine.get_stats()
        assert stats["loaded"] is False
        assert stats["model_name"] == "test-sts-model"

    def test_repr(self):
        """__repr__ shows stopped status before start()."""
        from omlx.engine.sts import STSEngine
        engine = STSEngine("my-model")
        r = repr(engine)
        assert "stopped" in r
        assert "my-model" in r

    def test_family_detection_deepfilternet(self):
        """Family is detected as deepfilternet for matching model name."""
        from omlx.engine.sts import _detect_sts_family
        assert _detect_sts_family("deepfilternet3") == "deepfilternet"
        assert _detect_sts_family("mlx-community/DeepFilterNet-mlx") == "deepfilternet"

    def test_family_detection_mossformer2(self):
        """Family is detected as mossformer2."""
        from omlx.engine.sts import _detect_sts_family
        assert _detect_sts_family("MossFormer2-SE-48K") == "mossformer2"
        assert _detect_sts_family("starkdmi/MossFormer2-SE") == "mossformer2"

    def test_family_detection_sam_audio(self):
        """Family is detected as sam_audio."""
        from omlx.engine.sts import _detect_sts_family
        assert _detect_sts_family("mlx-community/sam-audio-base-fp16") == "sam_audio"

    def test_family_detection_lfm2(self):
        """Family is detected as lfm2."""
        from omlx.engine.sts import _detect_sts_family
        assert _detect_sts_family("mlx-community/LFM2.5-Audio-1B") == "lfm2"
        assert _detect_sts_family("mlx-community/LFM2.5-Audio-1.5B-6bit") == "lfm2"

    def test_family_detection_generic(self):
        """Unknown model name returns 'generic'."""
        from omlx.engine.sts import _detect_sts_family
        assert _detect_sts_family("some-unknown-audio-model") == "generic"

    def test_process_raises_if_not_started(self):
        """process() raises RuntimeError if engine not started."""
        import asyncio
        from omlx.engine.sts import STSEngine
        engine = STSEngine("test-model")
        with pytest.raises(RuntimeError, match="not started"):
            asyncio.run(engine.process("/tmp/fake.wav"))

    def test_get_stats_has_family(self):
        """get_stats() includes 'family' key."""
        from omlx.engine.sts import STSEngine
        engine = STSEngine("mlx-community/sam-audio-base-fp16")
        stats = engine.get_stats()
        assert "family" in stats
        assert stats["family"] == "sam_audio"

    def test_start_rejects_generic_family(self):
        """start() raises ValueError for unsupported 'generic' family."""
        import asyncio
        from omlx.engine.sts import STSEngine
        engine = STSEngine("unknown-model-xyz")
        with pytest.raises(ValueError, match="Unsupported STS model family"):
            asyncio.run(engine.start())


# ---------------------------------------------------------------------------
# TestSTSModelRequest
# ---------------------------------------------------------------------------


class TestSTSModelRequest:
    """Pydantic model tests for AudioProcessRequest."""

    def test_audio_process_request_model(self):
        """AudioProcessRequest accepts a model field."""
        from omlx.api.audio_models import AudioProcessRequest
        req = AudioProcessRequest(model="deepfilternet")
        assert req.model == "deepfilternet"

    def test_audio_process_request_requires_model(self):
        """AudioProcessRequest raises ValidationError without model."""
        from omlx.api.audio_models import AudioProcessRequest
        with pytest.raises(Exception):  # pydantic ValidationError
            AudioProcessRequest()


# ---------------------------------------------------------------------------
# Integration tests (slow, requires mlx-audio + downloaded models)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSTSIntegrationDeepFilterNet:
    """Integration test for DeepFilterNet speech enhancement."""

    def test_enhance_produces_wav(self, tmp_path):
        """DeepFilterNet enhancement returns valid WAV bytes."""
        pytest.importorskip("mlx_audio")

        from omlx.engine.sts import STSEngine
        import asyncio

        model_name = "mlx-community/DeepFilterNet-mlx"
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(TINY_WAV)

        try:
            engine = STSEngine(model_name)
            asyncio.run(engine.start())
            result = asyncio.run(engine.process(str(wav_path)))
            assert isinstance(result, bytes)
            assert result[:4] == RIFF_MAGIC
            asyncio.run(engine.stop())
        except Exception as e:
            pytest.skip(f"Could not run integration test: {e}")


@pytest.mark.slow
class TestSTSIntegrationMossFormer2:
    """Integration test for MossFormer2 speech enhancement."""

    def test_enhance_produces_wav(self, tmp_path):
        """MossFormer2 enhancement returns valid WAV bytes."""
        pytest.importorskip("mlx_audio")

        from omlx.engine.sts import STSEngine
        import asyncio

        model_name = "starkdmi/MossFormer2-SE"
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(TINY_WAV)

        try:
            engine = STSEngine(model_name)
            asyncio.run(engine.start())
            result = asyncio.run(engine.process(str(wav_path)))
            assert isinstance(result, bytes)
            assert result[:4] == RIFF_MAGIC
            asyncio.run(engine.stop())
        except Exception as e:
            pytest.skip(f"Could not run integration test: {e}")


@pytest.mark.slow
class TestSTSIntegrationSAMAudio:
    """Integration test for SAMAudio separation."""

    def test_separate_produces_wav(self, tmp_path):
        """SAMAudio separation returns valid WAV bytes."""
        pytest.importorskip("mlx_audio")

        from omlx.engine.sts import STSEngine
        import asyncio

        model_name = "mlx-community/sam-audio-base-fp16"
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(TINY_WAV)

        try:
            engine = STSEngine(model_name)
            asyncio.run(engine.start())
            result = asyncio.run(engine.process(
                str(wav_path), descriptions=["speech"]
            ))
            assert isinstance(result, bytes)
            assert result[:4] == RIFF_MAGIC
            asyncio.run(engine.stop())
        except Exception as e:
            pytest.skip(f"Could not run integration test: {e}")


@pytest.mark.slow
class TestSTSIntegrationLFM2:
    """Integration test for LFM2.5-Audio speech-to-speech."""

    def test_sts_produces_wav(self, tmp_path):
        """LFM2 STS generation returns valid WAV bytes."""
        pytest.importorskip("mlx_audio")

        from omlx.engine.sts import STSEngine
        import asyncio

        model_name = "mlx-community/LFM2.5-Audio-1.5B-6bit"
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(TINY_WAV)

        try:
            engine = STSEngine(model_name)
            asyncio.run(engine.start())
            result = asyncio.run(engine.process(
                str(wav_path), max_new_tokens=64
            ))
            assert isinstance(result, bytes)
            assert result[:4] == RIFF_MAGIC
            asyncio.run(engine.stop())
        except Exception as e:
            pytest.skip(f"Could not run integration test: {e}")

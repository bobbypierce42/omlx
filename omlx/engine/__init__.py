# SPDX-License-Identifier: Apache-2.0
"""
Engine abstraction for oMLX inference.

Provides multiple engine implementations:
- BatchedEngine: Continuous batching for multiple concurrent users
- VLMBatchedEngine: Vision-language model engine with image support
- EmbeddingEngine: Batch embedding generation using mlx-embeddings
- RerankerEngine: Document reranking using SequenceClassification models

Also re-exports core engine components for backwards compatibility.
"""

# Re-export from parent engine.py for backwards compatibility
from ..engine_core import AsyncEngineCore, EngineConfig, EngineCore
from .base import BaseEngine, BaseNonStreamingEngine, GenerationOutput
from .batched import BatchedEngine
from .dflash import DFlashEngine
from .embedding import EmbeddingEngine
from .reranker import RerankerEngine
from .sts import STSEngine
from .stt import STTEngine
from .tts import TTSEngine
from .vlm import VLMBatchedEngine
from .voicechat import (
    VoiceChatBusyError,
    VoiceChatEngine,
    VoiceChatEngineStatus,
    VoiceChatError,
    VoiceChatEvent,
    VoiceChatEventKind,
    VoiceChatEventType,
    VoiceChatInferenceError,
    VoiceChatRealtimeSession,
    VoiceChatSession,
    VoiceChatStatus,
    normalize_voicechat_event,
    normalize_voicechat_events,
)

__all__ = [
    "BaseEngine",
    "BaseNonStreamingEngine",
    "GenerationOutput",
    "BatchedEngine",
    "DFlashEngine",
    "VLMBatchedEngine",
    "EmbeddingEngine",
    "RerankerEngine",
    "STTEngine",
    "STSEngine",
    "TTSEngine",
    "VoiceChatBusyError",
    "VoiceChatError",
    "VoiceChatEngine",
    "VoiceChatEngineStatus",
    "VoiceChatEvent",
    "VoiceChatEventKind",
    "VoiceChatEventType",
    "VoiceChatInferenceError",
    "VoiceChatRealtimeSession",
    "VoiceChatSession",
    "VoiceChatStatus",
    "normalize_voicechat_event",
    "normalize_voicechat_events",
    # Core engine components
    "EngineCore",
    "AsyncEngineCore",
    "EngineConfig",
]

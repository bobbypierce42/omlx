# SPDX-License-Identifier: Apache-2.0
"""Dependency and import smoke tests for the VoiceChat baseline."""

from __future__ import annotations

import importlib
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
MLX_VLM_REVISION = "e2bb5d68422ba28e51e530dc1c63b067dd2d1cb3"
MLX_AUDIO_REVISION = "8c4fb74b3393475f3080a46b204c511781a547d4"


def _revision(requirement: str) -> str:
    return requirement.rsplit("@", 1)[1]


def test_voicechat_dependency_pins_are_aligned():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project_dependencies = pyproject["project"]["dependencies"]
    audio_dependencies = pyproject["project"]["optional-dependencies"]["audio"]
    vlm_requirement = next(
        requirement
        for requirement in project_dependencies
        if requirement.startswith("mlx-vlm @ ")
    )
    audio_requirement = next(
        requirement
        for requirement in audio_dependencies
        if requirement.startswith("mlx-audio[")
    )
    transformers_requirement = next(
        requirement
        for requirement in project_dependencies
        if requirement.startswith("transformers")
    )

    build_text = (ROOT / "packaging/build.py").read_text()
    formula_text = (ROOT / "Formula/omlx.rb").read_text()
    build_match = re.search(r'_MLX_AUDIO_GIT = "[^@]+@([0-9a-f]{40})"', build_text)
    formula_match = re.search(r'revision: "([0-9a-f]{40})"', formula_text)

    assert build_match is not None
    assert formula_match is not None
    assert _revision(vlm_requirement) == MLX_VLM_REVISION
    assert _revision(audio_requirement) == MLX_AUDIO_REVISION
    assert transformers_requirement == "transformers>=5.14.0"
    assert build_match.group(1) == MLX_AUDIO_REVISION
    assert formula_match.group(1) == MLX_AUDIO_REVISION


@pytest.mark.parametrize(
    ("module_name", "symbol"),
    [
        ("mlx_vlm.models.nemotron_voicechat", "VoiceChatSession"),
        ("mlx_vlm.models.nemotron_voicechat.session", "VoiceChatSession"),
        (
            "mlx_vlm.models.nemotron_voicechat.streaming",
            "VoiceChatStreamingSession",
        ),
        (
            "mlx_audio.codec.models.nemotron_voicechat",
            "NemotronVoiceChatCodec",
        ),
        ("mlx_audio.stt.models.nemotron_asr", "ConformerStreamingState"),
        (
            "mlx_audio.stt.models.nemotron_asr.streaming",
            "ConformerStreamingState",
        ),
    ],
)
def test_voicechat_modules_import(module_name: str, symbol: str):
    pytest.importorskip("mlx")
    pytest.importorskip("mlx_vlm")
    pytest.importorskip("mlx_audio")

    module = importlib.import_module(module_name)
    assert hasattr(module, symbol)
